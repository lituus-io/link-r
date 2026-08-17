// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Durable, atomic index writes.
//!
//! Serialize to a sibling temp file, `fsync` it, then `rename` over the target
//! (atomic on POSIX same-filesystem). A crash leaves either the old index or the
//! new one — never a half-written file — and a reader never observes a partial
//! index.

use crate::error::Result;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Atomically write `bytes` to `path`.
///
/// # Errors
/// Returns [`crate::Error::Io`] on any filesystem failure; the destination is left
/// untouched unless the final rename succeeds.
pub fn atomic_write(path: impl AsRef<Path>, bytes: &[u8]) -> Result<()> {
    let path = path.as_ref();
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("index");

    // Unique-ish temp name in the same directory (same filesystem → atomic rename).
    let pid = std::process::id();
    let tmp_name = format!(".{file_name}.{pid}.tmp");
    let tmp_path = match dir {
        Some(d) => d.join(tmp_name),
        None => std::path::PathBuf::from(tmp_name),
    };

    {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }

    // Rename over the destination. On failure, clean up the temp file.
    if let Err(e) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e.into());
    }

    // Best-effort durability of the directory entry.
    if let Some(d) = dir {
        if let Ok(dir_file) = fs::File::open(d) {
            let _ = dir_file.sync_all();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_and_replaces_atomically() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("idx.lnkr");

        atomic_write(&path, b"first").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"first");

        // overwrite
        atomic_write(&path, b"second-larger").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second-larger");

        // no stray temp files left behind
        let leftovers: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files leaked: {leftovers:?}");
    }
}
