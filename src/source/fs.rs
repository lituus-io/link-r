// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The local-filesystem source: index a directory tree (offline / testing).
//!
//! Walks a directory, yielding one [`Page`] per indexable file with a `file://`
//! URL. Reads are lazy (per stream item) and synchronous, so this needs no async
//! runtime or `async-stream`.

use crate::error::{Error, Result};
use crate::payload::DocPayload;
use crate::resource::{Page, Resource, ResourceKind, SourceRef};
use crate::source::Source;
use bytes::Bytes;
use futures::StreamExt;
use std::path::PathBuf;
use url::Url;
use walkdir::WalkDir;

/// A source over a local directory tree.
#[derive(Clone, Copy, Debug, Default)]
pub struct FsSource;

impl FsSource {
    /// Create a filesystem source.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

fn read_page(path: &PathBuf) -> Result<Page<'static>> {
    let kind = ResourceKind::from_path(&path.to_string_lossy());
    let bytes = std::fs::read(path)?;
    let url = Url::from_file_path(path)
        .map_err(|()| Error::invalid_url(format!("not an absolute path: {}", path.display())))?;
    Ok(Page::new(
        Resource::new(url).with_kind(kind),
        DocPayload::Owned(Bytes::from(bytes)),
    ))
}

impl Source for FsSource {
    type Pages<'a>
        = impl futures::Stream<Item = Result<Page<'a>>> + Send + 'a
    where
        Self: 'a;

    fn kind(&self) -> &'static str {
        "fs"
    }

    fn discover<'a>(&'a self, root: &'a SourceRef) -> Self::Pages<'a> {
        // Collect the file list eagerly (cheap), read contents lazily per item.
        let paths: Vec<PathBuf> = match root {
            SourceRef::Fs { root } => WalkDir::new(root)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
                .map(|e| e.path().to_path_buf())
                .filter(|p| ResourceKind::from_path(&p.to_string_lossy()).is_indexable())
                .collect(),
            SourceRef::Http { .. } => Vec::new(),
        };

        let wrong_kind = !matches!(root, SourceRef::Fs { .. });
        futures::stream::iter(paths)
            .map(|path| read_page(&path))
            .chain(futures::stream::iter(wrong_kind.then(|| {
                Err(Error::crawl("fs", "FsSource requires an Fs SourceRef"))
            })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::io::Write;

    fn write_file(dir: &std::path::Path, name: &str, contents: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
    }

    #[tokio::test]
    async fn walks_directory_and_yields_indexable_files() {
        let tmp = tempfile::tempdir().unwrap();
        write_file(tmp.path(), "a.md", "# A\n\nalpha content");
        write_file(tmp.path(), "sub/b.txt", "bravo content");
        write_file(tmp.path(), "image.png", "not indexable");

        let source = FsSource::new();
        let root = SourceRef::fs(tmp.path());
        let stream = source.discover(&root);
        futures::pin_mut!(stream);

        let mut kinds = Vec::new();
        while let Some(item) = stream.next().await {
            let page = item.unwrap();
            kinds.push(page.resource.kind);
            assert_eq!(page.resource.url.scheme(), "file");
        }
        // png is filtered out as non-indexable; .md and .txt remain.
        assert_eq!(kinds.len(), 2);
        assert!(kinds.contains(&ResourceKind::Markdown));
        assert!(kinds.contains(&ResourceKind::Text));
    }

    #[tokio::test]
    async fn rejects_non_fs_source_ref() {
        let source = FsSource::new();
        let root = SourceRef::http("https://x.dev/").unwrap();
        let stream = source.discover(&root);
        futures::pin_mut!(stream);
        let first = stream.next().await.unwrap();
        assert!(first.is_err());
    }
}
