// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The single `unsafe` island in `link-r`.
//!
//! Two operations live here and nowhere else: memory-mapping an index file
//! (`Mmap::map`) and reinterpreting a validated, aligned byte run as `&[f32]`
//! (the zero-copy dense-vector view). Everything downstream consumes the safe
//! `&[u8]`/`&[f32]` these produce. The rest of the crate is `#![deny(unsafe_code)]`;
//! this module locally re-permits it under the safety contract documented at each
//! call site.
#![allow(unsafe_code)]

use crate::error::{Error, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// An owned, read-only memory map of an index file.
#[derive(Debug)]
pub struct MappedFile {
    mmap: Mmap,
}

impl MappedFile {
    /// Memory-map a file read-only.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the file cannot be opened or mapped.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: we open the file read-only and never expose a mutable view; the
        // `Mmap` owns the mapping for the lifetime of `MappedFile`, and callers
        // only ever receive `&[u8]` borrowed from it. The standard mmap caveat
        // (external truncation causing SIGBUS) is accepted, matching the house
        // `sapling-storage`/`roots-local` mmap usage.
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(Self { mmap })
    }

    /// The mapped bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }
}

/// Reinterpret a byte run as `&[f32]` without copying.
///
/// The index format guarantees the dense section starts at a 64-byte-aligned file
/// offset, and an mmap base is page-aligned, so the in-memory address is suitably
/// aligned; this function nonetheless verifies length and alignment and errors
/// (never UB) if either fails — the contract the loader fuzz target enforces.
///
/// # Errors
/// Returns [`Error::Format`] if `bytes` is not a whole number of `f32`s or is not
/// `f32`-aligned in memory.
pub fn cast_f32(bytes: &[u8]) -> Result<&[f32]> {
    // The reinterpret relies on native little-endian f32 layout.
    const _: () = assert!(
        cfg!(target_endian = "little"),
        "link-r index format is little-endian"
    );

    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(Error::format("dense section length is not a multiple of 4"));
    }
    let ptr = bytes.as_ptr();
    if (ptr as usize) % std::mem::align_of::<f32>() != 0 {
        return Err(Error::format("dense section is not f32-aligned"));
    }
    let len = bytes.len() / std::mem::size_of::<f32>();
    // SAFETY: `bytes` is a valid slice of `len * 4` initialized bytes; we verified
    // it is `f32`-aligned and a whole number of `f32`s; `f32` has no invalid bit
    // patterns (every 4 bytes is a valid float); and the returned slice borrows
    // `bytes`, so its lifetime cannot outlive the underlying mapping.
    #[allow(clippy::cast_ptr_alignment)] // alignment is verified above at runtime
    Ok(unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), len) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cast_roundtrips_aligned_bytes() {
        // A Vec<f32> is f32-aligned; viewing its bytes and casting back must agree.
        let floats = vec![1.0f32, 2.5, -3.25, 4.0];
        let bytes: &[u8] = bytemuck_free_as_bytes(&floats);
        let casted = cast_f32(bytes).unwrap();
        assert_eq!(casted, floats.as_slice());
    }

    #[test]
    fn cast_rejects_bad_length() {
        let bytes = [0u8; 6]; // not a multiple of 4
        assert!(cast_f32(&bytes).is_err());
    }

    #[test]
    fn cast_rejects_misalignment() {
        // Offset a 4-aligned buffer by one byte to force misalignment.
        let floats = vec![0.0f32; 4];
        let bytes = bytemuck_free_as_bytes(&floats);
        let misaligned = &bytes[1..5];
        // length is 4 (ok) but pointer is +1 from an aligned base → must error.
        assert!(cast_f32(misaligned).is_err());
    }

    /// View `&[f32]` as `&[u8]` for tests without pulling in bytemuck. The pointer
    /// stays f32-aligned, which is exactly what we want to exercise `cast_f32`.
    fn bytemuck_free_as_bytes(floats: &[f32]) -> &[u8] {
        // SAFETY (test-only): f32 is Copy and every bit pattern is a valid u8;
        // the slice borrows `floats` so the lifetime is sound.
        unsafe {
            std::slice::from_raw_parts(floats.as_ptr().cast::<u8>(), std::mem::size_of_val(floats))
        }
    }
}
