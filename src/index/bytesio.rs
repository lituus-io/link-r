// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Little-endian byte I/O shared across the index format, BM25, and metadata
//! codecs. Safe (no transmutes): fixed-width writers and a bounds-checked
//! [`Reader`] cursor whose decoders return [`Error::Format`] rather than panic on
//! truncated input — the property the loader fuzz target enforces.

use crate::error::{Error, Result};

/// Round `n` up to a multiple of `align` (a power of two).
#[inline]
#[must_use]
pub fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// Append a single byte.
#[inline]
pub fn put_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}
/// Append a little-endian `u16`.
#[inline]
pub fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
/// Append a little-endian `u32`.
#[inline]
pub fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
/// Append a little-endian `u64`.
#[inline]
pub fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
/// Append a little-endian `f32` (as its bit pattern).
#[inline]
pub fn put_f32(buf: &mut Vec<u8>, v: f32) {
    put_u32(buf, v.to_bits());
}
/// Append a length-prefixed (`u32` length) byte slice.
#[inline]
pub fn put_len_prefixed(buf: &mut Vec<u8>, bytes: &[u8]) {
    put_u32(buf, bytes.len() as u32);
    buf.extend_from_slice(bytes);
}

/// Read a little-endian `u16` at a known, in-bounds offset.
#[inline]
#[must_use]
pub fn get_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
/// Read a little-endian `u32` at a known, in-bounds offset.
#[inline]
#[must_use]
pub fn get_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
/// Read a little-endian `u64` at a known, in-bounds offset.
#[inline]
#[must_use]
pub fn get_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([
        b[at],
        b[at + 1],
        b[at + 2],
        b[at + 3],
        b[at + 4],
        b[at + 5],
        b[at + 6],
        b[at + 7],
    ])
}

/// A bounds-checked forward cursor over a byte buffer.
#[derive(Debug)]
pub struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Wrap a buffer.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Bytes not yet consumed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::format("read overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::format("unexpected end of section"));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// Read a single byte.
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    /// Read a little-endian `u16`.
    pub fn u16(&mut self) -> Result<u16> {
        Ok(get_u16(self.take(2)?, 0))
    }
    /// Read a little-endian `u32`.
    pub fn u32(&mut self) -> Result<u32> {
        Ok(get_u32(self.take(4)?, 0))
    }
    /// Read a little-endian `u64`.
    pub fn u64(&mut self) -> Result<u64> {
        Ok(get_u64(self.take(8)?, 0))
    }
    /// Read a little-endian `f32`.
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    /// Read `n` raw bytes (borrowed).
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }
    /// Read a `u32`-length-prefixed byte slice (borrowed).
    pub fn len_prefixed(&mut self) -> Result<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    /// Read a `u32`-length-prefixed UTF-8 string (borrowed).
    pub fn str(&mut self) -> Result<&'a str> {
        std::str::from_utf8(self.len_prefixed()?)
            .map_err(|_| Error::format("invalid UTF-8 in section"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_rounds_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    #[test]
    fn write_then_read_roundtrips() {
        let mut buf = Vec::new();
        put_u16(&mut buf, 0x1234);
        put_u32(&mut buf, 0xDEAD_BEEF);
        put_u64(&mut buf, 0x0102_0304_0506_0708);
        put_len_prefixed(&mut buf, b"hello");

        let mut r = Reader::new(&buf);
        assert_eq!(r.u16().unwrap(), 0x1234);
        assert_eq!(r.u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.u64().unwrap(), 0x0102_0304_0506_0708);
        assert_eq!(r.str().unwrap(), "hello");
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn reader_errs_on_truncation_not_panic() {
        let buf = [0u8; 3];
        let mut r = Reader::new(&buf);
        assert!(r.u32().is_err());
        let mut r2 = Reader::new(&buf);
        assert!(r2.len_prefixed().is_err());
    }

    #[test]
    fn reader_rejects_bad_utf8() {
        let mut buf = Vec::new();
        put_len_prefixed(&mut buf, &[0xff, 0xfe]);
        let mut r = Reader::new(&buf);
        assert!(r.str().is_err());
    }
}
