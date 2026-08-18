// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The on-disk index format — the contract every other index module builds on.
//!
//! A hand-rolled, little-endian, section-based format (not rkyv/bincode): the
//! file *is* the index, sections are 64-byte aligned so the dense f32 blob can be
//! reinterpreted with zero per-element decode, and integrity is layered xxh3
//! (header, per-section, whole-file trailer). Wrong-format buffers fail at the
//! 4-byte magic before any checksum work; truncated/corrupt buffers fail with a
//! typed [`Error::Format`] — never a panic (asserted by the loader fuzz target).
//!
//! ```text
//! [Header 128 B, page-aligned] [SectionDirEntry × N, 32 B each]
//! [section bytes, each 64-aligned] ... [trailer 8 B = xxh3(file[0..len-8])]
//! ```

use crate::error::{Error, Result};
use crate::index::bytesio::{align_up, get_u16, get_u32, get_u64, put_u16, put_u32, put_u64};
use xxhash_rust::xxh3::xxh3_64;

/// Magic number: ASCII `"LNKR"` read as a little-endian `u32`.
pub const MAGIC: u32 = 0x524B_4E4C;
/// Current format version. Additive features are gated by header flags, not bumps.
pub const VERSION: u16 = 1;
/// Fixed header length in bytes.
pub const HEADER_LEN: usize = 128;
/// Section directory entry length in bytes.
pub const DIR_ENTRY_LEN: usize = 32;
/// Section alignment: 64 B covers a cache line and AVX-512 lanes.
pub const ALIGN: usize = 64;
/// Trailer length (whole-file checksum) in bytes.
pub const TRAILER_LEN: usize = 8;

/// Header flag bits.
pub mod flags {
    /// The dense vectors are L2-normalized (cosine ≡ dot).
    pub const VECTORS_NORMALIZED: u16 = 1 << 0;
    /// The `DocMeta` section carries the per-document freshness columns
    /// (`fetched_at_ms`, `pinned`, `etag`). Files written before this feature leave
    /// it clear and decode those columns as defaults (stale, unpinned, no etag).
    pub const META_FRESHNESS: u16 = 1 << 1;
    /// An [`Edges`](super::SectionKind::Edges) section is present (the
    /// persisted link graph).
    pub const LINK_GRAPH: u16 = 1 << 2;
}

/// The kind of a section. Stable on-disk discriminants; unknown kinds are ignored
/// on read. Discriminants 1 (`UrlKeys`), 2 (`Urls`), and 4 (`DocLengths`) were
/// reserved in early drafts but never written — they stay burned so any future
/// reader keeps skipping them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum SectionKind {
    /// Per-document metadata records (URL, kind, content hash, freshness, title,
    /// snippet, tags).
    DocMeta = 3,
    /// `f32 × doc_count × dim`, row-major, 64-aligned: the dense vector blob.
    Dense = 5,
    /// BM25 term dictionary + roaring postings + term-frequency blob.
    Bm25 = 6,
    /// Per-document outbound link graph (fixed-width u32 count + u64
    /// canonical-URL keys per document — see `index::graph::encode`).
    Edges = 7,
}

/// Section directory entry flag bits.
pub mod sflags {
    /// The section payload is zstd-compressed (first 8 bytes = uncompressed length).
    pub const ZSTD: u16 = 1 << 0;
}

/// The fixed-size file header (128 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Header flag bits (see [`flags`]).
    pub flags: u16,
    /// Number of sections.
    pub section_count: u32,
    /// Embedding dimension.
    pub dim: u32,
    /// Document count.
    pub doc_count: u32,
    /// Distance metric tag (0 = cosine).
    pub metric: u32,
    /// Stable identity of the embedder used to build the index (open-time compat).
    pub embedder_id: u64,
    /// Total file length in bytes (including trailer).
    pub total_len: u64,
    /// BM25 `k1` (bit pattern of the `f32`).
    pub bm25_k1_bits: u32,
    /// BM25 `b` (bit pattern of the `f32`).
    pub bm25_b_bits: u32,
    /// Average document length (bit pattern of the `f32`).
    pub avgdl_bits: u32,
}

impl Header {
    fn write_into(&self, buf: &mut Vec<u8>) {
        let start = buf.len();
        put_u32(buf, MAGIC);
        put_u16(buf, VERSION);
        put_u16(buf, self.flags);
        put_u32(buf, HEADER_LEN as u32);
        put_u32(buf, self.section_count);
        put_u32(buf, self.dim);
        put_u32(buf, self.doc_count);
        put_u32(buf, self.metric);
        put_u64(buf, self.embedder_id);
        put_u64(buf, HEADER_LEN as u64); // section_dir_off (constant)
        put_u64(buf, self.total_len);
        put_u32(buf, self.bm25_k1_bits);
        put_u32(buf, self.bm25_b_bits);
        put_u32(buf, self.avgdl_bits);
        // reserved [0u8; 56] (must remain zero)
        buf.resize(start + 120, 0);
        // header checksum over the first 120 bytes
        let csum = xxh3_64(&buf[start..start + 120]);
        put_u64(buf, csum);
        debug_assert_eq!(buf.len() - start, HEADER_LEN);
    }

    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(Error::format("file shorter than header"));
        }
        let magic = get_u32(bytes, 0);
        if magic != MAGIC {
            return Err(Error::format(format!(
                "bad magic: expected {MAGIC:#010x}, found {magic:#010x}"
            )));
        }
        let version = get_u16(bytes, 4);
        if version != VERSION {
            return Err(Error::format(format!(
                "unsupported version {version} (expected {VERSION})"
            )));
        }
        let stored_csum = get_u64(bytes, 120);
        let actual_csum = xxh3_64(&bytes[0..120]);
        if stored_csum != actual_csum {
            return Err(Error::format("header checksum mismatch"));
        }
        // reserved bytes must be zero.
        if bytes[64..120].iter().any(|&b| b != 0) {
            return Err(Error::format("reserved header bytes are non-zero"));
        }
        let header_len = get_u32(bytes, 8);
        if header_len as usize != HEADER_LEN {
            return Err(Error::format("unexpected header length"));
        }
        Ok(Self {
            flags: get_u16(bytes, 6),
            section_count: get_u32(bytes, 12),
            dim: get_u32(bytes, 16),
            doc_count: get_u32(bytes, 20),
            metric: get_u32(bytes, 24),
            embedder_id: get_u64(bytes, 28),
            // section_dir_off at 36 is constant (HEADER_LEN); not stored on the struct.
            total_len: get_u64(bytes, 44),
            bm25_k1_bits: get_u32(bytes, 52),
            bm25_b_bits: get_u32(bytes, 56),
            avgdl_bits: get_u32(bytes, 60),
        })
    }

    /// BM25 `k1` as a float.
    #[must_use]
    pub fn bm25_k1(&self) -> f32 {
        f32::from_bits(self.bm25_k1_bits)
    }
    /// BM25 `b` as a float.
    #[must_use]
    pub fn bm25_b(&self) -> f32 {
        f32::from_bits(self.bm25_b_bits)
    }
    /// Average document length as a float.
    #[must_use]
    pub fn avgdl(&self) -> f32 {
        f32::from_bits(self.avgdl_bits)
    }
}

/// One section's directory entry.
#[derive(Clone, Copy, Debug)]
struct DirEntry {
    kind: u16,
    flags: u16,
    offset: u64,
    length: u64,
    xxh3: u64,
}

/// A section staged for writing.
#[derive(Debug)]
struct PendingSection {
    kind: SectionKind,
    flags: u16,
    bytes: Vec<u8>,
}

/// Accumulates sections and serializes a complete, integrity-checked index file.
#[derive(Debug)]
pub struct IndexWriter {
    header: Header,
    sections: Vec<PendingSection>,
}

impl IndexWriter {
    /// Start a writer for an index with the given header fields. `section_count`,
    /// `total_len`, and checksums are filled in by [`IndexWriter::finish`].
    #[must_use]
    pub fn new(header: Header) -> Self {
        Self {
            header,
            sections: Vec::new(),
        }
    }

    /// Add a raw section. `bytes` is stored verbatim (caller pre-compresses if it
    /// sets [`sflags::ZSTD`]).
    pub fn add_section(&mut self, kind: SectionKind, flags: u16, bytes: Vec<u8>) {
        self.sections.push(PendingSection { kind, flags, bytes });
    }

    /// Serialize the complete file (header + directory + 64-aligned sections + trailer).
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        let count = self.sections.len();
        self.header.section_count = count as u32;

        // Lay out section offsets: directory ends, then each section 64-aligned.
        let dir_end = HEADER_LEN + count * DIR_ENTRY_LEN;
        let mut cursor = align_up(dir_end, ALIGN);
        let mut entries: Vec<DirEntry> = Vec::with_capacity(count);
        for s in &self.sections {
            let offset = cursor;
            let length = s.bytes.len();
            entries.push(DirEntry {
                kind: s.kind as u16,
                flags: s.flags,
                offset: offset as u64,
                length: length as u64,
                xxh3: xxh3_64(&s.bytes),
            });
            cursor = align_up(offset + length, ALIGN);
        }
        let total_len = cursor + TRAILER_LEN;
        self.header.total_len = total_len as u64;

        let mut buf = Vec::with_capacity(total_len);
        self.header.write_into(&mut buf);
        for e in &entries {
            put_u16(&mut buf, e.kind);
            put_u16(&mut buf, e.flags);
            put_u32(&mut buf, 0); // reserved
            put_u64(&mut buf, e.offset);
            put_u64(&mut buf, e.length);
            put_u64(&mut buf, e.xxh3);
        }
        for (e, s) in entries.iter().zip(&self.sections) {
            buf.resize(e.offset as usize, 0); // pad to alignment
            buf.extend_from_slice(&s.bytes);
        }
        buf.resize(total_len - TRAILER_LEN, 0); // pad to final section's alignment
        let trailer = xxh3_64(&buf);
        put_u64(&mut buf, trailer);
        debug_assert_eq!(buf.len(), total_len);
        buf
    }
}

/// A parsed, validated view over an index file's bytes. Borrows for the file's
/// lifetime — section access is zero-copy (except zstd sections, decompressed on
/// demand).
#[derive(Debug)]
pub struct IndexFile<'a> {
    /// The validated header.
    pub header: Header,
    bytes: &'a [u8],
    dir: Vec<DirEntry>,
}

impl<'a> IndexFile<'a> {
    /// Parse and fully validate an index from raw bytes.
    ///
    /// Validates magic, version, header checksum, reserved-zero, the section
    /// directory (bounds + 64-alignment), every per-section checksum, and the
    /// whole-file trailer. Any failure is a typed [`Error::Format`].
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let header = Header::parse(bytes)?;
        let total = header.total_len as usize;
        if bytes.len() != total {
            return Err(Error::format(format!(
                "length mismatch: header says {total}, buffer is {}",
                bytes.len()
            )));
        }
        if total < HEADER_LEN + TRAILER_LEN {
            return Err(Error::format("file too small for trailer"));
        }
        // whole-file trailer.
        let trailer = get_u64(bytes, total - TRAILER_LEN);
        if trailer != xxh3_64(&bytes[..total - TRAILER_LEN]) {
            return Err(Error::format("whole-file checksum mismatch"));
        }

        let count = header.section_count as usize;
        let dir_end = HEADER_LEN + count * DIR_ENTRY_LEN;
        if dir_end > total - TRAILER_LEN {
            return Err(Error::format("section directory overflows file"));
        }
        let mut dir = Vec::with_capacity(count);
        for i in 0..count {
            let base = HEADER_LEN + i * DIR_ENTRY_LEN;
            let kind = get_u16(bytes, base);
            let flags = get_u16(bytes, base + 2);
            let offset = get_u64(bytes, base + 8);
            let length = get_u64(bytes, base + 16);
            let stored = get_u64(bytes, base + 24);
            let off = usize::try_from(offset).map_err(|_| Error::format("offset too large"))?;
            let len = usize::try_from(length).map_err(|_| Error::format("length too large"))?;
            let end = off
                .checked_add(len)
                .ok_or_else(|| Error::format("section offset+len overflow"))?;
            if end > total - TRAILER_LEN {
                return Err(Error::format("section overflows file"));
            }
            if off % ALIGN != 0 {
                return Err(Error::format("section not 64-byte aligned"));
            }
            if xxh3_64(&bytes[off..end]) != stored {
                return Err(Error::format("section checksum mismatch"));
            }
            dir.push(DirEntry {
                kind,
                flags,
                offset,
                length,
                xxh3: stored,
            });
        }
        Ok(Self { header, bytes, dir })
    }

    /// The `(offset, length)` of a section within the file, if present. Used to
    /// retain a zero-copy range (e.g. the dense blob) without holding the borrow.
    #[must_use]
    pub fn section_range(&self, kind: SectionKind) -> Option<(usize, usize)> {
        let want = kind as u16;
        self.dir
            .iter()
            .find(|e| e.kind == want)
            .map(|e| (e.offset as usize, e.length as usize))
    }

    /// The raw (still-compressed if zstd) bytes of a section, if present.
    fn raw_section(&self, kind: SectionKind) -> Option<(&'a [u8], u16)> {
        let want = kind as u16;
        self.dir.iter().find(|e| e.kind == want).map(|e| {
            let off = e.offset as usize;
            let end = off + e.length as usize;
            (&self.bytes[off..end], e.flags)
        })
    }

    /// A section's bytes, decompressing if it carries [`sflags::ZSTD`].
    ///
    /// Non-compressed sections return `Cow::Borrowed` (zero-copy); the dense vector
    /// section is never compressed, so it always borrows.
    pub fn section(&self, kind: SectionKind) -> Result<Option<std::borrow::Cow<'a, [u8]>>> {
        let Some((raw, flags)) = self.raw_section(kind) else {
            return Ok(None);
        };
        if flags & sflags::ZSTD != 0 {
            if raw.len() < 8 {
                return Err(Error::format("zstd section missing length prefix"));
            }
            let compressed = &raw[8..];
            let capacity = decompressed_capacity(get_u64(raw, 0), compressed.len())?;
            let decoded = zstd::bulk::decompress(compressed, capacity)
                .map_err(|e| Error::format(format!("zstd decode: {e}")))?;
            Ok(Some(std::borrow::Cow::Owned(decoded)))
        } else {
            Ok(Some(std::borrow::Cow::Borrowed(raw)))
        }
    }

    /// Borrow a section that must be uncompressed (e.g. the dense blob); errors if
    /// it is zstd-flagged.
    pub fn section_raw(&self, kind: SectionKind) -> Result<Option<&'a [u8]>> {
        match self.raw_section(kind) {
            None => Ok(None),
            Some((raw, flags)) if flags & sflags::ZSTD == 0 => Ok(Some(raw)),
            Some(_) => Err(Error::format("expected uncompressed section")),
        }
    }
}

/// Absolute ceiling on one decompressed section.
const MAX_DECOMPRESSED: u64 = 512 * 1024 * 1024;
/// Largest expansion we will honour from a section's declared length. Real
/// metadata and edge sections sit far below this; the bound exists only to make
/// a small hostile payload unable to name a large allocation.
const MAX_ZSTD_RATIO: u64 = 1024;

/// Decide how many bytes a zstd section may decompress into.
///
/// `zstd::bulk::decompress` pre-allocates exactly the capacity it is given, so
/// passing the file's declared length through unchecked lets a few crafted bytes
/// name a multi-gigabyte allocation — the same class of defect that fuzzing
/// already found in the metadata and BM25 decoders, which cap every count by the
/// bytes actually present. This applies the equivalent bound here: the declared
/// length must fit in `usize`, stay under [`MAX_DECOMPRESSED`], and stay within
/// [`MAX_ZSTD_RATIO`] of the compressed payload it claims to come from.
///
/// The capacity is a ceiling, not a promise: if the payload decodes to fewer
/// bytes that is fine, and if it decodes to more, zstd itself errors.
fn decompressed_capacity(declared: u64, compressed_len: usize) -> Result<usize> {
    let ratio_bound = (compressed_len as u64).saturating_mul(MAX_ZSTD_RATIO);
    let bound = ratio_bound.min(MAX_DECOMPRESSED);
    if declared > bound {
        return Err(Error::format(format!(
            "zstd section declares {declared} bytes from {compressed_len} compressed \
             (bound {bound}); refusing to allocate"
        )));
    }
    usize::try_from(declared).map_err(|_| Error::format("zstd section length overflows usize"))
}

/// Validate that an index can be parsed from `bytes` without retaining the view.
/// Used by the fuzz target and quick integrity checks.
pub fn validate(bytes: &[u8]) -> Result<()> {
    IndexFile::parse(bytes).map(|_| ())
}

/// Compress `data` for storage as a zstd section payload (length-prefixed).
#[must_use]
pub fn zstd_section(data: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 8);
    put_u64(&mut out, data.len() as u64);
    let compressed = zstd::bulk::compress(data, level).expect("zstd compress is infallible here");
    out.extend_from_slice(&compressed);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> Header {
        Header {
            flags: flags::VECTORS_NORMALIZED,
            section_count: 0,
            dim: 8,
            doc_count: 3,
            metric: 0,
            embedder_id: 0xDEAD_BEEF,
            total_len: 0,
            bm25_k1_bits: 1.2f32.to_bits(),
            bm25_b_bits: 0.75f32.to_bits(),
            avgdl_bits: 12.5f32.to_bits(),
        }
    }

    fn build_sample() -> Vec<u8> {
        let mut w = IndexWriter::new(sample_header());
        w.add_section(SectionKind::Dense, 0, vec![0u8; 96]); // 3*8 f32
        w.add_section(SectionKind::Edges, 0, b"edge-bytes".to_vec());
        w.add_section(
            SectionKind::DocMeta,
            sflags::ZSTD,
            zstd_section(b"hello hello hello", 3),
        );
        w.finish()
    }

    #[test]
    fn roundtrips_header_and_sections() {
        let bytes = build_sample();
        let file = IndexFile::parse(&bytes).unwrap();
        assert_eq!(file.header.dim, 8);
        assert_eq!(file.header.doc_count, 3);
        assert_eq!(file.header.embedder_id, 0xDEAD_BEEF);
        assert!((file.header.bm25_k1() - 1.2).abs() < 1e-6);
        assert_eq!(
            file.header.flags & flags::VECTORS_NORMALIZED,
            flags::VECTORS_NORMALIZED
        );

        let dense = file.section_raw(SectionKind::Dense).unwrap().unwrap();
        assert_eq!(dense.len(), 96);
        let edges = file.section(SectionKind::Edges).unwrap().unwrap();
        assert_eq!(&*edges, b"edge-bytes");
        let meta = file.section(SectionKind::DocMeta).unwrap().unwrap();
        assert_eq!(&*meta, b"hello hello hello");
    }

    #[test]
    fn dense_section_is_64_aligned_in_file() {
        let bytes = build_sample();
        let file = IndexFile::parse(&bytes).unwrap();
        // find the dense entry offset
        let dense_off = file
            .dir
            .iter()
            .find(|e| e.kind == SectionKind::Dense as u16)
            .unwrap()
            .offset as usize;
        assert_eq!(dense_off % ALIGN, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = build_sample();
        bytes[0] ^= 0xFF;
        assert!(IndexFile::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_truncation() {
        let bytes = build_sample();
        for cut in [HEADER_LEN, bytes.len() / 2, bytes.len() - 1] {
            assert!(IndexFile::parse(&bytes[..cut]).is_err());
        }
    }

    #[test]
    fn rejects_flipped_section_byte() {
        let mut bytes = build_sample();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x01;
        assert!(IndexFile::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_nonzero_reserved() {
        let mut bytes = build_sample();
        bytes[70] = 7; // inside reserved [64..120)
        assert!(IndexFile::parse(&bytes).is_err());
    }

    #[test]
    fn validate_never_panics_on_garbage() {
        // small and large arbitrary buffers must error, not panic.
        for len in [0usize, 1, 4, 16, 127, 128, 200, 1024] {
            let garbage: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
            let _ = validate(&garbage);
        }
    }

    #[test]
    fn missing_section_is_none() {
        let bytes = build_sample();
        let file = IndexFile::parse(&bytes).unwrap();
        assert!(file.section(SectionKind::Bm25).unwrap().is_none());
    }

    /// A zstd section's declared uncompressed length is attacker-controlled, and
    /// `zstd::bulk::decompress` allocates exactly the capacity it is handed. The
    /// bound must reject a decompression bomb before any allocation happens.
    #[test]
    fn a_declared_length_cannot_name_an_unbounded_allocation() {
        // 16 compressed bytes may not claim to expand to 16 GiB…
        assert!(decompressed_capacity(16 * 1024 * 1024 * 1024, 16).is_err());
        // …nor to u64::MAX, which would also overflow usize on 32-bit.
        assert!(decompressed_capacity(u64::MAX, 16).is_err());
        // …nor past the absolute ceiling, however large the payload.
        assert!(decompressed_capacity(MAX_DECOMPRESSED + 1, 10 * 1024 * 1024).is_err());
    }

    /// The bound must leave real sections plenty of headroom: metadata and edge
    /// sections compress well, and refusing a legitimate file would be worse
    /// than the bomb it prevents.
    #[test]
    fn the_bound_admits_realistic_compression_ratios() {
        // A 10 KiB payload expanding 100x is ordinary for repetitive metadata.
        assert_eq!(
            decompressed_capacity(1024 * 1024, 10 * 1024).unwrap(),
            1024 * 1024
        );
        // Exactly at the ratio bound is allowed; one byte past it is not.
        assert!(decompressed_capacity(1024 * MAX_ZSTD_RATIO, 1024).is_ok());
        assert!(decompressed_capacity(1024 * MAX_ZSTD_RATIO + 1, 1024).is_err());
        // An empty payload can only claim zero.
        assert!(decompressed_capacity(0, 0).is_ok());
        assert!(decompressed_capacity(1, 0).is_err());
    }

    /// The guard must not have broken the real read path.
    #[test]
    fn a_genuinely_compressed_section_still_round_trips() {
        let payload: Vec<u8> = (0..8192u32).flat_map(|i| (i % 251).to_le_bytes()).collect();
        let section = zstd_section(&payload, 3);
        assert!(
            section.len() < payload.len(),
            "payload should actually compress"
        );

        let mut w = IndexWriter::new(sample_header());
        w.add_section(SectionKind::DocMeta, sflags::ZSTD, section);
        let bytes = w.finish();

        let file = IndexFile::parse(&bytes).unwrap();
        let got = file.section(SectionKind::DocMeta).unwrap().unwrap();
        assert_eq!(&*got, &payload[..]);
    }
}
