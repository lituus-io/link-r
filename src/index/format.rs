// Author: support@lituus.io
// Copyright: lituus-io, all rights reserved
//
//! The on-disk index format — the contract every other index module builds on.
//!
//! The file is a [`graph_r::container`]: a 128-byte checksummed header whose
//! owner area carries the index fields below, a section directory, and
//! 64-aligned sections (so the dense `f32` blob is reinterpreted in place),
//! each checksummed once. Wrong-format buffers fail at the magic before any
//! checksum work; truncated or corrupt buffers fail with a typed
//! [`Error::Format`] — never a panic (asserted by the loader fuzz target).
//!
//! ```text
//! owner area: dim u32 | doc_count u32 | metric u32 | embedder_id u64
//!             | bm25_k1 f32 | bm25_b f32 | avgdl f32   (rest zero)
//! ```

use crate::error::{Error, Result};
use graph_r::bytesio::{get_u32, get_u64, put_f32, put_u32, put_u64};
use graph_r::container::{self, Dir, Format, OWNER_LEN};

/// Magic number: ASCII `"LNKR"` read as a little-endian `u32`.
pub const MAGIC: u32 = 0x524B_4E4C;
/// Current format version (2: the graph-r container).
pub const VERSION: u16 = 2;
/// Section alignment: 64 B covers a cache line and AVX-512 lanes.
pub const ALIGN: usize = container::ALIGN;

const FORMAT: Format = Format {
    magic: MAGIC,
    version: VERSION,
};
const OWNER_USED: usize = 32;

/// Header flag bits.
pub mod flags {
    /// The dense vectors are L2-normalized (cosine ≡ dot).
    pub const VECTORS_NORMALIZED: u16 = 1 << 0;
    /// The `DocMeta` section carries the per-document freshness columns
    /// (`fetched_at_ms`, `pinned`, `etag`).
    pub const META_FRESHNESS: u16 = 1 << 1;
    /// An [`Edges`](super::SectionKind::Edges) section is present (the
    /// persisted link graph).
    pub const LINK_GRAPH: u16 = 1 << 2;
}

/// The kind of a section. Stable on-disk discriminants; unknown kinds are
/// ignored on read.
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

/// Section flag bits.
pub mod sflags {
    /// The section payload is zstd-compressed (first 8 bytes = uncompressed length).
    pub const ZSTD: u16 = 1 << 0;
}

/// The index fields of the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Header flag bits (see [`flags`]).
    pub flags: u16,
    /// Embedding dimension.
    pub dim: u32,
    /// Number of documents.
    pub doc_count: u32,
    /// Distance metric tag.
    pub metric: u32,
    /// The embedder's id (open-time compatibility).
    pub embedder_id: u64,
    /// BM25 `k1`, as bits.
    pub bm25_k1_bits: u32,
    /// BM25 `b`, as bits.
    pub bm25_b_bits: u32,
    /// Average document length, as bits.
    pub avgdl_bits: u32,
}

impl Header {
    fn owner(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(OWNER_USED);
        put_u32(&mut o, self.dim);
        put_u32(&mut o, self.doc_count);
        put_u32(&mut o, self.metric);
        put_u64(&mut o, self.embedder_id);
        put_f32(&mut o, f32::from_bits(self.bm25_k1_bits));
        put_f32(&mut o, f32::from_bits(self.bm25_b_bits));
        put_f32(&mut o, f32::from_bits(self.avgdl_bits));
        o
    }

    fn from_owner(flags: u16, o: &[u8; OWNER_LEN]) -> Result<Self> {
        if o[OWNER_USED..].iter().any(|&b| b != 0) {
            return Err(Error::format("reserved header bytes are non-zero"));
        }
        Ok(Self {
            flags,
            dim: get_u32(o, 0),
            doc_count: get_u32(o, 4),
            metric: get_u32(o, 8),
            embedder_id: get_u64(o, 12),
            bm25_k1_bits: get_u32(o, 20),
            bm25_b_bits: get_u32(o, 24),
            avgdl_bits: get_u32(o, 28),
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

/// Accumulates sections and serializes a complete, integrity-checked index file.
#[derive(Debug)]
pub struct IndexWriter {
    inner: container::Writer,
}

impl IndexWriter {
    /// Start a writer for an index with the given header fields.
    #[must_use]
    pub fn new(header: Header) -> Self {
        let mut inner = container::Writer::new(FORMAT);
        inner.flags(header.flags).owner(&header.owner());
        Self { inner }
    }

    /// Add a section. `bytes` is stored verbatim (the caller pre-compresses
    /// when it sets [`sflags::ZSTD`]); an empty section is omitted and reads
    /// back as absent.
    pub fn add_section(&mut self, kind: SectionKind, flags: u16, bytes: Vec<u8>) {
        self.inner.section_flagged(kind as u16, flags, bytes);
    }

    /// Serialize the complete file.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.inner.finish()
    }
}

/// A parsed, validated view over an index file's bytes. Section access is
/// zero-copy (except zstd sections, decompressed on demand).
#[derive(Debug)]
pub struct IndexFile<'a> {
    /// The validated header.
    pub header: Header,
    bytes: &'a [u8],
    dir: Dir,
}

impl<'a> IndexFile<'a> {
    /// Parse and fully validate an index from raw bytes: header, directory
    /// and every section checksummed once, bounds, alignment and zero
    /// padding checked. Any failure is a typed [`Error::Format`].
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let dir = Dir::parse(bytes, FORMAT)?;
        let header = Header::from_owner(dir.flags, dir.owner())?;
        Ok(Self { header, bytes, dir })
    }

    /// The `(offset, length)` of a section within the file, if present. Used
    /// to retain a zero-copy range (the dense blob) without holding the borrow.
    #[must_use]
    pub fn section_range(&self, kind: SectionKind) -> Option<(usize, usize)> {
        let k = kind as u16;
        self.dir.has(k).then(|| {
            let r = self.dir.span(k);
            (r.start, r.len())
        })
    }

    /// A section's bytes, decompressing if it carries [`sflags::ZSTD`].
    /// Non-compressed sections borrow (zero-copy).
    pub fn section(&self, kind: SectionKind) -> Result<Option<std::borrow::Cow<'a, [u8]>>> {
        let k = kind as u16;
        if !self.dir.has(k) {
            return Ok(None);
        }
        let raw = self.dir.section(self.bytes, k);
        if self.dir.section_flags(k) & sflags::ZSTD == 0 {
            return Ok(Some(std::borrow::Cow::Borrowed(raw)));
        }
        if raw.len() < 8 {
            return Err(Error::format("zstd section missing length prefix"));
        }
        let compressed = &raw[8..];
        let capacity = decompressed_capacity(get_u64(raw, 0), compressed.len())?;
        let decoded = zstd::bulk::decompress(compressed, capacity)
            .map_err(|e| Error::format(format!("zstd decode: {e}")))?;
        Ok(Some(std::borrow::Cow::Owned(decoded)))
    }

    /// Borrow a section that must be uncompressed (the dense blob); errors if
    /// it is zstd-flagged.
    pub fn section_raw(&self, kind: SectionKind) -> Result<Option<&'a [u8]>> {
        let k = kind as u16;
        if !self.dir.has(k) {
            return Ok(None);
        }
        if self.dir.section_flags(k) & sflags::ZSTD != 0 {
            return Err(Error::format("expected uncompressed section"));
        }
        Ok(Some(self.dir.section(self.bytes, k)))
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
            dim: 8,
            doc_count: 3,
            metric: 0,
            embedder_id: 0xDEAD_BEEF,
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
        let (dense_off, _) = file.section_range(SectionKind::Dense).unwrap();
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
        for cut in [container::HEADER_LEN, bytes.len() / 2, bytes.len() - 1] {
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
        // Inside the owner area's reserved tail: rejected (the header
        // checksum is recomputed so only the reserved-zero rule can fire).
        let at = 32 + OWNER_USED + 4;
        bytes[at] = 7;
        let sum = graph_r::hash::xxh3_64(&bytes[..120]);
        bytes[120..128].copy_from_slice(&sum.to_le_bytes());
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
