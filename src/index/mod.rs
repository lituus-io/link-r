// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The hybrid index: on-disk format, zero-copy store, retrieval, and incremental
//! upsert.
//!
//! Correctness-first: the default retrieval path is brute-force cosine over a flat
//! f32 blob fused with BM25 via Reciprocal Rank Fusion, with structured metadata
//! prefilters. Quantization and ANN are additive, gated tiers layered on once this
//! exact path is recall-validated.

pub mod builder;
pub mod bytesio;
pub mod dense;
pub mod format;
pub mod fuse;
pub mod graph;
pub mod meta;
pub mod mmap;
#[cfg(feature = "quant")]
pub mod quant;
pub mod sparse;
pub mod writer;

pub use builder::{Document, IndexBuilder, UpsertOutcome};
pub use meta::{DocMeta, Filter};

use crate::error::{Error, Result};
use crate::index::format::{flags, Header, IndexFile, IndexWriter, SectionKind};
use crate::index::graph::Graph;
use crate::index::mmap::MappedFile;
use crate::index::sparse::Bm25;
use crate::metric::Metric;
use crate::query::{Fusion, Hits, PreparedQuery};
use crate::resource::DocId;
use crate::url_key::UrlKey;
use std::collections::HashMap;
use std::path::Path;

/// Where the dense vector blob lives.
#[derive(Debug)]
enum Backing {
    /// Owned f32 vectors (in-memory build).
    Owned(Vec<f32>),
    /// A zero-copy range into a memory-mapped file.
    Mapped {
        file: MappedFile,
        dense_off: usize,
        dense_len: usize,
    },
}

/// A queryable hybrid index.
///
/// Open from disk zero-copy ([`Index::open`]) for resolving queries, or obtain one
/// in memory from an [`IndexBuilder`]. The dense blob stays zero-copy when mapped;
/// small metadata/BM25 structures are decoded into owned form on open.
#[derive(Debug)]
pub struct Index {
    backing: Backing,
    dim: usize,
    metric: Metric,
    embedder_id: u64,
    bm25: Bm25,
    meta: Vec<DocMeta>,
    /// Per-document outbound edges (canonical-URL keys), parallel to `meta`.
    edges: Vec<Vec<UrlKey>>,
    /// Resolved link graph (out/in adjacency) for `related` and graph-boosted search.
    graph: Graph,
    by_url: HashMap<UrlKey, DocId>,
}

impl Index {
    /// Every document's metadata, in internal id order.
    #[must_use]
    pub fn documents(&self) -> &[DocMeta] {
        &self.meta
    }

    /// Per-document outbound edges (canonical-URL keys), parallel to
    /// [`Index::documents`].
    #[must_use]
    pub fn edge_lists(&self) -> &[Vec<UrlKey>] {
        &self.edges
    }

    /// Construct an owned (in-memory) index from builder parts.
    #[must_use]
    #[allow(clippy::too_many_arguments)] // an internal constructor mirroring the index parts
    pub(crate) fn from_owned(
        dim: usize,
        metric: Metric,
        embedder_id: u64,
        bm25: Bm25,
        meta: Vec<DocMeta>,
        dense: Vec<f32>,
        edges: Vec<Vec<UrlKey>>,
        url_index: HashMap<UrlKey, DocId>,
    ) -> Self {
        let graph = Graph::resolve(&edges, &url_index);
        Self {
            backing: Backing::Owned(dense),
            dim,
            metric,
            embedder_id,
            bm25,
            meta,
            edges,
            graph,
            by_url: url_index,
        }
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.meta.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// The embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The distance metric.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The id of the embedder used to build this index (open-time compatibility).
    #[must_use]
    pub fn embedder_id(&self) -> u64 {
        self.embedder_id
    }

    /// Whether a URL (by its canonical key) is already indexed.
    #[must_use]
    pub fn contains(&self, key: UrlKey) -> bool {
        self.by_url.contains_key(&key)
    }

    /// The dense vector blob (`doc_count × dim`, row-major). Zero-copy when mapped.
    pub fn dense(&self) -> Result<&[f32]> {
        match &self.backing {
            Backing::Owned(v) => Ok(v),
            Backing::Mapped {
                file,
                dense_off,
                dense_len,
            } => mmap::cast_f32(&file.bytes()[*dense_off..*dense_off + *dense_len]),
        }
    }

    /// Open an index from disk, memory-mapped and fully validated.
    ///
    /// # Errors
    /// Returns [`Error::Format`] on a corrupt/incompatible file, or [`Error::Io`]
    /// if the file cannot be opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = MappedFile::open(path)?;
        let (dim, metric, embedder_id, dense_off, dense_len, meta, bm25, edges) = {
            let parsed = IndexFile::parse(file.bytes())?;
            let h = parsed.header;
            let dim = h.dim as usize;
            let doc_count = h.doc_count as usize;

            let (doff, dlen) = parsed
                .section_range(SectionKind::Dense)
                .ok_or_else(|| Error::format("missing dense section"))?;
            let expected = doc_count
                .checked_mul(dim)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| Error::format("dense size overflow"))?;
            if dlen != expected {
                return Err(Error::format("dense section size mismatch"));
            }

            let meta_bytes = parsed
                .section(SectionKind::DocMeta)?
                .ok_or_else(|| Error::format("missing metadata section"))?;
            let has_freshness = h.flags & flags::META_FRESHNESS != 0;
            let meta = meta::decode(&meta_bytes, has_freshness)?;
            if meta.len() != doc_count {
                return Err(Error::format("metadata doc-count mismatch"));
            }

            let bm25_bytes = parsed
                .section(SectionKind::Bm25)?
                .ok_or_else(|| Error::format("missing BM25 section"))?;
            let bm25 = Bm25::from_bytes(&bm25_bytes)?;

            // The link graph is optional (flag-gated); absent ⇒ empty edges.
            let edges = if h.flags & flags::LINK_GRAPH != 0 {
                let edge_bytes = parsed
                    .section(SectionKind::Edges)?
                    .ok_or_else(|| Error::format("LINK_GRAPH flag set but no Edges section"))?;
                let decoded = graph::decode(&edge_bytes)?;
                if decoded.len() != doc_count {
                    return Err(Error::format("edges doc-count mismatch"));
                }
                decoded
            } else {
                vec![Vec::new(); doc_count]
            };

            (
                dim,
                Metric::from_tag(h.metric),
                h.embedder_id,
                doff,
                dlen,
                meta,
                bm25,
                edges,
            )
        };

        let url_index: HashMap<UrlKey, DocId> = meta
            .iter()
            .enumerate()
            .map(|(i, d)| (d.url_key, i as DocId))
            .collect();
        let graph = Graph::resolve(&edges, &url_index);

        let index = Self {
            backing: Backing::Mapped {
                file,
                dense_off,
                dense_len,
            },
            dim,
            metric,
            embedder_id,
            bm25,
            meta,
            edges,
            graph,
            by_url: url_index,
        };
        // Validate the dense alignment now so later queries cannot fail on it.
        index.dense()?;
        Ok(index)
    }

    /// Serialize and atomically save to `path`.
    ///
    /// # Errors
    /// Returns [`Error::Io`] on a filesystem failure.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = serialize_sections(
            self.dim,
            self.metric,
            self.embedder_id,
            &self.meta,
            self.dense()?,
            &self.edges,
            &self.bm25,
        );
        writer::atomic_write(path, &bytes)
    }

    /// Convert into a mutable builder for incremental upserts.
    ///
    /// # Errors
    /// Returns [`Error::Format`] if a mapped dense blob cannot be read.
    pub fn into_builder(self) -> Result<IndexBuilder> {
        let doc_terms = self.bm25.to_doc_terms();
        let (k1, b) = (self.bm25.k1(), self.bm25.b());
        let dense = match self.backing {
            Backing::Owned(v) => v,
            Backing::Mapped {
                file,
                dense_off,
                dense_len,
            } => mmap::cast_f32(&file.bytes()[dense_off..dense_off + dense_len])?.to_vec(),
        };
        Ok(IndexBuilder::from_parts(
            self.dim,
            self.metric,
            self.embedder_id,
            k1,
            b,
            self.meta,
            dense,
            doc_terms,
            self.edges,
            self.by_url,
        ))
    }

    /// Run a hybrid search (dense + BM25, fused), returning ranked hits.
    ///
    /// # Errors
    /// Returns [`Error::DimMismatch`] if the query vector dimension is wrong, or
    /// [`Error::Format`] if a mapped dense blob cannot be read.
    pub fn search_prepared(&self, query: &PreparedQuery<'_>) -> Result<Hits<'_>> {
        if query.vector.len() != self.dim {
            return Err(Error::DimMismatch {
                index: self.dim,
                query: query.vector.len(),
            });
        }
        if self.meta.is_empty() {
            return Ok(Hits::new(&self.meta, Vec::new()));
        }
        let dense_blob = self.dense()?;
        let allowed = query.filter.evaluate(&self.meta);

        let candidates = query
            .limit
            .saturating_mul(query.rank.candidate_multiplier.max(1))
            .max(query.limit)
            .max(32);

        let dense_hits = dense::top_k(
            query.vector,
            dense_blob,
            self.dim,
            candidates,
            self.metric,
            allowed.as_ref(),
        );
        let sparse_hits = self.bm25.score(query.terms, allowed.as_ref(), candidates);

        let dense_ids: Vec<DocId> = dense_hits.iter().map(|(id, _)| *id).collect();
        let sparse_ids: Vec<DocId> = sparse_hits.iter().map(|(id, _)| *id).collect();

        // Fuse over the wider candidate pool so a graph boost can re-rank within it
        // before truncating to the requested limit.
        let fuse_limit = if query.rank.graph_boost > 0.0 {
            candidates
        } else {
            query.limit
        };
        let mut ranked = match query.rank.fusion {
            Fusion::Rrf { k } => fuse::reciprocal_rank_fusion(
                &[
                    fuse::RankList::weighted(&dense_ids, query.rank.dense_weight),
                    fuse::RankList::weighted(&sparse_ids, query.rank.sparse_weight),
                ],
                k,
                fuse_limit,
            ),
        };
        if query.rank.graph_boost > 0.0 {
            ranked = self.graph.boosted(&ranked, query.rank.graph_boost);
            ranked.truncate(query.limit);
        }
        Ok(Hits::new(&self.meta, ranked))
    }

    /// The `k` documents most related to `url` in the link graph — outbound targets
    /// and co-cited siblings — as ranked hits. Empty if the URL is not indexed or
    /// the index has no link graph.
    #[must_use]
    pub fn related(&self, url_key: UrlKey, k: usize) -> Hits<'_> {
        let Some(&doc) = self.by_url.get(&url_key) else {
            return Hits::new(&self.meta, Vec::new());
        };
        let ranked = self
            .graph
            .related(doc, k)
            .into_iter()
            .map(|(id, score)| (id, score as f32))
            .collect();
        Hits::new(&self.meta, ranked)
    }
}

/// Serialize index parts into the on-disk byte format.
///
/// The dense blob is stored raw (64-aligned for zero-copy mmap); the text-heavy
/// metadata section is zstd-compressed.
#[must_use]
pub(crate) fn serialize_sections(
    dim: usize,
    metric: Metric,
    embedder_id: u64,
    docs: &[DocMeta],
    dense: &[f32],
    edges: &[Vec<UrlKey>],
    bm25: &Bm25,
) -> Vec<u8> {
    let has_edges = edges.iter().any(|e| !e.is_empty());
    let header = Header {
        // Always written with the freshness columns; VECTORS_NORMALIZED tracks the
        // dense blob's normalization; LINK_GRAPH marks a present Edges section.
        flags: flags::META_FRESHNESS
            | if metric.normalizes() {
                flags::VECTORS_NORMALIZED
            } else {
                0
            }
            | if has_edges { flags::LINK_GRAPH } else { 0 },
        section_count: 0,
        dim: dim as u32,
        doc_count: docs.len() as u32,
        metric: metric.as_tag(),
        embedder_id,
        total_len: 0,
        bm25_k1_bits: bm25.k1().to_bits(),
        bm25_b_bits: bm25.b().to_bits(),
        avgdl_bits: bm25.avgdl().to_bits(),
    };

    let mut dense_bytes = Vec::with_capacity(std::mem::size_of_val(dense));
    for &x in dense {
        dense_bytes.extend_from_slice(&x.to_le_bytes());
    }

    let mut w = IndexWriter::new(header);
    w.add_section(
        SectionKind::DocMeta,
        format::sflags::ZSTD,
        format::zstd_section(&meta::encode(docs), 3),
    );
    w.add_section(SectionKind::Dense, 0, dense_bytes);
    w.add_section(SectionKind::Bm25, 0, bm25.to_bytes());
    if has_edges {
        w.add_section(
            SectionKind::Edges,
            format::sflags::ZSTD,
            format::zstd_section(&graph::encode(edges), 3),
        );
    }
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::builder::Document;
    use crate::query::{PreparedQuery, RankParams};
    use crate::resource::ResourceKind;
    use compact_str::CompactString;
    use smallvec::SmallVec;
    use url::Url;

    fn doc(url: &str, terms: &[&str], vec: Vec<f32>) -> Document {
        Document {
            url: Url::parse(url).unwrap(),
            kind: ResourceKind::Html,
            content_hash: u64::from(vec.len() as u32) ^ url.len() as u64,
            title: Some(CompactString::from("Title")),
            snippet: CompactString::from("snippet"),
            lang: Some(CompactString::from("en")),
            tags: SmallVec::new(),
            terms: terms.iter().map(|t| CompactString::from(*t)).collect(),
            vector: vec,
            edges: Vec::new(),
            fetched_at_ms: 0,
            etag: None,
            pinned: false,
        }
    }

    fn sample_index() -> Index {
        let mut b = IndexBuilder::new(3, Metric::Cosine, 42);
        b.upsert(doc(
            "https://x.dev/cat",
            &["cat", "feline"],
            vec![1.0, 0.0, 0.0],
        ))
        .unwrap();
        b.upsert(doc(
            "https://x.dev/dog",
            &["dog", "canine"],
            vec![0.0, 1.0, 0.0],
        ))
        .unwrap();
        b.upsert(doc(
            "https://x.dev/bird",
            &["bird", "avian"],
            vec![0.0, 0.0, 1.0],
        ))
        .unwrap();
        b.build()
    }

    fn prepared<'a>(
        vector: &'a [f32],
        terms: &'a [CompactString],
        filter: &'a Filter,
    ) -> PreparedQuery<'a> {
        PreparedQuery {
            vector,
            terms,
            filter,
            limit: 5,
            rank: RankParams::default(),
        }
    }

    #[test]
    fn dense_search_finds_nearest() {
        let idx = sample_index();
        let q = vec![0.9, 0.1, 0.0];
        let terms: Vec<CompactString> = Vec::new();
        let filter = Filter::All;
        let hits = idx.search_prepared(&prepared(&q, &terms, &filter)).unwrap();
        let top = hits.iter().next().unwrap();
        assert_eq!(top.url, "https://x.dev/cat");
    }

    #[test]
    fn bm25_term_pulls_exact_match() {
        let idx = sample_index();
        // a query vector pointing nowhere useful, but the term "canine" is exact.
        let q = vec![0.0, 0.0, 0.0];
        let terms = vec![CompactString::from("canine")];
        let filter = Filter::All;
        let hits = idx.search_prepared(&prepared(&q, &terms, &filter)).unwrap();
        assert_eq!(hits.iter().next().unwrap().url, "https://x.dev/dog");
    }

    #[test]
    fn save_open_roundtrip_preserves_results() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("idx.lnkr");
        let idx = sample_index();
        idx.save(&path).unwrap();

        let reopened = Index::open(&path).unwrap();
        assert_eq!(reopened.len(), 3);
        assert_eq!(reopened.dim(), 3);
        assert_eq!(reopened.embedder_id(), 42);

        let q = vec![0.1, 0.9, 0.0];
        let terms: Vec<CompactString> = Vec::new();
        let filter = Filter::All;
        let a = idx.search_prepared(&prepared(&q, &terms, &filter)).unwrap();
        let b = reopened
            .search_prepared(&prepared(&q, &terms, &filter))
            .unwrap();
        let a_urls: Vec<_> = a.iter().map(|h| h.url.to_owned()).collect();
        let b_urls: Vec<_> = b.iter().map(|h| h.url.to_owned()).collect();
        assert_eq!(a_urls, b_urls);
        assert_eq!(b_urls[0], "https://x.dev/dog");
    }

    #[test]
    fn reopened_dense_blob_is_zero_copy_and_correct() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("idx.lnkr");
        sample_index().save(&path).unwrap();
        let reopened = Index::open(&path).unwrap();
        let dense = reopened.dense().unwrap();
        assert_eq!(dense.len(), 3 * 3);
        // doc 0 was the unit-x vector after normalization.
        assert!((dense[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn incremental_update_via_into_builder() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("idx.lnkr");
        sample_index().save(&path).unwrap();

        let mut builder = Index::open(&path).unwrap().into_builder().unwrap();
        // re-adding an unchanged doc must dedup (same url + content hash).
        let outcome = builder
            .upsert(doc(
                "https://x.dev/cat",
                &["cat", "feline"],
                vec![1.0, 0.0, 0.0],
            ))
            .unwrap();
        assert_eq!(outcome, UpsertOutcome::Unchanged);
        // a genuinely new doc is added.
        builder
            .upsert(doc("https://x.dev/fish", &["fish"], vec![0.5, 0.5, 0.5]))
            .unwrap();
        assert_eq!(builder.len(), 4);
        builder.save(&path).unwrap();
        assert_eq!(Index::open(&path).unwrap().len(), 4);
    }

    #[test]
    fn search_rejects_wrong_query_dim() {
        let idx = sample_index();
        let q = vec![1.0, 0.0];
        let terms: Vec<CompactString> = Vec::new();
        let filter = Filter::All;
        let err = idx
            .search_prepared(&prepared(&q, &terms, &filter))
            .unwrap_err();
        assert!(matches!(err, Error::DimMismatch { index: 3, query: 2 }));
    }

    #[test]
    fn filtered_search_restricts_candidates() {
        let mut b = IndexBuilder::new(2, Metric::Cosine, 1);
        let mut d0 = doc("https://x.dev/a", &["x"], vec![1.0, 0.0]);
        d0.tags = SmallVec::from_iter([CompactString::from("keep")]);
        let d1 = doc("https://x.dev/b", &["x"], vec![1.0, 0.0]);
        b.upsert(d0).unwrap();
        b.upsert(d1).unwrap();
        let idx = b.build();

        let q = vec![1.0, 0.0];
        let terms: Vec<CompactString> = Vec::new();
        let filter = Filter::Tag(CompactString::from("keep"));
        let hits = idx.search_prepared(&prepared(&q, &terms, &filter)).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits.iter().next().unwrap().url, "https://x.dev/a");
    }
}
