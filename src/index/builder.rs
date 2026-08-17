// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The owned, mutable index builder: incremental upsert keyed by canonical URL.
//!
//! Every link is unique in the index (`UrlKey`), so re-crawling a page updates its
//! entry in place instead of duplicating it; an unchanged page (matching
//! content-hash) is skipped entirely. `build` produces an in-memory queryable
//! [`Index`]; `save` serializes atomically to disk.

use crate::error::{Error, Result};
use crate::index::meta::DocMeta;
use crate::index::sparse::{Bm25, DEFAULT_B, DEFAULT_K1};
use crate::index::{dense, serialize_sections, writer, Index};
use crate::metric::Metric;
use crate::resource::{DocId, ResourceKind};
use crate::url_key::UrlKey;
use compact_str::CompactString;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::path::Path;
use url::Url;

/// A document to add to or update in the index. Produced by the extraction +
/// embedding stage; the index stores only this distilled form, never the body.
#[derive(Clone, Debug)]
pub struct Document {
    /// The canonical remote URL (the value the index resolves to).
    pub url: Url,
    /// Content kind.
    pub kind: ResourceKind,
    /// `xxh3` of the source body, for change detection on re-crawl.
    pub content_hash: u64,
    /// Display title, if extracted.
    pub title: Option<CompactString>,
    /// Short display snippet.
    pub snippet: CompactString,
    /// Detected language tag, if any.
    pub lang: Option<CompactString>,
    /// Structured tags for filtering.
    pub tags: SmallVec<[CompactString; 4]>,
    /// Normalized BM25 terms (also defines the document length).
    pub terms: Vec<CompactString>,
    /// The embedding vector (length must equal the index dimension).
    pub vector: Vec<f32>,
    /// Outbound link targets (canonical-URL keys) — the knowledge-graph edges.
    pub edges: Vec<UrlKey>,
    /// Wall-clock ms since the Unix epoch when the body was fetched (`0` = unknown).
    pub fetched_at_ms: u64,
    /// The source `ETag`, for conditional refresh.
    pub etag: Option<CompactString>,
    /// Whether to pin this document (exempt from TTL / age eviction).
    pub pinned: bool,
}

/// The result of an [`IndexBuilder::upsert`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpsertOutcome {
    /// A new link was added.
    Added,
    /// An existing link's content changed and was re-indexed.
    Updated,
    /// An existing link was unchanged (same content hash) and skipped.
    Unchanged,
}

/// Accumulates documents into an index, deduplicating by canonical URL.
#[derive(Debug)]
pub struct IndexBuilder {
    dim: usize,
    metric: Metric,
    embedder_id: u64,
    k1: f32,
    b: f32,
    meta: Vec<DocMeta>,
    dense: Vec<f32>,
    doc_terms: Vec<Vec<CompactString>>,
    /// Per-document outbound edges (canonical-URL keys), parallel to `meta`.
    edges: Vec<Vec<UrlKey>>,
    url_index: HashMap<UrlKey, DocId>,
}

impl IndexBuilder {
    /// Create an empty builder for `dim`-dimensional vectors.
    #[must_use]
    pub fn new(dim: usize, metric: Metric, embedder_id: u64) -> Self {
        Self {
            dim,
            metric,
            embedder_id,
            k1: DEFAULT_K1,
            b: DEFAULT_B,
            meta: Vec::new(),
            dense: Vec::new(),
            doc_terms: Vec::new(),
            edges: Vec::new(),
            url_index: HashMap::new(),
        }
    }

    /// Override the BM25 parameters.
    #[must_use]
    pub fn with_bm25(mut self, k1: f32, b: f32) -> Self {
        self.k1 = k1;
        self.b = b;
        self
    }

    /// Reconstruct a builder from the parts of an existing index (for the
    /// incremental update path).
    #[must_use]
    #[allow(clippy::too_many_arguments)] // an internal constructor mirroring the index parts
    pub(crate) fn from_parts(
        dim: usize,
        metric: Metric,
        embedder_id: u64,
        k1: f32,
        b: f32,
        meta: Vec<DocMeta>,
        dense: Vec<f32>,
        doc_terms: Vec<Vec<CompactString>>,
        edges: Vec<Vec<UrlKey>>,
        url_index: HashMap<UrlKey, DocId>,
    ) -> Self {
        Self {
            dim,
            metric,
            embedder_id,
            k1,
            b,
            meta,
            dense,
            doc_terms,
            edges,
            url_index,
        }
    }

    /// Number of documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.meta.len()
    }

    /// Whether the builder is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// The embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The id of the embedder this index was built with (open-time compatibility).
    #[must_use]
    pub fn embedder_id(&self) -> u64 {
        self.embedder_id
    }

    /// Whether `url` is already indexed with the same `content_hash` (so a
    /// re-crawl can skip re-embedding it).
    #[must_use]
    pub fn is_unchanged(&self, url: &url::Url, content_hash: u64) -> bool {
        let key = UrlKey::from_url(url);
        self.url_index
            .get(&key)
            .is_some_and(|&id| self.meta[id as usize].content_hash == content_hash)
    }

    /// Build a queryable in-memory [`Index`] without consuming the builder (clones
    /// the corpus; cheap at the crate's target scale).
    #[must_use]
    pub fn to_index(&self) -> Index {
        let bm25 = crate::index::sparse::Bm25::build(&self.doc_terms, self.k1, self.b);
        Index::from_owned(
            self.dim,
            self.metric,
            self.embedder_id,
            bm25,
            self.meta.clone(),
            self.dense.clone(),
            self.edges.clone(),
            self.url_index.clone(),
        )
    }

    /// Add or update a document, deduplicated by canonical URL.
    ///
    /// # Errors
    /// Returns [`Error::DimMismatch`] if the vector dimension is wrong.
    pub fn upsert(&mut self, doc: Document) -> Result<UpsertOutcome> {
        if doc.vector.len() != self.dim {
            return Err(Error::DimMismatch {
                index: self.dim,
                query: doc.vector.len(),
            });
        }
        let url_key = UrlKey::from_url(&doc.url);
        let doc_len = doc.terms.len() as u32;

        let mut vector = doc.vector;
        if self.metric.normalizes() {
            dense::l2_normalize(&mut vector);
        }
        let mut record = DocMeta {
            // Store the canonical form (drops userinfo/fragment, sorts query) so a
            // credential in the URL is never persisted and the display URL matches
            // the dedup key.
            url: crate::url_key::canonicalize(&doc.url),
            url_key,
            kind: doc.kind,
            content_hash: doc.content_hash,
            doc_len,
            title: doc.title,
            snippet: doc.snippet,
            lang: doc.lang,
            tags: doc.tags,
            fetched_at_ms: doc.fetched_at_ms,
            pinned: doc.pinned,
            etag: doc.etag,
        };

        if let Some(&id) = self.url_index.get(&url_key) {
            let id_usize = id as usize;
            if self.meta[id_usize].content_hash == doc.content_hash {
                return Ok(UpsertOutcome::Unchanged);
            }
            // A pin is a retention decision that must survive re-crawls.
            record.pinned = record.pinned || self.meta[id_usize].pinned;
            self.meta[id_usize] = record;
            let start = id_usize * self.dim;
            self.dense[start..start + self.dim].copy_from_slice(&vector);
            self.doc_terms[id_usize] = doc.terms;
            self.edges[id_usize] = doc.edges;
            Ok(UpsertOutcome::Updated)
        } else {
            let id = self.meta.len() as DocId;
            self.url_index.insert(url_key, id);
            self.meta.push(record);
            self.dense.extend_from_slice(&vector);
            self.doc_terms.push(doc.terms);
            self.edges.push(doc.edges);
            Ok(UpsertOutcome::Added)
        }
    }

    /// Remove a document by canonical URL. Returns whether it was present.
    ///
    /// O(1): swap-removes the metadata, dense row, and terms (moving the last doc
    /// into the hole), keeping the URL→id map consistent. Internal ids are not
    /// stable across removals, which is why the persisted link graph keys edges by
    /// [`UrlKey`], not id.
    pub fn remove(&mut self, url: &Url) -> bool {
        let key = UrlKey::from_url(url);
        let Some(removed_id) = self.url_index.remove(&key) else {
            return false;
        };
        let id = removed_id as usize;
        let last = self.meta.len() - 1;
        self.meta.swap_remove(id);
        self.doc_terms.swap_remove(id);
        self.edges.swap_remove(id);
        let dim = self.dim;
        if id != last {
            // Move the last dense row into the freed slot, then fix its id mapping.
            let (head, tail) = self.dense.split_at_mut(last * dim);
            head[id * dim..id * dim + dim].copy_from_slice(&tail[..dim]);
            let moved_key = self.meta[id].url_key;
            self.url_index.insert(moved_key, id as DocId);
        }
        self.dense.truncate(last * dim);
        true
    }

    /// Update only the fetch timestamp of an existing document (e.g. after a `304
    /// Not Modified`). Returns whether the document was present.
    pub fn touch(&mut self, url: &Url, now_ms: u64) -> bool {
        let key = UrlKey::from_url(url);
        match self.url_index.get(&key) {
            Some(&id) => {
                self.meta[id as usize].fetched_at_ms = now_ms;
                true
            }
            None => false,
        }
    }

    /// Set the `pinned` flag on every document whose URL starts with `url_prefix`.
    /// Returns how many documents changed state.
    pub fn set_pinned(&mut self, url_prefix: &str, pinned: bool) -> usize {
        let mut changed = 0;
        for doc in &mut self.meta {
            if doc.url.starts_with(url_prefix) && doc.pinned != pinned {
                doc.pinned = pinned;
                changed += 1;
            }
        }
        changed
    }

    /// Read-only view of the document metadata (for TTL/refresh planning).
    #[must_use]
    /// Per-document outbound edges, parallel to [`IndexBuilder::documents`].
    pub(crate) fn edge_lists(&self) -> &[Vec<UrlKey>] {
        &self.edges
    }

    pub(crate) fn documents(&self) -> &[DocMeta] {
        &self.meta
    }

    /// Build an in-memory, queryable index.
    #[must_use]
    pub fn build(self) -> Index {
        let bm25 = Bm25::build(&self.doc_terms, self.k1, self.b);
        Index::from_owned(
            self.dim,
            self.metric,
            self.embedder_id,
            bm25,
            self.meta,
            self.dense,
            self.edges,
            self.url_index,
        )
    }

    /// Serialize the index to bytes (header + sections + integrity).
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let bm25 = Bm25::build(&self.doc_terms, self.k1, self.b);
        serialize_sections(
            self.dim,
            self.metric,
            self.embedder_id,
            &self.meta,
            &self.dense,
            &self.edges,
            &bm25,
        )
    }

    /// Atomically save the index to `path`.
    ///
    /// # Errors
    /// Returns [`Error::Io`] on a filesystem failure.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        writer::atomic_write(path, &self.serialize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(url: &str, hash: u64, terms: &[&str], vec: Vec<f32>) -> Document {
        Document {
            url: Url::parse(url).unwrap(),
            kind: ResourceKind::Html,
            content_hash: hash,
            title: Some(CompactString::from("t")),
            snippet: CompactString::from("s"),
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

    #[test]
    fn upsert_dedups_by_canonical_url() {
        let mut b = IndexBuilder::new(2, Metric::Cosine, 1);
        assert_eq!(
            b.upsert(doc("https://x.dev/a", 1, &["x"], vec![1.0, 0.0]))
                .unwrap(),
            UpsertOutcome::Added
        );
        // same canonical URL (trailing slash + scheme case), same content → unchanged
        assert_eq!(
            b.upsert(doc("HTTPS://x.dev/a/", 1, &["x"], vec![1.0, 0.0]))
                .unwrap(),
            UpsertOutcome::Unchanged
        );
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn upsert_updates_on_content_change() {
        let mut b = IndexBuilder::new(2, Metric::Cosine, 1);
        b.upsert(doc("https://x.dev/a", 1, &["x"], vec![1.0, 0.0]))
            .unwrap();
        assert_eq!(
            b.upsert(doc("https://x.dev/a", 2, &["y"], vec![0.0, 1.0]))
                .unwrap(),
            UpsertOutcome::Updated
        );
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn upsert_rejects_wrong_dimension() {
        let mut b = IndexBuilder::new(3, Metric::Cosine, 1);
        let err = b
            .upsert(doc("https://x.dev/a", 1, &["x"], vec![1.0, 0.0]))
            .unwrap_err();
        assert!(matches!(err, Error::DimMismatch { index: 3, query: 2 }));
    }

    #[test]
    fn build_yields_searchable_index() {
        let mut b = IndexBuilder::new(2, Metric::Cosine, 7);
        b.upsert(doc("https://x.dev/a", 1, &["cat"], vec![1.0, 0.0]))
            .unwrap();
        b.upsert(doc("https://x.dev/b", 1, &["dog"], vec![0.0, 1.0]))
            .unwrap();
        let idx = b.build();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.dim(), 2);
    }

    #[test]
    fn remove_keeps_state_consistent() {
        let mut b = IndexBuilder::new(2, Metric::Cosine, 1);
        b.upsert(doc("https://x.dev/a", 1, &["a"], vec![1.0, 0.0]))
            .unwrap();
        b.upsert(doc("https://x.dev/b", 1, &["b"], vec![0.0, 1.0]))
            .unwrap();
        b.upsert(doc("https://x.dev/c", 1, &["c"], vec![0.7, 0.7]))
            .unwrap();
        // Remove the middle doc; the last (c) swaps into its slot.
        assert!(b.remove(&Url::parse("https://x.dev/b").unwrap()));
        assert_eq!(b.len(), 2);
        assert!(!b.remove(&Url::parse("https://x.dev/b").unwrap())); // gone
        // Surviving docs still searchable under their URLs, dense rows intact.
        let idx = b.build();
        assert_eq!(idx.len(), 2);
        assert!(idx.contains(UrlKey::from_url(&Url::parse("https://x.dev/a").unwrap())));
        assert!(idx.contains(UrlKey::from_url(&Url::parse("https://x.dev/c").unwrap())));
        assert!(!idx.contains(UrlKey::from_url(&Url::parse("https://x.dev/b").unwrap())));
        // c's dense row (0.7,0.7 normalized) moved into slot 1 correctly.
        let dense = idx.dense().unwrap();
        assert!((dense[2] - dense[3]).abs() < 1e-6);
    }

    #[test]
    fn touch_updates_only_timestamp() {
        let mut b = IndexBuilder::new(2, Metric::Cosine, 1);
        b.upsert(doc("https://x.dev/a", 1, &["a"], vec![1.0, 0.0]))
            .unwrap();
        let url = Url::parse("https://x.dev/a").unwrap();
        assert!(b.touch(&url, 12345));
        assert_eq!(b.documents()[0].fetched_at_ms, 12345);
        assert_eq!(b.documents()[0].content_hash, 1); // unchanged
        assert!(!b.touch(&Url::parse("https://x.dev/missing").unwrap(), 1));
    }

    #[test]
    fn set_pinned_by_prefix_and_sticky_on_update() {
        let mut b = IndexBuilder::new(2, Metric::Cosine, 1);
        b.upsert(doc("https://x.dev/docs/a", 1, &["a"], vec![1.0, 0.0]))
            .unwrap();
        b.upsert(doc("https://x.dev/blog/b", 1, &["b"], vec![0.0, 1.0]))
            .unwrap();
        assert_eq!(b.set_pinned("https://x.dev/docs/", true), 1);
        assert!(b.documents()[0].pinned);
        assert!(!b.documents()[1].pinned);
        // A re-crawl (content change) must not clear the pin.
        b.upsert(doc("https://x.dev/docs/a", 2, &["a2"], vec![0.5, 0.5]))
            .unwrap();
        assert!(b.documents()[0].pinned, "pin must survive re-crawl");
    }
}
