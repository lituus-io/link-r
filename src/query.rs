// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Query inputs and search results.
//!
//! [`PreparedQuery`] is the embedder-agnostic search input the index consumes: a
//! dense vector plus normalized BM25 terms plus a [`Filter`]. The text-level
//! `Query` (which turns text into a vector via an [`Embedder`](crate::embed)) is
//! assembled by the facade. [`Hit`]/[`Hits`] borrow the index, so results are
//! allocation-free to read.

use crate::index::fuse::DEFAULT_RRF_K;
use crate::index::meta::{DocMeta, Filter};
use crate::resource::{DocId, ResourceKind};
use compact_str::CompactString;

/// How dense and sparse result lists are combined.
#[derive(Clone, Copy, Debug)]
pub enum Fusion {
    /// Reciprocal Rank Fusion with the given constant (rank-based; scale-free).
    Rrf {
        /// The RRF damping constant.
        k: u32,
    },
}

impl Default for Fusion {
    fn default() -> Self {
        Self::Rrf { k: DEFAULT_RRF_K }
    }
}

/// Ranking parameters for a hybrid search.
#[derive(Clone, Copy, Debug)]
pub struct RankParams {
    /// Weight of the dense (semantic) arm.
    pub dense_weight: f32,
    /// Weight of the sparse (BM25) arm.
    pub sparse_weight: f32,
    /// How the arms are fused.
    pub fusion: Fusion,
    /// Per-arm candidate pool size as a multiple of `limit` (oversampling before
    /// fusion). Clamped to a sane floor internally.
    pub candidate_multiplier: usize,
    /// One-hop knowledge-graph boost applied after fusion (0 = off, the default):
    /// each candidate gains `graph_boost × mean(fused score of its in/out neighbors
    /// within the candidate pool)`, rewarding well-connected hub/authority pages.
    pub graph_boost: f32,
}

impl Default for RankParams {
    fn default() -> Self {
        Self {
            dense_weight: 1.0,
            sparse_weight: 1.0,
            fusion: Fusion::default(),
            candidate_multiplier: 10,
            graph_boost: 0.0,
        }
    }
}

/// An embedder-agnostic search input.
#[derive(Debug)]
pub struct PreparedQuery<'a> {
    /// The query embedding (same dimension and metric as the index).
    pub vector: &'a [f32],
    /// Normalized query terms for BM25.
    pub terms: &'a [CompactString],
    /// Structured prefilter.
    pub filter: &'a Filter,
    /// Maximum hits to return.
    pub limit: usize,
    /// Ranking parameters.
    pub rank: RankParams,
}

/// A single search result, borrowing the index it came from.
#[derive(Clone, Copy, Debug)]
pub struct Hit<'a> {
    /// The resolvable remote URL (the value of the index).
    pub url: &'a str,
    /// The fused relevance score (higher is better).
    pub score: f32,
    /// Display title, if any.
    pub title: Option<&'a str>,
    /// A short display snippet.
    pub snippet: &'a str,
    /// Content kind.
    pub kind: ResourceKind,
    doc: &'a DocMeta,
}

impl<'a> Hit<'a> {
    /// The document's structured tags.
    pub fn tags(&self) -> impl Iterator<Item = &'a str> {
        self.doc.tags.iter().map(CompactString::as_str)
    }
}

/// The ranked results of a search, borrowing the index.
#[derive(Debug)]
pub struct Hits<'a> {
    docs: &'a [DocMeta],
    ranked: Vec<(DocId, f32)>,
}

impl<'a> Hits<'a> {
    /// Construct from the index's metadata and a ranked id list.
    #[must_use]
    pub(crate) fn new(docs: &'a [DocMeta], ranked: Vec<(DocId, f32)>) -> Self {
        Self { docs, ranked }
    }

    /// Number of hits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranked.len()
    }

    /// Whether there are no hits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranked.is_empty()
    }

    /// Iterate hits in descending relevance.
    pub fn iter(&self) -> impl Iterator<Item = Hit<'a>> + '_ {
        let docs = self.docs;
        self.ranked.iter().map(move |&(id, score)| {
            let doc = &docs[id as usize];
            Hit {
                url: &doc.url,
                score,
                title: doc.title.as_deref(),
                snippet: &doc.snippet,
                kind: doc.kind,
                doc,
            }
        })
    }
}

impl<'a> IntoIterator for &Hits<'a> {
    type Item = Hit<'a>;
    type IntoIter = std::vec::IntoIter<Hit<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<_>>().into_iter()
    }
}
