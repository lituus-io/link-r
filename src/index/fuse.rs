// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Reciprocal Rank Fusion (RRF) for combining dense and sparse result lists.
//!
//! RRF is rank-based, so it sidesteps the scale mismatch between cosine
//! similarity and BM25: a document's contribution from each list is
//! `weight / (k + rank)` (1-based rank), and contributions sum across lists.

use crate::resource::DocId;
use std::cmp::Ordering;
use std::collections::HashMap;

/// The conventional RRF constant; damps the influence of top ranks.
pub const DEFAULT_RRF_K: u32 = 60;

/// One ranked result list to fuse: doc ids in descending relevance, plus a weight.
#[derive(Clone, Copy, Debug)]
pub struct RankList<'a> {
    /// Doc ids in rank order (best first).
    pub ids: &'a [DocId],
    /// Relative weight of this list.
    pub weight: f32,
}

impl<'a> RankList<'a> {
    /// A unit-weight list.
    #[must_use]
    pub fn new(ids: &'a [DocId]) -> Self {
        Self { ids, weight: 1.0 }
    }

    /// A weighted list.
    #[must_use]
    pub fn weighted(ids: &'a [DocId], weight: f32) -> Self {
        Self { ids, weight }
    }
}

/// Fuse ranked lists with RRF and return the top `limit` doc ids by fused score,
/// descending. `k` is the RRF constant (see [`DEFAULT_RRF_K`]).
#[must_use]
pub fn reciprocal_rank_fusion(lists: &[RankList<'_>], k: u32, limit: usize) -> Vec<(DocId, f32)> {
    let kf = k as f32;
    let mut scores: HashMap<DocId, f32> = HashMap::new();
    for list in lists {
        for (rank, &id) in list.ids.iter().enumerate() {
            let contribution = list.weight / (kf + (rank as f32 + 1.0));
            *scores.entry(id).or_insert(0.0) += contribution;
        }
    }
    let mut fused: Vec<(DocId, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_in_both_lists_outranks_one_list() {
        // doc 5 appears in both lists; doc 1 only in dense, doc 9 only in sparse.
        let dense = [5u32, 1, 2];
        let sparse = [9u32, 5, 3];
        let fused = reciprocal_rank_fusion(
            &[RankList::new(&dense), RankList::new(&sparse)],
            DEFAULT_RRF_K,
            10,
        );
        assert_eq!(fused[0].0, 5, "doc present in both lists should win");
    }

    #[test]
    fn weights_shift_ranking() {
        let dense = [1u32];
        let sparse = [2u32];
        // Heavily weight sparse → doc 2 wins.
        let fused = reciprocal_rank_fusion(
            &[
                RankList::weighted(&dense, 1.0),
                RankList::weighted(&sparse, 10.0),
            ],
            DEFAULT_RRF_K,
            10,
        );
        assert_eq!(fused[0].0, 2);
    }

    #[test]
    fn fusion_is_permutation_stable_on_ties() {
        // Identical single-element lists → deterministic ordering by id.
        let a = [3u32, 7];
        let b = [7u32, 3];
        let fused =
            reciprocal_rank_fusion(&[RankList::new(&a), RankList::new(&b)], DEFAULT_RRF_K, 10);
        // both docs get symmetric scores; tie broken by id ascending
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, 3);
        assert_eq!(fused[1].0, 7);
    }

    #[test]
    fn respects_limit() {
        let list = [1u32, 2, 3, 4, 5];
        let fused = reciprocal_rank_fusion(&[RankList::new(&list)], DEFAULT_RRF_K, 3);
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn empty_lists_yield_empty() {
        assert!(reciprocal_rank_fusion(&[], DEFAULT_RRF_K, 10).is_empty());
    }
}
