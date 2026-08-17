// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Dense vector kernels: normalization, similarity, and brute-force top-k.
//!
//! Vectors live in one contiguous flat `[f32]` (struct-of-arrays, row-major
//! `doc_count × dim`) so the inner loops are cache-friendly and autovectorize.
//! At our target scale (100s–10k docs) an exact scan is microseconds and beats an
//! ANN index on both recall and simplicity; the `ann` feature only matters past
//! ~100k vectors.

use crate::metric::Metric;
use crate::resource::DocId;
use roaring::RoaringBitmap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// L2-normalize a vector in place. A zero vector is left unchanged.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// L2-normalize each `dim`-wide row of a flat row-major blob in place. Rows are
/// independent; with the `parallel` feature this fans out per row with rayon. Each
/// row runs the identical [`l2_normalize`] op, so the result is bit-for-bit the
/// same as the sequential path (byte-reproducibility holds with the feature on/off).
pub fn l2_normalize_rows(v: &mut [f32], dim: usize) {
    if dim == 0 {
        return;
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        v.par_chunks_exact_mut(dim).for_each(l2_normalize);
    }
    #[cfg(not(feature = "parallel"))]
    for row in v.chunks_exact_mut(dim) {
        l2_normalize(row);
    }
}

/// Dot product of two equal-length slices.
///
/// A tight, `#[inline]` flat loop the compiler autovectorizes (FMA on x86, NEON
/// on aarch64). The SIMD-explicit kernel in the `quant`/optimization tier must
/// agree with this reference within tolerance.
#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Similarity under `metric`. For [`Metric::Cosine`] the vectors are assumed
/// pre-normalized, so this is a dot product.
#[inline]
#[must_use]
pub fn similarity(query: &[f32], doc: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine | Metric::Dot => dot(query, doc),
        Metric::L2 => {
            let mut acc = 0.0f32;
            for i in 0..query.len() {
                let d = query[i] - doc[i];
                acc += d * d;
            }
            -acc // negate so "higher is closer" holds uniformly
        }
    }
}

/// A `(DocId, score)` ordered by score for a bounded min-heap (smallest score on top).
#[derive(Clone, Copy, Debug)]
struct Scored {
    score: f32,
    id: DocId,
}
impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.id == other.id
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse score ordering → a max-heap of `Reverse` behaves as a min-heap.
        // NaN scores sort as smallest so they are evicted first.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Less)
            .then(self.id.cmp(&other.id))
    }
}

/// Brute-force top-`k` by similarity over the flat dense blob.
///
/// `dense` is `doc_count × dim` row-major. When `allowed` is `Some`, only those
/// doc ids are scored (the metadata prefilter result). Results are returned in
/// descending score order.
#[must_use]
pub fn top_k(
    query: &[f32],
    dense: &[f32],
    dim: usize,
    k: usize,
    metric: Metric,
    allowed: Option<&RoaringBitmap>,
) -> Vec<(DocId, f32)> {
    if k == 0 || dim == 0 || query.len() != dim {
        return Vec::new();
    }
    let doc_count = dense.len() / dim;
    let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k + 1);

    let consider = |id: DocId, heap: &mut BinaryHeap<Scored>| {
        let start = id as usize * dim;
        let doc = &dense[start..start + dim];
        let score = similarity(query, doc, metric);
        if heap.len() < k {
            heap.push(Scored { score, id });
        } else if let Some(top) = heap.peek() {
            // `top` is the current smallest score (min-heap via reversed Ord).
            if score > top.score {
                heap.pop();
                heap.push(Scored { score, id });
            }
        }
    };

    match allowed {
        Some(bitmap) => {
            for id in bitmap {
                if (id as usize) < doc_count {
                    consider(id, &mut heap);
                }
            }
        }
        None => {
            for id in 0..doc_count as DocId {
                consider(id, &mut heap);
            }
        }
    }

    let mut out: Vec<(DocId, f32)> = heap.into_iter().map(|s| (s.id, s.score)).collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_yields_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((dot(&v, &v).sqrt() - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_leaves_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_rows_matches_per_row_exactly() {
        // Bit-exact equivalence of the bulk kernel and a manual per-row loop —
        // this is what guarantees byte-reproducibility with `parallel` on or off.
        let dim = 8;
        let n = 600;
        let mut seed = 0x1234_5678u64;
        let mut a: Vec<f32> = (0..n * dim)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let mut b = a.clone();
        l2_normalize_rows(&mut a, dim);
        for row in b.chunks_exact_mut(dim) {
            l2_normalize(row);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn dot_matches_f64_reference() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [4.0f32, 3.0, 2.0, 1.0];
        let reference: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        assert!((f64::from(dot(&a, &b)) - reference).abs() < 1e-9);
    }

    #[test]
    fn top_k_finds_nearest() {
        // dim=2, three docs. Query closest to doc 1.
        let dense = vec![
            1.0, 0.0, // doc 0
            0.0, 1.0, // doc 1
            -1.0, 0.0, // doc 2
        ];
        let query = [0.0, 1.0];
        let hits = top_k(&query, &dense, 2, 2, Metric::Dot, None);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 1); // nearest
        assert!(hits[0].1 > hits[1].1);
    }

    #[test]
    fn top_k_respects_allowed_filter() {
        let dense = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let query = [0.0, 1.0];
        let mut allowed = RoaringBitmap::new();
        allowed.insert(0);
        allowed.insert(2);
        let hits = top_k(&query, &dense, 2, 5, Metric::Dot, Some(&allowed));
        // doc 1 (the true nearest) is filtered out.
        assert!(hits.iter().all(|(id, _)| *id != 1));
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn top_k_handles_k_larger_than_corpus() {
        let dense = vec![1.0, 0.0, 0.0, 1.0];
        let hits = top_k(&[1.0, 0.0], &dense, 2, 100, Metric::Dot, None);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn top_k_empty_for_degenerate_inputs() {
        assert!(top_k(&[1.0], &[1.0], 1, 0, Metric::Dot, None).is_empty());
        assert!(top_k(&[1.0, 2.0], &[1.0], 1, 5, Metric::Dot, None).is_empty());
        // dim mismatch
    }
}
