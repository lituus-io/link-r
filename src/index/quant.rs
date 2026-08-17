// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Vector quantization tiers for hypercompression (feature `quant`).
//!
//! Two compression tiers over the flat f32 blob, both with an exact-rerank safety
//! net so recall holds:
//! - **binary** (1 bit/dim, 32×): sign quantization; ranked by Hamming distance
//!   (`popcount`), faster than float cosine. Used as a wide, cheap shortlist.
//! - **int8** (1 byte/dim, ~4×): per-vector symmetric scalar quantization; an
//!   asymmetric f32×int8 dot is the middle-accuracy tier.
//!
//! The headline pattern is [`two_tier_search`]: a binary shortlist over all N,
//! then exact f32 cosine on the shortlist only. None of this is needed at the
//! crate's default scale (100s–10k docs); it is the path to 100k+.

use crate::index::dense::{dot, l2_normalize};
use crate::resource::DocId;
use std::cmp::Ordering;

/// Number of `u64` words needed to hold `dim` sign bits.
#[inline]
fn words_for(dim: usize) -> usize {
    dim.div_ceil(64)
}

/// A binary (sign-bit) quantization of a vector corpus.
#[derive(Clone, Debug)]
pub struct BinaryQuant {
    dim: usize,
    words: usize,
    /// `doc_count × words` packed sign bits (bit set iff component ≥ 0).
    bits: Vec<u64>,
}

impl BinaryQuant {
    /// Quantize a flat `doc_count × dim` corpus. Rows are independent; the
    /// `parallel` feature encodes them with rayon (bit-identical output).
    #[must_use]
    pub fn encode(dense: &[f32], dim: usize) -> Self {
        assert!(dim > 0);
        let words = words_for(dim);
        let doc_count = dense.len() / dim;
        let mut bits = vec![0u64; doc_count * words];
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            bits.par_chunks_exact_mut(words)
                .zip(dense.par_chunks_exact(dim))
                .for_each(|(out, row)| pack_signs(row, out));
        }
        #[cfg(not(feature = "parallel"))]
        for d in 0..doc_count {
            let row = &dense[d * dim..(d + 1) * dim];
            let out = &mut bits[d * words..(d + 1) * words];
            pack_signs(row, out);
        }
        Self { dim, words, bits }
    }

    /// The quantized vector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of quantized documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bits.len().checked_div(self.words).unwrap_or(0)
    }

    /// Whether there are no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Bytes used by the packed bits (for compression-ratio reporting).
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bits.len() * std::mem::size_of::<u64>()
    }

    /// Pack a query vector's signs into the same layout.
    #[must_use]
    pub fn encode_query(&self, query: &[f32]) -> Vec<u64> {
        let mut out = vec![0u64; self.words];
        pack_signs(query, &mut out);
        out
    }

    /// Hamming distance between document `doc` and the packed query bits.
    #[must_use]
    pub fn hamming(&self, doc: DocId, query_bits: &[u64]) -> u32 {
        let base = doc as usize * self.words;
        let row = &self.bits[base..base + self.words];
        row.iter()
            .zip(query_bits)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// The `n` documents with the smallest Hamming distance to `query_bits`,
    /// ascending (closest first).
    #[must_use]
    pub fn shortlist(&self, query_bits: &[u64], n: usize) -> Vec<DocId> {
        let doc_count = self.len();
        let mut scored: Vec<(u32, DocId)> = (0..doc_count as DocId)
            .map(|d| (self.hamming(d, query_bits), d))
            .collect();
        let n = n.min(scored.len());
        if n < scored.len() && !scored.is_empty() {
            scored.select_nth_unstable(n.saturating_sub(1));
        }
        scored.truncate(n);
        scored.sort_unstable();
        scored.into_iter().map(|(_, d)| d).collect()
    }
}

fn pack_signs(vector: &[f32], out: &mut [u64]) {
    for (i, &x) in vector.iter().enumerate() {
        if x >= 0.0 {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
}

/// Symmetric int8-quantize one row into `out`, returning the dequantization scale.
fn encode_row(row: &[f32], out: &mut [i8]) -> f32 {
    let max_abs = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (o, &x) in out.iter_mut().zip(row) {
        *o = (x * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// A per-vector symmetric int8 quantization of a corpus.
#[derive(Clone, Debug)]
pub struct Int8Quant {
    dim: usize,
    codes: Vec<i8>,
    /// Per-document dequantization scale.
    scales: Vec<f32>,
}

impl Int8Quant {
    /// Quantize a flat `doc_count × dim` corpus. Rows are independent; the
    /// `parallel` feature encodes them with rayon (bit-identical output).
    #[must_use]
    pub fn encode(dense: &[f32], dim: usize) -> Self {
        assert!(dim > 0);
        let doc_count = dense.len() / dim;
        let mut codes = vec![0i8; doc_count * dim];
        let mut scales = vec![0f32; doc_count];
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            codes
                .par_chunks_exact_mut(dim)
                .zip(scales.par_iter_mut())
                .zip(dense.par_chunks_exact(dim))
                .for_each(|((out, scale), row)| *scale = encode_row(row, out));
        }
        #[cfg(not(feature = "parallel"))]
        for d in 0..doc_count {
            let row = &dense[d * dim..(d + 1) * dim];
            let out = &mut codes[d * dim..(d + 1) * dim];
            scales[d] = encode_row(row, out);
        }
        Self { dim, codes, scales }
    }

    /// Bytes used by the codes + scales.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.codes.len() + self.scales.len() * std::mem::size_of::<f32>()
    }

    /// Asymmetric dot product of an f32 `query` against quantized document `doc`.
    #[must_use]
    pub fn dot_query(&self, doc: DocId, query: &[f32]) -> f32 {
        let base = doc as usize * self.dim;
        let codes = &self.codes[base..base + self.dim];
        let acc: f32 = codes
            .iter()
            .zip(query)
            .map(|(&c, &q)| f32::from(c) * q)
            .sum();
        acc * self.scales[doc as usize]
    }
}

/// Two-tier search: a wide binary-Hamming shortlist, then exact f32 cosine rerank.
///
/// `dense` is the full-precision `doc_count × dim` blob (vectors L2-normalized);
/// `shortlist` is the candidate pool size (oversample of `k`). Returns the top `k`
/// `(DocId, exact_cosine)` descending.
#[must_use]
pub fn two_tier_search(
    query: &[f32],
    dense: &[f32],
    dim: usize,
    binary: &BinaryQuant,
    shortlist: usize,
    k: usize,
) -> Vec<(DocId, f32)> {
    if k == 0 || query.len() != dim {
        return Vec::new();
    }
    // Normalize the query so the f32 rerank is a true cosine.
    let mut q = query.to_vec();
    l2_normalize(&mut q);

    let qbits = binary.encode_query(&q);
    let candidates = binary.shortlist(&qbits, shortlist.max(k));

    let mut rescored: Vec<(DocId, f32)> = candidates
        .into_iter()
        .map(|d| {
            let row = &dense[d as usize * dim..(d as usize + 1) * dim];
            (d, dot(&q, row))
        })
        .collect();
    rescored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    rescored.truncate(k);
    rescored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::dense::top_k;
    use crate::metric::Metric;

    /// Deterministic xorshift PRNG producing values in [-1, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn random_corpus(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed);
        let mut v = vec![0f32; n * dim];
        for x in &mut v {
            *x = rng.next();
        }
        for d in 0..n {
            l2_normalize(&mut v[d * dim..(d + 1) * dim]);
        }
        v
    }

    #[test]
    fn binary_hamming_is_symmetric_and_self_zero() {
        let dense = random_corpus(4, 16, 1);
        let bq = BinaryQuant::encode(&dense, 16);
        let q0 = bq.encode_query(&dense[0..16]);
        assert_eq!(bq.hamming(0, &q0), 0);
        assert!(bq.hamming(1, &q0) <= 16);
    }

    #[test]
    fn binary_compression_is_32x() {
        let dim = 256;
        let dense = random_corpus(100, dim, 7);
        let bq = BinaryQuant::encode(&dense, dim);
        let f32_bytes = dense.len() * 4;
        // 256 bits = 32 bytes/vec vs 1024 bytes/vec f32.
        assert!(bq.byte_len() * 30 <= f32_bytes, "expected ~32x compression");
    }

    #[test]
    fn int8_dot_approximates_f32_dot() {
        let dim = 64;
        let dense = random_corpus(8, dim, 3);
        let iq = Int8Quant::encode(&dense, dim);
        let query = &dense[0..dim];
        for d in 0..8u32 {
            let exact = dot(query, &dense[d as usize * dim..(d as usize + 1) * dim]);
            let approx = iq.dot_query(d, query);
            assert!(
                (exact - approx).abs() < 0.05,
                "int8 dot off by too much: {exact} vs {approx}"
            );
        }
    }

    #[test]
    fn int8_compression_is_about_4x() {
        let dim = 256;
        let dense = random_corpus(100, dim, 9);
        let iq = Int8Quant::encode(&dense, dim);
        let f32_bytes = dense.len() * 4;
        assert!(iq.byte_len() * 3 < f32_bytes);
    }

    #[test]
    fn two_tier_recall_matches_brute_force() {
        let dim = 64;
        let n = 300;
        let k = 10;
        let dense = random_corpus(n, dim, 42);
        let bq = BinaryQuant::encode(&dense, dim);

        let mut hits = 0usize;
        let mut total = 0usize;
        let mut rng = Rng(0xABCD);
        for _ in 0..20 {
            let mut q = vec![0f32; dim];
            for x in &mut q {
                *x = rng.next();
            }
            l2_normalize(&mut q);

            let truth: Vec<DocId> = top_k(&q, &dense, dim, k, Metric::Cosine, None)
                .into_iter()
                .map(|(d, _)| d)
                .collect();
            // Generous shortlist (oversample) → recall should be high.
            let approx: Vec<DocId> = two_tier_search(&q, &dense, dim, &bq, 64, k)
                .into_iter()
                .map(|(d, _)| d)
                .collect();
            for t in &truth {
                if approx.contains(t) {
                    hits += 1;
                }
            }
            total += truth.len();
        }
        let recall = hits as f32 / total as f32;
        assert!(recall >= 0.85, "two-tier recall@{k} too low: {recall:.3}");
    }

    #[test]
    fn two_tier_degenerate_inputs() {
        let dense = random_corpus(4, 8, 1);
        let bq = BinaryQuant::encode(&dense, 8);
        assert!(two_tier_search(&[1.0; 8], &dense, 8, &bq, 4, 0).is_empty());
        assert!(two_tier_search(&[1.0; 4], &dense, 8, &bq, 4, 2).is_empty()); // dim mismatch
    }
}
