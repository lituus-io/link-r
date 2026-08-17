// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The deterministic hash embedder — the zero-dependency, offline fallback.
//!
//! Signed feature hashing of word tokens and their character n-grams into a
//! fixed-dimensional vector, L2-normalized. It is a dense proxy for *lexical*
//! overlap (not semantics), so it pairs with BM25 to make tests/benches/fuzz
//! hermetic and byte-reproducible. Production semantic search uses the `onnx`
//! embedder; see the crate docs.

use crate::embed::{Embedder, EmbedderId};
use crate::error::{Error, Result};
use crate::index::dense::l2_normalize;
use crate::metric::Metric;
use crate::text;
use std::future::Ready;
use xxhash_rust::xxh3::xxh3_64_with_seed;

/// Default character n-gram size for subword robustness.
const DEFAULT_NGRAM: u8 = 3;
/// Default hashing seed (fixed → deterministic across runs/platforms).
const DEFAULT_SEED: u64 = 0x6C69_6E6B_725F_6873; // "linkr_hs"

/// A deterministic feature-hashing embedder.
#[derive(Clone, Copy, Debug)]
pub struct HashEmbedder {
    dim: u32,
    ngram: u8,
    seed: u64,
}

impl HashEmbedder {
    /// Create a `dim`-dimensional hash embedder with default n-gram and seed.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self::with_params(dim, DEFAULT_NGRAM, DEFAULT_SEED)
    }

    /// Create with explicit parameters. All three are folded into the
    /// [`EmbedderId`], so changing any of them invalidates an old index.
    #[must_use]
    pub fn with_params(dim: usize, ngram: u8, seed: u64) -> Self {
        assert!(dim > 0, "embedding dimension must be positive");
        Self {
            dim: dim as u32,
            ngram,
            seed,
        }
    }

    /// Hash a feature string into a `(bucket, sign)` and accumulate it into `row`.
    #[inline]
    fn add_feature(&self, feature: &str, row: &mut [f32]) {
        let h = xxh3_64_with_seed(feature.as_bytes(), self.seed);
        let bucket = (h % u64::from(self.dim)) as usize;
        // Use a high bit for the sign so it is independent of the bucket bits.
        let sign = if (h >> 63) & 1 == 1 { 1.0 } else { -1.0 };
        row[bucket] += sign;
    }

    /// Embed one text into `row` (length == dim).
    fn embed_into(&self, text_in: &str, row: &mut [f32]) {
        row.fill(0.0);
        for token in text::normalized_tokens(text_in) {
            self.add_feature(&token, row);
            text::for_each_char_ngram(&token, self.ngram as usize, |g| self.add_feature(g, row));
        }
        l2_normalize(row);
    }
}

impl Embedder for HashEmbedder {
    type EmbedFuture<'a> = Ready<Result<()>>;

    fn dim(&self) -> usize {
        self.dim as usize
    }

    fn metric(&self) -> Metric {
        Metric::Cosine
    }

    fn identity(&self) -> EmbedderId {
        let params = u64::from(self.ngram) ^ self.seed.rotate_left(8);
        EmbedderId::derive("hash", self.dim as usize, params)
    }

    fn embed_batch<'a>(
        &'a self,
        texts: &'a [&'a str],
        out: &'a mut [f32],
    ) -> Self::EmbedFuture<'a> {
        let dim = self.dim as usize;
        if out.len() != texts.len() * dim {
            return std::future::ready(Err(Error::embed(format!(
                "output buffer is {} floats, expected {} ({} texts × {dim})",
                out.len(),
                texts.len() * dim,
                texts.len()
            ))));
        }
        for (i, t) in texts.iter().enumerate() {
            self.embed_into(t, &mut out[i * dim..(i + 1) * dim]);
        }
        std::future::ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::embed_one;
    use crate::index::dense::dot;

    #[tokio::test]
    async fn embedding_is_unit_length() {
        let e = HashEmbedder::new(64);
        let v = embed_one(&e, "the quick brown fox").await.unwrap();
        assert_eq!(v.len(), 64);
        assert!((dot(&v, &v).sqrt() - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn embedding_is_deterministic() {
        let e = HashEmbedder::new(128);
        let a = embed_one(&e, "BigQuery row access policy").await.unwrap();
        let b = embed_one(&e, "BigQuery row access policy").await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn shared_vocabulary_scores_higher_than_disjoint() {
        let e = HashEmbedder::new(256);
        let q = embed_one(&e, "kubernetes autopilot cluster").await.unwrap();
        let near = embed_one(&e, "autopilot kubernetes cluster setup")
            .await
            .unwrap();
        let far = embed_one(&e, "banana smoothie recipe").await.unwrap();
        assert!(dot(&q, &near) > dot(&q, &far));
    }

    #[tokio::test]
    async fn batch_matches_single() {
        let e = HashEmbedder::new(96);
        let texts = ["alpha beta", "gamma delta"];
        let mut out = vec![0.0f32; 2 * 96];
        e.embed_batch(&texts, &mut out).await.unwrap();
        let single0 = embed_one(&e, "alpha beta").await.unwrap();
        assert_eq!(&out[..96], single0.as_slice());
    }

    #[tokio::test]
    async fn wrong_buffer_size_errors() {
        let e = HashEmbedder::new(32);
        let texts = ["x"];
        let mut out = vec![0.0f32; 16]; // wrong
        assert!(e.embed_batch(&texts, &mut out).await.is_err());
    }

    #[test]
    fn identity_depends_on_params() {
        assert_ne!(
            HashEmbedder::new(64).identity(),
            HashEmbedder::new(128).identity()
        );
        assert_ne!(
            HashEmbedder::with_params(64, 3, 1).identity(),
            HashEmbedder::with_params(64, 3, 2).identity()
        );
    }

    #[tokio::test]
    async fn empty_text_yields_zero_vector() {
        let e = HashEmbedder::new(32);
        let v = embed_one(&e, "").await.unwrap();
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
