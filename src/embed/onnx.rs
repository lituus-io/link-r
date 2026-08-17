// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The production semantic embedder: `bge-small-en-v1.5` via `fastembed`/ONNX.
//!
//! 384-dimensional, cosine. The model (~130 MB) is downloaded once on first
//! construction (over rustls) and cached by `hf-hub`. Inference runs inline
//! (returning a ready future); callers needing non-blocking behavior on a busy
//! executor should drive [`Embedder::embed_batch`] under `spawn_blocking`.

use crate::embed::{Embedder, EmbedderId};
use crate::error::{Error, Result};
use crate::index::dense::l2_normalize;
use crate::metric::Metric;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::fmt;
use std::future::Ready;

/// Embedding dimension of `bge-small-en-v1.5`.
const BGE_SMALL_DIM: usize = 384;
/// Inference batch size.
const BATCH_SIZE: usize = 256;

/// A `bge-small-en-v1.5` ONNX embedder.
pub struct OnnxEmbedder {
    model: TextEmbedding,
}

impl OnnxEmbedder {
    /// Load `bge-small-en-v1.5`, downloading and caching the model on first use.
    ///
    /// # Errors
    /// Returns [`Error::Embed`] if the model cannot be loaded.
    pub fn new() -> Result<Self> {
        let options =
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false);
        let model = TextEmbedding::try_new(options)
            .map_err(|e| Error::embed(format!("load bge-small-en-v1.5: {e}")))?;
        Ok(Self { model })
    }
}

impl fmt::Debug for OnnxEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnnxEmbedder")
            .field("model", &"bge-small-en-v1.5")
            .field("dim", &BGE_SMALL_DIM)
            .finish()
    }
}

impl Embedder for OnnxEmbedder {
    type EmbedFuture<'a> = Ready<Result<()>>;

    fn dim(&self) -> usize {
        BGE_SMALL_DIM
    }

    fn metric(&self) -> Metric {
        Metric::Cosine
    }

    fn identity(&self) -> EmbedderId {
        EmbedderId::derive("onnx:bge-small-en-v1.5", BGE_SMALL_DIM, 1)
    }

    fn embed_batch<'a>(
        &'a self,
        texts: &'a [&'a str],
        out: &'a mut [f32],
    ) -> Self::EmbedFuture<'a> {
        if out.len() != texts.len() * BGE_SMALL_DIM {
            return std::future::ready(Err(Error::embed(format!(
                "output buffer is {} floats, expected {}",
                out.len(),
                texts.len() * BGE_SMALL_DIM
            ))));
        }
        if texts.is_empty() {
            return std::future::ready(Ok(()));
        }
        let vectors = match self.model.embed(texts.to_vec(), Some(BATCH_SIZE)) {
            Ok(v) => v,
            Err(e) => return std::future::ready(Err(Error::embed(e.to_string()))),
        };
        if vectors.len() != texts.len() {
            return std::future::ready(Err(Error::embed("embedder returned wrong row count")));
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != BGE_SMALL_DIM {
                return std::future::ready(Err(Error::DimMismatch {
                    index: BGE_SMALL_DIM,
                    query: v.len(),
                }));
            }
            let row = &mut out[i * BGE_SMALL_DIM..(i + 1) * BGE_SMALL_DIM];
            row.copy_from_slice(v);
            // bge models output near-unit vectors; normalize to make cosine exact.
            l2_normalize(row);
        }
        std::future::ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::embed_one;
    use crate::index::dense::dot;

    #[test]
    fn identity_and_dim_are_fixed() {
        // These do not require loading the model.
        let id = EmbedderId::derive("onnx:bge-small-en-v1.5", BGE_SMALL_DIM, 1);
        assert_eq!(id, EmbedderId::derive("onnx:bge-small-en-v1.5", 384, 1));
    }

    // Requires a ~130 MB model download + ONNX runtime; run explicitly with
    // `cargo test --features onnx -- --ignored`.
    #[tokio::test]
    #[ignore = "downloads the bge-small model"]
    async fn semantic_similarity_beats_disjoint() {
        let e = OnnxEmbedder::new().unwrap();
        assert_eq!(e.dim(), 384);
        let q = embed_one(&e, "how do I keep query results private per user")
            .await
            .unwrap();
        let near = embed_one(
            &e,
            "row-level security restricts which rows a user can read",
        )
        .await
        .unwrap();
        let far = embed_one(&e, "a recipe for banana bread").await.unwrap();
        // semantic match despite no shared vocabulary — the case hash embedding fails.
        assert!(dot(&q, &near) > dot(&q, &far));
    }
}
