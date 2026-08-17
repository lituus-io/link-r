// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Embedders: turn distilled text into dense vectors.
//!
//! [`Embedder`] is a GAT-based async trait so a synchronous embedder (the
//! deterministic [`HashEmbedder`]) and an async one (a remote API) share one
//! interface with no `Box<dyn Future>`. The index records an [`EmbedderId`] so a
//! query can verify it is using a compatible embedder at open time.

pub mod hash;
#[cfg(feature = "onnx")]
pub mod onnx;

pub use hash::HashEmbedder;
#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbedder;

use crate::error::Result;
use crate::metric::Metric;
use std::future::Future;

/// A stable identity for an embedder configuration (model + dimension + params).
///
/// Persisted in the index header; a search validates that the query embedder's
/// identity matches, catching "indexed with model A, querying with model B".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EmbedderId(pub u64);

impl EmbedderId {
    /// Derive an id from a name and discriminating parameters.
    #[must_use]
    pub fn derive(name: &str, dim: usize, params: u64) -> Self {
        use xxhash_rust::xxh3::xxh3_64;
        let mut seed = xxh3_64(name.as_bytes());
        seed ^= (dim as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        seed ^= params.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        Self(seed)
    }

    /// The raw value (as stored in the index header).
    #[must_use]
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Turns text into dense vectors.
///
/// Implementors fill a caller-provided `out` buffer (`texts.len() × dim`,
/// row-major) to keep allocation under the caller's control. Synchronous
/// embedders return a ready future; remote ones return their request future.
pub trait Embedder: Send + Sync {
    /// The future returned by [`Embedder::embed_batch`].
    type EmbedFuture<'a>: Future<Output = Result<()>> + Send + 'a
    where
        Self: 'a;

    /// The embedding dimension.
    fn dim(&self) -> usize;

    /// The metric the produced vectors are intended for.
    fn metric(&self) -> Metric;

    /// This embedder's stable identity.
    fn identity(&self) -> EmbedderId;

    /// Embed `texts` into `out` (`texts.len() * dim` floats, row-major).
    ///
    /// Implementations must error (not panic) if `out.len() != texts.len() * dim`.
    fn embed_batch<'a>(&'a self, texts: &'a [&'a str], out: &'a mut [f32])
        -> Self::EmbedFuture<'a>;
}

/// Embed a single query string into a fresh vector.
pub async fn embed_one<E: Embedder>(embedder: &E, text: &str) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; embedder.dim()];
    let texts = [text];
    embedder.embed_batch(&texts, &mut out).await?;
    Ok(out)
}
