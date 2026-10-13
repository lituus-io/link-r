// Author: support@lituus.io
// Copyright: lituus-io, all rights reserved
//
//! The additive 0.3 surface: images in memory, a caller-chosen embedder, and
//! `freeze`.

#![cfg(feature = "crawl")]

#[path = "support/golden.rs"]
mod golden;

use link_r::embed::EmbedderId;
use link_r::metric::Metric;
use link_r::{Embedder, HashEmbedder, Index, LinkIndex};

#[tokio::test]
async fn bytes_round_trip_equals_the_file_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb.lnkr");
    let mut index = LinkIndex::open_or_create(&path).unwrap();
    golden::ingest(&mut index).await;
    index.save().unwrap();
    let image = index.to_bytes().unwrap();
    assert_eq!(image, std::fs::read(&path).unwrap());
    let a = LinkIndex::open(&path).unwrap();
    let b = LinkIndex::from_bytes(bytes::Bytes::from(image)).unwrap();
    assert_eq!(golden::dump(&a).await, golden::dump(&b).await);
    assert_eq!(golden::dump(&a).await, golden::dump(&index).await);
}

#[tokio::test]
async fn a_misaligned_image_still_loads_and_matches() {
    let mut index = LinkIndex::in_memory().unwrap();
    golden::ingest(&mut index).await;
    let image = index.to_bytes().unwrap();
    // Force every alignment of the buffer start.
    for shift in 0..8 {
        let mut buf = vec![0u8; shift];
        buf.extend_from_slice(&image);
        let shifted = bytes::Bytes::from(buf).slice(shift..);
        let loaded = LinkIndex::from_bytes(shifted).unwrap();
        assert_eq!(
            golden::dump(&loaded).await,
            golden::dump(&index).await,
            "shift {shift}"
        );
        let raw = Index::from_bytes(bytes::Bytes::from(image.clone())).unwrap();
        assert_eq!(raw.len(), index.len());
    }
}

#[tokio::test]
async fn corrupt_images_are_errors_not_panics() {
    let mut index = LinkIndex::in_memory().unwrap();
    golden::ingest(&mut index).await;
    let image = index.to_bytes().unwrap();
    for cut in [0, 1, 8, 64, image.len() / 2, image.len() - 1] {
        assert!(
            LinkIndex::from_bytes(bytes::Bytes::copy_from_slice(&image[..cut])).is_err(),
            "cut {cut}"
        );
    }
    let step = (image.len() / 97).max(1);
    for at in (0..image.len()).step_by(step) {
        let mut bad = image.clone();
        bad[at] ^= 0x5a;
        let _ = LinkIndex::from_bytes(bytes::Bytes::from(bad)); // typed error or clean load, never a panic
    }
}

/// A different embedder: same hashing, different dimension and identity.
#[derive(Debug)]
struct Wide(HashEmbedder);

impl Embedder for Wide {
    type EmbedFuture<'a> = <HashEmbedder as Embedder>::EmbedFuture<'a>;
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn metric(&self) -> Metric {
        Metric::Cosine
    }
    fn identity(&self) -> EmbedderId {
        EmbedderId::derive("wide-test", self.0.dim(), 7)
    }
    fn embed_batch<'a>(
        &'a self,
        texts: &'a [&'a str],
        out: &'a mut [f32],
    ) -> Self::EmbedFuture<'a> {
        self.0.embed_batch(texts, out)
    }
}

#[tokio::test]
async fn a_caller_chosen_embedder_is_used_and_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wide.lnkr");
    let mut index = LinkIndex::open_or_create_with(&path, Wide(HashEmbedder::new(64))).unwrap();
    let root = link_r::SourceRef::http("https://docs.example.com/").unwrap();
    index
        .ingest_from(&golden::MemSource, &root, 3, false)
        .await
        .unwrap();
    index.save().unwrap();
    let hits = index.search("autopilot nodes", 3).await.unwrap();
    assert_eq!(hits[0].url, "https://docs.example.com/compute/autopilot");

    // Reopen with the same embedder: fine. With the default embedder: refused.
    let again = LinkIndex::open_with(&path, Wide(HashEmbedder::new(64))).unwrap();
    assert_eq!(again.len(), 4);
    assert!(
        LinkIndex::open(&path).is_err(),
        "identity mismatch must be refused"
    );
    let image = bytes::Bytes::from(std::fs::read(&path).unwrap());
    assert!(LinkIndex::from_bytes(image.clone()).is_err());
    assert!(LinkIndex::from_bytes_with(image, Wide(HashEmbedder::new(64))).is_ok());
    let fresh = LinkIndex::in_memory_with(Wide(HashEmbedder::new(64)));
    assert!(fresh.is_empty());
}

#[tokio::test]
async fn freeze_keeps_results_and_allows_further_mutation() {
    let mut index = LinkIndex::in_memory().unwrap();
    golden::ingest(&mut index).await;
    let before = golden::dump(&index).await;
    let image = index.to_bytes().unwrap();
    index.freeze().unwrap();
    assert_eq!(golden::dump(&index).await, before);
    assert_eq!(
        index.to_bytes().unwrap(),
        image,
        "frozen image is the serialized builder"
    );
    index.freeze().unwrap(); // idempotent
    index.unpin("https://docs.example.com/guides/").unwrap();
    assert_ne!(
        golden::dump(&index).await,
        before,
        "mutation after freeze works"
    );
}
