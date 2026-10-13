// Author: support@lituus.io
// Copyright: lituus-io, all rights reserved
//
//! Format stability for indexes written by link-r 0.4.0 (format 2, the
//! graph-r container).
//!
//! `tests/fixtures/v0.4.0/facade.lnkr` was written by 0.4.0 from the corpus
//! in `tests/support/golden.rs`; its dump is the 0.2.0 dump unchanged, so the
//! format change kept every document, score and edge. Every later version
//! must read that file identically — from disk and from bytes — and keep
//! writing byte-identical output for it. `LINKR_WRITE_GOLDEN=1` rewrites the
//! fixture (only when the format deliberately changes).

#![cfg(all(feature = "crawl", not(feature = "onnx")))]

#[path = "support/golden.rs"]
mod golden;

use link_r::LinkIndex;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/v0.4.0")
        .join(name)
}

fn expected() -> String {
    std::fs::read_to_string(fixture("facade.dump")).unwrap()
}

#[tokio::test]
async fn the_fixture_is_current() {
    let mut index = LinkIndex::in_memory().unwrap();
    golden::ingest(&mut index).await;
    let written = index.to_bytes().unwrap();
    if std::env::var_os("LINKR_WRITE_GOLDEN").is_some() {
        std::fs::write(fixture("facade.lnkr"), &written).unwrap();
    }
    // Fetch times differ per run, so the fresh build matches by content.
    assert_eq!(golden::dump(&index).await, expected());
}

#[tokio::test]
async fn a_040_index_reads_identically_from_disk_and_from_bytes() {
    let from_disk = LinkIndex::open(fixture("facade.lnkr")).unwrap();
    assert_eq!(golden::dump(&from_disk).await, expected());
    let bytes = bytes::Bytes::from(std::fs::read(fixture("facade.lnkr")).unwrap());
    let from_bytes = LinkIndex::from_bytes(bytes).unwrap();
    assert_eq!(golden::dump(&from_bytes).await, expected());
}

#[tokio::test]
async fn saving_an_unmutated_040_index_is_byte_identical() {
    let original = std::fs::read(fixture("facade.lnkr")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let copy = dir.path().join("kb.lnkr");
    std::fs::write(&copy, &original).unwrap();
    let index = LinkIndex::open(&copy).unwrap();
    index.save().unwrap();
    assert_eq!(std::fs::read(&copy).unwrap(), original);
    assert_eq!(index.to_bytes().unwrap(), original);
}

#[tokio::test]
async fn today_serializes_the_040_index_byte_identically() {
    // Decode fully (materialize the builder) and re-encode with today's
    // writer: the same bytes 0.4.0 wrote.
    let original = std::fs::read(fixture("facade.lnkr")).unwrap();
    let mut index = LinkIndex::from_bytes(bytes::Bytes::from(original.clone())).unwrap();
    index.pin("https://docs.example.com/guides/").unwrap(); // a no-op mutation
    assert_eq!(index.to_bytes().unwrap(), original);
}

#[tokio::test]
async fn a_040_index_accepts_new_writes() {
    let dir = tempfile::tempdir().unwrap();
    let copy = dir.path().join("kb.lnkr");
    std::fs::copy(fixture("facade.lnkr"), &copy).unwrap();
    let mut index = LinkIndex::open(&copy).unwrap();
    golden::ingest(&mut index).await; // re-ingest: every page unchanged
    index.save().unwrap();
    let reopened = LinkIndex::open(&copy).unwrap();
    assert_eq!(golden::dump(&reopened).await, expected());
}
