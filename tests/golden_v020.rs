// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Compatibility with indexes written by link-r 0.2.0.
//!
//! `tests/fixtures/v0.2.0/facade.lnkr` and its dump were produced by the
//! released 0.2.0 code running `tests/support/golden.rs`. Every later version
//! must read that file identically — from disk and from bytes — and keep
//! writing byte-identical output for it.

#![cfg(all(feature = "crawl", not(feature = "onnx")))]

#[path = "support/golden.rs"]
mod golden;

use link_r::LinkIndex;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/v0.2.0")
        .join(name)
}

fn expected() -> String {
    std::fs::read_to_string(fixture("facade.dump")).unwrap()
}

#[tokio::test]
async fn a_020_index_reads_identically_from_disk_and_from_bytes() {
    let from_disk = LinkIndex::open(fixture("facade.lnkr")).unwrap();
    assert_eq!(golden::dump(&from_disk).await, expected());
    let bytes = bytes::Bytes::from(std::fs::read(fixture("facade.lnkr")).unwrap());
    let from_bytes = LinkIndex::from_bytes(bytes).unwrap();
    assert_eq!(golden::dump(&from_bytes).await, expected());
}

#[tokio::test]
async fn saving_an_unmutated_020_index_is_byte_identical() {
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
async fn today_serializes_the_020_index_byte_identically() {
    // Decode fully (materialize the builder) and re-encode with today's
    // writer: the same bytes 0.2.0 wrote.
    let original = std::fs::read(fixture("facade.lnkr")).unwrap();
    let mut index = LinkIndex::from_bytes(bytes::Bytes::from(original.clone())).unwrap();
    index.pin("https://docs.example.com/guides/").unwrap(); // a no-op mutation
    assert_eq!(index.to_bytes().unwrap(), original);
}

#[tokio::test]
async fn a_020_index_accepts_new_writes() {
    let dir = tempfile::tempdir().unwrap();
    let copy = dir.path().join("kb.lnkr");
    std::fs::copy(fixture("facade.lnkr"), &copy).unwrap();
    let mut index = LinkIndex::open(&copy).unwrap();
    golden::ingest(&mut index).await; // re-ingest: every page unchanged
    index.save().unwrap();
    let reopened = LinkIndex::open(&copy).unwrap();
    assert_eq!(golden::dump(&reopened).await, expected());
}
