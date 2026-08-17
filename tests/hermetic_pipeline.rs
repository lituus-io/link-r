// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Hermetic, offline end-to-end test: extract → embed → build → save → open →
//! search, with no network and the deterministic hash embedder. Mirrors what the
//! crawler and facade will drive in later phases.

use compact_str::CompactString;
use link_r::embed::{embed_one, Embedder};
use link_r::extract::{AutoExtractor, Extractor};
use link_r::index::IndexBuilder;
use link_r::metric::Metric;
use link_r::query::{PreparedQuery, RankParams};
use link_r::resource::{Resource, ResourceKind};
use link_r::{Filter, HashEmbedder, Index};
use url::Url;
use xxhash_rust::xxh3::xxh3_64;

const DIM: usize = 256;

/// A tiny in-memory "site": (url, content-type kind, raw bytes).
fn corpus() -> Vec<(&'static str, ResourceKind, &'static [u8])> {
    vec![
        (
            "https://docs.example.com/networking/psc",
            ResourceKind::Html,
            b"<html><head><title>Private Service Connect</title></head>
              <body><h1>PSC</h1><p>Reach Google APIs and services privately over
              an internal IP using Private Service Connect endpoints.</p></body></html>",
        ),
        (
            "https://docs.example.com/compute/autopilot.md",
            ResourceKind::Markdown,
            b"---\ntitle: GKE Autopilot\n---\n# GKE Autopilot\n\nAutopilot is a mode
              of operation in GKE in which Google manages your cluster nodes,
              scaling, and security for Kubernetes workloads.",
        ),
        (
            "https://docs.example.com/data/bigquery.md",
            ResourceKind::Markdown,
            b"---\ntitle: BigQuery Row Access\n---\n# Row Access Policies\n\nBigQuery
              row-level security uses row access policies to restrict which rows a
              principal can query in a table.",
        ),
    ]
}

fn build_index() -> Index {
    let embedder = HashEmbedder::new(DIM);
    let extractor = AutoExtractor;
    let mut builder = IndexBuilder::new(DIM, Metric::Cosine, embedder.identity().raw());

    // tokio runtime to drive the (synchronous, but async-typed) embedder.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    for (url, kind, bytes) in corpus() {
        let resource = Resource::new(Url::parse(url).unwrap()).with_kind(kind);
        let descriptor = extractor.extract(&resource, bytes).unwrap();
        let vector = rt
            .block_on(embed_one(&embedder, &descriptor.embed_text))
            .unwrap();
        let content_hash = xxh3_64(bytes);
        let doc = descriptor.into_document(resource.url.clone(), kind, content_hash, vector);
        builder.upsert(doc).unwrap();
    }
    builder.build()
}

fn search_top_url(index: &Index, embedder: &HashEmbedder, query: &str) -> String {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let vector = rt.block_on(embed_one(embedder, query)).unwrap();
    let terms: Vec<CompactString> = link_r::text::normalized_tokens(query).collect();
    let filter = Filter::All;
    let pq = PreparedQuery {
        vector: &vector,
        terms: &terms,
        filter: &filter,
        limit: 3,
        rank: RankParams::default(),
    };
    let hits = index.search_prepared(&pq).unwrap();
    let top = hits
        .iter()
        .map(|h| h.url.to_owned())
        .next()
        .expect("at least one hit");
    top
}

#[test]
fn pipeline_indexes_and_resolves_by_keyword() {
    let index = build_index();
    let embedder = HashEmbedder::new(DIM);
    assert_eq!(index.len(), 3);

    // Exact-term queries should resolve to the right page (BM25 arm carries these).
    assert_eq!(
        search_top_url(&index, &embedder, "autopilot kubernetes cluster nodes"),
        "https://docs.example.com/compute/autopilot.md"
    );
    assert_eq!(
        search_top_url(&index, &embedder, "row access policies bigquery"),
        "https://docs.example.com/data/bigquery.md"
    );
    assert_eq!(
        search_top_url(&index, &embedder, "private service connect endpoints"),
        "https://docs.example.com/networking/psc"
    );
}

#[test]
fn save_open_roundtrip_is_deterministic_and_stable() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("kb.lnkr");

    let index = build_index();
    index.save(&path).unwrap();
    let bytes_a = std::fs::read(&path).unwrap();

    // Rebuilding from identical inputs must yield a byte-identical index.
    let index2 = build_index();
    index2.save(&path).unwrap();
    let bytes_b = std::fs::read(&path).unwrap();
    assert_eq!(bytes_a, bytes_b, "index must be byte-reproducible");

    // Reopened index resolves identically.
    let reopened = Index::open(&path).unwrap();
    let embedder = HashEmbedder::new(DIM);
    assert_eq!(
        search_top_url(&reopened, &embedder, "row access policies bigquery"),
        "https://docs.example.com/data/bigquery.md"
    );
}

#[test]
fn recrawl_dedups_and_updates() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("kb.lnkr");
    build_index().save(&path).unwrap();

    // Re-run the same crawl into the existing index: every link is unchanged.
    let embedder = HashEmbedder::new(DIM);
    let extractor = AutoExtractor;
    let mut builder = Index::open(&path).unwrap().into_builder().unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let mut added = 0;
    let mut unchanged = 0;
    for (url, kind, bytes) in corpus() {
        let resource = Resource::new(Url::parse(url).unwrap()).with_kind(kind);
        let descriptor = extractor.extract(&resource, bytes).unwrap();
        let vector = rt
            .block_on(embed_one(&embedder, &descriptor.embed_text))
            .unwrap();
        let doc = descriptor.into_document(resource.url.clone(), kind, xxh3_64(bytes), vector);
        match builder.upsert(doc).unwrap() {
            link_r::UpsertOutcome::Added => added += 1,
            link_r::UpsertOutcome::Unchanged => unchanged += 1,
            link_r::UpsertOutcome::Updated => {}
        }
    }
    assert_eq!(added, 0, "no new links on identical re-crawl");
    assert_eq!(unchanged, 3, "every link deduped as unchanged");
    assert_eq!(builder.len(), 3, "no duplicate links");
}
