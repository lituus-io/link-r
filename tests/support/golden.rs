// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Shared by the golden-fixture generator (run once against the released
//! 0.2.0 code) and `tests/golden_v020.rs` (run against every later version):
//! a fixed in-memory corpus ingested through the public facade, and a
//! textual dump of everything observable. Uses only API present in 0.2.0.

#![allow(dead_code)]

use bytes::Bytes;
use link_r::source::Source;
use link_r::{DocPayload, Page, Resource, ResourceKind, SourceRef};
use link_r::{Filter, LinkIndex};
use std::fmt::Write as _;
use url::Url;

pub const QUERIES: [&str; 5] = [
    "private service connect",
    "autopilot nodes",
    "row access policy",
    "kubernetes security",
    "zzz nothing",
];

/// (url, kind, body) — HTML pages link to each other so the graph is exercised.
pub fn corpus() -> Vec<(&'static str, ResourceKind, &'static str)> {
    vec![
        (
            "https://docs.example.com/networking/psc",
            ResourceKind::Html,
            "<html><head><title>Private Service Connect</title></head><body><h1>PSC</h1>\
             <p>Reach Google APIs and services privately over an internal IP using Private Service Connect.</p>\
             <a href=\"/compute/autopilot\">Autopilot</a> <a href=\"/data/bigquery\">BigQuery</a></body></html>",
        ),
        (
            "https://docs.example.com/compute/autopilot",
            ResourceKind::Html,
            "<html><head><title>GKE Autopilot</title></head><body><h1>Autopilot</h1>\
             <p>Autopilot is a mode of operation in GKE in which Google manages your cluster nodes, \
             scaling, and security for Kubernetes workloads.</p><a href=\"/networking/psc\">PSC</a></body></html>",
        ),
        (
            "https://docs.example.com/data/bigquery",
            ResourceKind::Html,
            "<html><head><title>BigQuery Row Access</title></head><body><h1>Row Access Policies</h1>\
             <p>BigQuery row-level security uses row access policies to restrict which rows a principal \
             can query in a table.</p><a href=\"/networking/psc\">PSC</a></body></html>",
        ),
        (
            "https://docs.example.com/guides/security.md",
            ResourceKind::Markdown,
            "---\ntitle: Security overview\ntags: [security, iam]\n---\n# Security\n\n\
             Kubernetes security, IAM principals and least privilege for workloads.\n",
        ),
    ]
}

/// A source over [`corpus`], so ingestion needs no network.
#[derive(Debug, Default)]
pub struct MemSource;

impl Source for MemSource {
    type Pages<'a> = futures::stream::Iter<std::vec::IntoIter<link_r::Result<Page<'a>>>>;

    fn kind(&self) -> &'static str {
        "mem"
    }

    fn discover<'a>(&'a self, _root: &'a SourceRef) -> Self::Pages<'a> {
        let pages: Vec<link_r::Result<Page<'a>>> = corpus()
            .into_iter()
            .map(|(url, kind, body)| {
                Ok(Page::new(
                    Resource::new(Url::parse(url).unwrap()).with_kind(kind),
                    DocPayload::Owned(Bytes::from_static(body.as_bytes())),
                ))
            })
            .collect();
        futures::stream::iter(pages)
    }
}

pub async fn ingest(index: &mut LinkIndex) {
    let root = SourceRef::http("https://docs.example.com/").unwrap();
    index
        .ingest_from(&MemSource, &root, 2, false)
        .await
        .unwrap();
    index.pin("https://docs.example.com/guides/").unwrap();
}

pub async fn dump(index: &LinkIndex) -> String {
    let mut o = String::new();
    writeln!(o, "len {}", index.len()).unwrap();
    for d in index.export().unwrap() {
        let mut edges: Vec<u64> = d.edges.iter().map(|k| k.0).collect();
        edges.sort_unstable();
        writeln!(
            o,
            "doc {} kind={:?} hash={:x} len={} title={:?} snippet={:?} lang={:?} tags={:?} pinned={} etag={:?} edges={:x?}",
            d.meta.url, d.meta.kind, d.meta.content_hash, d.meta.doc_len, d.meta.title, d.meta.snippet,
            d.meta.lang, d.meta.tags, d.meta.pinned, d.meta.etag, edges
        )
        .unwrap();
    }
    for q in QUERIES {
        writeln!(o, "query {q:?}").unwrap();
        for r in index.search_filtered(q, 10, &Filter::All).await.unwrap() {
            writeln!(o, "  {} {:.4} {:?} {:?}", r.url, r.score, r.title, r.tags).unwrap();
        }
        for r in index.search_ranked(q, 10, &Filter::All, 0.5).await.unwrap() {
            writeln!(o, "  boosted {} {:.4}", r.url, r.score).unwrap();
        }
    }
    for r in index
        .related("https://docs.example.com/networking/psc", 5)
        .unwrap()
    {
        writeln!(o, "related {} {:.4}", r.url, r.score).unwrap();
    }
    o
}
