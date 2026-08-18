// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The stateless conditional re-crawl, end to end through the facade.
//!
//! The scenario these tests exist for: a persistent backend absorbed a crawl,
//! discarded the link-r index entirely (that is the design — bodies and vectors
//! do not outlive the session), and later hands its stored ETags and edges back
//! to a *fresh* index via [`UpdateBuilder::validators`] / `known_edges`. An
//! unchanged site must then revalidate with zero body transfers, and — just as
//! important — the report must *say* the pages were checked, or the backend's
//! adaptive freshness can never learn from the crawl.
//!
//! Runs against a real loopback HTTP server (same posture as tests/security.rs):
//! the conditional behaviour under test lives in the real fetcher stack.

#![cfg(feature = "crawl")]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

use compact_str::CompactString;
use link_r::facade::LinkIndex;
use link_r::url_key::UrlKey;

/// A tiny conditional-GET server: per-path (etag, body), answering 304 when
/// `If-None-Match` matches and counting the bodies it actually serves.
struct EtagServer {
    port: u16,
    bodies_served: Arc<AtomicU32>,
}

fn serve(pages: HashMap<String, (String, String)>) -> EtagServer {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let bodies_served = Arc::new(AtomicU32::new(0));
    let counter = bodies_served.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            let path = req.split_whitespace().nth(1).unwrap_or("/").to_string();
            let inm = req
                .lines()
                .find_map(|l| {
                    l.strip_prefix("if-none-match: ")
                        .or_else(|| l.strip_prefix("If-None-Match: "))
                })
                .map(str::trim);
            let response = match pages.get(&path) {
                Some((etag, body)) => {
                    if inm == Some(etag.as_str()) {
                        format!("HTTP/1.1 304 Not Modified\r\netag: {etag}\r\nconnection: close\r\n\r\n")
                    } else {
                        counter.fetch_add(1, Ordering::Relaxed);
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\netag: {etag}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                    }
                }
                None => "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    .to_string(),
            };
            let _ = stream.write_all(response.as_bytes());
        }
    });
    EtagServer {
        port,
        bodies_served,
    }
}

fn page(links: &[&str], text: &str) -> String {
    let mut html = format!("<html><body><h1>{text}</h1><p>{text} body words for the index</p>");
    for l in links {
        html.push_str(&format!("<a href=\"{l}\">link</a>"));
    }
    html.push_str("</body></html>");
    html
}

#[tokio::test]
async fn a_fresh_index_with_supplied_seeds_revalidates_for_free() {
    let pages: HashMap<String, (String, String)> = [
        (
            "/docs".to_string(),
            ("\"r1\"".to_string(), page(&["/docs/a"], "root")),
        ),
        (
            "/docs/a".to_string(),
            ("\"a1\"".to_string(), page(&["/docs/b"], "alpha")),
        ),
        (
            "/docs/b".to_string(),
            ("\"b1\"".to_string(), page(&[], "bravo")),
        ),
    ]
    .into();
    let server = serve(pages);
    let base = format!("http://127.0.0.1:{}/docs", server.port);

    // Session one: a plain crawl into an in-memory index. Three bodies.
    let mut first = LinkIndex::in_memory().unwrap();
    let report = first.update(&base).depth(3).run().await.unwrap();
    assert_eq!(report.added, 3, "{report:?}");
    assert_eq!(server.bodies_served.load(Ordering::Relaxed), 3);

    // The backend absorbs what it needs, then the index is DROPPED — this is
    // the "absorb and discard" design the seeds exist for.
    let url_of = |p: &str| format!("http://127.0.0.1:{}{p}", server.port);
    let validators: Vec<(UrlKey, CompactString)> = [
        ("/docs", "\"r1\""),
        ("/docs/a", "\"a1\""),
        ("/docs/b", "\"b1\""),
    ]
    .into_iter()
    .map(|(p, e)| (UrlKey::parse(&url_of(p)).unwrap(), CompactString::from(e)))
    .collect();
    let known_edges: Vec<(UrlKey, Vec<url::Url>)> =
        [("/docs", vec!["/docs/a"]), ("/docs/a", vec!["/docs/b"])]
            .into_iter()
            .map(|(p, kids)| {
                (
                    UrlKey::parse(&url_of(p)).unwrap(),
                    kids.into_iter()
                        .map(|k| url::Url::parse(&url_of(k)).unwrap())
                        .collect(),
                )
            })
            .collect();
    drop(first);

    // Session two: a FRESH index that has never seen this site, seeded from
    // the backend's records. Everything revalidates; zero bodies transfer;
    // and the report records all three checks so freshness can learn.
    let mut second = LinkIndex::in_memory().unwrap();
    let report = second
        .update(&base)
        .depth(3)
        .validators(validators.clone())
        .known_edges(known_edges.clone())
        .run()
        .await
        .unwrap();
    assert_eq!(
        server.bodies_served.load(Ordering::Relaxed),
        3,
        "no new bodies transferred"
    );
    assert_eq!(report.added, 0, "{report:?}");
    assert_eq!(
        report.unchanged, 3,
        "all three revalidations must be reported: {report:?}"
    );
    assert_eq!(report.pages.len(), 3);
    assert!(
        report
            .pages
            .iter()
            .all(|p| p.change == link_r::facade::PageChange::Unchanged),
        "{report:?}"
    );

    // The index's own knowledge wins over stale external seeds: crawl again
    // with a deliberately WRONG external validator for /docs/b while the index
    // itself now knows nothing (fresh again) except that seed — the wrong etag
    // forces exactly one body.
    let mut wrong = validators.clone();
    wrong[2].1 = CompactString::from("\"stale\"");
    let mut third = LinkIndex::in_memory().unwrap();
    let report = third
        .update(&base)
        .depth(3)
        .validators(wrong)
        .known_edges(known_edges)
        .run()
        .await
        .unwrap();
    assert_eq!(
        server.bodies_served.load(Ordering::Relaxed),
        4,
        "only /docs/b re-fetched"
    );
    assert_eq!(report.added, 1, "{report:?}");
    assert_eq!(report.unchanged, 2, "{report:?}");
}

#[tokio::test]
async fn within_a_session_the_index_own_etags_win_over_supplied_seeds() {
    let pages: HashMap<String, (String, String)> = [(
        "/docs".to_string(),
        ("\"v1\"".to_string(), page(&[], "solo")),
    )]
    .into();
    let server = serve(pages);
    let base = format!("http://127.0.0.1:{}/docs", server.port);

    let mut idx = LinkIndex::in_memory().unwrap();
    idx.update(&base).depth(0).run().await.unwrap();
    assert_eq!(server.bodies_served.load(Ordering::Relaxed), 1);

    // Re-crawl the SAME index, supplying a wrong external validator. The
    // index's own (correct) etag must take precedence: still no new body.
    let key = UrlKey::parse(&base).unwrap();
    let report = idx
        .update(&base)
        .depth(0)
        .validators([(key, CompactString::from("\"bogus\""))])
        .run()
        .await
        .unwrap();
    assert_eq!(
        server.bodies_served.load(Ordering::Relaxed),
        1,
        "the index's fresher etag must win over the stale seed"
    );
    assert_eq!(report.unchanged, 1, "{report:?}");
}
