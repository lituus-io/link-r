// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The GitHub tree-API sync, end to end through the facade, against a
//! loopback fake GitHub (tree JSON on the "API host", file bytes on the
//! "raw host" — one server playing both roles via its URL paths).
//!
//! The properties that make this source worth having are asserted as counts,
//! not vibes: an unchanged repository costs exactly ONE request; a one-file
//! change costs exactly one tree call plus one blob fetch; the unchanged
//! remainder is reported so a freshness tracker learns of every check.

#![cfg(all(feature = "crawl", feature = "github"))]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use link_r::facade::{LinkIndex, PageChange};
use link_r::UrlKey;

/// A fake GitHub: `/repos/{o}/{r}` (default branch), `/repos/{o}/{r}/git/trees/…`
/// (the tree), `/raw/{o}/{r}/{ref}/{path}` (blob bytes). Counts tree calls and
/// blob fetches, and records every Authorization header it sees.
struct FakeGithub {
    port: u16,
    tree_calls: Arc<AtomicU32>,
    blob_fetches: Arc<AtomicU32>,
    auth_seen: Arc<Mutex<Vec<String>>>,
}

type Files = HashMap<String, (String, String)>; // path -> (sha, body)

fn tree_json(files: &Files, truncated: bool) -> String {
    let entries: Vec<String> = files
        .iter()
        .map(|(path, (sha, body))| {
            format!(
                r#"{{"path":"{path}","mode":"100644","type":"blob","sha":"{sha}","size":{}}}"#,
                body.len()
            )
        })
        .collect();
    format!(
        r#"{{"sha":"root","truncated":{truncated},"tree":[{}]}}"#,
        entries.join(",")
    )
}

fn serve(files: Arc<Mutex<Files>>, truncated: bool) -> FakeGithub {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let tree_calls = Arc::new(AtomicU32::new(0));
    let blob_fetches = Arc::new(AtomicU32::new(0));
    let auth_seen = Arc::new(Mutex::new(Vec::new()));
    let (tc, bf, auth) = (tree_calls.clone(), blob_fetches.clone(), auth_seen.clone());
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            let path = req.split_whitespace().nth(1).unwrap_or("/").to_string();
            if let Some(a) = req.lines().find_map(|l| {
                l.strip_prefix("authorization: ")
                    .or_else(|| l.strip_prefix("Authorization: "))
            }) {
                auth.lock().unwrap().push(format!("{path} {}", a.trim()));
            }
            let files = files.lock().unwrap();
            let respond = |body: &str, ctype: &str| {
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                )
            };
            let response = if path.contains("/git/trees/") {
                tc.fetch_add(1, Ordering::Relaxed);
                respond(&tree_json(&files, truncated), "application/json")
            } else if path.starts_with("/repos/") {
                respond(r#"{"default_branch":"main"}"#, "application/json")
            } else if let Some(rest) = path.strip_prefix("/raw/o/r/main/") {
                match files.get(rest) {
                    Some((_, body)) => {
                        bf.fetch_add(1, Ordering::Relaxed);
                        // Like the real raw host: everything is text/plain.
                        respond(body, "text/plain; charset=utf-8")
                    }
                    None => {
                        "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                            .into()
                    }
                }
            } else {
                "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".into()
            };
            let _ = stream.write_all(response.as_bytes());
        }
    });
    FakeGithub {
        port,
        tree_calls,
        blob_fetches,
        auth_seen,
    }
}

fn corpus() -> Files {
    [
        (
            "stacks/README.md".to_string(),
            (
                "sha-r1".to_string(),
                "# Stacks\n\nexample corpus index words".to_string(),
            ),
        ),
        (
            "stacks/big_query/tables/Pulumi.yaml".to_string(),
            (
                "sha-y1".to_string(),
                "name: bq-tables\nruntime: yaml\ndescription: bigquery table example".to_string(),
            ),
        ),
        (
            "stacks/big_query/tables/README.md".to_string(),
            (
                "sha-m1".to_string(),
                "# BigQuery Tables\n\nHow to declare a bigquery table stack".to_string(),
            ),
        ),
    ]
    .into()
}

fn bases(port: u16) -> (String, String) {
    (
        format!("http://127.0.0.1:{port}"),
        format!("http://127.0.0.1:{port}/raw"),
    )
}

#[tokio::test]
async fn unchanged_repo_costs_exactly_one_request() {
    let files = Arc::new(Mutex::new(corpus()));
    let gh = serve(files, false);
    let (api, raw) = bases(gh.port);

    // Session one: everything is new — one tree call, three blob fetches.
    let mut first = LinkIndex::in_memory().unwrap();
    let report = first
        .update_github("github:o/r@main//stacks")
        .unwrap()
        .bases(&api, &raw)
        .token("sekrit")
        .run()
        .await
        .unwrap();
    assert_eq!(report.added, 3, "{report:?}");
    assert_eq!(gh.tree_calls.load(Ordering::Relaxed), 1);
    assert_eq!(gh.blob_fetches.load(Ordering::Relaxed), 3);

    // The backend absorbs (url, blob-sha) pairs, then the index is DROPPED.
    let seeds: Vec<(UrlKey, compact_str::CompactString)> = first
        .export()
        .unwrap()
        .map(|d| {
            (
                d.meta.url_key,
                d.meta.etag.clone().expect("blob sha stored as etag"),
            )
        })
        .collect();
    assert_eq!(seeds.len(), 3);
    drop(first);

    // Session two: fresh index, seeded validators. Exactly ONE request total
    // (the tree), zero blob fetches, and all three checks are reported.
    let mut second = LinkIndex::in_memory().unwrap();
    let report = second
        .update_github("github:o/r@main//stacks")
        .unwrap()
        .bases(&api, &raw)
        .token("sekrit")
        .validators(seeds.clone())
        .run()
        .await
        .unwrap();
    assert_eq!(
        gh.tree_calls.load(Ordering::Relaxed),
        2,
        "one more tree call"
    );
    assert_eq!(
        gh.blob_fetches.load(Ordering::Relaxed),
        3,
        "ZERO new blob fetches"
    );
    assert_eq!(report.added, 0, "{report:?}");
    assert_eq!(
        report.unchanged, 3,
        "every skipped file is reported: {report:?}"
    );
    assert!(report
        .pages
        .iter()
        .all(|p| p.change == PageChange::Unchanged));

    // The token reached both loopback "hosts" (api paths and raw paths alike):
    // the fetch layer applied Bearer auth to every request we served.
    let auth = gh.auth_seen.lock().unwrap();
    assert!(auth
        .iter()
        .any(|a| a.contains("/git/trees/") && a.contains("Bearer sekrit")));
    assert!(auth
        .iter()
        .any(|a| a.contains("/raw/") && a.contains("Bearer sekrit")));
}

#[tokio::test]
async fn a_one_file_change_costs_one_tree_call_and_one_fetch() {
    let files = Arc::new(Mutex::new(corpus()));
    let gh = serve(files.clone(), false);
    let (api, raw) = bases(gh.port);

    let mut first = LinkIndex::in_memory().unwrap();
    first
        .update_github("github:o/r@main//stacks")
        .unwrap()
        .bases(&api, &raw)
        .run()
        .await
        .unwrap();
    let seeds: Vec<(UrlKey, compact_str::CompactString)> = first
        .export()
        .unwrap()
        .map(|d| (d.meta.url_key, d.meta.etag.clone().unwrap()))
        .collect();
    drop(first);
    let fetched_before = gh.blob_fetches.load(Ordering::Relaxed);

    // One file's blob SHA moves.
    files.lock().unwrap().insert(
        "stacks/big_query/tables/README.md".into(),
        (
            "sha-m2".into(),
            "# BigQuery Tables\n\nNow with partitioned table guidance".into(),
        ),
    );

    let mut second = LinkIndex::in_memory().unwrap();
    let report = second
        .update_github("github:o/r@main//stacks")
        .unwrap()
        .bases(&api, &raw)
        .validators(seeds)
        .run()
        .await
        .unwrap();
    assert_eq!(
        gh.blob_fetches.load(Ordering::Relaxed),
        fetched_before + 1,
        "exactly the changed file transfers"
    );
    assert_eq!(
        report.added, 1,
        "fresh index: the changed file arrives as added: {report:?}"
    );
    assert_eq!(report.unchanged, 2, "{report:?}");
    // The new content is really in the index.
    let hits = second
        .search("partitioned table guidance", 5)
        .await
        .unwrap();
    assert!(
        hits[0].url.ends_with("/stacks/big_query/tables/README.md"),
        "{hits:?}"
    );
}

#[tokio::test]
async fn a_bare_spec_resolves_the_default_branch_first() {
    let files = Arc::new(Mutex::new(corpus()));
    let gh = serve(files, false);
    let (api, raw) = bases(gh.port);

    let mut idx = LinkIndex::in_memory().unwrap();
    let report = idx
        .update_github("github:o/r") // no ref: needs the /repos/o/r lookup
        .unwrap()
        .bases(&api, &raw)
        .run()
        .await
        .unwrap();
    assert_eq!(report.added, 3, "{report:?}");
}

#[tokio::test]
async fn a_truncated_tree_is_a_typed_error_not_a_partial_index() {
    let files = Arc::new(Mutex::new(corpus()));
    let gh = serve(files, true); // server sets truncated: true
    let (api, raw) = bases(gh.port);

    let mut idx = LinkIndex::in_memory().unwrap();
    let err = idx
        .update_github("github:o/r@main//stacks")
        .unwrap()
        .bases(&api, &raw)
        .run()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("truncated"), "{err}");
    assert_eq!(idx.len(), 0, "a partial listing must index nothing");
}

#[tokio::test]
async fn depth_and_extension_filters_confine_the_sync() {
    let files = Arc::new(Mutex::new(corpus()));
    let gh = serve(files, false);
    let (api, raw) = bases(gh.port);

    // depth 0 → only files directly in stacks/ (the README).
    let mut idx = LinkIndex::in_memory().unwrap();
    let report = idx
        .update_github("github:o/r@main//stacks")
        .unwrap()
        .bases(&api, &raw)
        .depth(0)
        .run()
        .await
        .unwrap();
    assert_eq!(report.added, 1, "{report:?}");

    // .md filter → the two markdown files, not the yaml.
    let mut idx = LinkIndex::in_memory().unwrap();
    let report = idx
        .update_github("github:o/r@main//stacks")
        .unwrap()
        .bases(&api, &raw)
        .accept_extension("md")
        .run()
        .await
        .unwrap();
    assert_eq!(report.added, 2, "{report:?}");
}
