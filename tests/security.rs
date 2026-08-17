// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Security tests for credential handling on private-source crawls.
//!
//! Covers the two credential-leak defenses: transport-layer (reqwest strips
//! `Authorization` on a cross-host redirect) and application-layer (host-scoped
//! tokens are never sent off-host; canonicalization drops URL userinfo before it
//! is persisted or logged).
#![cfg(feature = "http")]

use link_r::auth::{AuthProvider, Credential, StaticTokenAuth};
use link_r::fetch::{FetchOptions, Fetcher, HttpFetcher};
use link_r::resource::Resource;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;
use url::Url;

/// Accept one HTTP/1.1 connection, capture whether it carried an `Authorization`
/// header, send `response`, and report the capture over `tx`.
fn serve_once(listener: TcpListener, response: String, tx: mpsc::Sender<bool>) {
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_lowercase();
            let had_auth = req.contains("authorization:");
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            let _ = tx.send(had_auth);
        } else {
            let _ = tx.send(false);
        }
    });
}

#[tokio::test]
async fn authorization_is_stripped_on_cross_host_redirect() {
    // Two listeners on loopback; the client reaches the first via the "localhost"
    // hostname and is redirected to the second via "127.0.0.1" — a host change, so
    // reqwest must drop the Authorization header on the second request.
    let a = TcpListener::bind("127.0.0.1:0").unwrap();
    let b = TcpListener::bind("127.0.0.1:0").unwrap();
    let a_port = a.local_addr().unwrap().port();
    let b_port = b.local_addr().unwrap().port();

    let (tx_a, rx_a) = mpsc::channel();
    let (tx_b, rx_b) = mpsc::channel();
    serve_once(
        a,
        format!("HTTP/1.1 301 Moved Permanently\r\nLocation: http://127.0.0.1:{b_port}/t\r\nContent-Length: 0\r\n\r\n"),
        tx_a,
    );
    serve_once(
        b,
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nok".to_string(),
        tx_b,
    );

    // An *unscoped* bearer token, so we are testing reqwest's transport-layer
    // stripping (the scoped-token defense is tested separately).
    let fetcher = HttpFetcher::new(StaticTokenAuth::bearer("SUPER-SECRET-PAT")).unwrap();
    let resource = Resource::new(Url::parse(&format!("http://localhost:{a_port}/")).unwrap());
    let fetched = fetcher.fetch(&resource, FetchOptions::default()).await;
    assert!(fetched.is_ok(), "redirect should be followed: {fetched:?}");

    let a_had_auth = rx_a
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let b_had_auth = rx_b
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert!(a_had_auth, "the original host must receive the token");
    assert!(
        !b_had_auth,
        "the token must NOT survive a cross-host redirect"
    );
}

#[test]
fn scoped_token_withheld_from_foreign_host() {
    // Application-layer defense: a token scoped to one host yields no credential
    // for any other host, so an in-scope-by-config link to a foreign host cannot
    // exfiltrate it.
    let auth = StaticTokenAuth::bearer_scoped("SECRET", "github.com");
    let foreign = Resource::new(Url::parse("https://raw.githubusercontent.com/x").unwrap());
    assert!(matches!(auth.credential(&foreign), Credential::Anonymous));
    let home = Resource::new(Url::parse("https://github.com/o/r").unwrap());
    assert!(matches!(auth.credential(&home), Credential::Bearer(_)));
}

#[test]
fn token_is_never_in_debug_output() {
    let secret = "ghp_TopSecretValue123";
    let auth = StaticTokenAuth::bearer_scoped(secret, "github.com");
    assert!(!format!("{auth:?}").contains(secret));
    let fetcher = HttpFetcher::new(auth).unwrap();
    assert!(!format!("{fetcher:?}").contains(secret));
}

#[tokio::test]
async fn persisted_index_never_contains_url_userinfo() {
    // A crawl URL carrying `user:password@` must have its credentials dropped by
    // canonicalization before the URL is stored — the saved index bytes must not
    // contain the password.
    use link_r::facade::LinkIndex;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("kb.lnkr");

    // Drive one document with a userinfo-bearing URL through the public builder.
    let mut idx = LinkIndex::in_memory().unwrap();
    // The in-memory helper isn't public, so exercise it via a tiny manual upsert
    // through the index builder API.
    use link_r::index::{Document, IndexBuilder};
    use link_r::metric::Metric;
    use link_r::resource::ResourceKind;
    let mut builder = IndexBuilder::new(1, Metric::Cosine, 0);
    builder
        .upsert(Document {
            url: Url::parse("https://alice:hunter2@secret.dev/docs/a").unwrap(),
            kind: ResourceKind::Html,
            content_hash: 1,
            title: None,
            snippet: "s".into(),
            lang: None,
            tags: Default::default(),
            terms: vec!["term".into()],
            vector: vec![1.0],
            edges: Vec::new(),
            fetched_at_ms: 0,
            etag: None,
            pinned: false,
        })
        .unwrap();
    builder.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let haystack = String::from_utf8_lossy(&bytes);
    assert!(
        !haystack.contains("hunter2"),
        "index must not persist URL password"
    );
    assert!(
        !haystack.contains("alice:hunter2"),
        "index must not persist URL userinfo"
    );
    let _ = &mut idx;
}
