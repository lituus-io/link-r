// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The permanent index of fixed defects, asserted through the public API.
//!
//! Several regressions are also pinned as unit tests beside the code that owns
//! them — the chunked size-cap bypass in `fetch/http.rs`, the fuzz-found OOMs in
//! `index/meta.rs` and `index/sparse.rs`, the `Retry-After` HTTP-date parser,
//! and the refresh-without-ETag loop in `facade.rs`. Those stay where they are:
//! a unit test guards the function, this suite guards the behaviour a caller
//! depends on, and the two fail for different reasons.

use link_r::extract::Descriptor;
use link_r::resource::ResourceKind;
use link_r::url_key::{canonicalize, UrlKey};
use link_r::Error;

// ---- retry classification -------------------------------------------------

/// A 403 used to be classified as a retriable rate limit, so a permanently
/// forbidden URL was retried on every pass and never reached the eviction path
/// that was supposed to remove it. Access failures are terminal; only genuinely
/// transient conditions may be retried.
#[test]
fn access_failures_are_terminal_and_transient_ones_are_not() {
    // Terminal: retrying cannot change the answer.
    for e in [
        Error::permission_denied("https://x.dev/a"), // 403
        Error::unauthenticated("https://x.dev/a"),   // 401
        Error::not_found("https://x.dev/a"),         // 404
        Error::not_modified("https://x.dev/a"),      // 304 — success, not failure
        Error::http(400, "bad request"),
        Error::http(410, "gone"),
    ] {
        assert!(!e.is_retriable(), "{e} must not be retried");
    }

    // Transient: the same request may well succeed later.
    for e in [
        Error::rate_limited(1_000),
        Error::http(408, "timeout"),
        Error::http(429, "slow down"),
        Error::http(500, "oops"),
        Error::http(502, "bad gateway"),
        Error::http(503, "unavailable"),
        Error::http(504, "gateway timeout"),
    ] {
        assert!(e.is_retriable(), "{e} must be retried");
    }
}

// ---- the shared foreign key -----------------------------------------------

/// The contract graph-r's foreign key rests on: the streaming hash and the
/// string form must agree by construction, because one is computed over a chunk
/// stream and the other by concatenating the same chunks. If these ever diverge,
/// every persisted key silently stops resolving.
#[test]
fn the_url_key_is_exactly_the_hash_of_the_canonical_string() {
    for s in [
        "HTTP://Example.com:80/docs/?b=2&a=1#frag",
        "https://x.dev",
        "https://x.dev/",
        "https://x.dev:8443/a/b/c?z=9&a=1&m=5",
        "http://user:pass@Host.EXAMPLE.org:8080/Path/",
        "https://sub.domain.example.com/deep/path/segment",
        "https://x.dev/a?only=1",
    ] {
        let u = url::Url::parse(s).unwrap();
        assert_eq!(
            UrlKey::from_url(&u).raw(),
            xxhash_rust::xxh3::xxh3_64(canonicalize(&u).as_bytes()),
            "streaming hash diverged from the canonical string for {s:?}"
        );
    }
}

/// Canonicalization must be idempotent, or a URL that round-trips through the
/// index acquires a second, different key.
#[test]
fn canonicalization_is_idempotent() {
    for s in [
        "HTTP://Example.com:80/docs/?b=2&a=1#frag",
        "https://x.dev",
        "https://x.dev/a/",
    ] {
        let once = canonicalize(&url::Url::parse(s).unwrap());
        let twice = canonicalize(&url::Url::parse(&once).unwrap());
        assert_eq!(once, twice, "canonicalization is not idempotent for {s:?}");
    }
}

// ---- edge truncation ------------------------------------------------------

/// Outbound edges are capped per document. The cap used to be applied *after*
/// sorting by key, so a hub page kept a hash-random slice of its links and the
/// graph lost exactly the prominent navigation edges it most needed. The cap is
/// now applied in document order — which is salience order — and the result is
/// sorted afterwards for deterministic storage.
#[test]
fn the_edge_cap_keeps_the_prominent_prefix_not_a_hash_random_slice() {
    const MAX_EDGES_PER_DOC: usize = 64;
    let page = url::Url::parse("https://x.dev/hub").unwrap();
    let links: Vec<compact_str::CompactString> = (0..300)
        .map(|i| compact_str::CompactString::from(format!("/t{i}")))
        .collect();

    let descriptor = Descriptor {
        links: links.clone(),
        ..Descriptor::default()
    };
    let doc = descriptor.into_document(page.clone(), ResourceKind::Html, 0, Vec::new());

    assert_eq!(
        doc.edges.len(),
        MAX_EDGES_PER_DOC,
        "edge set must be capped"
    );

    // Every retained edge must come from the first 64 links in document order.
    let prefix: std::collections::HashSet<UrlKey> = links
        .iter()
        .take(MAX_EDGES_PER_DOC)
        .map(|h| UrlKey::from_url(&page.join(h).unwrap()))
        .collect();
    assert!(
        doc.edges.iter().all(|k| prefix.contains(k)),
        "the cap dropped prominent links in favour of a hash-random slice"
    );

    // …and storage order is deterministic (sorted), independent of input order.
    let mut sorted = doc.edges.clone();
    sorted.sort_unstable();
    assert_eq!(doc.edges, sorted, "edges must be stored sorted");
}

/// Duplicate links on a page must collapse to one edge, keeping the first
/// occurrence, so a repeated nav link cannot crowd out distinct targets.
#[test]
fn duplicate_links_collapse_to_one_edge() {
    let page = url::Url::parse("https://x.dev/hub").unwrap();
    let links: Vec<compact_str::CompactString> = ["/a", "/b", "/a", "/b", "/c"]
        .iter()
        .map(|s| compact_str::CompactString::from(*s))
        .collect();
    let descriptor = Descriptor {
        links,
        ..Descriptor::default()
    };
    let doc = descriptor.into_document(page, ResourceKind::Html, 0, Vec::new());
    assert_eq!(doc.edges.len(), 3, "duplicates must collapse");
}

// ---- decoder hardening ----------------------------------------------------

/// Found by fuzzing the index loader.
///
/// Each section decoder used to trust its own length prefix to size an
/// allocation, so a four-byte input declaring 269 million documents reserved
/// gigabytes before discovering there was no data behind it. Every count is now
/// capped by the bytes actually remaining. Also pinned as unit tests beside each
/// decoder.
#[test]
fn hostile_section_counts_error_instead_of_allocating() {
    // A doc count of u32::MAX with no bodies behind it.
    assert!(link_r::index::graph::decode(&[0xff, 0xff, 0xff, 0xff]).is_err());
    assert!(link_r::index::meta::decode(&[0xff, 0xff, 0xff, 0xff], true).is_err());
    assert!(link_r::index::sparse::Bm25::from_bytes(&[0xff; 8]).is_err());

    // Truncated inputs of every prefix length must error, never panic.
    let hostile = [0xffu8; 32];
    for cut in 0..hostile.len() {
        let _ = link_r::index::graph::decode(&hostile[..cut]);
        let _ = link_r::index::meta::decode(&hostile[..cut], true);
        let _ = link_r::index::sparse::Bm25::from_bytes(&hostile[..cut]);
    }
}

/// The graph section round-trips exactly, including empty edge lists — a
/// document with no outbound links must stay distinguishable from a missing one.
#[test]
fn the_edge_section_round_trips_including_empty_lists() {
    let edges = vec![vec![UrlKey(10), UrlKey(20)], vec![], vec![UrlKey(30)]];
    let bytes = link_r::index::graph::encode(&edges);
    assert_eq!(link_r::index::graph::decode(&bytes).unwrap(), edges);
    // Trailing bytes are corruption, not padding.
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(link_r::index::graph::decode(&trailing).is_err());
}
