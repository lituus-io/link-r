// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! URL canonicalization and the [`UrlKey`] dedup/uniqueness key.
//!
//! Every link in the index is keyed by its *canonical* URL, so a page reached
//! through equivalent URLs (`HTTP` vs `http`, default ports, reordered query
//! params, a trailing slash, a fragment) collapses to one entry — re-crawling
//! never duplicates a link, satisfying the index's uniqueness guarantee.
//!
//! # Canonicalization rules (deterministic)
//!
//! - lowercase scheme and host (paths stay case-sensitive),
//! - drop the default port (`:80` for http, `:443` for https),
//! - drop the fragment (`#...`),
//! - drop userinfo (`user:pass@`),
//! - sort query segments lexically (treated as opaque, not re-encoded),
//! - strip a single trailing `/` except on the root path.

use smallvec::SmallVec;
use url::Url;
use xxhash_rust::xxh3::Xxh3;

/// Emit the canonical form of `url` as a sequence of `&str` chunks — the single
/// source of truth shared by [`canonicalize`] (which concatenates the chunks into
/// a `String`) and [`UrlKey::from_url`] (which streams them through an `xxh3`
/// hasher). Because both consume the *same* emission, the key is identical to
/// `xxh3_64(canonicalize(url).as_bytes())` by construction — a property the
/// persisted on-disk keys depend on (see the `streaming_hash_matches_string`
/// test and the `tests/proptest.rs` equality property).
///
/// The canonicalization rules are the module-level ones: lowercase scheme/host,
/// drop default port / fragment / userinfo, strip a single trailing slash except
/// on root, sort opaque query segments.
fn canonical_chunks(url: &Url, sink: &mut impl FnMut(&str)) {
    // scheme + "://" (Url guarantees scheme is already lowercase).
    sink(url.scheme());
    sink("://");

    // host + non-default port. Userinfo intentionally dropped. The host is emitted
    // borrowed (zero-alloc) unless it contains ASCII uppercase, which `Url` already
    // normalizes away for domain hosts, so the owned path is effectively never hit.
    if let Some(host) = url.host_str() {
        if host.bytes().any(|b| b.is_ascii_uppercase()) {
            sink(&host.to_ascii_lowercase());
        } else {
            sink(host);
        }
    }
    if let Some(port) = url.port() {
        // `Url::port` already returns `None` for the scheme's default port.
        sink(":");
        let mut buf = [0u8; 5]; // u16 max is 65535 (5 digits)
        sink(fmt_u16(port, &mut buf));
    }

    // path, with a single trailing slash stripped except on root.
    let path = url.path();
    if path.len() > 1 && path.ends_with('/') {
        sink(&path[..path.len() - 1]);
    } else if path.is_empty() {
        sink("/");
    } else {
        sink(path);
    }

    // query: split on '&', sort opaque segments, rejoin. Deterministic & order-free.
    if let Some(query) = url.query() {
        if !query.is_empty() {
            let mut segs: SmallVec<[&str; 8]> =
                query.split('&').filter(|q| !q.is_empty()).collect();
            segs.sort_unstable();
            if !segs.is_empty() {
                sink("?");
                for (i, seg) in segs.iter().enumerate() {
                    if i > 0 {
                        sink("&");
                    }
                    sink(seg);
                }
            }
        }
    }

    // fragment intentionally dropped.
}

/// Format a `u16` into a stack buffer, returning the written decimal slice. No
/// allocation — used for the port in [`canonical_chunks`].
fn fmt_u16(mut v: u16, buf: &mut [u8; 5]) -> &str {
    if v == 0 {
        return "0";
    }
    let mut i = buf.len();
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    // Every written byte is an ASCII digit, so this slice is always valid UTF-8.
    std::str::from_utf8(&buf[i..]).unwrap_or("0")
}

/// Produce the canonical string form of a URL used for keying and dedup.
#[must_use]
pub fn canonicalize(url: &Url) -> String {
    let mut s = String::with_capacity(url.as_str().len());
    canonical_chunks(url, &mut |c| s.push_str(c));
    s
}

/// A 64-bit content-addressed key for a canonical URL.
///
/// Two URLs that canonicalize to the same string share a `UrlKey`; this is the
/// uniqueness key the index uses to upsert (never duplicate) links.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UrlKey(pub u64);

impl UrlKey {
    /// Compute the key from a parsed URL.
    ///
    /// Streams the canonical chunks straight through an `xxh3` hasher — no
    /// intermediate `String` — yielding a value identical to
    /// `xxh3_64(canonicalize(url).as_bytes())` (asserted by tests, and relied on
    /// by the persisted on-disk keys).
    #[must_use]
    pub fn from_url(url: &Url) -> Self {
        let mut h = Xxh3::new();
        canonical_chunks(url, &mut |c| h.update(c.as_bytes()));
        Self(h.digest())
    }

    /// Parse a URL string and compute its key.
    pub fn parse(url: &str) -> crate::Result<Self> {
        let parsed =
            Url::parse(url).map_err(|e| crate::Error::invalid_url(format!("{url:?}: {e}")))?;
        Ok(Self::from_url(&parsed))
    }

    /// The raw 64-bit value (for on-disk persistence).
    #[must_use]
    pub fn raw(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canon(s: &str) -> String {
        canonicalize(&Url::parse(s).unwrap())
    }

    #[test]
    fn lowercases_scheme_and_host_not_path() {
        assert_eq!(
            canon("HTTP://Example.COM/Docs/Intro"),
            "http://example.com/Docs/Intro"
        );
    }

    #[test]
    fn drops_default_ports() {
        assert_eq!(canon("http://x.dev:80/a"), "http://x.dev/a");
        assert_eq!(canon("https://x.dev:443/a"), "https://x.dev/a");
        assert_eq!(canon("https://x.dev:8443/a"), "https://x.dev:8443/a");
    }

    #[test]
    fn drops_fragment_and_userinfo() {
        assert_eq!(canon("https://x.dev/a#section-2"), "https://x.dev/a");
        assert_eq!(canon("https://user:pass@x.dev/a"), "https://x.dev/a");
    }

    #[test]
    fn sorts_query_segments() {
        assert_eq!(canon("https://x.dev/s?b=2&a=1"), "https://x.dev/s?a=1&b=2");
        // order-independence
        assert_eq!(
            canon("https://x.dev/s?a=1&b=2"),
            canon("https://x.dev/s?b=2&a=1")
        );
    }

    #[test]
    fn strips_trailing_slash_except_root() {
        assert_eq!(canon("https://x.dev/docs/"), "https://x.dev/docs");
        assert_eq!(canon("https://x.dev/"), "https://x.dev/");
        assert_eq!(canon("https://x.dev"), "https://x.dev/");
    }

    #[test]
    fn equivalent_urls_share_a_key() {
        let a = UrlKey::parse("HTTP://Example.com:80/docs/?b=2&a=1#frag").unwrap();
        let b = UrlKey::parse("http://example.com/docs?a=1&b=2").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn distinct_urls_differ() {
        let a = UrlKey::parse("https://x.dev/a").unwrap();
        let b = UrlKey::parse("https://x.dev/b").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn canonicalization_is_idempotent() {
        let once = canon("HTTP://Example.com:80/docs/?b=2&a=1#frag");
        let twice = canonicalize(&Url::parse(&once).unwrap());
        assert_eq!(once, twice);
    }

    #[test]
    fn streaming_hash_matches_string() {
        // The persisted key contract: from_url must equal xxh3_64(canonicalize()).
        use xxhash_rust::xxh3::xxh3_64;
        for s in [
            "HTTP://Example.com:80/docs/?b=2&a=1#frag",
            "https://x.dev",
            "https://x.dev/",
            "https://x.dev:8443/a/b/c?z=9&a=1&m=5",
            "http://user:pass@Host.EXAMPLE.org:8080/Path/",
            "https://x.dev/a?only=1",
            "https://sub.domain.example.com/deep/path/segment",
        ] {
            let u = Url::parse(s).unwrap();
            assert_eq!(
                UrlKey::from_url(&u).raw(),
                xxh3_64(canonicalize(&u).as_bytes()),
                "streaming hash diverged for {s:?}"
            );
        }
    }

    #[test]
    fn fmt_u16_matches_display() {
        for p in [0u16, 1, 9, 10, 80, 443, 8080, 8443, 65535] {
            let mut buf = [0u8; 5];
            assert_eq!(fmt_u16(p, &mut buf), p.to_string());
        }
    }
}
