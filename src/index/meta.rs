// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Per-document metadata: the resolvable URL plus the structured columns used for
//! display ([`DocMeta::title`], [`DocMeta::snippet`]) and exact-match filtering
//! ([`DocMeta::kind`], [`DocMeta::tags`], [`DocMeta::lang`], the URL path).
//!
//! Decoded into owned records on open (cheap — hundreds of short strings), while
//! the perf-critical dense blob stays zero-copy. Filter predicates evaluate to a
//! [`RoaringBitmap`] of allowed doc ids that prefilters both retrieval arms.

use crate::error::{Error, Result};
use crate::index::bytesio::{put_len_prefixed, put_u32, put_u64, put_u8, Reader};
use crate::resource::{DocId, ResourceKind};
use crate::url_key::UrlKey;
use compact_str::CompactString;
use roaring::RoaringBitmap;
use smallvec::SmallVec;

/// Metadata for one indexed document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocMeta {
    /// The canonical, resolvable remote URL.
    pub url: String,
    /// The dedup/uniqueness key for [`DocMeta::url`].
    pub url_key: UrlKey,
    /// Content class.
    pub kind: ResourceKind,
    /// `xxh3` of the source body, for change detection on re-crawl.
    pub content_hash: u64,
    /// Token count (BM25 document length).
    pub doc_len: u32,
    /// Display title, if extracted.
    pub title: Option<CompactString>,
    /// A short display snippet.
    pub snippet: CompactString,
    /// Detected language tag, if any.
    pub lang: Option<CompactString>,
    /// Structured tags for filtering.
    pub tags: SmallVec<[CompactString; 4]>,
    /// Wall-clock milliseconds since the Unix epoch when this document was last
    /// fetched. `0` for legacy files (treated as always stale). Drives TTL refresh
    /// and the [`Filter::FetchedBefore`]/[`Filter::FetchedAfter`] range filters.
    pub fetched_at_ms: u64,
    /// Whether this document is pinned — exempt from every TTL / max-age eviction.
    pub pinned: bool,
    /// The source `ETag`, used for conditional refresh (`If-None-Match`).
    pub etag: Option<CompactString>,
}

impl DocMeta {
    /// The URL path component (without query or fragment), for path-prefix
    /// filtering. Returns `"/"` when the URL has no path.
    #[must_use]
    pub fn path(&self) -> &str {
        // Strip "scheme://host" cheaply without re-parsing, then trim ?query/#frag.
        let after_host = match self.url.split_once("://") {
            Some((_, rest)) => rest.find('/').map_or("", |i| &rest[i..]),
            None => return &self.url,
        };
        if after_host.is_empty() {
            return "/";
        }
        let end = after_host.find(['?', '#']).unwrap_or(after_host.len());
        &after_host[..end]
    }
}

/// Encode all documents' metadata into the `DocMeta` section payload. Always
/// writes the freshness columns; the writer sets `flags::META_FRESHNESS` so the
/// reader knows to expect them.
#[must_use]
pub fn encode(docs: &[DocMeta]) -> Vec<u8> {
    let mut buf = Vec::new();
    put_u32(&mut buf, docs.len() as u32);
    for d in docs {
        put_len_prefixed(&mut buf, d.url.as_bytes());
        put_u64(&mut buf, d.url_key.raw());
        put_u8(&mut buf, d.kind.as_tag());
        put_u64(&mut buf, d.content_hash);
        put_u32(&mut buf, d.doc_len);
        put_opt_str(&mut buf, d.title.as_deref());
        put_len_prefixed(&mut buf, d.snippet.as_bytes());
        put_opt_str(&mut buf, d.lang.as_deref());
        put_u32(&mut buf, d.tags.len() as u32);
        for tag in &d.tags {
            put_len_prefixed(&mut buf, tag.as_bytes());
        }
        // Freshness columns (gated on read by flags::META_FRESHNESS).
        put_u64(&mut buf, d.fetched_at_ms);
        put_u8(&mut buf, u8::from(d.pinned));
        put_opt_str(&mut buf, d.etag.as_deref());
    }
    buf
}

/// Decode the `DocMeta` section payload into owned records. `has_freshness` comes
/// from `flags::META_FRESHNESS`: when clear (legacy files), the freshness columns
/// are absent and default to stale/unpinned/no-etag.
pub fn decode(bytes: &[u8], has_freshness: bool) -> Result<Vec<DocMeta>> {
    let mut r = Reader::new(bytes);
    let count = r.u32()? as usize;
    // Cap pre-allocation by remaining bytes: a malicious count cannot trigger a
    // huge allocation (each record consumes ≥1 byte, so the loop errors out first).
    let mut docs = Vec::with_capacity(count.min(r.remaining()));
    for _ in 0..count {
        let url = r.str()?.to_owned();
        let url_key = UrlKey(r.u64()?);
        let kind = ResourceKind::from_tag(r.u8()?);
        let content_hash = r.u64()?;
        let doc_len = r.u32()?;
        let title = get_opt_str(&mut r)?;
        let snippet = CompactString::from(r.str()?);
        let lang = get_opt_str(&mut r)?;
        let tag_count = r.u32()? as usize;
        let mut tags = SmallVec::new();
        for _ in 0..tag_count {
            tags.push(CompactString::from(r.str()?));
        }
        let (fetched_at_ms, pinned, etag) = if has_freshness {
            let fetched_at_ms = r.u64()?;
            let pinned = match r.u8()? {
                0 => false,
                1 => true,
                _ => return Err(Error::format("invalid pinned flag")),
            };
            let etag = get_opt_str(&mut r)?;
            (fetched_at_ms, pinned, etag)
        } else {
            (0, false, None)
        };
        docs.push(DocMeta {
            url,
            url_key,
            kind,
            content_hash,
            doc_len,
            title,
            snippet,
            lang,
            tags,
            fetched_at_ms,
            pinned,
            etag,
        });
    }
    if r.remaining() != 0 {
        return Err(Error::format("trailing bytes in DocMeta section"));
    }
    Ok(docs)
}

fn put_opt_str(buf: &mut Vec<u8>, s: Option<&str>) {
    match s {
        Some(v) => {
            put_u8(buf, 1);
            put_len_prefixed(buf, v.as_bytes());
        }
        None => put_u8(buf, 0),
    }
}

fn get_opt_str(r: &mut Reader<'_>) -> Result<Option<CompactString>> {
    match r.u8()? {
        0 => Ok(None),
        1 => Ok(Some(CompactString::from(r.str()?))),
        _ => Err(Error::format("invalid optional-string flag")),
    }
}

/// A structured metadata filter applied before ranking.
#[derive(Clone, Debug, Default)]
pub enum Filter {
    /// Match every document (the default).
    #[default]
    All,
    /// Documents carrying this tag.
    Tag(CompactString),
    /// Documents with this language tag.
    Lang(CompactString),
    /// Documents whose URL path starts with this prefix.
    PathPrefix(CompactString),
    /// Documents of this content kind.
    Kind(ResourceKind),
    /// Documents last fetched strictly before this Unix-epoch millisecond (i.e.
    /// older than the cutoff — useful for "stale since" queries).
    FetchedBefore(u64),
    /// Documents last fetched at or after this Unix-epoch millisecond.
    FetchedAfter(u64),
    /// Conjunction.
    And(Box<Filter>, Box<Filter>),
    /// Disjunction.
    Or(Box<Filter>, Box<Filter>),
    /// Negation.
    Not(Box<Filter>),
}

impl Filter {
    /// Whether this filter is the trivial match-all (lets retrieval skip prefiltering).
    #[must_use]
    pub fn is_all(&self) -> bool {
        matches!(self, Filter::All)
    }

    fn matches(&self, doc: &DocMeta) -> bool {
        match self {
            Filter::All => true,
            Filter::Tag(t) => doc.tags.iter().any(|x| x == t),
            Filter::Lang(l) => doc.lang.as_deref() == Some(l.as_str()),
            Filter::PathPrefix(p) => doc.path().starts_with(p.as_str()),
            Filter::Kind(k) => doc.kind == *k,
            Filter::FetchedBefore(t) => doc.fetched_at_ms < *t,
            Filter::FetchedAfter(t) => doc.fetched_at_ms >= *t,
            Filter::And(a, b) => a.matches(doc) && b.matches(doc),
            Filter::Or(a, b) => a.matches(doc) || b.matches(doc),
            Filter::Not(a) => !a.matches(doc),
        }
    }

    /// Evaluate over all documents, returning the set of allowed doc ids. Returns
    /// `None` for [`Filter::All`] (meaning "no restriction" — let retrieval scan
    /// everything without building a bitmap).
    #[must_use]
    pub fn evaluate(&self, docs: &[DocMeta]) -> Option<RoaringBitmap> {
        if self.is_all() {
            return None;
        }
        let mut allowed = RoaringBitmap::new();
        for (i, doc) in docs.iter().enumerate() {
            if self.matches(doc) {
                allowed.insert(i as DocId);
            }
        }
        Some(allowed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    fn doc(url: &str, kind: ResourceKind, lang: Option<&str>, tags: &[&str]) -> DocMeta {
        let parsed = Url::parse(url).unwrap();
        DocMeta {
            url: url.to_owned(),
            url_key: UrlKey::from_url(&parsed),
            kind,
            content_hash: 0xABCD,
            doc_len: 10,
            title: Some(CompactString::from("Title")),
            snippet: CompactString::from("snippet text"),
            lang: lang.map(CompactString::from),
            tags: tags.iter().map(|t| CompactString::from(*t)).collect(),
            fetched_at_ms: 1000,
            pinned: false,
            etag: Some(CompactString::from("W/\"abc\"")),
        }
    }

    fn corpus() -> Vec<DocMeta> {
        vec![
            doc(
                "https://x.dev/docs/a",
                ResourceKind::Markdown,
                Some("en"),
                &["guide"],
            ),
            doc(
                "https://x.dev/blog/b",
                ResourceKind::Html,
                Some("fr"),
                &["news"],
            ),
            doc(
                "https://x.dev/docs/c",
                ResourceKind::Html,
                Some("en"),
                &["guide", "ref"],
            ),
        ]
    }

    #[test]
    fn encode_decode_roundtrips() {
        let docs = corpus();
        let bytes = encode(&docs);
        let decoded = decode(&bytes, true).unwrap();
        assert_eq!(decoded, docs);
    }

    #[test]
    fn legacy_decode_defaults_freshness() {
        // A record encoded without the freshness columns (old layout) must decode
        // with default freshness when the flag is clear.
        let mut buf = Vec::new();
        put_u32(&mut buf, 1);
        put_len_prefixed(&mut buf, b"https://x.dev/a");
        put_u64(&mut buf, 0);
        put_u8(&mut buf, ResourceKind::Html.as_tag());
        put_u64(&mut buf, 0xABCD);
        put_u32(&mut buf, 10);
        put_opt_str(&mut buf, None); // title
        put_len_prefixed(&mut buf, b"snip"); // snippet
        put_opt_str(&mut buf, None); // lang
        put_u32(&mut buf, 0); // tags
        let decoded = decode(&buf, false).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].fetched_at_ms, 0);
        assert!(!decoded[0].pinned);
        assert_eq!(decoded[0].etag, None);
    }

    #[test]
    fn decode_rejects_truncation() {
        let bytes = encode(&corpus());
        assert!(decode(&bytes[..bytes.len() - 3], true).is_err());
    }

    #[test]
    fn decode_rejects_huge_count_without_oom() {
        // Regression for a fuzz-found OOM: a 269M doc count with only 4 bytes of
        // input must error (capped pre-allocation), not allocate gigabytes.
        let bytes = [0xff, 0xff, 0x0a, 0x10];
        assert!(decode(&bytes, true).is_err());
    }

    #[test]
    fn fetched_range_filters() {
        let mut docs = corpus();
        docs[0].fetched_at_ms = 100;
        docs[1].fetched_at_ms = 500;
        docs[2].fetched_at_ms = 900;
        let before = Filter::FetchedBefore(500).evaluate(&docs).unwrap();
        assert!(before.contains(0) && !before.contains(1) && !before.contains(2));
        let after = Filter::FetchedAfter(500).evaluate(&docs).unwrap();
        assert!(!after.contains(0) && after.contains(1) && after.contains(2));
    }

    #[test]
    fn path_extraction() {
        let d = doc("https://x.dev/docs/a?q=1", ResourceKind::Html, None, &[]);
        assert_eq!(d.path(), "/docs/a");
        let root = doc("https://x.dev", ResourceKind::Html, None, &[]);
        assert_eq!(root.path(), "/");
    }

    #[test]
    fn filter_all_is_unrestricted() {
        assert!(Filter::All.evaluate(&corpus()).is_none());
    }

    #[test]
    fn tag_and_kind_filters() {
        let docs = corpus();
        let guide = Filter::Tag(CompactString::from("guide"))
            .evaluate(&docs)
            .unwrap();
        assert_eq!(guide.len(), 2); // docs 0 and 2
        let html = Filter::Kind(ResourceKind::Html).evaluate(&docs).unwrap();
        assert_eq!(html.len(), 2); // docs 1 and 2
    }

    #[test]
    fn path_prefix_filter() {
        let docs = corpus();
        let f = Filter::PathPrefix(CompactString::from("/docs/"));
        let allowed = f.evaluate(&docs).unwrap();
        assert!(allowed.contains(0));
        assert!(!allowed.contains(1));
        assert!(allowed.contains(2));
    }

    #[test]
    fn boolean_combinators() {
        let docs = corpus();
        // English guides under /docs/
        let f = Filter::And(
            Box::new(Filter::Lang(CompactString::from("en"))),
            Box::new(Filter::Tag(CompactString::from("guide"))),
        );
        let allowed = f.evaluate(&docs).unwrap();
        assert_eq!(allowed.len(), 2);

        let not_html = Filter::Not(Box::new(Filter::Kind(ResourceKind::Html)));
        assert_eq!(not_html.evaluate(&docs).unwrap().len(), 1); // only the markdown doc
    }
}
