// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Extraction: distill a fetched document into the small descriptor the index
//! stores.
//!
//! An [`Extractor`] turns raw bytes into a [`Descriptor`]: a title, keywords/topics,
//! tags, a display snippet, the distilled `embed_text` that gets embedded, the
//! BM25 terms, and (for HTML) the outbound links the crawler follows. The body
//! itself never enters the index. [`AutoExtractor`] routes by [`ResourceKind`].

#[cfg(feature = "html")]
pub mod html;
pub mod markdown;

use crate::error::Result;
use crate::resource::{Resource, ResourceKind};
use crate::text;
use compact_str::CompactString;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Maximum number of BM25 terms retained per document (caps huge pages).
const MAX_BM25_TERMS: usize = 4096;
/// Maximum keywords surfaced per document.
const MAX_KEYWORDS: usize = 8;
/// Minimum token length to be considered a keyword.
const MIN_KEYWORD_LEN: usize = 3;
/// Snippet length in characters.
const SNIPPET_CHARS: usize = 180;
/// Maximum persisted outbound edges (knowledge-graph paths) per document.
const MAX_EDGES_PER_DOC: usize = 64;
/// Body characters folded into the embed text. Headings and keywords already
/// front-load the high-signal terms; a wider body window improves recall for a
/// knowledge base without materially growing the single per-document vector.
const EMBED_BODY_CHARS: usize = 1000;
/// Maximum headings retained per document.
const MAX_HEADINGS: usize = 8;

/// The distilled, indexable form of a document.
#[derive(Clone, Debug, Default)]
pub struct Descriptor {
    /// Display title, if found.
    pub title: Option<CompactString>,
    /// Top keywords/topics (most frequent discriminative terms).
    pub keywords: SmallVec<[CompactString; MAX_KEYWORDS]>,
    /// Section headings (H1–H3 / ATX), high-signal for retrieval.
    pub headings: SmallVec<[CompactString; MAX_HEADINGS]>,
    /// Heading depths parallel to `headings` (1 = H1 …); empty when the
    /// extractor cannot tell (code/plain text), in which case consumers fall
    /// back to position-derived depths.
    pub heading_levels: SmallVec<[u8; MAX_HEADINGS]>,
    /// Structured tags (currently URL path components) for filtering.
    pub tags: SmallVec<[CompactString; 8]>,
    /// Detected language tag, if any.
    pub lang: Option<CompactString>,
    /// A short display snippet.
    pub snippet: CompactString,
    /// The distilled text handed to the embedder.
    pub embed_text: String,
    /// Normalized BM25 terms (also defines document length).
    pub bm25_terms: Vec<CompactString>,
    /// Outbound link hrefs (raw, unresolved) discovered in the document.
    pub links: Vec<CompactString>,
}

impl Descriptor {
    /// Whether this descriptor carries no indexable content (no body/title terms).
    /// URL-path tags alone do not count as content.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bm25_terms.is_empty()
    }

    /// Convert into an index [`Document`](crate::index::Document), pairing with a
    /// resolved URL, content hash, and embedding vector.
    ///
    /// Freshness fields (`fetched_at_ms`, `etag`, `pinned`) default to
    /// unset/unpinned; callers that maintain a persisted knowledge base (the
    /// facade) set them on the returned document before upserting.
    #[must_use]
    pub fn into_document(
        self,
        url: url::Url,
        kind: ResourceKind,
        content_hash: u64,
        vector: Vec<f32>,
    ) -> crate::index::Document {
        // Resolve outbound hrefs against the page URL into canonical-URL keys — the
        // persisted knowledge-graph edges. Deduplicated in document order and
        // capped *before* sorting, so a hub page keeps its first (most
        // prominent) links rather than a hash-random subset, then sorted for
        // deterministic storage.
        let mut seen =
            std::collections::HashSet::with_capacity(self.links.len().min(MAX_EDGES_PER_DOC));
        let mut edges: Vec<crate::url_key::UrlKey> = Vec::new();
        for href in &self.links {
            let Ok(target) = url.join(href) else { continue };
            if target.scheme() != "http" && target.scheme() != "https" {
                continue;
            }
            let key = crate::url_key::UrlKey::from_url(&target);
            if seen.insert(key) {
                edges.push(key);
                if edges.len() == MAX_EDGES_PER_DOC {
                    break;
                }
            }
        }
        edges.sort_unstable();

        crate::index::Document {
            url,
            kind,
            content_hash,
            title: self.title,
            snippet: self.snippet,
            lang: self.lang,
            tags: self.tags.into_iter().take(4).collect(),
            terms: self.bm25_terms,
            vector,
            edges,
            fetched_at_ms: 0,
            etag: None,
            pinned: false,
        }
    }
}

/// Distill an extracted document into a [`Descriptor`]. Shared by every
/// content-type extractor so keywords/snippets/embed-text are produced uniformly.
/// `headings` (H1–H3 / ATX) and `lang` are optional high-signal inputs; code and
/// plain text pass an empty heading slice and `None`.
#[must_use]
pub fn distill(
    resource: &Resource,
    title: Option<CompactString>,
    headings: &[CompactString],
    body: &str,
    links: Vec<CompactString>,
    lang: Option<CompactString>,
) -> Descriptor {
    // BM25 terms: normalized title + heading tokens (high signal) then body, capped.
    let mut bm25_terms: Vec<CompactString> = Vec::new();
    if let Some(t) = &title {
        bm25_terms.extend(text::normalized_tokens(t));
    }
    for h in headings {
        bm25_terms.extend(text::normalized_tokens(h));
    }
    bm25_terms.extend(text::normalized_tokens(body).take(MAX_BM25_TERMS));
    bm25_terms.truncate(MAX_BM25_TERMS);

    // Keywords: term frequency over non-stopword, sufficiently-long tokens.
    let mut freq: HashMap<CompactString, u32> = HashMap::new();
    for term in &bm25_terms {
        if term.len() >= MIN_KEYWORD_LEN && !text::is_stopword(term) {
            *freq.entry(term.clone()).or_insert(0) += 1;
        }
    }
    let mut ranked: Vec<(CompactString, u32)> = freq.into_iter().collect();
    // Deterministic: by descending frequency, then term ascending.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let keywords: SmallVec<[CompactString; MAX_KEYWORDS]> = ranked
        .into_iter()
        .take(MAX_KEYWORDS)
        .map(|(t, _)| t)
        .collect();

    // Tags: URL path components.
    let tags = path_tags(resource);

    // Snippet: first SNIPPET_CHARS of whitespace-collapsed body.
    let snippet = make_snippet(body);

    // Headings retained for the descriptor (also folded into embed text below).
    let heading_vec: SmallVec<[CompactString; MAX_HEADINGS]> =
        headings.iter().take(MAX_HEADINGS).cloned().collect();

    // Embed text: title + headings + keywords + path tail + body head.
    let mut embed_text = String::new();
    if let Some(t) = &title {
        embed_text.push_str(t);
        embed_text.push(' ');
    }
    for h in &heading_vec {
        embed_text.push_str(h);
        embed_text.push(' ');
    }
    for kw in &keywords {
        embed_text.push_str(kw);
        embed_text.push(' ');
    }
    for tag in &tags {
        embed_text.push_str(tag);
        embed_text.push(' ');
    }
    push_char_prefix(&mut embed_text, body, EMBED_BODY_CHARS);

    Descriptor {
        title,
        keywords,
        headings: heading_vec,
        heading_levels: SmallVec::new(),
        tags,
        lang,
        snippet,
        embed_text,
        bm25_terms,
        links,
    }
}

fn path_tags(resource: &Resource) -> SmallVec<[CompactString; 8]> {
    let mut tags: SmallVec<[CompactString; 8]> = SmallVec::new();
    for seg in resource.url.path().split('/') {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }
        // strip an extension from the last segment
        let base = seg.rsplit_once('.').map_or(seg, |(b, _)| b);
        if base.is_empty() {
            continue;
        }
        let norm = text::normalize(base);
        if norm.len() >= 2 && !tags.contains(&norm) {
            tags.push(norm);
        }
    }
    tags.truncate(8);
    tags
}

fn make_snippet(body: &str) -> CompactString {
    let mut snippet = String::with_capacity(SNIPPET_CHARS);
    let mut prev_space = false;
    for ch in body.chars() {
        if snippet.chars().count() >= SNIPPET_CHARS {
            break;
        }
        if ch.is_whitespace() {
            if !prev_space && !snippet.is_empty() {
                snippet.push(' ');
                prev_space = true;
            }
        } else {
            snippet.push(ch);
            prev_space = false;
        }
    }
    CompactString::from(snippet.trim())
}

fn push_char_prefix(out: &mut String, body: &str, max_chars: usize) {
    let mut prev_space = false;
    let mut count = 0;
    for ch in body.chars() {
        if count >= max_chars {
            break;
        }
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
                count += 1;
            }
        } else {
            out.push(ch);
            prev_space = false;
            count += 1;
        }
    }
}

/// Extracts a [`Descriptor`] from raw document bytes.
pub trait Extractor: Send + Sync {
    /// Whether this extractor handles the given content kind.
    fn handles(&self, kind: ResourceKind) -> bool;

    /// Extract a descriptor from `bytes`.
    fn extract(&self, resource: &Resource, bytes: &[u8]) -> Result<Descriptor>;
}

/// The default extractor: routes by [`ResourceKind`] to the right content handler.
#[derive(Clone, Copy, Debug, Default)]
pub struct AutoExtractor;

impl Extractor for AutoExtractor {
    fn handles(&self, _kind: ResourceKind) -> bool {
        true
    }

    fn extract(&self, resource: &Resource, bytes: &[u8]) -> Result<Descriptor> {
        match resource.kind {
            #[cfg(feature = "html")]
            ResourceKind::Html => html::extract(resource, bytes),
            ResourceKind::Markdown => markdown::extract_markdown(resource, bytes),
            ResourceKind::Code => markdown::extract_code(resource, bytes),
            _ => markdown::extract_plain(resource, bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    fn resource(url: &str) -> Resource {
        Resource::new(Url::parse(url).unwrap())
    }

    #[test]
    fn distill_extracts_keywords_and_snippet() {
        let r = resource("https://x.dev/docs/guide.md");
        let body = "Kubernetes Autopilot clusters run Kubernetes without node management. \
                    Autopilot manages nodes for you.";
        let d = distill(
            &r,
            Some(CompactString::from("GKE Autopilot")),
            &[],
            body,
            Vec::new(),
            None,
        );
        // "autopilot" and "kubernetes" are the most frequent discriminative terms.
        assert!(d.keywords.iter().any(|k| k == "autopilot"));
        assert!(d.keywords.iter().any(|k| k == "kubernetes"));
        // stopwords excluded
        assert!(!d.keywords.iter().any(|k| k == "for" || k == "you"));
        assert!(!d.snippet.is_empty());
        assert!(d.embed_text.contains("GKE Autopilot"));
    }

    #[test]
    fn distill_derives_path_tags() {
        let r = resource("https://x.dev/docs/networking/psc.md");
        let d = distill(&r, None, &[], "private service connect", Vec::new(), None);
        assert!(d.tags.iter().any(|t| t == "docs"));
        assert!(d.tags.iter().any(|t| t == "networking"));
        assert!(d.tags.iter().any(|t| t == "psc")); // extension stripped
    }

    #[test]
    fn empty_body_is_empty_descriptor() {
        let r = resource("https://x.dev/empty");
        let d = distill(&r, None, &[], "   ", Vec::new(), None);
        assert!(d.is_empty());
    }

    #[test]
    fn into_document_carries_fields() {
        let r = resource("https://x.dev/a");
        let d = distill(
            &r,
            Some(CompactString::from("T")),
            &[],
            "cat dog cat",
            Vec::new(),
            None,
        );
        let doc = d.into_document(r.url.clone(), ResourceKind::Markdown, 99, vec![0.1, 0.2]);
        assert_eq!(doc.content_hash, 99);
        assert_eq!(doc.title.as_deref(), Some("T"));
        assert!(doc.terms.contains(&CompactString::from("cat")));
    }
}
