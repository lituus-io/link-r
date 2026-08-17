// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Resource addressing: what a source discovers and what gets indexed.
//!
//! A [`Resource`] is a remote document the index points *at* (its body stays at
//! the URL). A [`Page`] couples a resource with its fetched body for the indexing
//! pipeline. [`SourceRef`] names the parent a [`Source`](crate) crawls from.

use crate::payload::DocPayload;
use compact_str::CompactString;
use std::path::PathBuf;
use url::Url;

/// A dense, index-local document identifier.
pub type DocId = u32;

/// The content class of a resource, used to select an extractor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResourceKind {
    /// HTML page (`text/html`).
    Html,
    /// Markdown document.
    Markdown,
    /// Source code or config.
    Code,
    /// Plain text.
    Text,
    /// PDF document.
    Pdf,
    /// JSON document.
    Json,
    /// Unknown / unclassified.
    Unknown,
}

impl ResourceKind {
    /// Classify from an HTTP `Content-Type` header value (parameters ignored).
    #[must_use]
    pub fn from_content_type(content_type: &str) -> Self {
        let mime = content_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        match mime.as_str() {
            "text/html" | "application/xhtml+xml" => Self::Html,
            "text/markdown" | "text/x-markdown" => Self::Markdown,
            "application/json" | "text/json" => Self::Json,
            "application/pdf" => Self::Pdf,
            "text/plain" => Self::Text,
            m if m.starts_with("text/") => Self::Code,
            _ => Self::Unknown,
        }
    }

    /// Classify from a path or URL suffix.
    #[must_use]
    pub fn from_path(path: &str) -> Self {
        let lower = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
        let ext = lower.rsplit('.').next().unwrap_or("");
        match ext {
            "html" | "htm" | "xhtml" => Self::Html,
            "md" | "markdown" | "mdx" => Self::Markdown,
            "json" => Self::Json,
            "pdf" => Self::Pdf,
            "txt" | "text" => Self::Text,
            "rs" | "py" | "js" | "ts" | "go" | "c" | "h" | "cpp" | "hpp" | "java" | "rb"
            | "toml" | "yaml" | "yml" | "sh" | "sql" => Self::Code,
            _ => Self::Unknown,
        }
    }

    /// Whether this kind carries text worth extracting and indexing.
    #[must_use]
    pub fn is_indexable(self) -> bool {
        !matches!(self, Self::Pdf | Self::Unknown)
    }

    /// Whether this kind can contain crawlable `<a href>` links.
    #[must_use]
    pub fn is_linkable(self) -> bool {
        matches!(self, Self::Html | Self::Markdown)
    }

    /// A stable single-byte tag for on-disk metadata columns.
    #[must_use]
    pub fn as_tag(self) -> u8 {
        match self {
            Self::Html => 0,
            Self::Markdown => 1,
            Self::Code => 2,
            Self::Text => 3,
            Self::Pdf => 4,
            Self::Json => 5,
            Self::Unknown => 255,
        }
    }

    /// Inverse of [`ResourceKind::as_tag`]; unknown tags decode to [`ResourceKind::Unknown`].
    #[must_use]
    pub fn from_tag(tag: u8) -> Self {
        match tag {
            0 => Self::Html,
            1 => Self::Markdown,
            2 => Self::Code,
            3 => Self::Text,
            4 => Self::Pdf,
            5 => Self::Json,
            _ => Self::Unknown,
        }
    }
}

/// A remote document the index resolves to. The body lives at [`Resource::url`]
/// and is fetched lazily by the caller; the index stores only this addressing
/// plus a distilled descriptor.
#[derive(Clone, Debug)]
pub struct Resource {
    /// The canonical remote URL.
    pub url: Url,
    /// The content class.
    pub kind: ResourceKind,
    /// An opaque change token (HTTP `ETag`, git blob SHA, or content hash) used to
    /// skip re-embedding unchanged pages on re-crawl.
    pub etag: Option<CompactString>,
    /// The body size in bytes, if known.
    pub size: Option<u64>,
}

impl Resource {
    /// Create a resource, classifying its kind from the URL path.
    #[must_use]
    pub fn new(url: Url) -> Self {
        let kind = ResourceKind::from_path(url.path());
        Self {
            url,
            kind,
            etag: None,
            size: None,
        }
    }

    /// Set the content kind explicitly (e.g. from a `Content-Type` header).
    #[must_use]
    pub fn with_kind(mut self, kind: ResourceKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set the change token.
    #[must_use]
    pub fn with_etag(mut self, etag: impl Into<CompactString>) -> Self {
        self.etag = Some(etag.into());
        self
    }

    /// Set the known size.
    #[must_use]
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }
}

/// A fetched page handed to the indexing pipeline: a [`Resource`] plus its body.
///
/// Crawl sources fetch a page to enumerate its links *and* extract its keywords,
/// so they emit `Page` directly (the body is reused for both — no double fetch).
#[derive(Debug)]
pub struct Page<'a> {
    /// The resource this page addresses.
    pub resource: Resource,
    /// The fetched body, in whichever zero-copy representation the source produced.
    pub payload: DocPayload<'a>,
}

impl<'a> Page<'a> {
    /// Couple a resource with its fetched payload.
    #[must_use]
    pub fn new(resource: Resource, payload: DocPayload<'a>) -> Self {
        Self { resource, payload }
    }
}

/// The parent a source crawls/enumerates from.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SourceRef {
    /// An HTTPS (or HTTP) parent URL to crawl recursively.
    Http {
        /// The root URL the crawl starts from.
        root: Url,
    },
    /// A local directory tree to enumerate.
    Fs {
        /// The root directory.
        root: PathBuf,
    },
}

impl SourceRef {
    /// Build an HTTP source ref from a URL string.
    pub fn http(url: &str) -> crate::Result<Self> {
        let root =
            Url::parse(url).map_err(|e| crate::Error::invalid_url(format!("{url:?}: {e}")))?;
        if root.scheme() != "http" && root.scheme() != "https" {
            return Err(crate::Error::invalid_url(format!(
                "expected http(s), got scheme {:?}",
                root.scheme()
            )));
        }
        Ok(Self::Http { root })
    }

    /// Build a filesystem source ref from a path.
    #[must_use]
    pub fn fs(root: impl Into<PathBuf>) -> Self {
        Self::Fs { root: root.into() }
    }

    /// A stable label for the source kind.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Http { .. } => "http",
            Self::Fs { .. } => "fs",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_type_classification() {
        assert_eq!(
            ResourceKind::from_content_type("text/html; charset=utf-8"),
            ResourceKind::Html
        );
        assert_eq!(
            ResourceKind::from_content_type("application/json"),
            ResourceKind::Json
        );
        assert_eq!(
            ResourceKind::from_content_type("text/x-rust"),
            ResourceKind::Code
        );
        assert_eq!(
            ResourceKind::from_content_type("image/png"),
            ResourceKind::Unknown
        );
    }

    #[test]
    fn path_classification() {
        assert_eq!(
            ResourceKind::from_path("/docs/intro.html"),
            ResourceKind::Html
        );
        assert_eq!(ResourceKind::from_path("README.md"), ResourceKind::Markdown);
        assert_eq!(ResourceKind::from_path("src/lib.rs"), ResourceKind::Code);
        assert_eq!(ResourceKind::from_path("/a/b/"), ResourceKind::Unknown);
    }

    #[test]
    fn kind_tag_roundtrips() {
        for kind in [
            ResourceKind::Html,
            ResourceKind::Markdown,
            ResourceKind::Code,
            ResourceKind::Text,
            ResourceKind::Pdf,
            ResourceKind::Json,
            ResourceKind::Unknown,
        ] {
            assert_eq!(ResourceKind::from_tag(kind.as_tag()), kind);
        }
        assert_eq!(ResourceKind::from_tag(200), ResourceKind::Unknown);
    }

    #[test]
    fn linkable_and_indexable_flags() {
        assert!(ResourceKind::Html.is_linkable());
        assert!(ResourceKind::Markdown.is_linkable(), "markdown links are followed");
        assert!(!ResourceKind::Code.is_linkable());
        assert!(ResourceKind::Markdown.is_indexable());
        assert!(!ResourceKind::Pdf.is_indexable());
    }

    #[test]
    fn resource_builder_classifies_from_path() {
        let r = Resource::new(Url::parse("https://x.dev/a/b.md").unwrap())
            .with_etag("abc")
            .with_size(42);
        assert_eq!(r.kind, ResourceKind::Markdown);
        assert_eq!(r.etag.as_deref(), Some("abc"));
        assert_eq!(r.size, Some(42));
    }

    #[test]
    fn source_ref_http_validates_scheme() {
        assert!(SourceRef::http("https://example.com/docs/").is_ok());
        assert!(SourceRef::http("ftp://example.com").is_err());
        assert!(SourceRef::http("not a url").is_err());
        assert_eq!(SourceRef::fs("/tmp/x").kind(), "fs");
    }
}
