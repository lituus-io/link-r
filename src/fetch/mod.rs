// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Fetching: retrieve one resource's bytes.
//!
//! [`Fetcher`] is a GAT async trait so the auth refresh future flows through
//! unboxed. The trait itself is dependency-free (so the crawler can be tested with
//! a mock); the concrete [`HttpFetcher`](http::HttpFetcher) (feature `http`) is
//! gated. Filesystem reads live in the `fs` source, which reads files directly.

#[cfg(feature = "http")]
pub mod http;
#[cfg(feature = "http")]
#[cfg_attr(not(feature = "robots"), allow(unused_imports))] // consumed by the robots UA-parity check
pub(crate) use http::DEFAULT_UA;

#[cfg(feature = "http")]
pub use http::HttpFetcher;

use crate::error::Result;
use crate::payload::DocPayload;
use crate::resource::Resource;
use crate::resource::ResourceKind;
use compact_str::CompactString;
use std::future::Future;
use url::Url;

/// Metadata about a fetched resource.
#[derive(Clone, Debug)]
pub struct FetchMeta {
    /// The content kind, classified from the `Content-Type` (or path).
    pub kind: ResourceKind,
    /// A change token (`ETag`/SHA), if the source provided one.
    pub etag: Option<CompactString>,
    /// The transport status (e.g. HTTP status; 200 for non-HTTP).
    pub status: u16,
    /// The final URL after any redirects, when it differs from the requested one.
    /// The crawler scope-checks and canonicalizes against this so a redirect can
    /// neither escape the crawl scope nor split the canonical dedup key. `None`
    /// means "no redirect happened" (or the source doesn't redirect).
    pub final_url: Option<Url>,
}

/// A fetched resource: its metadata plus body payload.
#[derive(Debug)]
pub struct Fetched<'a> {
    /// Metadata about the fetch.
    pub meta: FetchMeta,
    /// The body, in whichever zero-copy representation the fetcher produced.
    pub payload: DocPayload<'a>,
}

/// Options controlling a fetch.
#[derive(Clone, Copy, Debug, Default)]
pub struct FetchOptions<'a> {
    /// Conditional fetch: send `If-None-Match` and treat 304 as "unchanged".
    pub if_none_match: Option<&'a str>,
    /// Skip bodies larger than this many bytes.
    pub max_bytes: Option<u64>,
    /// Override the User-Agent for this fetch.
    pub user_agent: Option<&'a str>,
}

/// Retrieves a single resource's bytes.
pub trait Fetcher: Send + Sync {
    /// The future returned by [`Fetcher::fetch`].
    type FetchFuture<'a>: Future<Output = Result<Fetched<'a>>> + Send + 'a
    where
        Self: 'a;

    /// Fetch `resource`.
    fn fetch<'a>(&'a self, resource: &'a Resource, opts: FetchOptions<'a>)
        -> Self::FetchFuture<'a>;
}
