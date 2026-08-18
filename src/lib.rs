// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! `link-r` — a tiny, embeddable crawl-and-resolve link index.
//!
//! Give it a parent HTTPS link. It recursively crawls sublinks to a configurable
//! depth, distills keywords/topics from each page, and builds a small, mmap-able
//! hybrid index (dense embeddings + BM25 + structured metadata). A query resolves
//! to ranked **remote URLs** — document bodies stay at the URL and are fetched
//! lazily by the caller. Each link is unique in the index (canonical-URL keyed),
//! so re-crawling never duplicates a link.
//!
//! # Design tenets
//!
//! - **Zero-copy**: the on-disk index *is* the file; search runs against mapped
//!   pages with no deserialization step.
//! - **Static dispatch**: async traits use GATs (no `Box<dyn Future>`); runtime
//!   variant selection uses enum-dispatch, not `dyn`.
//! - **Self-contained**: no server, no sidecar database; the production embedder
//!   (`OnnxEmbedder`, feature `onnx`) and a deterministic
//!   [`HashEmbedder`] fallback both run in-process.
//!
//! # Dead-simple API
//!
//! ```ignore
//! use link_r::prelude::*;
//!
//! let mut idx = LinkIndex::open_or_create("kb.lnkr")?;
//! idx.update("https://example.com/docs/").depth(2).run().await?;
//! idx.save()?;
//!
//! for hit in idx.search("how does auth work", 10).await? {
//!     println!("{:.3}  {}", hit.score, hit.url);
//! }
//!
//! // Maintain the knowledge base: re-validate links older than a day, evict dead ones.
//! idx.refresh().ttl(std::time::Duration::from_secs(86_400)).run().await?;
//! ```
// GAT TAIT for the async Source/Fetcher/Auth traits. Only the impls need it --
// source/crawl.rs, source/fs.rs, fetch/http.rs and auth/oauth.rs -- so the
// attribute is conditional: declared unconditionally it trips `unused_features`
// under -D warnings in a build where none of them is compiled. All four are
// listed rather than relying on `crawl`/`oauth` implying `http`, so a later
// change to the feature graph cannot silently drop the declaration.
#![cfg_attr(
    any(feature = "crawl", feature = "fs", feature = "http", feature = "oauth"),
    feature(impl_trait_in_assoc_type)
)]
#![deny(unsafe_code)] // `deny`, not `forbid`, so the single `index::mmap` island can locally override.
#![deny(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]
#![warn(clippy::all, clippy::pedantic, clippy::cargo)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::multiple_crate_versions,
    // This crate is a byte-format + scoring engine: document counts, dimensions,
    // and lengths are bounded by `u32`/`usize` by construction, and float scores
    // are intentionally lossy. These casts are deliberate, so the pedantic cast
    // family would only add noise.
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

pub mod auth;
pub mod embed;
pub mod error;
pub mod extract;
#[cfg(feature = "crawl")]
pub mod facade;
pub mod fetch;
pub mod index;
pub mod metric;
pub mod payload;
pub mod query;
pub mod resource;
pub mod source;
pub mod text;
pub mod url_key;

pub use auth::{AnonymousAuth, AuthProvider, Credential, DynAuthProvider, StaticTokenAuth};
pub use embed::{Embedder, EmbedderId, HashEmbedder};
pub use error::{Error, Result};
pub use extract::{AutoExtractor, Descriptor, Extractor};
pub use fetch::{FetchMeta, FetchOptions, Fetched, Fetcher};
pub use index::{DocMeta, Document, Filter, Index, IndexBuilder, UpsertOutcome};
pub use metric::Metric;
pub use payload::{ByteStream, DocPayload, MmapView};
pub use query::{Fusion, Hit, Hits, PreparedQuery, RankParams};
pub use resource::{DocId, Page, Resource, ResourceKind, SourceRef};
pub use source::Source;
pub use url_key::{canonicalize, UrlKey};

#[cfg(feature = "crawl")]
pub use facade::{LinkIndex, SearchResult, UpdateReport};
#[cfg(feature = "http")]
pub use fetch::HttpFetcher;
#[cfg(feature = "fs")]
pub use source::FsSource;
#[cfg(feature = "crawl")]
pub use source::{CrawlConfig, CrawlScope, CrawlSource};

/// Everything you need for the dead-simple API, in one glob import.
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::index::{Document, Filter, Index, IndexBuilder, UpsertOutcome};
    pub use crate::query::{Hit, Hits, PreparedQuery, RankParams};
    pub use crate::resource::{Resource, ResourceKind, SourceRef};
    pub use crate::url_key::UrlKey;

    #[cfg(feature = "crawl")]
    pub use crate::facade::{LinkIndex, SearchResult, UpdateReport};
    #[cfg(feature = "crawl")]
    pub use crate::source::CrawlScope;
}
