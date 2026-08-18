// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Sources: discover the pages to index from a parent.
//!
//! A [`Source`] streams already-fetched [`Page`]s (the crawler must fetch a page
//! to find its links, so discovery and content-fetch are fused — no double fetch).
//! The primary source is the recursive [`CrawlSource`]; a
//! local-directory [`FsSource`] serves offline/test use.

#[cfg(feature = "crawl")]
pub mod crawl;
#[cfg(feature = "fs")]
pub mod fs;
#[cfg(feature = "github")]
pub mod github;

#[cfg(feature = "crawl")]
pub use crawl::{CrawlConfig, CrawlScope, CrawlSource};
#[cfg(feature = "fs")]
pub use fs::FsSource;
#[cfg(feature = "github")]
pub use github::{GitHubSource, GithubAuth, GithubSpec};

use crate::error::Result;
use crate::resource::{Page, SourceRef};
use futures::Stream;

/// Discovers and yields the pages to index from a parent reference.
pub trait Source: Send + Sync {
    /// The stream of discovered pages.
    type Pages<'a>: Stream<Item = Result<Page<'a>>> + Send + 'a
    where
        Self: 'a;

    /// A stable label for this source kind (`"crawl"`, `"fs"`).
    fn kind(&self) -> &'static str;

    /// Begin discovery from `root`.
    fn discover<'a>(&'a self, root: &'a SourceRef) -> Self::Pages<'a>;
}
