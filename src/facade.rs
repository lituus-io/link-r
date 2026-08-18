// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! [`LinkIndex`] — the dead-simple, three-verb API over the trait layer, plus the
//! knowledge-base lifecycle (persist, refresh after a TTL, retention).
//!
//! ```no_run
//! # async fn run() -> link_r::Result<()> {
//! use link_r::facade::LinkIndex;
//!
//! let mut idx = LinkIndex::open_or_create("kb.lnkr")?;
//! idx.update("https://example.com/docs/").depth(2).run().await?;
//! idx.save()?;
//!
//! for hit in idx.search("how does auth work", 10).await? {
//!     println!("{:.3}  {}", hit.score, hit.url);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Defaults are wired so the common case needs no configuration: a path-prefix
//! crawl, [`AnonymousAuth`], [`AutoExtractor`], and the
//! best embedder compiled in — `OnnxEmbedder`
//! (`bge-small`) when the `onnx` feature is on, else the deterministic
//! [`HashEmbedder`](crate::embed::HashEmbedder). For private sources, inject auth
//! with [`UpdateBuilder::token`] or [`UpdateBuilder::auth`].
//!
//! # Lifecycle
//!
//! - [`LinkIndex::update`] discovers and indexes *new* links by crawling.
//! - [`LinkIndex::refresh`] re-validates *known* links after a TTL: unchanged pages
//!   (304) are just touched, changed pages re-embedded, dead pages evicted — so the
//!   index accumulates knowledge without growing endlessly.
//! - [`LinkIndex::pin`] marks links to retain forever (exempt from eviction).
//!
//! # Performance
//!
//! A freshly opened index searches **directly against the memory-mapped file**
//! (zero-copy, no rebuild); after a mutation, searches run against a cached
//! snapshot rebuilt lazily on the next query, so steady-state lookups are `O(query)`
//! rather than `O(index)`.

use crate::auth::{AnonymousAuth, DynAuthProvider, StaticTokenAuth};
use crate::embed::{embed_one, Embedder};
use crate::error::{Error, Result};
use crate::extract::{AutoExtractor, Descriptor, Extractor};
use crate::fetch::{FetchOptions, Fetcher, HttpFetcher};
use crate::index::{Index, IndexBuilder, UpsertOutcome};
use crate::metric::Metric;
use crate::query::{PreparedQuery, RankParams};
use crate::resource::{Resource, ResourceKind, SourceRef};
use crate::source::{CrawlScope, CrawlSource, Source};
use crate::{text, Filter};
use bytes::Bytes;
use compact_str::CompactString;
use futures::stream::{FuturesUnordered, StreamExt};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use xxhash_rust::xxh3::xxh3_64;

/// The embedder the facade uses by default.
#[cfg(feature = "onnx")]
pub type DefaultEmbedder = crate::embed::OnnxEmbedder;
/// The embedder the facade uses by default.
#[cfg(not(feature = "onnx"))]
pub type DefaultEmbedder = crate::embed::HashEmbedder;

/// Default dimension for the hash embedder fallback.
#[cfg(not(feature = "onnx"))]
const HASH_DIM: usize = 256;

/// Default number of pages embedded per model invocation.
const DEFAULT_EMBED_BATCH: usize = 64;
/// Per-page byte ceiling for conditional refresh fetches (2 MiB).
const REFRESH_MAX_BYTES: u64 = 2 * 1024 * 1024;

#[cfg(feature = "onnx")]
fn default_embedder() -> Result<DefaultEmbedder> {
    crate::embed::OnnxEmbedder::new()
}
#[cfg(not(feature = "onnx"))]
#[allow(clippy::unnecessary_wraps)] // signature must match the fallible onnx variant
fn default_embedder() -> Result<DefaultEmbedder> {
    Ok(crate::embed::HashEmbedder::new(HASH_DIM))
}

/// Milliseconds since the Unix epoch (0 if the clock is before the epoch).
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// One exported document: its metadata plus outbound canonical-URL edges.
#[derive(Clone, Copy, Debug)]
pub struct ExportDoc<'a> {
    /// Document metadata (url, `url_key`, title, snippet, tags, freshness).
    pub meta: &'a crate::index::DocMeta,
    /// Outbound edges as canonical-URL keys.
    pub edges: &'a [crate::url_key::UrlKey],
}

/// A single owned search result (no borrows — easy to return across FFI).
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// The resolvable remote URL.
    pub url: String,
    /// Fused relevance score (higher is better).
    pub score: f32,
    /// Display title, if any.
    pub title: Option<String>,
    /// Display snippet.
    pub snippet: String,
    /// Content kind.
    pub kind: ResourceKind,
    /// Structured tags.
    pub tags: Vec<String>,
}

/// What an [`UpdateBuilder::run`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UpdateReport {
    /// New links added.
    pub added: usize,
    /// Existing links whose content changed and were re-indexed.
    pub updated: usize,
    /// Existing links seen again but unchanged.
    pub unchanged: usize,
    /// Pages skipped (empty/no extractable content).
    pub skipped: usize,
    /// Pages whose fetch failed after retries (previously silent).
    pub failed: usize,
    /// Per-page outcomes with distilled structure (headings, keywords) for
    /// downstream consumers such as a graph backend; ephemeral — not
    /// persisted in the index file.
    pub pages: Vec<PageOutcome>,
}

/// How one page fared during an update or refresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageChange {
    /// Newly indexed.
    Added,
    /// Content changed and was re-indexed.
    Updated,
    /// Seen again, byte-identical (or revalidated 304).
    Unchanged,
    /// No extractable content.
    Skipped,
    /// Evicted (dead or aged out).
    Removed,
}

/// One page's outcome plus its distilled structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageOutcome {
    /// Canonical URL.
    pub url: String,
    /// Canonical-URL key (joins the persisted graph edges).
    pub url_key: crate::url_key::UrlKey,
    /// What happened.
    pub change: PageChange,
    /// Extracted title, when a descriptor was produced.
    pub title: Option<CompactString>,
    /// `(depth, text)` section headings, when a descriptor was produced.
    pub headings: Vec<(u8, CompactString)>,
    /// Top keywords, when a descriptor was produced.
    pub keywords: Vec<CompactString>,
}

impl UpdateReport {
    /// Total pages encountered.
    #[must_use]
    pub fn pages_seen(&self) -> usize {
        self.added + self.updated + self.unchanged + self.skipped
    }
}

/// What a [`RefreshBuilder::run`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RefreshReport {
    /// Stale links whose content changed and were re-indexed.
    pub refreshed: usize,
    /// Stale links revalidated as unchanged (304) and re-timestamped.
    pub unchanged: usize,
    /// Links removed (dead/unreachable, or past the max-age retention cap).
    pub removed: usize,
    /// Links that failed to refresh (kept for the next attempt).
    pub failed: usize,
    /// Per-page outcomes (see [`PageOutcome`]); ephemeral.
    pub pages: Vec<PageOutcome>,
}

impl RefreshReport {
    /// Total links acted on.
    #[must_use]
    pub fn total(&self) -> usize {
        self.refreshed + self.unchanged + self.removed + self.failed
    }
}

/// The dead-simple link index facade.
///
/// Reads run against whichever representation is current: a memory-mapped [`Index`]
/// straight after [`open`](LinkIndex::open) (zero-copy), or a cached snapshot of the
/// in-memory builder after a mutation. The builder is materialized lazily on the
/// first mutation, so opening and searching a persisted index never rebuilds it.
#[derive(Debug)]
pub struct LinkIndex {
    path: Option<PathBuf>,
    embedder: DefaultEmbedder,
    /// The mmap-backed index from `open`, present until the first mutation.
    opened: Option<Index>,
    /// The mutable builder, materialized lazily from `opened` (or created empty).
    builder: Option<IndexBuilder>,
    /// Search snapshot of `builder`, built lazily and invalidated on mutation.
    cached: OnceLock<Index>,
}

impl LinkIndex {
    fn from_builder(
        path: Option<PathBuf>,
        embedder: DefaultEmbedder,
        builder: IndexBuilder,
    ) -> Self {
        Self {
            path,
            embedder,
            opened: None,
            builder: Some(builder),
            cached: OnceLock::new(),
        }
    }

    fn from_opened(path: PathBuf, embedder: DefaultEmbedder, opened: Index) -> Self {
        Self {
            path: Some(path),
            embedder,
            opened: Some(opened),
            builder: None,
            cached: OnceLock::new(),
        }
    }

    /// Open an existing index at `path`, or create an empty one bound to it.
    ///
    /// # Errors
    /// Returns an error if the embedder cannot be loaded, the file is corrupt, or
    /// an existing index was built with an incompatible embedder.
    pub fn open_or_create(path: impl Into<PathBuf>) -> Result<Self> {
        let embedder = default_embedder()?;
        let path = path.into();
        if path.exists() {
            let index = open_compatible(&path, &embedder)?;
            Ok(Self::from_opened(path, embedder, index))
        } else {
            let builder =
                IndexBuilder::new(embedder.dim(), Metric::Cosine, embedder.identity().raw());
            Ok(Self::from_builder(Some(path), embedder, builder))
        }
    }

    /// Open an existing index (errors if it does not exist).
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let embedder = default_embedder()?;
        let path = path.into();
        if !path.exists() {
            return Err(Error::not_found(path.display().to_string()));
        }
        let index = open_compatible(&path, &embedder)?;
        Ok(Self::from_opened(path, embedder, index))
    }

    /// Create an in-memory index not bound to any file.
    pub fn in_memory() -> Result<Self> {
        let embedder = default_embedder()?;
        let builder = IndexBuilder::new(embedder.dim(), Metric::Cosine, embedder.identity().raw());
        Ok(Self::from_builder(None, embedder, builder))
    }

    /// Number of indexed links.
    #[must_use]
    pub fn len(&self) -> usize {
        match (&self.opened, &self.builder) {
            (Some(idx), _) => idx.len(),
            (None, Some(b)) => b.len(),
            (None, None) => 0,
        }
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Begin a crawl-and-index from a parent URL. Configure with the builder
    /// methods, then `.run().await`.
    pub fn update(&mut self, url: impl Into<String>) -> UpdateBuilder<'_> {
        UpdateBuilder::new(self, url.into())
    }

    /// Begin a TTL refresh of already-indexed links. Configure with the builder
    /// methods, then `.run().await`.
    pub fn refresh(&mut self) -> RefreshBuilder<'_> {
        RefreshBuilder::new(self)
    }

    /// Pin every link whose URL starts with `url_prefix` so it is retained forever
    /// (exempt from TTL / max-age eviction). Returns how many links changed.
    pub fn pin(&mut self, url_prefix: &str) -> Result<usize> {
        Ok(self.materialize_builder()?.set_pinned(url_prefix, true))
    }

    /// Unpin every link whose URL starts with `url_prefix`. Returns how many changed.
    pub fn unpin(&mut self, url_prefix: &str) -> Result<usize> {
        Ok(self.materialize_builder()?.set_pinned(url_prefix, false))
    }

    /// Search the index, returning ranked owned results.
    ///
    /// # Errors
    /// Returns an error if the query cannot be embedded or searched.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search_filtered(query, limit, &Filter::All).await
    }

    /// Search with a structured metadata prefilter (categorical retrieval): pass a
    /// [`Filter`] to confine results by tag, path prefix, kind, freshness, etc.
    ///
    /// # Errors
    /// Returns an error if the query cannot be embedded or searched.
    pub async fn search_filtered(
        &self,
        query: &str,
        limit: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        self.search_ranked(query, limit, filter, 0.0).await
    }

    /// Search with a metadata prefilter and an optional knowledge-graph boost
    /// (`graph_boost > 0` re-ranks results by link-graph connectivity — hub pages
    /// rise). The full-control entry point behind [`search`](LinkIndex::search) and
    /// [`search_filtered`](LinkIndex::search_filtered).
    ///
    /// # Errors
    /// Returns an error if the query cannot be embedded or searched.
    pub async fn search_ranked(
        &self,
        query: &str,
        limit: usize,
        filter: &Filter,
        graph_boost: f32,
    ) -> Result<Vec<SearchResult>> {
        let vector = embed_one(&self.embedder, query).await?;
        let terms: Vec<CompactString> = text::normalized_tokens(query).collect();
        let index = self.read_index();
        let pq = PreparedQuery {
            vector: &vector,
            terms: &terms,
            filter,
            limit,
            rank: RankParams {
                graph_boost,
                ..RankParams::default()
            },
        };
        let hits = index.search_prepared(&pq)?;
        Ok(hits.iter().map(SearchResult::from_hit).collect())
    }

    /// Follow the knowledge graph from a link: the `k` documents most related to
    /// `url` — its outbound targets and co-cited siblings — ranked by connectivity.
    ///
    /// # Errors
    /// Returns [`Error::InvalidUrl`] if `url` cannot be parsed.
    pub fn related(&self, url: &str, k: usize) -> Result<Vec<SearchResult>> {
        let key = crate::url_key::UrlKey::parse(url)?;
        let index = self.read_index();
        Ok(index
            .related(key, k)
            .iter()
            .map(SearchResult::from_hit)
            .collect())
    }

    /// Every indexed document with its outbound edges — the export seam for
    /// external backends (e.g. a persistent graph store) to absorb the index
    /// without re-deriving anything.
    #[allow(clippy::missing_errors_doc)]
    pub fn export(&self) -> Result<impl Iterator<Item = ExportDoc<'_>>> {
        let idx = self.read_index();
        Ok(idx
            .documents()
            .iter()
            .zip(idx.edge_lists())
            .map(|(meta, edges)| ExportDoc { meta, edges }))
    }

    /// Atomically save to the bound path.
    ///
    /// # Errors
    /// Returns [`Error::InvalidUrl`] if no path is bound, or [`Error::Io`] on a
    /// filesystem failure.
    pub fn save(&self) -> Result<()> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| Error::invalid_url("no path bound; use save_as"))?;
        self.save_to(path)
    }

    /// Save to a new path and bind to it.
    pub fn save_as(&mut self, path: impl Into<PathBuf>) -> Result<()> {
        let path = path.into();
        self.save_to(&path)?;
        self.path = Some(path);
        Ok(())
    }

    fn save_to(&self, path: &Path) -> Result<()> {
        match (&self.opened, &self.builder) {
            (Some(idx), _) => idx.save(path),
            (None, Some(b)) => b.save(path),
            (None, None) => Err(Error::format("index has no content to save")),
        }
    }

    /// The current search-ready index: the mmap'd one if unmutated, else the cached
    /// snapshot (built once, reused until the next mutation).
    fn read_index(&self) -> &Index {
        if let Some(idx) = &self.opened {
            return idx;
        }
        self.cached.get_or_init(|| {
            self.builder
                .as_ref()
                .expect("opened or builder is always present")
                .to_index()
        })
    }

    /// Ensure a mutable builder exists (materializing it from the opened mmap index
    /// on the first mutation) and invalidate any cached search snapshot.
    fn materialize_builder(&mut self) -> Result<&mut IndexBuilder> {
        if self.builder.is_none() {
            let idx = self
                .opened
                .take()
                .ok_or_else(|| Error::format("index has no builder or opened state"))?;
            self.builder = Some(idx.into_builder()?);
        }
        self.cached = OnceLock::new(); // any snapshot is now stale
        Ok(self.builder.as_mut().expect("just materialized"))
    }

    /// Drive an update from any [`Source`] (the hermetic seam behind `update`),
    /// micro-batching the embedder across pages.
    ///
    /// # Errors
    /// Propagates fetch/extract/embed failures.
    #[doc(hidden)]
    pub async fn ingest_from<S: Source>(
        &mut self,
        source: &S,
        root: &SourceRef,
        embed_batch: usize,
        pin: bool,
    ) -> Result<UpdateReport> {
        self.materialize_builder()?;
        let dim = self.embedder.dim();
        let embed_batch = embed_batch.max(1);
        let extractor = AutoExtractor;
        let LinkIndex {
            builder,
            embedder,
            cached,
            ..
        } = self;
        let builder = builder.as_mut().expect("materialized");

        let mut report = UpdateReport::default();
        let mut buf = vec![0.0f32; embed_batch * dim];
        let mut pending: Vec<(Resource, Descriptor, u64)> = Vec::with_capacity(embed_batch);

        let stream = source.discover(root);
        futures::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            let page = item?;
            let resource = page.resource;
            let bytes = page.payload.into_bytes().await?;
            let content_hash = xxh3_64(&bytes);
            if builder.is_unchanged(&resource.url, content_hash) {
                report.unchanged += 1;
                report
                    .pages
                    .push(page_outcome(&resource.url, PageChange::Unchanged, None));
                continue;
            }
            let descriptor = extractor.extract(&resource, &bytes)?;
            if descriptor.is_empty() {
                report.skipped += 1;
                report
                    .pages
                    .push(page_outcome(&resource.url, PageChange::Skipped, None));
                continue;
            }
            pending.push((resource, descriptor, content_hash));
            if pending.len() == embed_batch {
                flush_batch(
                    builder,
                    &*embedder,
                    &mut pending,
                    &mut buf,
                    dim,
                    pin,
                    &mut report,
                )
                .await?;
            }
        }
        flush_batch(
            builder,
            &*embedder,
            &mut pending,
            &mut buf,
            dim,
            pin,
            &mut report,
        )
        .await?;

        if report.added + report.updated > 0 {
            *cached = OnceLock::new();
        }
        Ok(report)
    }

    /// Refresh already-indexed links against any [`Fetcher`] (the hermetic seam
    /// behind `refresh`). Conditionally re-fetches stale links, applying the
    /// retention policy.
    ///
    /// # Errors
    /// Propagates extract/embed failures; per-link fetch failures are counted, not
    /// propagated.
    #[doc(hidden)]
    #[allow(clippy::too_many_lines)] // one linear pass over planned work
    pub async fn refresh_with<F: Fetcher>(
        &mut self,
        fetcher: &F,
        opts: RefreshOptions,
    ) -> Result<RefreshReport> {
        self.materialize_builder()?;
        let now = now_ms();
        let dim = self.embedder.dim();
        let extractor = AutoExtractor;

        // Plan the work up front (immutable read of the builder), then mutate.
        let mut evict: Vec<url::Url> = Vec::new();
        let mut stale: Vec<(url::Url, Option<CompactString>)> = Vec::new();
        {
            let builder = self.builder.as_ref().expect("materialized");
            for d in builder.documents() {
                if let Some(filter) = &opts.urls {
                    if !filter.contains(&d.url_key) {
                        continue;
                    }
                }
                let age = now.saturating_sub(d.fetched_at_ms);
                if !d.pinned {
                    if let Some(max_age) = opts.max_age {
                        if u128::from(age) > max_age.as_millis() {
                            if let Ok(url) = url::Url::parse(&d.url) {
                                evict.push(url);
                            }
                            continue; // evicted without a fetch
                        }
                    }
                }
                if u128::from(age) >= opts.ttl.as_millis() {
                    if let Ok(url) = url::Url::parse(&d.url) {
                        stale.push((url, d.etag.clone()));
                    }
                }
            }
        }

        let mut report = RefreshReport::default();
        let builder = self.builder.as_mut().expect("materialized");
        // One pass: dead-link handling below checks pin state per URL key.
        let pinned_keys: std::collections::HashSet<crate::url_key::UrlKey> = builder
            .documents()
            .iter()
            .filter(|d| d.pinned)
            .map(|d| d.url_key)
            .collect();

        // Hard-age eviction (no network).
        for url in &evict {
            if builder.remove(url) {
                report.removed += 1;
            }
        }

        // Conditionally re-fetch stale links with bounded concurrency.
        let mut pending_changed: Vec<(Resource, Descriptor, u64)> = Vec::new();
        let concurrency = opts.concurrency.max(1);
        let mut idx = 0;
        let mut in_flight = FuturesUnordered::new();
        loop {
            while in_flight.len() < concurrency && idx < stale.len() {
                let (url, etag) = stale[idx].clone();
                idx += 1;
                in_flight.push(conditional_fetch(fetcher, url, etag));
            }
            let Some((url, outcome)) = in_flight.next().await else {
                break;
            };
            match outcome {
                Ok(RefreshFetch::NotModified) => {
                    builder.touch(&url, now);
                    report.unchanged += 1;
                    report
                        .pages
                        .push(page_outcome(&url, PageChange::Unchanged, None));
                }
                Ok(RefreshFetch::Changed { kind, etag, bytes }) => {
                    let content_hash = xxh3_64(&bytes);
                    let mut resource = Resource::new(url.clone()).with_kind(kind);
                    if let Some(tag) = etag {
                        resource = resource.with_etag(tag);
                    }
                    let descriptor = extractor.extract(&resource, &bytes)?;
                    if descriptor.is_empty() {
                        builder.touch(&url, now);
                        report.unchanged += 1;
                        report
                            .pages
                            .push(page_outcome(&url, PageChange::Unchanged, None));
                        continue;
                    }
                    // Defer embedding: changed pages are batched below so the
                    // (potentially model-backed) embedder runs once per chunk,
                    // not once per document.
                    pending_changed.push((resource, descriptor, content_hash));
                }
                Err(RefreshError::Gone) => {
                    // Dead link: evict unless pinned or eviction disabled.
                    let pinned = pinned_keys.contains(&crate::url_key::UrlKey::from_url(&url));
                    if opts.evict_unreachable && !pinned && builder.remove(&url) {
                        report.removed += 1;
                        report
                            .pages
                            .push(page_outcome(&url, PageChange::Removed, None));
                    } else {
                        report.failed += 1;
                    }
                }
                Err(RefreshError::Transient) => report.failed += 1,
            }
        }

        // Batched embed + upsert of every changed page (micro-batches like
        // ingest, one model invocation per chunk).
        let mut buf = vec![0.0f32; DEFAULT_EMBED_BATCH * dim];
        while !pending_changed.is_empty() {
            let take = pending_changed.len().min(DEFAULT_EMBED_BATCH);
            let batch: Vec<(Resource, Descriptor, u64)> = pending_changed.drain(..take).collect();
            let texts: Vec<&str> = batch
                .iter()
                .map(|(_, d, _)| d.embed_text.as_str())
                .collect();
            let out = &mut buf[..texts.len() * dim];
            self.embedder.embed_batch(&texts, out).await?;
            drop(texts);
            for (i, (resource, descriptor, content_hash)) in batch.into_iter().enumerate() {
                let vector = out[i * dim..(i + 1) * dim].to_vec();
                let outcome_meta = descriptor_outcome_meta(&descriptor);
                let mut doc =
                    descriptor.into_document(resource.url, resource.kind, content_hash, vector);
                doc.fetched_at_ms = now;
                doc.etag = resource.etag;
                let url = doc.url.clone();
                // A byte-identical body from a server that sends no ETag comes
                // back as `Unchanged` without the record being rewritten, so the
                // freshness stamp must be advanced here or the page is re-fetched
                // in full on every subsequent refresh.
                if builder.upsert(doc)? == UpsertOutcome::Unchanged {
                    builder.touch(&url, now);
                    report.unchanged += 1;
                    report
                        .pages
                        .push(page_outcome(&url, PageChange::Unchanged, None));
                } else {
                    report.refreshed += 1;
                    report
                        .pages
                        .push(page_outcome(&url, PageChange::Updated, Some(outcome_meta)));
                }
            }
        }

        if report.refreshed + report.removed > 0 {
            self.cached = OnceLock::new();
        }
        Ok(report)
    }
}

impl SearchResult {
    fn from_hit(h: crate::query::Hit<'_>) -> Self {
        Self {
            url: h.url.to_owned(),
            score: h.score,
            title: h.title.map(str::to_owned),
            snippet: h.snippet.to_owned(),
            kind: h.kind,
            tags: h.tags().map(str::to_owned).collect(),
        }
    }
}

/// Embed a batch of pending pages in one model call, then upsert each.
async fn flush_batch<E: Embedder>(
    builder: &mut IndexBuilder,
    embedder: &E,
    pending: &mut Vec<(Resource, Descriptor, u64)>,
    buf: &mut [f32],
    dim: usize,
    pin: bool,
    report: &mut UpdateReport,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let now = now_ms();
    let texts: Vec<&str> = pending
        .iter()
        .map(|(_, d, _)| d.embed_text.as_str())
        .collect();
    let out = &mut buf[..texts.len() * dim];
    embedder.embed_batch(&texts, out).await?;
    drop(texts);

    for (i, (resource, descriptor, content_hash)) in pending.drain(..).enumerate() {
        let vector = out[i * dim..(i + 1) * dim].to_vec();
        let outcome_meta = descriptor_outcome_meta(&descriptor);
        let mut doc = descriptor.into_document(resource.url, resource.kind, content_hash, vector);
        doc.fetched_at_ms = now;
        doc.etag = resource.etag;
        doc.pinned = pin;
        let url = doc.url.clone();
        let change = match builder.upsert(doc)? {
            crate::index::UpsertOutcome::Added => {
                report.added += 1;
                PageChange::Added
            }
            crate::index::UpsertOutcome::Updated => {
                report.updated += 1;
                PageChange::Updated
            }
            crate::index::UpsertOutcome::Unchanged => {
                report.unchanged += 1;
                PageChange::Unchanged
            }
        };
        report
            .pages
            .push(page_outcome(&url, change, Some(outcome_meta)));
    }
    Ok(())
}

/// (title, `(depth, heading)` pairs, keywords) captured before a descriptor is
/// consumed. Missing levels fall back to position-derived depths.
type OutcomeMeta = (
    Option<CompactString>,
    Vec<(u8, CompactString)>,
    Vec<CompactString>,
);

fn descriptor_outcome_meta(d: &Descriptor) -> OutcomeMeta {
    let headings = d
        .headings
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let level = d
                .heading_levels
                .get(i)
                .copied()
                .unwrap_or((i.min(2) + 1) as u8);
            (level, h.clone())
        })
        .collect();
    (
        d.title.clone(),
        headings,
        d.keywords.iter().cloned().collect(),
    )
}

fn page_outcome(url: &url::Url, change: PageChange, meta: Option<OutcomeMeta>) -> PageOutcome {
    let (title, headings, keywords) = meta.unwrap_or_default();
    PageOutcome {
        url: crate::url_key::canonicalize(url),
        url_key: crate::url_key::UrlKey::from_url(url),
        change,
        title,
        headings,
        keywords,
    }
}

/// The outcome of a single conditional refresh fetch.
enum RefreshFetch {
    NotModified,
    Changed {
        kind: ResourceKind,
        etag: Option<CompactString>,
        bytes: Bytes,
    },
}

/// A refresh fetch failure, classified for the retention policy.
enum RefreshError {
    /// The link is gone (404/410) or access was denied — a candidate for eviction.
    Gone,
    /// A transient failure (timeout/5xx/etc.) — keep and retry next time.
    Transient,
}

/// One conditional (`If-None-Match`) fetch, owning its URL so the borrow handed to
/// [`Fetcher::fetch`] lives inside the future.
async fn conditional_fetch<F: Fetcher>(
    fetcher: &F,
    url: url::Url,
    etag: Option<CompactString>,
) -> (url::Url, std::result::Result<RefreshFetch, RefreshError>) {
    let resource = Resource::new(url);
    let opts = FetchOptions {
        if_none_match: etag.as_deref(),
        max_bytes: Some(REFRESH_MAX_BYTES),
        user_agent: None,
    };
    let outcome = match fetcher.fetch(&resource, opts).await {
        Ok(got) => {
            let kind = got.meta.kind;
            let etag = got.meta.etag;
            match got.payload.into_bytes().await {
                Ok(bytes) => Ok(RefreshFetch::Changed { kind, etag, bytes }),
                Err(_) => Err(RefreshError::Transient),
            }
        }
        Err(Error::NotModified { .. }) => Ok(RefreshFetch::NotModified),
        Err(
            Error::NotFound { .. } | Error::PermissionDenied { .. } | Error::Unauthenticated { .. },
        ) => Err(RefreshError::Gone),
        Err(_) => Err(RefreshError::Transient),
    };
    (resource.url, outcome)
}

fn open_compatible(path: &Path, embedder: &DefaultEmbedder) -> Result<Index> {
    let index = Index::open(path)?;
    if index.dim() != embedder.dim() || index.embedder_id() != embedder.identity().raw() {
        return Err(Error::embed(format!(
            "index at {} is incompatible with the current embedder (index dim {}, embedder dim {})",
            path.display(),
            index.dim(),
            embedder.dim()
        )));
    }
    Ok(index)
}

/// Options for [`LinkIndex::refresh_with`]; set via [`RefreshBuilder`].
#[derive(Clone, Debug)]
pub struct RefreshOptions {
    /// When set, only these canonical-URL keys are considered (due-list
    /// driven refresh); eviction and TTL checks apply to them alone.
    pub urls: Option<std::collections::HashSet<crate::url_key::UrlKey>>,
    /// Re-validate links older than this.
    pub ttl: Duration,
    /// Evict unpinned links older than this without fetching (hard retention cap).
    pub max_age: Option<Duration>,
    /// Evict unpinned links that return gone/denied on refresh.
    pub evict_unreachable: bool,
    /// Max conditional fetches in flight.
    pub concurrency: usize,
}

impl Default for RefreshOptions {
    fn default() -> Self {
        Self {
            urls: None,
            ttl: Duration::from_secs(24 * 60 * 60),
            max_age: None,
            evict_unreachable: true,
            concurrency: 8,
        }
    }
}

/// Configures and runs a crawl-and-index. Created by [`LinkIndex::update`].
pub struct UpdateBuilder<'a> {
    index: &'a mut LinkIndex,
    root: String,
    depth: u16,
    scope: CrawlScope,
    max_pages: usize,
    concurrency: usize,
    max_retries: u32,
    min_delay: Duration,
    embed_batch: usize,
    pin: bool,
    require_path: Vec<CompactString>,
    accept_extensions: Vec<CompactString>,
    index_path: Vec<CompactString>,
    #[cfg(feature = "robots")]
    respect_robots: bool,
    auth: Box<dyn DynAuthProvider>,
}

impl std::fmt::Debug for UpdateBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpdateBuilder")
            .field("root", &self.root)
            .field("depth", &self.depth)
            .field("max_pages", &self.max_pages)
            .field("concurrency", &self.concurrency)
            .finish_non_exhaustive()
    }
}

impl<'a> UpdateBuilder<'a> {
    fn new(index: &'a mut LinkIndex, root: String) -> Self {
        Self {
            index,
            root,
            depth: 2,
            scope: CrawlScope::PathPrefix,
            max_pages: 1000,
            concurrency: 8,
            max_retries: 2,
            min_delay: Duration::ZERO,
            embed_batch: DEFAULT_EMBED_BATCH,
            pin: false,
            require_path: Vec::new(),
            accept_extensions: Vec::new(),
            index_path: Vec::new(),
            #[cfg(feature = "robots")]
            respect_robots: false,
            auth: Box::new(AnonymousAuth),
        }
    }

    /// Set the crawl depth (default 2).
    #[must_use]
    pub fn depth(mut self, depth: u16) -> Self {
        self.depth = depth;
        self
    }

    /// Set the crawl scope (default same path-prefix).
    #[must_use]
    pub fn scope(mut self, scope: CrawlScope) -> Self {
        self.scope = scope;
        self
    }

    /// Cap the total pages crawled (default 1000).
    #[must_use]
    pub fn max_pages(mut self, max_pages: usize) -> Self {
        self.max_pages = max_pages;
        self
    }

    /// Set the number of concurrent fetches (default 8).
    #[must_use]
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Set the per-page retry budget for retriable errors (default 2).
    #[must_use]
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set a polite inter-request delay (paces request starts globally).
    #[must_use]
    pub fn min_delay(mut self, delay: Duration) -> Self {
        self.min_delay = delay;
        self
    }

    /// Set how many pages are embedded per model call (default 64).
    #[must_use]
    pub fn embed_batch(mut self, n: usize) -> Self {
        self.embed_batch = n.max(1);
        self
    }

    /// Pin every link indexed by this update (retained forever, exempt from eviction).
    #[must_use]
    pub fn pin(mut self) -> Self {
        self.pin = true;
        self
    }

    /// Honor `robots.txt` for this crawl (requires the `robots` feature).
    #[cfg(feature = "robots")]
    #[must_use]
    pub fn respect_robots(mut self, respect: bool) -> Self {
        self.respect_robots = respect;
        self
    }

    /// Only follow links whose URL path contains `substring` (additive — all
    /// required substrings must be present). Confines the crawl to a sub-path.
    #[must_use]
    pub fn require_path(mut self, substring: impl Into<CompactString>) -> Self {
        self.require_path.push(substring.into());
        self
    }

    /// Only index pages whose URL path ends with `extension` (additive), e.g.
    /// `"md"`. Other pages are still crawled for links.
    #[must_use]
    pub fn accept_extension(mut self, extension: impl Into<CompactString>) -> Self {
        self.accept_extensions.push(extension.into());
        self
    }

    /// Only index pages whose URL path contains `substring` (additive). Narrows
    /// *indexing*, not crawling — e.g. index only GitHub `/blob/` file views.
    #[must_use]
    pub fn index_path(mut self, substring: impl Into<CompactString>) -> Self {
        self.index_path.push(substring.into());
        self
    }

    /// Authenticate with a bearer token (e.g. a GitHub PAT) for private sources.
    /// The token is scoped to the crawl root's host, so it is never sent to a
    /// different host reached via an in-scope link.
    #[must_use]
    pub fn token(mut self, token: impl Into<String>) -> Self {
        if let Ok(SourceRef::Http { root }) = SourceRef::http(&self.root) {
            if let Some(host) = root.host_str() {
                self.auth = Box::new(StaticTokenAuth::bearer_scoped(token, host));
                return self;
            }
        }
        self.auth = Box::new(StaticTokenAuth::bearer(token));
        self
    }

    /// Inject a custom auth provider (e.g. OAuth/GDrive) for private sources.
    #[must_use]
    pub fn auth(mut self, auth: Box<dyn DynAuthProvider>) -> Self {
        self.auth = auth;
        self
    }

    /// Run the crawl, indexing every discovered page (deduplicated by URL).
    ///
    /// # Errors
    /// Returns an error if the root URL is invalid, the crawl/fetch fails on the
    /// root, or a page cannot be embedded.
    pub async fn run(self) -> Result<UpdateReport> {
        let root = SourceRef::http(&self.root)?;
        // Conditional re-crawl inputs from the existing knowledge base: known
        // entity tags avoid body transfers (304), and known outbound links
        // keep the frontier alive across revalidated pages.
        self.index.materialize_builder()?;
        let (validators, known_edges) = {
            let builder = self.index.builder.as_ref().expect("materialized");
            let docs = builder.documents();
            let by_key: std::collections::HashMap<crate::url_key::UrlKey, &str> =
                docs.iter().map(|d| (d.url_key, d.url.as_str())).collect();
            let mut validators = Vec::new();
            let mut known_edges = Vec::new();
            for (d, edges) in docs.iter().zip(builder.edge_lists()) {
                if let Some(etag) = &d.etag {
                    validators.push((d.url_key, etag.clone()));
                }
                let children: Vec<url::Url> = edges
                    .iter()
                    .filter_map(|k| by_key.get(k))
                    .filter_map(|u| url::Url::parse(u).ok())
                    .collect();
                if !children.is_empty() {
                    known_edges.push((d.url_key, children));
                }
            }
            (validators, known_edges)
        };
        let fetcher = HttpFetcher::new(self.auth)?;
        #[allow(unused_mut)]
        let mut source = CrawlSource::new(fetcher)
            .validators(validators)
            .known_edges(known_edges)
            .depth(self.depth)
            .scope(self.scope)
            .max_pages(self.max_pages)
            .concurrency(self.concurrency)
            .max_retries(self.max_retries)
            .min_delay(self.min_delay);
        for substring in self.require_path {
            source = source.require_path(substring);
        }
        for ext in self.accept_extensions {
            source = source.accept_extension(ext);
        }
        for substring in self.index_path {
            source = source.index_path(substring);
        }
        #[cfg(feature = "robots")]
        {
            source = source.respect_robots(self.respect_robots);
        }
        let mut report = self
            .index
            .ingest_from(&source, &root, self.embed_batch, self.pin)
            .await?;
        report.failed = source.failed_count() as usize;
        Ok(report)
    }
}

/// Configures and runs a TTL refresh. Created by [`LinkIndex::refresh`].
pub struct RefreshBuilder<'a> {
    index: &'a mut LinkIndex,
    opts: RefreshOptions,
    auth: Box<dyn DynAuthProvider>,
    scoped_host: Option<CompactString>,
}

impl std::fmt::Debug for RefreshBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshBuilder")
            .field("opts", &self.opts)
            .finish_non_exhaustive()
    }
}

impl<'a> RefreshBuilder<'a> {
    fn new(index: &'a mut LinkIndex) -> Self {
        Self {
            index,
            opts: RefreshOptions::default(),
            auth: Box::new(AnonymousAuth),
            scoped_host: None,
        }
    }

    /// Re-validate links older than `ttl` (default 24h).
    #[must_use]
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.opts.ttl = ttl;
        self
    }

    /// Evict unpinned links older than `max_age` without fetching (hard cap so the
    /// knowledge base never retains stale knowledge endlessly). Off by default.
    #[must_use]
    pub fn max_age(mut self, max_age: Duration) -> Self {
        self.opts.max_age = Some(max_age);
        self
    }

    /// Whether to evict unpinned links that are gone/denied on refresh (default true).
    #[must_use]
    pub fn evict_unreachable(mut self, evict: bool) -> Self {
        self.opts.evict_unreachable = evict;
        self
    }

    /// Set the number of concurrent conditional fetches (default 8).
    #[must_use]
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.opts.concurrency = concurrency;
        self
    }

    /// Refresh only these URLs (a due-list from an external scheduler such as
    /// a graph backend). URLs that fail to parse are ignored; TTL still
    /// applies within the subset, so combine with `ttl(Duration::ZERO)` to
    /// force revalidation of exactly this set.
    #[must_use]
    pub fn urls<I>(mut self, urls: I) -> Self
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let set: std::collections::HashSet<crate::url_key::UrlKey> = urls
            .into_iter()
            .filter_map(|u| url::Url::parse(u.as_ref()).ok())
            .map(|u| crate::url_key::UrlKey::from_url(&u))
            .collect();
        self.opts.urls = Some(set);
        self
    }

    /// Authenticate refresh fetches with a bearer token, scoped to `host` (pass the
    /// private host, e.g. `"github.com"`).
    #[must_use]
    pub fn token(mut self, token: impl Into<String>, host: impl Into<CompactString>) -> Self {
        let host = host.into();
        self.auth = Box::new(StaticTokenAuth::bearer_scoped(token, host.as_str()));
        self.scoped_host = Some(host);
        self
    }

    /// Inject a custom auth provider for refresh fetches.
    #[must_use]
    pub fn auth(mut self, auth: Box<dyn DynAuthProvider>) -> Self {
        self.auth = auth;
        self
    }

    /// Run the refresh.
    ///
    /// # Errors
    /// Propagates extract/embed failures; per-link fetch failures are counted.
    pub async fn run(self) -> Result<RefreshReport> {
        let fetcher = HttpFetcher::new(self.auth)?;
        self.index.refresh_with(&fetcher, self.opts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::Resource;

    #[tokio::test]
    async fn in_memory_search_over_manual_upserts() {
        let mut idx = LinkIndex::in_memory().unwrap();
        index_doc(&mut idx, "https://x.dev/cat", "cats are feline animals").await;
        index_doc(&mut idx, "https://x.dev/dog", "dogs are canine animals").await;

        let hits = idx.search("canine", 5).await.unwrap();
        assert_eq!(hits[0].url, "https://x.dev/dog");
        assert_eq!(idx.len(), 2);
    }

    #[tokio::test]
    async fn save_and_reopen_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("kb.lnkr");
        {
            let mut idx = LinkIndex::open_or_create(&path).unwrap();
            index_doc(&mut idx, "https://x.dev/a", "alpha content here").await;
            idx.save().unwrap();
        }
        let idx = LinkIndex::open(&path).unwrap();
        assert_eq!(idx.len(), 1);
        // A freshly opened index searches directly against the mmap (no builder).
        assert!(idx.opened.is_some() && idx.builder.is_none());
        let hits = idx.search("alpha", 5).await.unwrap();
        assert_eq!(hits[0].url, "https://x.dev/a");
    }

    #[tokio::test]
    async fn search_cache_is_built_and_invalidated() {
        let mut idx = LinkIndex::in_memory().unwrap();
        index_doc(&mut idx, "https://x.dev/a", "alpha").await;
        // No cache until the first search.
        assert!(idx.cached.get().is_none());
        let _ = idx.search("alpha", 5).await.unwrap();
        assert!(idx.cached.get().is_some());
        // A mutation invalidates the cache.
        index_doc(&mut idx, "https://x.dev/b", "beta").await;
        assert!(idx.cached.get().is_none());
        let hits = idx.search("beta", 5).await.unwrap();
        assert_eq!(hits[0].url, "https://x.dev/b");
    }

    #[tokio::test]
    async fn first_mutation_after_open_materializes_builder() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("kb.lnkr");
        {
            let mut idx = LinkIndex::open_or_create(&path).unwrap();
            index_doc(&mut idx, "https://x.dev/a", "alpha").await;
            idx.save().unwrap();
        }
        let mut idx = LinkIndex::open(&path).unwrap();
        assert!(idx.opened.is_some());
        idx.pin("https://x.dev/").unwrap();
        // Pinning is a mutation → builder materialized, opened consumed.
        assert!(idx.opened.is_none() && idx.builder.is_some());
        assert_eq!(idx.len(), 1);
    }

    use crate::fetch::{FetchMeta, Fetched};
    use crate::payload::DocPayload;
    use std::collections::HashMap;
    use std::time::Duration;

    /// A refresh mock returning a per-URL behavior.
    enum Behavior {
        NotModified,
        Changed(Vec<u8>),
        Gone,
    }

    struct RefreshMock {
        behavior: HashMap<String, Behavior>,
        /// Records which URLs were actually fetched (to prove no-fetch eviction).
        fetched: std::sync::Mutex<Vec<String>>,
    }

    impl RefreshMock {
        fn new(pairs: Vec<(&str, Behavior)>) -> Self {
            Self {
                behavior: pairs.into_iter().map(|(u, b)| (u.to_string(), b)).collect(),
                fetched: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl Fetcher for RefreshMock {
        type FetchFuture<'a>
            = std::future::Ready<Result<Fetched<'a>>>
        where
            Self: 'a;
        fn fetch<'a>(
            &'a self,
            resource: &'a Resource,
            _opts: FetchOptions<'a>,
        ) -> Self::FetchFuture<'a> {
            let key = crate::url_key::canonicalize(&resource.url);
            self.fetched.lock().unwrap().push(key.clone());
            let result = match self.behavior.get(&key) {
                Some(Behavior::NotModified) => Err(Error::not_modified(resource.url.as_str())),
                Some(Behavior::Changed(body)) => Ok(Fetched {
                    meta: FetchMeta {
                        kind: ResourceKind::Text,
                        etag: None,
                        status: 200,
                        final_url: None,
                    },
                    payload: DocPayload::Owned(Bytes::from(body.clone())),
                }),
                Some(Behavior::Gone) | None => Err(Error::not_found(resource.url.as_str())),
            };
            std::future::ready(result)
        }
    }

    #[tokio::test]
    async fn refresh_touches_changes_and_evicts() {
        let mut idx = LinkIndex::in_memory().unwrap();
        index_doc(&mut idx, "https://x.dev/keep", "keep body unchanged").await;
        index_doc(&mut idx, "https://x.dev/change", "old body").await;
        index_doc(&mut idx, "https://x.dev/dead", "dead body").await;
        assert_eq!(idx.len(), 3);

        let fetcher = RefreshMock::new(vec![
            ("https://x.dev/keep", Behavior::NotModified),
            (
                "https://x.dev/change",
                Behavior::Changed(b"brand new different content".to_vec()),
            ),
            ("https://x.dev/dead", Behavior::Gone),
        ]);
        let opts = RefreshOptions {
            urls: None,
            ttl: Duration::ZERO, // everything is stale
            max_age: None,
            evict_unreachable: true,
            concurrency: 4,
        };
        let report = idx.refresh_with(&fetcher, opts).await.unwrap();
        assert_eq!(report.unchanged, 1, "keep was 304");
        assert_eq!(report.refreshed, 1, "change re-indexed");
        assert_eq!(report.removed, 1, "dead evicted");
        assert_eq!(idx.len(), 2);
        let hits = idx.search("different", 5).await.unwrap();
        assert!(hits.iter().any(|h| h.url == "https://x.dev/change"));
    }

    #[tokio::test]
    async fn refresh_urls_subset_only_touches_listed() {
        let mut idx = LinkIndex::in_memory().unwrap();
        index_doc(&mut idx, "https://x.dev/one", "first body words").await;
        index_doc(&mut idx, "https://x.dev/two", "second body words").await;
        index_doc(&mut idx, "https://x.dev/three", "third body words").await;

        let fetcher = RefreshMock::new(vec![
            ("https://x.dev/one", Behavior::NotModified),
            ("https://x.dev/two", Behavior::NotModified),
            ("https://x.dev/three", Behavior::NotModified),
        ]);
        let opts = RefreshOptions {
            urls: Some(
                [crate::url_key::UrlKey::parse("https://x.dev/two").unwrap()]
                    .into_iter()
                    .collect(),
            ),
            ttl: Duration::ZERO,
            max_age: None,
            evict_unreachable: true,
            concurrency: 4,
        };
        let report = idx.refresh_with(&fetcher, opts).await.unwrap();
        assert_eq!(report.total(), 1, "only the listed URL was acted on");
        let requested = fetcher.fetched.lock().unwrap();
        assert_eq!(&*requested, &vec!["https://x.dev/two".to_string()]);
    }

    #[tokio::test]
    async fn export_surfaces_documents_and_edges() {
        let mut idx = LinkIndex::in_memory().unwrap();
        index_doc(
            &mut idx,
            "https://x.dev/hub",
            "<a href=\"/spoke\">s</a> hub body words",
        )
        .await;
        // index_doc uses AutoExtractor with Text kind, so use a second doc to
        // check metadata surfaces regardless of edges.
        index_doc(&mut idx, "https://x.dev/spoke", "spoke body words").await;

        let docs: Vec<_> = idx.export().unwrap().collect();
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().any(|d| d.meta.url == "https://x.dev/hub"));
        assert!(docs
            .iter()
            .all(|d| d.meta.url_key == crate::url_key::UrlKey::parse(&d.meta.url).unwrap()));
    }

    /// Regression: a refreshed body that is byte-identical but carries no `ETag`
    /// used to leave `fetched_at_ms` untouched, so the page was re-downloaded in
    /// full on every refresh forever (and miscounted as `refreshed`).
    #[tokio::test]
    async fn refresh_of_identical_body_without_etag_advances_freshness() {
        let mut idx = LinkIndex::in_memory().unwrap();
        index_doc(&mut idx, "https://x.dev/static", "stable page body words").await;

        let fetcher = RefreshMock::new(vec![(
            "https://x.dev/static",
            Behavior::Changed(b"stable page body words".to_vec()),
        )]);

        // First refresh: everything stale (fetched_at_ms == 0), body identical.
        let opts = RefreshOptions {
            urls: None,
            ttl: Duration::ZERO,
            max_age: None,
            evict_unreachable: true,
            concurrency: 2,
        };
        let report = idx.refresh_with(&fetcher, opts).await.unwrap();
        assert_eq!(report.unchanged, 1, "identical body counts as unchanged");
        assert_eq!(report.refreshed, 0);
        assert_eq!(fetcher.fetched.lock().unwrap().len(), 1);

        // Second refresh with a real TTL: the freshness stamp must have advanced,
        // so the page is no longer stale and nothing is fetched at all.
        let opts = RefreshOptions {
            urls: None,
            ttl: Duration::from_secs(3600),
            max_age: None,
            evict_unreachable: true,
            concurrency: 2,
        };
        let report = idx.refresh_with(&fetcher, opts).await.unwrap();
        assert_eq!(report.total(), 0, "no work: page is fresh now");
        assert_eq!(
            fetcher.fetched.lock().unwrap().len(),
            1,
            "no second network fetch for a fresh page"
        );
    }

    #[tokio::test]
    async fn max_age_evicts_unpinned_without_fetching() {
        let mut idx = LinkIndex::in_memory().unwrap();
        index_doc(&mut idx, "https://x.dev/old", "old").await; // fetched_at_ms = 0 → ancient
        let fetcher = RefreshMock::new(vec![]);
        let opts = RefreshOptions {
            urls: None,
            ttl: Duration::from_secs(u64::from(u32::MAX)), // never stale for refetch
            max_age: Some(Duration::from_millis(1)),       // but past the hard cap
            evict_unreachable: true,
            concurrency: 4,
        };
        let report = idx.refresh_with(&fetcher, opts).await.unwrap();
        assert_eq!(report.removed, 1);
        assert_eq!(idx.len(), 0);
        assert!(
            fetcher.fetched.lock().unwrap().is_empty(),
            "max-age eviction must not fetch"
        );
    }

    #[tokio::test]
    async fn graph_related_and_boost_roundtrip() {
        use crate::index::{Document, IndexBuilder};
        use crate::metric::Metric;
        use crate::resource::ResourceKind;
        use crate::url_key::UrlKey;

        // Build a tiny 3-node graph directly: hub -> a, hub -> b, a -> b.
        fn node(url: &str, terms: &[&str], targets: &[&str]) -> Document {
            let u = url::Url::parse(url).unwrap();
            let edges = targets
                .iter()
                .map(|t| UrlKey::from_url(&url::Url::parse(t).unwrap()))
                .collect();
            Document {
                url: u,
                kind: ResourceKind::Html,
                content_hash: url.len() as u64,
                title: None,
                snippet: "s".into(),
                lang: None,
                tags: smallvec::SmallVec::new(),
                terms: terms.iter().map(|t| (*t).into()).collect(),
                vector: vec![1.0],
                edges,
                fetched_at_ms: 0,
                etag: None,
                pinned: false,
            }
        }
        let mut b = IndexBuilder::new(1, Metric::Cosine, 0);
        b.upsert(node(
            "https://x.dev/hub",
            &["hub"],
            &["https://x.dev/a", "https://x.dev/b"],
        ))
        .unwrap();
        b.upsert(node("https://x.dev/a", &["alpha"], &["https://x.dev/b"]))
            .unwrap();
        b.upsert(node("https://x.dev/b", &["beta"], &[])).unwrap();

        // Round-trip through disk so the Edges section is exercised.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("g.lnkr");
        b.save(&path).unwrap();
        let index = crate::index::Index::open(&path).unwrap();

        // `related(a)` should surface b (direct target) and hub (co-parent).
        let rel = index.related(
            UrlKey::from_url(&url::Url::parse("https://x.dev/a").unwrap()),
            5,
        );
        let urls: Vec<&str> = rel.iter().map(|h| h.url).collect();
        assert!(urls.contains(&"https://x.dev/b"), "related: {urls:?}");
    }

    #[tokio::test]
    async fn pinned_survives_max_age() {
        let mut idx = LinkIndex::in_memory().unwrap();
        index_doc(&mut idx, "https://x.dev/pinned", "x").await;
        assert_eq!(idx.pin("https://x.dev/pinned").unwrap(), 1);
        let fetcher = RefreshMock::new(vec![]);
        let opts = RefreshOptions {
            urls: None,
            ttl: Duration::from_secs(u64::from(u32::MAX)),
            max_age: Some(Duration::from_millis(1)),
            evict_unreachable: true,
            concurrency: 4,
        };
        let report = idx.refresh_with(&fetcher, opts).await.unwrap();
        assert_eq!(report.removed, 0, "pinned exempt from max-age");
        assert_eq!(idx.len(), 1);
    }

    /// Test helper: extract + embed + upsert one in-memory document.
    async fn index_doc(idx: &mut LinkIndex, url: &str, body: &str) {
        idx.materialize_builder().unwrap(); // ensure a builder, invalidate cache
        let resource = Resource::new(url::Url::parse(url).unwrap()).with_kind(ResourceKind::Text);
        let descriptor = AutoExtractor.extract(&resource, body.as_bytes()).unwrap();
        let vector = embed_one(&idx.embedder, &descriptor.embed_text)
            .await
            .unwrap();
        let doc = descriptor.into_document(
            resource.url,
            resource.kind,
            xxh3_64(body.as_bytes()),
            vector,
        );
        idx.builder.as_mut().unwrap().upsert(doc).unwrap();
    }
}
