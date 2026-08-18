// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The recursive HTTP crawler — the primary source.
//!
//! Breadth-first from a parent URL to a configurable depth, scoped (by default to
//! the parent's path prefix), deduplicating by canonical URL so a page is fetched
//! once, with a polite inter-request delay. Generic over a [`Fetcher`], so it is
//! tested hermetically with a mock fetcher.

use crate::error::{Error, Result};
use crate::extract::html::extract_links;

/// Kind-aware outbound-link extraction for the crawl frontier: HTML uses the
/// byte scanner; Markdown uses the zero-dep Markdown link scanner (when that
/// extractor is compiled in); other kinds follow nothing.
fn extract_links_for(
    kind: crate::resource::ResourceKind,
    bytes: &[u8],
) -> Vec<compact_str::CompactString> {
    match kind {
        crate::resource::ResourceKind::Html => extract_links(bytes),
        #[cfg(feature = "markdown")]
        crate::resource::ResourceKind::Markdown => {
            crate::extract::markdown::extract_md_links(&String::from_utf8_lossy(bytes))
        }
        _ => Vec::new(),
    }
}
use crate::fetch::{FetchMeta, FetchOptions, Fetcher};
use crate::payload::DocPayload;
use crate::resource::{Page, Resource, SourceRef};
use crate::source::Source;
use crate::url_key::UrlKey;
use bytes::Bytes;
use compact_str::CompactString;
use futures::stream::{FuturesUnordered, StreamExt};
use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};
use url::Url;

/// Default crawl depth.
const DEFAULT_MAX_DEPTH: u16 = 2;
/// Default cap on total pages fetched.
const DEFAULT_MAX_PAGES: usize = 1000;
/// Default per-page byte ceiling (2 MiB).
const DEFAULT_MAX_BYTES: u64 = 2 * 1024 * 1024;
/// Default number of fetches in flight at once.
const DEFAULT_CONCURRENCY: usize = 8;
/// Default retry budget per page for retriable errors.
const DEFAULT_MAX_RETRIES: u32 = 2;
/// Base backoff for exponential retry (`RETRY_BASE * 2^attempt`).
const RETRY_BASE: Duration = Duration::from_millis(250);
/// Upper bound on any single backoff, clamping a hostile `Retry-After`.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(30);

/// Which discovered links the crawler follows.
#[derive(Clone, Debug, Default)]
pub enum CrawlScope {
    /// Only links at or below the parent URL's path (the default).
    #[default]
    PathPrefix,
    /// Any link on the same host.
    SameHost,
    /// The host and its subdomains.
    SameHostAndSubdomains,
    /// Only hosts matching the allowlist (exact host or `.suffix`).
    Allowlist(Vec<CompactString>),
}

impl CrawlScope {
    /// Whether `candidate` is in scope relative to the crawl `root`. Allocation-free
    /// (runs once per discovered link on the crawl hot path).
    #[must_use]
    pub fn allows(&self, root: &Url, candidate: &Url) -> bool {
        let (Some(root_host), Some(cand_host)) = (root.host_str(), candidate.host_str()) else {
            return false;
        };
        match self {
            CrawlScope::PathPrefix => {
                if cand_host != root_host {
                    return false;
                }
                // `cp` is under `rp` iff it equals `rp` or extends it at a '/'
                // boundary — a byte-slice check, no `format!` per link.
                let (rp, cp) = (root.path().as_bytes(), candidate.path().as_bytes());
                cp == rp
                    || (cp.len() > rp.len()
                        && cp.starts_with(rp)
                        && (rp.ends_with(b"/") || cp[rp.len()] == b'/'))
            }
            CrawlScope::SameHost => cand_host == root_host,
            CrawlScope::SameHostAndSubdomains => host_under(cand_host, root_host),
            CrawlScope::Allowlist(list) => list.iter().any(|h| {
                let base = h.as_str().strip_prefix('.').unwrap_or(h.as_str());
                host_under(cand_host, base)
            }),
        }
    }
}

/// Whether `host` equals `base` or is a subdomain of it (`a.b.dev` under `b.dev`).
/// Allocation-free.
fn host_under(host: &str, base: &str) -> bool {
    host == base
        || (host.len() > base.len()
            && host.ends_with(base)
            && host.as_bytes()[host.len() - base.len() - 1] == b'.')
}

/// Crawl configuration. Every field is honored; see [`CrawlSource`] builder methods.
#[derive(Clone, Debug)]
pub struct CrawlConfig {
    /// Maximum link depth from the root (root is depth 0).
    pub max_depth: u16,
    /// Which links to follow.
    pub scope: CrawlScope,
    /// Hard cap on total pages fetched (runaway guard).
    pub max_pages: usize,
    /// Polite delay between fetches.
    pub min_delay: Duration,
    /// Per-page byte ceiling.
    pub max_bytes: Option<u64>,
    /// User-Agent override.
    pub user_agent: Option<CompactString>,
    /// A followed link's URL path must contain *all* of these substrings (empty =
    /// no constraint). Composes with [`CrawlConfig::scope`]: use it to confine a
    /// crawl to a sub-path that the host's URL scheme splits across segments (e.g.
    /// GitHub's `/tree/` directories and `/blob/` files both under `/main/docs`).
    pub require_path_contains: Vec<CompactString>,
    /// Only index pages whose URL path ends with one of these extensions, e.g.
    /// `["md"]` (empty = index every indexable kind). Non-matching pages are still
    /// crawled for links — only their *indexing* is filtered.
    pub accept_extensions: Vec<CompactString>,
    /// Only index pages whose URL path contains *all* of these substrings (empty =
    /// no constraint). Independent of [`CrawlConfig::require_path_contains`]: follow
    /// broadly, index narrowly — e.g. follow GitHub `/tree/` dirs but index only the
    /// canonical `/blob/` file views, collapsing the `/raw/`,`/commits/` duplicates.
    pub index_path_contains: Vec<CompactString>,
    /// Maximum fetches in flight at once (clamped to at least 1). Higher values
    /// crawl faster; `min_delay` still paces request *starts* globally.
    pub concurrency: usize,
    /// Retry budget per page for retriable errors (timeouts, 429/5xx). Honors
    /// `Retry-After` on rate limits; other retries back off exponentially.
    pub max_retries: u32,
    /// Known entity tags by canonical-URL key: fetches send `If-None-Match`
    /// and a 304 skips the body transfer entirely (conditional re-crawl).
    pub validators: std::collections::HashMap<UrlKey, CompactString>,
    /// Known outbound links by canonical-URL key: when a page revalidates as
    /// unchanged (no body to parse), its previously indexed children keep the
    /// frontier going, so coverage never shrinks on a 304.
    pub known_edges: std::collections::HashMap<UrlKey, Vec<Url>>,
    /// Fetch and honor `robots.txt` per host (opt-in; requires the `robots` feature).
    #[cfg(feature = "robots")]
    pub respect_robots: bool,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_DEPTH,
            scope: CrawlScope::PathPrefix,
            max_pages: DEFAULT_MAX_PAGES,
            min_delay: Duration::ZERO,
            max_bytes: Some(DEFAULT_MAX_BYTES),
            user_agent: None,
            require_path_contains: Vec::new(),
            accept_extensions: Vec::new(),
            index_path_contains: Vec::new(),
            concurrency: DEFAULT_CONCURRENCY,
            max_retries: DEFAULT_MAX_RETRIES,
            validators: std::collections::HashMap::new(),
            known_edges: std::collections::HashMap::new(),
            #[cfg(feature = "robots")]
            respect_robots: false,
        }
    }
}

/// A recursive HTTP crawler over a [`Fetcher`].
#[derive(Debug)]
pub struct CrawlSource<F: Fetcher> {
    fetcher: F,
    config: CrawlConfig,
    /// Child pages that failed after retries during the last crawl (root
    /// failures abort instead). Readable after the stream completes.
    failed: std::sync::atomic::AtomicU32,
    /// Pages revalidated as unchanged (HTTP 304) during the last crawl. A 304
    /// transfers no body, so these pages never enter the page stream — without
    /// this record they would be invisible to the caller, and a consumer
    /// tracking freshness could never learn that the check happened.
    /// A `Mutex`, not a `RefCell`: the stream must stay `Send`.
    revalidated: std::sync::Mutex<Vec<Url>>,
}

impl<F: Fetcher> CrawlSource<F> {
    /// Create a crawler with default configuration.
    #[must_use]
    pub fn new(fetcher: F) -> Self {
        Self {
            fetcher,
            config: CrawlConfig::default(),
            failed: std::sync::atomic::AtomicU32::new(0),
            revalidated: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Create with explicit configuration.
    #[must_use]
    pub fn with_config(fetcher: F, config: CrawlConfig) -> Self {
        Self {
            fetcher,
            config,
            failed: std::sync::atomic::AtomicU32::new(0),
            revalidated: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Set the crawl depth.
    #[must_use]
    pub fn depth(mut self, max_depth: u16) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    /// Set the page cap.
    #[must_use]
    pub fn max_pages(mut self, max_pages: usize) -> Self {
        self.config.max_pages = max_pages;
        self
    }

    /// Set the crawl scope.
    #[must_use]
    pub fn scope(mut self, scope: CrawlScope) -> Self {
        self.config.scope = scope;
        self
    }

    /// Set the polite inter-request delay.
    #[must_use]
    pub fn min_delay(mut self, delay: Duration) -> Self {
        self.config.min_delay = delay;
        self
    }

    /// Set the maximum number of fetches in flight at once (clamped to ≥ 1).
    #[must_use]
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.config.concurrency = concurrency;
        self
    }

    /// Set the per-page retry budget for retriable errors.
    #[must_use]
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.config.max_retries = max_retries;
        self
    }

    /// Fetch and honor `robots.txt` per host (opt-in).
    #[cfg(feature = "robots")]
    #[must_use]
    pub fn respect_robots(mut self, respect: bool) -> Self {
        self.config.respect_robots = respect;
        self
    }

    /// Require that every followed link's URL path contains `substring` (additive;
    /// all required substrings must be present). Confines the crawl to a sub-path.
    #[must_use]
    pub fn require_path(mut self, substring: impl Into<CompactString>) -> Self {
        self.config.require_path_contains.push(substring.into());
        self
    }

    /// Only index pages whose URL path ends with `extension` (additive). Other
    /// pages are still crawled for links.
    #[must_use]
    pub fn accept_extension(mut self, extension: impl AsRef<str>) -> Self {
        self.config
            .accept_extensions
            .push(normalize_ext(extension.as_ref()));
        self
    }

    /// Only index pages whose URL path contains `substring` (additive). Other pages
    /// are still crawled for links — this narrows *indexing*, not crawling.
    #[must_use]
    pub fn index_path(mut self, substring: impl Into<CompactString>) -> Self {
        self.config.index_path_contains.push(substring.into());
        self
    }

    /// Provide known validators (`If-None-Match`) for conditional re-crawls.
    #[must_use]
    pub fn validators(
        mut self,
        entries: impl IntoIterator<Item = (UrlKey, CompactString)>,
    ) -> Self {
        self.config.validators = entries.into_iter().collect();
        self
    }

    /// Provide previously indexed outbound links so 304 pages keep feeding
    /// the frontier.
    #[must_use]
    pub fn known_edges(mut self, entries: impl IntoIterator<Item = (UrlKey, Vec<Url>)>) -> Self {
        self.config.known_edges = entries.into_iter().collect();
        self
    }

    /// Child pages that failed after retries during the most recent crawl.
    #[must_use]
    pub fn failed_count(&self) -> u32 {
        self.failed.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Drain the pages revalidated as unchanged (HTTP 304) during the most
    /// recent crawl. They transferred no body and yielded no [`Page`], so this
    /// is the only place their revalidation is visible — a freshness tracker
    /// consumes it to record "checked, unchanged" against each URL.
    #[must_use]
    pub fn take_revalidated(&self) -> Vec<Url> {
        std::mem::take(&mut self.revalidated.lock().expect("revalidated lock poisoned"))
    }

    /// Read-only view of the configuration.
    #[must_use]
    pub fn config(&self) -> &CrawlConfig {
        &self.config
    }
}

/// Strip a leading dot and lowercase an extension (`".MD"` → `"md"`).
fn normalize_ext(ext: &str) -> CompactString {
    CompactString::from(ext.trim_start_matches('.').to_ascii_lowercase())
}

/// Whether `url`'s path contains every required substring (vacuously true if none).
fn path_allowed(url: &Url, required: &[CompactString]) -> bool {
    let path = url.path();
    required.iter().all(|s| path.contains(s.as_str()))
}

/// The extension of a URL path's final segment, if any — borrowed, no allocation.
/// Comparisons against it use `eq_ignore_ascii_case`.
fn ext_of(path: &str) -> Option<&str> {
    path.rsplit('/')
        .next()
        .unwrap_or("")
        .rsplit_once('.')
        .map(|(_, e)| e)
}

/// Whether `url`'s final path segment ends with one of `extensions` (vacuously
/// true if the list is empty). `extensions` are already normalized (dot-stripped,
/// lowercased) by [`normalize_ext`]; the compare is case-insensitive regardless.
fn extension_allowed(url: &Url, extensions: &[CompactString]) -> bool {
    if extensions.is_empty() {
        return true;
    }
    ext_of(url.path()).is_some_and(|e| {
        extensions
            .iter()
            .any(|allow| allow.as_str().eq_ignore_ascii_case(e))
    })
}

/// Extensions that clearly point at non-document assets the crawler skips.
const ASSET_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "css", "js", "mjs", "woff", "woff2", "ttf",
    "eot", "zip", "tar", "gz", "mp4", "mp3", "wav", "avi", "mov", "wasm", "map",
];

/// Whether a URL clearly points at a non-document asset we should not crawl.
/// Allocation-free (`eq_ignore_ascii_case`, no per-link lowercasing).
fn is_asset(url: &Url) -> bool {
    ext_of(url.path()).is_some_and(|e| ASSET_EXTS.iter().any(|a| e.eq_ignore_ascii_case(a)))
}

/// The backoff before a given retry attempt: `Retry-After` (clamped) for rate
/// limits, otherwise exponential from [`RETRY_BASE`].
fn backoff_for(err: &Error, attempt: u32) -> Duration {
    match err {
        Error::RateLimited { retry_after_ms } => {
            Duration::from_millis(*retry_after_ms).min(MAX_RETRY_AFTER)
        }
        _ => RETRY_BASE.saturating_mul(2u32.saturating_pow(attempt)),
    }
}

/// One paced, retried fetch. The returned future **owns** its [`Resource`], so the
/// `&resource` borrow handed to [`Fetcher::fetch`] lives entirely inside the future
/// — this is what lets many fetches run concurrently in a `FuturesUnordered` while
/// keeping the trait's borrowing signature. The requested `Url` is moved back out
/// on completion (no clone), tagged with its crawl `depth`.
#[allow(clippy::too_many_arguments)] // one bounded call site in the fill loop
async fn fetch_one<'a, F: Fetcher>(
    fetcher: &'a F,
    url: Url,
    depth: u16,
    start_delay: Duration,
    max_bytes: Option<u64>,
    user_agent: Option<&'a str>,
    max_retries: u32,
    etag: Option<CompactString>,
) -> (Url, u16, Result<(FetchMeta, Bytes)>) {
    if !start_delay.is_zero() {
        futures_timer::Delay::new(start_delay).await;
    }
    let resource = Resource::new(url);
    let opts = FetchOptions {
        if_none_match: etag.as_deref(),
        max_bytes,
        user_agent,
    };
    let mut attempt = 0u32;
    let outcome = loop {
        match fetcher.fetch(&resource, opts).await {
            Ok(got) => break got.payload.into_bytes().await.map(|b| (got.meta, b)),
            Err(err) if attempt < max_retries && err.is_retriable() => {
                futures_timer::Delay::new(backoff_for(&err, attempt)).await;
                attempt += 1;
            }
            Err(err) => break Err(err),
        }
    };
    (resource.url, depth, outcome)
}

/// Global request-start pacer. Assigns each launch a monotonically increasing
/// start slot so request *starts* are at least `interval` apart regardless of how
/// many workers run — `interval == 0` yields full concurrency. It never blocks the
/// crawl loop: the returned delay is awaited inside the worker future.
struct Pacer {
    /// Next free start slot **per host**: politeness is a per-server
    /// courtesy, so a multi-host crawl (subdomains, allowlist) neither
    /// over-throttles across hosts nor lets one host absorb another's slots.
    next_free: std::collections::HashMap<CompactString, Instant>,
}

impl Pacer {
    fn new() -> Self {
        Self {
            next_free: std::collections::HashMap::new(),
        }
    }

    /// Reserve the next start slot for `host`; returns how long that worker
    /// must sleep first.
    fn reserve(&mut self, host: &str, interval: Duration) -> Duration {
        if interval.is_zero() {
            return Duration::ZERO;
        }
        let now = Instant::now();
        let entry = self
            .next_free
            .entry(CompactString::from(host))
            .or_insert(now);
        let slot = (*entry).max(now);
        *entry = slot + interval;
        slot - now
    }
}

/// Per-host `robots.txt` policy, fetched once and cached for the crawl.
#[cfg(feature = "robots")]
struct HostPolicy {
    /// The parsed rules, or `None` to allow everything (missing/invalid robots).
    robot: Option<texting_robots::Robot>,
    /// The host's advertised crawl delay (folded into the pacer).
    crawl_delay: Duration,
}

/// The decision for one candidate URL.
#[cfg(feature = "robots")]
enum Verdict {
    Allowed { crawl_delay: Duration },
    Disallowed,
}

/// Caches `robots.txt` per host (keyed by scheme+authority), fetched through the
/// same [`Fetcher`] (so a host-scoped auth token is applied on the same host and
/// no bearer leaks to a foreign host). Cheap: one fetch per host, and crawls are
/// typically single-host.
#[cfg(feature = "robots")]
struct RobotsCache {
    by_host: std::collections::HashMap<CompactString, HostPolicy>,
    agent: CompactString,
}

#[cfg(feature = "robots")]
impl RobotsCache {
    /// Maximum bytes read for a `robots.txt` (they are small; guards abuse).
    const ROBOTS_MAX_BYTES: u64 = 512 * 1024;

    fn new(agent: CompactString) -> Self {
        Self {
            by_host: std::collections::HashMap::new(),
            agent,
        }
    }

    /// The scheme+authority key a robots policy is cached under.
    fn host_key(url: &Url) -> CompactString {
        let mut key = CompactString::from(url.scheme());
        key.push_str("://");
        if let Some(host) = url.host_str() {
            key.push_str(host);
        }
        if let Some(port) = url.port() {
            key.push(':');
            key.push_str(&port.to_string());
        }
        key
    }

    async fn check<F: Fetcher>(
        &mut self,
        fetcher: &F,
        url: &Url,
        user_agent: Option<&str>,
    ) -> Verdict {
        let key = Self::host_key(url);
        if !self.by_host.contains_key(&key) {
            let policy = Self::fetch_policy(fetcher, url, &self.agent, user_agent).await;
            self.by_host.insert(key.clone(), policy);
        }
        let policy = &self.by_host[&key];
        match &policy.robot {
            Some(robot) if !robot.allowed(url.as_str()) => Verdict::Disallowed,
            _ => Verdict::Allowed {
                crawl_delay: policy.crawl_delay,
            },
        }
    }

    async fn fetch_policy<F: Fetcher>(
        fetcher: &F,
        url: &Url,
        agent: &str,
        user_agent: Option<&str>,
    ) -> HostPolicy {
        // Any failure (network, non-200, invalid syntax) means "allow everything".
        let allow_all = HostPolicy {
            robot: None,
            crawl_delay: Duration::ZERO,
        };
        let mut robots_url = url.clone();
        robots_url.set_path("/robots.txt");
        robots_url.set_query(None);
        robots_url.set_fragment(None);
        let resource = Resource::new(robots_url);
        let opts = FetchOptions {
            if_none_match: None,
            max_bytes: Some(Self::ROBOTS_MAX_BYTES),
            user_agent,
        };
        let Ok(resp) = fetcher.fetch(&resource, opts).await else {
            return allow_all;
        };
        let Ok(bytes) = resp.payload.into_bytes().await else {
            return allow_all;
        };
        match texting_robots::Robot::new(agent, &bytes) {
            Ok(robot) => {
                let crawl_delay = robot
                    .delay
                    .and_then(|d| (d.is_finite() && d >= 0.0).then(|| Duration::from_secs_f32(d)))
                    .unwrap_or(Duration::ZERO);
                HostPolicy {
                    robot: Some(robot),
                    crawl_delay,
                }
            }
            Err(_) => allow_all,
        }
    }
}

impl<F: Fetcher> Source for CrawlSource<F> {
    type Pages<'a>
        = impl futures::Stream<Item = Result<Page<'a>>> + Send + 'a
    where
        Self: 'a;

    fn kind(&self) -> &'static str {
        "crawl"
    }

    #[allow(clippy::too_many_lines)] // one cohesive crawl state machine
    fn discover<'a>(&'a self, root: &'a SourceRef) -> Self::Pages<'a> {
        async_stream::try_stream! {
            let SourceRef::Http { root: root_url } = root else {
                return Err(Error::crawl("crawl", "CrawlSource requires an Http SourceRef"))?;
            };

            let cfg = &self.config;
            let concurrency = cfg.concurrency.max(1);
            let ua = cfg.user_agent.as_deref();

            // Crawl state lives entirely in this single-threaded stream body — the
            // frontier, visited set, and pacer are plain locals, so there are no
            // locks and no shared-mutable state despite N concurrent fetches.
            let mut visited: HashSet<UrlKey> = HashSet::new();
            let mut frontier: VecDeque<(Url, u16)> = VecDeque::new();
            visited.insert(UrlKey::from_url(root_url));
            frontier.push_back((root_url.clone(), 0));

            let mut pacer = Pacer::new();
            let mut in_flight = FuturesUnordered::new();
            let mut successes = 0usize;
            // Per-crawl records: reset here so "most recent crawl" stays true
            // when one source runs discover() more than once.
            self.failed.store(0, std::sync::atomic::Ordering::Relaxed);
            self.revalidated.lock().expect("revalidated lock poisoned").clear();

            #[cfg(feature = "robots")]
            let mut robots = RobotsCache::new(
                cfg.user_agent
                    .clone()
                    .unwrap_or_else(|| CompactString::from(crate::fetch::DEFAULT_UA)),
            );

            loop {
                // Fill idle slots up to `concurrency`, never launching past the page
                // budget. A failed fetch releases its budget (successes doesn't grow),
                // matching the sequential semantics.
                while in_flight.len() < concurrency
                    && successes + in_flight.len() < cfg.max_pages
                {
                    let Some((url, depth)) = frontier.pop_front() else {
                        break;
                    };
                    // Pacing interval for this request start: `min_delay`, widened
                    // by any robots.txt crawl-delay when robots is honored.
                    #[cfg(feature = "robots")]
                    let interval = if cfg.respect_robots {
                        match robots.check(&self.fetcher, &url, ua).await {
                            Verdict::Allowed { crawl_delay } => cfg.min_delay.max(crawl_delay),
                            Verdict::Disallowed => {
                                if depth == 0 {
                                    return Err(Error::crawl(
                                        "crawl",
                                        "root URL is disallowed by robots.txt",
                                    ))?;
                                }
                                continue; // skip a disallowed child
                            }
                        }
                    } else {
                        cfg.min_delay
                    };
                    #[cfg(not(feature = "robots"))]
                    let interval = cfg.min_delay;

                    let delay = pacer.reserve(url.host_str().unwrap_or(""), interval);
                    let etag = cfg.validators.get(&UrlKey::from_url(&url)).cloned();
                    in_flight.push(fetch_one(
                        &self.fetcher,
                        url,
                        depth,
                        delay,
                        cfg.max_bytes,
                        ua,
                        cfg.max_retries,
                        etag,
                    ));
                }

                let Some((req_url, depth, outcome)) = in_flight.next().await else {
                    break; // frontier drained and nothing in flight
                };
                let (meta, bytes) = match outcome {
                    Ok(v) => v,
                    Err(Error::NotModified { .. }) => {
                        // Revalidated 304: no body transferred, nothing to
                        // re-index — but the page's previously indexed children
                        // keep feeding the frontier so coverage never shrinks,
                        // and the check itself is recorded for freshness
                        // consumers (it is invisible in the page stream).
                        let key = UrlKey::from_url(&req_url);
                        visited.insert(key);
                        self.revalidated
                            .lock()
                            .expect("revalidated lock poisoned")
                            .push(req_url.clone());
                        if depth < cfg.max_depth {
                            if let Some(children) = cfg.known_edges.get(&key) {
                                for child in children {
                                    if is_asset(child)
                                        || !cfg.scope.allows(root_url, child)
                                        || !path_allowed(child, &cfg.require_path_contains)
                                    {
                                        continue;
                                    }
                                    if visited.insert(UrlKey::from_url(child)) {
                                        frontier.push_back((child.clone(), depth + 1));
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    Err(e) => {
                        if depth == 0 {
                            return Err(e)?; // the root must succeed
                        }
                        self.failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        continue; // skip a single failed page, freeing its budget
                    }
                };
                successes += 1;

                // Follow any redirect: index/scope/link-resolve against the final
                // location so a redirect can't escape scope or split the dedup key.
                let effective = meta.final_url.unwrap_or(req_url);
                if depth > 0 && !cfg.scope.allows(root_url, &effective) {
                    continue; // a child redirected out of scope
                }
                visited.insert(UrlKey::from_url(&effective));
                let kind = meta.kind;

                // Enqueue in-scope child links before yielding (we own the bytes).
                if depth < cfg.max_depth && kind.is_linkable() {
                    for href in extract_links_for(kind, &bytes) {
                        let Ok(mut child) = effective.join(&href) else {
                            continue;
                        };
                        child.set_fragment(None);
                        if child.scheme() != "http" && child.scheme() != "https" {
                            continue;
                        }
                        if is_asset(&child)
                            || !cfg.scope.allows(root_url, &child)
                            || !path_allowed(&child, &cfg.require_path_contains)
                        {
                            continue;
                        }
                        if visited.insert(UrlKey::from_url(&child)) {
                            frontier.push_back((child, depth + 1));
                        }
                    }
                }

                // Index filters govern indexing only (links were already followed
                // above). Skip yielding pages that don't match file-type / path.
                if !extension_allowed(&effective, &cfg.accept_extensions)
                    || !path_allowed(&effective, &cfg.index_path_contains)
                {
                    continue;
                }

                let mut out = Resource::new(effective).with_kind(kind);
                if let Some(tag) = meta.etag {
                    out = out.with_etag(tag);
                }
                yield Page::new(out, DocPayload::Owned(bytes));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::{FetchMeta, Fetched};
    use crate::resource::ResourceKind;
    use bytes::Bytes;
    use futures::StreamExt;
    use std::collections::HashMap;
    use std::future::Ready;

    /// An in-memory mock site: URL → (content-type kind, body).
    struct MockSite {
        pages: HashMap<String, (ResourceKind, Vec<u8>)>,
    }

    impl Fetcher for MockSite {
        type FetchFuture<'a> = Ready<Result<Fetched<'a>>>;
        fn fetch<'a>(
            &'a self,
            resource: &'a Resource,
            _opts: FetchOptions<'a>,
        ) -> Self::FetchFuture<'a> {
            let key = crate::url_key::canonicalize(&resource.url);
            let result = match self.pages.get(&key) {
                Some((kind, body)) => Ok(Fetched {
                    meta: FetchMeta {
                        kind: *kind,
                        etag: None,
                        status: 200,
                        final_url: None,
                    },
                    payload: DocPayload::Owned(Bytes::from(body.clone())),
                }),
                None => Err(Error::not_found(resource.url.as_str())),
            };
            std::future::ready(result)
        }
    }

    fn page(href_targets: &[&str]) -> Vec<u8> {
        use std::fmt::Write as _;
        let mut html = String::from("<html><body><p>content here</p>");
        for t in href_targets {
            let _ = write!(html, "<a href=\"{t}\">link</a>");
        }
        html.push_str("</body></html>");
        html.into_bytes()
    }

    fn mock_site() -> MockSite {
        let mut pages = HashMap::new();
        let h = ResourceKind::Html;
        // root links to /docs/a and /docs/b; a links to /docs/c; also an off-scope link.
        pages.insert(
            "https://x.dev/docs".to_string(),
            (
                h,
                page(&["/docs/a", "/docs/b", "https://other.dev/x", "/about"]),
            ),
        );
        pages.insert("https://x.dev/docs/a".to_string(), (h, page(&["/docs/c"])));
        pages.insert("https://x.dev/docs/b".to_string(), (h, page(&[])));
        pages.insert("https://x.dev/docs/c".to_string(), (h, page(&[])));
        pages.insert("https://x.dev/about".to_string(), (h, page(&[]))); // out of path scope
        pages.insert("https://other.dev/x".to_string(), (h, page(&[]))); // off host
        MockSite { pages }
    }

    async fn crawl_urls<F: Fetcher>(source: &CrawlSource<F>, root: &SourceRef) -> Vec<String> {
        let mut urls = Vec::new();
        let stream = source.discover(root);
        futures::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            urls.push(item.unwrap().resource.url.to_string());
        }
        urls
    }

    #[tokio::test]
    async fn crawls_within_path_prefix_to_depth() {
        let source = CrawlSource::new(mock_site()).depth(2);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;

        // path-prefix scope: /about and other.dev excluded; /docs/* included to depth 2.
        assert!(urls.contains(&"https://x.dev/docs".to_string()));
        assert!(urls.contains(&"https://x.dev/docs/a".to_string()));
        assert!(urls.contains(&"https://x.dev/docs/b".to_string()));
        assert!(urls.contains(&"https://x.dev/docs/c".to_string()));
        assert!(!urls.iter().any(|u| u.contains("about")));
        assert!(!urls.iter().any(|u| u.contains("other.dev")));
    }

    #[tokio::test]
    async fn markdown_pages_are_followed_and_yield_links() {
        let mut pages = HashMap::new();
        pages.insert(
            "https://x.dev/docs".to_string(),
            (
                ResourceKind::Markdown,
                b"# Docs\n\nSee [the guide](/docs/guide) and <https://x.dev/docs/ref>.\n".to_vec(),
            ),
        );
        pages.insert(
            "https://x.dev/docs/guide".to_string(),
            (ResourceKind::Markdown, b"# Guide\n\nBody text.\n".to_vec()),
        );
        pages.insert(
            "https://x.dev/docs/ref".to_string(),
            (ResourceKind::Markdown, b"# Ref\n\nBody text.\n".to_vec()),
        );
        let source = CrawlSource::new(MockSite { pages }).depth(2);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        assert!(
            urls.contains(&"https://x.dev/docs/guide".to_string()),
            "inline md link followed"
        );
        assert!(
            urls.contains(&"https://x.dev/docs/ref".to_string()),
            "autolink followed"
        );
    }

    /// A site that answers 304 when the request carries the matching etag,
    /// and counts full-body responses.
    struct EtagSite {
        pages: HashMap<String, (ResourceKind, Vec<u8>, &'static str)>,
        bodies_served: std::sync::atomic::AtomicU32,
    }

    impl Fetcher for EtagSite {
        type FetchFuture<'a> = Ready<Result<Fetched<'a>>>;
        fn fetch<'a>(
            &'a self,
            resource: &'a Resource,
            opts: FetchOptions<'a>,
        ) -> Self::FetchFuture<'a> {
            let key = crate::url_key::canonicalize(&resource.url);
            let result = match self.pages.get(&key) {
                Some((_, _, etag)) if opts.if_none_match == Some(etag) => {
                    Err(Error::not_modified(resource.url.as_str()))
                }
                Some((kind, body, etag)) => {
                    self.bodies_served
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(Fetched {
                        meta: FetchMeta {
                            kind: *kind,
                            etag: Some((*etag).into()),
                            status: 200,
                            final_url: None,
                        },
                        payload: DocPayload::Owned(Bytes::from(body.clone())),
                    })
                }
                None => Err(Error::not_found(resource.url.as_str())),
            };
            std::future::ready(result)
        }
    }

    #[tokio::test]
    async fn revalidated_pages_transfer_no_body_and_known_edges_keep_coverage() {
        let mut pages = HashMap::new();
        let h = ResourceKind::Html;
        pages.insert(
            "https://x.dev/docs".to_string(),
            (h, page(&["/docs/a"]), "\"r\""),
        );
        pages.insert(
            "https://x.dev/docs/a".to_string(),
            (h, page(&["/docs/c"]), "\"a\""),
        );
        pages.insert("https://x.dev/docs/c".to_string(), (h, page(&[]), "\"c\""));
        let site = EtagSite {
            pages,
            bodies_served: std::sync::atomic::AtomicU32::new(0),
        };

        let root_url = url::Url::parse("https://x.dev/docs").unwrap();
        let a_url = url::Url::parse("https://x.dev/docs/a").unwrap();
        let c_url = url::Url::parse("https://x.dev/docs/c").unwrap();
        let (rk, ak) = (UrlKey::from_url(&root_url), UrlKey::from_url(&a_url));

        // Everything known + unchanged: zero bodies transferred, and the
        // stored edges keep the whole frontier reachable (c is visited via
        // known edges of a, itself revalidated).
        let source = CrawlSource::new(site)
            .depth(2)
            .validators([
                (rk, "\"r\"".into()),
                (ak, "\"a\"".into()),
                (UrlKey::from_url(&c_url), "\"c\"".into()),
            ])
            .known_edges([(rk, vec![a_url.clone()]), (ak, vec![c_url.clone()])]);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        assert!(urls.is_empty(), "no page yielded: everything revalidated");
        assert_eq!(
            source
                .fetcher
                .bodies_served
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );

        // Only the leaf changed: exactly one body transfers.
        let mut pages = HashMap::new();
        pages.insert(
            "https://x.dev/docs".to_string(),
            (h, page(&["/docs/a"]), "\"r\""),
        );
        pages.insert(
            "https://x.dev/docs/a".to_string(),
            (h, page(&["/docs/c"]), "\"a\""),
        );
        pages.insert("https://x.dev/docs/c".to_string(), (h, page(&[]), "\"c2\""));
        let site = EtagSite {
            pages,
            bodies_served: std::sync::atomic::AtomicU32::new(0),
        };
        let source = CrawlSource::new(site)
            .depth(2)
            .validators([
                (rk, "\"r\"".into()),
                (ak, "\"a\"".into()),
                (UrlKey::from_url(&c_url), "\"c\"".into()),
            ])
            .known_edges([(rk, vec![a_url]), (ak, vec![c_url])]);
        let urls = crawl_urls(&source, &root).await;
        assert_eq!(urls, vec!["https://x.dev/docs/c".to_string()]);
        assert_eq!(
            source
                .fetcher
                .bodies_served
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            source.take_revalidated().len(),
            2,
            "root and a revalidated; c changed"
        );
    }

    #[tokio::test]
    async fn child_failures_are_counted_not_silent() {
        let mut pages = HashMap::new();
        let h = ResourceKind::Html;
        pages.insert(
            "https://x.dev/docs".to_string(),
            (h, page(&["/docs/dead", "/docs/b"])),
        );
        pages.insert("https://x.dev/docs/b".to_string(), (h, page(&[])));
        // /docs/dead is absent → NotFound (non-retriable) → counted.
        let source = CrawlSource::new(MockSite { pages }).depth(1);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        assert_eq!(urls.len(), 2, "root + b");
        assert_eq!(source.failed_count(), 1, "the dead child is counted");
    }

    #[tokio::test]
    async fn depth_zero_only_fetches_root() {
        let source = CrawlSource::new(mock_site()).depth(0);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        assert_eq!(urls, vec!["https://x.dev/docs".to_string()]);
    }

    #[tokio::test]
    async fn depth_one_stops_before_grandchildren() {
        let source = CrawlSource::new(mock_site()).depth(1);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        // root + a + b, but not c (a grandchild reached via a).
        assert!(urls.contains(&"https://x.dev/docs/a".to_string()));
        assert!(!urls.contains(&"https://x.dev/docs/c".to_string()));
    }

    #[tokio::test]
    async fn max_pages_caps_the_crawl() {
        let source = CrawlSource::new(mock_site()).depth(5).max_pages(2);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        assert_eq!(urls.len(), 2);
    }

    #[tokio::test]
    async fn dedups_revisited_links() {
        // a page that links to itself and a sibling repeatedly.
        let mut pages = HashMap::new();
        pages.insert(
            "https://x.dev/docs".to_string(),
            (ResourceKind::Html, page(&["/docs", "/docs/a", "/docs/a"])),
        );
        pages.insert(
            "https://x.dev/docs/a".to_string(),
            (ResourceKind::Html, page(&["/docs"])),
        );
        let source = CrawlSource::new(MockSite { pages }).depth(3);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        assert_eq!(urls.len(), 2, "each canonical URL fetched once");
    }

    #[tokio::test]
    async fn same_host_scope_allows_other_paths() {
        let source = CrawlSource::new(mock_site())
            .depth(2)
            .scope(CrawlScope::SameHost);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        assert!(
            urls.iter().any(|u| u.contains("/about")),
            "same-host allows /about"
        );
        assert!(!urls.iter().any(|u| u.contains("other.dev")));
    }

    /// A GitHub-shaped site: `/tree/` directories and `/blob/` files, plus an
    /// out-of-docs source file, an external link, and an image asset.
    fn github_mock() -> MockSite {
        let mut pages = HashMap::new();
        let h = ResourceKind::Html;
        pages.insert(
            "https://gh.dev/o/s4/tree/main/docs".to_string(),
            (
                h,
                page(&[
                    "/o/s4/blob/main/docs/install.md",
                    "/o/s4/tree/main/docs/ops",
                    "/o/s4/blob/main/docs/diagram.png", // asset
                    "/o/s4/blob/main/src/lib.rs",       // out of /main/docs
                    "https://external.dev/x",           // off host
                ]),
            ),
        );
        pages.insert(
            "https://gh.dev/o/s4/blob/main/docs/install.md".to_string(),
            (h, page(&["/o/s4/blob/main/docs/configuration.md"])),
        );
        pages.insert(
            "https://gh.dev/o/s4/blob/main/docs/configuration.md".to_string(),
            (h, page(&[])),
        );
        pages.insert(
            "https://gh.dev/o/s4/tree/main/docs/ops".to_string(),
            (h, page(&["/o/s4/blob/main/docs/ops/runbook.md"])),
        );
        pages.insert(
            "https://gh.dev/o/s4/blob/main/docs/ops/runbook.md".to_string(),
            (h, page(&[])),
        );
        pages.insert(
            "https://gh.dev/o/s4/blob/main/src/lib.rs".to_string(),
            (h, page(&[])),
        );
        pages.insert("https://external.dev/x".to_string(), (h, page(&[])));
        MockSite { pages }
    }

    #[tokio::test]
    async fn path_filter_confines_crawl_to_repo_subpath() {
        // Same-host keeps us on gh.dev; require_path confines to this repo's docs,
        // following both `/tree/` dirs and `/blob/` files.
        let source = CrawlSource::new(github_mock())
            .depth(4)
            .scope(CrawlScope::SameHost)
            .require_path("/o/s4")
            .require_path("/main/docs");
        let root = SourceRef::http("https://gh.dev/o/s4/tree/main/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;

        assert!(urls.iter().any(|u| u.ends_with("/docs/install.md")));
        assert!(urls.iter().any(|u| u.ends_with("/docs/configuration.md")));
        assert!(
            urls.iter().any(|u| u.ends_with("/docs/ops/runbook.md")),
            "followed tree → blob"
        );
        // Excluded: source file (no /main/docs), external host, image asset.
        assert!(!urls.iter().any(|u| u.contains("/src/lib.rs")));
        assert!(!urls.iter().any(|u| u.contains("external.dev")));
        assert!(!urls.iter().any(|u| u.contains(".png")));
    }

    #[tokio::test]
    async fn file_type_filter_indexes_only_markdown() {
        let source = CrawlSource::new(github_mock())
            .depth(4)
            .scope(CrawlScope::SameHost)
            .require_path("/o/s4")
            .require_path("/main/docs")
            .accept_extension("md");
        let root = SourceRef::http("https://gh.dev/o/s4/tree/main/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;

        // Only `.md` pages are indexed; the `/tree/` directory pages are crawled
        // for links (so runbook.md is reached) but not themselves indexed.
        assert!(
            urls.iter().all(|u| u.rsplit('.').next() == Some("md")),
            "non-md indexed: {urls:?}"
        );
        assert!(
            urls.iter().any(|u| u.ends_with("/docs/ops/runbook.md")),
            "tree page was still followed"
        );
        assert!(!urls.iter().any(|u| u.ends_with("/tree/main/docs")));
        assert_eq!(urls.len(), 3); // install, configuration, runbook
    }

    #[tokio::test]
    async fn index_path_filter_collapses_duplicate_views() {
        // GitHub exposes each file at /blob/, /raw/, and /commits/. Follow broadly
        // within /main/docs, but index only the canonical /blob/ view.
        let mut pages = HashMap::new();
        let h = ResourceKind::Html;
        pages.insert(
            "https://gh.dev/o/s4/tree/main/docs".to_string(),
            (h, page(&["/o/s4/blob/main/docs/a.md"])),
        );
        pages.insert(
            "https://gh.dev/o/s4/blob/main/docs/a.md".to_string(),
            (
                h,
                page(&["/o/s4/raw/main/docs/a.md", "/o/s4/commits/main/docs/a.md"]),
            ),
        );
        pages.insert(
            "https://gh.dev/o/s4/raw/main/docs/a.md".to_string(),
            (h, page(&[])),
        );
        pages.insert(
            "https://gh.dev/o/s4/commits/main/docs/a.md".to_string(),
            (h, page(&[])),
        );

        let source = CrawlSource::new(MockSite { pages })
            .depth(4)
            .scope(CrawlScope::SameHost)
            .require_path("/main/docs")
            .index_path("/blob/");
        let root = SourceRef::http("https://gh.dev/o/s4/tree/main/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;

        // Exactly one indexed entry — the blob view; raw/commits/tree excluded.
        assert_eq!(urls.len(), 1, "duplicate views collapsed: {urls:?}");
        assert!(urls[0].contains("/blob/main/docs/a.md"));
        assert!(!urls
            .iter()
            .any(|u| u.contains("/raw/") || u.contains("/commits/")));
    }

    #[test]
    fn extension_and_path_helpers() {
        let md = Url::parse("https://gh.dev/o/s4/blob/main/docs/a.md").unwrap();
        let dir = Url::parse("https://gh.dev/o/s4/tree/main/docs").unwrap();
        let exts = [CompactString::from("md")];
        assert!(extension_allowed(&md, &exts));
        assert!(!extension_allowed(&dir, &exts));
        assert!(extension_allowed(&dir, &[])); // empty = all

        let req = [
            CompactString::from("/o/s4"),
            CompactString::from("/main/docs"),
        ];
        assert!(path_allowed(&md, &req));
        assert!(!path_allowed(
            &Url::parse("https://gh.dev/o/s4/blob/main/src/x.rs").unwrap(),
            &req
        ));
    }

    #[tokio::test]
    async fn concurrency_yields_same_set_as_sequential() {
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let seq = CrawlSource::new(mock_site()).depth(3).concurrency(1);
        let par = CrawlSource::new(mock_site()).depth(3).concurrency(4);
        let mut a = crawl_urls(&seq, &root).await;
        let mut b = crawl_urls(&par, &root).await;
        a.sort();
        b.sort();
        assert_eq!(a, b, "result set must be independent of concurrency");
    }

    #[test]
    fn backoff_is_bounded_and_honors_retry_after() {
        // Rate limit uses Retry-After, clamped to MAX_RETRY_AFTER.
        assert_eq!(
            backoff_for(&Error::rate_limited(5_000), 0),
            Duration::from_millis(5_000)
        );
        assert_eq!(
            backoff_for(&Error::rate_limited(10_000_000), 3),
            MAX_RETRY_AFTER
        );
        // Other retriable errors grow exponentially from RETRY_BASE.
        assert_eq!(backoff_for(&Error::http(503, "x"), 0), RETRY_BASE);
        assert_eq!(backoff_for(&Error::http(503, "x"), 1), RETRY_BASE * 2);
    }

    #[test]
    fn pacer_spaces_starts_per_host_and_is_free_when_zero() {
        let mut p = Pacer::new();
        assert_eq!(p.reserve("a.dev", Duration::ZERO), Duration::ZERO);
        assert_eq!(p.reserve("a.dev", Duration::ZERO), Duration::ZERO);
        // With an interval, successive reservations schedule ~interval apart.
        let mut p = Pacer::new();
        let d0 = p.reserve("a.dev", Duration::from_millis(20));
        let d1 = p.reserve("a.dev", Duration::from_millis(20));
        let d2 = p.reserve("a.dev", Duration::from_millis(20));
        assert!(d0 < Duration::from_millis(5), "first start is immediate");
        assert!(d1 >= Duration::from_millis(15), "second start paced");
        assert!(d2 >= Duration::from_millis(35), "third start paced");
        // A different host has its own schedule: politeness is per server.
        assert!(
            p.reserve("b.dev", Duration::from_millis(20)) < Duration::from_millis(5),
            "another host starts immediately"
        );
    }

    /// A mock that records peak concurrent in-flight fetches and holds each fetch
    /// open briefly so overlap is observable.
    struct PeakSite {
        pages: HashMap<String, Vec<u8>>,
        current: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        peak: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Fetcher for PeakSite {
        type FetchFuture<'a>
            = impl std::future::Future<Output = Result<Fetched<'a>>> + Send + 'a
        where
            Self: 'a;
        fn fetch<'a>(
            &'a self,
            resource: &'a Resource,
            _opts: FetchOptions<'a>,
        ) -> Self::FetchFuture<'a> {
            use std::sync::atomic::Ordering::SeqCst;
            async move {
                let cur = self.current.fetch_add(1, SeqCst) + 1;
                self.peak.fetch_max(cur, SeqCst);
                futures_timer::Delay::new(Duration::from_millis(10)).await;
                self.current.fetch_sub(1, SeqCst);
                match self.pages.get(&crate::url_key::canonicalize(&resource.url)) {
                    Some(body) => Ok(Fetched {
                        meta: FetchMeta {
                            kind: ResourceKind::Html,
                            etag: None,
                            status: 200,
                            final_url: None,
                        },
                        payload: DocPayload::Owned(Bytes::from(body.clone())),
                    }),
                    None => Err(Error::not_found(resource.url.as_str())),
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn peak_in_flight_never_exceeds_concurrency() {
        // A wide fan-out so many links are ready to fetch at once.
        let mut pages = HashMap::new();
        let children: Vec<String> = (0..12).map(|i| format!("/docs/{i}")).collect();
        let refs: Vec<&str> = children.iter().map(String::as_str).collect();
        pages.insert("https://x.dev/docs".to_string(), page(&refs));
        for c in &children {
            pages.insert(format!("https://x.dev{c}"), page(&[]));
        }
        let current = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let site = PeakSite {
            pages,
            current: current.clone(),
            peak: peak.clone(),
        };
        let source = CrawlSource::new(site).depth(2).concurrency(4);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        assert_eq!(urls.len(), 13); // root + 12 children
        let observed = peak.load(std::sync::atomic::Ordering::SeqCst);
        assert!(observed <= 4, "peak {observed} exceeded concurrency 4");
        assert!(observed >= 2, "expected real overlap, peak was {observed}");
    }

    /// A mock that fails a URL a fixed number of times (with a chosen error) before
    /// serving it, counting attempts per URL.
    struct FlakySite {
        body: Vec<u8>,
        fail_times: u32,
        status: u16,
        attempts: std::sync::Mutex<HashMap<String, u32>>,
    }

    impl Fetcher for FlakySite {
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
            let n = {
                let mut map = self.attempts.lock().unwrap();
                let e = map.entry(key).or_insert(0);
                *e += 1;
                *e
            };
            let result = if n <= self.fail_times {
                Err(Error::http(self.status, resource.url.as_str()))
            } else {
                Ok(Fetched {
                    meta: FetchMeta {
                        kind: ResourceKind::Html,
                        etag: None,
                        status: 200,
                        final_url: None,
                    },
                    payload: DocPayload::Owned(Bytes::from(self.body.clone())),
                })
            };
            std::future::ready(result)
        }
    }

    #[tokio::test]
    async fn retries_retriable_then_succeeds() {
        let site = FlakySite {
            body: page(&[]),
            fail_times: 2, // fails twice (503), succeeds on the third attempt
            status: 503,
            attempts: std::sync::Mutex::new(HashMap::new()),
        };
        let source = CrawlSource::new(site).depth(0).max_retries(2);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let urls = crawl_urls(&source, &root).await;
        assert_eq!(urls, vec!["https://x.dev/docs".to_string()]);
    }

    #[tokio::test]
    async fn non_retriable_is_not_retried() {
        // 404 is not retriable: exactly one attempt, root fails → crawl errors.
        let site = FlakySite {
            body: page(&[]),
            fail_times: 1,
            status: 404,
            attempts: std::sync::Mutex::new(HashMap::new()),
        };
        let source = CrawlSource::new(site).depth(0).max_retries(5);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let stream = source.discover(&root);
        futures::pin_mut!(stream);
        let first = stream.next().await;
        assert!(matches!(first, Some(Err(_))), "root 404 should error");
        // 404 is non-retriable, so exactly one attempt was made.
        let attempts = source.fetcher.attempts.lock().unwrap();
        assert_eq!(attempts.values().copied().max(), Some(1));
    }

    #[tokio::test]
    async fn exhausted_retries_abort_root() {
        // Always-503 root, exhausts retries → crawl errors (root must succeed).
        let site = FlakySite {
            body: page(&[]),
            fail_times: u32::MAX,
            status: 503,
            attempts: std::sync::Mutex::new(HashMap::new()),
        };
        let source = CrawlSource::new(site).depth(0).max_retries(1);
        let root = SourceRef::http("https://x.dev/docs").unwrap();
        let stream = source.discover(&root);
        futures::pin_mut!(stream);
        assert!(matches!(stream.next().await, Some(Err(_))));
        // 1 initial + 1 retry = 2 attempts.
        let attempts = source.fetcher.attempts.lock().unwrap();
        assert_eq!(attempts.values().copied().max(), Some(2));
    }

    #[cfg(feature = "robots")]
    mod robots_tests {
        use super::*;

        /// A mock site that also serves `/robots.txt` and counts how many times it
        /// was requested (to prove per-host caching).
        struct RobotsMock {
            pages: HashMap<String, Vec<u8>>,
            robots: HashMap<String, Vec<u8>>, // host_key -> robots.txt body
            robots_hits: std::sync::Mutex<HashMap<String, u32>>,
        }

        impl Fetcher for RobotsMock {
            type FetchFuture<'a>
                = std::future::Ready<Result<Fetched<'a>>>
            where
                Self: 'a;
            fn fetch<'a>(
                &'a self,
                resource: &'a Resource,
                _opts: FetchOptions<'a>,
            ) -> Self::FetchFuture<'a> {
                let url = &resource.url;
                let result = if url.path() == "/robots.txt" {
                    let key = RobotsCache::host_key(url).to_string();
                    *self
                        .robots_hits
                        .lock()
                        .unwrap()
                        .entry(key.clone())
                        .or_insert(0) += 1;
                    match self.robots.get(&key) {
                        Some(body) => Ok(ok_page(ResourceKind::Text, body.clone())),
                        None => Err(Error::not_found(url.as_str())),
                    }
                } else {
                    match self.pages.get(&crate::url_key::canonicalize(url)) {
                        Some(body) => Ok(ok_page(ResourceKind::Html, body.clone())),
                        None => Err(Error::not_found(url.as_str())),
                    }
                };
                std::future::ready(result)
            }
        }

        fn ok_page(kind: ResourceKind, body: Vec<u8>) -> Fetched<'static> {
            Fetched {
                meta: FetchMeta {
                    kind,
                    etag: None,
                    status: 200,
                    final_url: None,
                },
                payload: DocPayload::Owned(Bytes::from(body)),
            }
        }

        fn site(robots_txt: Option<&str>) -> RobotsMock {
            let mut pages = HashMap::new();
            pages.insert(
                "https://x.dev/docs".to_string(),
                page(&["/docs/public", "/docs/private"]),
            );
            pages.insert("https://x.dev/docs/public".to_string(), page(&[]));
            pages.insert("https://x.dev/docs/private".to_string(), page(&[]));
            let mut robots = HashMap::new();
            if let Some(txt) = robots_txt {
                robots.insert("https://x.dev".to_string(), txt.as_bytes().to_vec());
            }
            RobotsMock {
                pages,
                robots,
                robots_hits: std::sync::Mutex::new(HashMap::new()),
            }
        }

        #[tokio::test]
        async fn disallowed_child_is_skipped() {
            let src = CrawlSource::new(site(Some("User-agent: *\nDisallow: /docs/private")))
                .depth(2)
                .scope(CrawlScope::SameHost)
                .respect_robots(true);
            let root = SourceRef::http("https://x.dev/docs").unwrap();
            let urls = crawl_urls(&src, &root).await;
            assert!(urls.iter().any(|u| u.ends_with("/docs/public")));
            assert!(
                !urls.iter().any(|u| u.contains("/docs/private")),
                "robots-disallowed child must be skipped: {urls:?}"
            );
        }

        #[tokio::test]
        async fn disallowed_root_errors() {
            let src = CrawlSource::new(site(Some("User-agent: *\nDisallow: /")))
                .depth(1)
                .respect_robots(true);
            let root = SourceRef::http("https://x.dev/docs").unwrap();
            let stream = src.discover(&root);
            futures::pin_mut!(stream);
            assert!(matches!(stream.next().await, Some(Err(_))));
        }

        #[tokio::test]
        async fn missing_robots_allows_all() {
            let src = CrawlSource::new(site(None)) // 404 robots.txt => allow-all
                .depth(2)
                .scope(CrawlScope::SameHost)
                .respect_robots(true);
            let root = SourceRef::http("https://x.dev/docs").unwrap();
            let urls = crawl_urls(&src, &root).await;
            assert!(urls.iter().any(|u| u.ends_with("/docs/public")));
            assert!(urls.iter().any(|u| u.ends_with("/docs/private")));
        }

        #[tokio::test]
        async fn robots_fetched_once_per_host() {
            let src = CrawlSource::new(site(Some("User-agent: *\nDisallow:")))
                .depth(2)
                .scope(CrawlScope::SameHost)
                .respect_robots(true);
            let root = SourceRef::http("https://x.dev/docs").unwrap();
            let _ = crawl_urls(&src, &root).await;
            let hits = src.fetcher.robots_hits.lock().unwrap();
            assert_eq!(hits.get("https://x.dev").copied(), Some(1));
        }
    }
}
