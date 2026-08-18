// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Python bindings for `link-r` (PyO3 + maturin).
//!
//! A synchronous, dead-simple surface: the binding owns a private Tokio runtime
//! and `block_on`s the async crawl inside [`PyLinkIndex::update`], detaching from
//! the interpreter while it runs so other Python threads keep going. The same
//! three verbs as the Rust facade: `open_or_create`/`open`/`in_memory`, `update`,
//! `search`, `save`.
//!
//! The wheel is built against the stable ABI (`abi3-py312`), so one artifact per
//! platform serves every Python from 3.12 up.

use linkr::facade::{LinkIndex, RefreshReport, SearchResult, UpdateReport};
use linkr::source::CrawlScope;
use linkr::{Filter, StaticTokenAuth};
use pyo3::create_exception;
use pyo3::prelude::*;
use std::time::Duration;

create_exception!(link_r, LinkRError, pyo3::exceptions::PyException);

fn map_err(e: linkr::Error) -> PyErr {
    LinkRError::new_err(e.to_string())
}

/// Build a metadata filter from the Python keyword surface (categorical retrieval).
fn build_filter(path_prefix: Option<String>, tag: Option<String>) -> Filter {
    let mut filter: Option<Filter> = path_prefix.map(|p| Filter::PathPrefix(p.into()));
    if let Some(t) = tag {
        let tag_filter = Filter::Tag(t.into());
        filter = Some(match filter {
            Some(existing) => Filter::And(Box::new(existing), Box::new(tag_filter)),
            None => tag_filter,
        });
    }
    filter.unwrap_or(Filter::All)
}

/// A single search result.
#[pyclass(name = "Hit", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyHit {
    #[pyo3(get)]
    url: String,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    title: Option<String>,
    #[pyo3(get)]
    snippet: String,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    tags: Vec<String>,
}

#[pymethods]
impl PyHit {
    fn __repr__(&self) -> String {
        format!("Hit(score={:.3}, url={:?})", self.score, self.url)
    }
}

impl From<SearchResult> for PyHit {
    fn from(r: SearchResult) -> Self {
        Self {
            url: r.url,
            score: r.score,
            title: r.title,
            snippet: r.snippet,
            kind: format!("{:?}", r.kind),
            tags: r.tags,
        }
    }
}

/// What an update did.
#[pyclass(name = "UpdateReport", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyUpdateReport {
    #[pyo3(get)]
    added: usize,
    #[pyo3(get)]
    updated: usize,
    #[pyo3(get)]
    unchanged: usize,
    #[pyo3(get)]
    skipped: usize,
    #[pyo3(get)]
    failed: usize,
}

#[pymethods]
impl PyUpdateReport {
    #[getter]
    fn pages_seen(&self) -> usize {
        self.added + self.updated + self.unchanged + self.skipped
    }

    fn __repr__(&self) -> String {
        format!(
            "UpdateReport(added={}, updated={}, unchanged={}, skipped={}, failed={})",
            self.added, self.updated, self.unchanged, self.skipped, self.failed
        )
    }
}

impl From<UpdateReport> for PyUpdateReport {
    fn from(r: UpdateReport) -> Self {
        Self {
            added: r.added,
            updated: r.updated,
            unchanged: r.unchanged,
            skipped: r.skipped,
            failed: r.failed,
        }
    }
}

/// What a TTL refresh did.
#[pyclass(name = "RefreshReport", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyRefreshReport {
    #[pyo3(get)]
    refreshed: usize,
    #[pyo3(get)]
    unchanged: usize,
    #[pyo3(get)]
    removed: usize,
    #[pyo3(get)]
    failed: usize,
}

#[pymethods]
impl PyRefreshReport {
    #[getter]
    fn total(&self) -> usize {
        self.refreshed + self.unchanged + self.removed + self.failed
    }

    fn __repr__(&self) -> String {
        format!(
            "RefreshReport(refreshed={}, unchanged={}, removed={}, failed={})",
            self.refreshed, self.unchanged, self.removed, self.failed
        )
    }
}

impl From<RefreshReport> for PyRefreshReport {
    fn from(r: RefreshReport) -> Self {
        Self {
            refreshed: r.refreshed,
            unchanged: r.unchanged,
            removed: r.removed,
            failed: r.failed,
        }
    }
}

fn parse_scope(scope: Option<&str>) -> CrawlScope {
    match scope {
        Some("host") | Some("same_host") => CrawlScope::SameHost,
        Some("subdomains") => CrawlScope::SameHostAndSubdomains,
        _ => CrawlScope::PathPrefix,
    }
}

/// A crawl-and-resolve link index.
///
/// Deliberately NOT `unsendable`: that marker makes PyO3 raise if the object is
/// touched from a thread other than the one that created it, which breaks every
/// caller that dispatches through a thread pool -- `asyncio.to_thread` being the
/// usual one. Both `LinkIndex` and the tokio runtime are `Send`, so the
/// restriction bought nothing and cost correctness.
#[pyclass(name = "LinkIndex")]
struct PyLinkIndex {
    inner: LinkIndex,
    rt: tokio::runtime::Runtime,
}

fn runtime() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| LinkRError::new_err(format!("tokio runtime: {e}")))
}

#[pymethods]
impl PyLinkIndex {
    /// Open an existing index at `path`, or create one bound to it.
    #[staticmethod]
    fn open_or_create(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: LinkIndex::open_or_create(path).map_err(map_err)?,
            rt: runtime()?,
        })
    }

    /// Open an existing index (errors if missing).
    #[staticmethod]
    fn open(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: LinkIndex::open(path).map_err(map_err)?,
            rt: runtime()?,
        })
    }

    /// Create an in-memory index.
    #[staticmethod]
    fn in_memory() -> PyResult<Self> {
        Ok(Self {
            inner: LinkIndex::in_memory().map_err(map_err)?,
            rt: runtime()?,
        })
    }

    /// Crawl `url` to `depth` and index the pages (deduplicated by URL). Blocks
    /// until the crawl completes, detached from the interpreter throughout.
    #[pyo3(signature = (url, depth=2, max_pages=1000, concurrency=8, embed_batch=64, token=None, scope=None, min_delay_ms=0, path_contains=None, extensions=None, index_path_contains=None, pin=false))]
    #[allow(clippy::too_many_arguments)] // a flat keyword surface is the point for Python
    fn update(
        &mut self,
        py: Python<'_>,
        url: String,
        depth: u16,
        max_pages: usize,
        concurrency: usize,
        embed_batch: usize,
        token: Option<String>,
        scope: Option<String>,
        min_delay_ms: u64,
        path_contains: Option<Vec<String>>,
        extensions: Option<Vec<String>>,
        index_path_contains: Option<Vec<String>>,
        pin: bool,
    ) -> PyResult<PyUpdateReport> {
        let crawl_scope = parse_scope(scope.as_deref());
        let report = py.detach(|| {
            self.rt.block_on(async {
                let mut update = self
                    .inner
                    .update(url)
                    .depth(depth)
                    .max_pages(max_pages)
                    .concurrency(concurrency)
                    .embed_batch(embed_batch)
                    .scope(crawl_scope)
                    .min_delay(Duration::from_millis(min_delay_ms));
                for substring in path_contains.into_iter().flatten() {
                    update = update.require_path(substring);
                }
                for ext in extensions.into_iter().flatten() {
                    update = update.accept_extension(ext);
                }
                for substring in index_path_contains.into_iter().flatten() {
                    update = update.index_path(substring);
                }
                if let Some(t) = token {
                    update = update.token(t);
                }
                if pin {
                    update = update.pin();
                }
                update.run().await
            })
        });
        report.map(PyUpdateReport::from).map_err(map_err)
    }

    /// Re-validate already-indexed links older than `ttl_secs`: unchanged pages are
    /// re-timestamped, changed pages re-indexed, dead pages evicted. `max_age_secs`
    /// hard-evicts unpinned links older than it without fetching. Blocks, detached
    /// from the interpreter.
    #[pyo3(signature = (ttl_secs, max_age_secs=None, evict_unreachable=true, concurrency=8, token=None, token_host=None))]
    #[allow(clippy::too_many_arguments)] // a flat keyword surface is the point for Python
    fn refresh(
        &mut self,
        py: Python<'_>,
        ttl_secs: u64,
        max_age_secs: Option<u64>,
        evict_unreachable: bool,
        concurrency: usize,
        token: Option<String>,
        token_host: Option<String>,
    ) -> PyResult<PyRefreshReport> {
        let report = py.detach(|| {
            self.rt.block_on(async {
                let mut refresh = self
                    .inner
                    .refresh()
                    .ttl(Duration::from_secs(ttl_secs))
                    .evict_unreachable(evict_unreachable)
                    .concurrency(concurrency);
                if let Some(secs) = max_age_secs {
                    refresh = refresh.max_age(Duration::from_secs(secs));
                }
                match (token, token_host) {
                    (Some(t), Some(h)) => refresh = refresh.token(t, h),
                    (Some(t), None) => {
                        refresh = refresh.auth(Box::new(StaticTokenAuth::bearer(t)));
                    }
                    _ => {}
                }
                refresh.run().await
            })
        });
        report.map(PyRefreshReport::from).map_err(map_err)
    }

    /// Pin every link whose URL starts with `url_prefix` (retained forever, exempt
    /// from TTL/age eviction). Returns how many links changed.
    fn pin(&mut self, url_prefix: String) -> PyResult<usize> {
        self.inner.pin(&url_prefix).map_err(map_err)
    }

    /// Unpin every link whose URL starts with `url_prefix`. Returns how many changed.
    fn unpin(&mut self, url_prefix: String) -> PyResult<usize> {
        self.inner.unpin(&url_prefix).map_err(map_err)
    }

    /// Search the index, returning ranked hits. `path_prefix` and `tag` apply a
    /// categorical metadata prefilter; `graph_boost > 0` re-ranks by knowledge-graph
    /// connectivity (hub/authority pages rise).
    #[pyo3(signature = (query, k=10, path_prefix=None, tag=None, graph_boost=0.0))]
    fn search(
        &self,
        py: Python<'_>,
        query: String,
        k: usize,
        path_prefix: Option<String>,
        tag: Option<String>,
        graph_boost: f32,
    ) -> PyResult<Vec<PyHit>> {
        let filter = build_filter(path_prefix, tag);
        let results = py.detach(|| {
            self.rt
                .block_on(self.inner.search_ranked(&query, k, &filter, graph_boost))
        });
        results
            .map(|hits| hits.into_iter().map(PyHit::from).collect())
            .map_err(map_err)
    }

    /// Follow the knowledge graph from a link: the `k` documents most related to
    /// `url` (outbound targets + co-cited siblings), ranked by connectivity.
    #[pyo3(signature = (url, k=10))]
    fn related(&self, py: Python<'_>, url: String, k: usize) -> PyResult<Vec<PyHit>> {
        // Graph traversal is pure Rust work; staying attached to the
        // interpreter across it stalls other Python threads for no reason.
        // Every other method already detached.
        let results = py.detach(|| self.inner.related(&url, k));
        results
            .map(|hits| hits.into_iter().map(PyHit::from).collect())
            .map_err(map_err)
    }

    /// Atomically save to the bound path.
    fn save(&self) -> PyResult<()> {
        self.inner.save().map_err(map_err)
    }

    /// Save to a new path and bind to it.
    fn save_as(&mut self, path: String) -> PyResult<()> {
        self.inner.save_as(path).map_err(map_err)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// The `link_r` Python module.
#[pymodule]
fn link_r(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinkIndex>()?;
    m.add_class::<PyHit>()?;
    m.add_class::<PyUpdateReport>()?;
    m.add_class::<PyRefreshReport>()?;
    m.add("LinkRError", m.py().get_type::<LinkRError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
