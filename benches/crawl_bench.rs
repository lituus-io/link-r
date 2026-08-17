// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Crawl + facade benchmarks: end-to-end mock-fetcher crawl throughput (sequential
//! vs concurrent), the hot per-page kernels, and the facade pipeline (batched
//! embedding + cached search vs the naive per-query rebuild).
//!
//! Each arm emits an `AUTORESEARCH` telemetry line and the final `rss_growth_bps`
//! least-squares slope — the memory-expansion heuristic the atomizer gate consumes.
//! The crawler is executor-agnostic; the benches drive it on a Tokio runtime.

#[path = "support/mod.rs"]
#[allow(dead_code)] // shared across bench binaries; each uses a subset
mod support;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use link_r::extract::html::extract_links;
use link_r::fetch::{FetchMeta, FetchOptions, Fetched, Fetcher};
use link_r::payload::DocPayload;
use link_r::resource::{Resource, ResourceKind, SourceRef};
use link_r::{CrawlScope, CrawlSource, LinkIndex, Source, UrlKey};
use std::collections::HashMap;
use std::future::Ready;
use std::sync::Arc;
use std::time::Instant;
use support::{autoresearch, least_squares_slope, rss_kib, Rng};
use url::Url;

/// A deterministic in-memory site: canonical URL → HTML body (as `Bytes`, so a
/// mock fetch is a refcount bump, not a copy). `Arc`-shared so it can be handed to
/// many `CrawlSource`s cheaply.
#[derive(Clone)]
struct MockSite {
    pages: Arc<HashMap<String, bytes::Bytes>>,
}

impl Fetcher for MockSite {
    type FetchFuture<'a>
        = Ready<link_r::Result<Fetched<'a>>>
    where
        Self: 'a;
    fn fetch<'a>(
        &'a self,
        resource: &'a Resource,
        _opts: FetchOptions<'a>,
    ) -> Self::FetchFuture<'a> {
        let key = link_r::canonicalize(&resource.url);
        let result = match self.pages.get(&key) {
            Some(body) => Ok(Fetched {
                meta: FetchMeta {
                    kind: ResourceKind::Html,
                    etag: None,
                    status: 200,
                    final_url: None,
                },
                payload: DocPayload::Owned(body.clone()),
            }),
            None => Err(link_r::Error::not_found(resource.url.as_str())),
        };
        std::future::ready(result)
    }
}

const VOCAB: &[&str] = &[
    "cluster", "network", "policy", "service", "private", "endpoint", "access", "vector", "query",
    "table", "secret", "token", "region", "bucket", "stream", "index", "embed", "graph",
];

/// A synthetic site of `n` interlinked pages under `https://bench.dev/docs/`. Each
/// page has `fanout` in-scope links plus ~40 words of vocab prose, so a BFS crawl
/// reaches every page.
fn gen_site(n: usize, fanout: usize, seed: u64) -> MockSite {
    let mut rng = Rng::new(seed);
    let mut pages = HashMap::with_capacity(n);
    for i in 0..n {
        let mut html = String::from("<html><body><h1>Doc</h1><p>");
        for _ in 0..40 {
            html.push_str(VOCAB[(rng.next_u64() as usize) % VOCAB.len()]);
            html.push(' ');
        }
        html.push_str("</p>");
        for _ in 0..fanout {
            let target = (rng.next_u64() as usize) % n;
            html.push_str(&format!("<a href=\"/docs/{target}\">l</a>"));
        }
        html.push_str("</body></html>");
        pages.insert(
            format!("https://bench.dev/docs/{i}"),
            bytes::Bytes::from(html.into_bytes()),
        );
    }
    MockSite {
        pages: Arc::new(pages),
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .build()
        .unwrap()
}

/// Drain a crawl to completion, returning the page count.
async fn drain<F: Fetcher>(source: &CrawlSource<F>, root: &SourceRef) -> usize {
    use futures::StreamExt;
    let stream = source.discover(root);
    futures::pin_mut!(stream);
    let mut count = 0usize;
    while let Some(item) = stream.next().await {
        if let Ok(page) = item {
            black_box(&page);
            count += 1;
        }
    }
    count
}

fn bench_crawl(c: &mut Criterion) {
    let rt = runtime();
    let root = SourceRef::http("https://bench.dev/docs/0").unwrap();
    let mut group = c.benchmark_group("crawl");
    for &n in &[100usize, 1_000, 10_000] {
        let site = gen_site(n, 8, 0xC0FFEE ^ n as u64);
        // sequential (concurrency = 1) vs concurrent (concurrency = 8)
        for &(label, conc) in &[("crawl_seq", 1usize), ("crawl_conc8", 8usize)] {
            let source = CrawlSource::new(site.clone())
                .depth(30)
                .scope(CrawlScope::SameHost)
                .max_pages(n)
                .concurrency(conc);
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(label, n), &source, |b, source| {
                b.iter(|| black_box(rt.block_on(drain(source, &root))));
            });
            let rss_before = rss_kib();
            let start = Instant::now();
            let pages = rt.block_on(drain(&source, &root));
            let pps = pages as f64 / start.elapsed().as_secs_f64().max(1e-9);
            autoresearch("crawl", label, n, "pages_per_sec", pps, rss_before);
        }
    }
    group.finish();
}

fn bench_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("crawl_kernels");

    // A realistic ~8 KiB page with ~50 links for the link scanner.
    let site = gen_site(1, 50, 7);
    let page = site.pages.values().next().unwrap().clone();
    group.bench_function("extract_links", |b| {
        b.iter(|| black_box(extract_links(&page)));
    });
    let rss = rss_kib();
    let start = Instant::now();
    for _ in 0..1000 {
        black_box(extract_links(&page));
    }
    autoresearch(
        "crawl_kernels",
        "extract_links",
        1,
        "ns_per_page",
        start.elapsed().as_nanos() as f64 / 1000.0,
        rss,
    );

    // URL canonicalization + key over 10k synthetic URLs.
    let urls: Vec<Url> = (0..10_000)
        .map(|i| Url::parse(&format!("https://bench.dev/docs/{i}?b=2&a=1#frag")).unwrap())
        .collect();
    group.bench_function("url_key", |b| {
        b.iter(|| {
            for u in &urls {
                black_box(UrlKey::from_url(u));
            }
        });
    });
    let rss = rss_kib();
    let start = Instant::now();
    for u in &urls {
        black_box(UrlKey::from_url(u));
    }
    autoresearch(
        "crawl_kernels",
        "url_key",
        urls.len(),
        "ns_per_url",
        start.elapsed().as_nanos() as f64 / urls.len() as f64,
        rss,
    );

    // Scope check over 10k candidates.
    let root = Url::parse("https://bench.dev/docs/").unwrap();
    let scope = CrawlScope::PathPrefix;
    group.bench_function("scope_allows", |b| {
        b.iter(|| {
            for u in &urls {
                black_box(scope.allows(&root, u));
            }
        });
    });
    group.finish();
}

fn bench_facade(c: &mut Criterion) {
    let rt = runtime();
    let root = SourceRef::http("https://bench.dev/docs/0").unwrap();
    let mut group = c.benchmark_group("facade");

    for &n in &[100usize, 1_000] {
        let site = gen_site(n, 6, 0xF00D ^ n as u64);

        // Batched (B=64) vs one-at-a-time (B=1) end-to-end ingest.
        for &(label, batch) in &[("update_batched", 64usize), ("update_unbatched", 1usize)] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(label, n), &batch, |b, &batch| {
                b.iter(|| {
                    let mut idx = LinkIndex::in_memory().unwrap();
                    let source = CrawlSource::new(site.clone())
                        .depth(30)
                        .scope(CrawlScope::SameHost)
                        .max_pages(n);
                    rt.block_on(idx.ingest_from(&source, &root, batch, false))
                        .unwrap();
                    black_box(idx.len());
                });
            });
            let rss_before = rss_kib();
            let start = Instant::now();
            let mut idx = LinkIndex::in_memory().unwrap();
            let source = CrawlSource::new(site.clone())
                .depth(30)
                .scope(CrawlScope::SameHost)
                .max_pages(n);
            rt.block_on(idx.ingest_from(&source, &root, batch, false))
                .unwrap();
            let pps = idx.len() as f64 / start.elapsed().as_secs_f64().max(1e-9);
            autoresearch("facade", label, n, "pages_per_sec", pps, rss_before);
        }

        // Steady-state cached search vs the first (cache-building) query.
        let mut idx = LinkIndex::in_memory().unwrap();
        let source = CrawlSource::new(site.clone())
            .depth(30)
            .scope(CrawlScope::SameHost)
            .max_pages(n);
        rt.block_on(idx.ingest_from(&source, &root, 64, false))
            .unwrap();

        let rss = rss_kib();
        let start = Instant::now();
        black_box(rt.block_on(idx.search("network policy", 10)).unwrap());
        autoresearch(
            "facade",
            "search_cache_build",
            n,
            "ns_first_query",
            start.elapsed().as_nanos() as f64,
            rss,
        );

        group.bench_with_input(BenchmarkId::new("search_cached", n), &idx, |b, idx| {
            b.iter(|| black_box(rt.block_on(idx.search("network policy", 10)).unwrap()));
        });
        let rss = rss_kib();
        let start = Instant::now();
        for _ in 0..200 {
            black_box(rt.block_on(idx.search("network policy", 10)).unwrap());
        }
        autoresearch(
            "facade",
            "search_cached",
            n,
            "ns_per_query",
            start.elapsed().as_nanos() as f64 / 200.0,
            rss,
        );
    }
    group.finish();
}

fn bench_rss_growth(_c: &mut Criterion) {
    let rt = runtime();
    let root = SourceRef::http("https://bench.dev/docs/0").unwrap();
    let scales = [100usize, 500, 1_000, 5_000, 10_000];
    let baseline = rss_kib().max(1);
    let mut points: Vec<(f64, f64)> = Vec::new();
    for &n in &scales {
        let site = gen_site(n, 6, 0xBEEF ^ n as u64);
        let mut idx = LinkIndex::in_memory().unwrap();
        let source = CrawlSource::new(site.clone())
            .depth(30)
            .scope(CrawlScope::SameHost)
            .max_pages(n);
        rt.block_on(idx.ingest_from(&source, &root, 64, false))
            .unwrap();
        black_box(idx.len());
        points.push((n as f64, rss_kib() as f64));
    }
    let slope_bps = least_squares_slope(&points) / baseline as f64 * 10_000.0;
    println!(
        "AUTORESEARCH rss bench=crawl_growth scale={} process_rss_kib={} rss_growth_bps={slope_bps:.3}",
        scales[scales.len() - 1],
        rss_kib(),
    );
}

criterion_group!(
    benches,
    bench_crawl,
    bench_kernels,
    bench_facade,
    bench_rss_growth
);
criterion_main!(benches);
