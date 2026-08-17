// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Benchmarks for build, query, mmap-open, and the hot kernels.
//!
//! Each arm emits an `AUTORESEARCH` telemetry line (matching the house bench
//! style) carrying the primary metric plus `process_rss_kib`/`rss_delta_kib`, and
//! a final `rss_growth_bps` least-squares slope across scales — the memory-
//! expansion heuristic the atomizer regression-gate consumes.

#[path = "support/mod.rs"]
#[allow(dead_code)] // shared across bench binaries; each uses a subset
mod support;

use compact_str::CompactString;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use link_r::index::dense::{dot, top_k};
use link_r::index::sparse::{Bm25, DEFAULT_B, DEFAULT_K1};
use link_r::index::IndexBuilder;
use link_r::metric::Metric;
use link_r::query::{PreparedQuery, RankParams};
use link_r::resource::ResourceKind;
use link_r::{Document, Filter, Index};
use std::time::Instant;
use support::{autoresearch, least_squares_slope, rss_kib, Rng};
use url::Url;

const DIM: usize = 256;
const VOCAB: &[&str] = &[
    "cluster", "network", "policy", "service", "private", "endpoint", "access", "row", "query",
    "table", "secret", "token", "region", "bucket", "stream", "vector", "index", "embed",
];

fn gen_docs(n: usize, seed: u64) -> Vec<Document> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let mut vector: Vec<f32> = (0..DIM).map(|_| rng.unit()).collect();
            let norm = dot(&vector, &vector).sqrt();
            if norm > 0.0 {
                for x in &mut vector {
                    *x /= norm;
                }
            }
            let terms: Vec<CompactString> = (0..8)
                .map(|_| CompactString::from(VOCAB[(rng.next_u64() as usize) % VOCAB.len()]))
                .collect();
            Document {
                url: Url::parse(&format!("https://x.dev/doc/{i}")).unwrap(),
                kind: ResourceKind::Text,
                content_hash: i as u64,
                title: Some(CompactString::from("t")),
                snippet: CompactString::from("s"),
                lang: None,
                tags: smallvec::SmallVec::new(),
                terms,
                vector,
                edges: Vec::new(),
                fetched_at_ms: 0,
                etag: None,
                pinned: false,
            }
        })
        .collect()
}

fn build_index(docs: &[Document]) -> Index {
    let mut builder = IndexBuilder::new(DIM, Metric::Cosine, 1);
    for d in docs {
        builder.upsert(d.clone()).unwrap();
    }
    builder.build()
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    for &n in &[100usize, 1_000, 10_000] {
        let docs = gen_docs(n, 1);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &docs, |b, docs| {
            b.iter(|| black_box(build_index(docs)));
        });
        // Manual ns/doc + RSS for the atomizer line.
        let rss_before = rss_kib();
        let start = Instant::now();
        let idx = build_index(&docs);
        let ns_per_doc = start.elapsed().as_nanos() as f64 / n as f64;
        black_box(&idx);
        autoresearch(
            "build",
            "build_index",
            n,
            "ns_per_doc",
            ns_per_doc,
            rss_before,
        );
    }
    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    for &n in &[100usize, 1_000, 10_000] {
        let docs = gen_docs(n, 2);
        let index = build_index(&docs);
        let query = docs[0].vector.clone();
        let terms = docs[0].terms.clone();
        let filter = Filter::All;
        let pq = PreparedQuery {
            vector: &query,
            terms: &terms,
            filter: &filter,
            limit: 10,
            rank: RankParams::default(),
        };
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &pq, |b, pq| {
            b.iter(|| black_box(index.search_prepared(pq).unwrap()));
        });
        let rss_before = rss_kib();
        let start = Instant::now();
        let iters = 200;
        for _ in 0..iters {
            black_box(index.search_prepared(&pq).unwrap());
        }
        let ns_per_query = start.elapsed().as_nanos() as f64 / f64::from(iters);
        autoresearch(
            "query",
            "hybrid_rrf",
            n,
            "ns_per_query",
            ns_per_query,
            rss_before,
        );
    }
    group.finish();
}

fn bench_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("open");
    let tmp = tempfile::tempdir().unwrap();
    for &n in &[1_000usize, 10_000] {
        let path = tmp.path().join(format!("idx-{n}.lnkr"));
        build_index(&gen_docs(n, 3)).save(&path).unwrap();
        let bytes = std::fs::metadata(&path).unwrap().len();
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::from_parameter(n), &path, |b, path| {
            b.iter(|| black_box(Index::open(path).unwrap()));
        });
        let rss_before = rss_kib();
        let start = Instant::now();
        let idx = Index::open(&path).unwrap();
        let ns_open = start.elapsed().as_nanos() as f64;
        black_box(&idx);
        autoresearch("open", "mmap_open", n, "ns_open", ns_open, rss_before);
        println!(
            "AUTORESEARCH compress bench=index_bytes scale={n} bytes_per_doc={:.1}",
            bytes as f64 / n as f64
        );
    }
    group.finish();
}

fn bench_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernels");
    let docs = gen_docs(10_000, 4);
    let dense: Vec<f32> = docs.iter().flat_map(|d| d.vector.iter().copied()).collect();
    let query = docs[0].vector.clone();

    group.throughput(Throughput::Elements(10_000));
    group.bench_function("dense_top_k", |b| {
        b.iter(|| black_box(top_k(&query, &dense, DIM, 10, Metric::Cosine, None)));
    });

    let doc_terms: Vec<Vec<CompactString>> = docs.iter().map(|d| d.terms.clone()).collect();
    let bm25 = Bm25::build(&doc_terms, DEFAULT_K1, DEFAULT_B);
    let q_terms = docs[0].terms.clone();
    group.bench_function("bm25_score", |b| {
        b.iter(|| black_box(bm25.score(&q_terms, None, 10)));
    });

    // RSS expansion heuristic: rss vs build scale, least-squares slope (bps).
    let scales = [100usize, 500, 1_000, 5_000, 10_000];
    let mut points: Vec<(f64, f64)> = Vec::new();
    let baseline = rss_kib().max(1);
    for &n in &scales {
        let idx = build_index(&gen_docs(n, 5));
        black_box(&idx);
        points.push((n as f64, rss_kib() as f64));
    }
    let slope_bps = least_squares_slope(&points) / baseline as f64 * 10_000.0;
    println!(
        "AUTORESEARCH rss bench=build_growth scale={} process_rss_kib={} rss_growth_bps={:.3}",
        scales[scales.len() - 1],
        rss_kib(),
        slope_bps
    );
    group.finish();
}

criterion_group!(benches, bench_build, bench_query, bench_open, bench_kernels);
criterion_main!(benches);
