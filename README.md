# link-r

A tiny, embeddable **crawl-and-resolve knowledge index** for Rust (with Python bindings).

Give it a parent HTTPS link. It concurrently crawls sublinks to a configurable
depth, distills keywords/topics/headings from each page, and builds a small,
memory-mapped hybrid index (dense embeddings + BM25 + structured metadata + a
persisted **link graph**). A query resolves to ranked **remote URLs** — document
bodies stay at the URL and are fetched lazily by the caller. Each link is unique in
the index (canonical-URL keyed), so re-crawling never duplicates a link.

It functions as an **accumulating knowledge base with a TTL**: `update` discovers
new links, `refresh` re-validates known ones after a staleness window (unchanged →
touched, changed → re-embedded, dead → evicted), and `pin` retains chosen links
forever — so knowledge accumulates without growing endlessly.

No server, no sidecar database — the index is a single mmap-able file. A freshly
opened index searches **directly against the mapped file** (zero-copy, no rebuild).

## Rust

```rust
use link_r::prelude::*;

# async fn run() -> link_r::Result<()> {
let mut idx = LinkIndex::open_or_create("kb.lnkr")?;
idx.update("https://example.com/docs/")
    .depth(2)
    .concurrency(8)     // bounded-concurrency crawl
    .run().await?;
idx.save()?;

for hit in idx.search("how does auth work", 10).await? {
    println!("{:.3}  {}", hit.score, hit.url);
}

// Maintain the knowledge base: re-validate links older than a day; evict dead ones.
idx.refresh().ttl(std::time::Duration::from_secs(86_400)).run().await?;

// Follow the knowledge graph from a hit.
for related in idx.related("https://example.com/docs/auth", 5)? {
    println!("→ {}", related.url);
}
# Ok(())
# }
```

For private sources, inject auth: `.token("ghp_...")` (scoped to the crawl host, so
it is never sent off-host) or `.auth(Box::new(my_oauth))`.

## Python (drop-in RAG replacement)

Build a categorical knowledge base from GitHub links, maintain it on a TTL, and
retrieve context via the returned URIs:

```python
import link_r

kb = link_r.LinkIndex.open_or_create("kb.lnkr")
kb.update(
    "https://github.com/org/repo/tree/main/docs",
    depth=3, scope="host", concurrency=8,
    path_contains=["/org/repo", "/main/docs"],   # follow broadly…
    index_path_contains=["/blob/"], extensions=["md"],  # …index narrowly
    token="ghp_…",         # private repos; scoped to github.com
)
kb.save()

# Categorical retrieval, graph-aware ranking:
for hit in kb.search("how does auth work", k=10, path_prefix="/blob/", graph_boost=0.3):
    print(round(hit.score, 3), hit.url)   # resolve context by fetching hit.url

# Follow the knowledge graph:
for r in kb.related("https://github.com/org/repo/blob/main/docs/auth.md", k=5):
    print("→", r.url)

# Maintain after a TTL; hard-evict anything unpinned older than 30 days:
kb.pin("https://github.com/org/repo/blob/main/docs/reference")  # keep forever
report = kb.refresh(ttl_secs=86_400, max_age_secs=30*86_400, token="ghp_…", token_host="github.com")
print(report)
```

Build the wheel: `cd bindings/python && maturin develop` (or `maturin build --release
--features onnx` for the semantic embedder).

## Embedders

- **`HashEmbedder`** (default, zero-dep): deterministic feature-hashing — lexical
  proxy, makes tests/fuzz hermetic and offline.
- **`OnnxEmbedder`** (feature `onnx`): `bge-small-en-v1.5` (384-dim) for real
  semantic search. ~130 MB model downloaded once over rustls. Recommended for
  production; pairs with BM25 to cover exact terms the model blurs.

The `Embedder` trait is pluggable; bring your own.

## Design

- **Zero-copy**: the on-disk index *is* the file; the dense f32 blob is mmap'd and
  reinterpreted with no per-element decode. Integrity is layered xxh3 (header,
  per-section, whole-file); writes are atomic (write-tmp-then-rename). Opening and
  searching a persisted index never rebuilds it.
- **Concurrent crawl**: bounded-concurrency BFS (`FuturesUnordered`, no task
  spawning — executor-agnostic) with global request-start pacing, bounded retry +
  backoff, streaming body caps + timeouts, an explicit redirect policy (credentials
  stripped cross-host), and an allocation-lean byte-scanner for link discovery.
  Optional `robots.txt` honoring.
- **Static dispatch**: async traits use GATs (no `Box<dyn Future>`); one documented
  `Box<dyn>` escape hatch exists only for runtime-selected auth. Lifetimes over Arc.
- **`#![deny(unsafe_code)]`** everywhere except a single audited `index::mmap` island.
- **Hybrid retrieval**: brute-force SIMD-friendly cosine + BM25, fused via
  Reciprocal Rank Fusion, with metadata prefilters (tags, path, kind, freshness
  ranges) and an optional one-hop knowledge-graph boost. Optional `quant` tier adds
  int8/binary quantization with two-tier rerank for 100k+ corpora.
- **Knowledge graph**: each page's outbound links are persisted as canonical-URL
  keyed edges. `related(url)` follows outbound targets + co-cited siblings; edges
  live and die with their node (path TTL = node TTL), and dangling edges heal as
  the corpus accumulates.
- **Security**: bearer tokens are host-scoped by default; URL userinfo is dropped
  before persistence; single-flight OAuth refresh under concurrency.
- **Parallelism** (`parallel`): rayon-parallel BM25 counting, row normalization,
  and quantization — all byte-reproducible with the feature on or off.

## Features

`hash` `markdown` `html` `http` `crawl` `fs` (default) · `onnx` `oauth`
`robots` `quant` `parallel` · `full`.

## Testing

`cargo test` (unit + integration + property), `cargo bench` (criterion, with
AUTORESEARCH + RSS telemetry), `cargo +nightly fuzz run fuzz_index_loader` (the
loader is fuzzed against arbitrary bytes — it must never panic or OOM).

## graph-r: the persistent backend

link-r can run purely in memory as the hot working set; its sibling crate
[graph-r](../graph-r) is the durable historical layer. `LinkIndex::export()`
hands every document + its edges to graph-r's `bridge::absorb`, per-page
`UpdateReport::pages` outcomes (headings, keywords) become graph segments, and
graph-r's adaptive-TTL due-lists drive `refresh().urls(...)` so only what is
actually stale gets revalidated — with stored ETags, an unchanged site costs
zero body transfers (304s all the way down, and previously indexed links keep
the crawl frontier alive).

## License


Copyright: lituus-io, all rights reserved.
