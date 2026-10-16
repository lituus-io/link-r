# Changelog

All notable changes to link-r are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/spec/v2.0.0.html).

## 0.4.3

- Builds on uu-graph-r 0.4, 0.5 or 0.6 (`>=0.4.0, <0.7`), so a graph-r 0.6
  build -- one checksummed log record per transaction, whole or absent
  after a crash -- can link link-r without a duplicate graph-r core. No API
  or behaviour change here: `bridge` keeps calling `set_edges`, which stays
  infallible in 0.6, and the suite passes against 0.6.

## 0.4.2

- Builds on uu-graph-r 0.4 or 0.5 (`>=0.4.0, <0.6`), so a graph-r 0.5
  build -- writer lock taken per commit, so several processes can write one
  store -- can link link-r without a duplicate graph-r core. No API or
  behaviour change; the suite passes against both minor lines.

## 0.4.1

- Security: every lockfile (crate, Python binding, fuzz) refreshed to
  patched releases, among them rustls 0.23.46 (RUSTSEC-2026-0346, -0347,
  -0348, -0349) and the earlier rustls and quinn-proto fixes
  (RUSTSEC-2026-0285, -0338 to -0342). The PyPI wheels are built from these.
- The Security workflow audits all three lockfiles, not only the crate's.
- CI: cargo-deny runs on the host; the golden test reads byte-exact fixtures
  on Windows.
- Builds against uu-graph-r 0.4.1 and uu-objstore-r 0.2.1.

## 0.4.0

Breaking: index format 2 (indexes from 0.3 are rebuilt by re-indexing; the
golden fixture moved to `tests/fixtures/v0.4.0`, whose dump is the 0.2.0 dump
unchanged — every document, score and edge survives the change).

- The file is a `graph_r::container`: one checksum per byte, zero padding,
  per-section flags; the separate whole-file trailer is gone.
- Byte codecs, mmap (`cast_f32`), atomic writes, hashing and the text rules
  are graph-r's core; `link_r::UrlKey` is `graph_r::UrlKey`, built through
  the `UrlKeyExt` trait (`UrlKey::from_url`, `UrlKey::parse`).
- Retry status set, jittered backoff and `Retry-After` (both RFC 9110 forms)
  are objstore-r's `net` policy, shared with every HTTP client of the stack.
- `index::sparse::{idf, term_weight}` and `index::fuse::rrf_term`: the BM25
  and RRF formulas, public for indexes with their own layout (crab-r's
  schema linker, graph-r-kb's fused lookups).
- Published as **uu-link-r** on crates.io (the library is still `link_r`).
  The former satellite crates are features: `remote` (was link-r-objstore;
  `link_r::remote::Remote`, on `objstore_r::fenced`), `graph` (the graph-r
  bridge, was graph-r-link; `link_r::bridge`) and `kb` (was graph-r-kb;
  `link_r::kb::KnowledgeBase`), plus `gcs`.

## 0.3.0

Additive release: every 0.2 API, feature, file format and default behaviour
is unchanged (`cargo semver-checks` against 0.2.0: 196/196 checks pass as a
minor release), and indexes written by 0.2.0 read identically from disk and
from bytes, save byte-identically, and re-serialize byte-identically
(golden fixture in `tests/fixtures/v0.2.0`).

### In-memory images (object storage)

- `Index::from_bytes(Bytes)` and `Index::serialize()`; `LinkIndex::from_bytes`
  and `LinkIndex::to_bytes`. The dense block is borrowed from the buffer when
  it is `f32`-aligned (copied once otherwise), so an index fetched from a
  bucket is searched without a temp file or a second copy.
- New crate `objstore/` (**link-r-objstore**): `Remote::load`, `publish`
  (create-only generations + conditional `HEAD` swap; a racing publisher is
  fenced), `load_if_changed` / `load_if_changed_into` (one metadata request
  when unchanged; the latter refreshes a replica in place), and
  `load_or_empty_with` (a writer's first start), over
  objstore-r's GCS, local-directory and in-memory backends. Separate crate,
  so link-r's own manifest gains no dependency.

### Bring your own embedder

- `LinkIndex<E: Embedder = DefaultEmbedder>`: existing code infers exactly
  as before; `open_or_create_with`, `open_with`, `in_memory_with` and
  `from_bytes_with` take any `Embedder` (e.g. a remote embedding API), and
  the header's embedder-identity check still refuses a mismatched index.
  The update/refresh/GitHub builders follow the index's embedder.
- `facade::default_embedder()` is public; `LinkIndex::embedder()`.
- `LinkIndex::reload_from_bytes`: swap a newer image in place, keeping the
  embedder (the replica refresh path).

### Performance and resources

- `LinkIndex::freeze()`: after a sync, replace the builder and its search
  snapshot (two copies of the corpus) with one image searched zero-copy.
- Saving an index that was loaded and not mutated writes the original image
  (no decode/re-encode, no zstd pass).
- One HTTP client per `LinkIndex`, reused by every update/refresh/GitHub
  run (connection pool and TLS configuration survive between runs); same
  client configuration as `HttpFetcher::new`.

### Fixes

- `atomic_write` temporaries are named by pid **and** a process-wide counter
  and created exclusively: two threads (or two indexes) saving in one process
  could previously truncate each other's half-written temporary.

### Security

- Lockfiles (crate and Python binding) move to rustls 0.23.45 for
  RUSTSEC-2026-0285.

### Tooling

- Fuzz target `fuzz_index_from_bytes` (every buffer alignment; accepted
  images must re-serialize to themselves), seeded with the 0.2.0 fixture.
- Benchmarks for image load and the save fast path; CI jobs for the
  objstore crate (with the GCS emulator), semver and the golden fixture;
  `scripts/ci-local.sh`.
- `rust-version` is now 1.89; the pinned nightly is unchanged.

## 0.2.0

### GitHub tree-API source (feature `github`)

Index a GitHub repository — or one subdirectory of it — without crawling
anything, and without depending on its link structure:

- One `git/trees/{ref}?recursive=1` call lists every file with its blob SHA.
  The SHA is the change token: entries matching a caller-supplied validator
  are reported as revalidated and **never fetched**, so an unchanged
  repository costs **exactly one HTTPS request per sync**. Oversized files
  are skipped from the tree's own size field — no fetch ever issued.
- New/changed blobs are fetched from raw.githubusercontent.com, classified by
  path (raw serves everything as `text/plain`), and yielded with the blob SHA
  as their entity tag — which a persistent backend stores and hands back as
  the next sync's validators, closing the loop across process restarts.
- `GithubSpec::parse` accepts both spellings and normalizes them:
  `https://github.com/owner/repo[/tree/ref[/dir]]` and
  `github:owner/repo@ref[//dir]` (the only unambiguous form for branch names
  containing `/`). A bare repo resolves its default branch with one extra call.
- `GithubAuth` confines the bearer token to exactly `api.github.com` and
  `raw.githubusercontent.com` (overridable for GHE) — a document may link
  anywhere; the token must not follow. Works for public, private, and
  internal repositories (raw accepts `Bearer`).
- Facade: `LinkIndex::update_github(spec)` with token/depth (directory
  levels)/max-bytes/extension filters, folding revalidated files into the
  report exactly as the crawler folds 304s.
- Hermetic end-to-end tests against a loopback fake GitHub pin the counts:
  1 tree call + 0 fetches for an unchanged repo, +1 fetch for a one-file
  change, typed error on a truncated listing, depth/extension confinement,
  and token confinement to the two hosts.

### Fixed

- Content-type classification now refines a generic `text/plain` with path
  evidence (`ResourceKind::refine_with_path`): raw.githubusercontent serves
  `.md` as `text/plain`, which classified markdown as plain text and disabled
  link-following on the crawl path. A specific header claim is never
  overridden.

### CI

- The Python-binding job runs on Linux, macOS, and Windows on every push (it
  was ubuntu-only outside releases), so persisted-store read/write is proven
  per-OS continuously.

## 0.1.0

Initial release: a tiny, embeddable crawl-and-resolve link index. Give it a
parent URL; it crawls sublinks to a configurable depth, distills each page, and
builds a small mmap-able hybrid index. Queries resolve to ranked **remote URLs** —
document bodies stay at the URL and are fetched lazily by the caller.

### Acquisition

- Recursive crawler with settable depth, scope (path-prefix, same-host,
  subdomains, allowlist), page caps, bounded concurrency, per-host politeness
  pacing, retry with `Retry-After` honoured in both RFC 9110 forms, optional
  `robots.txt`, and independent follow/index filters.
- **Conditional revalidation**: stored ETags are sent as `If-None-Match`, a 304
  skips the body entirely, and previously indexed links keep the crawl frontier
  alive — so a re-crawl of an unchanged site transfers zero bodies.
- rustls-only HTTP fetcher with redirect handling, incremental size caps that
  chunked encoding cannot bypass, gzip, and pluggable auth (anonymous,
  host-scoped bearer tokens in zeroizing secrets, single-flight OAuth refresh).
- Filesystem source for local corpora and hermetic tests.

### Extraction

- Per-format extractors — HTML via a fuzz-hardened byte scanner plus DOM,
  Markdown with front-matter and ATX headings, code, and plain text — each
  distilling a page into a `Descriptor`: title, headings with depths, keywords,
  tags, snippet, BM25 terms, embed text, and outbound links.
- One canonicalization recipe and one `UrlKey` (xxh3 of the canonical URL),
  emitted from a single chunk stream so the key is identical to the hash of the
  canonical string by construction, not by convention.

### Index

- A single mmap-able file: dense vectors, BM25, document metadata, and a
  persisted link graph, all checksummed, byte-reproducible, and loaded
  zero-copy.
- Hybrid retrieval: dense embeddings fused with BM25 via reciprocal-rank
  fusion, categorical metadata prefilters, an optional one-hop graph boost, and
  `related(url)` over outbound targets and co-cited siblings.
- Knowledge-base lifecycle: TTL `refresh()`, subset refresh by URL, eviction,
  pinning, and `export()` for external backends to absorb.

### Embedders and features

- Deterministic zero-dependency hash embedder by default; ONNX `bge-small`
  behind the `onnx` feature.
- Fine-grained features so capability is installed by requirement:
  `hash`, `onnx`, `markdown`, `html`, `http`, `crawl`, `fs`, `oauth`, `robots`,
  `quant`, `parallel`.

### Verification

- Unit, hermetic pipeline, property, and security suites; three fuzz targets;
  criterion benchmarks; zero clippy warnings.
- Security posture is tested, not assumed: tokens are stripped on cross-host
  redirects, withheld from foreign hosts, never rendered in `Debug`, and URL
  userinfo is never persisted.

### Python bindings

- PyO3/maturin wheel exposing the three-verb facade (`update`, `search`,
  `refresh`) with the GIL released around blocking work.
