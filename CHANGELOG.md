# Changelog

All notable changes to link-r are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/spec/v2.0.0.html).

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
