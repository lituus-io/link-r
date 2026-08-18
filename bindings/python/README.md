# link-r

`pip install uu-link-r` → `import link_r` (Python ≥ 3.12, one `abi3` wheel per
platform).

A tiny, embeddable crawl-and-resolve link index. Give it a parent URL: it
crawls sublinks to a configurable depth, distills keywords and topics from
each page, and builds a compact hybrid index (dense embeddings + BM25 +
metadata filters). Queries resolve to ranked **remote URLs** — document bodies
stay at the URL and are fetched lazily by the caller.

```python
import link_r

idx = link_r.LinkIndex.open_or_create("kb.lnkr")
idx.update("https://docs.example.com/", depth=2)
idx.save()

for hit in idx.search("how does auth work", k=10):
    print(f"{hit.score:.3f}  {hit.url}")

# Keep the knowledge base fresh: 304s are free, dead links are evicted.
idx.refresh(ttl_secs=86_400)
```

- **Zero-copy.** The on-disk index *is* the file: search runs against mapped
  pages with no deserialization step, verified byte-reproducible.
- **Hybrid retrieval.** Dense embeddings (deterministic hash embedder by
  default; ONNX `bge-small` behind the `onnx` feature) fused with BM25 via
  reciprocal-rank fusion, with categorical prefilters and an optional
  link-graph boost.
- **Conditional everything.** Stored `ETag`s become `If-None-Match`; an
  unchanged page answers `304` and transfers no body, and previously indexed
  links keep the crawl frontier alive through it.
- **Credential-safe.** Tokens are host-scoped, stripped on cross-host
  redirects, never rendered in debug output; URL userinfo is never persisted.
- **Thread-safe.** Blocking calls detach from the interpreter; driving the
  index through `asyncio.to_thread` is the intended pattern. One `abi3` wheel
  per platform (Python ≥ 3.12).

Dual-licensed AGPL-3.0-or-later / commercial (contact spicyzhug@gmail.com).
