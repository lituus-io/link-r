# Copyright: lituus-io, all rights reserved.
# Author: terekete <spicyzhug@gmail.com>
#
# Hermetic end-to-end smoke test for the link_r Python bindings.
#
# Serves a tiny two-page site from a local http.server (no external network),
# crawls it, searches, and verifies dedup on re-crawl. Run with:
#   maturin develop && pytest -q

import functools
import http.server
import socketserver
import tempfile
import threading
from pathlib import Path

import link_r


def _serve(directory: str):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _make_site(root: Path):
    (root / "index.html").write_text(
        "<html><head><title>Home</title></head><body>"
        "<h1>Welcome</h1><p>alpha root content about widgets</p>"
        '<a href="page.html">details</a></body></html>'
    )
    (root / "page.html").write_text(
        "<html><head><title>Details</title></head><body>"
        "<h1>Private Service Connect</h1>"
        "<p>PSC reaches services privately over an internal endpoint.</p>"
        "</body></html>"
    )


def test_crawl_index_search_and_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_site(root)
        httpd, port = _serve(tmp)
        base = f"http://127.0.0.1:{port}/"
        try:
            idx = link_r.LinkIndex.in_memory()
            report = idx.update(base, depth=2)
            assert report.added >= 2, report
            assert len(idx) >= 2

            hits = idx.search("private service connect", k=5)
            assert hits, "expected at least one hit"
            assert any("page.html" in h.url for h in hits), [h.url for h in hits]

            # Re-crawling the unchanged site must add nothing (dedup by URL).
            report2 = idx.update(base, depth=2)
            assert report2.added == 0, report2
            assert report2.unchanged >= 2, report2
        finally:
            httpd.shutdown()


def test_save_and_reopen(tmp_path):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_site(root)
        httpd, port = _serve(tmp)
        base = f"http://127.0.0.1:{port}/"
        index_path = str(tmp_path / "kb.lnkr")
        try:
            idx = link_r.LinkIndex.open_or_create(index_path)
            idx.update(base, depth=1)
            n = len(idx)
            idx.save()
        finally:
            httpd.shutdown()

        reopened = link_r.LinkIndex.open(index_path)
        assert len(reopened) == n
        hits = reopened.search("widgets", k=5)
        assert hits


def test_refresh_pin_and_filtered_search():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_site(root)
        httpd, port = _serve(tmp)
        base = f"http://127.0.0.1:{port}/"
        try:
            idx = link_r.LinkIndex.in_memory()
            idx.update(base, depth=2, concurrency=4, embed_batch=8)
            n = len(idx)
            assert n >= 2

            # Pin retains links across a max-age refresh (nothing evicted).
            assert idx.pin(base) >= 1
            report = idx.refresh(ttl_secs=0, max_age_secs=0)
            assert report.removed == 0, report
            assert len(idx) == n

            # Categorical filtered search still resolves the target page.
            hits = idx.search("private service connect", k=5, path_prefix="/")
            assert hits
            assert isinstance(report.total, int)
        finally:
            httpd.shutdown()
