# Copyright: lituus-io, all rights reserved.
# Author: terekete <spicyzhug@gmail.com>
#
# Type stubs for the link_r native extension module.

from typing import List, Optional

__version__: str

class LinkRError(Exception):
    """Raised when a link-r operation fails."""

class Hit:
    url: str
    score: float
    title: Optional[str]
    snippet: str
    kind: str
    tags: List[str]
    def __repr__(self) -> str: ...

class UpdateReport:
    added: int
    updated: int
    unchanged: int
    skipped: int
    failed: int
    pages_seen: int
    def __repr__(self) -> str: ...

class RefreshReport:
    refreshed: int
    unchanged: int
    removed: int
    failed: int
    total: int
    def __repr__(self) -> str: ...

class LinkIndex:
    @staticmethod
    def open_or_create(path: str) -> "LinkIndex":
        """Open an index at ``path`` or create one bound to it."""
    @staticmethod
    def open(path: str) -> "LinkIndex":
        """Open an existing index (raises if missing)."""
    @staticmethod
    def in_memory() -> "LinkIndex":
        """Create an in-memory index not bound to a file."""
    def update(
        self,
        url: str,
        depth: int = 2,
        max_pages: int = 1000,
        concurrency: int = 8,
        embed_batch: int = 64,
        token: Optional[str] = None,
        scope: Optional[str] = None,
        min_delay_ms: int = 0,
        path_contains: Optional[List[str]] = None,
        extensions: Optional[List[str]] = None,
        index_path_contains: Optional[List[str]] = None,
        pin: bool = False,
    ) -> UpdateReport:
        """Crawl ``url`` to ``depth`` and index the pages (deduplicated by URL).

        ``scope`` is one of ``"path"`` (default), ``"host"``, ``"subdomains"``.
        ``path_contains`` confines *crawling* to links whose URL path contains *all*
        of the given substrings; ``index_path_contains`` narrows *indexing* the same
        way (follow broadly, index narrowly). ``extensions`` (e.g. ``["md"]``)
        indexes only pages with those file extensions (others are still crawled for
        links). ``token`` sets a bearer credential for private sources.
        """
    def refresh(
        self,
        ttl_secs: int,
        max_age_secs: Optional[int] = None,
        evict_unreachable: bool = True,
        concurrency: int = 8,
        token: Optional[str] = None,
        token_host: Optional[str] = None,
    ) -> RefreshReport:
        """Re-validate indexed links older than ``ttl_secs``.

        Unchanged pages (HTTP 304) are re-timestamped, changed pages re-indexed,
        and dead/unreachable pages evicted (unless ``evict_unreachable`` is False or
        the link is pinned). ``max_age_secs`` hard-evicts unpinned links older than
        it without fetching. ``token`` (scoped to ``token_host`` when given)
        authenticates private-source refreshes.
        """
    def pin(self, url_prefix: str) -> int:
        """Pin links whose URL starts with ``url_prefix`` (retained forever)."""
    def unpin(self, url_prefix: str) -> int:
        """Unpin links whose URL starts with ``url_prefix``."""
    def search(
        self,
        query: str,
        k: int = 10,
        path_prefix: Optional[str] = None,
        tag: Optional[str] = None,
        graph_boost: float = 0.0,
    ) -> List[Hit]:
        """Search, returning up to ``k`` ranked hits.

        ``path_prefix`` and ``tag`` apply a categorical metadata prefilter.
        ``graph_boost`` (>0) re-ranks results by knowledge-graph connectivity.
        """
    def related(self, url: str, k: int = 10) -> List[Hit]:
        """Return the ``k`` links most related to ``url`` in the knowledge graph
        (its outbound targets and co-cited siblings), ranked by connectivity.
        """
    def save(self) -> None:
        """Atomically save to the bound path."""
    def save_as(self, path: str) -> None:
        """Save to a new path and bind to it."""
    def __len__(self) -> int: ...
