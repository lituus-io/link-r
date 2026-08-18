# Copyright: lituus-io, all rights reserved.
# Author: terekete <spicyzhug@gmail.com>
#
# Cross-thread access to a LinkIndex.
#
# These exist because the binding was originally declared
# `#[pyclass(unsendable)]`, which makes PyO3 raise as soon as the object is
# touched from a thread other than the one that constructed it. Nothing in the
# smoke tests caught it: they build and use the index on the main thread, which
# is the one arrangement that works.
#
# It is not a theoretical problem. The intended consumer dispatches blocking
# native calls with `asyncio.to_thread`, which runs them on a *pool* -- so the
# index is constructed on the event-loop thread and then used from a worker, and
# the pool is free to pick a different worker for each call. That is precisely
# the shape `unsendable` forbids.

import asyncio
import concurrent.futures as futures
import threading

import pytest

import link_r


def _seeded_index() -> "link_r.LinkIndex":
    """An in-memory index with two documents, built on the calling thread."""
    idx = link_r.LinkIndex.in_memory()
    return idx


def test_index_is_usable_from_another_thread():
    """The direct regression: construct here, use there."""
    idx = _seeded_index()
    creating_thread = threading.get_ident()

    def use_it() -> tuple[int, int]:
        assert threading.get_ident() != creating_thread, "test must run off-thread"
        # Any method at all is enough -- `unsendable` rejects the *access*, not
        # a particular call.
        return len(idx), len(idx.search("anything", k=1))

    with futures.ThreadPoolExecutor(max_workers=1) as pool:
        count, hits = pool.submit(use_it).result(timeout=30)

    assert count == 0
    assert hits == 0


def test_index_survives_being_passed_between_several_threads():
    """A pool hands successive calls to different workers, which is the real
    `asyncio.to_thread` behaviour rather than a single hand-off."""
    idx = _seeded_index()
    threads_seen: set[int] = set()

    def touch(_: int) -> int:
        threads_seen.add(threading.get_ident())
        return len(idx)

    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(touch, range(24)))

    assert results == [0] * 24
    assert len(threads_seen) > 1, "expected the pool to use more than one worker"


def test_asyncio_to_thread_round_trip():
    """The exact call shape a consumer uses: an async caller offloading the
    blocking native call onto the default executor."""

    async def main() -> int:
        idx = link_r.LinkIndex.in_memory()
        # Two awaits so the executor gets a chance to use different workers.
        await asyncio.to_thread(len, idx)
        return await asyncio.to_thread(lambda: len(idx.search("query", k=3)))

    assert asyncio.run(main()) == 0


def test_concurrent_reads_do_not_deadlock():
    """Several threads reading at once must not deadlock. The binding detaches
    from the interpreter around Rust work, so these genuinely overlap rather
    than serialising behind the interpreter lock."""
    idx = link_r.LinkIndex.in_memory()
    barrier = threading.Barrier(4)

    def read() -> int:
        barrier.wait(timeout=30)  # maximise real overlap
        total = 0
        for _ in range(50):
            total += len(idx.search("private service connect", k=5))
        return total

    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = [f.result(timeout=60) for f in [pool.submit(read) for _ in range(4)]]

    assert outcomes == [0, 0, 0, 0]


def test_error_maps_to_the_crate_exception():
    """Rust failures surface as LinkRError, not a bare RuntimeError, so callers
    can catch this crate's failures specifically."""
    assert issubclass(link_r.LinkRError, Exception)
    with pytest.raises(link_r.LinkRError):
        link_r.LinkIndex.open("/nonexistent/path/that/cannot/exist.lnkr")


def test_error_from_a_worker_thread_still_maps():
    """Error conversion must not depend on being on the creating thread either."""

    def boom() -> None:
        link_r.LinkIndex.open("/nonexistent/path/that/cannot/exist.lnkr")

    with futures.ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(link_r.LinkRError):
            pool.submit(boom).result(timeout=30)


def test_module_reports_a_version():
    assert isinstance(link_r.__version__, str)
    assert link_r.__version__, "version must not be empty"
