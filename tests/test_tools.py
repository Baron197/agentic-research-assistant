"""Fake tools + caching behaviour."""

from __future__ import annotations

import pytest

from agent.cache import CachedSearch
from agent.config import Settings
from agent.tools.fetch import FakeFetch
from agent.tools.search import FakeSearch


def _corpus_dir():
    return Settings().corpus_dir


def test_fake_search_returns_relevant_docs():
    search = FakeSearch(corpus_dir=_corpus_dir(), top_k=5)
    results = search.search("reranking cross-encoder relevance")
    assert results, "expected relevant corpus hits"
    assert any("reranking" in r.url for r in results)
    assert all(r.url.startswith("local://") for r in results)


def test_fake_search_abstains_off_topic():
    search = FakeSearch(corpus_dir=_corpus_dir(), top_k=5)
    assert search.search("sourdough bread baking recipe") == []


def test_fake_fetch_resolves_local_urls():
    fetch = FakeFetch(corpus_dir=_corpus_dir())
    text = fetch.fetch("local://reranking.md")
    assert "cross-encoder" in text.lower()
    with pytest.raises(ValueError):
        fetch.fetch("https://example.com")


def test_fake_fetch_blocks_path_traversal():
    # A corpus URL is a bare filename; separators / '..' / absolute paths must
    # never resolve outside the corpus directory (regression).
    fetch = FakeFetch(corpus_dir=_corpus_dir())
    for url in ("local://../../README.md", "local://..", "local://sub/doc.md",
                "local://C:/windows/win.ini", "local://"):
        with pytest.raises((ValueError, FileNotFoundError)):
            fetch.fetch(url)


def test_web_search_unimplemented_backend_fails_fast():
    # 'brave'/'serpapi' are accepted by config but not implemented; constructing
    # the tool must fail immediately, not mid-run on the first search call.
    from agent.tools.search import OpenWebSearch

    with pytest.raises(NotImplementedError):
        OpenWebSearch(api_key="k", backend="brave")


def test_cached_search_marks_repeats():
    inner = FakeSearch(corpus_dir=_corpus_dir(), top_k=5)
    cached = CachedSearch(inner, maxsize=8)
    first = cached.search("hybrid search")
    assert cached.hits == 0 and cached.misses == 1
    second = cached.search("hybrid search")  # same query -> cache hit
    assert cached.hits == 1
    assert first == second


def test_cached_fetch_is_case_sensitive():
    # URLs are case-sensitive; the cache must not collapse case (regression).
    class _Stub:
        name = "stub"
        def fetch(self, url: str) -> str:
            return f"CONTENT::{url}"

    from agent.cache import CachedFetch

    cf = CachedFetch(_Stub())
    a = cf.fetch("local://Reranking.md")
    b = cf.fetch("local://reranking.md")
    assert a != b  # different URLs -> different content, no collision
    assert cf.hits == 0


def test_cached_search_returns_independent_copies():
    class _Stub:
        name = "stub"
        def search(self, query: str):
            return ["a", "b"]

    cached = CachedSearch(_Stub(), maxsize=4)
    first = cached.search("q")
    first.append("MUTATED")  # caller mutates the returned list
    second = cached.search("q")
    assert second == ["a", "b"]  # cache entry was not corrupted
