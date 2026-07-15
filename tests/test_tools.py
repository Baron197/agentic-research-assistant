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


# --- researching your OWN documents (multi-format corpus) --------------------

def test_corpus_supports_txt_and_md(tmp_path):
    # Point corpus_dir at an arbitrary folder of the user's own .txt/.md files.
    (tmp_path / "revenue.txt").write_text(
        "Revenue grew because the new pricing tier increased average contract "
        "value across enterprise accounts.", encoding="utf-8")
    (tmp_path / "onboarding.md").write_text(
        "# Onboarding\nNew engineers set up the toolchain on day one.",
        encoding="utf-8")

    search = FakeSearch(corpus_dir=tmp_path, top_k=5)
    hits = search.search("revenue pricing enterprise contract")
    assert any(h.url == "local://revenue.txt" for h in hits)

    fetch = FakeFetch(corpus_dir=tmp_path)
    assert "pricing tier" in fetch.fetch("local://revenue.txt")


def test_corpus_reads_pdf(tmp_path):
    # PDF support is optional (needs pypdfium2); the test PDF is built with
    # reportlab. Both are absent from the keyless install, so CI skips this.
    pytest.importorskip("pypdfium2")
    pytest.importorskip("reportlab")
    from reportlab.pdfgen.canvas import Canvas

    pdf_path = tmp_path / "brief.pdf"
    canvas = Canvas(str(pdf_path))
    canvas.drawString(72, 720, "Migration Brief")
    canvas.drawString(72, 700, "The database migration reduces query latency "
                                "for analytics workloads.")
    canvas.save()

    search = FakeSearch(corpus_dir=tmp_path, top_k=5)
    hits = search.search("database migration latency analytics")
    assert any(h.url == "local://brief.pdf" for h in hits)

    fetch = FakeFetch(corpus_dir=tmp_path)
    assert "migration" in fetch.fetch("local://brief.pdf").lower()


def test_corpus_skips_unreadable_files(tmp_path):
    # A corrupt/unsupported-content file must not break loading the good ones.
    (tmp_path / "good.md").write_text("# Good\nUseful notes about caching.",
                                      encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 not actually a valid pdf")

    search = FakeSearch(corpus_dir=tmp_path, top_k=5)
    hits = search.search("caching notes")
    assert any(h.url == "local://good.md" for h in hits)  # good file still found


def test_real_mode_factories_wire_up_without_network():
    # Constructing the real providers must not need network or the SDKs (imports
    # are lazy inside the call methods), so this runs on the keyless install too.
    from agent.llm import OpenAILLM, get_llm
    from agent.tools.fetch import HttpFetch, get_fetch
    from agent.tools.search import OpenWebSearch, get_search

    real = Settings(
        llm_provider="openai", openai_api_key="x",
        search_provider="web", search_api_key="x", search_backend="tavily",
        fetch_provider="http", enable_cache=False,
    )
    assert isinstance(get_search(real), OpenWebSearch)
    assert isinstance(get_fetch(real), HttpFetch)
    assert isinstance(get_llm(real), OpenAILLM)
