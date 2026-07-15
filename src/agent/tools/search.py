"""Search tool: ``SearchTool`` Protocol + keyless ``FakeSearch`` + real web search.

``FakeSearch`` performs a deterministic keyword search over the local Markdown
corpus and returns ``local://<file>`` URLs, so an entire research run is
reproducible offline. ``OpenWebSearch`` calls a real provider (Tavily by default)
with a lazy import, so the keyless path never needs the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..textutil import best_sentences, content_word_set
from .documents import iter_corpus_files, parse_doc


@dataclass(frozen=True)
class SearchResult:
    """A single search hit. ``url`` is ``local://<file>`` in keyless mode."""

    title: str
    url: str
    snippet: str


@runtime_checkable
class SearchTool(Protocol):
    name: str

    def search(self, query: str) -> list[SearchResult]:  # pragma: no cover
        ...


def list_corpus(corpus_dir: Path) -> list[dict[str, str]]:
    """List corpus documents as ``{filename, title, url}`` for the UI corpus view.

    Reuses the same title parsing as :class:`FakeSearch`, so the titles shown in
    the corpus-coverage panel match the source titles on gathered evidence. Files
    that cannot be read (e.g. a corrupt PDF) are skipped, not fatal.
    """
    docs: list[dict[str, str]] = []
    for path in iter_corpus_files(corpus_dir):
        try:
            title, _ = parse_doc(path)
        except Exception:  # noqa: BLE001 - one unreadable file must not break the listing
            continue
        docs.append({"filename": path.name, "title": title, "url": f"local://{path.name}"})
    return docs


class FakeSearch:
    """Deterministic keyword search over ``corpus_dir`` (``.md``/``.txt``/``.pdf``)."""

    name = "fake-search"

    def __init__(self, corpus_dir: Path, top_k: int = 5) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.top_k = top_k
        self._docs: list[tuple[str, str, str]] | None = None  # (filename, title, body)

    def _corpus(self) -> list[tuple[str, str, str]]:
        if self._docs is None:
            docs = []
            for path in iter_corpus_files(self.corpus_dir):
                try:
                    title, body = parse_doc(path)
                except Exception:  # noqa: BLE001 - skip an unreadable file, don't crash the run
                    continue
                docs.append((path.name, title, body))
            self._docs = docs
        return self._docs

    def search(self, query: str) -> list[SearchResult]:
        """Return up to ``top_k`` corpus docs ranked by keyword overlap.

        A query term scores once for appearing in the body and once more for
        appearing in the title, so a term present in both counts double. Docs
        with zero overlap are excluded, so an off-topic question (not covered by
        the corpus) returns nothing — which is what lets the pipeline honestly
        abstain instead of inventing facts.
        """
        q = content_word_set(query)
        if not q:
            return []
        scored = []
        for filename, title, body in self._corpus():
            body_set = content_word_set(body)
            title_set = content_word_set(title)
            score = len(q & body_set) + len(q & title_set)  # title hit adds to body hit
            if score > 0:
                scored.append((score, filename, title, body))
        scored.sort(key=lambda t: (-t[0], t[1]))  # score desc, filename asc (stable)
        results = []
        for _score, filename, title, body in scored[: self.top_k]:
            snippet = " ".join(best_sentences(query, body, k=2))
            results.append(SearchResult(title=title, url=f"local://{filename}", snippet=snippet))
        return results


class OpenWebSearch:
    """Real web search via a pluggable provider (Tavily by default)."""

    name = "web-search"

    def __init__(self, api_key: str, backend: str = "tavily", top_k: int = 5) -> None:
        # Fail fast at construction time, not mid-run on the first search call.
        if backend != "tavily":
            raise NotImplementedError(
                f"search backend {backend!r} is not implemented yet — only 'tavily' is; "
                "set SEARCH_BACKEND=tavily (or contribute the backend)."
            )
        self._api_key = api_key
        self.backend = backend
        self.top_k = top_k

    def search(self, query: str) -> list[SearchResult]:  # pragma: no cover
        from tavily import TavilyClient  # lazy: keyless path never imports

        client = TavilyClient(api_key=self._api_key)
        data: dict[str, Any] = client.search(query=query, max_results=self.top_k)
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
            )
            for r in data.get("results", [])
        ]


def get_search(settings: Any) -> SearchTool:
    """Factory: pick a search tool from config and optionally wrap with cache."""
    if settings.search_provider == "web":
        if not settings.search_api_key:
            raise ValueError("search_provider=web but SEARCH_API_KEY is empty.")
        tool: SearchTool = OpenWebSearch(
            api_key=settings.search_api_key,
            backend=settings.search_backend,
            top_k=settings.top_search_results,
        )
    else:
        tool = FakeSearch(corpus_dir=settings.corpus_dir, top_k=settings.top_search_results)

    if settings.enable_cache:
        from ..cache import CachedSearch

        return CachedSearch(tool, maxsize=settings.cache_size)  # type: ignore[return-value]
    return tool
