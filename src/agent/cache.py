"""A tiny thread-safe LRU cache plus caching wrappers for the tools.

The cache is deliberately minimal and self-contained (``OrderedDict`` + a
``threading.Lock``) so it has zero dependencies and is trivial to reason about.
Keys are normalised so that an omitted default and an explicit-but-equal default
never produce two different entries.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

_MISS = object()


def normalise_key(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace for stable keys."""
    return " ".join(text.lower().split())


class LRUCache(Generic[K, V]):
    """Bounded least-recently-used cache, safe for concurrent access."""

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = max(1, maxsize)
        self._data: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> tuple[bool, V | None]:
        """Return ``(hit, value)``; moves the key to most-recently-used on hit."""
        with self._lock:
            val = self._data.get(key, _MISS)
            if val is _MISS:
                self.misses += 1
                return False, None
            self._data.move_to_end(key)
            self.hits += 1
            return True, val  # type: ignore[return-value]

    def put(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class CachedSearch:
    """Wrap any ``SearchTool`` with an LRU cache keyed by the normalised query."""

    def __init__(self, inner: object, maxsize: int = 256) -> None:
        self._inner = inner
        self._cache: LRUCache[str, list] = LRUCache(maxsize)

    @property
    def name(self) -> str:
        return f"cached:{getattr(self._inner, 'name', 'search')}"

    @property
    def hits(self) -> int:
        return self._cache.hits

    @property
    def misses(self) -> int:
        return self._cache.misses

    def search(self, query: str) -> list:
        key = normalise_key(query)
        hit, val = self._cache.get(key)
        if hit:
            return list(val)  # copy: callers must not mutate the cached entry
        result = self._inner.search(query)  # type: ignore[attr-defined]
        self._cache.put(key, list(result))
        return list(result)


class CachedFetch:
    """Wrap any ``FetchTool`` with an LRU cache keyed by the URL."""

    def __init__(self, inner: object, maxsize: int = 256) -> None:
        self._inner = inner
        self._cache: LRUCache[str, str] = LRUCache(maxsize)

    @property
    def name(self) -> str:
        return f"cached:{getattr(self._inner, 'name', 'fetch')}"

    @property
    def hits(self) -> int:
        return self._cache.hits

    @property
    def misses(self) -> int:
        return self._cache.misses

    def fetch(self, url: str) -> str:
        # URLs are case-sensitive (e.g. local://Reranking.md != reranking.md), so
        # only strip surrounding whitespace — never lowercase or collapse them.
        key = url.strip()
        hit, val = self._cache.get(key)
        if hit:
            return val  # type: ignore[return-value]
        result = self._inner.fetch(url)  # type: ignore[attr-defined]
        self._cache.put(key, result)
        return result
