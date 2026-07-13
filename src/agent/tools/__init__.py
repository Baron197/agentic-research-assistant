"""Tools the agents use to gather evidence, each behind a small Protocol.

Both tools follow the same Strategy pattern as the LLM: a Protocol, a
deterministic ``Fake`` implementation for the keyless path, a real
implementation with a lazy SDK import, and a ``get_*`` factory. The interfaces
are intentionally MCP-server-friendly (single string in, structured out) so they
could be exposed as MCP tools later with no change to the agents.
"""

from __future__ import annotations

from .fetch import FakeFetch, FetchTool, HttpFetch, get_fetch
from .search import FakeSearch, OpenWebSearch, SearchResult, SearchTool, get_search

__all__ = [
    "SearchResult",
    "SearchTool",
    "FakeSearch",
    "OpenWebSearch",
    "get_search",
    "FetchTool",
    "FakeFetch",
    "HttpFetch",
    "get_fetch",
]
