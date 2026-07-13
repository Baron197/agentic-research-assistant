"""The dependency bundle every node closes over.

Nodes in the graph are pure functions of ``(state, ctx)``. ``AgentContext`` is the
``ctx``: the chosen providers, the tracer, settings, and an optional human-
approval callback. Building it once in the runner and threading it through keeps
the nodes free of global state and trivial to unit-test with fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .config import Settings
from .llm import LLM
from .observability import Tracer
from .schemas import Report
from .tools.fetch import FetchTool
from .tools.search import SearchTool


@dataclass
class AgentContext:
    """Everything the agent nodes need, injected explicitly (no globals)."""

    settings: Settings
    llm: LLM
    search: SearchTool
    fetch: FetchTool
    tracer: Tracer
    approval_fn: Callable[[Report | None], bool] | None = None
