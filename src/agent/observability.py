"""First-class observability: per-step tracing, costing, and run aggregation.

Every run records an ordered list of ``Step`` spans (node, tool, token, usd, ms).
Runs are persisted as ``runs/<id>.json`` plus a one-line summary appended to
``runs/index.jsonl`` (guarded by a lock). ``aggregate()`` summarises the index
and is deliberately resilient to a torn final line. The ``Step`` shape mirrors a
Langfuse / OpenTelemetry span so this layer can be swapped for a real backend.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .schemas import RunResult, Step

# --- Cost table -------------------------------------------------------------
# USD per 1,000 tokens (blended input/output estimate for simplicity). The fake
# provider is free, which is why keyless runs honestly report $0.00.
PRICES: dict[str, float] = {
    "fake-llm": 0.0,
    "gpt-4o-mini": 0.0004,
    "gpt-4o": 0.005,
    "gpt-4.1": 0.004,
    "gpt-4.1-mini": 0.0004,
}
DEFAULT_PRICE_PER_1K = 0.0004  # fall back to a mini-tier estimate for unknowns


def cost_usd(model: str, tokens: int) -> float:
    """USD cost for ``tokens`` on ``model`` using the blended price table."""
    rate = PRICES.get(model, DEFAULT_PRICE_PER_1K)
    return round((tokens / 1000.0) * rate, 6)


# --- Tracing ----------------------------------------------------------------
@dataclass
class StepBuilder:
    """Mutable span filled in inside a ``Tracer.span`` block."""

    node: str
    tool: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    tokens: int = 0
    usd: float = 0.0
    ms: float = 0.0

    def to_step(self) -> Step:
        return Step(
            node=self.node,
            tool=self.tool,
            input_summary=self.input_summary[:300],
            output_summary=self.output_summary[:300],
            tokens=self.tokens,
            usd=self.usd,
            ms=round(self.ms, 3),
        )


class Tracer:
    """Times spans and prices token usage for the configured model."""

    def __init__(self, model: str = "fake-llm") -> None:
        self.model = model

    @contextmanager
    def span(self, node: str, tool: str | None = None) -> Iterator[StepBuilder]:
        sb = StepBuilder(node=node, tool=tool)
        start = perf_counter()
        try:
            yield sb
        finally:
            sb.ms = (perf_counter() - start) * 1000.0

    def cost(self, tokens: int) -> float:
        return cost_usd(self.model, tokens)


# --- Persistence ------------------------------------------------------------
_INDEX_LOCK = threading.Lock()


def _summary_line(result: RunResult) -> dict[str, object]:
    return {
        "run_id": result.run_id,
        "created_at": result.created_at,
        # Truncated so the append-only index stays lean; enough to identify a run
        # in the history browser.
        "question": result.question[:200],
        "status": result.status,
        "iterations": result.iterations,
        "tool_calls": result.tool_calls,
        "tokens": result.tokens,
        "usd": result.usd,
        "latency_ms": result.latency_ms,
        "n_steps": len(result.trace),
        "n_claims": len(result.report.all_claims()),
        "citation_coverage": round(result.citation_coverage, 4),
    }


def persist_run(result: RunResult, traces_dir: Path) -> Path:
    """Write the full run JSON and append a summary line to the index.

    The index append is guarded by a module-level lock so concurrent API
    requests cannot interleave half-written lines.
    """
    traces_dir = Path(traces_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)
    run_path = traces_dir / f"{result.run_id}.json"
    run_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    line = json.dumps(_summary_line(result))
    with _INDEX_LOCK:
        with (traces_dir / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return run_path


def load_run(run_id: str, traces_dir: Path) -> RunResult | None:
    """Load a persisted run by id, or ``None`` if it does not exist.

    Defense in depth: an id containing path separators (traversal) resolves to
    ``None`` rather than a filesystem lookup, and a corrupt/foreign file reads
    as "not found" instead of raising into the caller.
    """
    if not run_id or Path(run_id).name != run_id:
        return None
    path = Path(traces_dir) / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:  # pydantic ValidationError subclasses ValueError
        return None


def recent_runs(traces_dir: Path, limit: int = 20) -> list[dict[str, object]]:
    """Most-recent-first run summaries from the index (torn-line tolerant).

    Backs the API's ``GET /runs`` and the UI's run-history browser. Reuses the
    same append-only index that ``aggregate`` reads, so nothing new is persisted.
    """
    index = Path(traces_dir) / "index.jsonl"
    if not index.exists():
        return []
    rows: list[dict[str, object]] = []
    for raw in index.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue  # tolerate a torn line
    rows.reverse()  # index is append-order; newest last -> reverse for newest-first
    return rows[: max(0, limit)]


def _percentile_nearest_rank(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: rank = ceil(pct * n), clamped to [1, n]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(pct * len(ordered))
    idx = min(max(rank - 1, 0), len(ordered) - 1)
    return ordered[idx]


def _empty_aggregate() -> dict[str, float | int]:
    """Stable zero-valued shape so /metrics never changes keys on empty data."""
    return {
        "runs": 0,
        "avg_cost_usd": 0.0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "avg_steps": 0.0,
        "avg_citation_coverage": 0.0,
    }


def aggregate(traces_dir: Path) -> dict[str, float | int]:
    """Aggregate the run index into headline observability metrics.

    Resilient to a torn/partial final line (skips anything that does not parse),
    which is the realistic failure mode for an append-only JSONL written by
    multiple processes.
    """
    index = Path(traces_dir) / "index.jsonl"
    if not index.exists():
        return _empty_aggregate()

    rows = []
    for raw in index.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue  # tolerate a torn line

    n = len(rows)
    if n == 0:
        return _empty_aggregate()

    latencies = [float(r.get("latency_ms", 0.0)) for r in rows]
    return {
        "runs": n,
        "avg_cost_usd": round(sum(float(r.get("usd", 0.0)) for r in rows) / n, 6),
        "avg_latency_ms": round(sum(latencies) / n, 3),
        "p95_latency_ms": round(_percentile_nearest_rank(latencies, 0.95), 3),
        "avg_steps": round(sum(int(r.get("n_steps", 0)) for r in rows) / n, 3),
        "avg_citation_coverage": round(
            sum(float(r.get("citation_coverage", 0.0)) for r in rows) / n, 4
        ),
    }
