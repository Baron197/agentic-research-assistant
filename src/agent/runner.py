"""The single entry point used by the API, UI, eval, and CLI.

``run(question, ...)`` builds the provider context from settings, executes the
compiled graph, persists the trace, and returns a ``RunResult``. Everything is
keyless by default; pass ``settings=`` or keyword overrides to change behaviour
(e.g. ``token_budget=10`` or ``enable_critic=False`` for the A/B).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from time import perf_counter
from typing import Any
from uuid import uuid4

from .config import Settings, get_settings
from .context import AgentContext
from .graph import build_graph
from .guardrails import clamp_input
from .llm import get_llm
from .observability import Tracer, persist_run
from .schemas import Budget, Report, RunResult
from .tools.fetch import get_fetch
from .tools.search import get_search

# Hard ceiling on graph supersteps; the budget/iteration guards stop us long
# before this, but it guarantees the graph can never loop forever.
RECURSION_LIMIT = 50


def _recursion_limit(max_iterations: int) -> int:
    """Scale the superstep ceiling with the iteration cap.

    Each revise loop replays researcher+writer+critic (3 supersteps); the fixed
    nodes add ~6 more. Scaling keeps the backstop meaningful for small caps while
    ensuring a legitimately-configured high ``max_iterations`` cannot crash into
    the limit and lose the run.
    """
    return max(RECURSION_LIMIT, 12 + 3 * max_iterations)


def _tracer_model(settings: Settings) -> str:
    """The model name used to PRICE the trace, matching the active backend.

    The DSPy backend uses a real LLM (``dspy_model``), so the tracer must price it
    as such — otherwise a paid DSPy run would silently report $0.00 and break the
    "cost == 0 iff keyless" honesty guarantee.
    """
    if settings.agent_backend == "dspy":
        return settings.dspy_model.split("/")[-1]  # "openai/gpt-4o-mini" -> "gpt-4o-mini"
    if settings.llm_provider == "openai":
        return settings.openai_model
    return "fake-llm"


def build_context(
    settings: Settings,
    approval_fn: Callable[[Report | None], bool] | None = None,
) -> AgentContext:
    """Assemble providers + tracer from typed settings (the Strategy wiring)."""
    return AgentContext(
        settings=settings,
        llm=get_llm(settings),
        search=get_search(settings),
        fetch=get_fetch(settings),
        tracer=Tracer(model=_tracer_model(settings)),
        approval_fn=approval_fn,
    )


def _resolve_settings(settings: Settings | None, overrides: dict[str, Any]) -> Settings:
    if settings is None:
        return Settings(**overrides) if overrides else get_settings()
    return settings.model_copy(update=overrides) if overrides else settings


def run(
    question: str,
    *,
    settings: Settings | None = None,
    approval_fn: Callable[[Report | None], bool] | None = None,
    persist: bool = True,
    **overrides: Any,
) -> RunResult:
    """Run the full research pipeline and return a structured ``RunResult``."""
    settings = _resolve_settings(settings, overrides)
    question = clamp_input(question, settings.max_question_length)

    ctx = build_context(settings, approval_fn=approval_fn)
    graph = build_graph(ctx)

    initial: dict[str, Any] = {
        "question": question,
        "plan": None,
        "evidence": [],
        "draft": None,
        "critique": None,
        "iteration": 0,
        "budget": Budget(token_limit=settings.token_budget),
        "status": "complete",
        "rejected": [],
        "approved": True,
        "tool_calls": 0,
        "dropped_claims": 0,
        "next_action": "finalize",
        "unresolved": False,
        "researched_sqs": [],
        "trace": [],
        "final": None,
    }

    start = perf_counter()
    final_state = graph.invoke(
        initial, config={"recursion_limit": _recursion_limit(settings.max_iterations)}
    )
    latency_ms = (perf_counter() - start) * 1000.0

    report = final_state.get("final") or Report(question=question, partial=True)
    budget = final_state["budget"]
    result = RunResult(
        run_id=uuid4().hex[:12],
        question=question,
        status=final_state.get("status", "complete"),
        report=report,
        iterations=int(final_state.get("iteration", 0)),
        tool_calls=int(final_state.get("tool_calls", 0)),
        tokens=budget.tokens_used,
        usd=budget.usd_used,
        latency_ms=round(latency_ms, 3),
        dropped_claims=int(final_state.get("dropped_claims", 0)),
        trace=final_state.get("trace", []),
        evidence=final_state.get("evidence", []),
    )

    if persist:
        persist_run(result, settings.traces_dir)
    return result


def render_report_markdown(report: Report) -> str:
    """Render a ``Report`` as Markdown with inline ``[n]`` citations + sources."""
    n_by_evidence = {c.evidence_id: c.n for c in report.sources}
    lines: list[str] = ["# Research Report\n", f"**Question:** {report.question}\n"]
    if report.partial:
        lines.append("> ⚠️ This report is **partial** (stopped at a budget/iteration "
                     "limit or pending approval).\n")
    if report.summary:
        lines.append(f"## Executive Summary\n\n{report.summary}\n")

    for section in report.sections:
        lines.append(f"## {section.heading}\n")
        for claim in section.claims:
            marks = "".join(
                f"[{n_by_evidence[eid]}]" for eid in claim.evidence_ids if eid in n_by_evidence
            )
            suffix = f" {marks}" if marks else ""
            lines.append(f"- {claim.text}{suffix}")
        lines.append("")

    if report.sources:
        lines.append("## Sources\n")
        for src in report.sources:
            lines.append(f"{src.n}. {src.title} — {src.url}")
        lines.append("")
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    """CLI: ``python -m agent.runner "your question"`` -> prints a cited report."""
    if len(argv) < 2 or not argv[1].strip():
        print('Usage: python -m agent.runner "your research question"', file=sys.stderr)
        return 2
    result = run(argv[1])
    print(render_report_markdown(result.report))
    print("\n---")
    print(
        f"run_id={result.run_id} status={result.status} iterations={result.iterations} "
        f"tool_calls={result.tool_calls} tokens={result.tokens} "
        f"usd=${result.usd:.4f} latency={result.latency_ms:.1f}ms "
        f"citation_coverage={result.citation_coverage:.0%}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv))
