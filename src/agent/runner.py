"""The single entry point used by the API, UI, eval, and CLI.

``run(question, ...)`` builds the provider context from settings, executes the
compiled graph, persists the trace, and returns a ``RunResult``. Everything is
keyless by default; pass ``settings=`` or keyword overrides to change behaviour
(e.g. ``token_budget=10`` or ``enable_critic=False`` for the A/B).
``resume(run_id, approved)`` finishes a run that paused for a person's approval.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from time import perf_counter
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from . import approvals
from .config import Settings, get_settings
from .context import AgentContext
from .graph import assemble_report, build_graph, pause_checkpointer
from .guardrails import clamp_input
from .llm import get_llm
from .observability import Tracer, persist_run
from .progress import Progress, ProgressFn
from .schemas import Budget, Report, RunResult, Step
from .tools.embed import get_embedder
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
        embedder=get_embedder(settings),
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
    progress: ProgressFn | None = None,
    **overrides: Any,
) -> RunResult:
    """Run the full research pipeline and return a structured ``RunResult``.

    ``progress(fraction, label)``, if given, is called as the run moves through
    its stages (see ``agent.progress``); it only observes and never changes the result.

    With ``require_approval`` set, ``approval_fn(draft)`` decides at once if given.
    Without one the run pauses before the finalizer and returns status
    ``awaiting_approval``, with the draft as approving it would release it;
    ``resume(run_id, approved)`` then finishes it.
    """
    settings = _resolve_settings(settings, overrides)
    question = clamp_input(question, settings.max_question_length)

    ctx = build_context(settings, approval_fn=approval_fn)
    if progress is not None:
        ctx.progress = Progress(progress, max_iterations=settings.max_iterations,
                                web=settings.search_provider == "web")
    can_pause = settings.require_approval and approval_fn is None
    graph = build_graph(ctx, checkpointer=pause_checkpointer() if can_pause else None)
    run_id = uuid4().hex[:12]
    config = {"recursion_limit": _recursion_limit(settings.max_iterations),
              "configurable": {"thread_id": run_id}}

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
    final_state = graph.invoke(initial, config=config)
    latency_ms = (perf_counter() - start) * 1000.0

    paused = can_pause and bool(graph.get_state(config).next)
    if paused:
        result = _paused_result(run_id, question, final_state, latency_ms, settings)
    else:
        result = _result(run_id, question, final_state, latency_ms, settings)

    if persist:
        persist_run(result, settings.traces_dir)
    if paused:
        approvals.put(run_id, approvals.PausedRun(
            graph=graph, config=config, ctx=ctx, question=question,
            created_at=result.created_at, latency_ms=latency_ms, persist=persist))
    elif ctx.progress is not None:
        ctx.progress.done()
    return result


def resume(run_id: str, approved: bool) -> RunResult:
    """Finish a run paused at the approval gate with a person's decision.

    Approving releases the report (``complete``, or ``partial`` if the run hit a
    limit); rejecting ends the run as ``rejected``. The run keeps its id, and its
    persisted record is replaced. Raises ``approvals.NotAwaitingApproval`` when no
    run with this id is paused: unknown, already decided, or forgotten.
    """
    paused = approvals.take(run_id)
    paused.ctx.progress = None  # whatever watched the first part has moved on
    start = perf_counter()
    final_state = paused.graph.invoke(Command(resume=bool(approved)), config=paused.config)
    latency_ms = paused.latency_ms + (perf_counter() - start) * 1000.0

    result = _result(run_id, paused.question, final_state, latency_ms,
                     paused.ctx.settings).model_copy(update={"created_at": paused.created_at})
    if paused.persist:
        persist_run(result, paused.ctx.settings.traces_dir)
    return result


def _result(run_id: str, question: str, state: dict[str, Any], latency_ms: float,
            settings: Settings) -> RunResult:
    report = state.get("final") or Report(question=question, partial=True)
    budget = state["budget"]
    return RunResult(
        run_id=run_id,
        question=question,
        status=state.get("status", "complete"),
        report=report,
        iterations=int(state.get("iteration", 0)),
        tool_calls=int(state.get("tool_calls", 0)),
        tokens=budget.tokens_used,
        usd=budget.usd_used,
        latency_ms=round(latency_ms, 3),
        dropped_claims=int(state.get("dropped_claims", 0)),
        removed_claims=list(state.get("rejected", [])),
        trace=state.get("trace", []),
        evidence=state.get("evidence", []),
        keyless=settings.is_keyless,
    )


def _paused_result(run_id: str, question: str, state: dict[str, Any],
                   latency_ms: float, settings: Settings) -> RunResult:
    """A run waiting at the approval gate: its draft, as approving would release it."""
    report, dropped = assemble_report(question, state.get("draft"), state["evidence"])
    waiting = Step(node="approval", input_summary="human approval requested",
                   output_summary="paused: waiting for a person's decision")
    return _result(run_id, question, {
        **state,
        "final": report.model_copy(update={"partial": True}),
        "status": "awaiting_approval",
        "dropped_claims": int(state.get("dropped_claims", 0)) + dropped,
        "trace": [*state.get("trace", []), waiting],
    }, latency_ms, settings)


def render_report_markdown(report: Report) -> str:
    """Render a ``Report`` as Markdown with inline ``[n]`` citations + sources."""
    n_by_evidence = {c.evidence_id: c.n for c in report.sources}
    lines: list[str] = ["# Research Report\n", f"**Question:** {report.question}\n"]
    if report.partial:
        lines.append("> ⚠️ This report is **partial**: the run stopped at a budget or "
                     "iteration limit, or a person has not approved it.\n")
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
    if result.status == "awaiting_approval":  # REQUIRE_APPROVAL=true: you are the reviewer
        try:
            answer = input("\nApprove this report? [y/N] ")
        except EOFError:  # nobody to ask (input is piped): not approved
            answer = ""
        result = resume(result.run_id, answer.strip().lower() in ("y", "yes"))
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
