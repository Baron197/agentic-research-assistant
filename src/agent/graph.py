"""The LangGraph ``StateGraph`` wiring the agents into a verifying research loop.

Flow:

    START -> planner -> researcher -> writer -> critic
    critic --accept--> [approval?] -> finalizer -> END
    critic --revise & iteration<max & budget ok--> researcher
    any node --budget exceeded--> finalizer (status="partial")
    (once a draft exists, a required approval gate still runs first)

A budget guard runs on the edge out of every node, so exceeding the token budget
always routes cleanly to the finalizer instead of hanging. The finalizer applies
the unconditional ``enforce_citations`` guarantee and numbers the sources.
"""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import critic, planner, researcher, writer
from .context import AgentContext
from .guardrails import build_sources, enforce_citations, within_budget
from .schemas import Budget, Critique, Report, ResearchPlan, Step


def _append_steps(left: list[Step] | None, right: list[Step] | None) -> list[Step]:
    """Reducer for the trace channel: concatenate steps from every node."""
    return (left or []) + (right or [])


class GraphState(TypedDict, total=False):
    """Typed state threaded through the graph (see ``schemas`` for the models)."""

    question: str
    plan: ResearchPlan | None
    evidence: list[Any]
    draft: Report | None
    critique: Critique | None
    iteration: int
    budget: Budget
    status: str
    rejected: list[str]
    approved: bool
    tool_calls: int
    dropped_claims: int
    next_action: str
    unresolved: bool
    researched_sqs: list[str]
    trace: Annotated[list[Step], _append_steps]
    final: Report | None


def _approval(state: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    """Optional human-in-the-loop gate before finalizing.

    Keyless/default mode auto-approves. When ``require_approval`` is set, the
    injected ``approval_fn`` decides; a production deployment would swap this node
    for a LangGraph ``interrupt()`` checkpoint (noted in the roadmap).
    """
    if not ctx.settings.require_approval:
        return {}
    with ctx.tracer.span("approval") as sp:
        approved = ctx.approval_fn(state.get("draft")) if ctx.approval_fn else True
        sp.input_summary = "human approval requested"
        sp.output_summary = f"approved={approved}"
    return {"approved": approved, "trace": [sp.to_step()]}


def _finalizer(state: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    """Assemble the final report: enforce citations, number sources, set status."""
    question = state["question"]
    draft = state.get("draft")
    evidence = state["evidence"]
    approved = state.get("approved", True)
    budget_exceeded = state["budget"].exceeded
    unresolved = bool(state.get("unresolved", False))

    with ctx.tracer.span("finalizer") as sp:
        if draft is None:
            report = Report(
                question=question,
                summary="No report could be produced (stopped before drafting, "
                "likely due to the token budget).",
                partial=True,
            )
            dropped = 0
        else:
            cleaned, dropped = enforce_citations(draft, evidence)
            sources = build_sources(cleaned, evidence)
            report = cleaned.model_copy(update={"sources": sources})

        if not approved:
            status, partial = "awaiting_approval", True
        elif budget_exceeded or unresolved or draft is None:
            status, partial = "partial", True
        else:
            status, partial = "complete", False

        report = report.model_copy(update={"partial": partial})
        sp.input_summary = f"{len(report.all_claims())} claims, {len(evidence)} evidence"
        sp.output_summary = f"status={status}, dropped={dropped}, sources={len(report.sources)}"

    return {
        "final": report,
        "status": status,
        "dropped_claims": int(state.get("dropped_claims", 0)) + dropped,
        "trace": [sp.to_step()],
    }


def build_graph(ctx: AgentContext):
    """Compile the StateGraph with all nodes and conditional (budget-aware) edges."""
    settings = ctx.settings

    def over_budget(state: dict[str, Any]) -> bool:
        return not within_budget(state["budget"])

    def route_after_planner(state: dict[str, Any]) -> str:
        return "finalizer" if over_budget(state) else "researcher"

    def route_after_researcher(state: dict[str, Any]) -> str:
        return "finalizer" if over_budget(state) else "writer"

    def route_after_writer(state: dict[str, Any]) -> str:
        # Once a draft exists, a required human gate must not be bypassed —
        # even a budget-exhausted (partial) report goes through approval.
        if over_budget(state):
            return "approval" if settings.require_approval else "finalizer"
        if settings.enable_critic:
            return "critic"
        return "approval" if settings.require_approval else "finalizer"

    def route_after_critic(state: dict[str, Any]) -> str:
        if over_budget(state):
            return "approval" if settings.require_approval else "finalizer"
        if state.get("next_action") == "revise":
            return "researcher"
        return "approval" if settings.require_approval else "finalizer"

    g: StateGraph = StateGraph(GraphState)
    g.add_node("planner", partial(planner, ctx=ctx))
    g.add_node("researcher", partial(researcher, ctx=ctx))
    g.add_node("writer", partial(writer, ctx=ctx))
    g.add_node("critic", partial(critic, ctx=ctx))
    g.add_node("approval", partial(_approval, ctx=ctx))
    g.add_node("finalizer", partial(_finalizer, ctx=ctx))

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner", route_after_planner,
        {"researcher": "researcher", "finalizer": "finalizer"},
    )
    g.add_conditional_edges(
        "researcher", route_after_researcher,
        {"writer": "writer", "finalizer": "finalizer"},
    )
    g.add_conditional_edges(
        "writer", route_after_writer,
        {"critic": "critic", "approval": "approval", "finalizer": "finalizer"},
    )
    g.add_conditional_edges(
        "critic", route_after_critic,
        {"researcher": "researcher", "approval": "approval", "finalizer": "finalizer"},
    )
    g.add_edge("approval", "finalizer")
    g.add_edge("finalizer", END)
    return g.compile()
