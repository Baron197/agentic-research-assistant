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

The approval gate either asks an injected ``approval_fn`` or pauses the run with
LangGraph's ``interrupt()``; a paused run's state waits in an in-memory
checkpoint until ``runner.resume`` brings a person's decision.
"""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents import critic, planner, researcher, writer
from .context import AgentContext
from .guardrails import build_sources, enforce_citations, within_budget
from .schemas import (
    Budget,
    Citation,
    Claim,
    Critique,
    Evidence,
    Report,
    ReportSection,
    ResearchPlan,
    Step,
    SubQuestion,
)


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

    With ``require_approval`` set, an injected ``approval_fn`` decides at once;
    without one the run pauses here (``interrupt()``) and LangGraph re-runs this
    node with the decision when ``runner.resume`` continues it.
    """
    if not ctx.settings.require_approval:
        return {}
    with ctx.tracer.span("approval") as sp:
        if ctx.approval_fn is not None:
            approved = bool(ctx.approval_fn(state.get("draft")))
        else:
            approved = bool(interrupt("approve or reject the draft"))
        sp.input_summary = "human approval requested"
        sp.output_summary = f"approved={approved}"
    return {"approved": approved, "trace": [sp.to_step()]}


def assemble_report(question: str, draft: Report | None,
                    evidence: list[Evidence]) -> tuple[Report, int]:
    """The report the finalizer releases: citations enforced, sources numbered.

    Also shows a paused run's draft exactly as approving it would release it.
    Returns the report and how many claims ``enforce_citations`` dropped.
    """
    if draft is None:
        return Report(
            question=question,
            summary="No report could be produced (stopped before drafting, "
            "likely due to the token budget).",
            partial=True,
        ), 0
    cleaned, dropped = enforce_citations(draft, evidence)
    return cleaned.model_copy(update={"sources": build_sources(cleaned, evidence)}), dropped


def _finalizer(state: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    """Assemble the final report: enforce citations, number sources, set status."""
    draft = state.get("draft")
    evidence = state["evidence"]
    approved = state.get("approved", True)
    budget_exceeded = state["budget"].exceeded
    unresolved = bool(state.get("unresolved", False))

    with ctx.tracer.span("finalizer") as sp:
        report, dropped = assemble_report(state["question"], draft, evidence)

        if not approved:
            status, partial = "rejected", True
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


# Every schema type the graph state holds. A checkpoint stores the state as
# msgpack, and LangGraph only rebuilds the classes it is told to trust.
_STATE_TYPES = [(cls.__module__, cls.__name__) for cls in (
    Budget, Citation, Claim, Critique, Evidence, Report, ReportSection,
    ResearchPlan, Step, SubQuestion)]


def pause_checkpointer() -> InMemorySaver:
    """Where a run paused for approval keeps its state until the decision comes."""
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES)
    except TypeError:  # langgraph-checkpoint before the allowlist trusts every type
        serde = JsonPlusSerializer()
    return InMemorySaver(serde=serde)


def build_graph(ctx: AgentContext, checkpointer: InMemorySaver | None = None):
    """Compile the StateGraph with all nodes and conditional (budget-aware) edges.

    ``checkpointer`` is needed only by a run that can pause for approval.
    """
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

    def node(name: str, fn: Any):
        """The node, reporting its start and end to the progress bar if one listens.

        Checked on every call, not once here: a run resumed after a pause has no
        progress bar any more, though its graph was built with one.
        """
        bound = partial(fn, ctx=ctx)

        def reporting(state: dict[str, Any]) -> dict[str, Any]:
            progress = ctx.progress
            if progress is None:
                return bound(state)
            progress.node_started(name, state)
            update = bound(state)
            progress.node_finished(name, state, update)
            return update

        return reporting

    g: StateGraph = StateGraph(GraphState)
    g.add_node("planner", node("planner", planner))
    g.add_node("researcher", node("researcher", researcher))
    g.add_node("writer", node("writer", writer))
    g.add_node("critic", node("critic", critic))
    g.add_node("approval", node("approval", _approval))
    g.add_node("finalizer", node("finalizer", _finalizer))

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
    return g.compile(checkpointer=checkpointer)
