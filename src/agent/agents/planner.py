"""Planner node: decompose the question into focused sub-questions."""

from __future__ import annotations

from typing import Any

from ..context import AgentContext
from ..llm import ROLE_PLANNER, LLMRequest
from ._common import parse_plan, structured_call


def planner(state: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    """Produce a ``ResearchPlan`` (3-6 sub-questions) and record a trace step."""
    question = state["question"]
    with ctx.tracer.span("planner") as sp:
        plan, resp = structured_call(
            ctx.llm,
            LLMRequest(role=ROLE_PLANNER, payload={"question": question}),
            lambda content: parse_plan(question, content),
        )
        sp.tokens = resp.tokens
        sp.usd = ctx.tracer.cost(resp.tokens)
        sp.input_summary = question
        sp.output_summary = f"{len(plan.sub_questions)} sub-questions"

    budget = state["budget"].charge(resp.tokens, sp.usd)
    return {"plan": plan, "budget": budget, "trace": [sp.to_step()]}
