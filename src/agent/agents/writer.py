"""Writer node: synthesise gathered evidence into a structured, cited draft.

The writer may only cite evidence ids that exist in state; the ``rejected`` list
(claims the critic previously removed) is passed through so the revise loop
converges instead of re-emitting the same unsupported claim.
"""

from __future__ import annotations

from typing import Any

from ..context import AgentContext
from ..llm import ROLE_WRITER, LLMRequest
from ._common import parse_report, structured_call


def writer(state: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    question = state["question"]
    evidence = state["evidence"]
    rejected = list(state.get("rejected", []))

    payload = {
        "question": question,
        "evidence": [e.model_dump() for e in evidence],
        "rejected": rejected,
    }
    with ctx.tracer.span("writer") as sp:
        draft, resp = structured_call(
            ctx.llm,
            LLMRequest(role=ROLE_WRITER, payload=payload),
            lambda content: parse_report(question, content),
        )
        sp.tokens = resp.tokens
        sp.usd = ctx.tracer.cost(resp.tokens)
        sp.input_summary = f"{len(evidence)} evidence items"
        sp.output_summary = f"{len(draft.all_claims())} claims in {len(draft.sections)} sections"

    budget = state["budget"].charge(resp.tokens, sp.usd)
    return {"draft": draft, "budget": budget, "trace": [sp.to_step()]}
