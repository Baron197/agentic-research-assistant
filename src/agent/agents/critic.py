"""Critic node: verify each claim against its cited evidence.

Produces a ``Critique`` (supported / unsupported claim ids + verdict). When claims
are unsupported it removes them from the draft and records their text in
``rejected`` so the writer will not re-emit them. It then decides — using the
iteration cap and the remaining budget — whether another revise loop is both
*warranted* and *permitted* (``next_action``), and flags ``unresolved`` when a
revise was needed but no loop was possible (cap/budget reached). This keeps the
loop decision in one place and makes the final status precise.
"""

from __future__ import annotations

from typing import Any

from ..context import AgentContext
from ..guardrails import can_iterate, within_budget
from ..llm import ROLE_CRITIC, LLMRequest
from ._common import parse_critique, structured_call


def critic(state: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    draft = state["draft"]
    evidence = state["evidence"]

    payload = {
        "draft": draft.model_dump(),
        "evidence": [e.model_dump() for e in evidence],
    }
    with ctx.tracer.span("critic") as sp:
        critique, resp = structured_call(
            ctx.llm,
            LLMRequest(role=ROLE_CRITIC, payload=payload),
            parse_critique,
        )
        sp.tokens = resp.tokens
        sp.usd = ctx.tracer.cost(resp.tokens)
        sp.input_summary = f"{len(draft.all_claims())} claims"
        sp.output_summary = f"verdict={critique.verdict}, unsupported={len(critique.unsupported)}"

    budget = state["budget"].charge(resp.tokens, sp.usd)
    iteration = int(state.get("iteration", 0))
    rejected = list(state.get("rejected", []))
    new_draft = draft

    if critique.unsupported:
        unsupported = set(critique.unsupported)
        new_sections = []
        for section in draft.sections:
            kept = []
            for claim in section.claims:
                if claim.id in unsupported:
                    rejected.append(claim.text)  # remember it so we don't re-add it
                else:
                    kept.append(claim)
            if kept:
                new_sections.append(section.model_copy(update={"claims": kept}))
        new_draft = draft.model_copy(update={"sections": new_sections})

    # A revise loop is taken only if it is both warranted (something was
    # unsupported) and permitted (budget left AND under the iteration cap).
    revise_needed = bool(critique.unsupported)
    can_loop = (
        revise_needed
        and within_budget(budget)
        and can_iterate(iteration, ctx.settings.max_iterations)
    )
    return {
        "critique": critique,
        "draft": new_draft,
        "rejected": rejected,
        "iteration": iteration + (1 if can_loop else 0),
        "next_action": "revise" if can_loop else "finalize",
        "unresolved": revise_needed and not can_loop,
        "budget": budget,
        "trace": [sp.to_step()],
    }
