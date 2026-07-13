"""Researcher node: gather evidence with the search + fetch tools.

For each not-yet-attempted sub-question it searches, fetches each new result,
extracts the most relevant sentences as an evidence snippet, and appends an
``Evidence`` record with a stable id and real source metadata. It is budget-aware
(checks the token budget before every tool call), de-duplicates by URL so the same
source is never gathered twice, and records which sub-questions it has *attempted*
(``researched_sqs``) so a revise loop never re-runs a facet — even one that yielded
no evidence — which keeps the revise loop cheap.
"""

from __future__ import annotations

from typing import Any

from ..context import AgentContext
from ..schemas import Evidence
from ..textutil import approx_tokens, best_sentences, strip_markdown
from ._common import clean_hint

# Cap evidence gathered per sub-question so distinct facets pick up distinct
# sources (keeps sections coherent instead of one facet grabbing everything).
MAX_EVIDENCE_PER_SUBQ = 2


def researcher(state: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    plan = state["plan"]
    evidence: list[Evidence] = list(state["evidence"])
    budget = state["budget"]
    tool_calls = int(state.get("tool_calls", 0))
    researched = list(state.get("researched_sqs", []))
    steps = []

    seen_urls = {e.source_url for e in evidence}
    next_id = len(evidence) + 1

    for sq in plan.sub_questions:
        if sq.id in researched:
            continue  # already attempted (e.g. on a revise loop) — keep it cheap
        if budget.exceeded:
            break
        hint = clean_hint(sq.question)
        gathered_for_sq = 0
        for query in sq.search_queries:
            if budget.exceeded or gathered_for_sq >= MAX_EVIDENCE_PER_SUBQ:
                break
            with ctx.tracer.span("researcher", tool="search") as sp:
                sp.input_summary = query
                try:
                    results = ctx.search.search(query)
                except Exception as exc:  # a real provider can time out / 4xx mid-run
                    results = []
                    sp.output_summary = f"search failed: {exc}"[:200]
                else:
                    sp.output_summary = f"{len(results)} results"
                cost_tokens = approx_tokens(query)
                sp.tokens = cost_tokens
                sp.usd = ctx.tracer.cost(cost_tokens)
            budget = budget.charge(sp.tokens, sp.usd)
            tool_calls += 1
            steps.append(sp.to_step())

            for result in results:
                if budget.exceeded or gathered_for_sq >= MAX_EVIDENCE_PER_SUBQ:
                    break
                if result.url in seen_urls:
                    continue
                with ctx.tracer.span("researcher", tool="fetch") as fp:
                    fp.input_summary = result.url
                    try:
                        doc = ctx.fetch.fetch(result.url)
                    except Exception as exc:  # a dead link must not abort the run
                        doc = None
                        fp.tokens = 1
                        fp.output_summary = f"fetch failed: {exc}"[:200]
                    else:
                        snippet = " ".join(best_sentences(query, strip_markdown(doc), k=2))
                        snippet = " ".join(snippet.split()) or result.snippet
                        fp.tokens = approx_tokens(snippet)
                        fp.output_summary = snippet[:80]
                    fp.usd = ctx.tracer.cost(fp.tokens)
                budget = budget.charge(fp.tokens, fp.usd)
                tool_calls += 1
                steps.append(fp.to_step())
                if doc is None:
                    continue  # try the next search result instead

                evidence.append(
                    Evidence(
                        id=f"E{next_id}",
                        claim_hint=hint,
                        source_title=result.title,
                        source_url=result.url,
                        snippet=snippet,
                    )
                )
                next_id += 1
                gathered_for_sq += 1
                seen_urls.add(result.url)
        researched.append(sq.id)  # mark attempted (even if it yielded nothing)

    return {
        "evidence": evidence,
        "budget": budget,
        "tool_calls": tool_calls,
        "researched_sqs": researched,
        "trace": steps,
    }
