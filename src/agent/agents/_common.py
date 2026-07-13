"""Shared helpers for the agent nodes (parsers + validate/retry call)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ..guardrails import validate_and_retry
from ..llm import LLM, LLMRequest, LLMResponse
from ..schemas import Claim, Critique, Report, ReportSection, ResearchPlan, SubQuestion

T = TypeVar("T")


def structured_call(
    llm: LLM,
    request: LLMRequest,
    parser: Callable[[dict[str, Any]], T],
    retries: int = 2,
) -> tuple[T, LLMResponse]:
    """Call the LLM and parse/validate its output via the validate-and-retry guard.

    Delegates the retry/validation logic to ``guardrails.validate_and_retry`` (the
    one place that owns it) while capturing the raw response so the caller can
    still read token usage. The fake LLM is deterministic and always valid (passes
    first try); the retry budget exists for the real model's occasional bad JSON.
    Tokens from *failed* attempts were still spent, so they are summed into the
    returned response — the budget must see everything the run actually consumed.
    """
    responses: list[LLMResponse] = []

    def call() -> dict[str, Any]:
        resp = llm.generate(request)
        responses.append(resp)
        return resp.content

    parsed = validate_and_retry(call, parser, retries=retries)
    last = responses[-1]
    if len(responses) > 1:
        last = LLMResponse(content=last.content, tokens=sum(r.tokens for r in responses))
    return parsed, last


def clean_hint(sub_question_text: str) -> str:
    """Turn ``"Core concepts: <question>"`` into a clean section heading."""
    return sub_question_text.split(":", 1)[0].strip()


def parse_plan(question: str, content: dict[str, Any]) -> ResearchPlan:
    subs = []
    seen_ids: set[str] = set()
    for i, sq in enumerate(content.get("sub_questions", []), start=1):
        text = str(sq.get("question", "")).strip()
        if not text:
            continue
        queries = [str(q) for q in sq.get("search_queries", []) if str(q).strip()]
        # A real LLM may repeat ids; `researched_sqs` tracking is id-based, so a
        # duplicate would silently skip a facet. Suffix duplicates until unique.
        sq_id, n = str(sq.get("id") or f"SQ{i}"), 2
        base = sq_id
        while sq_id in seen_ids:
            sq_id = f"{base}-{n}"
            n += 1
        seen_ids.add(sq_id)
        subs.append(
            SubQuestion(
                id=sq_id,
                question=text,
                search_queries=queries or [text],  # fall back to the sub-question itself
            )
        )
    if not subs:
        raise ValueError("planner returned no usable sub-questions")
    return ResearchPlan(question=question, sub_questions=subs)


def parse_report(question: str, content: dict[str, Any]) -> Report:
    sections = []
    counter = 0
    seen_ids: set[str] = set()
    for s in content.get("sections", []):
        claims = []
        for c in s.get("claims", []):
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            counter += 1
            # The critic removes claims *by id*; a real LLM repeating an id would
            # make it delete the wrong claims too. Suffix duplicates until unique.
            claim_id, n = str(c.get("id") or f"C{counter}"), 2
            base = claim_id
            while claim_id in seen_ids:
                claim_id = f"{base}-{n}"
                n += 1
            seen_ids.add(claim_id)
            claims.append(
                Claim(
                    id=claim_id,
                    text=text,
                    evidence_ids=[str(e) for e in c.get("evidence_ids", []) if str(e).strip()],
                )
            )
        sections.append(ReportSection(heading=str(s.get("heading", "Findings")), claims=claims))
    return Report(question=question, summary=str(content.get("summary", "")), sections=sections)


def parse_critique(content: dict[str, Any]) -> Critique:
    verdict = content.get("verdict", "accept")
    if verdict not in ("accept", "revise"):
        raise ValueError(f"invalid verdict {verdict!r}")
    return Critique(
        supported=[str(x) for x in content.get("supported", [])],
        unsupported=[str(x) for x in content.get("unsupported", [])],
        verdict=verdict,
        notes=str(content.get("notes", "")),
    )
