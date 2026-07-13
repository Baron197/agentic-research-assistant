"""Guardrails: the safety rails that make the system trustworthy.

The headline guarantee lives here: ``enforce_citations`` makes it *structurally
impossible* for the final report to cite a source that was not gathered. This is
applied unconditionally in the finalizer (independent of the critic), and is the
behaviour proven by the no-fabricated-sources test.

Also here: a validate-and-retry wrapper for structured LLM calls, budget
accounting helpers, an input-length cap, and the iteration cap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .schemas import Budget, Citation, Evidence, Report

T = TypeVar("T")


class GuardrailError(ValueError):
    """Raised when an input or model output violates a hard guardrail."""


class StructuredOutputError(GuardrailError):
    """Raised when the model's structured output fails validation after retries.

    Kept distinct from plain ``GuardrailError`` so the API can report it as a
    server-side (model) failure rather than a client input error.
    """


# --- Input limits -----------------------------------------------------------
def clamp_input(question: str, max_length: int) -> str:
    """Validate and normalise a user question, or raise ``GuardrailError``."""
    q = (question or "").strip()
    if not q:
        raise GuardrailError("question must not be empty")
    if len(q) > max_length:
        raise GuardrailError(f"question exceeds max length of {max_length} characters")
    return q


# --- Iteration cap ----------------------------------------------------------
def can_iterate(iteration: int, max_iterations: int) -> bool:
    """True if another revise loop is permitted under the iteration cap."""
    return iteration < max_iterations


# --- Budget accounting ------------------------------------------------------
def within_budget(budget: Budget) -> bool:
    """True while the run is still under its token budget."""
    return not budget.exceeded


# --- Structured-output validate & retry -------------------------------------
def validate_and_retry(call: Callable[[], object], parse: Callable[[object], T],
                       retries: int = 2) -> T:
    """Call a (possibly flaky real) LLM and parse/validate its output.

    Retries up to ``retries`` times if parsing/validation fails. The fake LLM is
    always valid so this passes first try; for a real model it absorbs the
    occasional malformed JSON response. Raises ``GuardrailError`` if every
    attempt fails.
    """
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            return parse(call())
        except Exception as exc:  # noqa: BLE001 - we re-raise as StructuredOutputError
            last_err = exc
    raise StructuredOutputError(f"structured output failed validation after retries: {last_err}")


# --- The no-fabricated-sources guarantee ------------------------------------
def enforce_citations(report: Report, evidence: list[Evidence]) -> tuple[Report, int]:
    """Strip any citation to an unknown evidence id; drop now-orphaned claims.

    Rules:
      * A citation to an id not present in ``evidence`` is removed.
      * A claim that *had* citations but loses them all (every id was invalid) is
        dropped entirely — it was relying on fabricated sources.
      * A claim that was uncited to begin with is left untouched (the critic, not
        this guardrail, is responsible for uncited claims).
      * Empty sections are removed.
      * Any pre-existing ``sources`` entry pointing at an ungathered id is also
        removed, so the guarantee holds for any caller (the finalizer rebuilds
        sources from scratch anyway via ``build_sources``).

    Returns the cleaned report and the number of dropped claims.
    """
    valid_ids = {e.id for e in evidence}
    dropped = 0
    new_sections = []
    for section in report.sections:
        kept_claims = []
        for claim in section.claims:
            if not claim.evidence_ids:
                kept_claims.append(claim)  # uncited: not this guardrail's job
                continue
            valid = [eid for eid in claim.evidence_ids if eid in valid_ids]
            if not valid:
                dropped += 1  # every cited id was fabricated -> drop the claim
                continue
            kept_claims.append(claim.model_copy(update={"evidence_ids": valid}))
        if kept_claims:
            new_sections.append(section.model_copy(update={"claims": kept_claims}))

    cleaned = report.model_copy(
        update={
            "sections": new_sections,
            "sources": [s for s in report.sources if s.evidence_id in valid_ids],
        }
    )
    return cleaned, dropped


def build_sources(report: Report, evidence: list[Evidence]) -> list[Citation]:
    """Build the numbered ``[n]`` source list from the claims' cited ids.

    Only evidence actually cited by a surviving claim appears, numbered in order
    of first appearance, so every ``[n]`` maps to a real gathered source.
    """
    by_id = {e.id: e for e in evidence}
    sources = []
    n = 0
    for eid in report.ordered_cited_evidence_ids():
        if eid not in by_id:
            continue
        n += 1
        ev = by_id[eid]
        sources.append(Citation(n=n, evidence_id=eid, title=ev.source_title, url=ev.source_url))
    return sources
