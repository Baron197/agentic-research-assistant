"""Typed data contracts shared across the whole pipeline.

Everything that crosses an agent/tool/API boundary is a small pydantic model so
that (a) the structure is validated, (b) it serialises cleanly to JSON for the
API and the persisted trace, and (c) the fake and real providers are forced to
produce the *same* shapes. Keeping these in one module makes the data flow easy
to read top-to-bottom.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["accept", "revise"]
RunStatus = Literal["complete", "partial", "awaiting_approval", "error"]


def _utcnow_iso() -> str:
    """3.11-compatible UTC timestamp (we avoid ``datetime.UTC`` for 3.11)."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
class SubQuestion(BaseModel):
    """One focused facet of the research question plus its search queries."""

    id: str
    question: str
    search_queries: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """The planner's decomposition of the user question (3-6 sub-questions)."""

    question: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evidence (gathered by tools)
# ---------------------------------------------------------------------------
class Evidence(BaseModel):
    """A single piece of gathered evidence with stable id and real source.

    ``id`` is the anchor for the no-fabricated-sources guarantee: the writer may
    only cite ids that exist here, and the finalizer maps each cited id to a
    numbered source.
    """

    id: str
    claim_hint: str
    source_title: str
    source_url: str
    snippet: str


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
class Claim(BaseModel):
    """A single assertion in the report and the evidence ids that back it."""

    id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)

    @property
    def is_cited(self) -> bool:
        return len(self.evidence_ids) > 0


class ReportSection(BaseModel):
    """A titled group of claims (usually one per sub-question)."""

    heading: str
    claims: list[Claim] = Field(default_factory=list)


class Citation(BaseModel):
    """A numbered source entry: ``[n]`` -> the gathered evidence behind it."""

    n: int
    evidence_id: str
    title: str
    url: str


class Report(BaseModel):
    """The structured, cited report returned to the caller."""

    question: str
    summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)
    partial: bool = False

    def all_claims(self) -> list[Claim]:
        return [c for s in self.sections for c in s.claims]

    def cited_evidence_ids(self) -> set[str]:
        return {eid for c in self.all_claims() for eid in c.evidence_ids}

    def ordered_cited_evidence_ids(self) -> list[str]:
        """Cited evidence ids in first-appearance order across the report."""
        seen: list[str] = []
        for claim in self.all_claims():
            for eid in claim.evidence_ids:
                if eid not in seen:
                    seen.append(eid)
        return seen


# ---------------------------------------------------------------------------
# Critique
# ---------------------------------------------------------------------------
class Critique(BaseModel):
    """The critic's verdict over the current draft.

    ``supported`` / ``unsupported`` hold *claim ids*. ``verdict == "revise"``
    means at least one claim was unsupported and another iteration is warranted.
    """

    supported: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    verdict: Verdict = "accept"
    notes: str = ""


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------
class Budget(BaseModel):
    """Token/cost budget tracked across the whole run."""

    token_limit: int = 60_000
    tokens_used: int = 0
    usd_used: float = 0.0

    @property
    def remaining(self) -> int:
        return self.token_limit - self.tokens_used

    @property
    def exceeded(self) -> bool:
        return self.tokens_used >= self.token_limit

    def charge(self, tokens: int, usd: float) -> Budget:
        """Return a new Budget with usage added (immutable-style update)."""
        return self.model_copy(
            update={
                "tokens_used": self.tokens_used + tokens,
                "usd_used": round(self.usd_used + usd, 6),
            }
        )


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
class Step(BaseModel):
    """One ordered entry in a run's execution trace.

    Shaped like a span you would forward to Langfuse / OpenTelemetry so the
    observability layer can be swapped without touching the agents.
    """

    node: str
    tool: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    tokens: int = 0
    usd: float = 0.0
    ms: float = 0.0


class RunResult(BaseModel):
    """The full result of a single run (returned by the runner, served by API)."""

    run_id: str
    question: str
    status: RunStatus
    report: Report
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0
    dropped_claims: int = 0
    trace: list[Step] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)

    @property
    def citation_coverage(self) -> float:
        """Fraction of claims carrying at least one citation (keyless metric)."""
        claims = self.report.all_claims()
        if not claims:
            return 0.0
        cited = sum(1 for c in claims if c.is_cited)
        return cited / len(claims)
