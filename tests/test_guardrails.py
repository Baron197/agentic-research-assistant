"""Guardrails, including the headline no-fabricated-sources guarantee."""

from __future__ import annotations

import pytest

from agent.guardrails import GuardrailError, build_sources, clamp_input, enforce_citations
from agent.runner import run
from agent.schemas import Citation, Claim, Evidence, Report, ReportSection
from tests.conftest import QUESTION


def _evidence():
    return [Evidence(id="E1", claim_hint="h", source_title="Doc One",
                     source_url="local://one.md", snippet="alpha beta gamma")]


def test_enforce_citations_drops_fabricated_source():
    evidence = _evidence()
    report = Report(
        question="q",
        sections=[ReportSection(heading="s", claims=[
            Claim(id="C1", text="valid claim", evidence_ids=["E1"]),
            Claim(id="C2", text="fabricated claim", evidence_ids=["E999"]),  # not gathered
        ])],
    )
    cleaned, dropped = enforce_citations(report, evidence)
    assert dropped == 1
    surviving = cleaned.cited_evidence_ids()
    # The crucial invariant: no surviving citation id is absent from evidence.
    assert surviving <= {e.id for e in evidence}
    assert "E999" not in surviving


def test_enforce_citations_keeps_uncited_claims():
    # Uncited claims are the critic's responsibility, not this guardrail's.
    report = Report(question="q", sections=[ReportSection(heading="s", claims=[
        Claim(id="C1", text="uncited synthesis", evidence_ids=[]),
    ])])
    cleaned, dropped = enforce_citations(report, _evidence())
    assert dropped == 0
    assert len(cleaned.all_claims()) == 1


def test_build_sources_only_references_gathered_evidence():
    evidence = _evidence()
    report = Report(question="q", sections=[ReportSection(heading="s", claims=[
        Claim(id="C1", text="x", evidence_ids=["E1"]),
    ])])
    sources = build_sources(report, evidence)
    assert [s.evidence_id for s in sources] == ["E1"]
    assert sources[0].url == "local://one.md"


def test_enforce_citations_strips_fabricated_sources_entries():
    # The sources list itself must also honour the guarantee, for any caller
    # that passes a report with pre-populated sources (regression).
    evidence = _evidence()
    report = Report(
        question="q",
        sections=[ReportSection(heading="s", claims=[
            Claim(id="C1", text="x", evidence_ids=["E1"]),
        ])],
        sources=[Citation(n=1, evidence_id="E1", title="t", url="u"),
                 Citation(n=2, evidence_id="E999", title="fabricated", url="u2")],
    )
    cleaned, _ = enforce_citations(report, evidence)
    assert [s.evidence_id for s in cleaned.sources] == ["E1"]


def test_full_run_never_cites_ungathered_source(settings):
    result = run(QUESTION, settings=settings, persist=False)
    gathered = {e.id for e in result.evidence}
    cited = result.report.cited_evidence_ids()
    # Guard against a vacuous pass: this in-corpus question must actually
    # produce cited claims before the subset invariant means anything.
    assert result.report.all_claims(), "expected a non-empty report"
    assert cited, "expected at least one citation"
    assert cited <= gathered


def test_clamp_input_rejects_empty_and_too_long():
    with pytest.raises(GuardrailError):
        clamp_input("   ", 100)
    with pytest.raises(GuardrailError):
        clamp_input("x" * 101, 100)
    assert clamp_input("  hello  ", 100) == "hello"


def test_build_sources_numbers_by_first_appearance_not_lexicographic():
    # Regression: with >=10 evidence items the old code sorted ids lexicographically
    # (E1, E10, E2, ...). Sources must follow first-appearance order: E1..E12.
    evidence = [Evidence(id=f"E{i}", claim_hint="h", source_title=f"D{i}",
                         source_url=f"local://{i}.md", snippet="x") for i in range(1, 13)]
    claims = [Claim(id=f"C{i}", text="t", evidence_ids=[f"E{i}"]) for i in range(1, 13)]
    report = Report(question="q", sections=[ReportSection(heading="s", claims=claims)])
    sources = build_sources(report, evidence)
    assert [(s.n, s.evidence_id) for s in sources] == [(i, f"E{i}") for i in range(1, 13)]
