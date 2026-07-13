"""Evaluation metric functions and the critic A/B delta."""

from __future__ import annotations

from pathlib import Path

from agent.config import Settings
from agent.schemas import Citation, Claim, Evidence, Report, ReportSection, RunResult
from eval.run_eval import (
    citation_coverage,
    evaluate_compare,
    load_tasks,
    point_coverage,
    source_validity,
    support_rate,
)


def _result_with(claim_evidence_ids, evidence_ids):
    evidence = [Evidence(id=e, claim_hint="h", source_title="t",
                         source_url=f"local://{e}.md", snippet="alpha beta gamma delta")
                for e in evidence_ids]
    report = Report(question="q", summary="alpha beta",
                    sections=[ReportSection(heading="s", claims=[
                        Claim(id="C1", text="alpha beta gamma", evidence_ids=claim_evidence_ids)])],
                    sources=[Citation(n=1, evidence_id=evidence_ids[0], title="t",
                                      url="local://x.md")] if evidence_ids else [])
    return RunResult(run_id="x", question="q", status="complete", report=report,
                     evidence=evidence)


def test_metric_functions():
    r = _result_with(["E1"], ["E1"])
    assert citation_coverage(r.report) == 1.0
    assert source_validity(r) == 1.0
    assert support_rate(r) == 1.0
    assert point_coverage(r.report, ["alpha beta"]) == 1.0
    assert point_coverage(r.report, ["zzz qqq"]) == 0.0


def test_source_validity_catches_fabrication():
    r = _result_with(["E999"], ["E1"])  # claim cites an ungathered id
    assert source_validity(r) == 0.0


def test_compare_shows_positive_delta():
    # Anchor at the repo root so the test passes regardless of pytest's CWD.
    tasks_path = Path(__file__).resolve().parents[1] / "eval" / "tasks.jsonl"
    tasks = load_tasks(tasks_path)[:4]
    report = evaluate_compare(tasks, Settings())
    agg = report["aggregate"]
    assert agg["coverage_on"] >= agg["coverage_off"]
    assert agg["coverage_delta"] > 0
