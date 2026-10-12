"""End-to-end keyless pipeline behaviour."""

from __future__ import annotations

from agent.runner import run
from tests.conftest import QUESTION


def test_end_to_end_produces_cited_report(settings):
    result = run(QUESTION, settings=settings, persist=False)
    assert result.status == "complete"
    assert result.report.sections, "expected a non-empty report"
    claims = result.report.all_claims()
    assert claims, "expected at least one claim"
    # Every surviving claim is cited (critic removed the uncited synthesis claim).
    assert all(c.is_cited for c in claims)
    # Sources exist and are numbered 1..n.
    assert result.report.sources
    assert [c.n for c in result.report.sources] == list(range(1, len(result.report.sources) + 1))


def test_runs_are_deterministic(settings):
    a = run(QUESTION, settings=settings, persist=False)
    b = run(QUESTION, settings=settings, persist=False)
    assert a.report.model_dump() == b.report.model_dump()


def test_run_persists_trace(settings):
    result = run("What is reranking?", settings=settings, persist=True)
    run_file = settings.traces_dir / f"{result.run_id}.json"
    index = settings.traces_dir / "index.jsonl"
    assert run_file.exists() and index.exists()
