"""Deeper research: parallel fan-out preserves output; the depth knob scales it.

The researcher runs its searches/fetches concurrently but replays the decision
logic sequentially, so the output must be identical to a one-at-a-time run while
``evidence_per_subquestion`` controls how thorough the report is.
"""

from __future__ import annotations

from agent.runner import run
from tests.conftest import QUESTION


def test_parallel_fanout_matches_serial(settings):
    # Concurrency is a latency win only: a parallel run must be byte-for-byte
    # identical to a one-at-a-time (concurrency=1) run.
    serial = run(QUESTION, settings=settings.model_copy(
        update={"research_concurrency": 1}), persist=False)
    parallel = run(QUESTION, settings=settings.model_copy(
        update={"research_concurrency": 8}), persist=False)

    assert serial.report.model_dump() == parallel.report.model_dump()
    assert [e.model_dump() for e in serial.evidence] == \
           [e.model_dump() for e in parallel.evidence]
    assert serial.tool_calls == parallel.tool_calls


def test_depth_setting_gathers_more_sources(settings):
    # Raising evidence_per_subquestion yields a more thorough report — more
    # evidence and more claims — while staying fully cited.
    shallow = run(QUESTION, settings=settings.model_copy(
        update={"evidence_per_subquestion": 1}), persist=False)
    deep = run(QUESTION, settings=settings.model_copy(
        update={"evidence_per_subquestion": 4, "top_search_results": 12}), persist=False)

    assert len(deep.evidence) > len(shallow.evidence)
    assert len(deep.report.all_claims()) > len(shallow.report.all_claims())
    assert deep.status == "complete"
    assert all(c.is_cited for c in deep.report.all_claims())
