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


def test_parallel_researcher_survives_failing_fetch(settings):
    # A provider that raises must degrade gracefully even on the threaded path
    # (the failure is captured per task and replayed as a "fetch failed" step).
    from agent.agents.researcher import researcher
    from agent.runner import build_context
    from agent.schemas import Budget, ResearchPlan, SubQuestion

    class FailingFetch:
        name = "failing-fetch"

        def fetch(self, url: str) -> str:
            raise RuntimeError("boom: connection reset")

    ctx = build_context(settings.model_copy(update={"research_concurrency": 8}))
    ctx.fetch = FailingFetch()
    state = {
        "plan": ResearchPlan(question="q", sub_questions=[
            SubQuestion(id="SQ1", question="Core concepts: what is RAG?",
                        search_queries=["retrieval augmented generation"]),
        ]),
        "evidence": [],
        "budget": Budget(token_limit=60_000),
        "tool_calls": 0,
        "researched_sqs": [],
    }
    out = researcher(state, ctx)  # must not raise, even with a thread pool
    assert out["evidence"] == []
    assert out["tool_calls"] > 0
    assert any("fetch failed" in s.output_summary
               for s in out["trace"] if s.tool == "fetch")
