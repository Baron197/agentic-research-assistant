"""Robustness against real-provider failure modes (malformed output, dead links).

The keyless fakes never misbehave, so these tests simulate the failure modes a
real LLM / real web introduce: repeated ids, invalid structured output, and
tool calls that raise mid-run.
"""

from __future__ import annotations

from agent.agents._common import parse_plan, parse_report, structured_call
from agent.agents.researcher import researcher
from agent.llm import LLMRequest, LLMResponse
from agent.runner import build_context
from agent.schemas import Budget, ResearchPlan, SubQuestion


def test_parse_plan_deduplicates_repeated_ids():
    # `researched_sqs` tracking is id-based; a repeated id would silently skip a
    # facet, so duplicates must be renamed (regression).
    content = {"sub_questions": [
        {"id": "SQ1", "question": "What is X?", "search_queries": ["x"]},
        {"id": "SQ1", "question": "What are the trade-offs of X?", "search_queries": ["x trade-offs"]},
    ]}
    plan = parse_plan("q", content)
    ids = [sq.id for sq in plan.sub_questions]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_parse_report_deduplicates_repeated_claim_ids():
    # The critic removes claims *by id*; duplicates would make it delete the
    # wrong claims too (regression).
    content = {"summary": "s", "sections": [{"heading": "h", "claims": [
        {"id": "C1", "text": "first claim", "evidence_ids": ["E1"]},
        {"id": "C1", "text": "second claim", "evidence_ids": ["E2"]},
    ]}]}
    report = parse_report("q", content)
    ids = [c.id for c in report.all_claims()]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_structured_call_charges_tokens_of_failed_attempts():
    # A failed attempt still consumed tokens; the budget must see all of them.
    class FlakyLLM:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            return LLMResponse(content={"ok": self.calls > 1}, tokens=10)

    def parse(content):
        if not content["ok"]:
            raise ValueError("malformed output")
        return content

    parsed, resp = structured_call(FlakyLLM(), LLMRequest(role="planner", payload={}), parse)
    assert parsed == {"ok": True}
    assert resp.tokens == 20  # 10 (failed attempt) + 10 (successful retry)


def test_researcher_survives_tool_failures(settings):
    # A dead link / provider outage must degrade to "no evidence for that
    # result", never abort the run (regression).
    class FailingFetch:
        name = "failing-fetch"

        def fetch(self, url: str) -> str:
            raise RuntimeError("boom: connection reset")

    ctx = build_context(settings)
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
    out = researcher(state, ctx)  # must not raise
    assert out["evidence"] == []
    assert out["tool_calls"] > 0  # search + failed fetch attempts were still traced
    assert any("fetch failed" in s.output_summary for s in out["trace"] if s.tool == "fetch")
