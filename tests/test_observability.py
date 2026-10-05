"""Costing and run aggregation (p95 nearest-rank, torn-line resilience)."""

from __future__ import annotations

import sys
import types

import pytest

from agent.llm import LLMRequest, LLMResponse, OpenAILLM
from agent.observability import Tracer, aggregate, cost_usd
from tests.conftest import QUESTION


def test_cost_usd_prices_input_and_output_separately():
    # List prices per 1M tokens — gpt-4o-mini: $0.15 input, $0.075 cached, $0.60 output.
    assert cost_usd("gpt-4o-mini", 1_000_000) == 0.15
    assert cost_usd("gpt-4o-mini", 0, 1_000_000) == 0.60
    assert cost_usd("gpt-4o-mini", 1_000_000, cached_input_tokens=1_000_000) == 0.075
    assert cost_usd("gpt-4o", 1_000_000, 1_000_000) == 12.5
    assert cost_usd("fake-llm", 999_999, 999_999) == 0.0  # keyless is free


def test_tracer_prices_a_response_by_its_split():
    tracer = Tracer(model="gpt-4o-mini")
    split = LLMResponse(content={}, tokens=1_000, input_tokens=800, output_tokens=200)
    assert tracer.cost(split) == cost_usd("gpt-4o-mini", 800, 200) == 0.00024
    # No split reported: priced as all input, never as the pricier output.
    assert tracer.cost(LLMResponse(content={}, tokens=1_000)) == 0.00015


def test_only_llm_calls_are_priced(settings, monkeypatch):
    # Price the fake LLM's estimated usage as if it were gpt-4o-mini.
    from agent import runner

    monkeypatch.setattr(runner, "_tracer_model", lambda _settings: "gpt-4o-mini")
    result = runner.run(QUESTION, settings=settings, persist=False)

    tool_steps = [s for s in result.trace if s.tool]
    llm_steps = [s for s in result.trace if s.node in ("planner", "writer", "critic")]
    assert tool_steps and all(s.usd == 0.0 and s.tokens > 0 for s in tool_steps)
    assert llm_steps and all(s.usd > 0.0 for s in llm_steps)
    assert result.usd == pytest.approx(sum(s.usd for s in llm_steps))


def test_openai_llm_reads_the_providers_usage_split(monkeypatch):
    # A stand-in ``openai`` module, so the test needs neither the SDK nor a key.
    usage = types.SimpleNamespace(
        prompt_tokens=900, completion_tokens=100, total_tokens=1_000,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=400),
    )
    message = types.SimpleNamespace(content='{"sub_questions": []}')
    completion = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                       usage=usage)

    class StubClient:
        def __init__(self, api_key: str) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **_: completion))

    stub = types.ModuleType("openai")
    stub.OpenAI = StubClient
    monkeypatch.setitem(sys.modules, "openai", stub)

    resp = OpenAILLM(api_key="test", model="gpt-4o-mini").generate(
        LLMRequest(role="planner", payload={"question": "q"}))
    assert (resp.tokens, resp.input_tokens, resp.output_tokens, resp.cached_input_tokens) == (
        1_000, 900, 100, 400)


def test_aggregate_handles_torn_line_and_p95(tmp_path):
    index = tmp_path / "index.jsonl"
    good = [
        '{"usd": 0.0, "latency_ms": 10, "n_steps": 5, "citation_coverage": 1.0}',
        '{"usd": 0.0, "latency_ms": 20, "n_steps": 7, "citation_coverage": 1.0}',
        '{"usd": 0.0, "latency_ms": 30, "n_steps": 9, "citation_coverage": 0.5}',
    ]
    torn = '{"usd": 0.0, "latency_ms": 40, "n_steps":'  # truncated -> must be skipped
    index.write_text("\n".join(good + [torn]) + "\n", encoding="utf-8")

    agg = aggregate(tmp_path)
    assert agg["runs"] == 3  # torn line skipped
    # nearest-rank p95 of [10,20,30] -> ceil(0.95*3)=3 -> index 2 -> 30
    assert agg["p95_latency_ms"] == 30
    assert agg["avg_steps"] == 7.0
    assert agg["avg_citation_coverage"] == 0.8333


def test_load_run_rejects_traversal_and_corrupt_files(tmp_path):
    # Path-separator ids and corrupt/foreign files read as "not found", never as
    # a filesystem escape or an exception into the API handler (regression).
    from agent.observability import load_run

    (tmp_path / "corrupt.json").write_text("{not valid json", encoding="utf-8")
    assert load_run("..\\corrupt", tmp_path) is None
    assert load_run("../corrupt", tmp_path) is None
    assert load_run("", tmp_path) is None
    assert load_run("corrupt", tmp_path) is None  # exists, but not a RunResult


def test_aggregate_empty(tmp_path):
    # Empty data returns a STABLE full-shape dict (zeros), so /metrics never
    # changes its keys between empty and non-empty states.
    agg = aggregate(tmp_path)
    assert agg["runs"] == 0
    assert agg["p95_latency_ms"] == 0.0
    assert set(agg) == {"runs", "avg_cost_usd", "avg_latency_ms", "p95_latency_ms",
                        "avg_steps", "avg_citation_coverage"}
