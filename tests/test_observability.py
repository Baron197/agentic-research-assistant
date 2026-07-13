"""Costing and run aggregation (p95 nearest-rank, torn-line resilience)."""

from __future__ import annotations

from agent.observability import aggregate, cost_usd


def test_cost_usd_uses_price_table():
    assert cost_usd("gpt-4o-mini", 1000) == 0.0004
    assert cost_usd("gpt-4o", 1000) == 0.005
    assert cost_usd("fake-llm", 999999) == 0.0  # keyless is free


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
