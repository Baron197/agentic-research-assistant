"""API contract: happy paths plus 422 on bad input."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import real_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate persisted runs to a temp dir by pointing settings there.
    monkeypatch.setenv("TRACES_DIR", str(tmp_path / "runs"))
    from agent.config import get_settings
    get_settings.cache_clear()
    from agent.api import app
    yield TestClient(app)
    get_settings.cache_clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["keyless"] is True
    # The UI's sidebar mode box and its slider limits are built from these two.
    assert body["mode"] == {"llm": "rule-based", "sources": "corpus", "corpus": "bundled",
                            "evidence": "words"}
    assert body["limits"] == {"token_budget": 60_000, "max_iterations": 2}


def test_research_happy_path(client):
    resp = client.post("/research", json={"question": "What is hybrid search in RAG?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "complete"
    assert data["citation_coverage"] == 1.0
    assert data["report"]["sources"]
    # support_rate powers the meta line and the critic A/B compare view.
    assert 0.0 <= data["support_rate"] <= 1.0


def _stream(client, body):
    import json

    with client.stream("POST", "/research/stream", json=body) as resp:
        assert resp.status_code == 200
        return [json.loads(line) for line in resp.iter_lines() if line]


def test_research_stream_sends_progress_then_the_result(client):
    events = _stream(client, {"question": "What is hybrid search in RAG?"})
    progress = [e["pct"] for e in events if e["type"] == "progress"]
    assert progress == sorted(progress) and progress[-1] == 100
    assert [e["type"] for e in events][-1] == "result"
    data = events[-1]["data"]
    assert data["status"] == "complete" and data["citation_coverage"] == 1.0
    assert client.get(f"/runs/{data['run_id']}").status_code == 200  # persisted like POST /research


def test_research_stream_reports_failures(client, monkeypatch):
    import agent.api as api
    from agent.guardrails import StructuredOutputError

    def failing_run(question, **kwargs):
        raise StructuredOutputError("not JSON")

    monkeypatch.setattr(api, "run", failing_run)
    assert _stream(client, {"question": "a valid question"})[-1] == {
        "type": "error", "status": 502, "detail": "model output failed validation: not JSON"}
    # Bad input is still refused before anything streams.
    assert client.post("/research/stream", json={"question": "      "}).status_code == 422


def test_research_rejects_short_question(client):
    assert client.post("/research", json={"question": "hi"}).status_code == 422


def test_research_rejects_out_of_range_iterations(client):
    resp = client.post("/research", json={"question": "valid question text", "max_iterations": 99})
    assert resp.status_code == 422


def test_corpus_endpoint_lists_documents(client):
    body = client.get("/corpus").json()
    assert body["provider"] == "fake"
    docs = body["documents"]
    assert len(docs) >= 5
    assert all(d["title"] and d["url"].startswith("local://") for d in docs)


def test_research_enable_critic_toggle_changes_coverage(client):
    # The UI's live critic on/off control. With the critic OFF the deliberately
    # uncited synthesis claim survives -> strictly lower citation coverage.
    q = "What is hybrid search in RAG?"
    on = client.post("/research", json={"question": q, "enable_critic": True}).json()
    off = client.post("/research", json={"question": q, "enable_critic": False}).json()
    assert on["citation_coverage"] == 1.0
    assert off["citation_coverage"] < on["citation_coverage"]


def test_missing_run_is_404(client):
    assert client.get("/runs/does-not-exist").status_code == 404


def test_run_id_with_path_characters_is_404(client):
    # Traversal-shaped ids must never reach the filesystem lookup (regression).
    assert client.get("/runs/..%2F..%2Fsecrets").status_code == 404
    assert client.get("/runs/name.with.dots").status_code == 404


def test_metrics_endpoint(client):
    # (The shape of an empty aggregate is covered in test_observability.)
    assert client.get("/metrics").json()["runs"] == 0
    client.post("/research", json={"question": "What is chunking in RAG?"})
    assert client.get("/metrics").json()["runs"] == 1


def test_list_runs_is_newest_first_with_summaries(client):
    client.post("/research", json={"question": "What is reranking?"})
    client.post("/research", json={"question": "What is hybrid search in RAG?"})
    runs = client.get("/runs?limit=5").json()["runs"]
    assert [r["question"] for r in runs] == ["What is hybrid search in RAG?",
                                             "What is reranking?"]  # most recent first
    assert "citation_coverage" in runs[0] and "status" in runs[0]  # the History page shows these


def test_list_runs_validates_limit(client):
    assert client.get("/runs?limit=0").status_code == 422
    assert client.get("/runs?limit=999").status_code == 422


def test_get_run_is_enriched_for_reload(client):
    # A persisted run must come back with the derived fields the UI re-renders.
    run_id = client.post("/research", json={"question": "What is reranking?"}).json()["run_id"]
    detail = client.get(f"/runs/{run_id}").json()
    assert detail["run_id"] == run_id
    assert detail["markdown"] and "citation_coverage" in detail
    assert detail["trace"] and "evidence" in detail  # tabs need these too


def test_research_rejects_whitespace_question(client):
    # A whitespace-only question must be a 422 (validation), not a 500.
    resp = client.post("/research", json={"question": "      "})
    assert resp.status_code == 422


def test_removed_claims_are_returned_and_persisted(client):
    data = client.post("/research", json={"question": "What is reranking?"}).json()
    assert len(data["removed_claims"]) == 1  # the keyless writer's planted claim
    detail = client.get(f"/runs/{data['run_id']}").json()
    assert detail["removed_claims"] == data["removed_claims"]


def test_real_mode_api_caps_the_requested_budget(client, monkeypatch):
    # A paid deployment's TOKEN_BUDGET / MAX_ITERATIONS bound what a caller asks for.
    import agent.api as api
    from agent.schemas import Report, RunResult

    real = real_settings(token_budget=40_000, max_iterations=2)
    seen = {}

    def fake_run(question, *, settings, **overrides):
        seen.update(overrides)
        return RunResult(run_id="r1", question=question, status="complete",
                         report=Report(question=question))

    monkeypatch.setattr(api, "get_settings", lambda: real)
    monkeypatch.setattr(api, "run", fake_run)
    resp = client.post("/research", json={"question": "a valid question",
                                          "token_budget": 500_000, "max_iterations": 5})
    assert resp.status_code == 200
    assert seen == {"token_budget": 40_000, "max_iterations": 2}
