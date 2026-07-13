"""API contract: happy paths plus 422 on bad input."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


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


def test_research_happy_path(client):
    resp = client.post("/research", json={"question": "What is hybrid search in RAG?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "complete"
    assert data["citation_coverage"] == 1.0
    assert data["report"]["sources"]
    # round-trip the run id
    got = client.get(f"/runs/{data['run_id']}")
    assert got.status_code == 200
    assert got.json()["run_id"] == data["run_id"]


def test_research_rejects_short_question(client):
    assert client.post("/research", json={"question": "hi"}).status_code == 422


def test_research_rejects_out_of_range_iterations(client):
    resp = client.post("/research", json={"question": "valid question text", "max_iterations": 99})
    assert resp.status_code == 422


def test_research_response_includes_support_rate(client):
    # support_rate powers the meta line and the critic A/B compare view.
    data = client.post("/research", json={"question": "What is reranking?"}).json()
    assert "support_rate" in data and 0.0 <= data["support_rate"] <= 1.0


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
    client.post("/research", json={"question": "What is chunking in RAG?"})
    body = client.get("/metrics").json()
    assert body["runs"] >= 1


def test_list_runs_returns_recent_with_question(client):
    client.post("/research", json={"question": "What is chunking in RAG?"})
    body = client.get("/runs?limit=5").json()
    assert body["runs"], "expected the just-persisted run to be listed"
    top = body["runs"][0]
    assert top["question"]  # question is carried in the summary for the browser
    assert "citation_coverage" in top and "status" in top


def test_list_runs_is_newest_first(client):
    client.post("/research", json={"question": "What is reranking?"})
    client.post("/research", json={"question": "What is hybrid search in RAG?"})
    runs = client.get("/runs").json()["runs"]
    assert runs[0]["question"] == "What is hybrid search in RAG?"  # most recent first


def test_list_runs_validates_limit(client):
    assert client.get("/runs?limit=0").status_code == 422
    assert client.get("/runs?limit=999").status_code == 422


def test_get_run_is_enriched_for_reload(client):
    # A persisted run must come back with the derived fields the UI re-renders.
    run_id = client.post("/research", json={"question": "What is reranking?"}).json()["run_id"]
    detail = client.get(f"/runs/{run_id}").json()
    assert detail["markdown"] and "citation_coverage" in detail
    assert detail["trace"] and "evidence" in detail  # tabs need these too


def test_research_rejects_whitespace_question(client):
    # A whitespace-only question must be a 422 (validation), not a 500.
    resp = client.post("/research", json={"question": "      "})
    assert resp.status_code == 422


def test_metrics_stable_shape_when_empty(client):
    # /metrics returns a stable full-shape dict even before any run is persisted.
    body = client.get("/metrics").json()
    for key in ("runs", "avg_cost_usd", "avg_latency_ms", "p95_latency_ms",
                "avg_steps", "avg_citation_coverage"):
        assert key in body
