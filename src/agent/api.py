"""FastAPI service exposing the research pipeline.

Endpoints:
  * ``POST /research``   — run the pipeline, return the cited report + run metadata.
  * ``GET  /runs``       — newest-first list of persisted run summaries (history).
  * ``GET  /runs/{id}``  — one persisted run, enriched with markdown + coverage.
  * ``GET  /corpus``     — list the local corpus documents (corpus-coverage view).
  * ``GET  /health``     — liveness probe.
  * ``GET  /metrics``    — aggregate observability metrics across all runs.

Inputs are validated declaratively by pydantic (length/range bounds -> HTTP 422).
Business-logic errors are wrapped into clean HTTP 500s so stack traces never leak.
The service is keyless by default — it boots and serves with no API key.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import get_settings
from .guardrails import GuardrailError, StructuredOutputError
from .metrics import support_rate as _support_rate
from .observability import aggregate, load_run, recent_runs
from .runner import render_report_markdown, run
from .schemas import Report
from .tools.search import list_corpus

# run_ids are uuid4().hex[:12]; anything else is by definition not a run (and
# must never reach the filesystem lookup — path separators, dots, drive letters).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = FastAPI(
    title="Agentic Research & Report Assistant",
    version=__version__,
    description="Multi-agent, LangGraph-orchestrated research with cited reports.",
)


class ResearchRequest(BaseModel):
    """Validated request body for ``POST /research``."""

    question: str = Field(min_length=3, max_length=2000,
                          description="The research question.")
    max_iterations: int | None = Field(default=None, ge=0, le=10)
    token_budget: int | None = Field(default=None, ge=1)
    require_approval: bool | None = None
    enable_critic: bool | None = Field(
        default=None,
        description="Toggle the verifying critic. False = the critic-OFF arm of "
                    "the A/B (uncited claims survive; lower citation coverage).",
    )

    @field_validator("question")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        # min_length counts whitespace; ensure there is real content (-> 422, not 500).
        stripped = v.strip()
        if len(stripped) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return stripped


class ResearchResponse(BaseModel):
    """Response body: the report plus headline run metadata."""

    run_id: str
    status: str
    question: str
    report: Report
    markdown: str
    iterations: int
    tool_calls: int
    tokens: int
    usd: float
    latency_ms: float
    dropped_claims: int
    citation_coverage: float
    support_rate: float


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {"status": "ok", "version": __version__, "keyless": settings.is_keyless}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    overrides = {
        k: v
        for k, v in {
            "max_iterations": req.max_iterations,
            "token_budget": req.token_budget,
            "require_approval": req.require_approval,
            "enable_critic": req.enable_critic,
        }.items()
        if v is not None
    }
    try:
        result = run(req.question, settings=get_settings(), **overrides)
    except StructuredOutputError as exc:
        # The model returned unusable output — a server-side failure, not the
        # client's fault; 502 keeps the distinction visible to callers.
        raise HTTPException(status_code=502, detail=f"model output failed validation: {exc}") from exc
    except GuardrailError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert to a clean 500
        raise HTTPException(status_code=500, detail=f"research failed: {exc}") from exc

    return ResearchResponse(
        run_id=result.run_id,
        status=result.status,
        question=result.question,
        report=result.report,
        markdown=render_report_markdown(result.report),
        iterations=result.iterations,
        tool_calls=result.tool_calls,
        tokens=result.tokens,
        usd=result.usd,
        latency_ms=result.latency_ms,
        dropped_claims=result.dropped_claims,
        citation_coverage=round(result.citation_coverage, 4),
        support_rate=round(_support_rate(result.report, result.evidence), 4),
    )


@app.get("/runs")
def list_runs(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    """Newest-first summaries of persisted runs (backs the history browser)."""
    return {"runs": recent_runs(get_settings().traces_dir, limit)}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    result = load_run(run_id, get_settings().traces_dir)
    if result is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    # Enrich with the two derived fields the UI needs so a historical run can be
    # re-rendered exactly like a fresh one (the model itself stays lean).
    data = result.model_dump()
    data["markdown"] = render_report_markdown(result.report)
    data["citation_coverage"] = round(result.citation_coverage, 4)
    data["support_rate"] = round(_support_rate(result.report, result.evidence), 4)
    return data


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return aggregate(get_settings().traces_dir)


@app.get("/corpus")
def corpus() -> dict[str, Any]:
    """List the local corpus documents (backs the UI's corpus-coverage view)."""
    settings = get_settings()
    return {"provider": settings.search_provider,
            "documents": list_corpus(settings.corpus_dir)}
