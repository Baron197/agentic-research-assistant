"""MCP server: expose the research pipeline to any MCP host as callable tools.

The Model Context Protocol (MCP) is the wire protocol between an AI application
(the *host* — Claude Desktop, an IDE, another agent) and an external capability
(the *server* — this module). Wrapping the pipeline this way lets a chat
assistant answer with a **cited report** produced by this project instead of
answering from its own memory.

Why this file is a thin adapter and not a rewrite: the pipeline already has
exactly one entry point, :func:`agent.runner.run`, shared by the CLI, the
FastAPI service, the Streamlit UI and the eval harness. The MCP server is simply
a fourth caller of that same function — no agent, graph, or tool is modified to
support it.

The decisions worth knowing about:

* **Keyless by default, and MCP callers cannot change that.** A server left
  running in real mode would spend the operator's OpenAI/Tavily budget for
  whoever connects to it. Real mode is therefore refused unless the operator
  opts in with ``MCP_ALLOW_REAL_MODE=true``; otherwise the providers are forced
  back to the deterministic fakes and ``assistant_status`` says so out loud.
* **The tool surface is a trust boundary.** The *model* picks the arguments, so
  only knobs that can improve an answer are exposed (``depth``). Knobs that
  would weaken the guarantee (``enable_critic=False``) or spend unbounded budget
  (``token_budget``) stay server-side.
* **Expected failures raise ``ToolError``**, whose message reaches the calling
  model so it can fix its input and retry. Anything unexpected becomes the SDK's
  ``UnexpectedToolError``, which deliberately hides internals from the caller.
* **stdio transport speaks JSON-RPC over stdout**, so nothing on this path may
  print to stdout — the project only does so from the ``runner`` CLI entry point.

Run it::

    python -m agent.mcp_server        # stdio transport

Requires the optional dependency::

    pip install -e ".[mcp]"
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import __version__
from .config import Settings, get_settings
from .guardrails import GuardrailError, StructuredOutputError
from .runner import render_report_markdown, run
from .schemas import RunResult
from .tools.search import list_corpus

SERVER_NAME = "agentic-research-assistant"
CORPUS_RESOURCE_URI = "corpus://documents"

# Upper bound on caller-requested depth. The model chooses this value, so it is
# capped here rather than left to whatever the caller asks for.
MAX_DEPTH = 5

# Applied when real mode is configured but not allowed for MCP callers. Forcing
# all four fields (not just the LLM) is what makes ``is_keyless`` true again.
KEYLESS_OVERRIDES: dict[str, Any] = {
    "llm_provider": "fake",
    "search_provider": "fake",
    "fetch_provider": "fake",
    "agent_backend": "manual",
}

# Server-level guidance, surfaced to the host during initialisation. This is the
# one place to tell a calling model what the server is *for* and what its output
# means, before it has looked at any individual tool.
INSTRUCTIONS = """\
Multi-agent research assistant. Given a question, it plans sub-questions,
gathers evidence with tools, drafts a report, has a critic check each claim
against its cited evidence, and returns Markdown in which each claim carries a
[n] citation to a source that was actually retrieved.

Citations cannot be fabricated: any citation to a source that was not gathered
is stripped, and a claim left with none is dropped. An empty or near-empty
report therefore means the available sources did not cover the question — report
that abstention to the user rather than filling the gap from memory.

Call `list_sources` first when you need to know what the assistant can cite.
"""

mcp = MCPServer(
    name=SERVER_NAME,
    title="Agentic Research & Report Assistant",
    version=__version__,
    instructions=INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Core implementation (settings-injectable, protocol-free, unit-testable)
# ---------------------------------------------------------------------------
def resolve_settings(settings: Settings | None = None) -> tuple[Settings, bool]:
    """Return ``(effective settings, downgraded)`` for an MCP-served run.

    ``downgraded`` is True when real providers were configured but the operator
    has not set ``MCP_ALLOW_REAL_MODE``, in which case the returned settings are
    forced back to the keyless fakes. Callers surface that flag rather than
    silently pretending real mode is active.
    """
    raw = settings if settings is not None else get_settings()
    if raw.is_keyless or raw.mcp_allow_real_mode:
        return raw, False
    return raw.model_copy(update=KEYLESS_OVERRIDES), True


def corpus_payload(settings: Settings) -> dict[str, Any]:
    """Describe what the assistant is able to cite, for the tool and the resource."""
    if settings.search_provider == "web":
        return {
            "mode": "web-search",
            "description": (
                "Sources are retrieved live from the web for each question, so "
                "there is no fixed list of documents to enumerate."
            ),
            "corpus_dir": None,
            "document_count": 0,
            "documents": [],
        }
    documents = list_corpus(settings.corpus_dir)
    return {
        "mode": "local-corpus",
        "description": (
            "Every citation comes from one of these documents. A question this "
            "corpus does not cover is answered with an abstention, not a guess."
        ),
        "corpus_dir": str(settings.corpus_dir),
        "document_count": len(documents),
        "documents": documents,
    }


def status_payload(settings: Settings | None = None) -> dict[str, Any]:
    """Report how the server is configured, including the real-mode cost guard."""
    raw = settings if settings is not None else get_settings()
    effective, downgraded = resolve_settings(raw)
    sources = corpus_payload(effective)
    if downgraded:
        note = (
            "Real providers are configured, but MCP callers are restricted to "
            "keyless mode so this server cannot spend the operator's budget. "
            "Set MCP_ALLOW_REAL_MODE=true to permit paid runs."
        )
    elif effective.is_keyless:
        note = "Keyless mode: deterministic local providers, no API calls, no cost."
    else:
        note = (
            "Real mode is enabled by the operator: each call uses paid model and "
            "search APIs."
        )
    return {
        "server": SERVER_NAME,
        "version": __version__,
        "mode": "keyless" if effective.is_keyless else "real",
        "keyless": effective.is_keyless,
        "real_mode_allowed": raw.mcp_allow_real_mode,
        "real_mode_downgraded": downgraded,
        "llm": "fake-llm" if effective.llm_provider == "fake" else effective.openai_model,
        "search_provider": effective.search_provider,
        "source_mode": sources["mode"],
        "document_count": sources["document_count"],
        "default_depth": effective.evidence_per_subquestion,
        "max_depth": MAX_DEPTH,
        "critic_enabled": effective.enable_critic,
        "token_budget": effective.token_budget,
        "note": note,
    }


def format_report(result: RunResult) -> str:
    """Render a run as the Markdown a calling model reads back.

    The footer is not decoration: the calling model needs the coverage and
    ``status`` signals to judge how much weight to put on the report, and the
    closing paragraph tells it how to treat an empty one.
    """
    dropped = (
        f" · {result.dropped_claims} claim(s) dropped for citing an ungathered source"
        if result.dropped_claims
        else ""
    )
    lines = [
        render_report_markdown(result.report).rstrip(),
        "",
        "---",
        "",
        f"- run_id `{result.run_id}` · status **{result.status}**",
        f"- {len(result.report.sources)} source(s) cited · "
        f"{result.citation_coverage:.0%} of claims carry a citation{dropped}",
        f"- {result.tokens} tokens · ${result.usd:.4f} · {result.latency_ms:.0f} ms",
        "",
        "Every `[n]` above resolves to a source that was actually retrieved during "
        "this run. An empty report means the available sources did not cover the "
        "question — treat that as an abstention, not as evidence of absence.",
    ]
    return "\n".join(lines)


def research_report(
    question: str,
    *,
    depth: int | None = None,
    settings: Settings | None = None,
) -> str:
    """Run the pipeline and return the cited Markdown report.

    Kept separate from the ``@mcp.tool`` wrapper below so it can be tested with
    injected settings without going through the protocol layer.
    """
    effective, _ = resolve_settings(settings)
    overrides: dict[str, Any] = {}
    if depth is not None:
        if not 1 <= depth <= MAX_DEPTH:
            raise ToolError(f"depth must be between 1 and {MAX_DEPTH}, got {depth}")
        overrides["evidence_per_subquestion"] = depth
    try:
        result = run(question, settings=effective, **overrides)
    except StructuredOutputError as exc:
        # A model-side failure, not the caller's fault — but worth telling the
        # caller so it can retry rather than treating the tool as broken.
        raise ToolError(f"the research model returned unusable output: {exc}") from exc
    except GuardrailError as exc:
        # Input the caller can fix (empty or over-long question).
        raise ToolError(str(exc)) from exc
    return format_report(result)


# ---------------------------------------------------------------------------
# MCP surface: tools (model-controlled) + one resource (application-controlled)
# ---------------------------------------------------------------------------
@mcp.tool(
    title="Research a question and return a cited report",
    annotations=ToolAnnotations(
        # Not read-only: every run appends a trace file under runs/ .
        read_only_hint=False,
        destructive_hint=False,  # append-only; nothing is overwritten or deleted
        idempotent_hint=False,  # a repeat run may gather different sources
        open_world_hint=True,  # in real mode the researcher fetches live pages
    ),
)
def research(
    question: Annotated[
        str,
        Field(
            min_length=3,
            description=(
                "The research question, phrased as a full question in natural "
                "language. Questions longer than the server's configured limit "
                "are rejected with an explanatory error."
            ),
        ),
    ],
    depth: Annotated[
        int | None,
        Field(
            ge=1,
            le=MAX_DEPTH,
            description=(
                "Sources to gather per sub-question. Higher is more thorough and "
                "slower. Omit to use the server's configured default."
            ),
        ),
    ] = None,
) -> str:
    """Research a question and return a Markdown report in which every claim
    carries a [n] citation to a source that was actually retrieved.

    Use this instead of answering from memory whenever the user wants a sourced
    answer, or wants the assistant's own document collection consulted. Citations
    are structurally guaranteed: a citation to a source that was not gathered is
    stripped, and a claim left with none is dropped, so a thin or empty report is
    an honest abstention rather than a failure to try.
    """
    return research_report(question, depth=depth)


@mcp.tool(
    title="List the sources this assistant can cite",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def list_sources() -> dict[str, Any]:
    """List the documents `research` is able to cite from.

    Call this before `research` when you need to know whether a question is in
    scope — the assistant abstains on anything its sources do not cover, and
    knowing that up front is cheaper than a run that returns nothing.
    """
    settings, _ = resolve_settings()
    return corpus_payload(settings)


@mcp.tool(
    title="Report how this server is configured",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def assistant_status() -> dict[str, Any]:
    """Report the server's mode, model, source count and per-call cost profile.

    Worth checking before a long run: it says whether calls are free (keyless,
    deterministic local providers) or billed to the operator's API keys.
    """
    return status_payload()


@mcp.resource(
    CORPUS_RESOURCE_URI,
    name="corpus",
    title="Available source documents",
    mime_type="application/json",
)
def corpus_resource() -> str:
    """The documents `research` can cite, as JSON.

    The same content is offered as a tool and as a resource on purpose: a
    resource is *application*-controlled context a host may attach to a
    conversation up front, whereas a tool is called by the *model* when it
    decides it needs one. Which primitive a host prefers varies, so both exist.
    """
    settings, _ = resolve_settings()
    return json.dumps(corpus_payload(settings), indent=2)


def main() -> None:
    """Serve over stdio — the transport an MCP host uses to launch a local server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
