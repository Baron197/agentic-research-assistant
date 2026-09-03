"""Keyless tests for the optional MCP server.

Skipped automatically when the ``mcp`` SDK is not installed, so the core suite is
unaffected. Two layers are covered on purpose:

  * the **core functions** (``research_report``, ``resolve_settings``,
    ``status_payload``) with injected settings — no protocol involved, which is
    where behaviour bugs actually live;
  * the **protocol surface** via ``MCPServer.call_tool`` / ``list_tools`` /
    ``read_resource`` — proving the tools are registered with usable schemas and
    that a real call reaches the pipeline and comes back cited.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

pytest.importorskip("mcp")  # skip the whole module if the MCP SDK isn't installed

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from agent import mcp_server  # noqa: E402
from agent.config import Settings  # noqa: E402
from tests.conftest import QUESTION  # noqa: E402

# "1. Title — url" lines under the report's "## Sources" heading.
_SOURCE_LINE = re.compile(r"^\d+\. .+ — .+$", re.MULTILINE)


def _await(coro):
    """Run one coroutine to completion (the suite has no async plugin)."""
    return asyncio.run(coro)


def _real_mode_settings(**extra) -> Settings:
    """Settings that look like real mode without ever touching the network.

    Only ``resolve_settings``/``status_payload`` see these — neither builds a
    provider, so no client is constructed and no key is used.
    """
    return Settings(_env_file=None, llm_provider="openai",
                    openai_api_key="sk-not-a-real-key", **extra)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------
def test_research_returns_a_fully_cited_report(settings):
    md = mcp_server.research_report(QUESTION, settings=settings)

    assert "# Research Report" in md
    assert "## Sources" in md
    assert "[1]" in md  # inline citation markers survived into the markdown
    assert _SOURCE_LINE.search(md) is not None
    # The footer is what tells the calling model how much to trust the report.
    assert "of claims carry a citation" in md
    assert "treat that as an abstention" in md


def test_depth_argument_reaches_the_pipeline(settings):
    shallow = mcp_server.research_report(QUESTION, depth=1, settings=settings)
    deep = mcp_server.research_report(
        QUESTION, depth=4, settings=settings.model_copy(update={"top_search_results": 12})
    )

    assert len(_SOURCE_LINE.findall(deep)) > len(_SOURCE_LINE.findall(shallow))


def test_blank_question_raises_a_tool_error_the_model_can_act_on(settings):
    # Long enough to pass the schema's min_length, empty once stripped: this is
    # the guardrail path, and its message must survive to the caller.
    with pytest.raises(ToolError, match="must not be empty"):
        mcp_server.research_report("   ", settings=settings)


def test_overlong_question_raises_a_tool_error(settings):
    tight = settings.model_copy(update={"max_question_length": 20})
    with pytest.raises(ToolError, match="max length"):
        mcp_server.research_report("x" * 50, settings=tight)


def test_depth_is_capped_for_direct_callers(settings):
    # The JSON schema bounds depth for protocol callers; the core function must
    # not rely on that, since it is also importable directly.
    with pytest.raises(ToolError, match="depth must be between"):
        mcp_server.research_report(QUESTION, depth=mcp_server.MAX_DEPTH + 1, settings=settings)


def test_nothing_is_written_to_stdout(settings, capsys):
    # stdio transport carries JSON-RPC on stdout; a stray print would corrupt
    # every message on the wire. This is the regression guard for that.
    mcp_server.research_report(QUESTION, settings=settings)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# The real-mode cost guard
# ---------------------------------------------------------------------------
def test_keyless_settings_pass_through_untouched(settings):
    effective, downgraded = mcp_server.resolve_settings(settings)
    assert downgraded is False
    assert effective is settings


def test_real_mode_is_refused_unless_the_operator_allows_it():
    raw = _real_mode_settings()
    assert raw.is_keyless is False  # the operator did configure real mode...

    effective, downgraded = mcp_server.resolve_settings(raw)

    assert downgraded is True  # ...but MCP callers must not spend that budget
    assert effective.is_keyless is True
    assert effective.llm_provider == "fake"
    assert effective.agent_backend == "manual"


def test_real_mode_is_honoured_once_the_operator_opts_in():
    raw = _real_mode_settings(mcp_allow_real_mode=True)

    effective, downgraded = mcp_server.resolve_settings(raw)

    assert downgraded is False
    assert effective.llm_provider == "openai"


def test_status_reports_the_downgrade_instead_of_hiding_it():
    status = mcp_server.status_payload(_real_mode_settings())

    assert status["mode"] == "keyless"
    assert status["real_mode_allowed"] is False
    assert status["real_mode_downgraded"] is True
    assert "MCP_ALLOW_REAL_MODE" in status["note"]


def test_status_describes_the_keyless_default(settings):
    status = mcp_server.status_payload(settings)

    assert status["mode"] == "keyless"
    assert status["llm"] == "fake-llm"
    assert status["max_depth"] == mcp_server.MAX_DEPTH
    assert status["document_count"] > 0


# ---------------------------------------------------------------------------
# What the assistant can cite
# ---------------------------------------------------------------------------
def test_corpus_payload_lists_the_local_documents(settings):
    payload = mcp_server.corpus_payload(settings)

    assert payload["mode"] == "local-corpus"
    assert payload["document_count"] == len(payload["documents"])
    assert payload["document_count"] > 0
    assert all(d["url"].startswith("local://") for d in payload["documents"])


def test_corpus_payload_admits_there_is_no_fixed_list_on_the_web(settings):
    web = settings.model_copy(update={"search_provider": "web", "fetch_provider": "http"})

    payload = mcp_server.corpus_payload(web)

    assert payload["mode"] == "web-search"
    assert payload["documents"] == []
    assert payload["corpus_dir"] is None  # shape stays stable across modes


# ---------------------------------------------------------------------------
# The protocol surface
# ---------------------------------------------------------------------------
def test_tools_are_registered_with_usable_schemas():
    tools = {t.name: t for t in _await(mcp_server.mcp.list_tools())}

    assert set(tools) == {"research", "list_sources", "assistant_status"}
    assert all(t.description for t in tools.values())  # the model reads these

    schema = tools["research"].input_schema
    assert schema["required"] == ["question"]
    assert schema["properties"]["question"]["description"]
    depth = schema["properties"]["depth"]
    assert depth["default"] is None  # omit it to use the server default
    integer = next(b for b in depth["anyOf"] if b.get("type") == "integer")
    assert (integer["minimum"], integer["maximum"]) == (1, mcp_server.MAX_DEPTH)

    # Hints a host uses to decide whether a call needs confirmation.
    assert tools["research"].annotations.read_only_hint is False
    assert tools["research"].annotations.destructive_hint is False
    assert tools["list_sources"].annotations.read_only_hint is True


def test_calling_research_over_the_protocol_returns_a_cited_report(settings, monkeypatch):
    # The tool wrapper takes no settings (the model must not choose them), so
    # point the module's settings lookup at the temp-dir fixture instead.
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)

    result = _await(mcp_server.mcp.call_tool("research", {"question": QUESTION}))

    assert result.is_error is False
    text = result.content[0].text
    assert "## Sources" in text
    assert _SOURCE_LINE.search(text) is not None


def test_protocol_errors_carry_a_message_the_model_can_use(settings, monkeypatch):
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)

    with pytest.raises(ToolError, match="must not be empty"):
        _await(mcp_server.mcp.call_tool("research", {"question": "   "}))


def test_schema_rejects_a_question_that_is_too_short(settings, monkeypatch):
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)

    # Rejected by the JSON schema before the pipeline is ever entered.
    with pytest.raises(ToolError):
        _await(mcp_server.mcp.call_tool("research", {"question": "hi"}))


def test_list_sources_over_the_protocol(settings, monkeypatch):
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)

    result = _await(mcp_server.mcp.call_tool("list_sources", {}))

    payload = json.loads(result.content[0].text)
    assert payload["mode"] == "local-corpus"
    assert payload["document_count"] > 0


def test_corpus_resource_serves_the_same_payload_as_the_tool(settings, monkeypatch):
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)

    resources = {str(r.uri): r for r in _await(mcp_server.mcp.list_resources())}
    assert mcp_server.CORPUS_RESOURCE_URI in resources

    contents = list(_await(mcp_server.mcp.read_resource(mcp_server.CORPUS_RESOURCE_URI)))
    assert json.loads(contents[0].content) == mcp_server.corpus_payload(settings)
