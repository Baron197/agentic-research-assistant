"""Keyless tests for the optional DSPy backend (driven by DSPy's DummyLM).

These are skipped automatically if ``dspy`` is not installed, so the core suite is
unaffected. With dspy installed they run with NO API key (DummyLM canned outputs)
and assert that the DSPy backend honours the same contracts as the manual backend —
including the no-fabricated-sources guarantee.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy")  # skip the whole module if DSPy isn't installed

from dspy.utils.dummies import DummyLM  # noqa: E402

from agent.agents._common import parse_critique, parse_plan, parse_report  # noqa: E402
from agent.config import Settings  # noqa: E402
from agent.dspy_modules import DSPyLLM  # noqa: E402
from agent.guardrails import enforce_citations  # noqa: E402
from agent.llm import ROLE_CRITIC, ROLE_PLANNER, ROLE_WRITER, LLMRequest  # noqa: E402
from agent.schemas import Evidence  # noqa: E402


def _dspy_settings():
    # agent_backend=dspy; no key needed because we inject a DummyLM below.
    return Settings(agent_backend="dspy", openai_api_key="")


def test_dspy_planner_returns_subquestions():
    lm = DummyLM([{"reasoning": "decompose", "subquestions":
                   ["What is X?", "What are the trade-offs of X?", "How is X evaluated?"]}])
    llm = DSPyLLM(_dspy_settings(), lm=lm)
    content = llm.generate(LLMRequest(role=ROLE_PLANNER, payload={"question": "What is X?"})).content
    plan = parse_plan("What is X?", content)
    assert len(plan.sub_questions) == 3
    assert all(sq.search_queries for sq in plan.sub_questions)  # topic queries attached


def test_dspy_writer_schema_and_no_fabrication():
    # The DSPy writer emits one valid claim (cites E1) and one citing a FABRICATED id.
    lm = DummyLM([{"reasoning": "draft", "summary": "Summary.",
                   "sections": [{"heading": "Core concepts", "claims": [
                       {"text": "RAG combines retrieval and generation", "evidence_ids": ["E1"]},
                       {"text": "An invented over-claim", "evidence_ids": ["E999"]}]}]}])
    llm = DSPyLLM(_dspy_settings(), lm=lm)
    evidence = [Evidence(id="E1", claim_hint="h", source_title="Doc", source_url="local://d.md",
                         snippet="RAG combines retrieval and generation")]
    payload = {"question": "q", "evidence": [e.model_dump() for e in evidence], "rejected": []}
    content = llm.generate(LLMRequest(role=ROLE_WRITER, payload=payload)).content

    draft = parse_report("q", content)
    assert draft.all_claims(), "DSPy writer should produce a Report with claims"

    # The no-fabricated-sources guarantee must hold for the DSPy backend too.
    cleaned, dropped = enforce_citations(draft, evidence)
    assert dropped == 1                                   # the E999 claim is removed
    assert cleaned.cited_evidence_ids() <= {"E1"}         # no surviving ungathered citation


def test_dspy_writer_honours_rejected_list():
    # Claims the critic already removed must not be re-emitted, or the revise
    # loop never converges on the DSPy backend (regression). The re-emitted text
    # carries trailing whitespace on purpose: 'rejected' holds the *stripped*
    # text (parse_report strips before the critic records it), so the filter
    # must compare stripped values or a real LM's whitespace defeats it.
    lm = DummyLM([{"reasoning": "draft", "summary": "S.",
                   "sections": [{"heading": "h", "claims": [
                       {"text": "keep me", "evidence_ids": ["E1"]},
                       {"text": "previously rejected claim ", "evidence_ids": []}]}]}])
    llm = DSPyLLM(_dspy_settings(), lm=lm)
    payload = {"question": "q", "evidence": [], "rejected": ["previously rejected claim"]}
    content = llm.generate(LLMRequest(role=ROLE_WRITER, payload=payload)).content
    texts = [c["text"] for s in content["sections"] for c in s["claims"]]
    assert "previously rejected claim" not in texts
    assert "previously rejected claim " not in texts
    assert "keep me" in texts


def test_dspy_critic_flags_unsupported():
    lm = DummyLM([{"reasoning": "verify", "verdict": "revise", "unsupported_claims": ["C2"]}])
    llm = DSPyLLM(_dspy_settings(), lm=lm)
    draft = {"sections": [{"heading": "h", "claims": [
        {"id": "C1", "text": "cited claim", "evidence_ids": ["E1"]},
        {"id": "C2", "text": "uncited claim", "evidence_ids": []}]}]}
    content = llm.generate(LLMRequest(role=ROLE_CRITIC, payload={"draft": draft, "evidence": []})).content
    crit = parse_critique(content)
    assert crit.verdict == "revise"
    assert "C2" in crit.unsupported
    assert "C1" in crit.supported


def test_dspy_backend_requires_key_without_injected_lm():
    # Mirrors the other real providers: dspy mode needs a key (unless a DummyLM is given).
    with pytest.raises(ValueError):
        DSPyLLM(Settings(agent_backend="dspy", openai_api_key=""))


def test_dspy_planner_empty_falls_back_without_crashing():
    # A real LM may return no sub-questions; the planner must degrade gracefully
    # (fall back to the question) rather than crash the run.
    lm = DummyLM([{"reasoning": "r", "subquestions": []}])
    llm = DSPyLLM(_dspy_settings(), lm=lm)
    content = llm.generate(LLMRequest(role=ROLE_PLANNER, payload={"question": "What is X?"})).content
    plan = parse_plan("What is X?", content)
    assert len(plan.sub_questions) >= 1
