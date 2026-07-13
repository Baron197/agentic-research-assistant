"""Optional human-in-the-loop approval gate."""

from __future__ import annotations

from agent.runner import run
from tests.conftest import QUESTION


def test_approval_granted_completes(settings):
    result = run(QUESTION, settings=settings.model_copy(update={"require_approval": True}),
                 approval_fn=lambda draft: True, persist=False)
    assert result.status == "complete"
    assert result.report.all_claims()


def test_approval_denied_marks_awaiting(settings):
    result = run(QUESTION, settings=settings.model_copy(update={"require_approval": True}),
                 approval_fn=lambda draft: False, persist=False)
    assert result.status == "awaiting_approval"
    assert result.report.partial is True


def test_approval_default_auto_approves(settings):
    # require_approval=True but no callback -> auto-approve (keyless default).
    result = run(QUESTION, settings=settings.model_copy(update={"require_approval": True}),
                 persist=False)
    assert result.status == "complete"


def test_budget_exhaustion_still_respects_approval_gate(settings):
    # A budget of 1000 exhausts right after the writer: a draft exists, so the
    # required human gate must still run — a denied approval wins over the
    # budget-partial status (regression: budget used to bypass the gate).
    result = run(QUESTION,
                 settings=settings.model_copy(update={"require_approval": True,
                                                      "token_budget": 1000}),
                 approval_fn=lambda draft: False, persist=False)
    assert result.status == "awaiting_approval"
    assert result.report.partial is True
