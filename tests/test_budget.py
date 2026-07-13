"""Budget/iteration guards: a tiny budget ends 'partial' without crashing."""

from __future__ import annotations

from agent.runner import run
from tests.conftest import QUESTION


def test_tiny_budget_returns_partial(settings):
    result = run(QUESTION, settings=settings.model_copy(update={"token_budget": 5}),
                 persist=False)
    assert result.status == "partial"
    assert result.report.partial is True
    # It stopped cleanly; nothing raised, and tokens were actually charged.
    assert result.tokens > 0


def test_moderate_budget_completes(settings):
    result = run(QUESTION, settings=settings, persist=False)
    assert result.status == "complete"
