"""The critic revise-loop: one revise + convergence, and a precise iteration cap."""

from __future__ import annotations

from agent.runner import run
from tests.conftest import QUESTION

SYNTH_PREFIX = "Taken together, the evidence suggests"


def _has_synthesis(result) -> bool:
    return any(c.text.startswith(SYNTH_PREFIX) for c in result.report.all_claims())


def test_critic_triggers_exactly_one_revise(settings):
    result = run(QUESTION, settings=settings.model_copy(update={"max_iterations": 2}),
                 persist=False)
    # The deliberately-unsupported synthesis claim forces one revise, then converges.
    assert result.iterations == 1
    assert result.status == "complete"
    assert not _has_synthesis(result)


def test_one_iteration_still_converges_to_complete(settings):
    # max_iterations=1 permits exactly one revise loop, after which the critic
    # accepts -> the run is COMPLETE (not partial) and the synthesis claim is gone.
    result = run(QUESTION, settings=settings.model_copy(update={"max_iterations": 1}),
                 persist=False)
    assert result.iterations == 1
    assert result.status == "complete"
    assert not _has_synthesis(result)


def test_zero_iterations_is_partial_but_clean(settings):
    # max_iterations=0 forbids any revise loop: the critic still removes the
    # unsupported claim, but cannot confirm via a second pass -> partial.
    result = run(QUESTION, settings=settings.model_copy(update={"max_iterations": 0}),
                 persist=False)
    assert result.iterations == 0
    assert result.status == "partial"
    assert result.report.partial is True
    assert not _has_synthesis(result)  # unsupported claim removed regardless


def test_critic_off_keeps_uncited_claim(settings):
    on = run(QUESTION, settings=settings.model_copy(update={"enable_critic": True}),
             persist=False)
    off = run(QUESTION, settings=settings.model_copy(update={"enable_critic": False}),
              persist=False)
    # With the critic OFF the uncited synthesis claim survives -> lower coverage.
    assert off.citation_coverage < on.citation_coverage
    assert on.citation_coverage == 1.0
    assert _has_synthesis(off)
