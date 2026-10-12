"""The critic revise-loop: one revise + convergence, and a precise iteration cap."""

from __future__ import annotations

import pytest

from agent.runner import run
from tests.conftest import QUESTION

SYNTH_PREFIX = "Taken together, the evidence suggests"


def _has_synthesis(result) -> bool:
    return any(c.text.startswith(SYNTH_PREFIX) for c in result.report.all_claims())


@pytest.mark.parametrize(("cap", "iterations", "status"), [
    (2, 1, "complete"),  # the default: one revise removes the planted claim, then converges
    (1, 1, "complete"),  # a cap of 1 still allows that revise -> complete, not partial
    (0, 0, "partial"),   # no revise allowed: the claim is still removed, but unconfirmed
])
def test_iteration_cap(settings, cap, iterations, status):
    result = run(QUESTION, settings=settings.model_copy(update={"max_iterations": cap}),
                 persist=False)
    assert result.iterations == iterations
    assert result.status == status
    assert result.report.partial is (status == "partial")
    assert not _has_synthesis(result)  # the unsupported claim is removed regardless


def test_critic_off_keeps_uncited_claim(settings):
    on = run(QUESTION, settings=settings.model_copy(update={"enable_critic": True}),
             persist=False)
    off = run(QUESTION, settings=settings.model_copy(update={"enable_critic": False}),
              persist=False)
    # With the critic OFF the uncited synthesis claim survives -> lower coverage.
    assert off.citation_coverage < on.citation_coverage
    assert on.citation_coverage == 1.0
    assert _has_synthesis(off)
    assert off.removed_claims == []


def test_run_records_the_claims_the_critic_removed(settings):
    # The removed claim is gone from the report but kept in the run record, so a
    # revise loop (or a partial run) shows what was cut and why the loop ran.
    result = run(QUESTION, settings=settings, persist=False)
    assert len(result.removed_claims) == 1
    assert result.removed_claims[0].startswith(SYNTH_PREFIX)
    assert all(c.text not in result.removed_claims for c in result.report.all_claims())
