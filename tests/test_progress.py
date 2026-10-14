"""The progress reported while a run works: every stage, in order, never backwards."""

from __future__ import annotations

import re

import pytest

from agent.runner import run
from tests.conftest import QUESTION


def _events(settings, **kw):
    events: list[tuple[float, str]] = []
    result = run(QUESTION, settings=settings, persist=False,
                 progress=lambda f, label: events.append((f, label)), **kw)
    return result, events


@pytest.mark.parametrize("concurrency", [1, 4], ids=["serial", "thread-pool"])
def test_progress_follows_every_stage_and_never_goes_back(settings, concurrency):
    result, events = _events(settings.model_copy(update={"research_concurrency": concurrency}))
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions)  # the bar never moves backwards
    assert events[0][1].startswith("Planning") and events[-1] == (1.0, "Done")

    labels = " | ".join(label for _, label in events)
    for stage in ("Searching", "Reading sources", "Writing the report", "Checking every claim",
                  "Revising (loop 1 of 2)", "Finalizing"):
        assert stage in labels
    # One event per page read, counting up to the pages actually fetched.
    reads = [re.search(r"(\d+) of (\d+) documents", label) for _, label in events]
    counts = [(int(m[1]), int(m[2])) for m in reads if m]
    fetched = sum(1 for s in result.trace if s.tool == "fetch")
    assert counts == [(i, fetched) for i in range(1, fetched + 1)]


def test_progress_only_observes(settings):
    plain = run(QUESTION, settings=settings, persist=False)
    watched, _ = _events(settings)
    assert watched.report.model_dump() == plain.report.model_dump()
    assert [e.model_dump() for e in watched.evidence] == [e.model_dump() for e in plain.evidence]
    assert [(s.node, s.tool, s.tokens) for s in watched.trace] == \
           [(s.node, s.tool, s.tokens) for s in plain.trace]


def test_a_broken_progress_display_never_breaks_a_run(settings):
    def broken(fraction, label):
        raise RuntimeError("the page went away")

    result = run(QUESTION, settings=settings, persist=False, progress=broken)
    assert result.status == "complete" and result.citation_coverage == 1.0
