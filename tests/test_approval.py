"""The human-approval gate: a callback decides at once, or the run pauses until a person does."""

from __future__ import annotations

import pytest

from agent import approvals
from agent.approvals import NotAwaitingApproval, PausedRun
from agent.observability import aggregate, load_run, recent_runs
from agent.runner import resume, run
from tests.conftest import QUESTION


def _gated(settings, **update):
    return settings.model_copy(update={"require_approval": True, **update})


@pytest.mark.parametrize(("decision", "status"), [(True, "complete"), (False, "rejected")])
def test_an_approval_fn_decides_at_once(settings, decision, status):
    result = run(QUESTION, settings=_gated(settings), approval_fn=lambda draft: decision,
                 persist=False)
    assert result.status == status
    assert result.report.partial is not decision
    assert result.report.all_claims()  # a rejected report stays readable


def test_without_one_the_run_pauses_and_approving_releases_the_draft(settings):
    labels: list[str] = []
    paused = run(QUESTION, settings=_gated(settings), persist=False,
                 progress=lambda fraction, label: labels.append(label))
    assert paused.status == "awaiting_approval" and paused.report.partial
    assert paused.trace[-1].node == "approval" and "waiting" in paused.trace[-1].output_summary
    assert labels[-1] == "Waiting for approval"  # not "Done": the run isn't
    seen = len(labels)

    final = resume(paused.run_id, True)
    assert final.status == "complete"
    assert (final.run_id, final.created_at) == (paused.run_id, paused.created_at)
    # The reviewer read exactly what approving released...
    assert final.report == paused.report.model_copy(update={"partial": False})
    # ...which is the report a run without the gate writes: pausing changes nothing else.
    plain = run(QUESTION, settings=settings, persist=False)
    assert final.report == plain.report and final.tokens == plain.tokens
    assert [s.node for s in final.trace] == [s.node for s in plain.trace][:-1] + [
        "approval", "finalizer"]
    assert len(labels) == seen  # the first part's progress display is not called again
    assert not settings.traces_dir.exists()  # persist=False holds for the second part too


def test_a_paused_run_is_decided_once(settings):
    paused = run(QUESTION, settings=_gated(settings), persist=False)
    assert approvals.is_paused(paused.run_id)
    rejected = resume(paused.run_id, False)
    assert rejected.status == "rejected" and rejected.report.partial
    assert not approvals.is_paused(paused.run_id)
    with pytest.raises(NotAwaitingApproval):
        resume(paused.run_id, True)  # a second decision (a double click, another tab)
    with pytest.raises(NotAwaitingApproval):
        resume("never-paused", True)


def test_budget_exhaustion_still_respects_approval_gate(settings):
    # A budget of 1000 exhausts right after the writer: a draft exists, so the
    # required human gate must still run (regression: budget used to bypass it).
    gated = _gated(settings, token_budget=1000)
    assert run(QUESTION, settings=gated, approval_fn=lambda draft: False,
               persist=False).status == "rejected"
    paused = run(QUESTION, settings=gated, persist=False)
    assert paused.status == "awaiting_approval"
    assert resume(paused.run_id, True).status == "partial"  # approved, but over budget


def test_the_decision_replaces_the_paused_record(settings):
    paused = run(QUESTION, settings=_gated(settings))
    assert load_run(paused.run_id, settings.traces_dir).status == "awaiting_approval"
    resume(paused.run_id, True)
    assert load_run(paused.run_id, settings.traces_dir).status == "complete"
    # History and the metrics count the run once, as decided.
    assert [r["status"] for r in recent_runs(settings.traces_dir)] == ["complete"]
    assert aggregate(settings.traces_dir)["runs"] == 1


def test_paused_runs_are_bounded(monkeypatch):
    monkeypatch.setattr(approvals, "_PAUSED", {})
    monkeypatch.setattr(approvals, "MAX_PAUSED", 2)

    def paused() -> PausedRun:
        return PausedRun(graph=None, config={}, ctx=None, question="q", created_at="",
                         latency_ms=0.0, persist=False)

    for run_id in ("a", "b", "c"):
        approvals.put(run_id, paused())
    assert [approvals.is_paused(r) for r in ("a", "b", "c")] == [False, True, True]  # oldest out
    stale = paused()
    stale.paused_at -= approvals.PAUSE_TTL_S + 1
    approvals.put("stale", stale)
    assert not approvals.is_paused("stale")  # too old to decide


@pytest.mark.parametrize(("reply", "status"), [("y", "complete"), (EOFError, "rejected")],
                         ids=["yes", "piped-input"])
def test_the_cli_asks_whoever_runs_it(tmp_path, monkeypatch, capsys, reply, status):
    from agent import runner
    from agent.config import get_settings

    monkeypatch.setenv("REQUIRE_APPROVAL", "true")
    monkeypatch.setenv("TRACES_DIR", str(tmp_path / "runs"))
    get_settings.cache_clear()

    def answer(prompt: str) -> str:
        if reply is EOFError:
            raise EOFError
        return reply

    monkeypatch.setattr("builtins.input", answer)
    assert runner._main(["runner", QUESTION]) == 0
    assert f"status={status}" in capsys.readouterr().out
