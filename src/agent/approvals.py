"""Runs paused at the human-approval gate, waiting for a person's decision.

A run with ``require_approval`` and no ``approval_fn`` stops before the finalizer
(LangGraph ``interrupt()``). Its compiled graph, holding the run's state in an
in-memory checkpoint, waits here until ``runner.resume`` brings the decision.

They live in this process's memory, which keeps the store simple and is enough
for one server: a restart forgets them, so a decision must reach the process
that paused the run (the API runs one worker). The store is bounded: a run waits
at most ``PAUSE_TTL_S``, and past ``MAX_PAUSED`` the oldest is forgotten.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

MAX_PAUSED = 32
PAUSE_TTL_S = 24 * 3600


class NotAwaitingApproval(LookupError):
    """No paused run has this id: unknown, already decided, or forgotten."""


@dataclass
class PausedRun:
    """Everything ``runner.resume`` needs to finish a paused run."""

    graph: Any  # the compiled graph; its checkpointer holds the run's state
    config: dict[str, Any]
    ctx: Any
    question: str
    created_at: str
    latency_ms: float  # pipeline time before the pause (a person's wait is not counted)
    persist: bool
    paused_at: float = field(default_factory=time.monotonic)


_LOCK = threading.Lock()
_PAUSED: dict[str, PausedRun] = {}


def _forget_expired(now: float) -> None:
    for run_id in [k for k, p in _PAUSED.items() if now - p.paused_at > PAUSE_TTL_S]:
        del _PAUSED[run_id]


def put(run_id: str, paused: PausedRun) -> None:
    with _LOCK:
        _forget_expired(time.monotonic())
        _PAUSED[run_id] = paused
        while len(_PAUSED) > MAX_PAUSED:
            del _PAUSED[next(iter(_PAUSED))]  # insertion order: the oldest first


def take(run_id: str) -> PausedRun:
    """Remove and return a paused run, so only one decision can ever resume it."""
    with _LOCK:
        _forget_expired(time.monotonic())
        try:
            return _PAUSED.pop(run_id)
        except KeyError:
            raise NotAwaitingApproval(run_id) from None


def is_paused(run_id: str) -> bool:
    with _LOCK:
        _forget_expired(time.monotonic())
        return run_id in _PAUSED
