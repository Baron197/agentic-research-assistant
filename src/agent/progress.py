"""Progress of a run, as (fraction, label) updates for a progress bar.

The graph reports when each node starts and finishes, and the researcher reports
each search and page read as it completes; this module turns those events into a
fraction of the run and a short label of what is happening now ("Reading
sources: 3 of 8"). It only observes: a run with a progress callback produces the
same report, evidence and trace as one without.

How much of the bar each stage gets comes from measured real runs on the live
web (gpt-4o-mini, about 33 s): planning ~14% of the time, searching and reading
pages ~56%, writing ~23%, checking ~5%. The bar never moves backwards; a revise
loop advances through the last few percent with its own label.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressFn = Callable[[float, str], None]

# Where each stage starts and ends, as fractions of the bar.
_PLAN = (0.0, 0.14)
_SEARCH = (0.14, 0.22)
_READ = (0.22, 0.70)
_WRITE = (0.70, 0.90)
_CHECK = (0.90, 0.95)
_REVISE = (0.95, 0.99)  # shared by every revise loop the run is allowed
_FINAL = 0.99


class Progress:
    """Maps pipeline events to monotonic ``(fraction, label)`` reports."""

    def __init__(self, report: ProgressFn, *, max_iterations: int, web: bool) -> None:
        self._report = report
        self._loops = max(1, max_iterations)
        self._sources = "pages" if web else "documents"
        self._where = "the web" if web else "the documents"
        self._at = 0.0

    def _emit(self, fraction: float, label: str) -> None:
        self._at = max(self._at, min(1.0, fraction))
        try:
            self._report(self._at, label)
        except Exception:  # noqa: BLE001 - a broken progress display must never break a run
            pass

    def _loop_span(self, loop: int) -> tuple[float, float]:
        width = (_REVISE[1] - _REVISE[0]) / self._loops
        start = _REVISE[0] + (min(loop, self._loops) - 1) * width
        return start, start + width

    def node_started(self, node: str, state: dict[str, Any]) -> None:
        loop = int(state.get("iteration", 0))
        if loop and node in ("researcher", "writer", "critic"):
            start, end = self._loop_span(loop)
            step = {"researcher": (0.0, "looking for missing evidence"),
                    "writer": (0.3, "rewriting the report"),
                    "critic": (0.7, "checking the new draft")}[node]
            self._emit(start + (end - start) * step[0],
                       f"Revising (loop {loop} of {self._loops}): {step[1]}")
            return
        if node == "planner":
            self._emit(_PLAN[0], "Planning: splitting the question into sub-questions")
        elif node == "researcher":
            self._emit(_SEARCH[0], f"Searching {self._where}")
        elif node == "writer":
            n = len(state.get("evidence") or [])
            self._emit(_WRITE[0], f"Writing the report from {n} sources")
        elif node == "critic":
            self._emit(_CHECK[0], "Checking every claim against its evidence")
        elif node == "approval":
            self._emit(_FINAL, "Waiting for approval")
        elif node == "finalizer":
            self._emit(_FINAL, "Finalizing: numbering the sources")

    def node_finished(self, node: str, state: dict[str, Any], update: dict[str, Any]) -> None:
        if int(state.get("iteration", 0)):
            if node == "critic":
                end = self._loop_span(int(state["iteration"]))[1]
                self._emit(end, f"Revising (loop {state['iteration']} of {self._loops}): done")
            return
        if node == "planner":
            plan = update.get("plan")
            n = len(plan.sub_questions) if plan is not None else 0
            self._emit(_PLAN[1], f"Planned {n} sub-questions")
        elif node == "researcher":
            n = len(update.get("evidence") or [])
            self._emit(_READ[1], f"Gathered {n} sources")
        elif node == "writer":
            self._emit(_WRITE[1], "Report drafted")
        elif node == "critic":
            self._emit(_CHECK[1], "Claims checked")

    def researched(self, phase: str, done: int, total: int) -> None:
        """One search ("search") or page read ("read") finished, of ``total``."""
        if total <= 0:
            return
        if phase == "search":
            lo, hi = _SEARCH
            label = f"Searching {self._where}: {done} of {total} " + ("query" if total == 1 else "queries")
        else:
            lo, hi = _READ
            label = f"Reading sources: {done} of {total} {self._sources}"
        self._emit(lo + (hi - lo) * done / total, label)

    def done(self) -> None:
        self._emit(1.0, "Done")
