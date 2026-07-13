"""The objective the DSPy optimizer maximizes — reuses the project's eval logic.

A higher score means a more *grounded* report: no fabricated citations
(source_validity, weighted heaviest), good citation coverage, and claims actually
supported by their evidence. This is intentionally the same notion of quality the
offline eval reports (via ``agent.metrics``), so optimizing toward it improves the
numbers the eval prints.
"""

from __future__ import annotations

from typing import Any

from . import metrics
from .schemas import Claim, Report, ReportSection


def _attr(obj: Any, name: str, default: Any) -> Any:
    """Read ``name`` from a pydantic object or a dict (DummyLM may yield either)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _prediction_to_report(question: str, prediction: Any) -> Report:
    """Turn a DSPy program prediction (summary + sections) into a ``Report``."""
    sections = []
    counter = 0
    for s in _attr(prediction, "sections", []) or []:
        heading = _attr(s, "heading", None)
        claims = []
        for c in _attr(s, "claims", []) or []:
            counter += 1
            claims.append(
                Claim(
                    id=f"C{counter}",
                    text=str(_attr(c, "text", "")),
                    evidence_ids=[str(e) for e in (_attr(c, "evidence_ids", []) or [])],
                )
            )
        sections.append(ReportSection(heading=str(heading or "Findings"), claims=claims))
    return Report(question=question, summary=str(_attr(prediction, "summary", "")), sections=sections)


def metric(example: Any, prediction: Any, trace: Any = None) -> float | bool:
    """Score a produced report in [0, 1]: grounding-first, then coverage + support.

    When DSPy passes a ``trace`` (bootstrapping), the return value is used as a
    pass/fail gate for demo selection — so return a strict boolean there. A
    truthy float like 0.5 would otherwise admit weakly-grounded traces as demos.
    """
    question = getattr(example, "question", "")
    evidence = list(getattr(example, "evidence", []) or [])
    report = _prediction_to_report(question, prediction)
    sv = metrics.source_validity(report, evidence)   # no fabricated citations (heaviest)
    cov = metrics.citation_coverage(report)          # every claim cited
    sup = metrics.support_rate(report, evidence)     # cited evidence supports the claim
    score = round(0.5 * sv + 0.3 * cov + 0.2 * sup, 4)
    if trace is not None:
        return score >= 0.99  # only fully-grounded traces qualify as demos
    return score
