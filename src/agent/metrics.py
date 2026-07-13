"""Shared, schema-level evaluation metrics.

These pure functions operate on a ``Report`` plus the gathered ``Evidence`` and are
reused by two callers so "quality" means one thing across the project:
  * ``eval/run_eval.py`` (the offline evaluation + CI gate), and
  * ``agent/dspy_metric.py`` (the objective the DSPy optimizer maximizes).

Keeping them here avoids duplicating the scoring logic and guarantees the optimizer
targets exactly the metric the eval reports.
"""

from __future__ import annotations

from .schemas import Evidence, Report
from .textutil import content_word_set, keyword_overlap

SUPPORT_THRESHOLD = 0.3   # claim/evidence overlap to count a claim "supported"
POINT_THRESHOLD = 0.6     # fraction of a point's words that must appear in the report


def report_text(report: Report) -> str:
    return " ".join([report.summary, *(c.text for c in report.all_claims())])


def citation_coverage(report: Report) -> float:
    """Fraction of claims carrying at least one citation."""
    claims = report.all_claims()
    if not claims:
        return 0.0
    return sum(1 for c in claims if c.is_cited) / len(claims)


def source_validity(report: Report, evidence: list[Evidence]) -> float:
    """Fraction of citation references that map to actually-gathered evidence."""
    valid = {e.id for e in evidence}
    refs = [eid for c in report.all_claims() for eid in c.evidence_ids]
    if not refs:
        return 1.0  # nothing cited -> vacuously valid
    return sum(1 for eid in refs if eid in valid) / len(refs)


def support_rate(report: Report, evidence: list[Evidence]) -> float:
    """Fraction of claims whose cited evidence snippet actually supports them."""
    claims = report.all_claims()
    if not claims:
        return 0.0
    by_id = {e.id: e.snippet for e in evidence}
    ok = 0
    for c in claims:
        if any(
            eid in by_id and keyword_overlap(c.text, by_id[eid]) >= SUPPORT_THRESHOLD
            for eid in c.evidence_ids
        ):
            ok += 1
    return ok / len(claims)


def point_coverage(report: Report, points: list[str]) -> float:
    """Fraction of expected key facts present in the report (keyword proxy)."""
    if not points:
        return 1.0
    text_words = content_word_set(report_text(report))
    covered = 0
    for point in points:
        pw = content_word_set(point)
        if pw and len(pw & text_words) / len(pw) >= POINT_THRESHOLD:
            covered += 1
    return covered / len(points)
