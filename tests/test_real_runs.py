"""The real-mode runs shipped in data/real_runs/ for the keyless demo's History page.

They were recorded on a real-mode server, so no test can produce them again; these
checks keep them honest instead. Each file must still load as a run, must be a real
(paid) run, and must show exactly what the current renderer writes for its report,
so an example can't drift from the code or carry markup the renderer would escape.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.config import PROJECT_ROOT
from agent.runner import render_report_markdown
from agent.schemas import RunResult

REAL_RUNS = PROJECT_ROOT / "data" / "real_runs"


def _examples() -> list[tuple[Path, dict]]:
    files = sorted(REAL_RUNS.glob("run_*.json"))
    assert len(files) >= 2, "expected the shipped real-mode examples"
    return [(f, json.loads(f.read_text(encoding="utf-8"))) for f in files]


def test_examples_are_complete_real_runs():
    for path, data in _examples():
        run = RunResult.model_validate(data)
        assert path.name == f"run_{run.run_id}.json"
        assert run.keyless is False, path.name
        assert run.status == "complete" and run.report.all_claims(), path.name


def test_example_reports_are_what_the_renderer_writes():
    for path, data in _examples():
        markdown = render_report_markdown(RunResult.model_validate(data).report)
        assert data["markdown"] == markdown, path.name
        report_md = REAL_RUNS / path.name.replace("run_", "report_").replace(".json", ".md")
        assert report_md.read_text(encoding="utf-8") == markdown, report_md.name


def test_example_citations_point_to_gathered_evidence():
    for path, data in _examples():
        run = RunResult.model_validate(data)
        cited = run.report.cited_evidence_ids()
        assert cited and cited <= {e.id for e in run.evidence}, path.name
