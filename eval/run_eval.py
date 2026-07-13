"""Evaluation harness with metrics, a critic on/off A/B, and a CI gate.

All metrics below are computed on the **keyless** deterministic path, so they
validate *structure and plumbing* — citation discipline, source validity, the
critic's effect — not human-judged answer quality. The single metric that
genuinely needs a real model (``faithfulness``, LLM-as-judge) is import-guarded
and reported as ``n/a`` unless real mode is configured. We never fabricate
numbers; everything here is recomputed from real runs.

Usage:
    python -m eval.run_eval                         # single mode -> metrics.{json,md}
    python -m eval.run_eval --compare               # critic ON vs OFF A/B
    python -m eval.run_eval --min-citation-coverage 0.8   # CI gate (exit 1 on fail)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from agent import metrics as _metrics
from agent.config import Settings
from agent.runner import run
from agent.schemas import Report, RunResult

SUPPORT_THRESHOLD = _metrics.SUPPORT_THRESHOLD
POINT_THRESHOLD = _metrics.POINT_THRESHOLD


# --- metric helpers (delegated to the shared agent.metrics module) ----------
def report_text(report: Report) -> str:
    return _metrics.report_text(report)


def citation_coverage(report: Report) -> float:
    return _metrics.citation_coverage(report)


def source_validity(result: RunResult) -> float:
    """Fraction of citation references that map to actually-gathered evidence."""
    return _metrics.source_validity(result.report, result.evidence)


def support_rate(result: RunResult) -> float:
    """Fraction of claims whose cited evidence snippet actually supports them."""
    return _metrics.support_rate(result.report, result.evidence)


def point_coverage(report: Report, points: list[str]) -> float:
    return _metrics.point_coverage(report, points)


def _mean(xs: list[float]) -> float:
    return round(statistics.mean(xs), 4) if xs else 0.0


# --- task loading -----------------------------------------------------------
def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


# --- single mode ------------------------------------------------------------
def evaluate_single(tasks: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    rows = []
    for task in tasks:
        result = run(task["question"], settings=settings, persist=False)
        report = result.report
        in_corpus = task.get("in_corpus", True)
        row = {
            "id": task["id"],
            "in_corpus": in_corpus,
            "n_claims": len(report.all_claims()),
            "citation_coverage": round(citation_coverage(report), 4),
            "source_validity": round(source_validity(result), 4),
            "support_rate": round(support_rate(result), 4),
            "point_coverage": round(point_coverage(report, task.get("expected_points", [])), 4),
            "abstained": len(report.all_claims()) == 0,
            "tool_calls": result.tool_calls,
            "tokens": result.tokens,
            "steps": len(result.trace),
            "latency_ms": result.latency_ms,
        }
        rows.append(row)

    in_rows = [r for r in rows if r["in_corpus"]]
    out_rows = [r for r in rows if not r["in_corpus"]]
    aggregate = {
        "n_tasks": len(rows),
        "n_in_corpus": len(in_rows),
        "citation_coverage": _mean([r["citation_coverage"] for r in in_rows]),
        "source_validity": _mean([r["source_validity"] for r in in_rows]),
        "support_rate": _mean([r["support_rate"] for r in in_rows]),
        "point_coverage": _mean([r["point_coverage"] for r in in_rows]),
        "avg_tool_calls": _mean([float(r["tool_calls"]) for r in in_rows]),
        "avg_tokens": _mean([float(r["tokens"]) for r in in_rows]),
        "avg_steps": _mean([float(r["steps"]) for r in in_rows]),
        "avg_latency_ms": _mean([r["latency_ms"] for r in in_rows]),
        # 0.0 would read as "always fails to abstain"; be explicit when the task
        # set simply contains no out-of-corpus checks.
        "abstention_accuracy": (
            _mean([1.0 if r["abstained"] else 0.0 for r in out_rows])
            if out_rows else "n/a (no out-of-corpus tasks)"
        ),
        "faithfulness": _faithfulness(settings),
    }
    return {"mode": "single", "rows": rows, "aggregate": aggregate}


def _faithfulness(settings: Settings) -> str:
    """LLM-as-judge faithfulness — only meaningful in real mode (import-guarded)."""
    if settings.llm_provider != "openai":
        return "n/a (keyless; requires real model + judge)"
    try:  # pragma: no cover - real path
        import openai  # noqa: F401
    except Exception:  # pragma: no cover
        return "n/a (openai not installed)"
    return "not-run (enable a judge to compute)"  # pragma: no cover


# --- compare (A/B) mode -----------------------------------------------------
def evaluate_compare(tasks: list[dict[str, Any]], base: Settings) -> dict[str, Any]:
    in_tasks = [t for t in tasks if t.get("in_corpus", True)]
    on_settings = base.model_copy(update={"enable_critic": True})
    off_settings = base.model_copy(update={"enable_critic": False})
    rows = []
    for task in in_tasks:
        on = run(task["question"], settings=on_settings, persist=False)
        off = run(task["question"], settings=off_settings, persist=False)
        rows.append({
            "id": task["id"],
            "coverage_on": round(citation_coverage(on.report), 4),
            "coverage_off": round(citation_coverage(off.report), 4),
            "support_on": round(support_rate(on), 4),
            "support_off": round(support_rate(off), 4),
        })
    cov_on = _mean([r["coverage_on"] for r in rows])
    cov_off = _mean([r["coverage_off"] for r in rows])
    sup_on = _mean([r["support_on"] for r in rows])
    sup_off = _mean([r["support_off"] for r in rows])
    aggregate = {
        "coverage_on": cov_on, "coverage_off": cov_off,
        "coverage_delta": round(cov_on - cov_off, 4),
        "support_on": sup_on, "support_off": sup_off,
        "support_delta": round(sup_on - sup_off, 4),
    }
    return {"mode": "compare", "rows": rows, "aggregate": aggregate}


# --- rendering --------------------------------------------------------------
def render_single_md(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    lines = ["# Evaluation Results (keyless)\n",
             "_All metrics validate structure/plumbing on the deterministic fake "
             "path. `faithfulness` requires a real model._\n",
             "## Aggregate\n",
             "| metric | value |", "|---|---|"]
    for k in ["n_tasks", "n_in_corpus", "citation_coverage", "source_validity",
              "support_rate", "point_coverage", "avg_tool_calls", "avg_tokens",
              "avg_steps", "avg_latency_ms", "abstention_accuracy", "faithfulness"]:
        lines.append(f"| {k} | {agg[k]} |")
    lines += ["\n## Per task\n",
              "| id | in_corpus | claims | cite_cov | src_valid | support | "
              "point_cov | abstained | tools | tokens |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        lines.append(
            f"| {r['id']} | {r['in_corpus']} | {r['n_claims']} | "
            f"{r['citation_coverage']} | {r['source_validity']} | {r['support_rate']} | "
            f"{r['point_coverage']} | {r['abstained']} | {r['tool_calls']} | {r['tokens']} |"
        )
    return "\n".join(lines) + "\n"


def render_compare_md(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    lines = ["# Critic A/B: ON vs OFF (keyless)\n",
             "The critic removes uncited / unsupported claims, so citation "
             "coverage and support rate are higher with it ON.\n",
             "## Aggregate\n",
             "| metric | critic ON | critic OFF | delta |", "|---|---|---|---|",
             f"| citation_coverage | {agg['coverage_on']} | {agg['coverage_off']} | "
             f"**{agg['coverage_delta']:+}** |",
             f"| support_rate | {agg['support_on']} | {agg['support_off']} | "
             f"**{agg['support_delta']:+}** |",
             "\n## Per task\n",
             "| id | cov ON | cov OFF | support ON | support OFF |",
             "|---|---|---|---|---|"]
    for r in report["rows"]:
        lines.append(f"| {r['id']} | {r['coverage_on']} | {r['coverage_off']} | "
                     f"{r['support_on']} | {r['support_off']} |")
    return "\n".join(lines) + "\n"


# --- backend A/B (manual vs DSPy) -------------------------------------------
def _dspy_available(settings: Settings) -> tuple[bool, str]:
    """Whether a DSPy A/B can actually run (needs the lib + a live LLM)."""
    import importlib.util
    if importlib.util.find_spec("dspy") is None:
        return False, "dspy not installed (pip install dspy-ai)"
    artifact = Path(settings.dspy_artifact_path).exists()
    if not settings.openai_api_key and not artifact:
        return False, "no OPENAI_API_KEY and no compiled artifact (run 'make optimize')"
    if not settings.openai_api_key:
        return False, "compiled artifact present but OPENAI_API_KEY is needed to run the LLM"
    return True, "ok"


def _backend_rows(settings: Settings, in_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in in_tasks:
        r = run(task["question"], settings=settings, persist=False)
        rows.append({
            "id": task["id"],
            "citation_coverage": round(citation_coverage(r.report), 4),
            "source_validity": round(source_validity(r), 4),
            "support_rate": round(support_rate(r), 4),
        })
    return rows


def _backend_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": rows,
        "citation_coverage": _mean([r["citation_coverage"] for r in rows]),
        "source_validity": _mean([r["source_validity"] for r in rows]),
        "support_rate": _mean([r["support_rate"] for r in rows]),
    }


def evaluate_compare_backends(tasks: list[dict[str, Any]], base: Settings) -> dict[str, Any]:
    """A/B the manual vs DSPy backend. The DSPy side is a REAL-LLM run and is
    skipped gracefully (manual-only) when DSPy / key / artifact are unavailable."""
    in_tasks = [t for t in tasks if t.get("in_corpus", True)]
    manual = _backend_summary(_backend_rows(base.model_copy(update={"agent_backend": "manual"}), in_tasks))
    available, note = _dspy_available(base)
    result: dict[str, Any] = {"mode": "compare_backends", "manual": manual,
                              "dspy_available": available, "dspy_note": note}
    if available:
        result["dspy"] = _backend_summary(
            _backend_rows(base.model_copy(update={"agent_backend": "dspy"}), in_tasks)
        )
    return result


def render_compare_backends_md(report: dict[str, Any]) -> str:
    m = report["manual"]
    lines = ["# Backend A/B: manual vs DSPy", ""]
    if not report.get("dspy_available"):
        lines += [
            f"_DSPy column skipped: {report['dspy_note']}. (A real-LLM run is required.)_",
            "",
            "| metric | manual (keyless) |",
            "|---|---|",
            f"| citation_coverage | {m['citation_coverage']} |",
            f"| source_validity | {m['source_validity']} |",
            f"| support_rate | {m['support_rate']} |",
        ]
    else:
        d = report["dspy"]
        lines += [
            "_DSPy is a REAL-LLM run; the manual baseline is keyless._",
            "",
            "| metric | manual | DSPy | delta |",
            "|---|---|---|---|",
            f"| citation_coverage | {m['citation_coverage']} | {d['citation_coverage']} | {round(d['citation_coverage'] - m['citation_coverage'], 4)} |",
            f"| source_validity | {m['source_validity']} | {d['source_validity']} | {round(d['source_validity'] - m['source_validity'], 4)} |",
            f"| support_rate | {m['support_rate']} | {d['support_rate']} | {round(d['support_rate'] - m['support_rate'], 4)} |",
        ]
    return "\n".join(lines) + "\n"


# --- CLI --------------------------------------------------------------------
def _apply_gate(coverage: float, threshold: float | None) -> int:
    """Shared CI gate: exit 1 when citation coverage regresses below threshold."""
    if threshold is None:
        return 0
    if coverage < threshold:
        print(f"[gate] FAIL: citation_coverage {coverage} < {threshold}", file=sys.stderr)
        return 1
    print(f"[gate] PASS: citation_coverage {coverage} >= {threshold}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the research assistant.")
    parser.add_argument("--tasks", default=None, help="path to tasks.jsonl")
    parser.add_argument("--out", default=None, help="results output dir")
    parser.add_argument("--compare", action="store_true", help="run critic ON/OFF A/B")
    parser.add_argument("--compare-backends", action="store_true",
                        help="A/B the manual vs DSPy backend (DSPy side needs a real LLM)")
    parser.add_argument("--min-citation-coverage", type=float, default=None,
                        help="CI gate: exit non-zero if avg citation coverage is below this")
    args = parser.parse_args(argv)

    settings = Settings()
    tasks_path = Path(args.tasks) if args.tasks else Path(__file__).parent / "tasks.jsonl"
    out_dir = Path(args.out) if args.out else settings.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(tasks_path)

    if args.compare:
        report = evaluate_compare(tasks, settings)
        (out_dir / "compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "compare.md").write_text(render_compare_md(report), encoding="utf-8")
        agg = report["aggregate"]
        print(f"[compare] citation_coverage ON={agg['coverage_on']} OFF={agg['coverage_off']} "
              f"(delta {agg['coverage_delta']:+}) | support ON={agg['support_on']} "
              f"OFF={agg['support_off']} (delta {agg['support_delta']:+})")
        print(f"[compare] wrote {out_dir/'compare.md'}")
        # The gate must not be silently ignored in this mode; apply it to the ON arm.
        return _apply_gate(agg["coverage_on"], args.min_citation_coverage)

    if args.compare_backends:
        report = evaluate_compare_backends(tasks, settings)
        (out_dir / "compare_backends.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "compare_backends.md").write_text(render_compare_backends_md(report), encoding="utf-8")
        m = report["manual"]
        if report.get("dspy_available"):
            d = report["dspy"]
            print(f"[backends] manual: cov={m['citation_coverage']} sv={m['source_validity']} | "
                  f"dspy: cov={d['citation_coverage']} sv={d['source_validity']}")
        else:
            print(f"[backends] manual: cov={m['citation_coverage']} sv={m['source_validity']} | "
                  f"dspy skipped ({report['dspy_note']})")
        print(f"[backends] wrote {out_dir/'compare_backends.md'}")
        return _apply_gate(m["citation_coverage"], args.min_citation_coverage)

    report = evaluate_single(tasks, settings)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "metrics.md").write_text(render_single_md(report), encoding="utf-8")
    agg = report["aggregate"]
    print(f"[eval] citation_coverage={agg['citation_coverage']} "
          f"source_validity={agg['source_validity']} support_rate={agg['support_rate']} "
          f"point_coverage={agg['point_coverage']} abstention_accuracy={agg['abstention_accuracy']}")
    print(f"[eval] avg_tool_calls={agg['avg_tool_calls']} avg_tokens={agg['avg_tokens']} "
          f"avg_steps={agg['avg_steps']} avg_latency_ms={agg['avg_latency_ms']}")
    print(f"[eval] wrote {out_dir/'metrics.md'}")

    return _apply_gate(agg["citation_coverage"], args.min_citation_coverage)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
