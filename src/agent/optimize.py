"""Compile (optimize) the DSPy program against the project's own eval metric.

This is the "programming, not prompting" payoff: instead of hand-tuning prompts,
DSPy's optimizer searches prompts/demos to maximize ``dspy_metric.metric`` (which
reuses ``agent.metrics`` — grounding-first). It REQUIRES a real LLM and is never run
in the default CI gate; it is a separate, manual command.

Usage:
    python -m agent.optimize          # needs OPENAI_API_KEY; saves the compiled program
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .dspy_metric import metric
from .runner import run


def _build_examples(settings: Settings, dspy: Any, tasks: list[dict[str, Any]]) -> list[Any]:
    """Gather evidence per task with the keyless manual pipeline -> DSPy Examples."""
    gather = settings.model_copy(update={"agent_backend": "manual", "llm_provider": "fake"})
    examples = []
    for task in tasks:
        if not task.get("in_corpus", True):
            continue  # need real gathered evidence to optimize the writer/critic
        result = run(task["question"], settings=gather, persist=False)
        context = "\n".join(
            f"{e.id} | {e.source_title} | {e.snippet}" for e in result.evidence
        )
        ex = dspy.Example(
            question=task["question"],
            context=context,
            evidence=result.evidence,
            expected_points=task.get("expected_points", []),
        ).with_inputs("question", "context")
        examples.append(ex)
    return examples


def _avg_metric(program: Any, examples: list[Any]) -> float:
    if not examples:
        return 0.0
    total = 0.0
    for ex in examples:
        pred = program(question=ex.question, context=ex.context)
        total += metric(ex, pred)
    return round(total / len(examples), 4)


def main(argv: list[str]) -> int:
    settings = get_settings().model_copy(update={"agent_backend": "dspy"})
    if not settings.openai_api_key:
        print("optimize requires OPENAI_API_KEY (DSPy compilation runs a real LLM).",
              file=sys.stderr)
        return 2

    import dspy  # lazy

    from .dspy_modules import build_program

    dspy.configure(lm=dspy.LM(settings.dspy_model, api_key=settings.openai_api_key))

    tasks_path = Path(__file__).resolve().parents[2] / "eval" / "tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = _build_examples(settings, dspy, tasks)
    split = max(1, int(len(examples) * 0.6))
    trainset, devset = examples[:split], (examples[split:] or examples[:1])

    program = build_program(settings, dspy=dspy)
    before = _avg_metric(program, devset)

    if settings.dspy_optimizer == "mipro":
        optimizer = dspy.MIPROv2(metric=metric, auto="light")
        compiled = optimizer.compile(program, trainset=trainset)
    else:
        optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
        compiled = optimizer.compile(program, trainset=trainset)

    artifact = Path(settings.dspy_artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(artifact))
    after = _avg_metric(compiled, devset)

    print(f"[optimize] dev metric: before={before:.3f}  after={after:.3f}  "
          f"(delta {after - before:+.3f})")
    print(f"[optimize] saved compiled program -> {artifact}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
