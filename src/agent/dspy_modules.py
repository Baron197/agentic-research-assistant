"""Optional DSPy backend: declarative reasoning modules behind the LLM Protocol.

Opt-in via ``settings.agent_backend == "dspy"``. ``import dspy`` is lazy (it only
happens inside this module, which is itself imported lazily by ``get_llm``), so the
keyless/manual path never loads DSPy.

The key idea: ``DSPyLLM`` conforms to the **same** ``LLM`` Protocol as ``FakeLLM`` /
``OpenAILLM`` and returns the **same** role-shaped ``content`` dicts. So the graph,
the agents, the guardrails (including the no-fabricated-sources enforcement in the
finalizer), and the eval all treat the DSPy backend identically — it simply swaps
*how* the plan/write/critique steps are produced (declarative DSPy modules whose
prompts/demos can be optimized) for *what* produces them.

What DSPy governs: the three LLM reasoning steps (planner, writer, critic).
What it does NOT touch: the tool-using researcher and the LangGraph orchestration.
"""

# NOTE: intentionally NO `from __future__ import annotations` here. DSPy's
# Signature metaclass reads the OutputField type annotations as real type
# objects; stringized (future) annotations arrive as ForwardRefs and are
# rejected. This is the one module that must opt out of that project default.
from pathlib import Path
from typing import Any

from .config import Settings
from .llm import ROLE_CRITIC, ROLE_PLANNER, ROLE_WRITER, LLMRequest, LLMResponse, _count
from .textutil import content_words


# --- text helpers (map our structured data <-> the DSPy modules' text I/O) ---
def _format_context(evidence: list[dict[str, Any]]) -> str:
    """Render gathered evidence as 'id | title | snippet' lines for the modules."""
    return "\n".join(
        f"{e.get('id')} | {e.get('source_title', '')} | {e.get('snippet', '')}"
        for e in evidence
    )


def _render_report_for_critic(draft: dict[str, Any]) -> str:
    """Render the draft as 'Cn: text [cites E1,E2]' lines so the critic can name ids."""
    lines = []
    for section in draft.get("sections", []):
        for claim in section.get("claims", []):
            cites = ",".join(claim.get("evidence_ids", [])) or "none"
            lines.append(f"{claim.get('id')}: {claim.get('text', '')} [cites {cites}]")
    return "\n".join(lines)


def _build_signatures(dspy):
    """Define the typed DSPy Signatures (needs the dspy module)."""
    from pydantic import BaseModel

    class _Claim(BaseModel):
        text: str
        evidence_ids: list[str] = []

    class _Section(BaseModel):
        heading: str
        claims: list[_Claim] = []

    class PlanResearch(dspy.Signature):
        """Decompose a research question into 3-6 focused, non-overlapping sub-questions."""

        question: str = dspy.InputField()
        subquestions: list[str] = dspy.OutputField(desc="3-6 focused sub-questions")

    class WriteReport(dspy.Signature):
        """Write a structured, cited report from the evidence.

        Cite ONLY evidence ids that appear in `context`; never invent ids. Each
        claim's `evidence_ids` must be a subset of the provided ids.
        """

        question: str = dspy.InputField()
        context: str = dspy.InputField(desc="evidence items, one per line: id | title | snippet")
        summary: str = dspy.OutputField()
        sections: list[_Section] = dspy.OutputField()

    class CritiqueReport(dspy.Signature):
        """Identify claims NOT supported by their cited evidence.

        Return the ids of unsupported claims and a verdict ('accept' if all are
        supported, otherwise 'revise').
        """

        report: str = dspy.InputField(desc="claims as 'Cn: text [cites ...]'")
        context: str = dspy.InputField(desc="evidence items: id | title | snippet")
        verdict: str = dspy.OutputField(desc="'accept' or 'revise'")
        unsupported_claims: list[str] = dspy.OutputField(desc="ids of unsupported claims, e.g. ['C2']")

    return PlanResearch, WriteReport, CritiqueReport


def build_program(settings: Settings, dspy=None):
    """Build the composable DSPy program (and load the optimized artifact if present)."""
    if dspy is None:
        import dspy  # lazy

    plan_sig, write_sig, crit_sig = _build_signatures(dspy)

    class ResearchProgram(dspy.Module):
        """The optimizable program: the write/critique reasoning steps.

        The planner module lives here so the whole program serialises together,
        but ``forward`` (what the optimizer compiles against) exercises only the
        writer and critic — the steps that produce the scored report. The
        planner runs un-bootstrapped at inference time.
        """

        def __init__(self) -> None:
            super().__init__()
            self.planner = dspy.ChainOfThought(plan_sig)
            self.writer = dspy.ChainOfThought(write_sig)
            self.critic = dspy.ChainOfThought(crit_sig)

        def forward(self, question: str, context: str):
            """Used by the optimizer: produce a report (writer) + a critic verdict."""
            draft = self.writer(question=question, context=context)
            # Render one line per CLAIM with its citations — the same format the
            # CritiqueReport signature documents and the runtime critic uses.
            lines = []
            n = 0
            for s in getattr(draft, "sections", []) or []:
                for c in _attr(s, "claims", []) or []:
                    n += 1
                    cites = ",".join(str(e) for e in (_attr(c, "evidence_ids", []) or [])) or "none"
                    lines.append(f"C{n}: {_attr(c, 'text', '')} [cites {cites}]")
            rendered = "\n".join(lines)
            crit = self.critic(report=rendered, context=context)
            return dspy.Prediction(
                summary=getattr(draft, "summary", ""),
                sections=getattr(draft, "sections", []),
                verdict=getattr(crit, "verdict", "accept"),
                unsupported=getattr(crit, "unsupported_claims", []),
            )

    program = ResearchProgram()
    path = Path(settings.dspy_artifact_path)
    if path.exists():
        program.load(str(path))  # restore the optimized prompts/demos
    return program


class DSPyLLM:
    """An ``LLM`` strategy whose plan/write/critique steps are DSPy modules.

    For tests, inject a DSPy ``DummyLM`` via ``lm=`` to run fully keyless. In real
    use it configures ``dspy.LM`` from settings and requires an OpenAI key (mirroring
    the other real providers).
    """

    name = "dspy-llm"

    def __init__(self, settings: Settings, *, lm: Any = None, program: Any = None) -> None:
        import dspy  # lazy: keyless/manual path never imports this

        self._dspy = dspy
        self.settings = settings
        if lm is not None:
            dspy.configure(lm=lm)
        else:
            if not settings.openai_api_key:
                raise ValueError(
                    "agent_backend=dspy requires OPENAI_API_KEY "
                    "(or inject a DummyLM via DSPyLLM(settings, lm=...) in tests)."
                )
            dspy.configure(lm=dspy.LM(settings.dspy_model, api_key=settings.openai_api_key))
        self._program = program if program is not None else build_program(settings, dspy=dspy)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if request.role == ROLE_PLANNER:
            content = self._plan(request.payload)
        elif request.role == ROLE_WRITER:
            content = self._write(request.payload)
        elif request.role == ROLE_CRITIC:
            content = self._critique(request.payload)
        else:  # pragma: no cover - defensive
            raise ValueError(f"DSPyLLM has no module for role {request.role!r}")
        return LLMResponse(content=content, tokens=_count(request.payload, content))

    # -- role dispatch -----------------------------------------------------
    def _plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        pred = self._program.planner(question=question)
        subs = [str(s).strip() for s in (getattr(pred, "subquestions", []) or []) if str(s).strip()]
        if not subs:
            # Robustness: a real LM may occasionally return no sub-questions. Fall
            # back to the question itself so the run degrades gracefully (abstains
            # / partial) instead of crashing the planner node.
            subs = [question] if question else []
        # Search queries reuse the topic-anchored approach so retrieval + abstention
        # behave exactly like the manual backend (the researcher is unchanged).
        terms = " ".join(dict.fromkeys(content_words(question))) or question
        sub_questions = [
            {"id": f"SQ{i}", "question": sq, "search_queries": [terms]}
            for i, sq in enumerate(subs, start=1)
        ]
        return {"sub_questions": sub_questions}

    def _write(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        evidence = list(payload.get("evidence", []))
        # Claims the critic already removed must not be re-emitted, or the
        # revise loop never converges (mirrors the manual backend's contract).
        rejected = {str(r) for r in payload.get("rejected", [])}
        pred = self._program.writer(question=question, context=_format_context(evidence))
        sections = []
        for s in getattr(pred, "sections", []) or []:
            heading = _attr(s, "heading", "Findings")
            claims = []
            for c in _attr(s, "claims", []) or []:
                # parse_report strips claim text before the critic records it in
                # 'rejected'; strip here too or the comparison can never match.
                text = str(_attr(c, "text", "")).strip()
                if text in rejected:
                    continue
                claims.append(
                    {"text": text,
                     "evidence_ids": [str(e) for e in (_attr(c, "evidence_ids", []) or [])]}
                )
            sections.append({"heading": str(heading), "claims": claims})
        return {"summary": str(getattr(pred, "summary", "")), "sections": sections}

    def _critique(self, payload: dict[str, Any]) -> dict[str, Any]:
        draft = payload.get("draft", {})
        evidence = list(payload.get("evidence", []))
        pred = self._program.critic(
            report=_render_report_for_critic(draft), context=_format_context(evidence)
        )
        unsupported = [str(x) for x in (getattr(pred, "unsupported_claims", []) or [])]
        unsupported_set = set(unsupported)
        all_ids = [
            c.get("id") for s in draft.get("sections", []) for c in s.get("claims", [])
        ]
        supported = [cid for cid in all_ids if cid not in unsupported_set]
        # Trust the unsupported list for the verdict (mirrors the manual critic and is
        # robust to a model returning an inconsistent verdict string).
        verdict = "revise" if unsupported else "accept"
        return {"supported": supported, "unsupported": unsupported,
                "verdict": verdict, "notes": str(getattr(pred, "verdict", ""))}


def _attr(obj: Any, name: str, default: Any) -> Any:
    """Read ``name`` from a pydantic object or a dict (DummyLM may yield either)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_dspy_llm(settings: Settings) -> DSPyLLM:
    """Factory used by ``get_llm`` for the real DSPy backend."""
    return DSPyLLM(settings)
