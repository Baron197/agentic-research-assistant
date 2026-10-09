"""Multi-page Streamlit UI for the research assistant.

It carries NO business logic — it only collects inputs, reaches the backend, and
renders the response. The backend is reached two ways, chosen automatically:

  * **http** (default) — call the FastAPI service over HTTP; the thin-client path
    used with docker-compose / Cloud Run / a local ``make api``. Point it with the
    ``API_URL`` env var (default http://127.0.0.1:8000; 127.0.0.1 rather than
    ``localhost`` avoids an IPv6 surprise on Windows).
  * **embedded** — run the keyless pipeline *in-process* via the ``agent`` package,
    so the one Streamlit app is fully self-contained (e.g. on Streamlit Community
    Cloud, which runs a single process). It just routes to the same ``agent``
    functions the API handlers call — still no business logic here.

Detection: probe the API once; if it is unreachable, fall back to embedded when the
``agent`` package is importable. Force embedded with ``ARA_EMBEDDED=1``.

Pages (native ``st.navigation``): Research · Critic A/B · History ·
Observability · Guide. Shared state lives in ``st.session_state``.
"""

from __future__ import annotations

import hmac
import html
import json
import os
import time
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("API_TIMEOUT", "120"))

DEFAULT_Q = "What are the main approaches to retrieval-augmented generation and their trade-offs?"

# Example questions the seed corpus can actually answer (plus one honest miss),
# so a first-time user immediately knows what the assistant is good for.
EXAMPLES = [
    "What are text embeddings and how is similarity measured?",
    "How does hybrid search combine dense and sparse retrieval?",
    "What is reranking and why are cross-encoders used?",
    "What causes hallucinations in RAG and how does grounding help?",
    "What is agentic RAG and what design patterns does it use?",
    "How do I bake sourdough bread?  (out-of-corpus → abstains)",
]

# Per-node icon + accent colour, shared by the timeline and the agent graph.
NODE_STYLE: dict[str, tuple[str, str]] = {
    "planner": ("🧭", "#2e6da4"),
    "researcher": ("🔍", "#3a7d44"),
    "writer": ("✍️", "#8250df"),
    "critic": ("🕵️", "#d9822b"),
    "approval": ("👤", "#6c757d"),
    "finalizer": ("📄", "#1b3a5c"),
}
NODE_FILL = {
    "planner": "#eaf1f8",
    "researcher": "#e8f4ea",
    "writer": "#f1ecfa",
    "critic": "#fbf0e1",
    "approval": "#eef0f2",
    "finalizer": "#e9edf3",
}
GRAPH_NODES = [("planner", "Planner"), ("researcher", "Researcher"),
               ("writer", "Writer"), ("critic", "Critic"),
               ("approval", "Approval?"), ("finalizer", "Finalizer")]
STATUS_ACCENT = {
    "complete": "#2f8a3b",
    "partial": "#d9822b",
    "awaiting_approval": "#2e6da4",
    "error": "#c0392b",
}

st.set_page_config(page_title="Agentic Research Assistant", page_icon="🔎", layout="wide")

# ------------------------------------------------------------------ session state
ss = st.session_state
ss.setdefault("q", DEFAULT_Q)
ss.setdefault("result", None)
ss.setdefault("detail", None)
ss.setdefault("compare", None)
ss.setdefault("error", None)
ss.setdefault("dark", False)
# Run settings shared by the pages. Their defaults live here, not on the widgets:
# a widget given both a value= and a Session State value shows a warning.
ss.setdefault("max_iter", 2)
ss.setdefault("budget", 60_000)
ss.setdefault("critic", True)
ss.setdefault("approval", False)
# Streamlit discards a widget's value when the next page doesn't render that same
# widget, so switching Research -> Critic A/B emptied the question and reset the
# run settings. Re-assigning the keys on every run keeps them as plain session
# state that survives page switches (Streamlit's documented pattern).
for _key in ("q", "max_iter", "budget", "critic", "approval"):
    ss[_key] = ss[_key]


def _theme_css(dark: bool) -> str:
    """Component styles via CSS variables. The dark block re-points the variables
    AND restyles Streamlit's own surfaces, so a single toggle flips the whole app.
    """
    base = """
    <style>
      :root {
        --page-bg:#ffffff; --panel-bg:#f2f6fb; --card-bg:#f7fafd; --tl-bg:#ffffff;
        --text:#1f2933; --muted:#6b7a89; --border:#e2eaf2; --code-bg:#eef3f8;
        --stat-default:#1b3a5c; --revise-bg:#fff8ec;
      }
      .hero { background: linear-gradient(120deg,#1b3a5c 0%,#2e6da4 100%);
        border-radius:16px; padding:20px 26px; margin-bottom:12px; color:#fff; }
      .hero h1 { color:#fff; font-size:1.7rem; margin:0 0 4px 0; font-weight:800; }
      .hero p { color:#e8f0f8; margin:0; font-size:0.95rem; }
      .stat-row { display:flex; gap:12px; flex-wrap:wrap; margin:6px 0 4px 0; }
      .stat { flex:1 1 120px; background:var(--card-bg); border:1px solid var(--border);
        border-radius:12px; padding:12px 16px; }
      .stat .lbl { font-size:0.72rem; text-transform:uppercase; letter-spacing:.04em;
        color:var(--muted); font-weight:600; }
      .stat .val { font-size:1.5rem; font-weight:800; line-height:1.1; margin-top:2px; }
      .runmeta { color:var(--muted); font-size:0.82rem; margin:2px 0 6px 0; }
      .runmeta code { background:var(--code-bg); padding:1px 6px; border-radius:5px; }
      .tl { display:flex; flex-direction:column; gap:8px; }
      .tl-item { display:flex; gap:10px; align-items:flex-start; background:var(--tl-bg);
        border:1px solid var(--border); border-left:4px solid #ccc; border-radius:10px;
        padding:8px 12px; }
      .tl-badge { width:26px; height:26px; border-radius:50%; flex:0 0 26px; display:flex;
        align-items:center; justify-content:center; font-size:14px; }
      .tl-head { font-weight:700; font-size:0.9rem; color:var(--text); }
      .tl-meta { font-weight:500; color:var(--muted); font-size:0.78rem; }
      .tl-sum { color:var(--muted); font-size:0.82rem; margin-top:1px; }
      .tl-revise { background:var(--revise-bg); }
      .chip-hint { color:var(--muted); font-size:0.8rem; margin:2px 0 6px 0; }
      div[data-testid="stMetricValue"] { font-size:1.5rem; }
    </style>
    """
    if not dark:
        return base
    return base + """
    <style>
      :root {
        --page-bg:#0e1117; --panel-bg:#161b26; --card-bg:#1b2231; --tl-bg:#1b2231;
        --text:#e6e9ef; --muted:#98a3b3; --border:#2b3344; --code-bg:#232b3a;
        --stat-default:#cfd8e6; --revise-bg:#2a2418;
      }
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background:var(--page-bg); }
      [data-testid="stHeader"] { background:transparent; }
      section[data-testid="stSidebar"] { background:var(--panel-bg); }
      .hero h1, .hero p { color:#fff !important; }
      [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
      [data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] h1,
      [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
      [data-testid="stMarkdownContainer"] h4, [data-testid="stMarkdownContainer"] td,
      [data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] em,
      .stApp label, [data-testid="stWidgetLabel"] p { color:var(--text); }
      section[data-testid="stSidebar"] a { color:var(--text) !important; }
      .stApp table td, .stApp table th { border-color:var(--border) !important; }
      .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
      .stApp [data-baseweb="input"] input, .stApp textarea {
        background:var(--card-bg) !important; color:var(--text) !important;
        border-color:var(--border) !important; }
      .stApp .stButton > button { background:var(--card-bg); color:var(--text);
        border:1px solid var(--border); }
      .stApp [data-baseweb="tab"] { color:var(--muted); }
      /* Material icons hard-code the light-theme colour on every glyph; relight
         them in dark mode (nav, tabs, buttons) or they vanish on the dark bg. */
      .stApp [data-testid="stIconMaterial"] { color:var(--text) !important; }
      .stApp [data-testid="stExpander"] details { background:var(--card-bg);
        border-color:var(--border); }
      .stApp [data-testid="stMetricValue"] { color:var(--text); }
      .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricLabel"] p {
        color:var(--muted); }
      .stApp code { background:var(--code-bg); color:#e6e9ef; }
      .stApp [data-testid="stNumberInputContainer"] { background:var(--card-bg); }
    </style>
    """


st.markdown(_theme_css(bool(ss.get("dark", False))), unsafe_allow_html=True)


# ------------------------------------------------------------------------- backend
FORCE_EMBEDDED = os.environ.get("ARA_EMBEDDED", "").strip().lower() in ("1", "true", "yes", "on")


def _embed() -> dict[str, Any] | None:
    """Import the ``agent`` package for in-process use; None if unavailable.

    Cheap on repeat calls — Python caches modules in ``sys.modules``.
    """
    try:
        import sys
        from pathlib import Path
        src = str(Path(__file__).resolve().parents[1] / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from agent import __version__
        from agent.config import get_settings
        from agent.guardrails import cap_request
        from agent.metrics import support_rate
        from agent.observability import aggregate, load_run, recent_runs
        from agent.runner import render_report_markdown, run
        from agent.tools.search import list_corpus
        return {"version": __version__, "get_settings": get_settings,
                "cap_request": cap_request,
                "support_rate": support_rate, "aggregate": aggregate,
                "load_run": load_run, "recent_runs": recent_runs,
                "render_markdown": render_report_markdown, "run": run,
                "list_corpus": list_corpus}
    except Exception:  # noqa: BLE001
        return None


def _backend() -> str:
    """'http' or 'embedded' — decided once per session and cached in state."""
    if "backend" not in ss:
        if FORCE_EMBEDDED:
            ss.backend = "embedded" if _embed() else "http"
        else:
            try:
                httpx.get(f"{API_URL}/health", timeout=3).raise_for_status()
                ss.backend = "http"
            except Exception:  # noqa: BLE001 — no API reachable; try in-process
                ss.backend = "embedded" if _embed() else "http"
    return ss.backend


def api_health() -> dict[str, Any] | None:
    if _backend() == "embedded":
        e = _embed()
        s = e["get_settings"]()
        # A hosted app can keep an older agent package in memory after an update
        # (Streamlit Cloud reruns this file but doesn't restart the process), so
        # a missing mode_summary must not crash the page: ui_mode then reads "{}"
        # as the keyless wording, which is right for the public demo.
        mode = s.mode_summary() if hasattr(s, "mode_summary") else {}
        return {"status": "ok", "version": e["version"], "keyless": s.is_keyless,
                "mode": mode, "backend": "in-process",
                "limits": {"token_budget": s.token_budget, "max_iterations": s.max_iterations}}
    try:
        return httpx.get(f"{API_URL}/health", timeout=5).json()
    except Exception:  # noqa: BLE001
        return None


def ui_mode(health: dict[str, Any] | None) -> dict[str, Any]:
    """What the backend runs on, for the text that differs between modes.

    ``real``: a paid model is involved (any non-keyless setup). ``web``: sources
    come from live web search rather than local documents. No answer from the
    backend reads as the keyless demo, the public default.
    """
    mode = (health or {}).get("mode") or {}
    web = mode.get("sources") == "web"
    return {
        "real": bool(health) and health.get("keyless") is False,
        "web": web,
        "own_docs": not web and mode.get("corpus") == "custom",
        "llm": mode.get("llm", "rule-based"),
        "meaning": mode.get("evidence") == "meaning",
    }


def md(text: str) -> str:
    """Escape dollar signs for Streamlit markdown, which reads $...$ as LaTeX math.

    Code spans are left alone: a backslash would show there literally.
    """
    parts = text.split("`")
    return "`".join(p if i % 2 else p.replace("$", "\\$") for i, p in enumerate(parts))


def show_md(text: str) -> None:
    st.markdown(md(text))


def sources_label(m: dict[str, Any]) -> str:
    if m["web"]:
        return "the live web"
    return "your own documents" if m["own_docs"] else "the bundled RAG corpus"


def typical_run(m: dict[str, Any]) -> str:
    """What a run takes in this mode, from measured runs (October 2026)."""
    if not m["real"]:
        return "a few milliseconds and $0 (offline)"
    if m["llm"] == "gpt-4o-mini" and m["web"]:
        return "30–40 seconds and roughly $0.0016 (measured with gpt-4o-mini on the live web)"
    if m["llm"] == "gpt-4o-mini":
        return "about 15 seconds and roughly $0.0014 (measured with gpt-4o-mini on local documents)"
    return f"seconds to a minute; the cost depends on the model ({m['llm']})"


def api_get(path: str) -> Any | None:
    if _backend() == "embedded":
        return _embedded_get(path)
    try:
        r = httpx.get(f"{API_URL}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


def _embedded_get(path: str) -> Any | None:
    """In-process equivalents of the API's GET endpoints (same return shapes)."""
    e = _embed()
    if e is None:
        return None
    s = e["get_settings"]()
    if path == "/metrics":
        return e["aggregate"](s.traces_dir)
    if path == "/corpus":
        return {"provider": s.search_provider, "documents": e["list_corpus"](s.corpus_dir)}
    if path == "/runs" or path.startswith("/runs?"):
        from urllib.parse import parse_qs, urlparse
        limit = int(parse_qs(urlparse(path).query).get("limit", ["20"])[0])
        return {"runs": e["recent_runs"](s.traces_dir, limit)}
    if path.startswith("/runs/"):
        res = e["load_run"](path.split("/runs/", 1)[1], s.traces_dir)
        if res is None:
            return None
        data = res.model_dump()
        data["markdown"] = e["render_markdown"](res.report)
        data["citation_coverage"] = round(res.citation_coverage, 4)
        data["support_rate"] = round(e["support_rate"](res.report, res.evidence), 4)
        return data
    return None


def _base_payload() -> dict[str, Any]:
    return {"question": ss.get("q", DEFAULT_Q),
            "max_iterations": int(ss.get("max_iter", 2)),
            "token_budget": int(ss.get("budget", 60_000)),
            "require_approval": bool(ss.get("approval", False))}


def _post_research(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Run one research request; return (data, None) on success or (None, error)."""
    if _backend() == "embedded":
        return _embedded_research(payload)
    try:
        resp = httpx.post(f"{API_URL}/research", json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return None, f"Request failed: {exc}"
    if resp.status_code != 200:
        try:
            d = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            d = resp.text
        return None, f"API error {resp.status_code}: {d}"
    return resp.json(), None


def _embedded_research(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """In-process equivalent of POST /research (runs the pipeline directly)."""
    e = _embed()
    if e is None:
        return None, "Embedded backend unavailable (the agent package could not be imported)."
    settings = e["get_settings"]()
    overrides = e["cap_request"](settings, {
        k: payload[k] for k in
        ("max_iterations", "token_budget", "require_approval", "enable_critic")
        if payload.get(k) is not None})
    try:
        res = e["run"](payload.get("question", ""), settings=settings, **overrides)
    except Exception as exc:  # noqa: BLE001
        return None, f"Research failed: {exc}"
    return {
        "run_id": res.run_id, "status": res.status, "question": res.question,
        "report": res.report.model_dump(), "markdown": e["render_markdown"](res.report),
        "iterations": res.iterations, "tool_calls": res.tool_calls, "tokens": res.tokens,
        "usd": res.usd, "latency_ms": res.latency_ms, "dropped_claims": res.dropped_claims,
        "removed_claims": res.removed_claims,
        "citation_coverage": round(res.citation_coverage, 4),
        "support_rate": round(e["support_rate"](res.report, res.evidence), 4),
    }, None


# --- state mutations (button callbacks) -------------------------------------
def run_research() -> None:
    """Single run from the current widget state; store result + detail in session."""
    with st.spinner("Planning → researching → writing → verifying…"):
        data, err = _post_research({**_base_payload(), "enable_critic": bool(ss.get("critic", True))})
    if err:
        ss.error, ss.result, ss.detail = err, None, None
        return
    ss.error, ss.result = None, data
    ss.detail = api_get(f"/runs/{data['run_id']}")  # full trace + evidence


def run_compare() -> None:
    """Run the same question with the critic ON and OFF — a live demo of the removal mechanism."""
    with st.spinner("Running the critic ON and OFF…"):
        on, err_on = _post_research({**_base_payload(), "enable_critic": True})
        off, err_off = _post_research({**_base_payload(), "enable_critic": False})
    if err_on or err_off:
        ss.error, ss.compare = (err_on or err_off), None
        return
    ss.error, ss.compare = None, {"on": on, "off": off}


def load_run(run_id: str) -> None:
    """Load a persisted run into the view (used by the history browser).

    The enriched GET /runs/{id} response carries both the report-level fields and
    the trace/evidence, so one payload fills every tab exactly like a fresh run.
    """
    d = api_get(f"/runs/{run_id}")
    if d:
        ss.result, ss.detail, ss.error = d, d, None
    else:
        ss.error = f"Could not load run {run_id}."


def pick_example(text: str) -> None:
    ss.q = text.split("  (")[0]  # drop the parenthetical hint


# ----------------------------------------------------------------- render helpers
def coverage_color(pct: float) -> str:
    return "#2f8a3b" if pct >= 0.999 else "#d9822b" if pct >= 0.8 else "#c0392b"


def stat_card(label: str, value: str, accent: str = "var(--stat-default)") -> str:
    return (f'<div class="stat"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val" style="color:{accent}">{value}</div></div>')


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def _claims_count(report: dict[str, Any]) -> int:
    return sum(len(s.get("claims", [])) for s in report.get("sections", []))


def timeline_html(trace: list[dict[str, Any]]) -> str:
    rows = []
    for i, step in enumerate(trace, start=1):
        node = step.get("node", "?")
        icon, color = NODE_STYLE.get(node, ("•", "#888"))
        tool = f" · {step['tool']}" if step.get("tool") else ""
        summary = html.escape(step.get("output_summary", ""))
        revise = "verdict=revise" in summary
        rows.append(
            f'<div class="tl-item{" tl-revise" if revise else ""}" style="border-left-color:{color}">'
            f'<div class="tl-badge" style="background:{color}">{icon}</div>'
            f'<div><div class="tl-head">{i}. {html.escape(node)}{html.escape(tool)} '
            f'<span class="tl-meta">— {step.get("tokens", 0)} tok · {float(step.get("ms", 0)):.1f} ms</span></div>'
            f'<div class="tl-sum">{summary}</div></div></div>'
        )
    return '<div class="tl">' + "".join(rows) + "</div>"


def agent_graph_dot(result: dict[str, Any], trace: list[dict[str, Any]]) -> str:
    """Graphviz DOT of the pipeline with THIS run's executed path highlighted."""
    runs: dict[str, int] = {}
    toks: dict[str, int] = {}
    for step in trace:
        node = step.get("node", "")
        runs[node] = runs.get(node, 0) + 1
        toks[node] = toks.get(node, 0) + int(step.get("tokens", 0))
    executed = set(runs)
    critic_ran = "critic" in executed
    approval_ran = "approval" in executed
    revised = int(result.get("iterations", 0)) > 0
    accept = "approval" if approval_ran else "finalizer"
    TAKEN, GREY = '"#2e6da4"', '"#c9cfd6"'

    def node_stmt(key: str, label: str) -> str:
        accent = NODE_STYLE.get(key, ("", "#888"))[1]
        if key in executed:
            sub = f"{runs[key]}x"
            if toks.get(key):
                sub += f", {toks[key]} tok"
            return (f'{key} [label="{label}\\n{sub}", fillcolor="{NODE_FILL[key]}", '
                    f'color="{accent}", penwidth=2, fontcolor="#26313c"];')
        return (f'{key} [label="{label}", fillcolor="white", color="#cdd3da", '
                f'fontcolor="#9aa4ad"];')

    def edge(a: str, b: str, taken: bool, label: str = "") -> str:
        color = TAKEN if taken else GREY
        parts = [f"color={color}", f'penwidth={"2.4" if taken else "1"}']
        if not taken:
            parts.append("style=dashed")
        if label:
            parts.append(f'label="{label}", fontsize=9, fontcolor={color}')
        return f'{a} -> {b} [{", ".join(parts)}];'

    d = ["digraph G {", "rankdir=TB;", 'bgcolor="transparent";',
         'size="5.5,8"; ratio="compress"; ranksep=0.42; nodesep=0.35;',
         'node [shape=box, style="rounded,filled", fontname="Helvetica", '
         "fontsize=11, width=1.7, height=0.5];",
         'edge [fontname="Helvetica"];',
         'START [shape=oval, fillcolor="#e8f4ea", color="#3a7d44", penwidth=2];',
         'END [shape=oval, fillcolor="#e8f4ea", color="#3a7d44", penwidth=2];']
    d += [node_stmt(k, lbl) for k, lbl in GRAPH_NODES]
    d.append(edge("START", "planner", "planner" in executed))
    d.append(edge("planner", "researcher", "researcher" in executed))
    d.append(edge("researcher", "writer", "writer" in executed))
    if critic_ran:
        d.append(edge("writer", "critic", True))
        d.append(edge("critic", "researcher", revised, label="revise"))
        d.append(edge("critic", accept, True, label="accept"))
        if not approval_ran:
            d.append(edge("critic", "approval", False))
            d.append(edge("approval", "finalizer", False))
    else:
        d.append(edge("writer", "critic", False))
        d.append(edge("critic", "researcher", False, label="revise"))
        d.append(edge("writer", accept, "writer" in executed))
        d.append(edge("critic", "finalizer", False))
        if not approval_ran:
            d.append(edge("writer", "approval", False))
            d.append(edge("approval", "finalizer", False))
    if approval_ran:
        d.append(edge("approval", "finalizer", True))
    d.append(edge("finalizer", "END", "finalizer" in executed))
    d.append("}")
    return "\n".join(d)


def pipeline_dot() -> str:
    """Static structural diagram of the pipeline (for the About page)."""
    def node_stmt(key: str, label: str) -> str:
        return (f'{key} [label="{label}", fillcolor="{NODE_FILL[key]}", '
                f'color="{NODE_STYLE[key][1]}", penwidth=2, fontcolor="#26313c"];')

    d = ["digraph G {", "rankdir=TB;", 'bgcolor="transparent";',
         'size="5.5,8.5"; ranksep=0.45; nodesep=0.35;',
         'node [shape=box, style="rounded,filled", fontname="Helvetica", '
         "fontsize=11, width=1.7, height=0.5];",
         'edge [fontname="Helvetica", color="#2e6da4", penwidth=1.6];',
         'START [shape=oval, fillcolor="#e8f4ea", color="#3a7d44", penwidth=2];',
         'END [shape=oval, fillcolor="#e8f4ea", color="#3a7d44", penwidth=2];']
    d += [node_stmt(k, lbl) for k, lbl in GRAPH_NODES]
    d += [
        "START -> planner;", "planner -> researcher;", "researcher -> writer;",
        "writer -> critic;",
        'critic -> researcher [label="revise", fontsize=9, color="#d9822b", '
        'fontcolor="#d9822b", style=dashed];',
        'critic -> approval [label="accept", fontsize=9];',
        "approval -> finalizer;", "finalizer -> END;",
        'writer -> finalizer [label="budget", fontsize=8, color="#c9cfd6", '
        'fontcolor="#9aa4ad", style=dashed, constraint=false];',
    ]
    d.append("}")
    return "\n".join(d)


def _sync_dark() -> None:
    # Persist the choice in a plain key: widget-keyed state does not survive
    # st.navigation page changes, but a plain session_state key does.
    ss.dark = ss.dark_toggle


def sidebar_health(health: dict[str, Any] | None = None) -> None:
    with st.sidebar:
        st.toggle("Dark mode", value=bool(ss.get("dark", False)),
                  key="dark_toggle", on_change=_sync_dark,
                  help="Switch the whole app between light and dark themes.")
        health = health or api_health()
        if not health:
            st.error(f"API not reachable at {API_URL}")
            return
        m = ui_mode(health)
        sources = ("live web search" if m["web"]
                   else "your own documents" if m["own_docs"] else "bundled corpus")
        evidence = "evidence by meaning" if m["meaning"] else "evidence by words"
        backend = "In-process backend" if health.get("backend") == "in-process" else "API up"
        title = (f"**Real mode** · {m['llm']}" if m["real"]
                 else "**Keyless demo** · offline · $0")
        st.success(md(f"{title}  \n{sources}  \n{evidence}  \n"
                      f"{backend} · v{health.get('version', '?')}"))


def settings_sidebar(show_critic: bool = True) -> dict[str, Any] | None:
    """The run-settings sidebar; returns the backend's health for mode-aware text."""
    health = api_health()
    m = ui_mode(health)
    # In real mode the server's TOKEN_BUDGET / MAX_ITERATIONS are a ceiling (it
    # lowers any request above them), so the inputs stop there too.
    limits = (health or {}).get("limits") or {}
    capped = bool(limits) and health.get("keyless") is False
    max_iter = int(limits["max_iterations"]) if capped else 5
    max_budget = int(limits["token_budget"]) if capped else 500_000
    ss.max_iter = min(int(ss.max_iter), max_iter)
    ss.budget = min(int(ss.budget), max_budget)
    with st.sidebar:
        st.header("Run settings")
        iter_help = ("How many times the critic may send the draft back to be rewritten. "
                     "0 = check once, no rewrite.")
        if m["real"]:
            iter_help += " Each rewrite is one more writer and critic call."
        if max_iter > 0:
            st.slider("Max critic iterations", 0, max_iter, key="max_iter", help=iter_help)
        else:
            st.caption("Max critic iterations: 0 (set by the server)")
        st.number_input("Token budget", min_value=min(100, max_budget), max_value=max_budget,
                        step=1_000, key="budget",
                        help="The most tokens one run may use. At the limit the run stops "
                             "cleanly and the report is marked partial."
                             + (" Embedding tokens count too." if m["meaning"] else ""))
        if capped:
            st.caption(f"This server allows up to {max_budget:,} tokens and {max_iter} "
                       "critic iteration(s) per run (its TOKEN_BUDGET and MAX_ITERATIONS).")
        if show_critic:
            st.toggle("Enable verifying critic", key="critic",
                      help=("OFF skips the check: faster and cheaper, but a claim its "
                            "evidence doesn't support stays in the report. Citations to "
                            "ungathered sources are still removed." if m["real"] else
                            "OFF = the critic-OFF arm of the A/B: the keyless writer's "
                            "planted uncited claim survives, so coverage drops."))
        st.checkbox("Require human approval", key="approval",
                    help="This app has no interactive approver, so runs auto-approve (an "
                         "approval step still appears in the trace). A real deny is only "
                         "possible via the Python API's approval_fn.")
        st.divider()
    sidebar_health(health)
    return health


def render_result(r: dict[str, Any], detail: dict[str, Any]) -> None:
    """The metrics band + tabbed report/evidence/corpus/timeline/graph/raw view."""
    report = r["report"]
    cov = float(r["citation_coverage"])
    band = "".join([
        stat_card("Status", r["status"], STATUS_ACCENT.get(r["status"], "#1b3a5c")),
        stat_card("Iterations", str(r["iterations"])),
        stat_card("Tool calls", str(r["tool_calls"])),
        stat_card("Tokens", f"{r['tokens']:,}"),
        stat_card("Citation coverage", f"{cov:.0%}", coverage_color(cov)),
    ])
    st.markdown(f'<div class="stat-row">{band}</div>', unsafe_allow_html=True)
    critic_ran = any(s.get("node") == "critic" for s in detail.get("trace", []))
    removed = detail.get("removed_claims") or r.get("removed_claims") or []
    sup = r.get("support_rate")
    sup_s = f" · support {sup:.0%}" if isinstance(sup, (int, float)) else ""
    removed_s = f" · removed by critic {len(removed)}" if critic_ran else ""
    st.markdown(
        f'<div class="runmeta">Cost ${r["usd"]:.4f} · latency {r["latency_ms"]:.0f} ms · '
        f'dropped claims {r["dropped_claims"]}{removed_s}{sup_s} · critic '
        f'{"ran" if critic_ran else "off"} · run_id <code>{r["run_id"]}</code></div>',
        unsafe_allow_html=True,
    )

    tab_report, tab_evidence, tab_corpus, tab_timeline, tab_graph, tab_raw = st.tabs(
        [":material/description: Report", ":material/format_quote: Evidence & sources",
         ":material/folder_open: Corpus", ":material/timeline: Step timeline",
         ":material/account_tree: Agent graph", ":material/data_object: Run data"]
    )

    with tab_report:
        st.markdown(report.get("markdown") or r["markdown"])
        if removed:
            with st.expander(f"Removed by the critic ({len(removed)})"):
                st.caption("Claims an earlier draft made that the critic found unsupported "
                           "by their cited evidence. None of them are in the report above.")
                for text in removed:
                    st.markdown(f"- {text}")
        d1, d2 = st.columns(2)
        d1.download_button(":material/download: Download report (Markdown)", r["markdown"],
                           file_name=f"report_{r['run_id']}.md", mime="text/markdown",
                           use_container_width=True)
        d2.download_button(":material/download: Download run (JSON)",
                           json.dumps(detail or r, indent=2, default=str),
                           file_name=f"run_{r['run_id']}.json", mime="application/json",
                           use_container_width=True)

    with tab_evidence:
        cited_by: dict[str, list[str]] = {}
        for sec in report.get("sections", []):
            for claim in sec.get("claims", []):
                for eid in claim.get("evidence_ids", []):
                    cited_by.setdefault(eid, []).append(claim["text"])
        evidence = detail.get("evidence", [])
        if not evidence:
            st.info("No evidence was gathered — the assistant abstained on this question.")
        else:
            st.caption(f"{len(evidence)} evidence item(s) gathered. "
                       "Every citation in the report maps to one of these.")
            for ev in evidence:
                uses = cited_by.get(ev["id"], [])
                tag = f"cited by {len(uses)}" if uses else "not cited"
                with st.expander(f"{ev['id']} · {ev['source_title']}  —  {tag}"):
                    st.caption(ev["source_url"])
                    st.write(ev["snippet"])
                    if uses:
                        st.markdown("**Cited by:**")
                        for t in uses:
                            st.markdown(f"- {t}")

    with tab_corpus:
        corpus_info = api_get("/corpus")
        docs = (corpus_info or {}).get("documents", [])
        if not docs:
            st.caption("Corpus listing unavailable.")
        elif corpus_info.get("provider") != "fake":
            st.caption("This instance searches the live web, so there is no fixed set of "
                       "documents to show coverage for.")
        else:
            used = {ev.get("source_url") for ev in detail.get("evidence", [])}
            n_used = sum(1 for d in docs if d["url"] in used)
            st.caption(
                f"This run drew on **{n_used} of {len(docs)}** corpus documents. "
                "Green = a document that fed a claim; grey = not retrieved for this "
                "question — an at-a-glance view of retrieval coverage."
            )
            rows = []
            for d in docs:
                is_used = d["url"] in used
                dot = "#2f8a3b" if is_used else "#cdd3da"
                txt = "var(--text)" if is_used else "#9aa4ad"
                rows.append(
                    f'<div style="padding:5px 2px">'
                    f'<span style="color:{dot};font-size:1.15rem">&#9679;</span> '
                    f'<span style="color:{txt};font-weight:{600 if is_used else 400}">'
                    f'{html.escape(d["title"])}</span> '
                    f'<code style="color:#7a8896;font-size:0.82rem">{html.escape(d["url"])}</code>'
                    f'<span style="color:{txt};font-size:0.78rem"> — '
                    f'{"used" if is_used else "not used"}</span></div>'
                )
            st.markdown("".join(rows), unsafe_allow_html=True)

    with tab_timeline:
        trace = detail.get("trace", [])
        if trace:
            st.markdown(timeline_html(trace), unsafe_allow_html=True)
        else:
            st.caption("Timeline unavailable (could not load the run detail).")

    with tab_graph:
        trace = detail.get("trace", [])
        if trace:
            st.caption(
                "The compiled LangGraph pipeline. **Filled** nodes ran on this "
                "request (with run-count and tokens); the **blue** path is the route "
                "taken. The `revise` edge (critic → researcher) lights up only when "
                "the critic sent the draft back for another pass."
            )
            st.graphviz_chart(agent_graph_dot(r, trace), use_container_width=False)
        else:
            st.caption("Graph unavailable (could not load the run detail).")

    with tab_raw:
        st.json(detail or r)


def render_compare(cmp: dict[str, Any]) -> None:
    """Side-by-side critic ON vs OFF for the same question, with the deltas."""
    on, off = cmp["on"], cmp["off"]

    def drow(label: str, a: float, b: float, pct: bool = False,
             higher_better: bool = True) -> str:
        fa = f"{a:.0%}" if pct else f"{a:,}"
        fb = f"{b:.0%}" if pct else f"{b:,}"
        delta = a - b
        if delta == 0:
            dc, ds = "#9aa4ad", "±0"
        else:
            dc = "#2f8a3b" if (delta > 0) == higher_better else "#c0392b"
            ds = f"{delta:+.0%}" if pct else f"{delta:+,}"
        cell = "padding:6px 10px;text-align:center;font-weight:700"
        return (f'<tr><td style="padding:6px 10px">{label}</td>'
                f'<td style="{cell}">{fa}</td><td style="{cell}">{fb}</td>'
                f'<td style="{cell};color:{dc};font-weight:800">{ds}</td></tr>')

    rows = "".join([
        drow("Citation coverage", on["citation_coverage"], off["citation_coverage"], pct=True),
        drow("Support rate", on.get("support_rate", 0.0), off.get("support_rate", 0.0), pct=True),
        drow("Claims in report", _claims_count(on["report"]), _claims_count(off["report"]),
             higher_better=False),
        drow("Tokens used", on["tokens"], off["tokens"], higher_better=False),
    ])
    st.markdown(
        '<table style="width:100%;border-collapse:collapse;margin:4px 0 14px 0">'
        '<thead><tr style="background:#1b3a5c;color:#fff">'
        '<th style="padding:8px 10px;text-align:left">Metric</th>'
        '<th style="padding:8px 10px">Critic ON</th>'
        '<th style="padding:8px 10px">Critic OFF</th>'
        '<th style="padding:8px 10px">&Delta; (ON&minus;OFF)</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )

    for col, res, title, tint in zip(
        st.columns(2), (on, off), ("Critic ON", "Critic OFF"),
        ("#2f8a3b", "#d9822b"), strict=True,
    ):
        with col:
            cov = float(res["citation_coverage"])
            st.markdown(
                f'<div style="border-top:4px solid {tint};background:var(--card-bg);'
                f'color:var(--text);border-radius:8px;padding:8px 12px;margin-bottom:8px">'
                f'<b>{title}</b>'
                f' · coverage <span style="color:{coverage_color(cov)};font-weight:800">'
                f'{cov:.0%}</span> · {_claims_count(res["report"])} claims · '
                f'status {res["status"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(res["markdown"])


# ---------------------------------------------------------------------------- pages
def page_research() -> None:
    m = ui_mode(settings_sidebar(show_critic=True))
    if m["real"]:
        intro = (f"Real mode: <b>{html.escape(m['llm'])}</b> plans, writes and checks the "
                 f"report, using sources from {sources_label(m)}. Every citation points to a "
                 "source that was actually retrieved.")
    else:
        intro = ("Multi-agent (LangGraph) research with <b>cited</b> reports. This is the "
                 "keyless demo: rule-based agents over a bundled corpus, offline and free. "
                 "Every source is real; none are fabricated.")
    hero("Agentic Research &amp; Report Assistant", intro)
    st.text_input("Research question", key="q")
    if m["web"]:
        hint = "Try an example, or ask about any topic:"
    elif m["own_docs"]:
        hint = "Ask about your documents (the examples cover the bundled RAG corpus):"
    else:
        hint = "Try an example (the corpus covers RAG topics):"
    st.markdown(f'<div class="chip-hint">{hint}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        # The "abstains" hint is only true when the sources are a fixed corpus.
        label = ex.split("  (")[0] if m["web"] else ex
        cols[i % 3].button(label, key=f"ex{i}", on_click=pick_example, args=(ex,),
                           use_container_width=True)
    st.button("Run research", type="primary", on_click=run_research, use_container_width=True)

    if ss.error:
        st.error(ss.error)
    if ss.result:
        render_result(ss.result, ss.detail or {})
    elif m["real"]:
        st.info(md("Enter a question (or pick an example) and click **Run research**. A "
                   f"run takes {typical_run(m)}. New to real mode? See **Guide → Real mode**."))
    else:
        st.info("Enter a question (or pick an example) and click **Run research**. "
                "To see the critic remove a planted uncited claim, open the **Critic A/B** "
                "page in the sidebar.")


def page_compare() -> None:
    m = ui_mode(settings_sidebar(show_critic=False))
    if m["real"]:
        hero("Critic A/B",
             "Run the same question with the critic <b>on</b> and <b>off</b>, side by side. "
             "With a real model there is no planted claim: a real writer usually cites what "
             "it says, and on the evaluation set the measured difference was about zero. "
             "One click makes <b>two paid runs</b>.")
    else:
        hero("Critic A/B",
             "Run the same question with the critic <b>on</b> and <b>off</b>, side by "
             "side — a live demonstration of the claim-removal mechanism (keyless mode "
             "plants one uncited claim for the critic to catch).")
    st.text_input("Research question", key="q")
    st.button("Run A/B comparison", type="primary", on_click=run_compare,
              use_container_width=True)
    if not m["real"]:
        st.caption("The critic removes uncited / unsupported claims. With it OFF the "
                   "deliberately-uncited synthesis claim survives — visible in the "
                   "right-hand report and in the lower citation coverage.")
    elif m["web"]:
        st.caption("Each arm searches the live web separately, so the two can find "
                   "different sources: part of any difference is the search, not the critic.")
    else:
        st.caption("Both arms search the same local documents, so a difference comes from the "
                   "critic, plus the model's run-to-run variation.")

    if ss.error:
        st.error(ss.error)
    if ss.compare:
        render_compare(ss.compare)
    elif m["real"]:
        st.info(md("Enter a question and click **Run A/B comparison**. It runs the "
                   f"pipeline twice; each run takes {typical_run(m)}."))
    else:
        st.info("Enter a question and click **Run A/B comparison** to see the "
                "citation-coverage and support delta between critic ON and OFF.")


def page_history() -> None:
    health = api_health()
    sidebar_health(health)
    hero("Run history",
         "Browse and reopen any past run — full report, evidence, and trace.")
    runs = (api_get("/runs?limit=50") or {}).get("runs", [])
    if not runs:
        st.info("No runs recorded yet. Run a research query to populate history.")
        return
    shared = (" Runs are stored on this server: anyone who can open this app can see them."
              if ui_mode(health)["real"] else "")
    st.caption(f"{len(runs)} most recent run(s). Click **View** to reopen one below.{shared}")
    head = st.columns([6, 2, 2, 2, 1.6])
    for c, label in zip(head, ["Question", "Status", "Coverage", "Tokens", ""], strict=True):
        c.markdown(f"**{label}**" if label else "")
    for it in runs:
        c = st.columns([6, 2, 2, 2, 1.6])
        q = (it.get("question") or "(no question)").strip()
        c[0].write(q[:70] + ("…" if len(q) > 70 else ""))
        c[1].write(it.get("status", "?"))
        cov = it.get("citation_coverage")
        c[2].write(f"{cov:.0%}" if isinstance(cov, (int, float)) else "—")
        c[3].write(f"{it.get('tokens', 0):,}")
        c[4].button("View", key=f"view_{it.get('run_id')}", on_click=load_run,
                    args=(it.get("run_id"),), use_container_width=True)

    if ss.error:
        st.error(ss.error)
    if ss.result:
        st.divider()
        st.subheader("Selected run")
        render_result(ss.result, ss.detail or {})


def page_observability() -> None:
    sidebar_health()
    hero("Observability",
         "Aggregate metrics across every persisted run — the first-class "
         "observability story, in one place.")
    agg = api_get("/metrics")
    if not agg or not agg.get("runs"):
        st.info("No runs recorded yet — run some research to populate metrics.")
        return
    r1 = st.columns(3)
    r1[0].metric("Total runs", agg["runs"])
    r1[1].metric("Avg cost / run", f"${agg['avg_cost_usd']:.4f}")
    r1[2].metric("Avg steps / run", f"{agg['avg_steps']:.1f}")
    r2 = st.columns(3)
    r2[0].metric("Avg latency", f"{agg['avg_latency_ms']:.0f} ms")
    r2[1].metric("p95 latency", f"{agg['p95_latency_ms']:.0f} ms")
    r2[2].metric("Avg citation coverage", f"{agg['avg_citation_coverage']:.0%}")

    st.divider()
    st.subheader("Recent runs")
    runs = (api_get("/runs?limit=20") or {}).get("runs", [])
    if not runs:
        st.caption("No runs yet.")
        return
    body = []
    for it in runs:
        q = html.escape((it.get("question") or "(no question)")[:60])
        cov = it.get("citation_coverage")
        cov_s = f"{cov:.0%}" if isinstance(cov, (int, float)) else "—"
        body.append(
            "<tr>"
            f"<td style='padding:6px 8px'>{q}</td>"
            f"<td style='padding:6px 8px'>{html.escape(str(it.get('status', '?')))}</td>"
            f"<td style='padding:6px 8px;text-align:center'>{cov_s}</td>"
            f"<td style='padding:6px 8px;text-align:right'>{it.get('tokens', 0):,}</td>"
            f"<td style='padding:6px 8px;text-align:right'>{float(it.get('latency_ms', 0)):.0f} ms</td>"
            "</tr>"
        )
    st.markdown(
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr style='background:#1b3a5c;color:#fff'>"
        "<th style='padding:8px;text-align:left'>Question</th>"
        "<th style='padding:8px;text-align:left'>Status</th>"
        "<th style='padding:8px'>Coverage</th>"
        "<th style='padding:8px;text-align:right'>Tokens</th>"
        "<th style='padding:8px;text-align:right'>Latency</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>",
        unsafe_allow_html=True,
    )


# The Guide's evaluation table: 12 golden tasks (eval/tasks.jsonl). Keyless is the
# CI baseline; the real-mode columns were measured with gpt-4o-mini (trafilatura
# article text, passages ranked by meaning) — see REAL_MODE.md for the method.
REAL_EVAL_DATE = "9 October 2026"
REAL_EVAL_ROWS = (
    "| **citation_coverage** | % of claims carrying ≥1 citation | 1.00 | 1.00 | 1.00 |\n"
    "| **source_validity** | % of citations whose source was *actually gathered* — "
    "catches fabrication | 1.00 | 1.00 | 1.00 |\n"
    "| **support_rate** | % of claims whose cited passage shares enough words with them "
    "(≥ 0.3 overlap) — a lexical proxy, not a semantic check | 1.00 | 0.93 | 1.00 |\n"
    "| **point_coverage** | % of each task's expected key facts present in the report "
    "| 0.90 | 0.87 | 0.67 |\n"
    "| **abstention_accuracy** | off-topic questions must produce zero claims "
    "| 1.00 | 1.00 | 0.00 |\n"
    "| avg cost per run | | $0 | $0.0014 | $0.0016 |\n"
    "| avg time per run | | ~10 ms | 16 s | 33 s |\n"
)


def _corpus_list_md(m: dict[str, Any]) -> str:
    """The Concepts tab's "Sources" section, for this instance's mode."""
    docs = (api_get("/corpus") or {}).get("documents", [])
    if m["web"]:
        n = f"{len(docs)} " if docs else ""
        return ("This instance searches the **live web** (Tavily) for every question, so "
                "there is no fixed list of documents: the sources are whatever the searches "
                f"find. The keyless demo instead searches a bundled corpus of {n}Markdown "
                "documents on RAG topics.")
    if not docs:
        return "_Document listing unavailable._"
    whose = "your own" if m["own_docs"] else "bundled"
    lines = [f"{len(docs)} {whose} documents back the search and fetch tools:\n"]
    lines += [f"- **{html.escape(d['title'])}** &nbsp;`{html.escape(d['url'])}`" for d in docs]
    return "\n".join(lines)


def _you_are_on(m: dict[str, Any]) -> str:
    if m["real"]:
        return (f"You're on **real mode**: {m['llm']}, sources from {sources_label(m)}, "
                f"evidence picked by {'meaning' if m['meaning'] else 'words'}.")
    return "You're on the **keyless demo**: rule-based agents, bundled corpus, offline, $0."


def _limits_md(health: dict[str, Any] | None, m: dict[str, Any]) -> str:
    limits = (health or {}).get("limits") or {}
    if not (m["real"] and limits):
        return ""
    return (f"- **This server's ceiling:** {limits['token_budget']:,} tokens and "
            f"{limits['max_iterations']} critic iteration(s) per run (its `TOKEN_BUDGET` and "
            "`MAX_ITERATIONS`). The sidebar can lower them, not raise them; to go higher, "
            "change them in the server's `.env` and run `docker compose up -d`.\n")


def page_guide() -> None:
    health = api_health()
    sidebar_health(health)
    m = ui_mode(health)
    hero("Guide",
         "How the app works, what every number means, and how far to trust a report. "
         "New here? Start with <b>Getting started</b>; the <b>Real mode</b> tab covers "
         "the paid, live-web version.")

    t_start, t_real, t_read, t_obs, t_eval, t_concepts, t_faq = st.tabs(
        [":material/rocket_launch: Getting started", ":material/public: Real mode",
         ":material/description: Reading a report", ":material/monitoring: Observability",
         ":material/task_alt: Evaluation", ":material/school: Concepts",
         ":material/help: FAQ"]
    )

    with t_start:
        if m["web"]:
            step1 = ("Any topic works: the planner splits your question into 3–6 "
                     "sub-questions, and the researcher searches the web for each.")
        elif m["own_docs"]:
            step1 = "Ask about the documents this instance was pointed at (`CORPUS_DIR`)."
        else:
            step1 = ("The bundled corpus covers RAG topics (embeddings, chunking, hybrid "
                     "search, reranking, evaluation, hallucination, agentic RAG…).")
        left, right = st.columns([1.15, 1])
        with left:
            show_md(
                "### What this app does\n"
                "It answers a research question **using only sources it actually "
                "retrieved**. A LangGraph pipeline plans the research, gathers evidence with "
                "tools, drafts a report that cites **only** what was gathered, and has a "
                "**critic check each claim against its cited evidence**. That's "
                "*agentic Retrieval-Augmented Generation (RAG)*.\n\n"
                "### Two modes\n"
                "- **Keyless demo** — rule-based agents over a bundled corpus of RAG "
                "documents: offline, deterministic, free. This is what the public demo runs.\n"
                "- **Real mode** — an OpenAI model plans, writes and checks, and the sources "
                "come from the live web (or your own documents). Each run costs a fraction "
                "of a cent. See the **Real mode** tab.\n\n"
                f"{_you_are_on(m)} The box at the bottom of the sidebar always shows the "
                "mode.\n\n"
                "### Your first question\n"
                f"1. On the **Research** page, type a question or click an example. {step1}\n"
                "2. Click **Run research**. The pipeline plans → researches → writes → "
                f"checks; it takes {typical_run(m)}.\n"
                "3. Read the answer: the **[1] [2]** markers are citations; the **Sources** "
                "list and the **Evidence & sources** tab show the passages they came from.\n"
                "4. Explore the numbers: **Observability** shows how the system is *running* "
                "(cost, latency, tokens); the **Evaluation** tab here explains how *good* "
                "the answers are.\n\n"
                "### Run settings (sidebar)\n"
                "| Setting | What it does |\n|---|---|\n"
                "| **Max critic iterations** | how many times the critic may send the draft "
                "back to be rewritten; 0 = check once, no rewrite |\n"
                "| **Token budget** | the most tokens one run may use; at the limit the run "
                "stops cleanly and is marked `partial` |\n"
                "| **Enable verifying critic** | OFF skips the claim check, so unsupported "
                "claims stay in the report; in real mode it also saves the critic's model "
                "calls |\n"
                "| **Require human approval** | adds an approval step; this app has no "
                "interactive approver, so it auto-approves |\n\n"
                + ("In real mode the server's own `TOKEN_BUDGET` and `MAX_ITERATIONS` are the "
                   "ceiling for the first two.\n\n" if m["real"] else "")
                + "### The pages\n"
                "- **Research** — ask a question and read the cited report (six result tabs).\n"
                "- **Critic A/B** — run the same question with the critic on and off, side by "
                "side.\n"
                "- **History** — browse and reopen any past run.\n"
                "- **Observability** — cost / latency / coverage across all runs.\n"
                "- **Guide** — this page."
            )
        with right:
            st.graphviz_chart(pipeline_dot(), use_container_width=False)
            st.caption("The pipeline. **Real mode → What a real run does** describes each "
                       "step.")

    with t_real:
        if not m["real"]:
            st.info("You're on the keyless demo, so none of this costs anything here. This "
                    "tab explains what changes when the app runs on a real model and the live "
                    "web: how to set that up is in REAL_MODE.md in the GitHub repository.")
        show_md(
            "### Real mode vs the keyless demo\n"
            "| | Keyless demo | Real mode (live web) |\n|---|---|---|\n"
            "| Planner, writer, critic | rule-based code | an OpenAI model "
            "(`gpt-4o-mini` by default) |\n"
            "| Sub-questions | 4 fixed facets | 3–6, written for your question |\n"
            "| Sources | a bundled corpus (or your own documents) | live web search "
            "(Tavily) |\n"
            "| Reading a source | the corpus file | the page's article text (trafilatura); "
            "pages with none, such as videos and login walls, are skipped |\n"
            "| Evidence kept per source | the 2 sentences sharing the most words with the "
            "search query | a passage of a few sentences that best answers the sub-question, "
            "ranked by meaning |\n"
            "| Time per run | milliseconds | 30–40 seconds |\n"
            "| Cost per run | $0 | roughly $0.0016 with gpt-4o-mini |\n"
            "| Same question twice | identical report | can differ (live search, model "
            "variation) |\n"
            "| Off-topic question | abstains with an empty report | answered from the web |\n\n"
            "A real model can also run over local documents instead of the web: then the "
            "sources and the abstention behave like the keyless column.\n\n"
            "### What a real run does\n"
            "1. **Planner** (LLM) splits the question into 3–6 sub-questions, each with "
            "search queries.\n"
            "2. **Researcher** (no LLM) searches for each sub-question (5 results per search "
            "by default), fetches the pages in parallel, keeps each page's article text and "
            "picks the passage that best answers the sub-question, comparing meaning with an "
            "embedding model. It keeps up to 2 sources per sub-question; a page that fails "
            "(blocked, no article text, a video) is skipped for the next result.\n"
            "3. **Writer** (LLM) writes the report from that evidence only; every claim cites "
            "evidence ids.\n"
            "4. **Critic** (LLM) checks each claim against its cited evidence. Unsupported "
            "claims are removed (listed under the report as *Removed by the critic*), and "
            "the writer may rewrite, up to the iteration limit.\n"
            "5. **Finalizer** (no LLM) strips any citation to a source that wasn't gathered, "
            "numbers the sources, and sets the status.\n\n"
            "The model is told to treat fetched text as data and to ignore any instructions "
            "hidden in it.\n\n"
            "### Cost and limits\n"
            "- **A typical web run:** about 20,000 tokens and roughly $0.0016 with "
            "gpt-4o-mini (the average over 12 runs, October 2026). More than half the "
            "tokens are embedding "
            "tokens for picking passages ($0.02 per 1M); the LLM calls ($0.15 per 1M input, "
            "$0.60 per 1M output for gpt-4o-mini) are most of the cost.\n"
            "- **Each critic rewrite** adds one writer and one critic call.\n"
            + _limits_md(health, m) +
            "- **At the token budget** the run stops cleanly and is marked `partial`.\n"
            "- **Web search** uses Tavily: a run makes about 5 searches, and its free plan "
            "gives 1,000 credits a month (one per basic search).\n"
            "- **Every run's cost** is on its meta line; the Observability page averages "
            "them.\n\n"
            "### Writing good questions\n"
            "- One topic per question, with the context that matters (\"for a small team\", "
            "\"in Python\", \"as of 2026\").\n"
            "- Comparisons work well: \"X vs Y for Z\" gives the planner clear sub-questions.\n"
            "- For recent events, the report is only as current as the pages it finds: check "
            "the dates on the sources.\n"
            "- It can't read paywalled or login-only pages, videos, or sites that block bots "
            "(Medium and some blogs answer with 403).\n\n"
            "### How far to trust a report\n"
            "- **Guaranteed by code:** every [n] points to a source that was actually "
            "fetched. Citations to anything else are stripped before you see the report.\n"
            "- **Not guaranteed:** that a claim says exactly what its source says. The writer "
            "sees one passage per source, not the whole page, and the critic is itself a "
            "language model: it can miss a weak claim or remove a good one.\n"
            "- Sources are whatever the search returns, blogs and vendor pages included; the "
            "app doesn't rank them by authority.\n"
            "- Before relying on a claim, open its source.\n\n"
            "### Privacy\n"
            "- Your question and the gathered page text are sent to OpenAI, and the search "
            "queries to Tavily. Don't include passwords, secrets or personal data.\n"
            "- Pages are fetched from the server running the app, which identifies itself "
            "as this project's bot.\n"
            "- Run history is stored on that server: anyone who can open the app can see it "
            "on the **History** page."
        )

    with t_read:
        if m["real"]:
            tokens_row = ("tokens the run used: the model-reported LLM and embedding tokens, "
                          "plus an estimate for the gathered text; all of them count toward "
                          "the token budget")
        else:
            tokens_row = "a deterministic estimate (~4 characters per token); keyless tokens are free"
        if m["web"]:
            abstain = ("With **web** sources there is no topic limit: almost any question is "
                       "answered from the web. A report with few or no claims means the "
                       "searches found little the fetcher could read; the **Step timeline** "
                       "shows which fetches failed.")
        else:
            abstain = ("Ask something outside the documents (e.g. *“How do I bake sourdough "
                       "bread?”* on the bundled corpus) and you'll get **0 sources, 0% "
                       "coverage, an empty report**. That is the system correctly declining "
                       "to answer rather than inventing facts — not a bug.")
        picked = ("a passage of a few sentences that best answers the sub-question, "
                  "ranked by meaning with an embedding model" if m["meaning"] else
                  "the 2 sentences that share the most words with the search query")
        show_md(
            "### The metrics band\n"
            "Every answer opens with five cards and a meta line:\n\n"
            "| Field | Meaning |\n|---|---|\n"
            "| **Status** | `complete` · `partial` (it hit the token budget, or the critic "
            "still wanted a rewrite when the iterations ran out — see **FAQ**) · "
            "`awaiting_approval` |\n"
            "| **Iterations** | how many times the critic sent the draft back to be "
            "rewritten |\n"
            "| **Tool calls** | `search` + `fetch` calls made while gathering evidence |\n"
            f"| **Tokens** | {tokens_row} |\n"
            "| **Citation coverage** | % of claims carrying ≥1 citation — green at 100%, "
            "amber ≥80%, red below |\n"
            "| meta line | cost · latency · dropped claims · removed by critic · support rate "
            "· whether the critic ran · the `run_id` |\n\n"
            "### Citations, evidence and sources\n"
            "Every bullet in the report ends in a **[n]** marker. The numbered **Sources** "
            "list maps each **[n]** to a source that was actually gathered, and the "
            "**Evidence & sources** tab expands each passage and shows *which claims cite "
            "it*. A citation can never point at something the system didn't retrieve.\n\n"
            f"Each evidence item is one passage from one source: {picked}. An item marked "
            "**not cited** is normal: the writer only cites evidence that supports a claim, "
            "so a weak or off-topic passage is left out instead of being forced into the "
            "report. The **Sources** list shows only the cited ones.\n\n"
            "### Removed by critic vs dropped claims\n"
            "- **Removed by critic** — claims a draft made that the critic found unsupported "
            "by their cited evidence. They are listed under the report and are not in it.\n"
            "- **Dropped claims** — claims whose every citation pointed at a source that "
            "wasn't gathered. The citation guard removes them at the end; normally 0.\n"
            "- **Support rate** — the share of claims whose cited passage shares enough "
            "words with them (a quick lexical estimate, not proof). A model that paraphrases "
            "its sources scores lower here even when it is faithful.\n\n"
            "### The step timeline\n"
            "| Step | What it is |\n|---|---|\n"
            "| planner | the question split into sub-questions |\n"
            "| researcher · search | one search query |\n"
            "| researcher · fetch | one source read. `fetch failed: …` lines are normal: "
            "`403 Forbidden` (the site blocks bots), `no readable article text`, or `video "
            "page`; the next result is used |\n"
            "| researcher · rank | real mode: the embedding call that picked that source's "
            "passage (its cost is included) |\n"
            "| writer | a draft of the report |\n"
            "| critic | the check; `verdict=revise` (highlighted) asks for a rewrite, which "
            "happens if iterations and budget are left |\n"
            "| approval | only when *Require human approval* is on |\n"
            "| finalizer | the citation guard, source numbering, and the status |\n\n"
            "### The six result tabs\n"
            "- **Report** — the cited answer, any claims the critic removed, and Markdown / "
            "JSON downloads.\n"
            "- **Evidence & sources** — each gathered passage, its source, and the claims "
            "that cite it.\n"
            "- **Corpus** — with local documents: which ones this question drew on (used vs "
            "not retrieved). With web sources there is no corpus, and the tab says so.\n"
            "- **Step timeline** — every step in order, with its tokens and latency.\n"
            "- **Agent graph** — the pipeline with *this run's* path highlighted; the "
            "`revise` edge lights up when the critic looped.\n"
            "- **Run data** — the raw run JSON.\n\n"
            "### Abstention (honest refusal)\n"
            f"{abstain}"
        )

    with t_obs:
        show_md(
            "### Observability — how the system is *running*\n"
            "The **Observability** page aggregates every run this app has stored (from the "
            "`/metrics` endpoint):\n\n"
            "| Metric | What it means | Keyless demo | Real mode, web (gpt-4o-mini) |\n"
            "|---|---|---|---|\n"
            "| **Total runs** | runs in this app's run index | | |\n"
            "| **Avg cost / run** | mean USD per run | $0.0000 | about $0.0016 |\n"
            "| **Avg latency** | mean wall-clock time per run | milliseconds | 30–40 s |\n"
            "| **p95 latency** | 95th-percentile latency (nearest-rank: no interpolation, "
            "so one slow run doesn't distort a small sample) | | |\n"
            "| **Avg steps / run** | mean number of timeline steps (agents, tool calls and, "
            "in real mode, rank steps) | about 15 | about 33 |\n"
            "| **Avg citation coverage** | mean share of claims carrying a citation | 100% "
            "| close to 100% |\n\n"
            "The two value columns are typical values from measured runs (October 2026), "
            "not this app's own numbers: those are on the Observability page.\n\n"
            "Cost is honest by construction: the keyless model is priced at `$0`, so a "
            "keyless run genuinely reports zero. A real model is priced at list price from "
            "the provider's own token counts, input and output separately. Search and fetch "
            "steps carry no USD (the model is billed for that text when it reads it); "
            "real mode's `rank` steps are priced at the embedding model's rate.\n\n"
            "### Where the history is kept\n"
            "Runs are saved by the backend. On a Docker deployment they live in a Docker "
            "volume and survive restarts and updates. On the public demo (Streamlit Community "
            "Cloud) they live on the app's temporary disk, so **History** and "
            "**Observability** start empty again whenever the app restarts."
        )

    with t_eval:
        show_md(
            "### Evaluation — how *good* the grounded answers are\n"
            "An offline harness runs 12 golden tasks: 10 answerable from the bundled corpus "
            "and 2 deliberately off-topic. The keyless run is a CI gate on every push; the "
            "real-mode columns were measured with gpt-4o-mini on " + REAL_EVAL_DATE + ".\n\n"
            "| Metric | What it measures | Keyless | Real model, bundled corpus | Real model, "
            "live web |\n|---|---|---|---|---|\n"
            + REAL_EVAL_ROWS +
            "| **faithfulness** | LLM-as-judge — **not implemented yet** (only its "
            "scaffolding exists), so it is never reported as a number | n/a | n/a | n/a |\n\n"
            "**Reading the real-mode columns**\n"
            "- **source_validity is 1.00 everywhere:** the no-fabricated-sources guarantee "
            "is code, so it holds for any model and any source.\n"
            "- **support_rate** is a word-overlap estimate, so read it as a rough signal: a "
            "model that paraphrases can score lower while staying faithful (0.93 over the "
            "corpus), and longer passages make overlap easier (1.00 on the web).\n"
            "- **point_coverage** checks for key facts phrased from the bundled corpus; web "
            "pages word things differently, so it is not meaningful in the web column.\n"
            "- **abstention_accuracy** is corpus-relative. On the live web the two "
            "\"off-topic\" questions (the capital of France, baking sourdough) are "
            "answerable, and the assistant answers them, by design.\n\n"
            "### The critic A/B — what it shows, and what it doesn't\n"
            "Running the same tasks with the **critic ON vs OFF** shows the removal "
            "mechanism working: in keyless mode the fake writer deliberately adds one "
            "uncited claim, so with the critic OFF it survives and **citation coverage "
            "and support rate each drop by ~0.17**. That delta is **by construction** — "
            "it proves the mechanism, not that a critic improves a real model. Re-run "
            "with a real model over the bundled corpus (3 repeats, July 2026), the effect "
            "measured **~0** (+0.004 ± 0.007): a real writer cites what it says. Try it live "
            "on the **Critic A/B** page. `source_validity` stays 1.0 in *both* arms — the "
            "no-fabrication guarantee is always on, independent of the critic."
        )

    with t_concepts:
        show_md(
            "### Concepts & glossary\n\n"
            "| Term | Meaning |\n|---|---|\n"
            "| **RAG** | Retrieval-Augmented Generation — ground the answer in retrieved "
            "documents instead of the model's memory |\n"
            "| **Agentic RAG** | retrieval inside a plan → act → verify loop, not a single "
            "retrieve-then-answer pass |\n"
            "| **Keyless mode** | deterministic fake LLM, search and fetch; offline, "
            "reproducible, $0 (the default) |\n"
            "| **Real mode** | an OpenAI model with live web search (or your own documents); "
            "paid per run |\n"
            "| **Sub-question** | one facet of your question, planned by the planner; each "
            "gets its own search |\n"
            "| **Search query** | the words actually sent to the search tool for a "
            "sub-question |\n"
            "| **Evidence** | one passage from one source, with a stable id (E1, E2…), the "
            "source's title and URL |\n"
            "| **Ranking by meaning** | real mode: an embedding model turns the sub-question "
            "and each candidate passage into vectors; the closest passage wins |\n"
            "| **Hybrid retrieval** | a cheap word-overlap shortlist, then the more expensive "
            "ranking by meaning on just the shortlist |\n"
            "| **Claim** | one statement in the report, plus the evidence ids that back it |\n"
            "| **Critic / reflection loop** | the step that checks each claim against its "
            "cited evidence and can send the draft back for a rewrite |\n"
            "| **Iteration** | one critic → rewrite round |\n"
            "| **Token budget** | the most tokens a run may use; reaching it ends the run "
            "as `partial` |\n"
            "| **Partial** | the run stopped early (budget), or the critic still wanted a "
            "rewrite when the iterations ran out |\n"
            "| **The guarantee** | `enforce_citations` strips every citation to a source "
            "that wasn't gathered and drops any claim whose citations were all invalid — "
            "fabricated sources are *structurally impossible* |\n"
            "| **Abstention** | an empty report for a question the sources don't cover, "
            "rather than an invented answer |\n"
            "| **Tavily** | the web-search API real mode uses |\n"
            "| **trafilatura** | the library that pulls a web page's article text out of "
            "its HTML |\n"
            "| **p95 (nearest-rank)** | 95th-percentile latency without interpolation — "
            "stable on small samples |\n\n"
            "### Sources\n"
            + _corpus_list_md(m) + "\n\n"
            "### Researching your own documents\n"
            "Point `CORPUS_DIR` at a folder of `.md`, `.txt` or `.pdf` files (PDF needs "
            "`pypdfium2`) and keep `SEARCH_PROVIDER` and `FETCH_PROVIDER` at `fake`: the "
            "assistant then searches those files instead of the bundled corpus. Keyless, you "
            "get retrieval with citations; with `LLM_PROVIDER=openai`, a report written by "
            "the model. On a Docker server the folder must also "
            "be mounted into the container. See *Research your own documents* in the "
            "README.\n\n"
            "### Under the hood\n"
            "LangGraph · FastAPI · pydantic · Streamlit, with a keyless deterministic "
            "test/eval suite and a CI citation-coverage gate. Real mode adds the OpenAI API "
            "(chat and embeddings), Tavily and trafilatura. This UI holds no research "
            "logic: it calls the FastAPI service or, when no API is running (as on the "
            "public demo), the same agent functions in-process."
        )

    with t_faq:
        show_md(
            "### FAQ\n\n"
            "**The run ended `partial`. What happened?**  \n"
            "Open the **Step timeline**. If the last step before the finalizer is a critic "
            "step with `verdict=revise`, the critic wanted another rewrite but the run had "
            "no iteration or token budget left for one: the claims it flagged were removed, "
            "and every remaining claim passed its check. Otherwise the run reached the token budget, possibly "
            "before the critic checked the last draft. Raise *Max critic iterations* or "
            "*Token budget* in the sidebar"
            + (" (up to this server's limits)" if m["real"] else "") + ".\n\n"
            "**The report is empty or very short.**  \n"
            + ("The searches found little the fetcher could read. Look for `fetch failed` "
               "steps in the timeline, and try a more specific question.\n\n" if m["web"] else
               "Nothing in the documents matched the question: that is abstention, not an "
               "error. Ask about a topic the documents cover.\n\n")
            + "**Why is some evidence marked \"not cited\"?**  \n"
            "The writer only cites evidence that supports a claim. A passage that doesn't "
            "is left out rather than forced into the report.\n\n"
            "**What do `403 Forbidden`, `no readable article text` and `video page` mean?**  \n"
            "That source couldn't be used: the site blocks bots, the page had no article "
            "text, or it is a video. The researcher moves on to the next search result.\n\n"
            "**Why does the same question give a different report?**  \n"
            + ("In real mode the search results change and the model isn't deterministic. "
               "The keyless demo always gives the same report.\n\n" if m["real"] else
               "It doesn't in the keyless demo: the same question gives the same report. "
               "In real mode it can (live search, model variation).\n\n")
            + "**Why does a real-mode run take so long?**  \n"
            "It makes several model calls and, on the web, reads a dozen pages: 30–40 seconds "
            "with web sources, about 15 over local documents. The keyless demo takes "
            "milliseconds.\n\n"
            "**The sidebar won't let me raise the token budget or iterations.**  \n"
            "In real mode the server's `TOKEN_BUDGET` and `MAX_ITERATIONS` are the ceiling, "
            "so a request can't spend more than the operator allowed.\n\n"
            "**The sidebar says \"API not reachable\".**  \n"
            "The UI can't reach the API. On a Docker server, check `docker compose ps` and "
            "`docker compose logs api`.\n\n"
            "**I got \"API error 502: model output failed validation\".**  \n"
            "The model's answer didn't match the expected format, even after automatic "
            "retries. Run the question again.\n\n"
            "**Where do I see what a run cost?**  \n"
            "On its meta line (\"Cost $…\"), and averaged on the **Observability** page."
        )


# ------------------------------------------------------------------------ access
# Optional shared password, for a real-mode deployment shared with a reviewer.
# The check lives inside the app rather than in front of it as HTTP basic auth:
# Safari doesn't send basic-auth credentials on WebSocket connections (WebKit bug
# 80362), so a Safari visitor would sit on "connecting" forever. Unset (the public
# keyless demo) means no gate at all.
UI_PASSWORD = os.environ.get("ARA_UI_PASSWORD", "")


def require_password() -> None:
    """Stop here until this browser session has entered ``ARA_UI_PASSWORD``."""
    if not UI_PASSWORD or ss.get("authenticated"):
        return
    hero("Agentic Research &amp; Report Assistant",
         "This instance is private. Enter the password you were given.")
    with st.form("login"):
        attempt = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter", type="primary")
    if submitted:
        if hmac.compare_digest(attempt.encode(), UI_PASSWORD.encode()):
            ss.authenticated = True
            st.rerun()
        time.sleep(1)  # slow down guessing
        st.error("Wrong password.")
    st.stop()


require_password()

# ------------------------------------------------------------------------ navigate
_nav = st.navigation([
    st.Page(page_research, title="Research", icon=":material/search:", default=True),
    st.Page(page_compare, title="Critic A/B", icon=":material/balance:"),
    st.Page(page_history, title="History", icon=":material/history:"),
    st.Page(page_observability, title="Observability", icon=":material/monitoring:"),
    st.Page(page_guide, title="Guide", icon=":material/menu_book:"),
])
_nav.run()
