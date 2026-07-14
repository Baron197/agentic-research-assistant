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

import html
import json
import os
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
        from agent.metrics import support_rate
        from agent.observability import aggregate, load_run, recent_runs
        from agent.runner import render_report_markdown, run
        from agent.tools.search import list_corpus
        return {"version": __version__, "get_settings": get_settings,
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
        return {"status": "ok", "version": e["version"], "keyless": s.is_keyless}
    try:
        return httpx.get(f"{API_URL}/health", timeout=5).json()
    except Exception:  # noqa: BLE001
        return None


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
    overrides = {k: payload[k] for k in
                 ("max_iterations", "token_budget", "require_approval", "enable_critic")
                 if payload.get(k) is not None}
    try:
        res = e["run"](payload.get("question", ""), settings=e["get_settings"](), **overrides)
    except Exception as exc:  # noqa: BLE001
        return None, f"Research failed: {exc}"
    return {
        "run_id": res.run_id, "status": res.status, "question": res.question,
        "report": res.report.model_dump(), "markdown": e["render_markdown"](res.report),
        "iterations": res.iterations, "tool_calls": res.tool_calls, "tokens": res.tokens,
        "usd": res.usd, "latency_ms": res.latency_ms, "dropped_claims": res.dropped_claims,
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
    """Run the same question with the critic ON and OFF — the headline A/B, live."""
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


def sidebar_health() -> None:
    with st.sidebar:
        st.toggle("Dark mode", value=bool(ss.get("dark", False)),
                  key="dark_toggle", on_change=_sync_dark,
                  help="Switch the whole app between light and dark themes.")
        health = api_health()
        if health:
            st.success(f"API up · v{health.get('version', '?')} · "
                       f"keyless={health.get('keyless')}")
        else:
            st.error(f"API not reachable at {API_URL}")


def settings_sidebar(show_critic: bool = True) -> None:
    with st.sidebar:
        st.header("Run settings")
        st.slider("Max critic iterations", 0, 5, 2, key="max_iter")
        st.number_input("Token budget", min_value=100, max_value=500_000, value=60_000,
                        step=1_000, key="budget")
        if show_critic:
            st.toggle("Enable verifying critic", value=True, key="critic",
                      help="The headline agent. OFF = the critic-OFF arm of the A/B: "
                           "deliberately uncited claims survive, so coverage drops.")
        st.checkbox("Require human approval", value=False, key="approval",
                    help="Over the HTTP API there is no interactive approver, so runs "
                         "auto-approve (an approval step still appears in the trace). A "
                         "real deny is only possible via the Python API's approval_fn.")
        st.divider()
    sidebar_health()


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
    sup = r.get("support_rate")
    sup_s = f" · support {sup:.0%}" if isinstance(sup, (int, float)) else ""
    st.markdown(
        f'<div class="runmeta">Cost ${r["usd"]:.4f} · latency {r["latency_ms"]:.0f} ms · '
        f'dropped claims {r["dropped_claims"]}{sup_s} · critic '
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
            st.caption("Corpus coverage is a keyless-mode view over the local corpus.")
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
    settings_sidebar(show_critic=True)
    hero("Agentic Research &amp; Report Assistant",
         "Multi-agent (LangGraph) research with <b>cited</b> reports. Runs keyless by "
         "default — every source is real, none are fabricated.")
    st.text_input("Research question", key="q")
    st.markdown('<div class="chip-hint">Try an example (the corpus covers RAG topics):</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        cols[i % 3].button(ex, key=f"ex{i}", on_click=pick_example, args=(ex,),
                           use_container_width=True)
    st.button("Run research", type="primary", on_click=run_research, use_container_width=True)

    if ss.error:
        st.error(ss.error)
    if ss.result:
        render_result(ss.result, ss.detail or {})
    else:
        st.info("Enter a question (or pick an example) and click **Run research**. "
                "To see the critic's impact, open the **Critic A/B** page in the sidebar.")


def page_compare() -> None:
    settings_sidebar(show_critic=False)
    hero("Critic A/B",
         "Run the same question with the verifying critic <b>on</b> and <b>off</b>, "
         "side by side — the project's headline result, live.")
    st.text_input("Research question", key="q")
    st.button("Run A/B comparison", type="primary", on_click=run_compare,
              use_container_width=True)
    st.caption("The critic removes uncited / unsupported claims. With it OFF the "
               "deliberately-uncited synthesis claim survives — visible in the "
               "right-hand report and in the lower citation coverage.")

    if ss.error:
        st.error(ss.error)
    if ss.compare:
        render_compare(ss.compare)
    else:
        st.info("Enter a question and click **Run A/B comparison** to see the "
                "citation-coverage and support delta between critic ON and OFF.")


def page_history() -> None:
    sidebar_health()
    hero("Run history",
         "Browse and reopen any past run — full report, evidence, and trace.")
    runs = (api_get("/runs?limit=50") or {}).get("runs", [])
    if not runs:
        st.info("No runs recorded yet. Run a research query to populate history.")
        return
    st.caption(f"{len(runs)} most recent run(s). Click **View** to reopen one below.")
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


def _corpus_list_md() -> str:
    docs = (api_get("/corpus") or {}).get("documents", [])
    if not docs:
        return "_Corpus listing unavailable._"
    lines = [f"{len(docs)} local Markdown documents back the keyless search/fetch tools:\n"]
    lines += [f"- **{html.escape(d['title'])}** &nbsp;`{html.escape(d['url'])}`" for d in docs]
    return "\n".join(lines)


def page_guide() -> None:
    sidebar_health()
    hero("Guide",
         "How the app works, and what every metric and evaluation number means. "
         "New here? Start with <b>Getting started</b>, then dip into the metric "
         "tabs whenever a number needs explaining.")

    t_start, t_read, t_obs, t_eval, t_concepts = st.tabs(
        [":material/rocket_launch: Getting started", ":material/description: Reading a report",
         ":material/monitoring: Observability metrics", ":material/task_alt: Evaluation metrics",
         ":material/school: Concepts"]
    )

    with t_start:
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown(
                "### What this app does\n"
                "It answers a research question **using only documents it actually "
                "retrieved**. Instead of guessing, five cooperating agents plan the "
                "research, gather evidence with tools, draft a report that cites "
                "**only** what was gathered, and a **critic verifies every claim** — "
                "abstaining honestly when the answer isn't in the corpus. That's "
                "*agentic Retrieval-Augmented Generation (RAG)*.\n\n"
                "### Your first question in four steps\n"
                "1. On the **Research** page, type a question — or click an example "
                "chip. The corpus covers RAG topics (embeddings, chunking, hybrid "
                "search, reranking, evaluation, hallucination, agentic RAG…).\n"
                "2. Click **Run research**. The pipeline plans → researches → writes → "
                "verifies, usually in a few milliseconds (it runs keyless, offline).\n"
                "3. Read the answer: the **[1] [2]** markers are citations; the "
                "**Sources** list and the **Evidence & sources** tab show the passages "
                "they came from.\n"
                "4. Explore the numbers: **Observability** shows how the system is "
                "*running* (cost, latency, tokens); the **Evaluation metrics** tab "
                "here explains how *good* the grounded answers are.\n\n"
                "### The pages\n"
                "- **Research** — ask a question and read the cited report (six "
                "result tabs).\n"
                "- **Critic A/B** — run the same question with the critic on and "
                "off, side by side.\n"
                "- **History** — browse and reopen any past run.\n"
                "- **Observability** — aggregate cost / latency / coverage across "
                "all runs.\n"
                "- **Guide** — this page."
            )
        with right:
            st.graphviz_chart(pipeline_dot(), use_container_width=False)
            st.caption("The five-agent pipeline. See **Concepts** for what each "
                       "agent does.")

    with t_read:
        st.markdown(
            "### The metrics band\n"
            "Every answer opens with five cards and a meta line:\n\n"
            "| Field | Meaning |\n|---|---|\n"
            "| **Status** | `complete` · `partial` (hit a budget/iteration limit) · "
            "`awaiting_approval` |\n"
            "| **Iterations** | how many times the critic sent the draft back to be "
            "re-written |\n"
            "| **Tool calls** | `search` + `fetch` calls made while gathering evidence |\n"
            "| **Tokens** | total tokens the run consumed (a deterministic estimate "
            "in keyless mode) |\n"
            "| **Citation coverage** | % of claims carrying ≥1 citation — the badge is "
            "green ≥100%, amber ≥80%, red below |\n"
            "| meta line | cost · latency · dropped claims · support rate · whether "
            "the critic ran · the `run_id` |\n\n"
            "### Citations & sources\n"
            "Every bullet in the report ends in a **[n]** marker. The numbered "
            "**Sources** list maps each **[n]** to a document that was actually "
            "gathered, and the **Evidence & sources** tab expands each passage and "
            "shows *exactly which claim cites it*. This is the no-fabricated-sources "
            "guarantee, made visible: a citation can never point at something the "
            "system didn't retrieve.\n\n"
            "### The six result tabs\n"
            "- **Report** — the cited answer, with Markdown / JSON download.\n"
            "- **Evidence & sources** — each gathered passage, its source, and the "
            "claims that cite it.\n"
            "- **Corpus** — which corpus documents this question drew on (used vs "
            "not retrieved).\n"
            "- **Step timeline** — the ordered trace: every agent step with its "
            "tokens and latency.\n"
            "- **Agent graph** — the pipeline with *this run's* path highlighted; "
            "the `revise` edge lights up when the critic looped.\n"
            "- **Run data** — the raw run JSON.\n\n"
            "### Abstention (honest refusal)\n"
            "Ask something outside the corpus (e.g. *“How do I bake sourdough "
            "bread?”*) and you'll get **0 sources, 0% coverage, an empty report**. "
            "That is the system correctly declining to answer rather than inventing "
            "facts — not a bug."
        )

    with t_obs:
        st.markdown(
            "### Observability — how the system is *running*\n"
            "The **Observability** page aggregates these across every persisted run "
            "(from the `/metrics` endpoint):\n\n"
            "| Metric | What it means |\n|---|---|\n"
            "| **Total runs** | how many runs are recorded in the run index |\n"
            "| **Avg cost / run** | mean USD per run — `$0.0000` in keyless mode, "
            "because the fake providers are free |\n"
            "| **Avg latency** | mean wall-clock time per run |\n"
            "| **p95 latency** | 95th-percentile latency by the *nearest-rank* method "
            "— resilient to small samples, so one slow run doesn't distort it |\n"
            "| **Avg steps / run** | mean number of trace spans (planner → researcher "
            "→ writer → critic → finalizer, plus tool calls) |\n"
            "| **Avg citation coverage** | mean fraction of claims that carry a "
            "citation, across runs |\n\n"
            "Cost is honest by construction: the keyless model is priced at `$0`, so a "
            "keyless run genuinely reports zero. A real model is priced from a small "
            "per-1K-token table."
        )

    with t_eval:
        st.markdown(
            "### Evaluation — how *good* the grounded answers are\n"
            "These are computed by the offline eval harness over a golden task set "
            "(and wired into CI as a gate). All validate *structure and grounding* on "
            "the deterministic keyless path:\n\n"
            "| Metric | What it measures | Keyless |\n|---|---|---|\n"
            "| **citation_coverage** | % of claims carrying ≥1 citation | 1.00 |\n"
            "| **source_validity** | % of citations whose id was *actually gathered* "
            "— catches fabrication | 1.00 |\n"
            "| **support_rate** | % of claims whose cited snippet actually supports "
            "them (keyword overlap ≥ 0.3) | 1.00 |\n"
            "| **point_coverage** | % of each task's expected key facts present in the "
            "report | 0.90 |\n"
            "| **abstention_accuracy** | out-of-corpus questions must produce zero "
            "claims | 1.00 |\n"
            "| **faithfulness** | LLM-as-judge; needs a real model, so reported `n/a` "
            "in keyless mode (never faked) | n/a |\n\n"
            "### The critic A/B (the headline result)\n"
            "Running the same tasks with the **critic ON vs OFF** isolates the "
            "critic's contribution: with it OFF, deliberately-uncited claims survive, "
            "so **citation coverage and support rate each drop by ~0.17**. Try it live "
            "on the **Critic A/B** page. `source_validity` stays 1.0 in *both* arms — "
            "the no-fabrication guarantee is always on, independent of the critic."
        )

    with t_concepts:
        st.markdown(
            "### Concepts & glossary\n\n"
            "| Term | Meaning |\n|---|---|\n"
            "| **RAG** | Retrieval-Augmented Generation — ground the answer in "
            "retrieved documents instead of the model's memory |\n"
            "| **Agentic RAG** | retrieval inside a plan → act → verify loop, not a "
            "single retrieve-then-answer pass |\n"
            "| **Evidence** | one gathered snippet with a stable id (E1, E2…), source "
            "title and URL |\n"
            "| **Claim** | one assertion in the report, plus the evidence ids that "
            "back it |\n"
            "| **Critic / reflection loop** | the agent that checks each claim against "
            "its cited evidence and triggers a re-draft |\n"
            "| **The guarantee** | `enforce_citations` strips any citation to an "
            "ungathered id and drops unsupported claims — fabricated sources are "
            "*structurally impossible* |\n"
            "| **Keyless mode** | deterministic fake LLM/search/fetch; offline, "
            "reproducible, $0.00 (the default) |\n"
            "| **Abstention** | producing an empty report for an off-topic question "
            "rather than inventing an answer |\n"
            "| **p95 (nearest-rank)** | 95th-percentile latency without interpolation "
            "— stable on small samples |\n\n"
            "### The knowledge corpus\n"
            + _corpus_list_md() + "\n\n"
            "### Under the hood\n"
            "LangGraph · FastAPI · pydantic · Streamlit · a keyless deterministic "
            "test/eval suite with a CI citation-coverage gate. This UI is a **thin "
            "client** — all logic lives behind the API; every page only calls it and "
            "renders the response."
        )


# ------------------------------------------------------------------------ navigate
_nav = st.navigation([
    st.Page(page_research, title="Research", icon=":material/search:", default=True),
    st.Page(page_compare, title="Critic A/B", icon=":material/balance:"),
    st.Page(page_history, title="History", icon=":material/history:"),
    st.Page(page_observability, title="Observability", icon=":material/monitoring:"),
    st.Page(page_guide, title="Guide", icon=":material/menu_book:"),
])
_nav.run()
