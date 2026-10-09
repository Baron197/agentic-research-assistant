"""Researcher node: gather evidence with the search + fetch tools.

For each not-yet-attempted sub-question it searches, fetches each new result,
extracts the most relevant sentences as an evidence snippet, and appends an
``Evidence`` record with a stable id and real source metadata. It is budget-aware
(checks the token budget before every tool call), de-duplicates by URL so the
same source is never gathered twice, and records which sub-questions it has
*attempted* (``researched_sqs``) so a revise loop never re-runs a facet — even
one that yielded no evidence — which keeps the revise loop cheap.

**Parallel fan-out.** The network-bound work — the search queries, then the page
fetches the report will actually use — runs concurrently on a small thread pool
(``research_concurrency``). The *decision* logic then runs sequentially over those
cached results: the URL de-duplication that distributes shared search results
across facets, stable evidence-id assignment, and budget accounting. Separating
I/O (parallel) from decisions (sequential replay) means a run gathers the **same
evidence, citations, and token accounting** as a one-at-a-time run — deterministic,
independent of thread timing (only each step's wall-clock ``ms`` differs, as it
always has) — while real-mode latency drops to roughly the slowest fetch instead
of their sum. Only as many pages as the replay can consume (``evidence_per_subquestion``
per facet) are prefetched, so report depth, not the raw search width, bounds the
fetching. Raise ``evidence_per_subquestion`` for deeper reports; the parallel
fetch keeps those deeper runs fast.

**Which text becomes the evidence.** Keyless runs quote the two sentences that
share the most words with the search query. In real mode (an embedder in the
context) the snippet is instead the passage that best answers the sub-question,
ranked by meaning (``agent.passages``). That choice is made inside the parallel
fetch, so its embedding call costs no wall-clock time; it is recorded as a
priced ``rank`` step after the page's ``fetch`` step.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..context import AgentContext
from ..observability import cost_usd
from ..passages import pick_passage
from ..schemas import Evidence
from ..textutil import approx_tokens, best_sentences, strip_markdown
from ._common import clean_hint

# Hard bound on a single evidence snippet.
#
# ``HttpFetch`` caps the *download* at 1 MB, but sentence-based extraction can
# still return the entire document as one snippet: ``split_sentences`` splits on
# ``.!?``, so a real page with no sentence punctuation (JS-heavy pages, nav walls,
# minified content) collapses to a single "sentence". That snippet is then charged
# at ~4 chars/token — roughly 250k tokens for one fetch, which exhausts the whole
# token budget in a single call and ends the run with no report. Bounding it here,
# before the snippet is charged or stored, keeps one hostile page from starving the
# rest of the run. 4,000 chars is far above any real extract (the longest snippet
# over the bundled corpus is ~420 chars), so normal runs are unaffected.
MAX_SNIPPET_CHARS = 4_000


@dataclass(frozen=True)
class _Outcome:
    """A captured tool result: a value, or the exception the call raised."""

    value: Any = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class _Page:
    """A fetched page and the snippet picked from it for one (query, sub-question)."""

    doc: str
    snippet: str
    picked_for: tuple[str, str]
    embed_tokens: int = 0
    candidates: int = 0
    note: str = ""  # set when semantic ranking failed and the word-overlap pick was kept


def _pick(ctx: AgentContext, doc: str, query: str, target: str) -> _Page:
    """Pick the page's evidence snippet: by meaning in real mode, by words otherwise."""
    if ctx.embedder is not None:
        try:
            p = pick_passage(target, query, doc, ctx.embedder)
            return _Page(doc, p.text, (query, target), p.tokens, p.candidates)
        except Exception as exc:  # noqa: BLE001 - keep the page; fall back to words
            note = f"ranking failed, kept the word-overlap pick: {exc}"[:200]
            text = " ".join(best_sentences(query, strip_markdown(doc), k=2))
            return _Page(doc, text, (query, target), note=note)
    return _Page(doc, " ".join(best_sentences(query, strip_markdown(doc), k=2)), (query, target))


def _read_page(ctx: AgentContext, url: str, query: str, target: str) -> _Page:
    return _pick(ctx, ctx.fetch.fetch(url), query, target)


def _run_parallel(fns: dict[str, Callable[[], Any]], max_workers: int) -> dict[str, _Outcome]:
    """Run each ``key -> thunk`` concurrently; return ``{key: _Outcome}``.

    Exceptions are captured per task (never raised) so the sequential replay can
    reproduce the exact graceful degradation a one-at-a-time run would show.
    Results are keyed, so thread-completion order can never affect the output.
    """
    if not fns:
        return {}
    workers = max(1, min(max_workers, len(fns)))
    out: dict[str, _Outcome] = {}
    if workers == 1:  # thread-free path for concurrency=1 (and simpler debugging)
        for key, fn in fns.items():
            try:
                out[key] = _Outcome(value=fn())
            except Exception as exc:  # noqa: BLE001 - captured, reproduced in replay
                out[key] = _Outcome(error=exc)
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {key: ex.submit(fn) for key, fn in fns.items()}
        for key, fut in futures.items():
            try:
                out[key] = _Outcome(value=fut.result())
            except Exception as exc:  # noqa: BLE001
                out[key] = _Outcome(error=exc)
    return out


def researcher(state: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
    plan = state["plan"]
    evidence: list[Evidence] = list(state["evidence"])
    budget = state["budget"]
    tool_calls = int(state.get("tool_calls", 0))
    researched = list(state.get("researched_sqs", []))
    steps = []

    seen_urls = {e.source_url for e in evidence}
    next_id = len(evidence) + 1
    cap = max(1, int(getattr(ctx.settings, "evidence_per_subquestion", 2)))
    concurrency = max(1, int(getattr(ctx.settings, "research_concurrency", 4)))

    pending = [sq for sq in plan.sub_questions if sq.id not in researched]
    if not pending or budget.exceeded:
        return {
            "evidence": evidence, "budget": budget, "tool_calls": tool_calls,
            "researched_sqs": researched, "trace": steps,
        }

    # --- Parallel I/O: overlap every distinct search, then the page fetches the
    # replay will actually use. (Distinct keys, so each network call happens at
    # most once; the replay below reads from these caches.)
    queries: list[str] = []
    for sq in pending:
        for q in sq.search_queries:
            if q not in queries:
                queries.append(q)
    search_out = _run_parallel(
        {q: (lambda q=q: ctx.search.search(q)) for q in queries}, concurrency
    )

    # Prefetch only the URLs a successful replay would consume — each facet's first
    # ``cap`` not-yet-taken results — so the fetching is bounded by report depth,
    # not the raw search width. (A fetch failure that pushes a facet past this
    # window is fetched lazily in the replay; rare.) This keeps the token/cost
    # budget meaningful in real mode: we never eagerly fetch pages the run drops.
    # Each prefetched page also gets its snippet picked for the (query, sub-question)
    # that will consume it, so real mode's embedding call overlaps the other fetches.
    eager: dict[str, tuple[str, str]] = {}
    provisional_seen = set(seen_urls)
    for sq in pending:
        taken = 0
        for q in sq.search_queries:
            if taken >= cap:
                break
            res = search_out.get(q)
            if not (res and res.ok):
                continue
            for r in res.value:
                if taken >= cap:
                    break
                if r.url in provisional_seen:
                    continue
                eager.setdefault(r.url, (q, sq.question))
                provisional_seen.add(r.url)
                taken += 1
    fetch_out = _run_parallel(
        {u: (lambda u=u, qt=qt: _read_page(ctx, u, *qt)) for u, qt in eager.items()}, concurrency
    )

    # --- Sequential replay: identical logic/ordering to a one-at-a-time run.
    for sq in pending:
        if budget.exceeded:
            break
        hint = clean_hint(sq.question)
        gathered = 0
        for query in sq.search_queries:
            if budget.exceeded or gathered >= cap:
                break
            with ctx.tracer.span("researcher", tool="search") as sp:
                sp.input_summary = query
                res = search_out.get(query)
                if res and res.ok:
                    results = res.value
                    sp.output_summary = f"{len(results)} results"
                else:
                    results = []
                    sp.output_summary = f"search failed: {res.error if res else 'no result'}"[:200]
                # Tool steps charge estimated tokens to the budget but no USD: the
                # LLM provider bills this text as input when the writer/critic read it.
                sp.tokens = approx_tokens(query)
            budget = budget.charge(sp.tokens, sp.usd)
            tool_calls += 1
            steps.append(sp.to_step())

            for result in results:
                if budget.exceeded or gathered >= cap:
                    break
                if result.url in seen_urls:
                    continue
                with ctx.tracer.span("researcher", tool="fetch") as fp:
                    fp.input_summary = result.url
                    fres = fetch_out.get(result.url)
                    if fres is None:  # beyond the prefetch window (a retry) — fetch now
                        try:
                            fres = _Outcome(value=_read_page(ctx, result.url, query, sq.question))
                        except Exception as exc:  # noqa: BLE001
                            fres = _Outcome(error=exc)
                        fetch_out[result.url] = fres
                    page = fres.value if fres.ok else None
                    if page is not None and page.picked_for != (query, sq.question):
                        # Prefetched for another facet (a failed fetch shifted the
                        # assignment): pick this facet's snippet from the same page.
                        page = _pick(ctx, page.doc, query, sq.question)
                    if page is None:
                        err = fres.error if not fres.ok else "no content"
                        fp.tokens = 1
                        fp.output_summary = f"fetch failed: {err}"[:200]
                    else:
                        snippet = " ".join(page.snippet.split()) or result.snippet
                        if len(snippet) > MAX_SNIPPET_CHARS:
                            # An unpunctuated page yields one document-sized "sentence";
                            # truncate before charging so it cannot eat the budget.
                            snippet = snippet[:MAX_SNIPPET_CHARS].rstrip() + "..."
                        fp.tokens = approx_tokens(snippet)
                        fp.output_summary = snippet[:80]
                budget = budget.charge(fp.tokens, fp.usd)
                tool_calls += 1
                steps.append(fp.to_step())
                if page is None:
                    continue  # try the next search result instead

                if ctx.embedder is not None and (page.embed_tokens or page.note):
                    # Real mode: the embedding call that picked the passage, priced
                    # at the embedding model's rate (it is billed on its own).
                    with ctx.tracer.span("researcher", tool="rank") as rp:
                        rp.input_summary = f"{page.candidates} passages · {ctx.embedder.model}"
                        rp.output_summary = page.note or "best passage by meaning"
                        rp.tokens = page.embed_tokens
                        rp.usd = cost_usd(ctx.embedder.model, page.embed_tokens)
                    budget = budget.charge(rp.tokens, rp.usd)
                    steps.append(rp.to_step())

                evidence.append(
                    Evidence(
                        id=f"E{next_id}",
                        claim_hint=hint,
                        source_title=result.title,
                        source_url=result.url,
                        snippet=snippet,
                    )
                )
                next_id += 1
                gathered += 1
                seen_urls.add(result.url)
        researched.append(sq.id)  # mark attempted (even if it yielded nothing)

    return {
        "evidence": evidence,
        "budget": budget,
        "tool_calls": tool_calls,
        "researched_sqs": researched,
        "trace": steps,
    }
