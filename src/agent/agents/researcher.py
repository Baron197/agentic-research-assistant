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
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..context import AgentContext
from ..schemas import Evidence
from ..textutil import approx_tokens, best_sentences, strip_markdown
from ._common import clean_hint


@dataclass(frozen=True)
class _Outcome:
    """A captured tool result: a value, or the exception the call raised."""

    value: Any = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


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
    eager_urls: list[str] = []
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
                if r.url not in eager_urls:
                    eager_urls.append(r.url)
                provisional_seen.add(r.url)
                taken += 1
    fetch_out = _run_parallel(
        {u: (lambda u=u: ctx.fetch.fetch(u)) for u in eager_urls}, concurrency
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
                cost_tokens = approx_tokens(query)
                sp.tokens = cost_tokens
                sp.usd = ctx.tracer.cost(cost_tokens)
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
                            fres = _Outcome(value=ctx.fetch.fetch(result.url))
                        except Exception as exc:  # noqa: BLE001
                            fres = _Outcome(error=exc)
                        fetch_out[result.url] = fres
                    doc = fres.value if fres.ok else None
                    if doc is None:
                        err = fres.error if not fres.ok else "no content"
                        fp.tokens = 1
                        fp.output_summary = f"fetch failed: {err}"[:200]
                    else:
                        snippet = " ".join(best_sentences(query, strip_markdown(doc), k=2))
                        snippet = " ".join(snippet.split()) or result.snippet
                        fp.tokens = approx_tokens(snippet)
                        fp.output_summary = snippet[:80]
                    fp.usd = ctx.tracer.cost(fp.tokens)
                budget = budget.charge(fp.tokens, fp.usd)
                tool_calls += 1
                steps.append(fp.to_step())
                if doc is None:
                    continue  # try the next search result instead

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
