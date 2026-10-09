"""Pick a page's evidence passage by meaning (real mode's ``evidence_ranking``).

The keyless researcher quotes the two sentences sharing the most words with the
search query (``textutil.best_sentences``). On real pages that often picks the
wrong text: for "definition of text embeddings", Wikipedia's *Word embedding*
page gave a sentence about transcribing games, because it happens to contain
"text" and "embeddings". Matching word forms or weighting rare words (BM25)
picked the same sentence; only ranking by meaning found the definition.

So a page is cut into overlapping 3-sentence passages (a definition stays next
to the sentence that explains it), a word-overlap shortlist keeps the
``SHORTLIST`` most promising ones, and an embedding model reranks them by
similarity to the sub-question: cheap lexical recall, then a dense rerank, the
usual hybrid retrieval pattern. On 21 real pages, a shortlist of 15 picked the
same passage as embedding every passage on 20 at about a third of the tokens,
and a shortlist of 12 matched 15 on every page it was compared on.

When the two best passages overlap, they are merged: neighbouring passages that
score almost the same usually means the answer runs on into the next sentence
(Google's similarity page states "choose one of these three similarity
measures" and lists them in the sentence after).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .textutil import content_words, split_sentences, strip_markdown
from .tools.embed import Embedder

WINDOW = 3               # sentences per passage
SHORTLIST = 12           # passages sent to the embedding model per page
MAX_EMBED_CHARS = 1_000  # of each passage, as embedding input (bounds the cost)
MAX_PASSAGE_CHARS = 1_500  # a merged passage longer than this keeps the best window only


@dataclass(frozen=True)
class Passage:
    text: str
    tokens: int = 0       # embedding tokens billed to pick it
    candidates: int = 0   # passages the embedding model compared


def _fold(word: str) -> str:
    """Crude suffix folding so "embeddings" and "embedding" count as one term."""
    for suffix in ("ings", "ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _terms(text: str) -> set[str]:
    return {_fold(w) for w in content_words(text)}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm if norm else 0.0


def pick_passage(target: str, query: str, text: str, embedder: Embedder) -> Passage:
    """The passage of ``text`` that best answers ``target`` (the sub-question).

    ``query`` (the search phrasing) only widens the word-overlap shortlist; the
    rerank compares meaning with the sub-question, which says what the evidence
    is for.
    """
    sentences = split_sentences(strip_markdown(text))
    if not sentences:
        return Passage("")
    windows = [" ".join(sentences[i:i + WINDOW]) for i in range(len(sentences))]
    terms = _terms(target) | _terms(query)
    overlap = [len(terms & _terms(w)) for w in windows]
    shortlist = sorted(range(len(windows)), key=lambda i: (-overlap[i], i))[:SHORTLIST]
    if len(shortlist) == 1:
        return Passage(windows[0])

    vectors, tokens = embedder.embed([target] + [windows[i][:MAX_EMBED_CHARS] for i in shortlist])
    score = {i: _cosine(vectors[0], v) for i, v in zip(shortlist, vectors[1:], strict=True)}
    best, second = sorted(shortlist, key=lambda i: (-score[i], i))[:2]

    start, end = best, best + WINDOW
    if abs(second - best) <= WINDOW:  # overlapping or touching: one passage
        start, end = min(best, second), max(best, second) + WINDOW
    passage = " ".join(sentences[start:end])
    if len(passage) > MAX_PASSAGE_CHARS:
        passage = windows[best]
    return Passage(passage, tokens=tokens, candidates=len(shortlist))
