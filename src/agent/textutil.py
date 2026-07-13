"""Small, dependency-free text helpers shared across the keyless path.

Centralising tokenisation, keyword overlap, sentence splitting and a cheap token
estimate here guarantees that the fake LLM, the fake search tool, the critic, and
the evaluator all use *exactly the same* notion of "relevant" — which is what
makes the offline pipeline deterministic and internally consistent.
"""

from __future__ import annotations

import re

# A compact English stopword list. Kept small on purpose: it only needs to strip
# the highest-frequency function words so keyword overlap reflects topical terms.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then else for to of in on at by with from into over
    under as is are was were be been being do does did has have had this that these
    those it its their there here we you they he she them his her our your my me i
    what which who whom whose how when where why can could should would may might
    will shall must not no nor so than too very just about also more most some any
    each every both either neither between within without across per via using use
    used uses one two three main key
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.MULTILINE)
_MD_NOISE_RE = re.compile(r"[*_`>]")


def tokenize(text: str) -> list[str]:
    """Lowercase the text and return alphanumeric word tokens."""
    return _WORD_RE.findall(text.lower())


def content_words(text: str) -> list[str]:
    """Tokens with stopwords and very short tokens removed (order preserved)."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


def content_word_set(text: str) -> set[str]:
    return set(content_words(text))


def keyword_overlap(a: str, b: str) -> float:
    """Overlap coefficient of content words: |A∩B| / min(|A|, |B|).

    The overlap coefficient (rather than Jaccard) is used so that a short claim
    fully contained in a longer snippet scores ~1.0, which is exactly the
    "is this claim supported by this snippet?" question the critic asks.
    """
    sa, sb = content_word_set(a), content_word_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def strip_markdown(text: str) -> str:
    """Remove Markdown heading lines and light inline noise for clean prose.

    Used before snippet extraction so a claim quotes the document's prose rather
    than its ``# Title`` line.
    """
    no_headings = _HEADING_RE.sub("", text)
    return _MD_NOISE_RE.sub("", no_headings)


def split_sentences(text: str) -> list[str]:
    """Deterministic sentence splitter; each sentence has whitespace collapsed."""
    parts = _SENT_RE.split(text.strip())
    return [" ".join(p.split()) for p in parts if p.strip()]


def best_sentences(query: str, text: str, k: int = 2) -> list[str]:
    """Return the ``k`` sentences of ``text`` most relevant to ``query``.

    Deterministic: sentences are ranked by content-word overlap with the query,
    ties broken by original order, so the same inputs always yield the same
    snippet. Falls back to the first sentences when nothing overlaps.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    q = content_word_set(query)
    scored = []
    for idx, sent in enumerate(sentences):
        s = content_word_set(sent)
        score = len(q & s) / (len(q) or 1)
        scored.append((score, idx, sent))
    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[:k]
    if all(score == 0 for score, _, _ in top):
        return sentences[:k]
    top.sort(key=lambda t: t[1])
    return [sent for _, _, sent in top]


def approx_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token), min 1."""
    return max(1, len(text) // 4)
