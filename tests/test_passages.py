"""Real mode's evidence ranking: passages picked by meaning, with a stand-in embedder.

No network and no SDK: ``ConceptEmbedder`` turns text into counts of a few
"meaning" words, which is enough to show the picker follows meaning rather than
shared words, and lets the researcher's wiring (pricing, budget, fallback,
parallel == serial) run end to end on the keyless providers.
"""

from __future__ import annotations

import pytest

from agent import runner
from agent.config import Settings
from agent.observability import cost_usd
from agent.passages import pick_passage
from tests.conftest import QUESTION

TARGET = "What are text embeddings?"
QUERY = "definition of text embeddings"
PAGE = (
    "The process turns game transcripts into text and then into word embeddings. "
    "Tournaments are recorded move by move. Players review them later. "
    "Chess clubs share the files online. Some clubs print them. "
    "A word embedding is a vector that encodes the meaning of a word. "
    "Words with similar meaning get vectors that are close together. "
    "Cats sleep most of the day."
)


class ConceptEmbedder:
    """Vectors over a few concept words; the target maps to the 'meaning' concept."""

    name = "concept-embedder"
    model = "text-embedding-3-small"
    CONCEPTS = ("vector", "meaning", "game", "cat")

    def __init__(self, target_vector=(1.0, 1.0, 0.0, 0.0)):
        self.target_vector = list(target_vector)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for t in texts:
            low = t.lower()
            if low.startswith(("what", "how", "why", "which")) or "?" in t:
                vectors.append(self.target_vector)
            else:
                vectors.append([float(low.count(c)) + 0.01 for c in self.CONCEPTS])
        return vectors, sum(max(1, len(t) // 4) for t in texts)


class BrokenEmbedder(ConceptEmbedder):
    def embed(self, texts):
        raise RuntimeError("embeddings API unavailable")


def test_picks_the_passage_that_means_the_answer_not_the_one_sharing_words():
    passage = pick_passage(TARGET, QUERY, PAGE, ConceptEmbedder())
    assert "encodes the meaning of a word" in passage.text
    assert "game transcripts" not in passage.text  # the word-overlap favourite
    assert passage.tokens > 0 and passage.candidates >= 2


class ScoredEmbedder:
    """Gives each passage an exact cosine with the target, from a table."""

    name = "scored-embedder"
    model = "text-embedding-3-small"

    def __init__(self, target, scores):
        self.target, self.scores = target, scores

    def embed(self, texts):
        def vec(t):
            if t == self.target:
                return [1.0, 0.0]
            s = self.scores.get(t, 0.1)
            return [s, (1 - s * s) ** 0.5]
        return [vec(t) for t in texts], len(texts)


SENTENCES = [f"Sentence {n} about similarity measures." for n in range(8)]


def _window(i):
    return " ".join(SENTENCES[i:i + 3])


def test_overlapping_best_passages_are_merged_into_one():
    # "Choose one of these measures:" in one window, the list in the next: when
    # the two best windows overlap, the snippet spans both.
    embedder = ScoredEmbedder(TARGET, {_window(2): 0.95, _window(4): 0.94})
    passage = pick_passage(TARGET, QUERY, " ".join(SENTENCES), embedder)
    assert passage.text == " ".join(SENTENCES[2:7])


def test_distant_runner_up_is_not_merged():
    embedder = ScoredEmbedder(TARGET, {_window(0): 0.95, _window(5): 0.94})
    passage = pick_passage(TARGET, QUERY, " ".join(SENTENCES), embedder)
    assert passage.text == _window(0)


def test_a_one_sentence_page_needs_no_embedding_call():
    embedder = ConceptEmbedder()
    passage = pick_passage(TARGET, QUERY, "A single sentence about embeddings.", embedder)
    assert passage.text == "A single sentence about embeddings."
    assert embedder.calls == 0 and passage.tokens == 0


def test_evidence_ranking_setting():
    keyless = Settings(_env_file=None)
    assert not keyless.uses_embeddings and keyless.is_keyless
    real = Settings(_env_file=None, llm_provider="openai", openai_api_key="x")
    assert real.uses_embeddings  # auto -> by meaning whenever the LLM is OpenAI
    assert not real.model_copy(update={"evidence_ranking": "lexical"}).uses_embeddings
    # Embeddings are paid, so a fake-LLM run that ranks by meaning is not keyless.
    semantic = Settings(_env_file=None, evidence_ranking="semantic", openai_api_key="x")
    assert semantic.uses_embeddings and not semantic.is_keyless
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings(_env_file=None, evidence_ranking="semantic")


def _run_with(settings, monkeypatch, embedder):
    monkeypatch.setattr(runner, "get_embedder", lambda _settings: embedder)
    return runner.run(QUESTION, settings=settings, persist=False)


def test_each_ranked_page_is_a_priced_rank_step(settings, monkeypatch):
    result = _run_with(settings, monkeypatch, ConceptEmbedder())
    ranks = [s for s in result.trace if s.tool == "rank"]
    fetched = [s for s in result.trace if s.tool == "fetch" and not s.output_summary.startswith("fetch failed")]
    assert ranks and len(ranks) <= len(fetched)
    for s in ranks:  # billed at the embedding model's own rate
        assert s.tokens > 0 and s.usd == pytest.approx(cost_usd("text-embedding-3-small", s.tokens))
    # The fake LLM is free, so the run's cost is exactly the embedding calls.
    assert result.usd == pytest.approx(sum(s.usd for s in ranks)) and result.usd > 0
    assert result.status == "complete" and result.citation_coverage == 1.0


def test_ranking_by_meaning_keeps_parallel_equal_to_serial(settings, monkeypatch):
    serial = _run_with(settings.model_copy(update={"research_concurrency": 1}),
                       monkeypatch, ConceptEmbedder())
    parallel = _run_with(settings.model_copy(update={"research_concurrency": 8}),
                         monkeypatch, ConceptEmbedder())
    assert [e.model_dump() for e in serial.evidence] == [e.model_dump() for e in parallel.evidence]
    assert [(s.tool, s.tokens) for s in serial.trace] == [(s.tool, s.tokens) for s in parallel.trace]


def test_a_failing_embedder_keeps_the_word_overlap_evidence(settings, monkeypatch):
    # An embeddings outage must not cost the run its evidence: each page falls back
    # to the keyless pick, and the trace says so.
    keyless = runner.run(QUESTION, settings=settings, persist=False)
    degraded = _run_with(settings, monkeypatch, BrokenEmbedder())
    assert [e.snippet for e in degraded.evidence] == [e.snippet for e in keyless.evidence]
    ranks = [s for s in degraded.trace if s.tool == "rank"]
    assert ranks and all("ranking failed" in s.output_summary and s.usd == 0 for s in ranks)
    assert degraded.status == "complete"
