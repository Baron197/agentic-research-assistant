"""Embedding tool: ``Embedder`` Protocol + the real ``OpenAIEmbedder``.

Real mode uses it to rank a page's passages by meaning (``agent.passages``).
There is deliberately no fake: the keyless path has *no* embedder and keeps the
word-overlap selection, so it stays offline, free and deterministic. Tests pass
their own stand-in.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    name: str
    model: str

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:  # pragma: no cover
        """Return one vector per text, plus the tokens the call was billed for."""
        ...


class OpenAIEmbedder:
    """OpenAI embeddings (lazy SDK import; one client shared by the researcher's threads)."""

    name = "openai-embeddings"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any = None
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                from openai import OpenAI  # lazy: never imported on the keyless path

                self._client = OpenAI(api_key=self.api_key)
            return self._client

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:  # pragma: no cover - network
        resp = self._get_client().embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data], int(resp.usage.total_tokens)


def get_embedder(settings: Any) -> Embedder | None:
    """The embedder for semantic evidence ranking, or None when the run ranks by words."""
    if not settings.uses_embeddings:
        return None
    return OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.embedding_model)
