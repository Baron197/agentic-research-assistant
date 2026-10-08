"""Shared fixtures: hermetic keyless settings with isolated temp dirs for artifacts."""

from __future__ import annotations

import os

import pytest

from agent.config import Settings, get_settings


def isolate_from_local_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``Settings()`` ignore the developer's ``.env`` and shell variables.

    Settings reads both, case-insensitively. Without this, a developer whose
    ``.env`` is in real mode (``LLM_PROVIDER=openai``, ``CORPUS_DIR=...``) would
    have a local ``pytest`` call OpenAI and Tavily with their own keys and test
    their own corpus instead of the bundled one. CI never saw it: it has no ``.env``.
    """
    fields = set(Settings.model_fields)
    for name in list(os.environ):
        if name.lower() in fields:
            monkeypatch.delenv(name)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()  # the API and run() fall back to this process-wide cache


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    """Run every test on the keyless defaults, whatever the local machine sets.

    A test may still set what it needs (e.g. ``monkeypatch.setenv("TRACES_DIR", ...)``):
    autouse fixtures run first, so that lands on top of this clean slate.
    """
    isolate_from_local_config(monkeypatch)
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path):
    """Keyless settings that write runs/results under a temp dir.

    The providers are spelled out (and ``_env_file=None`` kept) so the fixture
    reads as keyless on its own; ``_hermetic_settings`` above already guarantees it.
    """
    return Settings(
        _env_file=None,
        llm_provider="fake",
        search_provider="fake",
        fetch_provider="fake",
        agent_backend="manual",
        traces_dir=tmp_path / "runs",
        results_dir=tmp_path / "results",
    )


QUESTION = "What are the main approaches to retrieval-augmented generation and their trade-offs?"
