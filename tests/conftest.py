"""Shared fixtures: keyless settings with isolated temp dirs for artifacts."""

from __future__ import annotations

import pytest

from agent.config import Settings


@pytest.fixture
def settings(tmp_path):
    """Keyless settings that write runs/results under a temp dir.

    ``_env_file=None`` isolates the suite from a developer's local ``.env`` —
    otherwise e.g. ``AGENT_BACKEND=dspy`` there would silently change what the
    tests exercise.
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
