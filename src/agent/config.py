"""Typed configuration in one place (Strategy-pattern friendly).

All settings come from environment / ``.env`` via ``pydantic-settings``. The
defaults are deliberately *keyless and offline*: with no ``.env`` and no
environment variables the whole system runs with deterministic fake providers
and zero cost. Real mode (OpenAI + a real search/fetch tool) is opt-in purely
by flipping the ``*_provider`` fields and supplying keys.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = three levels up from this file (src/agent/config.py -> repo).
# Resolving paths against the root keeps corpus/runs lookups stable regardless
# of the current working directory (tests, uvicorn, Streamlit, eval all differ).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Why a single settings object: every swappable component (LLM, search, fetch)
    is chosen by a factory from these typed fields, so configuration lives in one
    auditable place instead of being scattered across modules.
    """

    model_config = SettingsConfigDict(
        # Anchored at the repo root (like the paths below) so the same .env is
        # found no matter which directory uvicorn/streamlit/pytest run from.
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider selection (the Strategy switches) -------------------------
    llm_provider: Literal["fake", "openai"] = "fake"
    search_provider: Literal["fake", "web"] = "fake"
    fetch_provider: Literal["fake", "http"] = "fake"

    # --- OpenAI (real LLM path only) ----------------------------------------
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- Real web search path only ------------------------------------------
    search_api_key: str = ""
    search_backend: Literal["tavily", "brave", "serpapi"] = "tavily"

    # --- Run knobs ----------------------------------------------------------
    max_iterations: int = 2
    token_budget: int = 60_000
    require_approval: bool = False
    top_search_results: int = 5
    # Depth: evidence gathered (and claims written) per sub-question. Raise for
    # more thorough reports; the researcher fetches sources in parallel so deeper
    # runs stay fast. Default 2 keeps the keyless demo output stable.
    evidence_per_subquestion: int = 2
    # Parallel search/fetch workers in the researcher (real-mode latency win; the
    # output is identical regardless of this value). 1 = no thread pool.
    research_concurrency: int = 4
    enable_critic: bool = True
    max_question_length: int = 2_000

    # --- Agent backend: hand-written ("manual") vs DSPy ---------------------
    agent_backend: Literal["manual", "dspy"] = "manual"
    dspy_model: str = "openai/gpt-4o-mini"
    dspy_optimizer: Literal["bootstrap", "mipro"] = "bootstrap"
    dspy_artifact_path: Path = PROJECT_ROOT / "artifacts" / "dspy_program.json"

    # --- Caching ------------------------------------------------------------
    enable_cache: bool = True
    cache_size: int = 256

    # --- Paths (absolute, anchored at the repo root) ------------------------
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    traces_dir: Path = PROJECT_ROOT / "runs"
    results_dir: Path = PROJECT_ROOT / "eval" / "results"

    # --- Cross-field validation ----------------------------------------------
    @model_validator(mode="after")
    def _check_provider_mix(self) -> Settings:
        """Fail fast on provider combinations that would crash every run."""
        if self.search_provider == "web" and self.fetch_provider == "fake":
            raise ValueError(
                "search_provider='web' returns real http(s) URLs, which FakeFetch "
                "cannot resolve; set FETCH_PROVIDER=http as well."
            )
        if self.search_provider == "fake" and self.fetch_provider == "http":
            raise ValueError(
                "search_provider='fake' returns local:// URLs, which HttpFetch "
                "cannot resolve; set SEARCH_PROVIDER=web too (or FETCH_PROVIDER=fake)."
            )
        return self

    # --- Convenience flags --------------------------------------------------
    @property
    def is_keyless(self) -> bool:
        """True when no external service is configured (pure offline mode)."""
        return (
            self.llm_provider == "fake"
            and self.search_provider == "fake"
            and self.fetch_provider == "fake"
            and self.agent_backend == "manual"  # the DSPy backend runs a real, paid LLM
        )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings.

    Tests that need different values construct ``Settings(**overrides)`` directly
    and pass them in, which avoids mutating global state.
    """
    return Settings()
