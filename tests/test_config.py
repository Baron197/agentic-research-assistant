"""Settings validation: provider-mix checks and the keyless flag."""

from __future__ import annotations

import pytest

from agent.config import Settings


def test_incompatible_provider_mix_is_rejected():
    # web search returns http(s) URLs FakeFetch cannot resolve (and vice versa);
    # this must fail at construction, not crash every run (regression).
    with pytest.raises(ValueError):
        Settings(_env_file=None, search_provider="web", fetch_provider="fake",
                 search_api_key="k")
    with pytest.raises(ValueError):
        Settings(_env_file=None, search_provider="fake", fetch_provider="http")


def test_matched_real_providers_are_accepted():
    s = Settings(_env_file=None, search_provider="web", fetch_provider="http",
                 search_api_key="k")
    assert s.search_provider == "web"


def test_api_keys_are_stripped_of_whitespace():
    # A key stored via `echo` / Secret Manager carries a trailing newline, and a
    # PowerShell pipe adds CRLF; httpx rejects either as an illegal header value,
    # which would fail every real-mode call (regression guard for deployments).
    s = Settings(_env_file=None, openai_api_key="sk-abc\r\n", search_api_key="  tvly-xyz\n")
    assert s.openai_api_key == "sk-abc"
    assert s.search_api_key == "tvly-xyz"


def test_is_keyless_false_for_dspy_backend():
    # The DSPy backend runs a real, paid LLM even with fake tools; /health must
    # not report such a deployment as keyless (regression).
    assert Settings(_env_file=None).is_keyless is True
    assert Settings(_env_file=None, agent_backend="dspy").is_keyless is False
