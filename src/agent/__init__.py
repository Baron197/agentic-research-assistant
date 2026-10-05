"""Agentic Research & Report Assistant.

A multi-agent (LangGraph) pipeline that plans research, gathers evidence with
tools, drafts a structured report, has a critic check each claim against its
cited evidence, enforces guardrails, and returns a cited report with full
observability. Runs fully keyless with deterministic fake providers.
"""

from __future__ import annotations

__version__ = "0.1.0"
