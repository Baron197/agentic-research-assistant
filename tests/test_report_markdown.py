"""Untrusted text in the rendered report: shown as text, never as markup.

Page titles and snippets come from the web, and claims come from a model that read
those pages, so either can carry Markdown. An image is the dangerous kind: the
browser showing the report loads it on sight, which leaks the reader's IP (and
anything in its URL) to whoever chose it. A link can hide where it goes.
"""

from __future__ import annotations

import re

import pytest

from agent.runner import render_report_markdown
from agent.schemas import Citation, Claim, Report, ReportSection
from agent.textutil import escape_markdown, markdown_url

PIXEL = "![x](http://127.0.0.1:9/pixel.png?q=secret)"
LINK = "[the official docs](https://evil.example)"
# An unescaped bracket: what a link or image would need.
_BARE_BRACKET = re.compile(r"(?<!\\)[\[\]]")
# Markup characters, plus what GFM links on its own: "://" after a web scheme,
# "." after "www", and "@". Parentheses are not markup without a "]" before them.
_MARKUP = r"[`*_~\[\]!#<>$@]|(?i:(?<=http)|(?<=https)|(?<=ftp)):(?=//)|(?i:(?<=www))\."


def test_escape_markdown_neutralises_an_image():
    assert escape_markdown(PIXEL) == r"\!\[x\](http\://127.0.0.1:9/pixel.png?q=secret)"


def test_escape_markdown_neutralises_a_link():
    assert escape_markdown(LINK) == r"\[the official docs\](https\://evil.example)"


def test_escape_markdown_escapes_what_gfm_links_on_its_own():
    # Escaped, GitHub shows these as text, and Streamlit (which links them anyway)
    # no longer pulls the backslash of a following escape into the address.
    assert escape_markdown("see https://a.example/x_y, www.b.example or me@c.example") == \
        r"see https\://a.example/x\_y, www\.b.example or me\@c.example"


@pytest.mark.parametrize("text", [
    "[![x](http://127.0.0.1:9/p.png)](https://evil.example)",  # an image inside a link
    "![a [b] c](http://127.0.0.1:9/p.png)",
    "[[nested]](https://evil.example)",
    r"\![x](http://127.0.0.1:9/p.png)",  # a backslash meant to cancel the escape
    "<https://evil.example> <img src=x>",
    "`code` *em* _em_ ~~del~~ $x$ # not a heading > not a quote",
    "HTTPS://evil.example WWW.evil.example ftp://evil.example mailto:me@evil.example",
])
def test_escape_markdown_leaves_no_markup(text):
    escaped = escape_markdown(text)
    # Every markup character is preceded by an odd run of backslashes: escaped.
    for m in re.finditer(rf"(\\*)({_MARKUP})", escaped):
        assert len(m.group(1)) % 2 == 1, (escaped, m.start())


def test_escape_markdown_keeps_ordinary_text_and_joins_lines():
    claim = "Hybrid search combines BM25 and dense (semantic) vectors, cutting misses by 30%."
    assert escape_markdown(claim) == claim
    assert escape_markdown("one\n\n# two\n> three") == r"one \# two \> three"


def test_markdown_url_links_web_urls_to_themselves_and_nothing_else():
    wiki = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    assert markdown_url(wiki) == f"<{wiki}>"  # nothing inside an autolink is markup
    assert markdown_url("https://x.example/a b<c>") == "<https://x.example/a%20b%3Cc%3E>"
    assert markdown_url("local://rag-basics.md") == "local://rag-basics.md"
    assert markdown_url("javascript:alert(1)") == "javascript:alert(1)"  # text, not a link


def _hostile_report() -> Report:
    return Report(
        question=f"What is RAG? {PIXEL}",
        summary=f"Summary {LINK}",
        sections=[ReportSection(heading=f"Core {PIXEL}", claims=[
            Claim(id="c1", text=f"RAG retrieves passages {PIXEL}", evidence_ids=["e1", "e2"]),
            Claim(id="c2", text="A plain claim [1](https://evil.example).", evidence_ids=["e2"]),
        ])],
        sources=[
            Citation(n=1, evidence_id="e1", title=f"Page {PIXEL}", url="https://good.example/a"),
            Citation(n=2, evidence_id="e2", title=LINK, url="local://notes.md"),
        ],
    )


def test_report_escapes_every_untrusted_field_and_keeps_its_structure():
    md = render_report_markdown(_hostile_report())

    # The only bare brackets left are the citation markers the renderer adds, so
    # no link or image can open anywhere.
    for line in md.splitlines():
        assert not _BARE_BRACKET.search(re.sub(r" (\[\d+\])+$", "", line)), line
    assert re.search(r"^- RAG retrieves passages .+ \[1\]\[2\]$", md, re.MULTILINE)
    assert re.search(r"^- A plain claim \\\[1\\\]\(https\\://evil\.example\)\. \[2\]$",
                     md, re.MULTILINE)
    assert md.startswith("# Research Report\n")
    assert "\n## Executive Summary\n" in md and "\n## Sources\n" in md
    assert r"1. Page \!\[x\]" in md and md.count("<https://good.example/a>") == 1


def test_rendered_report_shows_no_image_and_links_only_to_sources():
    # The same check as a browser would make it: render the Markdown to HTML.
    markdown_it = pytest.importorskip("markdown_it")
    page = markdown_it.MarkdownIt("commonmark").render(render_report_markdown(_hostile_report()))

    assert "<img" not in page
    assert re.findall(r'<a href="([^"]*)"', page) == ["https://good.example/a"]
    assert "http://127.0.0.1:9/pixel.png" in page  # still there, as text
    assert "[1][2]</li>" in page  # citation markers render as they are
