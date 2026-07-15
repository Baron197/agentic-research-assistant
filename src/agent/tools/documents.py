"""Corpus document handling shared by the keyless search and fetch tools.

The keyless pipeline researches a *folder of documents* (``corpus_dir``). By
default that is the bundled ``data/corpus`` RAG primer, but pointing
``CORPUS_DIR`` at your own folder turns the app into a researcher over **your own
notes** — which is the whole point of "work mode".

Supported formats: Markdown / plain text out of the box, and **PDF** when the
optional ``pypdfium2`` package is installed. Extraction is text-only, so a
scanned/image PDF with no text layer yields an empty string — the pipeline then
honestly abstains rather than inventing content (no OCR).
"""

from __future__ import annotations

from pathlib import Path

# Extensions treated as corpus documents. ``.pdf`` needs the optional pypdfium2.
SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".text", ".pdf")


def iter_corpus_files(corpus_dir: Path) -> list[Path]:
    """Return the supported corpus documents in ``corpus_dir``, sorted by name."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        return []
    return sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def read_doc_text(path: Path) -> str:
    """Return the plain-text content of a corpus document.

    Markdown/text are read directly (unknown bytes replaced, so odd encodings in
    real user files never crash a run); PDFs are extracted via ``pypdfium2``.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _read_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_text(path: Path) -> str:
    """Extract text from a PDF (text layer only) using the optional pypdfium2."""
    try:
        import pypdfium2 as pdfium  # optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            f"reading {path.name} needs the optional 'pypdfium2' package "
            "(pip install pypdfium2) to research PDF documents."
        ) from exc

    pdf = pdfium.PdfDocument(str(path))
    try:
        parts: list[str] = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            try:
                parts.append(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
        return "\n".join(parts).strip()
    finally:
        pdf.close()


def parse_doc(path: Path) -> tuple[str, str]:
    """Return ``(title, body)`` for a corpus document.

    The title is the first Markdown ``# heading`` when present, otherwise a
    prettified filename; the body excludes that heading line so snippets are real
    prose rather than the title.
    """
    text = read_doc_text(path)
    lines = text.splitlines()
    title = path.stem.replace("-", " ").replace("_", " ").title()
    body_lines = lines
    for i, line in enumerate(lines):
        if line.strip():
            if line.lstrip().startswith("#"):
                title = line.lstrip("#").strip()
                body_lines = lines[i + 1:]
            break
    return title, "\n".join(body_lines).strip()
