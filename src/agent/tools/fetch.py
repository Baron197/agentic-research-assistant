"""Fetch tool: ``FetchTool`` Protocol + keyless ``FakeFetch`` + real ``HttpFetch``.

``FakeFetch`` resolves ``local://<file>`` URLs to corpus files so a run is fully
offline. ``HttpFetch`` is a polite, bounded real fetcher (lazy ``httpx`` import)
that returns a page's article text (``readable_text``); it is never used on the
keyless path.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from .documents import read_doc_text

LOCAL_PREFIX = "local://"
_MAX_BYTES = 1_000_000  # politeness/safety bound for the real fetcher
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Whole elements whose text is code or markup, not prose. A bare tag strip keeps
# their contents, which is how CSS, JavaScript and JSON-LD ended up as "evidence".
_NON_TEXT_RE = re.compile(
    r"<(script|style|noscript|template|svg|head)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Less than this much article text means the page had none worth quoting: a video
# page, a login wall, a JS-only app. (A video page's extract is its footer links.)
MIN_ARTICLE_CHARS = 200
# Video pages carry no article text at all; skip them before spending a fetch.
_NO_TEXT_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "tiktok.com")


class NoReadableText(ValueError):
    """The page has no article text to quote; the researcher tries the next result."""


def _crude_text(raw: str) -> str:
    """Tag-stripped text with code blocks and comments removed first."""
    text = _COMMENT_RE.sub(" ", _NON_TEXT_RE.sub(" ", raw))
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text))).strip()


def readable_text(raw: str) -> str:
    """The article text of an HTML page, or ``NoReadableText``.

    trafilatura (installed in the Docker image and the ``real`` extra) finds the
    main content and leaves menus, cookie banners and footers behind. Without it,
    a crude fallback drops script/style blocks before stripping tags. Either way,
    a page with less than ``MIN_ARTICLE_CHARS`` of text is refused, so the run
    moves on to the next search result instead of citing page furniture.
    """
    try:
        import trafilatura
    except ImportError:
        text = _crude_text(raw)
    else:
        try:
            text = trafilatura.extract(raw) or trafilatura.extract(raw, favor_recall=True) or ""
        except Exception:  # noqa: BLE001 - a parser crash on odd HTML: fall back
            text = _crude_text(raw)
    if len(text.strip()) < MIN_ARTICLE_CHARS:
        raise NoReadableText(f"no readable article text ({len(text.strip())} chars)")
    return text


@runtime_checkable
class FetchTool(Protocol):
    name: str

    def fetch(self, url: str) -> str:  # pragma: no cover
        ...


class FakeFetch:
    """Resolve ``local://<file>`` URLs to local corpus document text.

    Handles the same formats as the corpus search (``.md``/``.txt``/``.pdf``) via
    the shared reader, so a run over your own documents fetches them the same way.
    """

    name = "fake-fetch"

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = Path(corpus_dir)

    def fetch(self, url: str) -> str:
        if not url.startswith(LOCAL_PREFIX):
            raise ValueError(f"FakeFetch only resolves {LOCAL_PREFIX} URLs, got {url!r}")
        filename = url[len(LOCAL_PREFIX) :]
        # A corpus URL is a bare filename; reject separators, dot-segments and
        # absolute paths so a crafted URL can never read outside the corpus dir.
        if (not filename or filename in (".", "..")
                or Path(filename).name != filename):
            raise ValueError(f"invalid corpus filename in {url!r}")
        path = self.corpus_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"corpus file not found for {url!r}: {path}")
        return read_doc_text(path)


class HttpFetch:
    """Real, bounded HTTP fetch that returns a page's article text.

    Guardrails: only ``http(s)`` URLs; redirects are followed manually so every
    hop is re-validated; hosts that resolve to private/loopback/link-local
    addresses are refused (SSRF); and the body is *streamed* and cut off at
    ``max_bytes`` actual bytes, so a huge page cannot exhaust memory.
    """

    name = "http-fetch"

    _MAX_REDIRECTS = 5
    # Identify the bot and where to find out about it: Wikipedia, for one, answers
    # a bare product name with 403 (its robot policy asks for contact details).
    _USER_AGENT = "agentic-research-assistant/0.1 (+https://github.com/Baron197/agentic-research-assistant)"

    def __init__(self, timeout: float = 10.0, max_bytes: int = _MAX_BYTES) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes

    @staticmethod
    def _check_url(url: str) -> None:
        """Reject non-http(s) schemes and hosts on private/internal networks."""
        import socket
        from ipaddress import ip_address
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"unsupported URL scheme in {url!r} (http/https only)")
        host = parsed.hostname or ""
        if not host:
            raise ValueError(f"URL has no host: {url!r}")
        for info in socket.getaddrinfo(host, None):
            addr = ip_address(info[4][0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                raise ValueError(f"refusing to fetch private/internal address for {url!r}")

    @staticmethod
    def _check_has_text(url: str) -> None:
        """Refuse hosts whose pages are video players, not articles."""
        host = (urlparse(url).hostname or "").lower()
        if any(host == h or host.endswith("." + h) for h in _NO_TEXT_HOSTS):
            raise NoReadableText(f"video page, no article text: {url!r}")

    def fetch(self, url: str) -> str:  # pragma: no cover - real network path
        import httpx  # lazy

        self._check_has_text(url)
        raw = ""
        with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
            for _ in range(self._MAX_REDIRECTS + 1):
                self._check_url(url)
                request = client.build_request("GET", url, headers={"User-Agent": self._USER_AGENT})
                resp = client.send(request, stream=True)
                try:
                    if resp.is_redirect and resp.next_request is not None:
                        url = str(resp.next_request.url)
                        continue
                    resp.raise_for_status()
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in resp.iter_bytes():
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= self.max_bytes:
                            break
                    body = b"".join(chunks)[: self.max_bytes]
                    raw = body.decode(resp.encoding or "utf-8", errors="replace")
                    break
                finally:
                    resp.close()
            else:
                raise ValueError(f"too many redirects fetching {url!r}")
        return readable_text(raw)


def get_fetch(settings: Any) -> FetchTool:
    """Factory: pick a fetch tool from config and optionally wrap with cache."""
    if settings.fetch_provider == "http":
        tool: FetchTool = HttpFetch()
    else:
        tool = FakeFetch(corpus_dir=settings.corpus_dir)

    if settings.enable_cache:
        from ..cache import CachedFetch

        return CachedFetch(tool, maxsize=settings.cache_size)  # type: ignore[return-value]
    return tool
