"""Fetch tool: ``FetchTool`` Protocol + keyless ``FakeFetch`` + real ``HttpFetch``.

``FakeFetch`` resolves ``local://<file>`` URLs to corpus files so a run is fully
offline. ``HttpFetch`` is a polite, bounded real fetcher (lazy ``httpx`` import)
with a minimal readable-text extraction; it is never used on the keyless path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .documents import read_doc_text

LOCAL_PREFIX = "local://"
_MAX_BYTES = 1_000_000  # politeness/safety bound for the real fetcher
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


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
    """Real, bounded HTTP fetch with naive readable-text extraction.

    Guardrails: only ``http(s)`` URLs; redirects are followed manually so every
    hop is re-validated; hosts that resolve to private/loopback/link-local
    addresses are refused (SSRF); and the body is *streamed* and cut off at
    ``max_bytes`` actual bytes, so a huge page cannot exhaust memory.
    """

    name = "http-fetch"

    _MAX_REDIRECTS = 5
    _USER_AGENT = "agentic-research-assistant/0.1"

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

    def fetch(self, url: str) -> str:  # pragma: no cover - real network path
        import httpx  # lazy

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
        # Prefer trafilatura if available; otherwise strip tags crudely.
        try:
            import trafilatura

            extracted = trafilatura.extract(raw)
            if extracted:
                return extracted
        except Exception:
            pass
        return _WS_RE.sub(" ", _TAG_RE.sub(" ", raw)).strip()


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
