"""HTTP helpers with retries and browser-like headers.

Some data providers (including OECD pages) may block default python-requests user agents.
We set conservative browser-like headers to reduce spurious 403s.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_HEADERS: Mapping[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_SESSION = requests.Session()
_SESSION.headers.update(DEFAULT_HEADERS)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_file(
    url: str,
    destination: Path,
    timeout: int = 120,
    headers: Optional[Mapping[str, str]] = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _SESSION.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_text(url: str, timeout: int = 60, headers: Optional[Mapping[str, str]] = None) -> str:
    response = _SESSION.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.text
