"""Fetch vendor page HTML."""

from __future__ import annotations

import urllib.request


def fetch_page(url: str) -> str:
    """Fetch one vendor page and return decoded HTML."""

    with urllib.request.urlopen(url) as response:
        body = response.read()
    html = body.decode("utf-8", errors="replace")
    return html
