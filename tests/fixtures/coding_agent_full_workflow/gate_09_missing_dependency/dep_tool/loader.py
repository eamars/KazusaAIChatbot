"""Config loading helpers with an intentionally incomplete implementation."""

from __future__ import annotations


def load_config(text: str) -> dict[str, str]:
    """Load configuration text into a mapping."""

    return {"raw": text.strip()}
