"""Slug normalization helpers."""

from __future__ import annotations


def slugify(value: str) -> str:
    """Convert user-visible text into a URL slug."""

    lowered = value.strip().lower()
    replaced = lowered.replace(" ", "-").replace("_", "-")
    stripped = "".join(
        character
        for character in replaced
        if character.isalnum() or character == "-"
    )
    return stripped.strip("-")
