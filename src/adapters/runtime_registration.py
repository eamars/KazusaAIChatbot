"""Shared runtime adapter registration response validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def character_name_from_registration_response(payload: Any) -> str:
    """Return the required brain-owned character name.

    Args:
        payload: Decoded registration or heartbeat response body.

    Returns:
        Non-empty character name supplied by the brain.

    Raises:
        ValueError: If the response is not an object or does not contain a
            non-empty string ``character_name``.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("runtime adapter registration response must be an object")

    raw_character_name = payload.get("character_name")
    if not isinstance(raw_character_name, str):
        raise ValueError(
            "runtime adapter registration response character_name "
            "must be a string"
        )

    character_name = raw_character_name.strip()
    if not character_name:
        raise ValueError(
            "runtime adapter registration response character_name is required"
        )
    return character_name
