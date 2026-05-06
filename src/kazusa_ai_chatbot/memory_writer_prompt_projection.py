"""Prompt payload projection for memory-writing LLM stages."""

from __future__ import annotations

import copy
from typing import Any


def project_memory_unit_extractor_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the memory-unit extractor payload with safe speaker metadata.

    Args:
        payload: Model-facing memory-unit extractor payload.
        character_name: Exact profile name used in prompt metadata.

    Returns:
        A deep-copied payload where chat speaker metadata exposes direct
        speaker ownership without changing message text.
    """

    required_character_name = _required_character_name(character_name)
    projected_payload = copy.deepcopy(payload)
    rows = projected_payload.get("chat_history_recent")
    if isinstance(rows, list):
        projected_rows = [
            _project_prompt_speaker_row(
                row,
                character_name=required_character_name,
            )
            if isinstance(row, dict)
            else row
            for row in rows
        ]
        projected_payload["chat_history_recent"] = projected_rows
    return projected_payload


def project_memory_unit_rewrite_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the memory-unit rewrite payload as an isolated prompt copy.

    Args:
        payload: Model-facing memory-unit rewrite payload.
        character_name: Exact profile name used in prompt metadata.

    Returns:
        A deep-copied payload. This stage has no speaker rows to project.
    """

    _required_character_name(character_name)
    projected_payload = copy.deepcopy(payload)
    return projected_payload


def project_relationship_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the relationship/global-state payload as an isolated copy.

    Args:
        payload: Model-facing relationship or global-state payload.
        character_name: Exact profile name used in prompt metadata.

    Returns:
        A deep-copied payload. This stage preserves text evidence unchanged.
    """

    _required_character_name(character_name)
    projected_payload = copy.deepcopy(payload)
    return projected_payload


def project_character_image_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the image-summary payload as an isolated prompt copy.

    Args:
        payload: Model-facing character-image payload.
        character_name: Exact profile name used in prompt metadata.

    Returns:
        A deep-copied payload. This stage preserves text evidence unchanged.
    """

    _required_character_name(character_name)
    projected_payload = copy.deepcopy(payload)
    return projected_payload


def project_reflection_promotion_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the reflection-promotion payload as an isolated prompt copy.

    Args:
        payload: Model-facing global reflection promotion payload.
        character_name: Exact profile name used in prompt metadata.

    Returns:
        A deep-copied payload. This stage preserves evidence text unchanged.
    """

    _required_character_name(character_name)
    projected_payload = copy.deepcopy(payload)
    evidence_cards = projected_payload.get("evidence_cards")
    if isinstance(evidence_cards, list):
        for card in evidence_cards:
            if not isinstance(card, dict):
                continue
            if "active_character_utterance" in card:
                card["source_utterance"] = card.pop("active_character_utterance")
    return projected_payload


def _required_character_name(character_name: str) -> str:
    """Validate the exact profile name used in prompt metadata."""

    if not isinstance(character_name, str) or not character_name.strip():
        raise ValueError("character_name is required")
    return character_name


def _project_prompt_speaker_row(
    row: dict,
    *,
    character_name: str,
) -> dict:
    """Project one chat row's speaker metadata without changing text."""

    projected_row: dict[str, Any] = copy.deepcopy(row)
    role = projected_row.pop("role", "")
    display_name = projected_row.pop("display_name", "")

    if role == "assistant":
        projected_row["speaker_name"] = character_name
    elif role == "user":
        if isinstance(display_name, str) and display_name.strip():
            projected_row["speaker_name"] = display_name
    else:
        if isinstance(display_name, str) and display_name.strip():
            projected_row["speaker_name"] = display_name

    return projected_row
