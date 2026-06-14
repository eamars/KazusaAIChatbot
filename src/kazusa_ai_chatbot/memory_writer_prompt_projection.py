"""Prompt payload projection for memory-writing LLM stages."""

from __future__ import annotations

import copy


def project_memory_unit_extractor_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the memory-unit extractor payload as an isolated prompt copy.

    Args:
        payload: Model-facing memory-unit extractor payload.
        character_name: Exact profile name used in prompt metadata.

    Returns:
        A deep-copied payload. Conversation history is already projected
        upstream before this prompt hook runs.
    """

    _required_character_name(character_name)
    projected_payload = copy.deepcopy(payload)
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


