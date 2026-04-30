"""Deterministic helpers for decontextualizer referent-resolution output.

This module owns validation and derived clarification decisions for the
``referents`` list emitted by the message decontextualizer. Inputs are graph
state dictionaries or raw LLM fields; outputs are plain dictionaries and
booleans consumed by RAG gating and cognition prompts. The extension slot is
the shared referent dictionary shape: future ambiguity producers can plug into
the same consumers by emitting ``phrase``, ``referent_role``, and ``status`` fields.
"""

from __future__ import annotations

_REFERENT_ROLES = {"subject", "object", "time"}
_REFERENT_STATUSES = {"resolved", "unresolved"}


def normalize_referents(value: object) -> list[dict[str, str]]:
    """Normalize referent rows emitted by the decontextualizer LLM.

    Args:
        value: Raw ``referents`` field from the model response or graph state.

    Returns:
        Valid referent dictionaries. Malformed rows are dropped so telemetry
        exposes weak model output instead of hiding it behind synthesized data.
    """
    if not isinstance(value, list):
        return_value = []
        return return_value

    normalized_referents = []
    for raw_referent in value:
        if not isinstance(raw_referent, dict):
            continue

        phrase = raw_referent.get("phrase")
        if not isinstance(phrase, str):
            continue
        phrase = phrase.strip()
        if not phrase:
            continue

        referent_role = raw_referent.get("referent_role")
        if not isinstance(referent_role, str):
            continue
        referent_role = referent_role.strip()
        if referent_role not in _REFERENT_ROLES:
            continue

        status = raw_referent.get("status")
        if not isinstance(status, str):
            continue
        status = status.strip()
        if status not in _REFERENT_STATUSES:
            continue

        normalized_referents.append({
            "phrase": phrase,
            "referent_role": referent_role,
            "status": status,
        })

    return_value = normalized_referents
    return return_value


def unresolved_referents(value: object) -> list[dict[str, str]]:
    """Return normalized referents whose status is unresolved.

    Args:
        value: Raw or normalized referents list.

    Returns:
        Normalized unresolved referent rows.
    """
    normalized_referents = normalize_referents(value)
    unresolved = [
        referent
        for referent in normalized_referents
        if referent["status"] == "unresolved"
    ]
    return_value = unresolved
    return return_value


def needs_referent_clarification(referents_value: object) -> bool:
    """Decide whether cognition should clarify unresolved referents.

    Args:
        referents_value: Raw or normalized referents list.

    Returns:
        True when any structured referent is unresolved.
    """
    referents = normalize_referents(referents_value)
    return_value = any(
        referent["status"] == "unresolved"
        for referent in referents
    )
    return return_value


def should_skip_rag_for_unresolved_referents(referents_value: object) -> bool:
    """Decide whether RAG should fail closed for unresolved-only inputs.

    Args:
        referents_value: Raw or normalized referents list.

    Returns:
        True when all structured referents are unresolved, meaning there is no
        concrete referent to retrieve against.
    """
    referents = normalize_referents(referents_value)
    return_value = bool(referents) and all(
        referent["status"] == "unresolved"
        for referent in referents
    )
    return return_value


def unresolved_referent_reason(referents_value: object) -> str:
    """Build the unresolved-referent reason exposed to cognition prompts.

    Args:
        referents_value: Raw or normalized referents list.

    Returns:
        Short reason based on unresolved structured referents, or an empty
        string when no structured ambiguity is present.
    """
    unresolved = unresolved_referents(referents_value)
    if unresolved:
        phrases = "、".join(referent["phrase"] for referent in unresolved)
        return_value = f'缺少以下指代对象: {phrases}'
        return return_value

    return_value = ""
    return return_value
