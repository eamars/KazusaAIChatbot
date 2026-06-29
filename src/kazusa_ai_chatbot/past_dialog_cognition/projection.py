"""Prompt-safe projection from parsed trace steps to L2a context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.past_dialog_cognition.models import (
    PastDialogCognitionCandidate,
)


PAST_DIALOG_COGNITION_STAGE_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "l2a_conscious_framing": (
        ("internal_monologue", "private thought"),
        ("logical_stance", "earlier stance"),
        ("character_intent", "earlier intent"),
    ),
    "l2c1_judgment_synthesis": (
        ("logical_stance", "settled stance"),
        ("character_intent", "settled intent"),
        ("judgment_note", "judgment note"),
    ),
}
PAST_DIALOG_COGNITION_STAGE_NAMES = tuple(
    PAST_DIALOG_COGNITION_STAGE_FIELDS.keys()
)
_VISIBLE_DIALOG_CHAR_LIMIT = 360
_TRACE_FIELD_CHAR_LIMIT = 520


def clip_prompt_text(text: str, *, limit: int) -> str:
    """Return stripped text within a prompt-facing character limit.

    Args:
        text: Raw text value to clip.
        limit: Maximum number of characters to keep.

    Returns:
        Stripped text, truncated with an ASCII ellipsis marker when needed.
    """

    clean_text = text.strip()
    if limit <= 0:
        clipped_text = ""
        return clipped_text
    if len(clean_text) <= limit:
        clipped_text = clean_text
        return clipped_text
    clipped_text = clean_text[: max(0, limit - 3)].rstrip() + "..."
    return clipped_text


def project_candidate_trace_context(
    candidate: PastDialogCognitionCandidate,
    trace_steps: Sequence[Mapping[str, Any]],
) -> str:
    """Build compact L2a context for one candidate and its parsed trace steps.

    Args:
        candidate: Already-attached past dialog candidate.
        trace_steps: Trace-step rows already filtered to approved stages.

    Returns:
        Prompt-facing natural-language context, or an empty string when no
        approved parsed fields are available.
    """

    projected_lines: list[str] = []
    for step in _sorted_steps(trace_steps):
        projected_lines.extend(_project_step_lines(step))

    if not projected_lines:
        projected_context = ""
        return projected_context

    visible_text = clip_prompt_text(
        candidate.visible_text,
        limit=_VISIBLE_DIALOG_CHAR_LIMIT,
    )
    context_lines = [
        "Attached earlier character dialog:",
        f"- Visible dialog: {visible_text}",
        "- Private cognition around that dialog:",
        *projected_lines,
    ]
    projected_context = "\n".join(context_lines)
    return projected_context


def join_projected_contexts(
    contexts: Sequence[str],
    *,
    context_char_limit: int,
) -> str:
    """Join candidate context blocks under a shared character budget.

    Args:
        contexts: Non-empty per-dialog context blocks.
        context_char_limit: Maximum characters in the combined prompt value.

    Returns:
        Combined context text bounded by ``context_char_limit``.
    """

    clean_contexts = [
        context.strip()
        for context in contexts
        if isinstance(context, str) and context.strip()
    ]
    joined_context = "\n\n".join(clean_contexts)
    clipped_context = clip_prompt_text(
        joined_context,
        limit=context_char_limit,
    )
    return clipped_context


def _sorted_steps(
    trace_steps: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return trace steps sorted by numeric sequence when available."""

    sorted_steps = sorted(trace_steps, key=_step_sequence)
    return sorted_steps


def _step_sequence(step: Mapping[str, Any]) -> int:
    """Return a stable sort key for one trace step."""

    sequence = step.get("sequence")
    if isinstance(sequence, int) and not isinstance(sequence, bool):
        return_value = sequence
    else:
        return_value = 0
    return return_value


def _project_step_lines(step: Mapping[str, Any]) -> list[str]:
    """Project approved parsed fields from one trace step."""

    stage_name_value = step.get("stage_name")
    stage_name = stage_name_value if isinstance(stage_name_value, str) else ""
    field_specs = PAST_DIALOG_COGNITION_STAGE_FIELDS.get(stage_name)
    if field_specs is None:
        projected_lines: list[str] = []
        return projected_lines

    parsed_output = step.get("parsed_output")
    if not isinstance(parsed_output, Mapping):
        projected_lines = []
        return projected_lines

    projected_lines = []
    for field_name, label in field_specs:
        raw_value = parsed_output.get(field_name)
        text = _prompt_field_text(raw_value)
        if not text:
            continue
        clipped_text = clip_prompt_text(text, limit=_TRACE_FIELD_CHAR_LIMIT)
        projected_lines.append(f"  - {label}: {clipped_text}")
    return projected_lines


def _prompt_field_text(value: object) -> str:
    """Return scalar trace text safe for prompt-facing residual projection."""

    if not isinstance(value, str):
        text = ""
        return text
    text = value.strip()
    return text
