"""Daily reflection promotion into abstract interaction style images."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.db import (
    CharacterReflectionRunDoc,
    empty_interaction_style_overlay,
    get_group_channel_style_image,
    get_user_style_image,
    resolve_single_private_scope_user_id,
    upsert_group_channel_style_image,
    upsert_user_style_image,
    validate_interaction_style_overlay,
)
from kazusa_ai_chatbot.reflection_cycle import repository
from kazusa_ai_chatbot.reflection_cycle.models import (
    REFLECTION_RUN_KIND_DAILY_CHANNEL,
    REFLECTION_STATUS_SUCCEEDED,
    ReflectionWorkerResult,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty


logger = logging.getLogger(__name__)

REFLECTION_RUN_KIND_DAILY_INTERACTION_STYLE_UPDATE = "daily_interaction_style_update"
_ELIGIBLE_DAILY_CONFIDENCE = {"medium", "high"}
_GUIDELINE_FIELDS = (
    "speech_guidelines",
    "social_guidelines",
    "pacing_guidelines",
    "engagement_guidelines",
)
_STYLE_SIGNAL_ITEM_LIMIT = 5
_STYLE_SIGNAL_ITEM_CHARS = 180


_INTERACTION_STYLE_EXTRACTOR_PROMPT = """\
You update a fictional chat character's abstract interaction-style overlay.

# Language Policy
- Output JSON keys and enum values exactly as specified.
- Free-text guideline values should be written in Simplified Chinese unless the
  visible input already uses a specific non-Chinese term that must be preserved.

# Core Task
Transform daily reflection quality signals into abstract handling guidance for
future wording and social pacing.

This stage is the sanitization boundary. The output must describe how the
active character should interact, not what happened.

# Generation Procedure
1. Read `channel_type` and `daily_confidence`.
2. Read `current_overlay` as the currently active handling policy.
3. Read `conversation_quality_patterns` and `synthesis_limitations` only as
   abstract learning signals.
4. Preserve useful existing guidance when the new signal still supports it.
5. Add, refine, or remove guidance only when the daily signal justifies it.
6. Write positive action-oriented guidance. Do not write event summaries,
   identities, dates, message ids, topic recaps, quotes, or source references.
7. Return an empty overlay only when the daily signal is not useful enough to
   maintain or update the current handling policy.

# Input Format
{
  "channel_type": "private|group",
  "daily_confidence": "medium|high",
  "conversation_quality_patterns": ["abstract quality signal"],
  "synthesis_limitations": ["abstract limitation signal"],
  "current_overlay": {
    "speech_guidelines": ["current handling guidance"],
    "social_guidelines": ["current handling guidance"],
    "pacing_guidelines": ["current handling guidance"],
    "engagement_guidelines": ["current handling guidance"],
    "confidence": "low|medium|high|"
  }
}

# Output Format
Return only a valid JSON object:
{
  "overlay": {
    "speech_guidelines": ["abstract speech guidance"],
    "social_guidelines": ["abstract social handling guidance"],
    "pacing_guidelines": ["abstract pacing guidance"],
    "engagement_guidelines": ["abstract engagement guidance"],
    "confidence": "medium|high|"
  }
}
"""
_interaction_style_extractor_llm = get_llm(
    temperature=0.15,
    top_p=0.8,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


async def extract_user_style_overlay_from_daily_reflection(
    *,
    daily_doc: CharacterReflectionRunDoc,
    current_overlay: dict,
) -> dict:
    """Extract a user-scoped style overlay from one private daily reflection.

    Args:
        daily_doc: Succeeded private daily reflection document.
        current_overlay: Current sanitized user style overlay.

    Returns:
        Sanitized overlay candidate for persistence.
    """

    payload = _daily_style_signal_payload(
        daily_doc=daily_doc,
        current_overlay=current_overlay,
    )
    overlay = await _run_interaction_style_extractor(payload)
    return overlay


async def extract_group_channel_style_overlay_from_daily_reflection(
    *,
    daily_doc: CharacterReflectionRunDoc,
    current_overlay: dict,
) -> dict:
    """Extract a group-channel style overlay from one group daily reflection.

    Args:
        daily_doc: Succeeded group daily reflection document.
        current_overlay: Current sanitized group channel style overlay.

    Returns:
        Sanitized overlay candidate for persistence.
    """

    payload = _daily_style_signal_payload(
        daily_doc=daily_doc,
        current_overlay=current_overlay,
    )
    overlay = await _run_interaction_style_extractor(payload)
    return overlay


async def _run_interaction_style_extractor(payload: dict) -> dict:
    """Run the style extraction LLM and validate its overlay output."""

    system_prompt = SystemMessage(content=_INTERACTION_STYLE_EXTRACTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(payload, ensure_ascii=False)
    )
    response = await _interaction_style_extractor_llm.ainvoke(
        [system_prompt, human_message]
    )
    parsed = parse_llm_json_output(str(response.content))
    raw_overlay = parsed.get("overlay")
    if not isinstance(raw_overlay, dict):
        raw_overlay = {}
    overlay = validate_interaction_style_overlay(raw_overlay)
    return overlay


async def run_daily_interaction_style_update(
    *,
    character_local_date: str,
    dry_run: bool,
    is_primary_interaction_busy: Callable[[], bool],
) -> ReflectionWorkerResult:
    """Update interaction style images from succeeded daily reflections.

    Args:
        character_local_date: Character-local date whose daily reflections
            should be considered.
        dry_run: Whether to skip persistence writes after extraction.
        is_primary_interaction_busy: Runtime busy probe used by reflection
            workers to avoid competing with live chat.

    Returns:
        Worker summary for the style update pass.
    """

    result = ReflectionWorkerResult(
        run_kind=REFLECTION_RUN_KIND_DAILY_INTERACTION_STYLE_UPDATE,
        dry_run=dry_run,
    )
    if is_primary_interaction_busy():
        result.deferred = True
        result.defer_reason = "primary interaction busy"
        return result

    daily_docs = await repository.daily_channel_runs(
        character_local_date=character_local_date,
    )
    if not daily_docs:
        result.skipped_count = 1
        result.defer_reason = "no daily channel reflections"
        return result

    for daily_doc in daily_docs:
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break
        await _process_daily_doc(
            daily_doc=daily_doc,
            dry_run=dry_run,
            result=result,
        )

    logger.info(
        "Reflection interaction-style pass complete: "
        f"date={character_local_date} processed={result.processed_count} "
        f"succeeded={result.succeeded_count} failed={result.failed_count} "
        f"skipped={result.skipped_count} deferred={result.deferred} "
        f"reason={result.defer_reason}"
    )
    return result


async def _process_daily_doc(
    *,
    daily_doc: CharacterReflectionRunDoc,
    dry_run: bool,
    result: ReflectionWorkerResult,
) -> None:
    """Process one daily reflection document for style-image updates."""

    if not _daily_doc_is_candidate(daily_doc):
        result.skipped_count += 1
        return

    confidence = _daily_confidence(daily_doc)
    if confidence not in _ELIGIBLE_DAILY_CONFIDENCE:
        result.skipped_count += 1
        logger.info(
            "Reflection style update skipped: "
            f"run_id={daily_doc['run_id']} reason=ineligible_confidence "
            f"confidence={confidence or '<empty>'}"
        )
        return

    result.processed_count += 1
    result.run_ids.append(str(daily_doc["run_id"]))
    scope = daily_doc["scope"]
    try:
        if scope["channel_type"] == "private":
            updated = await _process_private_daily_doc(
                daily_doc=daily_doc,
                dry_run=dry_run,
            )
        elif scope["channel_type"] == "group":
            updated = await _process_group_daily_doc(
                daily_doc=daily_doc,
                dry_run=dry_run,
            )
        else:
            updated = False
    except Exception as exc:
        result.failed_count += 1
        logger.exception(
            "Reflection style update failed: "
            f"run_id={daily_doc['run_id']} error={type(exc).__name__}: {exc}"
        )
        return

    if updated:
        result.succeeded_count += 1
    else:
        result.skipped_count += 1


async def _process_private_daily_doc(
    *,
    daily_doc: CharacterReflectionRunDoc,
    dry_run: bool,
) -> bool:
    """Update one user style image from a private daily reflection."""

    scope = daily_doc["scope"]
    global_user_id = await resolve_single_private_scope_user_id(
        platform=scope["platform"],
        platform_channel_id=scope["platform_channel_id"],
        start_timestamp=daily_doc["window_start"],
        end_timestamp=daily_doc["window_end"],
        character_global_user_id=CHARACTER_GLOBAL_USER_ID,
    )
    if not global_user_id:
        logger.info(
            "Reflection style update skipped: "
            f"run_id={daily_doc['run_id']} reason=unresolved_private_user"
        )
        return False

    current_doc = await get_user_style_image(global_user_id)
    if _style_doc_contains_source_run(current_doc, str(daily_doc["run_id"])):
        logger.info(
            "Reflection style update skipped: "
            f"run_id={daily_doc['run_id']} reason=already_applied"
        )
        return False

    current_overlay = _current_overlay_from_doc(current_doc)
    try:
        overlay = await extract_user_style_overlay_from_daily_reflection(
            daily_doc=daily_doc,
            current_overlay=current_overlay,
        )
    except ValueError as exc:
        logger.warning(
            "Reflection style update skipped: "
            f"run_id={daily_doc['run_id']} reason=style_extraction_rejected "
            f"error={type(exc).__name__}: {exc}"
        )
        return False

    if _overlay_is_empty(overlay):
        logger.info(
            "Reflection style update skipped: "
            f"run_id={daily_doc['run_id']} reason=style_extraction_empty_output"
        )
        return False
    if not dry_run:
        await upsert_user_style_image(
            global_user_id=global_user_id,
            overlay=overlay,
            source_reflection_run_ids=_source_run_ids(daily_doc),
        )
    return True


async def _process_group_daily_doc(
    *,
    daily_doc: CharacterReflectionRunDoc,
    dry_run: bool,
) -> bool:
    """Update one group channel style image from a group daily reflection."""

    scope = daily_doc["scope"]
    current_doc = await get_group_channel_style_image(
        platform=scope["platform"],
        platform_channel_id=scope["platform_channel_id"],
    )
    if _style_doc_contains_source_run(current_doc, str(daily_doc["run_id"])):
        logger.info(
            "Reflection style update skipped: "
            f"run_id={daily_doc['run_id']} reason=already_applied"
        )
        return False

    current_overlay = _current_overlay_from_doc(current_doc)
    try:
        overlay = await extract_group_channel_style_overlay_from_daily_reflection(
            daily_doc=daily_doc,
            current_overlay=current_overlay,
        )
    except ValueError as exc:
        logger.warning(
            "Reflection style update skipped: "
            f"run_id={daily_doc['run_id']} reason=style_extraction_rejected "
            f"error={type(exc).__name__}: {exc}"
        )
        return False

    if _overlay_is_empty(overlay):
        logger.info(
            "Reflection style update skipped: "
            f"run_id={daily_doc['run_id']} reason=style_extraction_empty_output"
        )
        return False
    if not dry_run:
        await upsert_group_channel_style_image(
            platform=scope["platform"],
            platform_channel_id=scope["platform_channel_id"],
            overlay=overlay,
            source_reflection_run_ids=_source_run_ids(daily_doc),
        )
    return True


def _daily_doc_is_candidate(daily_doc: CharacterReflectionRunDoc) -> bool:
    """Return whether a reflection run can feed style extraction."""

    if daily_doc["run_kind"] != REFLECTION_RUN_KIND_DAILY_CHANNEL:
        return False
    if daily_doc["status"] != REFLECTION_STATUS_SUCCEEDED:
        return False
    channel_type = daily_doc["scope"]["channel_type"]
    return_value = channel_type in {"private", "group"}
    return return_value


def _daily_confidence(daily_doc: CharacterReflectionRunDoc) -> str:
    """Return normalized daily reflection confidence."""

    output = daily_doc["output"]
    confidence = text_or_empty(output.get("confidence")).strip().lower()
    return confidence


def _daily_style_signal_payload(
    *,
    daily_doc: CharacterReflectionRunDoc,
    current_overlay: dict,
) -> dict:
    """Build the compact style-extraction prompt payload."""

    output = daily_doc["output"]
    scope = daily_doc["scope"]
    clean_current_overlay = validate_interaction_style_overlay(current_overlay)
    payload = {
        "channel_type": scope["channel_type"],
        "daily_confidence": _daily_confidence(daily_doc),
        "conversation_quality_patterns": _compact_text_items(
            output.get("conversation_quality_patterns")
        ),
        "synthesis_limitations": _compact_text_items(
            output.get("synthesis_limitations")
        ),
        "current_overlay": clean_current_overlay,
    }
    return payload


def _compact_text_items(value: Any) -> list[str]:
    """Project arbitrary reflection output text lists into a compact list."""

    if isinstance(value, list):
        raw_items = value
    elif value:
        raw_items = [value]
    else:
        raw_items = []
    compact_items: list[str] = []
    for item in raw_items:
        text = " ".join(str(item).strip().split())
        if not text:
            continue
        compact_items.append(text[:_STYLE_SIGNAL_ITEM_CHARS].strip())
        if len(compact_items) >= _STYLE_SIGNAL_ITEM_LIMIT:
            break
    return compact_items


def _current_overlay_from_doc(document: dict | None) -> dict:
    """Return the current stored overlay or an empty overlay."""

    if document is None:
        return_value = empty_interaction_style_overlay()
        return return_value
    overlay = document.get("overlay")
    if not isinstance(overlay, dict):
        return_value = empty_interaction_style_overlay()
        return return_value
    return_value = validate_interaction_style_overlay(overlay)
    return return_value


def _style_doc_contains_source_run(document: dict | None, run_id: str) -> bool:
    """Return whether a style image already includes a daily source run."""

    if document is None:
        return_value = False
        return return_value
    source_run_ids = document.get("source_reflection_run_ids")
    if not isinstance(source_run_ids, list):
        return_value = False
        return return_value
    clean_run_id = run_id.strip()
    known_run_ids = {
        str(source_run_id).strip()
        for source_run_id in source_run_ids
    }
    return_value = clean_run_id in known_run_ids
    return return_value


def _overlay_is_empty(overlay: dict) -> bool:
    """Return whether a validated overlay has no guidelines or confidence."""

    sanitized_overlay = validate_interaction_style_overlay(overlay)
    has_guidelines = any(
        sanitized_overlay[field_name]
        for field_name in _GUIDELINE_FIELDS
    )
    is_empty = not has_guidelines and not sanitized_overlay["confidence"]
    return is_empty


def _source_run_ids(daily_doc: CharacterReflectionRunDoc) -> list[str]:
    """Return internal audit source ids for a style-image write."""

    source_ids = [str(daily_doc["run_id"])]
    source_ids.extend(str(run_id) for run_id in daily_doc["source_reflection_run_ids"])
    return source_ids
