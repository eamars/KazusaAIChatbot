"""Persistence helpers for abstract interaction style images."""

from __future__ import annotations

import logging
import re

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.schemas import (
    InteractionStyleImageDoc,
    InteractionStyleOverlayDoc,
    InteractionStyleScopeType,
    InteractionStyleStatus,
)
from kazusa_ai_chatbot.config import (
    INTERACTION_STYLE_STORAGE_GUIDELINES_PER_FIELD_LIMIT,
    L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT,
    RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso
from kazusa_ai_chatbot.utils import text_or_empty

logger = logging.getLogger(__name__)

INTERACTION_STYLE_IMAGE_COLLECTION = "interaction_style_images"
_GUIDELINE_FIELDS = (
    "speech_guidelines",
    "social_guidelines",
    "pacing_guidelines",
    "engagement_guidelines",
)
_CONFIDENCE_DESCRIPTOR_MAX_CHARS = 80
_STYLE_LOAD_ERRORS = (PyMongoError, ValueError)
_STYLE_PROJECTION_ERRORS = (KeyError, TypeError, ValueError)
_EVENT_MARKER_PATTERNS = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Za-z0-9_-]{18,}\b"),
    re.compile(r"\d{6,}"),
    re.compile(r"run_id|reflection_run|source_reflection", re.IGNORECASE),
)
_QUOTE_EXAMPLE_CHARS = "\"“”‘’「」『』"


def empty_interaction_style_overlay() -> InteractionStyleOverlayDoc:
    """Return the empty runtime style overlay used by L3 consumers."""

    return_value: InteractionStyleOverlayDoc = {
        "speech_guidelines": [],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "",
    }
    return return_value


def _contains_event_marker(text: str) -> bool:
    """Return whether a guideline contains an obvious source detail marker."""

    for pattern in _EVENT_MARKER_PATTERNS:
        if pattern.search(text):
            return True
    quote_count = sum(text.count(char) for char in _QUOTE_EXAMPLE_CHARS)
    apostrophe_count = text.count("'")
    if quote_count >= 2 or apostrophe_count >= 2:
        return True
    return False


def _normalized_guidelines(
    overlay: dict,
    *,
    seen: set[str],
    field_name: str,
) -> list[str]:
    """Normalize one guideline field from an untrusted overlay payload.

    Args:
        overlay: Raw LLM or caller payload.
        seen: Case-insensitive strings already emitted across all fields.
        field_name: Name of the guideline field being normalized.

    Returns:
        Up to the configured number of unique, trimmed, source-detail-free
        guideline strings.
    """

    raw_items = overlay.get(field_name)
    if raw_items is None:
        return_value: list[str] = []
        return return_value
    if not isinstance(raw_items, list):
        raise ValueError(f"{field_name} must be a list of strings")

    normalized_items: list[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            continue
        text = " ".join(item.strip().split())
        if not text:
            continue
        if _contains_event_marker(text):
            raise ValueError(f"{field_name} contains source detail markers")
        text = text[:120].strip()
        dedup_key = text.casefold()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        normalized_items.append(text)
        if (
            len(normalized_items)
            >= INTERACTION_STYLE_STORAGE_GUIDELINES_PER_FIELD_LIMIT
        ):
            break
    return_value = normalized_items
    return return_value


def validate_interaction_style_overlay(overlay: dict) -> InteractionStyleOverlayDoc:
    """Normalize and validate an interaction-style overlay.

    Args:
        overlay: Candidate overlay from an LLM extraction or caller.

    Returns:
        A sanitized overlay containing only runtime L3 fields.

    Raises:
        ValueError: If the payload has invalid structure or obvious source
            details that should not be persisted.
    """

    if not isinstance(overlay, dict):
        raise ValueError("interaction style overlay must be a dict")

    seen: set[str] = set()
    normalized: InteractionStyleOverlayDoc = {}
    for field_name in _GUIDELINE_FIELDS:
        normalized[field_name] = _normalized_guidelines(
            overlay,
            seen=seen,
            field_name=field_name,
        )

    confidence = " ".join(text_or_empty(overlay.get("confidence")).split())
    confidence = confidence.lower()[:_CONFIDENCE_DESCRIPTOR_MAX_CHARS].strip()
    normalized["confidence"] = confidence

    has_guidelines = any(normalized[field_name] for field_name in _GUIDELINE_FIELDS)
    if not has_guidelines and confidence:
        raise ValueError("confidence cannot be set for an empty style overlay")
    return normalized


def _style_image_id_for_user(global_user_id: str) -> str:
    """Build the stable style-image id for one user scope."""

    clean_global_user_id = global_user_id.strip()
    if not clean_global_user_id:
        raise ValueError("global_user_id is required")
    style_image_id = f"user:{clean_global_user_id}"
    return style_image_id


def _style_image_id_for_group_channel(
    *,
    platform: str,
    platform_channel_id: str,
) -> str:
    """Build the stable style-image id for one group channel scope."""

    clean_platform = platform.strip()
    clean_platform_channel_id = platform_channel_id.strip()
    if not clean_platform or not clean_platform_channel_id:
        raise ValueError("platform and platform_channel_id are required")
    style_image_id = f"group_channel:{clean_platform}:{clean_platform_channel_id}"
    return style_image_id


def _overlay_has_guidelines(overlay: InteractionStyleOverlayDoc) -> bool:
    """Return whether a sanitized overlay contains any guideline text."""

    has_guidelines = any(overlay[field_name] for field_name in _GUIDELINE_FIELDS)
    return has_guidelines


def _source_run_ids(source_reflection_run_ids: list[str]) -> list[str]:
    """Normalize source reflection run ids for internal audit metadata."""

    normalized_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in source_reflection_run_ids:
        run_id = text_or_empty(raw_id).strip()
        if not run_id:
            continue
        if run_id in seen:
            continue
        seen.add(run_id)
        normalized_ids.append(run_id)
    return normalized_ids


async def ensure_interaction_style_image_indexes() -> None:
    """Create the collection and indexes required by style-image helpers."""

    db = await get_db()
    existing = set(await db.list_collection_names())
    if INTERACTION_STYLE_IMAGE_COLLECTION not in existing:
        await db.create_collection(INTERACTION_STYLE_IMAGE_COLLECTION)

    collection = db[INTERACTION_STYLE_IMAGE_COLLECTION]
    await collection.create_index(
        [("style_image_id", 1)],
        unique=True,
        name="interaction_style_image_id_unique",
    )
    await collection.create_index(
        [("scope_type", 1), ("global_user_id", 1)],
        name="interaction_style_user_scope",
    )
    await collection.create_index(
        [("scope_type", 1), ("platform", 1), ("platform_channel_id", 1)],
        name="interaction_style_group_channel_scope",
    )


async def get_user_style_image(global_user_id: str) -> InteractionStyleImageDoc | None:
    """Return the active style-image document for one user if it exists.

    Args:
        global_user_id: Internal user UUID.

    Returns:
        The stored style-image document without Mongo internals, or ``None``.
    """

    style_image_id = _style_image_id_for_user(global_user_id)
    db = await get_db()
    document = await db[INTERACTION_STYLE_IMAGE_COLLECTION].find_one(
        {
            "style_image_id": style_image_id,
            "scope_type": InteractionStyleScopeType.USER,
        },
        {"_id": 0},
    )
    if document is None:
        return_value = None
        return return_value
    return_value: InteractionStyleImageDoc = dict(document)
    return return_value


async def get_group_channel_style_image(
    *,
    platform: str,
    platform_channel_id: str,
) -> InteractionStyleImageDoc | None:
    """Return the active style-image document for one group channel.

    Args:
        platform: Platform namespace.
        platform_channel_id: Platform channel or group id.

    Returns:
        The stored style-image document without Mongo internals, or ``None``.
    """

    style_image_id = _style_image_id_for_group_channel(
        platform=platform,
        platform_channel_id=platform_channel_id,
    )
    db = await get_db()
    document = await db[INTERACTION_STYLE_IMAGE_COLLECTION].find_one(
        {
            "style_image_id": style_image_id,
            "scope_type": InteractionStyleScopeType.GROUP_CHANNEL,
        },
        {"_id": 0},
    )
    if document is None:
        return_value = None
        return return_value
    return_value: InteractionStyleImageDoc = dict(document)
    return return_value


async def _replace_style_image(document: InteractionStyleImageDoc) -> None:
    """Replace one style-image document by stable id."""

    db = await get_db()
    # Keep Mongo's immutable primary key aligned with the logical style key.
    await db[INTERACTION_STYLE_IMAGE_COLLECTION].replace_one(
        {"style_image_id": document["style_image_id"]},
        {**document, "_id": document["style_image_id"]},
        upsert=True,
    )


async def upsert_user_style_image(
    *,
    global_user_id: str,
    overlay: dict,
    source_reflection_run_ids: list[str],
    storage_timestamp_utc: str | None = None,
) -> InteractionStyleImageDoc:
    """Create or replace the current style image for one user.

    Args:
        global_user_id: Internal user UUID.
        overlay: Sanitized or candidate overlay payload.
        source_reflection_run_ids: Internal audit source run ids.
        storage_timestamp_utc: Optional storage UTC write timestamp.

    Returns:
        The persisted document without Mongo internals.
    """

    style_image_id = _style_image_id_for_user(global_user_id)
    existing = await get_user_style_image(global_user_id)
    sanitized_overlay = validate_interaction_style_overlay(overlay)
    write_time = storage_timestamp_utc or storage_utc_now_iso()
    status = (
        InteractionStyleStatus.ACTIVE
        if _overlay_has_guidelines(sanitized_overlay)
        else InteractionStyleStatus.EMPTY
    )
    created_at = existing["created_at"] if existing is not None else write_time
    revision = int(existing["revision"]) + 1 if existing is not None else 1
    document: InteractionStyleImageDoc = {
        "style_image_id": style_image_id,
        "scope_type": InteractionStyleScopeType.USER,
        "global_user_id": global_user_id.strip(),
        "platform": "",
        "platform_channel_id": "",
        "status": status,
        "overlay": sanitized_overlay,
        "source_reflection_run_ids": _source_run_ids(source_reflection_run_ids),
        "revision": revision,
        "created_at": created_at,
        "updated_at": write_time,
    }
    await _replace_style_image(document)
    return document


async def upsert_group_channel_style_image(
    *,
    platform: str,
    platform_channel_id: str,
    overlay: dict,
    source_reflection_run_ids: list[str],
    storage_timestamp_utc: str | None = None,
) -> InteractionStyleImageDoc:
    """Create or replace the current style image for one group channel.

    Args:
        platform: Platform namespace.
        platform_channel_id: Platform channel or group id.
        overlay: Sanitized or candidate overlay payload.
        source_reflection_run_ids: Internal audit source run ids.
        storage_timestamp_utc: Optional storage UTC write timestamp.

    Returns:
        The persisted document without Mongo internals.
    """

    clean_platform = platform.strip()
    clean_platform_channel_id = platform_channel_id.strip()
    style_image_id = _style_image_id_for_group_channel(
        platform=clean_platform,
        platform_channel_id=clean_platform_channel_id,
    )
    existing = await get_group_channel_style_image(
        platform=clean_platform,
        platform_channel_id=clean_platform_channel_id,
    )
    sanitized_overlay = validate_interaction_style_overlay(overlay)
    write_time = storage_timestamp_utc or storage_utc_now_iso()
    status = (
        InteractionStyleStatus.ACTIVE
        if _overlay_has_guidelines(sanitized_overlay)
        else InteractionStyleStatus.EMPTY
    )
    created_at = existing["created_at"] if existing is not None else write_time
    revision = int(existing["revision"]) + 1 if existing is not None else 1
    document: InteractionStyleImageDoc = {
        "style_image_id": style_image_id,
        "scope_type": InteractionStyleScopeType.GROUP_CHANNEL,
        "global_user_id": "",
        "platform": clean_platform,
        "platform_channel_id": clean_platform_channel_id,
        "status": status,
        "overlay": sanitized_overlay,
        "source_reflection_run_ids": _source_run_ids(source_reflection_run_ids),
        "revision": revision,
        "created_at": created_at,
        "updated_at": write_time,
    }
    await _replace_style_image(document)
    return document


def _runtime_overlay(
    document: InteractionStyleImageDoc | None,
) -> InteractionStyleOverlayDoc:
    """Project a stored style image into its sanitized runtime overlay."""

    if document is None:
        return_value = empty_interaction_style_overlay()
        return return_value
    if document["status"] != InteractionStyleStatus.ACTIVE:
        return_value = empty_interaction_style_overlay()
        return return_value
    overlay = document["overlay"]
    return_value = validate_interaction_style_overlay(dict(overlay))
    return return_value


def _project_l3_overlay(
    overlay: InteractionStyleOverlayDoc,
) -> InteractionStyleOverlayDoc:
    """Cap a sanitized style overlay for L3 prompt consumption."""

    return_value: InteractionStyleOverlayDoc = {
        "speech_guidelines": list(
            overlay["speech_guidelines"][
                :L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT
            ]
        ),
        "social_guidelines": list(
            overlay["social_guidelines"][
                :L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT
            ]
        ),
        "pacing_guidelines": list(
            overlay["pacing_guidelines"][
                :L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT
            ]
        ),
        "engagement_guidelines": list(
            overlay["engagement_guidelines"][
                :L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT
            ]
        ),
        "confidence": overlay["confidence"],
    }
    return return_value


async def build_interaction_style_context(
    *,
    global_user_id: str,
    channel_type: str,
    platform: str,
    platform_channel_id: str,
) -> dict:
    """Build the L3-facing style context for the active chat turn.

    Args:
        global_user_id: Current speaker's internal user UUID.
        channel_type: Current channel type.
        platform: Platform namespace.
        platform_channel_id: Platform channel or group id.

    Returns:
        Prompt-facing style context. Private chats omit group-channel style.
    """

    clean_global_user_id = text_or_empty(global_user_id).strip()
    if channel_type != "group" and not clean_global_user_id:
        raise ValueError("global_user_id is required")

    user_style = empty_interaction_style_overlay()
    if clean_global_user_id:
        try:
            user_document = await get_user_style_image(clean_global_user_id)
        except _STYLE_LOAD_ERRORS as exc:
            logger.exception(f"User interaction style load failed: {exc}")
        else:
            try:
                user_style = _project_l3_overlay(
                    _runtime_overlay(user_document)
                )
            except _STYLE_PROJECTION_ERRORS as exc:
                logger.exception(
                    f"User interaction style projection failed: {exc}"
                )

    context: dict = {
        "user_style": user_style,
        "application_order": ["user_style"],
    }
    if channel_type == "group":
        group_style = empty_interaction_style_overlay()
        try:
            group_document = await get_group_channel_style_image(
                platform=platform,
                platform_channel_id=platform_channel_id,
            )
        except _STYLE_LOAD_ERRORS as exc:
            logger.exception(f"Group interaction style load failed: {exc}")
        else:
            try:
                group_style = _project_l3_overlay(
                    _runtime_overlay(group_document)
                )
            except _STYLE_PROJECTION_ERRORS as exc:
                logger.exception(
                    f"Group interaction style projection failed: {exc}"
                )
        context["group_channel_style"] = group_style
        context["application_order"] = ["user_style", "group_channel_style"]
    return context


async def build_group_engagement_action_context(
    *,
    channel_type: str,
    platform: str,
    platform_channel_id: str,
) -> dict:
    """Project group-channel engagement guidance for action selection.

    Args:
        channel_type: Current channel type.
        platform: Platform namespace.
        platform_channel_id: Platform channel or group id.

    Returns:
        Compact group engagement guidance for L2d.
    """

    empty_context = {"engagement_guidelines": [], "confidence": ""}
    if channel_type != "group":
        return_value = empty_context
        return return_value

    try:
        group_document = await get_group_channel_style_image(
            platform=platform,
            platform_channel_id=platform_channel_id,
        )
    except _STYLE_LOAD_ERRORS as exc:
        logger.exception(f"Group engagement style load failed: {exc}")
        return_value = empty_context
        return return_value

    try:
        group_style = _runtime_overlay(group_document)
    except _STYLE_PROJECTION_ERRORS as exc:
        logger.exception(f"Group engagement style projection failed: {exc}")
        return_value = empty_context
        return return_value

    engagement_guidelines = list(
        group_style["engagement_guidelines"][
            :L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT
        ]
    )
    if not engagement_guidelines:
        return_value = empty_context
        return return_value

    return_value = {
        "engagement_guidelines": engagement_guidelines,
        "confidence": group_style["confidence"],
    }
    return return_value


async def build_user_engagement_relevance_context(global_user_id: str) -> dict:
    """Project one user's style image into relevance-facing engagement context.

    Args:
        global_user_id: Internal user UUID for the current speaker.

    Returns:
        Compact prompt-facing engagement guidance for persona relevance.
    """

    empty_context = {"engagement_guidelines": [], "confidence": ""}
    user_document = await get_user_style_image(global_user_id)
    try:
        user_style = _runtime_overlay(user_document)
    except (KeyError, TypeError, ValueError):
        return_value = empty_context
        return return_value

    engagement_guidelines = list(
        user_style["engagement_guidelines"][
            :RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT
        ]
    )
    if not engagement_guidelines:
        return_value = empty_context
        return return_value

    return_value = {
        "engagement_guidelines": engagement_guidelines,
        "confidence": user_style["confidence"],
    }
    return return_value
