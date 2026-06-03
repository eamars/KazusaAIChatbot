"""Build prompt-safe user-style source packets from reflection inputs."""

from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

from kazusa_ai_chatbot.db import CharacterReflectionRunDoc


class UserStyleEvidenceRow(TypedDict):
    """Prompt-facing style evidence row for one attributed user source."""

    role: Literal["target_user", "character"]
    text: str


class UserStyleSourcePacket(TypedDict):
    """Deterministic single-user source packet for style extraction."""

    source_id: str
    source_kind: Literal["private_daily", "group_participant_daily"]
    global_user_id: str
    channel_type: Literal["private", "group"]
    daily_confidence: Literal["medium", "high"]
    attribution_basis: str
    conversation_quality_patterns: list[str]
    synthesis_limitations: list[str]
    evidence_rows: list[UserStyleEvidenceRow]
    source_reflection_run_ids: list[str]


__all__ = [
    "UserStyleEvidenceRow",
    "UserStyleSourcePacket",
    "build_private_daily_user_style_source",
    "build_group_participant_user_style_sources",
    "user_style_source_to_extractor_payload",
]


_ELIGIBLE_DAILY_CONFIDENCE = {"medium", "high"}
_STYLE_SIGNAL_ITEM_LIMIT = 5
_STYLE_SIGNAL_ITEM_CHARS = 180
_GROUP_MIN_TARGET_USER_ROWS = 3
_GROUP_MIN_CHARACTER_ROWS = 3
_GROUP_MIN_TOTAL_ROWS = 8
_GROUP_SOURCE_LIMIT = 5
_EVIDENCE_ROW_LIMIT = 24
_EVIDENCE_ROW_CHARS = 160


def build_private_daily_user_style_source(
    *,
    daily_doc: CharacterReflectionRunDoc,
    global_user_id: str,
) -> UserStyleSourcePacket | None:
    """Build a private daily source packet for one resolved user.

    Args:
        daily_doc: Succeeded private daily reflection document.
        global_user_id: Internal user id resolved for the private scope.

    Returns:
        A single-user source packet, or ``None`` when the daily document is not
        eligible for user-style extraction.
    """

    clean_global_user_id = _compact_text(global_user_id)
    if not clean_global_user_id:
        return None

    scope = daily_doc["scope"]
    if str(scope["channel_type"]) != "private":
        return None

    confidence = _daily_confidence(daily_doc)
    if confidence not in _ELIGIBLE_DAILY_CONFIDENCE:
        return None

    source = UserStyleSourcePacket(
        source_id=_source_id(
            daily_doc=daily_doc,
            source_kind="private_daily",
            global_user_id=clean_global_user_id,
        ),
        source_kind="private_daily",
        global_user_id=clean_global_user_id,
        channel_type="private",
        daily_confidence=cast(Literal["medium", "high"], confidence),
        attribution_basis="private_scope_unique_user",
        conversation_quality_patterns=_daily_output_items(
            daily_doc=daily_doc,
            field_name="conversation_quality_patterns",
        ),
        synthesis_limitations=_daily_output_items(
            daily_doc=daily_doc,
            field_name="synthesis_limitations",
        ),
        evidence_rows=[],
        source_reflection_run_ids=_source_reflection_run_ids(daily_doc),
    )
    return source


def build_group_participant_user_style_sources(
    *,
    daily_doc: CharacterReflectionRunDoc,
    messages: list[dict[str, Any]],
    character_global_user_id: str,
) -> list[UserStyleSourcePacket]:
    """Build strongly attributed group participant style sources.

    Args:
        daily_doc: Succeeded group daily reflection document.
        messages: Bounded conversation rows from the reflection DB helper.
        character_global_user_id: Internal id for the active character.

    Returns:
        Up to five source packets, ranked by eligible evidence strength.
    """

    scope = daily_doc["scope"]
    if str(scope["channel_type"]) != "group":
        return []

    confidence = _daily_confidence(daily_doc)
    if confidence not in _ELIGIBLE_DAILY_CONFIDENCE:
        return []

    clean_character_id = _compact_text(character_global_user_id)
    if not clean_character_id:
        return []

    character_platform_ids = _character_platform_ids(
        messages=messages,
        character_global_user_id=clean_character_id,
    )
    target_platform_ids = _target_platform_ids(
        messages=messages,
        character_global_user_id=clean_character_id,
    )
    if not target_platform_ids:
        return []

    target_entries: dict[str, list[dict[str, Any]]] = {
        global_user_id: []
        for global_user_id in target_platform_ids
    }
    character_entries: dict[str, list[dict[str, Any]]] = {
        global_user_id: []
        for global_user_id in target_platform_ids
    }
    seen_keys: dict[str, set[tuple[str, str, str, str, str]]] = {
        global_user_id: set()
        for global_user_id in target_platform_ids
    }

    for index, message in enumerate(messages):
        role = _compact_text(message.get("role"))
        if role == "user":
            _collect_target_user_entry(
                target_entries=target_entries,
                seen_keys=seen_keys,
                message=message,
                index=index,
                character_global_user_id=clean_character_id,
                character_platform_ids=character_platform_ids,
            )
        elif role == "assistant":
            _collect_character_entry(
                character_entries=character_entries,
                seen_keys=seen_keys,
                message=message,
                index=index,
                character_global_user_id=clean_character_id,
                target_platform_ids=target_platform_ids,
            )

    source_rows: list[tuple[int, int, int, str, UserStyleSourcePacket]] = []
    for global_user_id in target_platform_ids:
        user_entries = target_entries[global_user_id]
        assistant_entries = character_entries[global_user_id]
        target_count = len(user_entries)
        character_count = len(assistant_entries)
        total_count = target_count + character_count
        if target_count < _GROUP_MIN_TARGET_USER_ROWS:
            continue
        if character_count < _GROUP_MIN_CHARACTER_ROWS:
            continue
        if total_count < _GROUP_MIN_TOTAL_ROWS:
            continue

        evidence_rows = _select_evidence_rows(
            target_entries=user_entries,
            character_entries=assistant_entries,
        )
        source = UserStyleSourcePacket(
            source_id=_source_id(
                daily_doc=daily_doc,
                source_kind="group_participant_daily",
                global_user_id=global_user_id,
            ),
            source_kind="group_participant_daily",
            global_user_id=global_user_id,
            channel_type="group",
            daily_confidence=cast(Literal["medium", "high"], confidence),
            attribution_basis="group_structural_single_target",
            conversation_quality_patterns=_daily_output_items(
                daily_doc=daily_doc,
                field_name="conversation_quality_patterns",
            ),
            synthesis_limitations=_daily_output_items(
                daily_doc=daily_doc,
                field_name="synthesis_limitations",
            ),
            evidence_rows=evidence_rows,
            source_reflection_run_ids=_source_reflection_run_ids(daily_doc),
        )
        source_rows.append((
            total_count,
            character_count,
            target_count,
            global_user_id,
            source,
        ))

    source_rows.sort(
        key=lambda row: (-row[0], -row[1], -row[2], row[3])
    )
    sources = [
        row[4]
        for row in source_rows[:_GROUP_SOURCE_LIMIT]
    ]
    return sources


def user_style_source_to_extractor_payload(
    *,
    source: UserStyleSourcePacket,
    current_overlay: dict,
) -> dict:
    """Project a source packet into the allowlisted extractor payload.

    Args:
        source: Deterministic single-user source packet.
        current_overlay: Current sanitized interaction-style overlay.

    Returns:
        Prompt-facing payload containing only semantic style signals.
    """

    evidence_rows = _payload_evidence_rows(source["evidence_rows"])
    payload = {
        "channel_type": source["channel_type"],
        "daily_confidence": source["daily_confidence"],
        "conversation_quality_patterns": list(
            source["conversation_quality_patterns"]
        ),
        "synthesis_limitations": list(source["synthesis_limitations"]),
        "evidence_rows": evidence_rows,
        "current_overlay": current_overlay,
    }
    return payload


def _daily_confidence(daily_doc: CharacterReflectionRunDoc) -> str:
    """Return normalized confidence from a daily reflection output."""

    output = daily_doc["output"]
    confidence = _compact_text(output.get("confidence")).lower()
    return confidence


def _daily_output_items(
    *,
    daily_doc: CharacterReflectionRunDoc,
    field_name: str,
) -> list[str]:
    """Return compact style signal strings from one daily output field."""

    output = daily_doc["output"]
    raw_value = output.get(field_name)
    compact_items = _compact_text_items(raw_value)
    return compact_items


def _compact_text_items(value: Any) -> list[str]:
    """Project arbitrary text-like values into bounded compact strings."""

    if isinstance(value, list):
        raw_items = value
    elif value:
        raw_items = [value]
    else:
        raw_items = []

    compact_items: list[str] = []
    for item in raw_items:
        text = _compact_text(item)
        if not text:
            continue
        compact_items.append(text[:_STYLE_SIGNAL_ITEM_CHARS].strip())
        if len(compact_items) >= _STYLE_SIGNAL_ITEM_LIMIT:
            break
    return compact_items


def _source_reflection_run_ids(
    daily_doc: CharacterReflectionRunDoc,
) -> list[str]:
    """Return daily and hourly source ids for internal write lineage."""

    source_run_ids = [str(daily_doc["run_id"])]
    for run_id in daily_doc["source_reflection_run_ids"]:
        source_run_ids.append(str(run_id))
    return source_run_ids


def _source_id(
    *,
    daily_doc: CharacterReflectionRunDoc,
    source_kind: str,
    global_user_id: str,
) -> str:
    """Build a stable internal source id for one user style packet."""

    source_id = f"{daily_doc['run_id']}:{source_kind}:{global_user_id}"
    return source_id


def _character_platform_ids(
    *,
    messages: list[dict[str, Any]],
    character_global_user_id: str,
) -> set[str]:
    """Collect platform account ids observed on active-character rows."""

    platform_ids: set[str] = set()
    for message in messages:
        role = _compact_text(message.get("role"))
        global_user_id = _compact_text(message.get("global_user_id"))
        if role != "assistant":
            continue
        if global_user_id != character_global_user_id:
            continue
        platform_user_id = _compact_text(message.get("platform_user_id"))
        if platform_user_id:
            platform_ids.add(platform_user_id)
    return platform_ids


def _target_platform_ids(
    *,
    messages: list[dict[str, Any]],
    character_global_user_id: str,
) -> dict[str, set[str]]:
    """Collect candidate group-user platform ids from user-authored rows."""

    platform_ids_by_user: dict[str, set[str]] = {}
    for message in messages:
        role = _compact_text(message.get("role"))
        if role != "user":
            continue
        global_user_id = _compact_text(message.get("global_user_id"))
        if not global_user_id:
            continue
        if global_user_id == character_global_user_id:
            continue
        platform_ids = platform_ids_by_user.setdefault(global_user_id, set())
        platform_user_id = _compact_text(message.get("platform_user_id"))
        if platform_user_id:
            platform_ids.add(platform_user_id)
    return platform_ids_by_user


def _collect_target_user_entry(
    *,
    target_entries: dict[str, list[dict[str, Any]]],
    seen_keys: dict[str, set[tuple[str, str, str, str, str]]],
    message: dict[str, Any],
    index: int,
    character_global_user_id: str,
    character_platform_ids: set[str],
) -> None:
    """Collect one user-authored row when it structurally targets character."""

    global_user_id = _compact_text(message.get("global_user_id"))
    if global_user_id not in target_entries:
        return
    is_targeted = _user_row_targets_character(
        message=message,
        character_global_user_id=character_global_user_id,
        character_platform_ids=character_platform_ids,
    )
    if not is_targeted:
        return
    _append_evidence_entry(
        entries=target_entries[global_user_id],
        seen_keys=seen_keys[global_user_id],
        message=message,
        index=index,
        evidence_role="target_user",
    )


def _collect_character_entry(
    *,
    character_entries: dict[str, list[dict[str, Any]]],
    seen_keys: dict[str, set[tuple[str, str, str, str, str]]],
    message: dict[str, Any],
    index: int,
    character_global_user_id: str,
    target_platform_ids: dict[str, set[str]],
) -> None:
    """Collect one assistant row when it maps to exactly one target user."""

    global_user_id = _compact_text(message.get("global_user_id"))
    if global_user_id != character_global_user_id:
        return

    target_global_user_id = _assistant_row_target_user(
        message=message,
        target_platform_ids=target_platform_ids,
    )
    if not target_global_user_id:
        return
    _append_evidence_entry(
        entries=character_entries[target_global_user_id],
        seen_keys=seen_keys[target_global_user_id],
        message=message,
        index=index,
        evidence_role="character",
    )


def _user_row_targets_character(
    *,
    message: dict[str, Any],
    character_global_user_id: str,
    character_platform_ids: set[str],
) -> bool:
    """Return whether a user row structurally targets the active character."""

    addressed_user_ids = _normalized_id_set(
        message.get("addressed_to_global_user_ids")
    )
    if character_global_user_id in addressed_user_ids:
        return True

    reply_context = _reply_context(message)
    reply_to_current_bot = reply_context.get("reply_to_current_bot")
    if reply_to_current_bot is True:
        return True

    reply_platform_user_id = _reply_to_platform_user_id(message)
    is_character_reply = reply_platform_user_id in character_platform_ids
    return is_character_reply


def _assistant_row_target_user(
    *,
    message: dict[str, Any],
    target_platform_ids: dict[str, set[str]],
) -> str:
    """Return the exact target user for one assistant row, if any."""

    addressed_user_ids = _normalized_id_set(
        message.get("addressed_to_global_user_ids")
    )
    if addressed_user_ids:
        if len(addressed_user_ids) != 1:
            return ""
        target_global_user_id = next(iter(addressed_user_ids))
        if target_global_user_id in target_platform_ids:
            return target_global_user_id
        return ""

    reply_platform_user_id = _reply_to_platform_user_id(message)
    if not reply_platform_user_id:
        return ""

    matched_user_ids = [
        global_user_id
        for global_user_id, platform_ids in target_platform_ids.items()
        if reply_platform_user_id in platform_ids
    ]
    if len(matched_user_ids) != 1:
        return ""
    target_global_user_id = matched_user_ids[0]
    return target_global_user_id


def _append_evidence_entry(
    *,
    entries: list[dict[str, Any]],
    seen_keys: set[tuple[str, str, str, str, str]],
    message: dict[str, Any],
    index: int,
    evidence_role: Literal["target_user", "character"],
) -> None:
    """Append a compact evidence entry unless the same row was already used."""

    body_text = _compact_evidence_text(message.get("body_text"))
    if not body_text:
        return

    timestamp = _compact_text(message.get("timestamp"))
    platform_user_id = _compact_text(message.get("platform_user_id"))
    global_user_id = _compact_text(message.get("global_user_id"))
    dedupe_key = (
        evidence_role,
        timestamp,
        platform_user_id,
        global_user_id,
        body_text,
    )
    if dedupe_key in seen_keys:
        return

    seen_keys.add(dedupe_key)
    entry = {
        "index": index,
        "role": evidence_role,
        "text": body_text,
    }
    entries.append(entry)


def _select_evidence_rows(
    *,
    target_entries: list[dict[str, Any]],
    character_entries: list[dict[str, Any]],
) -> list[UserStyleEvidenceRow]:
    """Select a bounded, recent, balanced evidence window for prompting."""

    per_role_limit = _EVIDENCE_ROW_LIMIT // 2
    selected_entries = (
        target_entries[-per_role_limit:]
        + character_entries[-per_role_limit:]
    )
    selected_keys = {
        (entry["index"], entry["role"], entry["text"])
        for entry in selected_entries
    }
    if len(selected_entries) < _EVIDENCE_ROW_LIMIT:
        remaining_entries = (
            target_entries[:-per_role_limit]
            + character_entries[:-per_role_limit]
        )
        remaining_entries.sort(
            key=lambda entry: int(entry["index"]),
            reverse=True,
        )
        for entry in remaining_entries:
            selected_key = (entry["index"], entry["role"], entry["text"])
            if selected_key in selected_keys:
                continue
            selected_entries.append(entry)
            selected_keys.add(selected_key)
            if len(selected_entries) >= _EVIDENCE_ROW_LIMIT:
                break

    selected_entries.sort(key=lambda entry: int(entry["index"]))
    evidence_rows = [
        UserStyleEvidenceRow(
            role=entry["role"],
            text=entry["text"],
        )
        for entry in selected_entries[:_EVIDENCE_ROW_LIMIT]
    ]
    return evidence_rows


def _payload_evidence_rows(
    evidence_rows: list[UserStyleEvidenceRow],
) -> list[UserStyleEvidenceRow]:
    """Return prompt evidence rows with only role and compact text fields."""

    payload_rows: list[UserStyleEvidenceRow] = []
    for row in evidence_rows[:_EVIDENCE_ROW_LIMIT]:
        role = row["role"]
        if role not in {"target_user", "character"}:
            continue
        text = _compact_evidence_text(row["text"])
        if not text:
            continue
        payload_rows.append(UserStyleEvidenceRow(role=role, text=text))
    return payload_rows


def _reply_context(message: dict[str, Any]) -> dict[str, Any]:
    """Return reply context metadata when the row carries a mapping."""

    raw_reply_context = message.get("reply_context")
    if not isinstance(raw_reply_context, dict):
        return {}
    reply_context = raw_reply_context
    return reply_context


def _reply_to_platform_user_id(message: dict[str, Any]) -> str:
    """Return the compact structural reply-target platform id."""

    reply_context = _reply_context(message)
    raw_platform_user_id = reply_context.get("reply_to_platform_user_id")
    platform_user_id = _compact_text(raw_platform_user_id)
    return platform_user_id


def _normalized_id_set(value: Any) -> set[str]:
    """Return non-empty normalized ids from a list-like field."""

    if not isinstance(value, list):
        return set()
    normalized_ids: set[str] = set()
    for item in value:
        normalized_id = _compact_text(item)
        if normalized_id:
            normalized_ids.add(normalized_id)
    return normalized_ids


def _compact_evidence_text(value: Any) -> str:
    """Return compact evidence text within the prompt-facing row cap."""

    text = _compact_text(value)
    compact_text = text[:_EVIDENCE_ROW_CHARS].strip()
    return compact_text


def _compact_text(value: Any) -> str:
    """Normalize arbitrary scalar text without inferring semantic meaning."""

    if value is None:
        return ""
    text = " ".join(str(value).strip().split())
    return text
