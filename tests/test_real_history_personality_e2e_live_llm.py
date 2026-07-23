"""Replay role-bound real conversation-history turns through the brain."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import pytest

from tests.stage3_fresh_database import validate_stage3_environment
from tests.test_stage3_fresh_database_e2e_live_llm import (
    _Stage3DebugAdapter,
    _json_safe,
    _wait_for_trace_run_finalization,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE_PATH = (
    _ROOT / "test_artifacts" / "chat_history_638473184_recent.json"
)
_DEFAULT_OUTPUT_ROOT = (
    _ROOT / "test_artifacts" / "personality_comparison"
    / "role_bound_20_case"
)
_DEFAULT_ROUTING_GUARD_ROOT = (
    _ROOT / "test_artifacts" / "personality_comparison"
    / "routing_guard"
)
_PROFILE_PATHS = (
    _ROOT / "personalities" / "asuna.json",
    _ROOT / "personalities" / "kazusa.json",
)
_CONTEXT_MESSAGE_LIMIT = 8
_COMPARISON_CASE_COUNT = 20
_ROUTING_GUARD_CASE_COUNT = 4
_MINIMUM_USER_TEXT_LENGTH = 10
_REPLAY_BOT_PLATFORM_ID = "real-history-bot"
_REPLAY_CURRENT_USER_PLATFORM_ID = "real-history-current-user"
_REPLAY_CURRENT_USER_GLOBAL_ID = "real-history-current-global"
_ASUNA_ALIAS = "明日奈"
_KAZUSA_ALIAS = "千纱"
_ASUNA_PROJECT_TOKEN = "AsunaAIChatbot"
_KAZUSA_PROJECT_TOKEN = "KazusaAIChatbot"


def _configure_utf8_streams() -> None:
    """Keep raw CJK input and model output printable on Windows."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_streams()


def _source_path() -> Path:
    """Return the database export selected for the replay suite."""

    configured_path = os.environ.get("REAL_HISTORY_SOURCE_PATH", "").strip()
    path = Path(configured_path) if configured_path else _DEFAULT_SOURCE_PATH
    return path.resolve()


def _profile_path() -> Path:
    """Return the active comparison personality file."""

    configured_path = os.environ.get("CHARACTER_PROFILE_PATH", "").strip()
    if not configured_path:
        raise AssertionError("CHARACTER_PROFILE_PATH is required")
    return Path(configured_path).resolve()


def _profile_label() -> str:
    """Return the stable profile label used for artifact isolation."""

    label = os.environ.get("REAL_HISTORY_PROFILE_LABEL", "").strip().casefold()
    if label not in {"asuna", "kazusa"}:
        raise AssertionError(
            "REAL_HISTORY_PROFILE_LABEL must be 'asuna' or 'kazusa'"
        )
    return label


def _output_root(*, routing_guard: bool = False) -> Path:
    """Return the profile-specific artifact directory."""

    configured_root = os.environ.get("REAL_HISTORY_OUTPUT_ROOT", "").strip()
    if configured_root:
        root = Path(configured_root)
    elif routing_guard:
        root = _DEFAULT_ROUTING_GUARD_ROOT
    else:
        root = _DEFAULT_OUTPUT_ROOT
    return (root / _profile_label()).resolve()


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object from a test fixture or profile file."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON object required: {path}")
    return value


def _load_source() -> dict[str, Any]:
    """Load and validate the persisted conversation-history export."""

    path = _source_path()
    if not path.is_file():
        raise AssertionError(f"conversation-history export is missing: {path}")
    source = _load_json_object(path)
    if source.get("query", {}).get("collection") != "conversation_history":
        raise AssertionError("source export is not from conversation_history")
    messages = source.get("messages")
    if not isinstance(messages, list) or len(messages) < 20:
        raise AssertionError("source export must contain at least 20 messages")
    if not all(isinstance(message, dict) for message in messages):
        raise AssertionError("source messages must be objects")
    return source


def _source_character_metadata(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Infer the historical character identity from typed bot mentions."""

    global_ids: Counter[str] = Counter()
    platform_ids: Counter[str] = Counter()
    display_names: Counter[str] = Counter()
    for message in messages:
        mentions = message.get("mentions")
        if not isinstance(mentions, list):
            continue
        for mention in mentions:
            if not isinstance(mention, Mapping):
                continue
            if mention.get("entity_kind") != "bot":
                continue
            global_user_id = mention.get("global_user_id")
            if isinstance(global_user_id, str) and global_user_id.strip():
                global_ids[global_user_id] += 1
            platform_user_id = mention.get("platform_user_id")
            if isinstance(platform_user_id, str) and platform_user_id.strip():
                platform_ids[platform_user_id] += 1
            display_name = mention.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                display_names[display_name] += 1
    if not global_ids:
        raise AssertionError("source export has no historical character id")
    source_global_id, _ = global_ids.most_common(1)[0]
    source_platform_ids = [
        value
        for value, _ in platform_ids.most_common()
        if value.strip()
    ]
    source_display_name = (
        display_names.most_common(1)[0][0]
        if display_names
        else "杏山千纱"
    )
    return {
        "global_user_id": source_global_id,
        "platform_user_ids": source_platform_ids,
        "display_name": source_display_name,
        "aliases": [_KAZUSA_ALIAS],
    }


def _source_character_global_id(messages: list[dict[str, Any]]) -> str:
    """Return the historical character global id."""

    return str(_source_character_metadata(messages)["global_user_id"])


def _message_body(message: Mapping[str, Any]) -> str:
    """Return the canonical text stored in one history row."""

    body = message.get("body_text")
    if not isinstance(body, str):
        raise AssertionError("conversation-history row has no body_text")
    return body.strip()


def _has_assistant_context(
    messages: list[dict[str, Any]],
    index: int,
) -> bool:
    """Return whether the selected user row has recent assistant context."""

    context = messages[max(0, index - _CONTEXT_MESSAGE_LIMIT):index]
    return any(item.get("role") == "assistant" for item in context)


def _is_direct_source_row(
    message: Mapping[str, Any],
    *,
    source_character: Mapping[str, Any],
) -> bool:
    """Return whether a row contains typed targeting of historical Kazusa."""

    source_global_id = str(source_character["global_user_id"])
    addressed_ids = message.get("addressed_to_global_user_ids", [])
    if isinstance(addressed_ids, list) and source_global_id in addressed_ids:
        return True
    mentions = message.get("mentions")
    if not isinstance(mentions, list):
        return False
    source_platform_ids = set(source_character["platform_user_ids"])
    for mention in mentions:
        if not isinstance(mention, Mapping):
            continue
        if mention.get("entity_kind") != "bot":
            continue
        if mention.get("global_user_id") == source_global_id:
            return True
        if mention.get("platform_user_id") in source_platform_ids:
            return True
    return False


def _eligible_direct_indexes(
    messages: list[dict[str, Any]],
    *,
    source_character: Mapping[str, Any],
) -> list[int]:
    """Find direct, text-bearing user rows with bounded bot context."""

    eligible: list[int] = []
    for index, message in enumerate(messages):
        body = _message_body(message)
        if message.get("role") != "user":
            continue
        if len(body) < _MINIMUM_USER_TEXT_LENGTH:
            continue
        if message.get("attachments"):
            continue
        if not _is_direct_source_row(
            message,
            source_character=source_character,
        ):
            continue
        if not _has_assistant_context(messages, index):
            continue
        eligible.append(index)
    return eligible


def _evenly_spaced_indexes(indexes: list[int], count: int) -> list[int]:
    """Select deterministic source rows across the eligible history range."""

    if len(indexes) < count:
        raise AssertionError(
            f"need {count} eligible source rows, found {len(indexes)}"
        )
    if count == 1:
        return [indexes[0]]
    selected = [
        indexes[round(position * (len(indexes) - 1) / (count - 1))]
        for position in range(count)
    ]
    if len(set(selected)) != count:
        raise AssertionError("source selection produced duplicate rows")
    return selected


def _build_cases() -> list[dict[str, Any]]:
    """Build exactly twenty direct cases from real Kazusa history."""

    source = _load_source()
    messages = source["messages"]
    source_character = _source_character_metadata(messages)
    direct_indexes = _evenly_spaced_indexes(
        _eligible_direct_indexes(
            messages,
            source_character=source_character,
        ),
        _COMPARISON_CASE_COUNT,
    )
    cases: list[dict[str, Any]] = []
    for position, source_index in enumerate(direct_indexes, 1):
        source_message = messages[source_index]
        context_start = max(0, source_index - _CONTEXT_MESSAGE_LIMIT)
        cases.append({
            "case_id": f"role_bound_comparison_{position:02d}",
            "category": "direct_kazusa_history",
            "source_index": source_index,
            "source_timestamp": source_message.get("timestamp", ""),
            "source_platform_message_id": source_message.get(
                "platform_message_id",
                "",
            ),
            "input_text": _message_body(source_message),
            "source_message": source_message,
            "source_context": messages[context_start:source_index],
            "source_query": source["query"],
            "source_character": source_character,
        })
    return cases


def _build_routing_guard_cases() -> list[dict[str, Any]]:
    """Build four source turns for the separate non-active-target guard."""

    source = _load_source()
    messages = source["messages"]
    source_character = _source_character_metadata(messages)
    direct_indexes = _evenly_spaced_indexes(
        _eligible_direct_indexes(
            messages,
            source_character=source_character,
        ),
        _ROUTING_GUARD_CASE_COUNT,
    )
    cases: list[dict[str, Any]] = []
    for position, source_index in enumerate(direct_indexes, 1):
        source_message = messages[source_index]
        context_start = max(0, source_index - _CONTEXT_MESSAGE_LIMIT)
        cases.append({
            "case_id": f"routing_guard_{position:02d}",
            "category": "exact_source_target_without_active_injection",
            "source_index": source_index,
            "source_timestamp": source_message.get("timestamp", ""),
            "source_platform_message_id": source_message.get(
                "platform_message_id",
                "",
            ),
            "input_text": _message_body(source_message),
            "source_message": source_message,
            "source_context": messages[context_start:source_index],
            "source_query": source["query"],
            "source_character": source_character,
        })
    return cases


_CASES = _build_cases()
_ROUTING_GUARD_CASES = _build_routing_guard_cases()


def _active_profile_name() -> str:
    """Return the configured personality's full display name."""

    profile = _load_json_object(_profile_path())
    name = profile.get("name")
    if not isinstance(name, str) or not name.strip():
        raise AssertionError("active personality has no name")
    return name.strip()


def _active_semantic_name(profile_name: str) -> str:
    """Return the Chinese semantic display name used in the envelope."""

    return profile_name.split("(", maxsplit=1)[0].strip()


def _active_alias(profile_label: str) -> str:
    """Return the Chinese short name used by the role-bound projection."""

    if profile_label == "asuna":
        return _ASUNA_ALIAS
    return _KAZUSA_ALIAS


def _source_identity_tokens(
    source_character: Mapping[str, Any],
) -> set[str]:
    """Return exact source identity tokens used by the leakage guard."""

    tokens = {
        str(source_character["global_user_id"]),
        str(source_character["display_name"]),
        _KAZUSA_ALIAS,
        _KAZUSA_PROJECT_TOKEN,
        "Kazusa",
        "kazusa",
    }
    tokens.update(
        str(value)
        for value in source_character.get("platform_user_ids", [])
        if str(value).strip()
    )
    return {token for token in tokens if token}


def _source_user_identity(
    message: Mapping[str, Any],
    *,
    source_current_global_id: str,
    current_platform_id: str,
    current_global_id: str,
) -> tuple[str, str]:
    """Create stable replay identities while retaining display names."""

    source_global_id = str(message.get("global_user_id") or "")
    source_platform_id = str(message.get("platform_user_id") or "")
    if (
        source_global_id == source_current_global_id
        or source_platform_id == str(
            message.get("_source_current_platform_id") or ""
        )
    ):
        return current_platform_id, current_global_id
    source_identity = (
        source_global_id
        or source_platform_id
        or str(message.get("display_name") or "unknown-source-user")
    )
    digest = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()[:12]
    return f"real-history-user-{digest}", f"real-history-global-{digest}"


def _replace_text(
    value: str,
    replacements: list[tuple[str, str]],
) -> str:
    """Apply deterministic longest-first replacement to one text value."""

    result = value
    for source, target in sorted(
        replacements,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if source and source != target:
            result = result.replace(source, target)
    return result


def _replace_nested(
    value: object,
    replacements: list[tuple[str, str]],
) -> object:
    """Replace source identity strings through any nested JSON-shaped value."""

    if isinstance(value, str):
        return _replace_text(value, replacements)
    if isinstance(value, list):
        return [
            _replace_nested(item, replacements)
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _replace_nested(item, replacements)
            for key, item in value.items()
        }
    return value


def _identity_replacements(
    *,
    profile_label: str,
    profile_name: str,
    character_global_user_id: str,
    source_character: Mapping[str, Any],
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    """Build the auditable source-to-runtime identity replacement table."""

    active_name = _active_semantic_name(profile_name)
    active_alias = _active_alias(profile_label)
    replacements: list[tuple[str, str]] = []
    mapping: list[dict[str, str]] = []

    def add(source: object, target: object, kind: str) -> None:
        source_text = str(source or "")
        target_text = str(target or "")
        if not source_text:
            return
        replacements.append((source_text, target_text))
        mapping.append({
            "kind": kind,
            "source": source_text,
            "target": target_text,
        })

    add(
        source_character["global_user_id"],
        character_global_user_id,
        "historical_character_global_id",
    )
    for source_platform_id in source_character["platform_user_ids"]:
        add(
            source_platform_id,
            _REPLAY_BOT_PLATFORM_ID,
            "historical_character_platform_id",
        )
    add(
        source_character["display_name"],
        active_name,
        "historical_character_display_name",
    )
    for source_alias in source_character["aliases"]:
        add(source_alias, active_alias, "historical_character_alias")
    if profile_label == "asuna":
        add(_KAZUSA_PROJECT_TOKEN, _ASUNA_PROJECT_TOKEN, "project_identity")
        add("Kazusa", "Asuna", "latin_character_identity")
        add("kazusa", "asuna", "latin_character_identity")
    return replacements, mapping


def _source_message_id_map(
    *,
    case_id: str,
    source_context: list[Mapping[str, Any]],
    source_message: Mapping[str, Any],
) -> dict[str, str]:
    """Map source message ids to isolated replay ids for Asuna prompts."""

    message_ids: dict[str, str] = {}
    for position, message in enumerate(source_context):
        source_id = str(message.get("platform_message_id") or "")
        if source_id:
            message_ids[source_id] = f"{case_id}-source-{position:02d}"
    source_id = str(source_message.get("platform_message_id") or "")
    if source_id:
        message_ids[source_id] = f"{case_id}-current"
    return message_ids


def _project_source_message(
    *,
    message: Mapping[str, Any],
    case_id: str,
    position: int | None,
    is_current: bool,
    profile_label: str,
    profile_name: str,
    character_global_user_id: str,
    source_character: Mapping[str, Any],
    source_current_global_id: str,
    current_global_id: str,
    replacements: list[tuple[str, str]],
    message_id_map: Mapping[str, str],
    preserve_source_raw_wire: bool,
) -> dict[str, Any]:
    """Project one source row into the isolated active-profile replay."""

    projected_value = _replace_nested(message, replacements)
    if not isinstance(projected_value, dict):
        raise AssertionError("projected source message is not an object")
    projected = projected_value
    role = str(message.get("role") or "")
    if role == "assistant":
        projected["platform_user_id"] = _REPLAY_BOT_PLATFORM_ID
        projected["global_user_id"] = character_global_user_id
        projected["display_name"] = _active_semantic_name(profile_name)
    elif is_current:
        projected["platform_user_id"] = _REPLAY_CURRENT_USER_PLATFORM_ID
        projected["global_user_id"] = current_global_id
        projected["display_name"] = str(
            projected.get("display_name") or message.get("display_name") or ""
        )
    else:
        source_message_with_marker = dict(message)
        source_message_with_marker["_source_current_platform_id"] = (
            str(source_current_global_id)
        )
        platform_user_id, global_user_id = _source_user_identity(
            source_message_with_marker,
            source_current_global_id=source_current_global_id,
            current_platform_id=_REPLAY_CURRENT_USER_PLATFORM_ID,
            current_global_id=current_global_id,
        )
        projected["platform_user_id"] = platform_user_id
        projected["global_user_id"] = global_user_id

    if position is None:
        projected["platform_message_id"] = f"{case_id}-current"
    else:
        projected["platform_message_id"] = f"{case_id}-source-{position:02d}"
    projected["platform"] = "debug"
    projected["platform_channel_id"] = f"real-history-{profile_label}-{case_id}"
    projected["channel_name"] = "real-history-replay"
    projected["timestamp"] = str(
        message.get("timestamp") or datetime.now(timezone.utc).isoformat()
    )

    original_reply = message.get("reply_context")
    if isinstance(original_reply, Mapping):
        reply = projected.get("reply_context")
        if isinstance(reply, dict):
            reply_id = str(reply.get("reply_to_message_id") or "")
            if reply_id in message_id_map:
                reply["reply_to_message_id"] = message_id_map[reply_id]
            elif reply_id and profile_label == "asuna":
                digest = hashlib.sha256(reply_id.encode("utf-8")).hexdigest()[:12]
                reply["reply_to_message_id"] = f"real-history-reply-{digest}"

    original_raw_wire = str(
        message.get("raw_wire_text") or _message_body(message)
    )
    if preserve_source_raw_wire:
        projected["raw_wire_text"] = original_raw_wire
    else:
        raw_wire_replacements = [
            *replacements,
            *[
                (source_id, target_id)
                for source_id, target_id in message_id_map.items()
            ],
        ]
        projected["raw_wire_text"] = _replace_text(
            original_raw_wire,
            raw_wire_replacements,
        )

    return projected


def _build_profile_case(
    *,
    case: Mapping[str, Any],
    profile_label: str,
    profile_name: str,
    character_global_user_id: str,
    current_global_id: str,
    routing_guard: bool = False,
) -> dict[str, Any]:
    """Build source and effective inputs for one profile mode."""

    source_message = case["source_message"]
    source_context = case["source_context"]
    source_character = case["source_character"]
    if not isinstance(source_message, Mapping):
        raise AssertionError("source message is invalid")
    if not isinstance(source_context, list):
        raise AssertionError("source context is invalid")
    if not isinstance(source_character, Mapping):
        raise AssertionError("source character metadata is invalid")
    source_current_global_id = str(
        source_message.get("global_user_id") or ""
    )
    case_id = str(case["case_id"])
    if routing_guard:
        effective_context = [
            dict(row)
            for row in source_context
            if isinstance(row, Mapping)
        ]
        effective_input = dict(source_message)
        identity_mapping: list[dict[str, str]] = []
        mode = "exact_source_target_without_active_injection"
        target_injected = False
    else:
        replacements, identity_mapping = _identity_replacements(
            profile_label=profile_label,
            profile_name=profile_name,
            character_global_user_id=character_global_user_id,
            source_character=source_character,
        )
        message_id_map = _source_message_id_map(
            case_id=case_id,
            source_context=source_context,
            source_message=source_message,
        )
        identity_mapping.extend({
            "kind": "replay_message_id",
            "source": source_id,
            "target": target_id,
        } for source_id, target_id in message_id_map.items())
        effective_context = []
        for position, row in enumerate(source_context):
            if not isinstance(row, Mapping):
                raise AssertionError("source context row is invalid")
            effective_context.append(
                _project_source_message(
                    message=row,
                    case_id=case_id,
                    position=position,
                    is_current=False,
                    profile_label=profile_label,
                    profile_name=profile_name,
                    character_global_user_id=character_global_user_id,
                    source_character=source_character,
                    source_current_global_id=source_current_global_id,
                    current_global_id=current_global_id,
                    replacements=replacements,
                    message_id_map=message_id_map,
                    preserve_source_raw_wire=profile_label == "kazusa",
                )
            )
        effective_input = _project_source_message(
            message=source_message,
            case_id=case_id,
            position=None,
            is_current=True,
            profile_label=profile_label,
            profile_name=profile_name,
            character_global_user_id=character_global_user_id,
            source_character=source_character,
            source_current_global_id=source_current_global_id,
            current_global_id=current_global_id,
            replacements=replacements,
            message_id_map=message_id_map,
            preserve_source_raw_wire=profile_label == "kazusa",
        )
        mode = "role_bound_identity_projection"
        target_injected = True
    return {
        "case_id": case_id,
        "category": case["category"],
        "mode": mode,
        "target_injected": target_injected,
        "source_input": dict(source_message),
        "source_context": [
            dict(row)
            for row in source_context
            if isinstance(row, Mapping)
        ],
        "effective_input": effective_input,
        "effective_context": effective_context,
        "identity_mapping": identity_mapping,
        "excluded_rows": [],
        "source_character": dict(source_character),
        "source_current_global_id": source_current_global_id,
        "active_character_global_user_id": character_global_user_id,
        "active_character_name": _active_semantic_name(profile_name),
        "current_user_global_id": current_global_id,
    }


def _current_request_reply(value: Mapping[str, Any]) -> dict[str, str] | None:
    """Project stored reply metadata into the typed request reply contract."""

    reply_context = value.get("reply_context")
    if not isinstance(reply_context, Mapping) or not reply_context:
        return None
    reply: dict[str, str] = {}
    field_map = {
        "reply_to_message_id": "platform_message_id",
        "reply_to_platform_user_id": "platform_user_id",
        "reply_to_display_name": "display_name",
        "reply_excerpt": "excerpt",
    }
    for source_key, target_key in field_map.items():
        item = reply_context.get(source_key)
        if isinstance(item, str) and item.strip():
            reply[target_key] = item
    if reply:
        reply["derivation"] = "platform_native"
    return reply or None


def _build_request(
    *,
    profile_case: Mapping[str, Any],
    character_global_user_id: str,
    channel_id: str,
    platform_message_id: str,
) -> Any:
    """Build a request from the already-projected input envelope."""

    from kazusa_ai_chatbot.brain_service.contracts import ChatRequest
    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    effective_input = profile_case["effective_input"]
    if not isinstance(effective_input, Mapping):
        raise AssertionError("effective input is invalid")
    raw_mentions = effective_input.get("mentions", [])
    mentions = [
        dict(mention)
        for mention in raw_mentions
        if isinstance(mention, Mapping)
    ] if isinstance(raw_mentions, list) else []
    body_text = str(effective_input.get("body_text") or "").strip()
    raw_wire_text = str(
        effective_input.get("raw_wire_text") or body_text
    )
    addressed_ids = [
        str(value)
        for value in effective_input.get(
            "addressed_to_global_user_ids",
            [],
        )
        if str(value).strip()
    ]
    envelope = {
        "body_text": body_text,
        "raw_wire_text": raw_wire_text,
        "mentions": mentions,
        "reply": _current_request_reply(effective_input),
        "attachments": (
            effective_input.get("attachments", [])
            if isinstance(effective_input.get("attachments", []), list)
            else []
        ),
        "addressed_to_global_user_ids": addressed_ids,
        "broadcast": bool(effective_input.get("broadcast", False)),
    }
    if profile_case["mode"] == "exact_source_target_without_active_injection":
        envelope["mentions"] = []
        envelope["reply"] = None
        envelope["addressed_to_global_user_ids"] = []
        envelope["broadcast"] = False
    request = ChatRequest.model_validate({
        "platform": "debug",
        "platform_channel_id": channel_id,
        "channel_type": "group",
        "platform_message_id": platform_message_id,
        "platform_user_id": _REPLAY_CURRENT_USER_PLATFORM_ID,
        "platform_bot_id": _REPLAY_BOT_PLATFORM_ID,
        "display_name": str(
            effective_input.get("display_name") or "real-history-user"
        ),
        "channel_name": "real-history-replay",
        "content_type": "text",
        "message_envelope": envelope,
        "local_timestamp": build_turn_clock()["local_timestamp"],
        "debug_modes": {},
    })
    return request


async def _seed_context(
    *,
    profile_case: Mapping[str, Any],
    channel_id: str,
) -> list[dict[str, Any]]:
    """Seed the selected source or projected history window."""

    from kazusa_ai_chatbot.db import (
        get_document_text_embeddings_batch,
        save_conversation,
    )

    context_rows = profile_case["effective_context"]
    if not isinstance(context_rows, list):
        raise AssertionError("effective context is invalid")
    seed_documents: list[dict[str, Any]] = []
    for position, message in enumerate(context_rows):
        if not isinstance(message, Mapping):
            raise AssertionError("effective context row is invalid")
        document = dict(message)
        document["platform"] = "debug"
        document["platform_channel_id"] = channel_id
        document["platform_message_id"] = (
            f"{profile_case['case_id']}-source-{position:02d}"
        )
        document["channel_name"] = "real-history-replay"
        document["channel_type"] = str(
            document.get("channel_type") or "group"
        )
        document["role"] = str(document.get("role") or "user")
        document["display_name"] = str(
            document.get("display_name") or "source-user"
        )
        document["body_text"] = _message_body(document)
        document["raw_wire_text"] = str(
            document.get("raw_wire_text") or document["body_text"]
        )
        document["addressed_to_global_user_ids"] = [
            str(value)
            for value in document.get("addressed_to_global_user_ids", [])
            if str(value).strip()
        ]
        document["mentions"] = (
            document.get("mentions", [])
            if isinstance(document.get("mentions", []), list)
            else []
        )
        document["broadcast"] = bool(document.get("broadcast", False))
        document["attachments"] = (
            document.get("attachments", [])
            if isinstance(document.get("attachments", []), list)
            else []
        )
        document["reply_context"] = (
            document.get("reply_context", {})
            if isinstance(document.get("reply_context", {}), Mapping)
            else {}
        )
        document["timestamp"] = str(
            document.get("timestamp")
            or datetime.now(timezone.utc).isoformat()
        )
        seed_documents.append(document)
    embedding_texts = [document["body_text"] for document in seed_documents]
    embeddings = await get_document_text_embeddings_batch(embedding_texts)
    if len(embeddings) != len(seed_documents):
        raise AssertionError("history embedding count does not match seed rows")
    seeded_rows: list[dict[str, Any]] = []
    for document, embedding in zip(seed_documents, embeddings, strict=True):
        document["embedding"] = embedding
        await save_conversation(document)
        seeded_rows.append(document)
    return seeded_rows


def _find_key(value: object, key: str) -> list[object]:
    """Find all values for one semantic field in a captured trace payload."""

    matches: list[object] = []
    if isinstance(value, Mapping):
        if key in value:
            matches.append(value[key])
        for nested in value.values():
            matches.extend(_find_key(nested, key))
    elif isinstance(value, list):
        for nested in value:
            matches.extend(_find_key(nested, key))
    return matches


def _extract_private_monologue(
    trace_steps: list[dict[str, Any]],
) -> tuple[str, str]:
    """Extract the cognition-owned private monologue from full trace rows."""

    candidates: list[tuple[str, str]] = []
    for step in trace_steps:
        parsed_output = step.get("parsed_output")
        for value in _find_key(parsed_output, "private_monologue"):
            if isinstance(value, str) and value.strip():
                candidates.append((str(step.get("stage_name", "")), value))
    if not candidates:
        return "", ""
    stage_name, monologue = candidates[-1]
    return monologue, stage_name


def _extract_decontextualized_input(
    trace_steps: list[dict[str, Any]],
) -> tuple[str, str]:
    """Extract the final message-decontextualizer output."""

    candidates: list[tuple[str, str]] = []
    for step in trace_steps:
        if step.get("stage_name") != "message_decontextualizer":
            continue
        parsed_output = step.get("parsed_output")
        if not isinstance(parsed_output, Mapping):
            continue
        for key in (
            "output",
            "decontexualized_input",
            "decontextualized_input",
        ):
            value = parsed_output.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append((key, value.strip()))
    if not candidates:
        return "", ""
    field_name, value = candidates[-1]
    return value, field_name


def _collect_source_leaks(
    value: object,
    *,
    source_tokens: set[str],
    path: str = "$",
) -> list[dict[str, str]]:
    """Find source identity tokens in model-visible JSON-shaped payloads."""

    leaks: list[dict[str, str]] = []
    if isinstance(value, str):
        folded = value.casefold()
        for token in sorted(source_tokens, key=len, reverse=True):
            if token.casefold() in folded:
                leaks.append({
                    "path": path,
                    "token": token,
                    "value": value[:400],
                })
        return leaks
    if isinstance(value, Mapping):
        for key, nested in value.items():
            leaks.extend(
                _collect_source_leaks(
                    nested,
                    source_tokens=source_tokens,
                    path=f"{path}.{key}",
                )
            )
        return leaks
    if isinstance(value, list):
        for index, nested in enumerate(value):
            leaks.extend(
                _collect_source_leaks(
                    nested,
                    source_tokens=source_tokens,
                    path=f"{path}[{index}]",
                )
            )
    return leaks


def _trace_model_visible_payload(
    trace_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only prompt/output fields used by the identity guard."""

    return [
        {
            "stage_name": step.get("stage_name", ""),
            "raw_messages": step.get("raw_messages", []),
            "parsed_output": step.get("parsed_output"),
            "raw_response_text": step.get("raw_response_text", ""),
        }
        for step in trace_steps
    ]


def _relevance_dispositions(
    trace_steps: list[dict[str, Any]],
) -> list[str]:
    """Return every observed frontline or settled disposition in order."""

    values: list[str] = []
    for step in trace_steps:
        if step.get("stage_name") not in {
            "persona_relevance_agent",
            "frontline_relevance_agent",
        }:
            continue
        parsed_output = step.get("parsed_output")
        if not isinstance(parsed_output, Mapping):
            continue
        value = parsed_output.get("semantic_disposition")
        if value is None:
            value = parsed_output.get("response_action")
        if value is None:
            value = parsed_output.get("intake_action")
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _response_surface_status(
    response_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the captured surface without rewriting its visible text."""

    operational_error = response_payload.get("operational_error")
    if isinstance(operational_error, Mapping):
        return {
            "status": "operational_error",
            "error_code": str(
                operational_error.get("error_code") or ""
            ),
            "trace_id": str(operational_error.get("trace_id") or ""),
        }
    messages = response_payload.get("messages")
    if isinstance(messages, list) and messages:
        return {
            "status": "visible_dialog",
            "error_code": "",
            "trace_id": "",
        }
    return {
        "status": "no_visible_dialog",
        "error_code": "",
        "trace_id": "",
    }


def _fixture_validity(
    *,
    profile_case: Mapping[str, Any],
    profile_label: str,
    request: Any,
) -> dict[str, Any]:
    """Validate the profile projection before a real LLM call."""

    failures: list[str] = []
    source_input = profile_case["source_input"]
    effective_input = profile_case["effective_input"]
    if not isinstance(source_input, Mapping):
        failures.append("source_input is not an object")
    if not isinstance(effective_input, Mapping):
        failures.append("effective_input is not an object")
    if isinstance(source_input, Mapping) and isinstance(
        effective_input,
        Mapping,
    ):
        source_body = _message_body(source_input)
        effective_body = _message_body(effective_input)
        if profile_label == "kazusa" and effective_body != source_body:
            failures.append("Kazusa body_text changed")
        if not effective_body:
            failures.append("effective body_text is empty")
    envelope = request.message_envelope.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )
    addressed_ids = envelope.get("addressed_to_global_user_ids", [])
    if profile_case["mode"] == "role_bound_identity_projection":
        if profile_case["target_injected"] is not True:
            failures.append("role-bound target marker is missing")
        if profile_case["active_character_global_user_id"] not in addressed_ids:
            failures.append("active character is not addressed")
    else:
        if profile_case["target_injected"] is not False:
            failures.append("routing guard injected an active target")
        if addressed_ids:
            failures.append("routing guard retained addressed ids")
        if envelope.get("mentions"):
            failures.append("routing guard retained typed mentions")
    return {
        "passed": not failures,
        "checks": [
            "source row is preserved as a separate evidence object",
            "effective request is constructed from the profile projection",
            "typed target fields are validated before the LLM call",
        ],
        "failures": failures,
    }


def _live_semantic_validity(
    *,
    profile_case: Mapping[str, Any],
    profile_label: str,
    trace_steps: list[dict[str, Any]],
    request: Any,
    decontextualized_input: str,
    private_monologue: str,
    visible_dialog: object,
    response_payload: Mapping[str, Any],
    response_surface: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate model-visible identity and required behavioral evidence."""

    failures: list[str] = []
    source_tokens = _source_identity_tokens(
        profile_case["source_character"]
    )
    if profile_case["mode"] == "role_bound_identity_projection":
        model_visible = {
            "effective_input": profile_case["effective_input"],
            "effective_context": profile_case["effective_context"],
            "effective_envelope": request.message_envelope.model_dump(
                exclude_none=True,
                exclude_defaults=True,
            ),
            "decontextualized_input": decontextualized_input,
            "private_monologue": private_monologue,
            "visible_dialog": visible_dialog,
            "trace_model_visible": _trace_model_visible_payload(
                trace_steps
            ),
        }
        if profile_label == "asuna":
            leaks = _collect_source_leaks(
                model_visible,
                source_tokens=source_tokens,
            )
            if leaks:
                failures.append(
                    "source identity leaked into model-visible evidence"
                )
        else:
            # The Kazusa side is the source-language baseline. Its source
            # identity is expected and is not a contamination signal.
            leaks = []
        if not decontextualized_input:
            failures.append("decontextualized_input is missing")
        if not private_monologue:
            failures.append("private_monologue is missing")
        if "messages" not in response_payload:
            failures.append("response messages field is missing")
        if not isinstance(visible_dialog, list):
            failures.append("response messages field is not a list")
        if response_surface.get("status") not in {
            "visible_dialog",
            "no_visible_dialog",
            "operational_error",
        }:
            failures.append("response surface status is invalid")
    else:
        leaks = []
        dispositions = _relevance_dispositions(trace_steps)
        if not dispositions:
            failures.append("routing guard has no relevance disposition")
        if any(
            disposition.casefold() in {"proceed", "respond"}
            for disposition in dispositions
        ):
            failures.append("routing guard proceeded for a non-active target")
        if isinstance(visible_dialog, list) and visible_dialog:
            failures.append("routing guard produced visible dialog")
    return {
        "passed": not failures,
        "checks": [
            "trace evidence was inspected after the live turn",
            "model-visible identity was checked against the source identity set",
            "required monologue and response surface or routing disposition was validated",
        ],
        "failures": failures,
        "source_leaks": leaks,
        "relevance_dispositions": _relevance_dispositions(trace_steps),
    }


def _write_artifact(
    payload: Mapping[str, Any],
    *,
    routing_guard: bool,
) -> Path:
    """Write one complete local raw-evidence case artifact."""

    output_root = _output_root(routing_guard=routing_guard)
    output_root.mkdir(parents=True, exist_ok=True)
    case_id = str(payload["case_id"])
    output_path = output_root / f"{case_id}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


async def _run_case(
    case: Mapping[str, Any],
    *,
    routing_guard: bool = False,
) -> dict[str, Any]:
    """Run one source turn through intake, cognition, dialog, and storage."""

    runtime_guard = validate_stage3_environment(os.environ)
    profile_label = _profile_label()
    if routing_guard and profile_label != "asuna":
        raise AssertionError("routing guard must run with Asuna active")
    character_name = _active_profile_name()
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.db import db_bootstrap, resolve_global_user_id
    from kazusa_ai_chatbot.db._client import close_db, get_db

    case_id = str(case["case_id"])
    channel_id = (
        f"real-history-routing-{case_id}"
        if routing_guard
        else f"real-history-{profile_label}-{case_id}"
    )
    platform_message_id = f"{channel_id}-current"
    started_at = time.perf_counter()
    artifact: dict[str, Any] = {
        "schema_version": "real_history_personality_e2e_case.v2",
        "case_id": case_id,
        "profile_label": profile_label,
        "character_name": character_name,
        "character_semantic_name": _active_semantic_name(character_name),
        "source_kind": "real_conversation_history_export",
        "source_path": str(_source_path()),
        "source_query": case["source_query"],
        "source_index": case["source_index"],
        "source_timestamp": case["source_timestamp"],
        "source_platform_message_id": case["source_platform_message_id"],
        "category": case["category"],
        "technical_status": "running",
    }
    try:
        await db_bootstrap()
        source_message = case["source_message"]
        source_display_name = str(
            source_message.get("display_name") or "real-history-user"
        )
        current_global_id = await resolve_global_user_id(
            "debug",
            _REPLAY_CURRENT_USER_PLATFORM_ID,
            source_display_name,
        )
        profile_case = _build_profile_case(
            case=case,
            profile_label=profile_label,
            profile_name=character_name,
            character_global_user_id=CHARACTER_GLOBAL_USER_ID,
            current_global_id=current_global_id,
            routing_guard=routing_guard,
        )
        artifact.update({
            "mode": profile_case["mode"],
            "target_injected": profile_case["target_injected"],
            "source_input": _json_safe(profile_case["source_input"]),
            "effective_input": _json_safe(profile_case["effective_input"]),
            "identity_mapping": _json_safe(
                profile_case["identity_mapping"]
            ),
            "excluded_rows": _json_safe(profile_case["excluded_rows"]),
            "source_context": _json_safe(profile_case["source_context"]),
            "effective_context": _json_safe(
                profile_case["effective_context"]
            ),
            "user_input": profile_case["effective_input"]["body_text"],
        })
        request = _build_request(
            profile_case=profile_case,
            character_global_user_id=CHARACTER_GLOBAL_USER_ID,
            channel_id=channel_id,
            platform_message_id=platform_message_id,
        )
        artifact["effective_envelope"] = _json_safe(
            request.message_envelope.model_dump(
                exclude_none=True,
                exclude_defaults=True,
            )
        )
        fixture_validity = _fixture_validity(
            profile_case=profile_case,
            profile_label=profile_label,
            request=request,
        )
        artifact["fixture_validity"] = fixture_validity
        if not fixture_validity["passed"]:
            raise AssertionError(
                "fixture validation failed: "
                + "; ".join(fixture_validity["failures"])
            )
        seeded_rows = await _seed_context(
            profile_case=profile_case,
            channel_id=channel_id,
        )
        adapter = _Stage3DebugAdapter()
        async with service.lifespan(service.app):
            if service._adapter_registry is None:
                raise AssertionError("service adapter registry was not started")
            service._adapter_registry.register(adapter)
            response = await service._enqueue_chat_request(request)
            db = await get_db()
            trace_run = await _wait_for_trace_run_finalization(
                db,
                platform_message_id=platform_message_id,
            )
            user_row = await db.conversation_history.find_one(
                {"platform_message_id": platform_message_id},
                {"_id": 0},
            )
            if not isinstance(user_row, Mapping):
                raise AssertionError("current user row was not persisted")
            trace_id = str(user_row.get("llm_trace_id", ""))
            if not trace_id:
                raise AssertionError("persisted user row has no LLM trace id")
            trace_steps = await db.llm_trace_steps.find(
                {"trace_id": trace_id},
                {"_id": 0},
            ).sort("created_at", 1).to_list(length=None)
            trace_steps = [
                dict(step)
                for step in trace_steps
                if isinstance(step, Mapping)
            ]
            private_monologue, monologue_stage = _extract_private_monologue(
                trace_steps
            )
            decontextualized_input, decontextualized_field = (
                _extract_decontextualized_input(trace_steps)
            )
            response_payload = response.model_dump(mode="json")
            visible_dialog = response_payload.get("messages", [])
            response_surface = _response_surface_status(response_payload)
            semantic_validity = _live_semantic_validity(
                profile_case=profile_case,
                profile_label=profile_label,
                trace_steps=trace_steps,
                request=request,
                decontextualized_input=decontextualized_input,
                private_monologue=private_monologue,
                visible_dialog=visible_dialog,
                response_payload=response_payload,
                response_surface=response_surface,
            )
            assistant_rows = await db.conversation_history.find(
                {"llm_trace_id": trace_id, "role": "assistant"},
                {"_id": 0},
            ).sort("timestamp", 1).to_list(length=None)
            artifact.update({
                "trace_run": _json_safe(trace_run),
                "trace_id": trace_id,
                "trace_steps": _json_safe(trace_steps),
                "decontextualized_input": decontextualized_input,
                "decontextualized_input_field": decontextualized_field,
                "private_monologue": private_monologue,
                "private_monologue_stage": monologue_stage,
                "visible_dialog": visible_dialog,
                "response": _json_safe(response_payload),
                "response_surface": response_surface,
                "adapter_calls": _json_safe(adapter.calls),
                "seeded_context": _json_safe(seeded_rows),
                "persisted_user_row": _json_safe(user_row),
                "persisted_assistant_rows": _json_safe(assistant_rows),
                "llm_trace_step_count": len(trace_steps),
                "semantic_validity": semantic_validity,
                "duration_ms": round(
                    (time.perf_counter() - started_at) * 1000
                ),
                "database_name": runtime_guard["database_name"],
            })
            if not semantic_validity["passed"]:
                raise AssertionError(
                    "live semantic validity failed: "
                    + "; ".join(semantic_validity["failures"])
                )
            artifact["technical_status"] = "passed"
    except Exception as exc:
        artifact.update({
            "technical_status": "failed",
            "failure_type": exc.__class__.__name__,
            "failure_message": str(exc),
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
        })
        _write_artifact(artifact, routing_guard=routing_guard)
        raise
    finally:
        await close_db()
    output_path = _write_artifact(artifact, routing_guard=routing_guard)
    dialog_text = "\n".join(
        str(item).strip()
        for item in artifact["visible_dialog"]
        if str(item).strip()
    )
    print(f"案例完成: {case_id}")
    print(f"用户输入: {artifact['user_input']}")
    print(f"私人独白: {artifact.get('private_monologue', '')}")
    print(f"可见对话: {dialog_text or '（无可见输出）'}")
    print(f"证据文件: {output_path}")
    return artifact


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[str(case["case_id"]) for case in _CASES],
)
async def test_live_real_history_personality_case(
    case: dict[str, Any],
) -> None:
    """Exercise one role-bound comparison case as an isolated live turn."""

    await _run_case(case)


@pytest.mark.parametrize(
    "case",
    _ROUTING_GUARD_CASES,
    ids=[str(case["case_id"]) for case in _ROUTING_GUARD_CASES],
)
async def test_live_real_history_non_active_target_routing_guard(
    case: dict[str, Any],
) -> None:
    """Verify source-target text cannot become an Asuna target implicitly."""

    if os.environ.get("REAL_HISTORY_RUN_ROUTING_GUARD", "").strip() != "1":
        pytest.skip("set REAL_HISTORY_RUN_ROUTING_GUARD=1 to run this live guard")
    await _run_case(case, routing_guard=True)
