"""Control one-case-at-a-time baseline/V2 differential executions."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from dotenv import dotenv_values


_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_ROOT = _ROOT / "tests" / "fixtures"
_EVIDENCE_ROOT = (
    _ROOT / "test_artifacts" / "cognition_core_v2"
    / "baseline_regression_hardening"
)
_PROFILE_PATH = (_ROOT / "personalities" / "asuna.json").resolve()
_HISTORY_PATH = (
    _ROOT / "test_artifacts" / "chat_history_638473184_recent.json"
).resolve()
_BASELINE_ROOT = Path("C:/workspace/kazusa_ai_chatbot_baseline_main")
_V2_ROOT = Path("C:/workspace/kazusa_ai_chatbot_v2_prefix")
_CURRENT_ROOT = _ROOT.resolve()
_BASELINE_REVISION = "8f834bf87a83ee42aca804934fb44af63788420c"
_CANDIDATE_REVISION = "0c2e929d51ac80c4519f564b61cbf8949efcca3d"
_PROFILE_SHA256 = (
    "7cd3d773c584fee7656da15eec827cd26b450825ec878716389f1e9a2ae1a484"
)
_HISTORY_SHA256 = (
    "e42ef1a7a454e1208f5723fd3b87ba70d0e64579a68838ede911b5286e576008"
)
_FIXED_LOCAL_TIMESTAMP = "2026-07-24 09:00:00"
_FIXED_SCHEDULED_LOCAL_TIMESTAMP = "2026-07-25 15:00:00"
_CHARACTER_GLOBAL_ID = "character-global"
_CURRENT_USER_GLOBAL_ID = "baseline-current-user"
_CURRENT_USER_PLATFORM_ID = "baseline-current-user-platform"
_BOT_PLATFORM_ID = "baseline-character-platform"
_CURRENT_USER_NAME = "基线测试用户"
_ASUNA_ALIAS = "明日奈"
_KAZUSA_PROJECT_TOKEN = "KazusaAIChatbot"
_KAZUSA_PROJECT_URL = "https://github.com/eamars/KazusaAIChatbot"
_REQUIRED_LLM_ENV = (
    "RELEVANCE_AGENT_LLM_BASE_URL",
    "RELEVANCE_AGENT_LLM_API_KEY",
    "RELEVANCE_AGENT_LLM_MODEL",
    "VISION_DESCRIPTOR_LLM_BASE_URL",
    "VISION_DESCRIPTOR_LLM_API_KEY",
    "VISION_DESCRIPTOR_LLM_MODEL",
    "MSG_DECONTEXTUALIZER_LLM_BASE_URL",
    "MSG_DECONTEXTUALIZER_LLM_API_KEY",
    "MSG_DECONTEXTUALIZER_LLM_MODEL",
    "RAG_PLANNER_LLM_BASE_URL",
    "RAG_PLANNER_LLM_API_KEY",
    "RAG_PLANNER_LLM_MODEL",
    "RAG_SUBAGENT_LLM_BASE_URL",
    "RAG_SUBAGENT_LLM_API_KEY",
    "RAG_SUBAGENT_LLM_MODEL",
    "WEB_SEARCH_LLM_BASE_URL",
    "WEB_SEARCH_LLM_API_KEY",
    "WEB_SEARCH_LLM_MODEL",
    "COGNITION_LLM_BASE_URL",
    "COGNITION_LLM_API_KEY",
    "COGNITION_LLM_MODEL",
    "BOUNDARY_CORE_LLM_BASE_URL",
    "BOUNDARY_CORE_LLM_API_KEY",
    "BOUNDARY_CORE_LLM_MODEL",
    "DIALOG_GENERATOR_LLM_BASE_URL",
    "DIALOG_GENERATOR_LLM_API_KEY",
    "DIALOG_GENERATOR_LLM_MODEL",
    "CONSOLIDATION_LLM_BASE_URL",
    "CONSOLIDATION_LLM_API_KEY",
    "CONSOLIDATION_LLM_MODEL",
    "JSON_REPAIR_LLM_BASE_URL",
    "JSON_REPAIR_LLM_API_KEY",
    "JSON_REPAIR_LLM_MODEL",
    "BACKGROUND_WORK_LLM_BASE_URL",
    "BACKGROUND_WORK_LLM_API_KEY",
    "BACKGROUND_WORK_LLM_MODEL",
    "CODING_AGENT_PM_LLM_BASE_URL",
    "CODING_AGENT_PM_LLM_API_KEY",
    "CODING_AGENT_PM_LLM_MODEL",
    "CODING_AGENT_PROGRAMMER_LLM_BASE_URL",
    "CODING_AGENT_PROGRAMMER_LLM_API_KEY",
    "CODING_AGENT_PROGRAMMER_LLM_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
)
_REQUIRED_DOTENV_ENV = (
    "MONGODB_URI",
    "MONGODB_DB_NAME",
    *_REQUIRED_LLM_ENV,
)
_GUARDED_DATABASE_NAMES = frozenset({
    "_test_kazusa_live_llm",
    "_test_kazusa_core_v2",
})


def _configure_utf8_streams() -> None:
    """Keep Chinese manifests and raw model output printable on Windows."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_streams()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from a UTF-8 harness file."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _configured_dotenv_environment() -> dict[str, str]:
    """Load the explicit project ``.env`` values for live child processes."""

    raw_values = dotenv_values(_ROOT / ".env")
    values = {
        str(key): str(value)
        for key, value in raw_values.items()
        if value is not None and str(value).strip()
    }
    missing = [
        name
        for name in _REQUIRED_DOTENV_ENV
        if not values.get(name, "").strip()
    ]
    if missing:
        raise ValueError(
            "configured .env is missing required live values: "
            + ", ".join(missing)
        )
    database_name = values["MONGODB_DB_NAME"].strip()
    if database_name not in _GUARDED_DATABASE_NAMES:
        raise ValueError(
            "configured MONGODB_DB_NAME is outside the recognized guarded "
            f"test databases: {database_name}"
        )
    return values


def _configured_database_name() -> str:
    """Return the exact Mongo database configured in the project ``.env``."""

    return _configured_dotenv_environment()["MONGODB_DB_NAME"].strip()


def _sha256(path: Path) -> str:
    """Return one lowercase file SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_name() -> str:
    """Return the fixed external profile's full name."""

    value = _load_json(_PROFILE_PATH).get("name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Asuna profile has no name")
    return value.strip()


def _semantic_name(full_name: str) -> str:
    """Return the name used in model-facing semantic fields."""

    return full_name.split("(", maxsplit=1)[0].strip()


def _replace_text(
    value: str,
    replacements: list[tuple[str, str]],
) -> str:
    """Replace only declared identity spans in one text field."""

    result = value
    for source, target in sorted(
        replacements,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if source and source != target:
            result = result.replace(source, target)
    return result


def _source_character_metadata(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Infer the historical character from typed bot mentions."""

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
            global_id = mention.get("global_user_id")
            platform_id = mention.get("platform_user_id")
            display_name = mention.get("display_name")
            if isinstance(global_id, str) and global_id.strip():
                global_ids[global_id] += 1
            if isinstance(platform_id, str) and platform_id.strip():
                platform_ids[platform_id] += 1
            if isinstance(display_name, str) and display_name.strip():
                display_names[display_name] += 1
    if not global_ids:
        raise ValueError("history export has no typed bot identity")
    source_global_id = global_ids.most_common(1)[0][0]
    return {
        "global_user_id": source_global_id,
        "platform_user_ids": [
            value
            for value, _ in platform_ids.most_common()
            if value.strip()
        ],
        "display_name": (
            display_names.most_common(1)[0][0]
            if display_names
            else "杏山千纱"
        ),
        "aliases": ["千纱"],
    }


def _stable_replay_user(source_identity: str) -> tuple[str, str]:
    """Build deterministic replay IDs for a non-current history participant."""

    digest = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()[:12]
    return (
        f"real-history-user-{digest}",
        f"real-history-global-{digest}",
    )


def _project_history_message(
    *,
    message: Mapping[str, Any],
    case_id: str,
    position: int | None,
    source_character: Mapping[str, Any],
    source_current_global_id: str,
    message_id_map: Mapping[str, str],
    active_name: str,
) -> dict[str, Any]:
    """Project typed history identity without recursively rewriting evidence."""

    source_character_global_id = str(source_character["global_user_id"])
    source_character_platform_ids = {
        str(value)
        for value in source_character["platform_user_ids"]
    }
    source_display_name = str(source_character["display_name"])
    identity_replacements = [
        (source_display_name, active_name),
        *[(str(alias), _ASUNA_ALIAS)
          for alias in source_character["aliases"]],
    ]
    projected = dict(message)
    role = str(message.get("role") or "")
    source_global_id = str(message.get("global_user_id") or "")
    source_platform_id = str(message.get("platform_user_id") or "")
    if role == "assistant":
        projected.update({
            "platform_user_id": _BOT_PLATFORM_ID,
            "global_user_id": _CHARACTER_GLOBAL_ID,
            "display_name": active_name,
        })
    elif source_global_id == source_current_global_id:
        projected.update({
            "platform_user_id": _CURRENT_USER_PLATFORM_ID,
            "global_user_id": _CURRENT_USER_GLOBAL_ID,
        })
    else:
        replay_platform_id, replay_global_id = _stable_replay_user(
            source_global_id or source_platform_id
            or str(message.get("display_name") or "unknown-user")
        )
        projected.update({
            "platform_user_id": replay_platform_id,
            "global_user_id": replay_global_id,
        })
    body_text = message.get("body_text")
    if isinstance(body_text, str):
        projected["body_text"] = _replace_text(
            body_text,
            identity_replacements,
        )
    addressed = message.get("addressed_to_global_user_ids")
    if isinstance(addressed, list):
        projected["addressed_to_global_user_ids"] = [
            _CHARACTER_GLOBAL_ID
            if str(value) == source_character_global_id
            else _CURRENT_USER_GLOBAL_ID
            if str(value) == source_current_global_id
            else str(value)
            for value in addressed
        ]
    mentions = message.get("mentions")
    if isinstance(mentions, list):
        projected_mentions: list[dict[str, Any]] = []
        for mention in mentions:
            if not isinstance(mention, Mapping):
                continue
            projected_mention = dict(mention)
            mention_global_id = str(mention.get("global_user_id") or "")
            mention_platform_id = str(
                mention.get("platform_user_id") or ""
            )
            is_character = (
                mention_global_id == source_character_global_id
                or mention_platform_id in source_character_platform_ids
            )
            if is_character:
                projected_mention.update({
                    "global_user_id": _CHARACTER_GLOBAL_ID,
                    "platform_user_id": _BOT_PLATFORM_ID,
                    "display_name": active_name,
                })
            elif mention_global_id == source_current_global_id:
                projected_mention.update({
                    "global_user_id": _CURRENT_USER_GLOBAL_ID,
                    "platform_user_id": _CURRENT_USER_PLATFORM_ID,
                })
            raw_text = projected_mention.get("raw_text")
            if isinstance(raw_text, str):
                mention_replacements = list(identity_replacements)
                if is_character:
                    mention_replacements.extend(
                        (source_id, _BOT_PLATFORM_ID)
                        for source_id in source_character_platform_ids
                    )
                projected_mention["raw_text"] = _replace_text(
                    raw_text,
                    mention_replacements,
                )
            projected_mentions.append(projected_mention)
        projected["mentions"] = projected_mentions
    reply_context = message.get("reply_context")
    if isinstance(reply_context, Mapping):
        projected_reply = dict(reply_context)
        reply_id = str(reply_context.get("reply_to_message_id") or "")
        if reply_id in message_id_map:
            projected_reply["reply_to_message_id"] = message_id_map[reply_id]
        reply_global_id = str(
            reply_context.get("reply_to_global_user_id") or ""
        )
        reply_platform_id = str(
            reply_context.get("reply_to_platform_user_id") or ""
        )
        if (
            reply_global_id == source_character_global_id
            or reply_platform_id in source_character_platform_ids
        ):
            projected_reply.update({
                "reply_to_global_user_id": _CHARACTER_GLOBAL_ID,
                "reply_to_platform_user_id": _BOT_PLATFORM_ID,
                "reply_to_display_name": active_name,
            })
        excerpt = projected_reply.get("reply_excerpt")
        if isinstance(excerpt, str):
            projected_reply["reply_excerpt"] = _replace_text(
                excerpt,
                identity_replacements,
            )
        projected["reply_context"] = projected_reply
    original_raw_wire = str(
        message.get("raw_wire_text") or message.get("body_text") or ""
    )
    raw_replacements = [
        *identity_replacements,
        *[
            (source_id, target_id)
            for source_id, target_id in message_id_map.items()
        ],
        *[
            (source_id, _BOT_PLATFORM_ID)
            for source_id in source_character_platform_ids
        ],
    ]
    projected["raw_wire_text"] = _replace_text(
        original_raw_wire,
        raw_replacements,
    )
    projected["platform"] = "debug"
    projected["platform_channel_id"] = f"real-history-asuna-{case_id}"
    projected["channel_name"] = "real-history-replay"
    projected["platform_message_id"] = (
        f"{case_id}-current"
        if position is None
        else f"{case_id}-source-{position:02d}"
    )
    return projected


def _history_case_payload(
    case: Mapping[str, Any],
    *,
    active_name: str,
) -> dict[str, Any]:
    """Build one neutral Asuna-projected real-history case payload."""

    source = _load_json(_HISTORY_PATH)
    messages = source.get("messages")
    if not isinstance(messages, list):
        raise ValueError("history export messages are missing")
    source_message_id = str(case["source_message_id"])
    source_index = next(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, Mapping)
            and str(message.get("platform_message_id") or "")
            == source_message_id
        ),
        None,
    )
    if source_index is None:
        raise ValueError(f"history source message is missing: {source_message_id}")
    source_message = messages[source_index]
    if not isinstance(source_message, Mapping):
        raise ValueError("history source message is not an object")
    source_character = _source_character_metadata(
        [dict(message) for message in messages if isinstance(message, Mapping)]
    )
    source_current_global_id = str(
        source_message.get("global_user_id") or ""
    )
    context_start = max(0, source_index - 8)
    source_context = messages[context_start:source_index]
    message_id_map = {
        str(message.get("platform_message_id")): (
            f"{case['case_id']}-source-{position:02d}"
        )
        for position, message in enumerate(source_context)
        if isinstance(message, Mapping)
        and str(message.get("platform_message_id") or "")
    }
    message_id_map[source_message_id] = f"{case['case_id']}-current"
    effective_context = [
        _project_history_message(
            message=message,
            case_id=str(case["case_id"]),
            position=position,
            source_character=source_character,
            source_current_global_id=source_current_global_id,
            message_id_map=message_id_map,
            active_name=active_name,
        )
        for position, message in enumerate(source_context)
        if isinstance(message, Mapping)
    ]
    effective_input = _project_history_message(
        message=source_message,
        case_id=str(case["case_id"]),
        position=None,
        source_character=source_character,
        source_current_global_id=source_current_global_id,
        message_id_map=message_id_map,
        active_name=active_name,
    )
    return {
        "case_id": case["case_id"],
        "source_kind": "user_message",
        "event_kind": "chat",
        "category": "direct_kazusa_history",
        "mode": "role_bound_identity_projection",
        "output_mode": "visible",
        "input_text": str(effective_input.get("body_text") or ""),
        "source_input": dict(source_message),
        "source_context": [
            dict(message)
            for message in source_context
            if isinstance(message, Mapping)
        ],
        "effective_input": effective_input,
        "effective_context": effective_context,
        "source_character": source_character,
        "source_message_id": source_message_id,
        "source_index": source_index,
        "identity_mapping": [
            {
                "kind": "historical_character_display_name",
                "source": source_character["display_name"],
                "target": active_name,
            },
            {
                "kind": "historical_character_alias",
                "source": "千纱",
                "target": _ASUNA_ALIAS,
            },
        ],
        "immutable_external_spans": _external_spans(
            effective_input,
            json_path="message_envelope.body_text",
        ),
        "hard_gates": [
            "visible_dialog",
            "canonical_monologue",
            "no_source_identity_leak",
            "immutable_external_text",
        ],
    }


def _external_spans(
    message: Mapping[str, Any],
    *,
    json_path: str,
) -> list[dict[str, str]]:
    """Declare immutable project text present in one projected message."""

    body_text = str(message.get("body_text") or "")
    spans: list[dict[str, str]] = []
    for token, kind in (
        (_KAZUSA_PROJECT_URL, "repository_url"),
        (_KAZUSA_PROJECT_TOKEN, "repository_name"),
    ):
        if token in body_text:
            spans.append({
                "json_path": json_path,
                "value": token,
                "kind": kind,
            })
    return spans


def _basic_case_payload(
    case: Mapping[str, Any],
    *,
    active_name: str,
) -> dict[str, Any]:
    """Build one controlled or owner case without semantic rewriting."""

    case_id = str(case["case_id"])
    source_kind = str(case.get("source_kind") or "user_message")
    event_kind = str(case.get("event_kind") or "chat")
    output_mode = str(case.get("output_mode") or "visible")
    addressee = str(case.get("addressee_role") or "none")
    is_chat = event_kind == "chat"
    addressed = addressee == "character" and is_chat
    effective_input = {
        "platform": "debug",
        "platform_channel_id": f"baseline-{case_id}",
        "role": "user" if is_chat else "system",
        "platform_message_id": f"{case_id}-current",
        "platform_user_id": _CURRENT_USER_PLATFORM_ID,
        "global_user_id": _CURRENT_USER_GLOBAL_ID,
        "display_name": _CURRENT_USER_NAME,
        "channel_type": "group" if is_chat else "private",
        "body_text": str(case.get("input_text") or ""),
        "raw_wire_text": str(case.get("input_text") or ""),
        "content_type": "text",
        "addressed_to_global_user_ids": (
            [_CHARACTER_GLOBAL_ID] if addressed else []
        ),
        "mentions": (
            [{
                "platform_user_id": _BOT_PLATFORM_ID,
                "global_user_id": _CHARACTER_GLOBAL_ID,
                "display_name": active_name,
                "entity_kind": "bot",
                "raw_text": f"@{active_name}",
            }]
            if addressed
            else []
        ),
        "broadcast": not addressed and is_chat,
        "attachments": [],
        "reply_context": {},
        "timestamp": _FIXED_LOCAL_TIMESTAMP,
    }
    state_seed = case.get("state_seed")
    if not isinstance(state_seed, Mapping):
        state_seed = {}
    if case_id == "O02":
        effective_input["reply_context"] = {
            "reply_to_message_id": f"{case_id}-prior",
            "reply_to_platform_user_id": _BOT_PLATFORM_ID,
            "reply_to_display_name": active_name,
            "reply_excerpt": str(state_seed.get("prior_message") or ""),
        }
    if case_id == "O03":
        effective_input["attachments"] = [{
            "media_type": "image/png",
            "storage_shape": "path",
            "path": str(state_seed.get("resource_path") or ""),
        }]
        effective_input["content_type"] = "mixed"
    return {
        "case_id": case_id,
        "source_kind": source_kind,
        "event_kind": event_kind,
        "mode": "controlled",
        "output_mode": output_mode,
        "input_text": str(case.get("input_text") or ""),
        "source_input": {},
        "source_context": [],
        "effective_input": effective_input,
        "effective_context": [],
        "source_character": {},
        "identity_mapping": [],
        "immutable_external_spans": _external_spans(
            effective_input,
            json_path="message_envelope.body_text",
        ),
        "hard_gates": list(case.get("hard_gates") or []),
        "state_seed": dict(state_seed),
    }


def _provenance_failures(case: Mapping[str, Any]) -> list[str]:
    """Reject mismatched typed roles, identity fields, or chronology."""

    failures: list[str] = []
    event_kind = str(case.get("event_kind") or "chat")
    effective_input = case.get("effective_input")
    if not isinstance(effective_input, Mapping):
        return ["effective input is missing"]

    experiencer_role = str(case.get("experiencer_role") or "")
    if experiencer_role != "character":
        failures.append(
            "experiencer role must remain the active character"
        )

    if event_kind == "chat":
        actor_role = str(case.get("actor_role") or "")
        if actor_role == "current_user" and str(
            effective_input.get("global_user_id") or ""
        ) != _CURRENT_USER_GLOBAL_ID:
            failures.append("actor identity does not match current user")
        if actor_role == "character" and str(
            effective_input.get("global_user_id") or ""
        ) != _CHARACTER_GLOBAL_ID:
            failures.append("actor identity does not match character")

        addressed = effective_input.get("addressed_to_global_user_ids")
        addressed_ids = [str(value) for value in addressed]
        addressee_role = str(case.get("addressee_role") or "none")
        character_addressed = _CHARACTER_GLOBAL_ID in addressed_ids
        current_user_addressed = _CURRENT_USER_GLOBAL_ID in addressed_ids
        if addressee_role == "character" and not character_addressed:
            failures.append("character addressee is missing")
        if addressee_role == "none" and character_addressed:
            failures.append("unexpected character addressee")
        if addressee_role == "current_user" and not current_user_addressed:
            failures.append("current-user addressee is missing")

        mentions = effective_input.get("mentions")
        mention_rows = [
            mention for mention in mentions
            if isinstance(mention, Mapping)
        ] if isinstance(mentions, list) else []
        character_mentioned = any(
            str(mention.get("global_user_id") or "")
            == _CHARACTER_GLOBAL_ID
            for mention in mention_rows
        )
        if addressee_role == "none" and character_mentioned:
            failures.append("unexpected character mention")

        reply_context = effective_input.get("reply_context")
        if isinstance(reply_context, Mapping) and reply_context.get(
            "reply_to_message_id"
        ):
            reply_role = str(case.get("reply_author_role") or "")
            reply_global_id = str(
                reply_context.get("reply_to_global_user_id") or ""
            )
            reply_platform_id = str(
                reply_context.get("reply_to_platform_user_id") or ""
            )
            if reply_role == "character" and not (
                reply_global_id == _CHARACTER_GLOBAL_ID
                or reply_platform_id == _BOT_PLATFORM_ID
            ):
                failures.append("reply author does not match character")
            if reply_role == "current_user" and not (
                reply_global_id == _CURRENT_USER_GLOBAL_ID
                or reply_platform_id == _CURRENT_USER_PLATFORM_ID
            ):
                failures.append("reply author does not match current user")

    source_context = case.get("source_context")
    effective_context = case.get("effective_context")
    source_input = case.get("source_input")
    message_id_map: dict[str, str] = {}
    if isinstance(source_context, list) or isinstance(effective_context, list):
        if not isinstance(source_context, list) or not isinstance(
            effective_context,
            list,
        ):
            failures.append("source/effective context shape differs")
        elif len(source_context) != len(effective_context):
            failures.append("source/effective context length differs")
        else:
            source_character = case.get("source_character")
            source_character_global_id = str(
                source_character.get("global_user_id") or ""
            ) if isinstance(source_character, Mapping) else ""
            source_character_platform_ids = {
                str(value)
                for value in source_character.get("platform_user_ids", [])
            } if isinstance(source_character, Mapping) else set()
            source_current_global_id = str(
                case.get("source_input", {}).get("global_user_id") or ""
            ) if isinstance(case.get("source_input"), Mapping) else ""
            source_message_ids = {
                str(row.get("platform_message_id") or "")
                for row in source_context
                if isinstance(row, Mapping)
            }
            source_input_id = (
                str(source_input.get("platform_message_id") or "")
                if isinstance(source_input, Mapping)
                else ""
            )
            message_id_map.update({
                source_id: (
                    f"{case['case_id']}-source-{index:02d}"
                    if source_id in source_message_ids
                    else f"{case['case_id']}-current"
                )
                for index, source_id in enumerate(
                    [
                        str(row.get("platform_message_id") or "")
                        for row in source_context
                        if isinstance(row, Mapping)
                    ]
                )
                if source_id
            })
            if source_input_id:
                message_id_map[source_input_id] = (
                    f"{case['case_id']}-current"
                )
            for index, (source_row, effective_row) in enumerate(
                zip(source_context, effective_context, strict=True)
            ):
                if not isinstance(source_row, Mapping) or not isinstance(
                    effective_row,
                    Mapping,
                ):
                    failures.append(f"context row {index} is not an object")
                    continue
                if source_row.get("timestamp") != effective_row.get(
                    "timestamp"
                ):
                    failures.append(
                        f"context chronology changed at index {index}"
                    )
                expected_message_id = f"{case['case_id']}-source-{index:02d}"
                if effective_row.get("platform_message_id") != (
                    expected_message_id
                ):
                    failures.append(
                        f"context message identity changed at index {index}"
                    )
                source_mentions = source_row.get("mentions")
                effective_mentions = effective_row.get("mentions")
                if not isinstance(source_mentions, list) or not isinstance(
                    effective_mentions,
                    list,
                ):
                    continue
                if len(source_mentions) != len(effective_mentions):
                    failures.append(
                        f"context mention count changed at index {index}"
                    )
                    continue
                for mention_index, (
                    source_mention,
                    effective_mention,
                ) in enumerate(zip(source_mentions, effective_mentions)):
                    if not isinstance(source_mention, Mapping) or not isinstance(
                        effective_mention,
                        Mapping,
                    ):
                        continue
                    source_global_id = str(
                        source_mention.get("global_user_id") or ""
                    )
                    source_platform_id = str(
                        source_mention.get("platform_user_id") or ""
                    )
                    is_character = (
                        source_global_id == source_character_global_id
                        or source_platform_id in source_character_platform_ids
                    )
                    is_current_user = (
                        source_global_id == source_current_global_id
                    )
                    if is_character or is_current_user:
                        continue
                    for key in (
                        "global_user_id",
                        "platform_user_id",
                        "display_name",
                        "entity_kind",
                        "raw_text",
                    ):
                        if effective_mention.get(key) != source_mention.get(
                            key
                        ):
                            failures.append(
                                "quoted third-party identity changed at "
                                f"context {index}, mention {mention_index}"
                            )
                            break

                source_reply = source_row.get("reply_context")
                effective_reply = effective_row.get("reply_context")
                if not isinstance(source_reply, Mapping) or not isinstance(
                    effective_reply,
                    Mapping,
                ):
                    continue
                source_reply_id = str(
                    source_reply.get("reply_to_message_id") or ""
                )
                effective_reply_id = str(
                    effective_reply.get("reply_to_message_id") or ""
                )
                if source_reply_id or effective_reply_id:
                    expected_reply_id = message_id_map.get(
                        source_reply_id,
                        source_reply_id,
                    )
                    if effective_reply_id != expected_reply_id:
                        failures.append(
                            "reply chronology changed at "
                            f"context {index}"
                        )
                reply_global_id = str(
                    source_reply.get("reply_to_global_user_id") or ""
                )
                reply_platform_id = str(
                    source_reply.get("reply_to_platform_user_id") or ""
                )
                is_character_reply = (
                    reply_global_id == source_character_global_id
                    or reply_platform_id in source_character_platform_ids
                )
                is_current_reply = reply_global_id == source_current_global_id
                if is_character_reply or is_current_reply:
                    continue
                for key in (
                    "reply_to_global_user_id",
                    "reply_to_platform_user_id",
                    "reply_to_display_name",
                ):
                    if effective_reply.get(key) != source_reply.get(key):
                        failures.append(
                            "quoted third-party reply identity changed at "
                            f"context {index}"
                        )
                        break

    if isinstance(source_input, Mapping) and source_input.get("timestamp"):
        if source_input.get("timestamp") != effective_input.get("timestamp"):
            failures.append("current input chronology changed")
        expected_input_id = f"{case['case_id']}-current"
        if effective_input.get("platform_message_id") != expected_input_id:
            failures.append("current input message identity changed")
        source_reply = source_input.get("reply_context")
        effective_reply = effective_input.get("reply_context")
        if isinstance(source_reply, Mapping) and isinstance(
            effective_reply,
            Mapping,
        ):
            source_reply_id = str(
                source_reply.get("reply_to_message_id") or ""
            )
            if source_reply_id and effective_reply.get(
                "reply_to_message_id"
            ) != message_id_map.get(source_reply_id, source_reply_id):
                failures.append("current input reply chronology changed")
    return failures


def _history_reply_author_role(payload: Mapping[str, Any]) -> str:
    """Classify a history reply target without changing its identity."""

    effective_input = payload.get("effective_input")
    if not isinstance(effective_input, Mapping):
        return "current_user"
    reply_context = effective_input.get("reply_context")
    if not isinstance(reply_context, Mapping):
        return "current_user"
    reply_platform_id = str(
        reply_context.get("reply_to_platform_user_id") or ""
    )
    if reply_platform_id == _BOT_PLATFORM_ID:
        return "character"
    if reply_platform_id:
        return "third_party"
    return "current_user"


def _load_case_rows() -> list[dict[str, Any]]:
    """Load and expand the fixed 50-case scored corpus."""

    controlled = _load_json(
        _FIXTURE_ROOT / "cognition_baseline_controlled_cases.json"
    )
    history = _load_json(
        _FIXTURE_ROOT / "cognition_baseline_real_history_cases.json"
    )
    owner = _load_json(
        _FIXTURE_ROOT / "cognition_baseline_owner_cases.json"
    )
    active_name = _semantic_name(_profile_name())
    rows: list[dict[str, Any]] = []
    repeated_ids = {"C07", "C09", "C11", "C16", "C18", "C19"}
    for case in controlled["cases"]:
        payload = _basic_case_payload(case, active_name=active_name)
        rows.append({
            **payload,
            "corpus_group": "controlled",
            "repetition_count": 3 if case["case_id"] in repeated_ids else 1,
            "applicable_dimensions": list(case["applicable_dimensions"]),
            "actor_role": case["actor_role"],
            "addressee_role": case["addressee_role"],
            "experiencer_role": case["experiencer_role"],
            "reply_author_role": case["reply_author_role"],
        })
    for case in history["cases"]:
        payload = _history_case_payload(case, active_name=active_name)
        rows.append({
            **payload,
            "corpus_group": "real_history",
            "repetition_count": 1,
            "applicable_dimensions": [
                "goal_completion",
                "role_identity",
                "character_judgment",
                "dialog_fidelity",
                "continuity",
                "source_output_mode",
            ],
            "actor_role": "current_user",
            "addressee_role": "character",
            "experiencer_role": "character",
            "reply_author_role": _history_reply_author_role(payload),
        })
    for case in owner["cases"]:
        payload = _basic_case_payload(case, active_name=active_name)
        rows.append({
            **payload,
            "corpus_group": "owner_source",
            "repetition_count": 3,
            "applicable_dimensions": list(case["applicable_dimensions"]),
            "actor_role": case["actor_role"],
            "addressee_role": case["addressee_role"],
            "experiencer_role": case["experiencer_role"],
            "reply_author_role": case["reply_author_role"],
        })
    if len(rows) != 50:
        raise ValueError(f"expected 50 scored cases, found {len(rows)}")
    return rows


def _corpus_config(corpus: str) -> tuple[str, Path, str]:
    """Return revision, target worktree and evidence label for one corpus."""

    if corpus == "pre_fix_main":
        return "main", _BASELINE_ROOT, _BASELINE_REVISION
    if corpus == "pre_fix_v2":
        return "v2", _V2_ROOT, _CANDIDATE_REVISION
    if corpus == "post_fix_v2":
        return "v2", _CURRENT_ROOT, _CANDIDATE_REVISION
    raise ValueError(f"unsupported scored corpus: {corpus}")


def _state_path(corpus: str) -> Path:
    """Return the local progress state path for one corpus."""

    return _EVIDENCE_ROOT / corpus / "run_state.json"


def _load_state(corpus: str) -> dict[str, Any]:
    """Load or initialize one corpus progress state."""

    path = _state_path(corpus)
    if not path.is_file():
        return {"schema_version": "cognition_baseline_run_state.v1", "completed": []}
    value = _load_json(path)
    if not isinstance(value.get("completed"), list):
        raise ValueError(f"invalid run state: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one generated harness manifest or state artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_digest(value: object) -> str:
    """Hash one canonical JSON-shaped manifest value."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _case_digest(case: Mapping[str, Any]) -> str:
    """Hash a scored case without execution-specific scheduling fields."""

    stable_case = {
        key: value
        for key, value in case.items()
        if key not in {"execution_id", "repetition_ordinal"}
    }
    return _json_digest(stable_case)


def _safe_slug(value: str) -> str:
    """Validate one manifest-controlled path component."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"unsafe path component: {value!r}")
    return value


def _case_workspace_root(case: Mapping[str, Any]) -> Path:
    """Return the guarded disposable coding workspace for one case."""

    case_id = _safe_slug(str(case["case_id"]))
    corpus_slug = _safe_slug(str(case["corpus_group"]))
    repetition = _safe_slug(str(case.get("repetition_ordinal", 1)))
    root = (
        _ROOT / "test_runs" / corpus_slug / case_id / repetition
    ).resolve()
    guard_root = (_ROOT / "test_runs").resolve()
    if guard_root not in root.parents:
        raise ValueError(f"coding workspace escaped guard root: {root}")
    return root


def _next_case(
    corpus: str,
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], int] | None:
    """Return the first unexecuted case/repetition pair."""

    completed = {
        str(value)
        for value in state.get("completed", [])
    }
    for case in _load_case_rows():
        for repetition in range(1, int(case["repetition_count"]) + 1):
            execution_id = f"{case['case_id']}::r{repetition}"
            if execution_id not in completed:
                selected = dict(case)
                selected["repetition_ordinal"] = repetition
                selected["execution_id"] = execution_id
                return selected, repetition
    return None


def _worker_environment(
    *,
    database_name: str,
    target_root: Path,
    workspace_root: Path,
) -> dict[str, str]:
    """Build the worker environment from the configured project ``.env``."""

    environment = dict(os.environ)
    configured = _configured_dotenv_environment()
    configured_database_name = configured["MONGODB_DB_NAME"].strip()
    if database_name != configured_database_name:
        raise ValueError(
            "worker database does not match configured MONGODB_DB_NAME: "
            f"{database_name} != {configured_database_name}"
        )
    environment.update(configured)
    environment.update({
        "PYTHON_DOTENV_DISABLED": "1",
        "CHARACTER_PROFILE_PATH": str(_PROFILE_PATH),
        "MONGODB_DB_NAME": configured_database_name,
        "CHARACTER_GLOBAL_USER_ID": _CHARACTER_GLOBAL_ID,
        "CHARACTER_TIME_ZONE": "Pacific/Auckland",
        "KAZUSA_TEST_DB_GUARD": "1",
        "STAGE3_DATABASE_GUARD": "1",
        "SELF_COGNITION_ENABLED": "false",
        "CALENDAR_SCHEDULER_ENABLED": "false",
        "BACKGROUND_WORK_WORKER_ENABLED": "false",
        "REFLECTION_CYCLE_ENABLED": "false",
        "CODING_AGENT_WORKSPACE_ROOT": str(workspace_root),
        "KAZUSA_BASELINE_TARGET_ROOT": str(target_root.resolve()),
    })
    return environment


def _database_name(corpus: str, case_id: str, repetition: int) -> str:
    """Return the exact configured guarded database for every isolated case."""

    del corpus, case_id, repetition
    return _configured_database_name()


def _run_next(corpus: str) -> int:
    """Spawn exactly one isolated worker execution."""

    revision, target_root, revision_sha = _corpus_config(corpus)
    if not target_root.is_dir():
        raise ValueError(f"target worktree is missing: {target_root}")
    state = _load_state(corpus)
    selected = _next_case(corpus, state)
    if selected is None:
        print(f"{corpus}: complete")
        return 0
    case, repetition = selected
    execution_id = str(case["execution_id"])
    workspace_root = _case_workspace_root(case)
    database_name = _database_name(corpus, str(case["case_id"]), repetition)
    case_root = _EVIDENCE_ROOT / corpus / str(case["case_id"])
    case_root.mkdir(parents=True, exist_ok=True)
    manifest_path = case_root / f"r{repetition}.input.json"
    output_path = case_root / f"r{repetition}.json"
    worker_manifest = {
        "schema_version": "cognition_baseline_worker_input.v1",
        "execution_id": execution_id,
        "corpus": corpus,
        "revision": revision,
        "revision_sha": revision_sha,
        "target_root": str(target_root.resolve()),
        "profile_path": str(_PROFILE_PATH),
        "profile_sha256": _PROFILE_SHA256,
        "history_path": str(_HISTORY_PATH),
        "history_sha256": _HISTORY_SHA256,
        "database_name": database_name,
        "fixed_local_timestamp": _FIXED_LOCAL_TIMESTAMP,
        "fixed_scheduled_local_timestamp": _FIXED_SCHEDULED_LOCAL_TIMESTAMP,
        "case_sha256": _case_digest(case),
        "workspace_root": str(workspace_root),
        "output_path": str(output_path.resolve()),
        "case": case,
    }
    _write_json(manifest_path, worker_manifest)
    worker_path = _ROOT / "tests" / "cognition_baseline_worker.py"
    python_path = _ROOT / "venv" / "Scripts" / "python.exe"
    if not python_path.is_file():
        raise ValueError(f"project virtual environment is missing: {python_path}")
    command = [
        str(python_path),
        str(worker_path),
        "--input",
        str(manifest_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        env=_worker_environment(
            database_name=database_name,
            target_root=target_root,
            workspace_root=workspace_root,
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (case_root / f"r{repetition}.worker.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (case_root / f"r{repetition}.worker.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if not output_path.is_file():
        raise RuntimeError(
            f"worker produced no artifact for {execution_id}; "
            f"exit={completed.returncode}"
        )
    artifact = _load_json(output_path)
    if not _is_executed_artifact(artifact):
        print(json.dumps({
            "execution_id": execution_id,
            "technical_status": artifact.get("technical_status"),
            "failure_type": artifact.get("failure_type"),
            "failure_message": artifact.get("failure_message"),
            "hard_gate_failures": artifact.get("hard_gate_failures", []),
            "artifact_path": str(output_path),
            "retryable": True,
        }, ensure_ascii=False, indent=2))
        return completed.returncode or 1
    state_rows = list(state.get("completed", []))
    if execution_id not in state_rows:
        state_rows.append(execution_id)
    state_value = {
        **state,
        "completed": state_rows,
        "last_execution_id": execution_id,
        "last_artifact": str(output_path),
    }
    _write_json(_state_path(corpus), state_value)
    return completed.returncode if artifact.get("technical_status") == "passed" else 1


def _preflight_paths() -> int:
    """Require the two fixed worktree paths to be absent before creation."""

    existing = [
        str(path)
        for path in (_BASELINE_ROOT, _V2_ROOT)
        if path.exists()
    ]
    if existing:
        raise ValueError(
            "preflight-paths requires absent paths: " + ", ".join(existing)
        )
    print("preflight-paths: passed")
    return 0


def _preflight() -> int:
    """Verify frozen revisions, inputs, fixture and target worktrees."""

    configured_database_name = _configured_database_name()
    if _sha256(_PROFILE_PATH) != _PROFILE_SHA256:
        raise ValueError("Asuna profile hash changed")
    if _sha256(_HISTORY_PATH) != _HISTORY_SHA256:
        raise ValueError("history export hash changed")
    for path, expected in (
        (_BASELINE_ROOT, _BASELINE_REVISION),
        (_V2_ROOT, _CANDIDATE_REVISION),
    ):
        if not path.is_dir():
            raise ValueError(f"frozen worktree is missing: {path}")
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = completed.stdout.strip()
        if actual != expected:
            raise ValueError(f"{path} SHA mismatch: {actual} != {expected}")
    rows = _load_case_rows()
    if len(rows) != 50:
        raise ValueError("scored case expansion is not exactly 50 rows")
    provenance_failures: dict[str, list[str]] = {}
    for row in rows:
        failures = _provenance_failures(row)
        if failures:
            provenance_failures[str(row["case_id"])] = failures
    if provenance_failures:
        raise ValueError(
            "provenance preflight failed: "
            f"{json.dumps(provenance_failures, ensure_ascii=False)}"
        )
    manifest_root = _EVIDENCE_ROOT / "manifest"
    _write_json(manifest_root / "frozen_inputs.json", {
        "schema_version": "cognition_baseline_frozen_inputs.v1",
        "profile_path": str(_PROFILE_PATH),
        "profile_sha256": _PROFILE_SHA256,
        "history_path": str(_HISTORY_PATH),
        "history_sha256": _HISTORY_SHA256,
        "main_revision": _BASELINE_REVISION,
        "v2_revision": _CANDIDATE_REVISION,
        "baseline_worktree": str(_BASELINE_ROOT),
        "v2_worktree": str(_V2_ROOT),
        "configured_database_name": configured_database_name,
        "fixed_local_timestamp": _FIXED_LOCAL_TIMESTAMP,
        "fixed_scheduled_local_timestamp": (
            _FIXED_SCHEDULED_LOCAL_TIMESTAMP
        ),
    })
    _write_json(manifest_root / "case_manifest.json", {
        "schema_version": "cognition_baseline_case_manifest.v1",
        "case_count": len(rows),
        "repetition_count": sum(
            int(row["repetition_count"]) for row in rows
        ),
        "cases": rows,
    })
    print(json.dumps({
        "preflight": "passed",
        "profile_sha256": _PROFILE_SHA256,
        "history_sha256": _HISTORY_SHA256,
        "scored_case_count": len(rows),
        "main_revision": _BASELINE_REVISION,
        "v2_revision": _CANDIDATE_REVISION,
        "configured_database_name": configured_database_name,
    }, ensure_ascii=False, indent=2))
    return 0


def _artifact_rows(corpus: str) -> list[dict[str, Any]]:
    """Load all completed case artifacts for one corpus."""

    root = _EVIDENCE_ROOT / corpus
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/r*.json")):
        if path.name.endswith(".input.json"):
            continue
        rows.append(_load_json(path))
    return rows


def _is_executed_artifact(row: Mapping[str, Any]) -> bool:
    """Identify semantic observations eligible for state and pairing."""

    status = str(row.get("technical_status") or "")
    if status == "passed":
        return True
    if status != "failed":
        return False
    hard_gate_failures = row.get("hard_gate_failures")
    if (
        isinstance(hard_gate_failures, list)
        and "worker exception" in hard_gate_failures
    ):
        return False
    return (
        isinstance(row.get("response"), Mapping)
        or isinstance(row.get("graph_result"), Mapping)
    )


def _completed_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep executed semantic artifacts eligible for differential pairing."""

    return [
        row
        for row in rows
        if _is_executed_artifact(row)
    ]


def _compare_blind() -> int:
    """Write a structural blinded comparison of completed artifacts."""

    main_rows = {
        str(row.get("execution_id")): row
        for row in _completed_artifacts(_artifact_rows("pre_fix_main"))
    }
    v2_rows = {
        str(row.get("execution_id")): row
        for row in _completed_artifacts(_artifact_rows("pre_fix_v2"))
    }
    blocked_rows = [
        {
            "corpus": corpus,
            "execution_id": row.get("execution_id"),
            "case_id": row.get("case_id"),
            "technical_status": row.get("technical_status"),
            "missing_environment": row.get("missing_environment", []),
        }
        for corpus in ("pre_fix_main", "pre_fix_v2")
        for row in _artifact_rows(corpus)
        if row.get("technical_status") == "blocked_environment"
    ]
    paired_ids = sorted(set(main_rows).intersection(v2_rows))
    comparisons = []
    for execution_id in paired_ids:
        main = main_rows[execution_id]
        v2 = v2_rows[execution_id]
        comparisons.append({
            "pair_id": hashlib.sha256(
                execution_id.encode("utf-8")
            ).hexdigest()[:12],
            "main_technical_status": main.get("technical_status"),
            "v2_technical_status": v2.get("technical_status"),
            "main_surface_status": main.get("response_surface_status"),
            "v2_surface_status": v2.get("response_surface_status"),
            "main_visible_message_count": main.get("visible_message_count", 0),
            "v2_visible_message_count": v2.get("visible_message_count", 0),
            "main_hard_gate_failures": main.get("hard_gate_failures", []),
            "v2_hard_gate_failures": v2.get("hard_gate_failures", []),
        })
    output_path = _EVIDENCE_ROOT / "blind_comparison.json"
    _write_json(output_path, {
        "schema_version": "cognition_baseline_blind_comparison.v1",
        "paired_count": len(comparisons),
        "blocked_artifact_count": len(blocked_rows),
        "blocked_artifacts": blocked_rows,
        "comparisons": comparisons,
    })
    print(f"blind comparison pairs: {len(comparisons)}")
    return 0


def _expected_execution_rows(corpus: str) -> dict[str, dict[str, Any]]:
    """Expand the immutable corpus into its exact execution ledger."""

    expected: dict[str, dict[str, Any]] = {}
    for row in _load_case_rows():
        for repetition in range(1, int(row["repetition_count"]) + 1):
            execution_id = f"{row['case_id']}::r{repetition}"
            expected[execution_id] = {
                "case_id": str(row["case_id"]),
                "repetition": repetition,
                "database_name": _database_name(
                    corpus,
                    str(row["case_id"]),
                    repetition,
                ),
                "case_sha256": _case_digest(row),
            }
    return expected


def _verify_ledger(phase: str = "pre_fix") -> int:
    """Verify that completed artifacts satisfy the execution ledger shape."""

    corpus_names = (
        ("post_fix_v2",) if phase == "post_fix" else
        ("pre_fix_main", "pre_fix_v2")
    )
    failures: list[str] = []
    for corpus in corpus_names:
        expected = _expected_execution_rows(corpus)
        artifacts = _artifact_rows(corpus)
        artifact_ids = [str(row.get("execution_id") or "") for row in artifacts]
        duplicate_ids = sorted({
            execution_id
            for execution_id in artifact_ids
            if artifact_ids.count(execution_id) > 1
        })
        missing_ids = sorted(set(expected) - set(artifact_ids))
        extra_ids = sorted(set(artifact_ids) - set(expected))
        if duplicate_ids:
            failures.append(
                f"{corpus}: duplicate execution ids {duplicate_ids}"
            )
        if missing_ids:
            failures.append(
                f"{corpus}: missing {len(missing_ids)} expected executions"
            )
        if extra_ids:
            failures.append(f"{corpus}: unexpected executions {extra_ids}")
        revision, _, revision_sha = _corpus_config(corpus)
        for row in artifacts:
            execution_id = str(row.get("execution_id") or "")
            expected_row = expected.get(execution_id)
            if expected_row is None:
                continue
            if not _is_executed_artifact(row):
                failures.append(
                    f"{corpus}/{execution_id}: non-executed artifact"
                )
            if row.get("case_id") != expected_row["case_id"]:
                failures.append(f"{corpus}/{execution_id}: case id")
            if row.get("database_name") != expected_row["database_name"]:
                failures.append(
                    f"{corpus}/{execution_id}: database guard"
                )
            if row.get("revision") != revision:
                failures.append(f"{corpus}/{execution_id}: revision")
            if row.get("revision_sha") != revision_sha:
                failures.append(f"{corpus}/{execution_id}: revision SHA")
            if row.get("profile_sha256") != _PROFILE_SHA256:
                failures.append(f"{corpus}/{execution_id}: profile SHA")
            if row.get("history_sha256") != _HISTORY_SHA256:
                failures.append(f"{corpus}/{execution_id}: history SHA")
            if row.get("case_sha256") != expected_row["case_sha256"]:
                failures.append(f"{corpus}/{execution_id}: case SHA")
    if failures:
        raise ValueError("ledger verification failed: " + "; ".join(failures))
    print(f"verify-ledger {phase}: passed")
    return 0


def _verify_deleted_selector_map() -> int:
    """Verify deleted baseline modules exist in main and replacements exist."""

    selector_fixture = _load_json(
        _FIXTURE_ROOT / "cognition_deleted_baseline_selectors.json"
    )
    failures: list[str] = []
    for row in selector_fixture["modules"]:
        main_path = _BASELINE_ROOT / row["main_module"]
        replacement_path = _CURRENT_ROOT / row["replacement_selector"]
        if not main_path.is_file():
            failures.append(f"missing main module: {main_path}")
        if not replacement_path.exists():
            failures.append(f"missing replacement: {replacement_path}")
    if failures:
        raise ValueError("deleted selector map failed: " + "; ".join(failures))
    print("verify-deleted-selector-map: passed")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the controller command-line parser."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight-paths")
    subparsers.add_parser("preflight")
    run_parser = subparsers.add_parser("run-next")
    run_parser.add_argument(
        "--corpus",
        choices=("pre_fix_main", "pre_fix_v2", "post_fix_v2"),
        required=True,
    )
    subparsers.add_parser("compare").add_argument(
        "--blind",
        action="store_true",
        required=True,
    )
    ledger_parser = subparsers.add_parser("verify-ledger")
    ledger_parser.add_argument("--phase", default="pre_fix")
    subparsers.add_parser("verify-deleted-selector-map")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one controller command."""

    args = _build_parser().parse_args(argv)
    if args.command == "preflight-paths":
        return _preflight_paths()
    if args.command == "preflight":
        return _preflight()
    if args.command == "run-next":
        return _run_next(args.corpus)
    if args.command == "compare":
        return _compare_blind()
    if args.command == "verify-ledger":
        return _verify_ledger(args.phase)
    if args.command == "verify-deleted-selector-map":
        return _verify_deleted_selector_map()
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
