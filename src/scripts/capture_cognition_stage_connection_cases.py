"""Capture real QQ conversation cases for cognition-stage connection tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._db_export import (
    configure_logging,
    configure_stdout,
    load_project_env,
)

load_project_env()

from kazusa_ai_chatbot.cognition_episode import (  # noqa: E402
    build_text_chat_cognitive_episode,
)
from kazusa_ai_chatbot.db import close_db, get_conversation_history, get_user_profile  # noqa: E402
from kazusa_ai_chatbot.db.script_operations import export_collection_rows  # noqa: E402
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (  # noqa: E402
    empty_user_memory_context,
)
from kazusa_ai_chatbot.time_context import build_character_time_context  # noqa: E402
from kazusa_ai_chatbot.utils import load_personality, text_or_empty  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "test_artifacts" / "cognition_stage_connection"
DEFAULT_PERSONALITY_PATH = ROOT / "personalities" / "kazusa.json"
CASE_SET_SCHEMA_VERSION = "cognition_stage_connection_case_set.v1"
CASE_SCHEMA_VERSION = "cognition_stage_connection_case.v1"
WINDOW_CONTEXT_LIMIT = 8
ASSISTANT_REPLY_LOOKAHEAD = 6
MAX_SELECTED_SCOPES_BY_TYPE = 8


async def _run(args: argparse.Namespace) -> None:
    """Capture private and group case sets from full conversation windows."""

    configure_stdout()
    configure_logging(args.verbose)
    character_profile = _load_character_profile(args.personality_path)
    authored_rows = await _discover_authored_rows(args)
    scopes = _discover_scopes(authored_rows)
    windows = await _load_scope_windows(args, scopes)
    private_cases = await _build_cases(
        windows,
        source_channel_type="private",
        max_cases=args.max_private,
        character_profile=character_profile,
        platform_user_id=args.platform_user_id,
    )
    group_cases = await _build_cases(
        windows,
        source_channel_type="group",
        max_cases=args.max_group,
        character_profile=character_profile,
        platform_user_id=args.platform_user_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "qq_673225019_scope_discovery.json",
        _scope_discovery_document(args, authored_rows, scopes),
    )
    _write_json(
        args.output_dir / "qq_673225019_private_windows.json",
        _window_document(args, windows, "private"),
    )
    _write_json(
        args.output_dir / "qq_673225019_group_windows.json",
        _window_document(args, windows, "group"),
    )
    private_path = args.output_dir / "qq_673225019_private_cases.json"
    group_path = args.output_dir / "qq_673225019_group_cases.json"
    _write_json(private_path, _case_set_document(args, "qq_private", private_cases))
    _write_json(group_path, _case_set_document(args, "qq_group", group_cases))

    summary = {
        "scope_count": len(scopes),
        "private_case_count": len(private_cases),
        "group_case_count": len(group_cases),
        "private_case_file": str(private_path),
        "group_case_file": str(group_path),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if not private_cases:
        raise RuntimeError("No private user/assistant pairs captured")
    if not group_cases:
        raise RuntimeError("No group user/assistant pairs captured")


async def _main(args: argparse.Namespace) -> None:
    """Run capture and close the database client in the same event loop."""

    try:
        await _run(args)
    finally:
        await close_db()


async def _discover_authored_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Find recent rows authored by the requested platform user."""

    rows = await export_collection_rows(
        collection_name="conversation_history",
        filter_doc={
            "platform": args.platform,
            "platform_user_id": args.platform_user_id,
        },
        projection={"embedding": 0},
        sort_doc={"timestamp": -1},
        limit=args.history_limit,
    )
    projected_rows = [_project_row(row) for row in rows]
    return projected_rows


def _discover_scopes(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build unique private/group conversation scopes from authored rows."""

    seen: set[tuple[str, str, str]] = set()
    scopes_by_type: dict[str, list[dict[str, str]]] = {"private": [], "group": []}
    for row in rows:
        platform = text_or_empty(row.get("platform"))
        channel_type = text_or_empty(row.get("channel_type"))
        platform_channel_id = text_or_empty(row.get("platform_channel_id"))
        if channel_type not in scopes_by_type:
            continue
        if not platform or not platform_channel_id:
            continue
        scope_key = (platform, channel_type, platform_channel_id)
        if scope_key in seen:
            continue
        seen.add(scope_key)
        scope = {
            "platform": platform,
            "channel_type": channel_type,
            "platform_channel_id": platform_channel_id,
        }
        scopes_by_type[channel_type].append(scope)

    scopes: list[dict[str, str]] = []
    for channel_type in ("private", "group"):
        scopes.extend(scopes_by_type[channel_type][:MAX_SELECTED_SCOPES_BY_TYPE])
    return scopes


async def _load_scope_windows(
    args: argparse.Namespace,
    scopes: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Load full conversation windows for each discovered scope."""

    windows: list[dict[str, Any]] = []
    for index, scope in enumerate(scopes, start=1):
        rows = await get_conversation_history(
            platform=scope["platform"],
            platform_channel_id=scope["platform_channel_id"],
            limit=args.history_limit,
        )
        window = {
            "scope_label": f"{scope['channel_type']}_scope_{index:03d}",
            "scope": dict(scope),
            "rows": [_project_row(row) for row in rows],
        }
        windows.append(window)
    return windows


async def _build_cases(
    windows: list[dict[str, Any]],
    *,
    source_channel_type: str,
    max_cases: int,
    character_profile: dict[str, Any],
    platform_user_id: str,
) -> list[dict[str, Any]]:
    """Build selected cases from loaded windows of one channel type."""

    cases: list[dict[str, Any]] = []
    for window in windows:
        scope = window["scope"]
        if scope["channel_type"] != source_channel_type:
            continue
        pairs = _select_pairs(window["rows"], platform_user_id)
        for pair in pairs:
            case_number = len(cases) + 1
            case = await _build_case(
                case_number,
                source_channel_type=source_channel_type,
                scope_label=window["scope_label"],
                pair=pair,
                character_profile=character_profile,
            )
            cases.append(case)
            if len(cases) >= max_cases:
                return cases
    return cases


def _select_pairs(
    rows: list[dict[str, Any]],
    platform_user_id: str,
) -> list[dict[str, Any]]:
    """Select user rows with nearby historical assistant replies."""

    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if text_or_empty(row.get("role")) != "user":
            continue
        if text_or_empty(row.get("platform_user_id")) != platform_user_id:
            continue
        if not text_or_empty(row.get("body_text")):
            continue
        assistant_index = _next_assistant_index(rows, index)
        if assistant_index is None:
            continue
        assistant_row = rows[assistant_index]
        start_index = max(0, index - WINDOW_CONTEXT_LIMIT)
        pair = {
            "user_row": row,
            "assistant_row": assistant_row,
            "history_rows": rows[start_index:index],
            "comparison_rows": rows[index:assistant_index + 1],
        }
        pairs.append(pair)
    return pairs


def _next_assistant_index(rows: list[dict[str, Any]], user_index: int) -> int | None:
    """Return the next nearby assistant reply row index."""

    end_index = min(len(rows), user_index + ASSISTANT_REPLY_LOOKAHEAD + 1)
    for index in range(user_index + 1, end_index):
        row = rows[index]
        role = text_or_empty(row.get("role"))
        body_text = text_or_empty(row.get("body_text"))
        if role == "assistant" and body_text:
            return_value = index
            return return_value
    return_value = None
    return return_value


async def _build_case(
    case_number: int,
    *,
    source_channel_type: str,
    scope_label: str,
    pair: dict[str, Any],
    character_profile: dict[str, Any],
) -> dict[str, Any]:
    """Build one live-LLM comparison case from a user/assistant pair."""

    user_row = pair["user_row"]
    assistant_row = pair["assistant_row"]
    global_user_id = text_or_empty(user_row.get("global_user_id"))
    user_profile = await _load_user_profile(global_user_id)
    seed_state = _build_seed_state(
        user_row,
        pair["history_rows"],
        assistant_row,
        character_profile=character_profile,
        user_profile=user_profile,
    )
    source_kind = f"qq_{source_channel_type}"
    case = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": f"{source_kind}_{case_number:03d}",
        "source_kind": source_kind,
        "source_channel_type": source_channel_type,
        "source_channel_id": scope_label,
        "seed_state": seed_state,
        "historical_user_message": text_or_empty(user_row.get("body_text")),
        "historical_assistant_reply": [
            text_or_empty(assistant_row.get("body_text")),
        ],
        "historical_comparison": {
            "expected_visible_surface": True,
            "comparison_basis": "assistant replied in stored history",
            "assistant_timestamp": text_or_empty(assistant_row.get("timestamp")),
        },
    }
    return case


async def _load_user_profile(global_user_id: str) -> dict[str, Any]:
    """Load or synthesize a minimal user profile for cognition."""

    if global_user_id:
        profile = await get_user_profile(global_user_id)
        if profile:
            normalized_profile = dict(profile)
            normalized_profile.setdefault("affinity", 500)
            normalized_profile.setdefault("facts", [])
            normalized_profile.setdefault("active_commitments", [])
            normalized_profile.setdefault("last_relationship_insight", "")
            return normalized_profile

    default_profile = {
        "affinity": 500,
        "facts": [],
        "active_commitments": [],
        "last_relationship_insight": "",
    }
    return default_profile


def _build_seed_state(
    user_row: dict[str, Any],
    history_rows: list[dict[str, Any]],
    assistant_row: dict[str, Any],
    *,
    character_profile: dict[str, Any],
    user_profile: dict[str, Any],
) -> dict[str, Any]:
    """Build the frozen state used by the live cognition test."""

    timestamp = _row_timestamp(user_row)
    time_context = build_character_time_context(timestamp)
    user_input = text_or_empty(user_row.get("body_text"))
    platform = text_or_empty(user_row.get("platform"))
    platform_channel_id = text_or_empty(user_row.get("platform_channel_id"))
    channel_type = text_or_empty(user_row.get("channel_type"))
    platform_message_id = text_or_empty(user_row.get("platform_message_id"))
    platform_user_id = text_or_empty(user_row.get("platform_user_id"))
    global_user_id = text_or_empty(user_row.get("global_user_id"))
    user_name = text_or_empty(user_row.get("display_name")) or "QQ User"
    conversation_row_id = text_or_empty(user_row.get("row_id"))
    episode = build_text_chat_cognitive_episode(
        episode_id=f"cognition_connection:qq:{platform_message_id or conversation_row_id}",
        percept_id=(
            "cognition_connection:qq:percept:"
            f"{platform_message_id or conversation_row_id}"
        ),
        timestamp=timestamp,
        time_context=time_context,
        user_input=user_input,
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type=channel_type,
        platform_message_id=platform_message_id,
        platform_user_id=platform_user_id,
        global_user_id=global_user_id,
        user_name=user_name,
        active_turn_platform_message_ids=[platform_message_id],
        active_turn_conversation_row_ids=[conversation_row_id],
        debug_modes={"no_visual_directives": True},
        output_mode="visible_reply",
        target_addressed_user_ids=[],
        target_broadcast=False,
    )
    seed_state = {
        "character_profile": character_profile,
        "timestamp": timestamp,
        "time_context": time_context,
        "user_input": user_input,
        "prompt_message_context": _prompt_message_context(user_row),
        "cognitive_episode": episode,
        "user_multimedia_input": [],
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "platform_message_id": platform_message_id,
        "platform_user_id": platform_user_id,
        "global_user_id": global_user_id,
        "user_name": user_name,
        "user_profile": user_profile,
        "platform_bot_id": text_or_empty(assistant_row.get("platform_user_id")),
        "chat_history_wide": _history_projection(history_rows),
        "chat_history_recent": _history_projection(history_rows),
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "conversation_progress": {},
        "promoted_reflection_context": {},
        "debug_modes": {"no_visual_directives": True},
        "should_respond": True,
        "decontexualized_input": user_input,
        "referents": [],
        "rag_result": _empty_rag_result(),
        "action_directives": {},
    }
    return seed_state


def _prompt_message_context(row: dict[str, Any]) -> dict[str, Any]:
    """Project the active row to prompt-safe current-message context."""

    context = {
        "body_text": text_or_empty(row.get("body_text")),
        "mentions": [],
        "attachments": _project_attachments(row.get("attachments")),
        "addressed_to_global_user_ids": [],
        "broadcast": bool(row.get("broadcast")),
    }
    return context


def _history_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project DB rows to the typed history fields cognition consumes."""

    projected_rows: list[dict[str, Any]] = []
    for row in rows:
        projected_row = {
            "role": text_or_empty(row.get("role")),
            "display_name": text_or_empty(row.get("display_name")),
            "body_text": text_or_empty(row.get("body_text")),
            "timestamp": text_or_empty(row.get("timestamp")),
            "platform_message_id": text_or_empty(row.get("platform_message_id")),
            "platform_user_id": text_or_empty(row.get("platform_user_id")),
            "global_user_id": text_or_empty(row.get("global_user_id")),
            "addressed_to_global_user_ids": _list_field(
                row,
                "addressed_to_global_user_ids",
            ),
            "mentions": [],
            "broadcast": bool(row.get("broadcast")),
            "reply_context": {},
        }
        projected_rows.append(projected_row)
    return projected_rows


def _empty_rag_result() -> dict[str, Any]:
    """Build the minimum stable RAG projection for cognition tests."""

    rag_result = {
        "answer": "",
        "user_image": {"user_memory_context": empty_user_memory_context()},
        "character_image": {
            "self_image": {
                "milestones": [],
                "historical_summary": "",
                "recent_window": [],
            }
        },
        "third_party_profiles": [],
        "memory_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }
    return rag_result


def _load_character_profile(path: Path) -> dict[str, Any]:
    """Load the active character profile with required runtime defaults."""

    profile = load_personality(path)
    profile.setdefault("mood", "Neutral")
    profile.setdefault("global_vibe", "Calm")
    profile.setdefault("reflection_summary", "")
    return profile


def _project_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project a conversation row to bounded local artifact data."""

    projected_row = {
        "row_id": str(row.get("_id") or ""),
        "platform": text_or_empty(row.get("platform")),
        "platform_channel_id": text_or_empty(row.get("platform_channel_id")),
        "channel_type": text_or_empty(row.get("channel_type")),
        "platform_message_id": text_or_empty(row.get("platform_message_id")),
        "platform_user_id": text_or_empty(row.get("platform_user_id")),
        "global_user_id": text_or_empty(row.get("global_user_id")),
        "role": text_or_empty(row.get("role")),
        "display_name": text_or_empty(row.get("display_name")),
        "body_text": text_or_empty(row.get("body_text")),
        "timestamp": text_or_empty(row.get("timestamp")),
        "addressed_to_global_user_ids": _list_field(
            row,
            "addressed_to_global_user_ids",
        ),
        "broadcast": bool(row.get("broadcast")),
        "attachments": _project_attachments(row.get("attachments")),
    }
    return projected_row


def _project_attachments(value: object) -> list[dict[str, str]]:
    """Keep only prompt-safe attachment descriptions."""

    if not isinstance(value, list):
        return_value: list[dict[str, str]] = []
        return return_value
    attachments: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        description = text_or_empty(item.get("description"))
        media_type = text_or_empty(item.get("media_type"))
        attachment: dict[str, str] = {}
        if media_type:
            attachment["media_type"] = media_type
        if description:
            attachment["description"] = description
        if attachment:
            attachments.append(attachment)
    return attachments


def _list_field(row: dict[str, Any], field_name: str) -> list[Any]:
    """Read an optional list field from an external row."""

    value = row.get(field_name)
    if not isinstance(value, list):
        return_value: list[Any] = []
        return return_value
    return value


def _row_timestamp(row: dict[str, Any]) -> str:
    """Return a row timestamp or current UTC timestamp when absent."""

    timestamp = text_or_empty(row.get("timestamp"))
    if timestamp:
        return timestamp
    return_value = datetime.now(timezone.utc).isoformat()
    return return_value


def _case_set_document(
    args: argparse.Namespace,
    source_kind: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a case-set document for one QQ source kind."""

    document = {
        "schema_version": CASE_SET_SCHEMA_VERSION,
        "source_platform": args.platform,
        "source_platform_user_id": args.platform_user_id,
        "source_kind": source_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }
    return document


def _scope_discovery_document(
    args: argparse.Namespace,
    authored_rows: list[dict[str, Any]],
    scopes: list[dict[str, str]],
) -> dict[str, Any]:
    """Build the scope-discovery artifact."""

    document = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform,
        "platform_user_id": args.platform_user_id,
        "authored_row_count": len(authored_rows),
        "scopes": scopes,
        "authored_rows": authored_rows,
    }
    return document


def _window_document(
    args: argparse.Namespace,
    windows: list[dict[str, Any]],
    channel_type: str,
) -> dict[str, Any]:
    """Build a full-window artifact for one channel type."""

    selected_windows = [
        window for window in windows
        if window["scope"]["channel_type"] == channel_type
    ]
    document = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform,
        "platform_user_id": args.platform_user_id,
        "channel_type": channel_type,
        "windows": selected_windows,
    }
    return document


def _write_json(path: Path, document: dict[str, Any]) -> None:
    """Write a local JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the case capture command parser."""

    parser = argparse.ArgumentParser(
        description="Capture real QQ cases for cognition-stage connection tests.",
    )
    parser.add_argument("--platform", default="qq")
    parser.add_argument("--platform-user-id", required=True)
    parser.add_argument("--history-limit", type=int, default=500)
    parser.add_argument("--max-private", type=int, default=20)
    parser.add_argument("--max-group", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--personality-path",
        type=Path,
        default=DEFAULT_PERSONALITY_PATH,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    """CLI entrypoint for the case capture command."""

    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
