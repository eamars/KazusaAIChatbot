"""Capture frozen upstream states for L2d routing verification."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.db import get_conversation_history, get_user_profile
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l1 import (
    call_cognition_subconscious,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (
    call_boundary_core_agent,
    call_cognition_consciousness,
    call_judgment_core_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2c2 import (
    call_social_context_appraisal,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.self_cognition import models as self_models
from kazusa_ai_chatbot.self_cognition import projection as self_projection
from kazusa_ai_chatbot.self_cognition import runner as self_runner
from kazusa_ai_chatbot.self_cognition import sources as self_sources
from kazusa_ai_chatbot.time_context import build_character_time_context
from kazusa_ai_chatbot.utils import load_personality, text_or_empty

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE_DIR = ROOT / "test_artifacts" / "l2d_action_initializer"
DEFAULT_PERSONALITY_PATH = ROOT / "personalities" / "kazusa.json"
DEFAULT_PLATFORM = "qq"
DEFAULT_QQ_CHANNEL_ID = "673225019"
DEFAULT_CAPTURE_LIMIT = 1
DEFAULT_HISTORY_LIMIT = 300
RECENT_HISTORY_LIMIT = 8
L2D_ROUTING_CASE_SET_SCHEMA_VERSION = "l2d_routing_case_set.v1"
L2D_ROUTING_CASE_SCHEMA_VERSION = "l2d_routing_case.v1"


async def _run(args: argparse.Namespace) -> None:
    """Capture requested cases and append them to a private case-set file."""

    character_profile = _load_character_profile(args.personality_path)
    if args.source == "qq_history":
        cases = await _capture_qq_history_cases(args, character_profile)
    else:
        cases = await _capture_self_cognition_cases(args, character_profile)

    output_path = _output_path(args)
    written_count = _write_case_set(output_path, cases)
    summary = {
        "output_path": str(output_path),
        "captured_count": len(cases),
        "case_set_count": written_count,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


async def _capture_qq_history_cases(
    args: argparse.Namespace,
    character_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Capture frozen L2d inputs from QQ user-message and reply windows."""

    rows = await get_conversation_history(
        platform=args.platform,
        platform_channel_id=args.platform_channel_id,
        limit=args.history_limit,
    )
    pairs = _select_qq_pairs(rows)
    selected_pairs = pairs[args.offset: args.offset + args.max_cases]
    cases: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(selected_pairs, start=args.offset + 1):
        user_row = pair["user_row"]
        history_rows = pair["history_rows"]
        global_user_id = text_or_empty(user_row.get("global_user_id"))
        user_profile = await _load_user_profile(global_user_id)
        state = _build_qq_cognition_state(
            user_row,
            history_rows,
            character_profile=character_profile,
            user_profile=user_profile,
        )
        frozen_state = await _run_upstream_to_l2c(state)
        case = _build_qq_case(pair_index, pair, frozen_state)
        cases.append(case)
    return cases


async def _capture_self_cognition_cases(
    args: argparse.Namespace,
    character_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Capture frozen L2d inputs from real self-cognition source cases."""

    now = _capture_now(args)
    source_cases = await self_sources.collect_self_cognition_cases(
        now=now,
        character_profile=character_profile,
        max_cases=args.offset + args.max_cases,
    )
    selected_cases = source_cases[args.offset: args.offset + args.max_cases]
    cases: list[dict[str, Any]] = []
    for case_index, source_case in enumerate(
        selected_cases,
        start=args.offset + 1,
    ):
        legacy_artifacts = await self_runner.build_self_cognition_case_artifacts_async(
            source_case,
            apply_consolidation=False,
        )
        frozen_state = await _capture_self_upstream_state(
            source_case,
            legacy_artifacts,
        )
        case = _build_self_case(case_index, source_case, legacy_artifacts, frozen_state)
        cases.append(case)
    return cases


async def _load_user_profile(global_user_id: str) -> dict[str, Any]:
    """Load the user profile when a conversation row has a global user id."""

    if not global_user_id:
        return_value = _default_user_profile()
        return return_value
    profile = await get_user_profile(global_user_id)
    if not profile:
        profile = _default_user_profile()
    profile.setdefault("affinity", 500)
    profile.setdefault("active_commitments", [])
    profile.setdefault("facts", [])
    profile.setdefault("last_relationship_insight", "")
    return profile


def _load_character_profile(path: Path) -> dict[str, Any]:
    """Load the character profile with runtime fields cognition expects."""

    profile = load_personality(path)
    profile.setdefault("mood", "Neutral")
    profile.setdefault("global_vibe", "Calm")
    profile.setdefault("reflection_summary", "")
    return profile


def _select_qq_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select user-message rows followed immediately by assistant replies."""

    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:-1]):
        next_row = rows[index + 1]
        role = text_or_empty(row.get("role"))
        next_role = text_or_empty(next_row.get("role"))
        body_text = text_or_empty(row.get("body_text"))
        assistant_text = text_or_empty(next_row.get("body_text"))
        if role != "user" or next_role != "assistant":
            continue
        if not body_text or not assistant_text:
            continue
        start = max(0, index - RECENT_HISTORY_LIMIT)
        pair = {
            "user_row": row,
            "assistant_row": next_row,
            "history_rows": rows[start:index],
        }
        pairs.append(pair)
    return pairs


def _build_qq_cognition_state(
    user_row: dict[str, Any],
    history_rows: list[dict[str, Any]],
    *,
    character_profile: dict[str, Any],
    user_profile: dict[str, Any],
) -> dict[str, Any]:
    """Build a cognition state seed from one QQ conversation row."""

    timestamp = _row_timestamp(user_row)
    time_context = build_character_time_context(timestamp)
    body_text = text_or_empty(user_row.get("body_text"))
    platform = text_or_empty(user_row.get("platform")) or DEFAULT_PLATFORM
    platform_channel_id = (
        text_or_empty(user_row.get("platform_channel_id"))
        or DEFAULT_QQ_CHANNEL_ID
    )
    channel_type = text_or_empty(user_row.get("channel_type")) or "private"
    platform_message_id = text_or_empty(user_row.get("platform_message_id"))
    platform_user_id = text_or_empty(user_row.get("platform_user_id"))
    global_user_id = text_or_empty(user_row.get("global_user_id"))
    display_name = text_or_empty(user_row.get("display_name")) or "qq_user"
    conversation_row_id = text_or_empty(user_row.get("_id"))
    episode = build_text_chat_cognitive_episode(
        episode_id=f"l2d_capture:qq:{platform_message_id or conversation_row_id}",
        percept_id=f"l2d_capture:qq:percept:{platform_message_id or conversation_row_id}",
        timestamp=timestamp,
        time_context=time_context,
        user_input=body_text,
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type=channel_type,
        platform_message_id=platform_message_id,
        platform_user_id=platform_user_id,
        global_user_id=global_user_id,
        user_name=display_name,
        active_turn_platform_message_ids=[platform_message_id],
        active_turn_conversation_row_ids=[conversation_row_id],
        debug_modes={"no_visual_directives": True},
        output_mode="visible_reply",
        target_addressed_user_ids=[],
        target_broadcast=bool(user_row.get("broadcast")),
    )
    state = {
        "character_profile": character_profile,
        "timestamp": timestamp,
        "time_context": time_context,
        "user_input": body_text,
        "prompt_message_context": _prompt_message_context(user_row),
        "cognitive_episode": episode,
        "user_multimedia_input": [],
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "platform_message_id": platform_message_id,
        "platform_user_id": platform_user_id,
        "global_user_id": global_user_id,
        "user_name": display_name,
        "user_profile": user_profile,
        "platform_bot_id": "l2d_capture_bot",
        "chat_history_wide": _history_projection(history_rows),
        "chat_history_recent": _history_projection(history_rows),
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "conversation_progress": {},
        "promoted_reflection_context": {},
        "debug_modes": {"no_visual_directives": True},
        "should_respond": True,
        "decontexualized_input": body_text,
        "referents": [],
        "rag_result": _empty_rag_result(),
        "internal_monologue": "",
        "action_directives": {},
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "",
        "logical_stance": "",
        "mood": "",
        "global_vibe": "",
        "reflection_summary": "",
        "subjective_appraisals": [],
        "affinity_delta": 0,
        "last_relationship_insight": "",
        "new_facts": [],
        "future_promises": [],
        "mention_target_user": False,
    }
    return state


async def _capture_self_upstream_state(
    source_case: dict[str, Any],
    legacy_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Run the shared upstream cognition stack for a self-cognition case."""

    rag_output = legacy_artifacts.get(self_models.ARTIFACT_RAG_OUTPUT)
    if not isinstance(rag_output, dict):
        rag_output = None
    source_packet = self_projection.build_source_packet(
        source_case,
        rag_output=rag_output,
    )
    rendered_packet = self_projection.render_source_packet_text(source_packet)
    state = self_runner._build_cognition_state(
        source_case,
        rendered_packet,
        rag_output=rag_output,
    )
    frozen_state = await _run_upstream_to_l2c(state)
    return frozen_state


async def _run_upstream_to_l2c(state: dict[str, Any]) -> dict[str, Any]:
    """Run L1, L2a, L2b, and L2c once and return L2d-facing state."""

    working_state = dict(state)
    l1_output = await call_cognition_subconscious(working_state)
    working_state.update(l1_output)
    l2a_output = await call_cognition_consciousness(working_state)
    working_state.update(l2a_output)
    l2b_output = await call_boundary_core_agent(working_state)
    working_state.update(l2b_output)
    l2c_output = await call_judgment_core_agent(working_state)
    working_state.update(l2c_output)
    l2c2_output = await call_social_context_appraisal(working_state)
    working_state.update(l2c2_output)
    frozen_state = _freeze_l2d_state(working_state)
    return frozen_state


def _freeze_l2d_state(state: dict[str, Any]) -> dict[str, Any]:
    """Project the post-L2c state to fields consumed by L2d."""

    frozen_state = {
        "internal_monologue": state["internal_monologue"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "judgment_note": state["judgment_note"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
        "boundary_core_assessment": state["boundary_core_assessment"],
        "social_distance": state["social_distance"],
        "emotional_intensity": state["emotional_intensity"],
        "vibe_check": state["vibe_check"],
        "relational_dynamic": state["relational_dynamic"],
        "cognitive_episode": state["cognitive_episode"],
        "channel_type": state["channel_type"],
        "rag_result": state["rag_result"],
        "decontexualized_input": state["decontexualized_input"],
        "conversation_progress": state.get("conversation_progress"),
    }
    return frozen_state


def _build_qq_case(
    case_number: int,
    pair: dict[str, Any],
    frozen_state: dict[str, Any],
) -> dict[str, Any]:
    """Build a routing fixture from one QQ historical reply."""

    assistant_row = pair["assistant_row"]
    assistant_text = text_or_empty(assistant_row.get("body_text"))
    case = {
        "schema_version": L2D_ROUTING_CASE_SCHEMA_VERSION,
        "case_id": f"qq_{case_number:03d}",
        "source_kind": "qq_history",
        "frozen_l2d_state": frozen_state,
        "historical_comparison": {
            "comparison_kind": "assistant_reply",
            "past_route": "visible_reply",
            "assistant_timestamp": text_or_empty(assistant_row.get("timestamp")),
            "past_text": assistant_text,
        },
        "expectations": {
            "required_action_kinds": ["speak"],
            "required_visibility_by_kind": {"speak": "user_visible"},
            "forbidden_action_kinds": ["send_message"],
        },
    }
    return case


def _build_self_case(
    case_number: int,
    source_case: dict[str, Any],
    legacy_artifacts: dict[str, Any],
    frozen_state: dict[str, Any],
) -> dict[str, Any]:
    """Build a routing fixture from one self-cognition source case."""

    run_record = legacy_artifacts[self_models.ARTIFACT_RUN_RECORD]
    action_candidate = legacy_artifacts.get(self_models.ARTIFACT_ACTION_CANDIDATE)
    selected_route = text_or_empty(run_record.get("selected_route"))
    case = {
        "schema_version": L2D_ROUTING_CASE_SCHEMA_VERSION,
        "case_id": f"self_{case_number:03d}",
        "source_kind": "self_cognition",
        "frozen_l2d_state": frozen_state,
        "historical_comparison": {
            "comparison_kind": "self_cognition_route",
            "source_case_id": text_or_empty(source_case.get("case_id")),
            "source_case_name": text_or_empty(source_case.get("case_name")),
            "past_route": selected_route,
            "legacy_action_candidate": action_candidate,
        },
        "expectations": _self_expectations(legacy_artifacts),
    }
    return case


def _self_expectations(legacy_artifacts: dict[str, Any]) -> dict[str, Any]:
    """Map historical self-cognition route artifacts to routing expectations."""

    run_record = legacy_artifacts.get(self_models.ARTIFACT_RUN_RECORD)
    selected_route = ""
    if isinstance(run_record, dict):
        selected_route = text_or_empty(run_record.get("selected_route"))

    action_candidate = legacy_artifacts.get(self_models.ARTIFACT_ACTION_CANDIDATE)
    if (
        selected_route == self_models.ROUTE_ACTION_CANDIDATE
        or _legacy_action_candidate_has_text(action_candidate)
    ):
        expectations = {
            "required_action_kinds": ["speak"],
            "required_visibility_by_kind": {"speak": "user_visible"},
            "forbidden_action_kinds": ["send_message"],
        }
        return expectations

    if selected_route in (
        self_models.ROUTE_PROGRESS_MAINTENANCE,
        self_models.ROUTE_AUDIT_ONLY,
        self_models.ROUTE_SILENT_NO_WRITE,
    ):
        expectations = {
            "forbidden_action_kinds": ["send_message"],
            "forbidden_user_visible_kinds": ["speak", "send_message"],
        }
        return expectations

    expectations = {
        "forbidden_action_kinds": ["send_message"],
    }
    return expectations


def _legacy_action_candidate_has_text(value: object) -> bool:
    """Return whether the old route produced a user-visible message candidate."""

    if not isinstance(value, dict):
        return_value = False
        return return_value
    text = text_or_empty(value.get("text"))
    return_value = bool(text)
    return return_value


def _prompt_message_context(row: dict[str, Any]) -> dict[str, Any]:
    """Project a DB conversation row to prompt-message context shape."""

    context = {
        "body_text": text_or_empty(row.get("body_text")),
        "addressed_to_global_user_ids": _list_field(
            row,
            "addressed_to_global_user_ids",
        ),
        "broadcast": bool(row.get("broadcast")),
        "mentions": _list_field(row, "mentions"),
        "attachments": _list_field(row, "attachments"),
    }
    return context


def _history_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project conversation rows to the compact fields cognition reads."""

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
            "mentions": _list_field(row, "mentions"),
            "broadcast": bool(row.get("broadcast")),
            "reply_context": {},
        }
        projected_rows.append(projected_row)
    return projected_rows


def _empty_rag_result() -> dict[str, Any]:
    """Build the minimum RAG result shape needed by cognition and L2d."""

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


def _default_user_profile() -> dict[str, Any]:
    """Build a minimum profile for rows without profile data."""

    profile = {
        "affinity": 500,
        "active_commitments": [],
        "facts": [],
        "last_relationship_insight": "",
    }
    return profile


def _row_timestamp(row: dict[str, Any]) -> str:
    """Return row timestamp or current UTC timestamp when absent."""

    timestamp = text_or_empty(row.get("timestamp"))
    if timestamp:
        return timestamp
    return_value = datetime.now(timezone.utc).isoformat()
    return return_value


def _list_field(row: dict[str, Any], field_name: str) -> list[Any]:
    """Read an optional list field from a database row."""

    value = row.get(field_name)
    if not isinstance(value, list):
        return_value: list[Any] = []
        return return_value
    return value


def _capture_now(args: argparse.Namespace) -> datetime:
    """Resolve the self-cognition capture timestamp."""

    if args.now:
        now = datetime.fromisoformat(args.now)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now
    return_value = datetime.now(timezone.utc)
    return return_value


def _write_case_set(path: Path, cases: list[dict[str, Any]]) -> int:
    """Append captured cases to a private case-set JSON file by case id."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        document = json.loads(text)
    else:
        document = {
            "schema_version": L2D_ROUTING_CASE_SET_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cases": [],
        }
    existing_cases = document["cases"]
    case_by_id = {
        text_or_empty(existing_case.get("case_id")): existing_case
        for existing_case in existing_cases
        if isinstance(existing_case, dict)
    }
    for case in cases:
        case_by_id[case["case_id"]] = case
    merged_cases = [
        case_by_id[case_id]
        for case_id in sorted(case_by_id)
        if case_id
    ]
    document["cases"] = merged_cases
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    written_count = len(merged_cases)
    return written_count


def _output_path(args: argparse.Namespace) -> Path:
    """Return the destination case-set path for the selected source."""

    if args.output is not None:
        return args.output
    if args.source == "qq_history":
        path = DEFAULT_CAPTURE_DIR / "qq_673225019_cases.json"
        return path
    path = DEFAULT_CAPTURE_DIR / "self_cognition_cases.json"
    return path


def _build_parser() -> argparse.ArgumentParser:
    """Build the capture command parser."""

    parser = argparse.ArgumentParser(
        description="Capture private frozen upstream states for L2d tests.",
    )
    parser.add_argument(
        "--source",
        choices=("qq_history", "self_cognition"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--personality-path",
        type=Path,
        default=DEFAULT_PERSONALITY_PATH,
    )
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--platform-channel-id", default=DEFAULT_QQ_CHANNEL_ID)
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--max-cases", type=int, default=DEFAULT_CAPTURE_LIMIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--now", default="")
    return parser


def main() -> None:
    """Entrypoint for the frozen L2d case capture command."""

    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
