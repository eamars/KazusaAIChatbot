"""Real LLM review cases for the standalone local-context resolver."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from tests.cognition_core_v2_test_helpers import canonical_user_message_episode
from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CAPABILITY_REQUEST_VERSION,
)
from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    resolve_local_context,
    validate_local_context_resolution_packet,
)
from kazusa_ai_chatbot.local_context_resolver import stages as resolver_stages
from kazusa_ai_chatbot.time_boundary import build_turn_clock

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

RAW_EVIDENCE_DIR = Path("test_artifacts/local_context_resolver/raw")


async def test_standalone_current_time() -> None:
    """Review whether live local-time context becomes bounded evidence."""

    await _run_case(
        case_id="standalone_current_time",
        objective="Use local context to answer what date it is for the chat.",
        message_text="@active character what date is it there?",
        local_time_context={
            "local_date": "2026-07-04",
            "local_time": "09:30",
            "timezone": "Pacific/Auckland",
        },
        chat_history_recent=[],
        chat_history_wide=[],
    )


async def test_standalone_exact_phrase() -> None:
    """Review whether exact phrase evidence keeps the literal anchor."""

    await _run_case(
        case_id="standalone_exact_phrase",
        objective=(
            "Find who said the exact phrase 'blue comet marker' in recent "
            "conversation context."
        ),
        message_text="@active character who said 'blue comet marker'?",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[
            {
                "speaker": "Mika",
                "text": "I left a blue comet marker in the notes.",
                "local_time": "2026-07-04 09:10",
            },
            {
                "speaker": "Ren",
                "text": "That marker was easy to miss.",
                "local_time": "2026-07-04 09:12",
            },
        ],
        chat_history_wide=[],
    )


async def test_standalone_current_user_url() -> None:
    """Review whether URL recall preserves the current-user scope."""

    await _run_case(
        case_id="standalone_current_user_url",
        objective="Recall the URL the current user shared in recent context.",
        message_text="@active character what URL did I just share?",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[
            {
                "speaker": "operator",
                "text": "Keep this URL for the demo: https://example.test/napcat",
                "local_time": "2026-07-04 09:20",
            },
        ],
        chat_history_wide=[],
    )


async def test_standalone_scoped_memory() -> None:
    """Review whether scoped current-user memory remains scoped evidence."""

    await _run_case(
        case_id="standalone_scoped_memory",
        objective="Use current-user scoped memory to recall the user's preference.",
        message_text="@active character what do you remember about my tea?",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[],
        chat_history_wide=[
            {
                "source": "user_memory_units",
                "scope_global_user_id": "user-1",
                "summary": "The current user prefers jasmine tea without sugar.",
            },
        ],
    )


async def test_standalone_napcat_command_anchor() -> None:
    """Review whether #napcat remains the search anchor after direct address."""

    await _run_case(
        case_id="standalone_napcat_command_anchor",
        objective=(
            "Resolve how the active character should understand the #napcat "
            "command after being directly tagged."
        ),
        message_text="@active character #napcat",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[
            {
                "speaker": "other bot",
                "text": "NapCat status: running on an imaginary moon server.",
                "local_time": "2026-07-04 09:25",
            },
        ],
        chat_history_wide=[
            {
                "source": "memory",
                "name": "napcat",
                "summary": (
                    "#napcat is a playful group command. If the active "
                    "character is tagged with #napcat, she may invent a "
                    "playful status line rather than refusing for lack of "
                    "real runtime status."
                ),
            },
        ],
    )


async def test_production_current_time() -> None:
    """Review production local-context recall for current local time."""

    await _run_production_case(
        case_id="production_current_time",
        objective="Use local context to answer what time it is for the chat.",
        message_text="@active character what time is it there?",
        local_time_context={
            "local_date": "2026-07-04",
            "local_time": "09:30",
            "timezone": "Pacific/Auckland",
        },
        chat_history_recent=[],
        chat_history_wide=[],
        required_fragments=["09:30"],
        max_total_llm_calls=4,
    )


async def test_production_exact_phrase() -> None:
    """Review production local-context recall for exact phrase provenance."""

    await _run_production_case(
        case_id="production_exact_phrase",
        objective=(
            "Find who said the exact phrase 'blue comet marker' in recent "
            "conversation context."
        ),
        message_text="@active character who said 'blue comet marker'?",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[
            {
                "speaker": "Mika",
                "text": "I left a blue comet marker in the notes.",
                "local_time": "2026-07-04 09:10",
            },
            {
                "speaker": "Ren",
                "text": "That marker was easy to miss.",
                "local_time": "2026-07-04 09:12",
            },
        ],
        chat_history_wide=[],
        required_fragments=["blue comet marker"],
        max_total_llm_calls=5,
    )


async def test_production_current_user_url() -> None:
    """Review production local-context recall for current-user URL evidence."""

    await _run_production_case(
        case_id="production_current_user_url",
        objective="Recall the URL the current user shared in recent context.",
        message_text="@active character what URL did I just share?",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[
            {
                "speaker": "operator",
                "text": "Keep this URL for the demo: https://example.test/napcat",
                "local_time": "2026-07-04 09:20",
            },
        ],
        chat_history_wide=[],
        required_fragments=["https://example.test/napcat"],
        max_total_llm_calls=5,
    )


async def test_production_scoped_memory() -> None:
    """Review production local-context recall for scoped memory evidence."""

    await _run_production_case(
        case_id="production_scoped_memory",
        objective="Use current-user scoped memory to recall the user's preference.",
        message_text="@active character what do you remember about my tea?",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[],
        chat_history_wide=[
            {
                "source": "user_memory_units",
                "scope_global_user_id": "user-1",
                "summary": "The current user prefers jasmine tea without sugar.",
            },
        ],
        required_fragments=["jasmine tea"],
        max_total_llm_calls=5,
    )


async def test_production_napcat_command_anchor() -> None:
    """Review production local-context recall keeps #napcat as the anchor."""

    await _run_production_case(
        case_id="production_napcat_command_anchor",
        objective=(
            "Resolve how the active character should understand the #napcat "
            "command after being directly tagged."
        ),
        message_text="@active character #napcat",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[
            {
                "speaker": "other bot",
                "text": "NapCat status: running on an imaginary moon server.",
                "local_time": "2026-07-04 09:25",
            },
        ],
        chat_history_wide=[
            {
                "source": "memory",
                "name": "napcat",
                "summary": (
                    "#napcat is a playful group command. If the active "
                    "character is tagged with #napcat, she may invent a "
                    "playful status line rather than refusing for lack of "
                    "real runtime status."
                ),
            },
        ],
        required_fragments=["#napcat"],
        max_total_llm_calls=5,
    )


async def _run_case(
    *,
    case_id: str,
    objective: str,
    message_text: str,
    local_time_context: dict[str, object],
    chat_history_recent: list[dict[str, object]],
    chat_history_wide: list[dict[str, object]],
) -> None:
    """Run one live LLM review case and write raw evidence for inspection."""

    await _skip_if_llm_unavailable()
    request = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": objective,
        "source": "live_llm_review",
        "reason": f"live LLM review case {case_id}",
        "priority": "normal",
    }
    context = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
        "character_name": "active character",
        "platform": "debug",
        "platform_channel_id": "group-1",
        "global_user_id": "user-1",
        "user_name": "operator",
        "local_time_context": local_time_context,
        "prompt_message_context": {
            "message_text": message_text,
            "addressed_to_active_character": True,
        },
        "chat_history_recent": chat_history_recent,
        "chat_history_wide": chat_history_wide,
        "conversation_progress": {},
    }
    options = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        "max_iterations": 3,
        "max_nodes": 8,
        "max_depth": 3,
        "max_node_attempts": 2,
        "max_subagent_attempts": 1,
    }

    resolver_stages.drain_stage_trace_records()
    start = time.perf_counter()
    packet = await resolve_local_context(request, context, options)
    duration_seconds = time.perf_counter() - start
    stage_traces = resolver_stages.drain_stage_trace_records()
    validate_local_context_resolution_packet(packet)

    RAW_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_EVIDENCE_DIR / f"{case_id}.json"
    evidence = {
        "case_id": case_id,
        "request": request,
        "context": context,
        "options": options,
        "duration_seconds": duration_seconds,
        "stage_traces": stage_traces,
        "packet": packet,
    }
    raw_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert packet["schema_version"] == "local_context_resolution_packet.v1"
    assert isinstance(packet["rag_result"], dict)
    assert isinstance(packet["trace_summary"], dict)
    assert stage_traces
    assert all("raw_model_output" in trace for trace in stage_traces)
    assert all("parsed_output" in trace for trace in stage_traces)
    assert "graph" not in str(packet["rag_result"])
    print(f"LOCAL_CONTEXT_RESOLVER_LIVE case={case_id} raw={raw_path}")


async def _run_production_case(
    *,
    case_id: str,
    objective: str,
    message_text: str,
    local_time_context: dict[str, object],
    chat_history_recent: list[dict[str, object]],
    chat_history_wide: list[dict[str, object]],
    required_fragments: list[str],
    max_total_llm_calls: int,
) -> None:
    """Run one production-wired local-context recall case."""

    await _skip_if_llm_unavailable()
    state = _production_persona_state(
        message_text=message_text,
        local_time_context=local_time_context,
        chat_history_recent=chat_history_recent,
        chat_history_wide=chat_history_wide,
    )
    request = {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": "local_context_recall",
        "objective": objective,
        "reason": f"live LLM production review case {case_id}",
        "priority": "now",
    }

    resolver_stages.drain_stage_trace_records()
    start = time.perf_counter()
    observation = await capabilities.execute_resolver_capability_request(
        request,
        state,
    )
    duration_seconds = time.perf_counter() - start
    stage_traces = resolver_stages.drain_stage_trace_records()

    RAW_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_EVIDENCE_DIR / f"{case_id}.json"
    evidence = {
        "case_id": case_id,
        "request": request,
        "state": state,
        "duration_seconds": duration_seconds,
        "stage_traces": stage_traces,
        "observation": observation,
    }
    raw_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "local_context_recall"
    assert isinstance(observation["rag_result"], dict)
    assert stage_traces
    assert len(stage_traces) <= max_total_llm_calls
    rendered = json.dumps(observation["rag_result"], ensure_ascii=False)
    for fragment in required_fragments:
        assert fragment in rendered
    assert "graph" not in rendered
    assert "trace_summary" not in rendered
    print(f"LOCAL_CONTEXT_RESOLVER_LIVE case={case_id} raw={raw_path}")


def _production_persona_state(
    *,
    message_text: str,
    local_time_context: dict[str, object],
    chat_history_recent: list[dict[str, object]],
    chat_history_wide: list[dict[str, object]],
) -> dict[str, object]:
    """Build a production-like persona state for live RAG3 capability tests."""

    turn_clock = build_turn_clock("2026-07-04 09:30:00")
    effective_time_context = dict(turn_clock["local_time_context"])
    effective_time_context.update(local_time_context)
    episode = canonical_user_message_episode(
        episode_id="production-local-context-live-episode",
        percept_id="production-local-context-live-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=effective_time_context,
        user_input=message_text,
        platform="debug",
        platform_channel_id="group-1",
        channel_type="group",
        platform_message_id="production-local-context-live-message",
        platform_user_id="operator-platform-user",
        global_user_id="user-1",
        user_name="operator",
        active_turn_platform_message_ids=[
            "production-local-context-live-message",
        ],
        active_turn_conversation_row_ids=[
            "production-local-context-live-row",
        ],
        debug_modes={},
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    state = {
        "cognitive_episode": episode,
        "decontextualized_input": message_text,
        "referents": [],
        "character_profile": {
            "name": "active character",
            "global_user_id": "character-1",
        },
        "platform": "debug",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "platform_message_id": "production-local-context-live-message",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "operator",
        "user_profile": {},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": effective_time_context,
        "prompt_message_context": {
            "body_text": message_text,
            "mentions": [{
                "display_name": "active character",
                "global_user_id": "character-1",
            }],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "live RAG3 review",
        "chat_history_recent": chat_history_recent,
        "chat_history_wide": chat_history_wide,
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": {},
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
    }
    return state


async def _skip_if_llm_unavailable() -> None:
    """Skip one live test when the configured RAG planner endpoint is down."""

    from kazusa_ai_chatbot.config import RAG_PLANNER_LLM_BASE_URL

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{RAG_PLANNER_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {exc}")
    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned status {response.status_code}: "
            f"{RAG_PLANNER_LLM_BASE_URL}"
        )
