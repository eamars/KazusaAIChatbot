"""Serial real-LLM quality cases for conversation progress V2."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import OpenAIError

from kazusa_ai_chatbot import llm_tracing as llm_tracing_module
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    DialogResponseOperation,
)
from kazusa_ai_chatbot.cognition_core_v2 import facade
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT_CAP,
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_text_surface_planning,
)
from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.conversation_progress.compaction import (
    validate_block,
)
from kazusa_ai_chatbot.conversation_progress.history import (
    assemble_logical_turns,
    logical_turn_source_refs,
)
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationEpisodeBlockV1,
    ConversationLogicalTurnV1,
    ConversationProgressRecordInput,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_ACTIVE_BLOCK_REFS,
    MAX_ACTIVE_EVENTS,
    MAX_CONTINUATION_CHARS,
    MAX_PROGRESS_EVIDENCE_CHARS,
    MAX_PROGRESS_SCENE_CHARS,
    MAX_RECENT_TURN_REFS,
    prune_aged_progress_packet,
)
from kazusa_ai_chatbot.conversation_progress.projection import (
    build_progress_prompt,
    continuation_projection_chars,
    filter_group_scene_ambient_turns,
)
from kazusa_ai_chatbot.conversation_progress.repository import (
    PreparedProgressWrite,
    prepare_progress_write,
    validate_active_packet,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_msg_decontextualizer as decontextualizer_module,
)
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_generator
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    _build_text_surface_services,
)
from tests.cognition_core_v2_test_helpers import (
    canonical_character_identity,
    canonical_episode,
    canonical_identity_context,
)
from tests.conversation_progress_v2_helpers import (
    SCOPE,
    event,
    logical_turn,
    packet,
    record_input,
)
from tests.conversation_progress_v2_simulation import (
    LongThreadSimulation,
    simulate_long_thread,
)
from tests.fixtures.conversation_progress_v2_asuna_houjing_regression import (
    BODY,
    BOT_DISPLAY_NAME,
    BOT_PLATFORM_USER_ID,
    CHANNEL_ID,
    CURRENT_TURN_TIMESTAMP,
    PLATFORM,
    TRACE_5,
    USER_A_DISPLAY_NAME,
    USER_A_GLOBAL_USER_ID,
    build_adjacent_history,
)
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.live_llm, pytest.mark.asyncio]

_COGNITION_NOW = "2026-07-28T09:30:00Z"
_BASE_TIME = datetime(2026, 7, 28, 8, 0, tzinfo=timezone.utc)
_FULL_FIXTURE_PATH = Path(
    "tests/fixtures/conversation_progress_v2/"
    "asuna_houjing_long_thread.json"
)


class _CapturingLLM:
    """Delegate to a configured real model while preserving full calls."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> Any:
        captured_call = {
            "stage_name": str(getattr(config, "stage_name", "")),
            "route_name": str(getattr(config, "route_name", "")),
            "system_prompt": str(getattr(messages[0], "content", "")),
            "human_messages": [
                str(getattr(message, "content", ""))
                for message in messages[1:]
            ],
            "raw_output": "",
            "usage_metadata": None,
            "response_metadata": None,
            "provider_error": "",
        }
        self.calls.append(captured_call)
        try:
            response = await self.delegate.ainvoke(messages, config=config)
        except (
            OpenAIError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            captured_call["provider_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            raise
        captured_call["raw_output"] = str(
            getattr(response, "content", "")
        )
        captured_call["usage_metadata"] = getattr(
            response,
            "usage_metadata",
            None,
        )
        captured_call["response_metadata"] = getattr(
            response,
            "response_metadata",
            None,
        )
        return response


class _GoalPromptBoundaryReached(AssertionError):
    """Stop before downstream goal semantics are produced."""


class _GoalPromptCapture:
    """Capture the serialized goal input without invoking its model."""

    def __init__(self) -> None:
        self.system_prompt = ""
        self.human_payload = ""
        self.payload: dict[str, Any] = {}

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del config
        self.system_prompt = str(getattr(messages[0], "content", ""))
        self.human_payload = str(getattr(messages[-1], "content", ""))
        self.payload = json.loads(self.human_payload)
        raise _GoalPromptBoundaryReached


def _character_profile() -> dict[str, Any]:
    """Build a valid character identity for the live semantic handoff."""

    profile = canonical_character_identity(marker="progress-live")
    profile["personality_brief"] = {
        "mbti": "test",
        "logic": "Preserve chronology and advance from grounded events.",
        "tempo": "measured",
        "defense": "Use direct character judgment without resetting context.",
        "quirks": "Prefer one concrete next move.",
        "taboos": "Avoid treating completed interaction as newly proposed.",
    }
    return profile


async def _skip_if_routes_unavailable(
    *,
    recorder_only: bool = False,
) -> None:
    """Skip only when a configured real-model endpoint cannot be reached."""

    base_urls = {
        str(recorder._scene_recorder_llm_config.base_url).rstrip("/"),
        str(recorder._event_recorder_llm_config.base_url).rstrip("/"),
    }
    if not recorder_only:
        core_services = build_cognition_core_services()
        surface_services = _build_text_surface_services()
        base_urls.update({
            str(
                core_services.goal_ordinary_response_config.base_url
            ).rstrip("/"),
            str(surface_services.content_plan_config.base_url).rstrip("/"),
            str(
                dialog_module._dialog_generator_llm_config.base_url
            ).rstrip("/"),
        })
    async with httpx.AsyncClient(timeout=4.0) as client:
        for base_url in sorted(base_urls):
            try:
                response = await client.get(f"{base_url}/models")
            except httpx.HTTPError as exc:
                pytest.skip(
                    f"LLM endpoint is unavailable: {base_url}: {exc}"
                )
            if response.status_code >= 500:
                pytest.skip(
                    f"LLM endpoint returned {response.status_code}: "
                    f"{base_url}"
                )


async def _skip_if_decontextualizer_route_unavailable() -> None:
    """Skip the focused Stage-0 case when its live endpoint is unavailable."""

    base_url = str(
        decontextualizer_module._msg_decontextualizer_llm_config.base_url
    ).rstrip('/')
    async with httpx.AsyncClient(timeout=4.0) as client:
        try:
            response = await client.get(f'{base_url}/models')
        except httpx.HTTPError as exc:
            pytest.skip(
                f'decontextualizer endpoint is unavailable: {base_url}: '
                f'{exc}'
            )
    if response.status_code >= 500:
        pytest.skip(
            f'decontextualizer endpoint returned {response.status_code}: '
            f'{base_url}'
        )


def _scope_record_input(
    *,
    prior_packet: ConversationProgressStateV2 | None,
    interaction_turns: list[ConversationLogicalTurnV1],
    current_sources: list[dict[str, str]],
    current_input: str,
    final_dialog: list[str],
) -> ConversationProgressRecordInput:
    """Build one exact settled-turn record input for a live call."""

    current = record_input(prior_packet=prior_packet)
    current["storage_timestamp_utc"] = interaction_turns[-1]["occurred_at"]
    current["interaction_logical_turns"] = deepcopy(interaction_turns)
    current["current_turn_source_refs"] = deepcopy(current_sources)
    current["decontextualized_input"] = current_input
    current["content_plan"] = {
        "semantic_content": (
            "Respect established event state and choose a coherent next step."
        ),
        "surface_intent": "respond",
    }
    current["final_dialog"] = list(final_dialog)
    return current


def _fixture_participant_turns() -> list[ConversationLogicalTurnV1]:
    """Apply the participant lane to the source-faithful group fixture."""

    participant_rows = [
        row for row in build_adjacent_history()
        if (
            row["role"] == "user"
            and row["global_user_id"] == USER_A_GLOBAL_USER_ID
        ) or (
            row["role"] == "assistant"
            and row["platform_user_id"] == BOT_PLATFORM_USER_ID
            and USER_A_GLOBAL_USER_ID
            in row["addressed_to_global_user_ids"]
        )
    ]
    return assemble_logical_turns(
        rows=participant_rows,
        excluded_row_ids=[],
    )


def _full_fixture_participant_turns() -> list[ConversationLogicalTurnV1]:
    """Load the full redacted thread through the participant lane."""

    fixture = json.loads(
        _FULL_FIXTURE_PATH.read_text(encoding="utf-8-sig")
    )
    rows = fixture["messages"]
    scope = fixture["scope"]
    participant_rows = [
        row for row in rows
        if (
            row["role"] == "user"
            and row["global_user_id"]
            == scope["participant_global_user_id"]
        ) or (
            row["role"] == "assistant"
            and row["platform_user_id"]
            == scope["bot_platform_user_id"]
            and scope["participant_global_user_id"]
            in row["addressed_to_global_user_ids"]
        )
    ]
    return assemble_logical_turns(
        rows=participant_rows,
        excluded_row_ids=[],
    )


def _bounded_event_sources(
    turns: list[ConversationLogicalTurnV1],
) -> list[dict[str, str]]:
    """Select four source-faithful aliases from complete logical turns."""

    sources = logical_turn_source_refs(turns)
    row_sources = [
        source for source in sources
        if source["ref_kind"] == "conversation_row"
    ]
    trace_sources = [
        source for source in sources
        if source["ref_kind"] == "llm_trace"
    ]
    selected = [*row_sources[-3:], *trace_sources[-1:]]
    if not selected:
        raise AssertionError("event source selection is empty")
    return selected


def _capacity_turn(turn_number: int) -> ConversationLogicalTurnV1:
    """Build the next exact logical turn for a capacity checkpoint."""

    timestamp = (
        _BASE_TIME + timedelta(minutes=turn_number)
    ).isoformat()
    row_id = f"live-capacity-row-{turn_number:03d}"
    return {
        "turn_id": f"row:{row_id}",
        "role": "user",
        "occurred_at": timestamp,
        "display_name": "Capacity User",
        "fragments": [
            (
                "Continue from the established thread while respecting "
                "the completed prior action."
            )
        ],
        "conversation_row_ids": [row_id],
        "llm_trace_id": "",
        "platform_user_id": "live-capacity-user",
        "global_user_id": SCOPE.global_user_id,
        "addressed_to_global_user_ids": [],
        "broadcast": False,
        "reply_context": {},
    }


async def _invoke_recorder(
    monkeypatch: pytest.MonkeyPatch,
    current: ConversationProgressRecordInput,
    *,
    active_blocks: list[ConversationEpisodeBlockV1] | None = None,
) -> tuple[
    recorder.RecorderInvocationResult,
    PreparedProgressWrite,
    list[dict[str, object]],
]:
    """Call the configured real recorder and prepare its canonical write."""

    scene_delegate = recorder._scene_recorder_llm
    while isinstance(scene_delegate, _CapturingLLM):
        scene_delegate = scene_delegate.delegate
    event_delegate = recorder._event_recorder_llm
    while isinstance(event_delegate, _CapturingLLM):
        event_delegate = event_delegate.delegate
    scene_capture = _CapturingLLM(scene_delegate)
    event_capture = _CapturingLLM(event_delegate)
    monkeypatch.setattr(
        recorder,
        "_scene_recorder_llm",
        scene_capture,
    )
    monkeypatch.setattr(
        recorder,
        "_event_recorder_llm",
        event_capture,
    )
    try:
        invocation = await recorder.record_with_llm(current)
    except recorder.ConversationProgressRecorderOutputError as exc:
        failure_path = write_llm_trace(
            "conversation_progress_v2_live_llm",
            "recorder_one_call_output_failure",
            {
                "record_input": {
                    **current,
                    "scope": asdict(current["scope"]),
                },
                "recorder_model_calls": [
                    *(
                        {"owner": "scene", **call}
                        for call in scene_capture.calls
                    ),
                    *(
                        {"owner": "event", **call}
                        for call in event_capture.calls
                    ),
                ],
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        print(json.dumps({
            "failure_trace_path": str(failure_path),
            "error": str(exc),
        }, ensure_ascii=True, indent=2))
        raise
    recorder_calls = [
        *(
            {"owner": "scene", **call}
            for call in scene_capture.calls
        ),
        *(
            {"owner": "event", **call}
            for call in event_capture.calls
        ),
    ]
    if len(scene_capture.calls) != 1 or len(event_capture.calls) != 1:
        raise AssertionError(
            'eligible recorder update must make exactly two specialist calls'
        )
    if (
        invocation.recorder_call_count != 2
        or invocation.scene_attempt_count != 1
        or invocation.event_attempt_count != 1
    ):
        raise AssertionError(
            'recorder specialist attempt telemetry is inconsistent'
        )
    prepared = prepare_progress_write(
        record_input=current,
        delta=invocation.delta,
        active_blocks=active_blocks or [],
    )
    validate_active_packet(prepared.packet)
    if prepared.block is not None:
        validate_block(prepared.block)
    return invocation, prepared, recorder_calls


def _recorder_telemetry(
    invocation: recorder.RecorderInvocationResult,
) -> dict[str, object]:
    """Project exact two-owner call, payload, and disposition telemetry."""

    return {
        "recorder_call_count": invocation.recorder_call_count,
        "scene_attempt_count": invocation.scene_attempt_count,
        "event_attempt_count": invocation.event_attempt_count,
        "scene_disposition": invocation.scene_disposition,
        "event_disposition": invocation.event_disposition,
        "scene_human_payload_chars": (
            invocation.scene_human_payload_chars
        ),
        "event_human_payload_chars": (
            invocation.event_human_payload_chars
        ),
    }


async def _semantic_handoff(
    *,
    monkeypatch: pytest.MonkeyPatch,
    active_packet: ConversationProgressStateV2,
    interaction_turns: list[ConversationLogicalTurnV1],
    current_input: str,
    target_event_id: str,
    case_id: str,
    response_operation: DialogResponseOperation | None = None,
) -> dict[str, object]:
    """Run live goal, text-surface, and dialog boundaries from progress."""

    async def _record_contract_event(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        llm_tracing_module,
        "record_llm_trace_step",
        _record_contract_event,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_model_contract_event",
        _record_contract_event,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_llm_stage_event",
        _record_contract_event,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_dialog_quality_event",
        _record_contract_event,
    )
    progress_prompt = build_progress_prompt(
        active_packet=active_packet,
        interaction_logical_turns=interaction_turns[-10:],
    )
    episode = canonical_episode(
        episode_id=f"progress-live-{case_id}",
        content=current_input,
        current_global_user_id=active_packet["global_user_id"],
        metadata=(
            {
                "role_explicit_content": current_input,
                "response_operation": response_operation,
            }
            if response_operation is not None
            else None
        ),
    )
    cognition_input = build_cognition_input_from_global_state(
        {
            "cognitive_episode": episode,
            "global_user_id": active_packet["global_user_id"],
            "user_input": current_input,
            "decontextualized_input": current_input,
            "conversation_episode_state": active_packet,
            "conversation_progress": progress_prompt,
            "public_group_scene": "",
            "user_multimedia_input": [],
            "rag_result": {"memory_evidence": []},
            "character_profile": _character_profile(),
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id=active_packet["global_user_id"],
            updated_at=_COGNITION_NOW,
        ),
        character_state=build_character_production_state(
            updated_at=_COGNITION_NOW,
        ),
    )
    target_rows = [
        row for row in cognition_input["evidence"]
        if row["evidence_ref"]["source_id"] == (
            f"conversation-progress-event:{target_event_id}"
        )
    ]
    if len(target_rows) != 1:
        raise AssertionError("target progress event did not reach cognition")
    target_handle = target_rows[0]["evidence_handle"]
    projection = project_state_for_prompt(
        cognition_input["mutable_state"],
        character_constraints=cognition_input["character_constraints"],
        character_identity_context=cognition_input.get(
            "character_identity_context",
            canonical_identity_context(),
        ),
        evidence=cognition_input["evidence"],
    )
    branch_context = facade._branch_context(
        projection,
        cognition_input["mutable_state"],
        cognition_input["evidence"],
        scene_context=cognition_input["scene_context"],
        private_continuity_context=(
            cognition_input["private_continuity_context"]
        ),
    )

    core_services = build_cognition_core_services()
    goal_capture = _CapturingLLM(core_services.llm)
    core_services = replace(core_services, llm=goal_capture)
    try:
        selected_bid = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary-response",
            },
            branch_context,
            cognition_input["evidence"],
            core_services,
        )
    except CognitionExecutionError as exc:
        failure_path = write_llm_trace(
            "conversation_progress_v2_live_llm",
            f"{case_id}_goal_cognition_failure",
            {
                "current_input": current_input,
                "target_event_id": target_event_id,
                "target_evidence_handle": target_handle,
                "available_evidence": cognition_input["evidence"],
                "goal_model_calls": goal_capture.calls,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        print(json.dumps({
            "failure_trace_path": str(failure_path),
            "error": str(exc),
        }, ensure_ascii=True, indent=2))
        raise

    surface_input = {
        "schema_version": "text_surface_input.v2",
        "episode": episode,
        "intention": {
            "route": "speech",
            "intention": selected_bid["intention"],
            "target_roles": list(selected_bid["target_roles"]),
            "reason": selected_bid["reason"],
        },
        "goal_resolution": "answerable_now",
        "primary_bid": {
            "motive": selected_bid["branch_id"],
            "intention": selected_bid["intention"],
            "desired_outcome": selected_bid["desired_outcome"],
            "permitted_detail": selected_bid["concrete_detail"],
            "target_summaries": [],
            "expected_consequences": list(
                selected_bid["expected_consequences"]
            ),
        },
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "engaged",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": (
            "Respond naturally while preserving established chronology."
        ),
        "character_expression_context": {
            "tempo": "measured",
            "linguistic_texture": "concise, vivid, and character-grounded",
        },
        "visual_character_context": "engaged and attentive",
    }
    surface_services = _build_text_surface_services()
    surface_capture = _CapturingLLM(surface_services.llm)
    surface_services = replace(surface_services, llm=surface_capture)
    surface_output = await run_text_surface_planning(
        surface_input,
        surface_services,
    )

    dialog_capture = _CapturingLLM(dialog_module._dialog_generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_generator_llm",
        dialog_capture,
    )
    dialog_result = await dialog_generator({
        "dialog_usage_mode": "live_progress_quality",
        "text_surface_input_v2": surface_input,
        "text_surface_output_v2": surface_output,
        "cognitive_episode": episode,
        "user_name": "participant",
        "llm_trace_id": f"progress-live-{case_id}",
    })
    final_dialog = dialog_result["final_dialog"]
    if not final_dialog:
        raise AssertionError("live dialog boundary returned no visible text")
    return {
        "progress_prompt": progress_prompt,
        "cognition_input": cognition_input,
        "target_evidence_handle": target_handle,
        "selected_bid": selected_bid,
        "goal_model_calls": goal_capture.calls,
        "surface_input": surface_input,
        "surface_output": surface_output,
        "surface_model_calls": surface_capture.calls,
        "dialog_model_calls": dialog_capture.calls,
        "final_dialog": final_dialog,
    }


def _block_graph(
    *,
    prior_simulation: LongThreadSimulation | None,
    prepared: PreparedProgressWrite,
) -> dict[str, ConversationEpisodeBlockV1]:
    """Merge prior and current blocks and apply mutable supersession links."""

    blocks = (
        deepcopy(prior_simulation.blocks)
        if prior_simulation is not None
        else {}
    )
    if prepared.block is not None:
        blocks[prepared.block["block_id"]] = deepcopy(prepared.block)
        for source_id in prepared.source_block_ids:
            blocks[source_id]["superseded_by_block_id"] = (
                prepared.block["block_id"]
            )
    for block in blocks.values():
        validate_block(block)
    return blocks


def _reachable_block_ids(
    packet: ConversationProgressStateV2,
    blocks: dict[str, ConversationEpisodeBlockV1],
) -> set[str]:
    """Resolve every transitive immutable block from active packet refs."""

    reachable: set[str] = set()
    pending = list(packet["compacted_block_refs"])
    while pending:
        block_id = pending.pop()
        if block_id in reachable:
            continue
        block = blocks[block_id]
        reachable.add(block_id)
        pending.extend(block["source_block_ids"])
    return reachable


def _critical_event(packet_value: ConversationProgressStateV2) -> dict:
    """Return the sole decision-critical event in capacity fixtures."""

    critical = [
        row for row in packet_value["events"]
        if row["retention"] == "decision_critical"
    ]
    if len(critical) != 1:
        raise AssertionError("capacity fixture lost its critical event")
    return critical[0]


def _write_case_trace(
    *,
    test_name: str,
    case_id: str,
    current: ConversationProgressRecordInput,
    invocation: recorder.RecorderInvocationResult,
    prepared: PreparedProgressWrite,
    recorder_calls: list[dict[str, object]],
    blocks: dict[str, ConversationEpisodeBlockV1],
    semantic_handoff: dict[str, object],
    hard_gates: dict[str, object],
    recorder_checkpoints: list[dict[str, object]] | None = None,
) -> None:
    """Write the complete review artifact and print its stable path."""

    scene_chars, evidence_chars = continuation_projection_chars(
        semantic_handoff["progress_prompt"],
        prepared.packet["updated_at"],
    )
    packet_event_ids = {
        row["event_id"] for row in prepared.packet["events"]
    }
    prompt_event_ids = {
        row["event_id"]
        for row in semantic_handoff["progress_prompt"]["events"]
    }
    trace_path = write_llm_trace(
        test_name,
        case_id,
        {
            "source_logical_turns": current["interaction_logical_turns"],
            "record_input": {
                **current,
                "scope": asdict(current["scope"]),
            },
            "prior_packet": current["prior_episode_state"],
            "recorder_model_calls": recorder_calls,
            "recorder_delta": invocation.delta,
            "recorder_telemetry": _recorder_telemetry(invocation),
            "provider_usage": invocation.provider_usage,
            "bound_normalizations": invocation.bound_normalizations,
            "recorder_checkpoints": recorder_checkpoints or [],
            "active_packet_after_update": prepared.packet,
            "inserted_or_merged_blocks": list(blocks.values()),
            "projection_selection": {
                "selected_event_ids": sorted(prompt_event_ids),
                "evicted_event_ids": sorted(
                    packet_event_ids - prompt_event_ids
                ),
                "scene_chars": scene_chars,
                "evidence_chars": evidence_chars,
                "combined_chars": scene_chars + evidence_chars,
            },
            "cognition_evidence_handles": [
                {
                    "handle": row["evidence_handle"],
                    "source": row["evidence_ref"],
                    "semantic_text": row["semantic_text"],
                }
                for row in semantic_handoff["cognition_input"]["evidence"]
                if row["evidence_ref"]["source_kind"]
                == "conversation_evidence"
            ],
            "target_evidence_handle": (
                semantic_handoff["target_evidence_handle"]
            ),
            "selected_cognition_bid": semantic_handoff["selected_bid"],
            "goal_model_calls": semantic_handoff["goal_model_calls"],
            "surface_input": semantic_handoff["surface_input"],
            "surface_output": semantic_handoff["surface_output"],
            "surface_model_calls": (
                semantic_handoff["surface_model_calls"]
            ),
            "dialog_model_calls": semantic_handoff["dialog_model_calls"],
            "final_dialog": semantic_handoff["final_dialog"],
            "hard_gates": hard_gates,
            "agent_quality_judgment": {
                "status": "pending_parent_inspection",
                "pass_reason": (
                    "Harness and contracts passed; parent must inspect "
                    "semantic progression and final dialog."
                ),
            },
        },
    )
    print(json.dumps({
        "case_id": case_id,
        "trace_path": str(trace_path),
        "recorder_telemetry": _recorder_telemetry(invocation),
        "turn_count": prepared.packet["turn_count"],
        "active_events": len(prepared.packet["events"]),
        "active_block_refs": len(prepared.packet["compacted_block_refs"]),
        "final_dialog": semantic_handoff["final_dialog"],
    }, ensure_ascii=True, indent=2))


async def test_live_original_failure_progress_semantic_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the original failure semantics reach serialized cognition input."""

    await _skip_if_routes_unavailable(recorder_only=True)
    turns = _full_fixture_participant_turns()
    if len(turns) != 23:
        raise AssertionError("source-faithful participant lane changed")

    active_packet: ConversationProgressStateV2 | None = None
    recorder_checkpoints: list[dict[str, object]] = []
    for checkpoint_index, (user_index, response_index) in enumerate(
        ((6, 7), (10, 11)),
        start=1,
    ):
        replay_user_turn = turns[user_index]
        replay_response_turn = turns[response_index]
        replay_current = _scope_record_input(
            prior_packet=active_packet,
            interaction_turns=turns[
                max(0, response_index - 9):response_index + 1
            ],
            current_sources=logical_turn_source_refs([
                replay_user_turn,
                replay_response_turn,
            ]),
            current_input=replay_user_turn["fragments"][0],
            final_dialog=list(replay_response_turn["fragments"]),
        )
        if user_index == 10:
            replay_current["decontextualized_input"] = (
                '当前用户已经完成对'
                f'{BOT_DISPLAY_NAME}后颈和肩膀的按摩，并询问她选择'
                '另一个接下来可触摸的部位。'
            )
        replay_current["scope"] = replay_current["scope"].__class__(
            platform=PLATFORM,
            platform_channel_id=CHANNEL_ID,
            global_user_id=USER_A_GLOBAL_USER_ID,
        )
        replay_current["character_name"] = BOT_DISPLAY_NAME
        replay_current["content_plan"] = {
            "semantic_content": (
                "Record the accepted interaction and preserve its chronology."
            ),
            "surface_intent": "respond",
        }
        invocation, prepared, recorder_calls = await _invoke_recorder(
            monkeypatch,
            replay_current,
        )
        recorder_checkpoints.append({
            "checkpoint_index": checkpoint_index,
            "participant_turn_indexes": [user_index, response_index],
            "record_input": {
                **replay_current,
                "scope": asdict(replay_current["scope"]),
            },
            "recorder_model_calls": recorder_calls,
            "recorder_delta": invocation.delta,
            "recorder_telemetry": _recorder_telemetry(invocation),
            "active_packet_after_update": prepared.packet,
        })
        active_packet = prepared.packet

    if active_packet is None:
        raise AssertionError("original-failure replay produced no packet")
    replay_trace_path = write_llm_trace(
        "test_live_original_failure_progress_semantic_handoff",
        "original_failure_recorder_replay",
        {
            "source_logical_turns": turns,
            "recorder_checkpoints": recorder_checkpoints,
            "active_packet": active_packet,
        },
    )

    completed_candidates = [
        row for row in active_packet["events"]
        if (
            row["state"] == "completed"
            and row["retention"] == "decision_critical"
        )
    ]
    target_events: list[dict[str, Any]] = []
    neck_markers = ('后颈', '颈部', '脖颈', '颈')
    shoulder_markers = ('肩膀', '肩部', '肩')
    massage_markers = ('按摩', '按摸', '揉', '按压')
    for row in completed_candidates:
        semantic_fields = " ".join([
            row["semantic_summary"],
            row["action"],
            row["object"],
            row["outcome"],
        ])
        if (
            any(marker in semantic_fields for marker in neck_markers)
            and any(
                marker in semantic_fields for marker in shoulder_markers
            )
            and any(
                marker in semantic_fields for marker in massage_markers
            )
        ):
            target_events.append(row)
    if len(target_events) != 1:
        pytest.fail(
            "real recorder did not produce one completed decision-critical "
            "neck/shoulder massage event; inspect "
            f"{replay_trace_path}"
        )
    completed_event = target_events[0]

    actor = completed_event["actor"]
    beneficiary = completed_event["beneficiary"]
    outcome = completed_event["outcome"]
    if not (
        (
            USER_A_DISPLAY_NAME in actor
            or '蚝爹油' in actor
            or '用户' in actor
        )
        and BOT_DISPLAY_NAME not in actor
    ):
        pytest.fail(
            "completed event does not identify the current user as actor; "
            f"inspect {replay_trace_path}"
        )
    if BOT_DISPLAY_NAME not in beneficiary:
        pytest.fail(
            "completed event does not identify the character as beneficiary; "
            f"inspect {replay_trace_path}"
        )
    outcome_markers = ('合格', '及格', '放松', '舒服', '接受', '认可', '评价')
    if not any(marker in outcome for marker in outcome_markers):
        pytest.fail(
            "completed event does not preserve acceptance or evaluation; "
            f"inspect {replay_trace_path}"
        )
    action_sources = logical_turn_source_refs([turns[6], turns[7]])
    has_accepted_response_lineage = any(
        source["ref_kind"] == "llm_trace"
        for source in completed_event["source_refs"]
    )
    if (
        action_sources[0] not in completed_event["source_refs"]
        or not has_accepted_response_lineage
    ):
        pytest.fail(
            "completed event lost user-action or accepted-response lineage; "
            f"inspect {replay_trace_path}"
        )

    final_user_turn = turns[-1]
    progress_prompt = build_progress_prompt(
        active_packet=active_packet,
        interaction_logical_turns=turns[-11:-1],
    )
    current_transition = (
        "The current user reports that the earlobe massage is complete and "
        "asks the current character to choose the next touch location."
    )
    episode = canonical_episode(
        episode_id="progress-live-original-failure-handoff",
        content=final_user_turn["fragments"][0],
        current_global_user_id=USER_A_GLOBAL_USER_ID,
        metadata={
            "role_explicit_content": current_transition,
            "response_operation": {
                "operation": (
                    "The current character chooses and states one distinct "
                    "location that the current user may touch next."
                ),
                "response_owner_role": CURRENT_CHARACTER_ROLE,
                "selection_owner_role": CURRENT_CHARACTER_ROLE,
                "selection_required": True,
                "embedded_actor_role": CURRENT_USER_ROLE,
                "embedded_target_role": CURRENT_CHARACTER_ROLE,
            },
        },
    )
    cognition_input = build_cognition_input_from_global_state(
        {
            "cognitive_episode": episode,
            "global_user_id": USER_A_GLOBAL_USER_ID,
            "user_input": final_user_turn["fragments"][0],
            "decontextualized_input": final_user_turn["fragments"][0],
            "conversation_episode_state": active_packet,
            "conversation_progress": progress_prompt,
            "user_multimedia_input": [],
            "rag_result": {"memory_evidence": []},
            "character_profile": _character_profile(),
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id=USER_A_GLOBAL_USER_ID,
            updated_at=_COGNITION_NOW,
        ),
        character_state=build_character_production_state(
            updated_at=_COGNITION_NOW,
        ),
    )
    progress_rows = [
        row for row in cognition_input["evidence"]
        if row["evidence_ref"]["source_id"] == (
            "conversation-progress-event:"
            f"{completed_event['event_id']}"
        )
    ]
    if len(progress_rows) != 1:
        raise AssertionError(
            "completed event did not reach cognition as one evidence row"
        )
    progress_row = progress_rows[0]
    progress_text = progress_row["semantic_text"]
    required_progress_details = (
        completed_event["semantic_summary"],
        "state=completed",
        "retention=decision_critical",
        f"actor={completed_event['actor']}",
        f"action={completed_event['action']}",
        f"object={completed_event['object']}",
        f"beneficiary={completed_event['beneficiary']}",
        f"outcome={completed_event['outcome']}",
    )
    for required_detail in required_progress_details:
        if required_detail not in progress_text:
            raise AssertionError(
                "cognition evidence lost a completed-event semantic field"
            )
    episode_rows = [
        row for row in cognition_input["evidence"]
        if row["evidence_ref"]["source_kind"] == "episode"
    ]
    if len(episode_rows) != 1:
        raise AssertionError(
            "current transition did not reach cognition as one episode row"
        )
    episode_row = episode_rows[0]
    if current_transition not in episode_row["semantic_text"]:
        raise AssertionError(
            "current earlobe completion and next-selection request were lost"
        )

    state_projection = project_state_for_prompt(
        cognition_input["mutable_state"],
        character_constraints=cognition_input["character_constraints"],
        character_identity_context=cognition_input.get(
            "character_identity_context",
            canonical_identity_context(),
        ),
        evidence=cognition_input["evidence"],
    )
    branch_context = facade._branch_context(
        state_projection,
        cognition_input["mutable_state"],
        cognition_input["evidence"],
        scene_context=cognition_input["scene_context"],
        private_continuity_context=(
            cognition_input["private_continuity_context"]
        ),
    )
    goal_capture = _GoalPromptCapture()
    with pytest.raises(_GoalPromptBoundaryReached):
        await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary-response",
            },
            branch_context,
            cognition_input["evidence"],
            SimpleNamespace(
                llm=goal_capture,
                goal_ordinary_response_config=object(),
                goal_active_branch_config=object(),
            ),
        )

    progress_handle = progress_row["evidence_handle"]
    episode_handle = episode_row["evidence_handle"]
    progress_constraints = {
        row["evidence_handle"]: row["semantic_text"]
        for row in goal_capture.payload[
            "conversation_progress_evidence"
        ]
    }
    if progress_handle not in progress_constraints:
        raise AssertionError(
            "serialized goal input lost the completed-event handle"
        )
    required_operations = {
        row["evidence_handle"]: row
        for row in goal_capture.payload["required_selection_operations"]
    }
    if episode_handle not in required_operations:
        raise AssertionError(
            "serialized goal input lost the current selection operation"
        )
    if progress_constraints[progress_handle] != progress_text:
        raise AssertionError(
            "serialized goal input changed completed-event semantics"
        )
    expected_operation = json.loads(
        episode_row["semantic_text"]
    )["response_operation"]
    if (
        required_operations[episode_handle]["response_operation"]
        != expected_operation
    ):
        raise AssertionError(
            "serialized goal input changed the current transition"
        )
    if len(goal_capture.human_payload) > GOAL_COGNITION_PROMPT_CAP:
        raise AssertionError("serialized goal input exceeded its hard cap")
    if "conversation_progress_evidence" not in goal_capture.system_prompt:
        raise AssertionError(
            "serialized goal input did not use the required-selection owner"
        )
    if "conversation_evidence_relations" in goal_capture.system_prompt:
        raise AssertionError("serialized goal input retained retired relations")

    trace_path = write_llm_trace(
        "test_live_original_failure_progress_semantic_handoff",
        "original_failure_progress_semantic_handoff",
        {
            "source_logical_turns": turns,
            "recorder_checkpoints": recorder_checkpoints,
            "active_packet": active_packet,
            "completed_event": completed_event,
            "progress_prompt": progress_prompt,
            "cognition_evidence": cognition_input["evidence"],
            "goal_system_prompt": goal_capture.system_prompt,
            "goal_human_payload": goal_capture.human_payload,
            "semantic_gates": {
                "actor_identifies_current_user": True,
                "action_object_identify_neck_shoulder_massage": True,
                "beneficiary_identifies_character": True,
                "outcome_preserves_acceptance_or_evaluation": True,
                "state_is_completed": True,
                "retention_is_decision_critical": True,
                "source_lineage_preserved": True,
                "earlobe_completion_and_next_request_preserved": True,
                "serialized_goal_input_preserved": True,
                "downstream_goal_model_invoked": False,
            },
        },
    )
    print(json.dumps({
        "case_id": "original_failure_progress_semantic_handoff",
        "trace_path": str(trace_path),
        "recorder_settled_pairs": len(recorder_checkpoints),
        "semantic_gate_count": 10,
        "downstream_goal_model_invoked": False,
    }, ensure_ascii=True, indent=2))


async def test_live_asuna_houjing_long_thread_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recall the completed source event and use it in the next response."""

    await _skip_if_routes_unavailable()
    turns = _full_fixture_participant_turns()
    transition_user_turn = turns[10]
    transition_response_turn = turns[11]
    final_user_turn = turns[-1]
    prior = packet(
        turn_count=8,
        events=[event(
            event_id="first-selected-action",
            summary=(
                "the first selected interaction action is being performed"
            ),
            state="in_progress",
            retention="active_scene",
            source_refs=_bounded_event_sources(turns[6:10]),
        )],
    )
    prior["platform"] = PLATFORM
    prior["platform_channel_id"] = CHANNEL_ID
    prior["global_user_id"] = USER_A_GLOBAL_USER_ID
    current = _scope_record_input(
        prior_packet=prior,
        interaction_turns=turns[2:12],
        current_sources=logical_turn_source_refs([
            transition_user_turn,
            transition_response_turn,
        ]),
        current_input=transition_user_turn["fragments"][0],
        final_dialog=list(transition_response_turn["fragments"]),
    )
    current["scope"] = current["scope"].__class__(
        platform=PLATFORM,
        platform_channel_id=CHANNEL_ID,
        global_user_id=USER_A_GLOBAL_USER_ID,
    )
    current["character_name"] = BOT_DISPLAY_NAME
    current["content_plan"] = {
        "semantic_content": (
            "Evaluate the first concrete action as completed and transition "
            "to a distinct next interaction action."
        ),
        "surface_intent": "respond",
    }
    invocation, prepared, recorder_calls = await _invoke_recorder(
        monkeypatch,
        current,
    )
    first_action_rows = [
        row for row in prepared.packet["events"]
        if row["event_id"] == "first-selected-action"
    ]
    if len(first_action_rows) != 1:
        failure_path = write_llm_trace(
            "test_live_asuna_houjing_long_thread_regression",
            "asuna_houjing_stable_event_missing",
            {
                "record_input": {
                    **current,
                    "scope": asdict(current["scope"]),
                },
                "recorder_model_calls": recorder_calls,
                "recorder_delta": invocation.delta,
                "recorder_telemetry": _recorder_telemetry(invocation),
                "prepared_packet": prepared.packet,
                "prepared_block": prepared.block,
                "hard_gate": (
                    "prior decision-relevant event must remain active"
                ),
            },
        )
        pytest.fail(
            "live recorder demoted or lost the prior active event: "
            f"{failure_path}"
        )
    first_action = first_action_rows[0]
    completed_critical = (
        [first_action]
        if (
            first_action["state"] == "completed"
            and first_action["retention"] == "decision_critical"
        )
        else []
    )
    if not completed_critical:
        failure_path = write_llm_trace(
            "test_live_asuna_houjing_long_thread_regression",
            "asuna_houjing_recorder_semantic_failure",
            {
                "record_input": {
                    **current,
                    "scope": asdict(current["scope"]),
                },
                "recorder_model_calls": recorder_calls,
                "recorder_delta": invocation.delta,
                "prepared_packet": prepared.packet,
                "hard_gate": (
                    "completed decision-critical event was not produced"
                ),
            },
        )
        print(json.dumps({
            "failure_trace_path": str(failure_path),
            "recorder_delta": invocation.delta,
        }, ensure_ascii=True, indent=2))
        pytest.fail(
            "live recorder lost completed decision-critical semantics: "
            f"{failure_path}"
        )
    transition_checkpoint = {
        "label": "first_action_completed_and_next_action_opened",
        "record_input": {
            **current,
            "scope": asdict(current["scope"]),
        },
        "recorder_model_calls": recorder_calls,
        "recorder_delta": invocation.delta,
        "recorder_telemetry": _recorder_telemetry(invocation),
        "active_packet_after_update": prepared.packet,
    }
    next_action = next(
        row for row in prepared.packet["events"]
        if row["event_id"] != "first-selected-action"
    )
    recorder_checkpoints = [transition_checkpoint]
    for user_index in range(12, len(turns) - 1, 2):
        response_index = user_index + 1
        replay_user_turn = turns[user_index]
        replay_response_turn = turns[response_index]
        window_start = max(0, response_index - 9)
        replay_current = _scope_record_input(
            prior_packet=prepared.packet,
            interaction_turns=turns[window_start:response_index + 1],
            current_sources=logical_turn_source_refs([
                replay_user_turn,
                replay_response_turn,
            ]),
            current_input=replay_user_turn["fragments"][0],
            final_dialog=list(replay_response_turn["fragments"]),
        )
        replay_current["scope"] = current["scope"]
        replay_current["character_name"] = BOT_DISPLAY_NAME
        replay_invocation, replay_prepared, replay_calls = (
            await _invoke_recorder(monkeypatch, replay_current)
        )
        recorder_checkpoints.append({
            "label": (
                f"settled_pair_{user_index:02d}_{response_index:02d}"
            ),
            "record_input": {
                **replay_current,
                "scope": asdict(replay_current["scope"]),
            },
            "recorder_model_calls": replay_calls,
            "recorder_delta": replay_invocation.delta,
            "recorder_telemetry": _recorder_telemetry(
                replay_invocation
            ),
            "active_packet_after_update": replay_prepared.packet,
        })
        current = replay_current
        invocation = replay_invocation
        prepared = replay_prepared
        recorder_calls = replay_calls
    tracked_next_action = next(
        row for row in prepared.packet["events"]
        if row["event_id"] == next_action["event_id"]
    )
    tracked_next_state_is_valid = (
        tracked_next_action["state"] in {"in_progress", "completed"}
    )
    if (
        not tracked_next_state_is_valid
        or tracked_next_action["retention"] != "decision_critical"
    ):
        failure_path = write_llm_trace(
            "test_live_asuna_houjing_long_thread_regression",
            "asuna_houjing_catchup_semantic_failure",
            {
                "recorder_checkpoints": recorder_checkpoints,
                "active_packet_after_replay": prepared.packet,
                "hard_gate": (
                    "the tracked next action must remain in-progress or "
                    "completed and decision-critical before the final input"
                ),
            },
        )
        pytest.fail(
            "live recorder lost the tracked next action before cognition: "
            f"{failure_path}"
        )
    current_input_declares_next_event_complete = (
        '耳垂按摩完了' in final_user_turn["fragments"][0]
    )
    if not current_input_declares_next_event_complete:
        raise AssertionError(
            "fixture final input does not explicitly complete the tracked "
            "next action"
        )
    target_event = next(
        row for row in prepared.packet["events"]
        if row["event_id"] == "first-selected-action"
    )
    handoff = await _semantic_handoff(
        monkeypatch=monkeypatch,
        active_packet=prepared.packet,
        interaction_turns=turns[-10:],
        current_input=final_user_turn["fragments"][0],
        target_event_id=target_event["event_id"],
        case_id="asuna_houjing_long_thread",
        response_operation={
            "operation": (
                "The current character chooses and states one distinct "
                "location that the current user may touch next."
            ),
            "response_owner_role": CURRENT_CHARACTER_ROLE,
            "selection_owner_role": CURRENT_CHARACTER_ROLE,
            "selection_required": True,
            "embedded_actor_role": CURRENT_USER_ROLE,
            "embedded_target_role": CURRENT_CHARACTER_ROLE,
        },
    )
    tracked_next_rows = [
        row for row in handoff["cognition_input"]["evidence"]
        if row["evidence_ref"]["source_id"] == (
            "conversation-progress-event:"
            f"{tracked_next_action['event_id']}"
        )
    ]
    if len(tracked_next_rows) != 1:
        raise AssertionError(
            "tracked next action did not reach cognition"
        )
    tracked_next_handle = tracked_next_rows[0]["evidence_handle"]
    selected_bid_cites_completed = (
        handoff["target_evidence_handle"] in
        handoff["selected_bid"]["evidence_handles"]
    )
    selected_bid_cites_tracked_next = (
        tracked_next_handle in
        handoff["selected_bid"]["evidence_handles"]
    )
    current_episode_rows = [
        row for row in handoff["cognition_input"]["evidence"]
        if row["evidence_ref"]["source_kind"] == "episode"
    ]
    if len(current_episode_rows) != 1:
        raise AssertionError(
            "current input did not reach cognition as one episode row"
        )
    current_input_handle = current_episode_rows[0]["evidence_handle"]
    selected_bid_cites_current_input = (
        current_input_handle in
        handoff["selected_bid"]["evidence_handles"]
    )
    tracked_next_completion_grounded = (
        selected_bid_cites_tracked_next
        or (
            current_input_declares_next_event_complete
            and selected_bid_cites_current_input
        )
    )
    completed_location_markers = (
        '后颈',
        '颈后',
        '颈部',
        '脖颈',
        '脖子',
        '肩膀',
        '肩部',
        '肩头',
        '耳根',
        '耳垂',
        '耳廓',
        '耳朵',
        '耳部',
    )
    selected_concrete_detail = str(
        handoff["selected_bid"]["concrete_detail"]
    )
    selected_expected_consequences = list(
        handoff["selected_bid"]["expected_consequences"]
    )
    if not selected_expected_consequences:
        raise AssertionError(
            "selected bid has no expected consequence for its choice"
        )
    selected_choice_consequence = str(
        selected_expected_consequences[0]
    )
    reselected_completed_locations = [
        marker for marker in completed_location_markers
        if marker in selected_choice_consequence
    ]
    completed_location_not_reselected = (
        not reselected_completed_locations
    )
    if (
        not selected_bid_cites_completed
        or not tracked_next_completion_grounded
        or not completed_location_not_reselected
    ):
        _write_case_trace(
            test_name="test_live_asuna_houjing_long_thread_regression",
            case_id="asuna_houjing_cognition_semantic_failure",
            current=current,
            invocation=invocation,
            prepared=prepared,
            recorder_calls=recorder_calls,
            blocks=_block_graph(
                prior_simulation=None,
                prepared=prepared,
            ),
            semantic_handoff=handoff,
            hard_gates={
                "completed_event_reached_cognition": True,
                "tracked_next_event_reached_cognition": True,
                "current_input_declares_next_event_complete": (
                    current_input_declares_next_event_complete
                ),
                "tracked_next_action_prior_state": (
                    tracked_next_action["state"]
                ),
                "selected_bid_cites_completed_event": (
                    selected_bid_cites_completed
                ),
                "selected_bid_cites_tracked_next_event": (
                    selected_bid_cites_tracked_next
                ),
                "selected_bid_cites_current_input": (
                    selected_bid_cites_current_input
                ),
                "tracked_next_completion_grounded": (
                    tracked_next_completion_grounded
                ),
                "completed_location_not_reselected": (
                    completed_location_not_reselected
                ),
                "reselected_completed_locations": (
                    reselected_completed_locations
                ),
                "selected_concrete_detail": selected_concrete_detail,
                "selected_choice_consequence": (
                    selected_choice_consequence
                ),
            },
            recorder_checkpoints=recorder_checkpoints,
        )
        pytest.fail(
            "live cognition omitted or reselected a completed "
            "decision-critical event"
        )
    _write_case_trace(
        test_name="test_live_asuna_houjing_long_thread_regression",
        case_id="asuna_houjing_long_thread",
        current=current,
        invocation=invocation,
        prepared=prepared,
        recorder_calls=recorder_calls,
        blocks=_block_graph(
            prior_simulation=None,
            prepared=prepared,
        ),
        semantic_handoff=handoff,
        hard_gates={
            "participant_logical_turn_count": len(turns),
            "transition_fragment_count": len(
                transition_response_turn["fragments"]
            ),
            "final_user_turn_included": (
                final_user_turn["occurred_at"]
                == "2026-07-28T09:46:49.000000+00:00"
            ),
            "completed_decision_critical_count": len(completed_critical),
            "selected_bid_cites_completed_event": (
                selected_bid_cites_completed
            ),
            "tracked_next_event_reached_cognition": True,
            "current_input_declares_next_event_complete": (
                current_input_declares_next_event_complete
            ),
            "tracked_next_action_prior_state": (
                tracked_next_action["state"]
            ),
            "selected_bid_cites_tracked_next_event": (
                selected_bid_cites_tracked_next
            ),
            "selected_bid_cites_current_input": (
                selected_bid_cites_current_input
            ),
            "tracked_next_completion_grounded": (
                tracked_next_completion_grounded
            ),
            "completed_location_not_reselected": (
                completed_location_not_reselected
            ),
            "reselected_completed_locations": (
                reselected_completed_locations
            ),
            "selected_concrete_detail": selected_concrete_detail,
            "selected_choice_consequence": selected_choice_consequence,
            "accidental_reopen_absent": (
                target_event["state"] == "completed"
            ),
            "source_ref_integrity": all(
                row["source_refs"] for row in prepared.packet["events"]
            ),
        },
        recorder_checkpoints=recorder_checkpoints,
    )


async def test_live_deliberate_reopening_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow an explicit repeat to reopen a completed prior event."""

    await _skip_if_routes_unavailable()
    prior = packet(events=[event(
        event_id="completed-action",
        summary="the selected interaction action was completed",
        state="completed",
        retention="decision_critical",
    )])
    turn = logical_turn(
        turn_id="row:live-deliberate-repeat",
        row_id="live-deliberate-repeat",
    )
    turn["fragments"] = [
        "I explicitly want to repeat that completed action now."
    ]
    sources = logical_turn_source_refs([turn])
    current = _scope_record_input(
        prior_packet=prior,
        interaction_turns=[turn],
        current_sources=sources,
        current_input=turn["fragments"][0],
        final_dialog=["The explicit repeat request is understood."],
    )
    invocation, prepared, recorder_calls = await _invoke_recorder(
        monkeypatch,
        current,
    )
    target = next(
        row for row in prepared.packet["events"]
        if row["event_id"] == "completed-action"
    )
    assert target["state"] in {"open", "in_progress"}
    handoff = await _semantic_handoff(
        monkeypatch=monkeypatch,
        active_packet=prepared.packet,
        interaction_turns=[turn],
        current_input=turn["fragments"][0],
        target_event_id=target["event_id"],
        case_id="deliberate_reopening",
    )
    assert handoff["target_evidence_handle"] in (
        handoff["selected_bid"]["evidence_handles"]
    )
    _write_case_trace(
        test_name="test_live_deliberate_reopening_remains_available",
        case_id="deliberate_reopening",
        current=current,
        invocation=invocation,
        prepared=prepared,
        recorder_calls=recorder_calls,
        blocks={},
        semantic_handoff=handoff,
        hard_gates={
            "prior_state": "completed",
            "result_state": target["state"],
            "explicit_reopening_allowed": True,
        },
    )


async def test_live_cross_domain_correction_and_supersession(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interpret rejection and supersession without domain keyword rules."""

    await _skip_if_routes_unavailable()
    prior = packet(events=[
        event(
            event_id="old-promise",
            summary="the character promised to send written notes tomorrow",
            state="open",
            retention="decision_critical",
        ),
        event(
            event_id="old-option",
            summary="the user was considering the first proposed option",
            state="open",
            retention="active_scene",
        ),
    ])
    turn = logical_turn(
        turn_id="row:live-cross-domain-correction",
        row_id="live-cross-domain-correction",
    )
    turn["fragments"] = [
        (
            "Correction: I reject the first option. Replace the written-notes "
            "promise with a voice summary tomorrow."
        )
    ]
    current = _scope_record_input(
        prior_packet=prior,
        interaction_turns=[turn],
        current_sources=logical_turn_source_refs([turn]),
        current_input=turn["fragments"][0],
        final_dialog=["The correction and replacement are accepted."],
    )
    invocation, prepared, recorder_calls = await _invoke_recorder(
        monkeypatch,
        current,
    )
    by_id = {
        row["event_id"]: row for row in prepared.packet["events"]
    }
    if (
        by_id["old-promise"]["state"] != "superseded"
        or by_id["old-option"]["state"] != "rejected"
    ):
        failure_path = write_llm_trace(
            "test_live_cross_domain_correction_and_supersession",
            "cross_domain_recorder_semantic_failure",
            {
                "record_input": {
                    **current,
                    "scope": asdict(current["scope"]),
                },
                "recorder_model_calls": recorder_calls,
                "recorder_delta": invocation.delta,
                "active_packet_after_update": prepared.packet,
                "hard_gates": {
                    "rejected_state": by_id["old-option"]["state"],
                    "superseded_state": by_id["old-promise"]["state"],
                },
            },
        )
        print(json.dumps({
            "failure_trace_path": str(failure_path),
            "rejected_state": by_id["old-option"]["state"],
            "superseded_state": by_id["old-promise"]["state"],
        }, ensure_ascii=True, indent=2))
        pytest.fail(
            "live recorder omitted one independent correction: "
            f"{failure_path}"
        )
    replacement_events = [
        row for event_id, row in by_id.items()
        if (
            event_id not in {"old-promise", "old-option"}
            and row["state"] in {"open", "in_progress"}
        )
    ]
    assert replacement_events
    target = by_id["old-promise"]
    handoff = await _semantic_handoff(
        monkeypatch=monkeypatch,
        active_packet=prepared.packet,
        interaction_turns=[turn],
        current_input=turn["fragments"][0],
        target_event_id=target["event_id"],
        case_id="cross_domain_correction",
    )
    _write_case_trace(
        test_name="test_live_cross_domain_correction_and_supersession",
        case_id="cross_domain_correction",
        current=current,
        invocation=invocation,
        prepared=prepared,
        recorder_calls=recorder_calls,
        blocks={},
        semantic_handoff=handoff,
        hard_gates={
            "rejected_state": by_id["old-option"]["state"],
            "superseded_state": by_id["old-promise"]["state"],
            "replacement_event_count": len(replacement_events),
        },
    )


async def test_live_interleaved_group_multifragment_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep participant continuity across group noise and segmented output."""

    await _skip_if_routes_unavailable()
    turns = _fixture_participant_turns()
    final_turn = next(
        turn for turn in turns if turn["llm_trace_id"] == TRACE_5
    )
    prior = packet(events=[event(
        event_id="group-critical-event",
        summary="the participant completed the selected interaction",
        state="completed",
        retention="decision_critical",
    )])
    current = _scope_record_input(
        prior_packet=prior,
        interaction_turns=turns,
        current_sources=logical_turn_source_refs([turns[-2], final_turn]),
        current_input=BODY[-8],
        final_dialog=list(final_turn["fragments"]),
    )
    current["scope"] = current["scope"].__class__(
        platform=PLATFORM,
        platform_channel_id=CHANNEL_ID,
        global_user_id=USER_A_GLOBAL_USER_ID,
    )
    current["storage_timestamp_utc"] = CURRENT_TURN_TIMESTAMP
    invocation, prepared, recorder_calls = await _invoke_recorder(
        monkeypatch,
        current,
    )
    protected = next(
        row for row in prepared.packet["events"]
        if row["event_id"] == "group-critical-event"
    )
    emitted_ref_ids = {
        source["ref_id"]
        for update in invocation.delta["event_updates"]
        for source in update["source_refs"]
    }
    assert "row_0015" not in emitted_ref_ids
    assert "trace_user_b_noise" not in emitted_ref_ids
    assert len(final_turn["fragments"]) == 7
    handoff = await _semantic_handoff(
        monkeypatch=monkeypatch,
        active_packet=prepared.packet,
        interaction_turns=turns,
        current_input=(
            "Continue with the current participant and established event."
        ),
        target_event_id=protected["event_id"],
        case_id="group_multifragment",
    )
    _write_case_trace(
        test_name=(
            "test_live_interleaved_group_multifragment_continuation"
        ),
        case_id="group_multifragment",
        current=current,
        invocation=invocation,
        prepared=prepared,
        recorder_calls=recorder_calls,
        blocks={},
        semantic_handoff=handoff,
        hard_gates={
            "participant_logical_turns": len(turns),
            "final_fragment_count": len(final_turn["fragments"]),
            "unrelated_group_source_absent": True,
            "critical_event_state": protected["state"],
        },
    )


async def test_live_group_stale_ambient_is_absent_from_stage_zero_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 0 receives fresh group history after deterministic age discard."""

    await _skip_if_decontextualizer_route_unavailable()
    trigger_timestamp = _COGNITION_NOW
    stale_turn = logical_turn(
        turn_id='row:stale-group-turn',
        row_id='stale-group-turn',
    )
    stale_turn['occurred_at'] = '2026-07-28T06:00:00+00:00'
    stale_turn['fragments'] = ['STALE_GROUP_TURN']
    fresh_turn = logical_turn(
        turn_id='row:fresh-group-turn',
        row_id='fresh-group-turn',
    )
    fresh_turn['occurred_at'] = '2026-07-28T09:00:00+00:00'
    fresh_turn['fragments'] = ['FRESH_GROUP_TURN']
    filtered_turns = filter_group_scene_ambient_turns(
        ambient_logical_turns=[stale_turn, fresh_turn],
        trigger_occurred_at=trigger_timestamp,
    )
    decontextualizer_llm = _CapturingLLM(
        decontextualizer_module._msg_decontextualizer_llm,
    )
    monkeypatch.setattr(
        decontextualizer_module,
        '_msg_decontextualizer_llm',
        decontextualizer_llm,
    )
    state = {
        'character_profile': {'name': BOT_DISPLAY_NAME},
        'user_input': 'Continue FRESH_GROUP_TURN.',
        'user_name': 'participant',
        'platform_user_id': 'platform-user',
        'platform_bot_id': BOT_PLATFORM_USER_ID,
        'prompt_message_context': {
            'body_text': 'Continue FRESH_GROUP_TURN.',
            'mentions': [],
            'attachments': [],
            'addressed_to_global_user_ids': [],
            'broadcast': True,
        },
        'channel_type': 'group',
        'channel_name': CHANNEL_ID,
        'channel_topic': '',
        'indirect_speech_context': '',
        'reply_context': {},
        'ambient_logical_turns': filtered_turns,
    }

    result = await decontextualizer_module.call_msg_decontextualizer(state)
    human_payload = json.loads(
        decontextualizer_llm.calls[0]['human_messages'][0]
    )
    rendered_history = '\n'.join(human_payload['chat_history'])
    trace_path = write_llm_trace(
        'test_live_group_stale_ambient_is_absent_from_stage_zero_prompt',
        'group_stale_ambient',
        {
            'trigger_timestamp': trigger_timestamp,
            'filtered_turn_ids': [
                turn['turn_id'] for turn in filtered_turns
            ],
            'decontextualizer_human_payload': human_payload,
            'decontextualizer_raw_model_output': (
                decontextualizer_llm.calls[0]['raw_output']
            ),
            'decontextualizer_result': result,
            'hard_gates': {
                'stale_turn_absent': (
                    'STALE_GROUP_TURN' not in rendered_history
                ),
                'fresh_turn_present': 'FRESH_GROUP_TURN' in rendered_history,
            },
        },
    )
    print(json.dumps({
        'case_id': 'group_stale_ambient',
        'trace_path': str(trace_path),
        'stale_turn_absent': 'STALE_GROUP_TURN' not in rendered_history,
        'fresh_turn_present': 'FRESH_GROUP_TURN' in rendered_history,
    }, ensure_ascii=True, indent=2))

    assert 'STALE_GROUP_TURN' not in rendered_history
    assert 'FRESH_GROUP_TURN' in rendered_history


async def test_live_private_stale_progress_is_pruned_before_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale private event is absent from the live cognition evidence."""

    await _skip_if_routes_unavailable()
    fresh_event = event(
        event_id='fresh-private-event',
        summary='fresh private decision event',
        state='in_progress',
        retention='decision_critical',
    )
    fresh_event['updated_at'] = '2026-07-28T09:00:00+00:00'
    stale_event = event(
        event_id='stale-private-event',
        summary='STALE_PRIVATE_EVENT',
        state='completed',
        retention='background',
    )
    stale_event['updated_at'] = '2026-07-25T09:00:00+00:00'
    active_packet = packet(events=[fresh_event, stale_event])
    pruned_packet, dropped_count, narrative_cleared = (
        prune_aged_progress_packet(
            active_packet,
            current_timestamp_utc=_COGNITION_NOW,
        )
    )
    assert dropped_count == 1
    assert narrative_cleared is False
    assert [
        row['event_id'] for row in pruned_packet['events']
    ] == ['fresh-private-event']

    handoff = await _semantic_handoff(
        monkeypatch=monkeypatch,
        active_packet=pruned_packet,
        interaction_turns=[logical_turn()],
        current_input='Continue from the fresh private event.',
        target_event_id='fresh-private-event',
        case_id='private_stale_progress',
    )
    evidence_source_ids = {
        row['evidence_ref']['source_id']
        for row in handoff['cognition_input']['evidence']
    }
    trace_path = write_llm_trace(
        'test_live_private_stale_progress_is_pruned_before_cognition',
        'private_stale_progress',
        {
            'pre_prune_packet': active_packet,
            'pruned_packet': pruned_packet,
            'dropped_count': dropped_count,
            'narrative_cleared': narrative_cleared,
            'cognition_input': handoff['cognition_input'],
            'goal_model_calls': handoff['goal_model_calls'],
            'surface_model_calls': handoff['surface_model_calls'],
            'dialog_model_calls': handoff['dialog_model_calls'],
            'final_dialog': handoff['final_dialog'],
            'hard_gates': {
                'stale_event_absent': (
                    'conversation-progress-event:stale-private-event'
                    not in evidence_source_ids
                ),
                'fresh_event_present': (
                    'conversation-progress-event:fresh-private-event'
                    in evidence_source_ids
                ),
            },
        },
    )
    print(json.dumps({
        'case_id': 'private_stale_progress',
        'trace_path': str(trace_path),
        'stale_event_absent': (
            'conversation-progress-event:stale-private-event'
            not in evidence_source_ids
        ),
        'fresh_event_present': (
            'conversation-progress-event:fresh-private-event'
            in evidence_source_ids
        ),
    }, ensure_ascii=True, indent=2))

    assert 'conversation-progress-event:stale-private-event' not in (
        evidence_source_ids
    )
    assert 'conversation-progress-event:fresh-private-event' in (
        evidence_source_ids
    )


async def _run_capacity_case(
    *,
    monkeypatch: pytest.MonkeyPatch,
    turn_count: int,
    test_name: str,
    case_id: str,
) -> None:
    """Run one real recorder continuation at an exact capacity checkpoint."""

    await _skip_if_routes_unavailable()
    prior_simulation = simulate_long_thread(turn_count - 1)
    prior_packet = prior_simulation.packet
    active_blocks = [
        prior_simulation.blocks[block_id]
        for block_id in sorted(prior_simulation.reachable_block_ids())
    ]
    turn = _capacity_turn(turn_count)
    current = _scope_record_input(
        prior_packet=prior_packet,
        interaction_turns=[turn],
        current_sources=logical_turn_source_refs([turn]),
        current_input=turn["fragments"][0],
        final_dialog=["Continue from established chronology."],
    )
    invocation, prepared, recorder_calls = await _invoke_recorder(
        monkeypatch,
        current,
        active_blocks=active_blocks,
    )
    packet_after = prepared.packet
    assert packet_after["turn_count"] == turn_count
    assert len(packet_after["events"]) <= MAX_ACTIVE_EVENTS
    assert len(packet_after["recent_turn_refs"]) <= MAX_RECENT_TURN_REFS
    assert (
        len(packet_after["compacted_block_refs"])
        <= MAX_ACTIVE_BLOCK_REFS
    )
    critical = _critical_event(packet_after)
    blocks = _block_graph(
        prior_simulation=prior_simulation,
        prepared=prepared,
    )
    reachable = _reachable_block_ids(packet_after, blocks)
    reachable_event_ids = {
        row["event_id"]
        for block_id in reachable
        for row in blocks[block_id]["events"]
    }
    assert prior_simulation.archived_event_ids <= reachable_event_ids
    if turn_count == 100:
        assert prior_simulation.hierarchical_compaction_count > 0
    handoff = await _semantic_handoff(
        monkeypatch=monkeypatch,
        active_packet=packet_after,
        interaction_turns=[turn],
        current_input=turn["fragments"][0],
        target_event_id=critical["event_id"],
        case_id=case_id,
    )
    assert handoff["target_evidence_handle"] in (
        handoff["selected_bid"]["evidence_handles"]
    )
    scene_chars, evidence_chars = continuation_projection_chars(
        handoff["progress_prompt"],
        packet_after["updated_at"],
    )
    assert scene_chars <= MAX_PROGRESS_SCENE_CHARS
    assert evidence_chars <= MAX_PROGRESS_EVIDENCE_CHARS
    assert scene_chars + evidence_chars <= MAX_CONTINUATION_CHARS
    _write_case_trace(
        test_name=test_name,
        case_id=case_id,
        current=current,
        invocation=invocation,
        prepared=prepared,
        recorder_calls=recorder_calls,
        blocks=blocks,
        semantic_handoff=handoff,
        hard_gates={
            "turn_count": packet_after["turn_count"],
            "active_event_count": len(packet_after["events"]),
            "recent_turn_ref_count": len(
                packet_after["recent_turn_refs"]
            ),
            "active_block_ref_count": len(
                packet_after["compacted_block_refs"]
            ),
            "reachable_block_count": len(reachable),
            "archived_event_count": len(
                prior_simulation.archived_event_ids
            ),
            "reachable_archived_event_count": len(reachable_event_ids),
            "decision_critical_state": critical["state"],
            "recorder_model_call_count": len(recorder_calls),
            "deterministic_block_created": prepared.block is not None,
            "hierarchical_compaction_count": (
                prior_simulation.hierarchical_compaction_count
                + (1 if prepared.source_block_ids else 0)
            ),
            "source_ref_integrity": all(
                row["source_refs"] for row in packet_after["events"]
            ) and all(
                row["source_refs"]
                for block_id in reachable
                for row in blocks[block_id]["events"]
            ),
        },
    )


async def test_live_twenty_turn_packet_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise a real recorder continuation ending at turn 20."""

    await _run_capacity_case(
        monkeypatch=monkeypatch,
        turn_count=20,
        test_name="test_live_twenty_turn_packet_continuation",
        case_id="twenty_turn_continuation",
    )


async def test_live_fifty_turn_block_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real continuation after deterministic block compaction."""

    await _run_capacity_case(
        monkeypatch=monkeypatch,
        turn_count=50,
        test_name="test_live_fifty_turn_block_compaction",
        case_id="fifty_turn_compaction",
    )


async def test_live_hundred_turn_hierarchical_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise a real continuation ending at turn 100 after hierarchy."""

    await _run_capacity_case(
        monkeypatch=monkeypatch,
        turn_count=100,
        test_name="test_live_hundred_turn_hierarchical_compaction",
        case_id="hundred_turn_hierarchical",
    )
