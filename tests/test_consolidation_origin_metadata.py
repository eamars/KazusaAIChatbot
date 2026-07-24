"""Tests for projecting canonical episode origins into consolidation state."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    build_self_cognition_episode,
    build_tool_result_episode,
    build_user_message_episode,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.consolidation import core as consolidator_module
from kazusa_ai_chatbot.consolidation.origin import (
    ConsolidationOriginError,
    build_self_cognition_consolidation_origin,
    build_tool_result_consolidation_origin,
    build_user_message_consolidation_origin,
    project_consolidation_origin_prompt_block,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock


def _target_scope(*, group: bool = True) -> dict[str, object]:
    """Return one valid adapter-neutral target scope."""

    return {
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "channel_type": "group" if group else "private",
        "current_platform_user_id": "platform-user-1",
        "current_global_user_id": "global-user-1",
        "current_display_name": "Test User",
        "target_addressed_user_ids": ["global-user-1"],
        "target_broadcast": False,
    }


def _local_time_context() -> dict[str, str]:
    """Return the configured-local context used by episode builders."""

    return {
        "current_local_datetime": "2026-05-10 17:00",
        "current_local_weekday": "Sunday",
    }


def _text_chat_episode():
    """Build a valid canonical user-message episode."""

    turn_clock = build_turn_clock("2026-05-10 09:00:00")
    return build_user_message_episode(
        episode_id="episode-1",
        origin={
            "platform": "qq",
            "platform_message_id": "message-1",
            "active_turn_platform_message_ids": ["message-1", "message-2"],
            "active_turn_conversation_row_ids": [
                "conversation-row-1",
                "conversation-row-2",
            ],
        },
        target_scope=_target_scope(),
        dialog_percept={
            "schema_version": "percept.v1",
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": "message-1",
            "content": {"semantic_text": "Please remember this."},
            "observed_at": turn_clock["storage_timestamp_utc"],
        },
        media_percepts=[],
        evidence_refs=[],
        local_time_context=turn_clock["local_time_context"],
        created_at=turn_clock["storage_timestamp_utc"],
        debug_controls={"think_only": False},
    )


def _self_cognition_episode():
    """Build a valid canonical self-cognition episode."""

    turn_clock = build_turn_clock("2026-05-10 21:00:00")
    return build_self_cognition_episode(
        case={
            "case_id": "case-1",
            "source_case_kind": "internal_monologue",
            "target_scope": _target_scope(group=False),
            "privacy_scope": "private",
        },
        percepts=[{
            "schema_version": "percept.v1",
            "percept_kind": "internal_context",
            "source_kind": "internal_thought",
            "source_id": "case-1",
            "content": {
                "summary": "The missed promise still feels unresolved.",
            },
            "observed_at": turn_clock["storage_timestamp_utc"],
        }],
        evidence_refs=[],
        local_time_context=turn_clock["local_time_context"],
        created_at=turn_clock["storage_timestamp_utc"],
    )


def _tool_result_episode():
    """Build a valid canonical completed-tool-result episode."""

    turn_clock = build_turn_clock("2026-05-10 22:00:00")
    return build_tool_result_episode(
        result={
            "schema_version": "tool_result_ready.v1",
            "task_id": "task-1",
            "task_kind": "background_work",
            "semantic_summary": "The requested work completed.",
            "artifact_text": "result text",
            "failure_text": "",
            "completed_at": turn_clock["storage_timestamp_utc"],
            "target_scope": _target_scope(group=False),
            "evidence_refs": [],
            "coding_run_context": {},
            "result_ref": "result-1",
        },
        evidence_refs=[],
        local_time_context=turn_clock["local_time_context"],
        created_at=turn_clock["storage_timestamp_utc"],
    )


def _global_state() -> dict:
    """Build the minimum state consumed by direct consolidation calls."""

    turn_clock = build_turn_clock("2026-05-10 09:00:00")
    return {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {
            "global_user_id": "global-user-1",
            "cognition_state": build_acquaintance_user_state(
                global_user_id="global-user-1",
                updated_at="2026-07-03T00:00:00Z",
            ),
        },
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "channel_type": "group",
        "platform_message_id": "message-1",
        "action_directives": {
            "linguistic_directives": {"content_plan": {}},
        },
        "internal_monologue": "Answer normally.",
        "final_dialog": ["ok"],
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "PROVIDE",
        "logical_stance": "CONFIRM",
        "character_profile": {"name": "Kazusa"},
        "rag_result": {
            "user_memory_unit_candidates": [],
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [],
                    "active_commitments": [],
                    "milestones": [],
                }
            },
        },
        "decontextualized_input": "Please remember this.",
        "chat_history_recent": [],
        "cognitive_episode": _text_chat_episode(),
    }


def test_user_message_origin_projects_exact_identifier_metadata() -> None:
    """User-message origin contains identifiers and source kinds only."""

    episode = _text_chat_episode()
    metadata = build_user_message_consolidation_origin(episode=episode)

    assert metadata == {
        "episode_id": "episode-1",
        "trigger_source": "user_message",
        "input_sources": ["dialog", "system_event"],
        "output_mode": "visible_reply",
        "storage_timestamp_utc": episode["created_at"],
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "channel_type": "group",
        "platform_message_id": "message-1",
        "active_turn_platform_message_ids": ["message-1", "message-2"],
        "active_turn_conversation_row_ids": [
            "conversation-row-1",
            "conversation-row-2",
        ],
        "current_platform_user_id": "platform-user-1",
        "current_global_user_id": "global-user-1",
        "current_display_name": "Test User",
    }


def test_origin_metadata_copies_list_fields_and_excludes_content() -> None:
    """Origin projections remain stable when the episode is later mutated."""

    episode = _text_chat_episode()
    metadata = build_user_message_consolidation_origin(episode=episode)
    episode["percepts"].append({
        "schema_version": "percept.v1",
        "percept_kind": "extra",
        "source_kind": "extra",
        "source_id": "extra",
        "content": {"text": "secret"},
        "observed_at": episode["created_at"],
    })
    episode["origin_metadata"]["active_turn_platform_message_ids"].append(
        "message-3"
    )

    assert metadata["input_sources"] == ["dialog", "system_event"]
    assert metadata["active_turn_platform_message_ids"] == [
        "message-1",
        "message-2",
    ]
    assert "secret" not in str(metadata)


def test_self_cognition_origin_supports_canonical_self_source() -> None:
    """Self-cognition origin maps the canonical private source."""

    episode = _self_cognition_episode()
    metadata = build_self_cognition_consolidation_origin(episode=episode)

    assert metadata["episode_id"] == "self-cognition:case-1"
    assert metadata["trigger_source"] == "self_cognition"
    assert metadata["input_sources"] == ["internal_thought", "system_event"]
    assert metadata["output_mode"] == "preview"
    assert metadata["current_global_user_id"] == "global-user-1"


def test_tool_result_origin_supports_completed_result_source() -> None:
    """Completed tool results enter consolidation with bounded identity."""

    episode = _tool_result_episode()
    metadata = build_tool_result_consolidation_origin(episode=episode)

    assert metadata["episode_id"] == "tool-result:task-1"
    assert metadata["trigger_source"] == "tool_result"
    assert metadata["input_sources"] == ["tool_result", "system_event"]
    assert metadata["output_mode"] == "visible_reply"


def test_origin_prompt_block_is_compact() -> None:
    """Model-facing consolidation origin excludes storage identifiers."""

    origin = build_user_message_consolidation_origin(
        episode=_text_chat_episode(),
    )

    assert project_consolidation_origin_prompt_block(origin) == {
        "episode_id": "episode-1",
        "trigger_source": "user_message",
        "input_sources": ["dialog", "system_event"],
        "output_mode": "visible_reply",
    }


def test_origin_builders_reject_wrong_source() -> None:
    """Each consolidation origin owner admits only its source family."""

    user_episode = deepcopy(_text_chat_episode())
    user_episode["trigger_source"] = "tool_result"
    with pytest.raises(ConsolidationOriginError):
        build_user_message_consolidation_origin(episode=user_episode)

    self_episode = deepcopy(_self_cognition_episode())
    self_episode["trigger_source"] = "user_message"
    with pytest.raises(ConsolidationOriginError):
        build_self_cognition_consolidation_origin(episode=self_episode)

    tool_episode = deepcopy(_tool_result_episode())
    tool_episode["trigger_source"] = "self_cognition"
    with pytest.raises(ConsolidationOriginError):
        build_tool_result_consolidation_origin(episode=tool_episode)


@pytest.mark.asyncio
async def test_call_consolidation_subgraph_threads_origin_to_all_nodes(
    monkeypatch,
) -> None:
    """The lane pipeline receives the canonical origin and target plan."""

    state = _global_state()
    expected_origin = build_user_message_consolidation_origin(
        episode=state["cognitive_episode"],
    )
    seen_pipeline_state = {}

    async def lane_pipeline(node_state: dict) -> dict:
        """Capture origin metadata seen by the patched lane pipeline."""

        seen_pipeline_state.update(node_state)
        pipeline_state = {
            **node_state,
            "mood": "calm",
            "vibe_check": "steady",
            "character_reflection": "summary",
            "subjective_appraisals": [],
            "relationship_delta": 0,
            "semantic_relationship_projection": "",
            "new_facts": [{"fact": "User likes tea"}],
            "future_promises": [],
            "metadata": {"write_success": {}},
        }
        return {"router_tasks": [], "state": pipeline_state}

    monkeypatch.setattr(
        consolidator_module,
        "run_consolidation_lane_pipeline",
        lane_pipeline,
    )

    await consolidator_module.call_consolidation_subgraph(state)

    assert seen_pipeline_state["consolidation_origin"] == expected_origin
    assert "consolidation_target_plan" in seen_pipeline_state


@pytest.mark.asyncio
async def test_unsupported_origin_fails_before_state_graph_construction(
    monkeypatch,
) -> None:
    """An unregistered Stage 3 source cannot enter consolidation."""

    state = _global_state()
    episode = deepcopy(state["cognitive_episode"])
    episode["trigger_source"] = "unregistered_source"
    state["cognitive_episode"] = episode
    pipeline_calls = []

    async def lane_pipeline(node_state: dict) -> dict:
        """Record unexpected lane-pipeline attempts."""

        pipeline_calls.append(node_state)
        raise AssertionError("lane pipeline must not run")

    monkeypatch.setattr(
        consolidator_module,
        "run_consolidation_lane_pipeline",
        lane_pipeline,
    )

    with pytest.raises(ConsolidationOriginError):
        await consolidator_module.call_consolidation_subgraph(state)

    assert pipeline_calls == []


@pytest.mark.asyncio
async def test_call_consolidation_subgraph_returns_sanitized_result(
    monkeypatch,
) -> None:
    """Consolidation returns durable outputs without origin internals."""

    async def lane_pipeline(node_state: dict) -> dict:
        """Return a minimal pipeline result for the public-call test."""

        assert node_state["consolidation_origin"]["episode_id"] == "episode-1"
        pipeline_state = {
            **node_state,
            "mood": "calm",
            "vibe_check": "steady",
            "character_reflection": "summary",
            "subjective_appraisals": [],
            "relationship_delta": 0,
            "semantic_relationship_projection": "",
            "new_facts": [],
            "future_promises": [],
            "metadata": {"write_success": {"character_state": True}},
        }
        return {"router_tasks": [], "state": pipeline_state}

    monkeypatch.setattr(
        consolidator_module,
        "run_consolidation_lane_pipeline",
        lane_pipeline,
    )

    result = await consolidator_module.call_consolidation_subgraph(
        _global_state(),
    )

    assert set(result) == {
        "new_facts",
        "future_promises",
        "consolidation_metadata",
    }
