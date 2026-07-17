"""Tests for efficient canonical V2 consolidator behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.consolidation import core as consolidator_module
from kazusa_ai_chatbot.consolidation import memory_units as memory_units_module
from kazusa_ai_chatbot.consolidation.origin import (
    build_user_message_consolidation_origin,
)
from kazusa_ai_chatbot.time_boundary import local_time_context_from_storage_utc
from tests.cognition_core_v2_test_helpers import canonical_cognition_output


STORAGE_TIMESTAMP_UTC = "2026-04-26T12:00:00+00:00"


def _cognitive_episode() -> dict:
    """Build a valid text-chat episode for direct consolidator calls."""

    return build_text_chat_cognitive_episode(
        episode_id="episode-1",
        percept_id="percept-1",
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        user_input="hello",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        platform_message_id="msg-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=["msg-1"],
        active_turn_conversation_row_ids=["conversation-row-1"],
        debug_modes={},
    )


def _global_state() -> dict:
    """Build a native V2 global state for consolidation."""

    return {
        "storage_timestamp_utc": STORAGE_TIMESTAMP_UTC,
        "local_time_context": local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {
            "global_user_id": "user-1",
            "cognition_state": build_acquaintance_user_state(
                global_user_id="user-1",
                updated_at="2026-04-26T12:00:00Z",
            ),
        },
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_message_id": "msg-1",
        "cognition_core_output": canonical_cognition_output(
            owner_user_id="user-1",
        ),
        "text_surface_output_v2": {
            "schema_version": "text_surface_output.v2",
            "content_plan": "acknowledge",
            "content_requirements": ["Acknowledge the current user."],
            "visible_boundaries": [],
            "addressee_plan": ["current user"],
            "style_guidance": "brief and grounded",
            "selected_surface_intent": "acknowledge",
        },
        "internal_monologue": "test",
        "final_dialog": ["ok"],
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "PROVIDE",
        "logical_stance": "CONFIRM",
        "character_profile": {"name": "Kazusa"},
        "rag_result": {"user_memory_unit_candidates": []},
        "decontexualized_input": "hello",
        "cognitive_episode": _cognitive_episode(),
    }


def test_build_existing_dedup_keys_ignores_memory_unit_context() -> None:
    """Transient RAG context must not become a dedup authority."""

    assert consolidator_module._build_existing_dedup_keys(_global_state()) == set()


def test_build_existing_dedup_keys_requires_canonical_candidates() -> None:
    """Missing canonical RAG candidates fail at the consolidation boundary."""

    global_state = _global_state()
    global_state["rag_result"] = {}

    with pytest.raises(KeyError, match="user_memory_unit_candidates"):
        consolidator_module._build_existing_dedup_keys(global_state)


def test_build_consolidator_state_forwards_native_v2_outputs() -> None:
    """The background state carries canonical cognition and surface outputs."""

    global_state = _global_state()
    origin = build_user_message_consolidation_origin(
        episode=_cognitive_episode(),
    )
    target_plan = {
        "origin_kind": "user_message",
        "targets": [],
    }

    state = consolidator_module._build_consolidator_state(
        global_state,
        consolidation_origin=origin,
        consolidation_target_plan=target_plan,
    )

    assert state["cognition_core_output"] == global_state[
        "cognition_core_output"
    ]
    assert state["text_surface_output_v2"] == global_state[
        "text_surface_output_v2"
    ]


def test_build_consolidator_state_projects_subjective_appraisal() -> None:
    """The memory lane receives the admitted V2 subjective reason."""

    global_state = _global_state()
    global_state["interaction_subtext"] = "  grounded subjective reason  "
    state = consolidator_module._build_consolidator_state(
        global_state,
        consolidation_origin=build_user_message_consolidation_origin(
            episode=_cognitive_episode(),
        ),
        consolidation_target_plan={
            "origin_kind": "user_message",
            "targets": [],
        },
    )

    assert state["subjective_appraisals"] == [
        "grounded subjective reason",
    ]
    payload = memory_units_module._json_payload(state)
    assert payload["subjective_appraisal_evidence"] == [
        "grounded subjective reason",
    ]


@pytest.mark.asyncio
async def test_empty_lane_router_output_skips_persistence_work(monkeypatch) -> None:
    """An empty lane decision returns without durable writer work."""

    writer = AsyncMock()

    async def _lane_pipeline(state):
        assert state["existing_dedup_keys"] == set()
        return {
            "router_tasks": [],
            "state": {
                **state,
                "new_facts": [],
                "future_promises": [],
                "metadata": {"write_success": {}},
            },
        }

    monkeypatch.setattr(
        consolidator_module,
        "run_consolidation_lane_pipeline",
        _lane_pipeline,
    )
    monkeypatch.setattr(consolidator_module, "db_writer", writer, raising=False)

    result = await consolidator_module.call_consolidation_subgraph(_global_state())

    assert result["new_facts"] == []
    assert result["future_promises"] == []
    assert result["consolidation_metadata"]["write_success"] == {}
    assert result["consolidation_metadata"]["cache_evicted_count"] == 0
    assert result["consolidation_metadata"]["cache_invalidated"] == []
    writer.assert_not_awaited()
