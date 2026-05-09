"""Tests for projecting cognitive episode origins into consolidation state."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    OutputMode,
    build_text_chat_cognitive_episode,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator as consolidator_module,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin import (
    ConsolidationOriginError,
    build_user_message_consolidation_origin,
)
from kazusa_ai_chatbot.time_context import build_character_time_context


def _text_chat_episode(
    output_mode: OutputMode = "visible_reply",
) -> CognitiveEpisode:
    """Build a valid text-chat cognitive episode for origin tests.

    Args:
        output_mode: Output mode to store on the episode.

    Returns:
        Valid current text-chat cognitive episode.
    """
    timestamp = "2026-05-10T09:00:00+12:00"
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-1",
        percept_id="percept-1",
        timestamp=timestamp,
        time_context=build_character_time_context(timestamp),
        user_input="Please remember this.",
        platform="qq",
        platform_channel_id="channel-1",
        channel_type="group",
        platform_message_id="message-1",
        platform_user_id="platform-user-1",
        global_user_id="global-user-1",
        user_name="Test User",
        active_turn_platform_message_ids=["message-1", "message-2"],
        active_turn_conversation_row_ids=["conversation-row-1", "conversation-row-2"],
        debug_modes={"think_only": False},
        output_mode=output_mode,
        target_addressed_user_ids=["character-user-1"],
        target_broadcast=False,
    )
    return episode


def _global_state() -> dict:
    """Build a minimal state for direct consolidation subgraph calls.

    Returns:
        Global persona state fields consumed by `call_consolidation_subgraph`.
    """
    state = {
        "timestamp": "2026-05-10T09:00:00+12:00",
        "time_context": build_character_time_context(
            "2026-05-10T09:00:00+12:00"
        ),
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {"affinity": 500},
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "channel_type": "group",
        "platform_message_id": "message-1",
        "action_directives": {
            "linguistic_directives": {"content_anchors": []},
        },
        "internal_monologue": "Answer normally.",
        "final_dialog": ["ok"],
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "PROVIDE",
        "logical_stance": "CONFIRM",
        "character_profile": {"name": "Kazusa"},
        "rag_result": {
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [],
                    "active_commitments": [],
                    "milestones": [],
                }
            }
        },
        "decontexualized_input": "Please remember this.",
        "chat_history_recent": [],
        "cognitive_episode": _text_chat_episode(),
    }
    return state


def test_build_user_message_consolidation_origin_returns_exact_metadata() -> None:
    episode = _text_chat_episode()

    metadata = build_user_message_consolidation_origin(episode=episode)

    expected_metadata = {
        "episode_id": "episode-1",
        "trigger_source": "user_message",
        "input_sources": ["dialog_text"],
        "output_mode": "visible_reply",
        "timestamp": "2026-05-10T09:00:00+12:00",
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
    assert metadata == expected_metadata


def test_origin_metadata_copies_list_fields() -> None:
    episode = _text_chat_episode()

    metadata = build_user_message_consolidation_origin(episode=episode)
    episode["input_sources"].append("image_observation")
    episode["origin_metadata"]["active_turn_platform_message_ids"].append("message-3")
    episode["origin_metadata"]["active_turn_conversation_row_ids"].append(
        "conversation-row-3"
    )

    assert metadata["input_sources"] == ["dialog_text"]
    assert metadata["active_turn_platform_message_ids"] == [
        "message-1",
        "message-2",
    ]
    assert metadata["active_turn_conversation_row_ids"] == [
        "conversation-row-1",
        "conversation-row-2",
    ]


def test_origin_metadata_excludes_content_and_prompt_fields() -> None:
    episode = _text_chat_episode()

    metadata = build_user_message_consolidation_origin(episode=episode)

    forbidden_keys = {
        "percepts",
        "content",
        "user_input",
        "decontexualized_input",
        "prompt_payload",
        "attachments",
        "rag_result",
        "facts",
        "promises",
        "debug_modes",
    }
    assert forbidden_keys.isdisjoint(set(metadata))
    assert "Please remember this." not in str(metadata)


def test_origin_rejects_non_user_message_trigger() -> None:
    episode = deepcopy(_text_chat_episode())
    episode["trigger_source"] = "reflection_signal"

    with pytest.raises(ConsolidationOriginError):
        build_user_message_consolidation_origin(episode=episode)


def test_origin_rejects_non_dialog_text_sources() -> None:
    episode = deepcopy(_text_chat_episode())
    episode["input_sources"] = ["dialog_text", "image_observation"]
    episode["percepts"].append(
        {
            "percept_id": "percept-2",
            "input_source": "image_observation",
            "content": "image observed",
            "visibility": "model_visible",
            "metadata": {},
        }
    )

    with pytest.raises(ConsolidationOriginError):
        build_user_message_consolidation_origin(episode=episode)


def test_origin_rejects_unsupported_output_mode() -> None:
    episode = _text_chat_episode(output_mode="preview")

    with pytest.raises(ConsolidationOriginError):
        build_user_message_consolidation_origin(episode=episode)


@pytest.mark.asyncio
async def test_call_consolidation_subgraph_threads_origin_to_all_nodes(
    monkeypatch,
) -> None:
    state = _global_state()
    expected_origin = build_user_message_consolidation_origin(
        episode=state["cognitive_episode"],
    )
    seen_origins = {}

    async def _global_state_updater(node_state: dict) -> dict:
        """Capture origin metadata seen by the global-state updater.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic global-state updater output.
        """
        seen_origins["global_state_updater"] = node_state["consolidation_origin"]
        return {
            "mood": "calm",
            "global_vibe": "steady",
            "reflection_summary": "summary",
        }

    async def _relationship_recorder(node_state: dict) -> dict:
        """Capture origin metadata seen by the relationship recorder.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic relationship-recorder output.
        """
        seen_origins["relationship_recorder"] = node_state["consolidation_origin"]
        return {
            "subjective_appraisals": [],
            "affinity_delta": 0,
            "last_relationship_insight": "",
        }

    async def _facts_harvester(node_state: dict) -> dict:
        """Capture origin metadata seen by the facts harvester.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic facts-harvester output that exercises the evaluator.
        """
        seen_origins["facts_harvester"] = node_state["consolidation_origin"]
        return {
            "new_facts": [{"fact": "User likes tea"}],
            "future_promises": [],
        }

    async def _fact_harvester_evaluator(node_state: dict) -> dict:
        """Capture origin metadata seen by the fact evaluator.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic evaluator output that ends the loop.
        """
        seen_origins["fact_harvester_evaluator"] = node_state[
            "consolidation_origin"
        ]
        return {"should_stop": True}

    async def _db_writer(node_state: dict) -> dict:
        """Capture origin metadata seen by the persistence boundary.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic persistence metadata.
        """
        seen_origins["db_writer"] = node_state["consolidation_origin"]
        return {"metadata": {"write_success": {}}}

    monkeypatch.setattr(
        consolidator_module,
        "global_state_updater",
        _global_state_updater,
    )
    monkeypatch.setattr(
        consolidator_module,
        "relationship_recorder",
        _relationship_recorder,
    )
    monkeypatch.setattr(consolidator_module, "facts_harvester", _facts_harvester)
    monkeypatch.setattr(
        consolidator_module,
        "fact_harvester_evaluator",
        _fact_harvester_evaluator,
    )
    monkeypatch.setattr(consolidator_module, "db_writer", _db_writer)

    await consolidator_module.call_consolidation_subgraph(state)

    assert seen_origins == {
        "global_state_updater": expected_origin,
        "relationship_recorder": expected_origin,
        "facts_harvester": expected_origin,
        "fact_harvester_evaluator": expected_origin,
        "db_writer": expected_origin,
    }


@pytest.mark.asyncio
async def test_unsupported_origin_fails_before_state_graph_construction(
    monkeypatch,
) -> None:
    state = _global_state()
    episode = deepcopy(state["cognitive_episode"])
    episode["trigger_source"] = "reflection_signal"
    state["cognitive_episode"] = episode
    graph_construction_calls = []

    def _state_graph(_state_type: object) -> object:
        """Record unexpected graph construction attempts.

        Args:
            _state_type: State type passed to the graph constructor.

        Returns:
            This helper never returns because graph construction is forbidden.

        Raises:
            AssertionError: Always, if graph construction is attempted.
        """
        graph_construction_calls.append(_state_type)
        raise AssertionError("StateGraph must not be constructed")

    monkeypatch.setattr(consolidator_module, "StateGraph", _state_graph)

    with pytest.raises(ConsolidationOriginError):
        await consolidator_module.call_consolidation_subgraph(state)

    assert graph_construction_calls == []


@pytest.mark.asyncio
async def test_call_consolidation_subgraph_does_not_return_origin_metadata(
    monkeypatch,
) -> None:
    async def _global_state_updater(node_state: dict) -> dict:
        """Return stable global-state output after seeing origin metadata.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic global-state updater output.
        """
        assert node_state["consolidation_origin"]["episode_id"] == "episode-1"
        return {
            "mood": "calm",
            "global_vibe": "steady",
            "reflection_summary": "summary",
        }

    async def _relationship_recorder(node_state: dict) -> dict:
        """Return stable relationship output after seeing origin metadata.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic relationship-recorder output.
        """
        assert node_state["consolidation_origin"]["episode_id"] == "episode-1"
        return {
            "subjective_appraisals": [],
            "affinity_delta": 0,
            "last_relationship_insight": "",
        }

    async def _facts_harvester(node_state: dict) -> dict:
        """Return no facts after seeing origin metadata.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Empty facts and promises so the evaluator is skipped.
        """
        assert node_state["consolidation_origin"]["episode_id"] == "episode-1"
        return {"new_facts": [], "future_promises": []}

    async def _fact_harvester_evaluator(_node_state: dict) -> dict:
        """Fail if the evaluator runs for empty harvester output.

        Args:
            _node_state: Unused consolidator state for an unexpected call.

        Returns:
            This helper never returns during a valid test path.

        Raises:
            AssertionError: Always, because the evaluator should be skipped.
        """
        raise AssertionError("fact evaluator should be skipped")

    async def _db_writer(node_state: dict) -> dict:
        """Return durable metadata without exposing origin metadata.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic durable metadata without origin fields.
        """
        assert node_state["consolidation_origin"]["episode_id"] == "episode-1"
        return {"metadata": {"write_success": {"character_state": True}}}

    monkeypatch.setattr(
        consolidator_module,
        "global_state_updater",
        _global_state_updater,
    )
    monkeypatch.setattr(
        consolidator_module,
        "relationship_recorder",
        _relationship_recorder,
    )
    monkeypatch.setattr(consolidator_module, "facts_harvester", _facts_harvester)
    monkeypatch.setattr(
        consolidator_module,
        "fact_harvester_evaluator",
        _fact_harvester_evaluator,
    )
    monkeypatch.setattr(consolidator_module, "db_writer", _db_writer)

    result = await consolidator_module.call_consolidation_subgraph(_global_state())

    assert set(result) == {
        "mood",
        "global_vibe",
        "reflection_summary",
        "subjective_appraisals",
        "affinity_delta",
        "last_relationship_insight",
        "new_facts",
        "future_promises",
        "consolidation_metadata",
    }
    assert "consolidation_origin" not in result
    assert "consolidation_origin" not in result["consolidation_metadata"]
