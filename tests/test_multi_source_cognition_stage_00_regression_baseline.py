"""Deterministic regression baseline for the current chat workflow."""

from __future__ import annotations

import pytest
pytest.skip("Stage 1 assertions replaced by the V2 contract suite", allow_module_level=True)

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.cognition_resolver import capabilities as capabilities_module
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2 as supervisor_module
from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import (
    project_known_facts,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from llm_test_helpers import bind_test_llm


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "multi_source_cognition_stage_00_cases.json"
)
BACKGROUND_HANDOFF_WAIT_POLLS = 100
BACKGROUND_HANDOFF_WAIT_SECONDS = 0.01


class _DummyResponse:
    """Small LangChain-like response wrapper for mocked LLM calls."""

    def __init__(self, content: str) -> None:
        """Store the response text exposed through ``content``.

        Args:
            content: JSON text returned by the fake LLM.
        """
        self.content = content


class _CaptureLLM:
    """Async LLM fake that captures rendered prompt messages."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Create a fake LLM with one deterministic JSON payload.

        Args:
            payload: JSON-serializable payload returned by ``ainvoke``.
        """
        self.payload = payload
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any], *, config=None) -> _DummyResponse:
        """Capture the prompt messages and return the configured payload.

        Args:
            messages: Rendered LangChain messages supplied by the handler.

        Returns:
            Dummy response containing the configured JSON payload.
        """
        self.messages = list(messages)
        response = _DummyResponse(json.dumps(self.payload))
        return response


class _FakeGraph:
    """Service graph fake that records invoked state and returns a result."""

    def __init__(self, result: dict[str, Any]) -> None:
        """Create a graph fake.

        Args:
            result: Graph result returned by ``ainvoke``.
        """
        self.result = result
        self.states: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Record state passed by the service and return the fixed result."""
        self.states.append(dict(state))
        return_value = self.result
        return return_value


def _fixture() -> dict[str, Any]:
    """Load the stage fixture file.

    Returns:
        Parsed fixture dictionary.
    """
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
    fixture_data = json.loads(fixture_text)
    return fixture_data


def _case(case_id: str) -> dict[str, Any]:
    """Return one fixture case by identifier.

    Args:
        case_id: Stable case id from the fixture file.

    Returns:
        Case dictionary.
    """
    cases = _fixture()["cases"]
    for item in cases:
        if item["case_id"] == case_id:
            return_value = dict(item)
            return return_value
    raise AssertionError(f"Missing fixture case: {case_id}")


def _character_profile() -> dict[str, Any]:
    """Build a complete character profile for prompt-render tests.

    Returns:
        Character profile containing every field the current prompts read.
    """
    return_value = {
        "name": "Character",
        "global_user_id": "character-1",
        "mood": "neutral",
        "vibe_check": "calm",
        "character_reflection": "quiet baseline",
        "description": "A test character.",
        "personality_brief": {
            "mbti": "INTJ",
            "logic": "precise and restrained",
            "tempo": "short replies",
            "defense": "keeps boundaries",
            "quirks": "dry humor",
            "taboos": "do not invent facts",
        },
        "boundary_profile": {
            "self_integrity": 0.8,
            "control_sensitivity": 0.3,
            "compliance_strategy": "comply",
            "relational_override": 0.25,
            "control_intimacy_misread": 0.2,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.35,
        },
        "linguistic_texture_profile": {
            "hesitation_density": 0.2,
            "fragmentation": 0.25,
            "emotional_leakage": 0.2,
            "rhythmic_bounce": 0.3,
            "direct_assertion": 0.6,
            "softener_density": 0.3,
            "counter_questioning": 0.2,
            "formalism_avoidance": 0.7,
            "abstraction_reframing": 0.4,
            "self_deprecation": 0.1,
        },
    }
    return return_value


def _rag_result() -> dict[str, Any]:
    """Build the current projected RAG payload shape."""
    return_value = {
        "answer": "Use the prior decision.",
        "user_image": {
            "global_user_id": "global-user-1",
            "display_name": "Test User",
            "user_memory_context": empty_user_memory_context(),
        },
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }
    return return_value


def _prompt_message_context(case: dict[str, Any]) -> dict[str, Any]:
    """Project a fixture case into the prompt-facing message context."""
    context = {
        "body_text": case["body_text"],
        "mentions": [],
        "attachments": [],
        "addressed_to_global_user_ids": case["addressed_to_global_user_ids"],
        "broadcast": case["broadcast"],
    }
    if "reply" in case:
        context["reply"] = case["reply"]
    return context


def _message_envelope(case: dict[str, Any]) -> dict[str, Any]:
    """Project a fixture case into a typed message envelope."""
    envelope = {
        "body_text": case["body_text"],
        "raw_wire_text": case["body_text"],
        "mentions": [],
        "attachments": [],
        "addressed_to_global_user_ids": case["addressed_to_global_user_ids"],
        "broadcast": case["broadcast"],
    }
    if "reply" in case:
        envelope["reply"] = case["reply"]
    return envelope


def _base_state(case_id: str = "private_text") -> dict[str, Any]:
    """Build a service/persona state from a fixture case."""
    case = _case(case_id)
    turn_clock = build_turn_clock("2026-05-01 09:00:00")
    return_value = {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "active_turn_platform_message_ids": [case["platform_message_id"]],
        "active_turn_conversation_row_ids": ["conversation-row-1"],
        "platform": case["platform"],
        "platform_message_id": case["platform_message_id"],
        "platform_user_id": case["platform_user_id"],
        "global_user_id": "global-user-1",
        "user_name": case["display_name"],
        "user_input": case["body_text"],
        "message_envelope": _message_envelope(case),
        "prompt_message_context": _prompt_message_context(case),
        "user_multimedia_input": [],
        "user_profile": {
            "relationship_state": 500,
            "semantic_relationship_projection": "steady baseline",
        },
        "platform_bot_id": "bot-1",
        "character_name": "Character",
        "character_profile": _character_profile(),
        "platform_channel_id": case["platform_channel_id"],
        "channel_type": case["channel_type"],
        "channel_name": "Test Channel",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": case.get("reply", {}),
        "should_respond": True,
        "reason_to_respond": "fixture",
        "use_reply_feature": False,
        "channel_topic": "stage baseline",
        "indirect_speech_context": "",
        "debug_modes": {
            "listen_only": False,
            "think_only": False,
            "no_remember": False,
        },
        "conversation_progress": {
            "status": "new_episode",
            "continuity": "sharp_transition",
            "conversation_mode": "casual_chat",
            "episode_phase": "opening",
            "topic_momentum": "stable",
            "current_thread": "baseline",
            "user_goal": "",
            "current_blocker": "",
            "user_state_updates": [],
            "overused_moves": [],
            "open_loops": [],
            "resolved_threads": [],
            "avoid_reopening": [],
            "emotional_trajectory": "neutral",
            "next_affordances": ["continue"],
            "progression_guidance": "continue normally",
        },
        "conversation_episode_state": None,
        "promoted_reflection_context": {},
    }
    return_value["cognitive_episode"] = build_text_chat_cognitive_episode(
        episode_id=(
            f"user_message:{return_value['platform']}:"
            f"{return_value['platform_channel_id']}:"
            f"{return_value['platform_message_id']}"
        ),
        percept_id=(
            f"user_message:{return_value['platform']}:"
            f"{return_value['platform_channel_id']}:"
            f"{return_value['platform_message_id']}:dialog_text:0"
        ),
        storage_timestamp_utc=return_value["storage_timestamp_utc"],
        local_time_context=return_value["local_time_context"],
        user_input=return_value["user_input"],
        platform=return_value["platform"],
        platform_channel_id=return_value["platform_channel_id"],
        channel_type=return_value["channel_type"],
        platform_message_id=return_value["platform_message_id"],
        platform_user_id=return_value["platform_user_id"],
        global_user_id=return_value["global_user_id"],
        user_name=return_value["user_name"],
        active_turn_platform_message_ids=return_value[
            "active_turn_platform_message_ids"
        ],
        active_turn_conversation_row_ids=return_value[
            "active_turn_conversation_row_ids"
        ],
        debug_modes=return_value["debug_modes"],
        output_mode="visible_reply",
        target_addressed_user_ids=return_value["prompt_message_context"][
            "addressed_to_global_user_ids"
        ],
        target_broadcast=return_value["prompt_message_context"]["broadcast"],
    )
    return return_value


def _cognition_state() -> dict[str, Any]:
    """Build a complete cognition-state fixture for prompt rendering."""
    state = _base_state()
    state.update(
        {
            "decontexualized_input": "User asks for the short plan.",
            "referents": [],
            "rag_result": _rag_result(),
            "emotional_appraisal": "steady",
            "interaction_subtext": "routine",
            "internal_monologue": "Answer directly.",
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "boundary_core_assessment": {
                "boundary_issue": "none",
                "boundary_summary": "No boundary issue.",
                "behavior_primary": "comply",
                "behavior_secondary": "none",
                "acceptance": "allow",
                "stance_bias": "confirm",
                "identity_policy": "accept",
                "pressure_policy": "absorb",
                "trajectory": "stable",
            },
            "social_distance": "neutral",
            "emotional_intensity": "low",
            "vibe_check": "routine",
            "relational_dynamic": "stable",
            "rhetorical_strategy": "answer briefly",
            "linguistic_style": "plain",
            "forbidden_phrases": [],
            "content_plan": {
                "visible_goal": "answer",
                "semantic_content": "keep it short",
                "rendering": "short",
            },
            "interaction_style_context": {
                "user_style": {
                    "speech_guidelines": [],
                    "social_guidelines": [],
                    "pacing_guidelines": [],
                    "engagement_guidelines": [],
                    "confidence": "low",
                },
                "application_order": ["user_style"],
            },
        }
    )
    state["cognitive_episode"] = build_text_chat_cognitive_episode(
        episode_id="user_message:debug:direct:message-1",
        percept_id="user_message:debug:direct:message-1:dialog_text:0",
        storage_timestamp_utc=state["storage_timestamp_utc"],
        local_time_context=state["local_time_context"],
        user_input=state["user_input"],
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
        channel_type=state["channel_type"],
        platform_message_id=state["platform_message_id"],
        platform_user_id=state["platform_user_id"],
        global_user_id=state["global_user_id"],
        user_name=state["user_name"],
        active_turn_platform_message_ids=state[
            "active_turn_platform_message_ids"
        ],
        active_turn_conversation_row_ids=state[
            "active_turn_conversation_row_ids"
        ],
        debug_modes=state["debug_modes"],
        output_mode="visible_reply",
        target_addressed_user_ids=state["prompt_message_context"][
            "addressed_to_global_user_ids"
        ],
        target_broadcast=state["prompt_message_context"]["broadcast"],
    )
    return state


def _speak_action_spec() -> dict[str, Any]:
    """Build a valid L2d-selected text-surface action spec."""

    action_spec = {
        "schema_version": "action_spec.v1",
        "kind": SPEAK_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "cognitive_episode",
                "ref_id": "current_cognitive_episode",
                "owner": "cognition",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {"intent": "answer normally"},
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The character should answer visibly.",
    }
    return action_spec


def _text_surface_update() -> dict[str, Any]:
    """Build a minimal selected L3 text-surface update."""

    update = {
        "action_directives": {
            "contextual_directives": {
                "social_distance": "friendly",
                "emotional_intensity": "low",
                "vibe_check": "routine",
                "relational_dynamic": "stable",
            },
            "linguistic_directives": {
                "rhetorical_strategy": "answer briefly",
                "linguistic_style": "plain",
                "accepted_user_preferences": [],
                "content_plan": {
                    "semantic_content": "answer",
                    "rendering": "short",
                },
                "forbidden_phrases": [],
            },
            "visual_directives": {
                "facial_expression": [],
                "body_language": [],
                "gaze_direction": [],
                "visual_vibe": [],
            },
        }
    }
    return update


def _resolver_update(
    *,
    rag_result: dict[str, Any] | None = None,
    action_specs: list[dict[str, Any]] | None = None,
    internal_monologue: str = "answer normally",
    character_intent: str = "PROVIDE",
    logical_stance: str = "CONFIRM",
) -> dict[str, Any]:
    """Build a patched resolver result for persona graph tests."""

    update = {
        "rag_result": rag_result or _rag_result(),
        "internal_monologue": internal_monologue,
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": character_intent,
        "logical_stance": logical_stance,
        "judgment_note": "ok",
        "social_distance": "",
        "emotional_intensity": "",
        "vibe_check": "",
        "relational_dynamic": "",
        "action_specs": action_specs or [],
    }
    return update


def _dialog_state() -> dict[str, Any]:
    """Build a dialog-agent state for generator render checks."""
    cognition_state = _cognition_state()
    return_value = {
        "internal_monologue": cognition_state["internal_monologue"],
        "action_directives": {
            "contextual_directives": {
                "social_distance": cognition_state["social_distance"],
                "emotional_intensity": cognition_state["emotional_intensity"],
                "vibe_check": cognition_state["vibe_check"],
                "relational_dynamic": cognition_state["relational_dynamic"],
            },
            "linguistic_directives": {
                "rhetorical_strategy": cognition_state["rhetorical_strategy"],
                "linguistic_style": cognition_state["linguistic_style"],
                "accepted_user_preferences": [],
                "content_plan": cognition_state["content_plan"],
                "forbidden_phrases": [],
            },
        },
        "chat_history_wide": cognition_state["chat_history_wide"],
        "chat_history_recent": cognition_state["chat_history_recent"],
        "platform_user_id": cognition_state["platform_user_id"],
        "platform_bot_id": cognition_state["platform_bot_id"],
        "global_user_id": cognition_state["global_user_id"],
        "user_name": cognition_state["user_name"],
        "user_profile": cognition_state["user_profile"],
        "character_profile": cognition_state["character_profile"],
        "dialog_usage_mode": "live_visible_reply",
        "final_dialog": ["ok"],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
    }
    return return_value


def _assert_prompt_messages(
    fake_llm: _CaptureLLM,
    required_payload_keys: set[str],
) -> dict[str, Any]:
    """Assert a mocked LLM received system and JSON human messages.

    Args:
        fake_llm: Fake LLM that captured messages.
        required_payload_keys: Keys expected in the human-message JSON payload.

    Returns:
        Parsed human-message payload.
    """
    assert len(fake_llm.messages) >= 2
    system_content = fake_llm.messages[0].content
    human_content = fake_llm.messages[1].content
    assert isinstance(system_content, str)
    assert system_content.strip()
    payload = json.loads(human_content)
    assert required_payload_keys <= set(payload)
    assert "cognitive_episode" not in payload
    assert "prompt_key" not in payload
    assert "trigger_source" not in payload
    assert "input_sources" not in payload
    return payload


def _selector_tracker() -> tuple[list[dict[str, Any]], Callable[..., Any]]:
    """Build a selector wrapper that records cognition prompt decisions.

    Returns:
        A mutable list of selector calls and a selector-compatible callable.
    """
    selector_calls: list[dict[str, Any]] = []

    def _tracked_selector(*, episode: Any, stage: Any) -> dict[str, Any]:
        """Record the selected prompt variant and return the real selection.

        Args:
            episode: Cognitive episode passed by the handler.
            stage: Cognition stage requesting a prompt variant.

        Returns:
            Prompt selection returned by the production selector.
        """
        selection = select_cognition_prompt_variant(
            episode=episode,
            stage=stage,
        )
        selector_calls.append(dict(selection))
        return_value = selection
        return return_value

    return_value = selector_calls, _tracked_selector
    return return_value


async def _reset_queue_state() -> None:
    """Reset process-local queue state around service endpoint tests."""
    await service_module._stop_chat_input_worker()
    service_module._chat_input_queue.reset_for_test()


def _graph_result() -> dict[str, Any]:
    """Build a successful service graph result with consolidation state."""
    consolidation_state = _base_state()
    consolidation_state.update(
        {
            "decontexualized_input": "please remember this",
            "rag_result": _rag_result(),
            "internal_monologue": "test",
            "action_directives": {
                "linguistic_directives": {"content_plan": {}},
                "contextual_directives": {},
            },
            "interaction_subtext": "",
            "emotional_appraisal": "",
            "character_intent": "PROVIDE",
            "logical_stance": "CONFIRM",
            "final_dialog": ["ok"],
        }
    )
    return_value = {
        "should_respond": True,
        "use_reply_feature": False,
        "final_dialog": ["ok"],
        "target_addressed_user_ids": ["global-user-1"],
        "target_broadcast": False,
        "future_promises": [],
        "consolidation_state": consolidation_state,
    }
    return return_value


def _assert_text_chat_cognitive_episode(
    state: dict[str, Any],
    *,
    output_mode: str,
) -> None:
    """Assert the inert episode mirrors existing service graph fields.

    Args:
        state: Service graph state captured by the fake graph.
        output_mode: Expected episode output mode for the debug-mode path.
    """

    episode = state["cognitive_episode"]
    channel_reference = state["platform_channel_id"] or "direct"
    episode_id = (
        f"user_message:{state['platform']}:"
        f"{channel_reference}:{state['platform_message_id']}"
    )

    assert episode["episode_id"] == episode_id
    assert episode["trigger_source"] == "user_message"
    assert episode["input_sources"] == ["dialog_text"]
    assert episode["output_mode"] == output_mode
    assert episode["percepts"][0]["percept_id"] == f"{episode_id}:dialog_text:0"
    assert episode["percepts"][0]["content"] == state["user_input"]
    assert episode["target_scope"]["target_addressed_user_ids"] == (
        state["prompt_message_context"]["addressed_to_global_user_ids"]
    )
    assert episode["target_scope"]["target_broadcast"] == (
        state["prompt_message_context"]["broadcast"]
    )
    assert episode["origin_metadata"]["debug_modes"] == state["debug_modes"]
    assert episode["origin_metadata"]["active_turn_platform_message_ids"] == (
        state["active_turn_platform_message_ids"]
    )
    assert episode["origin_metadata"]["active_turn_conversation_row_ids"] == (
        state["active_turn_conversation_row_ids"]
    )


def _chat_request(
    case_id: str = "private_text",
    *,
    debug_modes: service_module.DebugModesIn | None = None,
) -> service_module.ChatRequest:
    """Build a service chat request from a fixture case.

    Args:
        case_id: Fixture case id.
        debug_modes: Optional debug flags.

    Returns:
        ChatRequest ready for ``service.chat``.
    """
    case = _case(case_id)
    request = service_module.ChatRequest(
        platform=case["platform"],
        platform_channel_id=case["platform_channel_id"],
        channel_type=case["channel_type"],
        platform_message_id=case["platform_message_id"],
        platform_user_id=case["platform_user_id"],
        platform_bot_id="bot-1",
        display_name=case["display_name"],
        channel_name="Test Channel",
        message_envelope=_message_envelope(case),
        debug_modes=debug_modes or service_module.DebugModesIn(),
    )
    return request


def _patch_service_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    graph: _FakeGraph,
) -> dict[str, AsyncMock]:
    """Patch service dependencies outside the queue worker contract.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        graph: Fake graph object installed as the service graph.

    Returns:
        Patched async mocks keyed by contract name.
    """
    save_assistant_message = AsyncMock()
    progress_recorder = AsyncMock()
    consolidation_runner = AsyncMock()
    save_conversation = AsyncMock()
    for event_function_name in (
        "record_database_operation_event",
        "record_pipeline_turn_event",
        "record_queue_intake_event",
        "record_runtime_error_event",
    ):
        monkeypatch.setattr(
            service_module.event_logging,
            event_function_name,
            AsyncMock(),
        )
    monkeypatch.setattr(
        service_module,
        "COGNITION_VISUAL_DIRECTIVES_ENABLED",
        True,
    )
    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        _character_profile(),
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {
            "mood": "neutral",
            "vibe_check": "calm",
            "character_reflection": "quiet baseline",
        },
    )
    monkeypatch.setattr(
        service_module,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "mood": "neutral",
            "vibe_check": "calm",
            "character_reflection": "quiet baseline",
        }),
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-1"),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_global_user_id",
        AsyncMock(return_value="global-user-1"),
    )
    monkeypatch.setattr(
        service_module,
        "get_user_profile",
        AsyncMock(return_value={
            "relationship_state": 500,
            "semantic_relationship_projection": "steady baseline",
        }),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "_hydrate_reply_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(service_module, "save_conversation", save_conversation)
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        save_assistant_message,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        progress_recorder,
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        consolidation_runner,
    )
    monkeypatch.setattr(service_module, "_graph", graph)
    mocks = {
        "save_assistant_message": save_assistant_message,
        "progress_recorder": progress_recorder,
        "consolidation_runner": consolidation_runner,
        "save_conversation": save_conversation,
    }
    return mocks


async def _wait_for_mock_await(
    mock: AsyncMock,
    *,
    await_count: int = 1,
) -> None:
    """Wait until post-response queue work reaches a patched async seam."""

    for _ in range(BACKGROUND_HANDOFF_WAIT_POLLS):
        if mock.await_count >= await_count:
            return
        await asyncio.sleep(BACKGROUND_HANDOFF_WAIT_SECONDS)

    assert mock.await_count >= await_count


def test_stage_00_fixture_covers_required_cases() -> None:
    """The reusable fixture must name every required current-chat case."""
    fixture_data = _fixture()
    case_ids = {item["case_id"] for item in fixture_data["cases"]}

    assert {
        "private_text",
        "group_text",
        "reply",
        "silence",
        "rag_skip",
        "rag_hit",
        "think_only",
        "no_remember",
        "listen_only",
    } <= case_ids
    corpus = fixture_data["frozen_evidence_corpus"]
    assert corpus["corpus_id"] == "stage_00_synthetic_rag_equivalence_v1"
    assert len(corpus["known_facts"]) == 3


@pytest.mark.asyncio
async def test_persona_graph_response_route_preserves_consolidation_snapshot() -> None:
    """A normal persona turn should expose dialog and consolidation state."""
    state = _base_state("private_text")
    rag_result = _rag_result()

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={
                "decontexualized_input": "hello there",
                "referents": [],
            },
        ) as decontextualizer,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value=_resolver_update(
                rag_result=rag_result,
                action_specs=[_speak_action_spec()],
            ),
        ) as resolver,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_l3_text_surface_handler",
            new_callable=AsyncMock,
            return_value=_text_surface_update(),
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["ok"],
                "target_addressed_user_ids": ["global-user-1"],
                "target_broadcast": False,
            },
        ) as dialog,
    ):
        result = await supervisor_module.persona_supervisor2(state)

    decontextualizer.assert_awaited_once()
    resolver.assert_awaited_once()
    dialog.assert_awaited_once()
    assert result["should_respond"] is True
    assert result["final_dialog"] == ["ok"]
    assert result["target_addressed_user_ids"] == ["global-user-1"]
    assert result["target_broadcast"] is False
    assert result["future_promises"] == []
    consolidation_state = result["consolidation_state"]
    assert consolidation_state["decontexualized_input"] == "hello there"
    assert consolidation_state["rag_result"] == rag_result
    assert consolidation_state["final_dialog"] == ["ok"]


@pytest.mark.asyncio
async def test_persona_graph_silence_route_skips_dialog() -> None:
    """Cognition-selected silence should leave no visible dialog."""
    state = _base_state("silence")

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={
                "decontexualized_input": "no response needed",
                "referents": [],
            },
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value=_resolver_update(
                internal_monologue="stay quiet",
                character_intent="DISMISS",
                logical_stance="REFUSE",
            ),
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["should not send"],
                "target_addressed_user_ids": ["global-user-1"],
                "target_broadcast": False,
            },
        ) as dialog,
    ):
        result = await supervisor_module.persona_supervisor2(state)

    assert result["should_respond"] is False
    assert result["final_dialog"] == []
    assert result["target_addressed_user_ids"] == []
    assert result["target_broadcast"] is False
    assert result["consolidation_state"]["should_respond"] is False
    dialog.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_graph_scopes_group_history_after_decontextualizer() -> None:
    """Full channel history should stop before the resolver stage."""
    state = _base_state("group_text")
    state["chat_history_wide"] = [
        {
            "role": "user",
            "platform_user_id": "platform-user-1",
            "global_user_id": "global-user-1",
            "body_text": "current user secret",
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-05-01T09:00:00+12:00",
        },
        {
            "role": "assistant",
            "platform_user_id": "bot-1",
            "global_user_id": "character-1",
            "body_text": "current user reply",
            "addressed_to_global_user_ids": ["global-user-1"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-05-01T09:01:00+12:00",
        },
        {
            "role": "user",
            "platform_user_id": "platform-user-2",
            "global_user_id": "global-user-2",
            "body_text": "other user secret",
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-05-01T09:02:00+12:00",
        },
        {
            "role": "assistant",
            "platform_user_id": "bot-1",
            "global_user_id": "character-1",
            "body_text": "other user reply",
            "addressed_to_global_user_ids": ["global-user-2"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-05-01T09:03:00+12:00",
        },
    ]
    state["chat_history_recent"] = list(state["chat_history_wide"])

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={
                "decontexualized_input": "group message",
                "referents": [],
            },
        ) as decontextualizer,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value=_resolver_update(action_specs=[_speak_action_spec()]),
        ) as resolver,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_l3_text_surface_handler",
            new_callable=AsyncMock,
            return_value=_text_surface_update(),
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["ok"],
                "target_addressed_user_ids": ["global-user-1"],
                "target_broadcast": False,
            },
        ),
    ):
        await supervisor_module.persona_supervisor2(state)

    decontextualizer_state = decontextualizer.await_args.args[0]
    resolver_state = resolver.await_args.args[0]
    assert [
        row["body_text"]
        for row in decontextualizer_state["chat_history_recent"]
    ] == [
        "current user secret",
        "current user reply",
        "other user secret",
        "other user reply",
    ]
    assert [
        row["body_text"]
        for row in resolver_state["chat_history_recent"]
    ] == [
        "current user secret",
        "current user reply",
    ]


@pytest.mark.asyncio
async def test_service_normal_response_tracks_delivery_and_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visible service responses should track assistant delivery and handoff."""
    await _reset_queue_state()
    graph = _FakeGraph(_graph_result())
    mocks = _patch_service_dependencies(monkeypatch, graph)

    response = await service_module.chat(_chat_request(), BackgroundTasks())

    assert response.messages == ["ok"]
    assert response.delivery_tracking_id
    assert graph.states[0]["debug_modes"] == {
        "listen_only": False,
        "think_only": False,
        "no_remember": False,
    }
    _assert_text_chat_cognitive_episode(
        graph.states[0],
        output_mode="visible_reply",
    )
    await _wait_for_mock_await(mocks["consolidation_runner"])
    mocks["save_assistant_message"].assert_awaited_once()
    saved_result = mocks["save_assistant_message"].await_args.args[0]
    assert saved_result["delivery_tracking_id"] == response.delivery_tracking_id
    mocks["progress_recorder"].assert_awaited_once()
    mocks["consolidation_runner"].assert_awaited_once()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_service_config_disabled_visual_directives_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service config should seed the internal visual-directive skip flag."""
    await _reset_queue_state()
    graph = _FakeGraph(_graph_result())
    _patch_service_dependencies(monkeypatch, graph)
    monkeypatch.setattr(
        service_module,
        "COGNITION_VISUAL_DIRECTIVES_ENABLED",
        False,
    )

    response = await service_module.chat(_chat_request(), BackgroundTasks())

    assert response.messages == ["ok"]
    assert graph.states[0]["debug_modes"] == {
        "listen_only": False,
        "think_only": False,
        "no_remember": False,
        "no_visual_directives": True,
    }
    assert graph.states[0]["cognitive_episode"]["origin_metadata"][
        "debug_modes"
    ] == graph.states[0]["debug_modes"]
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_service_think_only_suppresses_visible_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """think_only should suppress returned text without skipping thinking."""
    await _reset_queue_state()
    graph = _FakeGraph(_graph_result())
    mocks = _patch_service_dependencies(monkeypatch, graph)

    response = await service_module.chat(
        _chat_request(
            "think_only",
            debug_modes=service_module.DebugModesIn(think_only=True),
        ),
        BackgroundTasks(),
    )

    assert response.messages == []
    assert response.delivery_tracking_id == ""
    assert graph.states[0]["debug_modes"]["think_only"] is True
    _assert_text_chat_cognitive_episode(
        graph.states[0],
        output_mode="think_only",
    )
    await _wait_for_mock_await(mocks["consolidation_runner"])
    mocks["save_assistant_message"].assert_awaited_once()
    mocks["progress_recorder"].assert_awaited_once()
    mocks["consolidation_runner"].assert_awaited_once()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_service_no_remember_skips_consolidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """no_remember should skip consolidation but preserve response delivery."""
    await _reset_queue_state()
    graph = _FakeGraph(_graph_result())
    mocks = _patch_service_dependencies(monkeypatch, graph)

    response = await service_module.chat(
        _chat_request(
            "no_remember",
            debug_modes=service_module.DebugModesIn(no_remember=True),
        ),
        BackgroundTasks(),
    )

    assert response.messages == ["ok"]
    assert response.delivery_tracking_id
    assert graph.states[0]["debug_modes"]["no_remember"] is True
    _assert_text_chat_cognitive_episode(
        graph.states[0],
        output_mode="visible_reply",
    )
    await _wait_for_mock_await(mocks["progress_recorder"])
    mocks["save_assistant_message"].assert_awaited_once()
    mocks["progress_recorder"].assert_awaited_once()
    mocks["consolidation_runner"].assert_not_awaited()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_service_listen_only_skips_graph_but_persists_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """listen_only should persist the inbound row and skip graph work."""
    await _reset_queue_state()
    graph = _FakeGraph(_graph_result())
    mocks = _patch_service_dependencies(monkeypatch, graph)

    response = await service_module.chat(
        _chat_request(
            "listen_only",
            debug_modes=service_module.DebugModesIn(listen_only=True),
        ),
        BackgroundTasks(),
    )

    assert response.messages == []
    assert response.delivery_tracking_id == ""
    assert graph.states == []
    mocks["save_conversation"].assert_awaited_once()
    mocks["save_assistant_message"].assert_not_awaited()
    mocks["progress_recorder"].assert_not_awaited()
    mocks["consolidation_runner"].assert_not_awaited()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_rag_skip_preserves_full_projected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipped RAG still returns the shape consumed by cognition."""
    state = _base_state("rag_skip")
    state["decontexualized_input"] = "what does that mean"
    state["referents"] = [
        {
            "phrase": "that",
            "referent_role": "object",
            "status": "unresolved",
        }
    ]

    async def _fail_local_context(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("RAG3 resolver should not run")

    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        _fail_local_context,
    )

    rag_result = await supervisor_module.run_rag_evidence_for_persona_state(
        state,
        agent_name="resolver_rag_evidence",
    )

    assert rag_result["answer"] == ""
    assert rag_result["user_image"]["user_memory_context"] == (
        empty_user_memory_context()
    )
    assert rag_result["user_memory_unit_candidates"] == []
    assert rag_result["character_image"] == {}
    assert rag_result["third_party_profiles"] == []
    assert rag_result["memory_evidence"] == []
    assert rag_result["recall_evidence"] == []
    assert rag_result["conversation_evidence"] == []
    assert rag_result["external_evidence"] == []
    assert rag_result["supervisor_trace"]["unknown_slots"] == []
    assert rag_result["supervisor_trace"]["loop_count"] == 0
    assert rag_result["supervisor_trace"]["dispatched"] == []


def test_frozen_evidence_corpus_projects_rag_hit_shape() -> None:
    """The frozen evidence corpus should project into current RAG shape."""
    corpus = _fixture()["frozen_evidence_corpus"]
    expected = corpus["expected_projection"]

    rag_result = project_known_facts(
        corpus["known_facts"],
        current_user_id=expected["current_user_id"],
        character_user_id=expected["character_user_id"],
        answer=expected["answer"],
        unknown_slots=[],
        loop_count=3,
    )

    assert rag_result["answer"] == expected["answer"]
    assert rag_result["conversation_evidence"] == [
        "结论：Prior discussion settled on a short plan.\n不确定性：无"
    ]
    assert rag_result["memory_evidence"][0]["source_system"] == (
        "user_memory_units"
    )
    assert rag_result["user_memory_unit_candidates"][0]["unit_id"] == (
        "unit-plan-1"
    )
    assert rag_result["external_evidence"][0]["url"] == (
        "https://example.com/stage-00"
    )
    assert len(rag_result["supervisor_trace"]["dispatched"]) == 3


@pytest.mark.asyncio
async def test_existing_cognition_and_dialog_prompts_render_with_mocked_llms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing L1/L2/L3 and dialog prompt paths should render deterministically."""
    state = _cognition_state()
    selector_calls, tracked_selector = _selector_tracker()
    monkeypatch.setattr(
        l1_module,
        "select_cognition_prompt_variant",
        tracked_selector,
    )
    monkeypatch.setattr(
        l2_module,
        "select_cognition_prompt_variant",
        tracked_selector,
    )
    monkeypatch.setattr(
        l2c2_module,
        "select_cognition_prompt_variant",
        tracked_selector,
    )
    monkeypatch.setattr(
        l3_module,
        "select_cognition_prompt_variant",
        tracked_selector,
    )

    subconscious_llm = _CaptureLLM({
        "emotional_appraisal": "steady",
        "interaction_subtext": "routine",
    })
    monkeypatch.setattr(l1_module, "_subconscious_llm", bind_test_llm(subconscious_llm, "subconscious_llm"))
    subconscious_result = await l1_module.call_cognition_subconscious(state)
    assert subconscious_result["emotional_appraisal"] == "steady"
    _assert_prompt_messages(
        subconscious_llm,
        {"user_input", "indirect_speech_context"},
    )

    conscious_llm = _CaptureLLM({
        "internal_monologue": "Answer directly.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
    })
    monkeypatch.setattr(l2_module, "_conscious_llm", bind_test_llm(conscious_llm, "conscious_llm"))
    consciousness_result = await l2_module.call_cognition_consciousness(state)
    assert consciousness_result["logical_stance"] == "CONFIRM"
    _assert_prompt_messages(
        conscious_llm,
        {
            "decontextualized_input",
            "rag_result",
            "user_memory_context",
            "promoted_reflection_context",
        },
    )

    boundary_llm = _CaptureLLM({
        "boundary_issue": "none",
        "boundary_summary": "No boundary issue.",
        "behavior_primary": "comply",
        "behavior_secondary": "none",
        "acceptance": "allow",
        "stance_bias": "confirm",
        "identity_policy": "accept",
        "pressure_policy": "absorb",
        "trajectory": "stable",
    })
    monkeypatch.setattr(l2_module, "_boundary_core_llm", bind_test_llm(boundary_llm, "boundary_core_llm"))
    boundary_result = await l2_module.call_boundary_core_agent(state)
    assert boundary_result["boundary_core_assessment"]["acceptance"] == "allow"
    _assert_prompt_messages(
        boundary_llm,
        {"decontextualized_input", "relationship_state_context"},
    )

    judgment_llm = _CaptureLLM({
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "stable",
    })
    monkeypatch.setattr(l2_module, "_judgement_core_llm", bind_test_llm(judgment_llm, "judgement_core_llm"))
    judgment_result = await l2_module.call_judgment_core_agent(state)
    assert judgment_result["character_intent"] == "PROVIDE"
    _assert_prompt_messages(
        judgment_llm,
        {
            "referents",
            "internal_monologue_candidate",
            "boundary_issue",
        },
    )

    contextual_llm = _CaptureLLM({
        "social_distance": "neutral",
        "emotional_intensity": "low",
        "vibe_check": "routine",
        "relational_dynamic": "stable",
    })
    monkeypatch.setattr(l2c2_module, "_contextual_agent_llm", bind_test_llm(contextual_llm, "contextual_agent_llm"))
    contextual_result = await l2c2_module.call_social_context_appraisal(state)
    assert contextual_result["social_distance"] == "neutral"
    _assert_prompt_messages(
        contextual_llm,
        {"decontexualized_input", "boundary_core_assessment"},
    )

    style_llm = _CaptureLLM({
        "rhetorical_strategy": "answer briefly",
        "linguistic_style": "plain",
        "forbidden_phrases": [],
    })
    monkeypatch.setattr(l3_module, "_style_agent_llm", bind_test_llm(style_llm, "style_agent_llm"))
    style_result = await l3_module.call_style_agent(state)
    assert style_result["rhetorical_strategy"] == "answer briefly"
    _assert_prompt_messages(
        style_llm,
        {"internal_monologue", "interaction_style_context"},
    )

    plan_llm = _CaptureLLM({
        "content_plan": {
            "visible_goal": "answer",
            "semantic_content": "keep it short",
            "rendering": "short",
        },
    })
    monkeypatch.setattr(l3_module, "_content_plan_agent_llm", bind_test_llm(plan_llm, "content_plan_agent_llm"))
    plan_result = await l3_module.call_content_plan_agent(state)
    assert plan_result["content_plan"]["semantic_content"] == "keep it short"
    _assert_prompt_messages(
        plan_llm,
        {"decontexualized_input", "referents", "rag_result"},
    )

    preference_llm = _CaptureLLM({"accepted_user_preferences": []})
    monkeypatch.setattr(l3_module, "_preference_adapter_llm", bind_test_llm(preference_llm, "preference_adapter_llm"))
    preference_result = await l3_module.call_preference_adapter(state)
    assert preference_result["accepted_user_preferences"] == []
    _assert_prompt_messages(
        preference_llm,
        {
            "decontexualized_input",
            "content_plan",
            "user_memory_context",
        },
    )

    visual_llm = _CaptureLLM({
        "facial_expression": ["neutral"],
        "body_language": ["still"],
        "gaze_direction": ["forward"],
        "visual_vibe": ["plain"],
    })
    monkeypatch.setattr(l3_module, "_visual_agent_llm", bind_test_llm(visual_llm, "visual_agent_llm"))
    visual_result = await l3_module.call_visual_agent(state)
    assert visual_result["visual_vibe"] == ["plain"]
    _assert_prompt_messages(
        visual_llm,
        {"prompt_message_context", "content_plan", "conversation_progress"},
    )
    assert [selection["stage"] for selection in selector_calls] == [
        "l1_subconscious",
        "l2a_conscious_framing",
        "l2b_boundary_appraisal",
        "l2c1_judgment_synthesis",
        "l2c2_social_context_appraisal",
        "l3_style_agent",
        "l3_content_plan_agent",
        "l3_preference_adapter",
        "l3_visual_agent",
    ]
    assert {selection["variant"] for selection in selector_calls} == {
        "text_chat_user_message"
    }
    assert [
        selection["prompt_key"]
        for selection in selector_calls
    ] == [
        f"{selection['stage']}.text_chat_user_message"
        for selection in selector_calls
    ]
    assert {selection["trigger_source"] for selection in selector_calls} == {
        "user_message"
    }
    assert [selection["input_sources"] for selection in selector_calls] == [
        ["dialog_text"],
        ["dialog_text"],
        ["dialog_text"],
        ["dialog_text"],
        ["dialog_text"],
        ["dialog_text"],
        ["dialog_text"],
        ["dialog_text"],
        ["dialog_text"],
    ]
    assert {selection["output_mode"] for selection in selector_calls} == {
        "visible_reply"
    }

    dialog_state = _dialog_state()
    generator_llm = _CaptureLLM({"final_dialog": ["ok"]})
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    generator_result = await dialog_module.dialog_generator(dialog_state)
    assert generator_result["final_dialog"] == ["ok"]
    generator_payload = _assert_prompt_messages(
        generator_llm,
        {"linguistic_directives", "contextual_directives", "user_name"},
    )
    assert set(generator_payload) == {
        "linguistic_directives",
        "contextual_directives",
        "user_name",
    }

@pytest.mark.asyncio
async def test_l2_consciousness_receives_local_time_for_same_day_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2 needs the turn-local date before interpreting appointment timing."""

    state = _cognition_state()
    turn_clock = build_turn_clock("2026-05-25 16:14:00")
    user_input = "还记得今天的约定吗？"
    state["storage_timestamp_utc"] = turn_clock["storage_timestamp_utc"]
    state["local_time_context"] = turn_clock["local_time_context"]
    state["user_input"] = user_input
    state["decontexualized_input"] = user_input
    state["cognitive_episode"] = build_text_chat_cognitive_episode(
        episode_id="user_message:qq:same-day-commitment:message-1",
        percept_id=(
            "user_message:qq:same-day-commitment:message-1:dialog_text:0"
        ),
        storage_timestamp_utc=state["storage_timestamp_utc"],
        local_time_context=state["local_time_context"],
        user_input=user_input,
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
        channel_type=state["channel_type"],
        platform_message_id=state["platform_message_id"],
        platform_user_id=state["platform_user_id"],
        global_user_id=state["global_user_id"],
        user_name=state["user_name"],
        active_turn_platform_message_ids=state[
            "active_turn_platform_message_ids"
        ],
        active_turn_conversation_row_ids=state[
            "active_turn_conversation_row_ids"
        ],
        debug_modes=state["debug_modes"],
        output_mode="visible_reply",
        target_addressed_user_ids=state["prompt_message_context"][
            "addressed_to_global_user_ids"
        ],
        target_broadcast=state["prompt_message_context"]["broadcast"],
    )
    state["rag_result"]["answer"] = (
        "已确认约定：2026-05-25 17:00 在学校大门口交接提拉米苏。"
    )

    conscious_llm = _CaptureLLM({
        "internal_monologue": "今天下午五点还有提拉米苏交接约定。",
        "logical_stance": "CONFIRM",
        "character_intent": "BANTAR",
    })
    monkeypatch.setattr(l2_module, "_conscious_llm", bind_test_llm(conscious_llm, "conscious_llm"))

    result = await l2_module.call_cognition_consciousness(state)

    assert result["logical_stance"] == "CONFIRM"
    payload = _assert_prompt_messages(
        conscious_llm,
        {
            "decontextualized_input",
            "local_time_context",
            "rag_result",
        },
    )
    assert payload["local_time_context"] == {
        "current_local_datetime": "2026-05-25 16:14",
        "current_local_weekday": "Monday",
    }
