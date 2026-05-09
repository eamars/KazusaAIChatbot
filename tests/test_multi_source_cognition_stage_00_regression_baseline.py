"""Deterministic regression baseline for the current chat workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2 as supervisor_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l1 as l1_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as l2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import (
    project_known_facts,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.time_context import build_character_time_context


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "multi_source_cognition_stage_00_cases.json"
)


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

    async def ainvoke(self, messages: list[Any]) -> _DummyResponse:
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
        "global_vibe": "calm",
        "reflection_summary": "quiet baseline",
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
    timestamp = "2026-05-01T09:00:00+12:00"
    return_value = {
        "timestamp": timestamp,
        "time_context": build_character_time_context(timestamp),
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
            "affinity": 500,
            "last_relationship_insight": "steady baseline",
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
            "expression_willingness": "open",
            "rhetorical_strategy": "answer briefly",
            "linguistic_style": "plain",
            "forbidden_phrases": [],
            "content_anchors": [
                "[DECISION] answer",
                "[ANSWER] keep it short",
                "[SCOPE] short",
            ],
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
    return state


def _dialog_state() -> dict[str, Any]:
    """Build a dialog-agent state for generator and evaluator render checks."""
    cognition_state = _cognition_state()
    return_value = {
        "internal_monologue": cognition_state["internal_monologue"],
        "action_directives": {
            "contextual_directives": {
                "social_distance": cognition_state["social_distance"],
                "emotional_intensity": cognition_state["emotional_intensity"],
                "vibe_check": cognition_state["vibe_check"],
                "relational_dynamic": cognition_state["relational_dynamic"],
                "expression_willingness": cognition_state["expression_willingness"],
            },
            "linguistic_directives": {
                "rhetorical_strategy": cognition_state["rhetorical_strategy"],
                "linguistic_style": cognition_state["linguistic_style"],
                "accepted_user_preferences": [],
                "content_anchors": cognition_state["content_anchors"],
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
        "messages": [],
        "should_stop": False,
        "retry": 0,
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
    return payload


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
                "linguistic_directives": {"content_anchors": []},
                "contextual_directives": {"expression_willingness": "open"},
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
            "global_vibe": "calm",
            "reflection_summary": "quiet baseline",
        },
    )
    monkeypatch.setattr(
        service_module,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "quiet baseline",
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
            "affinity": 500,
            "last_relationship_insight": "steady baseline",
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
            "kazusa_ai_chatbot.nodes.persona_supervisor2.stage_1_research",
            new_callable=AsyncMock,
            return_value={"rag_result": rag_result},
        ) as research,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_subgraph",
            new_callable=AsyncMock,
            return_value={
                "internal_monologue": "answer normally",
                "action_directives": {},
                "interaction_subtext": "",
                "emotional_appraisal": "",
                "character_intent": "PROVIDE",
                "logical_stance": "CONFIRM",
            },
        ) as cognition,
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
    research.assert_awaited_once()
    cognition.assert_awaited_once()
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
            "kazusa_ai_chatbot.nodes.persona_supervisor2.stage_1_research",
            new_callable=AsyncMock,
            return_value={"rag_result": _rag_result()},
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_subgraph",
            new_callable=AsyncMock,
            return_value={
                "internal_monologue": "stay quiet",
                "action_directives": {
                    "contextual_directives": {
                        "expression_willingness": "silent",
                    },
                },
                "interaction_subtext": "",
                "emotional_appraisal": "",
                "character_intent": "DISMISS",
                "logical_stance": "REFUSE",
            },
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
    """Full channel history should stop before RAG and cognition stages."""
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
            "kazusa_ai_chatbot.nodes.persona_supervisor2.stage_1_research",
            new_callable=AsyncMock,
            return_value={"rag_result": _rag_result()},
        ) as research,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_subgraph",
            new_callable=AsyncMock,
            return_value={
                "internal_monologue": "answer normally",
                "action_directives": {},
                "interaction_subtext": "",
                "emotional_appraisal": "",
                "character_intent": "PROVIDE",
                "logical_stance": "CONFIRM",
            },
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
    research_state = research.await_args.args[0]
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
        for row in research_state["chat_history_recent"]
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
    mocks["save_assistant_message"].assert_awaited_once()
    saved_result = mocks["save_assistant_message"].await_args.args[0]
    assert saved_result["delivery_tracking_id"] == response.delivery_tracking_id
    mocks["progress_recorder"].assert_awaited_once()
    mocks["consolidation_runner"].assert_awaited_once()
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

    async def _fail_rag_supervisor(
        *,
        original_query: str,
        character_name: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        raise AssertionError("RAG supervisor should not run")

    monkeypatch.setattr(
        supervisor_module,
        "call_rag_supervisor",
        _fail_rag_supervisor,
    )

    result = await supervisor_module.stage_1_research(state)
    rag_result = result["rag_result"]

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
        "Prior discussion settled on a short plan."
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

    subconscious_llm = _CaptureLLM({
        "emotional_appraisal": "steady",
        "interaction_subtext": "routine",
    })
    monkeypatch.setattr(l1_module, "_subconscious_llm", subconscious_llm)
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
    monkeypatch.setattr(l2_module, "_conscious_llm", conscious_llm)
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
    monkeypatch.setattr(l2_module, "_boundary_core_llm", boundary_llm)
    boundary_result = await l2_module.call_boundary_core_agent(state)
    assert boundary_result["boundary_core_assessment"]["acceptance"] == "allow"
    _assert_prompt_messages(
        boundary_llm,
        {"decontextualized_input", "affinity_context"},
    )

    judgment_llm = _CaptureLLM({
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "stable",
    })
    monkeypatch.setattr(l2_module, "_judgement_core_llm", judgment_llm)
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
        "expression_willingness": "open",
    })
    monkeypatch.setattr(l3_module, "_contextual_agent_llm", contextual_llm)
    contextual_result = await l3_module.call_contextual_agent(state)
    assert contextual_result["expression_willingness"] == "open"
    _assert_prompt_messages(
        contextual_llm,
        {"decontexualized_input", "boundary_core_assessment"},
    )

    style_llm = _CaptureLLM({
        "rhetorical_strategy": "answer briefly",
        "linguistic_style": "plain",
        "forbidden_phrases": [],
    })
    monkeypatch.setattr(l3_module, "_style_agent_llm", style_llm)
    style_result = await l3_module.call_style_agent(state)
    assert style_result["rhetorical_strategy"] == "answer briefly"
    _assert_prompt_messages(
        style_llm,
        {"internal_monologue", "interaction_style_context"},
    )

    anchor_llm = _CaptureLLM({
        "content_anchors": [
            "[DECISION] answer",
            "[ANSWER] keep it short",
            "[SCOPE] short",
        ],
    })
    monkeypatch.setattr(l3_module, "_content_anchor_agent_llm", anchor_llm)
    anchor_result = await l3_module.call_content_anchor_agent(state)
    assert anchor_result["content_anchors"][0].startswith("[DECISION]")
    _assert_prompt_messages(
        anchor_llm,
        {"decontexualized_input", "referents", "rag_result"},
    )

    preference_llm = _CaptureLLM({"accepted_user_preferences": []})
    monkeypatch.setattr(l3_module, "_preference_adapter_llm", preference_llm)
    preference_result = await l3_module.call_preference_adapter(state)
    assert preference_result["accepted_user_preferences"] == []
    _assert_prompt_messages(
        preference_llm,
        {
            "decontexualized_input",
            "content_anchors",
            "user_memory_context",
        },
    )

    visual_llm = _CaptureLLM({
        "facial_expression": ["neutral"],
        "body_language": ["still"],
        "gaze_direction": ["forward"],
        "visual_vibe": ["plain"],
    })
    monkeypatch.setattr(l3_module, "_visual_agent_llm", visual_llm)
    visual_result = await l3_module.call_visual_agent(state)
    assert visual_result["visual_vibe"] == ["plain"]
    _assert_prompt_messages(
        visual_llm,
        {"prompt_message_context", "content_anchors", "conversation_progress"},
    )

    dialog_state = _dialog_state()
    generator_llm = _CaptureLLM({"final_dialog": ["ok"]})
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    generator_result = await dialog_module.dialog_generator(dialog_state)
    assert generator_result["final_dialog"] == ["ok"]
    _assert_prompt_messages(
        generator_llm,
        {"internal_monologue", "linguistic_directives", "tone_history"},
    )

    evaluator_llm = _CaptureLLM({"feedback": "Passed", "should_stop": True})
    monkeypatch.setattr(dialog_module, "_dialog_evaluator_llm", evaluator_llm)
    evaluator_result = await dialog_module.dialog_evaluator(dialog_state)
    assert evaluator_result["should_stop"] is True
    _assert_prompt_messages(
        evaluator_llm,
        {"retry", "final_dialog", "linguistic_directives"},
    )
