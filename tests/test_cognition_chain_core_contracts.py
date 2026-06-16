"""Tests for the reusable cognition-chain core contracts."""

import json
from types import SimpleNamespace

import pytest

from llm_test_helpers import make_llm_call_config


def _chain_input() -> dict:
    return {
        "schema_version": "cognition_chain_input.v1",
        "episode": {
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "live_response",
            "model_visible_percepts": [{
                "percept_id": "percept-1",
                "input_source": "dialog_text",
                "content": "hello",
                "metadata_summary": [],
            }],
            "target_scope_summary": "current channel",
            "origin_summary": "user message",
        },
        "character": {
            "character_global_id": "character-1",
            "name": "Kazusa",
            "description": "character",
            "gender": "unknown",
            "age": "unknown",
            "birthday": "unknown",
            "backstory": "none",
            "personality_brief": {"mbti": "INTJ"},
            "boundary_profile": {"control_sensitivity": 0.5},
            "linguistic_texture_profile": {"fragmentation": 0.2},
            "mood": "neutral",
            "global_vibe": "calm",
        },
        "current_user": {
            "global_user_id": "user-1",
            "display_name": "User",
            "affinity": "neutral",
            "affinity_level": "known",
            "last_relationship_insight": "none",
            "memory_context": {
                "durable_profile_summary": "",
                "relationship_summary": "",
                "recent_commitments_summary": "",
                "known_preferences_summary": "",
            },
            "profile": {"affinity": 500},
        },
        "current_event": {
            "user_input": "hello",
            "decontextualized_input": "hello",
            "indirect_speech_context": "",
            "referents": [],
            "media_observations": [],
            "reply_context_summary": "",
            "prompt_message_context_summary": "",
            "reply_context": {},
            "prompt_message_context": {},
        },
        "scene": {
            "platform": "debug",
            "channel_type": "dm",
            "channel_topic": "",
            "local_time_context": {"local_time": "day"},
            "storage_timestamp_utc": "2026-06-12T00:00:00Z",
            "interaction_history_recent": [],
        },
        "conversation_context": {
            "conversation_progress": {},
            "promoted_reflection_context": {},
            "internal_monologue_residue_context": "",
            "previous_action_summary": "",
        },
        "evidence": {
            "rag_answer": "",
            "current_user_rag_bundle": "",
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "recall_evidence": [],
            "supervisor_trace": [],
            "rag_result": {"answer": ""},
        },
        "resolver": {
            "resolver_context": "",
            "pending_resume": "",
            "goal_progress": "",
            "recent_observations": [],
            "max_projected_observations": 3,
            "resolver_state": {},
        },
        "available_actions": [{
            "capability": "speak",
            "available": True,
            "visibility": "public",
            "semantic_input_summary": "visible text surface",
            "output_kind": "semantic_action_request",
        }],
        "runtime_context": {
            "language_policy": "simplified_chinese_internal_text",
            "visual_directives_enabled": True,
            "max_action_requests": 3,
            "max_resolver_requests": 3,
            "background_work_output_char_limit": 4000,
        },
    }


def _chain_output() -> dict:
    return {
        "schema_version": "cognition_chain_output.v1",
        "cognition_residue": {
            "emotional_appraisal": "neutral",
            "interaction_subtext": "plain",
            "internal_monologue": "considered",
            "logical_stance": "answer directly",
            "character_intent": "respond",
            "judgment_note": "safe",
            "social_distance": "near",
            "emotional_intensity": "low",
            "vibe_check": "calm",
            "relational_dynamic": "stable",
        },
        "semantic_action_requests": [{
            "capability": "speak",
            "decision": "visible_reply",
            "detail": "answer the user",
            "reason": "direct greeting",
        }],
        "resolver_capability_requests": [],
        "chain_trace": {
            "stage_order": ["l1", "l2a", "l2b", "l2c1", "l2c2", "l2d"],
            "selected_actions_summary": "speak",
            "resolver_summary": "",
            "warnings": [],
        },
    }


def test_cognition_chain_input_rejects_raw_graph_and_adapter_fields() -> None:
    from kazusa_ai_chatbot.cognition_chain_core.contracts import (
        CognitionChainContractError,
        validate_cognition_chain_input,
    )

    payload = _chain_input()
    payload["platform_channel_id"] = "raw-channel-id"

    with pytest.raises(CognitionChainContractError):
        validate_cognition_chain_input(payload)


def test_cognition_chain_output_rejects_action_specs() -> None:
    from kazusa_ai_chatbot.cognition_chain_core.contracts import (
        CognitionChainContractError,
        validate_cognition_chain_output,
    )

    payload = _chain_output()
    payload["action_specs"] = [{"kind": "speak"}]

    with pytest.raises(CognitionChainContractError):
        validate_cognition_chain_output(payload)


def test_cognition_chain_input_rejects_unknown_output_mode() -> None:
    from kazusa_ai_chatbot.cognition_chain_core.contracts import (
        CognitionChainContractError,
        validate_cognition_chain_input,
    )

    payload = _chain_input()
    payload["episode"]["output_mode"] = "not-a-mode"

    with pytest.raises(CognitionChainContractError):
        validate_cognition_chain_input(payload)


def test_cognition_chain_input_rejects_compatibility_source_label() -> None:
    from kazusa_ai_chatbot.cognition_chain_core.contracts import (
        CognitionChainContractError,
        validate_cognition_chain_input,
    )

    payload = _chain_input()
    payload["episode"]["input_sources"] = ["chat_message"]

    with pytest.raises(CognitionChainContractError):
        validate_cognition_chain_input(payload)


def test_cognition_chain_input_rejects_unknown_action_capability() -> None:
    from kazusa_ai_chatbot.cognition_chain_core.contracts import (
        CognitionChainContractError,
        validate_cognition_chain_input,
    )

    payload = _chain_input()
    payload["available_actions"][0]["capability"] = "not-a-capability"

    with pytest.raises(CognitionChainContractError):
        validate_cognition_chain_input(payload)


def test_text_surface_input_accepts_prompt_safe_projection() -> None:
    from kazusa_ai_chatbot.cognition_chain_core.contracts import (
        validate_text_surface_input,
    )

    payload = {
        "schema_version": "cognition_text_surface_input.v1",
        "chain_input": _chain_input(),
        "cognition_residue": _chain_output()["cognition_residue"],
        "selected_text_surface_intent": {
            "decision": "visible_reply",
            "original_goal": "answer greeting",
            "goal_progress_summary": "ready",
            "observation_summary": "",
            "speak_intent": "answer the user",
            "detail": "short greeting",
            "tone": "warm",
            "reason": "selected speak action",
        },
        "pre_surface_action_results": [{
            "action_kind": "background_work_request",
            "status": "queued",
            "queue_state": "queued",
            "task_summary": "draft notes",
            "objective_summary": "prepare notes",
            "acknowledgement_constraint": "promise_allowed",
        }],
        "memory_lifecycle_context": {
            "active_commitment_aliases": [],
            "pending_memory_updates_summary": "",
            "recent_memory_resolution_summary": "",
        },
        "interaction_style_context": {
            "user_style": {
                "speech_guidelines": [],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [],
                "confidence": "",
            },
            "application_order": ["user_style"],
        },
    }

    validated_payload = validate_text_surface_input(payload)

    assert validated_payload["schema_version"] == "cognition_text_surface_input.v1"


@pytest.mark.asyncio
async def test_public_text_surface_entrypoint_runs_without_legacy_rag_shape() -> None:
    from kazusa_ai_chatbot.cognition_chain_core import (
        CognitionChainServices,
        run_text_surface_planning,
    )

    payload = {
        "schema_version": "cognition_text_surface_input.v1",
        "chain_input": _surface_chain_input(),
        "cognition_residue": _chain_output()["cognition_residue"],
        "selected_text_surface_intent": {
            "decision": "visible_reply",
            "original_goal": "answer greeting",
            "goal_progress_summary": "ready",
            "observation_summary": "",
            "speak_intent": "answer the user",
            "detail": "short greeting",
            "tone": "warm",
            "reason": "selected speak action",
        },
        "pre_surface_action_results": [],
        "memory_lifecycle_context": {
            "active_commitment_aliases": [],
            "pending_memory_updates_summary": "",
            "recent_memory_resolution_summary": "",
        },
        "interaction_style_context": {
            "user_style": {
                "speech_guidelines": [],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [],
                "confidence": "",
            },
            "application_order": ["user_style"],
        },
    }
    services = CognitionChainServices(
        llm=_surface_fake_llm(),
        cognition_config=make_llm_call_config("cognition"),
        boundary_core_config=make_llm_call_config("boundary_core"),
        action_selection_config=make_llm_call_config("action_selection"),
        style_config=make_llm_call_config("style_agent"),
        content_plan_config=make_llm_call_config("content_plan_agent"),
        preference_config=make_llm_call_config("preference_adapter"),
        visual_config=make_llm_call_config("visual_agent"),
        parse_json=json.loads,
        logger=_Logger(),
    )

    result = await run_text_surface_planning(payload, services)

    assert result["schema_version"] == "cognition_text_surface_output.v1"
    assert result["action_directives"]["linguistic_directives"]["content_plan"][
        "semantic_content"
    ] == "answer directly"


@pytest.mark.asyncio
async def test_public_text_surface_entrypoint_resets_injected_parser() -> None:
    from kazusa_ai_chatbot.cognition_chain_core import (
        CognitionChainServices,
        run_text_surface_planning,
    )
    from kazusa_ai_chatbot.cognition_chain_core.utils import parse_llm_json_output

    def parse_json(raw_output: str) -> dict:
        parsed = json.loads(raw_output)
        if parsed.get("probe"):
            return_value = {"leaked": "parser"}
            return return_value
        return parsed

    payload = _text_surface_input()
    services = CognitionChainServices(
        llm=_surface_fake_llm(),
        cognition_config=make_llm_call_config("cognition"),
        boundary_core_config=make_llm_call_config("boundary_core"),
        action_selection_config=make_llm_call_config("action_selection"),
        style_config=make_llm_call_config("style_agent"),
        content_plan_config=make_llm_call_config("content_plan_agent"),
        preference_config=make_llm_call_config("preference_adapter"),
        visual_config=make_llm_call_config("visual_agent"),
        parse_json=parse_json,
        logger=_Logger(),
    )

    await run_text_surface_planning(payload, services)

    parsed = parse_llm_json_output('{"probe": true}')

    assert parsed == {"probe": True}


class _FakeLLM:
    """Return configured JSON responses from async LLM calls."""

    def __init__(self, responses_by_stage: dict[str, dict]) -> None:
        self._responses_by_stage = dict(responses_by_stage)

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config,
    ) -> SimpleNamespace:
        del messages
        stage_name = config.stage_name
        if stage_name not in self._responses_by_stage:
            raise AssertionError(f"unexpected LLM call: {stage_name}")
        response = SimpleNamespace(
            content=json.dumps(
                self._responses_by_stage.pop(stage_name),
                ensure_ascii=False,
            ),
        )
        return response


def _surface_fake_llm() -> _FakeLLM:
    """Build the scripted fake backend for text-surface stages."""

    fake_llm = _FakeLLM({
        "style_agent": {
            "rhetorical_strategy": "direct",
            "linguistic_style": "plain",
            "forbidden_phrases": [],
        },
        "content_plan_agent": {
            "content_plan": {
                "semantic_content": "answer directly",
                "rendering": "short visible reply",
            },
        },
        "preference_adapter": {"accepted_user_preferences": []},
        "visual_agent": {
            "facial_expression": [],
            "body_language": [],
            "gaze_direction": [],
            "visual_vibe": [],
        },
    })
    return fake_llm


class _Logger:
    """No-op logger for public-entrypoint smoke tests."""

    def debug(self, *args: object, **kwargs: object) -> None:
        return None

    def info(self, *args: object, **kwargs: object) -> None:
        return None

    def warning(self, *args: object, **kwargs: object) -> None:
        return None

    def error(self, *args: object, **kwargs: object) -> None:
        return None


def _text_surface_input() -> dict:
    return_value = {
        "schema_version": "cognition_text_surface_input.v1",
        "chain_input": _surface_chain_input(),
        "cognition_residue": _chain_output()["cognition_residue"],
        "selected_text_surface_intent": {
            "decision": "visible_reply",
            "original_goal": "answer greeting",
            "goal_progress_summary": "ready",
            "observation_summary": "",
            "speak_intent": "answer the user",
            "detail": "short greeting",
            "tone": "warm",
            "reason": "selected speak action",
        },
        "pre_surface_action_results": [],
        "memory_lifecycle_context": {
            "active_commitment_aliases": [],
            "pending_memory_updates_summary": "",
            "recent_memory_resolution_summary": "",
        },
        "interaction_style_context": {
            "user_style": {
                "speech_guidelines": [],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [],
                "confidence": "",
            },
            "application_order": ["user_style"],
        },
    }
    return return_value


def _surface_chain_input() -> dict:
    payload = _chain_input()
    payload["character"]["personality_brief"] = {
        "mbti": "INTJ",
        "logic": "structured",
        "tempo": "steady",
        "defense": "reserved",
        "quirks": "dry",
        "taboos": "none",
    }
    payload["character"]["linguistic_texture_profile"] = {
        "fragmentation": 0.2,
        "hesitation_density": 0.2,
        "counter_questioning": 0.2,
        "softener_density": 0.2,
        "formalism_avoidance": 0.2,
        "abstraction_reframing": 0.2,
        "direct_assertion": 0.5,
        "emotional_leakage": 0.2,
        "rhythmic_bounce": 0.2,
        "self_deprecation": 0.1,
    }
    payload["character"]["boundary_profile"] = {
        "self_integrity": 0.7,
        "control_sensitivity": 0.5,
        "compliance_strategy": "selective",
        "relational_override": 0.4,
        "control_intimacy_misread": 0.2,
        "boundary_recovery": "steady",
        "authority_skepticism": 0.5,
    }
    payload["evidence"].pop("rag_result", None)
    return payload
