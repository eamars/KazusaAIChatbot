"""Focused contract tests for the validation-only cognition core V2 facade."""

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    CognitionChainServices,
    validate_cognition_chain_output,
)
from kazusa_ai_chatbot.cognition_core_v2 import run_cognition_chain

from llm_test_helpers import make_llm_call_config


class _NoCallLLM:
    """Raise if an unambiguous contract fixture reaches a model call."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        raise AssertionError("the deterministic greeting fixture must not call an LLM")


class _Logger:
    """Provide the minimal cognition logger contract for facade tests."""

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        del message
        del args
        del kwargs

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        del message
        del args
        del kwargs

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        del message
        del args
        del kwargs

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        del message
        del args
        del kwargs


def _chain_input() -> dict[str, object]:
    """Build one prompt-safe V1 input with no ambiguous causal event."""

    return {
        "schema_version": "cognition_chain_input.v1",
        "episode": {
            "episode_id": "v2-contract-episode",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "live_response",
            "model_visible_percepts": [{
                "percept_id": "v2-contract-percept",
                "input_source": "dialog_text",
                "content": "hello",
                "metadata_summary": [],
            }],
            "target_scope_summary": "contract test scope",
            "origin_summary": "unambiguous greeting",
        },
        "character": {
            "character_global_id": "v2-contract-character",
            "name": "Test Character",
            "description": "test-only character",
            "gender": "unknown",
            "age": "unknown",
            "birthday": "unknown",
            "backstory": "none",
            "personality_brief": {},
            "boundary_profile": {},
            "linguistic_texture_profile": {},
            "mood": "neutral",
            "global_vibe": "calm",
        },
        "current_user": {
            "global_user_id": "v2-contract-user",
            "display_name": "Test User",
            "affinity": "neutral",
            "affinity_level": "known",
            "last_relationship_insight": "none",
            "memory_context": {
                "durable_profile_summary": "",
                "relationship_summary": "",
                "recent_commitments_summary": "",
                "known_preferences_summary": "",
            },
        },
        "current_event": {
            "user_input": "hello",
            "decontextualized_input": "hello",
            "indirect_speech_context": "",
            "referents": [],
            "media_observations": [],
            "reply_context_summary": "",
            "prompt_message_context_summary": "",
        },
        "scene": {
            "platform": "debug",
            "channel_type": "dm",
            "channel_topic": "",
            "local_time_context": {"time_of_day": "day"},
            "storage_timestamp_utc": "2026-07-13T00:00:00Z",
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
        },
        "resolver": {
            "resolver_context": "",
            "pending_resume": "",
            "goal_progress": "",
            "recent_observations": [],
            "max_projected_observations": 3,
        },
        "available_actions": [{
            "capability": "speak",
            "available": True,
            "visibility": "public",
            "semantic_input_summary": "visible text surface",
            "output_kind": "semantic_action_request",
        }],
        "runtime_context": {
            "language_policy": "english",
            "visual_directives_enabled": False,
            "task_willingness_boundary_enabled": False,
            "max_action_requests": 1,
            "max_resolver_requests": 1,
            "background_work_output_char_limit": 1000,
        },
        "action_selection_context": {
            "coding_runs": [],
            "group_engagement_action_context": {},
        },
    }


def _services() -> CognitionChainServices:
    """Build required V1 service bindings for a deterministic facade call."""

    services = CognitionChainServices(
        llm=_NoCallLLM(),
        cognition_config=make_llm_call_config("v2_cognition"),
        boundary_core_config=make_llm_call_config("v2_boundary"),
        action_selection_config=make_llm_call_config("v2_action_selection"),
        style_config=make_llm_call_config("v2_style"),
        content_plan_config=make_llm_call_config("v2_content"),
        preference_config=make_llm_call_config("v2_preference"),
        visual_config=make_llm_call_config("v2_visual"),
        parse_json=json.loads,
        logger=_Logger(),
    )
    return services


@pytest.mark.asyncio
async def test_v2_facade_returns_a_valid_v1_output_for_a_deterministic_case() -> None:
    """Keep the experimental implementation on the exact V1 chain boundary."""

    output = await run_cognition_chain(_chain_input(), _services())

    validated_output = validate_cognition_chain_output(output)

    assert validated_output["schema_version"] == "cognition_chain_output.v1"
    assert validated_output["chain_trace"]["stage_order"] == ["v2"]

