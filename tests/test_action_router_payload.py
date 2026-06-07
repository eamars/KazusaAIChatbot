"""Tests for the action-router prompt-safe JSON payload contract."""

from __future__ import annotations

import json


def test_action_router_payload_is_prompt_safe_json() -> None:
    """The action-router payload must be a JSON-serializable dict with only
    prompt-safe semantic sections. No raw IDs, no prose string payload."""

    from kazusa_ai_chatbot.action_router.payload import (
        build_action_router_payload,
    )

    from kazusa_ai_chatbot.action_spec.registry import (
        build_initial_action_capabilities,
    )

    capabilities = build_initial_action_capabilities()

    minimal_state = _minimal_cognition_state()
    payload = build_action_router_payload(minimal_state, capabilities)

    assert isinstance(payload, dict), "payload must be a dict, not a prose string"

    serialized = json.dumps(payload, ensure_ascii=False)
    reparsed = json.loads(serialized)
    assert reparsed == payload, "payload must round-trip through JSON"

    for required_section in (
        "source",
        "current_input",
        "cognition",
        "evidence",
        "resolver",
        "capabilities",
        "work_seed",
    ):
        assert required_section in payload, (
            f"missing required section: {required_section}"
        )

    serialized_lower = serialized.lower()
    for forbidden in (
        "global_user_id",
        "platform_channel_id",
        "message_id",
        "job_id",
        "action_attempt_id",
        "handler_id",
        "schema_version",
        "collection_name",
        "adapter_id",
        "lease",
    ):
        assert forbidden not in serialized_lower, (
            f"forbidden field leaked into payload: {forbidden}"
        )

    assert payload["source"]["trigger_source"] == "user_message"
    assert payload["current_input"]["decontextualized_input"] == (
        "Tell me about Fibonacci numbers."
    )
    assert payload["work_seed"]["source_summary"] == (
        "Tell me about Fibonacci numbers."
    )
    assert payload["work_seed"]["background_work_allowed"] is True
    assert payload["work_seed"]["max_output_chars"] > 0
    resolver_affordances = payload["capabilities"]["resolver_affordances"]
    assert resolver_affordances
    for affordance in resolver_affordances:
        assert affordance["semantic_input_summary"]


def _minimal_cognition_state() -> dict:
    """Build a minimal CognitionState-shaped dict for payload testing."""

    return {
        "cognitive_episode": {
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
        },
        "channel_type": "private",
        "decontexualized_input": "Tell me about Fibonacci numbers.",
        "media_summary": "",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "factual question, straightforward reply",
        "internal_monologue": "User wants a math explanation.",
        "emotional_appraisal": "neutral curiosity",
        "interaction_subtext": "learning intent",
        "boundary_core_assessment": {},
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "calm",
        "relational_dynamic": "peer",
        "rag_answer": "",
        "memory_evidence": [],
        "conversation_progress": {},
        "active_commitment_clues": [],
        "resolver_context": {},
        "resolver_pending_resume": None,
        "resolver_previous_observations": [],
        "group_engagement_context": None,
    }
