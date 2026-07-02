"""Tests for the action-selection prompt-safe JSON payload contract."""

from __future__ import annotations

import json


def test_action_selection_payload_is_prompt_safe_json() -> None:
    """The action-selection payload must be a JSON-serializable dict with only
    prompt-safe semantic sections. No raw IDs, no prose string payload."""

    from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
        build_action_selection_payload,
    )

    minimal_state = _minimal_cognition_state()
    payload = build_action_selection_payload(minimal_state)

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
    assert payload["work_seed"]["accepted_task_allowed"] is True
    assert payload["work_seed"]["max_output_chars"] > 0
    resolver_affordances = payload["capabilities"]["resolver_affordances"]
    assert resolver_affordances
    for affordance in resolver_affordances:
        assert affordance["semantic_input_summary"]


def test_action_selection_payload_omits_task_willingness_gate_metadata() -> None:
    """The willingness flag may select prompts but must not enter L2d JSON."""

    from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
        build_action_selection_payload,
    )

    state = _minimal_cognition_state()
    state["task_willingness_boundary_enabled"] = True

    payload = build_action_selection_payload(state)
    serialized = json.dumps(payload, ensure_ascii=False)

    for forbidden in (
        "task_willingness_boundary_enabled",
        "COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED",
        "affinity_task_gate",
        "mood_gate",
        "vibe_gate",
        "feature_enabled",
        "effort_score",
        "complexity_score",
        "patience_score",
        "willingness_score",
    ):
        assert forbidden not in serialized


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
        "available_action_affordances": [
            {
                "capability": "speak",
                "available": True,
                "visibility": "public",
                "semantic_input_summary": [
                    "Use when the character wants a text surface to exist."
                ],
            },
            {
                "capability": "accepted_task_request",
                "available": True,
                "visibility": "private",
                "semantic_input_summary": [
                    "Use only for accepted bounded delayed text work."
                ],
            },
        ],
        "background_work_output_char_limit": 4000,
    }
