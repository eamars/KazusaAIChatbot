"""Tests for modality-neutral action-spec contract validators."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.action_spec.models import (
    ACTION_SPEC_VERSION,
    LIFECYCLE_STATUS_BY_DECISION,
    ActionValidationError,
    validate_action_continuation,
    validate_action_source_ref,
    validate_action_spec,
    validate_action_target,
    validate_capability_spec,
)


def _source_ref() -> dict:
    return {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "memory_unit",
        "ref_id": "promise-001",
        "owner": "user_memory_units",
        "relationship": "basis",
        "evidence_refs": [
            {
                "schema_version": "evidence_ref.v1",
                "evidence_kind": "memory_unit",
                "evidence_id": "promise-001",
                "owner": "user_memory_units",
                "excerpt": "Promised to reveal the spice answer.",
                "observed_at": "2026-05-07T00:00:00+00:00",
            }
        ],
    }


def _current_channel_target() -> dict:
    return {
        "schema_version": "action_target.v1",
        "target_kind": "current_channel",
        "target_id": None,
        "owner": "dispatcher",
        "scope": {"channel_relation": "same"},
    }


def _no_continuation() -> dict:
    return {
        "schema_version": "action_continuation.v1",
        "mode": "none",
        "episode_type": None,
        "max_depth": 0,
        "include_result_as": None,
    }


def _send_message_action() -> dict:
    return {
        "schema_version": ACTION_SPEC_VERSION,
        "kind": "send_message",
        "cognition_mode": "deliberative",
        "source_refs": [_source_ref()],
        "target": _current_channel_target(),
        "params": {
            "target_channel": "same",
            "text": "Checking in now.",
            "execute_at": None,
            "delivery_mentions": [],
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": _no_continuation(),
        "reason": "The character decided the promised follow-up should be sent.",
    }


def _capability_spec() -> dict:
    return {
        "schema_version": "capability_spec.v1",
        "capability_kind": "send_message",
        "category": "action",
        "owner_module": "dispatcher",
        "input_schema": {"type": "object", "required": ["text"]},
        "output_schema": {"type": "object"},
        "handler_id": "dispatcher.send_message",
        "lifecycle_hooks": ["validate", "dispatch"],
        "permission_policy": "policy:dispatcher.send_message.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }


def test_action_spec_validator_accepts_deliberative_contract() -> None:
    """A complete deliberative action spec should satisfy the public contract."""

    validated = validate_action_spec(_send_message_action())

    assert validated["schema_version"] == ACTION_SPEC_VERSION
    assert validated["kind"] == "send_message"
    assert validated["cognition_mode"] == "deliberative"
    assert validated["continuation"]["mode"] == "none"


def test_action_spec_validator_rejects_reflex_for_initial_slice() -> None:
    """The schema reserves reflex mode, but no current capability may use it."""

    action_spec = _send_message_action()
    action_spec["cognition_mode"] = "reflex"

    with pytest.raises(ActionValidationError, match="reflex"):
        validate_action_spec(action_spec)


def test_supporting_shape_validators_reject_incomplete_payloads() -> None:
    """Source refs, targets, and continuation requests are separate contracts."""

    source_ref = _source_ref()
    source_ref["schema_version"] = "action_source_ref.v0"
    with pytest.raises(ActionValidationError, match="schema_version"):
        validate_action_source_ref(source_ref)

    target = _current_channel_target()
    target["scope"] = "same"
    with pytest.raises(ActionValidationError, match="scope"):
        validate_action_target(target)

    continuation = _no_continuation()
    continuation["mode"] = "immediate_followup"
    continuation["episode_type"] = None
    continuation["max_depth"] = 1
    with pytest.raises(ActionValidationError, match="episode_type"):
        validate_action_continuation(continuation)


def test_capability_spec_requires_action_category_and_policy_refs() -> None:
    """Capability specs are action-category entries, not raw dispatcher tools."""

    validated = validate_capability_spec(_capability_spec())
    assert validated["category"] == "action"

    capability = _capability_spec()
    capability["category"] = "tool"
    with pytest.raises(ActionValidationError, match="category"):
        validate_capability_spec(capability)

    capability = _capability_spec()
    capability["permission_policy"] = ""
    with pytest.raises(ActionValidationError, match="permission_policy"):
        validate_capability_spec(capability)


def test_capability_spec_accepts_l3_surface_owner() -> None:
    """Surface action handlers are registered action owners, not dispatcher tools."""

    capability = _capability_spec()
    capability["owner_module"] = "l3_text"

    validated = validate_capability_spec(capability)

    assert validated["owner_module"] == "l3_text"


def test_lifecycle_decision_status_mapping_matches_plan() -> None:
    """The LLM-owned semantic decisions map to existing collection statuses."""

    assert LIFECYCLE_STATUS_BY_DECISION == {
        "fulfilled": "completed",
        "abandoned": "cancelled",
        "obsolete": "archived",
        "deferred": "active",
    }
