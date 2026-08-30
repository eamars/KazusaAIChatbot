"""Canonical cognition V2 interaction-contract tests."""

from __future__ import annotations

import pytest


def _plan() -> dict[str, object]:
    return {
        "goal_resolution": "answerable_now",
        "response_goal": "回答当前问题",
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": "仅基于当前观察回答。",
    }


def test_response_plan_requires_exact_kind_compatible_dsh_decision_only_when_context_exists() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.facade import (
        CanonicalContractError,
        _validate_plan,
    )

    plan = _plan()
    validated = _validate_plan(
        plan,
        self_cognition=False,
        capabilities={"actions": [], "resolvers": []},
        response_plan_contract_variant="fresh_ordinary",
    )
    assert validated.response_goal == "回答当前问题"
    with pytest.raises(CanonicalContractError) as error:
        _validate_plan(
            {**plan, "dsh_interaction_decision": {"decision": "answer"}},
            self_cognition=False,
            capabilities={"actions": [], "resolvers": []},
            response_plan_contract_variant="fresh_ordinary",
        )
    assert str(error.value) == (
        "response plan: unexpected fields ['dsh_interaction_decision']"
    )


def test_p_stage_decision_survives_canonical_output_without_semantic_rewrite() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.facade import _validate_plan

    plan = {
        **_plan(),
        "dsh_interaction_decision": {
            "interaction_id": "i1",
            "kind": "question",
            "decision": "answer",
            "answer": "由 Brain 判断",
            "reason": "context",
        },
    }
    validated = _validate_plan(
        plan,
        self_cognition=False,
        capabilities={"actions": [], "resolvers": []},
        dsh_interaction_context={"interaction_id": "i1", "kind": "question"},
        response_plan_contract_variant="fresh_ordinary",
    )
    assert validated.dsh_interaction_decision["decision"] == "answer"


def test_pending_dsh_projection_is_exact_and_idempotent() -> None:
    from kazusa_ai_chatbot.cognition_shared.contracts import (
        project_pending_dsh_interaction_for_prompt,
    )

    full = {
        "schema_version": "dsh_brain_interaction.v2",
        "interaction_id": "i1",
        "kind": "approval",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "activation_id": "activation-1",
        "lease_epoch": 1,
        "dsh_call_id": "call-1",
        "tool_name": "pwsh",
        "operation_id": "operation-1",
        "operation_payload_digest": "sha256:operation",
        "arguments_digest": "sha256:args",
        "transient_detail": "Approve one bounded command.",
        "brain_conversation_ref": "chat:debug:one",
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "global_user_id": "user-1",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "profile_version": "kazusa-resolver-standard-v2",
        "catalog_digest": "sha256:catalog",
        "model_route_digest": "sha256:route",
        "workspace_fingerprint": "sha256:workspace",
        "policy_epoch": "dsh-standard-policy-v2",
        "issued_reference_digest": "sha256:issued-refs",
        "issuer": "dsh-sidecar",
        "nonce": "nonce-1",
        "issued_at": "2026-08-28T00:00:00Z",
        "expires_at": "2026-08-28T00:05:00Z",
    }
    projected = project_pending_dsh_interaction_for_prompt(full)
    assert project_pending_dsh_interaction_for_prompt(projected) == projected
    assert set(projected) == {
        "schema_version", "interaction_id", "kind", "tool_name",
        "transient_detail",
    }


def test_cognition_input_validates_pending_dsh_context_as_bounded_evidence() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.facade import _validate_canonical_input

    payload = {
        "episode": {},
        "scene_context": {},
        "evidence": [],
        "mutable_state": {},
        "state_scope": "user",
        "character_constraints": {},
        "character_identity_context": {},
        "available_actions": [],
            "available_resolver_capabilities": [],
            "overused_moves": [],
            "response_plan_contract_variant": "fresh_ordinary",
            "pending_dsh_interaction": {
            "schema_version": "dsh_brain_interaction.v2",
            "interaction_id": "i1",
            "kind": "question",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "activation_id": "activation-1",
            "lease_epoch": 1,
            "dsh_call_id": "call-1",
            "tool_name": "read_file",
            "operation_id": "operation-1",
            "operation_payload_digest": "sha256:operation",
            "arguments_digest": "sha256:args",
            "transient_detail": "bounded",
            "brain_conversation_ref": "chat:debug:one",
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
            "scope_fingerprint": "sha256:scope",
            "audience_fingerprint": "sha256:audience",
            "profile_version": "kazusa-resolver-standard-v2",
            "catalog_digest": "sha256:catalog",
            "model_route_digest": "sha256:route",
            "workspace_fingerprint": "sha256:workspace",
            "policy_epoch": "dsh-standard-policy-v2",
            "issued_reference_digest": "sha256:issued-refs",
            "issuer": "dsh-sidecar",
            "nonce": "nonce-1",
            "issued_at": "2026-08-28T00:00:00Z",
            "expires_at": "2026-08-28T00:05:00Z",
        },
    }
    validated = _validate_canonical_input(payload)
    assert validated["pending_dsh_interaction"]["interaction_id"] == "i1"
