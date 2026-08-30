"""Canonical cognition V2 interaction-contract tests."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module


def _plan() -> dict[str, object]:
    return {
        "goal_resolution": "answerable_now",
        "response_goal": "回答当前问题",
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": "仅基于当前观察回答。",
    }


def _prompt_workspace(kind: str | None = None) -> dict[str, object]:
    workspace: dict[str, object] = {
        "orientation": {},
        "capabilities": {"actions": [], "resolvers": []},
        "observation": {},
        "state": {},
        "continuity": {},
        "overused_moves": [],
        "response_plan_contract_variant": "fresh_ordinary",
    }
    if kind is not None:
        workspace["pending_dsh_interaction"] = {
            "schema_version": "dsh_brain_interaction.v2",
            "interaction_id": "interaction-123",
            "kind": kind,
            "tool_name": "read_file",
            "transient_detail": "Need a semantic decision",
        }
    return workspace


def _prompt_goal() -> dict[str, str]:
    return {
        "goal_kind": "open",
        "intent": "answer",
        "reason": "context",
        "cause_summary": "context",
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


def test_p_prompt_assigns_internal_decision_to_character_cognition() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.prompt import build_canonical_plan_question

    packet = build_canonical_plan_question(
        workspace=_prompt_workspace("question"),
        goal=_prompt_goal(),
        appraisal_summary=[],
    )
    contract = packet["output_contract"]
    assert "dsh_interaction_decision" in contract["required_fields"]
    assert "guidance" not in packet
    assert "内部 DSH 交互" in facade_module._P_DSH_INTERACTION_SYSTEM_PROMPT
    assert "response_goal" not in contract["dsh_interaction_decision_fields"]
    assert "relay_mode" not in contract["dsh_interaction_decision_fields"]


def test_p_prompt_keeps_the_normal_five_field_contract_without_dsh_context() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.prompt import build_canonical_plan_question

    packet = build_canonical_plan_question(
        workspace=_prompt_workspace(),
        goal=_prompt_goal(),
        appraisal_summary=[],
    )
    assert "guidance" not in packet
    assert packet["output_contract"]["required_fields"] == [
        "goal_resolution",
        "response_goal",
        "action_requests",
        "resolver_requests",
        "epistemic_boundary",
    ]


def test_p_prompt_emits_exact_kind_specific_internal_decisions() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.prompt import build_canonical_plan_question

    expected = {
        "approval": ["allow_once", "reject"],
        "question": ["answer", "reject"],
        "plan_review": ["answer", "allow_once", "reject"],
    }
    for kind, values in expected.items():
        packet = build_canonical_plan_question(
            workspace=_prompt_workspace(kind),
            goal=_prompt_goal(),
            appraisal_summary=[],
        )
        contract = packet["output_contract"]
        assert contract["dsh_interaction_decision_values"] == values
        assert contract["dsh_interaction_decision_fields"] == [
            "interaction_id", "kind", "decision", "answer", "reason",
        ]
        assert "dsh_interaction_decision_relay_mode_values" not in contract
        assert "guidance" not in packet


def test_dsh_plan_packet_has_no_user_solicitation_resolver() -> None:
    """The P packet carries only the internal self-goal affordance for DSH."""

    from kazusa_ai_chatbot.cognition_core_v3.prompt import (
        build_canonical_plan_question,
    )
    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        _available_resolver_affordances,
    )

    scene = {
        "channel_scope": "internal",
        "character_role": "Kazusa",
        "current_user_role": "internal caller",
        "semantic_scene": "The character is judging one DSH interaction.",
        "public_group_scene": "",
        "conversation_continuity": "The internal interaction is active.",
        "semantic_temporal_context": "Now.",
    }
    affordances = _available_resolver_affordances(
        {"dsh_interaction_episode": True},
        cognition_scene_context=scene,
    )
    workspace = _prompt_workspace("question")
    workspace["capabilities"] = {"actions": [], "resolvers": affordances}
    packet = build_canonical_plan_question(
        workspace=workspace,
        goal=_prompt_goal(),
        appraisal_summary=[],
    )
    resolvers = packet["capabilities"]["resolvers"]
    assert [row["capability"] for row in resolvers] == ["self_goal_resolution"]
    assert "human_clarification" not in str(resolvers)
    assert "approval_preparation" not in str(resolvers)
    assert "task_resolution_request" not in str(resolvers)


def test_pending_dsh_prompt_projection_is_exact_and_idempotent() -> None:
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
