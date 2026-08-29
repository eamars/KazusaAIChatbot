"""Canonical cognition interaction-contract tests."""

from __future__ import annotations

import pytest


def _plan():
    return {
        "goal_resolution": "answerable_now",
        "response_goal": '回答当前问题',
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": '仅基于当前观察回答。',
    }


def _prompt_workspace(kind: str | None = None) -> dict[str, object]:
    """Build the smallest workspace accepted by the P-stage prompt builder."""

    workspace: dict[str, object] = {
        "orientation": {},
        "capabilities": {"actions": [], "resolvers": []},
        "observation": {},
        "state": {},
        "continuity": {},
        "overused_moves": [],
    }
    if kind is not None:
        workspace["pending_dsh_interaction"] = {
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": "interaction-123",
            "kind": kind,
            "tool_name": "read_file",
            "transient_detail": "Need a semantic decision",
        }
    return workspace


def _prompt_goal() -> dict[str, str]:
    """Build the bounded goal payload consumed by the P-stage prompt builder."""

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
    validated = _validate_plan(plan, self_cognition=False, capabilities={"actions": [], "resolvers": []})
    assert validated.response_goal == '回答当前问题'
    with pytest.raises(CanonicalContractError) as error:
        _validate_plan({**plan, "dsh_interaction_decision": {"decision": "answer"}}, self_cognition=False, capabilities={"actions": [], "resolvers": []})
    assert str(error.value) == "ordinary response plan fields are not exact"


def test_p_stage_decision_survives_canonical_output_without_deterministic_semantic_rewrite() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.facade import _validate_plan

    plan = {
        **_plan(),
        "dsh_interaction_decision": {
            "interaction_id": "i1",
            "kind": "question",
            "decision": "answer",
            "answer": '由 Brain 判断',
            "response_goal": None,
            "relay_mode": None,
            "reason": "context",
        },
    }
    validated = _validate_plan(
        plan,
        self_cognition=False,
        capabilities={"actions": [], "resolvers": []},
        dsh_interaction_context={"interaction_id": "i1", "kind": "question"},
    )
    assert validated.dsh_interaction_decision["decision"] == "answer"


def test_p_prompt_assigns_decision_to_brain_and_visible_wording_to_dialog() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.prompt import build_canonical_plan_question

    workspace = {
        "orientation": {},
        "capabilities": {"actions": [], "resolvers": []},
        "observation": {},
        "state": {},
        "continuity": {},
        "overused_moves": [],
        "pending_dsh_interaction": {
            "schema_version": "dsh_brain_interaction.v1",
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
            "transient_detail": "Need a semantic answer",
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
    packet = build_canonical_plan_question(
        workspace=workspace,
        goal={"goal_kind": "open", "intent": "answer", "reason": "context", "cause_summary": "context"},
        appraisal_summary=[],
    )
    assert "dsh_interaction_decision" in packet["output_contract"]["required_fields"]
    assert "dialog" in packet["guidance"].lower()
    assert packet["pending_dsh_interaction"] == {
        "schema_version": "dsh_brain_interaction.v1",
        "interaction_id": "i1",
        "kind": "question",
        "tool_name": "read_file",
        "transient_detail": "Need a semantic answer",
    }
    assert "resolution_thread_id" not in packet["pending_dsh_interaction"]
    assert "scope_fingerprint" not in packet["pending_dsh_interaction"]


def test_p_prompt_keeps_the_normal_five_field_contract_without_dsh_context() -> None:
    """Normal P turns retain their original field set and guidance text."""

    from kazusa_ai_chatbot.cognition_core_v3.prompt import (
        ORDINARY_PLAN_GUIDANCE,
        build_canonical_plan_question,
    )

    packet = build_canonical_plan_question(
        workspace=_prompt_workspace(),
        goal=_prompt_goal(),
        appraisal_summary=[],
    )

    assert packet["guidance"] == ORDINARY_PLAN_GUIDANCE
    assert packet["output_contract"]["required_fields"] == [
        "goal_resolution",
        "response_goal",
        "action_requests",
        "resolver_requests",
        "epistemic_boundary",
    ]
    assert "dsh_interaction_decision" not in packet["output_contract"]


def test_p_prompt_emits_exact_kind_and_reply_specific_dsh_contracts() -> None:
    """Every DSH kind exposes only its valid decisions and field rules."""

    from kazusa_ai_chatbot.cognition_core_v3.prompt import (
        build_canonical_plan_question,
    )

    expected_by_kind = {
        "approval": ["allow_once", "reject", "relay_to_user"],
        "question": ["answer", "reject", "relay_to_user"],
        "plan_review": [
            "answer", "allow_once", "reject", "relay_to_user",
        ],
    }
    expected_required_fields = [
        "goal_resolution",
        "response_goal",
        "action_requests",
        "resolver_requests",
        "epistemic_boundary",
        "dsh_interaction_decision",
    ]

    for kind, expected_values in expected_by_kind.items():
        for dsh_reply in (False, True):
            packet = build_canonical_plan_question(
                workspace=_prompt_workspace(kind),
                goal=_prompt_goal(),
                appraisal_summary=[],
                dsh_reply=dsh_reply,
            )
            contract = packet["output_contract"]
            values = [
                *expected_values,
                *(["continue_waiting"] if dsh_reply else []),
            ]

            assert contract["required_fields"] == expected_required_fields
            assert contract["dsh_interaction_decision_bindings"] == {
                "interaction_id": "interaction-123",
                "kind": kind,
            }
            assert contract["dsh_interaction_decision_values"] == values
            expected_values_by_kind = {
                expected_kind: [
                    *expected_kind_values,
                    *(
                        ["continue_waiting"]
                        if dsh_reply and expected_kind == kind
                        else []
                    ),
                ]
                for expected_kind, expected_kind_values in (
                    expected_by_kind.items()
                )
            }
            assert contract["dsh_interaction_decision_values_by_kind"] == (
                expected_values_by_kind
            )
            assert contract["dsh_interaction_decision_relay_mode_values"] == [
                "question", "approval", "plan_review", None,
            ]
            field_rules = contract["dsh_interaction_decision_field_rules"]
            expected_rule_decisions = {
                "answer", "relay_to_user", "allow_once", "reject",
            }
            if dsh_reply:
                expected_rule_decisions.add("continue_waiting")
            for field_name in ("answer", "response_goal", "relay_mode"):
                assert set(field_rules[field_name]["by_decision"]) == (
                    expected_rule_decisions
                )
            assert field_rules["answer"]["by_decision"]["answer"] == (
                "required_non_empty_string"
            )
            assert field_rules["answer"]["by_decision"]["relay_to_user"] == "null"
            assert field_rules["response_goal"]["by_decision"]["answer"] == "null"
            assert field_rules["response_goal"]["by_decision"]["relay_to_user"] == (
                "required_non_empty_string"
            )
            assert field_rules["relay_mode"]["by_decision"]["answer"] == "null"
            assert field_rules["relay_mode"]["by_decision"]["relay_to_user"] == (
                "required_allowed_value"
            )
            decisions_with_null_fields = ["allow_once", "reject"]
            if dsh_reply:
                decisions_with_null_fields.append("continue_waiting")
            for decision in decisions_with_null_fields:
                assert field_rules["answer"]["by_decision"][decision] == "null"
                assert field_rules["response_goal"]["by_decision"][decision] == "null"
                assert field_rules["relay_mode"]["by_decision"][decision] == "null"
            assert field_rules["reason"]["required_for_every_decision"] is True
            assert "interaction-123" in packet["guidance"]
            assert f"`kind` 必须逐字返回 `{kind}`" in packet["guidance"]
            if dsh_reply:
                assert "continue_waiting" in packet["guidance"]
            else:
                assert "不得使用 `continue_waiting`" in packet["guidance"]


def test_pending_dsh_prompt_projection_is_exact_and_idempotent() -> None:
    """Permit repeated stage projection without restoring authority fields."""

    from kazusa_ai_chatbot.cognition_shared.contracts import (
        project_pending_dsh_interaction_for_prompt,
    )

    full = {
        "schema_version": "dsh_brain_interaction.v1",
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


def test_cognition_input_validates_pending_dsh_context_as_untrusted_bounded_evidence() -> None:
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
        "pending_dsh_interaction": {
            "schema_version": "dsh_brain_interaction.v1",
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
