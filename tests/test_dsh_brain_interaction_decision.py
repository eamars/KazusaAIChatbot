"""Brain-owned DSH interaction semantic decision tests."""

from __future__ import annotations

import pytest

from tests.test_dsh_brain_interaction_contracts import _request_mapping


@pytest.mark.asyncio
async def test_brain_semantic_decision_is_enacted_without_keyword_or_post_llm_reclassification() -> None:
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV2,
    )
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine

    request = DshBrainInteractionRequestV2.from_mapping(_request_mapping())

    async def judge(request, context):
        assert context == {"user_text": "allow this"}
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "answer",
            "answer": "A semantic answer",
            "reason": "judged",
        }

    decision = await BrainDecisionEngine(judge=judge).decide(
        request,
        context={"user_text": "allow this"},
    )
    assert decision.decision == "answer"
    assert decision.answer == "A semantic answer"


def test_allow_once_grant_binds_request_activation_and_lease() -> None:
    """Immediate approval grants retain the exact activation lease lineage."""

    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionDecisionV2,
        DshBrainInteractionRequestV2,
    )
    from kazusa_ai_chatbot.dsh_interaction.decision import enact_decision

    request_mapping = _request_mapping(
        kind="approval",
        tool_name="pwsh",
    )
    request = DshBrainInteractionRequestV2.from_mapping(request_mapping)
    decision = DshBrainInteractionDecisionV2.from_mapping({
        "schema_version": request.schema_version,
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": "allow_once",
        "answer": None,
        "reason": "the exact native operation is allowed once",
    })

    result = enact_decision(request, decision, now=request.issued_at)

    grant = result["grant"]
    assert grant["activation_id"] == request.activation_id
    assert grant["lease_epoch"] == request.lease_epoch
    assert result.keys() == {
        "schema_version", "interaction_id", "request_digest", "kind",
        "decision", "answer", "reason", "grant",
    }


def test_global_state_carries_typed_pending_interaction_and_semantic_decision() -> None:
    from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
        DshPendingInteractionContext,
        DshSemanticInteractionDecision,
    )

    assert DshPendingInteractionContext
    assert DshSemanticInteractionDecision


def test_dsh_interaction_full_loop_advertises_only_internal_resolvers() -> None:
    """The real cognition input projection hides every user-soliciting resolver."""

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
    internal = _available_resolver_affordances(
        {"dsh_interaction_episode": True},
        cognition_scene_context=scene,
    )
    assert [row["capability"] for row in internal] == ["self_goal_resolution"]
    ordinary = _available_resolver_affordances(
        {},
        cognition_scene_context=scene,
    )
    ordinary_capabilities = {row["capability"] for row in ordinary}
    assert {"human_clarification", "approval_preparation"} <= ordinary_capabilities


@pytest.mark.asyncio
async def test_dsh_interaction_runs_full_reusable_cognition_loop_and_returns_internal_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV2,
    )
    from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition

    request = DshBrainInteractionRequestV2.from_mapping(_request_mapping())
    pending = request.unsigned_dict()
    calls: list[str] = []
    captured: dict[str, object] = {}
    output = {
        "schema_version": "cognition_output.v3",
        "state_projection": {
            "state_scope": "user",
            "owner_key": request.global_user_id,
            "original_persisted_state": {},
            "replacement_state": {},
        },
        "response_plan": {
            "dsh_interaction_decision": {
                "interaction_id": request.interaction_id,
                "kind": request.kind,
                "decision": "reject",
                "answer": None,
                "reason": "the character has insufficient grounded reason",
            },
        },
    }

    async def fake_loop(state, **kwargs):
        calls.append("resolver_loop")
        captured["state"] = state
        assert kwargs["call_cognition_subgraph_func"]
        assert kwargs["execute_capability_func"]
        return {
            **state,
            "cognition_core_output": output,
            "character_cognition_base_updated_at": None,
        }

    async def fail_one_pass(*_args, **_kwargs):
        raise AssertionError("DSH interaction must use resolver recurrence")

    async def fake_commit(value, **_kwargs):
        calls.append("commit")
        assert value is output

    monkeypatch.setattr(cognition, "call_cognition_resolver_loop", fake_loop)
    monkeypatch.setattr(cognition, "run_cognition", fail_one_pass)
    monkeypatch.setattr(cognition, "commit_cognition_output", fake_commit)
    monkeypatch.setattr(
        cognition,
        "build_cognition_input_from_global_state",
        lambda state: {"episode": state["cognitive_episode"]},
    )
    state = {
        "llm_trace_id": "dsh-trace-id",
        "global_user_id": request.global_user_id,
        "platform": request.platform,
        "platform_channel_id": request.platform_channel_id,
        "decontextualized_input": request.transient_detail,
        "character_profile": {
            "global_user_id": "character-1",
            "name": "Kazusa",
        },
        "cognitive_episode": {
            "episode_id": "dsh-episode-id",
            "target_scope": {"platform": "debug"},
        },
    }
    result = await cognition.run_dsh_interaction_cognition(
        state,
        pending_interaction=pending,
        services=object(),
    )

    assert result["decision"] == "reject"
    staged = captured["state"]
    assert isinstance(staged, dict)
    assert staged["pending_dsh_interaction"] == pending
    assert staged["dsh_interaction_episode"] is True
    assert calls == ["resolver_loop", "commit"]


def test_dsh_decision_contract_has_no_visible_surface_fields() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.facade import (
        CanonicalContractError,
        _validate_dsh_interaction_decision,
    )

    context = {"interaction_id": "interaction-1", "kind": "question"}
    decision = {
        "interaction_id": "interaction-1",
        "kind": "question",
        "decision": "reject",
        "answer": None,
        "reason": "internal character judgment",
    }
    assert _validate_dsh_interaction_decision(decision, context=context)["decision"] == "reject"
    with pytest.raises(CanonicalContractError, match="fields are not exact"):
        _validate_dsh_interaction_decision(
            {**decision, "response_goal": "visible"},
            context=context,
        )
