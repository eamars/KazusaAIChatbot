"""Deterministic tests for V3 cache-affine goal cognition semantics."""

from __future__ import annotations

import asyncio

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import GOAL_KINDS
from kazusa_ai_chatbot.cognition_core_v3 import goal_cognition as gc
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    EXHAUSTION_FAILURE_CLASS,
    hash_static_prompt,
    validate_stage_result,
)
from kazusa_ai_chatbot.cognition_core_v3.execution import (
    AttemptLedger,
    ChainTaskSpec,
    StageAttemptOutcome,
    start_wave,
)


def _ordinary_bid() -> dict:
    return {
        "intention": '确认当前请求',
        "desired_outcome": '得到回应',
        "concrete_detail": '先核实证据',
        "reason": '角色需要先核实当前事件再作立场。',
        "private_monologue": '我先看看证据。',
        "target_role_handles": ["self"],
        "evidence_handles": ["ev_1"],
        "expected_consequences": ['回应保持有依据'],
        "confidence": "medium",
        "relational_willingness": {
            "applicability": "relationship_sensitive",
            "stance": "negotiate",
            "current_user_relationship_state": "developing_or_uncertain",
            "reason": '关系仍在发展中，先协商边界。',
            "evidence_handles": ["ev_1"],
        },
    }


def test_all_goal_chains_use_one_byte_identical_static_system_prompt():
    prompt_hash = hash_static_prompt(gc.STATIC_GOAL_SYSTEM_PROMPT)

    for goal_kind in GOAL_KINDS:
        tail = gc.build_goal_question_tail(goal_kind, {"goals": "g1"}, ["ev_9"])
        assert goal_kind in tail
        assert f"- ev_9" in tail
        determinism_check = gc.build_goal_question_tail(goal_kind, {"goals": "g1"}, ["ev_9"])
        assert tail == determinism_check

    for owner_kind in GOAL_KINDS:
        owner_tail = gc.build_goal_question_tail(owner_kind, {}, [])
        for other_kind in GOAL_KINDS:
            if other_kind != owner_kind:
                assert other_kind not in owner_tail

    static = gc.STATIC_GOAL_SYSTEM_PROMPT
    assert hash_static_prompt(static) == prompt_hash
    for dynamic_token in ("# 目标类型", "# 授权证据 handle", "loss_recovery", "ev_9"):
        assert dynamic_token not in static
    assert 'Kazusa' not in static


def test_ordinary_goal_remains_relational_willingness_owner():
    owners = {kind: gc.goal_kind_owns_relational_willingness(kind) for kind in GOAL_KINDS}
    assert owners["ordinary_response"] is True
    assert sum(owners.values()) == 1

    with pytest.raises(ValueError, match="unknown goal kind"):
        gc.goal_kind_owns_relational_willingness("invented_kind")

    normalized = gc.validate_goal_bid_draft(
        _ordinary_bid(),
        goal_kind="ordinary_response",
        evidence_handles={"ev_1"},
        role_handles={"self"},
    )
    assert normalized["relational_willingness"]["schema_version"] == (
        "relational_willingness.v2"
    )

    missing_field = {k: v for k, v in _ordinary_bid().items() if k != "relational_willingness"}
    with pytest.raises(ValueError, match="goal bid draft fields are not exact"):
        gc.validate_goal_bid_draft(
            missing_field,
            goal_kind="ordinary_response",
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )

    foreign_owner = _ordinary_bid()
    with pytest.raises(ValueError, match="goal bid draft fields are not exact"):
        gc.validate_goal_bid_draft(
            foreign_owner,
            goal_kind="safety",
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )

    code_owned_version = _ordinary_bid()
    code_owned_version["relational_willingness"] = dict(code_owned_version["relational_willingness"])
    code_owned_version["relational_willingness"]["schema_version"] = "forged"
    with pytest.raises(ValueError, match="code-owned"):
        gc.validate_goal_bid_draft(
            code_owned_version,
            goal_kind="ordinary_response",
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )

    bad_stance = _ordinary_bid()
    bad_stance["relational_willingness"] = dict(bad_stance["relational_willingness"])
    bad_stance["relational_willingness"]["stance"] = "invented_stance"
    with pytest.raises(CognitionContractError, match="stance is invalid"):
        gc.validate_goal_bid_draft(
            bad_stance,
            goal_kind="ordinary_response",
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )

    ordinary_bid_tail = gc.build_goal_question_tail("ordinary_response", {}, [])
    assert "goal bid" in ordinary_bid_tail
    assert '本分支拥有 relational_willingness，必须输出完整关系立场。' in ordinary_bid_tail

    selection_tail = gc.build_goal_question_tail(
        "ordinary_response", {}, [], selection_mode=True
    )
    assert "selection draft" in selection_tail
    assert 'relational_willingness，必须输出完整关系立场' not in selection_tail


def test_sibling_goal_transcripts_are_isolated():
    safety_tail = gc.build_goal_question_tail("safety", {"goals": "g-safety"}, ["ev_s"])
    loss_tail = gc.build_goal_question_tail("loss_recovery", {"goals": "g-loss"}, ["ev_l"])

    assert "loss_recovery" not in safety_tail
    assert "g-loss" not in safety_tail
    assert "safety" not in loss_tail
    assert "goals=g-safety" not in loss_tail

    async def scenario():
        ledger = AttemptLedger({"safety": 2, "loss_recovery": 2})
        seen_contexts = []

        def make_producer(chain_name):
            async def producer(ctx):
                assert ctx.chain_name == chain_name
                assert ctx.accepted_prefix == ()
                seen_contexts.append((ctx.chain_name, ctx.stage_name))
                return StageAttemptOutcome(
                    True,
                    {"goal_kind": chain_name},
                    f"{chain_name}-summary",
                    None,
                )

            return producer

        handle = start_wave(
            [
                ChainTaskSpec("safety", ("safety",), {"safety": make_producer("safety")}),
                ChainTaskSpec("loss_recovery", ("loss_recovery",), {"loss_recovery": make_producer("loss_recovery")}),
            ],
            ledger=ledger,
        )
        result = await handle.complete()
        return result, seen_contexts

    result, seen_contexts = asyncio.run(scenario())
    assert sorted(seen_contexts) == [("loss_recovery", "loss_recovery"), ("safety", "safety")]

    safety_outcome, loss_outcome = result.outcomes["safety"], result.outcomes["loss_recovery"]
    assert safety_outcome.results[0].semantic_summary == "safety-summary"
    assert loss_outcome.results[0].semantic_summary == "loss_recovery-summary"


def test_required_selection_preserves_fixed_roles_and_progress_evidence():
    authoritative_operation = {
        "operation": "",
        "embedded_actor_role": gc.NO_ROLE,
        "embedded_target_role": "current_user",
        "response_owner_role": "self",
        "selection_owner_role": "self",
        "selection_required": True,
    }

    bound = gc.bind_selected_response_operation(
        {"operation": '回答当前问题'},
        authoritative_operation,
    )
    assert bound == {
        "operation": '回答当前问题',
        "response_owner_role": "self",
        "selection_owner_role": "self",
        "selection_required": True,
        "embedded_actor_role": gc.NO_ROLE,
        "embedded_target_role": "current_user",
    }

    with pytest.raises(ValueError, match="includes code-owned fields"):
        gc.bind_selected_response_operation(
            {"operation": '回答当前问题', "selection_owner_role": "other"},
            authoritative_operation,
        )

    with pytest.raises(ValueError, match="conflicts with known input role"):
        gc.bind_selected_response_operation(
            {
                "operation": '回答当前问题',
                "embedded_target_role": "third_party",
            },
            authoritative_operation,
        )

    with pytest.raises(ValueError, match="unknown fields"):
        gc.bind_selected_response_operation(
            {"operation": '回答当前问题', "invented_field": True},
            authoritative_operation,
        )

    with pytest.raises(ValueError, match="lacks operation text"):
        gc.bind_selected_response_operation({}, authoritative_operation)

    progress_rows = [
        {
            "evidence_handle": "ev_p1",
            "semantic_text": '进度：已确认第一步。',
            "authority": "current_event",
            "evidence_ref": {"source_kind": "conversation_evidence", "source_id": "conversation-progress-event:42"},
            "temporal_provenance": {"turn": 3},
        },
        {
            "evidence_handle": "ev_x",
            "semantic_text": '无关。',
            "authority": "contextual_fact_only",
            "evidence_ref": {"source_kind": "episode", "source_id": "ep_1"},
        },
        {
            "evidence_handle": "ev_y",
            "semantic_text": '旧话题。',
            "authority": "current_event",
            "evidence_ref": {"source_kind": "conversation_evidence", "source_id": "old-topic:7"},
        },
    ]
    projected = gc.project_progress_evidence(progress_rows)
    assert [row["evidence_handle"] for row in projected] == ["ev_p1"]
    assert projected[0]["temporal_provenance"] == {"turn": 3}
    progress_rows[0]["temporal_provenance"]["turn"] = 99
    assert projected[0]["temporal_provenance"] == {"turn": 3}

    selection_draft = {
        "selection": '选择直接回应。',
        "selected_response_operation": {"operation": '回答当前问题'},
        "reason": '当前事件要求明确选择。',
        "private_monologue": '我给出具体选择。',
        "target_role_handles": ["self"],
        "evidence_handles": ["ev_1"],
        "expected_consequences": ['回应保持有依据'],
        "confidence": "medium",
    }
    normalized = gc.validate_selection_goal_draft(
        selection_draft,
        evidence_handles={"ev_1"},
        role_handles={"self"},
    )
    assert normalized["selection"] == '选择直接回应。'

    with pytest.raises(ValueError, match="selection goal draft fields are not exact"):
        gc.validate_selection_goal_draft(
            {**selection_draft, "invented_field": True},
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )


def test_required_goal_exhaustion_preserves_existing_fail_closed_contract():
    async def exhaustion_scenario():
        ledger = AttemptLedger({"loss_recovery": 2})

        async def always_structurally_invalid(ctx):
            return StageAttemptOutcome(False, None, None, "structural_contract")

        handle = start_wave(
            [ChainTaskSpec("loss_recovery", ("loss_recovery",), {"loss_recovery": always_structurally_invalid})],
            ledger=ledger,
        )
        result = await handle.complete()
        return result.outcomes["loss_recovery"]

    exhausted_outcome = asyncio.run(exhaustion_scenario())
    exhausted_stage = exhausted_outcome.results[0]
    assert not exhausted_stage.accepted
    assert exhausted_stage.failure is not None
    assert exhausted_stage.failure.failure_class == EXHAUSTION_FAILURE_CLASS
    assert exhausted_stage.failure.error_code == APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
    assert exhausted_stage.failure.repair_attempted is True

    disposition = gc.resolve_goal_disposition("loss_recovery", exhausted_outcome)
    assert disposition.available is False
    assert disposition.bid is None
    assert disposition.error_code == APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE

    async def accepted_scenario():
        ledger = AttemptLedger({"safety": 2})

        async def accepting(ctx):
            return StageAttemptOutcome(
                True,
                {
                    "intention": '核实边界',
                    "desired_outcome": '保持安全',
                    "concrete_detail": '先确认证据',
                    "reason": '角色需要确认当前事件边界。',
                    "private_monologue": '我先看证据。',
                    "target_role_handles": ["self"],
                    "evidence_handles": ["ev_1"],
                    "expected_consequences": ['回应保持有依据'],
                    "confidence": "medium",
                },
                "safety-summary",
                None,
            )

        handle = start_wave(
            [ChainTaskSpec("safety", ("safety",), {"safety": accepting})],
            ledger=ledger,
        )
        result = await handle.complete()
        return result.outcomes["safety"]

    accepted_outcome = asyncio.run(accepted_scenario())
    accepted_disposition = gc.resolve_goal_disposition("safety", accepted_outcome)
    assert accepted_disposition.available is True
    assert accepted_disposition.error_code is None
    assert accepted_disposition.bid == dict(accepted_outcome.results[0].local_state)

    with pytest.raises(ValueError, match="lacks the 'loss_recovery' stage record"):
        gc.resolve_goal_disposition("loss_recovery", accepted_outcome)


# The isolated goal chain registry identity: every kind is one single-stage chain.
def test_isolated_goal_chain_registry_identity():
    from kazusa_ai_chatbot.cognition_core_v3.registry import GOAL_CHAINS

    by_name = {chain.name: chain for chain in GOAL_CHAINS}
    assert set(by_name) == set(GOAL_KINDS)
    for goal_kind, chain in by_name.items():
        assert chain.stages == (goal_kind,)
