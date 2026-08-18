"""Deterministic tests for V3 action planning, goal resolution, and isolated authorizations."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
)
from kazusa_ai_chatbot.cognition_episode import GOAL_CONTINUATION_REF_VERSION
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_GOAL_PROGRESS_VERSION,
)
from kazusa_ai_chatbot.time_boundary import (
    local_llm_datetime_to_storage_utc_iso,
)
from kazusa_ai_chatbot.cognition_core_v3 import action_selection as act


def _decision() -> dict:
    return {
        "action_requests": [
            {"action_handle": "act_1", "decision": "run the effect"},
        ],
        "resolver_requests": [
            {"resolver_handle": "res_1", "bid_handle": "b1", "decision": "resume work"},
        ],
        "goal_resolution": "pending_work",
        "resolver_pending_resolution": {"choice": "defer"},
        "resolver_goal_progress": None,
    }


def _primary_bid(
    stance: str | None = "accept",
    applicability: str = "relationship_sensitive",
) -> dict:
    bid = {
        "branch_id": "ordinary_response",
        "intention": "respond to the request",
        "goal_ref": {"scope": "character", "kind": "goal", "entity_id": "g1"},
        "target_roles": ["self"],
        "reason": "the turn is relationship-sensitive",
    }
    if stance is not None:
        bid["relational_willingness"] = {
            "schema_version": "relational_willingness.v2",
            "applicability": applicability,
            "stance": stance,
            "current_user_relationship_state": "established",
            "reason": "关系已建立，按当前请求协商。",
            "evidence_handles": ["ev_1"],
        }
    return bid


def test_non_accepting_stance_suppresses_effects():
    for stance in ("reject", "deflect", "negotiate", "conditional_accept"):
        suppressed_decision, suppressed = act.apply_stance_suppression(
            _decision(),
            _primary_bid(stance),
        )
        assert suppressed is True
        assert suppressed_decision["action_requests"] == []
        assert suppressed_decision["resolver_requests"] == []
        assert suppressed_decision["goal_resolution"] == "answerable_now"
        assert suppressed_decision["resolver_pending_resolution"] is None
        assert suppressed_decision["resolver_goal_progress"] is None

    # An accepting sensitive stance passes the decision through unchanged.
    passthrough, suppressed = act.apply_stance_suppression(
        _decision(),
        _primary_bid("accept"),
    )
    assert suppressed is False
    assert passthrough == _decision()

    # A non-sensitive applicability never suppresses by itself.
    insensitive, suppressed = act.apply_stance_suppression(
        _decision(),
        _primary_bid("reject", applicability="not_relationship_sensitive"),
    )
    assert suppressed is False
    assert insensitive["action_requests"] == _decision()["action_requests"]

    # A bid without any willingness decision passes through unchanged.
    bare, suppressed = act.apply_stance_suppression(
        _decision(),
        {k: v for k, v in _primary_bid().items() if k != "relational_willingness"},
    )
    assert suppressed is False

    # Suppressed turns keep resolver effects empty without authorization work.
    rows, goal_resolution = act.settle_resolver_outcome(
        _decision(),
        suppressed=True,
        action_requests_materialized=0,
        resolver_requests_materialized=[],
    )
    assert rows == []
    assert goal_resolution == "pending_work"

    # Owner denial with no surviving effects escalates to blocked.
    rows, goal_resolution = act.settle_resolver_outcome(
        _decision(),
        suppressed=False,
        action_requests_materialized=0,
        resolver_requests_materialized=[],
    )
    assert rows == []
    assert goal_resolution == "blocked"

    # A settled answerable_now resolution never escalates and clears effects.
    settled = dict(_decision())
    settled["goal_resolution"] = "answerable_now"
    rows, goal_resolution = act.settle_resolver_outcome(
        settled,
        suppressed=False,
        action_requests_materialized=1,
        resolver_requests_materialized=[{"resolver_handle": "res_1"}],
    )
    assert rows == []
    assert goal_resolution == "answerable_now"

    # Authorized materialization without denial preserves the turn's resolution.
    rows, goal_resolution = act.settle_resolver_outcome(
        _decision(),
        suppressed=False,
        action_requests_materialized=1,
        resolver_requests_materialized=[{"resolver_handle": "res_1"}],
    )
    assert [row["resolver_handle"] for row in rows] == ["res_1"]
    assert goal_resolution == "pending_work"


def test_action_and_resolver_authorizers_receive_fresh_minimal_context():
    # A turn without scene context projects an empty fresh boundary.
    assert act.project_scene_context_for_action_planning(None) == {}

    scene = {
        "channel_scope": "group",
        "character_role": "self",
        "semantic_scene": "dormitory",
        "public_group_scene": True,
        "conversation_continuity": "continuous",
        "semantic_temporal_context": {"date": "2026-07-01"},
        "current_user_role": "friend",
        "character_sleep_phase": "awake",
        "participant_bindings": [{"role": "self", "handle": "r1"}],
        "sibling_chain_output": {"rejected_candidate": "must not leak"},
        "raw_trace": ["trace row must not leak"],
    }
    projected = act.project_scene_context_for_action_planning(scene)

    assert set(projected) == {
        "channel_scope",
        "character_role",
        "semantic_scene",
        "public_group_scene",
        "conversation_continuity",
        "semantic_temporal_context",
        "current_user_role",
        "character_sleep_phase",
        "participant_bindings",
    }

    # The projection is an independent copy: input mutation cannot pollute it.
    scene["participant_bindings"][0]["role"] = "mutated"
    assert projected["participant_bindings"][0]["role"] == "self"

    evidence = [
        {
            "evidence_handle": "ev_1",
            "evidence_ref": {"source_kind": "episode"},
            "semantic_text": "the current request text",
        },
        {
            "evidence_handle": "mem_1",
            "evidence_ref": {"source_kind": "promoted_memory"},
            "memory_scope": "current_user_continuity",
            "semantic_text": "a history note",
        },
    ]
    rows = act.project_authorizer_evidence_rows(evidence)
    assert [row["handle"] for row in rows] == ["ev_1", "mem_1"]
    assert rows[0]["provenance_role"] == "current_episode"
    assert rows[1]["provenance_role"] == "current_user_history_only"

    # Unknown provenance fails closed: no authorizer input ever carries an
    # inferred or free-text role.
    with pytest.raises(CognitionContractError):
        act.project_authorizer_evidence_rows(
            [
                {
                    "evidence_handle": "x_1",
                    "evidence_ref": {"source_kind": "invented_source"},
                    "semantic_text": "no role can be derived",
                }
            ]
        )


def test_invalid_authority_proposal_denies_all_effects():
    evidence = [
        {
            "evidence_handle": "ev_1",
            "evidence_ref": {"source_kind": "episode"},
            "semantic_text": "the current request text",
            "authority": "current_event",
        }
    ]

    def valid_proposal() -> dict:
        return {
            "schema_version": "scheduled_authority_proposal.v1",
            "temporal_alignment": "aligned",
            "authorized_content_summary": (
                "Remind about the earlier commitment at the agreed time."
            ),
            "authorized_detail_refs": [
                {
                    "evidence_handle": "ev_1",
                    "semantic_summary": "The current request text.",
                    "provenance_role": "current_event",
                }
            ],
        }

    valid_row = {
        "action_handle": "fs_1",
        "decision": "2026-07-02 09:00",
        "scheduled_authority_proposal": valid_proposal(),
    }
    kinds = {"fs_1": "future_speak"}

    # A fully valid future-speak row validates through with the trigger strictly
    # after a one-day-earlier accepted instant.
    validated = act.future_speak_proposal_contract(
        valid_row,
        action_kind="future_speak",
        evidence=evidence,
        accepted_at_utc=local_llm_datetime_to_storage_utc_iso("2026-07-01 09:00"),
    )
    assert validated["temporal_alignment"] == "aligned"

    # A non-future-speak row may never carry a proposal.
    foreign_row = {
        "action_handle": "act_1",
        "decision": "run the effect",
        "scheduled_authority_proposal": valid_proposal(),
    }
    with pytest.raises(ValueError, match="only valid for future_speak"):
        act.future_speak_proposal_contract(
            foreign_row,
            action_kind="background_work",
            evidence=evidence,
            accepted_at_utc="",
        )

    # A future-speak row without its proposal is a contract violation.
    missing_row = {
        "action_handle": "fs_1",
        "decision": "2026-07-02 09:00",
    }
    with pytest.raises(ValueError, match="requires scheduled_authority_proposal"):
        act.future_speak_proposal_contract(
            missing_row,
            action_kind="future_speak",
            evidence=evidence,
            accepted_at_utc="",
        )

    # An unaligned proposal is denied even though its fields are well-formed.
    misaligned = dict(valid_row)
    misaligned["scheduled_authority_proposal"] = valid_proposal()
    misaligned["scheduled_authority_proposal"]["temporal_alignment"] = (
        "past_or_not_future"
    )
    with pytest.raises(ValueError, match="not aligned"):
        act.future_speak_proposal_contract(
            misaligned,
            action_kind="future_speak",
            evidence=evidence,
            accepted_at_utc="",
        )

    # A trigger not strictly after the accepted instant is denied.
    same_instant = dict(valid_row)
    with pytest.raises(ValueError, match="must be later than accepted time"):
        act.future_speak_proposal_contract(
            same_instant,
            action_kind="future_speak",
            evidence=evidence,
            accepted_at_utc=local_llm_datetime_to_storage_utc_iso("2026-07-02 09:00"),
        )

    # A trigger text the storage converter cannot normalize is denied.
    bad_trigger = dict(valid_row)
    bad_trigger["decision"] = "next week sometime"
    with pytest.raises(ValueError, match="trigger time is invalid"):
        act.future_speak_proposal_contract(
            bad_trigger,
            action_kind="future_speak",
            evidence=evidence,
            accepted_at_utc=local_llm_datetime_to_storage_utc_iso("2026-07-01 09:00"),
        )

    # One invalid future-speak row denies every effect on the candidate.
    good_row = dict(valid_row)
    bad_row = {**valid_row, "action_handle": "fs_2"}
    bad_row["scheduled_authority_proposal"] = valid_proposal()
    bad_row["decision"] = "next week sometime"
    with pytest.raises(ValueError, match="trigger time is invalid"):
        act.validate_future_speak_proposal_rows(
            [good_row, bad_row],
            action_kind_by_handle={"fs_1": "future_speak", "fs_2": "future_speak"},
            evidence=evidence,
            accepted_at_utc=local_llm_datetime_to_storage_utc_iso("2026-07-01 09:00"),
        )

    # The same candidate without the offending row validates cleanly.
    act.validate_future_speak_proposal_rows(
        [good_row],
        action_kind_by_handle={"fs_1": "future_speak"},
        evidence=evidence,
        accepted_at_utc=local_llm_datetime_to_storage_utc_iso("2026-07-01 09:00"),
    )


def _current_progress() -> dict:
    return {
        "schema_version": RESOLVER_GOAL_PROGRESS_VERSION,
        "original_goal": "finish the report",
        "current_focus": "drafting the first section",
        "deliverables": [
            {"description": "report draft", "status": "pending", "note": ""},
        ],
        "missing_user_inputs": [],
        "evidence_dependencies": [],
        "attempted_paths": [],
        "source_backed_facts": [],
        "assumptions_or_inferences": [],
        "blockers": [],
        "final_response_requirements": [],
    }


def test_selected_operation_and_goal_progress_are_preserved():
    episode = {
        "episode_id": "ep_1",
        "trigger_source": "user_message",
        "origin_metadata": {"platform_message_id": "msg_9"},
    }
    primary_bid = _primary_bid("accept")
    del primary_bid["relational_willingness"]
    primary_bid.update(
        {
            "branch_id": "bond_protection",
            "intention": "protect the bond boundary",
            "reason": "the user needs an explicit boundary",
            "goal_ref": {"scope": "character", "kind": "goal", "entity_id": "g2"},
        }
    )

    continuation_ref = act.bind_goal_continuation_ref(episode, primary_bid)
    assert continuation_ref["schema_version"] == GOAL_CONTINUATION_REF_VERSION
    assert continuation_ref["source_episode_id"] == "ep_1"
    assert continuation_ref["source_message_id"] == "msg_9"
    assert continuation_ref["branch_id"] == "bond_protection"
    assert continuation_ref["goal_ref"]["entity_id"] == "g2"

    # A tool-result episode must present its validated origin metadata ref.
    tool_episode = {
        "episode_id": "ep_1",
        "trigger_source": "tool_result",
        "origin_metadata": {"goal_continuation_ref": dict(continuation_ref)},
    }
    assert act.bind_goal_continuation_ref(tool_episode, primary_bid) == (
        continuation_ref
    )

    tool_episode_without_meta = {
        "episode_id": "ep_1",
        "trigger_source": "tool_result",
    }
    with pytest.raises(ValueError, match="tool-result origin metadata is invalid"):
        act.bind_goal_continuation_ref(tool_episode_without_meta, primary_bid)

    operation = {
        "operation": "ask_clarification",
        "embedded_actor_role": "",
        "embedded_target_role": "",
        "response_owner_role": "self",
        "selection_owner_role": "character",
        "selection_required": True,
    }
    primary_bid["selected_response_operation"] = dict(operation)

    intention = act.build_selected_intention(
        primary_bid,
        route="speech",
        goal_continuation_ref=continuation_ref,
    )
    assert intention["selected_branch_id"] == "bond_protection"
    assert intention["route"] == "speech"
    assert intention["intention"] == "protect the bond boundary"
    assert intention["reason"] == "the user needs an explicit boundary"
    assert intention["selected_response_operation"] == operation

    # The carried operation and continuation ref are independent copies.
    primary_bid["selected_response_operation"]["operation"] = "mutated"
    continuation_ref["branch_id"] = "mutated"
    assert intention["selected_response_operation"]["operation"] == (
        "ask_clarification"
    )
    assert intention["goal_continuation_ref"]["branch_id"] == "bond_protection"

    # A bid without a selected operation emits no key at all.
    bare_bid = {k: v for k, v in primary_bid.items() if k != "selected_response_operation"}
    bare_intention = act.build_selected_intention(
        bare_bid,
        route="evidence",
        goal_continuation_ref=continuation_ref,
    )
    assert "selected_response_operation" not in bare_intention

    current = _current_progress()
    merged = act.validate_goal_progress_choice(
        {"current_focus": "revising with the user"},
        current_goal_progress=current,
    )
    assert merged is not None
    assert merged["current_focus"] == "revising with the user"
    assert merged["original_goal"] == "finish the report"
    assert merged["schema_version"] == RESOLVER_GOAL_PROGRESS_VERSION

    # A null choice carries no state change.
    assert (
        act.validate_goal_progress_choice(
            None,
            current_goal_progress=current,
        )
        is None
    )

    with pytest.raises(ValueError, match="must be an object or null"):
        act.validate_goal_progress_choice(
            "drafting",
            current_goal_progress=current,
        )

    with pytest.raises(ValueError, match="requires existing current state"):
        act.validate_goal_progress_choice(
            {"current_focus": "revising"},
            current_goal_progress=None,
        )

    empty_shell = _current_progress()
    empty_shell["current_focus"] = ""
    empty_shell["deliverables"] = []
    with pytest.raises(ValueError, match="cannot update an empty shell"):
        act.validate_goal_progress_choice(
            {"current_focus": "revising"},
            current_goal_progress=empty_shell,
        )

    with pytest.raises(ValueError, match="protocol fields are code-owned"):
        act.validate_goal_progress_choice(
            {"original_goal": "rewritten by the model"},
            current_goal_progress=current,
        )

    with pytest.raises(ValueError, match="update fields are invalid"):
        act.validate_goal_progress_choice(
            {"invented_field": True},
            current_goal_progress=current,
        )
