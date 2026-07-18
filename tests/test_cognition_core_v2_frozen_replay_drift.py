"""Regressions for defects exposed by the frozen affinity replay."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage
import pytest

from kazusa_ai_chatbot.cognition_episode import (
    build_text_chat_cognitive_episode,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    validate_text_surface_output,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    validate_goal_bid_draft,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_semantic_appraisals,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.consolidation import core as consolidation_core
from kazusa_ai_chatbot.conversation_progress import recorder as progress_recorder
from kazusa_ai_chatbot.internal_monologue_residue import recorder as residue_recorder
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_generator
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.time_boundary import (
    local_time_context_from_storage_utc,
)


STORAGE_TIMESTAMP = "2026-07-15T00:00:00+00:00"
V2_TIMESTAMP = "2026-07-15T00:00:00Z"


def _evidence(index: int) -> dict[str, object]:
    """Build one exact cognition evidence row."""

    return {
        "evidence_handle": f"e{index}",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": f"episode-{index}",
            "occurred_at": f"2026-07-15T00:00:{index:02d}Z",
            "semantic_summary": f"semantic evidence {index}",
        },
        "semantic_text": f"semantic evidence {index}",
        "visible_to": ["q:event_agency"],
    }


def _appraisal(
    *,
    evidence_handle: str,
    propositions: list[dict[str, object]] | None = None,
    deltas: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build one validated-shape semantic appraisal result."""

    return {
        "question_id": "q:event_agency",
        "selected_evidence_handles": [evidence_handle],
        "selected_role_handles": [],
        "propositions": propositions or [],
        "deltas": deltas or [],
        "explanation": "grounded test appraisal",
    }


def _completion_proposition() -> dict[str, object]:
    """Return one completion meaning for the same current event."""

    return {
        "proposition_kind": "completion_meaning",
        "subject_handle": "ce1",
        "evidence_handles": ["e1"],
        "role_assignments": [],
        "semantic_value": "the current event is complete",
    }


def test_same_batch_repeated_terminal_meaning_is_idempotent() -> None:
    """A repeated completion in one batch reinforces one terminal result."""

    state = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=V2_TIMESTAMP,
    )
    comparisons: list[dict[str, object]] = []

    updated = apply_semantic_appraisals(
        state,
        [_appraisal(propositions=[
            _completion_proposition(),
            _completion_proposition(),
            _completion_proposition(),
        ], deltas=[{
            "target_path": "active_events.ce1.outcome_impact",
            "delta": 30,
            "evidence_handles": ["e1"],
            "reason": "the completed event has a material positive outcome",
        }], evidence_handle="e1")],
        [_evidence(1)],
        {
            "ce1": {
                "scope": "user",
                "kind": "event",
                "entity_id": "candidate:event:e1",
            },
        },
        comparisons,
    )

    assert updated["active_events"][0]["status"] == "resolved"
    assert [row["outcome"] for row in comparisons] == [
        "create",
        "resolve",
        "resolve",
    ]


def test_relationship_evidence_retains_the_newest_eight_rows() -> None:
    """Long conversations keep bounded valid relationship provenance."""

    state = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=V2_TIMESTAMP,
    )
    evidence = [_evidence(index) for index in range(1, 11)]
    appraisals = [
        _appraisal(
            evidence_handle=f"e{index}",
            deltas=[{
                "target_path": "relationship.r1.care",
                "delta": 1,
                "evidence_handles": [f"e{index}"],
                "reason": "current interaction supports care",
            }],
        )
        for index in range(1, 11)
    ]

    updated = apply_semantic_appraisals(
        state,
        appraisals,
        evidence,
        {
            "r1": {
                "scope": "user",
                "kind": "relationship",
                "entity_id": state["relationship"]["relationship_id"],
            },
        },
    )

    validate_cognition_state(updated)
    assert [
        row["source_id"]
        for row in updated["relationship"]["evidence_refs"]
    ] == [f"episode-{index}" for index in range(3, 11)]


def test_relationship_connection_tracks_a_current_closeness_gap() -> None:
    """Attachment alone creates no permanent relationship-connection goal."""

    state = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=V2_TIMESTAMP,
    )
    relationship = state["relationship"]
    relationship.update({
        "salience": 90,
        "attachment": 100,
        "care": 100,
        "desired_closeness": 100,
        "perceived_closeness": 0,
        "evidence_refs": [_evidence(1)["evidence_ref"]],
    })
    with_gap = create_deterministic_goals(
        state,
        evidence=[_evidence(1)],
        updated_at=V2_TIMESTAMP,
    )
    connection_goal = next(
        goal
        for goal in with_gap["goals"]
        if goal["goal_kind"] == "relationship_connection"
    )
    assert connection_goal["status"] == "pursuing"

    with_gap["relationship"]["perceived_closeness"] = 100
    closed_gap = create_deterministic_goals(
        with_gap,
        evidence=[_evidence(1)],
        updated_at=V2_TIMESTAMP,
    )
    connection_goal = next(
        goal
        for goal in closed_gap["goals"]
        if goal["goal_kind"] == "relationship_connection"
    )
    assert connection_goal["status"] == "satisfied"


def _progress_payload() -> dict[str, object]:
    """Build one recorder response with an actor-preserving obligation."""

    return {
        "status": "active",
        "episode_label": "shared reward",
        "continuity": "same_episode",
        "conversation_mode": "playful exchange",
        "episode_phase": "waiting for the user's choice",
        "topic_momentum": "moving toward a choice",
        "current_thread": "the user may choose a reward",
        "user_goal": "choose a reward",
        "current_blocker": "",
        "user_state_updates": [],
        "assistant_moves": ["offered a choice"],
        "overused_moves": [],
        "open_loops": ["reward choice remains open"],
        "interaction_obligations": [{
            "actor": "the character",
            "action": "provide the reward selected by the user",
            "beneficiary": "the user",
            "precondition": "the user selects one reward",
            "expected_outcome": "the selected reward is provided",
            "status": "active",
            "source_kind": "assistant_response",
        }],
        "resolved_threads": [],
        "avoid_reopening": [],
        "emotional_trajectory": "warm and playful",
        "next_affordances": ["wait for the user's selection"],
        "progression_guidance": "do not swap who owes the reward",
    }


def test_progress_recorder_preserves_obligation_roles_and_source() -> None:
    """Generated surface content cannot become an unlabeled user debt."""

    validated = progress_recorder.validate_recorder_output(_progress_payload())

    assert validated["interaction_obligations"] == [
        _progress_payload()["interaction_obligations"][0]
    ]


def _episode(user_input: str = "Current request") -> dict[str, object]:
    """Build a canonical text episode for connector tests."""

    return build_text_chat_cognitive_episode(
        episode_id="episode-1",
        percept_id="percept-1",
        storage_timestamp_utc=STORAGE_TIMESTAMP,
        local_time_context=local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP,
        ),
        user_input=user_input,
        platform="qq",
        platform_channel_id="638473184",
        channel_type="group",
        platform_message_id="message-1",
        platform_user_id="673225019",
        global_user_id="user-1",
        user_name="Participant",
        active_turn_platform_message_ids=["message-1"],
        active_turn_conversation_row_ids=["row-1"],
        debug_modes={},
        target_addressed_user_ids=["kazusa"],
        target_broadcast=False,
    )


def test_connector_separates_current_event_continuity_and_private_residue() -> None:
    """Current input, public continuity, and private continuity stay distinct."""

    state = {
        "cognitive_episode": _episode(),
        "global_user_id": "user-1",
        "decontexualized_input": "Current request",
        "user_multimedia_input": [],
        "rag_result": {"memory_evidence": []},
        "conversation_progress": {
            **_progress_payload(),
            "turn_count": 5,
        },
        "internal_monologue_residue_context": (
            "I still care about answering without swapping roles."
        ),
    }
    payload = build_cognition_input_from_global_state(
        state,
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=V2_TIMESTAMP,
        ),
        character_state=build_character_production_state(
            updated_at=V2_TIMESTAMP,
        ),
    )

    assert payload["scene_context"]["semantic_scene"] == "Current request"
    continuity = payload["scene_context"]["conversation_continuity"]
    assert "the character" in continuity
    assert "provide the reward" in continuity
    assert payload["private_continuity_context"].startswith("I still care")
    assert "I still care" not in json.dumps(payload["scene_context"])


def test_goal_bid_contract_separates_reason_from_private_monologue() -> None:
    """An analytic bid reason is not reused as first-person inner speech."""

    validated = validate_goal_bid_draft(
        {
            "intention": "clarify who acts",
            "desired_outcome": "roles stay correct",
            "concrete_detail": "the character provides the selected reward",
            "reason": "the current request requires role-preserving clarification",
            "private_monologue": "I want to get this right for them.",
            "target_role_handles": [],
            "evidence_handles": ["e1"],
            "expected_consequences": ["the response preserves actor direction"],
            "confidence": "high",
        },
        evidence_handles={"e1"},
        role_handles=set(),
    )

    assert validated["reason"].startswith("the current request")
    assert validated["private_monologue"].startswith("I want")


def test_residue_recorder_reconciles_private_thought_with_visible_outcome() -> None:
    """Post-turn residue sees bounded actual dialog and surface constraints."""

    recorder_input = residue_recorder._build_recorder_input({
        "cognitive_episode": _episode(),
        "internal_monologue": "I want to preserve who acts for whom.",
        "character_profile": {"name": "Kazusa", "global_user_id": "kazusa"},
        "platform": "qq",
        "platform_channel_id": "638473184",
        "channel_type": "group",
        "global_user_id": "user-1",
        "user_name": "Participant",
        "logical_stance": "clarify roles",
        "character_intent": "reply clearly",
        "emotional_appraisal": "care",
        "interaction_subtext": "keep the exchange mutual",
        "social_distance": "close",
        "relational_dynamic": "trusted",
        "final_dialog": ["I will provide the reward after you choose it."],
        "text_surface_output_v2": {
            "content_plan": "Character provides the user-selected reward.",
            "visible_boundaries": ["Do not swap actor and beneficiary."],
        },
        "internal_monologue_residue_context": "",
    })

    assert recorder_input is not None
    assert "I will provide" in recorder_input["visible_outcome_summary"]
    assert "Character provides" in recorder_input["surface_content_plan"]
    assert recorder_input["visible_boundaries"] == [
        "Do not swap actor and beneficiary."
    ]


def _surface_output() -> dict[str, object]:
    """Build the post-fix semantic surface contract."""

    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": "The character gives the user the selected reward.",
        "content_requirements": [
            "Actor: character; action: give; beneficiary: user; condition: selection.",
        ],
        "visible_boundaries": ["Keep actor and beneficiary direction unchanged."],
        "addressee_plan": ["current user"],
        "style_guidance": "reserved, vivid, and warm",
        "selected_surface_intent": "confirm the reward direction",
        "permitted_action_results": [],
    }


def test_text_surface_contract_carries_requirements_without_raw_voice() -> None:
    """The final renderer receives requirements without raw character voice."""

    validated = validate_text_surface_output(_surface_output())

    assert "character_voice_context" not in validated
    assert validated["content_requirements"][0].startswith("Actor: character")


@pytest.mark.asyncio
async def test_dialog_semantic_verdict_triggers_one_bounded_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detected actor inversion is repaired once before dialog is returned."""

    for recorder_name in (
        "record_llm_stage_event",
        "record_model_contract_event",
        "record_dialog_quality_event",
    ):
        monkeypatch.setattr(
            dialog_module.event_logging,
            recorder_name,
            AsyncMock(),
        )
    trace_recorder = AsyncMock()
    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        trace_recorder,
    )
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps({
            "final_dialog": ["You will give me the reward after I choose."],
        })),
        AIMessage(content=json.dumps({
            "final_dialog": ["Choose it, and I will give you the reward."],
        })),
    ])
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps({
            "aligned": False,
            "issues": ["Actor and beneficiary are reversed."],
        })),
        AIMessage(content=json.dumps({
            "aligned": True,
            "issues": [],
        })),
    ])
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps({
            "aligned": True,
            "issues": [],
        })),
        AIMessage(content=json.dumps({
            "aligned": True,
            "issues": [],
        })),
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    result = await dialog_generator({
        "internal_monologue": "I should preserve the direction.",
        "cognitive_episode": _episode(),
        "text_surface_output_v2": _surface_output(),
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "673225019",
        "platform_bot_id": "kazusa",
        "global_user_id": "user-1",
        "user_name": "Participant",
        "user_profile": {},
        "character_profile": {"name": "Kazusa"},
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "dialog_usage_mode": "unit_test",
        "llm_trace_id": "trace-1",
    })

    assert result["final_dialog"] == [
        "Choose it, and I will give you the reward."
    ]
    assert generator_llm.ainvoke.await_count == 2
    assert semantic_llm.ainvoke.await_count == 2
    assert surface_llm.ainvoke.await_count == 2
    verifier_payloads = {}
    verifier_calls = (
        semantic_llm.ainvoke.await_args_list
        + surface_llm.ainvoke.await_args_list
    )
    for call in verifier_calls:
        messages = call.args[0]
        verifier_payloads[messages[0].content] = json.loads(
            messages[1].content
        )
    semantic_payload = next(
        payload
        for prompt, payload in verifier_payloads.items()
        if "semantic fidelity" in prompt
    )
    surface_payload = next(
        payload
        for prompt, payload in verifier_payloads.items()
        if "surface integrity" in prompt
    )
    assert set(semantic_payload) == {
        "candidate_final_dialog",
        "candidate_role_frame",
        "current_visible_percepts",
    }
    assert set(surface_payload) == {
        "candidate_final_dialog",
        "permitted_action_results",
    }
    trace_stage_names = [
        call.kwargs["stage_name"]
        for call in trace_recorder.await_args_list
    ]
    assert trace_stage_names[0] == "dialog_generator"
    assert set(trace_stage_names[1:3]) == {
        "dialog_semantic_fidelity_verifier",
        "dialog_surface_integrity_verifier",
    }
    assert trace_stage_names[3] == "dialog_generator_repair"
    assert set(trace_stage_names[4:]) == {
        "dialog_semantic_fidelity_recheck",
        "dialog_surface_integrity_recheck",
    }


def test_consolidation_dedup_uses_canonical_v2_memory_candidates() -> None:
    """Consolidation no longer requires the legacy user-image envelope."""

    dedup_keys = consolidation_core._build_existing_dedup_keys({
        "rag_result": {
            "user_memory_unit_candidates": [
                {"dedup_key": "Breakfast.Preference"},
                {"dedup_key": "breakfast.preference"},
            ],
        },
    })

    assert dedup_keys == {"breakfast.preference"}
