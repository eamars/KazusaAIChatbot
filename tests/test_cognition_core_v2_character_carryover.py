"""Focused native character carry-over decision and privacy tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import kazusa_ai_chatbot.cognition_core_v2.character_carryover as character_carryover
from kazusa_ai_chatbot.cognition_core_v2.character_carryover import (
    CharacterCarryoverServicesV1,
    _build_native_appraisal,
    _validate_decision_payload,
    run_character_carryover_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_semantic_appraisals,
    apply_state_update,
    canonical_event_entity_id,
)
from kazusa_ai_chatbot.llm_interface.contracts import LLMCallConfig


NOW = "2026-08-02T00:00:00Z"
OFFSET_OCCURRED_AT = "2026-08-12T23:28:57.048343+00:00"
OFFSET_EFFECTIVE_AT = "2026-08-12T23:30:01.676468+00:00"


def _evidence(source_id: str, *, text: str = "closed operational event") -> dict[str, str]:
    """Build one ref-complete current-episode evidence view."""

    return {
        "source_kind": "episode",
        "source_id": source_id,
        "occurred_at": NOW,
        "semantic_summary": text,
        "evidence_handle": f"evidence:{source_id}",
    }


def _normalized_evidence(
    handle: str,
    source_id: str,
) -> dict[str, object]:
    """Build reducer-facing evidence with a distinct source identity.

    Args:
        handle: Opaque evidence handle cited by the semantic appraisal.
        source_id: Source episode identity associated with the handle.

    Returns:
        Reducer-facing evidence row containing the opaque handle and ref.
    """

    return {
        "evidence_handle": handle,
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": source_id,
            "occurred_at": NOW,
            "semantic_summary": "closed operational event",
        },
        "semantic_text": "closed operational event",
    }


def _offset_evidence(source_key: str, *, text: str) -> dict[str, str]:
    """Build one router-shaped evidence row with an offset UTC timestamp."""

    return {
        "source_kind": "episode",
        "source_id": "episode-offset",
        "occurred_at": OFFSET_OCCURRED_AT,
        "semantic_summary": "character operational event",
        "semantic_text": text,
        "evidence_handle": f"evidence:{source_key}",
    }


class _NoCallLLM:
    """Fail if a source-free empty episode invokes the model."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        self.calls += 1
        raise AssertionError("empty carry-over input must not call the model")


class _UnsafeOutputLLM:
    """Return a model-authored emotion/cause class that must be rejected."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        return SimpleNamespace(
            content=json.dumps({
                "emotion_id": "anger",
                "cause_class": "relationship",
            })
        )


class _ValidNativeOutputLLM:
    """Return one source-free appraisal with no model-authored emotion."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        return SimpleNamespace(
            content=json.dumps({
                "action": "apply",
                "reason_code": "lingering_character_effect",
                "question_id": "character_carryover",
                "propositions": [
                    {
                        "kind": "event",
                        "semantic_value": "persistent operational consequence",
                        "evidence_handles": ["evidence:episode"],
                        "role_assignments": [
                            {
                                "role": "actor",
                                "entity_handle": "self",
                            },
                        ],
                        "deltas": {"outcome_impact": -40},
                    }
                ],
            })
        )


class _SequenceLLM:
    """Return bounded model outputs in order for retry tests."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        if self.calls >= len(self.outputs):
            raise AssertionError("carry-over called beyond its bounded outputs")
        output = self.outputs[self.calls]
        self.calls += 1
        return SimpleNamespace(content=output)


def _valid_apply_output(
    *,
    evidence_handle: str,
    axis: str,
    delta: int,
    actor_handle: str,
    target_handle: str,
) -> str:
    """Build one exact apply response for deterministic carry-over tests."""

    output = {
        "action": "apply",
        "reason_code": "lingering_character_effect",
        "question_id": "character_carryover",
        "propositions": [{
            "kind": "event",
            "semantic_value": "persistent operational consequence",
            "evidence_handles": [evidence_handle],
            "role_assignments": [
                {"role": "actor", "entity_handle": actor_handle},
                {"role": "target", "entity_handle": target_handle},
            ],
            "deltas": {axis: delta},
        }],
    }
    serialized_output = json.dumps(output)
    return serialized_output


def _services(llm: object) -> CharacterCarryoverServicesV1:
    """Build the carry-over service bundle with the required route config."""

    config = LLMCallConfig(
        stage_name="character_carryover_test",
        route_name="COGNITION_LLM_CHARACTER_CARRYOVER",
        base_url="http://test.invalid",
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        top_p=None,
        top_k=None,
        max_completion_tokens=8192,
        presence_penalty=None,
    )
    return CharacterCarryoverServicesV1(llm=llm, config=config)


@pytest.mark.asyncio
async def test_empty_or_incomplete_episode_returns_no_change_without_a_call() -> None:
    """No ref-complete evidence is a deterministic no-change terminal result."""

    llm = _NoCallLLM()
    result = await run_character_carryover_cognition(
        source_episode_id="episode-empty",
        evidence=[],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(llm),
    )

    assert result.disposition == "no_change"
    assert result.decision.action == "no_change"
    assert result.attempts == 0
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_model_authored_emotion_or_cause_class_is_rejected() -> None:
    """The model proposes typed deltas; native code owns causes and emotions."""

    result = await run_character_carryover_cognition(
        source_episode_id="episode-unsafe",
        evidence=[_evidence("unsafe")],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(_UnsafeOutputLLM()),
    )

    assert result.decision.privacy_disposition == "unsafe"
    assert result.disposition == "degraded"
    assert result.decision.semantic_appraisal is None
    assert result.state_update is None


@pytest.mark.asyncio
async def test_valid_carryover_can_apply_one_native_state_update() -> None:
    """One accepted appraisal yields at most one source-free replacement."""

    result = await run_character_carryover_cognition(
        source_episode_id="episode",
        evidence=[_evidence("episode")],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(_ValidNativeOutputLLM()),
    )

    assert result.disposition == "apply"
    assert result.state_update is not None
    assert result.state_update["state_scope"] == "character"
    replacement = result.state_update["replacement_state"]
    assert len(replacement["active_events"]) <= 32
    assert result.decision.semantic_appraisal is not None


def test_candidate_root_binds_to_first_opaque_evidence_handle() -> None:
    """Candidate identity follows the first cited handle and its source."""

    evidence = [
        _normalized_evidence("evidence:first", "episode:first"),
        _normalized_evidence("evidence:second", "episode:second"),
    ]
    appraisal = {
        "selected_evidence_handles": [
            "evidence:first",
            "evidence:second",
        ],
        "selected_role_handles": ["self"],
        "propositions": [{
            "kind": "event",
            "semantic_value": "persisted operational consequence",
            "evidence_handles": [
                "evidence:first",
                "evidence:second",
            ],
            "role_assignments": [
                {"role": "actor", "entity_handle": "self"},
            ],
            "deltas": {"outcome_impact": -20},
        }],
    }

    semantic_result, handle_to_ref = _build_native_appraisal(
        appraisal,
        evidence=evidence,
    )

    assert handle_to_ref["cc1"]["entity_id"] == (
        "candidate:event:evidence:first"
    )
    comparison_results: list[dict[str, object]] = []
    base_state = build_character_production_state(updated_at=NOW)
    apply_semantic_appraisals(
        base_state,
        [semantic_result],
        evidence,
        handle_to_ref,
        comparison_results,
    )

    assert comparison_results[0]["current_event_ref"]["entity_id"] == (
        canonical_event_entity_id(base_state, evidence[0]["evidence_ref"])
    )
    assert comparison_results[0]["evidence_refs"] == [
        evidence[0]["evidence_ref"]
    ]


@pytest.mark.parametrize(
    "candidate_id",
    [
        "candidate:event",
        "candidate:relationship:evidence:first",
        "candidate:event:evidence:missing",
    ],
)
def test_malformed_candidate_root_fails_closed(
    candidate_id: str,
) -> None:
    """Malformed or mismatched candidate ids cannot create a causal root."""

    evidence = [_normalized_evidence("evidence:first", "episode:first")]
    appraisal = {
        "selected_evidence_handles": ["evidence:first"],
        "selected_role_handles": ["self"],
        "propositions": [{
            "kind": "event",
            "semantic_value": "persisted operational consequence",
            "evidence_handles": ["evidence:first"],
            "role_assignments": [
                {"role": "actor", "entity_handle": "self"},
            ],
            "deltas": {"outcome_impact": -20},
        }],
    }
    semantic_result, handle_to_ref = _build_native_appraisal(
        appraisal,
        evidence=evidence,
    )
    handle_to_ref["cc1"]["entity_id"] = candidate_id

    with pytest.raises(
        CognitionStateError,
        match="causal candidate evidence does not match its source",
    ):
        apply_semantic_appraisals(
            build_character_production_state(updated_at=NOW),
            [semantic_result],
            evidence,
            handle_to_ref,
        )


def test_external_offence_roles_reach_native_reducer() -> None:
    """An external actor and self target remain distinct in native state."""

    evidence = [_normalized_evidence("evidence:offence", "episode:offence")]
    appraisal = {
        "selected_evidence_handles": ["evidence:offence"],
        "selected_role_handles": ["self", "unspecified_other"],
        "propositions": [{
            "kind": "event",
            "semantic_value": "external deliberate obstruction",
            "evidence_handles": ["evidence:offence"],
            "role_assignments": [
                {"role": "actor", "entity_handle": "unspecified_other"},
                {"role": "target", "entity_handle": "self"},
            ],
            "deltas": {
                "harm": 40,
                "unfairness": 40,
                "intentionality": 40,
            },
        }],
    }

    semantic_result, handle_to_ref = _build_native_appraisal(
        appraisal,
        evidence=evidence,
    )
    state = apply_semantic_appraisals(
        build_character_production_state(updated_at=NOW),
        [semantic_result],
        evidence,
        handle_to_ref,
    )["updated_state"]

    event = state["active_events"][0]
    assert {
        (role["role"], role["entity_kind"], role["entity_id"])
        for role in event["role_refs"]
    } == {
        ("actor", "third_party", "operational:unspecified_other"),
        ("target", "character", "character:global"),
    }


def test_disgust_target_role_is_preserved() -> None:
    """A typed target role satisfies the native disgust guard."""

    evidence = [_normalized_evidence("evidence:contamination", "episode:contamination")]
    appraisal = {
        "selected_evidence_handles": ["evidence:contamination"],
        "selected_role_handles": ["self", "unspecified_other"],
        "propositions": [{
            "kind": "event",
            "semantic_value": "external norm contamination",
            "evidence_handles": ["evidence:contamination"],
            "role_assignments": [
                {"role": "actor", "entity_handle": "unspecified_other"},
                {"role": "target", "entity_handle": "self"},
            ],
            "deltas": {
                "contamination_risk": 40,
                "norm_violation": 40,
            },
        }],
    }

    semantic_result, handle_to_ref = _build_native_appraisal(
        appraisal,
        evidence=evidence,
    )
    state = apply_semantic_appraisals(
        build_character_production_state(updated_at=NOW),
        [semantic_result],
        evidence,
        handle_to_ref,
    )["updated_state"]
    state = apply_state_update(state, updated_at=NOW)

    assert any(
        activation["emotion_id"] == "disgust"
        for activation in state["affect_activations"]
    )


def test_exact_carryover_schema_has_no_defaults() -> None:
    """Missing semantic fields and roles fail closed without parser defaults."""

    evidence = [_evidence("schema")]
    complete = json.loads(
        _valid_apply_output(
            evidence_handle="evidence:schema",
            axis="harm",
            delta=20,
            actor_handle="unspecified_other",
            target_handle="self",
        )
    )

    for missing_key in ("action", "reason_code"):
        candidate = dict(complete)
        candidate.pop(missing_key)
        assert _validate_decision_payload(
            candidate,
            evidence=evidence,
        ) is None

    for missing_key in ("semantic_value", "role_assignments"):
        candidate = dict(complete)
        proposition = dict(candidate["propositions"][0])
        proposition.pop(missing_key)
        candidate["propositions"] = [proposition]
        assert _validate_decision_payload(
            candidate,
            evidence=evidence,
        ) is None


@pytest.mark.asyncio
async def test_conflicting_role_assignment_requests_replacement() -> None:
    """A repeated semantic role is replaced before native reduction."""

    conflicting = json.loads(
        _valid_apply_output(
            evidence_handle="evidence:conflict",
            axis="harm",
            delta=40,
            actor_handle="self",
            target_handle="self",
        )
    )
    conflicting["propositions"][0]["role_assignments"] = [
        {"role": "actor", "entity_handle": "self"},
        {"role": "actor", "entity_handle": "unspecified_other"},
    ]
    conflicting_output = json.dumps(conflicting)
    evidence = [_evidence("conflict")]
    assert _validate_decision_payload(
        conflicting,
        evidence=evidence,
    ) is None

    llm = _SequenceLLM([
        conflicting_output,
        _valid_apply_output(
            evidence_handle="evidence:conflict",
            axis="harm",
            delta=40,
            actor_handle="unspecified_other",
            target_handle="self",
        ),
    ])
    result = await run_character_carryover_cognition(
        source_episode_id="episode:conflict",
        evidence=evidence,
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(llm),
    )

    assert llm.calls == 2
    assert result.disposition == "apply"
    assert result.state_update is not None


@pytest.mark.asyncio
async def test_oversized_semantic_value_requests_replacement() -> None:
    """An oversized semantic label is replaced instead of truncated."""

    oversized = json.loads(
        _valid_apply_output(
            evidence_handle="evidence:oversized",
            axis="harm",
            delta=40,
            actor_handle="unspecified_other",
            target_handle="self",
        )
    )
    oversized["propositions"][0]["semantic_value"] = "x" * 501
    valid = _valid_apply_output(
        evidence_handle="evidence:oversized",
        axis="harm",
        delta=40,
        actor_handle="unspecified_other",
        target_handle="self",
    )
    evidence = [_evidence("oversized")]

    assert _validate_decision_payload(oversized, evidence=evidence) is None

    llm = _SequenceLLM([json.dumps(oversized), valid])
    result = await run_character_carryover_cognition(
        source_episode_id="episode:oversized",
        evidence=evidence,
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(llm),
    )

    assert llm.calls == 2
    assert result.disposition == "apply"
    assert result.state_update is not None
    assert result.decision.semantic_appraisal is not None
    assert result.decision.semantic_appraisal["propositions"][0][
        "semantic_value"
    ] == "persistent operational consequence"


@pytest.mark.asyncio
async def test_zero_delta_requests_replacement() -> None:
    """A zero-effective proposal is replaced before a native update is accepted."""

    llm = _SequenceLLM([
        _valid_apply_output(
            evidence_handle="evidence:zero",
            axis="harm",
            delta=0,
            actor_handle="unspecified_other",
            target_handle="self",
        ),
        _valid_apply_output(
            evidence_handle="evidence:zero",
            axis="harm",
            delta=40,
            actor_handle="unspecified_other",
            target_handle="self",
        ),
    ])
    result = await run_character_carryover_cognition(
        source_episode_id="episode:zero",
        evidence=[_evidence("zero")],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(llm),
    )

    assert llm.calls == 2
    assert result.disposition == "apply"
    assert result.state_update is not None


@pytest.mark.asyncio
async def test_state_rejected_exhaustion_has_no_state_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated native rejection exhausts the owner cap without state output."""

    monkeypatch.setattr(
        character_carryover,
        "_reduce_apply_decision",
        lambda **kwargs: None,
    )
    output = _valid_apply_output(
        evidence_handle="evidence:rejected",
        axis="harm",
        delta=40,
        actor_handle="unspecified_other",
        target_handle="self",
    )
    llm = _SequenceLLM([output, output, output])

    result = await run_character_carryover_cognition(
        source_episode_id="episode:rejected",
        evidence=[_evidence("rejected")],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(llm),
    )

    assert llm.calls == 3
    assert result.disposition == "degraded"
    assert result.error_code == "state_rejected"
    assert result.attempts == 3
    assert result.state_update is None


@pytest.mark.asyncio
async def test_offset_utc_timestamps_normalize_before_native_reduction() -> None:
    """Storage offset timestamps commit as native UTC-Z without rejection."""

    evidence = [
        _offset_evidence(
            "current_turn_user_message",
            text="I am humiliating you publicly on purpose.",
        ),
        _offset_evidence(
            "assistant_final_dialog",
            text="That is a deliberate boundary violation.",
        ),
    ]
    output = json.dumps({
        "action": "apply",
        "reason_code": "lingering_character_effect",
        "question_id": "character_carryover",
        "propositions": [{
            "kind": "event",
            "semantic_value": "deliberate public boundary violation",
            "evidence_handles": [
                "evidence:current_turn_user_message",
                "evidence:assistant_final_dialog",
            ],
            "role_assignments": [
                {"role": "actor", "entity_handle": "unspecified_other"},
                {"role": "experiencer", "entity_handle": "self"},
            ],
            "deltas": {
                "harm": 30,
                "intentionality": 30,
                "norm_violation": 25,
            },
        }],
    })
    result = await run_character_carryover_cognition(
        source_episode_id="episode-offset",
        evidence=evidence,
        base_state=build_character_production_state(
            updated_at="2026-08-12T23:28:56.351488Z",
        ),
        effective_at=OFFSET_EFFECTIVE_AT,
        services=_services(_SequenceLLM([output])),
    )

    assert result.disposition == "apply"
    assert result.state_update is not None
    replacement = result.state_update["replacement_state"]
    assert replacement["updated_at"].endswith("Z")
    validate_cognition_state(replacement)
    assert result.state_update["changed_paths"]
    event = replacement["active_events"][0]
    assert event["evidence_refs"][0]["occurred_at"].endswith("Z")


@pytest.mark.asyncio
async def test_no_change_decision_stays_no_change_without_state_update() -> None:
    """A valid no-change decision still returns no state update."""

    output = json.dumps({
        "action": "no_change",
        "reason_code": "no_lingering_effect",
        "question_id": "character_carryover",
        "propositions": [],
    })
    result = await run_character_carryover_cognition(
        source_episode_id="episode-no-change",
        evidence=[_offset_evidence("episode-no-change", text="ordinary chat")],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=OFFSET_EFFECTIVE_AT,
        services=_services(_SequenceLLM([output])),
    )

    assert result.disposition == "no_change"
    assert result.state_update is None
    assert result.decision.action == "no_change"
    assert result.error_code is None
