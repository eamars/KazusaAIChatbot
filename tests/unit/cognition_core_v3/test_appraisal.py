"""Deterministic tests for V3 cache-affine appraisal semantics."""

from __future__ import annotations

import asyncio

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import appraisal
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    BOUNDARY_REJECTED_ERROR_CODE,
    CANDIDATE_ORIGIN_MISSING,
    EXHAUSTION_FAILURE_CLASS,
    StageFailure,
    StageResult,
    hash_static_prompt,
    validate_stage_result,
)
from kazusa_ai_chatbot.cognition_core_v3.execution import (
    AttemptLedger,
    ChainOutcome,
    ChainTaskSpec,
    StageAttemptOutcome,
    start_wave,
)

EXPECTED_FAMILY_PROPOSITION_KINDS = {
    "event_agency": ("responsibility", "intentionality"),
    "relationship_social": ("social_meaning", "relationship_threat"),
    "moral_identity": ("norm_meaning",),
    "goal_threat_outcome": (
        "goal_release",
        "goal_supersession",
        "goal_completed",
        "event_completed",
        "threat_resolved",
        "event_repaired",
        "knowledge_answered",
        "outcome_pending",
    ),
    "epistemic_comparison_memory": ("comparison_meaning", "memory_cue"),
    "existential_drive": ("meaning_relevance",),
}

SIX_FAMILIES = tuple(sorted(EXPECTED_FAMILY_PROPOSITION_KINDS))


def _accepted_stage_result(chain_name: str, stage_name: str, summary: str) -> StageResult:
    return validate_stage_result(
        StageResult(
            chain_name=chain_name,
            stage_name=stage_name,
            accepted=True,
            local_state={"selected_evidence_handles": ["ev_1"], "propositions": [], "deltas": []},
            semantic_summary=summary,
        )
    )


def test_v3_appraisal_preserves_six_family_domains():
    assert set(SIX_FAMILIES) == set(appraisal.FAMILY_PROPOSITION_KINDS)

    for family in SIX_FAMILIES:
        assert appraisal.family_proposition_kinds(family) == EXPECTED_FAMILY_PROPOSITION_KINDS[family]
        axes = appraisal.FAMILY_DELTA_AXES[family]
        assert isinstance(axes, tuple) and len(axes) >= 1
        assert len(set(axes)) == len(axes)

    assert appraisal.FAMILY_DELTA_AXES["event_agency"] == ("responsibility", "intentionality")
    assert appraisal.FAMILY_IDENTITY_CATEGORY_SETS["event_agency"] == frozenset({"personality", "boundaries"})
    assert appraisal.FAMILY_IDENTITY_OPTIONAL_CATEGORY_SETS["epistemic_comparison_memory"] == (frozenset(), frozenset({"core"}))

    tail = appraisal.build_family_question_tail(
        "event_agency",
        {"personality": "p-value", "boundaries": "b-value"},
        ["ev_1", "ev_2"],
    )
    assert "判断责任和意图" in tail
    assert "- responsibility：事件主体对结果负有责任" in tail
    assert "boundaries=b-value" in tail

    appraisal.validate_family_projection("event_agency", {"personality": "p"})
    appraisal.validate_family_projection("epistemic_comparison_memory", {"core": "c"})
    appraisal.validate_family_projection("epistemic_comparison_memory", {})

    with pytest.raises(ValueError, match="unknown semantic question kind"):
        appraisal.family_proposition_kinds("invented_family")

    with pytest.raises(ValueError, match="not owned by family 'event_agency'"):
        appraisal.validate_family_projection("event_agency", {"self_image": "x"})

    with pytest.raises(ValueError, match="invalid for family 'epistemic_comparison_memory'"):
        appraisal.validate_family_projection(
            "epistemic_comparison_memory", {"personality": "x"}
        )


def test_all_appraisal_chains_use_one_byte_identical_static_system_prompt():
    prompt_hash = hash_static_prompt(appraisal.STATIC_APPRAISAL_SYSTEM_PROMPT)

    for family in SIX_FAMILIES:
        assert hash_static_prompt(appraisal.STATIC_APPRAISAL_SYSTEM_PROMPT) == prompt_hash

        dynamic_description = appraisal.FAMILY_QUESTION_DESCRIPTIONS[family]
        assert dynamic_description not in appraisal.STATIC_APPRAISAL_SYSTEM_PROMPT

        tail = appraisal.build_family_question_tail(family, {}, ["ev_1"])
        assert dynamic_description in tail

    for tail_only_heading in (
        "# 问题类型",
        "# 语义问题",
        "# 允许的命题类型",
        "# 授权证据 handle",
        "# 状态投影",
        "# 已接受前驱摘要",
    ):
        assert tail_only_heading not in appraisal.STATIC_APPRAISAL_SYSTEM_PROMPT

    assert "Kazusa" not in appraisal.STATIC_APPRAISAL_SYSTEM_PROMPT


def test_causal_and_epistemic_chains_expose_only_accepted_predecessor_context():
    event_result = _accepted_stage_result("causal_normative", "event_agency", "causal-event-accepted-summary")

    moral_tail = appraisal.build_family_question_tail(
        "moral_identity",
        {"personality": "p-value"},
        ["ev_1"],
        accepted_prefix_summaries=appraisal.render_accepted_context([event_result]),
    )
    assert "causal-event-accepted-summary" in moral_tail

    exhausted_predecessor = StageResult(
        chain_name="epistemic_meaning",
        stage_name="epistemic_comparison_memory",
        accepted=False,
        local_state=None,
        semantic_summary=None,
        failure=StageFailure(
            chain_name="epistemic_meaning",
            stage_name="epistemic_comparison_memory",
            failure_class=EXHAUSTION_FAILURE_CLASS,
            error_code=APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
            repair_attempted=True,
        ),
    )
    rendered = appraisal.render_accepted_context([event_result, exhausted_predecessor])
    assert rendered == ("causal-event-accepted-summary",)

    epistemic_result = _accepted_stage_result(
        "epistemic_meaning", "epistemic_comparison_memory", "epistemic-accepted-summary"
    )
    existential_tail = appraisal.build_family_question_tail(
        "existential_drive",
        {},
        ["ev_2"],
        accepted_prefix_summaries=appraisal.render_accepted_context([epistemic_result]),
    )
    assert "epistemic-accepted-summary" in existential_tail
    assert "causal-event-accepted-summary" not in existential_tail


def test_terminal_outcome_runs_from_provisional_accepted_state():
    async def wave_a_scenario() -> ChainOutcome:
        ledger = AttemptLedger({"event_agency": 2, "moral_identity": 2})

        async def accepting(ctx):
            return StageAttemptOutcome(
                True,
                {"selected_evidence_handles": ["ev_1"], "propositions": [], "deltas": []},
                f"{ctx.stage_name}-accepted-summary",
                None,
            )

        handle = start_wave(
            [
                ChainTaskSpec(
                    "causal_normative",
                    ("event_agency", "moral_identity"),
                    {"event_agency": accepting, "moral_identity": accepting},
                )
            ],
            ledger=ledger,
        )
        result = await handle.complete()
        return result.outcomes["causal_normative"]

    causal_outcome = asyncio.run(wave_a_scenario())
    provisional_state = appraisal.reduce_appraisal_results([causal_outcome])

    assert set(provisional_state.local_state) == {"event_agency", "moral_identity"}
    assert not provisional_state.omitted_families

    terminal_request = appraisal.build_terminal_outcome_request(
        provisional_state,
        ["ev_1", "ev_2"],
    )
    assert "goal_threat_outcome" in terminal_request
    assert "event_agency=" in terminal_request
    assert "moral_identity=" in terminal_request
    assert "ev_1" in terminal_request and "ev_2" in terminal_request

    async def exhausted_terminal_scenario() -> ChainOutcome:
        ledger = AttemptLedger({"goal_threat_outcome": 2})

        async def always_structurally_invalid(ctx):
            return StageAttemptOutcome(False, None, None, "structural_contract")

        handle = start_wave(
            [
                ChainTaskSpec(
                    "terminal_outcome",
                    ("goal_threat_outcome",),
                    {"goal_threat_outcome": always_structurally_invalid},
                )
            ],
            ledger=ledger,
        )
        result = await handle.complete()
        return result.outcomes["terminal_outcome"]

    terminal_outcome = asyncio.run(exhausted_terminal_scenario())
    combined_state = appraisal.reduce_appraisal_results(
        [causal_outcome, terminal_outcome]
    )

    assert combined_state.omitted_families == {
        "terminal_outcome": APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
    }
    assert set(combined_state.local_state) == {"event_agency", "moral_identity"}


def test_optional_stage_exhaustion_preserves_accepted_prefix():
    async def scenario() -> ChainOutcome:
        ledger = AttemptLedger({"event_agency": 2, "moral_identity": 2})

        async def always_structurally_invalid(ctx):
            return StageAttemptOutcome(False, None, None, "structural_contract")

        async def accepting_from_root(ctx):
            assert ctx.accepted_prefix == ()
            return StageAttemptOutcome(
                True,
                {"selected_evidence_handles": ["ev_1"], "propositions": [], "deltas": []},
                "moral-accepted-from-root-summary",
                None,
            )

        handle = start_wave(
            [
                ChainTaskSpec(
                    "causal_normative",
                    ("event_agency", "moral_identity"),
                    {"event_agency": always_structurally_invalid, "moral_identity": accepting_from_root},
                )
            ],
            ledger=ledger,
        )
        result = await handle.complete()
        return result.outcomes["causal_normative"]

    outcome = asyncio.run(scenario())
    exhausted_stage, continued_stage = outcome.results

    assert not exhausted_stage.accepted
    assert exhausted_stage.failure is not None
    assert exhausted_stage.failure.failure_class == EXHAUSTION_FAILURE_CLASS
    assert exhausted_stage.failure.error_code == APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
    assert exhausted_stage.failure.repair_attempted is True

    assert continued_stage.accepted
    assert continued_stage.stage_name == "moral_identity"

    reduced_state = appraisal.reduce_appraisal_results([outcome])
    assert set(reduced_state.local_state) == {"moral_identity"}
    assert "causal_normative" not in reduced_state.omitted_families

    boundary_candidate = {
        "selected_evidence_handles": ["ev_1"],
        "propositions": [
            {"kind": "responsibility", "statement": "s", "origin_evidence_handle": None}
        ],
        "deltas": [],
        "explanation": "x",
    }
    origin_missing = appraisal.classify_appaisal_candidate(
        "event_agency", boundary_candidate, ["ev_1"]
    )
    assert not origin_missing.accepted
    assert origin_missing.failure_class == CANDIDATE_ORIGIN_MISSING

    async def terminal_boundary_scenario() -> tuple[ChainOutcome, AttemptLedger]:
        ledger = AttemptLedger({"event_agency": 2, "moral_identity": 2})

        async def boundary_producer(ctx):
            return StageAttemptOutcome(
                False, None, None, CANDIDATE_ORIGIN_MISSING
            )

        handle = start_wave(
            [
                ChainTaskSpec(
                    "causal_normative",
                    ("event_agency", "moral_identity"),
                    {"event_agency": boundary_producer, "moral_identity": boundary_producer},
                )
            ],
            ledger=ledger,
        )
        result = await handle.complete()
        return result.outcomes["causal_normative"], ledger

    boundary_outcome, ledger = asyncio.run(terminal_boundary_scenario())
    assert ledger.attempts_used("event_agency") == 1
    event_stage, moral_stage = boundary_outcome.results
    assert not event_stage.accepted
    assert event_stage.failure is not None
    assert event_stage.failure.failure_class == CANDIDATE_ORIGIN_MISSING
    assert event_stage.failure.error_code == BOUNDARY_REJECTED_ERROR_CODE
    assert event_stage.failure.repair_attempted is False

    # The independently valid next stage still runs after a terminal boundary rejection.
    assert ledger.attempts_used("moral_identity") == 1
    assert not moral_stage.accepted
