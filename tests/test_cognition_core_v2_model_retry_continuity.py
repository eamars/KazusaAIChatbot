"""V2 retry exhaustion and degraded-continuity contracts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_msg_decontextualizer as decontextualizer_module,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_l3_surface as l3_surface_module,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "cognition_core_v2_retry_exhaustion_cases.json"
)


def _fixture() -> dict[str, object]:
    """Load the sanitized V2 retry contract fixture."""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _surface_input() -> dict[str, object]:
    """Build one canonical input for deterministic degraded projection."""

    return {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            content="Choose a bounded response for this turn.",
        ),
        "intention": {
            "route": "speech",
            "intention": "state the selected response",
            "target_roles": [],
            "reason": "the turn requires a direct response",
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "neutral",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "runtime_capability_limits": ["No external action is available."],
        "interaction_style_context": "brief conversational speech",
        "character_expression_context": {
            "tempo": "steady",
            "linguistic_texture": "Concise spoken clauses.",
        },
        "visual_character_context": "A neutral visual frame.",
    }


def _surface_output() -> dict[str, object]:
    """Build one validated surface for dialog candidate tests."""

    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": "State the selected response.",
        "content_requirements": ["Preserve current-turn meaning."],
        "visible_boundaries": [],
        "addressee_plan": [],
        "delivery_profile": {
            "lexical_register": "plain",
            "sentence_shape": "concise",
            "rhythm": "steady",
            "hesitation": "light",
            "punctuation": "restrained",
        },
        "selected_surface_intent": "state the selected response",
        "permitted_action_results": [],
        "runtime_capability_limits": ["No external action is available."],
    }


def _dialog_state() -> dict[str, object]:
    """Build a direct dialog renderer state with retained V2 truth."""

    episode = canonical_episode(
        content="Tell me which option you choose.",
    )
    surface_input = _surface_input()
    surface_input["episode"] = episode
    return {
        "internal_monologue": "I can choose and answer.",
        "text_surface_input_v2": surface_input,
        "text_surface_output_v2": _surface_output(),
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "platform-user",
        "platform_bot_id": "platform-bot",
        "global_user_id": "global-user",
        "user_name": "Current User",
        "user_profile": {},
        "character_profile": {},
        "cognitive_episode": episode,
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "dialog_usage_mode": "unit_test",
        "llm_trace_id": "retry-continuity-test",
    }


def _compliance_result(*, aligned: bool) -> dict[str, object]:
    """Build one exact focused-verifier aggregate for dialog ledger tests."""

    return {
        "semantic_fidelity": {
            "status": "aligned",
            "issues": [],
        },
        "role_direction": {
            "status": "aligned" if aligned else "misaligned",
            "violations": [] if aligned else [{
                "kind": "selection_owner_transfer",
                "evidence": "I will follow your choice.",
                "explanation": "The required selection was transferred.",
            }],
        },
        "surface_integrity": {
            "status": "aligned",
            "issues": [],
        },
    }


def _patch_dialog_sequence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidates: list[object],
    compliance_results: list[dict[str, object]],
) -> tuple[MagicMock, AsyncMock]:
    """Install one deterministic generator and compliance sequence."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(side_effect=[
        SimpleNamespace(content=json.dumps(candidate))
        for candidate in candidates
    ])
    compliance = AsyncMock(side_effect=compliance_results)
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(dialog_module, "_verify_dialog_compliance", compliance)
    monkeypatch.setattr(
        dialog_module,
        "repair_text_surface_for_dialog",
        AsyncMock(return_value=_surface_output()),
    )
    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_llm_stage_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_model_contract_event",
        AsyncMock(),
    )
    return generator_llm, compliance


def test_v2_attempt_policy_matches_exact_owner_matrix() -> None:
    """Every scoped model call has one bounded owner and terminal disposition."""

    policy = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy"
    )
    fixture_owners = _fixture()["owners"]

    assert policy.V2_MODEL_TOTAL_ATTEMPTS == 3
    assert policy.V2_VERIFIER_TOTAL_ATTEMPTS == 3
    assert policy.V2_MODEL_OWNER_POLICIES == fixture_owners


def test_v2_attempt_record_validation_is_bounded_and_data_only() -> None:
    """Attempt telemetry accepts only the approved bounded metadata fields."""

    policy = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy"
    )
    record = {
        "stage": "dialog_generator",
        "failure_kind": "semantic",
        "attempt_count": 3,
        "total_attempt_limit": 3,
        "selected_attempt": 3,
        "disposition": "accepted_degraded",
        "safe_checkpoint": "post_cognition_commit",
    }

    assert policy.validate_v2_attempt_record(record) == record

    for invalid_record in (
        {**record, "candidate_text": "private output"},
        {**record, "attempt_count": 4},
        {**record, "selected_attempt": 0},
        {**record, "disposition": "fatal"},
    ):
        with pytest.raises(ValueError):
            policy.validate_v2_attempt_record(invalid_record)


def test_appraisal_uses_two_attempts_while_other_short_owners_use_three() -> None:
    """Appraisal keeps its local two-attempt contract beside other owners."""

    semantic_appraisal = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal"
    )
    goal_cognition = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.goal_cognition"
    )
    workspace = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.workspace"
    )
    action_selection = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.action_selection"
    )
    action_authorization = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.action_authorization"
    )
    surface_stages = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface_stages"
    )

    other_limits = {
        goal_cognition.GOAL_COGNITION_ATTEMPT_LIMIT,
        workspace.WORKSPACE_COLLAPSE_ATTEMPT_LIMIT,
        action_selection.ACTION_PLANNING_ATTEMPT_LIMIT,
        action_authorization.ACTION_AUTHORIZATION_ATTEMPT_LIMIT,
        surface_stages.SURFACE_STAGE_ATTEMPT_LIMIT,
        decontextualizer_module.IMAGE_DESCRIPTOR_ATTEMPT_LIMIT,
        decontextualizer_module.MSG_DECONTEXTUALIZER_ATTEMPT_LIMIT,
        dialog_module.DIALOG_VERIFIER_ATTEMPT_LIMIT,
        dialog_module.DIALOG_GENERATOR_TOTAL_ATTEMPTS,
    }

    assert semantic_appraisal.SEMANTIC_APPRAISAL_ATTEMPT_LIMIT == 2
    assert other_limits == {3}


def test_degraded_text_surface_projects_only_validated_v2_truth() -> None:
    """Surface exhaustion still produces one validated neutral text surface."""

    surface = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface"
    )

    output = surface.build_degraded_text_surface(_surface_input())

    assert output["content_plan"] == "state the selected response"
    assert output["selected_surface_intent"] == "state the selected response"
    assert output["content_requirements"]
    assert output["visible_boundaries"] == []
    assert output["addressee_plan"] == []
    assert output["permitted_action_results"] == []
    assert output["runtime_capability_limits"] == [
        "No external action is available.",
    ]
    assert set(output["delivery_profile"]) == {
        "lexical_register",
        "sentence_shape",
        "rhythm",
        "hesitation",
        "punctuation",
    }


def test_role_direction_verdict_requires_typed_violation_kinds() -> None:
    """Role rejection is limited to the two approved typed conditions."""

    fixture_cases = _fixture()["role_direction_cases"]
    violation_kinds = {
        case["expected_violation_kind"]
        for case in fixture_cases
        if "expected_violation_kind" in case
    }
    assert violation_kinds == {
        "selection_owner_transfer",
        "typed_operation_role_reversal",
    }

    valid = {
        "aligned": False,
        "violations": [{
            "kind": "selection_owner_transfer",
            "evidence": "I will follow your choice.",
            "explanation": "The candidate transfers the required selection.",
        }],
    }
    assert dialog_module._validate_role_direction_verdict(
        valid,
        generated_dialog=["I will follow your choice."],
    ) == valid

    invalid = {
        "aligned": False,
        "violations": [{
            **valid["violations"][0],
            "kind": "information_request",
        }],
    }
    with pytest.raises(ValueError):
        dialog_module._validate_role_direction_verdict(
            invalid,
            generated_dialog=["I will follow your choice."],
        )

    with pytest.raises(ValueError, match="evidence"):
        dialog_module._validate_role_direction_verdict(
            valid,
            generated_dialog=["I already made the required selection."],
        )


@pytest.mark.asyncio
async def test_focused_verifier_exhaustion_returns_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed verifier output cannot erase a valid dialog candidate."""

    verifier_llm = MagicMock()
    verifier_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"aligned": true, "Issues": []}',
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        verifier_llm,
    )
    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_model_contract_event",
        AsyncMock(),
    )

    verdict = await dialog_module._verify_dialog_semantic_fidelity(
        surface_output=_surface_output(),
        generated_dialog=["I choose the first option."],
        current_visible_percepts=[],
        llm_trace_id="verifier-unavailable-test",
    )

    assert verdict == {"status": "unavailable", "issues": []}
    assert verifier_llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_focused_verifier_provider_exhaustion_returns_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three classified provider failures cannot erase a dialog candidate."""

    verifier_llm = MagicMock()
    verifier_llm.ainvoke = AsyncMock(
        side_effect=ConnectionError("provider unavailable"),
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        verifier_llm,
    )
    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_model_contract_event",
        AsyncMock(),
    )

    verdict = await dialog_module._verify_dialog_semantic_fidelity(
        surface_output=_surface_output(),
        generated_dialog=["I choose the first option."],
        current_visible_percepts=[],
        llm_trace_id="verifier-provider-unavailable-test",
    )

    assert verdict == {"status": "unavailable", "issues": []}
    assert verifier_llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_unexpected_verifier_exception_remains_unrecoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An internal verifier invariant still reaches the fatal boundary."""

    monkeypatch.setattr(
        dialog_module,
        "_verify_dialog_semantic_fidelity",
        AsyncMock(side_effect=AssertionError("verifier invariant")),
    )
    monkeypatch.setattr(
        dialog_module,
        "_verify_dialog_role_direction",
        AsyncMock(return_value={"aligned": True, "violations": []}),
    )
    monkeypatch.setattr(
        dialog_module,
        "_verify_dialog_surface_integrity",
        AsyncMock(return_value={"aligned": True, "issues": []}),
    )

    with pytest.raises(AssertionError, match="verifier invariant"):
        await dialog_module._verify_dialog_compliance(
            surface_output=_surface_output(),
            generated_dialog=["I choose the first option."],
            current_visible_percepts=[],
            llm_trace_id="verifier-invariant-test",
        )


@pytest.mark.asyncio
async def test_dialog_third_candidate_is_terminal_degraded_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two semantic rejections retain text and render candidate three."""

    generator_llm, compliance = _patch_dialog_sequence(
        monkeypatch,
        candidates=[
            {
            "final_dialog": ["candidate one"],
            },
            {
            "final_dialog": ["candidate two"],
            },
            {
            "final_dialog": ["candidate three"],
            },
        ],
        compliance_results=[
            _compliance_result(aligned=False),
            _compliance_result(aligned=False),
        ],
    )

    result = await dialog_module.dialog_generator(_dialog_state())

    assert result["final_dialog"] == ["candidate three"]
    assert result["text_surface_output_v2"] == _surface_output()
    assert generator_llm.ainvoke.await_count == 3
    assert compliance.await_count == 2


@pytest.mark.asyncio
async def test_empty_terminal_candidate_falls_back_to_candidate_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable terminal render preserves the newest earlier candidate."""

    generator_llm, compliance = _patch_dialog_sequence(
        monkeypatch,
        candidates=[
            {"final_dialog": ["candidate one"]},
            {"final_dialog": ["candidate two"]},
            {"final_dialog": []},
        ],
        compliance_results=[
            _compliance_result(aligned=False),
            _compliance_result(aligned=False),
        ],
    )

    result = await dialog_module.dialog_generator(_dialog_state())

    assert result["final_dialog"] == ["candidate two"]
    assert generator_llm.ainvoke.await_count == 3
    assert compliance.await_count == 2


@pytest.mark.asyncio
async def test_unusable_second_and_third_candidates_fall_back_to_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid first candidate remains deliverable across later shape failures."""

    generator_llm, compliance = _patch_dialog_sequence(
        monkeypatch,
        candidates=[
            {"final_dialog": ["candidate one"]},
            {"final_dialog": []},
            {"final_dialog": []},
        ],
        compliance_results=[
            _compliance_result(aligned=False),
        ],
    )

    result = await dialog_module.dialog_generator(_dialog_state())

    assert result["final_dialog"] == ["candidate one"]
    assert generator_llm.ainvoke.await_count == 3
    assert compliance.await_count == 1


@pytest.mark.asyncio
async def test_zero_usable_dialog_candidates_remains_unrecoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Total generator exhaustion has no visible fallback to deliver."""

    generator_llm, compliance = _patch_dialog_sequence(
        monkeypatch,
        candidates=[{}, {}, {}],
        compliance_results=[],
    )

    with pytest.raises(dialog_module.StateContractError):
        await dialog_module.dialog_generator(_dialog_state())

    assert generator_llm.ainvoke.await_count == 3
    compliance.assert_not_awaited()


@pytest.mark.asyncio
async def test_optional_visual_exhaustion_preserves_text_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The L3 connector omits failed visual output after text succeeds."""

    input_payload = _surface_input()
    text_output = _surface_output()
    monkeypatch.setattr(
        l3_surface_module,
        "_load_interaction_style_context",
        AsyncMock(return_value="brief conversational speech"),
    )
    monkeypatch.setattr(
        l3_surface_module,
        "build_text_surface_input_from_global_state",
        MagicMock(return_value=input_payload),
    )
    monkeypatch.setattr(
        l3_surface_module,
        "run_text_surface_planning",
        AsyncMock(return_value=text_output),
    )
    monkeypatch.setattr(
        l3_surface_module,
        "run_visual_surface_planning",
        AsyncMock(side_effect=CognitionExecutionError(
            "visual surface contract exhausted",
            error_code="surface_visual_contract_exhausted",
            stage="surface.visual",
            attempt_count=3,
            safe_checkpoint="post_cognition_commit",
            retryable=False,
        )),
    )

    result = await l3_surface_module.call_l3_text_surface_handler({})

    assert result == {
        "text_surface_input_v2": input_payload,
        "text_surface_output_v2": text_output,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("recover_on_third", [True, False])
async def test_text_surface_retry_or_validated_degraded_projection(
    recover_on_third: bool,
) -> None:
    """Text planning recovers on attempt three or projects canonical truth."""

    surface = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface"
    )

    class _SurfaceLLM:
        def __init__(self) -> None:
            self.content_calls = 0
            self.preference_calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages
            stage_name = getattr(config, "stage_name", "")
            if stage_name == "content":
                self.content_calls += 1
                if recover_on_third and self.content_calls == 3:
                    return SimpleNamespace(content=json.dumps({
                        "content_plan": "recovered content plan",
                        "content_requirements": [
                            "Preserve current-turn meaning.",
                        ],
                        "delivery_profile": {
                            "lexical_register": "plain",
                            "sentence_shape": "concise",
                            "rhythm": "steady",
                            "hesitation": "light",
                            "punctuation": "restrained",
                        },
                    }))
                return SimpleNamespace(content='{"invalid": true}')
            if stage_name == "preference":
                self.preference_calls += 1
                return SimpleNamespace(content=json.dumps({
                    "visible_boundaries": [],
                    "addressee_plan": [],
                }))
            raise AssertionError("unexpected surface stage")

    llm = _SurfaceLLM()
    services = SimpleNamespace(
        llm=llm,
        content_plan_config=SimpleNamespace(stage_name="content"),
        preference_config=SimpleNamespace(stage_name="preference"),
    )

    output = await surface.run_text_surface_planning(
        _surface_input(),
        services,
    )

    expected_plan = (
        "recovered content plan"
        if recover_on_third
        else "state the selected response"
    )
    assert output["content_plan"] == expected_plan
    assert output["selected_surface_intent"] == "state the selected response"
    assert llm.content_calls == 3
    assert llm.preference_calls == 1


@pytest.mark.asyncio
async def test_visual_surface_owner_exhausts_after_three_attempts() -> None:
    """Optional visual failure is typed for containment by the L3 connector."""

    surface = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface"
    )

    class _VisualLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            self.calls += 1
            return SimpleNamespace(content='{"invalid": true}')

    llm = _VisualLLM()
    services = SimpleNamespace(
        llm=llm,
        visual_config=SimpleNamespace(stage_name="visual"),
    )

    with pytest.raises(CognitionExecutionError) as error_info:
        await surface.run_visual_surface_planning(
            _surface_input(),
            services,
        )

    assert error_info.value.stage == "surface.visual"
    assert error_info.value.attempt_count == 3
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_surface_non_string_content_keeps_contract_retry_behavior() -> None:
    """Tracing must not make a non-string provider candidate acceptable."""

    surface = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface"
    )

    class _VisualLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content={"visual_directives": "wrong transport type"},
                )
            return SimpleNamespace(
                content='{"visual_directives": "validated retry"}',
            )

    llm = _VisualLLM()
    services = SimpleNamespace(
        llm=llm,
        visual_config=SimpleNamespace(stage_name="visual"),
    )

    output = await surface.run_visual_surface_planning(
        _surface_input(),
        services,
    )

    assert output["visual_directives"] == "validated retry"
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_unexpected_text_surface_failure_remains_unrecoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An internal surface invariant is never converted to degraded output."""

    surface = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface"
    )
    monkeypatch.setattr(
        surface,
        "run_content_plan_stage",
        AsyncMock(side_effect=AssertionError("surface invariant")),
    )
    monkeypatch.setattr(
        surface,
        "run_preference_stage",
        AsyncMock(return_value=([], [])),
    )

    with pytest.raises(AssertionError, match="surface invariant"):
        await surface.run_text_surface_planning(
            _surface_input(),
            SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_dialog_surface_repair_exhaustion_retains_valid_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dialog repair continues from the latest validated semantic surface."""

    generator_llm, compliance = _patch_dialog_sequence(
        monkeypatch,
        candidates=[
            {"final_dialog": ["candidate one"]},
            {"final_dialog": ["candidate two"]},
        ],
        compliance_results=[
            _compliance_result(aligned=False),
            _compliance_result(aligned=True),
        ],
    )
    surface_repair = AsyncMock(side_effect=CognitionExecutionError(
        "dialog surface repair contract exhausted",
        error_code="surface_dialog_compliance_repair_contract_exhausted",
        stage="surface.dialog_compliance_repair",
        attempt_count=3,
        safe_checkpoint="post_cognition_commit",
        retryable=False,
    ))
    monkeypatch.setattr(
        dialog_module,
        "repair_text_surface_for_dialog",
        surface_repair,
    )

    result = await dialog_module.dialog_generator(_dialog_state())

    assert result == {
        "final_dialog": ["candidate two"],
        "text_surface_output_v2": _surface_output(),
    }
    assert generator_llm.ainvoke.await_count == 2
    assert compliance.await_count == 2
    surface_repair.assert_awaited_once()
