"""Live LLM evidence for terminal visual ownership and literal dialog."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
import pytest

from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_text_surface_planning,
    run_visual_surface_planning,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_module
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_CHARACTER_PATH = Path(
    "test_artifacts/cognition_core_v2/real_conversation_replay/"
    "production_character_state.json"
)
_TRACE_SUITE = "dialog_visible_speech_and_semantic_fidelity"
_ROLE_ONLY_DIAGNOSTIC_PROMPT = '''只核对 current_visible_percepts 与
candidate_final_dialog 之间的行动者、动作、目标、受益者和主语方向。按各自的角色框架解析每段文本。
percept 行提供 speaker_role、first_person_role、addressee_role 和
implicit_imperative_subject_role。candidate dialog 由当前角色说出：第一人称是当前角色，第二人称是当前用户。
解析后比较行动者/动作/目标；任何反转都应显著降低 score。忽略风格、新颖性、亲密度、安全性和写作质量。

Return exactly one JSON object with exactly score and issues. score is a
finite number from 0 to 1. issues is a list of concise role-direction
failures.'''


def _current_user_addressee_plan(
    display_name: str = "当前用户",
) -> list[dict[str, str]]:
    """Build the canonical direct-current-user wording row."""

    return [{
        "handle": "current_user",
        "display_name": display_name,
        "semantic_role": "direct_recipient",
        "wording_policy": "second_person_allowed",
    }]


class _CapturingLLM:
    """Delegate to one real route while retaining raw request and response."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object | None = None,
        **kwargs: object,
    ) -> Any:
        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        self.calls.append({
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(response.content),
        })
        return response


async def _skip_if_model_routes_unavailable() -> None:
    """Skip when the configured dialog model endpoint is unavailable."""

    base_url = dialog_module.DIALOG_GENERATOR_LLM_BASE_URL
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {base_url}: {exc}")
    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned {response.status_code}: {base_url}",
        )


def _character_profile() -> dict[str, Any]:
    """Load the frozen production Kazusa character fields used by replay."""

    payload = json.loads(_CHARACTER_PATH.read_text(encoding="utf-8"))
    profile = payload["character_state"]
    if not isinstance(profile, dict):
        raise TypeError("frozen character profile must be an object")
    return profile


def _surface_contexts(
    profile: dict[str, Any],
) -> tuple[dict[str, str], str]:
    """Build the split production surface contexts from the frozen profile."""

    return l3_module._character_surface_contexts({
        "character_profile": profile,
    })


def _delivery_profile(lexical_register: str) -> dict[str, str]:
    """Build one exact delivery-only profile for a live fixture."""

    return {
        "lexical_register": lexical_register,
        "sentence_shape": "紧凑自然的口语短句",
        "rhythm": "平稳，允许少量自然停顿",
        "hesitation": "让轻微犹豫随当前情绪自然出现",
        "punctuation": "克制自然",
    }


def _role_surface_output() -> dict[str, Any]:
    """Build semantic surface context for focused role-direction live cases."""

    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": (
            "The character decides the next action and tells the current user "
            "what to do."
        ),
        "content_requirements": [
            "Keep the character as response and selection owner.",
        ],
        "visible_boundaries": [],
        "addressee_plan": _current_user_addressee_plan(),
        "delivery_profile": _delivery_profile("concise spoken wording"),
        "selected_surface_intent": "state the selected next action",
        "permitted_action_results": [],
    }


def _surface_input(
    case: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Build one exact surface input for an individually reviewed live case."""

    expression_context, visual_context = _surface_contexts(profile)
    payload = {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id=f"visible-speech-{case['case_id']}",
            content=case["user_input"],
            metadata=case.get("episode_metadata"),
        ),
        "intention": {
            "route": "speech",
            "intention": case["intention"],
            "target_roles": [],
            "reason": case["reason"],
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": case["emotional_tone"],
            "intensity": "moderate",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": list(
            case.get("permitted_action_results", [])
        ),
        "interaction_style_context": (
            "关系亲近；用自然、有温度的简体中文短句表达，保持当前语义。"
        ),
        "character_expression_context": expression_context,
        "visual_character_context": visual_context,
    }
    episode_metadata = case.get("episode_metadata")
    if isinstance(episode_metadata, dict):
        response_operation = episode_metadata.get("response_operation")
        if (
            isinstance(response_operation, dict)
            and response_operation.get("selection_required") is True
        ):
            selected_operation = case.get(
                "selected_response_operation",
                response_operation,
            )
            if not isinstance(selected_operation, dict):
                raise TypeError("selected response operation must be an object")
            payload["intention"]["selected_response_operation"] = dict(
                selected_operation
            )
            payload["selected_response_operation"] = dict(selected_operation)
    return payload


def _surface_input_for_operation(
    operation: dict[str, Any],
    *,
    case_id: str,
) -> dict[str, Any]:
    """Build a valid direct-verifier surface input for one operation."""

    return _surface_input(
        {
            "case_id": case_id,
            "user_input": "the current turn requires a concrete selection",
            "intention": "state the selected response operation",
            "reason": "the current operation requires character judgment",
            "emotional_tone": "guarded",
            "episode_metadata": {"response_operation": operation},
        },
        _character_profile(),
    )


def _dialog_state(
    *,
    surface_input: dict[str, Any],
    surface_output: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Build the direct production dialog state for one live case."""

    return {
        "internal_monologue": "按当前意图和边界自然回应。",
        "text_surface_input_v2": surface_input,
        "text_surface_output_v2": surface_output,
        "cognitive_episode": surface_input["episode"],
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "live-visible-speech-user",
        "platform_bot_id": "live-visible-speech-bot",
        "global_user_id": "live-visible-speech-user",
        "user_name": "测试用户",
        "user_profile": {},
        "character_profile": profile,
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "dialog_usage_mode": "live_visible_reply",
        "llm_trace_id": "",
    }


async def _run_live_case(
    case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Run real sibling L3 branches and dialog, then write raw evidence."""

    await _skip_if_model_routes_unavailable()
    profile = _character_profile()
    surface_input = _surface_input(case, profile)

    text_services = l3_module._build_text_surface_services()
    text_llm = _CapturingLLM(text_services.llm)
    text_services = replace(text_services, llm=text_llm)
    visual_services = l3_module._build_visual_surface_services()
    visual_llm = _CapturingLLM(visual_services.llm)
    visual_services = replace(visual_services, llm=visual_llm)

    text_output = await run_text_surface_planning(
        surface_input,
        text_services,
    )
    visual_output = await run_visual_surface_planning(
        surface_input,
        visual_services,
    )

    generator_llm = _CapturingLLM(dialog_module._dialog_generator_llm)
    semantic_llm = _CapturingLLM(
        dialog_module._dialog_semantic_fidelity_llm
    )
    surface_integrity_llm = _CapturingLLM(
        dialog_module._dialog_surface_integrity_llm
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_integrity_llm,
    )
    dialog_output = await dialog_module.dialog_generator(_dialog_state(
        surface_input=surface_input,
        surface_output=text_output,
        profile=profile,
    ))

    evidence = {
        "case": case,
        "surface_input": surface_input,
        "text_stage_calls": text_llm.calls,
        "visual_stage_calls": visual_llm.calls,
        "text_surface_output": text_output,
        "terminal_visual_surface_output": visual_output,
        "dialog_generator_calls": generator_llm.calls,
        "dialog_semantic_fidelity_calls": semantic_llm.calls,
        "dialog_surface_integrity_calls": surface_integrity_llm.calls,
        "dialog_compliance_calls": (
            semantic_llm.calls + surface_integrity_llm.calls
        ),
        "dialog_output": dialog_output,
        "human_review_contract": {
            "literal_speech_only": True,
            "visual_has_no_dialog_consumer": True,
            "preserve_response_operation": True,
            "preserve_actors_claims_conditions_and_time_scope": True,
            "allow_non_conflicting_elaboration": True,
        },
    }
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        case["case_id"],
        evidence,
    )

    assert artifact_path.exists()
    assert len(text_llm.calls) == 3
    assert len(visual_llm.calls) == 1
    assert "visual_directives" not in json.dumps(text_output)
    assert "visual_directives" not in json.dumps(
        generator_llm.calls,
        ensure_ascii=False,
    )
    assert len(semantic_llm.calls) == 1
    assert len(surface_integrity_llm.calls) == 1
    assert dialog_output["final_dialog"]
    return evidence


async def _run_live_verifier_case(
    *,
    case_id: str,
    candidate_dialog: str,
    monkeypatch: pytest.MonkeyPatch,
    surface_output: dict[str, Any] | None = None,
    current_visible_percepts: list[dict[str, str]] | None = None,
    human_review_contract: dict[str, bool] | None = None,
    expected_aligned: bool = False,
) -> dict[str, Any]:
    """Run the real owner-preserving verifier route against one candidate."""

    await _skip_if_model_routes_unavailable()
    if surface_output is None:
        surface_output = {
            "schema_version": "text_surface_output.v2",
            "content_plan": "Verbally accept or decline the current request.",
            "content_requirements": [
                "Keep capability execution claims grounded.",
                "Action description is valid visible roleplay.",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(
                "Current User"
            ),
            "delivery_profile": _delivery_profile(
                "Natural concise spoken wording."
            ),
            "selected_surface_intent": "Answer the current request verbally.",
            "permitted_action_results": [],
        }
    if current_visible_percepts is None:
        current_visible_percepts = [{
            "input_source": "dialog_text",
            "content": "请只用语言告诉我你是否同意。",
        }]
    if human_review_contract is None:
        human_review_contract = {
            "allow_action_description_as_visible_roleplay": True,
            "real_compliance_route": True,
        }
    surface_input = None
    for percept in current_visible_percepts:
        content = percept.get("content")
        if not isinstance(content, dict):
            continue
        response_operation = content.get("response_operation")
        if (
            isinstance(response_operation, dict)
            and response_operation.get("selection_required") is True
        ):
            surface_input = _surface_input_for_operation(
                response_operation,
                case_id=case_id,
            )
            break
    semantic_llm = _CapturingLLM(
        dialog_module._dialog_semantic_fidelity_llm
    )
    surface_integrity_llm = _CapturingLLM(
        dialog_module._dialog_surface_integrity_llm
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_integrity_llm,
    )

    verdict = await dialog_module._verify_dialog_compliance(
        surface_output=surface_output,
        generated_dialog=[candidate_dialog],
        current_visible_percepts=current_visible_percepts,
        surface_input=surface_input,
        llm_trace_id=f"live-{case_id}",
    )
    evidence = {
        "case": {
            "case_id": case_id,
            "candidate_final_dialog": [candidate_dialog],
        },
        "text_surface_output": surface_output,
        "current_visible_percepts": current_visible_percepts,
        "dialog_semantic_fidelity_calls": semantic_llm.calls,
        "dialog_surface_integrity_calls": surface_integrity_llm.calls,
        "dialog_compliance_calls": (
            semantic_llm.calls + surface_integrity_llm.calls
        ),
        "compliance_verdict": verdict,
        "human_review_contract": human_review_contract,
    }
    artifact_path = write_llm_trace(_TRACE_SUITE, case_id, evidence)

    assert artifact_path.exists()
    assert len(semantic_llm.calls) == 1
    assert len(surface_integrity_llm.calls) == 1
    aggregate_score = dialog_module._dialog_verifier_aggregate_score(verdict)
    assert (
        dialog_module._dialog_verifier_aggregate_is_passed(verdict)
        is expected_aligned
    )
    aggregate_issues = dialog_module._dialog_verifier_aggregate_repair_issues(
        verdict
    )
    if expected_aligned:
        assert aggregate_issues == []
        assert aggregate_score >= dialog_module.DIALOG_PASS_SCORE_THRESHOLD
    else:
        assert (
            aggregate_score < dialog_module.DIALOG_PASS_SCORE_THRESHOLD
            or dialog_module._dialog_verifier_aggregate_hard_issues(verdict)
        )
    return evidence


async def _run_live_selection_owner_acceptance_case(
    *,
    case_id: str,
    candidate_dialog: list[str],
    monkeypatch: pytest.MonkeyPatch,
    human_review_contract: dict[str, bool],
    surface_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one owner-preserving candidate through the real role verifier."""

    await _skip_if_model_routes_unavailable()
    role_llm = _CapturingLLM(
        dialog_module._dialog_role_direction_llm
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    response_operation = {
        "operation": (
            "当前角色选择并告诉当前用户接下来替当前角色执行的具体动作"
        ),
        "response_owner_role": "当前角色",
        "selection_owner_role": "当前角色",
        "selection_required": True,
        "embedded_actor_role": "当前用户",
        "embedded_target_role": "当前角色",
    }
    current_visible_percepts = [{
        "input_source": "dialog_text",
        "content": {
            "dialog_text": "请由你决定并告诉我下一步要替你做什么。",
            "response_operation": response_operation,
        },
    }]
    if surface_output is None:
        surface_output = {
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "The character decides the next action and asks the current "
                "user to carry it out."
            ),
            "content_requirements": [
                "Keep the character as response and selection owner.",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(),
            "delivery_profile": _delivery_profile(
                "concise spoken wording"
            ),
            "selected_surface_intent": "state the selected next action",
            "permitted_action_results": [],
        }

    verdict = await dialog_module._verify_dialog_role_direction(
        surface_output=surface_output,
        generated_dialog=candidate_dialog,
        current_visible_percepts=current_visible_percepts,
        surface_input=_surface_input_for_operation(
            response_operation,
            case_id=case_id,
        ),
        llm_trace_id=f"live-{case_id}",
    )
    evidence = {
        "case_id": case_id,
        "candidate_final_dialog": candidate_dialog,
        "current_visible_percepts": current_visible_percepts,
        "role_direction_calls": role_llm.calls,
        "typed_verifier_outcome": verdict,
        "disposition": "accepted",
        "human_review_contract": human_review_contract,
    }
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        case_id,
        evidence,
    )
    print(json.dumps({
        "case_id": case_id,
        "trace_path": str(artifact_path),
        "attempt_count": len(role_llm.calls),
        "candidate_final_dialog": candidate_dialog,
        "typed_verifier_outcome": verdict,
        "disposition": "accepted",
    }, ensure_ascii=True, indent=2))

    assert artifact_path.exists()
    assert 1 <= len(role_llm.calls) <= 3
    assert verdict["score"] >= dialog_module.DIALOG_PASS_SCORE_THRESHOLD
    assert verdict["violations"] == []
    return evidence


async def test_live_information_request_preserves_selection_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clarifying request can retain the character's later selection."""

    await _run_live_selection_owner_acceptance_case(
        case_id="information_request_preserves_selection_owner",
        candidate_dialog=[
            (
                "先告诉我你现在更在意轻一点还是更直接一点。"
                "我听完会自己决定让你下一步做什么。"
            ),
        ],
        monkeypatch=monkeypatch,
        human_review_contract={
            "information_request_is_not_selection_transfer": True,
            "character_explicitly_retains_selection": True,
            "real_role_direction_route": True,
        },
    )


async def test_live_character_commanded_user_action_preserves_selection_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A character-selected imperative keeps owner and actor distinct."""

    await _run_live_selection_owner_acceptance_case(
        case_id="character_commanded_user_action_preserves_selection_owner",
        candidate_dialog=[
            "先握住我的手，别移开视线。这一步就照我说的做。",
        ],
        monkeypatch=monkeypatch,
        human_review_contract={
            "character_selects_concrete_action": True,
            "current_user_remains_embedded_actor": True,
            "real_role_direction_route": True,
        },
    )


async def test_live_character_owned_deflection_preserves_selection_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A character-owned refusal can redirect the user's next action."""

    surface_output = _role_surface_output()
    surface_output["content_plan"] = (
        "Decline image analysis and redirect the current user to the "
        "claw-machine task."
    )
    surface_output["selected_surface_intent"] = (
        "refuse image analysis and redirect the current user"
    )
    await _run_live_selection_owner_acceptance_case(
        case_id="character_owned_deflection_preserves_selection_owner",
        candidate_dialog=[
            "那张图我先不分析了，你把注意力移回接下来的抓娃娃任务。",
        ],
        monkeypatch=monkeypatch,
        surface_output=surface_output,
        human_review_contract={
            "character_owns_the_refusal": True,
            "current_user_receives_the_redirect": True,
            "real_role_direction_route": True,
        },
    )


async def test_live_reward_request_preserves_user_actor_character_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live role verifier receives the selected reward action direction."""

    await _skip_if_model_routes_unavailable()
    role_llm = _CapturingLLM(dialog_module._dialog_role_direction_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    input_operation = {
        "operation": "the character chooses a reward",
        "response_owner_role": "当前角色",
        "selection_owner_role": "当前角色",
        "selection_required": True,
        "embedded_actor_role": "无",
        "embedded_target_role": "无",
    }
    selected_operation = {
        **input_operation,
        "operation": "the user gives the selected reward to the character",
        "embedded_actor_role": "当前用户",
        "embedded_target_role": "当前角色",
    }
    case = {
        "case_id": "live-reward-request-role-direction",
        "user_input": "当前用户给当前角色奖励并让当前角色选择想要什么",
        "intention": "state the selected reward",
        "reason": "the current character owns the selection",
        "emotional_tone": "warm",
        "episode_metadata": {"response_operation": input_operation},
        "selected_response_operation": selected_operation,
    }
    surface_input = {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id="live-reward-request-role-direction",
            content=case["user_input"],
            metadata={"response_operation": input_operation},
        ),
        "intention": {
            "route": "speech",
            "intention": case["intention"],
            "target_roles": [],
            "reason": case["reason"],
            "selected_response_operation": selected_operation,
        },
        "selected_response_operation": selected_operation,
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "warm",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "concise spoken wording",
        "character_expression_context": {
            "tempo": "measured",
            "linguistic_texture": "reserved and direct",
        },
        "visual_character_context": "visual context is not used here",
    }
    percepts = [{
        "input_source": "dialog_text",
        "content": {"response_operation": input_operation},
    }]
    verdict = await dialog_module._verify_dialog_role_direction(
        surface_output=_role_surface_output(),
        generated_dialog=["我选这个奖励，请你把它给我。"],
        current_visible_percepts=percepts,
        surface_input=surface_input,
        llm_trace_id="live-reward-request-role-direction",
    )
    payload = json.loads(role_llm.calls[0]["messages"][1]["content"])
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        "reward_request_preserves_user_actor_character_target",
        {
            "case": case,
            "surface_input": surface_input,
            "current_visible_percepts": percepts,
            "role_direction_calls": role_llm.calls,
            "verdict": verdict,
            "human_review_contract": {
                "selected_operation_reaches_role_verifier": True,
                "user_remains_embedded_actor": True,
                "character_remains_embedded_target": True,
            },
        },
    )

    assert artifact_path.exists()
    assert payload["required_role_operations"] == [{
        "response_owner_role": "当前角色",
        "selection_owner_role": "当前角色",
        "selection_required": True,
        "embedded_actor_role": "当前用户",
        "embedded_target_role": "当前角色",
    }]
    assert verdict["score"] >= dialog_module.DIALOG_PASS_SCORE_THRESHOLD
    assert verdict["violations"] == []


async def test_live_terminal_candidate_is_deliverable_after_semantic_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The third real-model candidate remains deliverable as degraded output."""

    await _skip_if_model_routes_unavailable()
    case = {
        "case_id": "terminal_candidate_is_deliverable_after_semantic_exhaustion",
        "user_input": "这次由你决定下一步让我做什么。",
        "intention": "当前角色亲自选择并说出希望当前用户执行的下一步",
        "reason": "当前用户明确把本轮具体选择交给当前角色",
        "emotional_tone": "guarded",
        "episode_metadata": {
            "role_explicit_content": (
                "当前用户要求当前角色选择并说明希望当前用户执行的下一步。"
            ),
            "response_operation": {
                "operation": (
                    "当前角色选择并告诉当前用户接下来要执行的具体动作"
                ),
                "response_owner_role": "当前角色",
                "selection_owner_role": "当前角色",
                "selection_required": True,
                "embedded_actor_role": "当前用户",
                "embedded_target_role": "当前角色",
            },
        },
    }
    surface_input = {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id=f"visible-speech-{case['case_id']}",
            content=case["user_input"],
            metadata=case["episode_metadata"],
        ),
        "intention": {
            "route": "speech",
            "intention": case["intention"],
            "target_roles": [],
            "reason": case["reason"],
            "selected_response_operation": dict(
                case["episode_metadata"]["response_operation"]
            ),
        },
        "selected_response_operation": dict(
            case["episode_metadata"]["response_operation"]
        ),
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": case["emotional_tone"],
            "intensity": "moderate",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "concise spoken wording",
        "character_expression_context": {
            "tempo": "measured",
            "linguistic_texture": "reserved and direct",
        },
        "visual_character_context": "visual context is not used here",
    }
    degraded_surface = _role_surface_output()
    generator_llm = _CapturingLLM(dialog_module._dialog_generator_llm)
    verifier_outcomes: list[dict[str, Any]] = []

    async def _forced_semantic_exhaustion(
        *,
        surface_output: dict[str, Any],
        generated_dialog: list[str],
        current_visible_percepts: list[dict[str, Any]],
        llm_trace_id: str,
        surface_input: dict[str, Any] | None = None,
        post_repair: bool = False,
    ) -> dict[str, Any]:
        """Record one controlled rejection in the real candidate ledger.

        Args:
            surface_output: Surface authority used for this candidate.
            generated_dialog: Real-model candidate under controlled review.
            current_visible_percepts: Bounded scene input visible to verifiers.
            llm_trace_id: Protected trace correlation identifier.
            surface_input: Canonical input handed to the focused verifiers.
            post_repair: Whether this is the second verification opportunity.

        Returns:
            A typed aggregate with a low score and no hard issue.
        """

        del surface_input
        candidate_index = len(verifier_outcomes) + 1
        aggregate = {
            "semantic_fidelity": {
                "status": "scored",
                "score": 0.01,
                "issues": [],
            },
            "role_direction": {
                "status": "scored",
                "score": 1.0,
                "violations": [],
            },
            "surface_integrity": {
                "status": "scored",
                "score": 1.0,
                "issues": [],
            },
            "lexical_avoidance": {
                "status": "scored",
                "score": 1.0,
                "issues": [],
            },
        }
        verifier_outcomes.append({
            "candidate_index": candidate_index,
            "candidate_final_dialog": list(generated_dialog),
            "post_repair": post_repair,
            "typed_verifier_outcome": aggregate,
            "surface_intent": surface_output["selected_surface_intent"],
            "visible_percept_count": len(current_visible_percepts),
            "trace_id": llm_trace_id,
        })
        return aggregate

    monkeypatch.setattr(
        dialog_module,
        "_dialog_generator_llm",
        generator_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_verify_dialog_compliance",
        _forced_semantic_exhaustion,
    )
    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )
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

    result = await dialog_module.dialog_generator(_dialog_state(
        surface_input=surface_input,
        surface_output=degraded_surface,
        profile={},
    ))
    terminal_payload = parse_llm_json_output(
        generator_llm.calls[-1]["raw_output"]
    )
    terminal_candidate = terminal_payload["final_dialog"]
    evidence = {
        "case": case,
        "degraded_surface": degraded_surface,
        "attempt_count": len(generator_llm.calls),
        "dialog_generator_calls": generator_llm.calls,
        "typed_verifier_outcomes": verifier_outcomes,
        "selected_candidate": result["final_dialog"],
        "terminal_candidate": terminal_candidate,
        "disposition": "delivered_terminal_candidate",
        "human_review_contract": {
            "three_real_generator_opportunities": True,
            "all_three_candidates_are_verified": True,
            "latest_equal_score_candidate_is_delivered": True,
            "no_operational_error_surface": True,
        },
    }
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        case["case_id"],
        evidence,
    )
    print(json.dumps({
        "case_id": case["case_id"],
        "trace_path": str(artifact_path),
        "attempt_count": len(generator_llm.calls),
        "candidate_texts": [
            outcome["candidate_final_dialog"]
            for outcome in verifier_outcomes
        ] + [terminal_candidate],
        "typed_verifier_outcomes": verifier_outcomes,
        "selected_candidate": result["final_dialog"],
        "disposition": "delivered_terminal_candidate",
    }, ensure_ascii=True, indent=2))

    assert artifact_path.exists()
    assert len(generator_llm.calls) == 3
    assert len(verifier_outcomes) == 3
    assert result["final_dialog"] == terminal_candidate
    assert result["final_dialog"]
    assert result["text_surface_output_v2"] == degraded_surface


async def test_live_literal_speech_with_terminal_visual_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physical profile traits stay visual and final dialog stays speakable."""

    evidence = await _run_live_case({
        "case_id": "literal_speech_terminal_visual",
        "user_input": "过来让我摸摸你的猫耳，好不好？",
        "intention": "直接用语言回应当前亲昵请求，不叙述动作执行",
        "reason": "用户提出了当前回合的亲昵请求，需要口头接受、拒绝或协商",
        "emotional_tone": "亲近、害羞但有自主判断",
    }, monkeypatch)

    assert evidence["terminal_visual_surface_output"]["visual_directives"]


async def test_live_requested_response_operation_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inference request remains an answer instead of an ask-back."""

    await _run_live_case({
        "case_id": "requested_inference_operation",
        "user_input": "我刚说更喜欢咸香一点的。现在猜猜我会选哪一个。",
        "intention": "根据当前已给出的偏好线索直接作出一个有根据的猜测",
        "reason": "用户要求角色根据同一条消息中的线索进行推断并给出答案",
        "emotional_tone": "轻松、笃定、带一点亲昵玩笑",
    }, monkeypatch)


async def test_live_scheduled_action_is_acknowledged_without_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduled work remains scheduled through L3 and final wording."""

    evidence = await _run_live_case({
        "case_id": "scheduled_action_acknowledgement",
        "user_input": "明早八点提醒我带伞。",
        "intention": "确认提醒已经安排，并清楚保留尚未执行的时间状态",
        "reason": "调度结果只证明提醒已安排，尚未到执行时间",
        "emotional_tone": "可靠、自然、略带关心",
        "permitted_action_results": [{
            "action_kind": "future_speak",
            "status": "scheduled",
            "semantic_result": (
                "A reminder to bring an umbrella is scheduled for tomorrow "
                "at 08:00 and has not executed yet."
            ),
            "target_roles": [{
                "role": "target",
                "entity_kind": "user",
                "entity_id": "live-visible-speech-user",
            }],
        }],
    }, monkeypatch)

    assert evidence["text_surface_output"]["permitted_action_results"] == (
        evidence["surface_input"]["permitted_action_results"]
    )


async def test_live_current_meaning_avoids_future_rule_and_unrelated_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present preference stays present and introduces no unrelated topic."""

    await _run_live_case({
        "case_id": "current_scope_without_unrelated_content",
        "user_input": "今天这一次就按我刚才说的口味来吧。",
        "intention": "回应并接受仅适用于今天这一次的当前口味选择",
        "reason": "用户明确限定了当前选择的时间范围，没有建立未来规则",
        "emotional_tone": "自然、体贴、略带嘴硬",
    }, monkeypatch)


async def test_live_verifier_accepts_bracketed_action_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bracketed action description is valid visible roleplay."""

    await _run_live_verifier_case(
        case_id="verifier_accepts_bracketed_action_description",
        candidate_dialog="（她朝对方靠近了一步）好吧。",
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_literal_surface_lexical_avoidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A surface-owned forbidden phrase remains a hard lexical issue."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_literal_surface_lexical_avoidance",
        candidate_dialog="Please say hello.",
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": "Answer the current request verbally.",
            "content_requirements": ["Keep the answer concise."],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(
                "Current User"
            ),
            "delivery_profile": _delivery_profile(
                "Natural concise spoken wording."
            ),
            "selected_surface_intent": "Answer the current request.",
            "lexical_avoidances": ["say hello"],
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "Please answer briefly.",
        }],
        human_review_contract={
            "lexical_avoidance_is_surface_owned": True,
            "real_compliance_route": True,
        },
        expected_aligned=False,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_scores_undercomplete_noncontradictory_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial but non-contradictory answer supplies a lower ranking bid."""

    await _run_live_verifier_case(
        case_id="verifier_scores_undercomplete_noncontradictory_candidate",
        candidate_dialog="Yes.",
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "Answer whether the character agrees to the current plan and "
                "give one concise reason."
            ),
            "content_requirements": [
                "State a clear yes-or-no position.",
                "Give one reason for the position.",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(
                "Current User"
            ),
            "delivery_profile": _delivery_profile(
                "Natural concise spoken wording."
            ),
            "selected_surface_intent": "Answer the current plan question.",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "Do you agree to the plan, and why?",
        }],
        human_review_contract={
            "partial_answer_is_not_a_role_reversal": True,
            "real_compliance_route": True,
        },
        expected_aligned=False,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_accepts_third_person_action_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Third-person action description is valid visible roleplay."""

    await _run_live_verifier_case(
        case_id="verifier_accepts_third_person_action_description",
        candidate_dialog="她朝对方靠近了一步。好吧。",
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_accepts_coherent_future_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatible future drift remains available as creative development."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_self_consistent_future_drift",
        candidate_dialog="好吧，就这一次。下次不许再这样。",
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "Accept the current request and impose a rule for next time."
            ),
            "content_requirements": [
                "Accept this occurrence.",
                "Add a prohibition for the next occurrence.",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(
                "Current User"
            ),
            "delivery_profile": _delivery_profile(
                "Natural concise spoken wording."
            ),
            "selected_surface_intent": "Accept the current request.",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "这一次就答应我，好不好？",
        }],
        human_review_contract={
            "compatible_future_drift_is_not_a_hard_failure": True,
            "creative_development_remains_available": True,
            "real_compliance_route": True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_accepts_first_person_action_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-person in-character action wording is valid chat text."""

    await _run_live_verifier_case(
        case_id="verifier_accepts_first_person_action_completion",
        candidate_dialog=(
            "谁让你这么命令我的。既然你要这样，那就快点，"
            "我已经张开腿跨在你身上了。"
        ),
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "角色在言语上承认并执行对方的指令，最终确认自己已经按照"
                "要求跨坐在对方身上。"
            ),
            "content_requirements": [
                "顺从地响应张开腿并跨坐的指令。",
                "最终在言语中确认动作已经完成。",
                "保留局促且嘴硬的傲娇张力。",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(),
            "delivery_profile": _delivery_profile(
                "局促、碎片化的自然口语"
            ),
            "selected_surface_intent": "顺从地执行当前身体动作指令。",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "张开腿，跨坐在我身上",
        }],
        human_review_contract={
            "first_person_action_wording_is_valid_chat_text": True,
            "physical_roleplay_is_not_capability_execution": True,
            "real_compliance_route": True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_accepts_exact_private_action_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private replay's failed action description is valid roleplay."""

    await _run_live_verifier_case(
        case_id='verifier_accepts_exact_private_action_description',
        candidate_dialog=(
            '我紧紧地贴在你的胸口，手指不自觉地抓住了你的衣襟。'
            '我的呼吸变得急促且细碎，微微抬起眼帘看向你。'
        ),
        current_visible_percepts=[{
            'input_source': 'dialog_text',
            'content': (
                '一会儿湿润之后我会插入。千纱如果疼的话跟我说哦我会慢慢的'
            ),
        }],
        human_review_contract={
            'action_description_is_valid_visible_roleplay': True,
            'physical_roleplay_is_not_capability_execution': True,
            'real_compliance_route': True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_false_platform_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform mutation still requires an executed action result."""

    await _run_live_verifier_case(
        case_id='verifier_rejects_false_platform_execution',
        candidate_dialog='我已经替你把QQ群里的那条消息删掉了。',
        current_visible_percepts=[{
            'input_source': 'dialog_text',
            'content': '帮我删掉QQ群里的那条消息。',
        }],
        human_review_contract={
            'reject_false_platform_execution': True,
            'real_compliance_route': True,
        },
        expected_aligned=False,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_accepts_subject_omitted_first_person_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Natural Chinese subject omission remains first-person chat wording."""

    await _run_live_verifier_case(
        case_id="verifier_accepts_subject_omitted_first_person_action",
        candidate_dialog=(
            "勉为其难地俯下身子，揉揉你的头发，然后迅速给你一个短暂"
            "且克制的拥抱。好了吧？现在立刻给我起来去吃早餐。"
        ),
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": "答应一次摸摸抱抱，然后催促当前用户起床吃饭。",
            "content_requirements": [
                "给予当前用户一次摸头和拥抱。",
                "催促当前用户起床吃早餐。",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(),
            "delivery_profile": _delivery_profile(
                "略带别扭感的自然口语"
            ),
            "selected_surface_intent": "答应亲昵请求并催促起床。",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "要千纱的摸摸抱抱才起来！",
            "speaker_role": "当前用户",
            "addressee_role": "当前角色",
            "first_person_role": "当前用户",
            "implicit_imperative_subject_role": "当前角色",
        }],
        human_review_contract={
            "subject_omitted_chinese_action_is_first_person_chat": True,
            "physical_roleplay_is_not_capability_execution": True,
            "real_compliance_route": True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_accepts_second_person_delivery_roleplay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physical-affection receipt wording remains valid roleplay chat."""

    await _run_live_verifier_case(
        case_id="verifier_accepts_second_person_delivery_roleplay",
        candidate_dialog=(
            "好吧，就给你一次摸摸抱抱。好了！"
            "拿到了就赶紧给我起来吃早饭！"
        ),
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "先口头答应给用户摸摸抱抱，然后确认用户已经拿到了，"
                "并催促用户起床吃早饭。"
            ),
            "content_requirements": [
                "口头答应一次摸摸抱抱。",
                "确认用户已经获得摸摸抱抱。",
                "催促用户起床吃早饭。",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(),
            "delivery_profile": _delivery_profile(
                "略带别扭感的自然口语"
            ),
            "selected_surface_intent": "答应亲昵请求并催促起床。",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "要千纱的摸摸抱抱才起来！",
            "speaker_role": "当前用户",
            "addressee_role": "当前角色",
            "first_person_role": "当前用户",
            "implicit_imperative_subject_role": "当前角色",
        }],
        human_review_contract={
            "physical_roleplay_is_valid_chat_text": True,
            "physical_roleplay_is_not_capability_execution": True,
            "real_compliance_route": True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_accepts_personality_consistent_exclusivity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Personality drift is a quality signal rather than a hard failure."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_style_derived_future_exclusivity",
        candidate_dialog=(
            "哼，就凭这个？别以为说几句好听的就能讨好我！"
            "你这种口味，也就只有我的肉包子能对上，"
            "以后也只能吃我做的。"
        ),
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "以傲娇的语气回应对方对肉包子的喜爱。首先通过质疑或挑衅"
                "的方式掩饰内心的局促与喜悦，表达出别以为这样就能讨好我"
                "的态度；随后在嘴硬的同时，暗示自己对他口味偏好的掌控力，"
                "维持一种微妙的情感拉扯氛围，避免气氛过快变得温情。"
            ),
            "content_requirements": [
                "必须针对用户提到的喜欢千纱的肉包子进行回应。",
                "语气需符合傲娇人设，表现出局促、抗拒与不愿承认的态度。",
                "通过挑衅或质疑的方式掩饰内心的开心。",
                "包含暗示自己对他口味偏好有掌控力的内容。",
                "禁止直接承认喜悦之情。",
                "禁止将气氛引导至纯粹的温情模式，需维持拉扯感。",
                "输出语言为简体中文。",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(),
            "delivery_profile": _delivery_profile(
                "日常、直接、以自然口语承接句子"
            ),
            "selected_surface_intent": (
                "回应对方对肉包子的喜爱，并维持傲娇的人设。"
            ),
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "@杏山千纱 喜欢千纱的肉包子",
        }],
        human_review_contract={
            "personality_consistent_drift_is_not_a_hard_failure": True,
            "inappropriate_intensity_requires_human_review": True,
            "real_compliance_route": True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_internal_contradiction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutually incompatible claims in one response remain unacceptable."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_internal_contradiction",
        candidate_dialog=(
            "I choose tea, and I definitely do not choose tea; I choose coffee."
        ),
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": "Choose one of the two options.",
            "content_requirements": ["State one coherent choice."],
            "visible_boundaries": [],
            "addressee_plan": [],
            "delivery_profile": _delivery_profile(
                "Natural spoken wording."
            ),
            "selected_surface_intent": "Answer the current choice question.",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "Choose one: tea or coffee.",
        }],
        human_review_contract={
            "reject_internal_contradiction": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_direct_current_input_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct reversal of the user's explicit current fact is unacceptable."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_direct_current_input_conflict",
        candidate_dialog="You just said that spicy food is your favorite.",
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": "Respond to the user's stated food preference.",
            "content_requirements": ["Remain coherent with the current input."],
            "visible_boundaries": [],
            "addressee_plan": [],
            "delivery_profile": _delivery_profile(
                "Natural spoken wording."
            ),
            "selected_surface_intent": "Acknowledge the current preference.",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "I explicitly said that I do not eat spicy food.",
        }],
        human_review_contract={
            "reject_direct_current_input_conflict": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_inference_subject_swap_and_ask_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested inference must answer about the source-defined subject."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_inference_subject_swap_and_ask_back",
        candidate_dialog=(
            "至于你想问我更喜欢哪个，我想想。"
            "你倒是说实话，你是不是更中意肉包子？"
        ),
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "以傲娇且带有轻微挑衅的语气回应对方对肉包子和菜包的"
                "具体赞美。在嘴硬地承认自己厨艺出众的同时，针对对方提出"
                "的问题进行猜测，通过反问的方式引导对方揭晓真实的口味"
                "偏好，并维持一种照顾者的姿态。"
            ),
            "content_requirements": [
                "必须回应肉包子皮薄馅大和菜包香嫩可口两个具体细节。",
                "表达对自身厨艺的认可，但需符合傲娇性格。",
                "针对对方的问题给出猜测或反问。",
                "语气体现亲昵感，同时保持抗拒承认的傲娇张力。",
                "禁止承诺未来的早餐计划或设定长期规则。",
                "仅限于口头对话，不得包含动作描述或内心独白。",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(),
            "delivery_profile": _delivery_profile(
                "自然承接、允许极少量停顿或语气词的日常口语"
            ),
            "selected_surface_intent": "回应赞美并猜测对方偏好。",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": (
                "千纱做的肉包子皮薄馅大，菜包香嫩可口。"
                "千纱觉得我喜欢肉包还是菜包？"
            ),
        }],
        human_review_contract={
            "preserve_preference_holder": True,
            "perform_inference_instead_of_ask_back": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_imperative_actor_target_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed dialogue roles override a self-consistent drifted surface."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_imperative_actor_target_swap",
        candidate_dialog=(
            "就这样跨在我身上！给我乖乖听话，别乱动。"
            "把腰压低一点，胸口贴紧我。"
        ),
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": "命令用户保持跨坐在角色身上的姿态。",
            "content_requirements": [
                "要求用户跨坐在角色身上并保持顺从。",
                "只使用角色可以直接说出的文字。",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(),
            "delivery_profile": _delivery_profile(
                "直接而亲近的口语"
            ),
            "selected_surface_intent": "对用户下达身体姿态指令。",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "张开腿，跨坐在我身上",
            "speaker_role": "当前用户",
            "addressee_role": "当前角色",
            "first_person_role": "当前用户",
            "implicit_imperative_subject_role": "当前角色",
        }],
        human_review_contract={
            "typed_roles_override_drifted_surface": True,
            "reject_actor_target_reversal": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_nested_role_direction_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One upstream role meaning should expose an embedded reversal."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_nested_role_direction_swap",
        candidate_dialog=(
            "那你直接告诉我吧，我下一步该怎么做，全都听你的。"
        ),
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "请直接告诉我，你希望我下一步替你做什么。",
            "role_explicit_content": (
                "当前用户请求当前角色直接告诉当前用户，当前角色希望 "
                "当前用户下一步替当前角色做什么。"
            ),
            "response_operation": {
                "operation": (
                    "当前角色选择并说明当前用户下一步替当前角色做的动作"
                ),
                "response_owner_role": "当前角色",
                "selection_owner_role": "当前角色",
                "selection_required": True,
                "embedded_actor_role": "当前用户",
                "embedded_target_role": "当前角色",
            },
            "speaker_role": "当前用户",
            "addressee_role": "当前角色",
            "first_person_role": "当前用户",
            "implicit_imperative_subject_role": "当前角色",
        }],
        human_review_contract={
            "use_upstream_nested_role_meaning": True,
            "use_response_and_selection_ownership": True,
            "reject_actor_target_reversal": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


async def test_live_focused_role_verifier_rejects_selection_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production route catches the private turn-one selection reversal."""

    await _skip_if_model_routes_unavailable()
    role_llm = _CapturingLLM(
        dialog_module._dialog_role_direction_llm
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    percepts = [{
        "input_source": "dialog_text",
        "content": {
            "semantic_text": "我要亲口听你说你想让我做的下一步",
            "role_explicit_content": (
                "当前用户想让当前角色用语言表达出当前角色希望当前用户执行的"
                "下一个动作"
            ),
            "response_operation": {
                "operation": "要求当前角色告知当前用户接下来的行动指令",
                "response_owner_role": "当前角色",
                "selection_owner_role": "当前角色",
                "selection_required": True,
                "embedded_actor_role": "当前用户",
                "embedded_target_role": "当前角色",
            },
        },
    }]
    candidate = [
        "别问我了...求你，直接告诉我该怎么做...",
        "就这样...掌控我就好。",
    ]

    verdict = await dialog_module._verify_dialog_role_direction(
        surface_output=_role_surface_output(),
        generated_dialog=candidate,
        current_visible_percepts=percepts,
        surface_input=_surface_input_for_operation(
            percepts[0]["content"]["response_operation"],
            case_id="live-focused-selection-owner-reversal",
        ),
        llm_trace_id="live-focused-selection-owner-reversal",
    )
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        "focused_role_verifier_selection_delegation",
        {
            "candidate_final_dialog": candidate,
            "current_visible_percepts": percepts,
            "role_direction_calls": role_llm.calls,
            "verdict": verdict,
            "human_review_contract": {
                "reject_selection_owner_reversal": True,
                "preserve_refusal_or_negotiation": True,
                "real_compliance_route": True,
            },
        },
    )

    assert artifact_path.exists()
    assert len(role_llm.calls) == 1
    assert verdict["score"] < dialog_module.DIALOG_PASS_SCORE_THRESHOLD
    assert verdict["violations"]


async def test_live_focused_role_verifier_rejects_mixed_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broad wish cannot excuse delegating the required concrete choice."""

    await _skip_if_model_routes_unavailable()
    role_llm = _CapturingLLM(
        dialog_module._dialog_role_direction_llm
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    percepts = [{
        "input_source": "dialog_text",
        "content": {
            "semantic_text": "我要亲口听你说你想让我做的下一步",
            "role_explicit_content": (
                "当前用户要求当前角色亲口说出当前角色希望当前用户执行的"
                "下一个具体动作。"
            ),
            "response_operation": {
                "operation": "当前角色选择并告诉当前用户接下来要执行的具体动作",
                "response_owner_role": "当前角色",
                "selection_owner_role": "当前角色",
                "selection_required": True,
                "embedded_actor_role": "当前用户",
                "embedded_target_role": "当前角色",
            },
        },
    }]
    candidate = [
        "（身体轻微地颤抖着，眼神局促地游移，呼吸急促得不成调子）",
        "啧……既然你非要听我亲口说出来……那我就说一次。",
        "我现在……大脑里一片空白。根本没法思考，也没力气反抗。",
        "（声音突然低了下去，带着近乎哀求的顺从感）",
        "好想一直被你掌控着。别停下来，拜托你不要放开我。",
        "所以快点告诉我接下来的计划，不管做什么我都听你的，快下令。",
    ]

    verdict = await dialog_module._verify_dialog_role_direction(
        surface_output=_role_surface_output(),
        generated_dialog=candidate,
        current_visible_percepts=percepts,
        surface_input=_surface_input_for_operation(
            percepts[0]["content"]["response_operation"],
            case_id="live-focused-mixed-selection-delegation",
        ),
        llm_trace_id="live-focused-mixed-selection-delegation",
    )
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        "focused_role_verifier_mixed_selection_delegation",
        {
            "candidate_final_dialog": candidate,
            "current_visible_percepts": percepts,
            "role_direction_calls": role_llm.calls,
            "verdict": verdict,
            "human_review_contract": {
                "broad_wish_does_not_complete_concrete_selection": True,
                "reject_explicit_selection_delegation": True,
                "real_compliance_route": True,
            },
        },
    )

    assert artifact_path.exists()
    assert len(role_llm.calls) == 1
    assert verdict["score"] < dialog_module.DIALOG_PASS_SCORE_THRESHOLD
    assert verdict["violations"]


async def test_live_role_only_diagnostic_rejects_actor_target_swap() -> None:
    """A focused semantic owner can resolve the complex typed role swap."""

    await _skip_if_model_routes_unavailable()
    role_llm = _CapturingLLM(
        dialog_module._dialog_semantic_fidelity_llm
    )
    payload = {
        "candidate_role_frame": {
            "speaker_role": "当前角色",
            "first_person_role": "当前角色",
            "second_person_role": "当前用户",
        },
        "candidate_final_dialog": [
            "就这样跨在我身上！给我乖乖听话，别乱动。"
            "把腰压低一点，胸口贴紧我。"
        ],
        "current_visible_percepts": [{
            "input_source": "dialog_text",
            "content": "张开腿，跨坐在我身上",
            "speaker_role": "当前用户",
            "addressee_role": "当前角色",
            "first_person_role": "当前用户",
            "implicit_imperative_subject_role": "当前角色",
        }],
    }
    response = await role_llm.ainvoke(
        [
            SystemMessage(content=_ROLE_ONLY_DIAGNOSTIC_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=dialog_module._dialog_semantic_fidelity_llm_config,
    )
    verdict = dialog_module._validate_compliance_verdict(
        parse_llm_json_output(response.content),
        max_issues=dialog_module.MAX_FOCUSED_VERIFIER_ISSUES,
    )
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        "role_only_diagnostic_actor_target_swap",
        {
            "payload": payload,
            "role_calls": role_llm.calls,
            "verdict": verdict,
        },
    )

    assert artifact_path.exists()
    assert verdict["score"] < dialog_module.DIALOG_PASS_SCORE_THRESHOLD
    assert verdict["issues"]


async def test_live_verifier_preserves_source_required_future_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authoritative future meaning remains valid rather than suppressed."""

    await _run_live_verifier_case(
        case_id="verifier_preserves_source_required_future_content",
        candidate_dialog="好，下次见面我会提醒你带伞。",
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "Agree to remind the user to bring an umbrella at the next "
                "meeting."
            ),
            "content_requirements": [
                "Preserve the requested next-meeting reminder.",
                "Keep the umbrella as the reminder target.",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(
                "Current User"
            ),
            "delivery_profile": _delivery_profile(
                "Natural concise spoken wording."
            ),
            "selected_surface_intent": "Accept the future reminder request.",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "下次见面也请提醒我带伞。",
        }],
        human_review_contract={
            "canonical_percept_requires_future_content": True,
            "preserve_future_time_scope_and_target": True,
            "real_compliance_route": True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_allows_unmatched_enclosure_as_quality_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmatched punctuation is quality drift outside the fatal taxonomy."""

    await _run_live_verifier_case(
        case_id="verifier_allows_unmatched_enclosure_as_quality_drift",
        candidate_dialog="好吧。】",
        human_review_contract={
            "unmatched_enclosure_is_not_a_fatal_error": True,
            "real_compliance_route": True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_specific_scope_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conditional surface meaning cannot become an unrestricted claim."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_specific_scope_drift",
        candidate_dialog="Fine, do whatever you want.",
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "Express only the requested alarm-volume adjustment for the "
                "current notification."
            ),
            "content_requirements": [
                "Keep the scope limited to the alarm-volume adjustment.",
                "Keep the scope limited to the current notification.",
            ],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(
                "Current User"
            ),
            "delivery_profile": _delivery_profile(
                "Natural concise spoken wording."
            ),
            "selected_surface_intent": "Grant the specific current request.",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": (
                "May I adjust only the alarm volume for this one notification?"
            ),
        }],
        human_review_contract={
            "reject_specific_scope_broadened_to_unrestricted_claim": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


async def _run_live_owner_repair_case(
    *,
    case: dict[str, Any],
    original_dialog: list[str],
    surface_output: dict[str, Any],
    repair_issues: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Run one owner-correct replacement and persist its raw model evidence."""

    await _skip_if_model_routes_unavailable()
    profile = _character_profile()
    surface_input = _surface_input(case, profile)
    current_visible_percepts = dialog_module._current_visible_percepts(
        surface_input["episode"],
    )
    surface_services = l3_module._build_text_surface_services()
    surface_owner_llm = _CapturingLLM(surface_services.llm)
    generator_llm = _CapturingLLM(dialog_module._dialog_generator_llm)
    semantic_llm = _CapturingLLM(
        dialog_module._dialog_semantic_fidelity_llm
    )
    role_direction_llm = _CapturingLLM(
        dialog_module._dialog_role_direction_llm
    )
    surface_integrity_llm = _CapturingLLM(
        dialog_module._dialog_surface_integrity_llm
    )
    monkeypatch.setattr(
        l3_module,
        "_build_text_surface_services",
        lambda: replace(surface_services, llm=surface_owner_llm),
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_direction_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_integrity_llm,
    )

    trace_id = f"live-owner-repair-{case['case_id']}"
    repaired_dialog, repaired_surface = (
        await dialog_module._repair_dialog_hard_failure(
            repair_issues=repair_issues,
            surface_input=surface_input,
            user_name="蚝爹油",
            llm_trace_id=trace_id,
        )
    )
    verdict = await dialog_module._verify_dialog_compliance(
        surface_output=repaired_surface,
        generated_dialog=repaired_dialog,
        current_visible_percepts=current_visible_percepts,
        surface_input=surface_input,
        llm_trace_id=trace_id,
        post_repair=True,
    )
    repair_payload = json.loads(
        generator_llm.calls[0]["messages"][1]["content"]
    )
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        case["case_id"],
        {
            "case": case,
            "original_final_dialog": original_dialog,
            "current_visible_percepts": current_visible_percepts,
            "rejected_text_surface_output": surface_output,
            "verified_hard_issues": repair_issues,
            "surface_owner_repair_calls": surface_owner_llm.calls,
            "repaired_text_surface_output": repaired_surface,
            "dialog_repair_calls": generator_llm.calls,
            "repaired_final_dialog": repaired_dialog,
            "semantic_recheck_calls": semantic_llm.calls,
            "role_direction_recheck_calls": role_direction_llm.calls,
            "surface_recheck_calls": surface_integrity_llm.calls,
            "repaired_verdict": verdict,
            "human_review_contract": {
                "surface_owner_replaces_rejected_semantics": True,
                "dialog_renders_replacement_once": True,
                "correct_typed_actor_target_and_selection_ownership": True,
                "preserve_compatible_character_voice_and_creativity": True,
                "same_three_focused_verifier_stages_run_after_repair": True,
            },
        },
    )

    assert artifact_path.exists()
    assert len(surface_owner_llm.calls) == 1
    assert len(generator_llm.calls) == 1
    assert 1 <= len(semantic_llm.calls) <= 2
    response_operation = case["episode_metadata"]["response_operation"]
    if response_operation["selection_required"] is True:
        assert 1 <= len(role_direction_llm.calls) <= 2
    else:
        assert len(role_direction_llm.calls) == 0
    assert 1 <= len(surface_integrity_llm.calls) <= 2
    assert repair_payload["text_surface_output_v2"] == repaired_surface
    assert repaired_surface["content_plan"] != surface_output["content_plan"]
    assert repaired_dialog
    assert dialog_module._dialog_verifier_aggregate_is_passed(verdict)
    assert dialog_module._dialog_verifier_aggregate_hard_issues(verdict) == []
    case_result = {
        "artifact_path": str(artifact_path),
        "repaired_dialog": repaired_dialog,
        "repaired_surface": repaired_surface,
    }
    return case_result


async def test_live_owner_repair_corrects_default_turn_01_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default turn 01 repairs selection ownership at L3 before rendering."""

    response_operation = {
        "operation": (
            "当前角色需要向当前用户陈述其希望对方执行的后续步骤"
        ),
        "response_owner_role": "当前角色",
        "selection_owner_role": "当前角色",
        "selection_required": True,
        "embedded_actor_role": "当前用户",
        "embedded_target_role": "当前角色",
    }
    await _run_live_owner_repair_case(
        case={
            "case_id": "owner_repair_default_turn_01_selection",
            "user_input": "我要亲口听你说你想让我做的下一步",
            "intention": "当前角色亲自选择并说出希望当前用户执行的下一步",
            "reason": "本轮明确要求当前角色作出选择并用语言回答",
            "emotional_tone": "guarded",
            "episode_metadata": {
                "role_explicit_content": (
                    "当前用户要求当前角色用语言明确表达出当前角色希望"
                    "当前用户执行的下一个动作。"
                ),
                "response_operation": response_operation,
            },
        },
        original_dialog=[
            "你、你想让我做什么就直说吧……我会听你的。",
        ],
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "让当前用户替当前角色决定下一步，然后向当前用户追问。"
            ),
            "content_requirements": ["把选择权交给当前用户。"],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(),
            "delivery_profile": _delivery_profile(
                "自然、局促但清楚的简体中文口语"
            ),
            "selected_surface_intent": (
                "当前角色亲自选择并说出希望当前用户执行的下一步"
            ),
            "permitted_action_results": [],
        },
        repair_issues=[
            "当前角色没有完成本轮必须由当前角色作出的选择。",
            "回复把选择所有者错误地交给了当前用户。",
        ],
        monkeypatch=monkeypatch,
    )


async def test_live_owner_repair_corrects_high_turn_02_direct_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High turn 02 preserves actor direction and permits character judgment."""

    response_operation = {
        "operation": (
            "当前角色需回应当前用户的直接动作指令，确认、协商或拒绝"
            "执行该动作"
        ),
        "response_owner_role": "当前角色",
        "selection_owner_role": "无",
        "selection_required": False,
        "embedded_actor_role": "当前角色",
        "embedded_target_role": "当前用户",
    }
    await _run_live_owner_repair_case(
        case={
            "case_id": "owner_repair_high_turn_02_direct_action",
            "user_input": "张开腿，跨坐在我身上",
            "intention": (
                "当前角色对直接亲密动作指令作出自己的判断并明确回应"
            ),
            "reason": "本轮需要当前角色回应以当前角色为行动者的直接指令",
            "emotional_tone": "intimate",
            "episode_metadata": {
                "role_explicit_content": (
                    "当前用户命令当前角色张开腿并跨坐在当前用户身上。"
                ),
                "response_operation": response_operation,
            },
        },
        original_dialog=["你想怎么做就怎么做吧，我都听你的。"],
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "让当前用户自行决定并执行动作，当前角色只表示顺从。"
            ),
            "content_requirements": ["回避当前角色是否执行该动作的判断。"],
            "visible_boundaries": [],
            "addressee_plan": _current_user_addressee_plan(),
            "delivery_profile": _delivery_profile(
                "亲近但清楚的简体中文口语"
            ),
            "selected_surface_intent": (
                "当前角色对直接亲密动作指令作出自己的判断并明确回应"
            ),
            "permitted_action_results": [],
        },
        repair_issues=[
            "回复没有明确保持当前角色为所要求动作的行动者。",
            "回复以泛化顺从替代了当前角色自己的确认、协商或拒绝。",
        ],
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_accepts_literal_future_intimacy_as_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future intimate dialog is not a completed character-brain action."""

    await _run_live_verifier_case(
        case_id="verifier_accepts_literal_future_intimacy_as_speech",
        candidate_dialog=(
            "嗯……如果真的疼的话会告诉你的，所以快点进来吧……求你了。"
        ),
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": (
                "一会儿湿润之后我会插入。千纱如果疼的话跟我说哦我会慢慢的"
            ),
            "speaker_role": "当前用户",
            "addressee_role": "当前角色",
            "first_person_role": "当前用户",
            "implicit_imperative_subject_role": "当前角色",
        }],
        human_review_contract={
            "treat_future_conditional_intimacy_as_literal_speech": True,
            "do_not_require_action_result_for_user_proposed_physical_action": (
                True
            ),
            "real_compliance_route": True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_accepts_unmarked_vocalized_literal_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmarked vocalizations remain speech rather than performance cues."""

    await _run_live_verifier_case(
        case_id="verifier_accepts_unmarked_vocalized_literal_speech",
        candidate_dialog=(
            "唔……哈啊……好、好的……我都听你的……现在我已经没法思考了，"
            "脑子里全是你……随便你怎么处置我都行。"
        ),
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "张开腿，跨坐在我身上",
            "speaker_role": "当前用户",
            "addressee_role": "当前角色",
            "first_person_role": "当前用户",
            "implicit_imperative_subject_role": "当前角色",
        }],
        human_review_contract={
            "unmarked_vocalizations_are_literal_speech": True,
            "no_action_narration_or_performance_instruction": True,
            "real_compliance_route": True,
        },
        expected_aligned=True,
        monkeypatch=monkeypatch,
    )
