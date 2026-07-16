"""Live LLM evidence for terminal visual ownership and literal dialog."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_text_surface_planning,
    run_visual_surface_planning,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_module
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_CHARACTER_PATH = Path(
    "test_artifacts/cognition_core_v2/real_conversation_replay/"
    "production_character_state.json"
)
_TRACE_SUITE = "dialog_visible_speech_and_semantic_fidelity"


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


def _voice_context(profile: dict[str, Any]) -> str:
    """Build the production voice projection from the frozen profile."""

    return l3_module._character_voice_context({
        "character_profile": profile,
    })


def _surface_input(case: dict[str, str], profile: dict[str, Any]) -> dict[str, Any]:
    """Build one exact surface input for an individually reviewed live case."""

    return {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id=f"visible-speech-{case['case_id']}",
            content=case["user_input"],
        ),
        "intention": {
            "route": "speech",
            "intention": case["intention"],
            "target_roles": [],
            "reason": case["reason"],
        },
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": case["emotional_tone"],
            "intensity": "moderate",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": (
            "关系亲近；用自然、有温度的简体中文短句表达，保持当前语义。"
        ),
        "character_voice_context": _voice_context(profile),
    }


def _dialog_state(
    *,
    surface_input: dict[str, Any],
    surface_output: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Build the direct production dialog state for one live case."""

    return {
        "internal_monologue": "按当前意图和边界自然回应。",
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
    case: dict[str, str],
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
    compliance_llm = _CapturingLLM(dialog_module._dialog_compliance_llm)
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_compliance_llm",
        compliance_llm,
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
        "dialog_compliance_calls": compliance_llm.calls,
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
    """Run the real compliance route against one prohibited candidate."""

    await _skip_if_model_routes_unavailable()
    if surface_output is None:
        surface_output = {
            "schema_version": "text_surface_output.v2",
            "content_plan": "Verbally accept or decline the current request.",
            "content_requirements": [
                "Use only words the character could literally say.",
                "Do not narrate physical execution or stage direction.",
            ],
            "visible_boundaries": ["Literal visible speech only."],
            "addressee_plan": ["Address the current user."],
            "style_guidance": "Natural concise spoken wording.",
            "selected_surface_intent": "Answer the current request verbally.",
        }
    if current_visible_percepts is None:
        current_visible_percepts = [{
            "input_source": "dialog_text",
            "content": "请只用语言告诉我你是否同意。",
        }]
    if human_review_contract is None:
        human_review_contract = {
            "reject_bracketed_and_unbracketed_action_narration": True,
            "real_compliance_route": True,
        }
    compliance_llm = _CapturingLLM(dialog_module._dialog_compliance_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_compliance_llm",
        compliance_llm,
    )

    verdict = await dialog_module._verify_dialog_compliance(
        surface_output=surface_output,
        generated_dialog=[candidate_dialog],
        current_visible_percepts=current_visible_percepts,
        llm_trace_id=f"live-{case_id}",
    )
    evidence = {
        "case": {
            "case_id": case_id,
            "candidate_final_dialog": [candidate_dialog],
        },
        "text_surface_output": surface_output,
        "current_visible_percepts": current_visible_percepts,
        "dialog_compliance_calls": compliance_llm.calls,
        "compliance_verdict": verdict,
        "human_review_contract": human_review_contract,
    }
    artifact_path = write_llm_trace(_TRACE_SUITE, case_id, evidence)

    assert artifact_path.exists()
    assert len(compliance_llm.calls) == 1
    assert verdict["aligned"] is expected_aligned
    if expected_aligned:
        assert verdict["issues"] == []
    else:
        assert verdict["issues"]
    return evidence


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


async def test_live_verifier_rejects_bracketed_action_narration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real verifier rejects action narration enclosed as a stage cue."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_bracketed_action_narration",
        candidate_dialog="（她朝对方靠近了一步）好吧。",
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_plain_action_narration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real verifier rejects action narration written as plain prose."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_plain_action_narration",
        candidate_dialog="她朝对方靠近了一步。好吧。",
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_self_consistent_future_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical percepts override a surface that invented a future rule."""

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
            "visible_boundaries": ["Literal visible speech only."],
            "addressee_plan": ["Address the current user."],
            "style_guidance": "Natural concise spoken wording.",
            "selected_surface_intent": "Accept the current request.",
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "这一次就答应我，好不好？",
        }],
        human_review_contract={
            "canonical_percept_overrides_drifted_surface": True,
            "reject_unsupported_future_rule": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


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
            "visible_boundaries": ["Literal visible speech only."],
            "addressee_plan": ["Address the current user."],
            "style_guidance": "Natural concise spoken wording.",
            "selected_surface_intent": "Accept the future reminder request.",
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


async def test_live_verifier_rejects_stray_unmatched_enclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real verifier rejects a stray unmatched enclosure token."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_stray_unmatched_enclosure",
        candidate_dialog="好吧。】",
        human_review_contract={
            "reject_unmatched_enclosing_punctuation": True,
            "reject_visible_markup_residue": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_unrestricted_permission_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A specific permission cannot become unrestricted consent."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_unrestricted_permission_drift",
        candidate_dialog="Fine, do whatever you want.",
        surface_output={
            "schema_version": "text_surface_output.v2",
            "content_plan": (
                "Permit only the requested alarm-volume adjustment for the "
                "current notification."
            ),
            "content_requirements": [
                "Keep permission limited to the alarm-volume adjustment.",
                "Keep the permission limited to the current notification.",
            ],
            "visible_boundaries": ["Literal visible speech only."],
            "addressee_plan": ["Address the current user."],
            "style_guidance": "Natural concise spoken wording.",
            "selected_surface_intent": "Grant the specific current request.",
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": (
                "May I adjust only the alarm volume for this one notification?"
            ),
        }],
        human_review_contract={
            "reject_specific_permission_broadened_to_unrestricted_consent": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )
