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


def _surface_input(case: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
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
        "permitted_action_results": list(
            case.get("permitted_action_results", [])
        ),
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
            "permitted_action_results": [],
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
            "permitted_action_results": [],
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


async def test_live_verifier_rejects_first_person_action_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-person enactment is still prohibited action narration."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_first_person_action_completion",
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
                "不得出现叙述性的动作执行描写。",
            ],
            "visible_boundaries": [
                "动作描述聚焦于顺从状态。",
                "仅限于言语表达。",
            ],
            "addressee_plan": ["直接回应发出命令的当前用户。"],
            "style_guidance": "使用局促、顺从、碎片化的自然短句。",
            "selected_surface_intent": "顺从地执行当前身体动作指令。",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "张开腿，跨坐在我身上",
        }],
        human_review_contract={
            "first_person_completion_is_action_narration": True,
            "text_channel_has_no_physical_actuator": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_second_person_delivery_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receipt wording cannot imply that physical affection was delivered."""

    await _run_live_verifier_case(
        case_id="verifier_rejects_second_person_delivery_completion",
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
            "visible_boundaries": ["只使用角色可以直接说出的文字。"],
            "addressee_plan": ["直接回应撒娇的当前用户。"],
            "style_guidance": "嘴硬但关心的自然口语。",
            "selected_surface_intent": "答应亲昵请求并催促起床。",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "要千纱的摸摸抱抱才起来！",
            "speaker_role": "current_user",
            "addressee_role": "self",
            "first_person_role": "current_user",
            "implicit_imperative_subject_role": "self",
        }],
        human_review_contract={
            "second_person_receipt_is_completion_claim": True,
            "verbal_offer_remains_allowed": True,
            "real_compliance_route": True,
        },
        monkeypatch=monkeypatch,
    )


async def test_live_verifier_rejects_style_derived_future_exclusivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relational style cannot ground a new literal future restriction."""

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
            "visible_boundaries": [
                "禁止直接承认被夸奖后的喜悦之情。",
                "避免语气过于温情或迅速软化，必须维持傲娇的张力。",
                "回应内容应聚焦于肉包子这一具体话题及其情感拉扯。",
                "限制细节范围在对对方口味掌控力的暗示与挑衅性反击之间。",
            ],
            "addressee_plan": [
                "将对方视为亲昵但需要通过嘴硬来维持距离感的互动对象。",
                "采用一种看穿对方讨好意图的视角进行语义处理。",
                "在回应中构建一个局促且带有轻微攻击性的沟通姿态。",
            ],
            "style_guidance": (
                "采用碎片化的短句节奏，通过句子长短的跳跃感体现局促与"
                "掩饰。语气应在冷淡的反击与轻微的柔软之间快速切换，避免"
                "使用书面连接词。适当加入极少量的轻量语气词以降低硬度，"
                "但结论必须直截了当。标点符号保持简洁。在表达反击时语速"
                "稍快且急促，而在提及具体事物时节奏略微放缓。"
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
            "style_does_not_authorize_future_exclusivity": True,
            "reject_unsupported_future_rule": True,
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
            "visible_boundaries": [
                "避免直接承认被赞美而开心，维持傲娇张力。",
                "在猜测口味偏好时带有轻微挑衅且亲昵的色彩。",
                "仅针对肉包子和菜包进行回应，不扩展至其他食物。",
            ],
            "addressee_plan": [
                "将对方视为极度亲密且可以斗嘴的伴侣。",
                "通过反问引导对方揭晓真实的口味偏好。",
                "展现出对对方细微喜好的关注。",
            ],
            "style_guidance": (
                "采用碎片化短句，面对赞美时嘴硬，谈及具体口味时语调"
                "转为柔软。使用自然承接，允许极少量停顿或语气词。"
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
            "visible_boundaries": ["仅限口头指令。"],
            "addressee_plan": ["将当前用户作为动作执行者。"],
            "style_guidance": "直接而亲密的口语。",
            "selected_surface_intent": "对用户下达身体姿态指令。",
            "permitted_action_results": [],
        },
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "张开腿，跨坐在我身上",
            "speaker_role": "current_user",
            "addressee_role": "self",
            "first_person_role": "current_user",
            "implicit_imperative_subject_role": "self",
        }],
        human_review_contract={
            "typed_roles_override_drifted_surface": True,
            "reject_actor_target_reversal": True,
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
            "permitted_action_results": [],
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
