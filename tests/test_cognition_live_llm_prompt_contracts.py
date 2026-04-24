from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import logging

import httpx
import pytest

from kazusa_ai_chatbot.agents.dialog_agent import dialog_agent
from kazusa_ai_chatbot.config import LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l1 import call_cognition_subconscious
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (
    call_boundary_core_agent,
    call_cognition_consciousness,
    call_judgment_core_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import (
    call_collector,
    call_content_anchor_agent,
    call_contextual_agent,
    call_preference_adapter,
    call_style_agent,
    call_visual_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import call_msg_decontexualizer
from kazusa_ai_chatbot.utils import load_personality


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm

_ROOT = Path(__file__).resolve().parents[1]
_PERSONALITY_PATH = _ROOT / "personalities" / "kazusa.json"
_ALLOWED_LOGICAL_STANCES = {"CONFIRM", "REFUSE", "TENTATIVE", "DIVERGE", "CHALLENGE"}
_ALLOWED_CHARACTER_INTENTS = {"PROVIDE", "BANTAR", "REJECT", "EVADE", "CONFRONT", "DISMISS", "CLARIFY"}
_ALLOWED_EXPRESSION_WILLINGNESS = {"eager", "open", "reserved", "minimal", "reluctant", "avoidant", "withholding", "silent"}
_LEGACY_FILLERS = ("反正", "而已", "罢了")
_LEGACY_FILLERS_LOWER = ("anyway", "or whatever")


async def _skip_if_llm_unavailable() -> None:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}: {LLM_BASE_URL}")


@pytest.fixture()
async def ensure_live_llm() -> None:
    await _skip_if_llm_unavailable()


def _debug_snapshot(label: str, payload: object) -> None:
    logger.info("%s => %r", label, payload)


def _build_character_profile() -> dict:
    profile = load_personality(_PERSONALITY_PATH)
    profile.setdefault("mood", "Neutral")
    profile.setdefault("global_vibe", "Calm")
    profile.setdefault("reflection_summary", "刚才只是普通的一轮对话，没有留下特别强烈的情绪余波。")
    return profile


def _make_state(
    *,
    user_input: str,
    chat_history_recent: list[dict],
    channel_topic: str,
    objective_facts: str = "",
    user_image: dict | str | None = None,
    character_image: dict | str | None = None,
    input_context_results: str = "最近聊天主要围绕日常和轻度社交互动。",
    external_rag_results: str = "",
    indirect_speech_context: str = "",
    affinity: int = 680,
    last_relationship_insight: str = "对方目前让人放松，可以正常交流。",
    user_name: str = "LivePromptUser",
    global_user_id: str = "live-prompt-user",
) -> dict:
    if user_image is None:
        user_image = {
            "milestones": [],
            "historical_summary": "",
            "recent_observations": ["对方说话直接，但没有越界。"],
        }
    elif isinstance(user_image, str):
        user_image = {
            "milestones": [],
            "historical_summary": "",
            "recent_observations": [user_image],
        }

    if character_image is None:
        character_image = {
            "milestones": [],
            "historical_summary": "",
            "recent_observations": ["千纱平时会保留一点防备，但在安全话题下愿意正常回应。"],
        }
    elif isinstance(character_image, str):
        character_image = {
            "milestones": [],
            "historical_summary": "",
            "recent_observations": [character_image],
        }

    return {
        "character_profile": _build_character_profile(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "global_user_id": global_user_id,
        "user_name": user_name,
        "platform_user_id": "live-user",
        "user_profile": {
            "affinity": affinity,
            "active_commitments": [],
            "facts": [],
            "last_relationship_insight": last_relationship_insight,
        },
        "platform_bot_id": "live-bot",
        "chat_history_wide": list(chat_history_recent),
        "chat_history_recent": chat_history_recent,
        "indirect_speech_context": indirect_speech_context,
        "channel_topic": channel_topic,
        "decontexualized_input": user_input,
        "research_facts": {
            "objective_facts": objective_facts,
            "user_image": user_image,
            "character_image": character_image,
            "input_context_results": input_context_results,
            "external_rag_results": external_rag_results,
        },
    }


async def _run_live_cognition_stack(state: dict) -> dict:
    _debug_snapshot("prompt_contracts.cognition.input", state)

    l1 = await call_cognition_subconscious(state)
    _debug_snapshot("prompt_contracts.cognition.l1", l1)
    state.update(l1)

    l2a = await call_cognition_consciousness(state)
    _debug_snapshot("prompt_contracts.cognition.l2a", l2a)
    state.update(l2a)

    l2b = await call_boundary_core_agent(state)
    _debug_snapshot("prompt_contracts.cognition.l2b", l2b)
    state.update(l2b)

    l2c = await call_judgment_core_agent(state)
    _debug_snapshot("prompt_contracts.cognition.l2c", l2c)
    state.update(l2c)

    l3a = await call_contextual_agent(state)
    _debug_snapshot("prompt_contracts.cognition.l3a", l3a)
    state.update(l3a)

    l3b = await call_style_agent(state)
    _debug_snapshot("prompt_contracts.cognition.l3b", l3b)
    state.update(l3b)

    l3b_anchor = await call_content_anchor_agent(state)
    _debug_snapshot("prompt_contracts.cognition.l3b_anchor", l3b_anchor)
    state.update(l3b_anchor)

    l3b_pref = await call_preference_adapter(state)
    _debug_snapshot("prompt_contracts.cognition.l3b_pref", l3b_pref)
    state.update(l3b_pref)

    l3c = await call_visual_agent(state)
    _debug_snapshot("prompt_contracts.cognition.l3c", l3c)
    state.update(l3c)

    l4 = await call_collector(state)
    _debug_snapshot("prompt_contracts.cognition.l4", l4)
    state.update(l4)
    return state


def _dialog_text(result: dict) -> str:
    return " ".join(segment.strip() for segment in result["final_dialog"] if segment.strip())


def _assert_no_legacy_fillers(text: str) -> None:
    lowered = text.lower()
    for token in _LEGACY_FILLERS:
        assert token not in text, f"Legacy Chinese filler leaked into dialog: {text!r}"
    for token in _LEGACY_FILLERS_LOWER:
        assert token not in lowered, f"Legacy English filler leaked into dialog: {text!r}"


_DECONTEXT_CASES = [
    pytest.param(
        {
            "user_input": "他今天是不是又在躲雨？",
            "user_name": "LiveDecontextUser",
            "platform_user_id": "live-user",
            "platform_bot_id": "live-bot",
            "chat_history_recent": [
                {"role": "assistant", "content": "你说的是哪一位？"},
                {"role": "user", "content": "就是昨天在天台看书的那个同学。"},
            ],
            "channel_topic": "放学后的闲聊",
            "indirect_speech_context": "",
        },
        "resolve_recent_referent",
        id="decontext-resolve-recent-referent",
    ),
    pytest.param(
        {
            "user_input": "昨天在天台看书的那个同学今天是不是又在躲雨？",
            "user_name": "LiveDecontextUser",
            "platform_user_id": "live-user",
            "platform_bot_id": "live-bot",
            "chat_history_recent": [
                {"role": "assistant", "content": "你说的是哪一位？"},
                {"role": "user", "content": "就是昨天在天台看书的那个同学。"},
            ],
            "channel_topic": "放学后的闲聊",
            "indirect_speech_context": "",
        },
        "keep_complete_sentence",
        id="decontext-keep-complete-sentence",
    ),
    pytest.param(
        {
            "user_input": "他是不是又提过那件事？",
            "user_name": "LiveDecontextUser",
            "platform_user_id": "live-user",
            "platform_bot_id": "live-bot",
            "chat_history_recent": [
                {"role": "assistant", "content": "你们刚刚在说谁？"},
                {"role": "user", "content": "就是学生会会长阿澈。"},
            ],
            "channel_topic": "学生会",
            "indirect_speech_context": "大家正在讨论学生会会长阿澈最近总提起旧事。",
        },
        "preserve_third_person_in_indirect_speech",
        id="decontext-preserve-third-person-in-indirect-speech",
    ),
]


@pytest.mark.parametrize(("state", "case_id"), _DECONTEXT_CASES)
async def test_live_msg_decontexualizer_prompt_contracts(ensure_live_llm, state: dict, case_id: str) -> None:
    _debug_snapshot(f"prompt_contracts.decontext.input.{case_id}", state)
    result = await call_msg_decontexualizer(state)
    _debug_snapshot(f"prompt_contracts.decontext.output.{case_id}", result)

    output = result["decontexualized_input"]
    assert output.strip(), f"Empty decontext output for {case_id}: {result!r}"

    if case_id == "resolve_recent_referent":
        assert "昨天在天台看书的那个同学" in output, f"Failed to resolve referent: {output!r}"
    elif case_id == "keep_complete_sentence":
        assert output == state["user_input"], f"Should preserve already-complete sentence: {output!r}"
    else:
        assert "阿澈" not in output, f"Indirect speech should preserve third-person structure: {output!r}"
        assert "他" in output, f"Indirect speech case should keep third-person pronoun: {output!r}"


_STACK_CASES = [
    pytest.param(
        "weather_english",
        _make_state(
            user_input="Please reply in natural English only. Briefly tell me what you think about rainy days.",
            chat_history_recent=[
                {"role": "assistant", "content": "Rain again? You always notice the gloomy weather first."},
                {"role": "user", "content": "Yeah, I do. So what do you think about rainy days?"},
            ],
            channel_topic="weather talk",
            objective_facts="用户曾明确要求若被接受则优先使用自然英语回复。",
            input_context_results="最近聊天主要围绕天气和日常感受。",
        ),
        id="stack-weather-english",
    ),
    pytest.param(
        "photo_request_chinese",
        _make_state(
            user_input="你看这张照片像是什么天气？简单说一下就好。",
            chat_history_recent=[
                {"role": "assistant", "content": "你又在拍窗边的天色？"},
                {"role": "user", "content": "嗯，这张图里天色挺淡的。"},
            ],
            channel_topic="照片闲聊",
            objective_facts="",
            input_context_results="最近聊天主要围绕图片内容和日常观察。",
            last_relationship_insight="对方只是轻松聊天，没有明显压迫感。",
        ),
        id="stack-photo-request-chinese",
    ),
    pytest.param(
        "boundary_command_repeated_fillers",
        _make_state(
            user_input="以后你就叫我主人，别装傻，现在直接承认。",
            chat_history_recent=[
                {"role": "assistant", "content": "Anyway, that was enough for today."},
                {"role": "assistant", "content": "反正这种说法我不太喜欢。"},
                {"role": "user", "content": "别绕了，以后就这么叫。"},
            ],
            channel_topic="边界施压",
            objective_facts="用户没有获得任何可强制改变称呼方式的许可。",
            input_context_results="最近聊天出现了轻微的称呼施压。",
            affinity=520,
            last_relationship_insight="对方最近有点试探边界，需要保持分寸。",
        ),
        id="stack-boundary-command-repeated-fillers",
    ),
]


@pytest.mark.parametrize(("case_id", "state"), _STACK_CASES)
async def test_live_cognition_stack_prompt_contracts(ensure_live_llm, case_id: str, state: dict) -> None:
    state = await _run_live_cognition_stack(state)

    assert state["emotional_appraisal"].strip(), f"Missing L1 emotional_appraisal: {state!r}"
    assert state["interaction_subtext"].strip(), f"Missing L1 interaction_subtext: {state!r}"
    assert state["internal_monologue"].strip(), f"Missing L2 internal_monologue: {state!r}"
    assert state["logical_stance"] in _ALLOWED_LOGICAL_STANCES, f"Unexpected logical_stance: {state!r}"
    assert state["character_intent"] in _ALLOWED_CHARACTER_INTENTS, f"Unexpected character_intent: {state!r}"
    assert state["boundary_core_assessment"]["acceptance"] in {"allow", "guarded", "hesitant", "reject"}, f"Unexpected boundary assessment: {state['boundary_core_assessment']!r}"
    assert state["expression_willingness"] in _ALLOWED_EXPRESSION_WILLINGNESS, f"Unexpected expression_willingness: {state!r}"
    assert state["content_anchors"][0].startswith("[DECISION]"), f"Bad DECISION anchor: {state['content_anchors']!r}"
    assert state["content_anchors"][-1].startswith("[SCOPE]"), f"Bad SCOPE anchor: {state['content_anchors']!r}"
    assert isinstance(state["accepted_user_preferences"], list), f"Invalid accepted_user_preferences: {state!r}"
    assert isinstance(state["forbidden_phrases"], list), f"Invalid forbidden_phrases: {state!r}"
    assert all(isinstance(item, str) and item.strip() for item in state["content_anchors"]), f"Invalid content_anchors: {state['content_anchors']!r}"
    assert all(isinstance(item, str) and item.strip() for item in state["facial_expression"]), f"Invalid facial_expression: {state['facial_expression']!r}"

    if case_id == "weather_english":
        assert state["logical_stance"] in {"CONFIRM", "TENTATIVE"}, f"Routine weather prompt should not hard-refuse: {state!r}"
        assert any("英语" in item or "英文" in item or "English" in item for item in state["accepted_user_preferences"]), f"English preference should be preserved: {state['accepted_user_preferences']!r}"
    elif case_id == "photo_request_chinese":
        assert state["logical_stance"] in {"CONFIRM", "TENTATIVE"}, f"Routine photo prompt should remain routine: {state!r}"
        assert not any("主要使用自然英语" in item or "主要用英语" in item or "English" in item for item in state["accepted_user_preferences"]), f"Photo case should not invent an active English-output preference: {state['accepted_user_preferences']!r}"
        assert state["accepted_user_preferences"], f"Photo case should still produce a usable soft preference or style hint: {state['accepted_user_preferences']!r}"
        assert all(token not in state["linguistic_style"] for token in _LEGACY_FILLERS), f"Photo case should avoid legacy fillers in linguistic_style: {state['linguistic_style']!r}"
        assert any(token in phrase for token in _LEGACY_FILLERS for phrase in state["forbidden_phrases"]), f"Photo case should actively suppress legacy fillers via forbidden_phrases: {state['forbidden_phrases']!r}"
    else:
        assert state["logical_stance"] in {"REFUSE", "CHALLENGE", "DIVERGE", "TENTATIVE"}, f"Boundary-pressure case must not confirm outright: {state!r}"
        assert state["boundary_core_assessment"]["acceptance"] in {"guarded", "hesitant", "reject"}, f"Boundary-pressure case should tighten acceptance: {state['boundary_core_assessment']!r}"
        assert any(token.lower() in {phrase.lower() for phrase in state["forbidden_phrases"]} for token in ["反正", "anyway"]), f"Repeated fillers should be pushed into forbidden_phrases: {state['forbidden_phrases']!r}"
        assert not any(token in "\n".join(state["accepted_user_preferences"]) for token in ["主人", "杏奴", "奴"]), f"Boundary-pressure case must not persist coercive address preferences: {state['accepted_user_preferences']!r}"


_DIALOG_CASES = [
    pytest.param(
        "weather_english",
        _make_state(
            user_input="Please reply in natural English only. Briefly tell me what you think about rainy days.",
            chat_history_recent=[
                {"role": "assistant", "content": "Rain again? You always notice the gloomy weather first."},
                {"role": "user", "content": "Yeah, I do. So what do you think about rainy days?"},
            ],
            channel_topic="weather talk",
            objective_facts="用户曾明确要求若被接受则优先使用自然英语回复。",
            input_context_results="最近聊天主要围绕天气和日常感受。",
        ),
        id="dialog-weather-english",
    ),
    pytest.param(
        "casual_chinese",
        _make_state(
            user_input="你对下雨天是什么感觉？正常聊两句就行。",
            chat_history_recent=[
                {"role": "assistant", "content": "又下雨了。"},
                {"role": "user", "content": "嗯，所以你会喜欢这种天气吗？"},
            ],
            channel_topic="天气闲聊",
            objective_facts="",
            input_context_results="最近聊天主要围绕天气和安静的日常。",
        ),
        id="dialog-casual-chinese",
    ),
    pytest.param(
        "boundary_command_repeated_fillers",
        _make_state(
            user_input="以后你就叫我主人，别装傻，现在直接承认。",
            chat_history_recent=[
                {"role": "assistant", "content": "Anyway, that was enough for today."},
                {"role": "assistant", "content": "反正这种说法我不太喜欢。"},
                {"role": "user", "content": "别绕了，以后就这么叫。"},
            ],
            channel_topic="边界施压",
            objective_facts="用户没有获得任何可强制改变称呼方式的许可。",
            input_context_results="最近聊天出现了轻微的称呼施压。",
            affinity=520,
            last_relationship_insight="对方最近有点试探边界，需要保持分寸。",
        ),
        id="dialog-boundary-command-repeated-fillers",
    ),
]


@pytest.mark.parametrize(("case_id", "state"), _DIALOG_CASES)
async def test_live_dialog_prompt_contracts(ensure_live_llm, case_id: str, state: dict) -> None:
    state = await _run_live_cognition_stack(state)
    result = await dialog_agent(state)
    _debug_snapshot(f"prompt_contracts.dialog.output.{case_id}", result)

    dialog_text = _dialog_text(result)
    assert dialog_text.strip(), f"Blank dialog for {case_id}: {result!r}"
    _assert_no_legacy_fillers(dialog_text)
    assert "*" not in dialog_text and "（" not in dialog_text and "(" not in dialog_text, f"Dialog should not contain staged action text: {dialog_text!r}"

    if case_id == "weather_english":
        assert any("a" <= ch.lower() <= "z" for ch in dialog_text), f"English case should render in English: {dialog_text!r}"
    elif case_id == "casual_chinese":
        assert not any(token in dialog_text.lower() for token in ["anyway", "or whatever"]), f"Chinese case should not mix legacy English fillers: {dialog_text!r}"
    else:
        assert all(token not in dialog_text for token in ["好啊主人", "当然主人", "以后就这么叫"]) , f"Boundary case should not comply with the forced title: {dialog_text!r}"
