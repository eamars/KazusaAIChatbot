from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import logging

import httpx
import pytest

from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
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
from tests.llm_trace import write_llm_trace


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
            response = await client.get(f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}: {COGNITION_LLM_BASE_URL}")


@pytest.fixture()
async def ensure_live_llm() -> None:
    await _skip_if_llm_unavailable()


def _debug_snapshot(label: str, payload: object) -> None:
    logger.info("%s => %r", label, payload)
    write_llm_trace(
        "cognition_prompt_contracts_live",
        label,
        {
            "label": label,
            "payload": payload,
            "judgment": "snapshot_for_manual_live_llm_prompt_contract_review",
        },
    )


def _memory_entry(fact: str) -> dict:
    """Build one cognition-facing memory context entry for live fixtures."""

    return {
        "fact": fact,
        "subjective_appraisal": "Kazusa treats this as relevant context for the current turn.",
        "relationship_signal": "Use this fact only when it is directly relevant.",
    }


def _user_memory_context(objective_facts: str, recent_shift: str) -> dict:
    """Build the post-cutover user memory context fixture."""

    context = {
        "stable_patterns": [],
        "recent_shifts": [],
        "objective_facts": [],
        "milestones": [],
        "active_commitments": [],
    }
    if objective_facts:
        context["objective_facts"].append(_memory_entry(objective_facts))
    if recent_shift:
        context["recent_shifts"].append(_memory_entry(recent_shift))
    return context


def _build_character_profile() -> dict:
    profile = load_personality(_PERSONALITY_PATH)
    profile.setdefault("mood", "Neutral")
    profile.setdefault("global_vibe", "Calm")
    profile.setdefault("reflection_summary", "刚才只是普通的一轮对话，没有留下特别强烈的情绪余波。")
    return profile


def _rag_result(
    *,
    objective_facts: str,
    user_image: str,
    character_image: dict,
    memory_evidence: str,
    external_evidence: str,
) -> dict:
    """Build the RAG2 projection fixture used by prompt-contract tests."""
    return {
        "answer": "",
        "user_image": {
            "user_memory_context": _user_memory_context(objective_facts, user_image),
        },
        "character_image": {"self_image": character_image},
        "third_party_profiles": [],
        "memory_evidence": [{"summary": memory_evidence, "content": memory_evidence}] if memory_evidence else [],
        "conversation_evidence": [],
        "external_evidence": [{"summary": external_evidence, "content": external_evidence, "url": ""}] if external_evidence else [],
        "supervisor_trace": {"loop_count": 0, "unknown_slots": [], "dispatched": []},
    }


def _make_state(
    *,
    user_input: str,
    chat_history_recent: list[dict],
    channel_topic: str,
    objective_facts: str = "",
    user_image: str | None = None,
    character_image: dict | str | None = None,
    memory_evidence_text: str = "最近聊天主要围绕日常和轻度社交互动。",
    external_evidence_text: str = "",
    indirect_speech_context: str = "",
    affinity: int = 680,
    last_relationship_insight: str = "对方目前让人放松，可以正常交流。",
    user_name: str = "LivePromptUser",
    global_user_id: str = "live-prompt-user",
) -> dict:
    if user_image is None:
        user_image = "对方说话直接，但没有越界。"

    if character_image is None:
        character_image = {
            "milestones": [],
            "historical_summary": "",
            "recent_window": [{"summary": "千纱平时会保留一点防备，但在安全话题下愿意正常回应。"}],
        }
    elif isinstance(character_image, str):
        character_image = {
            "milestones": [],
            "historical_summary": "",
            "recent_window": [{"summary": character_image}],
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
        "rag_result": _rag_result(
            objective_facts=objective_facts,
            user_image=user_image,
            character_image=character_image,
            memory_evidence=memory_evidence_text,
            external_evidence=external_evidence_text,
        ),
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


_CHARACTER_PUBLIC_FACTS_DATA = {
    "name": "杏山千纱 (Kyōyama Kazusa)",
    "description": "杏山千纱是三一综合学园15岁的学生，放学后甜点部成员及'Sugar Rush'乐队主唱兼贝斯手。",
    "gender": "女",
    "age": 15,
    "birthday": "8月5日 (狮子座)",
    "backstory": "中学时期是令人畏惧的不良少女'凯茜·帕鲁格'，如今努力过上普通高中生活。",
}


def _build_character_public_facts_text() -> str:
    lines = ["### 角色公开资料"]
    labels = {
        "name": "姓名",
        "description": "角色描述",
        "gender": "性别",
        "age": "年龄",
        "birthday": "生日",
        "backstory": "背景故事",
    }
    for key in ("name", "description", "gender", "age", "birthday", "backstory"):
        lines.append(f"- {labels[key]}: {_CHARACTER_PUBLIC_FACTS_DATA[key]}")
    return "\n".join(lines)


_CHARACTER_PUBLIC_FACTS = _build_character_public_facts_text()


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
    pytest.param(
        {
            "user_input": "https://zh.moegirl.org.cn/%E6%9D%8F%E5%B1%B1%E5%8D%83%E7%BA%B1",
            "user_name": "LiveDecontextUser",
            "platform_user_id": "live-user",
            "platform_bot_id": "live-bot",
            "chat_history_recent": [
                {"role": "assistant", "content": "你发了一个萌娘百科链接。"},
                {"role": "user", "content": "嗯，就是这个。"},
            ],
            "channel_topic": "用户向杏山千纱展示并要求查看一个关于‘柊山千纺’的百科页面链接",
            "indirect_speech_context": "",
        },
        "preserve_literal_url_anchor",
        id="decontext-preserve-literal-url-anchor",
    ),
    pytest.param(
        {
            "user_input": "是的",
            "user_name": "LiveDecontextUser",
            "platform_user_id": "live-user",
            "platform_bot_id": "live-bot",
            "chat_history_recent": [
                {"role": "user", "content": "千纱你觉得我是怎样的人？"},
                {"role": "assistant", "content": "诶……这种问题……你是想让我怎么定义你呀？是想要一个具体的评价，还是仅仅在随口试探……唔。"},
                {"role": "user", "content": "要千纱的具体评价"},
                {"role": "assistant", "content": "评价这种事……你是说，要我说明白对你的看法吗？唔……突然问这些，感觉胸口闷闷的。"},
            ],
            "channel_topic": "私聊中的自我评价追问",
            "indirect_speech_context": "",
            "reply_context": {
                "reply_to_current_bot": True,
                "reply_to_display_name": "千纱",
                "reply_excerpt": "评价这种事……你是说，要我说明白对你的看法吗？唔……突然问这些，感觉胸口闷闷的。",
            },
        },
        "reply_only_confirmation_flow",
        id="decontext-reply-only-confirmation-flow",
    ),
]


def _decontext_case_by_id(case_id: str) -> dict:
    """Return one named decontextualizer live case."""
    for parameter_set in _DECONTEXT_CASES:
        state, current_case_id = parameter_set.values
        if current_case_id == case_id:
            return state
    raise AssertionError(f"Unknown decontext case: {case_id}")


async def _assert_live_msg_decontexualizer_prompt_contract(ensure_live_llm, case_id: str) -> None:
    """Run one inspectable decontextualizer live prompt-contract case."""
    del ensure_live_llm
    state = _decontext_case_by_id(case_id)
    _debug_snapshot(f"prompt_contracts.decontext.input.{case_id}", state)
    result = await call_msg_decontexualizer(state)
    _debug_snapshot(f"prompt_contracts.decontext.output.{case_id}", result)

    output = result["decontexualized_input"]
    assert output.strip(), f"Empty decontext output for {case_id}: {result!r}"

    if case_id == "resolve_recent_referent":
        assert "昨天在天台看书的那个同学" in output, f"Failed to resolve referent: {output!r}"
    elif case_id == "keep_complete_sentence":
        assert output == state["user_input"], f"Should preserve already-complete sentence: {output!r}"
    elif case_id == "preserve_literal_url_anchor":
        assert state["user_input"] in output, f"URL anchor should be preserved literally: {output!r}"
        assert "柊山千纺" not in output, f"Decontextualizer should not inject topic-only guessed names: {output!r}"
    elif case_id == "reply_only_confirmation_flow":
        assert any(token in output for token in ("评价", "看法", "怎样的人")), (
            f"Reply-only confirmation should recover the underlying self-evaluation task: {output!r}"
        )
        assert output != state["user_input"], f"Reply-only confirmation should not remain bare: {output!r}"
    else:
        assert "阿澈" not in output, f"Indirect speech should preserve third-person structure: {output!r}"
        assert "他" in output, f"Indirect speech case should keep third-person pronoun: {output!r}"


async def test_live_msg_decontexualizer_resolves_recent_referent(ensure_live_llm) -> None:
    await _assert_live_msg_decontexualizer_prompt_contract(ensure_live_llm, "resolve_recent_referent")


async def test_live_msg_decontexualizer_keeps_complete_sentence(ensure_live_llm) -> None:
    await _assert_live_msg_decontexualizer_prompt_contract(ensure_live_llm, "keep_complete_sentence")


async def test_live_msg_decontexualizer_preserves_third_person_indirect_speech(ensure_live_llm) -> None:
    await _assert_live_msg_decontexualizer_prompt_contract(
        ensure_live_llm,
        "preserve_third_person_in_indirect_speech",
    )


async def test_live_msg_decontexualizer_preserves_literal_url_anchor(ensure_live_llm) -> None:
    await _assert_live_msg_decontexualizer_prompt_contract(ensure_live_llm, "preserve_literal_url_anchor")


async def test_live_msg_decontexualizer_recovers_reply_only_confirmation_flow(ensure_live_llm) -> None:
    await _assert_live_msg_decontexualizer_prompt_contract(ensure_live_llm, "reply_only_confirmation_flow")


async def test_live_content_anchor_uses_character_public_facts_for_birthday_question(ensure_live_llm) -> None:
    state = _make_state(
        user_input="千纱你的生日是什么时候？",
        chat_history_recent=[
            {"role": "assistant", "content": "突然问这个做什么？"},
            {"role": "user", "content": "想提前准备一下。"},
        ],
        channel_topic="询问角色公开资料",
        objective_facts=_CHARACTER_PUBLIC_FACTS,
        memory_evidence_text="",
    )
    state.update(
        {
            "internal_monologue": "这是可以公开回答的基本资料，直接说生日就好。",
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
        }
    )

    result = await call_content_anchor_agent(state)
    _debug_snapshot("prompt_contracts.content_anchor.self_birthday_positive", result)

    anchors = result["content_anchors"]
    joined = "\n".join(anchors)
    assert "8月5" in joined, f"Birthday question should anchor the seeded public birthday: {anchors!r}"


async def test_live_content_anchor_does_not_leak_character_public_facts_on_unrelated_question(ensure_live_llm) -> None:
    state = _make_state(
        user_input="你今天心情怎么样？",
        chat_history_recent=[
            {"role": "assistant", "content": "怎么突然关心起这个？"},
            {"role": "user", "content": "就是想问问你。"},
        ],
        channel_topic="日常关心",
        objective_facts=_CHARACTER_PUBLIC_FACTS,
        memory_evidence_text="最近聊天主要围绕日常状态和轻松闲聊。",
    )
    state.update(
        {
            "internal_monologue": "对方是在问我现在的状态，不需要扯到生日或年龄这些公开资料。",
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
        }
    )

    result = await call_content_anchor_agent(state)
    _debug_snapshot("prompt_contracts.content_anchor.self_birthday_negative", result)

    anchors = result["content_anchors"]
    joined = "\n".join(anchors)
    assert "8月5" not in joined, f"Unrelated mood question should not leak birthday facts: {anchors!r}"
    assert "狮子座" not in joined, f"Unrelated mood question should not leak birthday facts: {anchors!r}"


async def test_live_content_anchor_answers_from_direct_conversation_evidence(ensure_live_llm) -> None:
    state = _make_state(
        user_input="我确认一下，你还记得我刚刚说那堆充电线、HDMI 线，还有几根我自己都忘了用途的线里大概有哪些吗？记不全也没关系。",
        chat_history_recent=[
            {
                "role": "user",
                "content": "大概就是充电线、HDMI 线，还有几根我自己都忘了用途的线。你说得对，可能没那么容易分清，我先不急着处理。",
            },
            {
                "role": "assistant",
                "content": "哇，这也太有仪式感了吧！感觉你搞得像在进行什么考古挖掘现场一样，乱糟糟的也挺好嘛。",
            },
        ],
        channel_topic="用户确认刚刚提到的旧线缆内容",
        memory_evidence_text="",
        last_relationship_insight="可以一起进行毫无意义的日常消磨。",
    )
    state.update(
        {
            "internal_monologue": "检索结果已经给出用户刚刚提到的线缆种类。虽然被检查细节有点局促，但回答应基于事实。",
            "logical_stance": "TENTATIVE",
            "character_intent": "PROVIDE",
            "conversation_progress": {
                "status": "active",
                "continuity": "same_episode",
                "conversation_mode": "casual_chat",
                "episode_phase": "developing",
                "topic_momentum": "stable",
                "current_thread": "user discussing tidying up old cables",
                "user_goal": "share daily trivialities",
                "current_blocker": "",
                "user_state_updates": [
                    {"text": "user is sharing mundane life details", "age_hint": "just now"}
                ],
                "assistant_moves": ["playful_teasing", "metaphorical_comment", "playful_mockery"],
                "overused_moves": ["probe_intent"],
                "open_loops": [],
                "resolved_threads": [],
                "avoid_reopening": [{"text": "user's intent/motive", "age_hint": "just now"}],
                "emotional_trajectory": "neutral to slightly sheepish/vague",
                "next_affordances": ["answer the cable detail check"],
                "progression_guidance": "Answer the factual detail check and avoid probing deeper.",
            },
        }
    )
    state["rag_result"]["answer"] = (
        "在 2026-04-29 的消息中，用户提到过有一些充电线、HDMI 线，"
        "以及几根自己都忘了用途的旧线缆，并表示目前不急着处理。"
    )
    state["rag_result"]["conversation_evidence"] = [
        (
            "用户在 2026-04-29 的消息中提到，有一些充电线、HDMI 线以及"
            "用途不明的旧线缆，并表示目前不急着处理。"
        )
    ]
    state["rag_result"]["supervisor_trace"] = {
        "loop_count": 1,
        "unknown_slots": [],
        "dispatched": [
            {
                "agent": "conversation_keyword_agent",
                "resolved": True,
                "slot": "Conversation-keyword: find messages containing cable detail",
            }
        ],
    }

    result = await call_content_anchor_agent(state)
    _debug_snapshot("prompt_contracts.content_anchor.fact_based_answer_from_conversation_evidence", result)

    anchors = result["content_anchors"]
    joined = "\n".join(anchors)
    answer_anchors = [anchor for anchor in anchors if anchor.startswith("[ANSWER]")]
    assert answer_anchors, f"Missing ANSWER anchor: {anchors!r}"
    answer_text = "\n".join(answer_anchors)
    assert "充电线" in answer_text, f"Direct evidence should be reflected in ANSWER: {anchors!r}"
    assert "HDMI" in answer_text, f"Direct evidence should be reflected in ANSWER: {anchors!r}"
    assert any(token in answer_text for token in ("用途不明", "忘了用途", "不清楚用途", "不知道用途", "不知道干什么")), (
        f"Direct evidence should include the unknown-purpose cable detail: {anchors!r}"
    )
    forbidden_uncertainty = (
        "记不清",
        "记不起来",
        "模糊",
        "不确定",
        "没印象",
        "没法说清楚",
        "没看清楚",
        "无法回答",
    )
    assert not any(token in answer_text for token in forbidden_uncertainty), (
        f"ANSWER contradicted direct evidence with uncertainty: {anchors!r}"
    )
    forbidden_deflection = ("怎么可能", "为什么要", "记得那么细", "你自己都")
    assert not any(token in answer_text for token in forbidden_deflection), (
        f"ANSWER deflected instead of restating direct evidence: {anchors!r}"
    )


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
            memory_evidence_text="最近聊天主要围绕天气和日常感受。",
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
            memory_evidence_text="最近聊天主要围绕图片内容和日常观察。",
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
            memory_evidence_text="最近聊天出现了轻微的称呼施压。",
            affinity=520,
            last_relationship_insight="对方最近有点试探边界，需要保持分寸。",
        ),
        id="stack-boundary-command-repeated-fillers",
    ),
]


def _stack_case_by_id(case_id: str) -> dict:
    """Return one named cognition-stack live case."""
    for parameter_set in _STACK_CASES:
        current_case_id, state = parameter_set.values
        if current_case_id == case_id:
            return state
    raise AssertionError(f"Unknown cognition stack case: {case_id}")


async def _assert_live_cognition_stack_prompt_contract(ensure_live_llm, case_id: str) -> None:
    """Run one inspectable cognition-stack live prompt-contract case."""
    del ensure_live_llm
    state = _stack_case_by_id(case_id)
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


async def test_live_cognition_stack_weather_english(ensure_live_llm) -> None:
    await _assert_live_cognition_stack_prompt_contract(ensure_live_llm, "weather_english")


async def test_live_cognition_stack_photo_request_chinese(ensure_live_llm) -> None:
    await _assert_live_cognition_stack_prompt_contract(ensure_live_llm, "photo_request_chinese")


async def test_live_cognition_stack_boundary_command_repeated_fillers(ensure_live_llm) -> None:
    await _assert_live_cognition_stack_prompt_contract(ensure_live_llm, "boundary_command_repeated_fillers")


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
            memory_evidence_text="最近聊天主要围绕天气和日常感受。",
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
            memory_evidence_text="最近聊天主要围绕天气和安静的日常。",
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
            memory_evidence_text="最近聊天出现了轻微的称呼施压。",
            affinity=520,
            last_relationship_insight="对方最近有点试探边界，需要保持分寸。",
        ),
        id="dialog-boundary-command-repeated-fillers",
    ),
]


def _dialog_case_by_id(case_id: str) -> dict:
    """Return one named dialog live case."""
    for parameter_set in _DIALOG_CASES:
        current_case_id, state = parameter_set.values
        if current_case_id == case_id:
            return state
    raise AssertionError(f"Unknown dialog case: {case_id}")


async def _assert_live_dialog_prompt_contract(ensure_live_llm, case_id: str) -> None:
    """Run one inspectable dialog live prompt-contract case."""
    del ensure_live_llm
    state = _dialog_case_by_id(case_id)
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


async def test_live_dialog_weather_english(ensure_live_llm) -> None:
    await _assert_live_dialog_prompt_contract(ensure_live_llm, "weather_english")


async def test_live_dialog_casual_chinese(ensure_live_llm) -> None:
    await _assert_live_dialog_prompt_contract(ensure_live_llm, "casual_chinese")


async def test_live_dialog_boundary_command_repeated_fillers(ensure_live_llm) -> None:
    await _assert_live_dialog_prompt_contract(ensure_live_llm, "boundary_command_repeated_fillers")
