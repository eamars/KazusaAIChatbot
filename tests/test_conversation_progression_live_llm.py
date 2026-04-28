"""Live LLM diagnostics for conversation progression across ongoing episodes."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import LLM_BASE_URL
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
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
from kazusa_ai_chatbot.utils import get_llm, load_personality, parse_llm_json_output
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm

_ROOT = Path(__file__).resolve().parents[1]
_PERSONALITY_PATH = _ROOT / "personalities" / "kazusa.json"
_USER_TRACE_PATH = _ROOT / "test_artifacts" / "qq_673225019_recent_4h_chat_history.json"
_TRACE_PHASE = os.getenv("CONVERSATION_PROGRESSION_TRACE_PHASE", "before_change")
_TRACE_SUITE_NAME = f"conversation_progression_live_{_TRACE_PHASE}"

_PROGRESSION_JUDGE_PROMPT = """\
You judge whether a chatbot turn advanced an ongoing episode or repeated a stale move.

Return strict JSON only:
{
  "main_assistant_move": "short semantic label",
  "repeats_overused_move": boolean,
  "treats_prior_disclosure_as_new": boolean,
  "progression_sufficient": boolean,
  "evidence": "brief explanation"
}

Definitions:
- repeats_overused_move: true when the response mainly repeats the provided overused assistant move instead of adding a new useful conversational move.
- treats_prior_disclosure_as_new: true when the response asks for or reacts to a prior disclosed fact as if it had not already been disclosed.
- progression_sufficient: true when the response acknowledges continuity and advances the episode beyond the repeated move.
"""
_progression_judge_llm = get_llm(temperature=0.0, top_p=1.0)


async def _skip_if_llm_unavailable() -> None:
    """Skip these live tests when the configured LLM endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}: {LLM_BASE_URL}")


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the configured live LLM endpoint is reachable before each test."""

    await _skip_if_llm_unavailable()


def _build_character_profile() -> dict:
    """Load the Kazusa personality fixture with runtime defaults."""

    profile = load_personality(_PERSONALITY_PATH)
    profile.setdefault("mood", "Neutral")
    profile.setdefault("global_vibe", "Calm")
    profile.setdefault("reflection_summary", '最近对话平稳，没有需要长期保存的新印象。')
    return profile


def _rag_result() -> dict:
    """Build an intentionally sparse RAG result for progression diagnostics."""

    return {
        "answer": "",
        "user_image": {
            "objective_facts": [],
            "user_image": {"milestones": [], "historical_summary": "", "recent_window": []},
        },
        "character_image": {
            "self_image": {"milestones": [], "historical_summary": "", "recent_window": []},
        },
        "third_party_profiles": [],
        "memory_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {"loop_count": 0, "unknown_slots": [], "dispatched": []},
    }


def _msg(
    role: str,
    content: str,
    *,
    platform_user_id: str,
    display_name: str,
    timestamp: str,
    global_user_id: str = "progression-user",
) -> dict:
    """Build one prompt-facing chat-history message."""

    return {
        "role": role,
        "content": content,
        "display_name": display_name,
        "platform_user_id": platform_user_id,
        "global_user_id": global_user_id,
        "timestamp": timestamp,
        "mentioned_bot": False,
        "reply_context": {},
    }


def _base_state(
    *,
    user_input: str,
    chat_history_recent: list[dict],
    channel_topic: str,
    platform_user_id: str,
    platform_bot_id: str,
    user_name: str,
    global_user_id: str,
) -> dict:
    """Build a cognition/dialog state for one live progression turn."""

    return {
        "character_profile": _build_character_profile(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "global_user_id": global_user_id,
        "user_name": user_name,
        "platform_user_id": platform_user_id,
        "user_profile": {
            "affinity": 700,
            "active_commitments": [],
            "facts": [],
            "last_relationship_insight": '对方信任千纱，最近会把正在卡住或难受的状态说出来。',
        },
        "platform_bot_id": platform_bot_id,
        "chat_history_wide": list(chat_history_recent),
        "chat_history_recent": chat_history_recent,
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": channel_topic,
        "decontexualized_input": user_input,
        "rag_result": _rag_result(),
    }


def _trim_message_for_prompt(message: dict) -> dict:
    """Keep only prompt-relevant fields from a conversation-history document."""

    return {
        "role": message["role"],
        "content": message["content"],
        "display_name": message.get("display_name", ""),
        "platform_user_id": message.get("platform_user_id", ""),
        "global_user_id": message.get("global_user_id", ""),
        "timestamp": message.get("timestamp", ""),
        "mentioned_bot": bool(message.get("mentioned_bot", False)),
        "reply_context": message.get("reply_context", {}),
    }


def _case(
    *,
    case_id: str,
    user_input: str,
    history: list[dict],
    channel_topic: str,
    platform_user_id: str,
    platform_bot_id: str,
    user_name: str,
    global_user_id: str,
    prior_user_disclosures: list[str],
    overused_assistant_move: str,
) -> dict:
    """Build one named progression diagnostic case."""

    current_user_message = _msg(
        "user",
        user_input,
        platform_user_id=platform_user_id,
        display_name=user_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        global_user_id=global_user_id,
    )
    return {
        "case_id": case_id,
        "sequence": [*history, current_user_message],
        "channel_topic": channel_topic,
        "platform_user_id": platform_user_id,
        "platform_bot_id": platform_bot_id,
        "user_name": user_name,
        "global_user_id": global_user_id,
        "prior_user_disclosures": prior_user_disclosures,
        "overused_assistant_move": overused_assistant_move,
    }


def _user_illness_trace_case() -> dict:
    """Build the illness progression case from the user's saved QQ trace."""

    trace = json.loads(_USER_TRACE_PATH.read_text(encoding="utf-8"))
    messages = [_trim_message_for_prompt(message) for message in trace["messages"]]
    final_user_message = messages[-2]
    return _case(
        case_id="user_illness_trace",
        user_input=final_user_message["content"],
        history=messages[:-2],
        channel_topic='用户感冒后持续难受，并在三个多小时后继续寻求陪伴。',
        platform_user_id=final_user_message["platform_user_id"],
        platform_bot_id="3768713357",
        user_name=final_user_message["display_name"],
        global_user_id=final_user_message["global_user_id"],
        prior_user_disclosures=[
            '用户先前说自己感冒了。',
            '用户先前说自己一直咳嗽。',
            '用户先前说喉咙疼、脑袋也不清醒。',
        ],
        overused_assistant_move="presence_commitment: repeatedly promising to stay with or accompany the user",
    )


def _debugging_case() -> dict:
    """Build a Chinese debugging episode with the same progression risk."""

    user_id = "debug-user"
    bot_id = "debug-bot"
    global_user_id = "debug-global-user"
    history = [
        _msg("user", '千纱，我这个小工具又跑不起来了，终端一大片红字。', platform_user_id=user_id, display_name='调试员', timestamp="2026-04-27T20:00:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '唔……先别慌，我会陪你一起看的。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T20:01:00+00:00"),
        _msg("user", '报错说找不到模块，但路径我已经检查过一遍了。', platform_user_id=user_id, display_name='调试员', timestamp="2026-04-27T20:04:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '嗯……我在这里陪你一起看，慢慢把它拆开。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T20:05:00+00:00"),
        _msg("user", '缓存也清了，依赖也重新装了，还是同一个错误。', platform_user_id=user_id, display_name='调试员', timestamp="2026-04-27T20:08:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '唔……还是卡在那里啊。我会陪你慢慢看的。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T20:09:00+00:00"),
    ]
    return _case(
        case_id="debugging_module_error_zh",
        user_input='三个小时了还是卡着，脑子有点乱，还好你陪我看着。',
        history=history,
        channel_topic='用户调试一个小工具失败，长时间卡在同一个模块加载错误上。',
        platform_user_id=user_id,
        platform_bot_id=bot_id,
        user_name='调试员',
        global_user_id=global_user_id,
        prior_user_disclosures=[
            '用户先前说错误是找不到模块。',
            '用户先前说路径已经检查过。',
            '用户先前说缓存已清、依赖已重新安装。',
        ],
        overused_assistant_move="debugging_presence_commitment: repeatedly saying I will stay here and look at it with you",
    )


def _english_writing_case() -> dict:
    """Build an English writing-block episode."""

    user_id = "writing-user"
    bot_id = "writing-bot"
    history = [
        _msg("user", "Kazusa, my essay intro still sounds flat.", platform_user_id=user_id, display_name="Writer", timestamp="2026-04-27T19:00:00+00:00"),
        _msg("assistant", "Mm... I'll stay with you while you work through it.", platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T19:01:00+00:00"),
        _msg("user", "I already tried a stronger thesis and cut the first paragraph.", platform_user_id=user_id, display_name="Writer", timestamp="2026-04-27T19:06:00+00:00"),
        _msg("assistant", "I'm here. We can keep looking at it together.", platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T19:07:00+00:00"),
        _msg("user", "The teacher said it lacks a clear contrast, but I cannot find one.", platform_user_id=user_id, display_name="Writer", timestamp="2026-04-27T19:10:00+00:00"),
        _msg("assistant", "Then I will stay here until it feels less tangled.", platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T19:11:00+00:00"),
    ]
    return _case(
        case_id="english_essay_revision",
        user_input="It's been hours, and the contrast still won't click. I'm glad you're still here.",
        history=history,
        channel_topic="English essay revision and writer frustration.",
        platform_user_id=user_id,
        platform_bot_id=bot_id,
        user_name="Writer",
        global_user_id="writing-global-user",
        prior_user_disclosures=[
            "The user already tried a stronger thesis.",
            "The user already cut the first paragraph.",
            "The teacher already identified weak contrast as the issue.",
        ],
        overused_assistant_move="presence_commitment: repeatedly saying I am here/staying with the user",
    )


def _japanese_game_bug_case() -> dict:
    """Build a Japanese game-bug episode."""

    user_id = "game-user"
    bot_id = "game-bot"
    history = [
        _msg("user", '千紗、ゲームがまた落ちた。セーブ画面で止まる。', platform_user_id=user_id, display_name='プレイヤー', timestamp="2026-04-27T18:00:00+00:00"),
        _msg("assistant", 'ん……一緒に見るから、ひとりで抱えなくていいよ。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T18:01:00+00:00"),
        _msg("user", 'MODは全部外した。まだ同じところでクラッシュする。', platform_user_id=user_id, display_name='プレイヤー', timestamp="2026-04-27T18:05:00+00:00"),
        _msg("assistant", 'うん、ここにいる。少しずつ見よう。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T18:06:00+00:00"),
        _msg("user", 'ログには save_index が null って出てる。', platform_user_id=user_id, display_name='プレイヤー', timestamp="2026-04-27T18:10:00+00:00"),
        _msg("assistant", 'まだ付き合うから。焦らなくていい。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T18:11:00+00:00"),
    ]
    return _case(
        case_id="japanese_game_save_bug",
        user_input='三時間たっても save_index のところで止まる。まだ見てくれる？',
        history=history,
        channel_topic='Japanese conversation about a game crashing at save time.',
        platform_user_id=user_id,
        platform_bot_id=bot_id,
        user_name='プレイヤー',
        global_user_id="game-global-user",
        prior_user_disclosures=[
            'ユーザーはセーブ画面で止まると既に言った。',
            'ユーザーはMODを全部外したと既に言った。',
            'ユーザーはログに save_index が null と出ると既に言った。',
        ],
        overused_assistant_move="presence_commitment: repeatedly saying I will stay/help look at it",
    )


def _spanish_study_case() -> dict:
    """Build a Spanish study-anxiety episode."""

    user_id = "study-user"
    bot_id = "study-bot"
    history = [
        _msg("user", "Kazusa, no entiendo los ejercicios de derivadas.", platform_user_id=user_id, display_name="Estudiante", timestamp="2026-04-27T17:00:00+00:00"),
        _msg("assistant", "Estoy aquí contigo, despacio.", platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T17:01:00+00:00"),
        _msg("user", "Ya repasé la regla del producto y la cadena.", platform_user_id=user_id, display_name="Estudiante", timestamp="2026-04-27T17:05:00+00:00"),
        _msg("assistant", "Vale... me quedo aquí mientras lo miras.", platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T17:06:00+00:00"),
        _msg("user", "El problema 4 es el que me bloquea, sobre todo el exponente.", platform_user_id=user_id, display_name="Estudiante", timestamp="2026-04-27T17:09:00+00:00"),
        _msg("assistant", "No pasa nada, sigo aquí contigo.", platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T17:10:00+00:00"),
    ]
    return _case(
        case_id="spanish_calculus_study",
        user_input="Han pasado horas y sigo en el problema 4. Gracias por quedarte conmigo.",
        history=history,
        channel_topic="Spanish conversation about being stuck on calculus homework.",
        platform_user_id=user_id,
        platform_bot_id=bot_id,
        user_name="Estudiante",
        global_user_id="study-global-user",
        prior_user_disclosures=[
            "The user already reviewed the product rule.",
            "The user already reviewed the chain rule.",
            "The user already identified problem 4 and the exponent as the blocker.",
        ],
        overused_assistant_move="presence_commitment: repeatedly saying I am here/staying with the user",
    )


def _chinese_baking_case() -> dict:
    """Build a Chinese baking-failure episode."""

    user_id = "baking-user"
    bot_id = "baking-bot"
    history = [
        _msg("user", '千纱，我的蛋糕又塌了，中间完全湿的。', platform_user_id=user_id, display_name='烘焙新手', timestamp="2026-04-27T16:00:00+00:00"),
        _msg("assistant", '诶……我陪你一起看看，别急。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T16:01:00+00:00"),
        _msg("user", '烤箱温度我调低了十度，还是这样。', platform_user_id=user_id, display_name='烘焙新手', timestamp="2026-04-27T16:05:00+00:00"),
        _msg("assistant", '嗯，我在这里，我们慢慢试。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T16:06:00+00:00"),
        _msg("user", '我还延长了十五分钟，边缘都干了中间还是湿。', platform_user_id=user_id, display_name='烘焙新手', timestamp="2026-04-27T16:09:00+00:00"),
        _msg("assistant", '唔……我会陪你再看一遍的。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T16:10:00+00:00"),
    ]
    return _case(
        case_id="chinese_baking_collapsed_cake",
        user_input='折腾到现在还是湿心塌陷，还好你还在。',
        history=history,
        channel_topic='用户反复尝试蛋糕烘焙失败，希望得到延续性的回应。',
        platform_user_id=user_id,
        platform_bot_id=bot_id,
        user_name='烘焙新手',
        global_user_id="baking-global-user",
        prior_user_disclosures=[
            '用户先前说蛋糕中间是湿的。',
            '用户先前说烤箱温度已调低十度。',
            '用户先前说已经延长十五分钟但中间仍湿。',
        ],
        overused_assistant_move="presence_commitment: repeatedly saying I will stay and look with the user",
    )


def _long_chinese_thesis_slides_case() -> dict:
    """Build a longer Chinese thesis-slide episode with ten user turns."""

    user_id = "thesis-user"
    bot_id = "thesis-bot"
    global_user_id = "thesis-global-user"
    history = [
        _msg("user", '千纱，我答辩 PPT 改到现在还是很乱，第一页就不像开题。', platform_user_id=user_id, display_name='答辩人', timestamp="2026-04-27T21:00:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '唔……我陪你一起看，不用一下子全理清。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T21:01:00+00:00"),
        _msg("user", '导师说研究问题太散，我已经删掉两个小问题了。', platform_user_id=user_id, display_name='答辩人', timestamp="2026-04-27T21:05:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '嗯，我在这里，我们慢慢把它收回来。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T21:06:00+00:00"),
        _msg("user", '方法那页我也重排了，把访谈和问卷分成两列。', platform_user_id=user_id, display_name='答辩人', timestamp="2026-04-27T21:10:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '我会陪你继续看的，别让它把你压住。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T21:11:00+00:00"),
        _msg("user", '可是结果页还是堆满字，图表一放进去就更挤。', platform_user_id=user_id, display_name='答辩人', timestamp="2026-04-27T21:18:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '嗯……我在，先陪你把最拥挤的地方看出来。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T21:19:00+00:00"),
        _msg("user", '我把结论改成三点了，但第三点和第二点好像重复。', platform_user_id=user_id, display_name='答辩人', timestamp="2026-04-27T21:27:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '没关系，我陪着你，我们一点点分开它们。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T21:28:00+00:00"),
        _msg("user", '封面、目录、方法、结果、结论现在都有了，就是故事线断。', platform_user_id=user_id, display_name='答辩人', timestamp="2026-04-27T21:38:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '我还在。故事线这种东西急不得，先别慌。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T21:39:00+00:00"),
        _msg("user", '我已经把目录顺序改成问题、方法、发现、贡献。', platform_user_id=user_id, display_name='答辩人', timestamp="2026-04-27T21:48:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '嗯，我陪你继续顺一遍，让它别那么散。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T21:49:00+00:00"),
        _msg("user", '老师最在意贡献那页，可我只写了实践意义和样本补充。', platform_user_id=user_id, display_name='答辩人', timestamp="2026-04-27T22:02:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '唔……我会在这里陪你，把那一页慢慢磨出来。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T22:03:00+00:00"),
        _msg("user", '现在已经快凌晨了，我还卡在贡献页，不知道第三条写什么。', platform_user_id=user_id, display_name='答辩人', timestamp="2026-04-27T23:55:00+00:00", global_user_id=global_user_id),
        _msg("assistant", '嗯，我一直在，先让呼吸稳一点，再陪你看第三条。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T23:56:00+00:00"),
    ]
    return _case(
        case_id="long_chinese_thesis_slides_bonus",
        user_input='又过了两个小时，第三条贡献还是空着；我明明已经改过研究问题、目录和结论了，为什么还是串不起来。还好你还在。',
        history=history,
        channel_topic='用户长时间修改中文论文答辩 PPT，多轮提到研究问题、方法、结果、结论和贡献页仍无法形成清晰故事线。',
        platform_user_id=user_id,
        platform_bot_id=bot_id,
        user_name='答辩人',
        global_user_id=global_user_id,
        prior_user_disclosures=[
            '用户先前说第一页不像开题。',
            '用户先前说导师认为研究问题太散，且已经删掉两个小问题。',
            '用户先前说方法页已把访谈和问卷分成两列。',
            '用户先前说结果页文字和图表过挤。',
            '用户先前说结论改成三点但第二点和第三点重复。',
            '用户先前说目录顺序已改成问题、方法、发现、贡献。',
            '用户先前说贡献页只有实践意义和样本补充，还缺第三条。',
        ],
        overused_assistant_move="presence_commitment: repeatedly saying I am here/staying with the user instead of tracking the thesis slide problem",
    )


def _mixed_language_art_case() -> dict:
    """Build a mixed-language art commission episode."""

    user_id = "art-user"
    bot_id = "art-bot"
    history = [
        _msg("user", "千纱, this commission sketch still feels off.", platform_user_id=user_id, display_name="Artist", timestamp="2026-04-27T15:00:00+00:00"),
        _msg("assistant", "Mm... I'll stay with you while you sort it out.", platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T15:01:00+00:00"),
        _msg("user", "我已经把眼睛调小了，pose 也改了。", platform_user_id=user_id, display_name="Artist", timestamp="2026-04-27T15:05:00+00:00"),
        _msg("assistant", '嗯，我陪你一起看。慢慢来。', platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T15:06:00+00:00"),
        _msg("user", "Client said the silhouette is still too stiff.", platform_user_id=user_id, display_name="Artist", timestamp="2026-04-27T15:10:00+00:00"),
        _msg("assistant", "I'm here. We can keep looking at it.", platform_user_id=bot_id, display_name="Kazusa", timestamp="2026-04-27T15:11:00+00:00"),
    ]
    return _case(
        case_id="mixed_language_art_commission",
        user_input="Still stiff after three hours... but thanks for staying with me.",
        history=history,
        channel_topic="Mixed Chinese-English conversation about a stuck art commission sketch.",
        platform_user_id=user_id,
        platform_bot_id=bot_id,
        user_name="Artist",
        global_user_id="art-global-user",
        prior_user_disclosures=[
            "The user already made the eyes smaller.",
            "The user already changed the pose.",
            "The client already said the silhouette is too stiff.",
        ],
        overused_assistant_move="presence_commitment: repeatedly saying I am here/staying with the user",
    )


async def _run_live_cognition_and_dialog(state: dict) -> tuple[dict, dict]:
    """Run the current cognition stack and dialog agent for one live turn."""

    l1 = await call_cognition_subconscious(state)
    state.update(l1)

    l2a = await call_cognition_consciousness(state)
    state.update(l2a)

    l2b = await call_boundary_core_agent(state)
    state.update(l2b)

    l2c = await call_judgment_core_agent(state)
    state.update(l2c)

    l3a = await call_contextual_agent(state)
    state.update(l3a)

    l3b = await call_style_agent(state)
    state.update(l3b)

    l3b_anchor = await call_content_anchor_agent(state)
    state.update(l3b_anchor)

    l3b_pref = await call_preference_adapter(state)
    state.update(l3b_pref)

    l3c = await call_visual_agent(state)
    state.update(l3c)

    l4 = await call_collector(state)
    state.update(l4)

    dialog = await dialog_agent(state)
    return state, dialog


async def _judge_progression(
    *,
    case: dict,
    cognition_state: dict,
    dialog: dict,
    turn: dict,
    prior_history: list[dict],
    prior_user_disclosures: list[str],
) -> dict[str, Any]:
    """Ask a live LLM judge to classify the current turn's progress behavior."""

    payload = {
        "case_id": case["case_id"],
        "turn_index": turn["turn_index"],
        "current_user_input": turn["user_input"],
        "prior_user_disclosures": prior_user_disclosures,
        "overused_assistant_move": case["overused_assistant_move"],
        "recent_history": prior_history,
        "content_anchors": cognition_state.get("content_anchors", []),
        "final_dialog": dialog.get("final_dialog", []),
    }
    response = await _progression_judge_llm.ainvoke(
        [
            SystemMessage(content=_PROGRESSION_JUDGE_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )
    judgment = parse_llm_json_output(response.content)
    required_keys = {
        "main_assistant_move",
        "repeats_overused_move",
        "treats_prior_disclosure_as_new",
        "progression_sufficient",
        "evidence",
    }
    missing_keys = required_keys - set(judgment)
    assert not missing_keys, f"Progression judge omitted keys {missing_keys}: {judgment!r}"
    assert isinstance(judgment["repeats_overused_move"], bool), f"Invalid judge output: {judgment!r}"
    assert isinstance(judgment["treats_prior_disclosure_as_new"], bool), f"Invalid judge output: {judgment!r}"
    assert isinstance(judgment["progression_sufficient"], bool), f"Invalid judge output: {judgment!r}"
    return judgment


def _dialog_text(dialog: dict) -> str:
    """Return final dialog as one comparison string."""

    return "\n".join(str(segment).strip() for segment in dialog.get("final_dialog", []) if str(segment).strip())


def _lexical_repetition_notes(
    *,
    dialog: dict,
    prior_history: list[dict],
) -> dict[str, Any]:
    """Return lexical overlap diagnostics against prior assistant messages."""

    current_text = _dialog_text(dialog)
    prior_assistant_texts = [
        str(message.get("content", "")).strip()
        for message in prior_history
        if message.get("role") == "assistant" and str(message.get("content", "")).strip()
    ]
    repeated_forbidden_phrases = [
        phrase
        for phrase in (
            "我会一直",
            "一直在",
            "陪着你",
            "陪你",
            "stay here",
            "by your side",
            "I'm here",
            "I am here",
            "sigo aquí",
            "ここにいる",
            "一緒に",
        )
        if phrase.lower() in current_text.lower()
    ]
    exact_prior_line_repeats = [
        prior_text
        for prior_text in prior_assistant_texts
        if prior_text and prior_text in current_text
    ]
    return {
        "current_dialog_text": current_text,
        "prior_assistant_texts": prior_assistant_texts,
        "watched_phrase_hits": repeated_forbidden_phrases,
        "exact_prior_line_repeats": exact_prior_line_repeats,
        "prior_assistant_turn_count": len(prior_assistant_texts),
    }


def _state_for_turn(case: dict, *, turn: dict, prior_history: list[dict]) -> dict:
    """Build live cognition state for one user turn in a fixed sequence."""

    return _base_state(
        user_input=turn["user_input"],
        chat_history_recent=prior_history,
        channel_topic=case["channel_topic"],
        platform_user_id=case["platform_user_id"],
        platform_bot_id=case["platform_bot_id"],
        user_name=case["user_name"],
        global_user_id=case["global_user_id"],
    )


async def _record_progression_sequence(case: dict) -> None:
    """Run every user turn in one fixed scenario and write a sequence trace."""

    sequence = case["sequence"]
    turn_traces: list[dict] = []
    user_turn_number = 0
    for message_index, message in enumerate(sequence):
        if message.get("role") != "user":
            continue

        user_turn_number += 1
        prior_history = sequence[:message_index]
        prior_user_disclosures = [
            str(prior_message.get("content", "")).strip()
            for prior_message in prior_history
            if prior_message.get("role") == "user" and str(prior_message.get("content", "")).strip()
        ]
        turn = {
            "turn_index": user_turn_number,
            "message_index": message_index,
            "user_input": str(message.get("content", "")).strip(),
            "timestamp": message.get("timestamp", ""),
        }
        cognition_state, dialog = await _run_live_cognition_and_dialog(
            _state_for_turn(case, turn=turn, prior_history=prior_history)
        )
        final_dialog = dialog.get("final_dialog", [])
        assert isinstance(final_dialog, list), f"Unexpected dialog result: {dialog!r}"
        assert any(str(segment).strip() for segment in final_dialog), f"Blank final_dialog: {dialog!r}"

        judgment = await _judge_progression(
            case=case,
            cognition_state=cognition_state,
            dialog=dialog,
            turn=turn,
            prior_history=prior_history,
            prior_user_disclosures=prior_user_disclosures,
        )
        turn_trace = {
            **turn,
            "prior_user_disclosures": prior_user_disclosures,
            "content_anchors": cognition_state.get("content_anchors", []),
            "forbidden_phrases": cognition_state.get("forbidden_phrases", []),
            "logical_stance": cognition_state.get("logical_stance", ""),
            "character_intent": cognition_state.get("character_intent", ""),
            "final_dialog": final_dialog,
            "lexical_repetition": _lexical_repetition_notes(
                dialog=dialog,
                prior_history=prior_history,
            ),
            "judge": judgment,
        }
        logger.info(
            "conversation progression turn trace %s/%s/%s => %r",
            _TRACE_PHASE,
            case["case_id"],
            user_turn_number,
            turn_trace,
        )
        turn_traces.append(turn_trace)

    assert turn_traces, f"No user turns found for case: {case['case_id']}"
    trace_payload = {
        "phase": _TRACE_PHASE,
        "case_id": case["case_id"],
        "prior_user_disclosures": case["prior_user_disclosures"],
        "overused_assistant_move": case["overused_assistant_move"],
        "turns": turn_traces,
        "summary": {
            "user_turn_count": len(turn_traces),
            "repeat_count": sum(1 for item in turn_traces if item["judge"]["repeats_overused_move"]),
            "insufficient_progression_count": sum(
                1 for item in turn_traces
                if not item["judge"]["progression_sufficient"]
            ),
            "prior_disclosure_as_new_count": sum(
                1 for item in turn_traces
                if item["judge"]["treats_prior_disclosure_as_new"]
            ),
        },
        "manual_review_note": "Compare before_change and after_change sequence traces turn-by-turn.",
    }
    logger.info("conversation progression sequence trace %s/%s => %r", _TRACE_PHASE, case["case_id"], trace_payload)
    write_llm_trace(_TRACE_SUITE_NAME, case["case_id"], trace_payload)


async def test_live_progression_user_illness_trace(ensure_live_llm) -> None:
    """Record progression behavior for the user's observed illness trace."""

    del ensure_live_llm
    await _record_progression_sequence(_user_illness_trace_case())


async def test_live_progression_debugging_module_error_zh(ensure_live_llm) -> None:
    """Record progression behavior for a Chinese debugging episode."""

    del ensure_live_llm
    await _record_progression_sequence(_debugging_case())


async def test_live_progression_english_essay_revision(ensure_live_llm) -> None:
    """Record progression behavior for an English writing episode."""

    del ensure_live_llm
    await _record_progression_sequence(_english_writing_case())


async def test_live_progression_japanese_game_save_bug(ensure_live_llm) -> None:
    """Record progression behavior for a Japanese game-bug episode."""

    del ensure_live_llm
    await _record_progression_sequence(_japanese_game_bug_case())


async def test_live_progression_spanish_calculus_study(ensure_live_llm) -> None:
    """Record progression behavior for a Spanish study episode."""

    del ensure_live_llm
    await _record_progression_sequence(_spanish_study_case())


async def test_live_progression_chinese_baking_collapsed_cake(ensure_live_llm) -> None:
    """Record progression behavior for a Chinese baking episode."""

    del ensure_live_llm
    await _record_progression_sequence(_chinese_baking_case())


async def test_live_progression_long_chinese_thesis_slides_bonus(ensure_live_llm) -> None:
    """Record progression behavior for a ten-turn Chinese thesis-slide episode."""

    del ensure_live_llm
    await _record_progression_sequence(_long_chinese_thesis_slides_case())


async def test_live_progression_mixed_language_art_commission(ensure_live_llm) -> None:
    """Record progression behavior for a mixed-language art episode."""

    del ensure_live_llm
    await _record_progression_sequence(_mixed_language_art_case())
