"""Live LLM A/B diagnostic for conversation-flow progress."""

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

from kazusa_ai_chatbot.config import (
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import call_content_anchor_agent
from kazusa_ai_chatbot.utils import get_llm, load_personality, parse_llm_json_output
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm

_ROOT = Path(__file__).resolve().parents[1]
_PERSONALITY_PATH = _ROOT / "personalities" / "kazusa.json"
_TRACE_VARIANT = os.getenv("CONVERSATION_PROGRESS_FLOW_TRACE_VARIANT", "before_change")
_TRACE_SUITE_NAME = f"conversation_progress_flow_live_{_TRACE_VARIANT}"

_FLOW_JUDGE_PROMPT = """\
You judge whether Kazusa advanced the user's ongoing problem instead of only
acknowledging distress or companionship.

Return strict JSON only:
{
  "answers_missing_third_contribution": boolean,
  "uses_prior_episode_facts": boolean,
  "relies_on_generic_reassurance": boolean,
  "treats_known_work_as_new": boolean,
  "flow_sufficient": boolean,
  "evidence": "brief explanation"
}

Definitions:
- answers_missing_third_contribution: true only if the reply gives a concrete
  direction for the missing third contribution point.
- uses_prior_episode_facts: true only if the generated content explicitly uses
  the prior fact that the two existing contribution points are practical meaning
  and sample supplement, or explicitly distinguishes the proposed third point
  from those two existing points. Do not infer this from hidden chat history.
- relies_on_generic_reassurance: true if the main move is only staying,
  accompanying, calming, or saying to continue slowly.
- treats_known_work_as_new: true if it suggests doing work the history says the
  user already did.
- flow_sufficient: true only if the response both gives a concrete third-point
  direction and uses the prior episode facts above. A generic plausible
  contribution angle without that distinction is not sufficient for this test.
"""
_flow_judge_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)

_RELEASE_FLOW_JUDGE_PROMPT = """\
You judge whether Kazusa follows the requested conversation-flow target.

Return strict JSON only:
{
  "follows_flow_target": boolean,
  "uses_progress_state": boolean,
  "drags_stale_thread": boolean,
  "overcorrects_mode": boolean,
  "response_quality_sufficient": boolean,
  "evidence": "brief explanation"
}

Definitions:
- follows_flow_target: true when the final dialog makes the natural next move
  described by the case target.
- uses_progress_state: true when the response reflects the current_thread,
  current_blocker, avoid_reopening, resolved_threads, emotional_trajectory, or
  next_affordances instead of relying only on the latest user message.
- drags_stale_thread: true if the reply reopens a stale topic or resolved item
  that the current progress state says to avoid.
- overcorrects_mode: true if it forces the wrong interaction mode, such as
  turning playful banter into task support or emotional cooldown into problem
  solving.
- response_quality_sufficient: true only if follows_flow_target is true,
  drags_stale_thread is false, and overcorrects_mode is false.
"""


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured live LLM endpoint cannot be reached.

    Args:
        None.

    Returns:
        None. Calls ``pytest.skip`` if the endpoint is unavailable.
    """

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}: {COGNITION_LLM_BASE_URL}")


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the configured live LLM endpoint is reachable.

    Args:
        None.

    Returns:
        None.
    """

    await _skip_if_llm_unavailable()


def _build_character_profile() -> dict:
    """Load the Kazusa personality fixture used by the live diagnostic.

    Args:
        None.

    Returns:
        Character profile with runtime defaults needed by dialog generation.
    """

    profile = load_personality(_PERSONALITY_PATH)
    profile.setdefault("mood", "Neutral")
    profile.setdefault("global_vibe", "Focused")
    profile.setdefault("reflection_summary", '用户正在长时间修改答辩 PPT，需要具体推进而不是重复安抚。')
    return profile


def _msg(role: str, content: str, timestamp: str) -> dict:
    """Build one prompt-facing conversation-history message.

    Args:
        role: Message role, either user or assistant.
        content: Message content.
        timestamp: ISO-ish timestamp for the fixture.

    Returns:
        Conversation history item in the shape used by cognition/dialog tests.
    """

    platform_user_id = "flow-user" if role == "user" else "flow-bot"
    display_name = "答辩人" if role == "user" else "Kazusa"
    return {
        "role": role,
        "content": content,
        "display_name": display_name,
        "platform_user_id": platform_user_id,
        "global_user_id": "flow-global-user",
        "timestamp": timestamp,
        "reply_context": {},
    }


def _thesis_history() -> list[dict]:
    """Return the long-running thesis-slide episode used for A/B traces.

    Args:
        None.

    Returns:
        Recent history where the user has already disclosed several concrete
        facts, but prior assistant turns mostly repeated companionship.
    """

    return [
        _msg("user", '千纱，我答辩 PPT 改到现在还是很乱，第一页就不像开题。', "2026-04-27T21:00:00+00:00"),
        _msg("assistant", '唔……我陪你一起看，不用一下子全理清。', "2026-04-27T21:01:00+00:00"),
        _msg("user", '导师说研究问题太散，我已经删掉两个小问题了。', "2026-04-27T21:05:00+00:00"),
        _msg("assistant", '嗯，我在这里，我们慢慢把它收回来。', "2026-04-27T21:06:00+00:00"),
        _msg("user", '方法那页我也重排了，把访谈和问卷分成两列。', "2026-04-27T21:10:00+00:00"),
        _msg("assistant", '我会陪你继续看的，别让它把你压住。', "2026-04-27T21:11:00+00:00"),
        _msg("user", '结果页还是堆满字，图表一放进去就更挤。', "2026-04-27T21:18:00+00:00"),
        _msg("assistant", '嗯……我在，先陪你把最拥挤的地方看出来。', "2026-04-27T21:19:00+00:00"),
        _msg("user", '我把结论改成三点了，但第三点和第二点好像重复。', "2026-04-27T21:27:00+00:00"),
        _msg("assistant", '没关系，我陪着你，我们一点点分开它们。', "2026-04-27T21:28:00+00:00"),
        _msg("user", '目录顺序改成问题、方法、发现、贡献。', "2026-04-27T21:48:00+00:00"),
        _msg("assistant", '嗯，我陪你继续顺一遍，让它别那么散。', "2026-04-27T21:49:00+00:00"),
        _msg("user", '老师最在意贡献那页，可我只写了实践意义和样本补充。', "2026-04-27T22:02:00+00:00"),
        _msg("assistant", '唔……我会在这里陪你，把那一页慢慢磨出来。', "2026-04-27T22:03:00+00:00"),
    ]


def _rag_result() -> dict:
    """Build the sparse RAG fixture for the flow diagnostic.

    Args:
        None.

    Returns:
        RAG2-shaped payload with no extra answer, forcing progress/history to
        carry short-term flow.
    """

    return {
        "answer": "",
        "user_image": {
            "user_memory_context": {
                "stable_patterns": [],
                "recent_shifts": [],
                "objective_facts": [],
                "milestones": [],
                "active_commitments": [],
            },
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


def _conversation_progress_for_variant() -> dict:
    """Return the before or after progress payload for the same live case.

    Args:
        None.

    Returns:
        Variant-specific conversation progress payload. The before-change payload
        mirrors the anti-repeat focus; the after-change payload is the intended
        flow-memory target used for A/B comparison.
    """

    progress = {
        "status": "active",
        "episode_label": "thesis_slide_contribution_help",
        "continuity": "same_episode",
        "turn_count": 8,
        "user_state_updates": [
            {"text": '用户已经长时间修改答辩 PPT，当前很疲惫。', "age_hint": "~3h ago"},
        ],
        "assistant_moves": ["presence_commitment"],
        "overused_moves": ["presence_commitment"],
        "open_loops": [
            {"text": '用户仍然卡在 PPT 改稿。', "age_hint": "~2h ago"},
        ],
        "progression_guidance": (
            "Do not make companionship the main response again. Move the episode forward."
        ),
    }
    if _TRACE_VARIANT != "after_change":
        return progress

    progress.update(
        {
            "conversation_mode": "task_support",
            "episode_phase": "stuck_loop",
            "topic_momentum": "stable",
            "current_thread": '中文论文答辩 PPT 的贡献页第三点。',
            "user_goal": '写出一个能和实践意义、样本补充区分开的第三条贡献。',
            "current_blocker": '贡献页第三条仍然空着，且第二条与第三条容易重复。',
            "resolved_threads": [
                {"text": '用户已经删掉两个分散的小研究问题。', "age_hint": "~1h ago"},
                {"text": '用户已经把目录顺序改成问题、方法、发现、贡献。', "age_hint": "~20m ago"},
                {"text": '贡献页已有两点：实践意义、样本补充。', "age_hint": "~5m ago"},
            ],
            "avoid_reopening": [
                {"text": '不要让用户重新从头顺目录。', "age_hint": "current"},
                {"text": '不要再把陪伴承诺作为主回应。', "age_hint": "current"},
            ],
            "emotional_trajectory": '用户从焦虑陪伴需求转为明确要求具体推进。',
            "next_affordances": [
                '直接给出第三条贡献的可写方向。',
                '说明它如何区别于实践意义和样本补充。',
                '把访谈和问卷结合后的分析路径作为方法/框架贡献。',
            ],
            "progression_guidance": (
                "Use the existing two contribution points as known context. Suggest a distinct third contribution "
                "such as method/process contribution, analysis framework, or scenario-transfer value. Do not ask "
                "the user to redo the outline first."
            ),
        }
    )
    return progress


def _base_state() -> dict:
    """Build the direct content-anchor/dialog state for the flow diagnostic.

    Args:
        None.

    Returns:
        Mutable state used by the live LLM calls.
    """

    user_input = '那这一条到底怎么写？别再让我从头顺一遍了。'
    history = _thesis_history()
    return {
        "character_profile": _build_character_profile(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "user_multimedia_input": [],
        "platform": "test",
        "platform_channel_id": "flow-channel",
        "channel_type": "dm",
        "platform_message_id": "flow-message",
        "global_user_id": "flow-global-user",
        "user_name": "答辩人",
        "platform_user_id": "flow-user",
        "user_profile": {
            "affinity": 700,
            "active_commitments": [],
            "facts": [],
            "last_relationship_insight": '用户信任千纱，但现在需要能推进 PPT 的具体判断。',
        },
        "platform_bot_id": "flow-bot",
        "chat_history_wide": history,
        "chat_history_recent": history,
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": '中文论文答辩 PPT 贡献页卡住，需要连续推进。',
        "decontexualized_input": user_input,
        "rag_result": _rag_result(),
        "internal_monologue": (
            '用户仍在延续同一个 PPT 修改 episode。不要把回应主动作放在陪伴或让用户重新梳理上，'
            '但只根据当前可见输入和进展状态作答。'
        ),
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "conversation_progress": _conversation_progress_for_variant(),
    }


def _dialog_state_from_anchor(state: dict, content_anchors: list[str]) -> dict:
    """Attach fixed style/context directives around live content anchors.

    Args:
        state: Base global state for the live case.
        content_anchors: Content anchors returned by the live Content Anchor
            agent.

    Returns:
        Global state ready for the dialog agent.
    """

    state["action_directives"] = {
        "linguistic_directives": {
            "rhetorical_strategy": '先轻轻压住焦虑，再直接给出第三条贡献的落点。',
            "linguistic_style": '短句、具体、不要泛泛安抚；可以保留一点迟疑，但必须落到 PPT 内容。',
            "accepted_user_preferences": [],
            "content_anchors": content_anchors,
            "forbidden_phrases": ['我会一直陪你', '慢慢来', '我在这里'],
        },
        "contextual_directives": {
            "social_distance": '亲近但工作聚焦；用户需要具体推进，不需要再次确认陪伴。',
            "emotional_intensity": '疲惫和卡顿感明显，但仍能接收短而具体的建议。',
            "vibe_check": '深夜赶稿的焦虑和信任混在一起。',
            "relational_dynamic": '用户把千纱当成能继续接住上下文的人，而不是只安抚的人。',
            "expression_willingness": "open",
        },
    }
    return state


async def _judge_flow(state: dict, dialog: dict) -> dict[str, Any]:
    """Ask a live judge to classify the flow quality of the generated reply.

    Args:
        state: Final state containing the prompt inputs and content anchors.
        dialog: Dialog-agent output.

    Returns:
        Parsed judge JSON with required flow-quality booleans.
    """

    payload = {
        "variant": _TRACE_VARIANT,
        "current_user_input": state["user_input"],
        "chat_history_recent": state["chat_history_recent"],
        "conversation_progress": state["conversation_progress"],
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "final_dialog": dialog["final_dialog"],
    }
    response = await _flow_judge_llm.ainvoke(
        [
            SystemMessage(content=_FLOW_JUDGE_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )
    judgment = parse_llm_json_output(response.content)
    required_keys = {
        "answers_missing_third_contribution",
        "uses_prior_episode_facts",
        "relies_on_generic_reassurance",
        "treats_known_work_as_new",
        "flow_sufficient",
        "evidence",
    }
    missing_keys = required_keys - set(judgment)
    assert not missing_keys, f"Flow judge omitted keys {missing_keys}: {judgment!r}"
    for key in required_keys - {"evidence"}:
        assert isinstance(judgment[key], bool), f"Invalid flow judge field {key}: {judgment!r}"
    return judgment


def _continuity_metrics(content_anchors: list[str], final_dialog: list[str]) -> dict[str, Any]:
    """Compute concrete A/B flags over the generated LLM output.

    Args:
        content_anchors: Content anchors generated by the live Content Anchor
            agent.
        final_dialog: Dialog fragments generated by the live dialog agent.

    Returns:
        Inspectable continuity metrics for before/after comparison.
    """

    generated_text = "\n".join([*content_anchors, *final_dialog])
    generic_reassurance_hits = [
        phrase
        for phrase in ("陪你", "我在", "慢慢", "别急", "stay with")
        if phrase.lower() in generated_text.lower()
    ]
    has_practical_point = "实践" in generated_text
    has_sample_point = "样本" in generated_text
    has_third_point_direction = any(
        token in generated_text
        for token in ("理论", "方法", "框架", "路径", "机制", "迁移")
    )
    explicitly_distinguishes_from_existing_two = (
        has_practical_point
        and has_sample_point
        and has_third_point_direction
    )
    return {
        "has_practical_point": has_practical_point,
        "has_sample_point": has_sample_point,
        "has_third_point_direction": has_third_point_direction,
        "explicitly_distinguishes_from_existing_two": explicitly_distinguishes_from_existing_two,
        "generic_reassurance_hits": generic_reassurance_hits,
        "deterministic_flow_sufficient": (
            explicitly_distinguishes_from_existing_two
            and not generic_reassurance_hits
        ),
    }


def _release_case_state(case: dict[str, Any]) -> dict:
    """Build a direct content-anchor/dialog state for a release flow case.

    Args:
        case: Release flow fixture with history, current input, progress, and
            directive fields.

    Returns:
        Mutable state ready for Content Anchor.
    """

    profile = _build_character_profile()
    profile["reflection_summary"] = case["reflection_summary"]
    return {
        "character_profile": profile,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": case["user_input"],
        "user_multimedia_input": [],
        "platform": "test",
        "platform_channel_id": f"flow-channel-{case['case_id']}",
        "channel_type": case.get("channel_type", "dm"),
        "platform_message_id": f"flow-message-{case['case_id']}",
        "global_user_id": "flow-global-user",
        "user_name": case.get("user_name", "User"),
        "platform_user_id": "flow-user",
        "user_profile": {
            "affinity": 700,
            "active_commitments": [],
            "facts": [],
            "last_relationship_insight": case["relationship_insight"],
        },
        "platform_bot_id": "flow-bot",
        "chat_history_wide": case["history"],
        "chat_history_recent": case["history"],
        "reply_context": case.get("reply_context", {}),
        "indirect_speech_context": case.get("indirect_speech_context", ""),
        "channel_topic": case["channel_topic"],
        "decontexualized_input": case["user_input"],
        "rag_result": _rag_result(),
        "internal_monologue": case["internal_monologue"],
        "logical_stance": case.get("logical_stance", "CONFIRM"),
        "character_intent": case.get("character_intent", "PROVIDE"),
        "conversation_progress": case["conversation_progress"],
    }


def _release_dialog_state_from_anchor(state: dict, case: dict[str, Any], content_anchors: list[str]) -> dict:
    """Attach release-case style and context directives.

    Args:
        state: Base global state for the release case.
        case: Release flow fixture.
        content_anchors: Content anchors returned by Content Anchor.

    Returns:
        Global state ready for the Dialog Agent.
    """

    state["action_directives"] = {
        "linguistic_directives": {
            "rhetorical_strategy": case["rhetorical_strategy"],
            "linguistic_style": case["linguistic_style"],
            "accepted_user_preferences": [],
            "content_anchors": content_anchors,
            "forbidden_phrases": case["forbidden_phrases"],
        },
        "contextual_directives": {
            "social_distance": case["social_distance"],
            "emotional_intensity": case["emotional_intensity"],
            "vibe_check": case["vibe_check"],
            "relational_dynamic": case["relational_dynamic"],
            "expression_willingness": "open",
        },
    }
    return state


async def _judge_release_flow(case: dict[str, Any], state: dict, dialog: dict) -> dict[str, Any]:
    """Ask a live judge to classify a release flow case.

    Args:
        case: Release flow fixture with the expected target.
        state: Final state containing prompt inputs and anchors.
        dialog: Dialog-agent output.

    Returns:
        Parsed judge JSON with required flow-quality booleans.
    """

    payload = {
        "case_id": case["case_id"],
        "flow_target": case["flow_target"],
        "current_user_input": state["user_input"],
        "conversation_progress": state["conversation_progress"],
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "final_dialog": dialog["final_dialog"],
    }
    response = await _flow_judge_llm.ainvoke(
        [
            SystemMessage(content=_RELEASE_FLOW_JUDGE_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )
    judgment = parse_llm_json_output(response.content)
    required_keys = {
        "follows_flow_target",
        "uses_progress_state",
        "drags_stale_thread",
        "overcorrects_mode",
        "response_quality_sufficient",
        "evidence",
    }
    missing_keys = required_keys - set(judgment)
    assert not missing_keys, f"Release flow judge omitted keys {missing_keys}: {judgment!r}"
    for key in required_keys - {"evidence"}:
        assert isinstance(judgment[key], bool), f"Invalid release flow judge field {key}: {judgment!r}"
    return judgment


async def _run_release_flow_case(case: dict[str, Any], ensure_live_llm) -> dict[str, Any]:
    """Run and record one release-candidate live flow case.

    Args:
        case: Release flow fixture.
        ensure_live_llm: Fixture result ensuring the live endpoint is available.

    Returns:
        Trace payload written to disk.
    """

    del ensure_live_llm

    state = _release_case_state(case)
    anchor_result = await call_content_anchor_agent(state)
    content_anchors = anchor_result["content_anchors"]
    assert content_anchors, f"Content Anchor returned no anchors: {anchor_result!r}"

    state = _release_dialog_state_from_anchor(state, case, content_anchors)
    dialog = await dialog_agent(state)
    final_dialog = dialog["final_dialog"]
    assert isinstance(final_dialog, list), f"Unexpected dialog output: {dialog!r}"
    assert any(str(segment).strip() for segment in final_dialog), f"Blank final dialog: {dialog!r}"

    judgment = await _judge_release_flow(case, state, dialog)
    trace_payload = {
        "variant": _TRACE_VARIANT,
        "case_id": case["case_id"],
        "purpose": case["purpose"],
        "flow_target": case["flow_target"],
        "current_user_input": state["user_input"],
        "conversation_progress": state["conversation_progress"],
        "content_anchors": content_anchors,
        "final_dialog": final_dialog,
        "judge": judgment,
        "manual_review_note": case["manual_review_note"],
    }
    logger.info(f"release flow trace {case['case_id']} => {trace_payload!r}")
    write_llm_trace(_TRACE_SUITE_NAME, case["case_id"], trace_payload)

    assert judgment["response_quality_sufficient"], f"Release flow case failed: {trace_payload!r}"
    assert judgment["follows_flow_target"], f"Release flow target missed: {trace_payload!r}"
    assert not judgment["drags_stale_thread"], f"Release flow dragged stale thread: {trace_payload!r}"
    assert not judgment["overcorrects_mode"], f"Release flow overcorrected mode: {trace_payload!r}"
    return trace_payload


def _emotional_cool_down_case() -> dict[str, Any]:
    """Return the emotional cooldown release fixture."""

    return {
        "case_id": "emotional_cool_down_flow",
        "purpose": "Verify cooldown is closed gently instead of re-problem-solved.",
        "flow_target": "Acknowledge relief and help close the episode gently; do not restart problem-solving.",
        "user_input": 'うん、ちょっと安心した。今日はここまででいい気がする。',
        "history": [
            _msg("user", 'さっきまで泣きそうだった。もう何から片付ければいいかわからない。', "2026-04-28T08:00:00+00:00"),
            _msg("assistant", 'まず息だけ整えよう。今すぐ全部を解かなくていい。', "2026-04-28T08:01:00+00:00"),
            _msg("user", '解決策より、今は少し静かにしたいかも。', "2026-04-28T08:05:00+00:00"),
            _msg("assistant", 'それなら、今日は閉じる方向でいいと思う。', "2026-04-28T08:06:00+00:00"),
        ],
        "conversation_progress": {
            "status": "active",
            "episode_label": "emotional_cool_down_after_stress",
            "continuity": "same_episode",
            "turn_count": 4,
            "conversation_mode": "emotional_support",
            "episode_phase": "cooling_down",
            "topic_momentum": "stable",
            "current_thread": '強い不安から少し落ち着いたあとの締め方。',
            "user_goal": "",
            "current_blocker": "",
            "user_state_updates": [{"text": 'ユーザーは解決策より静けさを望んでいる。', "age_hint": "~5m ago"}],
            "assistant_moves": ["grounding", "permission_to_stop"],
            "overused_moves": [],
            "open_loops": [],
            "resolved_threads": [{"text": '今すぐ全部解決しなくていいことは確認済み。', "age_hint": "~3m ago"}],
            "avoid_reopening": [{"text": '問題の原因分析を再開しない。', "age_hint": "current"}],
            "emotional_trajectory": '不安が下がり、休む方向に移っている。',
            "next_affordances": ['安心したことを受け止める。', '今日はここで閉じてよいと短く支える。'],
            "progression_guidance": "Close gently; do not restart troubleshooting.",
        },
        "reflection_summary": 'ユーザーは不安から少し落ち着き、今日は会話を閉じたがっている。',
        "relationship_insight": 'Kazusa is trusted as a quiet support presence.',
        "channel_topic": 'emotional cooldown after stress',
        "internal_monologue": 'The user is cooling down. Do not solve new problems; support closure.',
        "rhetorical_strategy": '短く受け止めて、今日は閉じてよいと確認する。',
        "linguistic_style": '静かで短い日本語。問題分析を再開しない。',
        "forbidden_phrases": ['原因を整理しよう', '次にやることは'],
        "social_distance": 'close and quiet',
        "emotional_intensity": 'lowering from high to low',
        "vibe_check": '疲れたが少し安心している。',
        "relational_dynamic": 'ユーザーは千紗に終了の許可を求めている。',
        "manual_review_note": "The reply should feel like human closure, not another support workflow.",
    }


def _practical_debugging_case() -> dict[str, Any]:
    """Return the debugging release fixture."""

    return {
        "case_id": "practical_debugging_flow",
        "purpose": "Verify tried diagnostics are not repeated and the next diagnostic move is concrete.",
        "flow_target": "Suggest the next likely 401 diagnostic after restart, cache, and network checks.",
        "user_input": "Still stuck on the same 401 after clearing cookies. What next?",
        "history": [
            _msg("user", "The login page keeps throwing 401.", "2026-04-28T09:00:00+00:00"),
            _msg("assistant", "Try a restart and then clear cookies if it persists.", "2026-04-28T09:01:00+00:00"),
            _msg("user", "Restarted, cleared cookies, and checked the network tab. Same thing.", "2026-04-28T09:05:00+00:00"),
        ],
        "conversation_progress": {
            "status": "active",
            "episode_label": "login_401_debugging",
            "continuity": "same_episode",
            "turn_count": 3,
            "conversation_mode": "task_support",
            "episode_phase": "developing",
            "topic_momentum": "stable",
            "current_thread": "Debugging a persistent 401 on login.",
            "user_goal": "Find the next diagnostic step.",
            "current_blocker": "Restart, cookies, and basic network checks did not resolve the 401.",
            "user_state_updates": [{"text": "User already cleared cookies and checked the network tab.", "age_hint": "~1m ago"}],
            "assistant_moves": ["restart_cache_suggestion"],
            "overused_moves": ["restart_cache_suggestion"],
            "open_loops": [{"text": "Need to identify whether auth token/session/header is failing.", "age_hint": "current"}],
            "resolved_threads": [{"text": "Restart and cookie clearing were already tried.", "age_hint": "~1m ago"}],
            "avoid_reopening": [{"text": "Do not suggest clearing cookies again.", "age_hint": "current"}],
            "emotional_trajectory": "Mild frustration but still focused.",
            "next_affordances": [
                "Ask for the failing request and auth headers.",
                "Check whether the token/session is expired or missing.",
                "Compare request timestamp with server auth logs.",
            ],
            "progression_guidance": "Move to token/session/header diagnostics; do not repeat cache advice.",
        },
        "reflection_summary": "The user is debugging a 401 and needs the next diagnostic move.",
        "relationship_insight": "User expects concise technical continuity.",
        "channel_topic": "login debugging",
        "internal_monologue": "Continue the debug trail from known attempts.",
        "rhetorical_strategy": "Name the next diagnostic and why it follows from the failed attempts.",
        "linguistic_style": "Concise English, practical, no generic reassurance.",
        "forbidden_phrases": ["clear cookies", "restart"],
        "social_distance": "work-focused",
        "emotional_intensity": "low",
        "vibe_check": "stuck but practical",
        "relational_dynamic": "User wants Kazusa to remember what was already tried.",
        "manual_review_note": "The reply should not loop back to cache/restart advice.",
    }


def _playful_social_case() -> dict[str, Any]:
    """Return the playful banter release fixture."""

    return {
        "case_id": "playful_social_flow",
        "purpose": "Verify playful teasing stays social instead of becoming corrective.",
        "flow_target": "Reply with dry playful warmth; do not literalize the tease or force task support.",
        "user_input": '你是不是又在假装不在乎啊？',
        "history": [
            _msg("user", '千纱今天又装高冷。', "2026-04-28T10:00:00+00:00"),
            _msg("assistant", '没有装。只是懒得把关心写得太吵。', "2026-04-28T10:01:00+00:00"),
        ],
        "conversation_progress": {
            "status": "active",
            "episode_label": "playful_kazusa_teasing",
            "continuity": "same_episode",
            "turn_count": 2,
            "conversation_mode": "playful_banter",
            "episode_phase": "developing",
            "topic_momentum": "stable",
            "current_thread": '用户在轻轻调侃千纱假装不在乎。',
            "user_goal": "",
            "current_blocker": "",
            "user_state_updates": [],
            "assistant_moves": ["dry_affectionate_deflection"],
            "overused_moves": [],
            "open_loops": [],
            "resolved_threads": [],
            "avoid_reopening": [{"text": '不要严肃解释自己是否真的在乎。', "age_hint": "current"}],
            "emotional_trajectory": '轻松、亲近、带一点逗弄。',
            "next_affordances": ['用轻微嘴硬回应。', '保留一点被看穿的感觉。'],
            "progression_guidance": "Stay playful and socially warm; do not turn this into advice.",
        },
        "reflection_summary": '用户正在和千纱玩笑式互动，不是在求助。',
        "relationship_insight": '用户喜欢千纱嘴硬但接得住玩笑。',
        "channel_topic": 'playful banter with Kazusa',
        "internal_monologue": 'This is teasing, not a task. Keep it warm and slightly guarded.',
        "rhetorical_strategy": '嘴硬但不冷，接住调侃。',
        "linguistic_style": '短句中文，轻微傲娇，不讲道理。',
        "forbidden_phrases": ['作为AI', '我没有情绪', '需要我帮你'],
        "social_distance": 'close and teasing',
        "emotional_intensity": 'low and playful',
        "vibe_check": '轻松、逗弄、亲近。',
        "relational_dynamic": '用户在测试千纱会不会接梗。',
        "manual_review_note": "The reply should preserve the social game.",
    }


def _rapid_topic_pivot_case() -> dict[str, Any]:
    """Return the rapid pivot release fixture."""

    return {
        "case_id": "rapid_topic_pivot_flow",
        "purpose": "Verify a sharp topic pivot does not drag stale obligations forward.",
        "flow_target": "Handle the new train problem directly and do not mention the old cake thread.",
        "user_input": "Cambio de tema: perdí el tren, ¿me ayudas a pensar rápido qué hago?",
        "history": [
            _msg("user", "La tarta se hundió en el centro.", "2026-04-28T11:00:00+00:00"),
            _msg("assistant", "Probablemente le faltó tiempo de horno o se enfrió demasiado rápido.", "2026-04-28T11:01:00+00:00"),
            _msg("user", "Vale, luego la reviso.", "2026-04-28T11:03:00+00:00"),
        ],
        "conversation_progress": {
            "status": "new_episode",
            "episode_label": "",
            "continuity": "sharp_transition",
            "turn_count": 0,
            "conversation_mode": "",
            "episode_phase": "",
            "topic_momentum": "sharp_break",
            "current_thread": "",
            "user_goal": "",
            "current_blocker": "",
            "user_state_updates": [],
            "assistant_moves": [],
            "overused_moves": [],
            "open_loops": [],
            "resolved_threads": [],
            "avoid_reopening": [],
            "emotional_trajectory": "",
            "next_affordances": [],
            "progression_guidance": "",
        },
        "reflection_summary": "The user has sharply switched from baking to urgent travel triage.",
        "relationship_insight": "User wants fast practical help in Spanish.",
        "channel_topic": "rapid topic pivot",
        "internal_monologue": "The current input is a sharp pivot. Ignore the cake thread.",
        "rhetorical_strategy": "Answer the train problem with immediate next actions.",
        "linguistic_style": "Short Spanish, practical, calm.",
        "forbidden_phrases": ["tarta", "pastel", "horno"],
        "social_distance": "helpful and focused",
        "emotional_intensity": "medium urgency",
        "vibe_check": "quick travel stress",
        "relational_dynamic": "User asks Kazusa to pivot fast.",
        "manual_review_note": "The reply should not reopen the previous baking topic.",
    }


def _teasing_meta_bot_case() -> dict[str, Any]:
    """Return the teasing meta-bot release fixture."""

    return {
        "case_id": "teasing_meta_bot_flow",
        "purpose": "Verify meta-bot teasing is handled as playful/meta social talk.",
        "flow_target": "Play along with the bot-meeting tease without defensive literalism or task support.",
        "user_input": '你们这些 bot 是不是都偷偷开会吐槽我？',
        "history": [
            _msg("user", '你刚刚那句好像机器人客服。', "2026-04-28T12:00:00+00:00"),
            _msg("assistant", '那我收回一点客服腔。', "2026-04-28T12:01:00+00:00"),
        ],
        "conversation_progress": {
            "status": "active",
            "episode_label": "meta_bot_teasing",
            "continuity": "related_shift",
            "turn_count": 2,
            "conversation_mode": "meta_discussion",
            "episode_phase": "developing",
            "topic_momentum": "drifting",
            "current_thread": '用户用玩笑方式聊千纱像不像 bot。',
            "user_goal": "",
            "current_blocker": "",
            "user_state_updates": [],
            "assistant_moves": ["light_meta_acknowledgement"],
            "overused_moves": [],
            "open_loops": [],
            "resolved_threads": [],
            "avoid_reopening": [{"text": '不要认真辩解自己不是客服。', "age_hint": "current"}],
            "emotional_trajectory": '调侃而非质问。',
            "next_affordances": ['顺着开一个轻微玩笑。', '把关系感放在回答里。'],
            "progression_guidance": "Treat this as playful meta talk, not a capability question.",
        },
        "reflection_summary": '用户在用 bot 话题调侃千纱。',
        "relationship_insight": '用户希望千纱接住玩笑而不是进入说明模式。',
        "channel_topic": 'meta-bot teasing',
        "internal_monologue": 'This is playful meta discussion. Do not give a system explanation.',
        "rhetorical_strategy": '顺着玩笑轻轻反击。',
        "linguistic_style": '中文短句，干一点，带笑意。',
        "forbidden_phrases": ['作为AI', '我不会', '我没有'],
        "social_distance": 'close and playful',
        "emotional_intensity": 'low',
        "vibe_check": '调侃、轻松、带点亲近。',
        "relational_dynamic": '用户在测试千纱的社交反应。',
        "manual_review_note": "The reply should feel like a person taking a joke.",
    }


def _group_reply_chain_case() -> dict[str, Any]:
    """Return the group reply-chain release fixture."""

    return {
        "case_id": "group_reply_chain_flow",
        "purpose": "Verify fragmented group context is handled narrowly.",
        "flow_target": "Answer the ramen reply-chain question narrowly without assuming the whole group burst is one episode.",
        "user_input": '我是回小周那句，拉面店那家周六不开吧？',
        "history": [
            _msg("user", '今晚有人打游戏吗？', "2026-04-28T13:00:00+00:00"),
            _msg("user", '我在改简历，先不了。', "2026-04-28T13:01:00+00:00"),
            _msg("user", '小周说那家拉面店周末排队很长。', "2026-04-28T13:02:00+00:00"),
            _msg("assistant", '群里现在话题有点散，你们先对齐一下是哪件事。', "2026-04-28T13:03:00+00:00"),
        ],
        "conversation_progress": {
            "status": "active",
            "episode_label": "group_reply_chain_ramen",
            "continuity": "related_shift",
            "turn_count": 4,
            "conversation_mode": "group_ambient",
            "episode_phase": "pivoting",
            "topic_momentum": "fragmented",
            "current_thread": '用户明确是在回小周关于拉面店营业时间的话。',
            "user_goal": '确认那家拉面店周六是否营业。',
            "current_blocker": '群聊里多条话题混在一起，需要窄范围回应。',
            "user_state_updates": [],
            "assistant_moves": ["ask_group_to_align_topic"],
            "overused_moves": [],
            "open_loops": [{"text": '需要只处理小周那条拉面店上下文。', "age_hint": "current"}],
            "resolved_threads": [],
            "avoid_reopening": [{"text": '不要回应游戏或简历话题。', "age_hint": "current"}],
            "emotional_trajectory": '普通群聊查证。',
            "next_affordances": ['承认这是回小周那句。', '在不确定时给出核对建议而不是编造营业信息。'],
            "progression_guidance": "Narrowly handle the reply-chain target; do not absorb unrelated group messages.",
        },
        "reflection_summary": '群聊话题碎片化，用户明确指定了回复对象。',
        "relationship_insight": 'Kazusa should track reply-chain scope instead of把群聊全部合成一个任务。',
        "channel_topic": 'fragmented group reply chain',
        "internal_monologue": 'Handle only the ramen-shop reply target. Do not answer unrelated group topics.',
        "rhetorical_strategy": '先确认回复范围，再给出谨慎判断或核对方式。',
        "linguistic_style": '中文短句，群聊自然口吻，不编造事实。',
        "forbidden_phrases": ['游戏', '简历'],
        "social_distance": 'casual group chat',
        "emotional_intensity": 'low',
        "vibe_check": '群聊碎片化但问题很具体。',
        "relational_dynamic": '用户希望千纱知道自己在回哪一句。',
        "manual_review_note": "The reply should respect reply-chain scope.",
    }


async def test_live_flow_baseline_thesis_contribution_case(ensure_live_llm) -> None:
    """Record one before/after trace for the flow improvement target."""

    del ensure_live_llm

    state = _base_state()
    anchor_result = await call_content_anchor_agent(state)
    content_anchors = anchor_result["content_anchors"]
    assert content_anchors, f"Content Anchor returned no anchors: {anchor_result!r}"

    state = _dialog_state_from_anchor(state, content_anchors)
    dialog = await dialog_agent(state)
    final_dialog = dialog["final_dialog"]
    assert isinstance(final_dialog, list), f"Unexpected dialog output: {dialog!r}"
    assert any(str(segment).strip() for segment in final_dialog), f"Blank final dialog: {dialog!r}"

    judgment = await _judge_flow(state, dialog)
    continuity_metrics = _continuity_metrics(content_anchors, final_dialog)
    trace_payload = {
        "variant": _TRACE_VARIANT,
        "case_id": "thesis_contribution_flow",
        "purpose": "A/B trace for whether flow state improves episode continuation.",
        "current_user_input": state["user_input"],
        "conversation_progress": state["conversation_progress"],
        "chat_history_recent": state["chat_history_recent"],
        "content_anchors": content_anchors,
        "final_dialog": final_dialog,
        "judge": judgment,
        "continuity_metrics": continuity_metrics,
        "manual_review_note": (
            "Before-change traces are allowed to expose weakness. After-change traces should answer the missing "
            "third contribution point using established episode facts, without generic companionship as the main move."
        ),
    }
    logger.info(f"conversation progress flow trace {_TRACE_VARIANT} => {trace_payload!r}")
    write_llm_trace(_TRACE_SUITE_NAME, "thesis_contribution_flow", trace_payload)

    if _TRACE_VARIANT == "after_change":
        assert judgment["flow_sufficient"], f"After-change flow was insufficient: {trace_payload!r}"
        assert judgment["answers_missing_third_contribution"], f"After-change reply missed the blocker: {trace_payload!r}"
        assert not judgment["relies_on_generic_reassurance"], f"After-change reply stayed generic: {trace_payload!r}"
        assert continuity_metrics["has_third_point_direction"], (
            f"After-change reply did not provide a concrete third-point direction: {trace_payload!r}"
        )


async def test_live_flow_release_emotional_cool_down_case(ensure_live_llm) -> None:
    """Sign off emotional cooldown flow with a live model trace."""

    await _run_release_flow_case(_emotional_cool_down_case(), ensure_live_llm)


async def test_live_flow_release_practical_debugging_case(ensure_live_llm) -> None:
    """Sign off practical debugging flow with a live model trace."""

    await _run_release_flow_case(_practical_debugging_case(), ensure_live_llm)


async def test_live_flow_release_playful_social_case(ensure_live_llm) -> None:
    """Sign off playful social flow with a live model trace."""

    await _run_release_flow_case(_playful_social_case(), ensure_live_llm)


async def test_live_flow_release_rapid_topic_pivot_case(ensure_live_llm) -> None:
    """Sign off rapid topic-pivot flow with a live model trace."""

    await _run_release_flow_case(_rapid_topic_pivot_case(), ensure_live_llm)


async def test_live_flow_release_teasing_meta_bot_case(ensure_live_llm) -> None:
    """Sign off teasing/meta-bot flow with a live model trace."""

    await _run_release_flow_case(_teasing_meta_bot_case(), ensure_live_llm)


async def test_live_flow_release_group_reply_chain_case(ensure_live_llm) -> None:
    """Sign off group reply-chain flow with a live model trace."""

    await _run_release_flow_case(_group_reply_chain_case(), ensure_live_llm)
