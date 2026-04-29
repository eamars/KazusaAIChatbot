"""Tests for cognition/dialog prompt integration with conversation progress."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.conversation_progress import projection
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module


class _FakeResponse:
    """Small LLM response stand-in."""

    def __init__(self, payload: dict):
        self.content = json.dumps(payload)


class _CapturingLLM:
    """Capture messages passed into one prompt call."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = []

    async def ainvoke(self, messages):
        self.messages = messages
        return _FakeResponse(self.payload)


@pytest.mark.asyncio
async def test_content_anchor_agent_receives_conversation_progress(monkeypatch) -> None:
    """Content Anchor input includes compact progress guidance."""

    fake_llm = _CapturingLLM({
        "content_anchors": [
            "[DECISION] answer the current question",
            "[AVOID_REPEAT] reassurance",
            "[PROGRESSION] provide the missing detail",
            "[SCOPE] ~30 words",
        ],
    })
    monkeypatch.setattr(l3_module, "_content_anchor_agent_llm", fake_llm)

    result = await l3_module.call_content_anchor_agent({
        "character_profile": {"name": "Kazusa"},
        "decontexualized_input": "what is the missing third point?",
        "rag_result": {},
        "internal_monologue": "answer directly",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "conversation_progress": {
            "status": "active",
            "episode_label": "slides_help",
            "continuity": "same_episode",
            "turn_count": 8,
            "conversation_mode": "task_support",
            "episode_phase": "stuck_loop",
            "topic_momentum": "stable",
            "current_thread": "slides contribution page",
            "user_goal": "write the missing third contribution point",
            "current_blocker": "third point overlaps the second",
            "user_state_updates": [{"text": "user still lacks a third contribution point", "age_hint": "~3h ago"}],
            "assistant_moves": ["reassurance"],
            "overused_moves": ["reassurance"],
            "open_loops": [{"text": "third point missing", "age_hint": "~3h ago"}],
            "resolved_threads": [{"text": "outline order already changed", "age_hint": "~2h ago"}],
            "avoid_reopening": [{"text": "do not ask to redo the outline", "age_hint": "~2h ago"}],
            "emotional_trajectory": "tired but still asking for help",
            "next_affordances": ["give a concrete third contribution angle"],
            "progression_guidance": "address the missing third point",
        },
    })

    human_payload = json.loads(fake_llm.messages[1].content)
    assert human_payload["conversation_progress"]["overused_moves"] == ["reassurance"]
    assert human_payload["conversation_progress"]["conversation_mode"] == "task_support"
    assert human_payload["conversation_progress"]["next_affordances"] == ["give a concrete third contribution angle"]
    assert result["content_anchors"][1].startswith("[AVOID_REPEAT]")


def test_content_anchor_prompt_allows_progression_anchor_labels() -> None:
    """Prompt contract explicitly allows the new progress labels."""

    assert "[AVOID_REPEAT]" in l3_module._CONTENT_ANCHOR_AGENT_PROMPT
    assert "[PROGRESSION]" in l3_module._CONTENT_ANCHOR_AGENT_PROMPT
    assert "conversation_progress" in l3_module._CONTENT_ANCHOR_AGENT_PROMPT
    assert "next_affordances" in l3_module._CONTENT_ANCHOR_AGENT_PROMPT
    assert "avoid_reopening" in l3_module._CONTENT_ANCHOR_AGENT_PROMPT


def test_content_anchor_prompt_requires_fact_based_answers_without_case_example() -> None:
    """Prompt keeps L3 bound to upstream stance while preserving direct facts."""

    prompt = l3_module._CONTENT_ANCHOR_AGENT_PROMPT

    assert "依赖树（先解析上游，再生成下游）" in prompt
    assert "logical_stance + character_intent" in prompt
    assert "不能反向改变 `logical_stance`、`character_intent` 或已选 `[FACT]`" in prompt
    assert "不要在这里修正上游立场" in prompt
    assert "若 `rag_result.answer` 直接回答当前问题，它是最高优先级事实摘要" in prompt
    assert "`[ANSWER]` 不得与 `[DECISION]` 或 `[FACT]` 矛盾" in prompt
    assert "应保留这些具体内容，避免替换成泛称" in prompt
    assert "`character_intent = CLARIFY` 时，`[ANSWER]` 必须是缩小歧义范围的追问" in prompt
    assert "在服从[DECISION]的前提下" in prompt
    assert "只按上方“依赖树”和“解析步骤”执行" in prompt
    assert "`TENTATIVE` 不是事实不确定" not in prompt
    assert "禁止把已知事实改写成第一人称认知失败" not in prompt
    assert "充电线" not in prompt
    assert "HDMI" not in prompt


def _profile_conformance_state() -> dict:
    return {
        "character_profile": {
            "name": "一之濑明日奈",
            "mood": "Neutral",
            "global_vibe": "Calm",
            "personality_brief": {
                "mbti": "ENFP",
            },
            "boundary_profile": {
                "self_integrity": 0.7,
                "control_sensitivity": 0.3,
                "compliance_strategy": "comply",
                "relational_override": 0.65,
                "control_intimacy_misread": 0.35,
                "boundary_recovery": "rebound",
                "authority_skepticism": 0.35,
            },
        },
        "decontexualized_input": "换个轻松点的话题，你现在会想吃点甜的吗？",
        "user_profile": {
            "affinity": 500,
            "last_relationship_insight": "普通协作关系，没有当前边界冲突。",
        },
        "chat_history_recent": [
            {"role": "user", "content": "刚才的问题只是分类。"},
            {"role": "assistant", "content": "那就按类别整理吧。"},
        ],
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "boundary_summary": "无边界问题。",
            "acceptance": "allow",
            "stance_bias": "confirm",
        },
        "internal_monologue": "这是轻松换话题，可以自然接住。",
        "emotional_appraisal": "轻松、好奇。",
    }


@pytest.mark.asyncio
async def test_contextual_agent_receives_boundary_profile_contract(monkeypatch) -> None:
    """Contextual prompt receives inherited boundary-profile constraints."""

    fake_llm = _CapturingLLM({
        "social_distance": "日常轻松",
        "emotional_intensity": "轻微活力",
        "vibe_check": "普通闲聊",
        "relational_dynamic": "自然接住轻松话题",
        "expression_willingness": "open",
    })
    monkeypatch.setattr(l3_module, "_contextual_agent_llm", fake_llm)

    result = await l3_module.call_contextual_agent(_profile_conformance_state())

    system_prompt = fake_llm.messages[0].content
    human_payload = json.loads(fake_llm.messages[1].content)
    assert "边界画像绑定规则" in system_prompt
    assert "Boundary Profile" in system_prompt
    assert "comply" in system_prompt
    assert "话题合法性" in system_prompt
    assert "场景时间压力" in system_prompt
    assert "已提供的检索记忆/事实上下文" in system_prompt
    assert "RAG" not in system_prompt
    assert human_payload["boundary_core_assessment"]["acceptance"] == "allow"
    assert "boundary_profile" not in human_payload
    assert human_payload["decontexualized_input"] == "换个轻松点的话题，你现在会想吃点甜的吗？"
    assert result["vibe_check"] == "普通闲聊"


@pytest.mark.asyncio
async def test_visual_agent_receives_boundary_profile_contract(monkeypatch) -> None:
    """Visual prompt uses boundary profile to avoid threat-framing benign turns."""

    fake_llm = _CapturingLLM({
        "facial_expression": ["自然地笑了一下"],
        "body_language": ["姿态放松"],
        "gaze_direction": ["看向话题本身"],
        "visual_vibe": ["轻松的日常氛围"],
    })
    monkeypatch.setattr(l3_module, "_visual_agent_llm", fake_llm)

    result = await l3_module.call_visual_agent(_profile_conformance_state())

    system_prompt = fake_llm.messages[0].content
    human_payload = json.loads(fake_llm.messages[1].content)
    assert "边界画像绑定规则" in system_prompt
    assert "Boundary Profile" in system_prompt
    assert "rebound" in system_prompt
    assert "被审查" in system_prompt
    assert "场景时间压力" in system_prompt
    assert "已提供的检索记忆/事实上下文" in system_prompt
    assert "RAG" not in system_prompt
    assert human_payload["boundary_core_assessment"]["stance_bias"] == "confirm"
    assert "boundary_profile" not in human_payload
    assert result["visual_vibe"] == ["轻松的日常氛围"]


def test_projection_preserves_relative_age_for_prior_disclosure() -> None:
    """Stored first_seen_at becomes an LLM-facing age_hint."""

    prompt_doc = projection.project_prompt_doc(
        document={
            "status": "active",
            "episode_label": "illness_support",
            "continuity": "same_episode",
            "turn_count": 6,
            "user_state_updates": [
                {"text": "user previously said their throat hurts", "first_seen_at": "2026-04-28T01:00:00+00:00"},
            ],
            "assistant_moves": [],
            "overused_moves": [],
            "open_loops": [],
            "progression_guidance": "",
        },
        current_timestamp="2026-04-28T04:00:00+00:00",
    )

    assert prompt_doc["user_state_updates"][0]["age_hint"] == "~3h ago"


def test_dialog_evaluator_prompt_uses_existing_feedback_for_avoid_repeat() -> None:
    """Evaluator prompt includes the move-level repeat backstop."""

    assert "[AVOID_REPEAT]" in dialog_module._DIALOG_EVALUATOR_PROMPT
    assert "[PROGRESSION]" in dialog_module._DIALOG_EVALUATOR_PROMPT
    assert "feedback" in dialog_module._DIALOG_EVALUATOR_PROMPT


def test_content_anchor_prompt_owns_topic_admission_decision() -> None:
    """Topic-admission decisions belong to L3 content anchors, not dialog."""

    prompt = l3_module._CONTENT_ANCHOR_AGENT_PROMPT

    assert "话题准入决定必须在这里完成" in prompt
    assert "若上游 `logical_stance` 已确认" in prompt
    assert "只有当上游立场或意图已经表达保留" in prompt
    assert "dialog" not in prompt.lower()
