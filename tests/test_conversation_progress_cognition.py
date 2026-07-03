"""Tests for cognition/dialog prompt integration with conversation progress."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.conversation_progress import projection
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_chain_core.stages import l3 as l3_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2c2 as l2c2_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from llm_test_helpers import bind_test_llm


class _FakeResponse:
    """Small LLM response stand-in."""

    def __init__(self, payload: dict):
        self.content = json.dumps(payload)


class _CapturingLLM:
    """Capture messages passed into one prompt call."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = []

    async def ainvoke(self, messages, *, config=None):
        self.messages = messages
        return _FakeResponse(self.payload)


class _FailingLLM:
    """LLM stand-in that fails if a skipped stage invokes it."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, *, config=None):
        self.calls += 1
        raise AssertionError("visual agent LLM should not be called")


def _minimal_text_chat_episode() -> dict:
    """Build a valid text-chat cognitive episode for direct L3 tests."""
    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-progress-cognition",
        percept_id="percept-progress-cognition",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="what is the missing third point?",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        platform_message_id="msg-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=[],
        active_turn_conversation_row_ids=[],
        debug_modes={},
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    return episode


@pytest.mark.asyncio
async def test_visual_agent_skip_returns_empty_directives_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal skip flag should bypass visual LLM generation completely."""
    episode = _minimal_text_chat_episode()
    episode["origin_metadata"]["debug_modes"] = {
        "no_visual_directives": True,
    }
    fake_llm = _FailingLLM()
    monkeypatch.setattr(l3_module, "_visual_agent_llm", bind_test_llm(fake_llm, "visual_agent_llm"))

    result = await l3_module.call_visual_agent({
        "cognitive_episode": episode,
    })

    assert result == {
        "facial_expression": [],
        "body_language": [],
        "gaze_direction": [],
        "visual_vibe": [],
    }
    assert fake_llm.calls == 0


@pytest.mark.asyncio
async def test_content_plan_agent_receives_conversation_progress(monkeypatch) -> None:
    """Content-plan input includes compact progress guidance."""

    fake_llm = _CapturingLLM({
        "content_plan": {
            "semantic_content": (
                "Answer the missing third point without repeating reassurance."
            ),
            "rendering": "One ordinary text message; concise.",
        },
    })
    monkeypatch.setattr(l3_module, "_content_plan_agent_llm", bind_test_llm(fake_llm, "content_plan_agent_llm"))

    result = await l3_module.call_content_plan_agent({
        "cognitive_episode": _minimal_text_chat_episode(),
        "character_profile": {"name": "Kazusa"},
        "user_input": "what is the missing third point?",
        "user_name": "Jigsaw",
        "prompt_message_context": {
            "body_text": "what is the missing third point?",
            "addressed_to_global_user_ids": ["character-user"],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "reply_context": {},
        "decontexualized_input": "what is the missing third point?",
        "rag_result": {},
        "internal_monologue": "answer directly",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "referents": [],
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
    assert "repeating reassurance" in result["content_plan"]["semantic_content"]


def test_content_plan_prompt_allows_progression_plan_roles() -> None:
    """Prompt contract reads progress and lifecycle plan roles."""

    prompt = l3_module._CONTENT_PLAN_AGENT_PROMPT
    assert "content_plan_roles" in prompt
    assert "avoid_reopening" in prompt
    assert "acknowledge_fulfillment" in prompt
    assert "keep_waiting" in prompt
    assert "conversation_progress" in prompt
    assert "next_affordances" in prompt
    assert "[AVOID_REPEAT]" not in prompt
    assert "[PROGRESSION]" not in prompt


def test_content_plan_prompt_requires_fact_based_answers_without_case_example() -> None:
    """Prompt keeps L3 bound to upstream stance while preserving direct facts."""

    prompt = l3_module._CONTENT_PLAN_AGENT_PROMPT

    assert "`logical_stance` 与 `character_intent` 是已定的 L2 立场和意图" in prompt
    assert "内容计划只能执行它们，不能改判" in prompt
    assert "不要在这里改写 `logical_stance`、`character_intent`、检索事实或上游意识判断" in prompt
    assert "`answer` 是最高优先级的直接检索结论" in prompt
    assert "技术对比：`semantic_content` 保留所有已给数值、单位和结论" in prompt
    assert "处理证据边界" in prompt
    assert "# 生成步骤" in prompt
    assert "确定来源和任务" in prompt
    assert "收集可见语义" in prompt
    assert "[ANSWER]" not in prompt
    assert "[DECISION]" not in prompt
    assert "[FACT]" not in prompt
    assert "`TENTATIVE` 不是事实不确定" not in prompt
    assert "禁止把已知事实改写成第一人称认知失败" not in prompt
    assert "充电线" not in prompt
    assert "HDMI" not in prompt


def _profile_conformance_state() -> dict:
    return {
        "cognitive_episode": _minimal_text_chat_episode(),
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
        "user_input": "换个轻松点的话题，你现在会想吃点甜的吗？",
        "prompt_message_context": {
            "body_text": "换个轻松点的话题，你现在会想吃点甜的吗？",
            "addressed_to_global_user_ids": ["character-user"],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
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
        "referents": [],
        "rag_result": {
            "answer": "用户在切换到轻松偏好闲聊。",
            "user_image": {"user_memory_context": {}},
            "character_image": {},
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
        },
        "internal_monologue": "这是轻松换话题，可以自然接住。",
        "logical_stance": "CONFIRM",
        "character_intent": "BANTAR",
        "judgment_note": "普通闲聊可以自然接住。",
        "content_plan": {
            "semantic_content": "可以说一个自然的甜食偏好。",
            "rendering": "1 条普通文字消息，约30字。",
        },
        "social_distance": "日常轻松",
        "emotional_intensity": "轻微活力",
        "vibe_check": "普通闲聊",
        "relational_dynamic": "自然接住轻松话题",
        "emotional_appraisal": "轻松、好奇。",
        "reply_context": {},
        "channel_topic": "",
    }


@pytest.mark.asyncio
async def test_contextual_agent_receives_boundary_profile_contract(monkeypatch) -> None:
    """Contextual prompt receives inherited boundary-profile constraints."""

    fake_llm = _CapturingLLM({
        "social_distance": "日常轻松",
        "emotional_intensity": "轻微活力",
        "vibe_check": "普通闲聊",
        "relational_dynamic": "自然接住轻松话题",
    })
    monkeypatch.setattr(l2c2_module, "_contextual_agent_llm", bind_test_llm(fake_llm, "contextual_agent_llm"))

    result = await l2c2_module.call_social_context_appraisal(_profile_conformance_state())

    system_prompt = fake_llm.messages[0].content
    human_payload = json.loads(fake_llm.messages[1].content)
    assert "边界画像绑定规则" in system_prompt
    assert "Boundary Profile" in system_prompt
    assert "顺从" in system_prompt
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
    monkeypatch.setattr(l3_module, "_visual_agent_llm", bind_test_llm(fake_llm, "visual_agent_llm"))

    result = await l3_module.call_visual_agent(_profile_conformance_state())

    system_prompt = fake_llm.messages[0].content
    human_payload = json.loads(fake_llm.messages[1].content)
    assert "边界画像绑定规则" in system_prompt
    assert "Boundary Profile" in system_prompt
    assert "rebound" in system_prompt
    assert "被审查" in system_prompt
    assert "场景时间压力" in system_prompt
    assert "已提供的检索记忆/事实上下文" in system_prompt
    assert "静态画面" in system_prompt
    assert "单帧" in system_prompt
    assert "content_plan" in system_prompt
    assert "RAG" not in system_prompt
    assert human_payload["boundary_core_assessment"]["stance_bias"] == "confirm"
    assert human_payload["logical_stance"] == "CONFIRM"
    assert human_payload["content_plan"]["semantic_content"].startswith("可以说")
    assert human_payload["contextual_directives"]["vibe_check"] == "普通闲聊"
    assert human_payload["prompt_message_context"]["body_text"] == "换个轻松点的话题，你现在会想吃点甜的吗？"
    assert "boundary_profile" not in human_payload
    assert result["visual_vibe"] == ["轻松的日常氛围"]


@pytest.mark.asyncio
async def test_visual_agent_prompt_omits_raw_runtime_ids(monkeypatch) -> None:
    """Visual prompt should receive semantic scene context, not raw ids."""

    fake_llm = _CapturingLLM({
        "facial_expression": ["自然"],
        "body_language": [],
        "gaze_direction": [],
        "visual_vibe": [],
    })
    monkeypatch.setattr(l3_module, "_visual_agent_llm", bind_test_llm(fake_llm, "visual_agent_llm"))
    state = _profile_conformance_state()
    state["prompt_message_context"] = {
        "body_text": "看这张图。",
        "addressed_to_global_user_ids": ["raw-character-global-id"],
        "broadcast": False,
        "mentions": [
            {
                "display_name": "测试用户",
                "entity_kind": "user",
                "platform_user_id": "raw-mentioned-platform-id",
                "global_user_id": "raw-mentioned-global-id",
            },
        ],
        "attachments": [
            {
                "media_kind": "image",
                "description": "桌上的甜点照片。",
                "summary_status": "available",
                "raw_attachment_id": "raw-attachment-id",
            },
        ],
        "reply": {
            "platform_message_id": "raw-reply-message-id",
            "platform_user_id": "raw-reply-platform-id",
            "global_user_id": "raw-reply-global-id",
            "display_name": "上条消息作者",
            "excerpt": "上一条可见文本。",
        },
    }
    state["reply_context"] = {
        "reply_to_message_id": "raw-service-reply-message-id",
        "reply_to_platform_user_id": "raw-service-reply-platform-id",
        "reply_to_display_name": "上条消息作者",
        "reply_excerpt": "上一条可见文本。",
    }
    state["rag_result"]["supervisor_trace"] = {
        "loop_count": 1,
        "unknown_slots": [],
        "dispatched": [
            {
                "slot": "conversation evidence",
                "agent": "conversation_search",
                "resolved": True,
                "source_refs": [
                    {
                        "conversation_row_id": "raw-row-id",
                        "platform_message_id": "raw-message-id",
                    },
                ],
            },
        ],
    }

    await l3_module.call_visual_agent(state)

    human_payload = json.loads(fake_llm.messages[1].content)
    rendered_payload = fake_llm.messages[1].content
    prompt_context = human_payload["prompt_message_context"]
    reply_context = human_payload["reply_context"]
    supervisor_trace = human_payload["rag_result"]["supervisor_trace"]
    assert prompt_context["mentions"] == [
        {"display_name": "测试用户", "entity_kind": "user"},
    ]
    assert prompt_context["reply"] == {
        "display_name": "上条消息作者",
        "excerpt": "上一条可见文本。",
    }
    assert reply_context == {
        "reply_to_display_name": "上条消息作者",
        "reply_excerpt": "上一条可见文本。",
    }
    assert "source_refs" not in supervisor_trace["dispatched"][0]
    for raw_id in (
        "raw-character-global-id",
        "raw-mentioned-platform-id",
        "raw-mentioned-global-id",
        "raw-reply-message-id",
        "raw-service-reply-message-id",
        "raw-row-id",
        "raw-message-id",
    ):
        assert raw_id not in rendered_payload


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
        current_timestamp_utc="2026-04-28T04:00:00+00:00",
    )

    assert prompt_doc["user_state_updates"][0]["age_hint"] == "~3h ago"


def test_content_plan_prompt_owns_topic_admission_decision() -> None:
    """Topic-admission decisions belong to L3 content plan, not dialog."""

    prompt = l3_module._CONTENT_PLAN_AGENT_PROMPT

    assert "你决定本轮可见台词需要承载的内容" in prompt
    assert "`semantic_content` 只承载用户可见台词需要说出的内容" in prompt
    assert "`semantic_content` 是下游可见内容的首要来源" in prompt
    assert "需要说出的事实、问题、代码、例子、边界、拒绝或下一步必须已经写在计划值里" in prompt
    assert "dialog" not in prompt.lower()
