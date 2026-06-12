"""Deterministic tests for split L2d action selection and materialization."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.action_spec.registry import (
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
)
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_chain_core.action_selection_prompt import (
    ACTION_ROUTER_PROMPT,
)
from kazusa_ai_chatbot.cognition_chain_core.stages import l2d as l2d_module
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_cognition_actions as action_connector,
)
from kazusa_ai_chatbot.self_cognition import models as self_cognition_models
from kazusa_ai_chatbot.self_cognition import runner as self_cognition_runner
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


class _FakeLLM:
    """Capture the L2d prompt call and return one configured JSON payload."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        self.messages = messages
        response = SimpleNamespace(content=self.content)
        return response


async def _select_and_materialize(state: dict) -> dict:
    selected = await l2d_module.select_semantic_actions(state)
    action_specs = action_connector.materialize_semantic_action_requests(
        selected.get("semantic_action_requests", []),
        state,
    )
    result = {
        **selected,
        "action_specs": action_specs,
    }
    return result


def _episode() -> dict:
    storage_timestamp_utc = "2026-05-15T21:00:00+00:00"
    turn_clock = build_turn_clock_from_storage_utc(storage_timestamp_utc)
    episode = build_text_chat_cognitive_episode(
        episode_id="user_message:debug:raw-channel-123:raw-message-456",
        percept_id="user_message:debug:raw-channel-123:raw-message-456:dialog_text:0",
        storage_timestamp_utc=storage_timestamp_utc,
        local_time_context=turn_clock["local_time_context"],
        user_input="Please handle the old spice promise naturally.",
        platform="debug",
        platform_channel_id="raw-channel-123",
        channel_type="private",
        platform_message_id="raw-message-456",
        platform_user_id="platform-user-raw",
        global_user_id="global-user-raw",
        user_name="Test User",
        active_turn_platform_message_ids=["raw-message-456"],
        active_turn_conversation_row_ids=["conversation-row-raw"],
        debug_modes={},
        output_mode="visible_reply",
        target_addressed_user_ids=[],
        target_broadcast=False,
    )
    return episode


def _state() -> dict:
    storage_timestamp_utc = "2026-05-15T21:00:00+00:00"
    turn_clock = build_turn_clock_from_storage_utc(storage_timestamp_utc)
    return {
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": turn_clock["local_time_context"],
        "cognitive_episode": _episode(),
        "channel_type": "private",
        "decontexualized_input": (
            "The user asks whether the active character should deal with an "
            "old promise."
        ),
        "internal_monologue": (
            "The spice promise is overdue, but it should be handled naturally."
        ),
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "The final decision is to handle the promise without forcing it.",
        "emotional_appraisal": "calm",
        "interaction_subtext": "low pressure",
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "acceptance": "allow",
            "stance_bias": "confirm",
        },
        "social_distance": "friendly but not intrusive",
        "emotional_intensity": "quiet and low pressure",
        "vibe_check": "relaxed daily conversation",
        "relational_dynamic": "stable trust with room to wait",
        "available_action_affordances": [
            {
                "capability": "speak",
                "available": True,
                "visibility": "public",
                "semantic_input_summary": [
                    "Use when the character wants a text surface to exist.",
                ],
                "output_kind": "semantic_action_request",
            },
            {
                "capability": MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
                "available": True,
                "visibility": "private",
                "semantic_input_summary": [
                    "Use when active commitments need semantic lifecycle review.",
                ],
                "output_kind": "semantic_action_request",
            },
            {
                "capability": "trigger_future_cognition",
                "available": True,
                "visibility": "private",
                "semantic_input_summary": [
                    "Use when the character wants a later private cognition cycle.",
                ],
                "output_kind": "semantic_action_request",
            },
            {
                "capability": "background_work_request",
                "available": True,
                "visibility": "private",
                "semantic_input_summary": [
                    "Use only for accepted bounded background text work.",
                ],
                "output_kind": "semantic_action_request",
            },
        ],
        "max_action_requests": 3,
        "max_resolver_requests": 3,
        "background_work_output_char_limit": 4000,
        "rag_result": {
            "answer": "The active commitment is overdue.",
            "user_image": {
                "user_memory_context": {
                    "active_commitments": [
                        {
                            "unit_id": "promise-001",
                            "fact": "Reveal the spice answer.",
                            "due_at": "2026-05-07T00:00:00+00:00",
                            "due_state": "past_due",
                        }
                    ]
                }
            },
            "memory_evidence": [
                {
                    "summary": "The spice promise is past due.",
                    "source_id": "promise-001",
                }
            ],
        },
        "conversation_progress": {
            "current_thread": "hardware discussion",
            "next_affordances": ["wait for a natural pause"],
        },
    }


def _self_cognition_commitment_case() -> dict:
    """Build a self-cognition source case for one active commitment target."""

    case = {
        "case_name": self_cognition_models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "active_commitment:promise-001:2026-05-07T00:00:00+00:00",
        "idle_timestamp_utc": "2026-05-15T21:00:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-14T21:00:00+00:00",
        "trigger_kind": (
            self_cognition_models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK
        ),
        "semantic_due_state": self_cognition_models.DUE_STATE_PAST_DUE,
        "actionability": "past_due_commitment_contact_socially_available",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "group",
            "user_id": "global-user-001",
        },
        "source_refs": [
            {
                "source_kind": "user_memory_unit",
                "source_id": "promise-001",
                "due_at": "2026-05-07T00:00:00+00:00",
                "summary": (
                    "The active character promised to reveal the spice answer."
                ),
            }
        ],
        "visible_context": [],
    }
    return case


def _speak_request(reason: str = "A visible text surface is needed.") -> dict:
    return {
        "capability": "speak",
        "decision": "visible_reply",
        "detail": "Use a natural and brief tone.",
        "reason": reason,
    }


def _memory_lifecycle_request() -> dict:
    return {
        "capability": MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "decision": "review_needed",
        "detail": "Review whether active commitment lifecycle changed.",
        "reason": "The current turn may affect active commitment lifecycle.",
    }


def _future_cognition_request() -> dict:
    return {
        "capability": "trigger_future_cognition",
        "decision": "future_self_check",
        "detail": "Check whether the topic has a natural pause.",
        "reason": "The character wants to revisit this later without speaking now.",
    }


def test_action_selection_payload_is_prompt_safe() -> None:
    """L2d should receive a JSON string, not raw transport internals."""

    action_context = l2d_module.build_action_selection_payload_text(_state())
    payload = json.loads(action_context)
    serialized = action_context.lower()

    assert isinstance(payload, dict)
    assert "source" in payload
    assert payload["source"]["trigger_source"] == "user_message"
    assert payload["source"]["output_mode"] == "visible_reply"
    assert payload["source"]["channel_type"] == "private"
    assert payload["cognition"]["social_distance"] == "friendly but not intrusive"
    assert payload["cognition"]["emotional_intensity"] == "quiet and low pressure"
    assert payload["cognition"]["vibe_check"] == "relaxed daily conversation"
    assert payload["cognition"]["relational_dynamic"] == "stable trust with room to wait"
    assert "Reveal the spice answer." in action_context
    assert "send_message" not in serialized
    for forbidden in (
        "raw-channel-123",
        "raw-message-456",
        "platform-user-raw",
        "global-user-raw",
        "handler_id",
        "dispatcher.send_message",
        "l3_text",
        "user_memory_units",
        "mongodb",
        "mongo",
        "credential",
        "platform_channel_id",
        "channel_id",
        "schema_version",
        "unit_id",
        "source_id",
        "promise-001",
    ):
        assert forbidden not in serialized


def test_self_cognition_source_ref_binds_active_commitment_context() -> None:
    """Self-cognition source refs should become deterministic L2d targets."""

    state = self_cognition_runner._build_cognition_state(
        _self_cognition_commitment_case(),
        "rendered self-cognition packet",
    )

    commitments = (
        state["rag_result"]["user_image"]["user_memory_context"][
            "active_commitments"
        ]
    )

    assert commitments == [
        {
            "unit_id": "promise-001",
            "fact": "The active character promised to reveal the spice answer.",
            "summary": "The active character promised to reveal the spice answer.",
            "due_at": "2026-05-07 12:00",
            "due_state": self_cognition_models.DUE_STATE_PAST_DUE,
            "status": "active",
        }
    ]


def test_action_selection_prompt_follows_cognition_prompt_structure() -> None:
    """The L2d prompt should follow the established cognition prompt pattern."""

    prompt = ACTION_ROUTER_PROMPT

    for required_section in (
        "# 语言政策",
        "# 可选动作",
        "# 选择流程",
        "# 未来认知判断",
        "# 输入格式",
        "# 输出格式",
    ):
        assert required_section in prompt
    assert "你是角色的语义行动路由层" in prompt
    assert "JSON 只包含语义上下文" in prompt
    assert "行动请求只描述我想做什么" in prompt
    assert "解析请求只描述下一步需要什么证据、事实、澄清或审批" in prompt
    assert "内心独白是证据，不是动作" in prompt
    assert "只返回合法 JSON 字符串" in prompt
    assert "不要从本提示词推断固定能力清单" in prompt
    assert "capabilities.resolver_affordances" in prompt
    assert "capabilities.action_affordances" in prompt
    assert "只有线索或只有未确认候选时" in prompt
    assert "不得把目标属性升格为已确认事实" in prompt
    assert "具体当前外部断言" in prompt
    assert "`send_message`" not in prompt
    assert "用户消息是一个 JSON 对象" in prompt
    assert "具体新信息" in prompt
    assert "具体问题、任务或承诺" in prompt
    assert "action_requests" in prompt
    assert "resolver_capability_requests" in prompt
    assert "work_seed" in prompt
    assert "resolver_capability_request.v1" not in prompt
    assert "resolver_goal_progress.v1" not in prompt
    assert "schema_version" not in prompt
    assert "rag_evidence" not in prompt
    assert "web_evidence" not in prompt
    assert "human_clarification" not in prompt
    assert "approval_preparation" not in prompt
    assert "self_goal_resolution" not in prompt
    assert "memory_lifecycle_update" not in prompt
    assert "background_work_request" not in prompt
    assert "trigger_future_cognition" not in prompt
    assert "available_capabilities" not in prompt
    assert "scheduled_recall" not in prompt
    assert "system_probe" not in prompt
    assert "action_spec.v1" not in prompt
    assert "action_target" not in prompt
    assert "action_spec" not in prompt
    assert "continuation" not in prompt
    assert "l3_text" not in prompt
    assert "handler_id" not in prompt
    assert "final_l2" not in prompt
    assert "trigger_context" not in prompt
    assert "social_context_appraisal" not in prompt
    assert "L2c1" not in prompt
    assert "L2c2" not in prompt
    assert "小判断例" not in prompt
    assert "5090 能跑什么人工智能模型" not in prompt
    assert "resume_id" not in prompt
    assert "系统会绑定当前 active pending row" in prompt


def test_action_selection_payload_includes_group_engagement_context() -> None:
    """Group self-cognition should expose bounded engagement before L2d."""

    state = _state()
    state["channel_type"] = "group"
    state["cognitive_episode"]["trigger_source"] = "internal_thought"
    state["cognitive_episode"]["input_sources"] = ["internal_monologue"]
    state["cognitive_episode"]["output_mode"] = "preview"
    state["group_engagement_action_context"] = {
        "engagement_guidelines": [
            "Join clear direct group openings.",
            "Stay with the current group topic.",
        ],
        "confidence": "high",
    }

    action_context = l2d_module.build_action_selection_payload_text(state)

    payload = json.loads(action_context)
    assert "group_engagement" in payload
    ge = payload["group_engagement"]
    assert "Join clear direct group openings." in ge["engagement_guidelines"]
    assert "Stay with the current group topic." in ge["engagement_guidelines"]
    assert ge["confidence"] == "high"
    assert "Do not leak this to L2d." not in action_context


def test_action_selection_payload_omits_group_engagement_for_private() -> None:
    """Private chat should not receive group-channel engagement guidance."""

    state = _state()
    state["group_engagement_action_context"] = {
        "engagement_guidelines": ["Join clear direct group openings."],
        "confidence": "high",
    }

    action_context = l2d_module.build_action_selection_payload_text(state)

    payload = json.loads(action_context)
    assert "group_engagement" not in payload


def test_action_selection_hides_lifecycle_without_active_commitments() -> None:
    """Lifecycle should not be offered when no active commitment exists."""

    state = _state()
    state["rag_result"]["user_image"]["user_memory_context"][
        "active_commitments"
    ] = []

    action_context = l2d_module.build_action_selection_payload_text(state)

    payload = json.loads(action_context)
    assert payload["evidence"]["active_commitment_clues"] == []
    assert "memory_lifecycle_update" not in action_context


def test_action_selection_offers_lifecycle_route_for_multiple_commitments() -> None:
    """Multiple active commitments should still allow specialist routing."""

    state = _state()
    state["rag_result"]["user_image"]["user_memory_context"][
        "active_commitments"
    ] = [
        {
            "unit_id": "promise-001",
            "fact": "Reveal the spice answer.",
            "due_at": "2026-05-07T00:00:00+00:00",
            "due_state": "past_due",
        },
        {
            "unit_id": "promise-002",
            "fact": "Check the tea result.",
            "due_at": "2026-05-08T00:00:00+00:00",
            "due_state": "past_due",
        },
    ]

    action_context = l2d_module.build_action_selection_payload_text(state)

    payload = json.loads(action_context)
    assert len(payload["evidence"]["active_commitment_clues"]) == 2
    assert "promise-001" not in action_context
    assert "promise-002" not in action_context


@pytest.mark.asyncio
async def test_action_selection_ignores_lifecycle_target_fields_from_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown target fields from L2d must not become DB lifecycle params."""

    state = _state()
    state["rag_result"]["user_image"]["user_memory_context"][
        "active_commitments"
    ] = [
        {
            "unit_id": "promise-001",
            "fact": "Reveal the spice answer.",
            "due_at": "2026-05-07T00:00:00+00:00",
            "due_state": "past_due",
        },
        {
            "unit_id": "promise-002",
            "fact": "Check the tea result.",
            "due_at": "2026-05-08T00:00:00+00:00",
            "due_state": "past_due",
        },
    ]
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [
            {
                "capability": MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
                "decision": "review_needed",
                "detail": "Review whether a promise changed.",
                "reason": "A commitment may have changed.",
                "unit_id": "promise-001",
                "target_alias": "commitment_1",
                "lifecycle_decision": "fulfilled",
            },
        ],
    }))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(state)

    assert len(result["action_specs"]) == 1
    action_spec = result["action_specs"][0]
    serialized = json.dumps(action_spec, ensure_ascii=False)
    assert action_spec["kind"] == MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    assert action_spec["target"] == {
        "schema_version": "action_target.v1",
        "target_kind": "cognitive_episode",
        "target_id": None,
        "owner": "memory_lifecycle_specialist",
        "scope": {"unit_type": "active_commitment"},
    }
    assert action_spec["params"] == {
        "review_kind": "active_commitment_lifecycle",
        "detail": "Review whether a promise changed.",
    }
    assert "promise-001" not in serialized
    assert "target_alias" not in serialized
    assert "lifecycle_decision" not in serialized


@pytest.mark.asyncio
async def test_future_cognition_materialization_binds_trusted_source_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future cognition source scope is copied by code, not emitted by L2d."""

    state = _state()
    state.update(
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "global_user_id": "673225019",
            "platform_bot_id": "bot-001",
            "character_profile": {"name": "TestCharacter"},
        }
    )
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [_future_cognition_request()],
    }))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(state)

    action_spec = result["action_specs"][0]
    assert action_spec["kind"] == "trigger_future_cognition"
    assert action_spec["continuation"] == {
        "schema_version": "action_continuation.v1",
        "mode": "scheduled_followup",
        "episode_type": "self_cognition",
        "max_depth": 1,
        "include_result_as": "scheduled_event",
    }
    assert action_spec["target"]["scope"] == {
        "episode_type": "self_cognition",
        "source_platform": "qq",
        "source_channel_id": "54369546",
        "source_channel_type": "group",
        "source_user_id": "673225019",
        "source_platform_bot_id": "bot-001",
        "source_character_name": "TestCharacter",
    }
    human_payload = fake_llm.messages[1].content
    assert "54369546" not in human_payload
    assert "bot-001" not in human_payload


@pytest.mark.asyncio
async def test_future_cognition_uses_own_detail_as_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future cognition should carry its own distinct next-cycle objective."""

    speak_request = {
        "capability": "speak",
        "decision": "visible_reply",
        "detail": '回应用户关于 5090 AI 模型运行能力的询问',
        "reason": "The user asked for a visible acknowledgement first.",
    }
    future_request = {
        "capability": "trigger_future_cognition",
        "decision": "future_self_check",
        "detail": '需要查阅目前已知的泄露参数、预测数据以及社区讨论的兼容性信息',
        "reason": "The character needs a later private cognition cycle.",
    }
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [speak_request, future_request],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(_state())

    future_specs = [
        spec for spec in result["action_specs"]
        if spec["kind"] == "trigger_future_cognition"
    ]
    assert len(future_specs) == 1
    params = future_specs[0]["params"]
    assert params["continuation_objective"] == (
        '需要查阅目前已知的泄露参数、预测数据以及社区讨论的兼容性信息'
    )
    assert "context_summary" not in params


@pytest.mark.asyncio
async def test_scheduled_future_cognition_cannot_chain_another_future_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A due future-cognition cycle must not schedule itself again."""

    state = _state()
    state["cognitive_episode"]["trigger_source"] = "internal_thought"
    state["conversation_progress"] = {
        "source": "scheduled_future_cognition",
        "continuation_objective": (
            "Check whether the earlier follow-up is still useful."
        ),
    }
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [_future_cognition_request()],
    }))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(state)

    assert result["action_specs"] == []


@pytest.mark.asyncio
async def test_action_selection_accepts_multiple_valid_action_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2d can select more than one independent action in one cognition cycle."""

    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [
            _speak_request(),
            _memory_lifecycle_request(),
            _future_cognition_request(),
        ],
    }))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(_state())

    assert [spec["kind"] for spec in result["action_specs"]] == [
        "speak",
        MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "trigger_future_cognition",
    ]
    lifecycle_spec = result["action_specs"][1]
    assert lifecycle_spec["target"]["target_kind"] == "cognitive_episode"
    assert lifecycle_spec["target"]["owner"] == "memory_lifecycle_specialist"
    assert lifecycle_spec["params"] == {
        "review_kind": "active_commitment_lifecycle",
        "detail": "Review whether active commitment lifecycle changed.",
    }
    human_context = fake_llm.messages[1].content
    human_payload = json.loads(human_context)
    assert isinstance(human_payload, dict)
    assert "trigger_context" not in human_context
    assert "available_capabilities" not in human_context


@pytest.mark.asyncio
async def test_action_selection_drops_misplaced_resolver_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver capability names in action rows are contract drift."""

    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [
            {
                "capability": "web_evidence",
                "decision": "retrieve current facts",
                "detail": "检索奥克兰 CBD 预算内晚间计划需要的当前事实。",
                "reason": "当前回答还缺少外部证据。",
            },
        ],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(_state())

    assert result["action_specs"] == []
    assert result["resolver_capability_requests"] == []


@pytest.mark.asyncio
async def test_action_selection_drops_misplaced_terminal_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal action names in resolver rows are contract drift."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_capability_requests": [
            {
                "schema_version": "resolver_capability_request.v1",
                "capability_kind": "speak",
                "objective": "回应当前直接提问。",
                "reason": "前序认知已经决定可以可见回应。",
                "priority": "now",
            },
        ],
        "action_requests": [],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(_state())

    assert result["resolver_capability_requests"] == []
    assert result["action_specs"] == []


@pytest.mark.asyncio
async def test_action_selection_returns_valid_goal_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2d should be able to emit the goal checklist beside resolver work."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_capability_requests": [
            {
                "schema_version": "resolver_capability_request.v1",
                "capability_kind": "web_evidence",
                "objective": "检索当前营业证据。",
                "reason": "需要当前事实才能推荐。",
                "priority": "now",
            },
        ],
        "resolver_goal_progress": {
            "schema_version": "resolver_goal_progress.v1",
            "original_goal": "安排两小时低预算计划。",
            "current_focus": "先确认当前营业事实。",
            "deliverables": [
                {
                    "description": "晚餐候选和证据边界",
                    "status": "partial",
                    "note": "需要当前事实。",
                },
                {
                    "description": "散步路线和时间切分",
                    "status": "pending",
                    "note": "最终回答必须覆盖。",
                },
            ],
            "missing_user_inputs": [],
            "evidence_dependencies": ["当前营业证据"],
            "attempted_paths": [],
            "source_backed_facts": ["用户给出地点和预算"],
            "assumptions_or_inferences": [],
            "blockers": [],
            "final_response_requirements": [
                "覆盖晚餐、路线、时间切分和核实清单",
            ],
        },
        "action_requests": [],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(_state())

    assert result["resolver_goal_progress"]["original_goal"] == (
        "安排两小时低预算计划。"
    )
    assert result["resolver_goal_progress"]["deliverables"][1]["status"] == (
        "pending"
    )
    assert result["action_specs"] == []


@pytest.mark.asyncio
async def test_action_selection_derives_visible_requirements_from_open_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open L2d deliverables should reach L3 even when requirements are empty."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_capability_requests": [],
        "resolver_goal_progress": {
            "schema_version": "resolver_goal_progress.v1",
            "original_goal": "安排两小时低预算计划。",
            "current_focus": "基于现有证据给最佳努力计划。",
            "deliverables": [
                {
                    "description": "散步路线和时间切分",
                    "status": "pending",
                    "note": "最终回答必须覆盖。",
                },
                {
                    "description": "已确认约束复述",
                    "status": "satisfied",
                    "note": "地点和预算已确认。",
                },
            ],
            "missing_user_inputs": [],
            "evidence_dependencies": [],
            "attempted_paths": [],
            "source_backed_facts": ["用户给出地点和预算"],
            "assumptions_or_inferences": [],
            "blockers": [],
            "final_response_requirements": [],
        },
        "action_requests": [
            _speak_request("基于现有证据给用户可见计划。"),
        ],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(_state())

    requirements = result["resolver_goal_progress"][
        "final_response_requirements"
    ]
    assert requirements == [
        "散步路线和时间切分：最终回答必须覆盖。",
    ]
    assert result["action_specs"][0]["kind"] == "speak"


@pytest.mark.asyncio
async def test_action_selection_completes_partial_visible_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing requirements should be extended for uncovered open deliverables."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_capability_requests": [],
        "resolver_goal_progress": {
            "schema_version": "wrong-version-from-model",
            "original_goal": "安排两小时低预算计划。",
            "current_focus": "给最终计划。",
            "deliverables": [
                {
                    "description": "晚餐候选和证据边界",
                    "status": "partial",
                    "note": "已有候选但营业状态要 caveat。",
                },
                {
                    "description": "散步路线和时间切分",
                    "status": "pending",
                    "note": "最终回答必须覆盖。",
                },
            ],
            "missing_user_inputs": [],
            "evidence_dependencies": [],
            "attempted_paths": [],
            "source_backed_facts": [],
            "assumptions_or_inferences": [],
            "blockers": [],
            "final_response_requirements": [
                "晚餐候选和证据边界：已有候选但营业状态要 caveat。",
            ],
        },
        "action_requests": [
            _speak_request("基于现有证据给用户可见计划。"),
        ],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(_state())

    goal_progress = result["resolver_goal_progress"]
    assert goal_progress["schema_version"] == "resolver_goal_progress.v1"
    assert goal_progress["final_response_requirements"] == [
        "晚餐候选和证据边界：已有候选但营业状态要 caveat。",
        "散步路线和时间切分：最终回答必须覆盖。",
    ]


@pytest.mark.asyncio
async def test_action_selection_binds_pending_resolution_to_active_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2d should judge pending semantics while Python binds the row id."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_pending_resolution": {
            "decision": "answered",
            "reason": "用户补充了缺失地点。",
        },
        "action_requests": [
            _speak_request("继续完成原始目标。"),
        ],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)
    state = _state()
    state["pending_resolver_resume"] = {
        "schema_version": "resolver_pending_resume.v1",
        "resume_id": "resolver-pending-city-1",
        "capability_kind": "human_clarification",
        "status": "waiting_for_user",
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "global_user_id": "global-user-123",
        "source_message_id": "message-123",
        "prompt_safe_original_goal": "安排两小时低预算计划。",
        "prompt_safe_question": "奥克兰哪个区域？",
        "prompt_safe_approval_summary": "",
        "created_at_utc": "2026-05-15T21:00:00+00:00",
        "expires_at_utc": "2026-05-16T21:00:00+00:00",
    }

    result = await _select_and_materialize(state)

    assert result["resolver_pending_resolution"] == {
        "schema_version": "resolver_pending_resolution.v1",
        "resume_id": "resolver-pending-city-1",
        "decision": "answered",
        "reason": "用户补充了缺失地点。",
    }
    human_context = fake_llm.messages[1].content
    assert "pending_resolver_resume:" in human_context
    assert "安排两小时低预算计划。" in human_context
    assert "奥克兰哪个区域？" in human_context
    assert "resume_id" not in human_context
    assert "resolver-pending-city-1" not in human_context
    assert "expires_at_utc" not in human_context
    assert "channel-123" not in human_context
    assert "global-user-123" not in human_context


@pytest.mark.asyncio
async def test_action_selection_ignores_model_supplied_pending_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opaque pending ids from L2d should never choose the row to update."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_pending_resolution": {
            "schema_version": "resolver_pending_resolution.v1",
            "resume_id": "wrong-model-id",
            "decision": "approved",
            "reason": "用户同意了待审批动作。",
        },
        "action_requests": [
            _speak_request("说明审批后的结果。"),
        ],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)
    state = _state()
    state["pending_resolver_resume"] = {
        "schema_version": "resolver_pending_resume.v1",
        "resume_id": "resolver-pending-approval-1",
        "capability_kind": "approval_preparation",
        "status": "waiting_for_approval",
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "global_user_id": "global-user-123",
        "source_message_id": "message-123",
        "prompt_safe_original_goal": "准备提醒但先等待确认。",
        "prompt_safe_question": "",
        "prompt_safe_approval_summary": "提醒用户晚上复查。",
        "created_at_utc": "2026-05-15T21:00:00+00:00",
        "expires_at_utc": "2026-05-16T21:00:00+00:00",
    }

    result = await _select_and_materialize(state)

    assert result["resolver_pending_resolution"] == {
        "schema_version": "resolver_pending_resolution.v1",
        "resume_id": "resolver-pending-approval-1",
        "decision": "approved",
        "reason": "用户同意了待审批动作。",
    }
    serialized = json.dumps(result, ensure_ascii=False)
    assert "wrong-model-id" not in serialized


@pytest.mark.asyncio
async def test_action_selection_keeps_action_when_pending_request_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated pending resolver request should not hide a valid answer."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_capability_requests": [
            {
                "schema_version": "resolver_capability_request.v1",
                "capability_kind": "approval_preparation",
                "objective": "再次准备同一个提醒审批。",
                "reason": "模型重复了同一个待审批能力。",
                "priority": "now",
            },
        ],
        "action_requests": [
            _speak_request("explain the approval preview"),
        ],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)
    state = _state()
    state["pending_resolver_resume"] = {
        "schema_version": "resolver_pending_resume.v1",
        "resume_id": "resolver-pending-approval-1",
        "capability_kind": "approval_preparation",
        "status": "waiting_for_approval",
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "global_user_id": "global-user-123",
        "source_message_id": "message-123",
        "prompt_safe_original_goal": "Explain then wait for approval.",
        "prompt_safe_question": "",
        "prompt_safe_approval_summary": "Explain reminder plan first.",
        "created_at_utc": "2026-05-15T21:00:00+00:00",
        "expires_at_utc": "2026-05-16T21:00:00+00:00",
    }

    result = await _select_and_materialize(state)

    assert result["resolver_capability_requests"] == []
    assert [spec["kind"] for spec in result["action_specs"]] == ["speak"]
    assert result["action_specs"][0]["reason"] == "explain the approval preview"


@pytest.mark.asyncio
async def test_action_selection_recovers_pending_capability_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending resolver answer should become text even with the wrong name."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_capability_requests": [
            {
                "schema_version": "resolver_capability_request.v1",
                "capability_kind": "approval_preparation",
                "objective": "Repeat the same approval preparation.",
                "reason": "The model repeated the active pending capability.",
                "priority": "now",
            },
        ],
        "action_requests": [
            {
                "capability": "approval_preparation",
                "decision": "",
                "detail": "",
                "reason": "explain the pending approval preview",
            },
        ],
    }))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)
    state = _state()
    state["pending_resolver_resume"] = {
        "schema_version": "resolver_pending_resume.v1",
        "resume_id": "resolver-pending-approval-1",
        "capability_kind": "approval_preparation",
        "status": "waiting_for_approval",
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "global_user_id": "global-user-123",
        "source_message_id": "message-123",
        "prompt_safe_original_goal": "Explain then wait for approval.",
        "prompt_safe_question": "",
        "prompt_safe_approval_summary": "Explain reminder plan first.",
        "created_at_utc": "2026-05-15T21:00:00+00:00",
        "expires_at_utc": "2026-05-16T21:00:00+00:00",
    }

    result = await _select_and_materialize(state)

    assert result["resolver_capability_requests"] == []
    assert [spec["kind"] for spec in result["action_specs"]] == ["speak"]
    surface_requirements = result["action_specs"][0]["params"][
        "surface_requirements"
    ]
    assert surface_requirements == {
        "decision": "visible_reply",
        "detail": "Explain reminder plan first.",
    }


@pytest.mark.asyncio
async def test_action_selection_surfaces_repeated_pending_without_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated pending resolver request should still reach L3 text."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_capability_requests": [
            {
                "schema_version": "resolver_capability_request.v1",
                "capability_kind": "approval_preparation",
                "objective": "Repeat the same approval preparation.",
                "reason": "The model repeated the active pending capability.",
                "priority": "now",
            },
        ],
        "action_requests": [],
    }))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)
    state = _state()
    state["pending_resolver_resume"] = {
        "schema_version": "resolver_pending_resume.v1",
        "resume_id": "resolver-pending-approval-1",
        "capability_kind": "approval_preparation",
        "status": "waiting_for_approval",
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "global_user_id": "global-user-123",
        "source_message_id": "message-123",
        "prompt_safe_original_goal": "Explain then wait for approval.",
        "prompt_safe_question": "",
        "prompt_safe_approval_summary": "Explain reminder plan first.",
        "created_at_utc": "2026-05-15T21:00:00+00:00",
        "expires_at_utc": "2026-05-16T21:00:00+00:00",
    }

    result = await _select_and_materialize(state)

    assert result["resolver_capability_requests"] == []
    assert [spec["kind"] for spec in result["action_specs"]] == ["speak"]
    surface_requirements = result["action_specs"][0]["params"][
        "surface_requirements"
    ]
    assert surface_requirements == {
        "decision": "visible_reply",
        "detail": "Explain reminder plan first.",
    }


@pytest.mark.asyncio
async def test_action_selection_drops_invalid_specs_and_caps_valid_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed rows should be skipped and valid rows capped before graph merge."""

    invalid_request = {"capability": "speak"}
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [
            invalid_request,
            _speak_request("first"),
            _speak_request("second"),
            _memory_lifecycle_request(),
            _speak_request("fourth"),
        ],
    }))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", fake_llm)

    result = await _select_and_materialize(_state())

    assert len(result["action_specs"]) == 3
    assert [spec["reason"] for spec in result["action_specs"]] == [
        "first",
        "second",
        "The current turn may affect active commitment lifecycle.",
    ]
