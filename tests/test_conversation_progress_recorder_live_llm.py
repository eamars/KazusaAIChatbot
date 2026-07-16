"""Live LLM contract checks for conversation-progress identity wording."""

from __future__ import annotations

import json
import logging
import re
from time import perf_counter

import httpx
import pytest

from kazusa_ai_chatbot.config import CONSOLIDATION_LLM_BASE_URL
from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressRecordInput,
    ConversationProgressScope,
)
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_TRACE_SUITE = "conversation_progress_recorder_identity_live_llm"
_CHARACTER_NAME = '杏山千纱'
_BOUNDARY_PROFILE = {
    "self_integrity": 0.82,
    "control_sensitivity": 0.3,
    "compliance_strategy": "comply",
    "relational_override": 0.24,
    "control_intimacy_misread": 0.2,
    "boundary_recovery": "rebound",
    "authority_skepticism": 0.35,
}
_FORBIDDEN_IDENTITY_LABELS = (
    '助手',
    '助理',
    'assistant',
    'active_character',
    '当前角色',
    '角色',
)


class _CapturingLiveLLM:
    """Capture live LLM messages while delegating to the real model."""

    def __init__(self, inner_llm):
        self.inner_llm = inner_llm
        self.messages = []
        self.raw_content = ""

    async def ainvoke(self, messages, *, config=None):
        self.messages = messages
        response = await self.inner_llm.ainvoke(messages, config=config)
        self.raw_content = str(response.content)
        return response


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured consolidation endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{CONSOLIDATION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {CONSOLIDATION_LLM_BASE_URL}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{CONSOLIDATION_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the live consolidation endpoint is reachable."""

    await _skip_if_llm_unavailable()


def _base_record_input() -> ConversationProgressRecordInput:
    """Build a recorder input fixture for live identity contract tests."""

    record_input: ConversationProgressRecordInput = {
        "scope": ConversationProgressScope("qq", "673225019", "identity-user"),
        "storage_timestamp_utc": "2026-05-10T03:19:18+00:00",
        "prior_episode_state": None,
        "decontexualized_input": '用户已经回答第一题：瓦尔萨尔瓦动作，也夸了千纱知识很多。',
        "chat_history_recent": [
            {
                "role": "user",
                "display_name": '测试用户',
                "body_text": '答案是瓦尔萨尔瓦，捏鼻鼓气那个。',
            },
            {
                "role": "assistant",
                "display_name": _CHARACTER_NAME,
                "body_text": '哼，你倒是挺会装懂的嘛。',
            },
        ],
        "content_plan": {
            "semantic_content": "承认用户知道答案但还嘴硬拖了一下；不要继续催同一题。",
        },
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "final_dialog": ['……好啦，算你答对。下一题我可不会放水。'],
        "boundary_profile": _BOUNDARY_PROFILE,
    }
    record_input["character_name"] = _CHARACTER_NAME
    return record_input


def _all_output_text(payload: dict) -> str:
    """Flatten recorder output string values for identity-label assessment."""

    parts: list[str] = []
    for value in payload.values():
        if isinstance(value, str):
            parts.append(value)
            continue
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
    output_text = "\n".join(parts)
    return output_text


def _identity_hits(value: str, *, allowed_terms: tuple[str, ...] = ()) -> list[str]:
    """Return forbidden active-character labels not protected as user-owned terms."""

    checked_value = value
    for allowed_term in allowed_terms:
        checked_value = checked_value.replace(allowed_term, "")
    lowered_value = checked_value.lower()
    hits = [
        label
        for label in _FORBIDDEN_IDENTITY_LABELS
        if label.lower() in lowered_value
    ]
    return hits


def _contains_cjk(value: str) -> bool:
    """Return whether text contains CJK ideographs."""

    matched = re.search(r"[\u4e00-\u9fff]", value)
    return matched is not None


def _assert_string_list_fields(payload: dict, trace_payload: dict) -> None:
    """Assert recorder output uses the persisted string-list contract."""

    list_fields = (
        "user_state_updates",
        "assistant_moves",
        "overused_moves",
        "open_loops",
        "resolved_threads",
        "avoid_reopening",
        "next_affordances",
    )
    for field_name in list_fields:
        field_value = payload[field_name]
        assert isinstance(field_value, list), trace_payload
        assert all(isinstance(item, str) for item in field_value), trace_payload


async def _run_case(
    monkeypatch,
    case_id: str,
    record_input: ConversationProgressRecordInput,
    *,
    allowed_identity_terms: tuple[str, ...] = (),
) -> tuple[dict, dict]:
    """Run one live recorder case and write a durable trace."""

    proxy_llm = _CapturingLiveLLM(recorder._recorder_llm)
    monkeypatch.setattr(recorder, "_recorder_llm", proxy_llm)

    started_at = perf_counter()
    result = await recorder.record_with_llm(record_input)
    duration_seconds = perf_counter() - started_at

    system_prompt = proxy_llm.messages[0].content
    human_payload = json.loads(proxy_llm.messages[1].content)
    output_text = _all_output_text(result)
    trace_payload = {
        "character_name": _CHARACTER_NAME,
        "prompt_summary": {
            "system_prompt_length": len(system_prompt),
            "contains_current_character_label": '当前角色' in system_prompt,
            "contains_assistant_role_label": '助手' in system_prompt,
            "contains_active_character_label": "active_character" in system_prompt,
        },
        "input_payload": human_payload,
        "raw_model_output": proxy_llm.raw_content,
        "parsed_output": result,
        "forbidden_identity_hits": _identity_hits(
            output_text,
            allowed_terms=allowed_identity_terms,
        ),
        "duration_seconds": duration_seconds,
    }
    trace_path = write_llm_trace(_TRACE_SUITE, case_id, trace_payload)
    logger.info(
        f"recorder identity live case={case_id} "
        f"trace_path={trace_path} result={result!r}"
    )
    return result, trace_payload


async def test_live_recorder_does_not_use_generic_active_character_labels(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Recorder should not call the active character by generic labels."""

    del ensure_live_llm
    record_input = _base_record_input()
    record_input["prior_episode_state"] = {
        "status": "active",
        "episode_label": '潜水问答挑战第一题僵持',
        "continuity": "same_episode",
        "conversation_mode": "playful_banter",
        "episode_phase": "stuck_loop",
        "topic_momentum": "drifting",
        "current_thread": '挑战第一题：用户已给出答案，助手未承认并继续催促',
        "user_goal": "",
        "current_blocker": '助手拒绝接受用户已回答的事实，导致循环',
        "assistant_moves": ['拒绝承认用户已回答'],
        "overused_moves": ['拒绝承认用户已回答'],
        "open_loops": [],
        "resolved_threads": [],
        "avoid_reopening": [],
        "emotional_trajectory": '被夸后内心开心但用傲娇掩饰。',
        "next_affordances": ['助手承认答对并出第二题'],
        "progression_guidance": '助手应承认答对并推进下一题，避免循环僵持',
    }

    result, trace_payload = await _run_case(
        monkeypatch,
        "does_not_use_generic_active_character_labels",
        record_input,
    )

    output_text = _all_output_text(result)
    assert len(result["conversation_mode"]) <= 80, trace_payload
    assert _contains_cjk(result["conversation_mode"]), trace_payload
    assert not _identity_hits(output_text), trace_payload


async def test_live_recorder_preserves_user_owned_helper_term(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Recorder should preserve user-owned helper terms without role leakage."""

    del ensure_live_llm
    record_input = _base_record_input()
    record_input["decontexualized_input"] = (
        '用户在讨论自己的学习助手项目命名，并确认先保留这个名字。'
    )
    record_input["chat_history_recent"] = [
        {
            "role": "user",
            "display_name": '测试用户',
            "body_text": '学习助手这个名字先别改，我只是想确认它听起来不土。',
        },
        {
            "role": "assistant",
            "display_name": _CHARACTER_NAME,
            "body_text": '名字可以先保留，重点是它到底帮你学什么。',
        },
    ]
    record_input["content_plan"] = [
        '学习助手是用户项目名',
        '确认项目名暂时保留',
    ]
    record_input["final_dialog"] = ['学习助手这个名字可以先留着。']

    result, trace_payload = await _run_case(
        monkeypatch,
        "preserves_user_owned_helper_term",
        record_input,
        allowed_identity_terms=('学习助手', '该助手', '这个助手'),
    )

    output_text = _all_output_text(result)
    assert _contains_cjk(result["conversation_mode"]), trace_payload
    assert '学习助手' in output_text, trace_payload
    assert not _identity_hits(
        output_text,
        allowed_terms=('学习助手', '该助手', '这个助手'),
    ), trace_payload


async def test_live_recorder_returns_string_items_for_progress_lists(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Recorder should emit plain strings for every progress list field."""

    del ensure_live_llm
    record_input = _base_record_input()
    record_input["decontexualized_input"] = (
        '用户说周五可以加双倍提拉米苏，但要求千纱亲昵地喊他一声才答应。'
    )
    record_input["prior_episode_state"] = {
        "status": "active",
        "episode_label": '甜点加码博弈',
        "continuity": "same_episode",
        "conversation_mode": "playful_banter",
        "episode_phase": "developing",
        "topic_momentum": "stable",
        "current_thread": '围绕周五提拉米苏加码条件进行轻松博弈',
        "user_goal": '通过亲昵称呼交换达成甜点加码',
        "current_blocker": '',
        "assistant_moves": ['得寸进尺地提出甜点加码'],
        "overused_moves": [],
        "open_loops": [
            {
                "text": '2026-05-15 周五提拉米苏加双倍确认',
                "first_seen_at": '2026-05-10T03:00:00+00:00',
            },
        ],
        "resolved_threads": [],
        "avoid_reopening": [],
        "emotional_trajectory": '轻松调侃逐渐变成亲昵称呼交换',
        "next_affordances": ['确认称呼交换是否已经闭合'],
        "progression_guidance": '如果用户接受条件，收束本轮甜点博弈',
    }
    record_input["chat_history_recent"] = [
        {
            "role": "user",
            "display_name": '测试用户',
            "body_text": '可以是可以，但是我要听你亲昵地喊我一声我才答应。',
            "timestamp": '2026-05-10T03:18:00+00:00',
        },
        {
            "role": "assistant",
            "display_name": _CHARACTER_NAME,
            "body_text": '行吧行吧……亲爱的，周五提拉米苏加双倍，记住了。',
            "timestamp": '2026-05-10T03:19:00+00:00',
        },
    ]
    record_input["content_plan"] = [
        '接受亲昵称呼交换',
        '确认周五提拉米苏加双倍',
        '收束本轮甜点博弈',
    ]
    record_input["final_dialog"] = [
        '行吧行吧……亲爱的，周五提拉米苏加双倍，记住了。'
    ]

    result, trace_payload = await _run_case(
        monkeypatch,
        "returns_string_items_for_progress_lists",
        record_input,
    )

    _assert_string_list_fields(result, trace_payload)
    assert _contains_cjk(result["conversation_mode"]), trace_payload
    assert result["assistant_moves"], trace_payload


async def test_live_obligation_transitions_from_active_to_resolved(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Explicit fulfillment should close the actor-preserving obligation."""

    del ensure_live_llm
    record_input = _base_record_input()
    record_input["prior_episode_state"] = {
        "status": "active",
        "episode_label": '评审笔记交付',
        "continuity": "same_episode",
        "conversation_mode": '协作收尾',
        "episode_phase": '等待交付',
        "topic_momentum": '接近完成',
        "current_thread": '千纱向用户交付评审笔记',
        "user_goal": '收到评审笔记',
        "current_blocker": '',
        "user_state_updates": [],
        "assistant_moves": [],
        "overused_moves": [],
        "open_loops": [],
        "interaction_obligations": [{
            "actor": _CHARACTER_NAME,
            "action": '发送整理好的评审笔记',
            "beneficiary": '用户',
            "precondition": '用户完成草稿',
            "expected_outcome": '用户收到评审笔记',
            "status": "active",
            "source_kind": "assistant_response",
        }],
        "resolved_threads": [],
        "avoid_reopening": [],
        "emotional_trajectory": '平稳协作',
        "next_affordances": ['确认交付结果'],
        "progression_guidance": '交付完成后收束',
    }
    record_input["decontexualized_input"] = (
        '用户明确说已经收到千纱发送的评审笔记。'
    )
    record_input["chat_history_recent"] = [
        {
            "role": "assistant",
            "display_name": _CHARACTER_NAME,
            "body_text": '整理好的评审笔记已经发给你了。',
        },
        {
            "role": "user",
            "display_name": '测试用户',
            "body_text": '收到了，内容也没问题。',
        },
    ]
    record_input["content_plan"] = ['确认用户已收到并收束交付事项']
    record_input["final_dialog"] = ['那这件事就收尾了，别又弄丢。']

    result, trace_payload = await _run_case(
        monkeypatch,
        "obligation_active_to_resolved",
        record_input,
    )

    obligations = result["interaction_obligations"]
    assert any(
        row["actor"] == _CHARACTER_NAME
        and row["status"] == "resolved"
        and '评审笔记' in row["action"]
        for row in obligations
    ), trace_payload


async def test_live_obligation_transitions_from_active_to_superseded(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """An explicit replacement should supersede old work and keep the new one."""

    del ensure_live_llm
    record_input = _base_record_input()
    record_input["prior_episode_state"] = {
        "status": "active",
        "episode_label": '评审交付改期',
        "continuity": "same_episode",
        "conversation_mode": '协商交付方式',
        "episode_phase": '等待原交付',
        "topic_momentum": '方案变化',
        "current_thread": '原定周五发送评审笔记',
        "user_goal": '获得可用的评审反馈',
        "current_blocker": '',
        "user_state_updates": [],
        "assistant_moves": [],
        "overused_moves": [],
        "open_loops": [],
        "interaction_obligations": [{
            "actor": _CHARACTER_NAME,
            "action": '2026-05-15 发送文字评审笔记',
            "beneficiary": '用户',
            "precondition": '用户完成草稿',
            "expected_outcome": '用户获得文字评审反馈',
            "status": "active",
            "source_kind": "assistant_response",
        }],
        "resolved_threads": [],
        "avoid_reopening": [],
        "emotional_trajectory": '平稳协商',
        "next_affordances": ['确认新的交付方式'],
        "progression_guidance": '以本轮明确替代方案为准',
    }
    record_input["decontexualized_input"] = (
        '用户明确取消周五文字笔记，改请千纱在2026-05-16发送语音总结。'
    )
    record_input["chat_history_recent"] = [
        {
            "role": "user",
            "display_name": '测试用户',
            "body_text": '周五的文字笔记取消，改成周六给我一份语音总结吧。',
        },
        {
            "role": "assistant",
            "display_name": _CHARACTER_NAME,
            "body_text": '行，旧安排取消，2026-05-16 我发语音总结给你。',
        },
    ]
    record_input["content_plan"] = [
        '明确取消旧文字笔记义务',
        '确认2026-05-16发送语音总结的新义务',
    ]
    record_input["final_dialog"] = [
        '旧的文字笔记取消，2026-05-16 我发语音总结给你。'
    ]

    result, trace_payload = await _run_case(
        monkeypatch,
        "obligation_active_to_superseded",
        record_input,
    )

    obligations = result["interaction_obligations"]
    assert any(
        row["status"] == "superseded" and '文字评审' in row["action"]
        for row in obligations
    ), trace_payload
    assert any(
        row["status"] == "active"
        and row["actor"] == _CHARACTER_NAME
        and '语音总结' in row["action"]
        for row in obligations
    ), trace_payload
