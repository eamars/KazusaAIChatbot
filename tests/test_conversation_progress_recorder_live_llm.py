"""Live LLM contract checks for conversation-progress identity wording."""

from __future__ import annotations

import json
import logging
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

    async def ainvoke(self, messages):
        self.messages = messages
        response = await self.inner_llm.ainvoke(messages)
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
        "timestamp": "2026-05-10T03:19:18+00:00",
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
        "content_anchors": [
            '承认用户知道答案但还嘴硬拖了一下',
            '不要继续催同一题',
        ],
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
    record_input["content_anchors"] = [
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
    assert '学习助手' in output_text, trace_payload
    assert not _identity_hits(
        output_text,
        allowed_terms=('学习助手', '该助手', '这个助手'),
    ), trace_payload
