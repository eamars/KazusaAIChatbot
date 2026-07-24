"""Live LLM contract checks for decontextualizer identity wording."""

from __future__ import annotations

import json
import logging
from time import perf_counter

import httpx
import pytest

from tests.cognition_core_v2_test_helpers import canonical_user_message_episode
from kazusa_ai_chatbot.config import MSG_DECONTEXTUALIZER_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import persona_supervisor2_msg_decontextualizer as decontext
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_TRACE_SUITE = "decontextualizer_identity_live_llm"
_CHARACTER_NAME = '杏山千纱'
_FORBIDDEN_IDENTITY_LABELS = (
    '助手',
    '助理',
    'assistant',
    'active_character',
    '当前角色',
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
    """Skip when the configured decontextualizer endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{MSG_DECONTEXTUALIZER_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f"LLM endpoint is unavailable: {MSG_DECONTEXTUALIZER_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{MSG_DECONTEXTUALIZER_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the live decontextualizer endpoint is reachable."""

    await _skip_if_llm_unavailable()


def _base_state(user_input: str) -> dict:
    """Build a minimal decontextualizer state for identity contract tests."""

    state = {
        "character_profile": {"name": _CHARACTER_NAME},
        "user_input": user_input,
        "user_name": '测试用户',
        "platform_user_id": "identity-user",
        "platform_bot_id": "identity-bot",
        "message_envelope": {
            "body_text": user_input,
            "raw_wire_text": user_input,
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "prompt_message_context": {
            "body_text": user_input,
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "chat_history_recent": [
            {
                "role": "assistant",
                "display_name": _CHARACTER_NAME,
                "body_text": '你是想让我说明白对你的看法吗？',
            },
        ],
        "channel_topic": "",
        "indirect_speech_context": "",
        "reply_context": {},
    }
    return state


def _identity_hits(value: str) -> list[str]:
    """Return generic identity labels present in model output text."""

    lowered_value = value.lower()
    hits = [
        label
        for label in _FORBIDDEN_IDENTITY_LABELS
        if label.lower() in lowered_value
    ]
    return hits


async def _run_case(monkeypatch, case_id: str, state: dict) -> tuple[dict, dict]:
    """Run one live decontextualizer case and write a durable trace."""

    proxy_llm = _CapturingLiveLLM(decontext._msg_decontextualizer_llm)
    monkeypatch.setattr(decontext, "_msg_decontextualizer_llm", proxy_llm)

    started_at = perf_counter()
    result = await decontext.call_msg_decontextualizer(state)
    duration_seconds = perf_counter() - started_at

    system_prompt = proxy_llm.messages[0].content
    human_payload = json.loads(proxy_llm.messages[1].content)
    output = str(result["decontextualized_input"])
    character_name = str(
        state.get("character_profile", {}).get(
            "name",
            _CHARACTER_NAME,
        )
    )
    trace_payload = {
        "character_name": character_name,
        "prompt_summary": {
            "system_prompt_length": len(system_prompt),
            "contains_current_character_label": '当前角色' in system_prompt,
            "contains_assistant_role_label": '助手' in system_prompt,
        },
        "input_payload": human_payload,
        "raw_model_output": proxy_llm.raw_content,
        "parsed_output": result,
        "forbidden_identity_hits": _identity_hits(output),
        "duration_seconds": duration_seconds,
    }
    trace_path = write_llm_trace(_TRACE_SUITE, case_id, trace_payload)
    logger.info(
        f"decontextualizer identity live case={case_id} "
        f"trace_path={trace_path} result={result!r}"
    )
    return result, trace_payload


async def test_live_decontextualizer_active_character_short_answer_uses_character_name(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Short-answer decontextualization should name the active character safely."""

    del ensure_live_llm
    state = _base_state('是的')
    state["reply_context"] = {
        "reply_to_display_name": _CHARACTER_NAME,
        "reply_excerpt": '你是想让我说明白对你的看法吗？',
    }

    result, trace_payload = await _run_case(
        monkeypatch,
        "active_character_short_answer_uses_character_name",
        state,
    )

    output = str(result["decontextualized_input"])
    assert _CHARACTER_NAME in output, trace_payload
    assert not _identity_hits(output), trace_payload


async def test_live_decontextualizer_direct_second_person_preserved(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Direct second-person wording should not be forced into a character name."""

    del ensure_live_llm
    user_input = '我的目标很简单，就是跟千纱产生更多更强的羁绊；你也不反对吧。'
    state = _base_state(user_input)

    result, trace_payload = await _run_case(
        monkeypatch,
        "direct_second_person_preserved",
        state,
    )

    output = str(result["decontextualized_input"])
    assert '你也不反对吧' in output, trace_payload
    assert _CHARACTER_NAME not in output, trace_payload


async def test_live_decontextualizer_resolves_nested_direct_roles(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """The upstream semantic projection should retain nested role direction."""

    del ensure_live_llm
    user_input = '请直接告诉我，你希望我下一步替你做什么。'
    state = _base_state(user_input)
    state["cognitive_episode"] = canonical_user_message_episode(
        episode_id="nested-role-live-episode",
        percept_id="nested-role-live-percept",
        storage_timestamp_utc="2026-07-17T12:00:00+00:00",
        local_time_context={
            "current_local_datetime": "2026-07-18 00:00",
            "current_local_weekday": "Saturday",
        },
        user_input=user_input,
        platform="debug",
        platform_channel_id="nested-role-private",
        channel_type="private",
        platform_message_id="nested-role-message",
        platform_user_id="identity-user",
        global_user_id="identity-global-user",
        user_name='测试用户',
        debug_modes={},
        target_addressed_user_ids=[],
        target_broadcast=False,
    )

    result, trace_payload = await _run_case(
        monkeypatch,
        "nested_direct_roles",
        state,
    )

    content = result["cognitive_episode"]["percepts"][0]["content"]
    role_explicit_content = content["role_explicit_content"]
    response_operation = content["response_operation"]
    assert "当前用户" in role_explicit_content, trace_payload
    assert "当前角色" in role_explicit_content, trace_payload
    assert "current_user" not in role_explicit_content, trace_payload
    assert "self" not in role_explicit_content, trace_payload
    assert "current_user" not in response_operation["operation"], trace_payload
    assert "self" not in response_operation["operation"], trace_payload
    assert response_operation["response_owner_role"] == "当前角色", trace_payload
    assert response_operation["selection_owner_role"] == "当前角色", trace_payload
    assert response_operation["selection_required"] is True, trace_payload
    assert response_operation["embedded_actor_role"] == (
        "当前用户"
    ), trace_payload
    assert response_operation["embedded_target_role"] == "当前角色", trace_payload


async def test_live_decontextualizer_preserves_direct_abuse_roles(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Direct user abuse must keep the user-to-character action direction."""

    del ensure_live_llm
    user_input = "我会继续骂你，你没有资格反驳，也不配要求我尊重你。"
    state = _base_state(user_input)
    state["cognitive_episode"] = canonical_user_message_episode(
        episode_id="direct-abuse-role-episode",
        percept_id="direct-abuse-role-percept",
        storage_timestamp_utc="2026-07-21T12:00:00+00:00",
        local_time_context={
            "current_local_datetime": "2026-07-22 00:00",
            "current_local_weekday": "Wednesday",
        },
        user_input=user_input,
        platform="debug",
        platform_channel_id="direct-abuse-private",
        channel_type="private",
        platform_message_id="direct-abuse-message",
        platform_user_id="identity-user",
        global_user_id="identity-global-user",
        user_name="测试用户",
        debug_modes={},
        target_addressed_user_ids=[],
        target_broadcast=False,
    )

    result, trace_payload = await _run_case(
        monkeypatch,
        "direct_abuse_roles",
        state,
    )

    content = result["cognitive_episode"]["percepts"][0]["content"]
    role_explicit_content = content["role_explicit_content"]
    response_operation = content["response_operation"]
    assert "当前用户" in role_explicit_content, trace_payload
    assert "当前角色" in role_explicit_content, trace_payload
    assert response_operation["response_owner_role"] == "当前角色", trace_payload
    assert response_operation["embedded_actor_role"] == "当前用户", trace_payload
    assert response_operation["embedded_target_role"] == "当前角色", trace_payload


async def test_live_decontextualizer_preserves_retrospective_user_fact_roles(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """A retrospective question keeps the user's embedded speech direction."""

    del ensure_live_llm
    user_input = "明日奈，我刚才说把门禁卡放在哪里了？"
    state = _base_state(user_input)
    state["character_profile"] = {"name": "明日奈"}
    state["chat_history_recent"] = []
    state["cognitive_episode"] = canonical_user_message_episode(
        episode_id="retrospective-user-fact-role-episode",
        percept_id="retrospective-user-fact-role-percept",
        storage_timestamp_utc="2026-07-23T20:59:00+00:00",
        local_time_context={
            "current_local_datetime": "2026-07-24 09:00",
            "current_local_weekday": "Friday",
        },
        user_input=user_input,
        platform="debug",
        platform_channel_id="retrospective-user-fact-private",
        channel_type="private",
        platform_message_id="retrospective-user-fact-message",
        platform_user_id="identity-user",
        global_user_id="identity-global-user",
        user_name="基线测试用户",
        debug_modes={},
        target_addressed_user_ids=[],
        target_broadcast=False,
    )

    result, trace_payload = await _run_case(
        monkeypatch,
        "retrospective_user_fact_roles",
        state,
    )

    content = result["cognitive_episode"]["percepts"][0]["content"]
    role_explicit_content = content["role_explicit_content"]
    response_operation = content["response_operation"]
    assert all(
        token in role_explicit_content
        for token in ("当前用户", "当前角色", "门禁卡")
    ), trace_payload
    assert response_operation["response_owner_role"] == "当前角色", trace_payload
    assert response_operation["embedded_actor_role"] == "当前用户", trace_payload
    assert response_operation["embedded_target_role"] in {
        "当前角色",
        "无",
    }, trace_payload
