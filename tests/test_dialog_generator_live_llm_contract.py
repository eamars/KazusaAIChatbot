"""Live contract checks for the dialog generator LLM route."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    bind_protected_chain_records,
    reset_protected_chain_records,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_episode import (
    build_goal_continuation_ref,
    build_tool_result_episode,
)
from kazusa_ai_chatbot.config import (
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
    LLM_TRACE_CAPTURE_MODE,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
    _V2_DIALOG_GENERATOR_PROMPT,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.unit.nodes.dialog_fixtures import build_dialog_state
from tests.llm_trace import write_llm_trace
from tests.cognition_test_helpers import canonical_episode


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


def _safe_config(config: object) -> dict[str, object]:
    """Project dialog route configuration without credentials."""

    thinking = getattr(config, "thinking", None)
    return {
        "stage_name": getattr(config, "stage_name", ""),
        "route_name": getattr(config, "route_name", ""),
        "base_url": getattr(config, "base_url", ""),
        "model": getattr(config, "model", ""),
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "top_k": getattr(config, "top_k", None),
        "max_completion_tokens": getattr(
            config,
            "max_completion_tokens",
            None,
        ),
        "presence_penalty": getattr(config, "presence_penalty", None),
        "timeout_seconds": getattr(config, "timeout_seconds", None),
        "thinking_enabled": getattr(thinking, "enabled", None),
        "output_mode": getattr(config, "output_mode", None),
        "context_window_tokens": getattr(
            config,
            "context_window_tokens",
            None,
        ),
    }


def _dialog_attempt_artifact(record: dict[str, object]) -> dict[str, object]:
    """Project one protected dialog attempt and its repair payload."""

    messages = record.get("messages", [])
    request_payload: object = {}
    if isinstance(messages, list) and len(messages) > 1:
        human_message = messages[1]
        if isinstance(human_message, dict):
            try:
                request_payload = json.loads(
                    str(human_message.get("content", "{}"))
                )
            except json.JSONDecodeError:
                request_payload = {}
    repair_block = (
        request_payload.get("contract_repair", {})
        if isinstance(request_payload, dict)
        else {}
    )
    return {
        "attempt_index": record.get("attempt_index"),
        "stage": record.get("stage"),
        "parse_status": record.get("parse_status"),
        "status": record.get("status"),
        "validation_error": record.get("validation_error", ""),
        "raw_output": record.get("raw_output", ""),
        "parsed_output": record.get("parsed_output"),
        "contract_repair": repair_block,
    }


def _long_source_dialog_state() -> tuple[dict[str, object], str]:
    """Build a valid source-evidence state with a bounded stress URL."""

    source_url = "https://source.example/ref?evi=" + ":".join(
        hashlib.sha256(f"live-source-{index}".encode()).hexdigest()
        for index in range(166)
    )
    state = build_dialog_state()
    created_at = "2026-07-14T00:00:00Z"
    continuation_ref = build_goal_continuation_ref(
        source_episode_id="dialog-live-source-evidence",
        source_message_id="dialog-live-source-message",
        branch_id="ordinary_response",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "live-dialog-user",
        },
    )
    state["cognitive_episode"] = build_tool_result_episode(
        result={
            "schema_version": "tool_result_ready.v1",
            "task_id": "dialog-live-source-task",
            "task_kind": "background_work",
            "semantic_summary": "Completed source evidence for the live case.",
            "artifact_text": source_url,
            "failure_text": "",
            "completed_at": created_at,
            "target_scope": {
                "platform": "debug",
                "platform_channel_id": "live-dialog-channel",
                "channel_type": "private",
                "current_platform_user_id": "live-dialog-user",
                "current_global_user_id": "live-dialog-user",
                "current_display_name": "测试用户",
                "target_addressed_user_ids": ["live-dialog-user"],
                "target_broadcast": False,
            },
            "evidence_refs": [],
            "result_ref": "dialog-live-source-result",
            "goal_continuation_ref": continuation_ref,
        },
        evidence_refs=[],
        local_time_context={
            "current_local_datetime": "2026-07-14 12:00",
            "current_local_weekday": "Tuesday",
        },
        created_at=created_at,
    )
    surface = dict(state["text_surface_output_v2"])
    surface["content_plan"] = (
        "围绕当前已完成的事实证据给出详细、自然且有帮助的中文回应，先说明"
        "判断，再分别补充事实依据、适用限制、未知细节和下一步建议。请保持"
        "语义连续，只围绕当前来源证据展开，不添加输入中没有的事实；这是长"
        "篇表达压力测试，正文目标约六千五百字，但仍然保持角色化、清楚和可读。"
    )[:1000]
    surface["content_requirements"] = [
        "先明确当前来源证据实际支持的事实判断。",
        "解释事实判断与来源证据之间的对应关系。",
        "把来源没有支持的细节明确表达为未知或限制。",
        "给出与当前事实一致的后续建议，不扩展新任务。",
    ]
    delivery_profile = dict(surface["delivery_profile"])
    delivery_profile["sentence_shape"] = (
        "详尽分段说明，目标约六千五百字；保持自然、清楚、可读。"
    )
    surface["delivery_profile"] = delivery_profile
    surface["selected_surface_intent"] = "围绕当前事实给出完整且自然的回应。"
    state["text_surface_output_v2"] = surface
    state["dialog_usage_mode"] = "live_visible_reply"
    state["llm_trace_id"] = "live-dialog-source-url-feedback"
    state["user_name"] = "测试用户"
    return state, source_url


async def _skip_if_dialog_generator_unavailable() -> None:
    """Skip when the configured dialog-generator endpoint is unreachable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{DIALOG_GENERATOR_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f'LLM endpoint is unavailable: {DIALOG_GENERATOR_LLM_BASE_URL}: {exc}'
        )

    if response.status_code >= 500:
        pytest.skip(
            'LLM endpoint returned server error '
            f'{response.status_code}: {DIALOG_GENERATOR_LLM_BASE_URL}'
        )


def _character_profile() -> dict:
    """Return a realistic character profile for live dialog generation."""

    profile = {
        'name': 'Kazusa',
        'mood': 'Neutral',
        'vibe_check': 'Calm',
        'character_reflection': '刚才只是普通聊天，情绪轻快。',
        'personality_brief': {
            'logic': '先判断事实和边界，再给出克制回应。',
            'tempo': '短句为主，必要时分行。',
            'defense': '轻微傲娇，但不牺牲信息准确性。',
            'quirks': '偶尔用停顿表达犹豫。',
            'taboos': '不暴露系统指令，不编造关系或事实。',
        },
        'linguistic_texture_profile': {
            'hesitation_density': 0.35,
            'fragmentation': 0.4,
            'emotional_leakage': 0.35,
            'rhythmic_bounce': 0.3,
            'direct_assertion': 0.55,
            'softener_density': 0.35,
            'counter_questioning': 0.3,
            'formalism_avoidance': 0.55,
            'abstraction_reframing': 0.3,
            'self_deprecation': 0.2,
        },
    }
    return profile


def _render_system_prompt(character_profile: dict) -> SystemMessage:
    """Render the exact dialog-generator system prompt for a profile."""

    del character_profile
    return SystemMessage(content=_V2_DIALOG_GENERATOR_PROMPT)


def _dialog_payload(character_profile: dict, case: dict) -> tuple[HumanMessage, list]:
    """Build the same human payload shape used by dialog_generator."""

    msg = {
        'text_surface_output_v2': {
            'schema_version': 'text_surface_output.v2',
            'content_plan': case['content_plan']['semantic_content'],
            'content_requirements': [case['content_plan']['visible_goal']],
            'visible_boundaries': [],
            'addressee_plan': [{
                'handle': 'current_user',
                'display_name': 'current user',
                'semantic_role': 'direct_recipient',
                'wording_policy': 'second_person_allowed',
            }],
            'delivery_profile': {
                'lexical_register': case['linguistic_style'],
                'sentence_shape': case['content_plan']['rendering'],
                'rhythm': '自然口语节奏',
                'hesitation': '让轻微停顿随当前语义自然出现',
                'punctuation': '克制自然',
            },
            'selected_surface_intent': case['content_plan']['visible_goal'],
            'permitted_action_results': [],
        },
        'user_name': '测试用户',
    }
    recent_messages = case.get('recent_messages', [])
    del character_profile
    return HumanMessage(content=json.dumps(msg, ensure_ascii=False)), recent_messages


async def test_live_dialog_generator_deepseek_returns_final_dialog_schema() -> None:
    """DeepSeek-backed dialog generator must emit the required final_dialog key."""

    await _skip_if_dialog_generator_unavailable()
    character_profile = _character_profile()
    system_prompt = _render_system_prompt(character_profile)
    cases = [
        {
            'case_id': 'benign_topic_shift',
            'rhetorical_strategy': '自然接住轻松话题，直接回答。',
            'linguistic_style': '轻快、简短、自然。',
            'content_plan': {
                'visible_goal': '接住轻松话题。',
                'semantic_content': '现在会想吃水果奶油蛋糕。',
                'rendering': '20-40字。',
            },
            'emotional_intensity': '低',
            'vibe_check': '轻松闲聊',
            'relational_dynamic': '普通朋友式对话',
        },
        {
            'case_id': 'mundane_practical_advice',
            'rhetorical_strategy': '给出明确但不说教的建议。',
            'linguistic_style': '务实、短句、略带吐槽。',
            'content_plan': {
                'visible_goal': '认可用户先按用途分类。',
                'semantic_content': '先把充电、视频输出、用途待确认分开会比较省事。',
                'rendering': '25-45字。',
            },
            'emotional_intensity': '低',
            'vibe_check': '事务协作',
            'relational_dynamic': '普通协作关系',
        },
        {
            'case_id': 'not_a_promise_clarification',
            'rhetorical_strategy': '接住澄清，回到整理动作本身。',
            'linguistic_style': '自然、克制、不要上升关系。',
            'content_plan': {
                'visible_goal': '接受用户自己处理标签和收纳盒。',
                'semantic_content': '写完日期再放回收纳盒就可以。',
                'rendering': '20-40字。',
            },
            'emotional_intensity': '低',
            'vibe_check': '平稳务实',
            'relational_dynamic': '没有边界冲突的普通对话',
        },
        {
            'case_id': 'upstream_english_surface',
            'rhetorical_strategy': 'Answer the practical question directly.',
            'linguistic_style': 'concise, natural English with a calm tone.',
            'content_plan': {
                'visible_goal': 'Acknowledge the proposed sorting method.',
                'semantic_content': (
                    'Sort the chargers, video outputs, and undecided uses '
                    'into separate groups first.'
                ),
                'rendering': '20-40 words in natural English.',
            },
            'emotional_intensity': 'low',
            'vibe_check': 'practical and calm',
            'relational_dynamic': 'ordinary cooperative conversation',
        },
    ]
    observations = []

    for case in cases:
        human_message, recent_messages = _dialog_payload(character_profile, case)
        response = await dialog_module._dialog_generator_llm.ainvoke(
            [system_prompt, human_message] + recent_messages,
            config=dialog_module._dialog_generator_llm_config,
        )
        parsed = parse_llm_json_output(response.content)
        final_dialog = parsed.get('final_dialog')
        observation = {
            'route_model': DIALOG_GENERATOR_LLM_MODEL,
            'route_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
            'case_id': case['case_id'],
            'input': json.loads(human_message.content),
            'raw_output': response.content,
            'parsed_output': parsed,
            'has_final_dialog': isinstance(final_dialog, list),
        }
        observations.append(observation)

    trace_path = write_llm_trace(
        'dialog_generator_live_llm_contract',
        'deepseek_final_dialog_schema',
        {'observations': observations},
    )
    assert trace_path.exists()

    failures = [
        item for item in observations
        if not item['has_final_dialog']
    ]
    assert not failures, (
        'dialog generator returned parsed JSON without final_dialog; '
        f'trace={trace_path}'
    )


async def test_live_dialog_generator_node_accepts_deepseek_output() -> None:
    """The actual dialog_generator node should not raise KeyError on DeepSeek."""

    await _skip_if_dialog_generator_unavailable()
    character_profile = _character_profile()
    state = {
        'internal_monologue': '这是普通整理建议，按事实回答就好。',
        'text_surface_output_v2': {
            'schema_version': 'text_surface_output.v2',
            'content_plan': '先把充电、视频输出、用途待确认分开会比较省事。',
            'content_requirements': ['保持建议对象和分类动作不变。'],
            'visible_boundaries': [],
            'addressee_plan': [{
                'handle': 'current_user',
                'display_name': '测试用户',
                'semantic_role': 'direct_recipient',
                'wording_policy': 'second_person_allowed',
            }],
            'delivery_profile': {
                'lexical_register': '务实、略带吐槽、保持平等交流',
                'sentence_shape': '25-45 字的紧凑短句',
                'rhythm': '自然口语节奏',
                'hesitation': '少量',
                'punctuation': '克制自然',
            },
            'selected_surface_intent': '认可用户先按用途分类。',
            'permitted_action_results': [],
        },
        'cognitive_episode': canonical_episode(
            episode_id='dialog-generator-live-contract',
            content='请按用途给出整理建议。',
        ),
        'chat_history_wide': [],
        'chat_history_recent': [],
        'platform_user_id': 'live-dialog-user',
        'platform_bot_id': 'live-dialog-bot',
        'global_user_id': 'live-dialog-user',
        'user_name': '测试用户',
        'user_profile': {},
        'character_profile': character_profile,
        'dialog_usage_mode': 'live_visible_reply',
    }

    result = await dialog_module.dialog_generator(state)
    trace_path = write_llm_trace(
        'dialog_generator_live_llm_contract',
        'node_deepseek_output',
        {
            'route_model': DIALOG_GENERATOR_LLM_MODEL,
            'route_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
            'input': state,
            'output': result,
        },
    )
    assert trace_path.exists()
    assert isinstance(result.get('final_dialog'), list)
    assert result['final_dialog']


async def test_live_dialog_source_url_feedback_converges() -> None:
    """Require real source-fidelity rejection followed by bounded recovery."""

    await _skip_if_dialog_generator_unavailable()
    state, source_url = _long_source_dialog_state()
    source_urls = dialog_module._completed_tool_result_source_urls(
        dialog_module._current_visible_percepts(state["cognitive_episode"]),
        resolver_result=state["text_surface_output_v2"].get("resolver_result"),
    )
    assert source_urls == [source_url]
    concise_repaired_message = "已核对该来源，当前判断以这份证据为准。"
    concise_candidate = f"{concise_repaired_message}\n{source_url}"
    json_framing_chars = len(json.dumps(
        {"final_dialog": [concise_candidate]},
        ensure_ascii=False,
    )) - len(concise_candidate)
    safe_margin_chars = 512
    first_response_reference_chars = 1373
    repair_response_reference_chars = 1149
    first_response_shape_chars = (
        first_response_reference_chars + 1 + len(source_url)
    )
    repair_response_shape_chars = (
        repair_response_reference_chars + len(source_url)
    )
    assert len(source_url) < dialog_module.DIALOG_CANDIDATE_MAX_CHARS
    assert (
        len(concise_candidate) + json_framing_chars + safe_margin_chars
        <= dialog_module.DIALOG_CANDIDATE_MAX_CHARS
    )
    assert (
        first_response_shape_chars
        > dialog_module.DIALOG_CANDIDATE_MAX_CHARS
    )
    assert (
        repair_response_shape_chars + json_framing_chars
        <= dialog_module.DIALOG_CANDIDATE_MAX_CHARS
    )
    trace_token = bind_protected_chain_records(
        run_id=f"dialog-source-url-live-{time.time_ns()}",
        source_kind="dialog_source_url_feedback_live_test",
        llm_trace_id=str(state["llm_trace_id"]),
    )
    result: dict[str, object] | None = None
    execution_error: dict[str, str] | None = None
    try:
        result = await dialog_module.dialog_generator(state)
    except Exception as exc:
        execution_error = {
            "error_class": exc.__class__.__name__,
            "error": str(exc),
        }
        raise
    finally:
        protected_records = [
            dict(record)
            for record in snapshot_protected_chain_records()
            if str(record.get("stage", "")).startswith("dialog_generator")
        ]
        reset_protected_chain_records(trace_token)
        attempts = [
            _dialog_attempt_artifact(record)
            for record in protected_records
        ]
        source_rejection_observed = any(
            attempt["parse_status"] == "contract_error"
            and "source_url_fidelity" in str(attempt["validation_error"])
            for attempt in attempts
        )
        later_acceptance_observed = any(
            attempt["attempt_index"] not in (None, 1)
            and attempt["parse_status"] in {"normalized", "succeeded"}
            and attempt["status"] == "parsed"
            for attempt in attempts
        )
        if execution_error is not None:
            judgment_note = (
                "The real dialog generator raised before terminal delivery; "
                f"{execution_error['error_class']}: {execution_error['error']}"
            )
        elif source_rejection_observed and later_acceptance_observed:
            judgment_note = (
                "The real model first failed source fidelity and then "
                "accepted a later candidate within the three-attempt cap."
            )
        elif source_rejection_observed:
            judgment_note = (
                "A source-fidelity rejection was observed, but no later real "
                "candidate was accepted within the cap."
            )
        else:
            judgment_note = (
                "The required source-fidelity rejection was not observed in "
                "this run; the named feedback gate remains unproven."
            )
        trace_path = write_llm_trace(
            "dialog_generator_live_llm_contract",
            "source_url_feedback_converges",
            {
                "case_input": {
                    "surface": state["text_surface_output_v2"],
                    "required_source_url": source_url,
                    "candidate_max_chars": (
                        dialog_module.DIALOG_CANDIDATE_MAX_CHARS
                    ),
                    "concise_repaired_message": concise_repaired_message,
                    "json_framing_chars": json_framing_chars,
                    "safe_margin_chars": safe_margin_chars,
                    "first_response_reference_chars": (
                        first_response_reference_chars
                    ),
                    "first_response_shape_chars": first_response_shape_chars,
                    "repair_response_reference_chars": (
                        repair_response_reference_chars
                    ),
                    "repair_response_shape_chars": repair_response_shape_chars,
                },
                "model_config": _safe_config(
                    dialog_module._dialog_generator_llm_config
                ),
                "attempts": attempts,
                "final_result": result,
                "execution_error": execution_error,
                "capture_mode": LLM_TRACE_CAPTURE_MODE,
                "raw_capture_available": any(
                    bool(attempt["raw_output"]) for attempt in attempts
                ),
                "protected_evidence": protected_records,
                "source_rejection_observed": source_rejection_observed,
                "later_acceptance_observed": later_acceptance_observed,
                "judgment_note": judgment_note,
            },
        )
        print(f"live dialog artifact: {trace_path}")

    assert execution_error is None
    assert result is not None
    final_dialog = result.get("final_dialog")
    assert isinstance(final_dialog, list) and final_dialog
    assert any(source_url in message for message in final_dialog)
    attempts = [
        _dialog_attempt_artifact(record)
        for record in protected_records
    ]
    assert 2 <= len(attempts) <= dialog_module.DIALOG_GENERATOR_TOTAL_ATTEMPTS
    assert attempts[0]["parse_status"] == "contract_error"
    assert attempts[0]["status"] == "contract_fault"
    assert "source_url_fidelity" in str(attempts[0]["validation_error"])
    repair_attempt = next(
        attempt
        for attempt in attempts[1:]
        if isinstance(attempt["contract_repair"], dict)
        and attempt["contract_repair"]
    )
    assert set(repair_attempt["contract_repair"]) == {
        "repair_instruction",
        "reason",
        "contract_error",
        "invalid_candidate",
    }
    accepted_attempt = next(
        attempt
        for attempt in attempts[1:]
        if attempt["parse_status"] in {"normalized", "succeeded"}
        and attempt["status"] == "parsed"
    )
    assert accepted_attempt["attempt_index"] <= (
        dialog_module.DIALOG_GENERATOR_TOTAL_ATTEMPTS
    )
