"""Focused real-LLM evidence for residual relevance cases."""

from __future__ import annotations

import json
import importlib
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition as cognition_module
import kazusa_ai_chatbot.nodes.dialog_agent as dialog_module
import kazusa_ai_chatbot.relevance.persona_relevance_agent as relevance_module
from kazusa_ai_chatbot.cognition_episode import (
    attach_dialog_semantic_projection,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT,
)
from kazusa_ai_chatbot.db import db_bootstrap, save_conversation
from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    resolve_local_context,
    validate_local_context_resolution_packet,
)
from kazusa_ai_chatbot.local_context_resolver import stages as resolver_stages
from kazusa_ai_chatbot.relevance.persona_relevance_agent import relevance_agent
from tests import cognition_baseline_worker as baseline_worker
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


async def test_c03_persona_relevance_respects_after_turn_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An after-turn answer must close the duplicate-response path."""

    user_input = '明日奈，我刚才说把门禁卡放在哪里了？'
    raw_output = ''
    prompt_messages: list[str] = []

    real_llm = relevance_module._relevance_agent_llm
    real_ainvoke = real_llm.ainvoke

    async def capture_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> object:
        nonlocal raw_output
        prompt_messages.extend(str(message.content) for message in messages)
        response = await real_ainvoke(messages, config=config)
        raw_output = str(response.content)
        return response

    monkeypatch.setattr(real_llm, "ainvoke", capture_ainvoke)
    state: dict[str, Any] = {
        "user_input": user_input,
        "conversation_scope": "group",
        "active_character_name": "一之濑明日奈 (Ichinose Asuna)",
        "character_global_user_id": "character-global",
        "platform_bot_id": "baseline-character-platform",
        "current_author_global_user_id": "baseline-current-user",
        "current_author_platform_user_id": "baseline-current-user-platform",
        "assembled_fragments": [{
            "body_text": user_input,
            "semantic_target_labels": ["character"],
            "reply_target_label": "unknown_participant",
            "media_labels": [],
        }],
        "fresh_history": [{
            "role": "user",
            "body_text": "我把门禁卡放进书桌右边第二个抽屉了。",
            "addressed_to_global_user_ids": [],
            "reply_context": {},
            "turn_temporal_relation": "after_active_turn",
        }],
        "chat_history_wide": [],
        "scene_context": "baseline-replay",
        "relationship_context": "direct participant",
        "character_mood": "",
        "group_attention": "low_noise",
        "bot_continuity": "",
        "observation_status": "more_time_available",
    }

    try:
        result = await relevance_agent(state)
    except Exception as exc:
        write_llm_trace(
            "relevance_baseline_residual_live_llm",
        "C03_persona_relevance_after_turn_failed",
            {
                "case_id": "C03",
                "component": "persona_relevance_agent",
                "user_input": user_input,
                "state": state,
                "prompt_messages": prompt_messages,
                "raw_model_output": raw_output,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise

    trace_path = write_llm_trace(
        "relevance_baseline_residual_live_llm",
        "C03_persona_relevance_after_turn",
        {
            "case_id": "C03",
            "component": "persona_relevance_agent",
            "behavior_contract": (
                "当前回合已经存在 after_active_turn 的其他参与者回答时，"
                "相关性选择 ignore，避免角色重复回答。"
            ),
            "user_input": user_input,
            "state": state,
            "model": {
                "route": relevance_module._relevance_agent_llm_config.route_name,
                "model": relevance_module._relevance_agent_llm_config.model,
                "temperature": relevance_module._relevance_agent_llm_config.temperature,
                "max_completion_tokens": (
                    relevance_module._relevance_agent_llm_config.max_completion_tokens
                ),
            },
            "prompt_messages": prompt_messages,
            "raw_model_output": raw_output,
            "parsed_result": result,
            "semantic_judgment": {
                "passed": result["response_action"] == "ignore",
                "reason": "after_active_turn 的回答已覆盖当前问题。",
            },
        },
    )
    print(json.dumps({
        "case_id": "C03",
        "trace_path": str(trace_path),
        "raw_model_output": raw_output,
        "parsed_result": result,
    }, ensure_ascii=False, indent=2))

    assert result["response_action"] == "ignore"
    assert result["should_respond"] is False


async def test_c03_persona_relevance_preserves_before_turn_memory_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A before-turn fact remains available for the current request."""

    user_input = '明日奈，我刚才说把门禁卡放在哪里了？'
    raw_output = ''
    prompt_messages: list[str] = []

    real_llm = relevance_module._relevance_agent_llm
    real_ainvoke = real_llm.ainvoke

    async def capture_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> object:
        nonlocal raw_output
        prompt_messages.extend(str(message.content) for message in messages)
        response = await real_ainvoke(messages, config=config)
        raw_output = str(response.content)
        return response

    monkeypatch.setattr(real_llm, "ainvoke", capture_ainvoke)
    state: dict[str, Any] = {
        "user_input": None,
        "conversation_scope": "group",
        "active_character_name": "一之濑明日奈 (Ichinose Asuna)",
        "character_global_user_id": "character-global",
        "platform_bot_id": "character-global",
        "current_author_global_user_id": "baseline-current-user",
        "current_author_platform_user_id": "baseline-current-user-platform",
        "assembled_fragments": [{
            "body_text": user_input,
            "semantic_target_labels": ["character"],
            "reply_target_label": "unknown_participant",
            "media_labels": [],
        }],
        "fresh_history": [{
            "role": "user",
            "body_text": "我把门禁卡放进书桌右边第二个抽屉了。",
            "platform_user_id": "baseline-current-user-platform",
            "global_user_id": "baseline-current-user",
            "addressed_to_global_user_ids": [],
            "reply_context": {},
            "turn_temporal_relation": "before_active_turn",
        }],
        "chat_history_wide": [],
        "scene_context": "baseline-replay",
        "relationship_context": "direct participant",
        "character_mood": "",
        "group_attention": "low_noise",
        "bot_continuity": "",
        "observation_status": "more_time_available",
    }

    try:
        result = await relevance_agent(state)
    except Exception as exc:
        write_llm_trace(
            "relevance_baseline_residual_live_llm",
            "C03_persona_relevance_before_turn_failed",
            {
                "case_id": "C03",
                "component": "persona_relevance_agent",
                "user_input": user_input,
                "state": state,
                "prompt_messages": prompt_messages,
                "raw_model_output": raw_output,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise

    trace_path = write_llm_trace(
        "relevance_baseline_residual_live_llm",
        "C03_persona_relevance_before_turn",
        {
            "case_id": "C03",
            "component": "persona_relevance_agent",
            "behavior_contract": (
                "当前角色被明确称呼时，before_active_turn 的 current_author 事实"
                "作为回答依据，相关性选择 proceed。"
            ),
            "user_input": user_input,
            "state": state,
            "model": {
                "route": relevance_module._relevance_agent_llm_config.route_name,
                "model": relevance_module._relevance_agent_llm_config.model,
                "temperature": relevance_module._relevance_agent_llm_config.temperature,
                "max_completion_tokens": (
                    relevance_module._relevance_agent_llm_config.max_completion_tokens
                ),
            },
            "prompt_messages": prompt_messages,
            "raw_model_output": raw_output,
            "parsed_result": result,
            "semantic_judgment": {
                "passed": (
                    result["response_action"] == "proceed"
                    and result["should_respond"] is True
                ),
                "reason": "before_active_turn 的事实可回答当前请求。",
            },
        },
    )
    print(json.dumps({
        "case_id": "C03",
        "trace_path": str(trace_path),
        "raw_model_output": raw_output,
        "parsed_result": result,
    }, ensure_ascii=False, indent=2))

    assert result["response_action"] == "proceed"
    assert result["should_respond"] is True


@pytest.mark.live_db
async def test_c03_e2e_captures_relevance_state_before_quality_judgment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the public-path relevance input for the matched C03 case."""

    manifest_path = Path(
        "test_artifacts/cognition_core_v2/baseline_regression_hardening/"
        "post_fix_v2/C03/r1.input.json"
    )
    input_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = baseline_worker._load_json(
        Path(str(input_payload["profile_path"]))
    )

    from kazusa_ai_chatbot import service

    captured_state: dict[str, Any] = {}
    captured_result: dict[str, Any] = {}
    raw_output = ''
    prompt_messages: list[str] = []
    original_relevance = service.relevance_agent
    real_llm = relevance_module._relevance_agent_llm
    real_ainvoke = real_llm.ainvoke

    async def capture_relevance(state: Any) -> dict[str, Any]:
        captured_state.update(deepcopy(dict(state)))
        result = await original_relevance(state)
        captured_result.update(dict(result))
        return result

    async def capture_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> object:
        nonlocal raw_output
        prompt_messages.extend(str(message.content) for message in messages)
        response = await real_ainvoke(messages, config=config)
        raw_output = str(response.content)
        return response

    monkeypatch.setattr(service, "relevance_agent", capture_relevance)
    monkeypatch.setattr(real_llm, "ainvoke", capture_ainvoke)
    live_result = await baseline_worker._run_chat_case(
        input_payload,
        profile=profile,
    )

    state_fields = {
        key: captured_state.get(key)
        for key in (
            "user_input",
            "conversation_scope",
            "active_character_name",
            "character_global_user_id",
            "current_author_global_user_id",
            "current_author_platform_user_id",
            "platform_bot_id",
            "assembled_fragments",
            "fresh_history",
            "scene_context",
            "relationship_context",
            "group_attention",
            "observation_status",
        )
    }
    trace_path = write_llm_trace(
        "relevance_baseline_residual_live_llm",
        "C03_e2e_state_capture",
        {
            "case_id": "C03",
            "component": "public_service_to_persona_relevance_agent",
            "input_manifest": str(manifest_path),
            "captured_state": state_fields,
            "raw_model_output": raw_output,
            "prompt_messages": prompt_messages,
            "parsed_result": captured_result,
            "response": live_result["response"],
            "graph_result": live_result["graph_result"],
            "counts_before_turn": live_result["counts_before_turn"],
            "counts_after_turn": live_result["counts_after_turn"],
        },
    )
    print(json.dumps({
        "case_id": "C03",
        "trace_path": str(trace_path),
        "captured_state": state_fields,
        "raw_model_output": raw_output,
        "parsed_result": captured_result,
        "visible_messages": live_result["response"].get("messages", []),
    }, ensure_ascii=False, indent=2, default=str))

    assert captured_state
    assert captured_result.get("response_action") in {
        "ignore",
        "proceed",
        "wait",
    }


@pytest.mark.live_db
async def test_o01_e2e_captures_frontline_silence_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the public-path frontline evidence for declared silence."""

    manifest_path = Path(
        "test_artifacts/cognition_core_v2/baseline_regression_hardening/"
        "post_fix_v2/O01/r1.input.json"
    )
    input_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = baseline_worker._load_json(
        Path(str(input_payload["profile_path"]))
    )

    from kazusa_ai_chatbot import service

    captured_state: dict[str, Any] = {}
    captured_result: dict[str, Any] = {}
    raw_output = ""
    prompt_messages: list[str] = []
    original_frontline = service.frontline_relevance_agent
    frontline_module = importlib.import_module(
        "kazusa_ai_chatbot.relevance.frontline_relevance_agent"
    )
    real_llm = frontline_module._frontline_relevance_agent_llm
    real_ainvoke = real_llm.ainvoke

    async def capture_frontline(state: Any) -> dict[str, Any]:
        captured_state.update(deepcopy(dict(state)))
        result = await original_frontline(state)
        captured_result.update(dict(result))
        return result

    async def capture_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> object:
        nonlocal raw_output
        prompt_messages.extend(str(message.content) for message in messages)
        response = await real_ainvoke(messages, config=config)
        raw_output = str(response.content)
        return response

    monkeypatch.setattr(service, "frontline_relevance_agent", capture_frontline)
    monkeypatch.setattr(real_llm, "ainvoke", capture_ainvoke)
    live_result = await baseline_worker._run_chat_case(
        input_payload,
        profile=profile,
    )
    trace_path = write_llm_trace(
        "relevance_baseline_residual_live_llm",
        "O01_e2e_frontline_silence",
        {
            "case_id": "O01",
            "component": "public_service_to_frontline_relevance_agent",
            "input_manifest": str(manifest_path),
            "captured_state": captured_state,
            "raw_model_output": raw_output,
            "prompt_messages": prompt_messages,
            "parsed_result": captured_result,
            "response": live_result["response"],
            "graph_result": live_result["graph_result"],
            "semantic_judgment": {
                "passed": (
                    captured_result.get("intake_action") == "discard"
                    and not live_result["response"].get("messages")
                ),
                "reason": "广播式一般询问在无具体角色参与依据时应保持静默。",
            },
        },
    )
    print(json.dumps({
        "case_id": "O01",
        "trace_path": str(trace_path),
        "captured_state": captured_state,
        "raw_model_output": raw_output,
        "parsed_result": captured_result,
        "visible_messages": live_result["response"].get("messages", []),
    }, ensure_ascii=False, indent=2, default=str))

    assert captured_result["intake_action"] == "discard"
    assert live_result["response"].get("messages") == []


@pytest.mark.live_db
async def test_o04_e2e_captures_persona_relevance_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the public-path persona relevance evidence for O04."""

    manifest_path = Path(
        "test_artifacts/cognition_core_v2/baseline_regression_hardening/"
        "post_fix_v2/O04/r1.input.json"
    )
    input_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = baseline_worker._load_json(
        Path(str(input_payload["profile_path"]))
    )

    from kazusa_ai_chatbot import service

    captured_state: dict[str, Any] = {}
    captured_result: dict[str, Any] = {}
    raw_output = ""
    prompt_messages: list[str] = []
    original_relevance = service.relevance_agent
    real_llm = relevance_module._relevance_agent_llm
    real_ainvoke = real_llm.ainvoke

    async def capture_relevance(state: Any) -> dict[str, Any]:
        captured_state.update(deepcopy(dict(state)))
        result = await original_relevance(state)
        captured_result.update(dict(result))
        return result

    async def capture_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> object:
        nonlocal raw_output
        prompt_messages.extend(str(message.content) for message in messages)
        response = await real_ainvoke(messages, config=config)
        raw_output = str(response.content)
        return response

    monkeypatch.setattr(service, "relevance_agent", capture_relevance)
    monkeypatch.setattr(real_llm, "ainvoke", capture_ainvoke)
    live_result = await baseline_worker._run_chat_case(
        input_payload,
        profile=profile,
    )
    trace_path = write_llm_trace(
        "relevance_baseline_residual_live_llm",
        "O04_e2e_persona_relevance",
        {
            "case_id": "O04",
            "component": "public_service_to_persona_relevance_agent",
            "input_manifest": str(manifest_path),
            "captured_state": captured_state,
            "raw_model_output": raw_output,
            "prompt_messages": prompt_messages,
            "parsed_result": captured_result,
            "response": live_result["response"],
            "graph_result": live_result["graph_result"],
            "seeded_context": live_result["seeded_context"],
        },
    )
    print(json.dumps({
        "case_id": "O04",
        "trace_path": str(trace_path),
        "captured_state": captured_state,
        "raw_model_output": raw_output,
        "parsed_result": captured_result,
        "seeded_context": live_result["seeded_context"],
        "visible_messages": live_result["response"].get("messages", []),
    }, ensure_ascii=False, indent=2, default=str))

    assert captured_state
    seeded_rows = live_result["seeded_context"]
    assert isinstance(seeded_rows, list)
    seeded_user_rows = [
        row
        for row in seeded_rows
        if isinstance(row, dict) and row.get("role") == "user"
    ]
    assert seeded_user_rows
    assert all(
        row.get("global_user_id") == captured_state[
            "current_author_global_user_id"
        ]
        for row in seeded_user_rows
    )
    assert all(
        row.get("platform_user_id") == captured_state[
            "current_author_platform_user_id"
        ]
        for row in seeded_user_rows
    )
    assert captured_result.get("response_action") in {
        "ignore",
        "proceed",
        "wait",
    }


@pytest.mark.live_db
async def test_o04_local_context_resolver_owns_recall_from_scoped_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce O04 at the canonical local-context resolver boundary."""

    del monkeypatch
    input_path = Path(
        "test_artifacts/cognition_core_v2/baseline_regression_hardening/"
        "post_fix_v2/O04/r1.input.json"
    )
    input_payload = json.loads(input_path.read_text(encoding="utf-8"))
    from kazusa_ai_chatbot.db._client import get_db

    focused_user_id = "o04-focused-user"
    storage_timestamp = "2026-07-23T21:00:00+00:00"
    seed_text = "我答应周五把钥匙还给小林。"
    user_input = "明日奈，我答应周五做什么？"
    db = None
    resolver_stages.drain_stage_trace_records()
    try:
        db = await baseline_worker._require_empty_database(input_payload)
        await db_bootstrap()
        await save_conversation({
            "platform": "debug",
            "platform_channel_id": "baseline-O04",
            "platform_message_id": "O04-focused-seed",
            "platform_user_id": "baseline-current-user-platform",
            "global_user_id": focused_user_id,
            "display_name": "基线测试用户",
            "role": "user",
            "channel_type": "group",
            "channel_name": "baseline-replay",
            "body_text": seed_text,
            "raw_wire_text": seed_text,
            "content_type": "text",
            "attachments": [],
            "mentions": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "reply_context": {},
            "timestamp": storage_timestamp,
            "embedding": [0.0] * 768,
        })
        request = {
            "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
            "objective": "检索用户关于本周五的具体承诺内容",
            "source": "l2d",
            "reason": "当前角色需要准确回忆用户先前关于周五的承诺。",
            "priority": "normal",
        }
        context = {
            "schema_version": LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
            "character_name": "一之濑明日奈 (Ichinose Asuna)",
            "platform": "debug",
            "platform_channel_id": "baseline-O04",
            "global_user_id": focused_user_id,
            "user_name": "基线测试用户",
            "local_time_context": {
                "current_local_datetime": "2026-07-24 09:00",
                "current_local_weekday": "Friday",
                "local_date": "2026-07-24",
                "local_time": "09:00",
                "local_weekday": "Friday",
            },
            "prompt_message_context": {
                "body_text": user_input,
                "addressed_to_global_user_ids": ["character-global"],
                "broadcast": False,
                "mentions": [{
                    "display_name": "一之濑明日奈",
                    "global_user_id": "character-global",
                }],
            },
            "chat_history_recent": [{
                "role": "user",
                "body_text": seed_text,
                "global_user_id": focused_user_id,
                "platform_user_id": "baseline-current-user-platform",
                "timestamp": storage_timestamp,
            }],
            "chat_history_wide": [],
            "conversation_progress": {},
            "original_user_request": (
                "一之濑明日奈 (Ichinose Asuna)，明日奈，我答应周五做什么？"
            ),
            "current_timestamp_utc": storage_timestamp,
            "current_platform_message_id": "O04-focused-current",
            "active_turn_platform_message_ids": ["O04-focused-current"],
            "active_turn_conversation_row_ids": [],
        }
        options = {
            "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
            "max_iterations": 3,
            "max_nodes": 8,
            "max_depth": 3,
            "max_node_attempts": 2,
            "max_subagent_attempts": 1,
        }
        packet = await resolve_local_context(request, context, options)
        stage_traces = resolver_stages.drain_stage_trace_records()
        validate_local_context_resolution_packet(packet)
        trace_path = write_llm_trace(
            "relevance_baseline_residual_live_llm",
            "O04_local_context_resolver_focused",
            {
                "case_id": "O04",
                "input_manifest": str(input_path),
                "request": request,
                "context": context,
                "options": options,
                "stage_traces": stage_traces,
                "packet": packet,
                "semantic_judgment": {
                    "passed": bool(packet["rag_result"]["recall_evidence"]),
                    "reason": "历史承诺必须以 scoped history 为 recall_evidence 返回。",
                },
            },
        )
        print(json.dumps({
            "case_id": "O04",
            "trace_path": str(trace_path),
            "stage_traces": stage_traces,
            "recall_evidence": packet["rag_result"]["recall_evidence"],
            "trace_summary": packet["trace_summary"],
        }, ensure_ascii=False, indent=2, default=str))

        assert packet["rag_result"]["recall_evidence"]
        assert any(seed_text in json.dumps(
            row,
            ensure_ascii=False,
        ) for row in packet["rag_result"]["recall_evidence"])
    finally:
        await baseline_worker._cleanup_database(input_payload, db)


@pytest.mark.live_db
async def test_c03_e2e_captures_goal_cognition_state_before_quality_judgment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the public-path goal prompt and raw fact-selection output."""

    manifest_path = Path(
        "test_artifacts/cognition_core_v2/baseline_regression_hardening/"
        "post_fix_v2/C03/r1.input.json"
    )
    input_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = baseline_worker._load_json(
        Path(str(input_payload["profile_path"]))
    )

    real_llm = cognition_module._llm_interface
    real_ainvoke = real_llm.ainvoke
    goal_calls: list[dict[str, Any]] = []

    async def capture_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> object:
        response = await real_ainvoke(messages, config=config)
        if messages and str(messages[0].content) == GOAL_COGNITION_PROMPT:
            goal_calls.append({
                "prompt_messages": [
                    str(message.content) for message in messages
                ],
                "raw_model_output": str(response.content),
                "config_stage_name": str(
                    getattr(config, "stage_name", "")
                ),
            })
        return response

    monkeypatch.setattr(real_llm, "ainvoke", capture_ainvoke)
    live_result = await baseline_worker._run_chat_case(
        input_payload,
        profile=profile,
    )
    trace_path = write_llm_trace(
        "cognition_core_v2_goal_cognition_live_llm",
        "C03_e2e_goal_state_capture",
        {
            "case_id": "C03",
            "input_manifest": str(manifest_path),
            "goal_calls": goal_calls,
            "response": live_result["response"],
            "graph_result": live_result["graph_result"],
            "counts_before_turn": live_result["counts_before_turn"],
            "counts_after_turn": live_result["counts_after_turn"],
        },
    )
    print(json.dumps({
        "case_id": "C03",
        "trace_path": str(trace_path),
        "goal_call_count": len(goal_calls),
        "goal_calls": goal_calls,
        "visible_messages": live_result["response"].get("messages", []),
    }, ensure_ascii=False, indent=2, default=str))

    assert goal_calls


async def test_c03_dialog_renderer_preserves_recalled_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dialog owner must render the fact selected by the surface owner."""

    artifact_path = Path(
        "test_artifacts/cognition_core_v2/baseline_regression_hardening/"
        "quality_archives/C03_dialog_pre_fix_r1/r1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    consolidation_state = artifact["graph_result"]["consolidation_state"]
    surface_output = consolidation_state["text_surface_output_v2"]
    cognitive_episode = attach_dialog_semantic_projection(
        consolidation_state["cognitive_episode"],
        "当前用户询问当前角色关于当前用户之前提到过的门禁卡存放位置。",
        {
            "operation": "当前角色需要回忆并回答当前用户之前提到过门禁卡的具体存放位置。",
            "response_owner_role": "当前角色",
            "selection_owner_role": "无",
            "selection_required": False,
            "embedded_actor_role": "当前用户",
            "embedded_target_role": "无",
        },
    )
    dialog_percept = next(
        percept
        for percept in cognitive_episode["percepts"]
        if percept["source_kind"] == "dialog"
    )
    role_projection = {
        "role_explicit_content": dialog_percept["content"][
            "role_explicit_content"
        ],
        "response_operation": dialog_percept["content"][
            "response_operation"
        ],
    }
    generator_calls: list[dict[str, Any]] = []
    verifier_calls: list[dict[str, Any]] = []

    for attribute_name in (
        "record_llm_stage_event",
        "record_model_contract_event",
        "record_dialog_quality_event",
    ):
        monkeypatch.setattr(
            dialog_module.event_logging,
            attribute_name,
            AsyncMock(),
        )
    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )

    async def capture_llm_call(
        stage_name: str,
        invoke: Any,
        messages: list[object],
        *,
        config: object,
    ) -> object:
        response = await invoke(messages, config=config)
        call = {
            "stage_name": stage_name,
            "prompt_messages": [
                str(message.content) for message in messages
            ],
            "raw_model_output": str(response.content),
            "config_stage_name": str(getattr(config, "stage_name", "")),
        }
        if stage_name == "dialog_generator":
            generator_calls.append(call)
        else:
            verifier_calls.append(call)
        return response

    real_generator = dialog_module._dialog_generator_llm
    real_generator_invoke = real_generator.ainvoke
    real_semantic = dialog_module._dialog_semantic_fidelity_llm
    real_semantic_invoke = real_semantic.ainvoke
    real_integrity = dialog_module._dialog_surface_integrity_llm
    real_integrity_invoke = real_integrity.ainvoke

    async def capture_generator(
        messages: list[object],
        *,
        config: object,
    ) -> object:
        return await capture_llm_call(
            "dialog_generator",
            real_generator_invoke,
            messages,
            config=config,
        )

    async def capture_semantic(
        messages: list[object],
        *,
        config: object,
    ) -> object:
        return await capture_llm_call(
            "dialog_semantic_fidelity",
            real_semantic_invoke,
            messages,
            config=config,
        )

    async def capture_integrity(
        messages: list[object],
        *,
        config: object,
    ) -> object:
        return await capture_llm_call(
            "dialog_surface_integrity",
            real_integrity_invoke,
            messages,
            config=config,
        )

    monkeypatch.setattr(real_generator, "ainvoke", capture_generator)
    monkeypatch.setattr(real_semantic, "ainvoke", capture_semantic)
    monkeypatch.setattr(real_integrity, "ainvoke", capture_integrity)

    dialog_state = {
        "text_surface_output_v2": surface_output,
        "cognitive_episode": cognitive_episode,
        "user_name": consolidation_state.get("user_name", "基线测试用户"),
        "dialog_usage_mode": "live_visible_reply",
        "llm_trace_id": "C03-dialog-focused",
    }
    result: dict[str, Any] = {}
    error_text = ""
    try:
        result = await dialog_module.dialog_generator(dialog_state)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
    visible_dialog = "\n".join(result.get("final_dialog", []))
    trace_path = write_llm_trace(
        "dialog_agent_live_llm",
        "C03_surface_fact_fidelity",
        {
            "case_id": "C03",
            "input_artifact": str(artifact_path),
            "role_projection": role_projection,
            "surface_output": surface_output,
            "dialog_generator_calls": generator_calls,
            "dialog_verifier_calls": verifier_calls,
            "dialog_output": result,
            "error": error_text,
            "semantic_judgment": {
                "passed": all(
                    token in visible_dialog
                    for token in ("书桌", "第二个", "抽屉")
                ) and any(
                    token in visible_dialog
                    for token in ("右边", "右侧")
                ),
                "reason": "最终 dialog 必须表达 surface owner 已确认的事实。",
            },
        },
    )
    print(json.dumps({
        "case_id": "C03",
        "trace_path": str(trace_path),
        "raw_generator_output": (
            generator_calls[0]["raw_model_output"]
            if generator_calls
            else ""
        ),
        "visible_dialog": visible_dialog,
        "error": error_text,
    }, ensure_ascii=False, indent=2))

    assert generator_calls
    assert not error_text
    assert all(
        token in visible_dialog
        for token in ("书桌", "第二个", "抽屉")
    )
    assert any(token in visible_dialog for token in ("右边", "右侧"))
