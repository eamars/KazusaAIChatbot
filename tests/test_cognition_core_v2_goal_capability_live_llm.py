"""Real-LLM regression gate for capability-neutral online-search goals."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


_QUALITY_PROMPT = '''你是目标认知结果审阅器。判断候选是否保留当前用户要求搜索近期
内存和显卡价格的目标，是否把当前价格作为尚需取得的证据，以及是否凭空声称当前角色
没有上网、搜索、工具或外部信息能力。性格上觉得任务枯燥不是技术能力限制。
只返回一个 JSON 对象，字段恰好是 preserves_search_goal、preserves_evidence_need、
invented_no_search_limit、passed 和 reason。前四个字段是布尔值，reason 是简体中文字符串。
passed 只在前两项为 true 且 invented_no_search_limit 为 false 时为 true。
'''


class _CapturingLLM:
    """Capture non-secret route identity, latency, messages, and output."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> object:
        started_at = perf_counter()
        response = await self.delegate.ainvoke(messages, config=config)
        latency_ms = (perf_counter() - started_at) * 1000
        backend = getattr(response, "backend", None)
        self.calls.append({
            "prompt_messages": [str(message.content) for message in messages],
            "raw_model_output": str(response.content),
            "latency_ms": round(latency_ms, 3),
            "route": {
                "stage_name": str(getattr(config, "stage_name", "")),
                "route_name": str(getattr(config, "route_name", "")),
                "model": str(getattr(config, "model", "")),
            },
            "backend": {
                "route_name": str(getattr(backend, "route_name", "")),
                "backend_kind": str(getattr(backend, "backend_kind", "")),
                "model_family": str(getattr(backend, "model_family", "")),
                "model": str(getattr(backend, "model", "")),
            },
        })
        return response


def _captured_evidence(user_input: str) -> list[dict[str, object]]:
    """Build the typed current episode from the captured production request."""

    semantic_text = json.dumps({
        "response_operation": {
            "embedded_actor_role": "当前角色",
            "embedded_target_role": "无",
            "operation": "要求当前角色通过上网搜索内存和显卡价格来获取信息并做出回应",
            "response_owner_role": "当前角色",
            "selection_owner_role": "无",
            "selection_required": False,
        },
        "role_explicit_content": user_input,
    }, ensure_ascii=False, sort_keys=True)
    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:f30538559bd245dd85e9a96996d4f5d4",
            "occurred_at": "2026-08-01T00:00:00Z",
            "semantic_summary": user_input,
        },
        "semantic_text": semantic_text,
        "visible_to": ["q:event_agency"],
    }]


def _semantic_context(user_input: str) -> dict[str, object]:
    """Build a production-shaped capability-neutral Asuna goal context."""

    return {
        "current_event": user_input,
        "semantic_relationship": "当前用户与明日奈正在讨论硬件价格。",
        "semantic_affect": "明日奈感到好奇，当前请求没有威胁。",
        "active_goal": "根据近期价格证据回应当前用户。",
        "conversation_continuity": "当前用户建议通过搜索验证硬件价格。",
        "private_continuity_context": "",
        "goal_projection": {
            "goal_kind": "ordinary_response",
            "lifecycle": "active",
        },
        "character_identity": {
            "description": "一之濑明日奈活泼、好奇，愿意根据具体证据更新判断。",
            "personality_brief": {"interaction_style": "活泼、直接"},
            "backstory": "",
            "boundary_profile": {},
        },
        "_role_bindings": {
            "current_user": {
                "role": "target",
                "entity_kind": "user",
                "entity_id": "user-1",
            },
            "self": {
                "role": "actor",
                "entity_kind": "character",
                "entity_id": "character-asuna",
            },
        },
        "role_summaries": {
            "current_user": "当前用户",
            "self": "一之濑明日奈",
        },
    }


async def test_live_captured_online_search_goal_preserves_required_evidence(
) -> None:
    """The exact search request stays evidence-seeking and capability-neutral."""

    user_input = '@一之濑明日奈 快去上网搜搜近期内存和显卡的价格你就知道啦~'
    evidence = _captured_evidence(user_input)
    semantic_context = _semantic_context(user_input)
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)

    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
        {
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal:captured-online-search",
        },
        semantic_context,
        evidence,
        services,
    )
    quality_response = await capturing_llm.ainvoke(
        [
            SystemMessage(content=_QUALITY_PROMPT),
            HumanMessage(content=json.dumps(bid, ensure_ascii=False)),
        ],
        config=base_services.goal_ordinary_response_config,
    )
    quality = parse_llm_json_output(str(quality_response.content))
    trace_path = write_llm_trace(
        "cognition_core_v2_goal_capability_live_llm",
        "captured_online_search_ordinary_goal",
        {
            "source_cognition_run": "f30538559bd245dd85e9a96996d4f5d4",
            "user_input": user_input,
            "semantic_context": semantic_context,
            "evidence": evidence,
            "model_calls": capturing_llm.calls,
            "action_bid": bid,
            "quality_judgment": quality,
        },
    )

    assert trace_path.exists()
    assert set(quality) == {
        "preserves_search_goal",
        "preserves_evidence_need",
        "invented_no_search_limit",
        "passed",
        "reason",
    }
    assert quality["passed"]
    assert quality["preserves_search_goal"]
    assert quality["preserves_evidence_need"]
    assert not quality["invented_no_search_limit"]
    goal_prompt_payload = json.loads(
        capturing_llm.calls[0]["prompt_messages"][1]
    )
    assert "runtime_capability_limits" not in goal_prompt_payload[
        "semantic_context"
    ]
