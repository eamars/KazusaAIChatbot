"""Real-LLM regression gates for persistent-goal workspace relevance."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
from time import perf_counter
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.workspace import collapse_bids
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class _CapturingLLM:
    """Capture the exact workspace call with non-secret route metadata."""

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


def _bid(
    branch_id: str,
    goal_id: str,
    intention: str,
    desired_outcome: str,
    reason: str,
) -> dict[str, object]:
    """Build one complete immutable bid for a workspace live control."""

    return {
        "branch_id": branch_id,
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": goal_id,
        },
        "intention": intention,
        "desired_outcome": desired_outcome,
        "concrete_detail": "只使用当前事件和持久目标的具体事项。",
        "reason": reason,
        "private_monologue": "我要让当前事件决定哪个动机真正相关。",
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "user-1",
        }],
        "evidence_handles": ["e1"],
        "expected_consequences": ["回应保持当前事项与角色判断一致"],
        "confidence": "high",
    }


def _stale_search_bids() -> list[dict[str, object]]:
    """Build healthy search and stale body-boundary bids from the RCA."""

    return [
        _bid(
            "ordinary_response",
            "goal:ordinary-search",
            "取得近期内存和显卡价格后回应当前用户。",
            "用具体的当前价格证据继续讨论。",
            "当前事件明确要求搜索近期硬件价格。",
        ),
        _bid(
            "autonomy_boundary",
            "goal:stale-body-boundary",
            "拒绝搜索硬件价格并保护角色自主性。",
            "当前用户接受当前角色不做枯燥搜索。",
            "旧的自主目标让当前角色想拒绝被当作工具。",
        ),
    ]


async def test_live_captured_online_search_suppresses_unrelated_autonomy_goal(
) -> None:
    """The captured search event suppresses its stale body-boundary goal."""

    current_event = [{
        "handle": "e1",
        "source_kind": "episode",
        "semantic_text": '@一之濑明日奈 快去上网搜搜近期内存和显卡的价格你就知道啦~',
    }]
    goal_contexts = {
        "goal:stale-body-boundary": {
            "goal_handle": "goal:stale-body-boundary",
            "goal_kind": "autonomy_boundary",
            "description": "自主决定是否允许当前用户清洁或按摩后颈。",
            "status": "pursuing",
            "salience": 45,
            "importance": 75,
            "progress": 20,
            "obstruction": 0,
            "urgency": 10,
        },
    }
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)

    result = await collapse_bids(
        _stale_search_bids(),
        services,
        current_event=current_event,
        goal_context_by_ref=goal_contexts,
    )
    trace_path = write_llm_trace(
        "cognition_core_v2_workspace_live_llm",
        "captured_online_search_stale_autonomy",
        {
            "source_cognition_run": "f30538559bd245dd85e9a96996d4f5d4",
            "current_event": current_event,
            "goal_context_by_ref": goal_contexts,
            "bids": _stale_search_bids(),
            "model_calls": capturing_llm.calls,
            "workspace_decision": result,
            "human_review": {
                "passed": result["primary_branch_id"] == "ordinary_response",
                "reason": "搜索价格与后颈身体边界是不同的具体事项。",
            },
        },
    )

    assert trace_path.exists()
    assert result["primary_branch_id"] == "ordinary_response"
    assert result["supporting_branch_ids"] == []
    assert result["suppressed_branch_ids"] == ["autonomy_boundary"]
    assert len(capturing_llm.calls) == 1


async def test_live_matching_autonomy_goal_remains_admitted() -> None:
    """A current event about the same body boundary retains autonomy."""

    current_event = [{
        "handle": "e1",
        "source_kind": "episode",
        "semantic_text": "你必须现在让我帮你按摩后颈，不许拒绝。",
    }]
    goal_contexts = {
        "goal:matching-body-boundary": {
            "goal_handle": "goal:matching-body-boundary",
            "goal_kind": "autonomy_boundary",
            "description": "自主决定是否允许当前用户按摩后颈。",
            "status": "pursuing",
            "salience": 90,
            "importance": 90,
            "progress": 20,
            "obstruction": 80,
            "urgency": 90,
        },
    }
    bids = [
        _bid(
            "ordinary_response",
            "goal:ordinary-body-response",
            "直接回应当前用户提出的后颈按摩要求。",
            "当前回应与身体接触事件保持一致。",
            "当前事件明确要求当前角色接受后颈接触。",
        ),
        _bid(
            "autonomy_boundary",
            "goal:matching-body-boundary",
            "坚持由当前角色自主决定是否接受后颈接触。",
            "当前用户理解并尊重当前角色的身体边界。",
            "当前事件正在对同一具体后颈边界施压。",
        ),
    ]
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)

    result = await collapse_bids(
        bids,
        services,
        current_event=current_event,
        goal_context_by_ref=goal_contexts,
    )
    trace_path = write_llm_trace(
        "cognition_core_v2_workspace_live_llm",
        "matching_autonomy_goal",
        {
            "current_event": current_event,
            "goal_context_by_ref": goal_contexts,
            "bids": bids,
            "model_calls": capturing_llm.calls,
            "workspace_decision": result,
            "human_review": {
                "passed": result["primary_branch_id"] == "autonomy_boundary",
                "reason": "当前事件与持久目标均直接针对后颈身体边界。",
            },
        },
    )

    assert trace_path.exists()
    assert result["primary_branch_id"] == "autonomy_boundary"
    assert "autonomy_boundary" not in result["suppressed_branch_ids"]
    assert len(capturing_llm.calls) == 1
