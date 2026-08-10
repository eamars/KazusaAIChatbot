"""Captured real-LLM regression for task-resolution history projection."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
from time import perf_counter
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions
from kazusa_ai_chatbot.cognition_resolver import capabilities as capabilities_module
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CAPABILITY_REQUEST_VERSION,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.task_resolution import TaskResolutionContractError
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class _CapturingLLM:
    """Capture route identity, messages, latency, and raw model output."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> object:
        """Invoke the configured real model and retain inspectable evidence."""

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


def _captured_history() -> list[dict[str, object]]:
    """Return the ten logical turns projected in the failed production run."""

    messages = [
        ('2026-07-31 01:55', '就只够修bug了'),
        ('2026-07-31 01:55', '我上次不小心把codex给关了也是这个感觉'),
        ('2026-07-31 01:56', '气得我用了一张重置卡'),
        ('2026-07-31 01:57', '你忍得下去？'),
        ('2026-07-31 01:57', '千纱要是没蹬玩任务卡一半我当场就去世了'),
        ('2026-07-31 02:00', '@xxxsam 7月大概三天重置一次'),
        ('2026-07-31 09:53', '@总是跌倒的企鹅 万一没有呢'),
        ('2026-07-31 15:12', '嗨'),
        ('2026-07-31 15:13', '信oai还不如信饿哦'),
        ('2026-07-31 15:14', '信oai还不如信我'),
    ]
    history = [
        {
            "role": "user",
            "timestamp": timestamp,
            "display_name": "蚝爹油",
            "body_text": body_text,
            "platform_user_id": "673225019",
            "global_user_id": "4759394b-a4d2-4634-9d12-b6423a92a248",
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "reply_context": {},
            "llm_trace_id": "",
        }
        for timestamp, body_text in messages
    ]
    return history


def _captured_state(
    episode: dict[str, object],
) -> dict[str, object]:
    """Build the resolver-facing state from the failed QQ turn."""

    turn_clock = build_turn_clock("2026-08-02 02:35:53")
    history = _captured_history()
    return {
        "decontextualized_input": (
            "一之濑明日奈，当前用户请求搜索 RTX 5090 的实时价格。"
        ),
        "referents": [],
        "character_profile": {
            "name": "一之濑明日奈",
            "global_user_id": "00000000-0000-4000-8000-000000000001",
        },
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "platform_message_id": "908689583",
        "platform_bot_id": "3768713357",
        "global_user_id": "4759394b-a4d2-4634-9d12-b6423a92a248",
        "platform_user_id": "673225019",
        "user_name": "蚝爹油",
        "user_profile": {},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "prompt_message_context": {
            "body_text": "@一之濑明日奈 明日奈酱，能帮我搜一下rtx5090的实时价格么？",
            "mentions": [{
                "platform_user_id": "3768713357",
                "global_user_id": "00000000-0000-4000-8000-000000000001",
                "display_name": "一之濑明日奈",
                "entity_kind": "bot",
                "raw_text": "[CQ:at,qq=3768713357]",
            }],
            "attachments": [],
            "addressed_to_global_user_ids": [
                "00000000-0000-4000-8000-000000000001",
            ],
            "broadcast": False,
        },
        "channel_topic": "QQ group hardware discussion",
        "chat_history_recent": history,
        "chat_history_wide": history,
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": {},
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
        "active_turn_platform_message_ids": ["908689583"],
        "active_turn_conversation_row_ids": [
            "6a6e04495d9c0243331865a3",
        ],
        "cognitive_episode": episode,
    }


async def test_live_captured_rtx_price_turn_bounds_task_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live evidence route receives bounded task-resolution history."""

    user_input = '@一之濑明日奈 明日奈酱，能帮我搜一下rtx5090的实时价格么？'
    episode = canonical_episode(
        episode_id="user_message:qq:54369546:908689583",
        content=user_input,
        current_global_user_id="4759394b-a4d2-4634-9d12-b6423a92a248",
    )
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:user_message:qq:54369546:908689583",
            "occurred_at": "2026-08-01T14:35:53.558061+00:00",
            "semantic_summary": user_input,
        },
        "semantic_text": user_input,
        "visible_to": ["q:event_agency"],
    }]
    bid = {
        "branch_id": "ordinary_response",
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal:rtx5090-live-price",
        },
        "intention": "搜索并核实 RTX 5090 当前的实时价格。",
        "desired_outcome": "向当前用户提供有来源边界的当前价格信息。",
        "concrete_detail": "当前价格属于尚未取得的外部证据。",
        "reason": "当前用户明确要求搜索实时价格，现有证据不包含价格。",
        "private_monologue": "我需要先取得当前价格证据，再回答用户。",
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "4759394b-a4d2-4634-9d12-b6423a92a248",
        }],
        "evidence_handles": ["e1"],
        "expected_consequences": ["用户得到经过检索的当前价格信息"],
        "confidence": "high",
    }
    available_resolvers = [{
        "capability": "task_resolution_request",
        "semantic_capability": "解决需要当前外部证据的有界任务。",
        "availability": "available",
    }]
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)

    action_result = await plan_actions(
        primary_bid=bid,
        supporting_bids=[],
        episode=episode,
        evidence=evidence,
        available_actions=[],
        available_resolvers=available_resolvers,
        resolver_context="resolver_state: status=idle",
        runtime_capability_limits=[],
        services=services,
    )
    assert [
        request["capability"]
        for request in action_result["resolver_requests"]
    ] == ["task_resolution_request"]
    resolver_request = action_result["resolver_requests"][0]
    request = {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": resolver_request["capability"],
        "objective": resolver_request["semantic_goal"],
        "reason": resolver_request["reason"],
        "priority": "now",
    }

    captured_execution: dict[str, object] = {}

    async def resolve_task_inline(
        task_request: dict[str, object],
        execution_context: dict[str, object],
        *,
        inline_budget_seconds: float,
    ) -> dict[str, object]:
        """Return a bounded result once context validation succeeds."""

        captured_execution["task_request"] = task_request
        captured_execution["execution_context"] = execution_context
        captured_execution["inline_budget_seconds"] = inline_budget_seconds
        return {
            "schema_version": "task_resolution_result.v1",
            "status": "unavailable",
            "prompt_safe_summary": "No public price source was configured.",
            "evidence": [],
            "completed_subgoals": [],
            "remaining_needs": [],
            "checkpoint": {},
            "coding_run_context": {},
        }

    monkeypatch.setattr(
        capabilities_module,
        "resolve_task_inline",
        resolve_task_inline,
    )
    observation: dict[str, object] | None = None
    error = ""
    try:
        observation = (
            await capabilities_module.execute_resolver_capability_request(
                request,
                _captured_state(episode),
            )
        )
    except TaskResolutionContractError as exc:
        error = f"{type(exc).__name__}: {exc}"

    trace_path = write_llm_trace(
        "task_resolution_history_cap_live_llm",
        "chat_qq_ch_73987da21ae6b88a_908689583_after_fix",
        {
            "source_correlation_id": (
                "chat:qq:ch_73987da21ae6b88a:908689583"
            ),
            "input_kind": "captured_failure",
            "user_input": user_input,
            "captured_chat_history_recent": _captured_history(),
            "model_calls": capturing_llm.calls,
            "action_result": action_result,
            "resolver_request": request,
            "bounded_execution_context": captured_execution.get(
                "execution_context"
            ),
            "resolver_observation": observation,
            "execution_error": error,
        },
    )
    print(json.dumps({
        "trace_path": str(trace_path),
        "action_result": action_result,
        "bounded_history_count": len(
            captured_execution["execution_context"]["chat_history_recent"]
        ),
        "resolver_observation": observation,
        "execution_error": error,
    }, ensure_ascii=False, indent=2))

    expected_history = _captured_history()[-8:]
    execution_context = captured_execution["execution_context"]
    assert execution_context["chat_history_recent"] == expected_history
    assert execution_context["chat_history_wide"] == expected_history
    assert observation is not None
    assert observation["status"] == "failed"
    assert error == ""
