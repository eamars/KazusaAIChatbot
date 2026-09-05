"""Tests for cognition resolver capability execution."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
from kazusa_ai_chatbot.cognition_resolver import capabilities as capabilities_module
from kazusa_ai_chatbot.cognition_resolver import loop as loop_module
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    PENDING_TASK_CONTINUATION_VERSION,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_GOAL_PROGRESS_VERSION,
    RESOLVER_OBSERVATION_VERSION,
    RESOLVER_PENDING_RESOLUTION_VERSION,
    RESOLVER_PENDING_RESUME_VERSION,
    ResolverValidationError,
)
from kazusa_ai_chatbot.cognition_resolver.loop import call_cognition_resolver_loop
from kazusa_ai_chatbot.cognition_resolver.pending import (
    RESOLVER_PENDING_APPROVAL_ACTION_KIND,
    RESOLVER_PENDING_HIL_ACTION_KIND,
    apply_pending_resolution,
    build_pending_resume_record,
    load_matching_pending_resume,
    load_matching_pending_resume_into_state,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    ensure_initial_resolver_inputs,
    project_resolver_context,
)
from kazusa_ai_chatbot.cognition_resolver.telemetry import (
    build_resolver_cycle_event,
    build_resolver_terminal_event,
    write_human_readable_resolver_trace,
)
from kazusa_ai_chatbot.cognition_shared.contracts import CognitionExecutionError
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_test_helpers import (
    canonical_character_identity,
    canonical_episode,
    canonical_identity_context,
)


def _resolver_request(
    *,
    capability_kind: str = "task_resolution_request",
    objective: str = "检索当前用户与这个问题有关的关系和记忆证据。",
) -> dict:
    return {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": capability_kind,
        "objective": objective,
        "reason": "当前认知循环缺少足够证据。",
        "priority": "now",
        "goal_continuation_ref": (
            _resolver_goal_ref()
            if capability_kind == "task_resolution_request"
            else None
        ),
    }


def _resolver_goal_ref() -> dict[str, object]:
    """Build the continuation identity shared by resolver loop fixtures."""

    return build_goal_continuation_ref(
        source_episode_id="resolver-capability-episode",
        source_message_id="message-123",
        branch_id="ordinary_response",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "resolver-goal-001",
        },
    )


def _resolver_scene_context() -> dict[str, object]:
    """Build one bounded scene for resolver capability tests."""

    return {
        "channel_scope": "private",
        "character_role": "Kazusa",
        "current_user_role": "Test User",
        "semantic_scene": "The resolver is handling one bounded evidence goal.",
        "public_group_scene": "",
        "conversation_continuity": "The current turn continues the evidence goal.",
        "semantic_temporal_context": "The current test turn is active.",
    }


def _task_observation_fields(request: dict) -> dict[str, object]:
    """Add the continuation field only to task-resolution observations."""

    if request["capability_kind"] != "task_resolution_request":
        return {}
    return {"goal_continuation_ref": request["goal_continuation_ref"]}


def _task_result(
    *,
    status: str = "resolved",
    specialist: str = "dsh",
    summary: str = "找到一条相关证据。",
    semantic_objective: str = "检索当前用户与这个问题有关的关系和记忆证据。",
) -> dict[str, object]:
    """Build one canonical task-resolution result for resolver tests."""

    evidence = []
    if status in {"resolved", "partial"}:
        evidence = [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "evidence-1",
            "task_node_id": "node-1",
            "specialist": specialist,
            "summary": "evidence-1",
            "provenance_refs": ["source:item-1"],
            "limitations": [],
        }]
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": semantic_objective,
        "status": status,
        "scene_context": _resolver_scene_context(),
        "goal_continuation_ref": _resolver_goal_ref(),
        "evidence_state": {
            "resolved": "complete",
            "partial": "partial",
            "deferred": "pending",
            "needs_user_input": "pending",
            "approval_required": "pending",
            "unavailable": "missing",
            "failed": "blocked",
        }.get(status, "blocked"),
        "evidence_excerpts": [summary]
        if status in {"resolved", "partial"}
        else [],
        "evidence_handles": ["evidence-1"]
        if status in {"resolved", "partial"}
        else [],
        "prompt_safe_summary": summary,
        "evidence": evidence,
        "completed_subgoals": [],
        "remaining_needs": (
            ["需要用户补充缺失指代。"]
            if status == "needs_user_input"
            else ["The bounded task requires a truthful terminal disposition."]
            if status in {"unavailable", "failed"}
            else []
        ),
        "checkpoint": {},
        "coding_run_context": {},
    }


def _resolver_state() -> dict:
    turn_clock = build_turn_clock("2026-05-30 09:00:00")
    episode = canonical_episode(
        episode_id="resolver-capability-episode",
        content="Need an evidence-backed answer.",
        current_global_user_id="global-user-123",
    )
    return {
        "decontextualized_input": "Original user request about trust.",
        "referents": [],
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-123",
        },
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "channel_type": "private",
        "platform_message_id": "message-123",
        "platform_bot_id": "bot-123",
        "global_user_id": "global-user-123",
        "platform_user_id": "platform-user-123",
        "user_name": "Test User",
        "user_profile": {"relationship_state": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "prompt_message_context": {
            "body_text": "Need an evidence-backed answer.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-123"],
            "broadcast": False,
        },
        "channel_topic": "debug",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": {
            "current_thread": "trust question",
        },
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
        "cognition_scene_context": _resolver_scene_context(),
        "active_turn_platform_message_ids": ["message-123"],
        "active_turn_conversation_row_ids": ["row-123"],
        "cognitive_episode": episode,
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "no_task_admission",
        },
    }


def _internal_thought_resolver_state() -> dict:
    """Return resolver state for a private internal-thought cognition source."""

    state = _resolver_state()
    episode = canonical_episode(
        episode_id="resolver-internal-thought-episode",
        trigger_source="internal_thought",
        content="整理一个内部目标。",
    )
    state["cognitive_episode"] = episode
    state["channel_type"] = "group"
    return_value = state
    return return_value


def _cognition_result(
    *,
    internal_monologue: str,
    action_specs: list[dict] | None = None,
    resolver_requests: list[dict] | None = None,
    goal_resolution: str = "requires_required_evidence",
) -> dict:
    return {
        "internal_monologue": internal_monologue,
        "interaction_subtext": f"{internal_monologue} 的互动潜台词",
        "emotional_appraisal": f"{internal_monologue} 的情绪判断",
        "character_intent": f"{internal_monologue} 的角色意图",
        "logical_stance": f"{internal_monologue} 的逻辑立场",
        "judgment_note": f"{internal_monologue} 的判断备注",
        "social_distance": "close",
        "emotional_intensity": "low",
        "vibe_check": "steady",
        "relational_dynamic": "trusted",
        "action_specs": action_specs or [],
        "resolver_capability_requests": resolver_requests or [],
        "goal_resolution": goal_resolution,
    }


def _speak_action_spec(
    reason: str = "已经有足够证据，可以进入可见回复。",
    *,
    surface_role: str = "ordinary",
    goal_continuation_ref: dict[str, object] | None = None,
) -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "speak",
        "cognition_mode": "deliberative",
        "surface_role": surface_role,
        "goal_continuation_ref": goal_continuation_ref,
        "source_refs": [],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_user",
            "target_id": None,
            "owner": "l2d",
            "scope": {},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {},
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": reason,
    }


def _pending_resume(*, capability_kind: str = "human_clarification") -> dict:
    status = "waiting_for_user"
    question = "你现在在哪个城市？"
    approval_summary = ""
    if capability_kind == "approval_preparation":
        status = "waiting_for_approval"
        question = ""
        approval_summary = "准备创建提醒，但需要用户确认。"
    pending_resume = {
        "schema_version": RESOLVER_PENDING_RESUME_VERSION,
        "resume_id": f"resolver-pending-{capability_kind}",
        "capability_kind": capability_kind,
        "status": status,
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "global_user_id": "global-user-123",
        "source_message_id": "previous-message-123",
        "prompt_safe_original_goal": "Original user request about trust.",
        "prompt_safe_question": question,
        "prompt_safe_approval_summary": approval_summary,
        "created_at_utc": "2026-05-29T21:00:00+00:00",
        "expires_at_utc": "2026-05-30T21:00:00+00:00",
    }
    if capability_kind == "human_clarification":
        pending_resume["pending_task_continuation"] = {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "no_task_admission",
        }
    return pending_resume


def _pending_resolution(
    *,
    decision: str = "answered",
    resume_id: str = "resolver-pending-human_clarification",
) -> dict:
    return {
        "schema_version": RESOLVER_PENDING_RESOLUTION_VERSION,
        "resume_id": resume_id,
        "decision": decision,
        "reason": "用户已经回答了澄清问题。",
    }


def _goal_progress(*, focus: str = "继续完成原始目标。") -> dict:
    return {
        "schema_version": RESOLVER_GOAL_PROGRESS_VERSION,
        "original_goal": "今晚安排一个两小时低预算计划。",
        "current_focus": focus,
        "deliverables": [
            {
                "description": "晚餐候选和证据边界",
                "status": "partial",
                "note": "候选方向已有，实时营业仍需 caveat。",
            },
            {
                "description": "两小时散步路线和时间切分",
                "status": "pending",
                "note": "最终回复必须覆盖。",
            },
        ],
        "missing_user_inputs": [],
        "evidence_dependencies": ["当前营业状态"],
        "attempted_paths": [],
        "source_backed_facts": ["用户在奥克兰 CBD，预算 20 NZD"],
        "assumptions_or_inferences": ["可以给出海滨散步路线骨架"],
        "blockers": ["无法确认所有店 19:30 营业"],
        "final_response_requirements": [
            "覆盖晚餐、散步、时间切分和核实清单",
        ],
    }














async def _hil_pending_trace_state() -> dict:
    """Build a resolver result containing one pending HIL terminal trace."""

    request = _resolver_request(
        capability_kind="human_clarification",
        objective="请只问用户所在城市。",
    )

    async def call_cognition(state: dict) -> dict:
        resolver_context = state["resolver_context"]
        if "pending_resolver_resume" not in resolver_context:
            return _cognition_result(
                internal_monologue="第一轮：缺少用户城市",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="第二轮：提出最小澄清问题",
            action_specs=[_speak_action_spec("只询问用户所在城市。")],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_hil_city",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "prompt_safe_summary": (
                "Human clarification required: 请只问用户所在城市。"
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    async def upsert_pending_resume(state: dict, observation: dict) -> dict:
        record = build_pending_resume_record(state, observation)
        return record["execution_result"]["pending_resume"]

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending_resume,
    )
    return_value = result
    return return_value
















def test_loop_blocks_rephrased_task_request_with_same_continuation_ref() -> None:
    """One typed goal cannot be admitted twice under paraphrased objectives."""

    first_request = _resolver_request(
        objective="Resolve the bounded task objective.",
    )
    rephrased_request = _resolver_request(
        objective="Resolve the same bounded task with different wording.",
    )
    resolver_state = {
        "observations": [{
            "capability_kind": "task_resolution_request",
            "request_objective": first_request["objective"],
            "status": "succeeded",
            "goal_continuation_ref": first_request["goal_continuation_ref"],
        }],
    }

    assert loop_module._is_repeated_capability_request(
        rephrased_request,
        resolver_state,
    ) is True


def _production_resolver_state(*, original_goal: str) -> dict:
    """Build resolver state that crosses the production cognition projector."""

    state = _resolver_state()
    timestamp = "2026-05-29T21:00:00Z"
    state.pop("conversation_progress", None)
    profile = canonical_character_identity(marker="resolver-continuity")
    profile["global_user_id"] = "character-123"
    state["character_profile"] = profile
    state["character_identity_context"] = canonical_identity_context(
        marker="resolver-continuity",
    )
    state["character_cognition_state"] = build_character_production_state(
        updated_at=timestamp,
    )
    state["cognition_state"] = build_acquaintance_user_state(
        global_user_id="global-user-123",
        updated_at=timestamp,
    )
    state["decontextualized_input"] = original_goal
    return state










































@pytest.mark.asyncio
async def test_pending_helpers_load_and_close_matching_pending_rows() -> None:
    """Pending helper should filter by scope, expiry, and L2d resolution."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_hil_city",
        "capability_kind": "human_clarification",
        "request_objective": "你现在在哪个城市？",
        "request_reason": "缺少用户所在地。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 你现在在哪个城市？",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    record = build_pending_resume_record(state, observation)
    expired_record = build_pending_resume_record(
        state,
        observation,
        expires_at_utc="2026-05-29T20:00:00+00:00",
    )
    expired_record["attempt_id"] = "expired"
    expired_record["execution_result"]["pending_resume"]["resume_id"] = "expired"
    expired_record["resolver_pending_resume"]["resume_id"] = "expired"
    rows = [expired_record, record]
    upserted: list[dict] = []

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return list(rows)

    async def upsert_row(row: dict) -> None:
        upserted.append(row)

    follow_up_state = dict(state)
    follow_up_state["platform_message_id"] = "follow-up-message-id"
    follow_up_state["storage_timestamp_utc"] = "2026-05-29T21:05:00+00:00"
    follow_up_state["reply_context"] = {
        "reply_to_message_id": "message-123",
    }

    loaded = await load_matching_pending_resume(
        follow_up_state,
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert loaded["resume_id"] == (
        record["execution_result"]["pending_resume"]["resume_id"]
    )
    assert upserted[0]["status"] == "expired"

    resume_id = record["execution_result"]["pending_resume"]["resume_id"]
    await apply_pending_resolution(
        follow_up_state,
        _pending_resolution(resume_id=resume_id),
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert upserted[-1]["status"] == "closed"
    assert upserted[-1]["execution_result"]["pending_resolution"]["decision"] == (
        "answered"
    )
    assert upserted[-1]["execution_result"][
        "resolver_pending_resolution"
    ]["decision"] == "answered"

    approval_observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_approval",
        "capability_kind": "approval_preparation",
        "request_objective": "准备创建提醒，但需要用户确认。",
        "request_reason": "侧效应需要确认。",
        "status": "blocked",
        "prompt_safe_summary": "Approval required: 准备创建提醒。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    approval_record = build_pending_resume_record(state, approval_observation)
    rows = [approval_record]
    upserted.clear()
    approval_resume_id = approval_record["resolver_pending_resume"]["resume_id"]

    await apply_pending_resolution(
        follow_up_state,
        _pending_resolution(
            decision="approved",
            resume_id=approval_resume_id,
        ),
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert upserted[-1]["status"] == "closed"
    assert upserted[-1]["execution_result"][
        "resolver_pending_resolution"
    ]["decision"] == "approved"

    superseded_record = build_pending_resume_record(state, observation)
    rows = [superseded_record]
    upserted.clear()
    superseded_resume_id = superseded_record["resolver_pending_resume"][
        "resume_id"
    ]
    await apply_pending_resolution(
        follow_up_state,
        _pending_resolution(
            decision="superseded",
            resume_id=superseded_resume_id,
        ),
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert upserted[-1]["status"] == "superseded"
    assert upserted[-1]["execution_result"][
        "resolver_pending_resolution"
    ]["decision"] == "superseded"


@pytest.mark.asyncio
async def test_superseded_pending_row_leaves_the_next_turn_unbound() -> None:
    """A superseded clarification cannot constrain fresh normal cognition."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_superseded_pending",
        "capability_kind": "human_clarification",
        "request_objective": "请补充原始任务所需的用户事实。",
        "request_reason": "缺少一个用户控制的事实。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required.",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    record = build_pending_resume_record(state, observation)
    rows = [record]

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return list(rows)

    async def upsert_row(row: dict) -> None:
        rows[:] = [row]

    resume_id = record["resolver_pending_resume"]["resume_id"]
    updated = await apply_pending_resolution(
        state,
        _pending_resolution(decision="superseded", resume_id=resume_id),
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert updated is not None
    assert updated["status"] == "superseded"
    fresh_state = ensure_initial_resolver_inputs(_resolver_state(), max_cycles=3)
    fresh_state["platform_message_id"] = "fresh-message-after-supersession"
    fresh_state = await load_matching_pending_resume_into_state(
        fresh_state,
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )
    assert "pending_resolver_resume" not in fresh_state
    fresh_production_state = ensure_initial_resolver_inputs(
        _production_resolver_state(
            original_goal="Handle the replacement request as a fresh task.",
        ),
        max_cycles=3,
    )
    fresh_input = build_cognition_input_from_global_state(
        fresh_production_state,
    )
    assert fresh_input["response_plan_contract_variant"] == "fresh_ordinary"


@pytest.mark.asyncio
async def test_pending_loader_ignores_future_pending_rows() -> None:
    """Replay or delayed turns must not load pending rows from the future."""

    current_state = _resolver_state()
    current_state["storage_timestamp_utc"] = "2026-05-29T21:00:00+00:00"
    future_state = dict(current_state)
    future_state["storage_timestamp_utc"] = "2026-05-30T21:00:00+00:00"
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_future_hil",
        "capability_kind": "human_clarification",
        "request_objective": "确认未来消息里的对象。",
        "request_reason": "缺少未来消息中的指代对象。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 确认未来消息里的对象。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-30T21:00:00+00:00",
    }
    future_record = build_pending_resume_record(future_state, observation)

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [future_record]

    loaded = await load_matching_pending_resume(
        current_state,
        list_action_attempts_func=list_rows,
    )

    assert loaded is None


@pytest.mark.asyncio
async def test_pending_loader_ignores_same_source_message_rows() -> None:
    """A source message should not resume the pending row it created."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_same_message_hil",
        "capability_kind": "human_clarification",
        "request_objective": "确认当前消息里的对象。",
        "request_reason": "缺少当前消息中的指代对象。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 确认当前消息里的对象。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    same_message_record = build_pending_resume_record(state, observation)

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [same_message_record]

    loaded = await load_matching_pending_resume(
        state,
        list_action_attempts_func=list_rows,
    )

    assert loaded is None


@pytest.mark.asyncio
async def test_pending_loader_selects_unrelated_same_scope_candidate() -> None:
    """Cognition decides whether an ordinary same-scope turn answers pending."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_unrelated_hil",
        "capability_kind": "human_clarification",
        "request_objective": "确认原始计划所需的用户信息。",
        "request_reason": "原始计划缺少用户输入。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 确认原始计划所需的用户信息。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    pending_record = build_pending_resume_record(state, observation)
    unrelated_state = dict(state)
    unrelated_state["platform_message_id"] = "unrelated-message-456"
    unrelated_state["decontextualized_input"] = "我今天心情变好了，谢谢你陪我说话。"

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [pending_record]

    loaded = await load_matching_pending_resume(
        unrelated_state,
        list_action_attempts_func=list_rows,
    )

    assert loaded is not None
    assert loaded["resume_id"] == pending_record["resolver_pending_resume"][
        "resume_id"
    ]


@pytest.mark.asyncio
async def test_pending_loader_selects_newest_candidate_without_reply_metadata() -> None:
    """Ordinary adjacency selects the newest exact-scope clarification row."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_newest_hil",
        "capability_kind": "human_clarification",
        "request_objective": "确认当前计划缺少的用户事实。",
        "request_reason": "原始目标仍缺少用户控制的信息。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 缺少用户事实。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    older_state = dict(state)
    older_state["storage_timestamp_utc"] = "2026-05-29T20:00:00+00:00"
    newer_state = dict(state)
    newer_state["storage_timestamp_utc"] = "2026-05-29T20:30:00+00:00"
    older_record = build_pending_resume_record(older_state, observation)
    newer_record = build_pending_resume_record(newer_state, observation)
    follow_up_state = dict(state)
    follow_up_state["storage_timestamp_utc"] = "2026-05-29T21:00:00+00:00"
    follow_up_state["platform_message_id"] = "ordinary-follow-up-456"
    follow_up_state["reply_context"] = {}

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [older_record, newer_record]

    loaded = await load_matching_pending_resume(
        follow_up_state,
        list_action_attempts_func=list_rows,
    )

    assert loaded is not None
    assert loaded["resume_id"] == newer_record["resolver_pending_resume"][
        "resume_id"
    ]




@pytest.mark.asyncio
async def test_pending_resume_load_restores_original_goal_progress() -> None:
    """HIL follow-up turns should inherit the first-turn deliverable checklist."""

    state = ensure_initial_resolver_inputs(_resolver_state(), max_cycles=3)
    resolver_state = dict(state["resolver_state"])
    resolver_state["goal_progress"] = _goal_progress()
    state["resolver_state"] = resolver_state
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_hil_city",
        "capability_kind": "human_clarification",
        "request_objective": "你在奥克兰哪个区域？",
        "request_reason": "缺少用户所在地。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 你在奥克兰哪个区域？",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    record = build_pending_resume_record(state, observation)
    follow_up_state = _resolver_state()
    follow_up_state["platform_message_id"] = "message-456"
    follow_up_state["decontextualized_input"] = "就在奥克兰 CBD。"
    follow_up_state["reply_context"] = {}
    follow_up_state = ensure_initial_resolver_inputs(
        follow_up_state,
        max_cycles=3,
    )

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [record]

    async def upsert_row(row: dict) -> None:
        del row

    loaded_state = await load_matching_pending_resume_into_state(
        follow_up_state,
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    loaded_progress = loaded_state["resolver_state"]["goal_progress"]
    assert loaded_progress["original_goal"] == (
        "今晚安排一个两小时低预算计划。"
    )
    assert loaded_progress["deliverables"][1]["description"] == (
        "两小时散步路线和时间切分"
    )
    assert loaded_state["resolver_state"]["original_decontextualized_input"] == (
        "今晚安排一个两小时低预算计划。"
    )
    assert "就在奥克兰 CBD" not in loaded_state["resolver_context"]


















@pytest.mark.asyncio
async def test_empty_resolver_objective_fails_before_task_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed resolver requests fail before unified task dispatch."""

    resolve_task_inline = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "resolve_task_inline",
        resolve_task_inline,
    )
    request = _resolver_request(objective=" ")

    with pytest.raises(ResolverValidationError, match="objective"):
        await capabilities_module.execute_resolver_capability_request(
            request,
            _resolver_state(),
        )

    resolve_task_inline.assert_not_awaited()


@pytest.mark.asyncio
async def test_blocked_capabilities_return_prompt_safe_observations() -> None:
    """Clarification and approval capabilities should block without side effects."""

    state = _resolver_state()

    clarification = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="human_clarification",
            objective="请只问用户所在城市。",
        ),
        state,
    )
    approval = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="approval_preparation",
            objective="说明准备创建提醒，但等待用户确认。",
        ),
        state,
    )

    assert clarification["status"] == "blocked"
    assert clarification["capability_kind"] == "human_clarification"
    assert "请只问用户所在城市" in clarification["prompt_safe_summary"]
    assert approval["status"] == "blocked"
    assert approval["capability_kind"] == "approval_preparation"
    assert "等待用户确认" in approval["prompt_safe_summary"]
    assert "approval preparation only" in approval["prompt_safe_summary"]
    assert "file inspection" in approval["prompt_safe_summary"]


@pytest.mark.asyncio
async def test_self_goal_resolution_blocks_user_message_source() -> None:
    """User-message turns must not spawn private self-goal execution."""

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="self_goal_resolution",
            objective="整理一个内部目标。",
        ),
        _resolver_state(),
    )

    assert observation["status"] == "blocked"
    assert observation["capability_kind"] == "self_goal_resolution"


@pytest.mark.asyncio
async def test_self_goal_resolution_allows_internal_thought_source() -> None:
    """Internal thought may produce a private self-resolution observation."""

    state = _resolver_state()
    episode = canonical_episode(
        episode_id="resolver-internal-thought-episode",
        trigger_source="internal_thought",
        content="整理一个内部目标。",
    )
    state["cognitive_episode"] = episode

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="self_goal_resolution",
            objective="整理一个内部目标。",
        ),
        state,
    )

    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "self_goal_resolution"
    assert "internal cognition source" in observation["prompt_safe_summary"]
    assert "rag_result" not in observation
