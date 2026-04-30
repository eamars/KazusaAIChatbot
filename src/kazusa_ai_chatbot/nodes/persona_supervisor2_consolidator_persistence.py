"""Stage 4 consolidator persistence and scheduling helpers."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import (
    AFFINITY_DECREMENT_BREAKPOINTS,
    AFFINITY_DEFAULT,
    AFFINITY_INCREMENT_BREAKPOINTS,
    AFFINITY_RAW_DEAD_ZONE,
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.db import (
    update_affinity,
    update_last_relationship_insight,
    upsert_character_self_image,
    upsert_character_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_images import (
    _update_character_image,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_memory_units import (
    update_user_memory_units_from_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import (
    ConsolidatorState,
)
from kazusa_ai_chatbot.dispatcher import (
    DispatchContext,
    RawToolCall,
    TaskDispatcher,
    ToolRegistry,
)
from kazusa_ai_chatbot.dispatcher.task import parse_iso_datetime
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.utils import get_llm, log_list_preview, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_task_dispatcher: TaskDispatcher | None = None
_task_registry: ToolRegistry | None = None

_TASK_DISPATCHER_PROMPT = """\
你负责把角色已经接受的未来约定，转换成可执行的工具调用。

# 目标
- 输入是角色本轮最终说出口的话、已提取的 `future_promises`、来源平台上下文，以及当前可用工具。
- 输出必须是零个或多个工具调用；若没有把握，就输出空列表。

# 规则
1. 只能使用 `available_tools` 中出现的工具名与参数字段。
2. 所有延迟执行任务都必须使用绝对 UTC 时间 `execute_at`（ISO 8601）；禁止输出相对时间字段。
2a. 输入里的 `source_channel_type` 表示来源会话类型，常见值有 `group` 和 `private`。
2b. 如果某个 `future_promises` 已经给出 `due_time`，那就是权威执行时间；`execute_at` 必须与该 `due_time` 表示同一个精确时刻，不能改写时区含义，不能把本地时间原样误写成 UTC。
3. 若要回到原会话发送消息，使用 `target_channel: "same"`。
4. 若目标平台与来源平台相同，可以省略 `target_platform`。
4a. 如果用户明确指定了另一个群/频道/房间 ID（例如 `54369546群`），`target_channel` 必须写那个明确的 ID，而不是 `"same"`。
4b. 若来源是私聊/DM，但用户要求发到另一个群或频道，仍然按用户指定的目标 ID 填写 `target_channel`；不要因为来源是私聊就回退到 `"same"`。
4c. `target_channel` 必须是平台实际使用的纯 ID 字符串；不要保留“群”“频道”“房间”“#”之类的人类描述后缀。
5. 对“持续生效的回复规则/称呼规则/语言偏好/格式约定”这类承诺，如果 `due_time` 为 null，说明它属于长期状态，不是未来某个时刻要发送的新消息。此时必须返回空列表。
5a. 但如果 `commitment_type` 是 `future_promise`，且语义是模糊但明确的近未来（例如 `later`、`一会儿`、`待会`、`稍后`），并且 `due_time` 为 null，你可以默认 `execute_at = current_utc + 5 minutes`。仍然必须输出绝对 UTC 时间，不能输出相对时间。
6. 只有“未来某个时刻真的要额外发送一条消息”时，才生成 `send_message`。
7. 若无法形成可靠工具调用，返回空列表，不要解释。
8. `text` 是届时角色真正要发送的消息正文，不要只是复述 promise 字段。

# 生成步骤
1. 先读取 `future_promises`，只保留仍然需要在未来某一刻额外执行的承诺。
2. 对每个候选承诺，检查 `available_tools` 是否存在可执行工具及必要参数字段。
3. 根据 `due_time` 或允许的近未来默认规则确定绝对 UTC `execute_at`。
4. 根据来源会话和用户明确指定的目标，确定 `target_platform` 与 `target_channel`。
5. 生成届时角色真正要发送的 `text`。如果无法可靠生成工具调用，返回空数组。

# 输入格式
{{
  "instruction": "当前调度意图摘要",
  "current_utc": "当前 UTC ISO 时间",
  "source_platform": "来源平台",
  "source_channel_id": "来源会话 ID",
  "source_channel_type": "group | private",
  "source_message_id": "来源消息 ID",
  "decontexualized_input": "用户本轮真实意图摘要",
  "final_dialog": ["角色本轮最终实际说出口的话"],
  "content_anchors": ["回复前的内容锚点"],
  "future_promises": [
    {{
      "target": "承诺目标",
      "action": "可执行承诺本体",
      "due_time": "ISO 时间或 null",
      "commitment_type": "future_promise | language_preference | address_preference | other"
    }}
  ],
  "available_tools": [
    {{
      "name": "工具名",
      "description": "工具说明",
      "args_schema": {{"参数名": "参数定义"}}
    }}
  ]
}}

# 输出格式
请只返回合法 JSON：
{{
  "tool_calls": [
    {{
      "tool": "工具名",
      "args": {{
        "参数名": "参数值"
      }}
    }}
  ]
}}
"""
_task_dispatcher_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


def configure_task_dispatcher(dispatcher: TaskDispatcher, tool_registry: ToolRegistry) -> None:
    """Install the runtime dispatcher used by the consolidator background path.

    Args:
        dispatcher: Task dispatcher instance shared by the service runtime.
        tool_registry: Tool registry used to expose visible tool specs to the LLM.
    """

    global _task_dispatcher, _task_registry
    _task_dispatcher = dispatcher
    _task_registry = tool_registry


def _get_task_dispatcher() -> TaskDispatcher | None:
    """Return the configured task dispatcher, if the service installed one."""

    return _task_dispatcher


def _build_dispatch_context(state: ConsolidatorState, *, timestamp: str) -> DispatchContext:
    """Build the dispatcher context from consolidator state.

    Args:
        state: Consolidator state for the current turn.
        timestamp: Current turn timestamp.

    Returns:
        Source-side dispatch context.
    """

    try:
        now = parse_iso_datetime(timestamp)
    except ValueError as exc:
        logger.debug(f"Handled exception in _build_dispatch_context: {exc}")
        now = datetime.now(timezone.utc)

    return_value = DispatchContext(
        source_platform=state.get("platform", ""),
        source_channel_id=state.get("platform_channel_id", ""),
        source_user_id=state.get("global_user_id", ""),
        source_message_id=state.get("platform_message_id", ""),
        guild_id=None,
        bot_permission_role="user",
        now=now,
    )
    return return_value


def _build_dispatch_instruction(state: ConsolidatorState) -> str:
    """Summarize the current turn intent for tracing and task generation.

    Args:
        state: Consolidator state for the current turn.

    Returns:
        Natural-language instruction summary.
    """

    character_profile = state.get("character_profile", {})
    character_name = character_profile["name"]
    user_name = state.get("user_name", "user")
    promise_count = len(state.get("future_promises") or [])
    return_value = (
        f"{character_name} should follow through on {promise_count} accepted promise(s)"
        f" for {user_name} based on the finalized dialog."
    )
    return return_value


async def _generate_raw_tool_calls(
    state: ConsolidatorState,
    ctx: DispatchContext,
) -> list[RawToolCall]:
    """Ask the LLM to convert accepted promises into raw tool calls.

    Args:
        state: Consolidator state containing finalized dialog and harvested promises.
        ctx: Dispatch context used to expose source defaults and visible tools.

    Returns:
        Zero or more raw tool calls emitted by the LLM.
    """

    if _task_registry is None:
        return_value = []
        return return_value

    available_tools = _task_registry.filter(ctx)
    future_promises = _normalize_future_promises(
        state.get("future_promises") or [],
        timestamp=ctx.now.isoformat(),
    )
    if not available_tools or not future_promises:
        return_value = []
        return return_value

    action_directives = state.get("action_directives", {})
    linguistic_directives = action_directives.get("linguistic_directives", {})
    content_anchors = linguistic_directives.get("content_anchors", [])

    msg = {
        "instruction": _build_dispatch_instruction(state),
        "current_utc": ctx.now.isoformat(),
        "source_platform": ctx.source_platform,
        "source_channel_id": ctx.source_channel_id,
        "source_channel_type": state.get("channel_type", "group"),
        "source_message_id": ctx.source_message_id,
        "decontexualized_input": state.get("decontexualized_input", ""),
        "final_dialog": state.get("final_dialog", []),
        "content_anchors": content_anchors,
        "future_promises": future_promises,
        "available_tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "args_schema": spec.args_schema,
            }
            for spec in available_tools
        ],
    }
    response = await _task_dispatcher_llm.ainvoke(
        [
            SystemMessage(content=_TASK_DISPATCHER_PROMPT),
            HumanMessage(content=json.dumps(msg, ensure_ascii=False)),
        ]
    )
    result = parse_llm_json_output(response.content)
    tool_calls = result.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return_value = []
        return return_value

    raw_calls: list[RawToolCall] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool", "")).strip()
        args = item.get("args")
        if not tool_name or not isinstance(args, dict):
            continue
        raw_calls.append(RawToolCall(tool=tool_name, args=args))

    logger.debug(f'Task dispatcher LLM: raw_calls={log_list_preview([{"tool": raw.tool, "args": raw.args} for raw in raw_calls])}')
    return raw_calls


def process_affinity_delta(current_affinity: int, raw_delta: int) -> int:
    """Scale a raw affinity delta by direction-specific breakpoints.

    Args:
        current_affinity: Current affinity score (0-1000).
        raw_delta: Raw delta from the relationship recorder (-5..+5).

    Returns:
        Scaled delta with sign preserved.
    """
    if raw_delta == 0:
        return 0

    if abs(raw_delta) <= AFFINITY_RAW_DEAD_ZONE:
        return 0

    if raw_delta > 0:
        breakpoints = AFFINITY_INCREMENT_BREAKPOINTS
    else:
        breakpoints = AFFINITY_DECREMENT_BREAKPOINTS

    scaling_factor = 1.0
    for i in range(len(breakpoints) - 1):
        x1, y1 = breakpoints[i]
        x2, y2 = breakpoints[i + 1]

        if x1 <= current_affinity <= x2:
            if x2 == x1:
                scaling_factor = y1
            else:
                scaling_factor = y1 + (current_affinity - x1) * (y2 - y1) / (x2 - x1)
            break

    return_value = int(round(raw_delta * scaling_factor, 0))
    return return_value


def _default_future_promise_due_time(timestamp: str) -> str:
    """Return the immediate fallback due time for untimed future promises.

    Args:
        timestamp: Turn timestamp used as the reference clock.

    Returns:
        ISO-8601 UTC timestamp for now or the next minute boundary.
    """

    try:
        reference_time = parse_iso_datetime(timestamp)
    except ValueError as exc:
        logger.debug(f"Handled exception in _default_future_promise_due_time: {exc}")
        reference_time = datetime.now(timezone.utc)

    if reference_time.second == 0 and reference_time.microsecond == 0:
        return_value = reference_time.isoformat()
        return return_value

    next_minute = reference_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
    return_value = next_minute.isoformat()
    return return_value


def _normalize_future_promises(
    future_promises: list[dict],
    *,
    timestamp: str,
) -> list[dict]:
    """Fill deterministic due-time defaults for actionable future promises.

    Args:
        future_promises: Raw promise rows from the harvester.
        timestamp: Turn timestamp used to resolve immediate fallbacks.

    Returns:
        Promise rows with ``future_promise`` due times normalized.
    """

    fallback_due_time = _default_future_promise_due_time(timestamp)
    normalized: list[dict] = []

    for promise in future_promises:
        normalized_promise = dict(promise)
        commitment_type = text_or_empty(normalized_promise.get("commitment_type"))
        due_time = normalized_promise.get("due_time")
        due_time_is_missing = due_time is None or (isinstance(due_time, str) and not due_time.strip())

        if commitment_type == "future_promise" and due_time_is_missing:
            normalized_promise["due_time"] = fallback_due_time

        normalized.append(normalized_promise)

    return normalized


async def db_writer(state: ConsolidatorState) -> dict:
    timestamp = state.get("timestamp") or datetime.now(timezone.utc).isoformat()
    global_user_id = state.get("global_user_id", "")
    user_name = state.get("user_name", "")

    metadata = dict(state.get("metadata", {}) or {})
    write_log: dict[str, bool] = {}
    cache_invalidated: list[str] = []

    # ── Step 1: character_state (mood / vibe / reflection) ──────────
    mood = state.get("mood", "")
    global_vibe = state.get("global_vibe", "")
    reflection_summary = state.get("reflection_summary", "")
    try:
        await upsert_character_state(
            mood=mood,
            global_vibe=global_vibe,
            reflection_summary=reflection_summary,
            timestamp=timestamp,
        )
        write_log["character_state"] = True
    except PyMongoError as exc:
        logger.debug(f"Handled exception in db_writer: {exc}")
        logger.exception("db_writer: failed to upsert character_state")
        write_log["character_state"] = False

    # ── Step 2: last relationship insight ───────────────────────────
    last_relationship_insight = state.get("last_relationship_insight", "")
    if global_user_id and last_relationship_insight:
        try:
            await update_last_relationship_insight(global_user_id, last_relationship_insight)
            write_log["relationship_insight"] = True
        except PyMongoError as exc:
            logger.debug(f"Handled exception in db_writer: {exc}")
            logger.exception("db_writer: failed to update_last_relationship_insight")
            write_log["relationship_insight"] = False

    # ── Step 3: unified user-memory units ────────────────────────────
    future_promises = _normalize_future_promises(
        state.get("future_promises") or [],
        timestamp=timestamp,
    )
    try:
        memory_unit_results = await update_user_memory_units_from_state(state)
    except Exception as exc:
        logger.debug(f"Handled exception in db_writer: {exc}")
        logger.exception("db_writer: failed to update user_memory_units")
        memory_unit_results = []
        write_log["user_memory_units"] = False
    else:
        write_log["user_memory_units"] = bool(memory_unit_results)
        metadata["user_memory_unit_results"] = memory_unit_results

    scheduled_event_ids: list[str] = []
    dispatch_rejections: list[str] = []
    dispatcher = _get_task_dispatcher()
    if dispatcher is not None:
        dispatch_ctx = _build_dispatch_context(state, timestamp=timestamp)
        dispatch_state = {
            **state,
            "future_promises": future_promises,
        }
        raw_calls = await _generate_raw_tool_calls(dispatch_state, dispatch_ctx)
        dispatch_result = await dispatcher.dispatch(
            raw_calls,
            dispatch_ctx,
            instruction=_build_dispatch_instruction(state),
        )
        scheduled_event_ids = [event_id for _task, event_id in dispatch_result.scheduled]
        dispatch_rejections = [
            f"{raw.tool}: {reason}"
            for raw, reason in dispatch_result.rejected
        ]
        if dispatch_rejections:
            if any("no adapters registered" in rejection for rejection in dispatch_rejections):
                logger.warning(f'Task dispatch unavailable: raw_calls={len(raw_calls)} platform={dispatch_ctx.source_platform} channel={dispatch_ctx.source_channel_id} future_promises={len(future_promises)} rejections={dispatch_rejections}')
            else:
                logger.debug(f'Task dispatch rejected {len(dispatch_rejections)} call(s): {dispatch_rejections}')

    # ── Step 4: affinity (direction-scaled) ─────────────────────────
    user_profile = state.get("user_profile", {})
    user_affinity_score = user_profile.get("affinity", AFFINITY_DEFAULT)
    raw_affinity_delta = state.get("affinity_delta", 0) or 0
    processed_affinity_delta = process_affinity_delta(user_affinity_score, raw_affinity_delta)
    if global_user_id:
        try:
            await update_affinity(global_user_id, processed_affinity_delta)
            write_log["affinity"] = True
        except PyMongoError as exc:
            logger.debug(f"Handled exception in db_writer: {exc}")
            logger.exception("db_writer: failed to update_affinity")
            write_log["affinity"] = False

    logger.debug(f'User {user_name}(@{global_user_id}) affinity {user_affinity_score} -> {user_affinity_score + processed_affinity_delta}')

    # ── Step 5: character image ──────────────────────────────────────
    image_results = await asyncio.gather(
        _update_character_image(state, timestamp=timestamp),
        return_exceptions=True,
    )
    character_image_result = image_results[0]

    if isinstance(character_image_result, Exception):
        logger.error(
            "db_writer: failed to update character_image",
            exc_info=(
                type(character_image_result),
                character_image_result,
                character_image_result.__traceback__,
            ),
        )
        write_log["character_image"] = False
    elif character_image_result is not None:
        try:
            await upsert_character_self_image(character_image_result)
            write_log["character_image"] = True
        except PyMongoError as exc:
            logger.debug(f"Handled exception in db_writer: {exc}")
            logger.exception("db_writer: failed to upsert_character_self_image")
            write_log["character_image"] = False

    # ── Step 6: Cache2 invalidation events (after persistence) ──────
    runtime = get_rag_cache2_runtime()
    events: list[CacheInvalidationEvent] = []

    if global_user_id and (
        write_log.get("affinity")
        or write_log.get("relationship_insight")
        or write_log.get("user_memory_units")
    ):
        events.append(CacheInvalidationEvent(
            source="user_profile",
            platform=state["platform"],
            platform_channel_id=state["platform_channel_id"],
            global_user_id=global_user_id,
            timestamp=timestamp,
            reason="consolidator: user_profile",
        ))

    if write_log.get("character_state") or write_log.get("character_image"):
        events.append(CacheInvalidationEvent(
            source="character_state",
            reason="consolidator: character_state",
        ))

    evicted_total = 0
    for event in events:
        evicted_total += await runtime.invalidate(event)
    cache_invalidated = [event.source for event in events]
    metadata["cache_evicted_count"] = evicted_total

    metadata.update({
        "write_success": write_log,
        "cache_invalidated": cache_invalidated,
        "scheduled_event_ids": scheduled_event_ids,
        "task_dispatch_rejected": dispatch_rejections,
        "affinity_before": user_affinity_score,
        "affinity_delta_processed": processed_affinity_delta,
    })

    logger.debug(f'db_writer summary: user={user_name} global_user={global_user_id} writes={write_log} cache_invalidated={cache_invalidated} scheduled={len(scheduled_event_ids)} affinity_before={user_affinity_score} affinity_delta={processed_affinity_delta}')

    return_value = {"metadata": metadata}
    return return_value
