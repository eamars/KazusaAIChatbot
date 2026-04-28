"""Stage 4 consolidator persistence and scheduling helpers."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

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
    ActiveCommitmentDoc,
    CharacterDiaryEntry,
    MemoryType,
    ObjectiveFactEntry,
    insert_profile_memories,
    update_affinity,
    update_last_relationship_insight,
    upsert_character_self_image,
    upsert_character_state,
    upsert_user_image,
    UserProfileMemoryDoc,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_images import (
    _update_character_image,
    _update_user_image,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import (
    ConsolidatorState,
    normalize_diary_entries,
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
    except ValueError:
        now = datetime.now(timezone.utc)

    return DispatchContext(
        source_platform=state.get("platform", ""),
        source_channel_id=state.get("platform_channel_id", ""),
        source_user_id=state.get("global_user_id", ""),
        source_message_id=state.get("platform_message_id", ""),
        guild_id=None,
        bot_role="user",
        now=now,
    )


def _build_dispatch_instruction(state: ConsolidatorState) -> str:
    """Summarize the current turn intent for tracing and task generation.

    Args:
        state: Consolidator state for the current turn.

    Returns:
        Natural-language instruction summary.
    """

    character_name = state.get("character_profile", {}).get("name", "Kazusa")
    user_name = state.get("user_name", "user")
    promise_count = len(state.get("future_promises") or [])
    return (
        f"{character_name} should follow through on {promise_count} accepted promise(s)"
        f" for {user_name} based on the finalized dialog."
    )


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
        return []

    available_tools = _task_registry.filter(ctx)
    future_promises = _normalize_future_promises(
        state.get("future_promises") or [],
        timestamp=ctx.now.isoformat(),
    )
    if not available_tools or not future_promises:
        return []

    msg = {
        "instruction": _build_dispatch_instruction(state),
        "current_utc": ctx.now.isoformat(),
        "source_platform": ctx.source_platform,
        "source_channel_id": ctx.source_channel_id,
        "source_channel_type": state.get("channel_type", "group"),
        "source_message_id": ctx.source_message_id,
        "decontexualized_input": state.get("decontexualized_input", ""),
        "final_dialog": state.get("final_dialog", []),
        "content_anchors": state.get("action_directives", {}).get("linguistic_directives", {}).get("content_anchors", []),
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
        return []

    raw_calls: list[RawToolCall] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool", "")).strip()
        args = item.get("args")
        if not tool_name or not isinstance(args, dict):
            continue
        raw_calls.append(RawToolCall(tool=tool_name, args=args))

    logger.debug(
        "Task dispatcher LLM: raw_calls=%s",
        log_list_preview([{"tool": raw.tool, "args": raw.args} for raw in raw_calls]),
    )
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

    return int(round(raw_delta * scaling_factor, 0))


def _build_diary_entries(
    diary_strings: list[str],
    *,
    timestamp: str,
    interaction_subtext: str,
) -> list[CharacterDiaryEntry]:
    """Convert raw diary strings into ``CharacterDiaryEntry`` dicts."""
    entries: list[CharacterDiaryEntry] = []
    for text in diary_strings or []:
        if not text:
            continue
        entry: CharacterDiaryEntry = {
            "entry": text,
            "timestamp": timestamp,
            "confidence": 0.8,
            "context": interaction_subtext or "",
        }
        entries.append(entry)
    return entries


def _default_future_promise_due_time(timestamp: str) -> str:
    """Return the immediate fallback due time for untimed future promises.

    Args:
        timestamp: Turn timestamp used as the reference clock.

    Returns:
        ISO-8601 UTC timestamp for now or the next minute boundary.
    """

    try:
        reference_time = parse_iso_datetime(timestamp)
    except ValueError:
        reference_time = datetime.now(timezone.utc)

    if reference_time.second == 0 and reference_time.microsecond == 0:
        return reference_time.isoformat()

    next_minute = reference_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
    return next_minute.isoformat()


def _text_or_default(value: object, default: str) -> str:
    text = text_or_empty(value)
    if text:
        return text
    return default


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


def _build_active_commitment_entries(
    future_promises: list[dict],
    *,
    timestamp: str,
) -> list[ActiveCommitmentDoc]:
    """Convert accepted future promises into authoritative active commitments.

    Args:
        future_promises: Sanitized harvester promise rows.
        timestamp: Current turn timestamp.

    Returns:
        A list of structured commitment rows for ``user_profile.active_commitments``.
    """
    commitments: list[ActiveCommitmentDoc] = []
    normalized_promises = _normalize_future_promises(future_promises, timestamp=timestamp)
    for promise in normalized_promises:
        action = text_or_empty(promise.get("action"))
        if not action:
            continue
        dedup_key = text_or_empty(promise.get("dedup_key")) or action
        commitments.append(
            {
                "commitment_id": uuid4().hex,
                "target": text_or_empty(promise.get("target")),
                "action": action,
                "commitment_type": text_or_empty(promise.get("commitment_type")),
                "status": "active",
                "source": "conversation_extracted",
                "created_at": timestamp,
                "updated_at": timestamp,
                "due_time": promise.get("due_time"),
                "dedup_key": dedup_key.lower(),
            }
        )
    return commitments


def _build_objective_fact_entries(
    new_facts: list[dict],
    *,
    timestamp: str,
) -> list[ObjectiveFactEntry]:
    """Convert harvester ``new_facts`` rows into ``ObjectiveFactEntry`` dicts."""
    entries: list[ObjectiveFactEntry] = []
    for fact in new_facts or []:
        description = text_or_empty(fact.get("description"))
        if not description:
            continue
        entry: ObjectiveFactEntry = {
            "fact": description,
            "category": _text_or_default(fact.get("category"), "general"),
            "timestamp": timestamp,
            "source": "conversation_extracted",
            "confidence": 0.85,
        }
        entries.append(entry)
    return entries


def _build_memory_docs(
    *,
    diary_entries: list[CharacterDiaryEntry],
    objective_facts: list[ObjectiveFactEntry],
    active_commitments: list[ActiveCommitmentDoc],
    new_facts: list[dict],
    timestamp: str,
) -> list[UserProfileMemoryDoc]:
    """Build unified profile-memory docs from consolidator outputs.

    Args:
        diary_entries: Subjective diary entries from the relationship recorder.
        objective_facts: Objective facts built from the facts harvester output.
        active_commitments: Accepted future promises in prompt-facing shape.
        new_facts: Raw fact rows, used to preserve LLM-emitted milestone fields.
        timestamp: Current turn timestamp.

    Returns:
        Memory docs ready for ``insert_profile_memories``.
    """
    memories: list[UserProfileMemoryDoc] = []

    for entry in diary_entries:
        content = text_or_empty(entry.get("entry"))
        if not content:
            continue
        created_at = entry.get("timestamp")
        context = entry.get("context")
        memories.append({
            "memory_type": MemoryType.DIARY_ENTRY,
            "content": content,
            "created_at": created_at if isinstance(created_at, str) and created_at.strip() else timestamp,
            "updated_at": timestamp,
            "confidence": entry.get("confidence", 0.8),
            "context": context if isinstance(context, str) else "",
        })

    fact_by_description = {
        text_or_empty(fact.get("description")): fact
        for fact in new_facts or []
        if text_or_empty(fact.get("description"))
    }
    for fact in objective_facts:
        content = text_or_empty(fact.get("fact"))
        if not content:
            continue
        raw_fact = fact_by_description.get(content, {})
        dedup_key = (text_or_empty(raw_fact.get("dedup_key")) or content).lower()
        scope = text_or_empty(raw_fact.get("scope"))
        created_at = fact.get("timestamp")
        category = _text_or_default(fact.get("category"), "general")
        source = _text_or_default(fact.get("source"), "conversation_extracted")
        # A milestone fact is stored ONLY as a MILESTONE memory. The read
        # layer folds milestones into the objective_facts block so prompt-
        # facing fact lists still include them — no need to duplicate-write.
        if raw_fact.get("is_milestone"):
            memories.append({
                "memory_type": MemoryType.MILESTONE,
                "content": content,
                "created_at": created_at if isinstance(created_at, str) and created_at.strip() else timestamp,
                "updated_at": timestamp,
                "category": category,
                "source": source,
                "confidence": fact.get("confidence", 0.85),
                "event_category": text_or_empty(raw_fact.get("milestone_category")),
                "scope": scope,
                "dedup_key": dedup_key,
                "superseded_by": None,
            })
        else:
            memories.append({
                "memory_type": MemoryType.OBJECTIVE_FACT,
                "content": content,
                "created_at": created_at if isinstance(created_at, str) and created_at.strip() else timestamp,
                "updated_at": timestamp,
                "category": category,
                "source": source,
                "confidence": fact.get("confidence", 0.85),
                "dedup_key": dedup_key,
                "scope": scope,
            })

    for commitment in active_commitments:
        action = text_or_empty(commitment.get("action"))
        if not action:
            continue
        dedup_key = text_or_empty(commitment.get("dedup_key")) or action
        created_at = commitment.get("created_at")
        updated_at = commitment.get("updated_at")
        memories.append({
            "memory_type": MemoryType.COMMITMENT,
            "content": action,
            "action": action,
            "commitment_id": text_or_empty(commitment.get("commitment_id")),
            "target": text_or_empty(commitment.get("target")),
            "commitment_type": text_or_empty(commitment.get("commitment_type")),
            "status": _text_or_default(commitment.get("status"), "active"),
            "source": _text_or_default(commitment.get("source"), "conversation_extracted"),
            "created_at": created_at if isinstance(created_at, str) and created_at.strip() else timestamp,
            "updated_at": updated_at if isinstance(updated_at, str) and updated_at.strip() else timestamp,
            "due_time": commitment.get("due_time"),
            "dedup_key": dedup_key.lower(),
        })

    return memories


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
    except PyMongoError:
        logger.exception("db_writer: failed to upsert character_state")
        write_log["character_state"] = False

    # ── Step 2a: build character diary (subjective per-user notes) ──
    diary_entries = _build_diary_entries(
        normalize_diary_entries(state.get("diary_entry")),
        timestamp=timestamp,
        interaction_subtext=state.get("interaction_subtext", ""),
    )

    # ── Step 2b: last relationship insight ──────────────────────────
    last_relationship_insight = state.get("last_relationship_insight", "")
    if global_user_id and last_relationship_insight:
        try:
            await update_last_relationship_insight(global_user_id, last_relationship_insight)
            write_log["relationship_insight"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to update_last_relationship_insight")
            write_log["relationship_insight"] = False

    # ── Step 3a: objective facts and commitments as profile memories ─
    new_facts = state.get("new_facts") or []
    objective_facts = _build_objective_fact_entries(new_facts, timestamp=timestamp)
    future_promises = _normalize_future_promises(
        state.get("future_promises") or [],
        timestamp=timestamp,
    )
    active_commitments = _build_active_commitment_entries(future_promises, timestamp=timestamp)
    profile_memories = _build_memory_docs(
        diary_entries=diary_entries,
        objective_facts=objective_facts,
        active_commitments=active_commitments,
        new_facts=new_facts,
        timestamp=timestamp,
    )
    if global_user_id and profile_memories:
        try:
            await insert_profile_memories(global_user_id, profile_memories)
            write_log["user_profile_memories"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to insert_profile_memories")
            write_log["user_profile_memories"] = False

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
                logger.warning(
                    "Task dispatch unavailable: raw_calls=%d platform=%s channel=%s future_promises=%d rejections=%s",
                    len(raw_calls),
                    dispatch_ctx.source_platform,
                    dispatch_ctx.source_channel_id,
                    len(future_promises),
                    dispatch_rejections,
                )
            else:
                logger.debug(
                    "Task dispatch rejected %d call(s): %s",
                    len(dispatch_rejections),
                    dispatch_rejections,
                )

    # ── Step 4: affinity (direction-scaled) ─────────────────────────
    user_affinity_score = state.get("user_profile", {}).get("affinity", AFFINITY_DEFAULT)
    raw_affinity_delta = state.get("affinity_delta", 0) or 0
    processed_affinity_delta = process_affinity_delta(user_affinity_score, raw_affinity_delta)
    if global_user_id:
        try:
            await update_affinity(global_user_id, processed_affinity_delta)
            write_log["affinity"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to update_affinity")
            write_log["affinity"] = False

    logger.debug(
        "User %s(@%s) affinity %s -> %s",
        user_name, global_user_id,
        user_affinity_score, user_affinity_score + processed_affinity_delta,
    )

    # ── Step 5: user / character images (three-tier rolling) ─────────
    user_image_task = None
    if global_user_id:
        user_image_task = _update_user_image(
            state,
            timestamp=timestamp,
            processed_affinity_delta=processed_affinity_delta,
        )
    image_results = await asyncio.gather(
        user_image_task if user_image_task is not None else asyncio.sleep(0, result=None),
        _update_character_image(state, timestamp=timestamp),
        return_exceptions=True,
    )
    user_image_result, character_image_result = image_results

    if global_user_id:
        if isinstance(user_image_result, Exception):
            logger.error(
                "db_writer: failed to update user_image",
                exc_info=(type(user_image_result), user_image_result, user_image_result.__traceback__),
            )
            write_log["user_image"] = False
        elif user_image_result is not None:
            try:
                await upsert_user_image(global_user_id, user_image_result)
                write_log["user_image"] = True
            except PyMongoError:
                logger.exception("db_writer: failed to upsert_user_image")
                write_log["user_image"] = False

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
        except PyMongoError:
            logger.exception("db_writer: failed to upsert_character_self_image")
            write_log["character_image"] = False

    # ── Step 6: Cache2 invalidation events (after persistence) ──────
    runtime = get_rag_cache2_runtime()
    events: list[CacheInvalidationEvent] = []

    if global_user_id and (
        write_log.get("user_profile_memories")
        or write_log.get("affinity")
        or write_log.get("relationship_insight")
        or write_log.get("user_image")
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

    logger.debug(
        "db_writer summary: user=%s global_user=%s writes=%s cache_invalidated=%s scheduled=%d affinity_before=%s affinity_delta=%s",
        user_name,
        global_user_id,
        write_log,
        cache_invalidated,
        len(scheduled_event_ids),
        user_affinity_score,
        processed_affinity_delta,
    )

    return {"metadata": metadata}
