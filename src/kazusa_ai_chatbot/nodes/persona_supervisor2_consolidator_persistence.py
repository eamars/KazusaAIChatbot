"""Stage 4 consolidator persistence and scheduling helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import (
    AFFINITY_DECREMENT_BREAKPOINTS,
    AFFINITY_DEFAULT,
    AFFINITY_INCREMENT_BREAKPOINTS,
    AFFINITY_RAW_DEAD_ZONE,
)
from kazusa_ai_chatbot.db import (
    ActiveCommitmentDoc,
    CharacterDiaryEntry,
    MemoryType,
    ObjectiveFactEntry,
    ScheduledEventDoc,
    increment_rag_version,
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
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_knowledge import _update_knowledge_base
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import (
    ConsolidatorState,
    normalize_diary_entries,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _get_rag_cache
from kazusa_ai_chatbot.scheduler import schedule_event

logger = logging.getLogger(__name__)

AFFINITY_CACHE_NUKE_THRESHOLD = 50


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
    for promise in future_promises:
        action = str(promise.get("action", "")).strip()
        if not action:
            continue
        commitments.append(
            {
                "commitment_id": uuid4().hex,
                "target": str(promise.get("target", "")).strip(),
                "action": action,
                "commitment_type": str(promise.get("commitment_type", "")).strip(),
                "status": "active",
                "source": "conversation_extracted",
                "created_at": timestamp,
                "updated_at": timestamp,
                "due_time": promise.get("due_time"),
                "dedup_key": str(promise.get("dedup_key") or action).strip().lower(),
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
        description = fact.get("description", "")
        if not description:
            continue
        entry: ObjectiveFactEntry = {
            "fact": description,
            "category": fact.get("category", "general"),
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
        content = str(entry.get("entry", "")).strip()
        if not content:
            continue
        memories.append({
            "memory_type": MemoryType.DIARY_ENTRY,
            "content": content,
            "created_at": entry.get("timestamp") or timestamp,
            "updated_at": timestamp,
            "confidence": entry.get("confidence", 0.8),
            "context": entry.get("context", ""),
        })

    fact_by_description = {
        str(fact.get("description", "")).strip(): fact
        for fact in new_facts or []
        if str(fact.get("description", "")).strip()
    }
    for fact in objective_facts:
        content = str(fact.get("fact", "")).strip()
        if not content:
            continue
        raw_fact = fact_by_description.get(content, {})
        dedup_key = str(raw_fact.get("dedup_key") or content).strip().lower()
        scope = str(raw_fact.get("scope", "")).strip()
        # A milestone fact is stored ONLY as a MILESTONE memory. The read
        # layer folds milestones into the objective_facts block so prompt-
        # facing fact lists still include them — no need to duplicate-write.
        if raw_fact.get("is_milestone"):
            memories.append({
                "memory_type": MemoryType.MILESTONE,
                "content": content,
                "created_at": fact.get("timestamp") or timestamp,
                "updated_at": timestamp,
                "category": fact.get("category", "general"),
                "source": fact.get("source", "conversation_extracted"),
                "confidence": fact.get("confidence", 0.85),
                "event_category": str(raw_fact.get("milestone_category", "")).strip(),
                "scope": scope,
                "dedup_key": dedup_key,
                "superseded_by": None,
            })
        else:
            memories.append({
                "memory_type": MemoryType.OBJECTIVE_FACT,
                "content": content,
                "created_at": fact.get("timestamp") or timestamp,
                "updated_at": timestamp,
                "category": fact.get("category", "general"),
                "source": fact.get("source", "conversation_extracted"),
                "confidence": fact.get("confidence", 0.85),
                "dedup_key": dedup_key,
                "scope": scope,
            })

    for commitment in active_commitments:
        action = str(commitment.get("action", "")).strip()
        if not action:
            continue
        memories.append({
            "memory_type": MemoryType.COMMITMENT,
            "content": action,
            "action": action,
            "commitment_id": commitment.get("commitment_id", ""),
            "target": commitment.get("target", ""),
            "commitment_type": commitment.get("commitment_type", ""),
            "status": commitment.get("status", "active"),
            "source": commitment.get("source", "conversation_extracted"),
            "created_at": commitment.get("created_at") or timestamp,
            "updated_at": commitment.get("updated_at") or timestamp,
            "due_time": commitment.get("due_time"),
            "dedup_key": str(commitment.get("dedup_key") or action).strip().lower(),
        })

    return memories


async def _schedule_future_promises(
    promises: list[dict],
    *,
    active_commitments: list[ActiveCommitmentDoc],
    global_user_id: str,
    user_name: str,
    character_name: str,
    decontexualized_input: str,
) -> list[str]:
    """Persist each promise as a ``future_promise`` scheduled event.

    Returns the list of event_ids scheduled. Events with no ``due_time`` are
    skipped — there's nothing to fire on.
    """
    scheduled: list[str] = []
    commitment_by_action = {
        str(item.get("action", "")): item
        for item in active_commitments
        if item.get("action")
    }
    for promise in promises or []:
        due_time = promise.get("due_time")
        if not due_time:
            continue
        try:
            datetime.fromisoformat(due_time)
        except ValueError:
            logger.warning("Skipping promise with unparseable due_time: %r", due_time)
            continue

        event: ScheduledEventDoc = {
            "event_type": "future_promise",
            "target_platform": "",
            "target_channel_id": "",
            "target_global_user_id": global_user_id,
            "payload": {
                "promise_text": promise.get("action", ""),
                "commitment_id": (commitment_by_action.get(promise.get("action", "")) or {}).get("commitment_id", ""),
                "target": promise.get("target", user_name),
                "character_name": character_name,
                "original_input": decontexualized_input,
                "context_summary": f"promise by {character_name} to {promise.get('target', user_name)}",
            },
            "scheduled_at": due_time,
        }
        try:
            event_id = await schedule_event(event)
            scheduled.append(event_id)
        except PyMongoError:
            logger.exception("Failed to persist future_promise event for user %s", global_user_id)
    return scheduled


async def db_writer(state: ConsolidatorState) -> dict:
    timestamp = state.get("timestamp") or datetime.now(timezone.utc).isoformat()
    global_user_id = state.get("global_user_id", "")
    user_name = state.get("user_name", "")
    character_name = state.get("character_profile", {}).get("name", "")

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
    future_promises = state.get("future_promises") or []
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

    scheduled_event_ids = await _schedule_future_promises(
        future_promises,
        active_commitments=active_commitments,
        global_user_id=global_user_id,
        user_name=user_name,
        character_name=character_name,
        decontexualized_input=state.get("decontexualized_input", ""),
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

    # ── Step 5: cache invalidation (best-effort, after writes) ──────
    # The RAG cache is the hot read-path; stale entries now outweigh recency.
    if global_user_id:
        try:
            rag_cache = await _get_rag_cache()
            if profile_memories:
                for cache_type in (
                    "user_profile_memories",
                    "character_diary",
                    "objective_user_facts",
                    "user_promises",
                    "boundary_cache",
                ):
                    await rag_cache.invalidate_pattern(
                        cache_type=cache_type,
                        global_user_id=global_user_id,
                    )
                    cache_invalidated.append(cache_type)
            if abs(processed_affinity_delta) > AFFINITY_CACHE_NUKE_THRESHOLD:
                await rag_cache.clear_all_user(global_user_id)
                cache_invalidated.append("ALL_USER")

            if cache_invalidated:
                await increment_rag_version(global_user_id)
        except PyMongoError:
            logger.exception("db_writer: cache invalidation failed")

    # ── Step 6: user image (three-tier rolling) ─────────────────────
    if global_user_id:
        try:
            user_image_doc = await _update_user_image(
                state,
                timestamp=timestamp,
                processed_affinity_delta=processed_affinity_delta,
            )
            if user_image_doc is not None:
                await upsert_user_image(global_user_id, user_image_doc)
                write_log["user_image"] = True
        except Exception:
            logger.exception("db_writer: failed to update user_image")
            write_log["user_image"] = False

    # ── Step 7: character self-image (three-tier rolling) ────────────
    try:
        character_image_doc = await _update_character_image(state, timestamp=timestamp)
        if character_image_doc is not None:
            await upsert_character_self_image(character_image_doc)
            write_log["character_image"] = True
    except Exception:
        logger.exception("db_writer: failed to update character_image")
        write_log["character_image"] = False

    # ── Step 8: knowledge-base distillation (DEEP passes only) ───────
    kb_count = 0
    try:
        kb_count = await _update_knowledge_base(state)
        write_log["knowledge_base"] = kb_count > 0
    except Exception:
        logger.exception("db_writer: failed to update knowledge_base")
        write_log["knowledge_base"] = False

    metadata.update({
        "write_success": write_log,
        "cache_invalidation_scope": cache_invalidated,
        "scheduled_event_ids": scheduled_event_ids,
        "affinity_before": user_affinity_score,
        "affinity_delta_processed": processed_affinity_delta,
        "knowledge_base_entries_written": kb_count,
    })

    logger.debug(
        "db_writer summary: user=%s global_user=%s writes=%s cache_invalidated=%s scheduled=%d affinity_before=%s affinity_delta=%s kb_written=%d",
        user_name,
        global_user_id,
        write_log,
        cache_invalidated,
        len(scheduled_event_ids),
        user_affinity_score,
        processed_affinity_delta,
        kb_count,
    )

    return {"metadata": metadata}
