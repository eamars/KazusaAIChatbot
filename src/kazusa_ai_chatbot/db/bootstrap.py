"""One-shot startup routine that ensures collections, indices, and seeded
documents exist.

All operations are idempotent — safe to run on every service start.
"""

from __future__ import annotations

import logging

from kazusa_ai_chatbot.config import (
    MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.db._client import enable_vector_index, get_db
from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_JOBS_COLLECTION,
)
from kazusa_ai_chatbot.accepted_task.models import ACCEPTED_TASKS_COLLECTION
from kazusa_ai_chatbot.db.accepted_tasks import ensure_accepted_task_indexes
from kazusa_ai_chatbot.db.background_work_jobs import (
    ensure_background_work_job_indexes,
)
from kazusa_ai_chatbot.db.conversation import (
    CONVERSATION_VECTOR_FILTER_FIELDS,
    CONVERSATION_VECTOR_INDEX_NAME,
)
from kazusa_ai_chatbot.db.interaction_style_images import (
    ensure_interaction_style_image_indexes,
)
from kazusa_ai_chatbot.db.internal_monologue_residue import (
    INTERNAL_MONOLOGUE_RESIDUE_COLLECTION,
    ensure_internal_monologue_residue_indexes,
)
from kazusa_ai_chatbot.db.global_character_growth import (
    GLOBAL_CHARACTER_GROWTH_RUNS_COLLECTION,
    GLOBAL_CHARACTER_GROWTH_TRAITS_COLLECTION,
    ensure_global_character_growth_indexes,
)
from kazusa_ai_chatbot.db.event_logging import (
    EVENT_LOG_EVENTS_COLLECTION,
    EVENT_LOG_SNAPSHOTS_COLLECTION,
    ensure_event_log_indexes,
)
from kazusa_ai_chatbot.db.llm_tracing import (
    LLM_TRACE_RUNS_COLLECTION,
    LLM_TRACE_STEPS_COLLECTION,
    ensure_llm_trace_indexes,
)
from kazusa_ai_chatbot.db.reflection_cycle import ensure_reflection_run_indexes
from kazusa_ai_chatbot.db.rag_cache2_persistent import (
    PERSISTENT_CACHE_COLLECTION,
    PERSISTENT_CACHE_LOOKUP_INDEX,
    PERSISTENT_CACHE_LOOKUP_KEYS,
    prune_media_descriptor_entries,
    purge_stale_media_descriptor_entries,
)
from kazusa_ai_chatbot.db.self_cognition import (
    SELF_COGNITION_ACTION_ATTEMPTS_COLLECTION,
    SELF_COGNITION_GROUP_REVIEW_WINDOWS_COLLECTION,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso

logger = logging.getLogger(__name__)

CALENDAR_SCHEDULES_COLLECTION = "calendar_schedules"
CALENDAR_RUNS_COLLECTION = "calendar_runs"


def _now_iso() -> str:
    timestamp = storage_utc_now_iso()
    if timestamp.endswith("+00:00"):
        return f"{timestamp[:-6]}Z"
    return timestamp


async def db_bootstrap() -> None:
    """Create all required collections, indices, and seeded documents.

    Called once at service startup. Safe to call repeatedly.
    """
    db = await get_db()
    existing = set(await db.list_collection_names())

    for legacy in (
        "rag_cache_index",
        "rag_metadata_index",
        "background_artifact_jobs",
    ):
        if legacy in existing:
            await db.drop_collection(legacy)
            logger.info(f'Dropped legacy collection \'{legacy}\'')
            existing.discard(legacy)

    required_collections = [
        "conversation_history",
        "user_profiles",
        "character_state",
        "memory",
        "user_memory_units",
        "scheduled_events",
        CALENDAR_SCHEDULES_COLLECTION,
        CALENDAR_RUNS_COLLECTION,
        "conversation_episode_state",
        "character_reflection_runs",
        "interaction_style_images",
        GLOBAL_CHARACTER_GROWTH_TRAITS_COLLECTION,
        GLOBAL_CHARACTER_GROWTH_RUNS_COLLECTION,
        PERSISTENT_CACHE_COLLECTION,
        EVENT_LOG_EVENTS_COLLECTION,
        EVENT_LOG_SNAPSHOTS_COLLECTION,
        LLM_TRACE_RUNS_COLLECTION,
        LLM_TRACE_STEPS_COLLECTION,
        SELF_COGNITION_ACTION_ATTEMPTS_COLLECTION,
        SELF_COGNITION_GROUP_REVIEW_WINDOWS_COLLECTION,
        ACCEPTED_TASKS_COLLECTION,
        BACKGROUND_WORK_JOBS_COLLECTION,
        INTERNAL_MONOLOGUE_RESIDUE_COLLECTION,
    ]
    for name in required_collections:
        if name not in existing:
            await db.create_collection(name)
            logger.info(f'Created collection \'{name}\'')
        else:
            logger.debug(f'Collection \'{name}\' already exists')

    # ── Seed singleton character_state ─────────────────────────────
    existing_state = await db.character_state.find_one({"_id": "global"})
    if existing_state is None:
        await db.character_state.insert_one({
            "_id": "global",
            "updated_at": _now_iso(),
            "cognition_state": build_character_production_state(
                updated_at=_now_iso(),
            ),
        })
        logger.info("Seeded default character_state document")
    elif existing_state.get("cognition_state") is None:
        await db.character_state.update_one(
            {"_id": "global"},
            {
                "$set": {
                    "cognition_state": build_character_production_state(
                        updated_at=_now_iso(),
                    )
                }
            },
        )
    else:
        validate_cognition_state(existing_state["cognition_state"])

    # ── Standard regular indexes (idempotent) ──────────────────────
    await db.conversation_history.create_index(
        [("platform", 1), ("platform_channel_id", 1), ("timestamp", -1)],
        name="conv_platform_channel_ts",
    )
    await db.conversation_history.create_index(
        [
            ("platform", 1),
            ("platform_channel_id", 1),
            ("addressed_to_global_user_ids", 1),
            ("timestamp", -1),
        ],
        name="conv_platform_channel_addressee_ts",
    )
    await db.conversation_history.create_index(
        "body_text",
        name="conv_body_text",
    )
    await db.conversation_history.create_index(
        [
            ("role", 1),
            ("timestamp", -1),
            ("platform", 1),
            ("platform_channel_id", 1),
        ],
        name="conv_role_ts_platform_channel",
    )
    await db.conversation_history.create_index(
        [
            ("platform", 1),
            ("platform_channel_id", 1),
            ("platform_message_id", 1),
        ],
        name="conv_platform_channel_message_id",
    )
    await db.conversation_history.create_index(
        [
            ("platform", 1),
            ("platform_channel_id", 1),
            ("delivery_tracking_id", 1),
            ("logical_message_index", 1),
        ],
        name="conv_delivery_tracking_logical_index",
    )
    await db.user_profiles.create_index(
        "global_user_id", unique=True, name="user_global_id_unique",
    )
    await db.scheduled_events.create_index(
        "event_id", unique=True, name="event_id_unique",
    )
    await db.scheduled_events.create_index(
        [("status", 1), ("execute_at", 1)], name="event_status_execute_at",
    )
    await db.scheduled_events.create_index(
        "source_user_id", name="event_source_user",
    )
    await db[CALENDAR_SCHEDULES_COLLECTION].create_index(
        "idempotency_key",
        unique=True,
        name="calendar_schedule_idempotency_unique",
    )
    await db[CALENDAR_SCHEDULES_COLLECTION].create_index(
        [("status", 1), ("next_run_at", 1), ("trigger_kind", 1)],
        name="calendar_schedule_status_next_trigger",
    )
    await db[CALENDAR_RUNS_COLLECTION].create_index(
        "idempotency_key",
        unique=True,
        name="calendar_run_idempotency_unique",
    )
    await db[CALENDAR_RUNS_COLLECTION].create_index(
        "run_id",
        unique=True,
        name="calendar_run_id_unique",
    )
    await db[CALENDAR_RUNS_COLLECTION].create_index(
        [("status", 1), ("due_at", 1), ("trigger_kind", 1)],
        name="calendar_run_status_due_trigger",
    )
    await db[CALENDAR_RUNS_COLLECTION].create_index(
        [("lease_expires_at", 1), ("status", 1)],
        name="calendar_run_lease_expiry_status",
    )
    await db[CALENDAR_RUNS_COLLECTION].create_index(
        [("trigger_kind", 1), ("period_start_utc", 1), ("run_id", 1)],
        name="calendar_run_reflection_phase_period",
    )
    await db[SELF_COGNITION_ACTION_ATTEMPTS_COLLECTION].create_index(
        "idempotency_key",
        unique=True,
        name="self_cognition_attempt_idempotency_unique",
    )
    await db[SELF_COGNITION_ACTION_ATTEMPTS_COLLECTION].create_index(
        [("status", 1), ("recorded_at", -1)],
        name="self_cognition_attempt_status_recorded",
    )
    await db[SELF_COGNITION_GROUP_REVIEW_WINDOWS_COLLECTION].create_index(
        "source_id",
        unique=True,
        name="self_cognition_group_review_window_source_unique",
    )
    await db[SELF_COGNITION_GROUP_REVIEW_WINDOWS_COLLECTION].create_index(
        [("scope_ref", 1), ("status", 1), ("window_start", 1)],
        name="self_cognition_group_review_window_scope_status",
    )
    await db[SELF_COGNITION_GROUP_REVIEW_WINDOWS_COLLECTION].create_index(
        "reviewed_at",
        name="self_cognition_group_review_window_reviewed_at",
    )
    await ensure_accepted_task_indexes()
    await ensure_background_work_job_indexes()
    await db.conversation_episode_state.create_index(
        [("platform", 1), ("platform_channel_id", 1), ("global_user_id", 1)],
        unique=True,
        name="conversation_episode_scope_unique",
    )
    await db.conversation_episode_state.create_index(
        "expires_at",
        expireAfterSeconds=0,
        name="conversation_episode_expires_at_ttl",
    )
    await db.memory.create_index(
        "memory_name", name="memory_name_idx",
    )
    await db.memory.create_index(
        "source_global_user_id", name="memory_source_user_idx",
    )
    await db.memory.create_index(
        "memory_unit_id",
        unique=True,
        name="memory_unit_id_unique",
        partialFilterExpression={"memory_unit_id": {"$exists": True}},
    )
    await db.memory.create_index(
        [("lineage_id", 1), ("version", -1)],
        name="memory_lineage_version",
    )
    await db.memory.create_index(
        [("status", 1), ("memory_type", 1), ("source_kind", 1), ("updated_at", -1)],
        name="memory_active_lookup",
    )
    await db.memory.create_index(
        [("memory_name", 1), ("source_global_user_id", 1), ("source_kind", 1)],
        name="memory_seed_sync_lookup",
    )

    await db.user_memory_units.create_index(
        "unit_id",
        unique=True,
        name="user_memory_unit_id_unique",
    )
    await db.user_memory_units.create_index(
        [("global_user_id", 1), ("unit_type", 1), ("status", 1), ("last_seen_at", -1)],
        name="user_memory_unit_owner_type_status_recent",
    )
    await db.user_memory_units.create_index(
        [("global_user_id", 1), ("status", 1), ("updated_at", -1)],
        name="user_memory_unit_owner_status_updated",
    )
    await db[PERSISTENT_CACHE_COLLECTION].create_index(
        PERSISTENT_CACHE_LOOKUP_KEYS,
        name=PERSISTENT_CACHE_LOOKUP_INDEX,
    )
    await ensure_reflection_run_indexes()
    await ensure_interaction_style_image_indexes()
    await ensure_global_character_growth_indexes()
    await ensure_event_log_indexes()
    await ensure_llm_trace_indexes()
    await ensure_internal_monologue_residue_indexes()

    await purge_stale_media_descriptor_entries()
    await prune_media_descriptor_entries(
        max_entries=MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES,
    )

    # ── Vector search indexes (best-effort — requires Atlas) ──────
    for collection, index_name, path in (
        ("conversation_history", CONVERSATION_VECTOR_INDEX_NAME, "embedding"),
        ("memory",               "memory_vector_index",              "embedding"),
        ("user_memory_units",     "user_memory_units_vector",         "embedding"),
    ):
        try:
            filter_paths = None
            if collection == "conversation_history":
                filter_paths = list(CONVERSATION_VECTOR_FILTER_FIELDS)
            if collection == "memory":
                filter_paths = [
                    "status",
                    "memory_type",
                    "source_kind",
                    "source_global_user_id",
                    "authority",
                    "lineage_id",
                ]
            if collection == "user_memory_units":
                filter_paths = ["global_user_id", "unit_type", "status"]
            await enable_vector_index(collection, index_name, path=path, filter_paths=filter_paths)
        except Exception as exc:
            logger.exception(
                f"Could not create vector index {index_name!r} "
                f"on {collection}.{path} (requires Atlas): {exc}"
            )

    logger.info("Database bootstrap complete")
