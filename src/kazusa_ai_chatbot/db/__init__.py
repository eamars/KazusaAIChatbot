"""``kazusa_ai_chatbot.db`` — MongoDB layer.

Submodule map:

* ``_client``      — MongoDB connection, embedding client, vector index helper
* ``schemas``      — TypedDict document shapes
* ``bootstrap``    — startup: collections and indices
* ``conversation`` — ``conversation_history`` operations
* ``users``        — ``user_profiles`` operations (identity, profile, V2 state)
* ``character``    — ``character_state`` operations
* ``memory``       — ``memory`` operations
* ``interaction_style_images`` — L3-only interaction style overlays
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ── Client + embedding ─────────────────────────────────────────────
from kazusa_ai_chatbot.db._client import (
    DatabaseTestGuardError,
    close_db,
    enable_vector_index,
    get_document_text_embedding,
    get_document_text_embeddings_batch,
    get_query_text_embedding,
    get_query_text_embeddings_batch,
    get_text_embedding,
    get_text_embeddings_batch,
)
from kazusa_ai_chatbot.db.background_work_jobs import (
    claim_background_work_job,
    complete_background_work_job,
    ensure_background_work_job_indexes,
    fail_background_work_job,
    find_deliverable_background_work_jobs,
    insert_background_work_job,
    mark_background_work_delivered,
    mark_background_work_delivery_failed,
    mark_background_work_delivery_in_progress,
)
from kazusa_ai_chatbot.db.character_identity_growth import (
    CANDIDATES_COLLECTION,
    GROWTH_COLLECTION_NAMES,
    IDENTITY_INDEX_NAMES,
    REVISIONS_COLLECTION,
    RUNS_COLLECTION,
    CharacterIdentityPersistenceError,
    ConcurrentIdentityPromotionError,
    IdentityCandidateConflictError,
    IdentityLedgerCorruptionError,
    IdentityLedgerNotFoundError,
    IdentityRootAlreadyClaimedError,
    IdentityRunConflictError,
    IdentityTransactionUnavailableError,
    SeedIdentityConflictError,
    count_inferred_identity_promotions_on_local_date,
    create_operator_reset_revision,
    ensure_character_identity_growth_indexes,
    ensure_seed_identity,
    get_current_identity,
    get_growth_run,
    insert_growth_candidate,
    insert_growth_run,
    list_current_growth_candidates,
    list_identity_revisions,
    promote_ready_candidate,
    reject_growth_candidates,
    update_growth_candidate,
)
from kazusa_ai_chatbot.db.cognition_chain_runs import (
    ensure_cognition_chain_run_indexes,
    get_cognition_chain_run,
    save_cognition_chain_run,
)

# ── Conversation history ──────────────────────────────────────────
from kazusa_ai_chatbot.db.conversation import (
    aggregate_conversation_by_user,
    apply_assistant_delivery_receipt,
    get_ambient_conversation_history,
    get_conversation_by_platform_message_id,
    get_conversation_history,
    get_latest_private_channel_for_user,
    get_participant_conversation_history,
    get_user_message_by_platform_message_id,
    get_user_message_by_row_id,
    has_inbound_after,
    list_conversation_rows_by_row_ids,
    list_recent_group_summaries,
    save_conversation,
    save_conversation_receipt,
    search_conversation_history,
    set_conversation_source_episode_id,
    update_conversation_attachment_descriptions,
    update_conversation_row_llm_trace_id,
)
from kazusa_ai_chatbot.db.conversation_reflection import (
    explain_monitored_channel_query,
    list_recent_character_message_channels,
    list_reflection_scope_messages,
    resolve_single_private_scope_user_id,
)
from kazusa_ai_chatbot.db.errors import (
    DatabaseBackendError,
    DatabaseOperationError,
)
from kazusa_ai_chatbot.db.health import check_database_connection
from kazusa_ai_chatbot.db.interaction_style_images import (
    build_group_engagement_action_context,
    build_interaction_style_context,
    build_user_engagement_relevance_context,
    empty_interaction_style_overlay,
    ensure_interaction_style_image_indexes,
    get_group_channel_style_image,
    get_user_style_image,
    upsert_group_channel_style_image,
    upsert_user_style_image,
    validate_interaction_style_overlay,
)
from kazusa_ai_chatbot.db.internal_action_latches import (
    claim_due_internal_action_latch,
    consume_internal_action_latch,
    ensure_internal_action_latch_indexes,
    expire_due_internal_action_latches,
    fail_internal_action_latch,
    issue_internal_action_latch,
    release_internal_action_latch,
)
from kazusa_ai_chatbot.db.internal_monologue_residue import (
    INTERNAL_MONOLOGUE_RESIDUE_COLLECTION,
    ensure_internal_monologue_residue_indexes,
    insert_internal_monologue_residue_row,
    list_internal_monologue_residue_rows,
)
from kazusa_ai_chatbot.db.llm_tracing import (
    list_llm_trace_steps_for_trace_ids,
)
from kazusa_ai_chatbot.db.post_turn_lifecycle import (
    build_character_operational_lifecycle_record,
    claim_character_operational_receipt,
    commit_character_operational_update,
    complete_character_operational_receipt,
    ensure_post_turn_lifecycle_record_indexes,
    expire_character_operational_receipts,
    get_character_operational_receipt,
    upsert_post_turn_lifecycle_record,
)
from kazusa_ai_chatbot.db.rag_cache2_persistent import (
    build_initializer_version_key,
    build_media_descriptor_version_key,
    load_initializer_entries,
    load_media_descriptor_entries,
    prune_media_descriptor_entries,
    prune_persistent_entries,
    purge_stale_initializer_entries,
    purge_stale_media_descriptor_entries,
    record_initializer_hit,
    record_media_descriptor_hit,
    upsert_initializer_entry,
    upsert_media_descriptor_entry,
)
from kazusa_ai_chatbot.db.reflection_cycle import (
    ensure_reflection_run_indexes,
    find_reflection_run_by_id,
    list_daily_channel_runs,
    list_existing_run_ids,
    list_hourly_runs_for_channel_day,
    list_reflection_runs_for_kind_date,
    upsert_reflection_run,
)

# ── Schemas ────────────────────────────────────────────────────────
from kazusa_ai_chatbot.db.schemas import (
    AttachmentDoc,
    CalendarRunDoc,
    CalendarScheduleDoc,
    CharacterOperationalClaimV1,
    CharacterOperationalReceiptV1,
    CharacterProfileDoc,
    CharacterReflectionRunDoc,
    ConversationEpisodeBlockDoc,
    ConversationEpisodeStateDoc,
    ConversationMessageDoc,
    ConversationProgressEventDoc,
    ConversationProgressSourceRefDoc,
    InteractionStyleImageDoc,
    InteractionStyleOverlayDoc,
    InteractionStyleScopeType,
    InteractionStyleStatus,
    InternalActionLatchClaimV1,
    InternalActionLatchV1,
    InternalMonologueResidueSourceRefDoc,
    InternalMonologueResidueV2Doc,
    MemoryDoc,
    MentionDoc,
    PlatformAccountDoc,
    PostTurnLifecycleRecordV1,
    PostTurnLifecycleRecordV2,
    RAGCache2PersistentEntryDoc,
    ReflectionMessageRefDoc,
    ReflectionScopeDoc,
    ScheduledEventDoc,
    SelfCognitionActionAttemptDoc,
    SelfCognitionGroupReviewWindowDoc,
    UserMemoryContextDoc,
    UserMemoryContextEntry,
    UserMemoryUnitDoc,
    UserMemoryUnitMergeHistoryEntry,
    UserMemoryUnitSourceRef,
    UserMemoryUnitStatus,
    UserMemoryUnitType,
    UserProfileDoc,
    build_memory_doc,
)
from kazusa_ai_chatbot.db.self_cognition import (
    find_self_cognition_group_review_window,
    list_group_review_windows,
    list_self_cognition_action_attempts,
    upsert_self_cognition_action_attempt,
    upsert_self_cognition_group_review_window,
)
from kazusa_ai_chatbot.db.user_memory_units import (
    build_user_memory_unit_doc,
    get_user_memory_unit_by_unit_id,
    insert_user_memory_units,
    query_active_commitment_memory_units,
    query_active_commitment_memory_units_for_user,
    query_user_memory_units,
    search_user_memory_units_by_keyword,
    search_user_memory_units_by_vector,
    update_user_memory_unit_semantics,
    update_user_memory_unit_window,
    validate_user_memory_unit_semantics,
)

_LAZY_MEMORY_EXPORTS = {
    "enable_memory_vector_index",
    "get_active_promises",
    "save_memory",
    "search_memory",
}

_LAZY_DB_EXPORTS = {
    "db_bootstrap": "kazusa_ai_chatbot.db.bootstrap",
    "load_active_episode_state": "kazusa_ai_chatbot.db.conversation_progress",
    "replace_episode_state_guarded": "kazusa_ai_chatbot.db.conversation_progress",
    "insert_conversation_progress_block": (
        "kazusa_ai_chatbot.db.conversation_progress_blocks"
    ),
    "load_conversation_progress_block_graph": (
        "kazusa_ai_chatbot.db.conversation_progress_blocks"
    ),
    "search_conversation_progress_blocks": (
        "kazusa_ai_chatbot.db.conversation_progress_blocks"
    ),
    "supersede_conversation_progress_blocks": (
        "kazusa_ai_chatbot.db.conversation_progress_blocks"
    ),
    "touch_conversation_progress_blocks": (
        "kazusa_ai_chatbot.db.conversation_progress_blocks"
    ),
}

_LAZY_USER_EXPORTS = {
    "add_suspected_alias",
    "backfill_character_conversation_identity",
    "create_user_profile",
    "ensure_character_identity",
    "find_user_profile_by_identifier",
    "get_user_cognition_state",
    "get_user_profile",
    "link_platform_account",
    "list_users_by_relationship",
    "list_users_by_display_name",
    "list_recent_user_profiles",
    "resolve_global_user_id",
    "compare_and_replace_user_cognition_state",
    "replace_user_cognition_state",
    "search_users_by_display_name",
}

_LAZY_CHARACTER_EXPORTS = {
    "LegacyCharacterStateError",
    "RUNTIME_CHARACTER_STATE_FIELDS",
    "compare_and_replace_character_cognition_state",
    "compose_character_profile",
    "ensure_operational_character_state",
    "get_character_cognition_state",
    "get_character_profile",
    "get_character_runtime_state",
    "get_character_state",
    "replace_character_cognition_state",
    "split_character_profile_runtime_state",
}


def __getattr__(name: str) -> Any:
    """Resolve legacy memory helpers without creating import-time cycles.

    The evolving-memory repository imports its DB submodule during package
    import. Loading the legacy memory facade eagerly from here would pull that
    repository back in before it finishes initialising.
    """
    if name in _LAZY_MEMORY_EXPORTS:
        memory_module = import_module("kazusa_ai_chatbot.db.memory")
        resolved_value = getattr(memory_module, name)
        return resolved_value

    module_name = _LAZY_DB_EXPORTS.get(name)
    if module_name is not None:
        module = import_module(module_name)
        resolved_value = getattr(module, name)
        return resolved_value

    if name in _LAZY_USER_EXPORTS:
        module = import_module("kazusa_ai_chatbot.db.users")
        resolved_value = getattr(module, name)
        return resolved_value

    if name in _LAZY_CHARACTER_EXPORTS:
        module = import_module("kazusa_ai_chatbot.db.character")
        resolved_value = getattr(module, name)
        return resolved_value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Client
    "check_database_connection", "close_db", "DatabaseBackendError",
    "DatabaseTestGuardError",
    "DatabaseOperationError",
    "enable_vector_index",
    "get_document_text_embedding", "get_document_text_embeddings_batch",
    "get_query_text_embedding", "get_query_text_embeddings_batch",
    "get_text_embedding", "get_text_embeddings_batch",
    # Schemas
    "AttachmentDoc", "CalendarRunDoc", "CalendarScheduleDoc",
    "CharacterOperationalClaimV1", "CharacterOperationalReceiptV1",
    "CharacterProfileDoc",
    "CharacterReflectionRunDoc",
    "ConversationEpisodeBlockDoc", "ConversationEpisodeStateDoc",
    "ConversationProgressEventDoc", "ConversationProgressSourceRefDoc",
    "ConversationMessageDoc", "InteractionStyleImageDoc",
    "InternalMonologueResidueSourceRefDoc", "InternalMonologueResidueV2Doc",
    "InternalActionLatchClaimV1", "InternalActionLatchV1",
    "InteractionStyleOverlayDoc", "InteractionStyleScopeType",
    "InteractionStyleStatus", "MemoryDoc", "MentionDoc",
    "PostTurnLifecycleRecordV1", "PostTurnLifecycleRecordV2",
    "PlatformAccountDoc", "RAGCache2PersistentEntryDoc",
    "ReflectionMessageRefDoc", "ReflectionScopeDoc",
    "ScheduledEventDoc", "SelfCognitionActionAttemptDoc",
    "SelfCognitionGroupReviewWindowDoc",
    "UserMemoryContextDoc", "UserMemoryContextEntry",
    "UserMemoryUnitDoc", "UserMemoryUnitMergeHistoryEntry", "UserMemoryUnitSourceRef",
    "UserMemoryUnitStatus", "UserMemoryUnitType",
    "UserProfileDoc", "build_memory_doc",
    # Bootstrap
    "db_bootstrap",
    # Conversation
    "aggregate_conversation_by_user", "apply_assistant_delivery_receipt",
    "get_ambient_conversation_history",
    "get_conversation_by_platform_message_id", "get_conversation_history",
    "get_latest_private_channel_for_user",
    "get_participant_conversation_history",
    "get_user_message_by_platform_message_id", "get_user_message_by_row_id",
    "has_inbound_after",
    "list_conversation_rows_by_row_ids", "list_recent_group_summaries",
    "save_conversation", "save_conversation_receipt",
    "load_active_episode_state", "replace_episode_state_guarded",
    "insert_conversation_progress_block",
    "load_conversation_progress_block_graph",
    "search_conversation_progress_blocks",
    "supersede_conversation_progress_blocks",
    "touch_conversation_progress_blocks",
    "search_conversation_history", "set_conversation_source_episode_id",
    "update_conversation_row_llm_trace_id",
    "update_conversation_attachment_descriptions",
    "list_llm_trace_steps_for_trace_ids",
    "ensure_cognition_chain_run_indexes",
    "get_cognition_chain_run",
    "save_cognition_chain_run",
    "explain_monitored_channel_query",
    "list_recent_character_message_channels",
    "list_reflection_scope_messages",
    "resolve_single_private_scope_user_id",
    "ensure_reflection_run_indexes", "find_reflection_run_by_id",
    "list_daily_channel_runs", "list_existing_run_ids",
    "list_hourly_runs_for_channel_day", "list_reflection_runs_for_kind_date",
    "upsert_reflection_run",
    "build_group_engagement_action_context", "build_interaction_style_context",
    "build_user_engagement_relevance_context",
    "empty_interaction_style_overlay", "ensure_interaction_style_image_indexes",
    "get_group_channel_style_image",
    "get_user_style_image", "upsert_group_channel_style_image",
    "upsert_user_style_image", "validate_interaction_style_overlay",
    "INTERNAL_MONOLOGUE_RESIDUE_COLLECTION",
    "ensure_internal_monologue_residue_indexes",
    "insert_internal_monologue_residue_row",
    "list_internal_monologue_residue_rows",
    # Users
    "add_suspected_alias", "backfill_character_conversation_identity",
    "create_user_profile",
    "ensure_character_identity",
    "find_user_profile_by_identifier",
    "get_user_cognition_state",
    "get_user_profile", "link_platform_account",
    "list_users_by_relationship",
    "list_users_by_display_name",
    "list_recent_user_profiles",
    "resolve_global_user_id", "compare_and_replace_user_cognition_state",
    "replace_user_cognition_state",
    "search_users_by_display_name",
    "build_user_memory_unit_doc", "get_user_memory_unit_by_unit_id",
    "insert_user_memory_units",
    "query_active_commitment_memory_units",
    "query_active_commitment_memory_units_for_user",
    "query_user_memory_units",
    "search_user_memory_units_by_keyword",
    "search_user_memory_units_by_vector",
    "update_user_memory_unit_semantics", "update_user_memory_unit_window",
    "validate_user_memory_unit_semantics",
    # Character
    "LegacyCharacterStateError",
    "RUNTIME_CHARACTER_STATE_FIELDS",
    "compare_and_replace_character_cognition_state",
    "compose_character_profile",
    "ensure_operational_character_state",
    "get_character_cognition_state",
    "get_character_profile", "get_character_runtime_state",
    "get_character_state",
    "replace_character_cognition_state",
    "split_character_profile_runtime_state",
    # Character identity growth
    "CANDIDATES_COLLECTION", "GROWTH_COLLECTION_NAMES",
    "IDENTITY_INDEX_NAMES", "REVISIONS_COLLECTION", "RUNS_COLLECTION",
    "CharacterIdentityPersistenceError",
    "ConcurrentIdentityPromotionError", "IdentityCandidateConflictError",
    "IdentityLedgerCorruptionError", "IdentityLedgerNotFoundError",
    "IdentityRootAlreadyClaimedError", "IdentityRunConflictError",
    "IdentityTransactionUnavailableError", "SeedIdentityConflictError",
    "count_inferred_identity_promotions_on_local_date",
    "create_operator_reset_revision",
    "ensure_character_identity_growth_indexes", "ensure_seed_identity",
    "get_current_identity", "get_growth_run", "insert_growth_candidate",
    "insert_growth_run", "list_current_growth_candidates",
    "list_identity_revisions", "promote_ready_candidate",
    "reject_growth_candidates", "update_growth_candidate",
    # Internal action latches and post-turn lifecycle
    "claim_due_internal_action_latch", "consume_internal_action_latch",
    "ensure_internal_action_latch_indexes", "expire_due_internal_action_latches",
    "fail_internal_action_latch", "issue_internal_action_latch",
    "release_internal_action_latch",
    "ensure_post_turn_lifecycle_record_indexes",
    "build_character_operational_lifecycle_record",
    "claim_character_operational_receipt",
    "commit_character_operational_update",
    "complete_character_operational_receipt",
    "expire_character_operational_receipts",
    "get_character_operational_receipt",
    "upsert_post_turn_lifecycle_record",
    # Memory
    "enable_memory_vector_index", "get_active_promises", "save_memory", "search_memory",
    # Self-cognition action attempts
    "find_self_cognition_group_review_window",
    "list_group_review_windows",
    "list_self_cognition_action_attempts",
    "upsert_self_cognition_group_review_window",
    "upsert_self_cognition_action_attempt",
    "claim_background_work_job",
    "complete_background_work_job",
    "ensure_background_work_job_indexes",
    "fail_background_work_job",
    "find_deliverable_background_work_jobs",
    "insert_background_work_job",
    "mark_background_work_delivered",
    "mark_background_work_delivery_failed",
    "mark_background_work_delivery_in_progress",
    # Persistent Cache2
    "build_initializer_version_key", "build_media_descriptor_version_key",
    "load_initializer_entries", "load_media_descriptor_entries",
    "prune_media_descriptor_entries", "prune_persistent_entries",
    "purge_stale_initializer_entries", "purge_stale_media_descriptor_entries",
    "record_initializer_hit", "record_media_descriptor_hit",
    "upsert_initializer_entry", "upsert_media_descriptor_entry",
]
