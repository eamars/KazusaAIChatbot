# Script Registry for `src/scripts`

## Document Control

- Owning package: `scripts`
- Runtime role: operator, diagnostics, export, maintenance, and migration CLIs
- Normal startup owner: `kazusa-control-console`
- Related runbook: [HOWTO](../../docs/HOWTO.md)

## Purpose

This package guide lists active script entrypoints. Normal local operation is
`kazusa-control-console`, then service lifecycle from the console. Direct
scripts and console entrypoints remain fallback, diagnostic, maintenance, and
data-repair commands.

The table is maintained as documentation. Source modules and `pyproject.toml`
entry points remain the implementation source of truth for exact invocation
availability.

## Public Interfaces

| File | Invocation | Purpose |
|---|---|---|
| `_db_export.py` | Shared module only | Internal helper utilities for export scripts (`load_project_env`, JSON writers, env setup). |
| `apply_logging_retention.py` | `python -m scripts.apply_logging_retention --dry-run|--apply` | Assign or delete legacy logging rows under `AUDIT_LOG_TTL_DAYS` and `DEBUG_LOG_TTL_DAYS`. |
| `character_state_snapshot.py` | `python -m scripts.character_state_snapshot snapshot|restore` / `character-state-snapshot` | Snapshot and restore the singleton `character_state` document. |
| `count_project_artifacts.py` | `python -m scripts.count_project_artifacts` / `count-project-artifacts` | Count repo production, test, and docs artifacts for quick repo health checks. |
| `create_conversation_history_embedding.py` | `create-embeddings` | Rebuild conversation-history embeddings and refresh vector-search index. |
| `drop_legacy_rag_collections.py` | `python scripts/drop_legacy_rag_collections.py` | One-shot cleanup for legacy RAG collections (`rag_cache_index`, `rag_metadata_index`). |
| `ensure_vector_search_indexes.py` | `python -m scripts.ensure_vector_search_indexes` | Inspect and optionally recreate approved vector-search indexes. |
| `export_character_state.py` | `python -m scripts.export_character_state` / `export-character-state` | Export singleton character state JSON. |
| `export_chat_history.py` | `python -m scripts.export_chat_history <channel-id>` / `export-chat-history` | Export conversation history rows for a channel with optional time filters. |
| `export_collection.py` | `python -m scripts.export_collection <collection>` / `export-collection` | Export arbitrary collection rows by filter, sort, and limit. |
| `export_dialog_trace_review_input.py` | `python -m scripts.export_dialog_trace_review_input --dialog-text <text>` | Export compact review input for one dialog's protected LLM trace. |
| `export_event_log.py` | `python -m scripts.export_event_log --hours 24` | Export sanitized event-log aggregate diagnostics. |
| `export_llm_trace.py` | `python -m scripts.export_llm_trace --trace-id <id>` | Export protected LLM trace rows plus linked audit and conversation rows. |
| `export_memory.py` | `python -m scripts.export_memory` / `export-memory` | Export memory rows, optionally filtered by memory type or status. |
| `export_user_image.py` | `python -m scripts.export_user_image <user>` / `export-user-image` | Export a user image profile bundle. |
| `export_user_memories.py` | `python -m scripts.export_user_memories <user>` / `export-user-memories` | Export user memory rows. |
| `export_user_profile.py` | `python -m scripts.export_user_profile <user>` / `export-user-profile` | Export normalized user profile documents. |
| `fetch_ops_status.py` | `python -m scripts.fetch_ops_status` / `fetch-ops-status` | Query runtime operational status snapshots. |
| `identify_group_image.py` | `python -m scripts.identify_group_image <group-id>` / `identify-group-image` | Export or inspect group image metadata by ID. |
| `identify_user_image.py` | `python -m scripts.identify_user_image <user-id>` / `identify-user-image` | Export or inspect user image diagnostics. |
| `inspect_consolidation_target_lifecycle.py` | `python -m scripts.inspect_consolidation_target_lifecycle [--apply]` | Dry-run report and approved apply cleanup for synthetic consolidation user rows and malformed target lifecycle data. |
| `load_character_profile.py` | `python -m scripts.load_character_profile personalities/<file>.json` | Load a character profile JSON into MongoDB before brain startup. |
| `manage_memory_knowledge.py` | `manage-memory-knowledge` | Edit and sync local memory knowledge entries. |
| `migrate_conversation_history_envelope.py` | `python -m scripts.migrate_conversation_history_envelope [--apply]` | Repair conversation rows that violate typed-envelope storage fields or semantic-text cleanliness. |
| `profile_embedding_prefix_modes.py` | `python -m scripts.profile_embedding_prefix_modes` | Compare embedding prefix strategies for RAG tuning. |
| `profile_rag_retrieval.py` | `python -m scripts.profile_rag_retrieval` | Generate RAG retrieval profile cases for tuning and validation. |
| `reembed_text_vector_embeddings.py` | `python -m scripts.reembed_text_vector_embeddings` | Replay text embedding generation for documents or collections. |
| `reset_memory_lore.py` | `python -m scripts.reset_memory_lore --dry-run|--apply` | Reset and regenerate shared memory lore under operator control. |
| `run_global_character_growth.py` | `python -m scripts.run_global_character_growth --dry-run` | Manual global growth pass from reflection-promoted memory. |
| `run_reflection_cycle.py` | `python -m scripts.run_reflection_cycle hourly|daily|promote` | Run production reflection worker modes with dry-run support. |
| `run_reflection_cycle_readonly.py` | `python -m scripts.run_reflection_cycle_readonly --lookback-hours 24` | Read-only reflection-cycle evaluator for diagnostics. |
| `sanitize_memory_writer_perspective.py` | `python -m scripts.sanitize_memory_writer_perspective` | Offline migration and sanitization of durable memory perspective wording. |
| `search_conversation.py` | `search-conversations` | Search conversation history with keyword or vector search. |
| `search_memory.py` | `python -m scripts.search_memory` | Search memory entries with keyword or vector modes. |
| `user_state_snapshot.py` | `python -m scripts.user_state_snapshot snapshot|restore` / `user-state-snapshot` | Snapshot or restore authoritative per-user state across user collections. |

## Testing Contract

Script tests should verify argument parsing, dry-run behavior, output shape,
and read/write boundaries without requiring live DB or live LLM access unless
the test is explicitly marked. Maintenance scripts that can mutate data should
provide a dry-run path and focused tests for validation before apply behavior.

## Forbidden Paths

- Do not treat this registry as authorization to run maintenance scripts.
- Do not read `.env` directly in documentation or tests.
- Do not add a script entry here unless the source module or entry point exists.
- Do not use scripts to bypass module ICD boundaries, accepted-task lifecycle,
  scheduler ownership, adapter delivery rules, or database write validation.
