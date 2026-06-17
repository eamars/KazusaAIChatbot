# Script Registry for `src/scripts`

Generated from active entry points, runtime docs, and tests on `2026-05-17`.

## Active scripts

Normal operator startup is `kazusa-control-console`, then service lifecycle
from the console. The direct scripts and entry points below remain fallback,
diagnostic, and maintenance commands.

| File | Invocation | Purpose |
|---|---|---|
| `_db_export.py` | Shared module only | Internal helper utilities for export scripts (`load_project_env`, JSON writers, env setup). |
| `character_state_snapshot.py` | `python -m scripts.character_state_snapshot snapshot|restore` / `character-state-snapshot` | Snapshot and restore the singleton `character_state` document. |
| `count_project_artifacts.py` | `python -m scripts.count_project_artifacts` / `count-project-artifacts` | Count repo production/test/docs artifacts for quick repo health checks. |
| `create_conversation_history_embedding.py` | `create-embeddings` | Rebuild conversation-history embeddings and refresh vector-search index. |
| `drop_legacy_rag_collections.py` | `python scripts/drop_legacy_rag_collections.py` | One-shot cleanup for legacy RAG collections (`rag_cache_index`, `rag_metadata_index`). |
| `ensure_vector_search_indexes.py` | `python -m scripts.ensure_vector_search_indexes` | Inspect and optionally recreate approved vector-search indexes. |
| `export_character_state.py` | `python -m scripts.export_character_state` / `export-character-state` | Export singleton character state JSON. |
| `export_chat_history.py` | `python -m scripts.export_chat_history <channel-id>` / `export-chat-history` | Export conversation history rows for a channel with optional time filters. |
| `export_collection.py` | `python -m scripts.export_collection <collection>` / `export-collection` | Export arbitrary collection rows by filter/sort/limit. |
| `export_event_log.py` | `python -m scripts.export_event_log --hours 24` | Export sanitized event-log aggregate diagnostics. |
| `export_memory.py` | `python -m scripts.export_memory` / `export-memory` | Export memory rows (optionally filtered by memory type/status). |
| `export_user_image.py` | `python -m scripts.export_user_image <user>` / `export-user-image` | Export a user image profile bundle. |
| `export_user_memories.py` | `python -m scripts.export_user_memories <user>` / `export-user-memories` | Export user memory rows. |
| `export_user_profile.py` | `python -m scripts.export_user_profile <user>` / `export-user-profile` | Export normalized user profile documents. |
| `fetch_ops_status.py` | `python -m scripts.fetch_ops_status` / `fetch-ops-status` | Query runtime operational status snapshots. |
| `identify_group_image.py` | `python -m scripts.identify_group_image <group-id>` / `identify-group-image` | Export/inspect group image metadata by ID. |
| `identify_user_image.py` | `python -m scripts.identify_user_image <user-id>` / `identify-user-image` | Export/inspect user image diagnostics. |
| `inspect_consolidation_target_lifecycle.py` | `python -m scripts.inspect_consolidation_target_lifecycle [--apply]` | Dry-run report and approved apply cleanup for synthetic consolidation user rows and malformed target lifecycle data. |
| `load_character_profile.py` | `python -m scripts.load_character_profile personalities/<file>.json` | Load a character profile JSON into MongoDB (service bootstrap prerequisite). |
| `manage_memory_knowledge.py` | `manage-memory-knowledge` | Edit and sync local memory knowledge entries. |
| `migrate_conversation_history_envelope.py` | `python -m scripts.migrate_conversation_history_envelope [--apply]` | Repair conversation rows that violate typed-envelope storage fields or semantic-text cleanliness. |
| `profile_embedding_prefix_modes.py` | `python -m scripts.profile_embedding_prefix_modes` | Compare embedding prefix strategies for RAG tuning. |
| `profile_rag_retrieval.py` | `python -m scripts.profile_rag_retrieval` | Generate RAG retrieval profile cases for tuning/validation. |
| `reembed_text_vector_embeddings.py` | `python -m scripts.reembed_text_vector_embeddings` | Replay text embedding generation for documents/collections. |
| `reset_memory_lore.py` | `python -m scripts.reset_memory_lore --dry-run|--apply` | Reset and regenerate shared memory lore under operator control. |
| `run_global_character_growth.py` | `python -m scripts.run_global_character_growth --dry-run` | Manual global growth pass from reflection-promoted memory. |
| `run_reflection_cycle.py` | `python -m scripts.run_reflection_cycle hourly|daily|promote` | Run production reflection worker modes (dry-run supported). |
| `run_reflection_cycle_readonly.py` | `python -m scripts.run_reflection_cycle_readonly --lookback-hours 24` | Read-only reflection-cycle evaluator for diagnostics. |
| `sanitize_memory_writer_perspective.py` | `python -m scripts.sanitize_memory_writer_perspective` | Offline migration/sanitization of durable memory perspective wording. |
| `search_conversation.py` | `search-conversations` | Search conversation history with keyword or vector search. |
| `search_memory.py` | `python -m scripts.search_memory` | Search memory entries with keyword or vector modes. |
| `user_state_snapshot.py` | `python -m scripts.user_state_snapshot snapshot|restore` / `user-state-snapshot` | Snapshot/restore authoritative per-user state across user collections. |
