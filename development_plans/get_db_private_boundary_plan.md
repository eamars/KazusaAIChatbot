# get_db private boundary plan

## Summary

- Goal: Keep `get_db` as the internal DB-client helper, remove it from the public `kazusa_ai_chatbot.db` facade, and route runtime/application and script database access through semantic DB interfaces.
- Plan class: medium
- Status: draft
- Mandatory skills: `py-style`, `test-style-and-execution`, `local-llm-architecture`.
- Overall cutover strategy: compatible for runtime and script behavior; incompatible only for unsupported public imports of `kazusa_ai_chatbot.db.get_db`.
- Highest-risk areas: service health, scheduler event status transitions, pending-task index rebuild, conversation-progress guarded upsert semantics, script export/snapshot behavior, and tests that currently patch raw DB handles.
- Acceptance criteria: production application code and `src/scripts` code outside `src/kazusa_ai_chatbot/db/` no longer import or call `get_db`; do not import any symbol from `kazusa_ai_chatbot.db._client`; do not import any underscore-prefixed name from any `kazusa_ai_chatbot.db.*` submodule; do not import `motor` or `pymongo`; and do not contain any of the forbidden raw-backend tokens enumerated in `Mandatory Rules`. `get_db` is absent from `db.__init__` and `db.__all__`. Existing runtime/script behavior remains unchanged.

## Context

The DB package already contains most raw MongoDB access inside `src/kazusa_ai_chatbot/db/`, but `get_db` is still re-exported from `kazusa_ai_chatbot.db`. That makes the backend handle part of the public application API and encourages callers to work with Mongo collections directly.

Current runtime/application leaks found before this plan:

- `src/kazusa_ai_chatbot/service.py` imports `get_db` only to pass it into health construction.
- `src/kazusa_ai_chatbot/brain_service/health.py` calls `db.client.admin.command("ping")`.
- `src/kazusa_ai_chatbot/scheduler.py` performs direct `scheduled_events` inserts, finds, and status updates.
- `src/kazusa_ai_chatbot/dispatcher/pending_index.py` rebuilds from `db.scheduled_events.find(...)`.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py` imports `kazusa_ai_chatbot.db._client.get_db`, imports `pymongo.errors.DuplicateKeyError`, and performs raw `db[COLLECTION_NAME].find_one(...)` and guarded `update_one(..., upsert=True)` operations.
- `src/kazusa_ai_chatbot/db/__init__.py` imports and exports `get_db`.

Additional private-boundary leaks discovered during plan review (must also be closed by this plan):

- `src/scripts/user_state_snapshot.py` imports private DB-internal helpers `kazusa_ai_chatbot.db.conversation._embedding_source_text`, `kazusa_ai_chatbot.db.memory.memory_embedding_source_text`, and `kazusa_ai_chatbot.db.user_memory_units._semantic_text`. These bypass the public facade exactly like `get_db` does and must be replaced with public DB functions or kept inside the DB package.
- `src/scripts/inject_knowledge.py` and `src/scripts/drop_legacy_rag_collections.py` import directly from `kazusa_ai_chatbot.db._client` (not just `get_db`). The boundary must ban any symbol from `db._client` outside the DB package, not only `get_db`.
- No production code outside `src/kazusa_ai_chatbot/db/` may import `motor`, `pymongo`, or any submodule of either (including `pymongo.errors`). The DB package is the sole owner of backend driver dependencies.

This plan is not a production data migration. It only makes Mongo an implementation detail of the DB package; it does not switch backends, rename collections, redesign schemas, or mutate existing production documents. Future backend replacement requires a separate plan after this boundary is enforced.

`src/scripts/` contains backend-specific maintenance and export tools that also import `get_db` or `db._client`. This plan includes updating those affected script callers to use public `kazusa_ai_chatbot.db` interfaces, or the public maintenance-only `kazusa_ai_chatbot.db.script_operations` submodule, instead of raw DB handles.

Affected scripts found during discovery:

- `src/scripts/create_conversation_history_embedding.py`
- `src/scripts/character_state_snapshot.py`
- `src/scripts/drop_legacy_rag_collections.py`
- `src/scripts/export_collection.py`
- `src/scripts/export_memory.py`
- `src/scripts/export_user_image.py`
- `src/scripts/export_user_memories.py`
- `src/scripts/export_user_profile.py`
- `src/scripts/inject_knowledge.py`
- `src/scripts/insert_memory.py`
- `src/scripts/migrate_conversation_history_envelope.py`
- `src/scripts/sanitize_memory_writer_perspective.py`
- `src/scripts/search_conversation.py`
- `src/scripts/search_memory.py`
- `src/scripts/user_state_snapshot.py`

Do not silently migrate those scripts to `db._client.get_db`; add or reuse public semantic DB interfaces instead.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before touching RAG, scheduler-dispatch, conversation progress, or prompt-adjacent runtime paths so the implementation preserves existing LLM call boundaries.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Do not remove `get_db` from `src/kazusa_ai_chatbot/db/_client.py`.
- Do not rename `get_db` to `_get_db` in this plan.
- Do remove `get_db` from `src/kazusa_ai_chatbot/db/__init__.py` imports and `__all__`.
- Do document `get_db` as private DB-package infrastructure in `db/_client.py` and `db/README.md`.
- Do not add a compatibility `__getattr__`, alias, re-export, or fallback that keeps `kazusa_ai_chatbot.db.get_db` working. The existing `_LAZY_MEMORY_EXPORTS` `__getattr__` in `db/__init__.py` is permitted to stay because it does not expose `get_db` or any raw backend handle; do not extend it to cover `get_db`.
- Do not import any symbol from `kazusa_ai_chatbot.db._client` outside `src/kazusa_ai_chatbot/db/`. This covers `get_db`, `close_db`, `enable_vector_index`, `get_text_embedding`, `get_text_embeddings_batch`, and any future symbol in that module. Runtime and script callers must import these names from the public `kazusa_ai_chatbot.db` facade only.
- Do not import any private (underscore-prefixed) name from any `kazusa_ai_chatbot.db.*` submodule outside `src/kazusa_ai_chatbot/db/`. Currently leaking private names that must be eliminated: `kazusa_ai_chatbot.db.conversation._embedding_source_text`, `kazusa_ai_chatbot.db.user_memory_units._semantic_text`. (`kazusa_ai_chatbot.db.memory.memory_embedding_source_text` is not underscore-prefixed but is undocumented internal-use only and must be treated the same way — either elevate it to a public, documented DB interface or move its callers behind a public DB function.)
- Do not import `motor`, `pymongo`, or any submodule of either (including `pymongo.errors`) outside `src/kazusa_ai_chatbot/db/`. The DB package is the sole owner of backend driver dependencies; raise application-level exceptions from semantic DB functions instead of letting `pymongo.errors.*` propagate to callers.
- Do not move raw Mongo collection, query, projection, aggregation, or update details into application modules. The forbidden raw-backend tokens that must not appear anywhere under `src/kazusa_ai_chatbot/` outside `src/kazusa_ai_chatbot/db/` (and must not appear anywhere under `src/scripts/`) are at least: `db.client`, `.admin.command`, `.insert_one(`, `.insert_many(`, `.update_one(`, `.update_many(`, `.replace_one(`, `.find(`, `.find_one(`, `.delete_one(`, `.delete_many(`, `.aggregate(`, `.count_documents(`, `.distinct(`, `.bulk_write(`, `.create_index(`, `.list_indexes(`, `.list_collection_names(`, `.drop(`, and indexed access of the form `db[<collection_literal>]`.
- Do not change DB schemas, collection names, indexes, write shapes, scheduling semantics, cache invalidation, prompts, graph nodes, LLM call count, or endpoint schemas.
- Do update affected `src/scripts/` callers so they use public `kazusa_ai_chatbot.db` interfaces only.
- Scripts must not import `get_db`, `db._client`, Motor, PyMongo, raw collection handles, or DB implementation modules.
- Scripts must not contain raw Mongo collection, query, projection, aggregation, or update details after this plan.
- Do not mutate existing production database contents as part of this plan.
- If a caller needs a DB capability, add a narrow semantic function in `src/kazusa_ai_chatbot/db/` and call that function from application code.
- If a needed semantic DB function would require changing behavior or data shape, stop and report the blocker instead of expanding scope.

## Must Do

- Add static boundary tests before implementation.
- Add a public DB health interface so service health no longer receives a raw DB handle.
- Move scheduler persistence operations behind `kazusa_ai_chatbot.db.scheduled_events` semantic functions.
- Move pending-index rebuild reads behind a DB semantic function.
- Move conversation-progress raw persistence calls into `src/kazusa_ai_chatbot/db/conversation_progress.py`.
- Update affected `src/scripts/` callers to use public DB interfaces only.
- Add narrow public DB interfaces needed by affected scripts.
- Remove `get_db` from the public `kazusa_ai_chatbot.db` facade.
- Update runtime/application imports and tests to use semantic DB interfaces.
- Preserve all runtime behavior and currently tested request/response behavior.
- Record command output and grep evidence before marking stages complete.

## Deferred

- Do not implement SQL, an ORM, a repository base class, or a backend plugin system.
- Do not change the concrete Mongo implementation inside `src/kazusa_ai_chatbot/db/`.
- Do not rename `get_db` in `db/_client.py`.
- Do not remove `close_db`, `db_bootstrap`, embedding helpers, or vector-index helpers from the public DB facade in this plan.
- Do not add a data-migration script or production database content update path in this plan.
- Do not change script CLI arguments, output formats, selected collections, or exported document shapes except where the existing shape exposes raw backend implementation details that must be removed.
- Do not change tests that intentionally validate DB-package internals, except where needed to patch `db._client.get_db` directly inside DB-layer tests.

## Cutover Policy

| Area | Policy | Instruction |
|---|---|---|
| Public DB facade | compatible except `get_db` import | Remove only the raw DB handle export; keep existing semantic DB exports. |
| Runtime application behavior | compatible | Scheduler, health, progress, chat, RAG, and reflection behavior must remain unchanged. |
| DB implementation | compatible | Mongo remains the concrete backend behind the DB package. |
| Tests | compatible | Update tests to assert interfaces, not raw handles, outside DB-layer tests. |
| Affected scripts/admin tools | compatible | Update existing script callers to use public DB interfaces while preserving CLI and output behavior. |
| Deployment | compatible | No Docker, environment variable, or startup command changes. |

## Agent Autonomy Boundaries

- The target ownership boundary is `src/kazusa_ai_chatbot/db/` as the only package allowed to own raw database handles and backend query language.
- The agent may add narrow semantic DB functions only for capabilities already used by current runtime code.
- The agent may add narrow public DB functions only for capabilities already used by affected scripts.
- The agent must not invent generic repositories, query builders, or backend abstraction frameworks.
- The agent must not use string-built dynamic DB operations in application code to work around the boundary.
- The agent must keep compatibility wrappers only when they do not re-expose `get_db` or raw collection handles.
- The agent must treat any edit outside `src/kazusa_ai_chatbot/db/`, `src/kazusa_ai_chatbot/service.py`, `src/kazusa_ai_chatbot/brain_service/health.py`, `src/kazusa_ai_chatbot/scheduler.py`, `src/kazusa_ai_chatbot/dispatcher/pending_index.py`, `src/kazusa_ai_chatbot/conversation_progress/repository.py`, and focused tests as out of scope unless this plan names it.
- The agent may edit only the affected `src/scripts/` files listed in `Context`.
- If existing tests rely on monkeypatching raw DB handles outside DB-layer tests, update those tests to patch semantic DB functions instead of preserving the raw handle dependency.

## Target State

`get_db` still exists at:

```python
kazusa_ai_chatbot.db._client.get_db
```

It is documented as private infrastructure for DB-package internals. Runtime code outside `src/kazusa_ai_chatbot/db/` does not import it, call it, receive it through dependency injection, or operate on raw Mongo collections.

The public DB facade still exposes semantic capabilities such as:

```python
close_db
db_bootstrap
get_character_profile
get_conversation_history
save_conversation
query_pending_scheduled_events
resolve_global_user_id
```

New public or package-level semantic capabilities are added only where current runtime or affected script code already needs them.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Private helper name | Keep `get_db` in `_client.py` | The user explicitly does not want the function removed; DB internals can continue using it. |
| Public facade | Remove `get_db` from `db.__init__` and `__all__` | The public package should expose behavior, not backend handles. |
| Health check | Add DB-owned health function | Service health should ask whether DB is available, not ping Mongo directly. |
| Scheduler persistence | Move to `db.scheduled_events` semantic functions | Scheduler owns in-memory timers; DB package owns event document persistence. |
| Pending index rebuild | Read through scheduler-event DB interface | Dedup index should not know collection names or query shape. |
| Conversation progress persistence | DB package owns raw guarded upsert | Race-guard persistence is backend-specific and belongs behind DB interface. |
| Script policy | Update affected `src/scripts` callers | Scripts are application callers and must follow the same no-direct-DB boundary as runtime code. |

## Interface Contract

### DB health

Create `src/kazusa_ai_chatbot/db/health.py` and export from `kazusa_ai_chatbot.db`:

```python
async def check_database_connection() -> bool:
    """Return whether the configured DB backend is reachable."""
```

The function may call `get_db()` and backend-specific ping logic internally. Callers receive only `True` or `False`.

### Scheduled events

Extend `src/kazusa_ai_chatbot/db/scheduled_events.py` with semantic functions:

```python
async def insert_scheduled_event(event: ScheduledEventDoc) -> None: ...

async def list_pending_scheduler_events() -> list[ScheduledEventDoc]: ...

async def mark_scheduled_event_running(event_id: str) -> bool: ...

async def mark_scheduled_event_completed(event_id: str) -> bool: ...

async def mark_scheduled_event_failed(event_id: str) -> bool: ...

async def cancel_pending_scheduled_event(
    event_id: str,
    *,
    cancelled_at: str,
) -> bool: ...
```

Keep existing `query_pending_scheduled_events(...)` for RAG Recall. Do not change its result shape.

### Conversation progress persistence

Create `src/kazusa_ai_chatbot/db/conversation_progress.py` with:

```python
async def load_episode_state(
    *,
    scope: ConversationProgressScope,
) -> ConversationEpisodeStateDoc | None: ...

async def upsert_episode_state_guarded(
    *,
    document: ConversationEpisodeStateDoc,
) -> bool: ...
```

`conversation_progress.repository` may remain as the domain-facing module for pure document construction and compatibility, but it must not import `db._client.get_db` or perform raw collection operations after this plan.

### Script-facing DB interfaces

For each affected script capability, reuse existing public DB functions where they already express the semantic operation. Add new public DB functions only when the script currently performs a necessary operation that has no public interface.

Script-facing DB functions must be named by domain behavior, not by backend mechanics. Examples:

```python
async def export_collection_rows(...) -> list[dict]: ...

async def replace_user_state_snapshot(...) -> None: ...

async def update_conversation_history_envelope_rows(...) -> MigrationResult: ...
```

The implementation may live inside existing DB submodules or newly focused DB submodules. Runtime public callers remain on `kazusa_ai_chatbot.db`. Maintenance scripts may also call the public `kazusa_ai_chatbot.db.script_operations` submodule, which is the public operator-facing DB surface for export, snapshot, and maintenance operations. Scripts must not import private DB implementation modules, must not import any underscore-prefixed name from any `kazusa_ai_chatbot.db.*` submodule, and must not import `motor` or `pymongo`.

#### Private DB-internal helpers currently used by `user_state_snapshot.py`

`user_state_snapshot.py` builds embedding-derived fields by calling `_embedding_source_text` (from `db.conversation`), `memory_embedding_source_text` (from `db.memory`), and `_semantic_text` (from `db.user_memory_units`). These are DB-internal text-derivation helpers and must not remain importable from the script. Acceptable resolutions, in order of preference:

1. Add a single public DB function that performs the snapshot read and returns documents already enriched with the derived fields, so the script never recomputes embedding source text on its own.
2. Promote the three helpers to public, documented DB interfaces with a stable name (no leading underscore, listed in `db/__init__.py` `__all__`) only if the snapshot script genuinely needs them as building blocks and option 1 is unworkable without changing snapshot output shape.

Either resolution must keep the script's snapshot output bytes-identical to current behavior, per `Cutover Policy`.

### Private client documentation

Update `src/kazusa_ai_chatbot/db/_client.py` docstrings to state:

- The module is backend implementation infrastructure.
- `get_db` is private to `kazusa_ai_chatbot.db` submodules.
- Runtime/application callers must use semantic functions exported by `kazusa_ai_chatbot.db`.

Update `src/kazusa_ai_chatbot/db/README.md` with the same policy and examples.

## LLM Call And Context Budget

This plan must not alter LLM prompts, call count, state keys, RAG routing, reflection routing, scheduler-dispatch prompt payloads, or consolidation behavior.

Expected budget:

- Response path: unchanged.
- Background consolidation path: unchanged.
- Scheduler task-generation LLM path: unchanged.
- RAG Recall path: unchanged functionally; only the scheduled-event DB read implementation boundary changes.

Any implementation that requires changing prompts, graph nodes, RAG capability routing, or LLM output parsing is out of scope.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/db/health.py`
  - DB-owned health check interface.
- `src/kazusa_ai_chatbot/db/conversation_progress.py`
  - DB-owned conversation-progress persistence interface; pure document construction remains in `conversation_progress.repository`.
- `tests/test_db_public_boundary.py`
  - Static boundary tests that scan every `.py` file under `src/kazusa_ai_chatbot/` outside `src/kazusa_ai_chatbot/db/` (no hand-listed allow/deny set) and assert: no `get_db` references; no imports from `kazusa_ai_chatbot.db._client`; no underscore-prefixed imports from any `kazusa_ai_chatbot.db.*` submodule; no `motor` / `pymongo` imports; and none of the forbidden raw-backend tokens enumerated in `Mandatory Rules`. Also assert `get_db` is not in `kazusa_ai_chatbot.db.__all__` and not resolvable as `kazusa_ai_chatbot.db.get_db` (including via `__getattr__`).
- Focused DB submodules if needed for script-facing semantic interfaces.
  - Only create when an affected script capability has no existing public DB interface.
- `tests/test_script_db_boundary.py`
  - Static boundary tests that scan every `.py` file under `src/scripts/` (no hand-listed allow/deny set) and assert: no `get_db` references; no imports from `kazusa_ai_chatbot.db._client`; no underscore-prefixed imports from any `kazusa_ai_chatbot.db.*` submodule; no `motor` / `pymongo` imports; and none of the forbidden raw-backend tokens enumerated in `Mandatory Rules`.

### Modify

- `src/kazusa_ai_chatbot/db/__init__.py`
  - Remove `get_db` import and `__all__` entry.
  - Export new semantic DB functions.
- `src/kazusa_ai_chatbot/db/_client.py`
  - Document `get_db` as private. Keep behavior unchanged.
- `src/kazusa_ai_chatbot/db/README.md`
  - Document public/private DB boundary.
- `src/kazusa_ai_chatbot/db/scheduled_events.py`
  - Add scheduler persistence functions.
- `src/kazusa_ai_chatbot/service.py`
  - Stop importing `get_db`; import `check_database_connection` from `kazusa_ai_chatbot.db` and pass it to `brain_health.build_health_response`.
- `src/kazusa_ai_chatbot/brain_service/health.py`
  - Replace `get_db_func` with `check_database_connection_func`; call that dependency directly and do not inspect a raw DB object.
- `src/kazusa_ai_chatbot/scheduler.py`
  - Replace raw scheduled-event DB operations with semantic DB functions.
- `src/kazusa_ai_chatbot/dispatcher/pending_index.py`
  - Replace raw pending-event read with semantic DB function.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py`
  - Remove raw DB calls; delegate persistence to DB-owned functions while keeping pure document-building helpers.
- Affected scripts listed in `Context`
  - Replace raw DB access with public `kazusa_ai_chatbot.db` calls while preserving CLI behavior and output shape.
  - For `src/scripts/user_state_snapshot.py` specifically: remove imports of `kazusa_ai_chatbot.db.conversation._embedding_source_text`, `kazusa_ai_chatbot.db.memory.memory_embedding_source_text`, and `kazusa_ai_chatbot.db.user_memory_units._semantic_text`; route through the public DB function added per `Interface Contract > Private DB-internal helpers currently used by user_state_snapshot.py`.
  - For `src/scripts/inject_knowledge.py` and `src/scripts/drop_legacy_rag_collections.py` specifically: remove direct imports from `kazusa_ai_chatbot.db._client` and use only public `kazusa_ai_chatbot.db` exports.
- Focused tests under `tests/`
  - Update patches from raw DB handles to semantic DB functions where appropriate.
  - Update `tests/test_service_health.py` to patch `check_database_connection` (or the dependency parameter that replaces `get_db_func`) instead of patching `get_db` or its returned client.

### Keep

- `src/kazusa_ai_chatbot/db/_client.py:get_db`
- Existing DB schemas and collection names.
- Existing endpoint schemas and service entrypoints.
- Existing scheduler, dispatcher, RAG, reflection, and consolidation behavior.
- Unaffected `src/scripts/*` behavior and imports.

## Implementation Order

1. Baseline and boundary inventory.
   - Run focused tests before edits.
   - Record current `get_db` grep output.
2. Add failing boundary tests.
   - Add `tests/test_db_public_boundary.py` and `tests/test_script_db_boundary.py`.
   - Assert `get_db` is not exported by `kazusa_ai_chatbot.db` and is not resolvable as `kazusa_ai_chatbot.db.get_db` (including via `__getattr__`).
   - Assert that for every `.py` file under `src/kazusa_ai_chatbot/` outside `src/kazusa_ai_chatbot/db/` and every `.py` file under `src/scripts/`: no `get_db` references; no imports from `kazusa_ai_chatbot.db._client`; no underscore-prefixed imports from any `kazusa_ai_chatbot.db.*` submodule; no imports of `motor`, `pymongo`, or any submodule of either; and none of the forbidden raw-backend tokens enumerated in `Mandatory Rules`.
   - The tests must walk the file tree at runtime; a hard-coded list of files is not acceptable because it cannot catch leaks introduced after this plan ships.
3. Add DB health interface and update service health.
   - Implement `check_database_connection()`.
   - Update `brain_service.health` and `service.py`.
   - Run service health tests.
4. Add scheduled-event semantic functions and update scheduler paths.
   - Implement functions in `db.scheduled_events`.
   - Update `scheduler.py` and `dispatcher/pending_index.py`.
   - Run dispatcher and scheduler-adjacent tests.
5. Move conversation-progress persistence behind DB package.
   - Add DB-owned persistence functions.
   - Update `conversation_progress.repository`.
   - Run conversation-progress tests.
6. Move affected scripts behind public DB interfaces.
   - Add or reuse public DB functions needed by affected scripts.
   - Update affected scripts listed in `Context`.
   - Run script-boundary tests and focused script tests.
7. Remove public `get_db` export and document privacy.
   - Update `db.__init__`, `_client.py`, and `db/README.md`.
   - Do not add compatibility aliases.
8. Run final verification.
   - Run static greps, focused runtime tests, DB tests, and broader smoke.
   - Record script boundary grep evidence.

## Progress Checklist

- [ ] Stage 1 - Baseline and boundary tests added.
  - Covers: initial test run, current grep evidence, `tests/test_db_public_boundary.py`.
  - Verify: boundary test fails for current public `get_db` leaks before implementation.
  - Evidence: record command output and current offenders.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - Health boundary moved behind DB interface.
  - Covers: `db/health.py`, `brain_service/health.py`, `service.py`.
  - Verify: service health tests pass and service no longer imports `get_db`.
  - Evidence: changed files and test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 3 - Scheduler and pending-index DB access moved behind DB interface.
  - Covers: `db/scheduled_events.py`, `scheduler.py`, `dispatcher/pending_index.py`.
  - Verify: dispatcher/scheduler tests pass and no raw scheduled-event DB operations remain outside `db`.
  - Evidence: changed files and test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 4 - Conversation-progress persistence moved behind DB interface.
  - Covers: `db/conversation_progress.py`, `conversation_progress/repository.py`.
  - Verify: conversation-progress tests pass and repository no longer imports `db._client.get_db`.
  - Evidence: changed files and test output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 5 - Public facade cleanup complete.
  - Covers: `db/__init__.py`, `db/_client.py`, `db/README.md`.
  - Verify: `get_db` is absent from public facade and docs identify it as private.
  - Evidence: changed files and focused test output.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 6 - Affected scripts moved behind public DB interfaces.
  - Covers: affected scripts listed in `Context`, any narrow DB functions they require, and script boundary tests.
  - Verify: scripts do not import `get_db`, `db._client`, Motor, PyMongo, raw collection handles, or raw Mongo operation tokens.
  - Evidence: changed files, script-boundary test output, and focused script test output.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 7 - Final verification complete.
  - Covers: final static greps, focused tests, broader smoke, and script boundary evidence.
  - Verify: all commands in `Verification` pass.
  - Evidence: final command output and static grep results.
  - Handoff: plan complete.
  - Sign-off: `<agent/date>` after evidence is recorded.

## Verification

### Static Greps

Run after implementation. The greps below are exhaustive across the relevant trees, not scoped to known-offender modules — any new leak introduced anywhere outside `src/kazusa_ai_chatbot/db/` must surface here.

```powershell
# (1) get_db must only appear inside the DB package.
rg "\bget_db\b" src\kazusa_ai_chatbot -g "*.py"

# (2) get_db must be absent from the public facade.
rg "\bget_db\b" src\kazusa_ai_chatbot\db\__init__.py

# (3) No production code outside src/kazusa_ai_chatbot/db/ may import any symbol from db._client.
rg "from kazusa_ai_chatbot\.db\._client\b|kazusa_ai_chatbot\.db\._client\." src\kazusa_ai_chatbot src\scripts -g "*.py" -g "!src/kazusa_ai_chatbot/db/**"

# (4) No production code outside src/kazusa_ai_chatbot/db/ may import motor or pymongo (including pymongo.errors).
rg "^\s*(from|import)\s+(motor|pymongo)(\b|\.)" src\kazusa_ai_chatbot src\scripts -g "*.py" -g "!src/kazusa_ai_chatbot/db/**"

# (5) No private (underscore-prefixed) name may be imported from any db.* submodule outside the DB package.
rg "from kazusa_ai_chatbot\.db\.[A-Za-z0-9_]+ import [^#\n]*\b_[A-Za-z0-9_]+" src\kazusa_ai_chatbot src\scripts -g "*.py" -g "!src/kazusa_ai_chatbot/db/**"

# (6) Raw Mongo operation tokens must not appear in any non-db production module or any script.
rg "db\.client|\.admin\.command|\.insert_one\(|\.insert_many\(|\.update_one\(|\.update_many\(|\.replace_one\(|\.find\(|\.find_one\(|\.delete_one\(|\.delete_many\(|\.aggregate\(|\.count_documents\(|\.distinct\(|\.bulk_write\(|\.create_index\(|\.list_indexes\(|\.list_collection_names\(|\.drop\(|db\[" src\kazusa_ai_chatbot src\scripts -g "*.py" -g "!src/kazusa_ai_chatbot/db/**"
```

Expected interpretation:

- Grep (1): `get_db` may appear only under `src/kazusa_ai_chatbot/db/`, including `db/_client.py` and DB-owned implementation modules.
- Grep (2): must return no matches — `get_db` is absent from `src/kazusa_ai_chatbot/db/__init__.py`.
- Grep (3): must return no matches.
- Grep (4): must return no matches. The DB package is the sole owner of `motor` and `pymongo`.
- Grep (5): must return no matches. Private DB-internal helpers stay inside the DB package.
- Grep (6): must return no matches. Any false positive (e.g., a non-Mongo `.find(` on a string) must be removed by renaming or by using an explicit static-test allowlist file referenced from `tests/test_db_public_boundary.py`. Do not add inline `noqa`-style suppression comments.

If a script not currently listed in `Context` is discovered by grep (3), (4), (5), or (6), either add it to the affected script list and update it under this plan, or stop and report a scope blocker before continuing.

### Tests

Run focused tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_db.py tests\test_db_public_boundary.py -q
venv\Scripts\python.exe -m pytest tests\test_script_db_boundary.py -q
venv\Scripts\python.exe -m pytest tests\test_service_health.py tests\test_reflection_cycle_stage1c_service.py -q
venv\Scripts\python.exe -m pytest tests\test_dispatcher.py tests\test_rag_recall_agent.py -q
venv\Scripts\python.exe -m pytest tests\test_conversation_episode_state.py tests\test_conversation_progress_runtime.py tests\test_conversation_progress_flow.py -q
```

Run broader smoke:

```powershell
venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_runtime_adapter_registration.py tests\test_rag_cache2_persistent.py -q
```

If live LLM or real database tests are not run, record that explicitly in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `get_db` still exists in `src/kazusa_ai_chatbot/db/_client.py`.
- `get_db` is documented as private DB-package infrastructure.
- `get_db` is not imported or exported by `src/kazusa_ai_chatbot/db/__init__.py`.
- Production application code outside `src/kazusa_ai_chatbot/db/` does not import or call `get_db`.
- No file under `src/kazusa_ai_chatbot/` outside `src/kazusa_ai_chatbot/db/` and no file under `src/scripts/` imports any symbol from `kazusa_ai_chatbot.db._client`.
- No file under `src/kazusa_ai_chatbot/` outside `src/kazusa_ai_chatbot/db/` and no file under `src/scripts/` imports any underscore-prefixed name from any `kazusa_ai_chatbot.db.*` submodule. The previously-leaking names `_embedding_source_text`, `_semantic_text`, and `memory_embedding_source_text` are no longer referenced from script callers.
- No file under `src/kazusa_ai_chatbot/` outside `src/kazusa_ai_chatbot/db/` and no file under `src/scripts/` imports `motor`, `pymongo`, or any submodule of either (including `pymongo.errors`).
- No file under `src/kazusa_ai_chatbot/` outside `src/kazusa_ai_chatbot/db/` and no file under `src/scripts/` contains any of the forbidden raw-backend tokens enumerated in `Mandatory Rules`.
- Service health uses a semantic DB health function.
- Scheduler and pending-index runtime code use semantic scheduled-event DB functions.
- Conversation-progress runtime code no longer performs raw DB operations outside the DB package and no longer imports `pymongo.errors`.
- Affected `src/scripts` callers use public `kazusa_ai_chatbot.db` functions only.
- No production database content migration or repair script is introduced.
- Static boundary tests enforce the rules above by scanning every file under `src/kazusa_ai_chatbot/` outside `src/kazusa_ai_chatbot/db/` and every file under `src/scripts/`, not just a hand-listed set of known offenders.
- Existing focused runtime tests pass.
- Script boundary tests pass.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Scheduler status transitions change | Use one semantic function per current status update | Dispatcher and scheduler-adjacent tests |
| Pending index rebuild changes dedup state | Return the same pending event documents in the same effective shape | Pending-index and dispatcher tests |
| Conversation-progress guarded upsert changes race behavior | Move existing logic into DB package without changing filters or return semantics | Conversation-progress repository/runtime tests |
| Tests keep relying on raw DB handles | Update non-DB tests to patch semantic DB functions | Static boundary test |
| `get_db` remains public through lazy export | Ban compatibility `__getattr__` and assert `db.__all__` excludes `get_db` | Boundary tests and greps |
| Script behavior changes while removing raw DB access | Add narrow public DB functions that preserve current query/update behavior; run focused script tests | Script tests and before/after output checks |
| Affected script remains on `get_db` | Static script boundary test and broad `src/scripts` grep | Script boundary test and grep |
| Agent adds a data migration anyway | Explicitly forbid production content mutation and migration scripts | Plan review and static change-surface check |
| Private DB submodule symbols (e.g., `_embedding_source_text`, `_semantic_text`, `memory_embedding_source_text`) keep leaking through underscore-prefixed imports outside the DB package | Ban all underscore-prefixed imports from any `db.*` submodule outside `src/kazusa_ai_chatbot/db/` and route the snapshot script through a public DB function | Boundary grep (5) and `tests/test_db_public_boundary.py` |
| `motor` / `pymongo` imports leak into runtime modules (e.g., `pymongo.errors.DuplicateKeyError`) | Ban backend driver imports outside the DB package and surface DB races via semantic return values, not driver exceptions | Boundary grep (4) and `tests/test_db_public_boundary.py` |
| Boundary test only covers a hand-listed set of files and misses a new offender added later | Boundary tests must enumerate every `.py` under `src/kazusa_ai_chatbot/` outside `db/` and every `.py` under `src/scripts/` rather than a fixed allow/deny list | Boundary test code review and grep (1)–(6) |

## Execution Evidence

- Baseline focused tests:
- Baseline boundary grep:
- Stage 1 boundary test result:
- Stage 2 health result:
- Stage 3 scheduler/pending-index result:
- Stage 4 conversation-progress result:
- Stage 5 public facade result:
- Stage 6 script boundary result:
- Stage 7 final static grep result:
- Broader smoke result:
- Skipped verification, if any:
