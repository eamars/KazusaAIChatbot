# get_db private boundary plan

## Summary

- Goal: Keep `get_db` as the internal DB-client helper, remove it from the public `kazusa_ai_chatbot.db` facade, and route runtime/application database access through semantic DB interfaces.
- Plan class: medium
- Status: draft
- Mandatory skills: `py-style`, `test-style-and-execution`, `local-llm-architecture`.
- Overall cutover strategy: compatible for runtime behavior; incompatible only for unsupported public imports of `kazusa_ai_chatbot.db.get_db`.
- Highest-risk areas: service health, scheduler event status transitions, pending-task index rebuild, conversation-progress guarded upsert semantics, and tests that currently patch raw DB handles.
- Acceptance criteria: production application code outside `src/kazusa_ai_chatbot/db/` no longer imports or calls `get_db`, `get_db` is absent from `db.__init__` and `db.__all__`, and existing runtime behavior remains unchanged.

## Context

The DB package already contains most raw MongoDB access inside `src/kazusa_ai_chatbot/db/`, but `get_db` is still re-exported from `kazusa_ai_chatbot.db`. That makes the backend handle part of the public application API and encourages callers to work with Mongo collections directly.

Current runtime/application leaks found before this plan:

- `src/kazusa_ai_chatbot/service.py` imports `get_db` only to pass it into health construction.
- `src/kazusa_ai_chatbot/brain_service/health.py` calls `db.client.admin.command("ping")`.
- `src/kazusa_ai_chatbot/scheduler.py` performs direct `scheduled_events` inserts, finds, and status updates.
- `src/kazusa_ai_chatbot/dispatcher/pending_index.py` rebuilds from `db.scheduled_events.find(...)`.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py` imports `kazusa_ai_chatbot.db._client.get_db` and performs raw collection operations.
- `src/kazusa_ai_chatbot/db/__init__.py` imports and exports `get_db`.

This plan is not a SQL migration. It only makes Mongo an implementation detail of the DB package. Future backend replacement requires a separate plan after this boundary is enforced.

`src/scripts/` contains backend-specific maintenance and export tools that also import `get_db`. This plan treats those scripts as deferred admin tooling, not production application code. Do not silently migrate scripts to `db._client.get_db`; a later admin-interface plan must decide which scripts remain supported and which narrow DB admin interfaces they need.

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
- Do not add a compatibility `__getattr__`, alias, re-export, or fallback that keeps `kazusa_ai_chatbot.db.get_db` working.
- Do not import `kazusa_ai_chatbot.db._client.get_db` outside `src/kazusa_ai_chatbot/db/`.
- Do not move raw Mongo collection, query, projection, aggregation, or update details into application modules.
- Do not change DB schemas, collection names, indexes, write shapes, scheduling semantics, cache invalidation, prompts, graph nodes, LLM call count, or endpoint schemas.
- Do not change `src/scripts/` in this plan except to record current out-of-scope imports in evidence if a verification grep includes them.
- If a caller needs a DB capability, add a narrow semantic function in `src/kazusa_ai_chatbot/db/` and call that function from application code.
- If a needed semantic DB function would require changing behavior or data shape, stop and report the blocker instead of expanding scope.

## Must Do

- Add static boundary tests before implementation.
- Add a public DB health interface so service health no longer receives a raw DB handle.
- Move scheduler persistence operations behind `kazusa_ai_chatbot.db.scheduled_events` semantic functions.
- Move pending-index rebuild reads behind a DB semantic function.
- Move conversation-progress raw persistence calls into `src/kazusa_ai_chatbot/db/conversation_progress.py`.
- Remove `get_db` from the public `kazusa_ai_chatbot.db` facade.
- Update runtime/application imports and tests to use semantic DB interfaces.
- Preserve all runtime behavior and currently tested request/response behavior.
- Record command output and grep evidence before marking stages complete.

## Deferred

- Do not implement SQL, an ORM, a repository base class, or a backend plugin system.
- Do not change the concrete Mongo implementation inside `src/kazusa_ai_chatbot/db/`.
- Do not rename `get_db` in `db/_client.py`.
- Do not remove `close_db`, `db_bootstrap`, embedding helpers, or vector-index helpers from the public DB facade in this plan.
- Do not migrate `src/scripts/` admin tooling to new DB admin interfaces in this plan.
- Do not change tests that intentionally validate DB-package internals, except where needed to patch `db._client.get_db` directly inside DB-layer tests.
- Do not broaden static boundary tests to fail on `src/scripts/` until a script/admin-tooling plan exists.

## Cutover Policy

| Area | Policy | Instruction |
|---|---|---|
| Public DB facade | compatible except `get_db` import | Remove only the raw DB handle export; keep existing semantic DB exports. |
| Runtime application behavior | compatible | Scheduler, health, progress, chat, RAG, and reflection behavior must remain unchanged. |
| DB implementation | compatible | Mongo remains the concrete backend behind the DB package. |
| Tests | compatible | Update tests to assert interfaces, not raw handles, outside DB-layer tests. |
| Scripts/admin tools | deferred | Do not treat script breakage as handled by this plan; create a later plan. |
| Deployment | compatible | No Docker, environment variable, or startup command changes. |

## Agent Autonomy Boundaries

- The target ownership boundary is `src/kazusa_ai_chatbot/db/` as the only package allowed to own raw database handles and backend query language.
- The agent may add narrow semantic DB functions only for capabilities already used by current runtime code.
- The agent must not invent generic repositories, query builders, or backend abstraction frameworks.
- The agent must not use string-built dynamic DB operations in application code to work around the boundary.
- The agent must keep compatibility wrappers only when they do not re-expose `get_db` or raw collection handles.
- The agent must treat any edit outside `src/kazusa_ai_chatbot/db/`, `src/kazusa_ai_chatbot/service.py`, `src/kazusa_ai_chatbot/brain_service/health.py`, `src/kazusa_ai_chatbot/scheduler.py`, `src/kazusa_ai_chatbot/dispatcher/pending_index.py`, `src/kazusa_ai_chatbot/conversation_progress/repository.py`, and focused tests as out of scope unless this plan names it.
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

New public or package-level semantic capabilities are added only where current runtime code already needs them.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Private helper name | Keep `get_db` in `_client.py` | The user explicitly does not want the function removed; DB internals can continue using it. |
| Public facade | Remove `get_db` from `db.__init__` and `__all__` | The public package should expose behavior, not backend handles. |
| Health check | Add DB-owned health function | Service health should ask whether DB is available, not ping Mongo directly. |
| Scheduler persistence | Move to `db.scheduled_events` semantic functions | Scheduler owns in-memory timers; DB package owns event document persistence. |
| Pending index rebuild | Read through scheduler-event DB interface | Dedup index should not know collection names or query shape. |
| Conversation progress persistence | DB package owns raw guarded upsert | Race-guard persistence is backend-specific and belongs behind DB interface. |
| Script policy | Defer `src/scripts` | Admin tools need a separate decision between supported semantic interfaces and backend-specific tooling. |

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
  - Static boundary tests for `get_db` and raw DB operations outside the DB package.

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
- Focused tests under `tests/`
  - Update patches from raw DB handles to semantic DB functions where appropriate.

### Keep

- `src/kazusa_ai_chatbot/db/_client.py:get_db`
- Existing DB schemas and collection names.
- Existing endpoint schemas and service entrypoints.
- Existing scheduler, dispatcher, RAG, reflection, and consolidation behavior.
- `src/scripts/*` behavior and imports for this plan, except as explicitly documented in evidence.

## Implementation Order

1. Baseline and boundary inventory.
   - Run focused tests before edits.
   - Record current `get_db` grep output.
2. Add failing boundary tests.
   - Add `tests/test_db_public_boundary.py`.
   - Assert `get_db` is not exported by `kazusa_ai_chatbot.db`.
   - Assert production code outside `src/kazusa_ai_chatbot/db/` does not import `get_db` or `db._client`.
   - Assert selected runtime modules do not contain raw Mongo operation tokens.
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
6. Remove public `get_db` export and document privacy.
   - Update `db.__init__`, `_client.py`, and `db/README.md`.
   - Do not add compatibility aliases.
7. Run final verification.
   - Run static greps, focused runtime tests, DB tests, and broader smoke.
   - Record skipped script/admin-tooling migration explicitly.

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
- [ ] Stage 5 - Public facade cleanup and final verification complete.
  - Covers: `db/__init__.py`, `db/_client.py`, `db/README.md`, final greps.
  - Verify: all commands in `Verification` pass or deferred script findings are recorded.
  - Evidence: final command output and static grep results.
  - Handoff: plan complete; script/admin DB boundary requires a separate plan.
  - Sign-off: `<agent/date>` after evidence is recorded.

## Verification

### Static Greps

Run after implementation:

```powershell
rg "get_db" src\kazusa_ai_chatbot -g "*.py"
rg "get_db" src\kazusa_ai_chatbot\db\__init__.py
rg "db\.client|\.admin\.command|\.insert_one\(|\.update_one\(|\.find\(|\.aggregate\(|db\[" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\brain_service src\kazusa_ai_chatbot\scheduler.py src\kazusa_ai_chatbot\dispatcher src\kazusa_ai_chatbot\conversation_progress -g "*.py"
```

Expected interpretation:

- `get_db` may appear under `src/kazusa_ai_chatbot/db/` only, including `db/_client.py` and DB-owned implementation modules.
- The second grep must return no matches; `get_db` must not appear in `src/kazusa_ai_chatbot/db/__init__.py`.
- Raw Mongo operation tokens must not appear in the named runtime/application modules, except false positives that are documented and justified by static-test allowlists.
- `src/scripts` is intentionally not included in these greps for this plan.

### Tests

Run focused tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_db.py tests\test_db_public_boundary.py -q
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
- Service health uses a semantic DB health function.
- Scheduler and pending-index runtime code use semantic scheduled-event DB functions.
- Conversation-progress runtime code no longer performs raw DB operations outside the DB package.
- Static boundary tests enforce the new rule.
- Existing focused runtime tests pass.
- `src/scripts` direct DB usage is explicitly deferred, not silently treated as solved.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Scheduler status transitions change | Use one semantic function per current status update | Dispatcher and scheduler-adjacent tests |
| Pending index rebuild changes dedup state | Return the same pending event documents in the same effective shape | Pending-index and dispatcher tests |
| Conversation-progress guarded upsert changes race behavior | Move existing logic into DB package without changing filters or return semantics | Conversation-progress repository/runtime tests |
| Tests keep relying on raw DB handles | Update non-DB tests to patch semantic DB functions | Static boundary test |
| `get_db` remains public through lazy export | Ban compatibility `__getattr__` and assert `db.__all__` excludes `get_db` | Boundary tests and greps |
| Script imports break later unnoticed | Record script/admin tooling as deferred and require a separate plan | Execution evidence |

## Execution Evidence

- Baseline focused tests:
- Baseline boundary grep:
- Stage 1 boundary test result:
- Stage 2 health result:
- Stage 3 scheduler/pending-index result:
- Stage 4 conversation-progress result:
- Stage 5 final static grep result:
- Broader smoke result:
- Skipped verification, if any:
