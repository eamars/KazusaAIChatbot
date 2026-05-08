# rag active turn conversation row exclusion plan

## Summary

- Goal: Prevent RAG conversation evidence from projecting the active user turn when `platform_message_id` is missing, without adding a new persisted service-generated identifier.
- Plan class: medium
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`
- Overall cutover strategy: forward-only internal use of MongoDB's existing inserted row `_id`; no data migration, no new conversation-history field, and no prompt changes.
- Highest-risk areas: changing pre-graph persistence ordering, leaking internal row IDs into LLM prompts, losing collapsed-turn coverage, widening retrieval behavior, and replacing exact identity matching with fuzzy matching.
- Acceptance criteria: active-turn user rows are excluded from conversation evidence by existing conversation row identity before projection, including rows with blank `platform_message_id`; existing platform-message-ID exclusion remains as fallback.

## Context

The completed `rag_current_turn_exclusion_plan.md` fixed active-turn self-hits only when both sides had comparable platform message IDs. Its accepted boundary was intentionally narrow: platform message IDs only, no body-text fallback, and no exclusion for rows without `platform_message_id`.

The new incident shows the remaining gap:

```text
conversation_evidence_agent output:
selected_summary=蚝爹油 at 2026-05-08 11:58:
南方小岛的香草荚，用在熔岩巧克力蛋糕或展示甜点里可以让蛋糕变得更好吃！
```

The absence of `conversation_evidence_agent active-turn rows excluded` in the log indicates that the projection filter did not recognize the returned row as active-turn. The likely immediate cause is a missing or non-matching `platform_message_id` on the saved user row.

The first proposal added a new brain-local source ID. A follow-up audit shows that a stronger existing identity is already available: every saved `conversation_history` row receives a MongoDB `_id` from `insert_one`. The active user row is persisted before graph invocation, and collapsed rows are persisted before the survivor is processed. Therefore the fix can use the already assigned database row identity instead of assigning another service ID.

The system invariant is:

```text
conversation worker may retrieve current saved row
  -> conversation_evidence_agent must remove active-turn source rows
  -> evaluator/finalizer/cognition must never see active user input as historical evidence
```

This must remain deterministic. The LLM must not be asked to obey "do not cite current message" as a prompt instruction.

## Mandatory Skills

- `local-llm-architecture`: load before changing RAG context, helper-agent boundaries, prompt visibility, or response-path behavior.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Check `git status --short` before editing.
- Do not edit `.env`.
- Do not add a new persisted service-generated active-turn ID.
- Do not add a new `conversation_history` document field for this fix.
- Do not add body-text, timestamp, display-name, sequence-number, or fuzzy fallback matching for active-turn exclusion.
- Do not move user-message persistence after graph invocation or after RAG.
- Do not move the survivor user-message save before `chat_history_wide` and `chat_history_recent` are loaded; the active message must not enter normal chat history context.
- Do not change dropped or collapsed queued-message persistence semantics.
- Do not change conversation worker argument generation, judge prompts, tool query behavior, cache key policy, or cache invalidation policy.
- Do not include `active_turn_conversation_row_ids` in conversation helper cache keys; active-row filtering is a post-worker deterministic projection step.
- Do not let graph/RAG invocation begin until the active user row insert and its existing cache invalidation have both completed.
- Do not change RAG prompts, initializer prompts, evaluator/finalizer prompts, cognition prompts, or dialog prompts.
- Do not add a database migration, backfill, index, cleanup job, or collection rewrite.
- Do not expose Mongo `_id`, `conversation_row_id`, or `active_turn_conversation_row_ids` to LLM prompt projections.
- Do not add over-retrieval or replacement-row fill behavior. If all returned rows are active-turn rows, the capability must return unresolved/empty.
- Preserve the existing platform-message-ID fallback for rows and contexts that have platform IDs.

## Must Do

- Change `src/kazusa_ai_chatbot/db/conversation.py::save_conversation(...)` to:
  - Capture `insert_result = await db.conversation_history.insert_one(doc)` and immediately bind `inserted_id_str = str(insert_result.inserted_id)`.
  - Then run the existing Cache2 invalidation. Treat invalidation as best-effort relative to the identity boundary: catch a narrow `Exception` from `invalidate(...)` only, log it at WARN, and still return `inserted_id_str`. The identity caller cannot recover a missing row ID, but a stale in-memory cache is recoverable on the next event. Do not widen this catch to cover the insert itself — insert failure must continue to propagate.
  - Return `inserted_id_str` to the caller.
- Preserve existing callers that ignore the return value.
- Update persistence protocol/type aliases that currently require `Awaitable[None]` so injected save functions may return `str | None`. The new contract is: a non-empty string means the row was committed; `None` means the row was not committed.
- Update `src/kazusa_ai_chatbot/brain_service/intake.py::save_user_message_from_item(...)` to return the inserted row ID string when the save succeeds, or `None` when the existing exception path handles a persistence failure. Keep the existing narrow `except Exception` shape; do not split it.
- Update `src/kazusa_ai_chatbot/service.py::_save_user_message_from_item(...)` to return that row ID.
- Add a local holder on `QueuedChatItem` for the already assigned conversation row ID, for example `conversation_row_id: str = ""`. The default empty string means "row was not committed yet or save failed"; never compare against this default in active-turn matching.
- In `_persist_collapsed_queued_chat_item(...)`, set the collapsed item's `conversation_row_id` from `_save_user_message_from_item(...)` when available. If `None` is returned, leave the holder at its default empty string and emit one WARN log line so production telemetry surfaces the failure.
- In `_process_queued_chat_item(...)`, save the survivor user row before constructing `initial_state`, set `item.conversation_row_id`, then build active-turn row IDs from the survivor and collapsed items. If the survivor save returns `None`, log WARN and continue: the active-turn list omits the empty row ID, the platform-message-ID fallback still applies, and the graph still runs.
- Keep `_process_queued_chat_item(...)` history loading before the survivor save, so `chat_history_wide` and `chat_history_recent` keep their current active-message exclusion behavior.
- Preserve `_chat_input_worker(...)` ordering: drop loop, then collapse loop, then survivor processing — collapsed-row IDs must be assigned before the survivor builds `active_turn_conversation_row_ids`. This order is load-bearing for the fix; flag it in a comment so future refactors do not reorder these phases.
- Add a helper in `brain_service/intake.py` that returns deduplicated `active_turn_conversation_row_ids` from the survivor plus collapsed items. Skip empty strings explicitly so a save failure does not pollute the active-turn list with `""`.
- Add a service wrapper mirroring `_active_turn_platform_message_ids(...)`.
- Add optional `active_turn_conversation_row_ids: list[str]` to graph state typing and persona supervisor schema typing.
- Pass `active_turn_conversation_row_ids` from service initial state into `stage_1_research(...)`, then into `call_rag_supervisor(...)` context.
- Add raw internal `conversation_row_id` projection in `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py` for:
  - `search_conversation`
  - `search_conversation_keyword`
  - `get_conversation`
- Derive `conversation_row_id` from existing Mongo `_id` only, using `str(message.get("_id", ""))`. Do not derive it from text, timestamp, platform IDs, or queue sequence. ObjectId is not a str — the cast must be applied at every projection site so deterministic equality with the active-turn list holds.
- Update `src/kazusa_ai_chatbot/rag/prompt_projection.py` to strip `conversation_row_id` at one choke point: define a module-level `_STRIPPED_RAW_KEYS = ("conversation_row_id",)` and remove these keys inside `_project_dict_for_llm(...)` before recursion into nested fields. Stripping must happen in the recursive dict projection, not at individual call sites, so future tool projections inherit the strip automatically.
- Update `conversation_evidence_agent` active-turn filtering so it excludes message rows by:
  - first: `row.conversation_row_id` in `context.active_turn_conversation_row_ids`
  - second: existing `row.platform_message_id` in `context.active_turn_platform_message_ids`
  - Both checks must skip empty strings. Treat `conversation_row_id == ""` and `platform_message_id == ""` as "no comparable identity on this side"; never match on a default-empty value.
- Apply the filter only to message-row workers:
  - `conversation_keyword_agent`
  - `conversation_search_agent`
  - `conversation_filter_agent`
- Keep `conversation_aggregate_agent` unchanged.
- Log active-turn exclusions with counts split by `conversation_row_id` and `platform_message_id`, without logging the ID values at INFO.
- Add deterministic tests for save return value, queue handoff, retrieval raw projection, prompt projection stripping, and capability-level filtering.

## Race Condition Review

- In-process queue worker creation is not a practical race in the current asyncio service because `_ensure_chat_input_worker_started()` has no `await`; two endpoint tasks cannot interleave inside the check/set block on the same event loop.
- The queue worker processes dropped, collapsed, and surviving items sequentially in one coroutine. Collapsed-row IDs can be set before the survivor graph invocation without cross-task mutation. The survivor reads `item.collapsed_items`, which holds the same Python references that `_persist_collapsed_queued_chat_item(...)` mutates, so `conversation_row_id` propagation is by reference and does not require a queue-level handoff.
- The survivor row must be saved after history loading and before graph invocation. This preserves current chat-history behavior while guaranteeing RAG receives the active row ID.
- Mongo insert visibility is sufficient only after `await insert_one(...)` returns. The plan must not derive the row ID before the insert has completed. `insert_result.inserted_id` is the canonical source — do not read `doc["_id"]` even though `pymongo` mutates the doc in place, because future driver behavior is not part of this contract.
- Cache2 invalidation today is purely in-memory dict mutation (see `cache2_runtime.RAGCache2Runtime.invalidate(...)`), so the invalidation step itself does not introduce IO latency between insert and the row-ID return. Identity capture is therefore strictly stronger than cache invalidation: if invalidate raises (programmer error or future regression), the row is already in the database and the row ID is the only thing standing between RAG and a self-citation. `save_conversation(...)` must protect identity capture — return the inserted ID even if invalidation raises, with a logged WARN — rather than dropping the ID and falling through to the failure path.
- The active row IDs must not be part of conversation helper cache keys. Cached worker results, if any, remain raw retrieval payloads; the top-level conversation evidence agent filters active rows after a cache hit or miss. This means a cache hit between turns cannot leak the prior turn's filter list — the projection happens after the cache layer, against the current turn's `active_turn_conversation_row_ids`.
- A failed survivor save (returning `None`) must not be silent. The active-turn list will omit the row, RAG retrieval may surface it, and the only fallback is the platform-message-ID match — which is exactly the scenario the original incident shows is fragile. The intake/service wrappers must emit a WARN log so the operator can correlate a self-citation incident with a save failure post hoc.
- Multi-process duplicate delivery is outside this fix. If the same user payload is inserted twice by two independent service processes without platform message IDs, each process can exclude its own active row by Mongo `_id`; deduplicating a separate duplicate row requires adapter/message idempotency and is deferred.
- Same-process duplicate delivery (adapter retry of the same `/chat` call) is handled by the existing queue collapse policy when `platform_message_id` matches; when it does not, the second item is enqueued, persisted, and processed normally. Each survivor sees its own row's `_id` in the active-turn list, and the prior turn's row is legitimately historical evidence by the time the second turn runs RAG.

## Deferred

- Do not backfill or rewrite historical `conversation_history` rows.
- Do not add DB indexes for `_id` or any active-turn field.
- Do not remove `active_turn_platform_message_ids`.
- Do not remove the legacy `exclude_current_question` context key.
- Do not add synthetic `platform_message_id` generation to debug adapters in this fix. The row-ID fix covers debug traffic because Mongo `_id` exists even when adapter message IDs do not.
- Do not change cache storage or cache invalidation for conversation helper agents.

## Cutover Policy

- This is a forward-only runtime fix.
- No migration is required because MongoDB already assigns `_id` to every row.
- Existing historical rows remain compatible because retrieval rows can derive `conversation_row_id` from `_id` at read time.
- If row ID capture fails because persistence failed, retain current behavior: the graph continues through the existing exception-handled path, and active-turn filtering falls back to platform message IDs when present.
- If a worker returns only active-turn rows, the top-level conversation evidence capability returns unresolved with `missing_context=["conversation_evidence"]`.

## Agent Autonomy Boundaries

- The implementation agent may rename helper functions if the names are clearer, but must preserve the contracts in this plan.
- The implementation agent may update focused tests to match exact local fixtures.
- The implementation agent must not introduce a replacement identity strategy.
- The implementation agent must not add prompt instructions as a substitute for deterministic filtering.
- The implementation agent must not broaden the fix into RAG initializer routing, continuation planning, memory evidence retrieval, cognition stance, or dialog wording.

## Target State

For a normal active turn:

```text
enqueue request
  -> QueuedChatItem exists with no conversation_row_id yet
  -> save active user row before graph
  -> Mongo insert_one assigns _id
  -> service stores str(inserted_id) on the queued item
  -> initial_state carries active_turn_conversation_row_ids
  -> RAG context carries active_turn_conversation_row_ids
  -> conversation worker retrieves matching current row
  -> conversation_evidence_agent removes it before projection/finalizer/cognition
```

For collapsed turns:

```text
collapsed item is saved before survivor processing
  -> collapsed item stores str(inserted_id)
  -> survivor initial_state includes survivor + collapsed conversation row IDs
  -> all current source rows are excluded from conversation evidence
```

For adapter/platform rows with good platform IDs:

```text
row ID exclusion runs first
  -> platform_message_id exclusion remains fallback
```

## Design Decisions

- Use MongoDB `_id` because it is already assigned by the existing persistence backend and uniquely identifies the exact row that retrieval returns.
- Surface the row identity to deterministic Python as `conversation_row_id` rather than exposing raw `_id` in public-ish helper result shapes.
- Store only the string form of `inserted_id` in runtime state because ObjectId instances are not needed outside persistence and are less convenient in test fixtures.
- Keep row identity out of LLM-facing prompt projections; it is a deterministic filter key, not semantic evidence.
- Keep platform-message-ID exclusion because it already protects rows in fixtures or alternate retrieval paths that do not expose `conversation_row_id`.
- Do not use timestamp matching. Timestamp plus platform/channel/user is existing information but is not exact row identity and can false-positive under coarse adapter timestamps or duplicate messages.
- Do not use `QueuedChatItem.sequence`. It is existing process-local information, but it is not persisted and cannot be matched to retrieval rows.

## Change Surface

- `src/kazusa_ai_chatbot/db/conversation.py`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/brain_service/intake.py`
- `src/kazusa_ai_chatbot/chat_input_queue.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/state.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py`
- `src/kazusa_ai_chatbot/rag/prompt_projection.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
- Focused tests under `tests/`

## Implementation Order

1. Load mandatory skills and reread this plan.
2. Run `git status --short`.
3. Update `save_conversation(...)` to return `str(insert_result.inserted_id)` only after the existing Cache2 invalidation await has completed.
4. Update save protocol/type aliases and service/intake wrappers to propagate `str | None`.
5. Add the `QueuedChatItem.conversation_row_id` runtime holder.
6. Capture collapsed item row IDs in `_persist_collapsed_queued_chat_item(...)`.
7. Move the survivor save in `_process_queued_chat_item(...)` only far enough upward to capture the row ID before `initial_state`; keep history loading before the save.
8. Add `active_turn_conversation_row_ids` helpers and include the value in initial graph state and RAG context.
9. Add `conversation_row_id` to raw conversation retrieval results.
10. Strip `conversation_row_id` from LLM-facing prompt projections.
11. Update `conversation_evidence_agent` filtering and logging.
12. Add and update focused deterministic tests.
13. Run the focused test suite listed in `Verification`.
14. Record execution evidence if the plan is later executed.

## Progress Checklist

- [x] `save_conversation(...)` returns inserted row ID.
- [x] Intake/service save wrappers propagate inserted row ID without changing failure behavior.
- [x] Queue items hold the existing conversation row ID after save.
- [x] Survivor and collapsed active-turn conversation row IDs enter graph state.
- [x] Persona supervisor passes active row IDs into RAG context.
- [x] Conversation retrieval raw results include `conversation_row_id`.
- [x] Prompt projection strips `conversation_row_id` before LLM prompts.
- [x] Conversation evidence filters by row ID first and platform message ID second.
- [x] Active-turn exclusion logs show reason counts without ID values.
- [x] Deterministic tests pass.

## Verification

Run focused tests after implementation:

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_db.py `
  tests\test_service_input_queue.py `
  tests\test_rag_phase3_capability_agents.py `
  -q
```

If the full files are too broad for quick iteration, start with the added or changed tests only, then run the file-level command above before final sign-off.

Required test coverage:

- `save_conversation(...)` returns a non-empty string row ID after insert.
- `save_conversation(...)` returns the row ID after the existing Cache2 invalidation completes successfully.
- `save_conversation(...)` still returns the row ID and emits a WARN when a patched Cache2 runtime raises inside `invalidate(...)`; insert failure (raised before invalidation) still propagates.
- Existing callers may await `save_conversation(...)` and ignore the return value.
- `brain_service.intake.save_user_message_from_item(...)` returns the injected save function's string row ID, and returns `None` plus a logged WARN when the injected save function raises.
- Collapsed queued messages store row IDs and the survivor state includes both survivor and collapsed row IDs in arrival order without duplicates and without empty-string entries from failed saves.
- `_chat_input_worker(...)` ordering test: collapsed-item `conversation_row_id` is set before `_process_queued_chat_item(survivor)` reads `survivor.collapsed_items`.
- `memory_retrieval_tools` raw result rows include `conversation_row_id` derived from `str(message["_id"])`; the value type is `str`, not `ObjectId`.
- `project_tool_result_for_llm(...)` removes `conversation_row_id` from prompt-facing copies at every nesting depth covered by `_NESTED_LIST_FIELDS` and `_NESTED_OBJECT_FIELDS`.
- `conversation_evidence_agent` excludes a row whose `conversation_row_id` is active even when `platform_message_id` is blank.
- Existing platform-message-ID exclusion still excludes active rows when `conversation_row_id` is absent.
- Rows without active row ID and without matching platform message ID remain eligible evidence.
- Rows whose `conversation_row_id` is the empty string are NOT matched by an active-turn list that contains an empty string (defense against accidental default-value collisions).

## Acceptance Criteria

- The incident pattern cannot recur when the current user message has no `platform_message_id` but has been persisted before RAG.
- RAG finalizer and downstream cognition/dialog do not receive active-turn conversation evidence rows.
- No new persisted service-generated ID is added.
- No data migration or index change is required.
- No prompt-only instruction is used as the fix.
- Existing tests for the previous platform-message-ID exclusion still pass.

## Risks

- Moving the survivor save before `initial_state` creation can accidentally put the active row into history if moved too far upward. Mitigation: keep history retrieval before the save and test this ordering with a deterministic fixture that asserts the active row is not in `chat_history_wide`.
- Returning the row ID before cache invalidation completes can let graph/RAG run against stale helper-agent cache state. Mitigation: invoke invalidation before returning under normal flow; only fall through to "return ID with WARN" when invalidation raises. Test both paths: (1) normal happy path returns ID after invalidation, (2) patched runtime that raises in `invalidate(...)` still returns the ID and logs WARN.
- Internal row IDs can leak into LLM prompts if projection stripping is missed. Mitigation: strip at one recursive choke point in `_project_dict_for_llm(...)` rather than at every call site, and add a dedicated prompt-projection test that constructs nested rows under each `_NESTED_LIST_FIELDS` key (`messages`, `results`, `rows`, `conversation_evidence`) and asserts `conversation_row_id` is absent at every depth.
- Mock save functions in tests may return `None`. Mitigation: wrappers tolerate `None` and only include non-empty row IDs in active-turn context. Production-side mitigation: WARN log on `None` so the operator can distinguish "test mock" from "real save failure" in incident review.
- Some retrieval fixtures may not include `_id`. Mitigation: platform-message-ID fallback remains, and rows without either exact active identifier are not excluded; both filters explicitly skip empty-string identities to avoid false matches against default holders.
- ObjectId vs str equality drift: a retrieval site that forgets `str(...)` would compare `ObjectId("…")` against a string list and silently fail to filter. Mitigation: centralize the cast in the three tool projections plus a focused test that asserts `conversation_row_id` is `str` and equals the active-turn list element by `==`.
- A future refactor of `_chat_input_worker(...)` could reorder collapse handling after survivor processing, breaking the assumption that collapsed-row IDs are populated before the survivor reads them. Mitigation: comment the ordering invariant inline at the worker, and assert it in the queue-handoff test.

## Execution Evidence

- Implemented on 2026-05-08 by Codex.
- Plan lifecycle: moved from `active/bugfix/` to `archive/completed/bugfix/`.
- Scope note: `src/kazusa_ai_chatbot/db/schemas.py` did not need a code change because this fix adds no persisted `conversation_history` field.
- Pre-implementation focused tests failed as expected for the new row-ID contract:
  `venv\Scripts\python.exe -m pytest tests\test_db.py::test_save_conversation_returns_row_id_after_cache_invalidation tests\test_service_input_queue.py::test_active_turn_conversation_row_ids_skip_empty_and_dedupe tests\test_memory_retrieval_tools.py::test_search_conversation_delegates_to_vector_history_search tests\test_llm_time_payload_projection.py::test_tool_result_strips_conversation_row_id_from_nested_payloads tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_excludes_active_turn_row_id_hit -q`
- Post-implementation focused slice passed:
  `venv\Scripts\python.exe -m pytest tests\test_db.py::test_save_conversation_returns_row_id_after_cache_invalidation tests\test_service_input_queue.py::test_active_turn_conversation_row_ids_skip_empty_and_dedupe tests\test_memory_retrieval_tools.py::test_search_conversation_delegates_to_vector_history_search tests\test_llm_time_payload_projection.py::test_tool_result_strips_conversation_row_id_from_nested_payloads tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_excludes_active_turn_row_id_hit -q`
  -> `5 passed`.
- Final deterministic verification command:
  `venv\Scripts\python.exe -m pytest tests\test_db.py tests\test_service_input_queue.py tests\test_rag_phase3_capability_agents.py tests\test_memory_retrieval_tools.py tests\test_llm_time_payload_projection.py -q`
  -> `137 passed, 13 deselected`.
- Queue-only rerun after test helper cleanup:
  `venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py -q`
  -> `25 passed`.
- Syntax check:
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\db\conversation.py src\kazusa_ai_chatbot\brain_service\intake.py src\kazusa_ai_chatbot\brain_service\post_turn.py src\kazusa_ai_chatbot\chat_input_queue.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\rag\memory_retrieval_tools.py src\kazusa_ai_chatbot\rag\prompt_projection.py src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py`
  -> passed.
- Whitespace check:
  `git diff --check`
  -> passed; Git reported only existing line-ending normalization warnings.
