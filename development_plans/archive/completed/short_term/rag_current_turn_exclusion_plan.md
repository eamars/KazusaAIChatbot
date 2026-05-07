# rag current turn exclusion plan

## Summary

- Goal: Prevent active-turn user messages from being projected as RAG conversation evidence for the same turn.
- Plan class: small
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`
- Overall cutover strategy: compatible code-only safety filter at the conversation evidence capability boundary.
- Highest-risk areas: expanding the fix into every retriever, hiding the rule in prompts, adding body-text fallback matching, and accidentally changing queued dropped/collapsed persistence.
- Acceptance criteria: current active-turn message rows do not appear in `selected_summary`, `projection_payload.summaries`, `evidence`, `resolved_refs`, RAG fact summaries, or final RAG projection; older eligible rows still project when already returned by the worker.

## Context

The observed failure was a conversation RAG self-hit:

```text
RAG2 projection output: answer="蚝爹油 at 2026-05-06 00:09: 是不是刚刚在 905393941 群有人欺负你了"
```

The active user message had already been saved to `conversation_history` before graph execution. A conversation worker retrieved that same active row, then `conversation_evidence_agent` projected it as historical evidence.

This plan intentionally narrows the fix to the smallest safety boundary:

```text
worker returns raw message rows
  -> conversation_evidence_agent removes active-turn rows
  -> conversation_evidence_agent builds canonical evidence projection
  -> RAG evaluator/finalizer/cognition only see filtered canonical evidence
```

This prevents false evidence. It does not attempt to improve recall quality by over-retrieving replacement rows.

## Discovery Evidence

Discovery was tightened against the current queue and service behavior before this plan was finalized.

Verified code paths:

- `src/kazusa_ai_chatbot/chat_input_queue.py`
  - `QueuedChatItem` already stores `combined_content` and `collapsed_items`.
  - `_append_collapsed_item(...)` appends later queued messages into the survivor and joins body text into `combined_content`.
  - Private same-scope follow-ups and addressed group follow-ups both use this collapsed-turn mechanism.
- `src/kazusa_ai_chatbot/service.py`
  - `_chat_input_worker(...)` persists `dequeued_turn.collapsed_items` before processing `dequeued_turn.next_item`.
  - `_process_queued_chat_item(...)` uses `item.combined_content or message_envelope["body_text"]` as graph `user_input`.
  - The graph state currently carries only the survivor `platform_message_id`.
- `tests/test_service_input_queue.py`
  - Existing tests prove private-message collapse, addressed-group collapse, and collapsed-message persistence ordering.

Discovery test command run:

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_service_input_queue.py::test_private_messages_same_scope_coalesce `
  tests\test_service_input_queue.py::test_addressed_group_followups_coalesce `
  tests\test_service_input_queue.py::test_worker_saves_collapsed_messages_before_graph -q
```

Discovery result: `3 passed`.

Important discovered fact: collapsed originals are saved before the survivor graph runs, and the graph answers the combined content. Therefore collapsed source message IDs must be treated as active-turn source IDs and excluded from conversation evidence projection for that response.

## Mandatory Skills

- `local-llm-architecture`: load before changing RAG boundary behavior or LLM-facing context.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not move active user-message insertion after RAG.
- Do not add a database visibility flag, migration, index, cleanup job, or backfill.
- Do not change dropped or collapsed queued-message persistence.
- Do not edit any prompt for this fix.
- Do not change worker generator, worker judge, retrieval tool, or database query behavior.
- Do not add over-retrieval in this plan.
- Do not bump conversation worker cache versions in this plan.
- Do not expose new current-turn identity fields to LLM prompt projection.
- Do not rely on `exclude_current_question` as an LLM instruction.
- Do not filter by body text alone.
- Use platform message IDs only for this minimum fix.
- If a platform message ID is missing from the active-turn source list or a retrieved row, do not exclude by fallback in this plan.
- Deduplicate active-turn source IDs while preserving first-seen order.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.

## Must Do

- Add active-turn source message IDs to graph state for the processed survivor message and any collapsed messages.
- Build active-turn source IDs as `[survivor.platform_message_id, *collapsed.platform_message_id]`, omitting blank IDs and deduplicating.
- Pass active-turn source IDs into the RAG runtime context.
- Filter active-turn message rows inside `conversation_evidence_agent` before `_message_projection(...)`.
- Apply the filter to message-row workers only:
  - `conversation_keyword_agent`
  - `conversation_search_agent`
  - `conversation_filter_agent`
- Leave `conversation_aggregate_agent` unchanged in this plan.
- Log when active-turn rows are removed from conversation evidence projection.
- Add deterministic tests for the capability-level filter and runtime context handoff.

## Deferred

- Do not add over-retrieve-and-fill behavior. This means the minimum fix may return no evidence when the active row displaced older eligible rows.
- Do not correct aggregate counts that include the active message.
- Do not remove the existing `exclude_current_question` key from prompt projection.
- Do not change `conversation_keyword_agent`, `conversation_search_agent`, or `conversation_filter_agent`.
- Do not change `memory_retrieval_tools.py` or `db/conversation.py`.
- Do not change helper-agent cache keys or policy versions.
- Do not add body/timestamp fallback matching for platforms without message IDs.
- Do not implement cross-group disclosure policy.
- Do not fix channel-id routing or target-channel scope.

## Cutover Policy

| Area | Policy | Instruction |
|---|---|---|
| Active user-message persistence | compatible | Keep save-before-graph behavior. |
| Dropped queued messages | compatible | No behavior change. |
| Collapsed queued messages | compatible | Keep persistence behavior; only carry their message IDs as active-turn source IDs. |
| Conversation workers | unchanged | Do not edit worker generator, judge, cache, or tool calls. |
| Conversation evidence capability | compatible safety filter | Filter active-turn message rows before capability projection. |
| Cache | unchanged | No cache version bump; cached worker results are filtered at projection time. |
| Prompt payloads | unchanged | Do not add new prompt-visible fields. |
| Database | unchanged | No schema or query changes. |

## Agent Autonomy Boundaries

- The implementation agent must preserve this plan's minimum-blast-radius scope.
- The agent must not convert this into a worker/tool over-retrieval fix.
- The agent must not add database fields or change persistence timing.
- The agent must not broaden filtering to non-RAG readers such as exports, relevance history loading, consolidation, or debug tooling.
- If active-turn source IDs cannot be carried without larger state changes, stop and report the blocker.
- If an existing test expects `exclude_current_question` to be prompt-visible, leave that behavior unchanged unless the test directly conflicts with this plan's safety filter.

## Target State

The target behavior is:

```text
active turn contains one or more source platform_message_id values
  -> RAG conversation worker may return those rows
  -> conversation_evidence_agent removes those rows before canonical projection
  -> if rows remain, project them normally
  -> if no rows remain, mark conversation evidence unresolved/empty
```

The result is a safety-first behavior: missing evidence is allowed; false self-evidence is not.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Fix location | `conversation_evidence_agent` projection boundary | One capability owns conversation evidence projection; filtering here protects downstream RAG without touching every retriever. |
| Unit of exclusion | Active-turn source messages | Collapsed turns may contain multiple user messages answered as one turn. |
| Matching method | Platform message ID set only | This is the verified, narrow identity already present in queue/service state; body/timestamp fallback would widen scope and risk false positives. |
| Persistence timing | Unchanged | Moving insert-after-RAG has hidden coupling with media descriptions, failure handling, and cache invalidation. |
| Worker/tool changes | Deferred | They are needed only for recall quality recovery, not for preventing false evidence. |
| Cache changes | Deferred | Projection-time filtering applies after worker cache reads, so stale raw worker payloads do not become projected evidence. |
| Prompt changes | Forbidden | This is an operational invariant, not an LLM instruction-following problem. |
| Aggregate worker | Deferred | Aggregate counts can be slightly skewed by the active message, but fixing that requires query-level changes outside the minimum echo fix. |

## Change Surface

Primary files:

- `src/kazusa_ai_chatbot/service.py`
  - Build `active_turn_platform_message_ids` from the surviving queued item and its `collapsed_items`.
  - Omit blank IDs and deduplicate while preserving order.
  - Add it to initial graph state.

- `src/kazusa_ai_chatbot/state.py`
  - Add optional `active_turn_platform_message_ids: list[str]`.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Add optional `active_turn_platform_message_ids: list[str]` to `GlobalPersonaState`.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Pass `active_turn_platform_message_ids` into `call_rag_supervisor(...)` context.

- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
  - Add local private helpers to identify and remove active-turn message rows.
  - Filter rows before `_message_projection(...)`.
  - Include `excluded_active_turn_rows=N` in logs when rows are removed.

Tests:

- `tests/test_rag_phase3_capability_agents.py`
  - Add focused tests for keyword/filter/semantic worker payloads returning an active-turn row.
  - Add a collapsed-turn test with two active source IDs.
  - Add a test proving an older non-active row still projects.

- `tests/test_service_input_queue.py`
  - Extend `test_worker_saves_collapsed_messages_before_graph` or add an adjacent focused test proving graph state receives active-turn IDs for survivor plus collapsed originals.

## Helper Contract

Keep helpers private to `conversation_evidence_agent.py`; do not create a new module for this minimum fix.

Required private behavior:

```python
def _active_turn_message_ids(context: dict[str, Any]) -> set[str]:
    """Return active-turn platform message IDs from capability context."""


def _is_active_turn_row(row: dict[str, Any], context: dict[str, Any]) -> bool:
    """Return whether a conversation row belongs to the active turn."""


def _filter_active_turn_rows(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Remove active-turn rows before evidence projection."""
```

Matching rules:

- If `row.platform_message_id` is present and appears in `active_turn_platform_message_ids`, exclude the row when platform/channel are compatible with the current context.
- If there are no active-turn IDs, do not exclude anything.
- If a row has no `platform_message_id`, do not exclude it.
- Do not compare or normalize body text for exclusion in this plan.
- Platform/channel compatibility is a guardrail only:
  - If a row has `platform`, it must match current context `platform`.
  - If a row has `platform_channel_id`, it must match current context `platform_channel_id`.
  - Missing row platform/channel fields do not block ID-based exclusion because older tests and worker fixtures may omit those fields.

## Implementation Order

1. Reread this plan and load mandatory skills.
2. Add `active_turn_platform_message_ids` to state types.
3. Populate `active_turn_platform_message_ids` in `service.py` from the survivor and collapsed items.
4. Pass `active_turn_platform_message_ids` through `stage_1_research` into RAG context.
5. Add private filtering helpers in `conversation_evidence_agent.py`.
6. Change `_projection_from_worker(...)` or its caller so message rows are filtered before `_message_projection(...)`.
7. Add logs for removed active-turn rows.
8. Add deterministic tests.
9. Run targeted tests and syntax validation.
10. Record execution evidence if implementation is performed under this plan.

## Progress Checklist

- [x] Plan accepted for implementation.
- [x] Mandatory skills loaded.
- [x] Active-turn source IDs added to state.
- [x] Service populates active-turn source IDs.
- [x] RAG context receives active-turn source IDs.
- [x] Conversation evidence projection filter implemented.
- [x] Keyword self-hit projection test added.
- [x] Filter/recent self-hit projection test added.
- [x] Semantic self-hit projection test added.
- [x] Collapsed active-turn projection test added.
- [x] Targeted tests passed.
- [x] Execution evidence recorded.

## Verification

Syntax validation:

```powershell
venv\Scripts\python.exe -m py_compile `
  src\kazusa_ai_chatbot\service.py `
  src\kazusa_ai_chatbot\state.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2.py `
  src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py
```

Targeted deterministic tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py -q
```

Required test evidence:

- A keyword worker result containing only an active-turn message becomes unresolved/empty at `conversation_evidence_agent`.
- A filter worker result containing only an active-turn message becomes unresolved/empty.
- A semantic worker result containing only an active-turn message becomes unresolved/empty.
- A mixed result removes the active row and keeps the older eligible row.
- A collapsed active turn removes multiple active source rows.
- A same-body older row with a different `platform_message_id` remains eligible.
- A row without `platform_message_id` remains eligible because fallback matching is out of scope.
- Service graph state contains survivor and collapsed source IDs in a collapsed turn.

Real LLM tests are not required for this fix because the changed behavior is deterministic projection filtering.

## Acceptance Criteria

- The original failure shape cannot produce this projected conversation evidence when that row is the active message:

```text
蚝爹油 at 2026-05-06 00:09: 是不是刚刚在 905393941 群有人欺负你了
```

- Active-turn rows are removed from `selected_summary`, `projection_payload.summaries`, `evidence`, and `resolved_refs`.
- If no rows remain after filtering, `conversation_evidence_agent` returns unresolved with `missing_context=["conversation_evidence"]`.
- If older eligible rows remain after filtering, they project normally.
- Collapsed active turns exclude all non-blank platform message IDs in the survivor plus collapsed source list.
- No conversation worker, retrieval tool, DB query, prompt, cache policy, or persistence timing is changed.

## Risks

- This minimum fix can turn a self-hit into no evidence, even when an older matching row exists but was displaced by worker `top_k`.
- Aggregate conversation counts may still include the active message.
- Existing worker LLM prompts may still see the legacy `exclude_current_question` context key; this plan does not remove it.
- If a platform omits `platform_message_id`, this minimum fix does not exclude that active row. This is accepted to avoid body-text fallback risk and keep blast radius minimal.

## LLM Call And Context Budget

- No new LLM calls.
- No prompt text changes.
- No new LLM-visible context fields.
- The safety decision is made in deterministic code after worker return and before capability projection.

## Data Migration

No data migration is required.

No database fields, indexes, row updates, backfills, re-embedding, or cleanup jobs are part of this plan.

## Operational Steps

- Deploy code normally.
- No manual database action is required.
- No cache purge is required because projection-time filtering applies after worker cache reads.

## Execution Evidence

Implementation completed under this plan on 2026-05-06.

Plan continuity:

- Reread this plan after marking plan acceptance complete and before implementation.

Files changed:

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/state.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
- `tests/test_rag_phase3_capability_agents.py`
- `tests/test_service_input_queue.py`
- `development_plans/rag_current_turn_exclusion_plan.md`

Implementation summary:

- Added `active_turn_platform_message_ids` to graph state.
- Populated active-turn IDs from the survivor queued item plus collapsed items, omitting blank IDs and deduplicating in arrival order.
- Passed active-turn IDs into RAG runtime context.
- Filtered active-turn message rows inside `conversation_evidence_agent` before canonical conversation evidence projection.
- Left conversation workers, retrieval tools, DB queries, prompts, cache policy, and persistence timing unchanged.

Verification commands:

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py tests\test_rag_phase3_capability_agents.py tests\test_service_input_queue.py
```

Result: passed.

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py -q
```

Result: `42 passed`.

```powershell
venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py::test_private_messages_same_scope_coalesce tests\test_service_input_queue.py::test_addressed_group_followups_coalesce tests\test_service_input_queue.py::test_worker_saves_collapsed_messages_before_graph -q
```

Result: `3 passed`.

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_excludes_active_turn_keyword_row tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_excludes_active_turn_filter_row tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_excludes_active_turn_semantic_row tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_excludes_collapsed_active_turn_rows tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_keeps_row_without_message_id tests\test_service_input_queue.py::test_worker_saves_collapsed_messages_before_graph -q
```

Result: `6 passed`.

Discovery evidence already collected:

- Ran targeted queue/service tests for collapse mechanics and persistence ordering.
- Command:

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_service_input_queue.py::test_private_messages_same_scope_coalesce `
  tests\test_service_input_queue.py::test_addressed_group_followups_coalesce `
  tests\test_service_input_queue.py::test_worker_saves_collapsed_messages_before_graph -q
```

- Result: `3 passed`.

Skipped verification:

- Real LLM tests were not run because this fix is deterministic projection filtering and does not change prompts or LLM routing behavior.

## Glossary

- Active-turn source message: a user message that is being answered as part of the current graph turn, including collapsed follow-up messages.
- Self-hit: retrieval returning an active-turn source message as evidence for that same turn.
- Conversation evidence capability: `conversation_evidence_agent`, the top-level RAG helper that turns worker results into canonical conversation evidence.
- Projection safety filter: deterministic removal of active-turn rows before canonical evidence summaries and refs are built.
