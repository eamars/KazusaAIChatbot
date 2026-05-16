# consolidator text dispatch decommission plan

## Summary

- Goal: remove consolidator-driven user-visible text output so future contact
  is decided by scheduled cognition and L2d/L3 speak, not by a background
  task-dispatch prompt.
- Plan class: high_risk_migration
- Status: in_progress
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `no-prepost-user-input`, `cjk-safety`,
  `database-data-pull`, `character-test`
- Overall cutover strategy: bigbang for runtime behavior; migration for
  existing pending scheduler rows.
- Highest-risk areas: lost legitimate delayed follow-up behavior, stale
  pending `send_message` events, self-cognition legacy delivery handoff,
  service startup coupling to dispatcher objects, and tests that still assert
  the old path.
- Acceptance criteria: no consolidator task-dispatch LLM exists, no
  consolidator persistence path schedules `send_message`, no self-cognition
  production handoff schedules prewritten text, the dispatcher cluster orphaned
  by these removals is deleted with no dead code or stale exports, pending
  `send_message` rows are cancelled, future follow-up uses
  `trigger_future_cognition`, and live smoke shows user-visible text still
  comes from cognition plus L3 text.

## Context

The latest QQ private-channel trace for user `673225019` showed the newest
character message came from a completed `scheduled_events.tool=send_message`
row. The row was correlated to the prior user message, and no
`self_cognition_action_attempts` row referenced it. The source was therefore
the legacy consolidator task-dispatch path:

```text
final dialog + future_promises
-> consolidator task-dispatch LLM
-> RawToolCall(tool="send_message")
-> TaskDispatcher
-> scheduled_events
-> adapter delivery
```

That path can create user-visible text after the main cognition turn has
finished. It violates the action-spec target invariant:

```text
No user-visible text is authored outside cognition and L3 text.
```

The original use case was instruction-like delayed contact, such as asking the
character to send a message to a group after ten minutes. The accepted
replacement is:

```text
user or self-cognition trigger
-> shared cognition / RAG / L2d
-> ActionSpec(kind="trigger_future_cognition")
-> scheduled future cognition slot
-> later self-cognition episode
-> shared cognition / RAG / L2d
-> optional ActionSpec(kind="speak")
-> L3 text
-> normal delivery
```

The character may decide not to speak during the later cognition cycle. That is
acceptable and preferred over a deterministic delayed send.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing prompt, graph, cognition,
  action-spec, dispatcher, scheduler, or self-cognition behavior.
- `no-prepost-user-input`: load before changing promise or commitment
  interpretation paths.
- `cjk-safety`: load before editing Python files that contain Chinese prompt
  strings or Chinese test fixture strings.
- `database-data-pull`: load before exporting or changing production MongoDB
  scheduler data.
- `character-test`: load before running real service endpoint or live character
  behavior smoke tests.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.
- LLM stages own semantic judgment. Deterministic code owns validation,
  persistence, scheduling, adapter delivery, permissions, deduplication, and
  audit.
- Do not add deterministic keyword classification for delayed-message user
  instructions. If a future follow-up should exist, L2d must select
  `trigger_future_cognition`.
- Do not preserve a compatibility path from consolidator `future_promises` to
  `send_message`.
- Do not preserve self-cognition production delivery through a prewritten
  `send_message` candidate. Self-cognition remains a trigger source and may
  still record private tracking artifacts.
- Do not add new prompt examples, negative constraints, feature flags, fallback
  LLM calls, repair loops, or alternate delivery paths.
- Do not expose raw scheduler ids, adapter ids, channel ids, collection names,
  handler ids, credentials, or schema versions to model-facing prompts.
- Do not delete completed historical `scheduled_events` rows. Only pending or
  running prewritten `send_message` rows are migration targets.
- Regular deterministic tests may run in batches. Real LLM and real service
  tests must run one case at a time with trace inspection.

## Must Do

- Remove the consolidator task-dispatch LLM prompt, LLM instance, raw tool-call
  generator, dispatcher configuration hook, and db-writer scheduling branch.
- Remove `task_dispatch` from consolidator write policy so persistence policy
  no longer advertises a visible-output side effect.
- Remove service startup wiring that installs the consolidator task dispatcher.
- Stop production self-cognition worker handoff from dispatching prewritten
  `send_message` candidates.
- Remove action-spec bridge surfaces that exist only to convert a legacy
  self-cognition delivery candidate into `send_message`.
- Delete the dispatcher symbols orphaned by the above removals
  (`TaskDispatcher`, `ToolCallEvaluator`, `EvalResult`, `RawToolCall`,
  `DispatchResult`), including the `dispatcher/dispatcher.py` and
  `dispatcher/evaluator.py` files, and trim every now-stale import and
  `__all__` entry. Leave no dead code, dead branch, or unused export.
- Keep future cognition scheduling through
  `ActionSpec(kind="trigger_future_cognition")`.
- Cancel existing pending or running `scheduled_events.tool="send_message"`
  rows through an explicit DB migration step with backup/export evidence.
- Update documentation that currently describes consolidator or self-cognition
  `send_message` handoff as an active production path.
- Replace tests that assert `future_promises -> send_message` with tests that
  assert no scheduled text is produced outside cognition.
- Add or update tests proving the future-cognition path remains available.

## Deferred

- Do not delete the entire `kazusa_ai_chatbot.dispatcher` package. This plan
  deletes every symbol the decommission orphans (`TaskDispatcher`,
  `ToolCallEvaluator`, `EvalResult`, `RawToolCall`, `DispatchResult`), but the
  package keeps its deterministic delivery and adapter primitives
  (`handlers.py`, `tool_spec.py`, `pending_index.py`, `adapter_iface.py`,
  `remote_adapter.py`, and `Task`/`DispatchContext`). Those still have live
  callers, so a clean bigbang leaves no dead code without removing them.
- Do not redesign adapters, the scheduler, or runtime `/send_message`
  endpoints. The scheduler's deterministic `send_message` handler path is kept
  intact even though no in-scope code produces new `send_message` rows.
- Do not redesign final live-chat delivery.
- `proactive_output` (`PO-ICD-001`) is a dormant, test-only contract package
  with no production sender; it is unaffected by this plan and out of scope.
- Do not redesign RAG, L1, L2d, L3 text, or dialog prompts except where tests
  require a prompt to stop naming the removed dispatcher path.
- Do not implement web research, notes, open-loop tools, or image generation.
- Do not add a deterministic delayed-message fallback when the character
  declines to schedule future cognition.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Consolidator task-dispatch LLM | bigbang | Delete the prompt, model call, raw tool-call generator, and db-writer branch. No fallback. |
| Consolidator write policy | bigbang | Remove `task_dispatch` as a write/effect category. |
| Service startup | bigbang | Do not configure consolidator with `TaskDispatcher`; do not pass a production dispatcher to self-cognition. |
| Self-cognition production delivery | bigbang | Remove production `send_message` handoff. Tracking artifacts remain private and non-delivering. |
| Dispatcher orphaned cluster | bigbang | Delete `TaskDispatcher`, `ToolCallEvaluator`, `EvalResult`, `RawToolCall`, `DispatchResult` and the files that hold them once their last callers are removed. Leave no dead code or unused exports. |
| Existing pending scheduler rows | migration | Export then cancel pending/running `scheduled_events.tool="send_message"` rows. Completed rows remain historical. |
| Future follow-up | bigbang | Use only `trigger_future_cognition` for delayed reasoning. L3 speak may happen only during the later cognition episode. |
| Tests | bigbang | Rewrite tests to assert the new invariant instead of preserving old dispatch expectations. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- For bigbang areas, delete or rewrite legacy references instead of preserving
  compatibility shims.
- For the migration area, follow the exact backup, cancel, and verification
  gates in `Data Migration`.
- Any change to cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, feature flags, or extra
  delivery features.
- The agent must treat changes outside the listed change surface as
  high-scrutiny changes and stop unless the plan is updated.
- The agent may remove code from the existing codebase with lighter
  justification when the removal is explicitly in scope and verified by greps
  and tests.
- The agent must search for existing helper behavior before adding any helper.
  If equivalent behavior exists, reuse or move it instead of duplicating it.
- The agent must not perform unrelated formatting churn, dependency upgrades,
  prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve the plan's stated intent and record
  the discrepancy in `Execution Evidence`.
- If an instruction is impossible, stop and report the blocker instead of
  inventing a substitute.

## Target State

Runtime target:

```text
user_message | self_cognition | scheduled_tick | tool_result
-> shared cognition / RAG / L2d
-> action specs
   -> speak -> L3 text -> normal visible delivery
   -> trigger_future_cognition -> scheduler row -> later cognition
   -> memory_lifecycle_update -> private memory owner
-> episode trace
-> consolidation persistence
```

Removed runtime path:

```text
consolidator future_promises
-> task-dispatch LLM
-> RawToolCall(send_message)
-> scheduled prewritten text
```

The consolidator remains responsible for durable state updates and episode
trace learning. It no longer selects, generates, schedules, or dispatches any
user-visible text.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Delayed user-visible text | Replace prewritten delayed `send_message` with scheduled future cognition. | The character must decide at execution time whether to speak. |
| Consolidator role | Consolidator persists and learns; it does not execute or schedule visible actions. | Keeps consolidation downstream of cognition and action results. |
| Existing pending sends | Cancel pending/running `send_message` rows. | Otherwise old prewritten text can still fire after code cutover. |
| Dispatcher cluster | Delete every symbol the decommission orphans (`TaskDispatcher`, `ToolCallEvaluator`, `EvalResult`, `RawToolCall`, `DispatchResult`) in this plan; keep the deterministic delivery and adapter primitives. | A bigbang must not leave dead code. The orphaned symbols have zero live callers after cutover; the kept primitives still serve the scheduler and adapters. |
| Self-cognition delivery | Production self-cognition must not dispatch prewritten text. | Self-cognition is a trigger source, not an alternate output executor. |
| Future cognition | Preserve and test `trigger_future_cognition`. | This is the approved replacement for delayed follow-up intent. |

## Contracts And Data Shapes

### Removed Model-Facing Contract

Delete the consolidator task-dispatch prompt and payload:

```python
{
    "final_dialog": list[str],
    "future_promises": list[dict],
    "available_tools": list[dict],
}
```

No LLM should receive this payload for tool-call generation after this plan.

### Preserved Scheduler Contract

`scheduled_events` remains the persistence collection for scheduled work.
Allowed new rows from this workstream:

```python
{
    "tool": "trigger_future_cognition",
    "args": {
        "episode_type": "self_cognition",
        "continuation_objective": str,
    },
    "execute_at": "ISO-8601 UTC timestamp",
    "status": "pending",
}
```

Forbidden new rows from this workstream:

```python
{
    "tool": "send_message",
    "args": {"text": "prewritten future character text"},
}
```

### Consolidation Metadata

The consolidation metadata keys `metadata["scheduled_event_ids"]` and
`metadata["task_dispatch_rejected"]` exist only to describe the removed
task-dispatch branch. A clean bigbang deletes them; keeping empty-list
placeholders for a removed feature is stale code and would also break the
`rg "task_dispatch"` static grep. Before deleting, grep for readers of each
key; if a reader exists, remove that reader in the same change. The consolidator
no longer emits any dispatch-related metadata.

## LLM Call And Context Budget

Before:

| Call | Path | Blocking | Purpose |
|---|---|---|---|
| Consolidator task-dispatch LLM | background post-turn | non-response-path | Convert `future_promises` into `send_message` tool calls. |

After:

| Call | Path | Blocking | Purpose |
|---|---|---|---|
| Consolidator task-dispatch LLM | removed | none | No replacement call. |
| L2d action initializer | existing cognition path | existing response or self-cognition path | May select `trigger_future_cognition` when delayed reasoning is semantically appropriate. |
| Later cognition/L3 text | existing scheduled self-cognition path | background | May select `speak` and produce text at execution time. |

This plan removes one background LLM call. It does not add a response-path LLM
call or increase context budget.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - Delete `_TASK_DISPATCHER_PROMPT`, `_TASK_DISPATCHER_OUTPUT_FORMAT`,
    `_task_dispatcher_llm`, `_TASK_DISPATCHER_LLM_HIDDEN_ARGS`,
    `configure_task_dispatcher`, `_get_task_dispatcher`,
    `_build_dispatch_context`, `_build_dispatch_instruction`,
    `_dispatcher_llm_promises`, `_dispatcher_llm_args_schema`,
    `_normalize_raw_tool_call_args`, and `_generate_raw_tool_calls`.
  - Delete the module-level globals `_task_dispatcher` and `_task_registry`.
  - Remove the `kazusa_ai_chatbot.dispatcher` import line
    (`DispatchContext`, `RawToolCall`, `TaskDispatcher`, `ToolRegistry`) and any
    LLM message imports used only by that path. Keep the
    `dispatcher.task.parse_iso_datetime` import only if surviving promise
    normalization still uses it; otherwise remove it too.
  - Remove the db-writer branch that calls `dispatcher.dispatch`.
  - Keep promise normalization used for user-memory-unit persistence.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
  - Remove `task_dispatch` from `WritePolicyKey`,
    `ConsolidationWritePolicy`, and returned policy dictionaries.
- `src/kazusa_ai_chatbot/service.py`
  - Remove the `configure_task_dispatcher` call and the
    `dispatcher=_task_dispatcher` argument passed into
    `start_self_cognition_worker`.
  - Remove the now-dead `evaluator = ToolCallEvaluator(...)` and
    `_task_dispatcher = TaskDispatcher(...)` lines in the "Build the
    task-dispatch runtime" block, the module-level `_task_dispatcher` global,
    and the `TaskDispatcher`, `ToolCallEvaluator`, `configure_task_dispatcher`
    imports.
  - Keep `tool_registry` (including `build_send_message_tool()` registration),
    `adapter_registry`, `pending_index`, and the `scheduler.configure_runtime`
    call. The scheduler still owns deterministic `send_message` delivery for
    any remaining scheduler rows and is out of scope.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Remove the `TaskDispatcher` import, every `dispatcher` parameter, and the
    `dispatch_candidate_func` wiring into `dispatch_action_candidate`.
  - Collapse `_handle_case_action_outputs` so it records the action attempt
    (and any action candidate) as private tracking artifacts only, with no
    dispatcher handoff and no `_not_requested_dispatch_result` branch.
  - Remove the `event_logging.record_dispatcher_event(...)` call that reported
    `action_kind=send_message`; there is no dispatch left to report.
  - Remove now-always-zero dispatch counters (`dispatched_count`,
    `rejected_count` and their `SelfCognitionWorkerResult` fields / tick-event
    arguments) so the collapse leaves no dead counter.
  - Keep scheduled future-cognition claim/complete behavior.
- `src/kazusa_ai_chatbot/self_cognition/__init__.py`
  - Remove the `build_send_message_action_spec` and `dispatch_action_candidate`
    imports and their `__all__` entries.
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`
  - Delete `build_raw_tool_call_from_action_spec`,
    `_validate_send_message_contract`, the `SEND_MESSAGE_CAPABILITY` branch in
    `_validate_kind_specific_contract`, and the now-unused imports
    (`build_dispatcher_bridge_capabilities`, `RawToolCall`,
    `SEND_MESSAGE_CAPABILITY`).
  - Keep `speak`, `memory_lifecycle_update`, and `trigger_future_cognition`
    validation.
- `src/kazusa_ai_chatbot/action_spec/registry.py`
  - Delete `build_dispatcher_bridge_capabilities`, `_send_message_capability`,
    `_send_message_projection`, the `SEND_MESSAGE_CAPABILITY` constant, and the
    `SEND_MESSAGE_CAPABILITY` branch in `project_prompt_affordances`.
  - Keep `build_initial_action_capabilities` and the remaining capability
    builders.
- `src/kazusa_ai_chatbot/action_spec/models.py`
  - Delete `SendMessageParamsV1`.
  - Remove `"dispatcher"` from `ALLOWED_CAPABILITY_OWNERS` and from the
    `CapabilitySpecV1.owner_module` literal. After this cutover, action-spec
    capability owners are only `memory_lifecycle`, `orchestrator`, `l3_text`,
    and `l3_image`.
  - Keep `ActionTargetV1.target_kind` values such as `current_channel` and
    `channel`; those are still valid semantic target kinds for non-dispatcher
    actions and future surface handlers.
- `src/kazusa_ai_chatbot/action_spec/__init__.py`
  - Remove the `build_raw_tool_call_from_action_spec` and
    `build_dispatcher_bridge_capabilities` imports and their `__all__` entries.
- `src/kazusa_ai_chatbot/action_spec/README.md`
  - Update `send_message` wording to state it is not a cognition-visible or
    self-cognition handoff capability in this runtime.
- `src/kazusa_ai_chatbot/dispatcher/README.md`
  - State that `TaskDispatcher`, `ToolCallEvaluator`, `EvalResult`,
    `RawToolCall`, and `DispatchResult` are deleted, and that the consolidator
    task-dispatch generation path is decommissioned.
  - Keep documentation only for the remaining deterministic delivery mechanics
    (`handlers.py`, `tool_spec.py`, adapters) and historical scheduler rows.
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Remove production `send_message` handoff instructions.
  - State that self-cognition output must re-enter the shared cognition/L2d/L3
    path or remain private.
- `development_plans/reference/designs/cognition_contracts_design.md`
  - Update the global invariant: scheduler may persist future cognition slots,
    but scheduled user-visible text cannot be authored outside cognition and
    L3 text.
- `development_plans/reference/designs/action_spec_effector_expansion_architecture.md`
  - Update subordinate design rationale to remove dispatcher as an active
    semantic text bridge.
- `src/kazusa_ai_chatbot/dispatcher/task.py`
  - Delete the `RawToolCall` and `DispatchResult` classes. Both are orphaned
    once the consolidator path, `handoff.py`, and the action-spec bridge are
    removed.
  - Keep `Task`, `DispatchContext`, `BotPermissionRole`, and
    `parse_iso_datetime`; the scheduler still depends on them.
- `src/kazusa_ai_chatbot/dispatcher/__init__.py`
  - Remove the `TaskDispatcher`, `ToolCallEvaluator`, `EvalResult`,
    `RawToolCall`, and `DispatchResult` imports and their `__all__` entries.
  - Keep all adapter, handler, `ToolRegistry`/`ToolSpec`, `PendingTaskIndex`,
    `Task`, and `DispatchContext` exports.

### Delete

- `src/kazusa_ai_chatbot/self_cognition/handoff.py`
  - Delete when imports and tests are updated; this file only exists to bridge
    legacy self-cognition candidates to `send_message`.
- `src/kazusa_ai_chatbot/dispatcher/dispatcher.py`
  - Delete the whole file. `TaskDispatcher` has no remaining caller once the
    consolidator and self-cognition `dispatcher.dispatch` call sites are gone.
- `src/kazusa_ai_chatbot/dispatcher/evaluator.py`
  - Delete the whole file. `ToolCallEvaluator` and `EvalResult` are used only
    by the deleted `TaskDispatcher` and the removed `service.py` wiring.

### Tests To Modify Or Delete

This list was reconciled against the `Static Greps`. Every test file that
imports or references a removed symbol is listed here.

- `tests/test_consolidator_origin_policy_db_writer.py`
  - Replace scheduler-dispatch preservation test with a no-dispatch invariant.
  - Remove the `DispatchResult` and `RawToolCall` imports and their fixtures.
- `tests/test_consolidation_origin_policy.py`
  - Remove `task_dispatch` from the expected write-policy key list.
- `tests/test_consolidator_efficiency.py`
  - Remove the `persistence_module._task_dispatcher` monkeypatch.
- `tests/test_db_writer_cache2_invalidation.py`
  - Remove the `persistence_module._task_dispatcher` monkeypatch.
- `tests/test_service_event_logging.py`
  - Remove the `configure_task_dispatcher` monkeypatch.
- `tests/test_reflection_cycle_stage1c_service.py`
  - Remove the `configure_task_dispatcher` monkeypatch.
- `tests/test_dispatcher.py`
  - Delete the whole file. It exercises `TaskDispatcher`, `ToolCallEvaluator`,
    and `RawToolCall`, all of which are removed.
- `tests/test_dispatcher_event_logging.py`
  - Remove the `TaskDispatcher` and `ToolCallEvaluator` imports and the
    `_dispatcher_with_adapter` fixture and its tests.
  - Keep the direct `handle_send_message` tests; that handler is retained.
- `tests/test_scheduler_future_promise.py`
  - Rewrite fixtures that build scheduler rows through `ToolCallEvaluator` and
    `RawToolCall` so they construct `Task`/`ScheduledEventDoc` rows directly.
- `tests/test_action_spec_self_cognition_bridge.py`
  - Delete the file. The bridge it tested is removed, and no replacement bridge
    test should preserve that surface.
- `tests/test_self_cognition_integration.py`
  - Replace dispatcher-handoff assertions with no-production-dispatch
    assertions. Remove the `DispatchResult` import.
- `tests/test_self_cognition_tracking.py`
  - Remove the `"task_dispatch": True` entries from tracking-artifact fixtures.
- `tests/test_self_cognition_event_logging.py`
  - Update or remove the fake `DispatchResult` self-cognition dispatcher-event
    case so it asserts no production dispatch.
- `tests/test_action_spec_evaluator.py`
  - Update capability expectations so the registry no longer exposes
    `send_message`, and remove bridge-evaluator assertions.
- `tests/test_action_spec_models.py`
  - Remove `send_message` capability/model examples that the registry no
    longer produces.
- `tests/test_e2e_live_llm.py`
  - Update any expectation that a future promise creates a scheduled
    `send_message`.

### Keep

- `src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py`
  - Keep and verify; this is the replacement delayed-follow-up mechanism.
- `src/kazusa_ai_chatbot/scheduler.py`
  - Keep scheduler persistence, the deterministic tool-handler dispatch path,
    and future-cognition slot behavior. Do not change the scheduler.
- `src/kazusa_ai_chatbot/self_cognition/models.py`,
  `src/kazusa_ai_chatbot/self_cognition/tracking.py`
  - Keep `ACTION_KIND_SEND_MESSAGE` and the artifact builders
    (`build_idempotency_key`, `build_action_attempt`, `build_action_candidate`).
    They produce private, non-delivering self-cognition tracking artifacts
    (`production_write=False`, `production_handoff=False`). The constant is a
    descriptive artifact label, not a dispatch instruction, and is not orphaned
    by this plan. The artifact's now-unread `dispatch_shape` field stays as
    stored metadata; reshaping the tracking-artifact schema is out of scope.
- `src/kazusa_ai_chatbot/dispatcher/handlers.py`,
  `src/kazusa_ai_chatbot/dispatcher/tool_spec.py`,
  `src/kazusa_ai_chatbot/dispatcher/pending_index.py`,
  `src/kazusa_ai_chatbot/dispatcher/adapter_iface.py`,
  `src/kazusa_ai_chatbot/dispatcher/remote_adapter.py`
  - Keep. These are deterministic delivery and adapter primitives the
    scheduler and adapters still use. `build_send_message_tool` and
    `handle_send_message` stay as the scheduler's deterministic delivery
    handler.
- Adapter `/send_message` endpoints
  - Keep normal adapter delivery APIs. This plan removes the legacy authoring
    path, not adapter transport.

## Data Migration

The kept scheduler still executes `send_message` rows. To avoid a window where
old prewritten text fires after the code cutover, run this migration inside the
deployment window while the service is stopped (or with
`SCHEDULED_TASKS_ENABLED=false`), so the scheduler cannot execute or claim
`send_message` rows during export and cancel.

0. Precondition: confirm the brain service is stopped or the scheduler is
   disabled. Do not run this migration against a service that is actively
   processing `scheduled_events`.

1. Export rows matching:

```python
{
    "tool": "send_message",
    "status": {"$in": ["pending", "running"]}
}
```

2. Store the export artifact path in `Execution Evidence`.
3. Update those rows only:

```python
{
    "$set": {
        "status": "cancelled",
        "cancelled_at": "<current UTC ISO timestamp>",
        "cleanup_reason": "consolidator_text_dispatch_decommission"
    }
}
```

4. Verify the count of pending/running `send_message` rows is zero.
5. Do not modify completed, failed, or already-cancelled historical rows.

## Overdesign Guardrail

- Actual problem: a background consolidator task-dispatch LLM can create
  user-visible delayed text without a fresh cognition/L3 decision at execution
  time.
- Minimal change: remove that text-authoring path and cancel existing pending
  prewritten sends; preserve future cognition scheduling as the delayed
  follow-up mechanism.
- Ownership boundaries: L2d chooses action semantics; L3 text authors
  user-visible wording; scheduler persists future cognition slots; adapters
  deliver already-authored live text; consolidator persists and learns only.
- Rejected complexity: no compatibility shim, no deterministic delayed-message
  fallback, no feature flag, no generic dispatcher replacement, no new prompt
  examples, no repair LLM, no web/tool expansion, and no adapter redesign.
- Evidence threshold: add new delayed delivery mechanics only after an
  observed product need requires user-visible scheduled contact that cannot be
  represented as future cognition, and after a separate approved plan defines
  cognition-time revalidation, permissions, audit, and real LLM tests.

## Implementation Order

1. Add failing deterministic tests for the new invariant.
   - First run every `Static Greps` command and reconcile the result against
     `Tests To Modify Or Delete`. Any file that references a removed symbol but
     is not listed is a plan gap; stop and update the plan.
   - Update `tests/test_consolidator_origin_policy_db_writer.py` so allowed
     user-message origins do not call dispatcher generation or dispatch.
   - Update self-cognition integration tests so production worker ticks do not
     dispatch action candidates.
   - Run the focused tests and record the expected failures.
2. Remove consolidator task-dispatch code.
   - Modify `persona_supervisor2_consolidator_persistence.py` and
     `persona_supervisor2_consolidator_origin_policy.py`.
   - Rerun the focused consolidator tests.
3. Remove service startup wiring into consolidator task dispatch.
   - Modify `service.py`.
   - Preserve adapter registry setup needed for runtime adapter registration.
   - Run import/compile checks.
4. Remove self-cognition production send-message handoff.
   - Delete `self_cognition/handoff.py`.
   - Update `self_cognition/worker.py`, `self_cognition/__init__.py`, and
     tests.
   - Rerun focused self-cognition tests.
5. Remove unused action-spec send-message bridge support and delete the
   orphaned dispatcher cluster.
   - Remove the action-spec bridge from `evaluator.py`, `registry.py`, and
     `action_spec/__init__.py`. Keep `speak`, `memory_lifecycle_update`, and
     `trigger_future_cognition`.
   - With every caller now removed, delete `dispatcher/dispatcher.py` and
     `dispatcher/evaluator.py`, delete `RawToolCall`/`DispatchResult` from
     `dispatcher/task.py`, and trim `dispatcher/__init__.py` exports.
   - Update evaluator/registry/model tests and the dispatcher tests
     (`test_dispatcher.py`, `test_dispatcher_event_logging.py`,
     `test_scheduler_future_promise.py`).
   - Run the static greps in `Verification` and confirm zero matches for the
     removed symbols before proceeding.
6. Update documentation.
   - Update README and reference docs listed in `Change Surface`.
   - Run static greps for stale claims.
7. Run scheduler data migration.
   - Stop the brain service or disable the scheduler first, per the
     `Data Migration` precondition.
   - Export pending/running `send_message` rows.
   - Cancel exported rows.
   - Verify zero pending/running `send_message` rows remain.
   - Bring the service back up only after verification passes.
8. Run deterministic verification.
9. Run real LLM and real service smoke tests one case at a time.
10. Run independent code review and remediate findings.

## Progress Checklist

- [x] Stage 1 - failing invariant tests added
  - Covers: implementation order step 1.
  - Verify: focused tests fail for the old dispatch behavior.
  - Evidence: 2026-05-17 added/remapped invariant tests for no
    consolidator-driven dispatch metadata, no self-cognition delivery handoff,
    no action-spec send-message bridge, and no removed dispatcher symbols.
    The red phase was not isolated because production edits were assigned to a
    worker in parallel; this is recorded as execution-method variance, not a
    plan contract change.
  - Sign-off: Codex/2026-05-17.
- [x] Stage 2 - consolidator dispatch path removed
  - Covers: implementation order step 2.
  - Verify: consolidator focused tests pass.
  - Evidence: 2026-05-17 `venv\Scripts\python.exe -m pytest
    tests\test_consolidation_origin_policy.py
    tests\test_consolidator_origin_policy_db_writer.py
    tests\test_consolidator_efficiency.py
    tests\test_db_writer_cache2_invalidation.py
    tests\test_service_event_logging.py
    tests\test_reflection_cycle_stage1c_service.py -q` -> 28 passed.
  - Sign-off: Codex/2026-05-17.
- [x] Stage 3 - service and self-cognition production handoff removed
  - Covers: implementation order steps 3-4.
  - Verify: self-cognition focused tests and compile checks pass.
  - Evidence: 2026-05-17 `venv\Scripts\python.exe -m pytest
    tests\test_self_cognition_integration.py
    tests\test_self_cognition_tracking.py
    tests\test_self_cognition_event_logging.py -q` -> 54 passed. Changed
    Python and test files also passed `py_compile`.
  - Sign-off: Codex/2026-05-17.
- [x] Stage 4 - action-spec bridge and orphaned dispatcher cluster removed
  - Covers: implementation order step 5.
  - Verify: action-spec tests pass; `dispatcher/dispatcher.py` and
    `dispatcher/evaluator.py` are deleted; static greps show no matches for the
    bridge helpers or the `TaskDispatcher`/`ToolCallEvaluator`/`EvalResult`/
    `RawToolCall`/`DispatchResult` symbols.
  - Evidence: 2026-05-17 action-spec suite
    (`tests\test_action_spec_evaluator.py`,
    `tests\test_action_spec_models.py`,
    `tests\test_action_spec_attempt_ledger.py`,
    `tests\test_l2d_action_selection_cases.py`,
    `tests\test_persona_supervisor2_action_initializer.py`) -> 34 passed;
    scheduler/adapter suite (`tests\test_scheduler_future_promise.py`,
    `tests\test_dispatcher_event_logging.py`,
    `tests\test_runtime_adapter_registration.py`) -> 51 passed; static grep
    for removed dispatcher/action-spec bridge symbols -> no matches.
  - Sign-off: Codex/2026-05-17.
- [x] Stage 5 - documentation updated
  - Covers: implementation order step 6.
  - Verify: static greps for stale task-dispatch claims pass.
  - Evidence: 2026-05-17 updated reference and subsystem docs. Static grep for
    removed dispatcher symbols, action-spec bridge helpers, `task_dispatch`,
    task-dispatch wording, and send-message bridge claims -> no matches.
  - Sign-off: Codex/2026-05-17.
- [x] Stage 6 - scheduler migration complete
  - Covers: implementation order step 7.
  - Verify: the service was stopped or scheduler disabled during migration, the
    exported artifact exists, and pending/running `send_message` count is zero.
  - Evidence: 2026-05-17 user confirmed all services were shut down before
    migration. Export artifact:
    `test_artifacts\scheduler_migration\send_message_pending_running_20260516T234719Z.json`.
    Migration query `{"tool":"send_message","status":{"$in":["pending","running"]}}`
    returned `before_count=0`, `matched_count=0`, `modified_count=0`, and
    `after_count=0`; no historical completed/failed/cancelled rows were
    modified.
  - Sign-off: Codex/2026-05-17.
- [ ] Stage 7 - deterministic, real LLM, and service verification complete
  - Covers: implementation order steps 8-9.
  - Verify: all commands in `Verification` pass or have recorded accepted
    residual risk.
  - Evidence: record command output and live trace artifact paths.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 8 - independent code review complete
  - Covers: implementation order step 10.
  - Verify: full diff reviewed, findings fixed or explicitly recorded.
  - Evidence: record review findings, fixes, rerun commands, residual risks,
    and approval status.
  - Sign-off: `<agent/date>` after evidence is recorded.

## Verification

### Static Greps

- `rg "_TASK_DISPATCHER_PROMPT|_task_dispatcher_llm|_task_dispatcher\b|_task_registry|_generate_raw_tool_calls|configure_task_dispatcher" src tests`
  - Expected: no matches.
- `rg "build_send_message_action_spec|dispatch_action_candidate|build_dispatcher_bridge_capabilities|build_raw_tool_call_from_action_spec" src tests`
  - Expected: no matches.
- `rg "\b(TaskDispatcher|ToolCallEvaluator|EvalResult|RawToolCall|DispatchResult)\b" src tests`
  - Expected: no matches. The orphaned dispatcher cluster is fully deleted.
    `ActionEvalResult` and other unrelated names are not matched because of the
    word boundaries.
- `rg "task_dispatch" src tests`
  - Expected: no matches.
- `rg "future_promises.*send_message|send_message.*future_promises" src tests`
  - Expected: no active runtime or test expectations. Historical docs under
    `development_plans/archive/` are allowed only if this grep is run against
    the full repository.

### Deterministic Tests

- `venv\Scripts\python.exe -m pytest tests\test_consolidator_origin_policy_db_writer.py tests\test_consolidation_origin_policy.py tests\test_consolidator_efficiency.py tests\test_db_writer_cache2_invalidation.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_service_event_logging.py tests\test_reflection_cycle_stage1c_service.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py tests\test_self_cognition_event_logging.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_action_spec_evaluator.py tests\test_action_spec_models.py tests\test_action_spec_future_cognition.py tests\test_action_spec_results.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_scheduler_future_promise.py tests\test_dispatcher_event_logging.py -q`
  - Expected: `test_dispatcher.py` is deleted. Remaining dispatcher tests cover
    raw deterministic delivery mechanics only and must not import
    `TaskDispatcher`, `ToolCallEvaluator`, `RawToolCall`, consolidator
    generation helpers, or action-spec send-message bridge helpers.
- `venv\Scripts\python.exe -m py_compile <each changed Python file>`

### Database Verification

- Export pending/running `scheduled_events.tool="send_message"` rows before
  update.
- Verify after update:

```python
{
    "tool": "send_message",
    "status": {"$in": ["pending", "running"]}
}
```

returns count `0`.

### Real LLM Tests

Run one case at a time and inspect traces:

- L2d delayed-follow-up case: user asks the character to contact later. Pass
  condition: L2d may select `trigger_future_cognition`; it must not select or
  expose `send_message`.
- Scheduled future-cognition chain case: a pending
  `trigger_future_cognition` slot is processed. Pass condition: the later
  cognition episode receives the continuation objective, is allowed to choose
  speak or no visible action, and any visible text is produced by L3 text.

### Real Service Smoke

Use the `character-test` skill against the real service endpoint:

- Normal private-chat message still produces a visible reply through cognition
  and L3 text.
- A delayed-contact request does not create a pending
  `scheduled_events.tool="send_message"` row.
- If a future cognition slot is created, it uses
  `scheduled_events.tool="trigger_future_cognition"` and no prewritten message
  body.

## Independent Plan Review

Gate completed on 2026-05-17 from a fresh-review posture. Inputs reviewed:
this plan, `development_plans/README.md`, the development-plan-writing
contract references, `local-llm-architecture`, current source touchpoints under
`persona_supervisor2_consolidator_persistence.py`, `service.py`,
`self_cognition/worker.py`, `action_spec/*`, `dispatcher/*`, and static greps
for removed symbols.

Review scope:

- The plan aligns with `cognition_contracts_design.md` and the action-spec
  architecture: self-cognition is a trigger source, L3 owns final text, and the
  consolidator consumes traces only.
- The plan fully removes the observed failure path from latest chat trace:
  consolidator `future_promises -> send_message`.
- The plan does not preserve compatibility for the removed semantic path.
- The data migration gate prevents old pending prewritten text from firing
  after code cutover.
- Tests cover both removal of old behavior and preservation of future
  cognition.
- Agent creativity is bounded: no feature flags, fallback dispatchers,
  deterministic keyword routing, new tools, or prompt repair loops.

Record blockers, non-blocking findings, required edits, and approval status in
`Execution Evidence`.

Review result:

- Blockers: none after the inline fixes below.
- Required edits applied before approval:
  - Added `src/kazusa_ai_chatbot/action_spec/models.py` to the change surface
    so bridge-only `SendMessageParamsV1` and the `dispatcher` capability-owner
    enum cannot remain as stale action-spec surface.
  - Resolved the `tests/test_action_spec_self_cognition_bridge.py` choice by
    requiring file deletion.
- Non-blocking findings: the plan deliberately keeps scheduler delivery
  primitives while deleting the orphaned dispatch-planning cluster. This is
  acceptable because scheduler execution still needs `Task`,
  `DispatchContext`, `ToolRegistry`, adapter registry, and
  `handle_send_message`.
- Approval status: approved for execution.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Alignment with `Must Do`, `Deferred`, cutover policy, autonomy boundaries,
  change surface, implementation order, verification gates, and acceptance
  criteria.
- Code quality and design weaknesses: hidden delivery fallbacks, stale bridge
  helpers, prompt/context leaks, scheduler persistence risk, adapter coupling,
  broad exception handling, and unnecessary abstractions.
- Regression quality: tests prove old path removal, future cognition survival,
  normal chat survival, and DB migration safety.

Fix concrete findings only when the fix is inside this plan's change surface.
If a finding requires new scope, update the plan or request approval before
changing code.

## Acceptance Criteria

This plan is complete when:

- The consolidator no longer contains a task-dispatch LLM, visible tool
  registry projection, raw tool-call generation, or dispatcher call.
- `task_dispatch` is no longer a consolidator write-policy category.
- Service startup no longer configures consolidator task dispatch.
- Self-cognition production ticks do not schedule prewritten `send_message`
  rows.
- No action-spec bridge remains that lets cognition or self-cognition create a
  `send_message` action directly.
- The orphaned dispatcher cluster (`TaskDispatcher`, `ToolCallEvaluator`,
  `EvalResult`, `RawToolCall`, `DispatchResult`) is deleted, with no remaining
  imports, `__all__` entries, dead branches, or tests referencing it.
- Pending/running `scheduled_events.tool="send_message"` rows are exported and
  cancelled.
- Future delayed follow-up uses `trigger_future_cognition`.
- Deterministic tests, real LLM checks, real service smoke, static greps, DB
  verification, and independent code review are recorded in `Execution
  Evidence`.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Legitimate delayed-contact requests stop sending messages | Route delayed intent through `trigger_future_cognition`; accept that the character may later decline to speak. | Real LLM delayed-follow-up and scheduled-chain cases. |
| Old pending sends fire after code cutover | Run the migration inside the deploy window with the service stopped or scheduler disabled, then export and cancel pending/running `send_message` rows before restart. | DB count verification; Stage 6 records that the service was stopped during migration. |
| Normal live replies break | Do not change L3 text/dialog delivery path. | Real service private-chat smoke. |
| Self-cognition loses audit visibility | Keep action-attempt and event logging for private/non-delivered outcomes. | Self-cognition integration tests. |
| Dispatcher cluster left as dead code | Delete the orphaned cluster (`TaskDispatcher`, `ToolCallEvaluator`, `EvalResult`, `RawToolCall`, `DispatchResult`) and trim all imports/exports in the same plan. | Static grep for the removed symbols returns zero matches; `py_compile` of every changed file. |
| Over-deleting kept delivery primitives | Enumerate kept files in `Change Surface` -> `Keep`; do not touch the scheduler or `dispatcher/handlers.py`. | Scheduler and adapter tests still pass. |

## Execution Evidence

- Plan review: 2026-05-17 independent review approved after two inline
  specificity fixes: added `action_spec/models.py` to the change surface and
  made `tests/test_action_spec_self_cognition_bridge.py` deletion mandatory.
- Failing test evidence: invariant tests were added/remapped on 2026-05-17,
  but the red phase was not isolated because the production worker implemented
  code changes in parallel while the parent owned tests. The new tests enforce
  the removed-path contract directly: no consolidator dispatch metadata, no
  self-cognition delivery handoff, no L2d `send_message` bridge, and no removed
  dispatcher symbols.
- Implementation evidence: production worker removed the consolidator
  task-dispatch path, removed service/self-cognition dispatcher wiring, deleted
  `dispatcher/dispatcher.py`, `dispatcher/evaluator.py`, and
  `self_cognition/handoff.py`, and kept only deterministic scheduler/adapter
  delivery primitives.
- Static grep results: 2026-05-17 no matches for
  `TaskDispatcher|ToolCallEvaluator|EvalResult|RawToolCall|DispatchResult`,
  bridge helper names, `task_dispatch`, `task-dispatch`, or send-message bridge
  doc claims under `src`, `tests`, docs, and reference design documents.
- Deterministic test results: 2026-05-17 focused batch passed:
  `venv\Scripts\python.exe -m pytest` over the consolidator, service,
  reflection, action-spec, self-cognition, scheduler, adapter, and future
  cognition suites -> 179 passed in 9.99s.
- Database export artifact:
  `test_artifacts\scheduler_migration\send_message_pending_running_20260516T234719Z.json`
  with 0 exported pending/running `send_message` documents.
- Database migration result: 2026-05-17 service shutdown was confirmed by the
  user before the migration command. Exact filter:
  `{"tool":"send_message","status":{"$in":["pending","running"]}}`.
  Update result: `before_count=0`, `matched_count=0`, `modified_count=0`,
  `after_count=0`, `cleanup_reason=consolidator_text_dispatch_decommission`.
- Real LLM trace artifacts:
- Real service smoke artifacts:
- Independent code review:
