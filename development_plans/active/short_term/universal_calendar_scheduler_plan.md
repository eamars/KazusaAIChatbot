# universal calendar scheduler plan

## Summary

- Goal: Replace the fragmented delayed-execution paths with a durable Kazusa
  calendar scheduler that can run typed cognition, reflection, and recurring
  internal triggers without scheduling delayed user-visible text directly.
- Plan class: high_risk_migration
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, `database-data-pull` for optional
  live diagnostics, and `cjk-safety` if Python prompt or source-packet text
  with CJK is edited.
- Overall cutover strategy: bigbang with one-time migration. No compatibility
  layer, no dual reads, no dual writes, and no fallback to `scheduled_events`
  after migration verification.
- Highest-risk areas: losing pending future-cognition rows during migration,
  directly sending delayed visible text without fresh cognition, duplicate or
  stuck calendar runs, commitment due-time drift, reflection daily synthesis
  reading incomplete hourly input, recurrence drift across character-local
  timezone boundaries, and introducing a generic arbitrary-code job runner.
- Acceptance criteria: implementation is complete only when calendar schedules
  and runs are durable, due runs are atomically claimed with leases,
  `trigger_future_cognition` and precise active-commitment due checks use the
  new calendar path, active-commitment schedule reconciliation covers create,
  merge/evolve due changes, and lifecycle closure, pending legacy
  `scheduled_events` rows are migrated or cancelled by the approved script,
  production no longer reads
  `scheduled_events`, recurring schedules have deterministic next-run
  materialization, reflection integration provides durable
  `reflection_phase_slot` run intents through a provider-compatible handler
  seam while preserving hourly/daily idempotency, the `calendar_scheduler`
  package has an ICD-style README, docs and ops status describe the new stack,
  deterministic tests pass, migration dry-run/apply evidence is recorded, and
  independent code review approves the result.

## Context

Kazusa currently has several unrelated timing mechanisms:

- `scheduler.py` persists `scheduled_events` rows and creates process-local
  `asyncio` sleep tasks for dispatcher tools.
- `trigger_future_cognition` is persisted into `scheduled_events` but is
  explicitly skipped by the dispatcher scheduler and later polled by the
  self-cognition worker.
- Active commitments with `due_at` are not scheduled precisely. The
  self-cognition worker periodically queries active commitment memory units
  and prioritizes due rows inside the worker tick cap.
- Reflection runs on its own 15-minute worker loop, computes missing hourly
  run ids reactively, and then runs group self-cognition review on the same
  reflection cadence.
- RAG recall has a scheduled-event collector that reads pending
  `scheduled_events` as future-action evidence.

Codebase refresh as of 2026-06-04:

- `reflection_cycle.phase_scheduler` now exists as a pure materializer for
  `ReflectionPhaseRunIntent` rows. It defines
  `REFLECTION_PHASE_TRIGGER_KIND="reflection_phase_slot"`,
  `REFLECTION_PHASE_GROUPS_PER_SLOT=1`, and allowed payload actions
  `reflection_hourly_slot` and `group_self_cognition_review`.
- `reflection_cycle.worker.LocalReflectionPhaseRunProvider` is the current
  process-local control plane. It snapshots monitor-eligible scopes at
  `period_start_utc`, executes due phase intents through
  `_run_reflection_phase_intent`, coalesces older group review windows, and
  asks the provider for expected hourly runs before daily synthesis.
- `action_spec.handlers.future_cognition` still builds a
  `scheduled_events` document and persists it through `scheduler.schedule_event`.
- `self_cognition.sources.collect_scheduled_future_cognition_cases` still
  polls due `trigger_future_cognition` rows from `scheduled_events`, and
  `collect_active_commitment_cases` still scans active commitments by due
  ordering during the standalone worker tick.
- `rag.recall.collectors.scheduled_events.ScheduledEventCollector` is still
  imported by the recall agent and uses `query_pending_scheduled_events`.
- Service startup still configures `PendingTaskIndex`, registers the
  dispatcher `send_message` tool, calls `scheduler.configure_runtime`, and
  loads process-local scheduler tasks when `SCHEDULED_TASKS_ENABLED=true`.

The short-term reflection phase plan establishes the reflection-side contract
that this universal scheduler must consume. Reflection phase work is a single
calendar-compatible `ReflectionPhaseRunIntent` with
`trigger_kind="reflection_phase_slot"`, one selected source scope, one
`due_at`, `period_start_utc`, `slot_index`, `offset_seconds`, `payload`, and a
deterministic `idempotency_key`. The selected reflection handler may run at
most one hourly reflection slot and at most one group self-cognition review
case for that same selected scope. The universal scheduler must replace the
local reflection phase run provider control plane with a durable
calendar-backed adapter that feeds the same handler shape; it must not invent
separate `reflection_hourly_slot` and
`group_self_cognition_review` calendar trigger kinds.

The target architecture is a calendar system, not a dispatcher queue. The
calendar decides when an internal Kazusa trigger becomes due. The owning
subsystem then evaluates the current state and decides what to do. For delayed
contact, the calendar must trigger cognition first; it must not schedule a
prewritten user-visible message.

The user has explicitly accepted a one-time migration and no compatibility
layer. Historical rows may remain for audit, but production code must stop
reading and writing the old `scheduled_events` control plane after cutover.

## Mandatory Skills

- `development-plan`: load before editing, reviewing, approving, executing,
  verifying, signing off, or handing off this plan.
- `local-llm-architecture`: load before changing future-cognition,
  self-cognition source packets, reflection-to-cognition handoff, prompt-facing
  fields, LLM budgets, RAG recall evidence, cognition, dialog, or
  consolidation behavior.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before optional live MongoDB diagnostics or
  migration evidence collection; do not read `.env`.
- `cjk-safety`: load before editing Python files that contain CJK prompt text,
  source-packet text, or CJK string literals.

## Mandatory Rules

- Do not implement this draft plan until the user explicitly approves it.
- Do not read `.env`.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Preserve the live chat response path. Do not serialize normal `/chat`
  processing behind calendar, reflection, or self-cognition work.
- The calendar scheduler must schedule typed Kazusa triggers, not arbitrary
  Python callbacks, raw adapter operations, MongoDB commands, or delayed
  visible text.
- The new `calendar_scheduler` package must include an ICD-style
  `README.md`, matching the existing module documentation pattern. It must
  document document control, ownership boundary, public interfaces, collection
  contracts, worker lifecycle, trigger kinds, migration/cutover behavior,
  event logging, verification, and forbidden paths.
- Reflection phase work uses the single composite trigger kind
  `reflection_phase_slot`. Do not split it into separate hourly-reflection and
  group-review calendar triggers.
- Calendar-owned reflection runs must map mechanically to
  `ReflectionPhaseRunIntent` records consumed by reflection phase execution
  through the provider-compatible handler seam.
- Exactly one component may claim and complete `reflection_phase_slot`
  `calendar_runs` after cutover. In this plan, the calendar worker owns the
  durable claim, lease, completion, and failure transitions.
- Reflection phase materialization must use the monitor-eligible snapshot as
  of `period_start_utc`, not each slot's wall-clock execution time.
- Reflection phase materialization must happen at the phase-period boundary or
  the first calendar materializer pass after that boundary. Do not precompute
  future reflection phase runs before the eligible-scope snapshot can be taken.
- The pure reflection phase materializer remains in `reflection_cycle`; the
  calendar owns durable run storage, claims, leases, status transitions, and
  retry policy.
- Do not add a `reflection_phase_runs` collection. After universal scheduler
  cutover, `calendar_runs` is the only durable reflection phase control plane.
- Delayed visible contact must go through a fresh cognition decision. The
  calendar may create a `future_cognition` or `commitment_due_cognition`
  source case; it must not directly call `send_message`.
- Deterministic code owns schedule creation, recurrence materialization,
  leases, claims, status transitions, idempotency, source scope validation,
  migration, cancellation, cleanup, and capacity limits.
- LLM stages own only semantic judgment inside an already-selected cognition
  or reflection case.
- Do not add Celery, Redis, APScheduler, Temporal, external cron services, or
  a new database. Use the current Python async service and MongoDB stack.
- Do not implement a general arbitrary job framework. The public contract is a
  closed set of typed trigger kinds with explicit handlers.
- Do not create a compatibility layer for `scheduled_events`. After migration
  verification, production code must not read or write `scheduled_events`.
- Do not keep dual reads, dual writes, adapter shims, old scheduler fallbacks,
  or hidden feature flags that preserve the old control plane.
- Preserve prompt safety. Calendar ids, run ids, scheduler internals, lease
  fields, raw channel ids, and database names must not enter model-facing
  source packets unless an existing trusted source-scope field already permits
  the semantic projection.
- Keep recurring schedule calculation deterministic and testable. Numeric
  limits must be named config values or named module constants; do not embed
  magic numbers inside worker logic.
- Calendar lease recovery must be deterministic and auditable. Stuck `running`
  rows are recovered only through named lease-expiry rules.
- Active-commitment calendar reconciliation must run after every active
  commitment create, merge/evolve semantic update that can write `due_at`, and
  lifecycle closure. The handler must re-read the memory unit at execution
  time and skip stale, missing, non-active, or due-mismatched rows rather than
  trusting old calendar payload.
- Event logs are observability only. They must not be used as the source of
  truth for schedule state, run state, idempotency, or migration completion.
- Migration scripts must default to dry-run. Apply mode must require an
  explicit `--apply` flag and must write a bounded summary artifact.
- Before running an apply migration against live MongoDB, run the dry-run and
  record row counts, unknown-tool counts, pending future-cognition counts, and
  pending legacy dispatcher-send counts in `Execution Evidence`.
- Pending legacy `send_message` rows in `scheduled_events` must be cancelled
  during migration rather than migrated. Direct delayed visible sends are not
  preserved.
- Unknown pending `scheduled_events.tool` values must block apply migration.
  The migration must report them and exit without partial writes.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.

## Must Do

- Add a new `calendar_scheduler` package with focused modules for models,
  repository operations, recurrence calculation, run materialization, worker
  orchestration, and typed handlers.
- Add `src/kazusa_ai_chatbot/calendar_scheduler/README.md` as the package ICD.
  It must follow the style of other subsystem READMEs and describe the
  calendar scheduler as a closed typed trigger scheduler, not a generic job
  runner.
- Add MongoDB collections and bootstrap indexes for `calendar_schedules` and
  `calendar_runs`.
- Add config values for calendar worker enablement, poll interval, claim limit,
  lease duration, retry limit, and per-trigger capacity.
- Add durable schedule definitions for one-time and recurring triggers.
- Add durable run rows with atomic pending-to-running claim semantics and
  lease expiry metadata.
- Add deterministic idempotency keys for schedules and runs.
- Add typed trigger handlers for:
  - `future_cognition`
  - `commitment_due_cognition`
  - `reflection_phase_slot`
  - `recurring_self_check`
- Move `trigger_future_cognition` creation from `scheduled_events` to the
  calendar scheduler.
- Move due future-cognition collection from `scheduled_events` to due
  `calendar_runs`.
- Replace production active-commitment due polling with calendar-created
  `commitment_due_cognition` runs.
- Upsert commitment due schedules when active-commitment memory units are
  created, merged, rescheduled, completed, or cancelled.
- Add active-commitment calendar reconciliation at the current write sites:
  `consolidation.memory_units.process_memory_unit_candidate` after
  `insert_user_memory_units(...)`, after
  `update_user_memory_unit_semantics(...)` when lifecycle fields can change
  `due_at`, and
  `action_spec.handlers.memory_lifecycle.execute_user_memory_lifecycle_action`
  after a successful lifecycle close.
- Add a targeted active-commitment due handler that re-reads the memory unit by
  `unit_id`, validates `unit_type="active_commitment"`, `status="active"`,
  and matching `due_at`, then builds exactly one normal active-commitment
  self-cognition case through the existing source-builder contract.
- Keep post-turn active-commitment lifecycle review intact. That path reviews
  current user commitments after a turn and is not the due-time scheduler.
- Replace the RAG recall scheduled-event collector with a calendar pending-run
  collector and update the recall agent import/contract names so production no
  longer imports `ScheduledEventCollector`.
- Add a durable reflection phase materialization path that upserts bounded
  `reflection_phase_slot` `calendar_runs` from the reflection
  `ReflectionPhaseRunIntent` materializer.
- Add a calendar-backed reflection phase adapter that converts claimed
  `reflection_phase_slot` calendar runs into `ReflectionPhaseRunIntent`
  records and invokes the same reflection phase execution handler used by the
  short-term provider seam.
- Promote or wrap the current reflection phase execution seam without changing
  reflection prompt contracts: calendar code may call a public
  reflection-cycle facade or a narrow package-private adapter, but it must not
  duplicate `_run_reflection_phase_intent` logic.
- Replace production use of the local reflection phase run provider as a
  control plane after cutover. Keep the pure reflection phase materializer as
  the typed source of run-intent construction if the calendar materializer
  needs it.
- Preserve `REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD`,
  `REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS`, and
  `REFLECTION_PHASE_GROUPS_PER_SLOT` semantics. Do not duplicate those
  reflection slot-budget decisions in unrelated calendar config names.
- Integrate reflection slot scheduling without changing the deterministic
  reflection run id contract for hourly and daily reflection documents.
- Add a one-time migration script that converts pending
  `trigger_future_cognition` rows to calendar rows and cancels pending
  legacy `send_message` rows.
- Remove production startup and shutdown use of the old `scheduler.py`
  process-local task runtime.
- Update tests, docs, ops status, and HOWTO configuration for the calendar
  scheduler.
- Add migration dry-run and apply tests.
- Add focused deterministic tests before production code changes.

## Deferred

- Do not implement multi-brain coordination.
- Do not add external scheduler infrastructure.
- Do not add arbitrary plugin-defined calendar handlers.
- Do not schedule delayed visible text.
- Do not preserve `scheduled_events` as a compatibility read or write path.
- Do not add adapter-specific calendar behavior.
- Do not add a `reflection_phase_runs` collection.
- Do not add separate calendar trigger kinds for hourly reflection and group
  self-cognition review. They are allowed actions inside one
  `reflection_phase_slot` run.
- Do not persist reflection phase slot control state outside `calendar_runs`
  after universal scheduler cutover.
- Do not add calendar-owned reflection slot budget config that duplicates the
  reflection phase config values.
- Do not migrate completed, failed, cancelled, or historical
  `scheduled_events` rows into active calendar control state.
- Do not backfill calendar schedules for active commitments without a valid
  absolute storage UTC `due_at`.
- Do not add natural-language recurrence parsing in this plan.
- Do not expose calendar internals to LLM prompts.
- Do not rewrite cognition, dialog, RAG, consolidation, reflection prompts, or
  memory promotion beyond the explicit integration points in this plan.
- Do not change reflection prompt versions solely because scheduling changed.
- Do not make daily reflection summarize silently partial input caused by
  missing expected hourly docs.

## Cutover Policy

Overall strategy: bigbang with one-time migration

| Area                                           | Policy                             | Instruction                                                                                                                                                                                               |
| ---------------------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Calendar collections                           | migration                          | Create `calendar_schedules` and `calendar_runs` through DB bootstrap before service wiring uses them.                                                                                                     |
| `trigger_future_cognition` creation            | bigbang                            | Replace `scheduler.schedule_event(...)` with calendar schedule/run creation. Do not dual-write to `scheduled_events`.                                                                                     |
| Due future-cognition processing                | bigbang                            | Self-cognition must collect due `future_cognition` calendar runs. Do not query `scheduled_events`.                                                                                                        |
| Active commitment due checks                   | bigbang                            | Replace periodic due polling in production with `commitment_due_cognition` calendar runs. Keep user-specific active-commitment readers only for live/post-turn lifecycle review and RAG evidence.         |
| Active commitment write hooks                  | bigbang                            | Create, merge/evolve due changes, and lifecycle closure must reconcile calendar schedules/runs. Do not rely on a periodic full scan after cutover.                                                        |
| RAG recall future-action evidence              | bigbang                            | Replace the scheduled-event collector with a calendar collector. Do not query `scheduled_events`.                                                                                                         |
| Reflection phase local provider                | bigbang                            | Replace production `LocalReflectionPhaseRunProvider` control-plane use with the calendar worker's `reflection_phase_slot` claim and handler adapter. Do not retain local provider fallback after cutover. |
| Reflection phase materializer                  | compatible as internal helper only | Keep the pure reflection materializer as an internal run-intent builder for calendar materialization if needed. It must not remain a separate production control plane.                                   |
| Reflection phase trigger shape                 | bigbang                            | Use one composite `reflection_phase_slot` calendar trigger. Do not create separate `reflection_hourly_slot` or `group_self_cognition_review` trigger kinds.                                               |
| Legacy pending `trigger_future_cognition` rows | migration                          | Convert pending rows to calendar schedules and runs through the approved script. Mark legacy rows migrated for audit.                                                                                     |
| Legacy pending `send_message` rows             | migration                          | Cancel pending rows through the approved script. Do not migrate delayed visible sends.                                                                                                                    |
| Unknown pending `scheduled_events` tools       | migration                          | Block apply migration and report the unknown rows. Do not partially migrate.                                                                                                                              |
| Old scheduler runtime                          | bigbang                            | Remove service startup/shutdown use of `configure_runtime`, `load_pending_events`, and process-local sleep tasks.                                                                                         |
| `scheduled_events` collection                  | compatible as historical data only | The collection may remain in MongoDB for audit and migration evidence. Production must not read or write it after cutover.                                                                                |
| Tests                                          | bigbang                            | Rewrite scheduler, future-cognition, self-cognition, recall, service startup, and DB tests around calendar behavior. Remove old production scheduler expectations.                                        |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- Bigbang areas must delete or rewrite legacy production references instead
  of preserving old behavior behind fallbacks.
- Migration areas must follow the exact migration phases and cleanup gates
  listed in this plan.
- Compatible areas must preserve only the historical-data surface explicitly
  listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Data Migration

Create `src/scripts/migrate_scheduled_events_to_calendar_scheduler.py`.

The migration must:

1. Default to dry-run.
2. Read pending rows from `scheduled_events`.
3. Classify rows by `tool`.
4. For `trigger_future_cognition` rows, create:
   - one one-time `calendar_schedules` row with `trigger_kind` set to
     `future_cognition`;
   - one pending `calendar_runs` row with `due_at` copied from `execute_at`;
   - legacy linkage fields that identify the source scheduled-event id.
5. For `send_message` rows, set legacy row status to `cancelled` with
   migration metadata. Do not create calendar rows.
6. For unknown tools, stop before writes in apply mode.
7. Mark migrated `trigger_future_cognition` legacy rows with status
   `migrated` and migration metadata after calendar writes succeed.
8. Write a JSON summary artifact containing counts, sample ids, blocked
   unknown-tool ids, and apply/dry-run mode.
9. Be idempotent. A rerun must not create duplicate calendar schedules or
   duplicate calendar runs.

The migration must not:

- read `.env`;
- delete the `scheduled_events` collection;
- migrate or backfill local reflection phase slots;
- create a `reflection_phase_runs` collection;
- migrate historical terminal legacy rows into active calendar state;
- create calendar runs for direct delayed visible sends;
- silently skip unknown pending tools.

Reflection phase schedule creation is not part of the `scheduled_events`
migration. The calendar implementation must create or reconcile its built-in
`reflection_phase_slot` schedule through the approved calendar
bootstrap/service-start path. The calendar materializer must create
`calendar_runs` for a phase period only when the `period_start_utc` snapshot is
available, using the same monitor-eligible scope selection semantics as the
current `LocalReflectionPhaseRunProvider`.

## Target State

Service startup creates and starts one calendar worker when
`CALENDAR_SCHEDULER_ENABLED=true`. The worker polls due `calendar_runs`,
atomically claims a bounded batch, executes typed handlers, records terminal
state, and materializes next runs for recurring schedules.

The calendar worker does not call adapters directly. For cognition triggers it
creates normal self-cognition source cases or routes through the existing
self-cognition worker runner. For reflection triggers it calls reflection
helpers that preserve existing reflection run document ids and prompt
contracts. Reflection phase triggers are converted into
`ReflectionPhaseRunIntent` records by a calendar-backed adapter, so the
reflection phase handler path stays the same even though durable claim,
lease, completion, and failure transitions belong to the calendar worker. For
recurring self checks it creates bounded internal source cases.

Daily-channel reflection readiness reads expected hourly work from the
calendar-backed reflection phase runs for the previous character-local day.
It must not fall back to a fresh monitored-channel snapshot after calendar
cutover, because that can silently summarize partial hourly input.

The old `scheduled_events` collection is no longer a production control
plane. Historical rows remain only for audit and migration evidence.

## Design Decisions

- Use MongoDB as the durable scheduler store because the project already uses
  MongoDB, `motor`, and process-local async workers.
- Use Python `asyncio` for the worker loop because the service is already
  async and no external scheduler stack is required.
- Use `datetime`, `zoneinfo`, existing `tzdata`, and existing time-boundary
  helpers for UTC and character-local time conversion.
- Use repository functions for all DB access. Production code must not write
  calendar collections ad hoc.
- Use atomic update filters for claims:
  `{"run_id": run_id, "status": "pending"}` or the lease-recovery equivalent.
- Store source scope at the schedule/run boundary so future handlers can bind
  to the original platform, channel, user, character identity, and permission
  context without asking the LLM to infer routing details.
- Treat recurrence as deterministic schedule materialization, not as an LLM
  planning task.
- Materialize one future run per recurring schedule at a time unless a named
  trigger explicitly needs a bounded catch-up policy.
- Use a closed trigger-kind registry. New trigger kinds require code review,
  tests, docs, config review, and handler registration.
- Preserve reflection hourly/daily run ids. Calendar run ids are scheduling
  ids; reflection run ids remain reflection data ids.
- Use a single composite `reflection_phase_slot` trigger for reflection phase
  work. The run payload may list allowed reflection actions, but the calendar
  registry must not register those actions as separate trigger kinds.
- Reuse the reflection phase run-intent contract as the migration seam. The
  universal calendar replaces the provider, not the reflection execution
  handlers.
- Allow `phase_period` recurrence materialization to create multiple child
  runs for one period only through a closed typed materializer. This is for
  reflection phase slots, not a generic arbitrary fan-out job system.
- Re-read active-commitment memory units at calendar execution time. Calendar
  payloads identify the intended unit and due timestamp; they are not the
  source of truth for commitment status, due date, semantic content, or
  delivery binding.
- Replace the recall scheduled-event collector with a new calendar collector
  rather than keeping a production collector whose class or file name implies
  `scheduled_events` ownership.

## Contracts And Data Shapes

### `calendar_schedules`

Required fields:

- `schedule_id: str`
- `schema_version: "calendar_schedule.v1"`
- `trigger_kind: str`
- `owner: str`
- `status: "active" | "paused" | "completed" | "cancelled"`
- `recurrence: dict`
- `timezone: str`
- `next_run_at: str | None`
- `created_at: str`
- `updated_at: str`
- `source_scope: dict`
- `payload: dict`
- `idempotency_key: str`
- `legacy_source: dict | None`

Supported `recurrence.kind` values for this plan:

- `once`
- `fixed_interval_seconds`
- `daily_local_time`
- `phase_period`

For `recurrence.kind="phase_period"`, the recurrence payload must include
named values equivalent to:

```python
{
    "period_seconds": int,
    "min_slot_spacing_seconds": int,
    "max_slots_per_period": int,
}
```

The only approved `phase_period` trigger in this plan is
`reflection_phase_slot`.

### `calendar_runs`

Required fields:

- `run_id: str`
- `schema_version: "calendar_run.v1"`
- `schedule_id: str`
- `trigger_kind: str`
- `owner: str`
- `status: "pending" | "running" | "completed" | "failed" | "cancelled" | "skipped"`
- `due_at: str`
- `period_start_utc: str | None`
- `slot_index: int | None`
- `offset_seconds: int | None`
- `created_at: str`
- `updated_at: str`
- `claimed_at: str | None`
- `completed_at: str | None`
- `lease_owner: str | None`
- `lease_expires_at: str | None`
- `attempt_count: int`
- `max_attempts: int`
- `source_scope: dict`
- `payload: dict`
- `idempotency_key: str`
- `legacy_source: dict | None`
- `result_summary: dict | None`
- `failure_summary: dict | None`

For `trigger_kind="reflection_phase_slot"`, `period_start_utc`,
`slot_index`, and `offset_seconds` are required and must match:

```text
due_at == period_start_utc + offset_seconds
```

For non-phase triggers, those fields must be `None`.

### Trigger Kinds

`future_cognition`

- Owner: `self_cognition`.
- Input payload: semantic continuation objective, source refs,
  continuation metadata, original action-attempt id, and optional original
  trigger time.
- Output: one normal scheduled-future self-cognition case.

`commitment_due_cognition`

- Owner: `self_cognition`.
- Input payload: `unit_id`, `global_user_id`, `due_at`, source scope, and
  prompt-safe commitment summary fields.
- Execution: re-read the memory unit by `unit_id`, verify it is still an
  active `active_commitment` for the expected `global_user_id`, and verify the
  stored `due_at` still equals the run payload `due_at`.
- Output: one normal active-commitment self-cognition case. Missing, closed,
  type-mismatched, user-mismatched, or due-mismatched units mark the calendar
  run `skipped` with a sanitized reason.

`reflection_phase_slot`

- Owner: `reflection_cycle`.
- Input payload: selected monitor-eligible channel scope, phase period
  seconds, max slots per period, prompt version, and allowed actions:
  `reflection_hourly_slot` and `group_self_cognition_review`.
- Required run fields: `period_start_utc`, `slot_index`, `offset_seconds`,
  `due_at`, `source_scope`, and deterministic `idempotency_key`.
- Output: one composite reflection phase execution for the selected scope.
  The reflection handler may run at most one due hourly reflection slot and at
  most one group self-cognition review case for that same selected scope,
  using the existing reflection and self-cognition run/idempotency contracts.
- Boundary: the calendar claim marks the phase run execution state only.
  Hourly reflection documents remain in `character_reflection_runs`, and
  group review once-only state remains in the self-cognition reviewed-window
  ledger defined by the reflection phase plan.

`recurring_self_check`

- Owner: `self_cognition`.
- Input payload: semantic check label, source scope, recurrence metadata, and
  bounded objective.
- Output: one internal self-cognition case.

### Reflection Phase Run Intent Compatibility

Every `calendar_runs` document with
`trigger_kind="reflection_phase_slot"` must map mechanically to the
reflection-side run intent:

```python
ReflectionPhaseRunIntent = {
    "run_id": calendar_run["run_id"],
    "trigger_kind": "reflection_phase_slot",
    "due_at": calendar_run["due_at"],
    "period_start_utc": calendar_run["period_start_utc"],
    "slot_index": calendar_run["slot_index"],
    "offset_seconds": calendar_run["offset_seconds"],
    "source_scope": calendar_run["source_scope"],
    "payload": calendar_run["payload"],
    "idempotency_key": calendar_run["idempotency_key"],
}
```

The calendar-backed adapter must pass these intents to reflection phase
handler code. Reflection handlers must not depend on calendar lease fields,
attempt-count fields, database collection names, or migration metadata.

Calendar-backed daily readiness must expose an equivalent of
`expected_hourly_runs_for_character_local_date(...)` by reading terminal and
pending `reflection_phase_slot` calendar runs for the target
character-local day and deriving expected hourly reflection run ids with the
same reflection helper logic used by the current local provider.

### Calendar Scheduler ICD README Contract

`src/kazusa_ai_chatbot/calendar_scheduler/README.md` must be written as a
module ICD, consistent with the existing subsystem READMEs. It must include:

- Document control: ICD id, owning package, interface boundary, runtime
  consumers, upstream owners, and downstream owners.
- Purpose and boundary: the calendar scheduler owns durable typed trigger
  timing, not arbitrary jobs, direct adapter sends, or delayed visible text.
- Public interfaces: package facade imports, worker start/stop functions,
  repository functions, materializer functions, and handler registration.
- Collection contracts: `calendar_schedules` and `calendar_runs` fields,
  statuses, idempotency keys, indexes, lease semantics, and allowed historical
  legacy linkage.
- Trigger contracts: `future_cognition`, `commitment_due_cognition`,
  `reflection_phase_slot`, and `recurring_self_check`, including ownership and
  model-facing context safety.
- Reflection integration: single composite `reflection_phase_slot`,
  `ReflectionPhaseRunIntent` mapping, no `reflection_phase_runs` collection,
  no split reflection trigger kinds, and daily readiness from durable calendar
  phase runs.
- Migration and cutover: one-time `scheduled_events` migration, pending
  `send_message` cancellation, no compatibility layer, and
  `scheduled_events` as historical data only.
- Runtime lifecycle and ops: service startup/shutdown, worker polling, claim
  limits, leases, retries, event logging, `/ops/runtime-status` fields, and
  health semantics.
- Forbidden paths: no arbitrary Python callbacks, external scheduler stack,
  plugin trigger kinds, natural-language recurrence parsing, direct adapter
  calls, prompt-visible calendar internals, or fallback to `scheduled_events`.
- Testing and verification: focused deterministic tests, migration dry-run,
  old-path greps, and live DB migration apply approval rule.

## LLM Call And Context Budget

- The calendar scheduler itself must make zero LLM calls.
- Recurrence calculation, due checks, schedule creation, and migration must be
  deterministic.
- Calendar handlers may call existing self-cognition or reflection paths only
  through their existing bounded runner interfaces.
- Prompt-facing source packets must receive semantic due-state facts and
  source context, not raw calendar ids, leases, status-transition internals,
  MongoDB collection names, or migration metadata.
- No prompt change is authorized by this plan unless an existing source packet
  must rename `scheduled_event` evidence to calendar evidence. Such a change
  must be minimal and covered by prompt-contract tests.
- If source packets rename `source_kind="scheduled_event"` to a calendar
  source kind, the visible summary must remain semantic, for example
  "scheduled future cognition slot". Raw `calendar_run`, `run_id`,
  `schedule_id`, lease, attempt, collection, and migration terms must stay out
  of model-facing text.

## Change Surface

Expected new files:

- `src/kazusa_ai_chatbot/calendar_scheduler/__init__.py`
- `src/kazusa_ai_chatbot/calendar_scheduler/README.md`
- `src/kazusa_ai_chatbot/calendar_scheduler/models.py`
- `src/kazusa_ai_chatbot/calendar_scheduler/repository.py`
- `src/kazusa_ai_chatbot/calendar_scheduler/recurrence.py`
- `src/kazusa_ai_chatbot/calendar_scheduler/materializer.py`
- `src/kazusa_ai_chatbot/calendar_scheduler/reflection_phase.py`
- `src/kazusa_ai_chatbot/calendar_scheduler/handlers.py`
- `src/kazusa_ai_chatbot/calendar_scheduler/worker.py`
- `src/kazusa_ai_chatbot/rag/recall/collectors/calendar_runs.py`
- `src/scripts/migrate_scheduled_events_to_calendar_scheduler.py`
- `tests/test_calendar_scheduler_active_commitments.py`
- `tests/test_calendar_scheduler_models.py`
- `tests/test_calendar_scheduler_repository.py`
- `tests/test_calendar_scheduler_recurrence.py`
- `tests/test_calendar_scheduler_worker.py`
- `tests/test_calendar_scheduler_migration.py`
- `tests/test_calendar_scheduler_reflection_phase.py`

Expected modified files:

- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/db/bootstrap.py`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/db/__init__.py`
- `src/kazusa_ai_chatbot/db/scheduled_events.py`
- `src/kazusa_ai_chatbot/db/script_operations.py`
- `src/kazusa_ai_chatbot/db/user_memory_units.py`
- `src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py`
- `src/kazusa_ai_chatbot/self_cognition/sources.py`
- `src/kazusa_ai_chatbot/self_cognition/worker.py`
- `src/kazusa_ai_chatbot/rag/recall/agent.py`
- `src/kazusa_ai_chatbot/rag/recall/contracts.py`
- `src/kazusa_ai_chatbot/rag/recall/README.md`
- `src/kazusa_ai_chatbot/consolidation/memory_units.py`
- `src/kazusa_ai_chatbot/action_spec/handlers/memory_lifecycle.py`
- `src/kazusa_ai_chatbot/reflection_cycle/__init__.py`
- `src/kazusa_ai_chatbot/reflection_cycle/phase_scheduler.py`
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
- `src/kazusa_ai_chatbot/dispatcher/README.md`
- `src/kazusa_ai_chatbot/action_spec/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
- `src/kazusa_ai_chatbot/db/README.md`
- `src/kazusa_ai_chatbot/brain_service/README.md`
- `src/kazusa_ai_chatbot/event_logging/README.md`
- `docs/HOWTO.md`
- `README.md`
- `README_CN.md`
- `tests/test_action_spec_future_cognition.py`
- `tests/test_self_cognition_integration.py`
- `tests/test_service_event_logging.py`
- `tests/test_scheduler_future_promise.py`
- `tests/test_rag_recall_agent.py`
- `tests/test_reflection_phase_scheduler.py`
- `tests/test_service_ops_status.py`
- `tests/test_reflection_cycle_stage1c_worker.py`
- `tests/test_config.py`
- `tests/test_db.py`

Expected deleted files:

- `src/kazusa_ai_chatbot/scheduler.py`
- `src/kazusa_ai_chatbot/dispatcher/pending_index.py`
- `src/kazusa_ai_chatbot/rag/recall/collectors/scheduled_events.py`

Expected deleted or production-decommissioned behavior:

- Service startup use of old `scheduler.configure_runtime` and
  `scheduler.load_pending_events`.
- Production write path from future cognition to `scheduled_events`.
- Production self-cognition due-slot reads from `scheduled_events`.
- Production active-commitment due polling inside the default self-cognition
  source collector.
- Production RAG recall reads from `scheduled_events`, including the
  `ScheduledEventCollector` production import path.
- Runtime facade access to scheduled-event helpers. Legacy scheduled-event
  operations may remain only inside the migration script or maintenance-only
  DB helpers used by that script.
- Production reflection worker use of the local phase provider as the durable
  control plane. The pure materializer may remain as an internal calendar
  materialization helper.

## Overdesign Guardrail

- Actual problem: Kazusa has multiple process-local or polling-based delayed
  work paths, so future cognition, active-commitment due checks, RAG future
  evidence, and reflection phase work cannot share durable claim, retry,
  audit, and migration semantics.
- Minimal change: add one MongoDB-backed calendar scheduler for the closed
  trigger kinds in this plan, migrate only pending legacy scheduler rows that
  are in scope, and wire existing subsystem handlers through narrow adapters.
- Ownership boundaries: deterministic calendar code owns schedule/run storage,
  recurrence materialization, idempotency, claim/lease/status transitions,
  migration, and capacity limits; reflection owns reflection run ids and
  prompt contracts; self-cognition owns source-case construction, route
  tracking, dialog rendering, delivery handoff, and consolidation; action-spec
  owns semantic action residues and handler validation; adapters own delivery.
- Rejected complexity: arbitrary job execution, external scheduler services,
  plugin-defined trigger kinds, generic fan-out, natural-language recurrence
  parsing, delayed visible text, compatibility layers, dual reads/writes,
  fallback to `scheduled_events`, calendar-owned reflection budget config, a
  `reflection_phase_runs` collection, and prompt-visible calendar internals.
- Evidence threshold: add any rejected complexity only after a separate
  approved plan identifies a concrete production failure or near-term
  integration that cannot be handled by the closed trigger-kind model, MongoDB
  atomic claims, and existing subsystem handlers.

## Agent Autonomy Boundaries

- The implementation agent may choose internal helper names inside the approved
  files only when they do not change public contracts.
- The implementation agent may add focused test fixtures inside listed test
  files.
- The implementation agent may add indexes required by the listed repository
  queries.
- The implementation agent must not add new trigger kinds.
- The implementation agent must not add separate reflection calendar trigger
  kinds for hourly reflection or group self-cognition review.
- The implementation agent must not preserve old `scheduled_events` reads or
  writes.
- The implementation agent must not add external dependencies.
- The implementation agent must not change prompt wording beyond the minimal
  calendar evidence rename allowed in `LLM Call And Context Budget`.
- The implementation agent must request approval before changing change
  surface, cutover policy, trigger contracts, recurrence kinds, or migration
  behavior.

## Implementation Order

1. Parent establishes focused calendar module tests.
   - Add model, recurrence, repository, worker-claim, and migration dry-run
     tests before production code.
   - Add active-commitment schedule reconciliation tests for create,
     merge/evolve due changes, lifecycle closure, execution-time re-read, and
     stale-run skip behavior.
   - Add reflection-phase calendar tests that prove
     `reflection_phase_slot` run rows map to `ReflectionPhaseRunIntent`,
     phase run materialization respects `period_start_utc`, and no
     `reflection_phase_runs` collection is used.
   - Run the focused tests and record expected failures for missing modules or
     missing functions.
2. Parent starts the production-code subagent with this approved plan,
   mandatory skills, focused tests, and production-code boundary.
3. Production-code subagent implements the calendar package, package ICD
   README, DB schema, config, and bootstrap indexes.
4. Parent runs focused calendar module tests.
5. Parent adds integration tests for future cognition, self-cognition,
   commitment due scheduling, recall evidence, reflection phase provider
   cutover, service startup, and migration.
6. Production-code subagent wires action-spec, self-cognition, RAG recall,
   commitment writes, reflection phase provider/materialization, and service
   lifecycle to the calendar package.
7. Parent runs integration tests and updates docs.
8. Parent runs migration dry-run test and deterministic migration script test.
9. Parent runs grep-based old-path checks to prove production no longer reads
   or writes `scheduled_events`.
10. Parent runs broader deterministic verification.
11. Parent runs the independent code review gate.
12. Parent remediates review findings inside approved scope and reruns affected
    tests.
13. Parent records final evidence and leaves the plan in `approved` or
    `in_progress` state according to the actual lifecycle stage.

## Execution Model

Execution must be parent-led and use native subagents unless the user
explicitly approves fallback execution.

- Parent agent owns test-contract creation, integration tests, migration
  evidence, docs, verification, independent review dispatch, review
  remediation, execution evidence, and lifecycle updates.
- Production-code subagent owns planned production code changes only. It must
  not change plan scope, cutover policy, trigger contracts, or migration
  behavior.
- Independent code-review subagent owns review only. It must receive the
  approved plan, full diff, migration evidence, and verification evidence. It
  must not implement fixes.

If native subagent capability is unavailable at execution time, stop before
production-code execution and report the blocker unless the user explicitly
requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused calendar module contract established

  - Covers: model, recurrence, repository, worker claim, and migration dry-run
    tests, including active-commitment reconciliation and reflection-phase
    run-intent mapping.
  - Verify: `venv\Scripts\python -m pytest tests/test_calendar_scheduler_models.py tests/test_calendar_scheduler_recurrence.py tests/test_calendar_scheduler_repository.py tests/test_calendar_scheduler_worker.py tests/test_calendar_scheduler_migration.py tests/test_calendar_scheduler_active_commitments.py tests/test_calendar_scheduler_reflection_phase.py -q`.
  - Evidence: record expected pre-implementation failures and changed test
    files in `Execution Evidence`.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 2 - calendar package, ICD README, DB schema, config, and indexes implemented

  - Covers: `calendar_scheduler` package, package ICD README, `config.py`, DB
    schemas, DB exports, and bootstrap indexes.
  - Verify: focused calendar tests pass.
  - Evidence: record changed files and focused test output.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 3 - future-cognition calendar integration complete

  - Covers: action-spec future-cognition handler, self-cognition due-run
    collection, worker claim/completion, and source-packet contract updates.
  - Verify: `venv\Scripts\python -m pytest tests/test_action_spec_future_cognition.py tests/test_self_cognition_integration.py -q`.
  - Evidence: record test output and changed files.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 4 - active-commitment due scheduling complete

  - Covers: active commitment create/update/reschedule/cancel hooks and
    `commitment_due_cognition` run handling.
  - Verify: `venv\Scripts\python -m pytest tests/test_calendar_scheduler_active_commitments.py tests/test_action_spec_memory_lifecycle.py tests/test_self_cognition_integration.py -q`.
  - Evidence: record test output and changed files.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 5 - recall and reflection integrations complete

  - Covers: RAG recall calendar evidence collector and reflection slot
    provider cutover, materialization, and trigger handlers.
  - Verify: `venv\Scripts\python -m pytest tests/test_rag_recall_agent.py tests/test_calendar_scheduler_reflection_phase.py tests/test_reflection_phase_scheduler.py tests/test_reflection_cycle_stage1c_worker.py -q`.
  - Evidence: record test output and changed files.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 6 - legacy scheduler decommission and migration script complete

  - Covers: service startup/shutdown removal of old scheduler runtime,
    migration script, scheduler tests rewritten or removed, and old-path grep
    checks.
  - Verify: migration tests pass and grep checks show no production
    `scheduled_events` scheduler reads or writes outside migration,
    historical cleanup diagnostics, and archived DB helpers.
  - Evidence: record test output, grep output, and migration dry-run summary.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 7 - docs, ops status, and config tests complete

  - Covers: top-level README, HOWTO, the `calendar_scheduler` ICD README,
    adjacent subsystem READMEs, ops status, and config tests.
  - Verify: `venv\Scripts\python -m pytest tests/test_config.py tests/test_service_ops_status.py -q`.
  - Evidence: record test output and doc files changed.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 8 - full deterministic verification complete

  - Covers: focused and integration tests listed in `Verification`.
  - Verify: all required commands pass or blocked commands are recorded with
    exact reason.
  - Evidence: record command outputs and residual risks.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 9 - independent code review complete

  - Covers: full diff, migration behavior, old-path decommission, tests, docs,
    and plan alignment.
  - Verify: review findings are recorded, required fixes are complete, and
    affected verification is rerun.
  - Evidence: record review result, fixes, rerun commands, and approval status.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

Required deterministic commands:

```powershell
venv\Scripts\python -m pytest tests/test_calendar_scheduler_models.py -q
venv\Scripts\python -m pytest tests/test_calendar_scheduler_recurrence.py -q
venv\Scripts\python -m pytest tests/test_calendar_scheduler_repository.py -q
venv\Scripts\python -m pytest tests/test_calendar_scheduler_worker.py -q
venv\Scripts\python -m pytest tests/test_calendar_scheduler_migration.py -q
venv\Scripts\python -m pytest tests/test_calendar_scheduler_active_commitments.py -q
venv\Scripts\python -m pytest tests/test_calendar_scheduler_reflection_phase.py -q
venv\Scripts\python -m pytest tests/test_action_spec_future_cognition.py -q
venv\Scripts\python -m pytest tests/test_self_cognition_integration.py -q
venv\Scripts\python -m pytest tests/test_action_spec_memory_lifecycle.py -q
venv\Scripts\python -m pytest tests/test_rag_recall_agent.py -q
venv\Scripts\python -m pytest tests/test_reflection_phase_scheduler.py -q
venv\Scripts\python -m pytest tests/test_reflection_cycle_stage1c_worker.py -q
venv\Scripts\python -m pytest tests/test_config.py tests/test_service_ops_status.py -q
venv\Scripts\python -m pytest tests/test_db.py -q
```

Required grep checks:

```powershell
rg -n "scheduled_events|schedule_event|load_pending_events|PendingTaskIndex|SCHEDULED_TASKS_ENABLED" src/kazusa_ai_chatbot tests
rg -n "query_pending_scheduled_events|list_due_future_cognition_events|trigger_future_cognition" src/kazusa_ai_chatbot tests
rg -n "ScheduledEventCollector|collectors.scheduled_events" src/kazusa_ai_chatbot tests
rg -n "reflection_phase_runs|reflection_hourly_slot|group_self_cognition_review" src/kazusa_ai_chatbot/calendar_scheduler tests/test_calendar_scheduler_reflection_phase.py
Test-Path -LiteralPath src/kazusa_ai_chatbot/calendar_scheduler/README.md
rg -n "Document Control|Owning package|Interface boundary|Public Interfaces|Collection Contracts|Trigger Contracts|Forbidden Paths" src/kazusa_ai_chatbot/calendar_scheduler/README.md
```

The grep checks must show only accepted historical references, migration
script references, decommissioned test references being deleted, or docs that
explicitly describe historical migration. Any production read or write against
`scheduled_events` blocks sign-off.

The reflection grep must show no `reflection_phase_runs` references. The
strings `reflection_hourly_slot` and `group_self_cognition_review` may appear
only as allowed actions inside the `reflection_phase_slot` payload contract,
not as calendar trigger-kind registrations.

The README path check must print `True`. The README grep must match each ICD
section heading or equivalent heading text in
`src/kazusa_ai_chatbot/calendar_scheduler/README.md`.

Migration verification:

```powershell
venv\Scripts\python -m scripts.migrate_scheduled_events_to_calendar_scheduler --dry-run --output test_artifacts/calendar_migration_dry_run.json
```

Apply migration is not part of normal deterministic tests. It may run only
after dry-run evidence is reviewed:

```powershell
venv\Scripts\python -m scripts.migrate_scheduled_events_to_calendar_scheduler --apply --output test_artifacts/calendar_migration_apply.json
```

Live DB migration apply requires explicit user approval at execution time.

## Independent Plan Review

Before this draft is promoted to `approved`, run an independent plan review
that checks:

- no compatibility layer is retained;
- migration behavior is complete and one-time;
- old scheduler production reads and writes are fully removed;
- trigger-kind contracts are narrow and typed;
- recurrence support is sufficient for approved use cases and not broader;
- active-commitment due scheduling has a write-time materialization path;
- active-commitment due scheduling reconciles create, merge/evolve due change,
  lifecycle closure, and stale-run skip behavior;
- reflection integration preserves hourly/daily idempotency;
- reflection phase materialization happens only when the `period_start_utc`
  monitor snapshot is available;
- reflection integration uses one composite `reflection_phase_slot` trigger
  and does not create split reflection calendar trigger kinds;
- reflection calendar runs map mechanically to `ReflectionPhaseRunIntent`
  records and are consumed through the provider-compatible handler seam;
- no `reflection_phase_runs` collection or other temporary reflection phase
  control plane is introduced;
- verification commands cover migration, scheduler core, and integration
  risks.

Record plan-review findings and fixes in `Execution Evidence` before approval.

## Independent Code Review

Before final completion, run an independent code review over the full diff.

Review scope:

- calendar scheduler repository correctness, atomic claims, lease expiry, and
  idempotency;
- recurrence calculation and timezone behavior;
- migration script dry-run/apply safety;
- removal of production `scheduled_events` reads and writes;
- future-cognition and commitment due-case source-packet safety;
- active-commitment calendar reconciliation across create, merge/evolve due
  changes, lifecycle closure, and stale-run skips;
- reflection run id preservation;
- reflection phase run-intent compatibility, provider cutover, and absence of
  a `reflection_phase_runs` control plane;
- absence of separate calendar trigger-kind registrations for
  `reflection_hourly_slot` and `group_self_cognition_review`;
- config defaults and no magic numbers;
- test coverage and verification accuracy;
- `calendar_scheduler/README.md` exists, follows the module ICD style, and
  accurately documents the implemented package boundary and forbidden paths;
- docs and ops status consistency.

The review subagent must not implement fixes. The parent agent records all
findings, remediates approved in-scope fixes, reruns affected tests, and
records final approval status in `Execution Evidence`.

## Acceptance Criteria

- `calendar_schedules` and `calendar_runs` schemas and indexes exist.
- `src/kazusa_ai_chatbot/calendar_scheduler/README.md` exists and follows the
  subsystem ICD style, including document control, ownership boundary, public
  interfaces, collection contracts, trigger contracts, worker lifecycle,
  migration/cutover behavior, ops/event logging, verification, and forbidden
  paths.
- Calendar worker startup and shutdown are wired into service lifecycle.
- Due runs are claimed atomically and use lease expiry for recovery.
- Calendar handlers are typed and closed by trigger kind.
- `trigger_future_cognition` no longer writes `scheduled_events`.
- Self-cognition no longer reads due future-cognition rows from
  `scheduled_events`.
- Production active-commitment due checks use `commitment_due_cognition`
  calendar runs.
- Active-commitment create/update/reschedule/cancel flows maintain calendar
  schedule state.
- Calendar execution skips stale active-commitment runs when the underlying
  memory unit is missing, no longer active, not an active commitment, owned by
  a different user, or has a different `due_at`.
- RAG recall future-action evidence reads calendar state, not
  `scheduled_events`.
- Reflection slot integration uses one composite `reflection_phase_slot`
  calendar trigger while preserving existing reflection run id contracts.
- Calendar-backed reflection phase runs map mechanically to
  `ReflectionPhaseRunIntent` and are consumed through the
  provider-compatible handler seam.
- The pure reflection phase materializer remains reusable, but the local
  reflection phase provider is not retained as a production control-plane
  fallback after universal scheduler cutover.
- No `reflection_phase_runs` collection is introduced.
- `reflection_hourly_slot` and `group_self_cognition_review` are not calendar
  trigger kinds; they may appear only as allowed actions inside the
  `reflection_phase_slot` payload contract.
- The old process-local scheduler runtime is not started by the service.
- The one-time migration script converts pending `trigger_future_cognition`
  rows, cancels pending legacy `send_message` rows, blocks unknown tools, and
  is idempotent.
- No production code reads or writes `scheduled_events` after cutover.
- All verification commands pass or any blocked command is recorded with exact
  environment reason.
- Independent code review approves the implementation.

## Risks

- Migration can strand pending future cognition if idempotency keys or legacy
  linkage are wrong.
- Cancelling legacy pending `send_message` rows can drop old direct delayed
  sends. This is intentional because delayed visible sends are not preserved.
- Commitment schedules can become stale if active-commitment write hooks miss
  a create, merge/evolve due change, lifecycle closure, or invalid due-date
  removal path.
- Stale commitment calendar runs can reopen closed or rescheduled commitments
  unless handlers re-read the memory unit and skip mismatches.
- Calendar recurrence can drift around character-local timezone and daylight
  saving transitions if UTC/local conversion is not centralized.
- Reflection daily synthesis can become partial if calendar slot execution and
  daily readiness are not coordinated.
- Reflection phase integration can duplicate work if the calendar provider and
  local provider both remain active after cutover.
- Splitting reflection hourly and group review into separate calendar triggers
  would reintroduce burst and ordering ambiguity that the short-term
  reflection phase plan deliberately removed.
- Stuck `running` rows can block work if lease expiry is not covered by tests.
- A broad scheduler abstraction can become a hidden arbitrary job runner if
  trigger-kind ownership is not enforced.

## Execution Evidence

Plan-review evidence may be recorded here before approval. Implementation
execution evidence starts only after this plan is approved and implementation
begins.

- 2026-06-04 plan refresh: reread plan registry, top-level README, HOWTO,
  development-plan references, dispatcher/self-cognition/reflection/db/action
  READMEs, current `scheduler.py`, reflection phase materializer/worker,
  future-cognition handler, self-cognition sources, RAG recall scheduled-event
  collector, DB scheduled-event helpers, active-commitment memory-unit helpers,
  service startup, config docs, and reflection phase tests.
- 2026-06-04 independent plan review findings fixed:
  - Blocker: active-commitment schedule reconciliation covered creation only
    implicitly and did not name merge/evolve due changes, lifecycle closure,
    or execution-time stale-run validation. Fixed in `Mandatory Rules`,
    `Must Do`, trigger contracts, implementation order, checklist, and
    verification.
  - Blocker: reflection phase materialization allowed ambiguous future
    precomputation even though eligibility depends on the `period_start_utc`
    monitor snapshot. Fixed by requiring boundary-time materialization and
    calendar-backed daily readiness.
  - Important: recall change surface kept the old scheduled-event collector
    path ambiguous. Fixed by adding a `calendar_runs` collector, recall agent
    and contract updates, and `ScheduledEventCollector` grep verification.
  - Blocker: old process-local scheduler decommission did not name
    `scheduler.py`, `dispatcher.pending_index`, runtime facade scheduled-event
    helper access, or the old recall collector file in the change surface.
    Fixed by adding explicit deleted/decommissioned module entries and
    maintenance-only legacy scheduled-event access.
  - Important: change surface named a nonexistent dispatcher test file.
    Fixed by replacing it with the existing scheduler/service event logging
    tests that currently cover the affected startup and pending-index seams.
  - Important: the `Overdesign Guardrail` did not use the project-required
    actual-problem/minimal-change/ownership/rejected-complexity/evidence
    shape. Rewritten to match the plan contract.
  - Important: `Execution Evidence` said not to pre-fill drafting evidence
    while `Independent Plan Review` required plan-review findings to be
    recorded before approval. Fixed by separating plan-review evidence from
    implementation execution evidence.
- 2026-06-04 user-directed plan update: added
  `src/kazusa_ai_chatbot/calendar_scheduler/README.md` as a required
  ICD-style module README. The requirement is now present in `Mandatory Rules`,
  `Must Do`, `Contracts And Data Shapes`, `Change Surface`,
  `Implementation Order`, `Progress Checklist`, `Verification`,
  `Independent Code Review`, and `Acceptance Criteria`.
