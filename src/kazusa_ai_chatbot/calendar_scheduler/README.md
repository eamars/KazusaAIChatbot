# Calendar Scheduler Interface Control Document

## Document Control

- ICD id: `CALENDAR-SCHEDULER-ICD-001`
- Owning package: `kazusa_ai_chatbot.calendar_scheduler`
- Interface boundary: runtime packages and maintenance scripts -> typed
  calendar schedules and due runs
- Storage owner: `calendar_schedules` and `calendar_runs`
- Stage status: Stage 2 package, schema, repository, migration, and focused
  tests are present. Service lifecycle wiring, production action-spec cutover,
  RAG recall cutover, reflection worker cutover, and old scheduler deletion are
  planned later-stage work and are not wired by this package slice.

## Owning package

`kazusa_ai_chatbot.calendar_scheduler` owns durable calendar schedule/run
document builders, recurrence calculation, calendar collection repository
operations, lease-aware worker dispatch, active-commitment due reconciliation
helpers, reflection phase run mapping, and the migration helper surface that
moves reviewed legacy rows into calendar rows.

The package schedules typed Kazusa triggers. It is not a generic dispatcher,
adapter sender, arbitrary callback runner, or MongoDB command queue. RAG,
cognition, dialog, reflection, persistence, adapter delivery, and legacy
operator migration keep their own semantic ownership.

## Interface boundary

The calendar answers one deterministic question: when should a typed Kazusa
internal trigger become due? Owning subsystems re-read current state at
execution time and decide what the current semantic action should be.

Current Stage 2 boundaries:

- runtime callers may build schedules/runs through `models.py`;
- runtime or later lifecycle wiring may claim and finish runs through
  `repository.py` and `worker.py`;
- active-commitment write sites may call `handlers.py` to reconcile one due
  schedule, but those write sites are not wired in Stage 2;
- reflection code may convert `ReflectionPhaseRunIntent` values to calendar
  runs through `reflection_phase.py`, but the reflection worker is not cut over
  in Stage 2;
- operator migration uses the script under `src/scripts`, not the calendar
  runtime repository, to read or mutate legacy `scheduled_events`.

## Public Interfaces

`models.py` exposes:

- trigger constants:
  `future_cognition`, `commitment_due_cognition`,
  `reflection_phase_slot`, and `recurring_self_check`;
- schedule statuses: `active`, `paused`, `completed`, and `cancelled`;
- run statuses: `pending`, `running`, `completed`, `failed`, `cancelled`,
  and `skipped`;
- schema constants `calendar_schedule.v1` and `calendar_run.v1`;
- `build_one_time_calendar_schedule(...)`, which returns an active one-time
  schedule with `recurrence={"kind": "once"}`, `next_run_at` equal to the due
  timestamp, `timezone="UTC"` by default, and `legacy_source=None` by default;
- `build_calendar_run_from_schedule(...)`, which creates a pending due run
  with nullable lease/result/failure fields and no delayed visible text.

`recurrence.py` exposes deterministic recurrence helpers:

- `compute_next_run_at(...)` supports `once`, `fixed_interval_seconds`, and
  `daily_local_time`;
- `compute_phase_period_offsets(...)` validates that reflection phase slots
  fit inside a period.

`repository.py` is the narrow package-owned MongoDB adapter for only
`calendar_schedules` and `calendar_runs`. It exposes idempotent schedule/run
upserts, due-run claiming, terminal run transitions, skipped transitions, and
schedule cancellation by idempotency key. It intentionally imports
`db._client.get_db` because the active calendar plan assigns durable calendar
collection mechanics to this package. It must not read or write legacy
`scheduled_events`.

`worker.py` exposes `CalendarRunHandlerRegistry` and
`run_calendar_worker_tick(...)`. The registry is closed by trigger kind. A run
with an unsupported trigger kind fails closed with
`unsupported calendar trigger kind: {kind}`. A handler result with
`status="skipped"` marks the run skipped instead of completed.

`handlers.py` exposes active-commitment helpers:

- `reconcile_active_commitment_calendar_schedule(...)` creates or cancels the
  idempotent schedule `commitment_due:{unit_id}`;
- `handle_commitment_due_cognition_run(...)` re-reads the memory unit and
  skips missing, inactive, wrong-type, wrong-user, or stale-due rows before any
  source case is built.

`reflection_phase.py` exposes mechanical reflection mapping:

- `build_reflection_phase_calendar_runs(...)` wraps
  `ReflectionPhaseRunIntent` values as `reflection_phase_slot` calendar runs;
- `calendar_run_to_reflection_phase_intent(...)` restores the original intent
  from the run payload and strips scheduler metadata by construction.

The migration script exposes:

- `build_migration_plan(...)` for deterministic dry-run planning;
- `apply_migration_plan(...)` for dry-run or apply execution using a
  migration repository adapter;
- CLI flags `--dry-run`, `--apply`, and `--output`.

## Collection Contracts

`calendar_schedules` stores durable schedule definitions. Required Stage 2
fields are:

- `schema_version`: always `calendar_schedule.v1`;
- `owner`: always `calendar_scheduler`;
- `schedule_id`: stable generated id;
- `trigger_kind`: one closed calendar trigger kind;
- `status`: one of `active`, `paused`, `completed`, or `cancelled`;
- `start_at`: first schedule timestamp;
- `next_run_at`: next materialized due timestamp;
- `recurrence`: currently `{"kind": "once"}` for one-time builders, with
  recurrence helpers supporting the approved recurring shapes;
- `payload`: trigger-owned scheduler metadata;
- `source_scope`: structural source identity for the owning subsystem;
- `idempotency_key`: unique schedule duplicate-suppression key;
- `timezone`: IANA timezone label, defaulting to `UTC` for one-time schedules;
- `legacy_source`: nullable migration provenance, defaulting to `None`;
- `created_at`, `updated_at`, and optional cancellation fields.

Bootstrap creates `calendar_schedules` and indexes:

- unique `idempotency_key` as `calendar_schedule_idempotency_unique`;
- `(status, next_run_at, trigger_kind)` as
  `calendar_schedule_status_next_trigger`.

`calendar_runs` stores due executions. Required Stage 2 fields are:

- `schema_version`: always `calendar_run.v1`;
- `owner`: always `calendar_scheduler`;
- `run_id`: stable generated id, or the reflection phase run id for phase
  slots;
- `schedule_id`, `trigger_kind`, `status`, `due_at`, `payload`,
  `source_scope`, and `idempotency_key`;
- `attempt_count`: initialized to `0`;
- `max_attempts`: schedule value or the package default;
- nullable lease fields `claimed_at`, `lease_owner`, and `lease_expires_at`;
- nullable terminal fields `completed_at`, `failed_at`, and `skipped_at`;
- nullable `result_summary`, `failure_summary`, and `legacy_source`;
- reflection slot fields `period_start_utc`, `slot_index`, and
  `offset_seconds`, nullable for non-phase runs;
- `created_at` and `updated_at`.

Bootstrap creates `calendar_runs` and indexes:

- unique `idempotency_key` as `calendar_run_idempotency_unique`;
- unique `run_id` as `calendar_run_id_unique`;
- `(status, due_at, trigger_kind)` as `calendar_run_status_due_trigger`;
- `(lease_expires_at, status)` as `calendar_run_lease_expiry_status`.

Lease semantics:

- claims require `due_at <= now`, trigger kind membership, attempt count below
  the configured maximum, and either `pending` status or `running` with an
  expired lease;
- claims use `find_one_and_update(..., ReturnDocument.AFTER)`, set
  `status="running"`, `claimed_at`, lease owner, lease expiry, `updated_at`,
  and increment `attempt_count`;
- completion and failure match `run_id`, `status="running"`, and matching
  `lease_owner`;
- terminal transitions clear lease owner and expiry to `None`;
- completion stores `result_summary`;
- failure stores `failure_summary`;
- skipped runs are terminal for the current run and carry a bounded skip
  reason through `failure_summary`.

## Trigger Contracts

The calendar trigger roster is closed:

- `future_cognition`: due run should create fresh cognition context in a later
  wiring stage. It must not send prewritten text directly.
- `commitment_due_cognition`: due run re-reads the active-commitment memory
  unit and skips stale or structurally mismatched rows. Payload is restricted
  to `unit_id`, `global_user_id`, and `due_at`; semantic memory text is not
  copied into the schedule payload.
- `reflection_phase_slot`: one composite reflection phase slot. Hourly
  reflection and group self-cognition review remain allowed payload actions,
  not calendar trigger kinds.
- `recurring_self_check`: recurring internal self-check trigger reserved by
  the Stage 2 closed roster. Later wiring owns any runtime source-case
  behavior.

The calendar roster must not include `send_message`,
`reflection_hourly_slot`, or `group_self_cognition_review`.

## Reflection Integration

Reflection phase integration uses exactly one calendar trigger kind:
`reflection_phase_slot`.

The calendar representation maps mechanically to and from
`ReflectionPhaseRunIntent`:

- calendar `run_id` equals the intent `run_id`;
- calendar `due_at`, `period_start_utc`, `slot_index`, `offset_seconds`,
  `idempotency_key`, and `source_scope` preserve the intent values;
- the full intent is stored under `payload["reflection_phase_intent"]`;
- restoring an intent reads only that payload value, so scheduler metadata such
  as leases, attempts, migration provenance, or worker result fields does not
  enter the restored intent.

`reflection_hourly_slot` and `group_self_cognition_review` may appear only as
values inside the intent payload's `allowed_actions`. They must not become
calendar trigger kinds. The calendar package must not introduce a
`reflection_phase_runs` collection or any side control plane. In Stage 2, this
mapping exists and is tested, but the reflection worker cutover remains later
planned work.

## Migration And Cutover

The one-time migration script converts reviewed legacy scheduler rows into
calendar rows. The script defaults to dry-run unless `--apply` is supplied.
Dry-run returns a bounded plan and writes nothing.

Migration planning rules:

- pending `trigger_future_cognition` rows become one-time
  `future_cognition` schedules and due runs;
- pending `send_message` rows are listed for cancellation because delayed
  visible sends are not preserved;
- terminal legacy rows are ignored;
- unknown pending tools block the plan and produce no calendar rows.

Apply rules:

- unknown pending tools block without partial writes;
- schedule rows are upserted before run rows;
- run rows are upserted before legacy rows are mutated;
- pending legacy `send_message` rows are cancelled;
- pending legacy `trigger_future_cognition` rows are marked migrated after
  calendar writes.

There is no compatibility layer, dual-write path, dual-read path, or fallback
from calendar runs to `scheduled_events` in this package. Later stages must
remove production reads and writes against the old scheduler runtime before
full cutover sign-off.

## Runtime Lifecycle, Ops, Event Logging, And Health

Stage 2 provides the worker tick function and repository transitions, but it
does not start a background service task. Later service lifecycle wiring must
own:

- calendar worker startup and shutdown;
- worker lease owner identity;
- poll interval, claim limit, lease duration, retry limit, and per-trigger
  capacity from config;
- operator health surfaces that report whether the worker is enabled, alive,
  claiming, completing, failing, or blocked;
- event logging for claim counts, completed counts, failed counts, skipped
  counts, unsupported trigger kinds, stale commitment skips, and migration
  summaries.

Event logs are observability only. They must not become the source of truth for
schedule state, run state, idempotency, lease recovery, or migration
completion. The durable collections remain the source of truth.

## Testing And Verification

Stage 2 deterministic verification covers:

- model builders and closed trigger/status contracts;
- recurrence calculation and rejection of unsupported recurrence shapes;
- repository idempotent upserts, atomic claims, `claimed_at`, terminal summary
  fields, and nullable lease clearing;
- worker dispatch, unsupported trigger failure, and skipped handler results;
- migration planning, dry-run no-write behavior, apply ordering, send
  cancellation, migrated future-cognition ids, and unknown-tool blocking;
- active-commitment schedule reconciliation and stale due-run skips;
- reflection phase mechanical mapping and absence of a side collection;
- config defaults and fail-fast positive integer validation;
- DB bootstrap collections, unique idempotency indexes, run claim indexes, and
  facade schema exports.

Required local checks for this package slice are:

```powershell
venv\Scripts\python -m pytest tests/test_calendar_scheduler_models.py tests/test_calendar_scheduler_recurrence.py tests/test_calendar_scheduler_repository.py tests/test_calendar_scheduler_worker.py tests/test_calendar_scheduler_migration.py tests/test_calendar_scheduler_active_commitments.py tests/test_calendar_scheduler_reflection_phase.py -q
venv\Scripts\python -m pytest tests/test_config.py tests/test_db.py -q
Test-Path -LiteralPath src/kazusa_ai_chatbot/calendar_scheduler/README.md
rg -n "Document Control|Owning package|Interface boundary|Public Interfaces|Collection Contracts|Trigger Contracts|Forbidden Paths" src/kazusa_ai_chatbot/calendar_scheduler/README.md
git diff --check
```

## Forbidden Paths

The calendar scheduler must not:

- schedule delayed user-visible text;
- register `send_message` as a calendar trigger kind;
- directly call adapter send APIs;
- store adapter credentials, callback paths, Python import paths, raw MongoDB
  commands, or arbitrary job payloads;
- parse raw Discord, QQ, or debug-wire syntax as a scheduler contract;
- let scheduler ids, lease fields, raw channel ids, database names, or
  migration internals enter model-facing cognition or reflection payloads;
- treat RAG evidence as persona, stance, or final wording;
- split `reflection_phase_slot` into separate calendar trigger kinds for
  `reflection_hourly_slot` or `group_self_cognition_review`;
- create a `reflection_phase_runs` collection;
- keep a compatibility layer or fallback to legacy `scheduled_events` after
  migration and cutover;
- use event logs as state, idempotency, or migration truth.
