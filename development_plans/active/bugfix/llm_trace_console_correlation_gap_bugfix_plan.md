# trace correlation retrieval and web-availability gap bugfix plan

Date: 2026-08-09

## Summary

- Goal: make a value copied from the Control Console sufficient to start a
  deterministic, protected diagnostic workflow that identifies the parent
  `llm_trace_id`, retrieves the available `global_user_id` and Cognition
  invocation evidence, and follows exact links to action, background-work,
  calendar, and future-cognition records.
- Status: draft
- Scope boundary: the protected trace lane, its read-only diagnostic export
  path, the trace-debug and diagnostic skills, and forward correlation fields
  written by the live action/background/calendar/self-cognition path. The
  Control Console remains a read-only projection with its current redaction
  boundary.
- Change direction: record the web-console source surface before resolving an
  opaque value; resolve parent traces only through typed exact candidates;
  export a bounded correlation manifest; add forward source-trace links to
  durable action and scheduling records; and bind a protected trace to future
  self-cognition execution. Historical rows remain immutable and report
  missing capture explicitly.
- Acceptance state: pending user approval. This plan creation does not start
  implementation or authorize production-code edits.

## Scope And Change Direction

### Evidence-only current state

The current implementation establishes the following facts:

1. `POST /api/debug-chat` returns a console request id, a response
   `delivery_tracking_id`, and a bounded cognition graph. The browser visibly
   renders the tracking id and the graph's closed `Run reference` disclosure.
2. The graph run reference is not a declared protected trace identifier. It
   can be the delivery tracking id or another generic graph correlation value.
   A copied value such as `d63003933a924c03a258fcf9d891e6b5` therefore has no
   safe type inference from its shape.
3. The event monitor can display generic `correlation_id`, `run_id`,
   `trigger_id`, and `attempt_id` values. The event projection does not expose
   a canonical `llm_trace_id` field as a dedicated browser field.
4. The user directory exposes platform account identifiers and display names,
   but intentionally excludes `global_user_id`. Calendar projections exclude
   schedule and run ids. Background-work projections exclude job ids,
   accepted-task ids, action-attempt ids, and ownership identifiers.
5. Protected `llm_trace_runs` already retain the turn's `global_user_id`,
   platform/channel metadata, platform message id, and delivery tracking id
   when trace capture records the run. Protected Cognition failure capsules
   retain `cognition_invocation_id` only for captured failure/degraded cases.
6. `src/scripts/export_llm_trace.py` exports protected trace rows, failure
   capsules, sanitized event rows, and conversation rows, but it does not
   export action-attempt, background-work, calendar, or future self-cognition
   companion records.
7. Existing action-attempt, background-work, and calendar records carry
   links among themselves, such as `source_action_attempt_id`,
   `calendar_schedule_id`, and `calendar_run_id`, but the live action
   execution path does not persist a canonical source `llm_trace_id` on each
   durable companion record.
8. The self-cognition state builder does not currently carry a protected
   `llm_trace_id` into its Cognition V2 invocation. A future-cognition run can
   therefore have a durable `calendar_run_id` while lacking a corresponding
   protected future trace run or failure capsule.
9. The existing trace-debug skill describes protected trace resolution but
   does not yet define the complete web-surface availability matrix or the
   cross-subsystem companion-id retrieval contract.

These facts are evidence from the current source, ICDs, and tests. They are
separate from the implementation decisions below.

### Fixed implementation direction

The implementation uses two connected workstreams:

- Historical and current diagnostic retrieval: create a typed web-anchor
  resolver and a bounded `trace_correlation_manifest.v1` exporter. It reads
  the protected trace run first, then follows only explicit durable links.
- Forward observability propagation: add a source-trace field to durable
  action-attempt, background-work, and calendar records; propagate the field
  through direct and background future-cognition scheduling; and create/bind a
  protected trace for each future self-cognition execution.

The canonical meaning of “future cognition id” is fixed as follows:

- `calendar_schedule_id` identifies the durable scheduled definition.
- `calendar_run_id` identifies the concrete future-cognition source run and is
  the canonical future-cognition scheduling id.
- The future execution's own `llm_trace_id` identifies its protected model
  trace when capture is enabled.
- Its `cognition_invocation_id` identifies a captured Cognition V2 failure or
  degraded invocation when a failure capsule exists.

The implementation never introduces a second alias named `future_cognition_id`.

The web console is not expanded in this plan. Its current redaction contract
continues to hide global, operational, and scheduler identifiers. The web
availability matrix becomes explicit in the skills and ICDs so the agent can
distinguish “not shown by the web” from “not captured by the runtime.” A
separate security-approved console disclosure plan is required for any future
browser panel that displays hidden identifiers.

## Mandatory Skills

- `development-plan`: read the lifecycle registry before implementation and
  enforce the approved-plan and explicit-production-change gates.
- `llm-trace-debug`: own the copied-anchor classification, protected trace
  resolution, invocation selection, and evidence handoff workflow.
- `database-data-pull`: use the project-maintained read-only database export
  boundary and write bounded diagnostic artifacts under
  `test_artifacts/diagnostics`.
- `debug-llm`: review the protected trace and correlation manifest as evidence
  artifacts, keeping observed facts separate from diagnosis and proposal.
- `control-console-web-development`: preserve the console projection and
  redaction contracts while documenting the web-source matrix.
- `skill-creator`: update and validate the existing trace-debug skill without
  adding redundant auxiliary skill documentation.
- `local-llm-architecture`: review the trace-context ownership boundary across
  Cognition, action execution, background work, calendar scheduling, and
  self-cognition.
- `py-style`: apply the repository Python policy to the new exporter, database
  helpers, and runtime propagation changes.
- `test-style-and-execution`: add and run deterministic contract tests and
  keep any live database or live LLM evidence explicitly separated.
- `python-venv`: use `venv\\Scripts\\python` for Python validation.

## Mandatory Rules

- Read `development_plans/README.md`, `README.md`, `docs/HOWTO.md`, and every
  directly affected subsystem README before implementation. Do not read
  `.env`.
- Preserve every pre-existing worktree edit observed at implementation
  handoff. Unrelated files are outside this plan's implementation ownership.
- Keep the trace lane protected. The manifest may contain stable diagnostic
  identifiers, but it must not copy raw prompts, raw model output, full
  conversation bodies, embeddings, secrets, adapter payloads, or unbounded
  action/job payloads.
- Keep the console a projection boundary. Do not add hidden identifiers to
  Control Console routes, projections, HTML, CSS, or JavaScript in this plan.
- Resolve identifiers through typed, allowlisted fields and exact matches.
  Zero matches produce `not_found`; multiple parent candidates produce
  `ambiguous`; missing historical fields produce `not_captured`.
- Never select the newest row to break a tie. Never use timestamp proximity,
  user display name, message text similarity, or hexadecimal shape as a
  cross-subsystem join.
- Use the named database maintenance boundary for MongoDB reads. The skill
  and exporter must not open MongoDB clients or issue ad hoc queries directly.
- Preserve capture-mode semantics. `off` yields `not_available`; metadata
  mode yields only the fields retained by metadata capture; full mode does
  not authorize raw content in the correlation manifest.
- Trace-write failures must not change response delivery, action semantics,
  background scheduling, or future-cognition behavior. They produce bounded
  diagnostic availability status and existing operational telemetry.
- Do not add an LLM call, prompt instruction, deterministic semantic gate, or
  RCA conclusion to the runtime path.
- Use a big-bang canonical field vocabulary. Do not add compatibility aliases,
  parallel collection names, or a fallback mapper for old runtime call shapes.
  Historical documents are read as historical documents and are reported as
  `not_captured` when the new source field is absent.

## Must Do

1. Freeze the protected correlation contract in a new
   `trace_correlation_context.v1` helper owned by `llm_tracing`. The contract
   must distinguish the current trace from its source records and use these
   exact durable fields:

   - `source_llm_trace_id` on action-attempt, background-work, and calendar
     records;
   - `parent_llm_trace_id` on a child trace run created by background result
     delivery or future self-cognition;
   - `source_calendar_run_id` on a future self-cognition trace run;
   - `source_background_work_job_id` on a trace created to deliver a
     background-work result.

   Empty source fields are valid only for historical rows or non-persisting
   deterministic previews. Persisted live action/background/calendar rows
   created after cutover must carry the applicable source trace.

2. Extend the live action execution path so the owning state supplies its
   canonical `llm_trace_id` to every action-attempt write. Propagate that
   source trace into:

   - `self_cognition_action_attempts`;
   - direct `background_work_jobs` created by an accepted task action;
   - direct `calendar_schedules` and `calendar_runs` created by
     `trigger_future_cognition`;
   - `future_speak` schedules created from a background-work job.

   Preserve the existing `source_action_attempt_id`, `accepted_task_id`,
   schedule id, and run id links. The new source trace field supplements those
   links and does not replace their ownership semantics.

3. Add the required bounded indexes and named read helpers for exact
   correlation lookups. The read contract must support:

   - one parent `llm_trace_runs` row by `trace_id`;
   - all linked conversation rows by `llm_trace_id`, projected without body or
     embedding content;
   - Cognition failure capsules and their
     `cognition_invocation_id` values by trace;
   - action attempts by `source_llm_trace_id`;
   - background jobs by `source_llm_trace_id`,
     `source_action_attempt_id`, or exact `job_id`;
   - calendar schedules and runs by `source_llm_trace_id`, exact action
     attempt, exact schedule id, or exact run id;
   - child trace runs by `parent_llm_trace_id` plus
     `source_background_work_job_id` or `source_calendar_run_id`.

4. Create `src/scripts/export_trace_correlation_manifest.py`. The command
   must accept one typed source anchor using this fixed interface:

   ```text
   python -m scripts.export_trace_correlation_manifest \
       --identifier <copied-value> \
       --source-surface <enum> \
       --output test_artifacts/diagnostics/<manifest>.json
   ```

   The supported `source-surface` values are:

   - `web_debug_tracking`;
   - `web_cognition_run_reference`;
   - `web_event_request`;
   - `web_event_run`;
   - `web_event_trigger`;
   - `web_event_attempt`;
   - `web_debug_request_id`;
   - `web_debug_platform_message_id`;
   - `protected_llm_trace_id`;
   - `protected_cognition_invocation_id`;
   - `protected_global_user_id`;
   - `protected_action_attempt_id`;
   - `protected_background_work_job_id`;
   - `protected_accepted_task_id`;
   - `protected_calendar_schedule_id`;
   - `protected_calendar_run_id`;
   - `unknown`.

   The command must also support `--trace-id` as an explicit parent-trace
   override and `--cognition-invocation-id` as the existing bounded capsule
   selector. The source-surface value remains in the manifest even when an
   explicit trace id is supplied.

5. Refactor `export_llm_trace.py` and the new manifest exporter to share one
   strict candidate resolver. A candidate is accepted only when its typed
   field produces exactly one parent trace. A bare opaque value, including
   `d63003933a924c03a258fcf9d891e6b5`, must remain unclassified when the
   source surface is `unknown`; the resolver must never infer `llm_trace_id`
   from a 32-character hexadecimal shape.

   The existing dialog-text, delivery-tracking, and platform-message lookup
   paths must return all candidate trace ids before deciding. The resolver
   must preserve zero and multiple candidates in the manifest and must not
   silently choose a newest row.

6. Define `trace_correlation_manifest.v1` with these bounded sections:

   - `input`: supplied value, source surface, and explicit trace/invocation
     overrides;
   - `parent_trace`: resolution status, trace id, capture availability, and
     protected run metadata;
   - `identifiers`: one entry per canonical id with value, status, owner,
     source collection/field, and exact evidence reference;
   - `joins`: bounded match counts and the exact relation used for every
     companion collection;
   - `availability`: `rendered`, `api_only`, `protected_only`,
     `not_available_from_web`, `not_captured`, or `not_applicable`;
   - `unresolved`: typed reasons for zero, multiple, expired, or unavailable
     evidence.

   The manifest must include `global_user_id`, all selected
   `cognition_invocation_id` values, action-attempt ids, background job ids,
   accepted-task ids, calendar schedule/run ids, source/parent trace ids, and
   future execution trace ids when exact evidence exists.

7. Add a web-console availability matrix to
   `.agents/skills/llm-trace-debug/SKILL.md`,
   `src/control_console/README.md`, and the trace runbook. The matrix must
   label each identifier as rendered in the browser, available only in an API
   payload/request, protected-only, or absent from the current console. It
   must state the next exact retrieval route for every hidden identifier.

8. Update the database-pull and debug-LLM skills to hand off the correlation
   manifest as a protected evidence artifact. The workflow must require an
   agent to record the copied value and source surface before querying, then
   inspect `parent_trace`, `identifiers`, `joins`, and `unresolved` before
   forming any failure diagnosis.

9. Add forward self-cognition trace binding. For each live self-cognition
   case, allocate and bind a protected `llm_trace_id`, ensure the trace run
   with its `global_user_id`, platform/message metadata, and
   `source_calendar_run_id` when applicable, and finalize the trace without
   making trace persistence a delivery prerequisite. The existing failure
   capsule must then receive the bound trace id and retain its normal
   `cognition_invocation_id` behavior.

10. Propagate background-result ownership into the child trace created for
    accepted-task result delivery through `source_background_work_job_id` and
    `parent_llm_trace_id`. Preserve the distinction between the original
    action trace, the background job, and the later result-delivery trace.

11. Add deterministic tests for the web-source matrix, strict resolution, the
    bare opaque example, zero/multiple candidates, capture-mode availability,
    exact companion joins, historical `not_captured` rows, direct future
    cognition, `future_speak` background scheduling, and child future/result
    traces. Add redaction assertions that ensure no protected raw content or
    hidden identifier leaks into the console projections or sanitized event
    log.

12. Produce one parent-authored evidence review artifact for the supplied
    anchors `0a04c1db64e24dd7870cd3d865179f37`,
    `a1a573b590a3494786c4edebdee55342`, and
    `d63003933a924c03a258fcf9d891e6b5`. The review must separate observed
    records, availability status, unresolved joins, and any later diagnostic
    interpretation.

## Deferred

- Do not add a Control Console route, panel, browser label, or API field for
  `global_user_id`, `llm_trace_id`, `cognition_invocation_id`, job ids,
  action-attempt ids, schedule ids, or run ids.
- Do not change current Control Console redaction of internal global ids,
  operational ids, prompts, raw model output, raw messages, embeddings, or
  scheduler payloads.
- Do not expose protected trace rows through `/chat`, health, adapter
  responses, event-log payloads, or public browser APIs.
- Do not backfill historical action, background, calendar, or self-cognition
  rows. A missing forward field is reported as `not_captured`.
- Do not change the trace retention period, capture-mode defaults, failure
  capsule semantics, Cognition V2 prompts, action decisions, delivery
  behavior, background worker semantics, calendar trigger roster, or future
  cognition meaning.
- Do not treat `event_log_events` or `event_log_snapshots` as durable source
  truth for action, job, schedule, run, or trace state.
- Do not add timestamp-based joins, newest-row selection, hexadecimal regex
  classification, dialog-text similarity matching, or user-display-name
  inference.
- Do not add an LLM call, automatic RCA, or semantic diagnosis to the export
  or runtime path.
- Do not modify the unrelated dirty files listed in `Mandatory Rules`.

## Target State

The diagnostic flow is:

```text
copied console value + source surface
    -> strict typed anchor resolution
    -> unique parent llm_trace_id
    -> protected trace run and conversation evidence
    -> cognition invocation capsules when captured
    -> exact action-attempt / background-job / calendar joins
    -> future child trace by calendar_run_id or background_work_job_id
    -> bounded correlation manifest + separate trace export
```

The operator-facing availability contract is:

| Identifier | Current web availability | Canonical protected or durable source |
|---|---|---|
| `platform_user_id` | Debug input and bounded account projection | `conversation_history`, `user_profiles` |
| `platform_channel_id` | Debug input/request context; not a result identifier | `llm_trace_runs`, `conversation_history` |
| `platform_message_id` | Debug request/API payload only | `conversation_history`, `llm_trace_runs` |
| `console request_id` | API response only; not the rendered debug result | Control Console request envelope |
| `delivery_tracking_id` | Rendered in Debug Chat response metadata | `llm_trace_runs`, `conversation_history`, delivery records |
| graph `Run reference` | Rendered in the closed cognition graph disclosure | Generic graph correlation; type requires resolver evidence |
| event `correlation_id` | Rendered in event detail when present | Sanitized event row; not automatically an LLM trace |
| event `run_id` / `trigger_id` / `attempt_id` | Rendered as generic event detail when present | Sanitized event row and owning durable record |
| `llm_trace_id` | Not exposed by the current web console | Protected `llm_trace_runs`, `llm_trace_steps` |
| `global_user_id` | Not exposed by the current web console | Protected trace run and user profile |
| `cognition_invocation_id` | Not exposed by the current web console | Protected Cognition failure capsule |
| `background_work_job_id` | Hidden by the current background projection | `background_work_jobs` |
| `accepted_task_id` | Hidden by the current background projection | `background_work_jobs`, accepted-task records |
| `action_attempt_id` | Hidden by the current background/calendar projections | `self_cognition_action_attempts` and action results |
| `calendar_schedule_id` | Hidden by the current calendar projection | `calendar_schedules` |
| `calendar_run_id` | Hidden by the current calendar projection | `calendar_runs`; future source case |
| future execution `llm_trace_id` | Not exposed by the current web console | Protected child trace run |

An agent can therefore explain both dimensions separately: the browser may
provide a usable anchor while the protected runtime lacks a captured companion
field, or the runtime may contain a protected companion that the browser never
discloses.

## Change Surface

### Delete

- None.

### Modify

- `development_plans/README.md` — register this draft under active bugfix
  plans.
- `.agents/skills/llm-trace-debug/SKILL.md` — add typed web-source input,
  availability matrix, exact companion workflow, and manifest handoff.
- `.agents/skills/database-data-pull/SKILL.md` — document the protected
  correlation-manifest export boundary.
- `.agents/skills/debug-llm/SKILL.md` — require evidence-first manifest review
  and explicit unresolved-join reporting.
- `src/control_console/README.md` — document the current rendered/API-only/
  hidden identifier contract without changing the web surface.
- `src/kazusa_ai_chatbot/llm_tracing/README.md` — document source/parent trace
  fields, future self-cognition trace binding, and manifest ownership.
- `src/kazusa_ai_chatbot/db/README.md` — document the read-only maintenance
  correlation helpers and additive diagnostic fields/indexes.
- `src/kazusa_ai_chatbot/background_work/README.md` — document source-trace
  ownership and background/result-trace separation.
- `src/kazusa_ai_chatbot/calendar_scheduler/README.md` — document source-trace
  fields while preserving scheduler payload and trigger boundaries.
- `src/kazusa_ai_chatbot/self_cognition/README.md` — document future
  self-cognition trace creation and `calendar_run_id` terminology.
- `src/scripts/export_llm_trace.py` — share the strict no-newest-only parent
  resolver with the correlation manifest.
- `src/scripts/README.md` — register the new read-only manifest command.
- `src/kazusa_ai_chatbot/db/script_operations.py` — add allowlisted,
  bounded exact-correlation read helpers.
- `src/kazusa_ai_chatbot/db/bootstrap.py` and the owning persistence modules —
  add indexes required for source-trace and child-trace lookups.
- `src/kazusa_ai_chatbot/llm_tracing/__init__.py` and the new correlation
  helper — persist and validate bounded trace-source metadata.
- `src/kazusa_ai_chatbot/action_spec/attempt_ledger.py` and
  `src/kazusa_ai_chatbot/action_spec/execution.py` — carry the canonical live
  source trace into persisted action attempts.
- `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py` and
  `src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py` — carry
  source-trace and parent-owner metadata into durable records.
- `src/kazusa_ai_chatbot/background_work/models.py`, `jobs.py`, and
  `subagent/future_speak.py` — validate and propagate background source trace
  and result ownership.
- `src/kazusa_ai_chatbot/calendar_scheduler/models.py` and repository call
  sites — preserve source trace metadata on schedules and runs.
- `src/kazusa_ai_chatbot/self_cognition/runner.py` and `worker.py` — allocate,
  bind, finalize, and source-link future self-cognition traces.
- `src/kazusa_ai_chatbot/service.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`, and
  `src/kazusa_ai_chatbot/brain_service/post_turn.py` — pass the canonical
  trace context across live action and background-result ownership boundaries.
- Directly affected deterministic test files under `tests/` for tracing,
  action specs, background work, calendar scheduling, self-cognition,
  post-turn lifecycle, and Control Console redaction.

### Create

- `src/kazusa_ai_chatbot/llm_tracing/correlation.py` — typed bounded
  source/parent correlation contract and validation helpers.
- `src/scripts/export_trace_correlation_manifest.py` — read-only protected
  manifest exporter.
- `tests/test_trace_correlation_manifest.py` — resolver, manifest, status,
  and redaction contract tests.
- `tests/fixtures/trace_correlation_manifest_cases.json` — synthetic exact,
  ambiguous, absent, historical, direct-future, and background-future cases
  without real user or prompt data.
- `test_artifacts/diagnostics/trace_correlation_manifest_review_20260809.md`
  and bounded JSON outputs during verification; these remain diagnostic
  artifacts rather than source contracts.

### Keep

- Control Console routes, projections, static HTML/CSS/JavaScript, and its
  intentional exclusion of hidden identifiers.
- Protected `llm_trace_runs` and `llm_trace_steps` as the source of model-stage
  evidence.
- Existing `cognition_invocation_id` failure-capsule behavior and capture-mode
  retention rules.
- Existing action-attempt, background-work, calendar, and self-cognition
  ownership boundaries and stable ids.
- Existing event-log sanitization and the separation between event telemetry
  and durable state.

## Agent Autonomy Boundaries

The implementing agent may choose helper decomposition, private function
names, bounded JSON field ordering, fixture values, and the order of focused
verification commands within the fixed scope.

The implementing agent must obtain explicit approval and a production-change
command before editing runtime Python. It must keep the canonical field names,
source-surface enum, status vocabulary, exact-join rules, console redaction
boundary, and historical `not_captured` behavior fixed.

The implementing agent has no authority to add a browser disclosure surface,
alter semantic prompts, change action/calendar/background behavior, add a
compatibility mapper, backfill existing rows, relax redaction, or infer a
trace from an opaque value. Any such request requires a new or amended plan.

## Confirmed Decisions

| Topic | Decision |
|---|---|
| Web console | Keep the current redacted projection; document availability and use protected export for hidden ids. |
| Parent trace resolution | Require a typed source surface and one exact parent candidate; preserve zero and multiple results. |
| Bare opaque ids | Treat them as `unknown` until the operator supplies the source surface; never infer from length or hex characters. |
| Global user id | Read from protected trace/user records; do not disclose it in the web console. |
| Background id | Report both `background_work_job_id` and `accepted_task_id` with separate ownership labels. |
| Future cognition id | Use `calendar_run_id` for the scheduled source run; use the child `llm_trace_id` for execution evidence. |
| Invocation id | Use `cognition_invocation_id` only when a protected failure/degraded capsule captured it. |
| Historical data | Do not backfill; report absent forward fields as `not_captured`. |
| Evidence artifact | Keep the correlation manifest bounded and identifier-focused; keep raw protected trace export separate. |
| Runtime impact | Correlation write/read failures degrade evidence availability and never alter semantic or delivery outcomes. |

## Cutover Policy

Overall strategy: bigbang for forward writers, immutable historical handling
for existing rows.

| Area | Policy | Instruction |
|---|---|---|
| Copied-anchor input | bigbang | Every diagnostic invocation records a typed source surface. |
| Trace resolution | bigbang | All resolver paths use the strict shared candidate resolver; no newest-only path remains. |
| Action/background/calendar writers | bigbang | Live persisted records receive the canonical source trace field from the owning state. |
| Future self-cognition | bigbang | Each live case receives a bound protected trace and source calendar-run metadata. |
| Historical rows | immutable | Missing forward fields remain absent and produce `not_captured`; no migration or timestamp reconstruction occurs. |
| Console disclosure | unchanged | Existing UI/API projections and redaction assertions remain authoritative. |
| Manifest schema | additive | New `trace_correlation_manifest.v1` is written beside, not inside, the existing raw protected trace export. |

Rollback is a source revert for runtime writers and exporter behavior. No
rollback script, data rewrite, or field-clearing operation is permitted. Newly
written additive diagnostic fields remain harmless historical data after a
source revert; the manifest reports them only when present.

## Contracts And Data Shapes

The implementation must use the following status vocabulary:

- `confirmed`: exact evidence found and the relation is valid;
- `not_found`: the exact query found no row;
- `ambiguous`: the supplied anchor maps to more than one parent candidate;
- `not_captured`: the owning historical row exists but lacks the forward
  correlation field;
- `not_available`: capture or retention prevents the protected source from
  existing;
- `not_available_from_web`: the value is outside the current console surface;
- `not_applicable`: the relation does not apply to the run path.

The protected `trace_correlation_context.v1` contract is identifier-only and
bounded. It carries source ownership rather than semantic payload:

```json
{
  "schema_version": "trace_correlation_context.v1",
  "source_llm_trace_id": "",
  "source_episode_id": "",
  "source_background_work_job_id": "",
  "source_calendar_run_id": ""
}
```

The trace-run metadata uses `parent_llm_trace_id`,
`source_background_work_job_id`, and `source_calendar_run_id` as top-level
protected fields. Action-attempt, background-job, schedule, and run records
use `source_llm_trace_id` as the exact parent link. These fields never enter a
model-facing source packet, prompt, dialog result, or Control Console graph
detail.

## Runtime Or Resource Constraints

- Live response and background delivery paths remain bounded and inspectable.
- Correlation writes use the existing trace-write timeout and must not block
  normal delivery beyond that bound.
- Manifest queries use explicit per-collection limits and projections; the
  manifest cannot become a database dump.
- Protected trace retention remains governed by `DEBUG_LOG_TTL_DAYS`.
- Test artifacts containing protected identifiers stay under
  `test_artifacts/diagnostics` and are not emitted in chat output.
- No `.env` inspection is part of implementation or verification.

## Verification

The implementing agent must run the following focused checks with
`venv\\Scripts\\python` after approval and implementation:

1. Validate the new and modified Python modules with `py_compile` and run
   `git diff --check`.
2. Run the skill validator on `.agents/skills/llm-trace-debug` and the
   deterministic skill/export tests:

   ```powershell
   venv\Scripts\python -m pytest `
     tests/test_llm_tracing.py `
     tests/test_llm_trace_export.py `
     tests/test_llm_trace_skill_contract.py `
     tests/test_trace_correlation_manifest.py -q
   ```

3. Run the focused runtime correlation tests:

   ```powershell
   venv\Scripts\python -m pytest `
     tests/test_action_spec_attempt_ledger.py `
     tests/test_action_spec_future_cognition.py `
     tests/test_action_spec_results.py `
     tests/test_background_work_jobs.py `
     tests/test_background_work_future_speak.py `
     tests/test_calendar_scheduler_models.py `
     tests/test_calendar_scheduler_repository.py `
     tests/test_self_cognition_integration.py `
     tests/test_self_cognition_event_logging.py -q
   ```

4. Run the Control Console contract and redaction tests. Because this plan
   does not change the rendered web surface, deterministic route/static
   checks are sufficient and no browser sign-off is claimed:

   ```powershell
   venv\Scripts\python -m pytest `
     tests/test_control_console_web_surface.py `
     tests/test_control_console_kazusa_client.py `
     tests/test_control_console_cognition_graph.py `
     tests/test_control_console_redaction.py `
     tests/test_console_lookup_limits.py -q
   ```

5. Run the existing skill quick validator and inspect its output. Run no live
   LLM case as part of this plan; the change is diagnostic plumbing and
   deterministic correlation. A bounded live MongoDB smoke is required when
   the configured database is available and must query only the three supplied
   anchors one at a time. If the database is unavailable, record the exact
   unavailable status in the evidence artifact rather than substituting a
   mock result for live evidence.

6. Inspect each generated manifest before starting the next live anchor. The
   inspection must verify the copied input surface, parent resolution status,
   all identifier statuses, exact join provenance, and absence of raw prompt,
   response, message-body, embedding, and secret values.

7. Obtain an independent code review focused on source ownership, strict
   ambiguity handling, capture-mode behavior, future self-cognition trace
   binding, and the unchanged Control Console redaction boundary.

## Acceptance Criteria

- The web availability matrix names every requested identifier and states
  whether it is rendered, API-only, protected-only, or absent from the current
  console.
- The skill requires a source surface for copied values and does not classify
  `d63003933a924c03a258fcf9d891e6b5` as a trace from its hexadecimal shape.
- Every parent-trace resolver path preserves zero and multiple candidates;
  no path silently selects the newest row.
- A unique protected parent trace yields its retained `global_user_id`,
  platform/message/tracking metadata, and all captured Cognition invocation
  ids with collection/field provenance.
- A forward live action path yields an exact action-attempt to source-trace
  relation; background and calendar records retain both their existing stable
  links and the source trace.
- A direct future-cognition path yields exact
  `calendar_schedule_id` -> `calendar_run_id` -> child trace relationships.
- A `future_speak` path yields exact source-trace -> background-job -> calendar
  schedule/run relationships, and a background result trace identifies its
  source job without confusing it with the original action trace.
- A self-cognition failure/degraded case has a bound protected trace and a
  `cognition_invocation_id` capsule when capture mode and retention permit it.
- Historical rows without the new fields are reported as `not_captured`; no
  timestamp, newest-row, text-similarity, or shape heuristic fills the gap.
- The manifest contains identifiers and bounded evidence references only. No
  raw prompt, raw output, full message, embedding, secret, or unrestricted
  payload reaches the manifest, console, or event log.
- Existing Control Console redaction, graph, event, calendar, background, and
  user-directory tests remain green.
- The three supplied anchors have individually inspected diagnostic artifacts
  that separate facts, availability gaps, unresolved joins, and later
  interpretation.
- The focused tests, skill validation, Python compilation, diff check, and
  required independent review pass before the plan can move to `completed`.

## Progress Checklist

- [ ] Baseline worktree and existing trace-skill diff recorded.
- [ ] Web-console availability matrix frozen in the skill and ICDs.
- [ ] Correlation context and durable source-field contracts implemented.
- [ ] Strict shared resolver and correlation manifest implemented.
- [ ] Future self-cognition trace binding implemented.
- [ ] Deterministic runtime, exporter, skill, and redaction tests pass.
- [ ] Supplied-anchor live smoke artifacts inspected one at a time.
- [ ] Independent code review completed.
- [ ] Execution evidence and residual `not_captured` gaps recorded.
- [ ] User-approved sign-off received before lifecycle completion.

## Execution Evidence

This draft is based on the current Control Console, protected trace, database,
background-work, calendar, self-cognition, exporter, skill, and test contracts
reviewed on 2026-08-09. No production implementation or live database query is
performed while the plan is in `draft`.

At implementation handoff, record:

- the pre-handoff `git status --short` and explicitly owned file set;
- focused test and compilation output;
- one manifest and one review record per supplied anchor;
- exact unavailable/expired/not-captured causes;
- protected artifact hashes when artifacts are copied or preserved;
- the independent review result and any accepted residual gaps.

## Independent Plan Review

Before changing `Status` to `approved`, an independent reviewer must verify
that the plan distinguishes web availability from runtime capture, names exact
joins for every requested id, preserves the no-newest-only rule, and does not
silently authorize a Control Console privacy expansion. The reviewer records a
PASS or blocking findings in a diagnostic artifact; blocking findings require
plan revision before approval.

## Independent Code Review

After implementation and focused verification, a separate reviewer must inspect
the diff and generated manifest contracts. The review must confirm that source
trace fields are written by the owning runtime boundaries, self-cognition binds
the trace before Cognition V2 begins, historical rows are not backfilled, and
hidden identifiers remain absent from web and event-log projections.

## Execution Handoff

This plan is executable only after the user approves it and explicitly commands
the production implementation. The implementing agent must then re-read the
plan, lifecycle registry, required skills, current worktree status, and all
directly affected source/tests before editing. A draft plan is a discussion
artifact and is not an implementation authorization.
