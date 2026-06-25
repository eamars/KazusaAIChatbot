# llm trace observability and retrieval plan

## Summary

- Goal: Add turn-scoped LLM traceability so an operator can reconstruct the
  instruction handoff, stage decisions, parsed outputs, and final dialog path
  for a generated response, then teach future agents how to retrieve and
  review those traces.
- Plan class: large
- Status: completed
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `debug-llm`, `database-data-pull`,
  `skill-creator`, `development-plan`.
- Overall cutover strategy: bigbang
- Highest-risk areas: sensitive raw prompt/output capture, live-path latency,
  write amplification, hidden global instrumentation, and retrieval workflows
  that future agents cannot discover.
- Acceptance criteria: each dialog-producing turn has a stable trace id across
  conversation rows and sanitized event-log rows; metadata trace capture is
  the default; full LLM trace capture is explicit and protected; trace storage
  has database contracts and export scripts; a repo skill guides agents from a
  visible dialog to the complete trace and a human-readable LLM debug review.

## Context

The investigation of the dialog `14:30了。怎么，雪凪你难道没有表吗？` showed a
debugging gap. `conversation_history` proved the visible input and output, and
`event_log_events` proved that `dialog_generator` ran, but the event log did
not contain the stage handoff, rendered prompt, raw model output, parsed
output, or state deltas needed to reconstruct why the final text used
`14:30`.

This gap is intentional in the current Event Logging ICD: `event_log_events`
must not store raw prompts, raw model outputs, parsed model outputs, message
bodies, generated dialog, retrieved evidence text, or adapter wire payloads.
That contract should stay intact. The sanitized event log is the operational
index, not the raw trace store.

The new requirement is therefore two-part:

- add a protected LLM trace lane for prompt/output and LLM-to-LLM handoff
  reconstruction;
- add or update repo skills so future agents can retrieve the trace from the
  database and produce a readable review instead of rediscovering ad hoc
  collection names and filters.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing prompt, graph, RAG,
  cognition, dialog, resolver, or LLM-stage boundaries.
- `debug-llm`: load before creating trace review artifacts or changing LLM
  debug output workflows.
- `database-data-pull`: load before adding or using MongoDB export scripts.
- `skill-creator`: load before creating or updating repo skills.
- `development-plan`: load before moving this plan through approval,
  execution, or completion.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Do not store raw prompts, raw model outputs, parsed model outputs, final
  dialog, message bodies, evidence text, or adapter payloads in
  `event_log_events` or `/ops/*` aggregate payloads.
- Do not implement raw trace capture as a broad hidden `LLInterface`
  interceptor. Stage-owned trace recording must name the semantic stage,
  prompt role, parser result, and downstream state field it affects.
- Do not add new LLM calls for tracing or trace retrieval.
- Do not let trace write failures change chat behavior, reflection,
  self-cognition, dispatcher delivery, memory writes, or adapter sends.
- Do not expose trace retrieval through unauthenticated public service
  endpoints in this plan.
- Use `LLM_TRACE_CAPTURE_MODE=metadata` as the default capture mode. `off`
  disables trace-run and trace-step writes but still propagates `trace_id`
  through conversation and sanitized event-log rows. `full` stores raw
  prompt/output payloads in protected trace collections only.
- Use exactly two logging retention knobs:
  `AUDIT_LOG_TTL_DAYS=90` and `DEBUG_LOG_TTL_DAYS=14`. Audit retention governs
  sanitized operational/audit records. Debug retention governs high-detail
  trace/debug records. No per-collection TTL config is allowed in this plan.
- Pre-populate `AUDIT_LOG_TTL_DAYS=90` and `DEBUG_LOG_TTL_DAYS=14` in `.env`
  during implementation without reading or printing unrelated `.env` content.
- Retention means per-row `expires_at` plus MongoDB TTL indexes, not archival
  or compaction.
- Use repo-relative paths and PowerShell `-LiteralPath` for filesystem
  commands.
- Use `apply_patch` for manual edits.

## Must Do

- Propagate one stable `trace_id` for each accepted `/chat` turn and use it as
  the join key across conversation rows, event-log rows, pipeline state, LLM
  trace runs, and LLM trace steps when trace capture mode is `metadata` or
  `full`.
- Add protected trace storage separate from `event_log_events`.
- Classify every logging/debug database collection touched by this plan as
  either audit-retained or debug-retained. Do not leave a covered logging
  collection without `expires_at`.
- Preserve the Event Logging ICD privacy contract by keeping
  `event_log_events` sanitized and linking to traces only through ids, counts,
  statuses, hashes, and refs.
- Record explicit stage-owned trace steps for the live persona path:
  relevance when present, message decontextualizer, resolver cognition cycles,
  L1, L2a, L2b, L2c1, L2c2, L2d, resolver capability observations, L3 text
  surface stages, dialog generator, dialog quality, and post-turn LLM stages
  that directly affect the episode.
- Capture for each trace step the rendered system prompt, rendered human
  payload, model route, model name, raw normalized output, parsed output,
  validation result, and bounded state delta when full trace capture is
  explicitly enabled.
- Always record trace metadata needed for joins and absence diagnosis:
  trace id, step id, stage name, stage order, route name, model name, prompt
  hash, output hash, parse status, duration, token or character counts, and
  capture mode whenever `LLM_TRACE_CAPTURE_MODE` is `metadata` or `full`.
- Add deterministic export scripts that let an operator start from a visible
  dialog, message id, delivery tracking id, event correlation id, or trace id
  and pull all related rows into `test_artifacts/`.
- Add or update repo skills so future agents know the retrieval workflow and
  expected review artifact shape.
- Add tests for storage privacy, join propagation, capture-mode behavior,
  export script behavior, and skill validation.

## Deferred

- Do not add a trace dashboard or control-console UI in this plan.
- Do not add external telemetry products, OpenTelemetry, Prometheus, or new
  runtime dependencies.
- Do not add prompt rewrites or model-routing changes.
- Do not backfill old turns into the new trace collections.
- Do not expose raw trace data through `/ops/*`.
- Do not add autonomous diagnosis or an LLM trace-analysis agent.
- Do not implement retention archival beyond the bounded two-class TTL/index
  policy in this plan.
- Do not add separate retention knobs for individual logging collections,
  event families, trace stages, or raw/full capture modes.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Event log | bigbang | Add audit-class `expires_at` to event-log rows and create TTL indexes. Existing event-log rows must be deleted if expired or assigned default audit expiry before sign-off. |
| Event-log snapshots | bigbang | Add audit-class `expires_at` to snapshot rows and create TTL indexes. Existing snapshot rows must be deleted if expired or assigned default audit expiry before sign-off. |
| Trace storage | bigbang | Create new trace collections and indexes through DB bootstrap. Trace rows use debug retention. No legacy trace TTL config remains. |
| Retention config | bigbang | Use only `AUDIT_LOG_TTL_DAYS` and `DEBUG_LOG_TTL_DAYS`; do not add `LLM_TRACE_TTL_DAYS` or per-collection TTL knobs. |
| Existing logging rows | bigbang | Before sign-off, existing rows in covered logging collections must either be deleted as expired or updated with `expires_at` computed from their timestamp and the matching TTL. |
| `.env` defaults | bigbang | Pre-populate `.env` with `AUDIT_LOG_TTL_DAYS=90` and `DEBUG_LOG_TTL_DAYS=14` without inspecting unrelated secrets. |
| LLM stage code | compatible | Add explicit trace recording around existing stage-owned calls without changing prompts, parsers, or decisions. |
| Scripts | compatible | Add new export scripts without removing existing database-data-pull scripts. |
| Skills | compatible | Add a focused trace retrieval skill and cross-link existing debug/data-pull skills. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a broader compatibility layer or hidden wrapper by
  default.
- If a local area is `bigbang`, do not preserve an old retention shape, old
  unlimited logging behavior, or a per-collection TTL compatibility path.
- Any change to cutover policy requires user approval before implementation.

## Target State

For a suspicious dialog, an operator can run one retrieval command such as:

```powershell
venv\Scripts\python.exe -m scripts.export_llm_trace `
  --dialog-text "14:30了。怎么，雪凪你难道没有表吗？" `
  --output test_artifacts\llm_traces\kazusa_1430_trace.json
```

The exported artifact contains:

- matched conversation rows and delivery metadata;
- sanitized event-log rows joined by `trace_id`, `correlation_id`, or
  `run_id`;
- one trace run row;
- ordered trace step rows for every captured LLM stage;
- a summary of missing steps when a stage has metadata but full trace capture
  was disabled.

A future agent loads a repo skill, follows the workflow from visible dialog to
  trace export, and writes a human-readable review that explains which stage
  introduced or authorized the disputed claim.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Event log role | Keep `event_log_events` sanitized and use it as the trace index | The current ICD explicitly forbids raw prompt and output capture. |
| Raw trace role | Store raw prompt/output in separate protected trace collections | Debugging needs replay evidence that the event log is not allowed to contain. |
| Capture default | Use metadata capture by default and require `full` for raw prompt/output payloads | Metadata gives joinability without copying sensitive text by default. |
| Correlation | Always propagate `trace_id`; raw step payloads depend on capture mode | Even when raw capture is off, absence can be diagnosed and rows can be joined. |
| Instrumentation style | Use stage-owned trace recording, not a generic interceptor | Stage code knows semantic names, parsed output, validation, and state deltas. |
| Retrieval workflow | Create a focused repo skill for trace retrieval | The workflow spans conversation history, event logs, trace collections, export scripts, and human-readable review. |
| Review output | Keep readable assessment agent-authored | Scripts should export raw structured evidence; agents write the human review. |

## Contracts And Data Shapes

### Trace Run

Create `llm_trace_runs` for one logical turn or background episode:

```python
{
    "trace_id": str,
    "source_kind": str,
    "platform": str,
    "platform_channel_ref": str,
    "global_user_id": str,
    "conversation_user_message_id": str,
    "conversation_assistant_message_id": str,
    "delivery_tracking_id": str,
    "started_at": str,
    "completed_at": str,
    "status": str,
    "capture_mode": "off | metadata | full",
    "step_count": int,
    "missing_full_payload_count": int,
    "expires_at": str,
}
```

### Trace Step

Create `llm_trace_steps` for each stage-owned LLM call:

```python
{
    "trace_id": str,
    "step_id": str,
    "stage_name": str,
    "stage_order": int,
    "cycle_index": int | None,
    "route_name": str,
    "model_name": str,
    "prompt_hash": str,
    "output_hash": str,
    "system_prompt": str,
    "human_payload": str,
    "raw_model_output": str,
    "parsed_output": object,
    "validation_status": str,
    "state_delta": object,
    "parse_status": str,
    "duration_ms": int,
    "prompt_chars": int,
    "output_chars": int,
    "created_at": str,
    "capture_mode": "metadata | full",
    "expires_at": str,
}
```

When full capture is disabled, raw payload fields must be absent or empty and
hash/count/status fields remain available.

`llm_trace_runs.expires_at` and `llm_trace_steps.expires_at` must be set from
`DEBUG_LOG_TTL_DAYS`. The default is 14 days. A future archival or long-term
retention policy requires a separate approved plan.

### Retention Classes

Use only two retention classes:

| Retention class | Config | Applies to | Default |
|---|---|---|---:|
| Audit | `AUDIT_LOG_TTL_DAYS` | `event_log_events`, `event_log_snapshots`, future DB-backed control audit rows | 90 |
| Debug | `DEBUG_LOG_TTL_DAYS` | `llm_trace_runs`, `llm_trace_steps`, future raw/debug trace collections | 14 |

Both config values must be positive integers. Config loading must fail fast for
missing, zero, negative, or non-integer values.

The implementation must pre-populate `.env` with:

```env
AUDIT_LOG_TTL_DAYS=90
DEBUG_LOG_TTL_DAYS=14
```

The `.env` update must be narrow and must not read, print, or inspect
unrelated secret-bearing values.

### Conversation And Event Joins

Conversation rows for accepted `/chat` turns must include:

```python
{
    "trace_id": str,
}
```

The user row and assistant row for one turn must carry the same `trace_id`.
When no assistant row is produced because the graph selected silence, the user
row still carries the trace id and `llm_trace_runs.status` records the terminal
outcome.

Sanitized event-log rows may include only trace-safe fields:

```python
{
    "correlation_id": str,
    "run_id": str,
    "refs": [
        {"ref_type": "llm_trace", "ref_id": trace_id}
    ],
    "labels": {
        "trace_capture_mode": "off | metadata | full"
    }
}
```

Event-log payloads must not contain raw trace payload fields. Adding trace refs
to an event family must preserve that family's allowed payload and forbidden
data rules in the Event Logging ICD.

### Public Trace API

Add a dedicated trace module with explicit functions. Runtime stage modules
call only this public trace API; DB collection helpers remain private to the
DB package or public only through `db.script_operations` for maintenance.

The public API must support:

- creating or ensuring a trace run;
- recording a stage metadata-only step;
- recording a full stage step;
- finalizing a trace run;
- returning a best-effort write result without affecting production behavior.

The public trace API must not accept arbitrary raw dictionaries for event-log
metadata. Full trace payload arguments must be explicit by semantic role, such
as `system_prompt`, `human_payload`, `raw_model_output`, `parsed_output`, and
`state_delta`.

Public return values must mirror event logging's best-effort style:

```python
{
    "accepted": bool,
    "trace_id": str,
    "step_id": str,
    "status": "recorded | skipped | rejected | failed",
    "reason": str,
}
```

`skipped` means trace capture is `off`. `rejected` means caller input violated
the trace API contract. `failed` means storage failed or timed out. None of
these statuses may alter production response behavior.

### Export Scripts

Add operator scripts under `src/scripts/`:

- `scripts.export_llm_trace`: resolve by `--trace-id`, `--dialog-text`,
  `--delivery-tracking-id`, `--platform-message-id`, or time-bounded filters,
  then export all joined evidence.
- `scripts.export_dialog_trace_review_input`: emit a compact JSON bundle for
  `debug-llm` review artifacts from a trace export or the same lookup keys as
  `scripts.export_llm_trace`.

Scripts must write raw evidence only. They must not generate the readable
review prose.

### Skill Deliverable

Create `.agents/skills/llm-trace-debug/SKILL.md`.

The skill must instruct future agents to:

- start from visible dialog text, message id, delivery tracking id, event id,
  or trace id;
- use `database-data-pull` style safeguards and project scripts;
- export `conversation_history`, `event_log_events`, `llm_trace_runs`, and
  `llm_trace_steps`;
- inspect missing or disabled capture modes before claiming no trace exists;
- produce a `debug-llm` style review artifact under
  `test_artifacts/llm_reviews/`;
- avoid dumping raw trace content into chat unless the user explicitly asks.

Update `.agents/skills/database-data-pull/SKILL.md` to list the trace export
scripts. Update `.agents/skills/debug-llm/SKILL.md` to direct agents to use
`llm-trace-debug` when production trace data exists.

## LLM Call And Context Budget

This plan must add no new LLM calls.

Before this plan, response-path LLM call counts are unchanged by tracing:
relevance, decontextualizer, resolver cognition cycles, L3 text stages, dialog,
and post-turn LLM stages run only when their existing graph paths call them.

After this plan, response-path LLM call counts remain identical. Tracing records
inputs and outputs from existing calls only. The response-path latency budget
changes only by bounded local serialization and one best-effort trace write per
captured stage. Full capture is operator-enabled and may increase write volume;
metadata-only trace id propagation must remain cheap.

The implementation must not raise resolver cycle caps, prompt budgets, model
route budgets, or retry counts.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/llm_tracing/README.md`: trace ICD and privacy
  contract.
- `src/kazusa_ai_chatbot/llm_tracing/`: public trace recording API and
  bounded sanitization/hash helpers.
- `src/kazusa_ai_chatbot/db/llm_tracing.py`: private DB storage adapter.
- `src/scripts/export_llm_trace.py`: trace export script.
- `src/scripts/export_dialog_trace_review_input.py`: compact raw-evidence
  bundler for `debug-llm` review artifacts.
- `src/scripts/apply_logging_retention.py`: one-shot migration script that
  assigns `expires_at` or deletes expired rows in covered logging collections.
- `.agents/skills/llm-trace-debug/SKILL.md`: retrieval skill for future
  agents.
- `tests/test_llm_tracing.py`: trace API, capture modes, privacy, and
  bootstrap contract tests.
- `tests/test_llm_trace_export.py`: export script lookup and joined output
  tests.
- `tests/test_llm_trace_skill_contract.py`: skill existence and required
  retrieval workflow checks.

### Modify

- `src/kazusa_ai_chatbot/db/README.md`: add trace collection contracts.
- `src/kazusa_ai_chatbot/event_logging/README.md`: document trace refs while
  preserving raw-data exclusions.
- `src/kazusa_ai_chatbot/llm_interface/README.md`: clarify that tracing is
  stage-owned and not an interface-level semantic interceptor.
- `src/kazusa_ai_chatbot/nodes/README.md`: document how cognition/dialog trace
  artifacts reconstruct the thought-to-dialog path.
- `docs/HOWTO.md`: add operator trace export commands, capture-mode
  configuration, and the two logging retention `.env` keys.
- `.env`: pre-populate `AUDIT_LOG_TTL_DAYS=90` and
  `DEBUG_LOG_TTL_DAYS=14` through a narrow update that does not inspect or
  print unrelated secret-bearing content.
- `.agents/skills/database-data-pull/SKILL.md`: list trace export scripts.
- `.agents/skills/debug-llm/SKILL.md`: link trace retrieval to readable review
  artifact requirements.
- Stage modules that call LLMs on the persona path, only to add explicit
  trace recording around existing calls.

### Keep

- Existing prompt text, schema decisions, model routes, retry behavior, and
  response semantics remain unchanged.
- Existing event-log privacy exclusions remain unchanged.
- Existing database export scripts remain available.

## Data Migration

- Add `llm_trace_runs` and `llm_trace_steps` through `db_bootstrap()`.
- Add audit-class `expires_at` support to `event_log_events` and
  `event_log_snapshots`.
- Create indexes:
  - `event_log_events_expires_at_ttl` on `expires_at` with
    `expireAfterSeconds=0`.
  - `event_log_snapshots_expires_at_ttl` on `expires_at` with
    `expireAfterSeconds=0`.
  - `llm_trace_runs_trace_id_unique` on `trace_id`, unique.
  - `llm_trace_runs_started_at` on `started_at`.
  - `llm_trace_runs_conversation_user_message` on
    `conversation_user_message_id`.
  - `llm_trace_runs_delivery_tracking` on `delivery_tracking_id`.
  - `llm_trace_runs_expires_at_ttl` on `expires_at` with
    `expireAfterSeconds=0`.
  - `llm_trace_steps_trace_order` on `trace_id`, `stage_order`, `step_id`.
  - `llm_trace_steps_stage_time` on `stage_name`, `created_at`.
  - `llm_trace_steps_expires_at_ttl` on `expires_at` with
    `expireAfterSeconds=0`.
- Do not backfill historical turns.
- Before sign-off, run the one-shot retention migration over existing covered
  logging rows:
  - `event_log_events`: compute expiry from `occurred_at + AUDIT_LOG_TTL_DAYS`.
  - `event_log_snapshots`: compute expiry from
    `generated_at + AUDIT_LOG_TTL_DAYS`.
  - `llm_trace_runs`: compute expiry from `started_at + DEBUG_LOG_TTL_DAYS`.
  - `llm_trace_steps`: compute expiry from `created_at + DEBUG_LOG_TTL_DAYS`.
- If computed `expires_at <= now`, delete the row during the migration.
- If a covered existing row lacks a reliable timestamp, assign
  `expires_at = now + matching TTL` rather than leaving it immortal.
- Do not remove or rewrite `conversation_history` rows as part of this
  retention migration. Conversation rows are not logging records for this
  two-class rollover policy.

## Overdesign Guardrail

- Actual problem: current diagnostics can show that a dialog LLM stage ran but
  cannot reconstruct the prompt handoff or stage output that produced a
  disputed final claim.
- Minimal change: propagate a stable trace id, add protected trace storage for
  existing LLM calls, and provide scripts plus a skill for retrieval.
- Ownership boundaries: LLM stages own semantic trace content; deterministic
  code owns ids, hashing, persistence, capture mode, export filters, and
  validation; event logging owns sanitized operational metadata only.
- Rejected complexity: no dashboard, no new LLM diagnosis agent, no global
  monkeypatching, no prompt rewrites, no hidden `LLInterface` semantic
  interception, no public raw-trace endpoint, and no historical backfill.
- Evidence threshold: add richer UI, archival/compaction tooling, or
  autonomous analysis only after repeated operator trace retrieval proves the
  raw JSON workflow is insufficient.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan's contracts.
- The agent must not introduce alternate storage paths, compatibility shims,
  fallback trace ids, prompt rewrites, route changes, or additional LLM calls.
- The agent must treat changes outside tracing, DB bootstrap, event logging
  metadata, retention config, the retention migration script, scripts, skills,
  docs, and explicitly listed LLM stage call sites as out of scope.
- If equivalent export or hashing behavior already exists, reuse or extract it
  rather than duplicating it.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Add focused tests that define trace id propagation, capture modes, protected
   storage shape, export behavior, and skill validation expectations:
   `tests/test_llm_tracing.py`, `tests/test_llm_trace_export.py`, and
   `tests/test_llm_trace_skill_contract.py`.
2. Run the focused tests before implementation and record the expected missing
   module/script/skill failures in `Execution Evidence`.
3. Add `llm_tracing` ICD, public API, DB adapter, bootstrap indexes,
   two-class retention config, `.env` prepopulation behavior, and the
   retention migration script.
4. Re-run `tests/test_llm_tracing.py` and record the passing result before
   wiring runtime stages.
5. Run the retention migration test/dry-run path and record evidence that
   existing covered logging rows will be deleted if expired or assigned
   `expires_at` before sign-off.
6. Propagate `trace_id` through chat request processing, conversation writes,
   event-log calls, resolver state, and dialog generation.
7. Add focused integration tests for trace id joins and event-log raw-data
   exclusions, then run them before continuing.
8. Instrument explicit LLM stage call sites with trace recording while leaving
   prompt text and parser semantics unchanged.
9. Add export scripts and validate they can join conversation, event-log, and
   trace rows from a visible dialog or trace id.
10. Create `.agents/skills/llm-trace-debug/SKILL.md` and update existing
   database/debug skills.
11. Update docs and run focused plus regression verification.
12. Run independent code review and record execution evidence.

## Execution Model

- Parent agent owns orchestration, test contract, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract before production
  implementation starts.
- Production-code subagent: exactly one native subagent after tests are
  established; owns production code changes only.
- Parent agent owns skill edits, documentation, export-script review, and
  final artifact review unless delegated explicitly in the plan execution
  record.
- Independent code-review subagent: exactly one native subagent after planned
  verification passes.
- If native subagent capability is unavailable, stop before production
  execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused test contract and baseline recorded.
  - Covers: implementation order steps 1-2.
  - Files: `tests/test_llm_tracing.py`, `tests/test_llm_trace_export.py`,
    `tests/test_llm_trace_skill_contract.py`.
  - Verify: run the focused test command in `Verification`.
  - Evidence: record missing module/script/skill failures in
    `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-06-25`.
- [x] Stage 2 - trace storage contract implemented.
  - Covers: implementation order steps 3-5.
  - Files: `src/kazusa_ai_chatbot/llm_tracing/`,
    `src/kazusa_ai_chatbot/db/llm_tracing.py`, DB bootstrap code,
    retention config code, `src/scripts/apply_logging_retention.py`,
    narrow `.env` default update path,
    `src/kazusa_ai_chatbot/llm_tracing/README.md`,
    `src/kazusa_ai_chatbot/db/README.md`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_llm_tracing.py -q`.
  - Evidence: record changed files, index names, capture-mode behavior,
    two-class retention config validation, migration dry-run/apply behavior,
    and test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-06-25`.
- [x] Stage 3 - runtime trace id propagation complete.
  - Covers: implementation order steps 6-7.
  - Files: service/chat queue persistence, conversation write paths, event-log
    calls, resolver state handoff, and dialog state handoff.
  - Verify: focused trace join tests plus
    `venv\Scripts\python.exe -m pytest tests\test_event_logging_interface.py -q`.
  - Evidence: record proof that event rows contain only trace refs/capture mode
    and no raw prompt/output/message/dialog fields.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-06-25`.
- [x] Stage 4 - stage-owned LLM trace recording complete.
  - Covers: implementation order step 8.
  - Files: explicit LLM call sites in relevance/decontextualizer/resolver
    cognition stages, L3 text surface stages, dialog generator, and affected
    post-turn LLM stages.
  - Verify: focused stage trace tests plus
    `venv\Scripts\python.exe -m pytest tests\test_rag_dialog_event_logging.py tests\test_dialog_agent.py tests\test_cognition_resolver_contracts.py -q`.
  - Evidence: record stage list, capture-mode behavior, and proof that prompts,
    parsers, model routes, and retry behavior are unchanged.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-06-25`.
- [x] Stage 5 - trace export scripts complete.
  - Covers: implementation order step 9.
  - Files: `src/scripts/export_llm_trace.py`,
    `src/scripts/export_dialog_trace_review_input.py`,
    `tests/test_llm_trace_export.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_llm_trace_export.py -q`.
  - Evidence: record supported lookup keys, output paths, and missing-capture
    reporting behavior.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-06-25`.
- [x] Stage 6 - retrieval skill workflow complete.
  - Covers: implementation order step 10.
  - Files: `.agents/skills/llm-trace-debug/SKILL.md`,
    `.agents/skills/database-data-pull/SKILL.md`,
    `.agents/skills/debug-llm/SKILL.md`,
    `tests/test_llm_trace_skill_contract.py`.
  - Verify: skill validation command and
    `venv\Scripts\python.exe -m pytest tests\test_llm_trace_skill_contract.py -q`.
  - Evidence: record the trigger description, required export workflow, and
    validation output.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `Codex/2026-06-25`.
- [x] Stage 7 - docs and full verification complete.
  - Covers: implementation order step 11.
  - Files: `docs/HOWTO.md`, `src/kazusa_ai_chatbot/event_logging/README.md`,
    `src/kazusa_ai_chatbot/llm_interface/README.md`,
    `src/kazusa_ai_chatbot/nodes/README.md`.
  - Verify: all focused, regression, static grep, and skill validation gates in
    `Verification`.
  - Evidence: record every command, allowed skipped live DB smoke reason, and
    static grep results.
  - Handoff: next agent starts at Stage 8.
  - Sign-off: `Codex/2026-06-25`.
- [x] Stage 8 - independent code review completed.
  - Covers: implementation order step 12.
  - Files: full implementation diff and execution evidence.
  - Verify: independent code-review subagent reports approval or all concrete
    findings are remediated and affected commands rerun.
  - Evidence: record review findings, fixes, rerun commands, residual risks,
    and approval status.
  - Handoff: next agent may complete lifecycle update only after this stage is
    signed off.
  - Sign-off: `Codex/2026-06-25`, separate subagent review waived by user
    instruction to execute without subagents; parent self-review recorded in
    `Execution Evidence`.
- [x] Stage 9 - lifecycle completion recorded.
  - Covers: plan completion after implementation and review.
  - Files: this plan and `development_plans/README.md`.
  - Verify: `git diff --check` and final `git status --short`.
  - Evidence: record final changed-file list and completion status update.
  - Handoff: no further plan work remains.
  - Sign-off: `Codex/2026-06-25`.

## Verification

### Static Greps

- `rg "system_prompt|human_payload|raw_model_output|parsed_output|final_dialog|body_text" src/kazusa_ai_chatbot/event_logging src/kazusa_ai_chatbot/db/event_logging.py`
  returns no matches introduced by this plan. Pre-existing documented forbidden
  names in `src/kazusa_ai_chatbot/event_logging/README.md` are allowed only as
  documentation of prohibited fields.
- `rg "record_event\\(" src/kazusa_ai_chatbot/event_logging src/kazusa_ai_chatbot/llm_tracing`
  returns no matches. Public observability APIs must remain explicit named
  functions.
- `rg "from kazusa_ai_chatbot.db import event_logging|from kazusa_ai_chatbot.db import llm_tracing" src/kazusa_ai_chatbot`
  returns no runtime caller matches. DB adapters may be imported only by their
  owning public modules, DB bootstrap, and focused tests.
- `rg "LLM_TRACE_TTL_DAYS" docs src tests .agents`
  returns no matches. New code, docs, tests, and skills must not introduce the
  retired trace-specific TTL config.
- `rg "AUDIT_LOG_TTL_DAYS|DEBUG_LOG_TTL_DAYS|LLM_TRACE_CAPTURE_MODE" docs src tests .agents`
  returns matches only in config, docs, tracing implementation, retention
  migration script, tests, `.env` update logic, and the trace retrieval skill.
- `rg "AUDIT_LOG_TTL_DAYS|DEBUG_LOG_TTL_DAYS" .env docs\\HOWTO.md`
  returns the two configured default keys after implementation. The `.env`
  check must not print unrelated `.env` content.
- `rg "LLM_TRACE_TTL_DAYS|EVENT_LOG_EVENTS_TTL_DAYS|EVENT_LOG_SNAPSHOTS_TTL_DAYS|LLM_TRACE_METADATA_TTL_DAYS|LLM_TRACE_FULL_TTL_DAYS|CONTROL_AUDIT_EVENTS_TTL_DAYS" docs src tests .agents`
  returns no matches. Per-collection logging TTL knobs are forbidden.
- `rg "llm_trace_debug|llm-trace-debug|export_llm_trace" .agents docs src tests development_plans`
  returns matches proving the export workflow is discoverable from the new
  skill, database-data-pull skill, debug-llm skill, HOWTO, tests, and this
  plan.

### Tests

Run deterministic focused tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_llm_tracing.py tests\test_llm_trace_export.py -q
```

Run affected regression tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_event_logging_interface.py tests\test_rag_dialog_event_logging.py tests\test_dialog_agent.py tests\test_cognition_resolver_contracts.py -q
```

Run skill validation for the new skill and any updated skill metadata:

```powershell
python "C:\Users\Ran Bao\.codex\skills\.system\skill-creator\scripts\quick_validate.py" .agents\skills\llm-trace-debug
```

Run a live DB smoke only when MongoDB is intentionally available and trace
capture has been explicitly enabled:

```powershell
venv\Scripts\python.exe -m scripts.apply_logging_retention --dry-run --output test_artifacts\diagnostics\logging_retention_dry_run.json
venv\Scripts\python.exe -m scripts.apply_logging_retention --apply --output test_artifacts\diagnostics\logging_retention_apply.json
venv\Scripts\python.exe -m scripts.export_llm_trace --trace-id <trace_id> --output test_artifacts\llm_traces\smoke_trace.json
```

Inspect the export and confirm:

- `event_log_events` contains no raw prompt/output/message/dialog content;
- covered logging rows have `expires_at`, or expired rows were deleted by the
  migration;
- no covered logging collection has immortal rows after apply;
- `llm_trace_steps` contains full payloads only when capture mode is `full`;
- missing capture mode is reported as disabled rather than silently absent;
- a `debug-llm` review can identify which stage introduced the disputed claim.

## Independent Plan Review

This review was requested before implementation. No separate reviewer was used
because the user did not explicitly authorize subagent delegation, so the
drafting agent performed a fresh plan-contract review after rereading
`README.md`, `docs/HOWTO.md`, the Event Logging ICD, DB ICD, LLM Interface ICD,
Nodes ICD, Cognition Resolver ICD, `plan_contract.md`, and
`execution_gates.md`.

Findings and resolutions:

| Severity | Finding | Resolution |
|---|---|---|
| Blocker | Capture mode was ambiguous: full capture was disabled by default, but metadata was also required for joins. | Fixed by defining `LLM_TRACE_CAPTURE_MODE=metadata` as default, `off` as explicit trace-write disable, and `full` as explicit raw capture. |
| Blocker | Trace retention was mentioned but not defined. | First fixed by defining a trace-specific TTL, then superseded by the user-approved two-class retention model with `AUDIT_LOG_TTL_DAYS=90` and `DEBUG_LOG_TTL_DAYS=14`. |
| Blocker | `scripts.export_dialog_trace_review_input` was not mandatory, leaving future agents without a guaranteed review bundle path. | Fixed by making the script required. |
| Blocker | Conversation and event-log join fields were not explicit enough for execution. | Fixed by adding the `trace_id` conversation-row contract and event-log `refs`/`labels` contract. |
| Blocker | Progress checklist was too coarse for handoff and sign-off. | Fixed by replacing it with staged checkpoints that name files, commands, evidence, handoff, and sign-off requirements. |
| Non-blocking | Static privacy gates were implied by tests but not listed as commands. | Fixed by adding exact static grep gates and expected results. |
| Non-blocking | Public trace API result behavior was underspecified. | Fixed by adding a best-effort result shape and meanings for `recorded`, `skipped`, `rejected`, and `failed`. |

Review status: draft is improved but remains `draft`; implementation still
requires explicit user approval before production-code changes.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Privacy boundary between `event_log_events` and raw trace collections.
- Stage-owned trace recording without prompt rewrites or semantic behavior
  changes.
- Trace id propagation correctness and absence of fallback ids that break
  joins.
- Export script correctness, output safety, and evidence completeness.
- Skill usefulness for a future agent starting from only a suspicious dialog.
- Alignment with `Must Do`, `Deferred`, `Change Surface`, and verification
  gates.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Every accepted dialog-producing turn has a stable `trace_id` visible in the
  relevant conversation and event-log evidence.
- When capture mode is `metadata` or `full`, every accepted dialog-producing
  turn has a trace-run row and ordered trace-step evidence. When capture mode
  is `off`, export scripts report tracing disabled for that trace id rather
  than silently returning no evidence.
- Full raw prompt/output capture is explicit, protected, and separate from
  event-log storage.
- Logging retention uses only `AUDIT_LOG_TTL_DAYS` and `DEBUG_LOG_TTL_DAYS`,
  both pre-populated in `.env`.
- Existing covered logging rows are either deleted as expired or assigned
  default `expires_at` before sign-off.
- Event-log rows remain sanitized and contain no raw prompt, raw output,
  message body, final dialog, or retrieved evidence text.
- Trace export scripts can reconstruct an ordered stage timeline from visible
  dialog text, delivery tracking id, platform message id, event id, or trace id.
- A repo-local `llm-trace-debug` skill exists and instructs agents how to pull
  trace evidence and write a human-readable debug review.
- Existing `database-data-pull` and `debug-llm` skills point to the trace
  retrieval workflow.
- Focused tests, affected regressions, and skill validation pass, or any live
  DB smoke skipped status is recorded with a reason.
- Independent code review is completed and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Raw trace leaks sensitive content | Store raw traces outside event logs, keep metadata as default, require explicit `full` for raw payloads, avoid public endpoints, add privacy tests. | Forbidden-field greps and trace export inspection. |
| Tracing changes live behavior | Best-effort writes and no prompt/parser changes. | Regression tests and code review of stage diffs. |
| Existing logs remain immortal after bigbang | Run the one-shot retention migration before sign-off and verify no covered logging row lacks `expires_at`. | Migration dry-run/apply evidence and live DB smoke when MongoDB is intentionally available. |
| Generic interception misses semantic state | Require explicit stage-owned trace calls with parsed output and state delta. | Focused tests assert stage names and parsed fields. |
| Future agents still cannot retrieve evidence | Add a dedicated skill and export scripts. | Skill validation and a smoke retrieval from a known dialog. |
| Write volume grows unexpectedly | Capture mode defaults to `metadata`, `off` is explicit, and `full` is explicit plus TTL-indexed. | Config tests and DB index tests. |

## Execution Evidence

- Draft created from the 2026-06-25 investigation of the `14:30` dialog and the
  user's added requirement that storage work must include an agent retrieval
  skill.
- Plan review on 2026-06-25: reviewed against `plan_contract.md`,
  `execution_gates.md`, README/HOWTO, and the Event Logging, DB, LLM
  Interface, Nodes, and Cognition Resolver ICDs. Resolved blockers around
  capture-mode ambiguity, missing TTL/index contract, non-mandatory review
  export script, underspecified conversation/event joins, coarse progress
  checklist, missing static privacy gates, and public trace API result
  behavior. Placeholder scan for prohibited open-ended planning terms returned
  no matches after fixes. Plan remains `draft` and does not authorize
  production-code implementation.
- Retention agreement on 2026-06-25: user approved simplified rollover config
  with one audit TTL and one debug TTL, required `.env` prepopulation, and
  required a bigbang migration where existing covered logging rows are deleted
  if expired or assigned default `expires_at` before sign-off. This supersedes
  the earlier trace-specific TTL design.
- Execution on 2026-06-25: implemented protected `llm_trace_runs` and
  `llm_trace_steps` storage, two-class TTL config, event-log/snapshot
  `expires_at` handling, trace id propagation through service, persona,
  cognition, RAG, dialog, and conversation persistence, protected trace export
  scripts, retention migration, docs, and the `llm-trace-debug` retrieval
  skill.
- Stage-owned trace coverage on 2026-06-25: added best-effort trace steps for
  relevance, message decontextualizer, RAG initializer/dispatcher/evaluator
  summarizer/continuation/finalizer, memory lifecycle specialist, L1, L2a,
  L2b, L2c1, L2c2, L2d action selection, L3 style/content/preference/visual
  stages, and dialog generator. Vision descriptor LLM calls remain untraced in
  this plan to avoid storing base64 media payloads under full capture.
- Retention migration on 2026-06-25: `.env` was narrowly populated with
  `AUDIT_LOG_TTL_DAYS=90`, `DEBUG_LOG_TTL_DAYS=14`, and
  `LLM_TRACE_CAPTURE_MODE=metadata` without printing unrelated `.env` values.
  `scripts.apply_logging_retention --apply` updated existing covered rows, and
  the final dry-run reported zero missing `expires_at`, zero pending deletes,
  and zero pending updates across `event_log_events`, `event_log_snapshots`,
  `llm_trace_runs`, and `llm_trace_steps`.
- Verification on 2026-06-25: focused trace/export/retention/skill tests passed
  with `9 passed`; affected event logging, dialog, conversation history, DB,
  service, RAG adapter, and selected multimodal regression tests passed with
  `95 passed`; plan-named RAG/dialog/cognition resolver regressions passed
  with `28 passed`; skill validation reported `Skill is valid!`;
  `git diff --check` returned no whitespace errors. A broader unrelated
  multimodal fixture test still fails if run directly because its test state
  omits required L2 current-event fields; production fail-fast behavior was not
  relaxed for that out-of-scope fixture.
- Static privacy/config checks on 2026-06-25: forbidden raw event-log field grep
  matched only Event Logging README prohibition text; retired per-collection TTL
  keys returned no matches; `.env` and HOWTO expose only the agreed audit/debug
  TTL keys; trace retrieval is discoverable from HOWTO, scripts, skills, tests,
  and this plan.
- Code review gate on 2026-06-25: user explicitly instructed execution without
  subagents, so the independent review subagent was not run. Parent self-review
  checked the privacy boundary, absence of hidden LLInterface interception,
  two-class retention enforcement, stage-owned trace writes, RAG trace-id
  propagation, and export/skill discoverability. Residual risk: no independent
  reviewer inspected the final diff.
