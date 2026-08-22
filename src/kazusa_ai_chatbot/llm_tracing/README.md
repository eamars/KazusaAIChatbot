# LLM Tracing ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.llm_tracing`
- Runtime role: protected model-stage debug trace lane
- Related runbook: [HOWTO](../../../docs/HOWTO.md)
- Related persistence docs: [Database ICD](../db/README.md),
  [Event Logging ICD](../event_logging/README.md)

## Purpose

`llm_tracing` owns protected debug trace records for model-stage prompts,
outputs, parsed results, and state handoff metadata. It is separate from
sanitized operational event logging so local diagnosis can inspect model-stage
behavior without turning raw prompts or outputs into public audit data.

## Boundary

`event_log_events` remains sanitized audit data. It may carry `llm_trace_id`
as a correlation label or reference, but it must not store raw prompt text,
raw model output, parsed model output, final dialog text, evidence text, or
adapter payloads.

Protected trace rows are debug data. They must be governed by capture mode,
retention, and export tooling rather than shown in normal operator status,
adapter responses, or public health endpoints.

## Public Interfaces

Runbook-level inspection commands are documented in `docs/HOWTO.md`:

```bash
python -m scripts.export_llm_trace --dialog-text "..."
python -m scripts.export_dialog_trace_review_input --trace-id llmtrace_<id>
```

Runtime callers may propagate `llm_trace_id` for correlation. Trace export
scripts are diagnostic tools and must keep protected trace content out of
sanitized event-log exports.

## Capture Modes

`LLM_TRACE_CAPTURE_MODE` controls capture sensitivity:

- `off`: no trace-run or trace-step rows are written, but callers may still
  propagate `llm_trace_id`.
- `metadata`: default mode. Stores hashes, character counts, model/route
  metadata, parse status, state fields, and timing.
- `full`: stores raw prompt messages, raw response text, and parsed output in
  protected trace collections.

While `metadata` or `full` capture is enabled, cognition keeps one
invocation-local exact-input and model-attempt buffer. A clean invocation
discards that buffer without a capsule write. A terminal, recovered, partial,
or degraded invocation schedules one protected `llm_trace_steps` row with
`capture_reason="cognition_failure_capsule"`. The row contains the raw public
entrypoint input, ordered attempts, concrete validation errors, and the final
failure disposition. `cognition_failure_capsule.v3` adds an optional bounded
attempt ledger. Goal-attempt rows identify the cognition invocation, service
graph attempt, branch, producing stage, local attempt, cumulative producer
attempt, configured limit, and attempt disposition; the ledger also records
final branch dispositions. Failure events may include an outermost-first
exception cause chain capped at four entries. Every cause message passes
through the protected session's secret redaction before persistence. Model
configuration excludes API keys. Historical V2 capsule rows remain immutable
and exportable as recorded evidence.

Past-dialog cognition residual can only use selected `parsed_output` fields
from protected full-capture trace steps. Metadata-mode trace steps
intentionally store empty parsed output for this purpose, so a past dialog with
only metadata trace rows contributes no residual context and is treated as
forgotten.

### V3 protected chain transcript

The scoped `record_cognition_chain_transcript(...)` API writes one protected
`cognition_chain_transcript.v1` row for a V3 invocation when capture is
enabled. `off` writes no row; `metadata` keeps hashes, lengths, step metadata,
and dispositions; `full` may keep the exact system/user/assistant messages and
accepted step records. The row carries the V3 `run_id` and `llm_trace_id` but
stays behind the protected trace-store boundary. It is never copied into
sanitized event, service, or console payloads.

The protected chain writer is best effort and uses the existing debug-trace
retention and write timeout. A capture failure returns a bounded failed/skipped
result and does not change cognition-facing completion. Runtime timing remains
non-streaming elapsed duration; no TTFT field is part of this trace contract.

## Storage Contract

Retention is governed by shared logging retention settings:

- `AUDIT_LOG_TTL_DAYS` for sanitized audit and event-log rows.
- `DEBUG_LOG_TTL_DAYS` for protected debug trace rows.

Trace storage must preserve the distinction between protected trace payloads
and sanitized audit/event-log payloads. Event-log rows may reference trace ids;
they must not duplicate protected trace bodies.

Failure capsules reuse the trace-step collection, indexes, and
`DEBUG_LOG_TTL_DAYS` expiry. Their `cognition_invocation_id` distinguishes safe
retries and concurrent cognition calls under the same turn trace.

### Parent guardrail lineage

The live persona path has a separate protected outer writer named
`cognition_parent_guardrail_capsule.v1`. It is created only after the
canonical connector checkpoint has produced an eligible parent-recovery
trigger. The outer row stores the trace reference, scope, cycle index,
checkpoint SHA-256, bounded trigger coordinates, parent disposition, and the
epoch-aware `cognition_attempt_ledger.v2` aggregate. It stores no checkpoint
state, user content, prompts, model responses, credentials, or raw exception
messages.

The existing `cognition_failure_capsule.v3` rows remain the owners of exact
inner model attempts and retain the unchanged `cognition_attempt_ledger.v1`
shape when used without the guardrail. Clean guarded invocations discard the
outer session; a recovered or exhausted parent child schedules one additional
bounded outer row without delaying the chat response. Direct and idle
self-cognition calls do not create this outer lineage unless they explicitly
bind the live persona coordinator.

## Failure Behavior

Trace capture must not be required for normal chat delivery. Capture failures
should degrade diagnostics and be visible through operational logging or
event-log metadata, but they must not expose raw prompts or outputs through
fallback public paths.

Cognition failure-capsule persistence is scheduled in the background and is
never awaited by the response path. Snapshot, scheduling, and persistence
failures emit sanitized warnings containing no protected input, model output,
API key, or exception message. The original cognition output or exception
continues unchanged.

## Testing Contract

Tests should cover:

- capture-mode behavior for `off`, `metadata`, and `full`;
- retention command behavior for audit rows versus debug rows;
- export command filtering and correlation behavior;
- past-dialog residual behavior when parsed output is unavailable in metadata
  mode;
- absence of protected prompt/output text from sanitized event-log surfaces.

## Forbidden Paths

- Do not store raw prompts, raw model outputs, parsed model outputs, final
  dialog text, evidence text, or adapter payloads in `event_log_events`.
- Do not expose protected trace rows through public health or adapter-facing
  endpoints.
- Do not treat metadata-mode trace rows as usable past-dialog residual
  content.
- Do not bypass `DEBUG_LOG_TTL_DAYS` for protected trace retention.

## Correlation Contract

`trace_correlation_context.v1` carries only source ownership:

```json
{
  "schema_version": "trace_correlation_context.v1",
  "source_llm_trace_id": "",
  "source_episode_id": "",
  "source_background_work_job_id": "",
  "source_calendar_run_id": ""
}
```

Action attempts, background jobs, calendar schedules, and calendar runs use
`source_llm_trace_id`. Child trace runs use `parent_llm_trace_id` plus the
applicable `source_background_work_job_id` or `source_calendar_run_id`.
Historical rows may be empty and are reported as `not_captured`; the runtime
does not backfill them. The bounded
`scripts.export_trace_correlation_manifest` command is the exact typed lookup
boundary for a value copied from the Control Console. It reports zero,
multiple, conflict, and protected-read-unavailable outcomes explicitly before
the separate raw trace exporter is used.

Conflict metadata is identifier-only: a durable owner keeps its first non-empty
`source_llm_trace_id`, while the rejected competing value is retained as
`correlation_conflict_source_llm_trace_id` for bounded diagnostic review.
