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

While `metadata` or `full` capture is enabled, Cognition Core V2 keeps one
invocation-local exact-input and model-attempt buffer. A clean invocation
discards that buffer without a capsule write. A terminal, recovered, partial,
or degraded invocation schedules one protected `llm_trace_steps` row with
`capture_reason="cognition_failure_capsule"`. The row contains the raw public
entrypoint input, ordered attempts, concrete validation errors, and the final
failure disposition. `cognition_failure_capsule.v2` failure events may include
an outermost-first exception cause chain capped at four entries. Every cause
message passes through the protected session's secret redaction before
persistence. Model configuration excludes API keys.

Past-dialog cognition residual can only use selected `parsed_output` fields
from protected full-capture trace steps. Metadata-mode trace steps
intentionally store empty parsed output for this purpose, so a past dialog with
only metadata trace rows contributes no residual context and is treated as
forgotten.

## Storage Contract

Retention is governed by shared logging retention settings:

- `AUDIT_LOG_TTL_DAYS` for sanitized audit and event-log rows.
- `DEBUG_LOG_TTL_DAYS` for protected debug trace rows.

Trace storage must preserve the distinction between protected trace payloads
and sanitized audit/event-log payloads. Event-log rows may reference trace ids;
they must not duplicate protected trace bodies.

Failure capsules reuse the trace-step collection, indexes, and
`DEBUG_LOG_TTL_DAYS` expiry. Their `cognition_invocation_id` distinguishes safe
retries and concurrent Cognition V2 calls under the same turn trace.

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
