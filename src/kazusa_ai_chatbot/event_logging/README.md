# Event Logging Interface Control Document

## Document Control

- ICD id: `EL-ICD-001`
- Owning package: `kazusa_ai_chatbot.event_logging`
- Interface boundary: runtime callers -> explicit event recorder functions
- Storage owner: `kazusa_ai_chatbot.db.event_logging`

## Purpose

The event logging module provides one caller-facing interface for durable,
sanitized runtime telemetry. It is append-only, best-effort, and intended for
operator status plus long-term analysis of worker behavior, reflection,
self-cognition, dispatcher handoff, queue outcomes, and model-stage contracts.

## Public Imports

Runtime callers may import only from:

```python
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.event_logging import record_worker_event
```

Runtime callers must not import event logging repository, schema, sanitizer, or
database internals directly. New event families require this ICD, public
function signatures, sanitizer coverage, and deterministic tests.

## Public Recorder Contract

All recorders are `async def` functions with keyword-only arguments. Each
recorder awaits one bounded private repository write and returns
`EventLogWriteResult`. The timeout is
`EVENT_LOG_WRITE_TIMEOUT_SECONDS = 0.25`.

Recorder failures return `status="failed"` or `status="rejected"` and must not
change live chat, reflection, self-cognition routing, scheduler execution, or
adapter delivery.

## Event Families

Approved event families are:

- `process`
- `worker`
- `llm_stage`
- `runtime_error`
- `pipeline_turn`
- `queue_intake`
- `rag_stage`
- `dialog_quality`
- `dispatcher`
- `database_operation`
- `self_cognition`
- `model_contract`
- `resource_health`

Caller code must use the named `record_*` function for the family it needs.
There is no public generic recorder.

## Privacy Rules

Event rows store IDs, refs, counts, statuses, durations, components, route
names, sanitized warning codes, short error previews, and deterministic
semantic labels. They do not store full user text, full model input, full model
answers, adapter wire envelopes, binary attachment data, vector arrays,
credentials, callback tokens, private user details, or generated dialog.

Callers may pass `EventScopeInput.platform_channel_id`; persistence converts it
to `scope.platform_channel_ref`. Raw channel IDs are not stored in event-log
documents.

## Storage

The DB package owns:

- `event_log_events`
- `event_log_snapshots`

Only event logging internals, DB bootstrap, and focused DB tests may call the
DB event-log adapter directly.

## Operator Status

The public builders are:

- `build_runtime_status(window_hours=24)`
- `build_reflection_stats(window_hours=24)`
- `build_self_cognition_stats(window_hours=24)`
- `write_analysis_snapshot(window_hours=24)`

These return bounded aggregate payloads. `/ops/*` endpoints expose those
payloads for trusted local operators only until a separate auth plan exists.

## Retention

Event collections are append-only in this plan. Retention, archival, and
external supervisor health belong to later approved plans.
