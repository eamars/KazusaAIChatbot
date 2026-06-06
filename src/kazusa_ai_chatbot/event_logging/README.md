# Event Logging Interface Control Document

## Document Control

- ICD id: `EL-ICD-001`
- Status: implementation contract
- Owning package: `kazusa_ai_chatbot.event_logging`
- Caller boundary: runtime code -> explicit public `event_logging` API
- Storage owner: `kazusa_ai_chatbot.db.event_logging`
- Collections: `event_log_events`, `event_log_snapshots`
- Primary implementation files:
  - `src/kazusa_ai_chatbot/event_logging/__init__.py`
  - `src/kazusa_ai_chatbot/event_logging/models.py`
  - `src/kazusa_ai_chatbot/event_logging/recording.py`
  - `src/kazusa_ai_chatbot/event_logging/status.py`
  - `src/kazusa_ai_chatbot/event_logging/snapshots.py`
  - `src/kazusa_ai_chatbot/db/event_logging.py`

This ICD is the caller-facing contract for event capture. Runtime modules use
only the public functions exported by `kazusa_ai_chatbot.event_logging`.
Runtime modules do not know MongoDB collection names, raw event-document
layout, sanitizer internals, or repository helpers.

## Purpose

The event logging module records durable, sanitized operational telemetry for
long-term agent and operator analysis. It covers process lifecycle, worker
ticks, LLM-stage health, recoverable runtime failures, live pipeline outcomes,
queue decisions, RAG stages, dialog quality checks, calendar scheduler and
dispatcher delivery events, approved database-operation outcomes,
self-cognition cases, model contract drift, and resource health.
Background artifact runtime ticks use the existing `worker` family; failures
use `runtime_error`. No new event family is introduced for Stage 1.

The module is append-only and best-effort. A telemetry write must never change
live chat behavior, reflection routing, self-cognition routing, dispatcher
planning, calendar scheduler execution, memory promotion, or adapter delivery.
On the live chat input path, routine successful enqueue, message persistence,
assistant-message persistence, and turn completion are intentionally not
mirrored into `event_log_events`; `conversation_history` is the canonical
record for those normal message writes. Event logging remains anomaly-first for
queue drops/collapses, failed persistence, runtime errors, and degraded
resource or model-contract health.

## Non-Goals

This module is not a tracing framework and does not expose arbitrary spans. It
does not persist raw prompts, raw model outputs, raw user messages, adapter
wire envelopes, embeddings, binary payloads, credentials, or backend
query/update documents. It does not add new LLM calls for telemetry. Analysis
snapshots are deterministic aggregate summaries only.

Authentication for `/ops/*`, retention, archival, and external supervisor
health are outside this ICD and require later approved plans.

## Public Imports

Approved caller imports:

```python
from kazusa_ai_chatbot import event_logging
```

or:

```python
from kazusa_ai_chatbot.event_logging import record_worker_event
```

Forbidden caller imports:

```python
from kazusa_ai_chatbot.db import event_logging
from kazusa_ai_chatbot.event_logging import repository
from kazusa_ai_chatbot.event_logging import schemas
from kazusa_ai_chatbot.event_logging import sanitization
```

The only approved direct users of `kazusa_ai_chatbot.db.event_logging` are:

- `kazusa_ai_chatbot.event_logging` internals
- DB bootstrap code that creates collections and indexes
- focused DB tests for the event-log adapter

## Caller Responsibilities

Callers must:

- call one explicit `record_*` function for the matching event family;
- pass keyword-only arguments;
- await the returned `EventLogWriteResult`;
- keep event values to ids, refs, counts, statuses, route names, component
  names, durations, sanitized warning codes, and short sanitized error previews;
- pass `EventScopeInput.platform_channel_id` only when a private channel
  reference is needed;
- treat `status="failed"` and `status="rejected"` as telemetry outcomes, not
  production-control outcomes.

Callers must not:

- pass raw prompt text, raw model output, user message body, final dialog,
  retrieved evidence text, adapter wire payloads, image bytes, embeddings,
  credentials, connection strings, callback tokens, API keys, or backend query
  documents;
- branch production behavior on whether telemetry was persisted;
- import event-log internals;
- create local wrappers that accept arbitrary payload dictionaries;
- expose per-user, per-message, per-query, per-prompt, or per-cache-key details
  through operator endpoints.

## Public Types

The public module exports these shared types from `event_logging.models`:

```python
EventSeverity = Literal["debug", "info", "warning", "error", "critical"]

class EventScopeInput(TypedDict, total=False):
    platform: str
    platform_channel_id: str
    channel_type: str

class SelfCognitionBudget(TypedDict):
    rag_calls: int
    cognition_calls: int
    dialog_calls: int
    topic_limit: int

class EventRefRecord(TypedDict):
    ref_type: str
    ref_id: str

class EventLogWriteResult(TypedDict):
    accepted: bool
    event_id: str
    status: Literal["recorded", "rejected", "failed"]
    reason: str
```

`EventScopeInput.platform_channel_id` is caller input only. Persistence stores
it as `scope.platform_channel_ref`, a deterministic salted hash, and never as
the raw channel id.

## Public Recorder Contract

All public recorder functions are `async def` functions with keyword-only
arguments. No public recorder accepts arbitrary `payload`, `metrics`, `labels`,
or `refs` dictionaries. The private implementation may generate those fields
from explicit public arguments.

The write timeout is:

```python
EVENT_LOG_WRITE_TIMEOUT_SECONDS = 0.25
```

Recorder return behavior:

| Condition | Returned `status` | `accepted` | Production effect |
|---|---:|---:|---|
| Event persisted | `recorded` | `True` | None beyond telemetry |
| Invalid severity or denied field path | `rejected` | `False` | Caller continues normal work |
| Timeout, cancellation, or DB failure | `failed` | `False` | Caller continues normal work |

MongoDB write failures, timeouts, and cancellation are contained inside the
event logging module. They may emit a sanitized local warning log, but they must
not propagate into caller control flow.

## Public API

`kazusa_ai_chatbot.event_logging.__init__` exports this API:

```python
async def record_process_event(
    *,
    event_type: Literal["startup", "shutdown", "lifespan_error"],
    phase: str,
    component: str,
    status: str,
    pid: int,
    host_label: str,
    config_snapshot_id: str = "",
    git_commit: str = "",
    severity: EventSeverity = "info",
    correlation_id: str = "",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_worker_event(
    *,
    event_type: Literal["started", "stopped", "tick", "disabled", "cancelled"],
    component: str,
    worker_name: str,
    enabled: bool,
    dry_run: bool,
    run_kind: str,
    status: str,
    processed_count: int = 0,
    succeeded_count: int = 0,
    failed_count: int = 0,
    skipped_count: int = 0,
    deferred: bool = False,
    defer_reason: str = "",
    run_id: str = "",
    duration_ms: int | None = None,
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_llm_stage_event(
    *,
    component: str,
    stage_name: str,
    route_name: str,
    model_name: str,
    status: str,
    prompt_chars: int,
    output_chars: int,
    parse_status: str,
    retry_count: int,
    json_repair_used: bool,
    run_id: str = "",
    correlation_id: str = "",
    duration_ms: int | None = None,
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_runtime_error_event(
    *,
    component: str,
    error_class: str,
    error_preview: str,
    stack_fingerprint: str,
    top_frame_module: str,
    recovered: bool,
    status: str = "failed",
    run_id: str = "",
    correlation_id: str = "",
    severity: EventSeverity = "error",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_pipeline_turn_event(
    *,
    component: str,
    correlation_id: str,
    status: str,
    queue_wait_ms: int,
    stages_reached: Sequence[str],
    final_outcome: str,
    scheduled_followups: int,
    debug_modes: Sequence[str] = (),
    scope: EventScopeInput | None = None,
    duration_ms: int | None = None,
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_queue_intake_event(
    *,
    component: str,
    correlation_id: str,
    status: str,
    queue_depth: int,
    coalesced_count: int,
    dropped_count: int,
    protected_by_reply: bool,
    listen_only: bool,
    scope: EventScopeInput | None = None,
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_rag_stage_event(
    *,
    component: str,
    correlation_id: str,
    agent_name: str,
    status: str,
    slot_count: int,
    retrieval_count: int,
    cache_hit: bool,
    no_evidence: bool,
    latency_ms: int,
    run_id: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_dialog_quality_event(
    *,
    component: str,
    correlation_id: str,
    usage_mode: str,
    evaluator_status: str,
    retry_count: int,
    failure_codes: Sequence[str],
    anchor_count: int,
    status: str,
    run_id: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_dispatcher_event(
    *,
    component: str,
    action_kind: str,
    validation_status: str,
    adapter_available: bool,
    status: str,
    scheduled_event_ids: Sequence[str] = (),
    rejection_codes: Sequence[str] = (),
    attempt_id: str = "",
    correlation_id: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_database_operation_event(
    *,
    component: str,
    collection: str,
    operation_kind: str,
    status: str,
    idempotency_result: str,
    latency_ms: int,
    document_ref: str = "",
    run_id: str = "",
    correlation_id: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_self_cognition_event(
    *,
    component: str,
    case_id: str,
    trigger_kind: str,
    selected_route: str,
    output_mode: str,
    budget: SelfCognitionBudget,
    dispatch_status: str,
    status: str,
    trigger_id: str = "",
    run_id: str = "",
    attempt_id: str = "",
    consolidation_outcome: Mapping[str, object] | None = None,
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_model_contract_event(
    *,
    component: str,
    stage_name: str,
    violation_kind: str,
    missing_fields: Sequence[str],
    invalid_fields: Sequence[str],
    repair_used: bool,
    status: str,
    run_id: str = "",
    correlation_id: str = "",
    severity: EventSeverity = "warning",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def record_resource_health_event(
    *,
    component: str,
    resource_name: str,
    resource_kind: str,
    availability: str,
    latency_ms: int,
    failure_class: str = "",
    status: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult: ...

async def build_runtime_status(
    *,
    window_hours: int = 24,
) -> dict[str, object]: ...

async def build_reflection_stats(
    *,
    window_hours: int = 24,
) -> dict[str, object]: ...

async def build_self_cognition_stats(
    *,
    window_hours: int = 24,
) -> dict[str, object]: ...

async def write_analysis_snapshot(
    *,
    window_hours: int = 24,
    snapshot_kind: Literal["event_log_snapshot"] = "event_log_snapshot",
) -> EventLogWriteResult: ...
```

The public module must not export `record_event`.

## Event Families And Payloads

Every event row has one event family. The payload for each family is generated
by the matching public recorder.

| Family | Recorder | Event type source | Payload fields |
|---|---|---|---|
| `process` | `record_process_event` | `event_type` | `phase`, `pid`, `host_label`, `config_snapshot_id`, `git_commit` |
| `worker` | `record_worker_event` | `event_type` | `worker_name`, `enabled`, `dry_run`, `run_kind`, `processed_count`, `succeeded_count`, `failed_count`, `skipped_count`, `deferred`, `defer_reason` |
| `llm_stage` | `record_llm_stage_event` | `stage_name` | `stage_name`, `route_name`, `model_name`, `prompt_chars`, `output_chars`, `parse_status`, `retry_count`, `json_repair_used` |
| `runtime_error` | `record_runtime_error_event` | `runtime_error` | `error_class`, `error_preview`, `stack_fingerprint`, `top_frame_module`, `recovered` |
| `pipeline_turn` | `record_pipeline_turn_event` | `turn` | `queue_wait_ms`, `stages_reached`, `final_outcome`, `scheduled_followups`, `debug_modes` |
| `queue_intake` | `record_queue_intake_event` | `queue_intake` | `queue_depth`, `coalesced_count`, `dropped_count`, `protected_by_reply`, `listen_only` |
| `rag_stage` | `record_rag_stage_event` | `agent_name` | `agent_name`, `slot_count`, `retrieval_count`, `cache_hit`, `no_evidence`, `latency_ms` |
| `dialog_quality` | `record_dialog_quality_event` | `dialog_quality` | `usage_mode`, `evaluator_status`, `retry_count`, `failure_codes`, `anchor_count` |
| `dispatcher` | `record_dispatcher_event` | `action_kind` | `action_kind`, `validation_status`, `adapter_available`, legacy `scheduled_event_ids`, `rejection_codes` |
| `database_operation` | `record_database_operation_event` | `operation_kind` | `collection`, `operation_kind`, `idempotency_result`, `latency_ms`, `document_ref` |
| `self_cognition` | `record_self_cognition_event` | `trigger_kind` | `case_id`, `trigger_kind`, `selected_route`, `output_mode`, `budget`, `dispatch_status`, `consolidation_outcome` |
| `model_contract` | `record_model_contract_event` | `violation_kind` | `stage_name`, `violation_kind`, `missing_fields`, `invalid_fields`, `repair_used` |
| `resource_health` | `record_resource_health_event` | `resource_kind` | `resource_name`, `resource_kind`, `availability`, `latency_ms`, `failure_class` |

### Family-Specific Forbidden Data

| Family | Forbidden data |
|---|---|
| `process` | environment variables, connection strings, credentials |
| `worker` | raw worker output, raw reflection artifacts, raw self-cognition artifacts |
| `llm_stage` | system prompt, human prompt, full input messages, raw model output, parsed model output |
| `runtime_error` | full traceback with local values, request bodies, secrets |
| `pipeline_turn` | user message body, final dialog, adapter payload, conversation row body |
| `queue_intake` | dropped message text, collapsed message text, attachment bytes |
| `rag_stage` | raw query, retrieved evidence text, cache entry body |
| `dialog_quality` | generated dialog text, evaluator prompt, evaluator raw output |
| `dispatcher` | action candidate text, tool args containing message body, adapter raw response |
| `database_operation` | inserted or updated document body, query filters with user text |
| `self_cognition` | source packet text, candidate text, private reasoning text, dialog candidate |
| `model_contract` | offending raw model output, repaired output body |
| `resource_health` | credentials, secret-bearing URLs, request or response bodies |

## Canonical Event Document

`event_log_events` is the append-only canonical event stream.

```python
class EventScopeRecord(TypedDict):
    platform: str
    platform_channel_ref: str
    channel_type: str

class EventErrorRecord(TypedDict):
    error_class: str
    error_preview: str
    stack_fingerprint: str
    recovered: bool

class EventLogEventDoc(TypedDict):
    event_id: str
    event_family: str
    event_type: str
    component: str
    severity: EventSeverity
    status: str
    correlation_id: str
    run_id: str
    trigger_id: str
    attempt_id: str
    occurred_at: str
    created_at: str
    duration_ms: int | None
    scope: EventScopeRecord
    metrics: dict[str, int | float | bool | str]
    labels: dict[str, str]
    refs: list[EventRefRecord]
    warning_codes: list[str]
    error: EventErrorRecord
    payload: dict[str, Any]
```

Rules:

- `event_id` is generated by event logging internals.
- `occurred_at` is caller-provided when a source event has its own timestamp;
  otherwise it is generated at record time.
- `created_at` is always generated at record time.
- `metrics`, `labels`, and `refs` are internal projections from public
  recorder arguments.
- `warning_codes` is a capped sanitized list.
- `error_preview` is a short sanitized preview only.
- `payload` contains only the family-specific fields listed above.

## Snapshot Document

`event_log_snapshots` stores deterministic aggregate summaries for later
operator or agent review.

```python
class EventLogSnapshotDoc(TypedDict):
    snapshot_id: str
    snapshot_kind: str
    window_start: str
    window_end: str
    generated_at: str
    source_counts: dict[str, int]
    semantic_descriptors: dict[str, str]
    findings: list[dict[str, str]]
    source_event_refs: list[str]
```

`write_analysis_snapshot` currently writes `snapshot_kind="event_log_snapshot"`.
It computes source counts and semantic descriptors deterministically. It does
not invoke an LLM and does not copy raw event payloads into the snapshot.

Approved snapshot source counts:

- `reflection_succeeded`
- `reflection_failed`
- `self_cognition_runs`
- `self_cognition_dispatch_accepted`
- `llm_failed`
- `llm_repaired`
- `worker_errors`

Approved semantic descriptors:

- `reflection_health`: `healthy`, `mixed`, `failing`, `inactive`
- `self_cognition_liveness`: `inactive`, `active_with_handoff`,
  `active_internal_only`
- `llm_parse_stability`: `stable`, `watch`, `degraded`
- `worker_error_level`: `none`, `low`, `elevated`, `high`

## Sanitization Contract

The sanitizer owns prompt-safe event document construction.

Current sanitizer behavior:

- removes ASCII control bytes from short text;
- strips surrounding whitespace;
- caps short text to 300 characters by default;
- caps string lists to 25 items;
- converts `EventScopeInput.platform_channel_id` to
  `scope.platform_channel_ref` with a deterministic salted SHA-256 hash;
- rejects event documents containing denied field names anywhere in the nested
  document.

Denied field names:

```text
human_prompt
system_prompt
raw_output
base64_data
embedding
api_key
shared_secret
message_envelope
```

These denied names are a final guardrail, not permission to pass nearby raw
data under different names. The caller contract above remains stricter than the
key-name rejection list.

## Storage Contract

The DB package owns collection creation, indexes, inserts, finds, counts, and
aggregations. `event_logging.repository` is the only bridge from public event
logging code to DB helpers.

Collections:

| Collection | Purpose |
|---|---|
| `event_log_events` | append-only sanitized event stream |
| `event_log_snapshots` | deterministic aggregate snapshots |

Indexes created by DB bootstrap:

| Collection | Index name | Fields |
|---|---|---|
| `event_log_events` | `event_log_event_id_unique` | `event_id` unique |
| `event_log_events` | `event_log_family_component_time` | `event_family`, `component`, `occurred_at` |
| `event_log_events` | `event_log_component_type_status_time` | `component`, `event_type`, `status`, `occurred_at` |
| `event_log_events` | `event_log_correlation_time` | `correlation_id`, `occurred_at` |
| `event_log_events` | `event_log_run_time` | `run_id`, `occurred_at` |
| `event_log_events` | `event_log_trigger_time` | `trigger_id`, `occurred_at` |
| `event_log_events` | `event_log_attempt_time` | `attempt_id`, `occurred_at` |
| `event_log_snapshots` | `event_log_snapshot_id_unique` | `snapshot_id` unique |
| `event_log_snapshots` | `event_log_snapshot_kind_time` | `snapshot_kind`, `generated_at` |

Retention is not implemented by this ICD. Event collections are append-only
until a later retention or archival plan is approved.

## Operator Status Contract

The public status builders expose bounded aggregate payloads:

| Builder | Service endpoint | Purpose |
|---|---|---|
| `build_runtime_status(window_hours=24)` | `GET /ops/runtime-status` | process and worker event summaries plus semantic worker error level; service route adds calendar, reflection, self-cognition, and background artifact liveness |
| `build_reflection_stats(window_hours=24)` | `GET /ops/reflection/stats` | reflection counts, latest event refs, reflection health |
| `build_self_cognition_stats(window_hours=24)` | `GET /ops/self-cognition/stats` | self-cognition run counts, latest refs, liveness label; service endpoint adds `enabled` and `task_alive` |

Operator payloads must remain aggregate. They may include counts, status
strings, event ids, run ids, trigger ids, attempt ids, timestamps, service-owned
worker state such as `enabled` and `task_alive`, and semantic descriptors. They
must not expose raw prompt, message, output, evidence, or per-user detail.

Deployment must restrict `/ops/*` exposure to trusted operators until a future
auth plan exists.

## Resolver Telemetry Helper Boundary

`kazusa_ai_chatbot.cognition_resolver.telemetry` builds sanitized
resolver-cycle and resolver-terminal dictionaries for local inspection and
future event-log integration. These helpers are not a new public event-log
family and do not persist by themselves.

The helper payloads may contain cycle counts, capability kinds, observation
statuses, duration labels, pending-resume status, and bounded L1/L2/L2d
summaries. They must not contain raw user message bodies, raw prompts, raw
model output, raw platform ids, raw database ids, adapter wire payloads,
credentials, or callback URLs.

Persisting resolver telemetry to `event_log_events` requires a later dedicated
public recorder and the extension review below. Until then, local human-readable
resolver artifacts belong under `test_artifacts/cognition_resolver/` and remain
debug artifacts rather than production control state.

## Approved Instrumentation Boundaries

Caller modules may emit only through the public API. The approved production
instrumentation surface is:

| Module | Approved event families |
|---|---|
| `service.py` | `process`, `resource_health`, `queue_intake`, `pipeline_turn`, `runtime_error`, `database_operation` |
| `background_artifact.runtime` | `worker`, `runtime_error` |
| `reflection_cycle.worker` | `worker`, `llm_stage`, `model_contract`, `database_operation`, `runtime_error` |
| `self_cognition.worker` | `worker`, `self_cognition`, `dispatcher`, `runtime_error` |
| `self_cognition.runner` | `self_cognition`, `llm_stage`, `model_contract` |
| `dispatcher.dispatcher` | `dispatcher`, `database_operation` |
| `dispatcher.handlers` | `dispatcher` |
| `persona_supervisor2*` RAG nodes | `rag_stage`, `llm_stage`, `model_contract` |
| `dialog_agent` | `llm_stage`, `model_contract`, `dialog_quality` |

Do not add global monkeypatches, broad DB wrappers, broad LLM interceptors, or
runtime object scanners for telemetry under this ICD.

## Extension Rules

A new event family requires all of the following in one reviewed change:

1. A public keyword-only `record_*` function exported from
   `kazusa_ai_chatbot.event_logging`.
2. An ICD update naming the family, recorder, event type source, allowed
   payload fields, and forbidden data.
3. Sanitizer coverage for any new reference, label, warning, or scope behavior.
4. Deterministic tests proving:
   - the public function is async and keyword-only;
   - there is no public generic `record_event`;
   - unsafe field names are rejected;
   - raw prompt/message/output/secret fields are not stored;
   - timeout, cancellation, and DB failure stay best-effort;
   - operator endpoints remain aggregate if the new family affects stats.
5. Documentation updates for affected runtime modules and operator docs.

A new field in an existing family requires the same review standard unless it
is a count, id/ref, status label, duration, route name, component name, or
sanitized warning code already covered by this ICD.

## Review Checklist

Before signing off event logging changes, reviewers must verify:

- caller code imports only from `kazusa_ai_chatbot.event_logging`;
- `kazusa_ai_chatbot.db.event_logging` is used only by event logging internals,
  DB bootstrap, and focused DB tests;
- no public recorder accepts `payload`, `metrics`, `labels`, or `refs`;
- no `record_event` public export exists;
- forbidden raw fields are absent from persisted event documents and operator
  payloads;
- event writes remain bounded by `EVENT_LOG_WRITE_TIMEOUT_SECONDS`;
- timeout, cancellation, invalid severity, unsafe field paths, and DB failures
  return `EventLogWriteResult` without changing production behavior;
- `/ops/*` payloads expose only aggregate counts, refs, timestamps, status
  values, and semantic descriptors;
- snapshot documents contain deterministic counts and descriptors only;
- tests cover any new family, field, or operator aggregate.
