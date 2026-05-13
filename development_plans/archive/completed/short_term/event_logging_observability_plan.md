# event logging observability plan

## Summary

- Goal: Add a dedicated `event_logging` module and ICD that provide explicit public interfaces for durable, sanitized runtime event capture and long-term agent-based audits.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `py-style`, `test-style-and-execution`, `local-llm-architecture`, `development-plan-writing`.
- Overall cutover strategy: compatible
- Highest-risk areas: privacy leakage, accidental live-response latency, broad LLM instrumentation, MongoDB write amplification, and confusing dry-run artifacts with production activity.
- Acceptance criteria: `event_logging` is the only caller-facing event capture interface; its ICD defines concrete async public APIs, ownership, and forbidden data; new append-only event collections exist; reflection, self-cognition, LLM failures, recoverable runtime errors, process lifecycle failures, dispatcher outcomes, and pipeline outcomes are recorded through the explicit module API; operator stats endpoints return bounded aggregate payloads; tests and docs prove no raw prompt/message/secret data is exposed.

## Context

The current audit could answer reflection health from `character_reflection_runs`, `/health`, process state, and local artifacts. That is enough for a point-in-time manual audit, but not enough for long-term agent-based analysis.

Current state:

- Reflection is database-led. It persists `character_reflection_runs`, promoted memory, interaction-style overlays, and global character growth rows.
- Self-cognition is artifact-led. It writes local files under `SELF_COGNITION_TRACKING_DIR` and a local action-attempt ledger, but it has no production MongoDB run ledger.
- `/health` exposes service health, scheduler status, and sanitized Cache2 agent stats.
- Worker liveness, effective sanitized config, last tick results, self-cognition DB state, dispatch outcomes, and LLM-stage latency/parse telemetry are not durable.

Target state:

- Operators and analysis agents can query aggregate reflection and self-cognition stats without scanning arbitrary files or raw conversations.
- Self-cognition production activity is distinguishable from dry-run/test artifacts.
- Background telemetry failures never block `/chat`, reflection, dispatcher delivery, or self-cognition decisions.
- Any prompt-facing agent-analysis snapshot converts numeric measurements into semantic descriptors before LLM consumption.

## Mandatory Skills

- `py-style`: load before editing Python files. Follow project docstring, import, exception, default, and fail-fast rules.
- `test-style-and-execution`: load before adding, changing, or running tests. Regular deterministic tests may run in batches; live DB tests require explicit MongoDB availability.
- `local-llm-architecture`: load before changing background LLM telemetry, prompt-facing analysis snapshots, self-cognition cognition wrappers, or reflection LLM instrumentation.
- `development-plan-writing`: load before modifying this plan or moving it through lifecycle states.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the active agent must run the `Independent Code Review` gate and record the result in `Execution Evidence`.
- Do not store raw prompts, full LLM outputs, raw message bodies, raw `message_envelope`, base64 attachments, embeddings, API keys, adapter tokens, callback secrets, private user details, or full generated dialog in event-log collections or operator endpoints by default.
- Store IDs, run refs, status values, counts, timestamps, route names, component names, sanitized warning codes, sanitized error classes, and short sanitized error previews only.
- Observability writes must be best-effort. A telemetry write failure may log a sanitized warning, but must not change live chat, reflection, self-cognition routing, dispatcher handoff, scheduler execution, memory promotion, or adapter delivery.
- Public recorder functions must be `async def` functions with keyword-only arguments. They must await one bounded repository write through the private event-logging repository and return `EventLogWriteResult`.
- Event-log writes must use an internal timeout constant `EVENT_LOG_WRITE_TIMEOUT_SECONDS = 0.25`. Timeout, cancellation, and MongoDB failures return `status="failed"` with a sanitized reason and must not propagate to caller control flow.
- The implementation must not create a custom telemetry queue, untracked `asyncio.create_task` fire-and-forget writes, or alternate background event writers. Live-path callers may await only the bounded public recorder call at approved call sites.
- Do not add new LLM calls on the live `/chat` response path.
- Do not add new background LLM calls for telemetry capture. Agent-analysis snapshots in this plan are deterministic summaries only.
- If a future LLM analysis consumer reads these stats, feed it semantic labels generated by deterministic helper functions, not raw measurement tables alone.
- Do not record event-log collection writes as `database_operation` events. This prevents recursive event logging.
- Keep reflection outside live chat. Raw reflection outputs still must not enter normal cognition directly.
- Keep self-cognition controlled-handoff behavior. It may hand a selected `send_message` candidate to the dispatcher; it must not call adapters directly or write live-chat conversation rows.
- Use repo-relative paths and PowerShell `-LiteralPath` for filesystem commands.
- Use `apply_patch` for manual edits.

## Must Do

- Add `src/kazusa_ai_chatbot/event_logging/README.md` as the canonical Event Logging ICD.
- Add a dedicated `kazusa_ai_chatbot.event_logging` module with explicit public functions for all caller code.
- Require caller code to use only the public `kazusa_ai_chatbot.event_logging` interface. Reflection, self-cognition, service routes, dispatcher, RAG, dialog, and future callers must not write directly to event-log DB helpers.
- Add an append-only event-log DB interface under the database package for use by the event-logging module internals only.
- Add bootstrap support and indexes for the event-log collections.
- Add sanitized event recording for process lifecycle, worker lifecycle/ticks, LLM failures, recoverable runtime errors, lifespan failures, accepted chat turns, queue/intake decisions, RAG stages, dialog quality checks, dispatcher outcomes, approved database write outcomes, model contract drift, resource health, reflection, and self-cognition.
- Mirror self-cognition live trigger/run/route/action-attempt/dispatch-result artifacts into the event log while preserving local artifact output.
- Add bounded operator endpoints:
  - `GET /ops/runtime-status`
  - `GET /ops/reflection/stats`
  - `GET /ops/self-cognition/stats`
- Add deterministic analysis-snapshot support that writes aggregate, semantic, prompt-safe snapshots for later agent review.
- Add tests for the event-logging public interface, DB contracts, worker recording, endpoint response shape, privacy exclusions, import-boundary rules, and documentation examples.
- Update `docs/HOWTO.md`, `src/kazusa_ai_chatbot/brain_service/README.md`, `src/kazusa_ai_chatbot/db/README.md`, `src/kazusa_ai_chatbot/reflection_cycle/README.md`, and `src/kazusa_ai_chatbot/self_cognition/README.md`.

## Deferred

- Do not add authentication, authorization, or a UI dashboard in this plan.
- Do not add Prometheus, OpenTelemetry, external metrics services, or new runtime dependencies.
- Do not add raw OpenTelemetry-style spans or arbitrary structured logging. The `event_logging` ICD owns the approved event families and fields.
- Do not backfill historical local self-cognition test artifacts into production MongoDB.
- Do not change reflection prompt schemas or self-cognition route semantics.
- Do not redesign `/health`; keep new operator status under `/ops/*`.
- Do not expose per-user, per-message, per-cache-key, per-query, or per-prompt payload details through operator endpoints.
- Do not add an autonomous diagnosis LLM agent. This plan stores deterministic snapshots that a later approved analysis agent can consume.
- Do not claim hard-process crash coverage in this plan. In-process event logging can capture startup, shutdown, lifespan failures, handled exceptions, and worker loop exceptions only. OS kills, interpreter aborts, host crashes, and power loss require a future external supervisor plan.
- Do not instrument every database helper globally. `database_operation` events are limited to the exact call sites listed in `Fixed Instrumentation Scope`.
- Do not store raw `platform_channel_id` in event-log documents. Event logging may accept a scope input from callers, but persisted scope uses `platform_channel_ref`, a deterministic salted hash or empty string.
- Operator endpoints are local-service endpoints in this plan. Authentication and authorization are deferred, so documentation must state that deployment must restrict `/ops/*` exposure to trusted operators until a future auth plan exists.

## Cutover Policy

Overall strategy: compatible

| Area | Policy | Instruction |
|---|---|---|
| Existing `/health` | compatible | Preserve current fields and Cache2 stats. Do not move worker status into `/health`. |
| New `/ops/*` endpoints | compatible | Add optional operator endpoints without changing existing adapter APIs. |
| Event logging module | bigbang | New event capture must use `kazusa_ai_chatbot.event_logging`; do not create alternate event-writing helpers in caller modules. |
| Reflection persistence | compatible | Keep `character_reflection_runs` as the canonical reflection run ledger. Add supplemental telemetry only. |
| Self-cognition persistence | compatible | Preserve local artifacts and ledger. Add event-log mirror rows for production worker activity. |
| MongoDB collections | migration | Create new collections and indexes idempotently through `db_bootstrap`. No historical backfill is required. |
| LLM telemetry | compatible | Add explicit event-logging calls at approved stage boundaries. Do not store raw prompts or outputs. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The implementation agent must not choose a more conservative or broader strategy by default.
- If an area is `compatible`, preserve only the compatibility surfaces listed above.
- If an area is `bigbang`, route all new event capture through the named public interface instead of preserving or adding local event-write paths.
- If an area is `migration`, use the exact idempotent collection/index creation path in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The agent must not introduce alternate storage paths, fallback endpoints, compatibility shims, telemetry queues, dashboard UI, or external metrics dependencies.
- The agent must treat changes outside `event_logging`, `db`, `brain_service`, `service`, `reflection_cycle`, `self_cognition`, `dispatcher`, `rag`, and `nodes` as high-scrutiny changes and justify them in `Execution Evidence`.
- The agent must not expose or call `kazusa_ai_chatbot.db.event_logging` outside `kazusa_ai_chatbot.event_logging`, `db_bootstrap`, and focused tests.
- The agent must not import from `kazusa_ai_chatbot.event_logging.repository`, `schemas`, `sanitization`, or other internals in caller code. Caller code imports only from `kazusa_ai_chatbot.event_logging`.
- The agent may add small projection helpers only when they convert raw measurements into semantic descriptors or sanitize telemetry payloads.
- If equivalent repository behavior already exists, reuse or extract it instead of duplicating it.
- If the plan and code disagree, preserve this plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

The brain service exposes bounded operator status:

```json
{
  "status": "ok",
  "generated_at": "2026-05-13T11:14:38Z",
  "config": {
    "reflection_cycle_enabled": true,
    "self_cognition_enabled": false,
    "reflection_worker_interval_seconds": 900,
    "self_cognition_worker_interval_seconds": 3600,
    "self_cognition_max_cases_per_tick": 3
  },
  "workers": {
    "reflection_cycle": {
      "enabled": true,
      "task_alive": true,
      "last_event_at": "2026-05-13T11:09:10Z",
      "last_status": "completed"
    },
    "self_cognition": {
      "enabled": false,
      "task_alive": false,
      "last_event_at": "",
      "last_status": "disabled"
    }
  }
}
```

Stats endpoints return aggregate windows and latest refs, not raw content. Self-cognition production rows can be counted from MongoDB even when local files rotate or the service restarts. All runtime capture flows through `kazusa_ai_chatbot.event_logging`; caller modules do not construct MongoDB rows, collection names, or raw event envelopes themselves.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Ownership | Add `kazusa_ai_chatbot.event_logging` as the caller-facing module plus `kazusa_ai_chatbot.db.event_logging` as its private storage adapter | Event logging needs one explicit interface. Caller modules should not know collection names, MongoDB expressions, or sanitizer internals. |
| ICD | Make `src/kazusa_ai_chatbot/event_logging/README.md` canonical | The interface must be inspectable and enforceable like the brain service, DB, reflection, and self-cognition ICDs. |
| Public API | Export named `record_*` and `build_*` functions from `kazusa_ai_chatbot.event_logging` only | Named APIs prevent arbitrary payloads and make greps/reviews enforce the boundary. |
| Storage | Use append-only MongoDB event-log collections | Long-term analysis needs durable history across process restarts. |
| Event model | Store one canonical event stream with event-family-specific typed payloads | This groups process, worker, LLM, runtime-error, turn, RAG, dialog, dispatcher, DB, and self-cognition evidence without scattering persistence across runtime modules. |
| Recorder async contract | Make every public recorder `async def` and bounded by `EVENT_LOG_WRITE_TIMEOUT_SECONDS = 0.25` | MongoDB helpers are async. A concrete async contract prevents hidden task spawning and keeps live-path latency bounded. |
| Caller payloads | Public functions accept only typed keyword arguments and never raw payload dictionaries | The explicit ICD must prevent accidental raw prompt, output, message, or document capture at the call boundary. |
| `/health` | Keep unchanged | Adapters already consume `/health`; operator stats should not bloat readiness checks. |
| Self-cognition event mirror | Mirror production artifacts into the event stream in addition to local files | Local artifacts are useful for debug but not reliable as a long-term production ledger. |
| Analysis snapshots | Deterministic snapshots only | This enables future agent review without adding new LLM calls or hidden reasoning loops now. |
| LLM telemetry scope | Approved metadata at explicit stage boundaries | LLM failures and contract drift are important, but raw prompts/outputs and broad wrappers are not allowed. |
| Privacy | Deny raw prompt/message/output storage by default | Observability must be safe to query and attach to plans. |
| Scope identity | Persist `platform_channel_ref` instead of raw `platform_channel_id` | Event logs need stable grouping without duplicating raw platform identifiers into operator-facing telemetry. |
| DB operation telemetry | Record only the three approved production write groups in `Fixed Instrumentation Scope` | Global DB interception would increase blast radius and risk recursive or private document capture. |

## Contracts And Data Shapes

### Event Logging ICD

Create `src/kazusa_ai_chatbot/event_logging/README.md` as `EL-ICD-001`.

The ICD must define:

- module purpose and ownership boundary,
- public imports and forbidden imports,
- caller responsibilities,
- event families and allowed payload fields,
- sanitizer behavior,
- storage ownership,
- best-effort failure behavior,
- endpoint/statistics integration,
- privacy and retention rules,
- tests required for new event families.

Caller rule:

```python
from kazusa_ai_chatbot import event_logging
```

or:

```python
from kazusa_ai_chatbot.event_logging import record_worker_event
```

Caller modules must not import:

```python
from kazusa_ai_chatbot.db import event_logging
from kazusa_ai_chatbot.event_logging import repository
from kazusa_ai_chatbot.event_logging import schemas
from kazusa_ai_chatbot.event_logging import sanitization
```

The only approved direct users of `kazusa_ai_chatbot.db.event_logging` are `kazusa_ai_chatbot.event_logging` internals, `db_bootstrap`, and focused DB tests.

### Public Interface

`src/kazusa_ai_chatbot/event_logging/__init__.py` must export these explicit functions:

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

async def build_runtime_status(*, window_hours: int = 24) -> dict[str, object]: ...
async def build_reflection_stats(*, window_hours: int = 24) -> dict[str, object]: ...
async def build_self_cognition_stats(*, window_hours: int = 24) -> dict[str, object]: ...
async def write_analysis_snapshot(
    *,
    window_hours: int = 24,
    snapshot_kind: Literal["event_log_snapshot"] = "event_log_snapshot",
) -> EventLogWriteResult: ...
```

Public caller functions must match these signatures. Callers must not pass raw
payload dictionaries, arbitrary event-family names, arbitrary event-type names,
raw prompts, raw outputs, message bodies, or backend query/update documents. A
generic `record_event(...)` helper may exist only as a private internal helper.

Shared public types exported from `event_logging.models`:

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

Recorder functions are best-effort. They must not raise on MongoDB write failure. They may reject invalid enum values or unsafe payloads by returning `status="rejected"` and a sanitized reason.

### `event_log_events`

Append-only canonical event stream.

```python
{
    "event_id": str,
    "event_family": (
        "process" | "worker" | "llm_stage" | "runtime_error" |
        "pipeline_turn" | "queue_intake" | "rag_stage" |
        "dialog_quality" | "dispatcher" | "database_operation" |
        "self_cognition" | "model_contract" | "resource_health"
    ),
    "event_type": str,
    "component": str,
    "severity": "debug" | "info" | "warning" | "error" | "critical",
    "status": str,
    "correlation_id": str,
    "run_id": str,
    "trigger_id": str,
    "attempt_id": str,
    "occurred_at": str,
    "created_at": str,
    "duration_ms": int | None,
    "scope": {
        "platform": str,
        "platform_channel_ref": str,
        "channel_type": str,
    },
    "metrics": dict[str, int | float | bool | str],
    "labels": dict[str, str],
    "refs": list[EventRefRecord],
    "warning_codes": list[str],
    "error": {
        "error_class": str,
        "error_preview": str,
        "stack_fingerprint": str,
        "recovered": bool,
    },
    "payload": (
        ProcessPayload | WorkerPayload | LLMStagePayload |
        RuntimeErrorPayload | PipelineTurnPayload |
        QueueIntakePayload | RAGStagePayload |
        DialogQualityPayload | DispatcherPayload |
        DatabaseOperationPayload | SelfCognitionPayload |
        ModelContractPayload | ResourceHealthPayload
    ),
}
```

`EventScopeInput.platform_channel_id` must be converted to
`scope.platform_channel_ref` before persistence. `platform_channel_ref` is a
deterministic salted hash or empty string; raw channel ids are not stored in
event-log documents. `metrics`, `labels`, `refs`, and `payload` are generated
from the explicit public function arguments and internal typed schemas only.
No public function accepts arbitrary `metrics`, `labels`, `refs`, or `payload`
dictionaries. `error_preview` is capped at 300 characters and must pass the
sanitizer.

### Event Family Payloads

Approved payload fields by family:

| Family | Required payload fields | Forbidden payload fields |
|---|---|---|
| `process` | `phase`, `pid`, `host_label`, `config_snapshot_id`, `git_commit` | env vars, secrets, connection strings |
| `worker` | `enabled`, `dry_run`, `run_kind`, `processed_count`, `succeeded_count`, `failed_count`, `skipped_count`, `deferred`, `defer_reason` | raw run output |
| `llm_stage` | `stage_name`, `route_name`, `model_name`, `prompt_chars`, `output_chars`, `parse_status`, `retry_count`, `json_repair_used` | system prompt, human prompt, raw output, parsed output |
| `runtime_error` | `error_class`, `error_preview`, `stack_fingerprint`, `top_frame_module`, `recovered` | full stack with local values |
| `pipeline_turn` | `queue_wait_ms`, `stages_reached`, `final_outcome`, `scheduled_followups`, `debug_modes` | message body, final dialog |
| `queue_intake` | `queue_depth`, `coalesced_count`, `dropped_count`, `protected_by_reply`, `listen_only` | dropped message text |
| `rag_stage` | `agent_name`, `slot_count`, `retrieval_count`, `cache_hit`, `no_evidence`, `latency_ms` | raw query, retrieved text |
| `dialog_quality` | `evaluator_status`, `retry_count`, `failure_codes`, `anchor_count` | generated dialog text |
| `dispatcher` | `action_kind`, `validation_status`, `adapter_available`, `scheduled_event_ids`, `rejection_codes` | action candidate text, adapter raw response |
| `database_operation` | `collection`, `operation_kind`, `idempotency_result`, `latency_ms` | raw document body |
| `self_cognition` | `case_id`, `trigger_kind`, `selected_route`, `output_mode`, `budget`, `dispatch_status` | source packet text, candidate text |
| `model_contract` | `stage_name`, `violation_kind`, `missing_fields`, `invalid_fields`, `repair_used` | offending raw model output |
| `resource_health` | `resource_name`, `resource_kind`, `availability`, `latency_ms`, `failure_class` | credentials, URLs with secrets |

### Fixed Instrumentation Scope

Instrumentation is limited to these call sites. The implementation agent must
not add broader wrappers, monkeypatch DB helpers, intercept every LLM call, or
scan arbitrary runtime objects for telemetry.

| File | Symbols | Approved event families |
|---|---|---|
| `src/kazusa_ai_chatbot/service.py` | `lifespan`, `_enqueue_chat_request`, `_drop_queued_chat_item`, `_persist_collapsed_queued_chat_item`, `_process_queued_chat_item`, new `/ops/*` route handlers | `process`, `runtime_error`, `queue_intake`, `pipeline_turn`, `database_operation`, `resource_health` |
| `src/kazusa_ai_chatbot/reflection_cycle/worker.py` | `start_reflection_cycle_worker`, `stop_reflection_cycle_worker`, `_reflection_worker_loop`, `_run_worker_tick` | `worker`, `llm_stage`, `model_contract`, `runtime_error` |
| `src/kazusa_ai_chatbot/self_cognition/worker.py` | `start_self_cognition_worker`, `stop_self_cognition_worker`, `run_self_cognition_worker_tick`, `_self_cognition_worker_loop` | `worker`, `self_cognition`, `dispatcher`, `runtime_error` |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | `run_self_cognition_case` only after route classification and artifact construction | `self_cognition`, `llm_stage`, `model_contract` |
| `src/kazusa_ai_chatbot/dispatcher/dispatcher.py` | `TaskDispatcher.dispatch` | `dispatcher`, `database_operation` |
| `src/kazusa_ai_chatbot/dispatcher/handlers.py` | `handle_send_message` | `dispatcher`, `runtime_error` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | `stage_1_research` | `rag_stage` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` | `rag_initializer`, `rag_dispatcher`, `rag_executor`, `rag_evaluator`, `rag_finalizer`, `call_rag_supervisor` | `rag_stage`, `llm_stage`, `model_contract` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py` | `rag_dispatcher`, `rag_executor` | `rag_stage` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py` | `rag_evaluator`, `rag_finalizer` | `rag_stage`, `llm_stage`, `model_contract` |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | `dialog_generator`, `dialog_evaluator`, `dialog_agent` | `dialog_quality`, `llm_stage`, `model_contract` |

`database_operation` events are limited to these operations:

| Component | Collection | Operation | Recording location |
|---|---|---|---|
| `service.chat_queue` | `conversation_history` | `save_conversation` for dropped, collapsed, user, and assistant rows | `service.py` after the awaited DB helper returns or raises |
| `dispatcher.scheduler` | `scheduled_events` | scheduler event insert through `scheduler.schedule_event(...)` | `TaskDispatcher.dispatch` after the awaited scheduler call returns or raises |
| `reflection_cycle.worker` | `character_reflection_runs` | reflection run upsert result | `reflection_cycle/worker.py` around the existing run invocation, without changing repository semantics |

Hard crash coverage is out of scope. This plan records process startup,
graceful shutdown, lifespan startup failures that are caught before re-raise,
handled request/worker exceptions, and worker loop exceptions. External
supervisor restarts, OS kills, interpreter aborts, and host failures require a
future supervisor-health plan.

### `event_log_snapshots`

Deterministic aggregate snapshot for later agent review.

```python
{
    "snapshot_id": str,
    "snapshot_kind": "event_log_snapshot",
    "window_start": str,
    "window_end": str,
    "generated_at": str,
    "source_counts": dict,
    "semantic_descriptors": dict,
    "findings": list[dict],
    "source_event_refs": list[str],
}
```

`semantic_descriptors` must include labels such as `reflection_health="healthy"`, `self_cognition_liveness="inactive"`, `llm_parse_stability="degraded"`, or `worker_error_level="none"` generated by deterministic helper functions.

## LLM Call And Context Budget

Before this plan:

- Reflection hourly/daily/promotion LLM calls run in the background and persist reflection run documents.
- Self-cognition may call RAG once, shared cognition once, and dialog once per case.
- No LLM call is made for event logging.

After this plan:

- Live `/chat` LLM call count remains unchanged.
- Reflection LLM call count remains unchanged.
- Self-cognition LLM call count remains unchanged.
- Event logging adds zero LLM calls.
- Agent-analysis snapshots are deterministic and add zero LLM calls.

Prompt-facing future analysis payload cap:

- Default cap: 50k tokens.
- This plan stores deterministic snapshots designed to fit under 20k characters per analysis window by using aggregate counts, labels, warning codes, and refs.
- If a future plan adds an LLM analysis agent, it must read `event_log_snapshots.semantic_descriptors` first and use raw numeric counts only as supporting evidence with deterministic explanations.

## Change Surface

### Create

| Path | Purpose |
|---|---|
| `src/kazusa_ai_chatbot/event_logging/README.md` | Canonical Event Logging ICD (`EL-ICD-001`). |
| `src/kazusa_ai_chatbot/event_logging/__init__.py` | Public package exports for all approved caller interfaces. |
| `src/kazusa_ai_chatbot/event_logging/models.py` | Typed constants, result shapes, event family enums, and semantic descriptor helpers. |
| `src/kazusa_ai_chatbot/event_logging/schemas.py` | Internal typed event payload shapes used by the public `record_*` functions. |
| `src/kazusa_ai_chatbot/event_logging/sanitization.py` | Shared sanitizer and forbidden-field checks used before persistence or endpoint projection. |
| `src/kazusa_ai_chatbot/event_logging/recording.py` | Best-effort public `record_*` implementations used by runtime callers. |
| `src/kazusa_ai_chatbot/event_logging/status.py` | Build `/ops/runtime-status` and stats endpoint payloads through the public module boundary. |
| `src/kazusa_ai_chatbot/event_logging/snapshots.py` | Build deterministic aggregate snapshots for later agent review. |
| `src/kazusa_ai_chatbot/event_logging/repository.py` | Private adapter from event-logging APIs to DB helpers. Only event-logging internals may import it. |
| `src/kazusa_ai_chatbot/db/event_logging.py` | MongoDB collection names, indexes, insert helpers, and aggregate stats helpers for event logging. |
| `src/scripts/export_event_log.py` | Operator export/snapshot command writing sanitized JSON under `test_artifacts/diagnostics/`. |
| `tests/test_event_logging_repository.py` | DB helper contract tests with fake collections. |
| `tests/test_event_logging_interface.py` | Public interface, sanitizer, result-shape, and forbidden-import tests. |
| `tests/test_event_logging_status.py` | Endpoint/status payload shape and descriptor tests. |
| `tests/test_self_cognition_event_logging.py` | Self-cognition event mirror and privacy tests. |
| `tests/test_reflection_event_logging.py` | Reflection worker event recording tests. |
| `tests/test_service_event_logging.py` | Service process, queue/intake, pipeline-turn, runtime-error, resource-health, and allowed database-operation recording tests. |
| `tests/test_dispatcher_event_logging.py` | Dispatcher validation, scheduler write, adapter availability, and forbidden text/privacy tests. |
| `tests/test_rag_dialog_event_logging.py` | RAG and dialog node instrumentation tests proving no prompt, raw evidence, generated dialog, or graph behavior changes. |
| `tests/test_service_ops_status.py` | FastAPI route handler tests for `/ops/*`. |

### Modify

| Path | Change |
|---|---|
| `src/kazusa_ai_chatbot/db/bootstrap.py` | Add required event-log collections and call `ensure_event_log_indexes()`. |
| `src/kazusa_ai_chatbot/db/__init__.py` | Re-export event-log DB helpers only if existing DB package conventions require it; caller code still must not use them. |
| `src/kazusa_ai_chatbot/db/README.md` | Document event-log collections, ownership, and the rule that runtime callers use `kazusa_ai_chatbot.event_logging`. |
| `src/kazusa_ai_chatbot/brain_service/contracts.py` | Add Pydantic response models for `/ops/runtime-status`, `/ops/reflection/stats`, and `/ops/self-cognition/stats`. |
| `src/kazusa_ai_chatbot/brain_service/README.md` | Document new operator endpoints and privacy limits. |
| `src/kazusa_ai_chatbot/service.py` | Add `/ops/*` route handlers and pass worker handles/config into `event_logging` status builders. Add bounded recorder calls only in `lifespan`, `_enqueue_chat_request`, `_drop_queued_chat_item`, `_persist_collapsed_queued_chat_item`, and `_process_queued_chat_item`. |
| `src/kazusa_ai_chatbot/reflection_cycle/worker.py` | Record sanitized worker lifecycle/tick, reflection LLM-stage, model-contract, and worker-loop exception events through `event_logging` at the symbols listed in `Fixed Instrumentation Scope`. |
| `src/kazusa_ai_chatbot/self_cognition/worker.py` | Record sanitized worker lifecycle/tick, self-cognition production run summary, dispatch-result mirror, and worker-loop exception events through `event_logging`. |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | Mirror trigger, run, route effect, and action attempt artifacts through `event_logging` only after existing artifact records are built; do not change local artifact output or route classification. |
| `src/kazusa_ai_chatbot/dispatcher/dispatcher.py` | Record dispatcher validation, duplicate, scheduler write, and scheduling outcomes through `event_logging` without storing raw tool-call args or candidate text. |
| `src/kazusa_ai_chatbot/dispatcher/handlers.py` | Record adapter availability/send exceptions for `handle_send_message` without storing outgoing text or raw adapter response. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | Record one top-level RAG stage event from `stage_1_research`; do not alter persona graph state or prompt payloads. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` | Record RAG stage, LLM-stage, and model-contract metadata at existing supervisor boundaries only; do not alter prompts, graph topology, retry behavior, or continuation semantics. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py` | Record dispatcher/executor stage metadata only; do not instrument every helper agent internally. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py` | Record evaluator/finalizer stage metadata only; do not store refined queries, raw evidence, or model output. |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | Record dialog evaluator quality, LLM-stage, and model-contract metadata without storing generated dialog, evaluator feedback text, or prompt payloads. |
| `src/kazusa_ai_chatbot/self_cognition/README.md` | State that production tracking is mirrored through event logging while local artifacts remain canonical dry-run output. |
| `src/kazusa_ai_chatbot/reflection_cycle/README.md` | State that event logging supplements `character_reflection_runs`. |
| `docs/HOWTO.md` | Add operator endpoint and export command documentation. |

### Keep

| Path | Instruction |
|---|---|
| `src/adapters/*` | Do not change adapters. |
| `src/kazusa_ai_chatbot/nodes/*` | Do not change live cognition/dialog prompt text, graph topology, state shape, retry policy, or semantics. Node edits are limited to the exact instrumentation call sites listed in `Fixed Instrumentation Scope`. |
| `src/kazusa_ai_chatbot/reflection_cycle/prompts.py` | Do not change reflection prompt schemas. |
| `src/kazusa_ai_chatbot/self_cognition/tracking.py` | Preserve route classification and artifact shapes. Only add DB mirror calls outside semantic classification. |

## Data Migration

- `db_bootstrap()` creates these collections if missing:
  - `event_log_events`
  - `event_log_snapshots`
- Add indexes:
  - `event_id` unique on `event_log_events`.
  - `(event_family, component, occurred_at)`.
  - `(component, event_type, status, occurred_at)`.
  - `(correlation_id, occurred_at)`.
  - `(run_id, occurred_at)`.
  - `(trigger_id, occurred_at)`.
  - `(attempt_id, occurred_at)`.
  - `snapshot_id` unique on `event_log_snapshots`.
  - `(snapshot_kind, generated_at)`.
- Do not import existing `test_artifacts/self_cognition_*` files into MongoDB.
- No destructive migration, deletion, or backfill is authorized.

## Implementation Order

1. Add the Event Logging ICD skeleton and public-interface tests that assert the concrete async signatures, keyword-only arguments, typed result shape, no arbitrary payload arguments, and forbidden imports.
2. Add DB repository tests for `event_log_events`, `event_log_snapshots`, collection/index creation, bounded insert timeout behavior, sanitized insert helpers, and raw channel-id hashing.
3. Implement `db.event_logging` and bootstrap integration.
4. Implement `event_logging.models`, `sanitization`, `repository`, and `recording` with `EVENT_LOG_WRITE_TIMEOUT_SECONDS = 0.25`.
5. Add status and semantic descriptor tests.
6. Implement `event_logging.status` and `event_logging.snapshots`.
7. Add self-cognition event mirror tests for trigger/run/action/dispatch artifacts and privacy exclusions.
8. Wire self-cognition worker/runner recording through `event_logging` while preserving local artifacts and route classification.
9. Add reflection worker event tests for lifecycle, tick, LLM-stage, model-contract, and worker-loop exception events.
10. Wire reflection worker recording through `event_logging` without changing `character_reflection_runs`, prompts, or promotion behavior.
11. Add focused tests for service process lifecycle, queue/intake, pipeline-turn, allowed `database_operation` writes, runtime-error, and resource-health events.
12. Wire the approved `service.py` instrumentation call sites only.
13. Add focused tests for dispatcher validation, scheduler write outcomes, adapter availability/send exceptions, and database-operation events.
14. Wire `dispatcher/dispatcher.py` and `dispatcher/handlers.py` instrumentation only.
15. Add focused tests for RAG-stage, dialog-quality, LLM-stage, and model-contract event recording at the approved node call sites.
16. Wire approved RAG and dialog node instrumentation without changing prompts, graph topology, state shape, retry behavior, or generated dialog.
17. Add service route tests for `/ops/*`.
18. Implement Pydantic contracts and FastAPI route handlers using `event_logging` status builders.
19. Add deterministic export/snapshot script and tests.
20. Update docs, including local-only `/ops/*` exposure, no hard-crash coverage, and deferred retention policy.
21. Run verification.
22. Run independent code review and remediate findings.

## Progress Checklist

- [x] Stage 1 - Event Logging ICD, public interface, DB contract, and bootstrap
  - Covers: steps 1-4.
  - Verify: `venv\Scripts\python -m pytest tests/test_event_logging_interface.py tests/test_event_logging_repository.py tests/test_db.py -q`
  - Evidence: record concrete async signatures, timeout constant, channel-ref hashing behavior, index names, and test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 2 - Runtime projection and status builders
  - Covers: steps 5-6.
  - Verify: `venv\Scripts\python -m pytest tests/test_event_logging_status.py -q`
  - Evidence: record endpoint payload examples and descriptor labels in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 3 - Self-cognition production mirror
  - Covers: steps 7-8.
  - Verify: `venv\Scripts\python -m pytest tests/test_self_cognition_tracking.py tests/test_self_cognition_integration.py tests/test_self_cognition_event_logging.py -q`
  - Evidence: record proof that local artifacts still write, route classification is unchanged, and DB mirror rows omit action candidate text.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 4 - Reflection worker telemetry
  - Covers: steps 9-10.
  - Verify: `venv\Scripts\python -m pytest tests/test_reflection_cycle_stage1c_worker.py tests/test_reflection_event_logging.py -q`
  - Evidence: record proof that skipped/deferred/reflection result events are recorded without changing `character_reflection_runs`.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 5 - Service process, queue, pipeline, DB-operation, runtime-error, and resource-health events
  - Covers: steps 11-12.
  - Verify: `venv\Scripts\python -m pytest tests/test_service_event_logging.py tests/test_event_logging_interface.py tests/test_service_input_queue.py tests/test_service_health.py -q`
  - Evidence: record exact `service.py` symbols wired, prove bounded awaited calls are used, and prove no raw message/envelope content is persisted.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 6 - Dispatcher and scheduler write telemetry
  - Covers: steps 13-14.
  - Verify: `venv\Scripts\python -m pytest tests/test_dispatcher_event_logging.py tests/test_scheduler_future_promise.py tests/test_event_logging_interface.py -q`
  - Evidence: record exact dispatcher symbols wired and prove scheduled-event rows omit candidate text and raw adapter responses.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 7 - RAG and dialog node telemetry
  - Covers: steps 15-16.
  - Verify: `venv\Scripts\python -m pytest tests/test_rag_dialog_event_logging.py tests/test_persona_supervisor2_rag2_integration.py tests/test_dialog_agent.py tests/test_event_logging_interface.py -q`
  - Evidence: record exact node symbols wired and prove prompts, graph topology, state shape, retry behavior, generated dialog, refined queries, and raw evidence are unchanged.
  - Handoff: next agent starts at Stage 8.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 8 - Operator endpoints and export script
  - Covers: steps 17-19.
  - Verify: `venv\Scripts\python -m pytest tests/test_service_ops_status.py tests/test_event_logging_status.py -q`
  - Evidence: record sample sanitized `/ops/*` payloads and script output path.
  - Handoff: next agent starts at Stage 9.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 9 - Documentation and full verification
  - Covers: steps 20-21.
  - Verify: all commands in `Verification`.
  - Evidence: record static grep, deterministic tests, local-only `/ops/*` docs, no-hard-crash docs, deferred retention docs, and any live DB smoke results.
  - Handoff: next agent starts at Stage 10.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 10 - Independent code review
  - Covers: step 22.
  - Verify: review gate completed, findings resolved or explicitly accepted as residual risk, affected tests rerun.
  - Evidence: record reviewer mode, files reviewed, findings, fixes, reruns, and approval status.
  - Handoff: plan can move to `completed` only after this stage is signed off.
  - Sign-off: `Codex/2026-05-14` after review and evidence are recorded.

## Verification

### Static Checks

- `rg "event_log_events|event_log_snapshots|record_llm_stage_event|record_runtime_error_event" src tests docs`
  - Expected: matches in the new event-logging modules, bootstrap, tests, and docs.
- `rg "def record_.*\\.\\.\\.|payload: dict|record_event\\(" src/kazusa_ai_chatbot/event_logging tests docs`
  - Expected: no public `record_*` ellipsis signatures, no `payload: dict` public contract, and no public generic `record_event` export.
  - Allowed: private internal `_record_event(...)` in `src/kazusa_ai_chatbot/event_logging/recording.py` or `repository.py`, and test strings asserting that `record_event` is not exported.
- `rg "human_prompt|system_prompt|raw_output|base64_data|embedding|api_key|shared_secret|message_envelope" src/kazusa_ai_chatbot/event_logging src/kazusa_ai_chatbot/db/event_logging.py src/kazusa_ai_chatbot/brain_service src/kazusa_ai_chatbot/service.py`
  - Expected: no matches showing those fields added to observability payloads or `/ops/*` responses.
  - Allowed: existing unrelated `brain_service.contracts` request models may still define `message_envelope` outside new `/ops/*` models.
- `rg "platform_channel_id" src/kazusa_ai_chatbot/event_logging src/kazusa_ai_chatbot/db/event_logging.py tests/test_event_logging_interface.py tests/test_event_logging_repository.py`
  - Expected: matches only in scope input handling, hashing tests, and forbidden raw-storage assertions. Persisted event documents and endpoint payloads use `platform_channel_ref`, not `platform_channel_id`.
- `rg "db\\.event_logging|event_logging\\.repository|event_logging\\.schemas|event_logging\\.sanitization" src/kazusa_ai_chatbot tests`
  - Expected: no runtime caller imports outside `src/kazusa_ai_chatbot/event_logging`, `src/kazusa_ai_chatbot/db/bootstrap.py`, and focused tests.
- `rg "create_task\\(|BackgroundTasks|add_task\\(" src/kazusa_ai_chatbot/event_logging src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/reflection_cycle src/kazusa_ai_chatbot/self_cognition src/kazusa_ai_chatbot/dispatcher src/kazusa_ai_chatbot/nodes`
  - Expected: no new untracked event-log background write path. Existing service and worker task creation may remain only for pre-existing chat, reflection, self-cognition, scheduler, or FastAPI behavior, not for event logging.
- `rg "self_cognition_action_attempts.jsonl" src/kazusa_ai_chatbot/self_cognition tests`
  - Expected: existing local ledger behavior remains and tests still cover it.
- `rg "@app.get\\(\"/ops/" src/kazusa_ai_chatbot/service.py`
  - Expected: exactly the three approved operator routes.

### Deterministic Tests

- `venv\Scripts\python -m pytest tests/test_event_logging_repository.py -q`
- `venv\Scripts\python -m pytest tests/test_event_logging_interface.py -q`
- `venv\Scripts\python -m pytest tests/test_event_logging_status.py -q`
- `venv\Scripts\python -m pytest tests/test_self_cognition_event_logging.py -q`
- `venv\Scripts\python -m pytest tests/test_reflection_event_logging.py -q`
- `venv\Scripts\python -m pytest tests/test_service_event_logging.py -q`
- `venv\Scripts\python -m pytest tests/test_dispatcher_event_logging.py -q`
- `venv\Scripts\python -m pytest tests/test_rag_dialog_event_logging.py -q`
- `venv\Scripts\python -m pytest tests/test_service_ops_status.py tests/test_service_health.py -q`
- `venv\Scripts\python -m pytest tests/test_self_cognition_tracking.py tests/test_self_cognition_integration.py -q`
- `venv\Scripts\python -m pytest tests/test_reflection_cycle_stage1c_worker.py tests/test_reflection_cycle_stage1c_service.py -q`
- `venv\Scripts\python -m pytest tests/test_db.py -q`

### Live DB Smoke

Run only when MongoDB is intentionally available:

- `venv\Scripts\python -m scripts.export_event_log --hours 24 --output test_artifacts/diagnostics/event_log_smoke.json`
  - Expected: writes a JSON file with aggregate stats and no raw message bodies, prompts, embeddings, or secrets.
- Start the brain service and call:
  - `GET /ops/runtime-status`
  - `GET /ops/reflection/stats`
  - `GET /ops/self-cognition/stats`
  - Expected: HTTP 200 with bounded aggregate JSON payloads.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that did not draft the plan. If no separate reviewer is available, the active agent must reread `README.md`, `docs/HOWTO.md`, `src/kazusa_ai_chatbot/brain_service/README.md`, `src/kazusa_ai_chatbot/db/README.md`, `src/kazusa_ai_chatbot/reflection_cycle/README.md`, `src/kazusa_ai_chatbot/self_cognition/README.md`, this plan, and relevant tests from a fresh-review posture.

Review scope:

- The proposed scope aligns with the project boundary: adapter/debug client -> brain service -> queue/intake -> RAG -> cognition -> dialog -> persistence/consolidation -> scheduler/reflection.
- Reflection remains outside live chat and raw reflection output does not enter cognition directly.
- Self-cognition remains controlled-handoff only and does not call adapters directly.
- The plan gives concrete file paths, contracts, data shapes, endpoint names, verification gates, checklist stages, and evidence requirements.
- No unresolved choices, broad helper freedom, fallback routes, dashboard scope, external telemetry dependency, or raw-data capture path remains.
- New MongoDB collection ownership is in the DB package, not scattered through runtime modules.

Record blockers, non-blocking findings, required edits, and approval status. Approve only when blockers are resolved.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. Prefer a reviewer that did not implement the change. If no separate reviewer is available, the active agent must reread this plan, inspect the full diff from a fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, privacy leaks, unbounded payloads, persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including execution evidence, static checks, deterministic tests, live DB smoke if run, and lifecycle registry updates.

Fix concrete findings directly only when the fix is inside the approved change surface or this review gate explicitly allows review-only fixture/documentation corrections. If a fix would cross the approved boundary or alter the contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `src/kazusa_ai_chatbot/event_logging/README.md` defines the canonical ICD and public interface contract.
- Runtime callers use only the public `kazusa_ai_chatbot.event_logging` interface for event capture.
- Static import checks prove no runtime caller writes directly to `db.event_logging` or event-logging internals.
- `db_bootstrap()` creates all event-log collections and indexes idempotently.
- Reflection worker lifecycle/tick/LLM-stage telemetry is durably recorded without changing reflection run persistence semantics.
- Self-cognition production trigger/run/action/dispatch tracking is mirrored to the event log while local artifacts and local duplicate-suppression ledger still work.
- Process lifecycle, recoverable runtime errors, lifespan failures, accepted chat-turn outcomes, queue/intake decisions, RAG-stage metadata, dialog quality, dispatcher outcomes, approved database operation outcomes, model contract drift, and resource health can be recorded through approved public functions.
- `/ops/runtime-status`, `/ops/reflection/stats`, and `/ops/self-cognition/stats` return bounded, sanitized aggregate payloads.
- Deterministic analysis snapshots can be exported and stored without new LLM calls.
- Static checks show no raw prompts, raw outputs, raw message bodies, embeddings, or secrets are exposed by event-logging code or `/ops/*` payloads.
- Focused and regression tests listed in `Verification` pass, or any blocked live DB smoke is recorded with reason.
- Documentation describes the endpoints, collections, privacy contract, and export command.
- Independent code review is completed and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Telemetry leaks private content | Sanitizer helpers, payload denylist, tests, and static greps. | Privacy tests and forbidden-field grep. |
| Telemetry write failure changes runtime behavior | Best-effort recording helpers catch DB errors and log sanitized warnings. | Worker tests simulate DB failure and assert original result still returns. |
| `/ops/*` payloads become too large | Endpoints use aggregate helpers, date windows, result caps, and latest refs only. | Endpoint tests assert caps and response shape. |
| Self-cognition dry-run artifacts pollute production stats | Mongo mirror writes only from production worker path or explicit approved script mode. | Tests distinguish dry-run runner calls from worker recording. |
| Broad LLM instrumentation changes behavior | Scope LLM metadata to approved call sites only and store counts/statuses without prompt/output text. | Static grep and focused tests prove prompts, graph topology, retry behavior, and generated dialog are untouched. |
| New collections grow without bound | Append-only is required for analysis; retention policy is deferred and must be planned separately before production scale-out. | Documentation records deferred retention policy and names retention as a required follow-up plan. |
| `/ops/*` endpoints expose operator data too broadly | Keep payloads aggregate-only and document trusted-network/local-service assumption because auth is deferred. | Endpoint tests assert aggregate shape; docs state `/ops/*` must not be internet exposed before auth exists. |

## Operational Steps

- After implementation, restart the brain service so `db_bootstrap()` creates the new collections.
- Call `GET /ops/runtime-status` to confirm effective config and worker liveness.
- Run the export command for a short window and inspect the JSON for aggregate-only data.
- If `SELF_COGNITION_ENABLED=false`, expect self-cognition liveness to report disabled and no production run rows.
- Before exposing `/ops/*` beyond localhost or a trusted operator network, create and approve a separate authentication/authorization plan.
- Before high-volume production retention becomes a concern, create and approve a separate event-log retention or archival plan.

## Execution Evidence

- Plan review: 2026-05-14 independent review found blockers around plan class, concrete public APIs, async write behavior, arbitrary payloads, broad instrumentation scope, crash coverage, retention, auth assumptions, and bundled verification. This revision resolves those blockers by reclassifying to `high_risk_migration`, defining concrete async keyword-only APIs, bounding MongoDB writes, forbidding raw payload dictionaries, hashing channel scope, pinning exact instrumentation call sites, clarifying no hard-crash coverage, documenting local-only `/ops/*` exposure, recording retention as deferred follow-up scope, and splitting implementation/verification stages.
- Stage 1 evidence: Implemented `src/kazusa_ai_chatbot/event_logging/README.md`, public async keyword-only recorder exports, `EVENT_LOG_WRITE_TIMEOUT_SECONDS = 0.25`, private event-log repository adapter, MongoDB collection/index helper, and bootstrap delegation. Verified deterministic channel refs hash raw `platform_channel_id` into `scope.platform_channel_ref` and raw channel IDs are absent from persisted event documents. Index names include `event_log_event_id_unique`, `event_log_family_component_time`, `event_log_component_type_status_time`, `event_log_correlation_time`, `event_log_run_time`, `event_log_trigger_time`, `event_log_attempt_time`, `event_log_snapshot_id_unique`, and `event_log_snapshot_kind_time`. Verification command `venv\Scripts\python -m pytest tests/test_event_logging_interface.py tests/test_event_logging_repository.py tests/test_db.py -q` passed with 62 passed, 13 deselected on 2026-05-14.
- Stage 2 evidence: Implemented aggregate status builders for runtime, reflection, and self-cognition windows plus deterministic snapshot descriptors. Sample descriptor labels verified: `reflection_health=healthy|mixed`, `self_cognition_liveness=inactive|active_internal_only|active_with_handoff`, `llm_parse_stability=stable|watch|degraded`, and `worker_error_level=none|low|elevated|high`. Verification command `venv\Scripts\python -m pytest tests/test_event_logging_status.py -q` passed with 5 passed on 2026-05-14.
- Stage 3 evidence: Added opt-in runner mirroring and production worker mirroring through the public `event_logging` API after local self-cognition artifacts are built. Existing local artifacts, route classification, duplicate suppression, and dispatcher handoff behavior remained unchanged. Event-log calls store case/run/trigger/attempt refs, selected route, output mode, budget counters, and dispatch status; tests assert candidate text is omitted. Existing self-cognition integration tests stub event-log recorders so deterministic tests do not touch MongoDB. Verification command `venv\Scripts\python -m pytest tests/test_self_cognition_tracking.py tests/test_self_cognition_integration.py tests/test_self_cognition_event_logging.py -q` passed with 28 passed on 2026-05-14.
- Stage 4 evidence: Added reflection worker tick/lifecycle metadata, recoverable worker-loop error recording, reflection-run upsert DB-operation recording for `character_reflection_runs`, and LLM-stage/model-contract metadata around hourly and daily reflection calls. Existing reflection run persistence and prompt schemas remained unchanged; tests stub event-log recorders in deterministic worker coverage. Verification command `venv\Scripts\python -m pytest tests/test_reflection_cycle_stage1c_worker.py tests/test_reflection_event_logging.py -q` passed with 10 passed on 2026-05-14.
- Stage 5 evidence: Wired `lifespan`, `_enqueue_chat_request`, `_drop_queued_chat_item`, `_persist_collapsed_queued_chat_item`, and `_process_queued_chat_item` in `service.py` through the public `event_logging` API. Process startup/shutdown/lifespan failures, resource health for Mongo/RAG initializer/MCP manager, accepted/dropped/collapsed queue decisions, approved `conversation_history` user/assistant write outcomes, graph/runtime failures, and pipeline-turn outcomes are awaited through bounded public recorders. Focused tests assert queue, DB-operation, runtime-error, resource-health, and pipeline calls omit raw user text and generated dialog; older deterministic queue/lifespan tests stub event-log recorders and DB-backed promoted-reflection context reads. Verification command `venv\Scripts\python -m pytest tests/test_service_event_logging.py tests/test_event_logging_interface.py tests/test_service_input_queue.py tests/test_service_health.py -q` passed with 36 passed on 2026-05-14.
- Stage 6 evidence: Wired `TaskDispatcher.dispatch` and `handle_send_message` through the public `event_logging` API. Dispatcher validation failures, duplicate rejections, scheduler insert successes/failures, scheduling outcomes, adapter delivery successes, adapter availability failures, and send exceptions are recorded without storing raw tool-call arguments, action candidate text, raw target channel ids in correlation ids, or raw adapter responses. Existing scheduler tests stub handler telemetry so deterministic scheduler execution does not touch event-log storage. Verification command `venv\Scripts\python -m pytest tests/test_dispatcher_event_logging.py tests/test_scheduler_future_promise.py tests/test_event_logging_interface.py -q` passed with 15 passed on 2026-05-14.
- Stage 7 evidence: Wired `stage_1_research`, `rag_initializer`, `rag_dispatcher`, `rag_executor`, `rag_evaluator`, `rag_finalizer`, `call_rag_supervisor`, dispatcher/evaluator LLM-stage boundaries, `dialog_generator`, `dialog_evaluator`, and `dialog_agent` through the public `event_logging` API. Focused tests assert RAG skip/success, dispatcher LLM metadata, executor wrapper metadata, dialog generator/evaluator contract drift, and dialog quality events omit raw query text, raw evidence, helper result payloads, generated dialog, and raw channel ids; existing RAG/dialog regression tests stub event-log recorders and continue proving public RAG keys, request shape, skip behavior, dialog state, prompt text, and retry behavior. Verification command `venv\Scripts\python -m pytest tests/test_rag_dialog_event_logging.py tests/test_persona_supervisor2_rag2_integration.py tests/test_dialog_agent.py tests/test_event_logging_interface.py -q` passed with 30 passed on 2026-05-14.
- Stage 8 evidence: Added `OpsRuntimeStatusResponse` and aggregate stats response contracts, plus `/ops/runtime-status`, `/ops/reflection/stats`, and `/ops/self-cognition/stats` route handlers. Sample runtime payload shape includes `status`, `generated_at`, `window_hours`, `config.reflection_cycle_enabled`, `config.self_cognition_enabled`, worker `enabled`/`task_alive` flags, latest event refs, and semantic descriptors only. Reflection and self-cognition stats expose counts, latest refs, and labels only. Added `python -m scripts.export_event_log --hours 24 --output test_artifacts/diagnostics/event_log_smoke.json`, with default output under `test_artifacts/diagnostics/event_log_<UTC>.json`; the script writes aggregate status/stats and the deterministic snapshot write result, not raw event documents. Verification command `venv\Scripts\python -m pytest tests/test_service_ops_status.py tests/test_event_logging_status.py -q` passed with 9 passed on 2026-05-14.
- Stage 9 evidence: Updated operator and ICD docs in `docs/HOWTO.md`, `src/kazusa_ai_chatbot/brain_service/README.md`, `src/kazusa_ai_chatbot/db/README.md`, `src/kazusa_ai_chatbot/reflection_cycle/README.md`, and `src/kazusa_ai_chatbot/self_cognition/README.md`. Docs now state `/ops/*` is trusted-operator/local-service only until a future auth plan, in-process logging does not prove OS kills/interpreter aborts/host crashes/power loss/external restarts, and event-log retention/archival is deferred to a separate approved plan. Static checks were reviewed: event-log collection/API names appeared only in event-logging modules, bootstrap, focused tests, and docs; the focused public event-logging surface had no `payload: dict`, public ellipsis signatures, or public generic `record_event`; forbidden raw-field grep had no matches in `event_logging` or `db.event_logging`, with remaining matches limited to pre-existing brain-service/service request-envelope handling; `platform_channel_id` matches were limited to scope input handling and hashing assertions; import-boundary matches were limited to event-logging internals, DB bootstrap, docs, and focused DB tests; task-creation matches were pre-existing service/worker/FastAPI/RAG-cache task paths, not event-log writers; the local self-cognition ledger file remained present; and `/ops/*` route grep returned exactly `/ops/runtime-status`, `/ops/reflection/stats`, and `/ops/self-cognition/stats`. Deterministic verification passed: `tests/test_event_logging_repository.py` 3 passed, `tests/test_event_logging_interface.py` 6 passed, `tests/test_event_logging_status.py` 5 passed, `tests/test_self_cognition_event_logging.py` 2 passed, `tests/test_reflection_event_logging.py` 3 passed, `tests/test_service_event_logging.py` 4 passed, `tests/test_dispatcher_event_logging.py` 5 passed, `tests/test_rag_dialog_event_logging.py` 7 passed, `tests/test_service_ops_status.py tests/test_service_health.py` 5 passed, `tests/test_self_cognition_tracking.py tests/test_self_cognition_integration.py` 26 passed, `tests/test_reflection_cycle_stage1c_worker.py tests/test_reflection_cycle_stage1c_service.py` 13 passed, and `tests/test_db.py` 53 passed with 13 deselected. Live DB smoke command was not run because the plan requires intentional MongoDB availability for the write-producing operator snapshot command; deterministic route/export tests covered the aggregate payload and sanitized export shape.
- Independent code review: No separate reviewer was available in this session, so Codex performed the required fresh-review gate on 2026-05-14 after rereading the plan and inspecting the full diff, new event-logging modules, DB adapter/bootstrap changes, service/operator routes, reflection/self-cognition/dispatcher/RAG/dialog instrumentation, docs, and focused tests. Findings fixed: `write_analysis_snapshot` did not catch cancellation even though event-log writes must be cancellation-safe; the self-cognition sync wrapper had malformed call indentation and missing `event_log_mirror` docstring coverage; dialog evaluator telemetry normalized a malformed top-level list into an empty dict, changing failure behavior, so the test now uses a missing-field object and the runtime keeps the old failure behavior for non-mapping results; `record_self_cognition_event` copied the budget mapping wholesale, so it now persists only the four approved budget counters; reflection run DB-operation telemetry recorded `latency_ms=0`, so reflection upserts now use a measured helper; and `_BootstrapDb` collection-name indentation drift was corrected. Reruns after fixes: `tests/test_event_logging_status.py` 6 passed, `tests/test_self_cognition_event_logging.py tests/test_self_cognition_tracking.py` 19 passed, `tests/test_event_logging_interface.py` 7 passed, `tests/test_rag_dialog_event_logging.py tests/test_dialog_agent.py` 18 passed, `tests/test_reflection_event_logging.py tests/test_reflection_cycle_stage1c_worker.py` 10 passed, and the complete deterministic verification list passed again with `tests/test_event_logging_repository.py` 3 passed, `tests/test_event_logging_interface.py` 7 passed, `tests/test_event_logging_status.py` 6 passed, `tests/test_self_cognition_event_logging.py` 2 passed, `tests/test_reflection_event_logging.py` 3 passed, `tests/test_service_event_logging.py` 4 passed, `tests/test_dispatcher_event_logging.py` 5 passed, `tests/test_rag_dialog_event_logging.py` 7 passed, `tests/test_service_ops_status.py tests/test_service_health.py` 5 passed, `tests/test_self_cognition_tracking.py tests/test_self_cognition_integration.py` 26 passed, `tests/test_reflection_cycle_stage1c_worker.py tests/test_reflection_cycle_stage1c_service.py` 13 passed, and `tests/test_db.py` 53 passed with 13 deselected. Final static checks showed no event-logging-surface `payload: dict` public contract, no forbidden raw fields in `event_logging` or `db.event_logging`, exactly three `/ops/*` routes, and no event-log background task writer. Broad legacy greps still show unrelated pre-existing test helper names and brain-service message-envelope request handling; those are outside the event-log interface and operator payloads. Approval status: approved with residual risks limited to deferred auth/retention/external-supervisor coverage already documented by the plan.
