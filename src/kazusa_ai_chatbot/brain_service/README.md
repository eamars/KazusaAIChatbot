# Brain Service Interface Control Document

## Document Control

- ICD id: `BS-ICD-001`
- Owning package: `kazusa_ai_chatbot.brain_service`
- Interface boundary: runtime adapters and debug clients -> brain service HTTP API
- Runtime consumers: platform adapters, debug adapter, calendar scheduler,
  dispatcher, service queue, persona graph, persistence, and health checks
- Contract owners: brain service API models in `contracts.py` and FastAPI route
  handlers in `service.py`

This document defines the service-level protocol between platform adapters and
the Kazusa brain service. It is the source of truth for request/response
lifecycle, delivery tracking, runtime adapter registration, and the ownership
boundary between adapter transport code and the platform-agnostic brain.

For local setup, environment variables, service startup commands, adapter run
commands, and test commands, use the operational
[HOWTO](../../../docs/HOWTO.md). This ICD owns normative API schemas.

## Purpose

The brain service is the platform-agnostic character runtime. It accepts typed
adapter input, queues and runs the persona pipeline, returns outbound text, and
records durable state after a turn. Platform adapters own platform transport;
the brain service owns semantic processing, persistence coordination, and API
validation.

The service protocol must stay explicit because adapters now participate in
both inbound and outbound state:

- inbound `/chat` sends normalized message metadata and the typed message
  envelope,
- outbound `ChatResponse.delivery_tracking_id` lets an adapter report the
  platform message id after delivery,
- `/delivery_receipt` stores delivered platform message ids for future native
  reply hydration,
- runtime adapter registration lets dispatcher/proactive sends call adapter
  callback endpoints.
- background-work worker lifecycle is started, stopped, and reported by the
  service runtime as an internal executor while accepted-task result delivery
  still returns through cognition/dialog.

Local process lifecycle management is owned by the separate top-level
`control_console` package and `kazusa-control-console` command. The brain
service does not mount, import, or control console routes.

## Scope

This ICD covers:

- Public FastAPI endpoints owned by the brain service.
- Pydantic request and response models in `brain_service.contracts`.
- Adapter obligations before and after calling `/chat`.
- Delivery tracking and delivery receipt lifecycle.
- Runtime adapter registration and heartbeat protocol for dispatcher or
  proactive callback delivery.
- Accepted-task delayed-work lifecycle plus internal background-work worker
  enablement, liveness, and result-ready delivery boundary.
- Reply-context hydration behavior when an adapter supplies only a platform
  reply message id.
- Compatibility rules for adding service fields and endpoints.

Related contracts are owned by their subsystem ICDs: platform parsing lives in
adapter docs/code, database collection ownership lives in the database ICD, and
the typed `message_envelope` schema lives in the message envelope ICD.

## Parties

### Runtime Adapter

A runtime adapter is a platform-facing process or module that receives platform
events, normalizes them, calls `/chat`, renders returned messages to the
platform, and optionally reports delivery receipts.

Examples: Discord adapter, NapCat QQ adapter, debug adapter, and future
platform adapters.

### Brain Service

The brain service is the FastAPI service that validates adapter requests,
hydrates reply context, queues live chat work, runs the persona graph, persists
conversation rows, exposes health, and receives delivery receipts.

The brain service consumes typed adapter metadata and normalized envelope
fields.

### Dispatcher Or Proactive Callback Caller

The dispatcher or a future proactive-output owner can use registered adapter
callbacks for trusted non-`/chat` delivery. Callback delivery and normal
`/chat` delivery receipts stay on separate service paths. Calendar scheduler
triggers do not by themselves send prewritten visible text.

## Boundary Summary

```text
platform event
  -> adapter platform parser
  -> adapter-owned envelope normalizer
  -> POST /chat
  -> brain queue and persona graph
  -> selected text surface outputs
  -> ChatResponse(messages, use_reply_feature, delivery_mentions, delivery_tracking_id, cognition_graph?)
  -> adapter ordered platform send sequence
  -> platform returns outbound message id
  -> POST /delivery_receipt
  -> conversation row gains delivered platform message id
```

For dispatcher or proactive callback delivery:

```text
adapter startup
  -> POST /runtime/adapters/register
  -> periodic POST /runtime/adapters/heartbeat
  -> brain delivery owner selects registered adapter
  -> brain calls adapter callback /send_message
  -> adapter returns SendResult(message_id)
```

The two flows are related but separate. Normal `/chat` delivery receipts update
assistant conversation rows. Runtime callback sends return `SendResult` for
dispatcher/proactive execution and are not backfilled into normal-chat rows by
this ICD.

Normal `/chat` responses and dispatcher/proactive callback sends may include
optional `delivery_mentions` metadata. This is adapter-owned rendering
metadata: the brain keeps outbound text platform-neutral with visible
`@display_name` tokens, and the adapter replaces matching tokens with native
user mentions only when feasible. Missing, empty, or unrenderable mention
metadata must not block text delivery.

Normal `/chat` `messages` are ordered logical outbound chat messages. The
adapter sends each string as a separate normal chat message in order. The first
message is sent immediately and may use native reply rendering when requested;
follow-up messages are adapter-owned background sends with short
length-derived delays. Inline delivery mentions are applied per logical
message before any platform chunking or segment conversion. Dispatcher or
proactive callback `/send_message` remains a single-message delivery surface.

Normal `/chat` responses may also include optional `cognition_graph` telemetry
for local operator inspection. It is a bounded graph snapshot derived from the
current graph result and consolidation state. It must not include prompts,
embeddings, raw messages, message envelopes, raw user input, secrets, or
unbounded memory content. Adapters may ignore this field.

The service also keeps a process-local latest cognition graph snapshot for
trusted local operator inspection. Normal chat/debug turns update it from the
`/chat` response graph. Completed self-cognition cases update it from bounded
self-cognition artifacts. Reading this snapshot is observational only and must
not trigger cognition.

The service is one caller of the generic runtime coordination API in
`kazusa_ai_chatbot.runtime_coordination`. For the canonical channel scope
`(platform, platform_channel_id, channel_type)`, inbound foreground work asks
the coordinator to cancel lower-precedence background pipelines in the same
scope before the queued turn is admitted. This rule is scoped and generic:
future foreground applications must call the same interface instead of
importing `/chat` queue state or adding their own gates. Different channel
scopes remain independent.

### Live Chat Intake And Settlement

The normal active-chat path persists each typed inbound fragment, runs the
compact frontline stage from `kazusa_ai_chatbot.relevance` through the
existing `RELEVANCE_AGENT_LLM` route, and keeps only `discard`, `start`, or
`append` as the frontline vocabulary. Group turns use a six-second quiet
window and ten-second hard deadline. The service owns the pending-turn heap,
fragment chronology, same-author candidate and silent-prelude slots, one
bounded settled `wait`, stale-version invalidation, a pre-deadline ingress
barrier, and the atomic cognition claim. The same
`kazusa_ai_chatbot.relevance`
package, using the same `RELEVANCE_AGENT_LLM` route, owns the settled
character-level
`ignore/proceed/wait` judgment; deterministic code applies the validated action
and never rewrites a valid semantic choice.

The model-facing intake projection carries typed private/group scope, runtime
character identity, semantic target/reply labels, and only eligible
same-author/same-channel candidates. A present reply with unresolved author is
`unknown_participant`. Explicit-third-party and unresolved-reply discards do
not become later preludes. Latest bot dialog is continuity evidence only within
the 180-second active scene and is never an append slot. Frontline renders a
scope-specific prompt and exposes only candidate-supported actions; returned
slot labels must exist in the exact capped projection shown to the model.
Settled fresh history
maps production identities to character, current-author, and other-participant
relations before it reaches the model. Conversation row order also marks each
external history row as before, after, or unknown relative to the active turn;
when the bounded history window has evicted the active row, persisted server
timestamps retain that relation, while missing or equal timestamps remain
unknown. Rows between active fragments are `during_active_turn` and may resolve
requests expressed in earlier fragments; after-turn rows may resolve the whole
assembled request. The settled projection
labels the assembled author as the current human and repeats only the bounded
final fragment to make recipient correction salient without another model
call. The first assessment may choose one bounded `wait`; the hard-deadline
prompt offers only `ignore` or `proceed`.

Private adjacency-only coalescing retains the existing immediate-ready timing
and shows the full coalesced logical input to frontline before attaching its
individual fragments to the exact survivor turn. An appended request completes
with an empty response after attachment; only the response owner receives the
assembled turn's visible response. A settled native-reply request reaches the
adapter only when that response owner is also the effective latest fragment;
otherwise the visible response is delivered without a misleading quote.
Fresh settled history excludes active-turn rows. The opening/newest four-image
budget is shared across reassessments, with overflow exposed to the settled
fail-closed judgment.
Group burst pruning and group pre-relevance coalescing are not part of the
active queue contract. A claimable `proceed` is the only path into the existing
cognition and dialog graph.

Visible `/chat` delivery follows selected `SurfaceOutputV1` text surfaces.
Private action results, private finalization, calendar-triggered action
results, and no-visible-output decisions may still make an episode
consolidatable, but they do not create adapter sends or delivery receipts by
themselves.

## Public Endpoints

### `GET /health`

Response model: `HealthResponse`.

Purpose:

- Report service readiness for the database, service graph, and Cache2.
- Provide operational visibility without running the persona graph.

Adapters can use this endpoint for startup diagnostics. Chat availability is
reported by the health status, database status, and service-graph status
fields.

The response field named `scheduler` is a legacy readiness field in
`HealthResponse`; it is not the calendar scheduler liveness contract. Trusted
operators use `/ops/runtime-status` for calendar scheduler enablement,
configuration, and task liveness.

### `GET /ops/latest-cognition-graph`

Response model: `OpsLatestCognitionGraphResponse`.

Purpose:

- Return the latest bounded cognition graph snapshot reported by live chat,
  debug chat, or self-cognition.
- Support the local control-console Overview graph without running cognition.
- Return `cognition_graph: null` when no completed run has published telemetry
  since service startup.

The endpoint is process-local and read-only. It must not expose prompts,
embeddings, raw messages, message envelopes, raw source packets, secrets, or
unbounded memory content.

### `GET /ops/runtime-status`

Response model: `OpsRuntimeStatusResponse`.

Purpose:

- Report aggregate runtime observability for trusted local operators.
- Keep worker state and event-log health out of adapter `/health` checks.

The response exposes:

- `status`, `generated_at`, and `window_hours`;
- `config` values for calendar, reflection, self-cognition, and background-work
  worker
  enablement and intervals;
- calendar claim, lease, and retry settings;
- process-local worker `enabled` and `task_alive` values;
- latest event timestamp/status for process, reflection, self-cognition, and
  background-work worker activity;
- deterministic `semantic_descriptors`.

This endpoint must not expose message bodies, prompt text, generated dialog,
event-log row bodies, channel ids, secrets, callback tokens, or per-user
details.

### `GET /ops/reflection/stats`

Response model: `OpsStatsResponse`.

Purpose:

- Report aggregate reflection event-log counts for a bounded window.
- Return latest event/run refs and semantic health labels.

This endpoint supplements `character_reflection_runs`; it does not replace the
reflection run ledger and does not expose raw reflection output.

### `GET /ops/self-cognition/stats`

Response model: `OpsSelfCognitionStatsResponse`.

Purpose:

- Report service-owned self-cognition `enabled` and `task_alive` state.
- Report aggregate self-cognition event-log counts for a bounded window.
- Distinguish internal-only activity from private action-attempt activity
  through deterministic labels.

`self_cognition_liveness=inactive` only means no self-cognition run events were
recorded in the window. Callers must use `enabled` and `task_alive` to
distinguish a disabled worker from an enabled idle worker.

This endpoint does not expose source packet text, route reasoning, action
candidate text, delivery arguments, or generated dialog.

`/ops/*` endpoints are local-service/trusted-operator endpoints in the current
contract. They have no auth layer here and must not be internet exposed until a
separate authentication and authorization plan exists.

### `POST /chat`

Request model: `ChatRequest`.

Response model: `ChatResponse`.

Purpose:

- Submit one normalized inbound platform message to the brain.
- Receive zero or more outbound text messages plus platform-rendering intent.

`ChatRequest` fields:

| Field | Required | Owner | Meaning |
| --- | --- | --- | --- |
| `platform` | yes | adapter | Stable platform key, such as `qq`, `discord`, or `debug`. |
| `platform_channel_id` | no | adapter | Platform channel or DM identifier. Empty string only when the platform has no channel id. |
| `channel_type` | no | adapter | Channel class. Current operational values are `group` and `private`. |
| `platform_message_id` | no | adapter | Inbound platform message id, if available. |
| `platform_user_id` | yes | adapter | Platform id of the inbound message author. |
| `platform_bot_id` | no | adapter | Platform id of the active bot account, if available. |
| `display_name` | no | adapter | Display name of the inbound author. |
| `channel_name` | no | adapter | Optional human-readable channel label. The brain may use a sanitized usable group label as weak scene text, but synthetic ids or generic labels are ignored. |
| `content_type` | no | adapter | High-level input type, currently usually `text`. |
| `message_envelope` | yes | adapter | Typed envelope defined by the message envelope ICD. |
| `local_timestamp` | no | adapter | Adapter event time as configured-local wall-clock text. Empty string means service receive time is used. |
| `debug_modes` | no | adapter/debug client | Debug toggles: `listen_only`, `think_only`, `no_remember`. |

`ChatResponse` fields:

| Field | Owner | Meaning |
| --- | --- | --- |
| `messages` | brain | Ordered text messages the adapter should render as separate normal chat sends. Empty means no user-visible reply. |
| `content_type` | brain | Outbound content type. Current normal value is `text`. |
| `attachments` | brain | Outbound attachments. Currently reserved for future use. |
| `use_reply_feature` | brain | Adapter should use native reply rendering for the first outbound message when possible. |
| `delivery_mentions` | brain then adapter | Optional platform-neutral inline mention render candidates. The brain emits these only for authored `@display_name` text with matching scoped user identity; adapters decide native rendering, channel feasibility, and no-op fallback. |
| `scheduled_followups` | brain | Count of scheduled future-cognition follow-ups accepted during the turn. |
| `delivery_tracking_id` | brain | Brain-generated identifier for the ordered assistant response sequence. Empty means no receipt should be posted. |

Adapter responsibilities:

- Send a valid typed `message_envelope`.
- Represent visible platform mentions in `message_envelope.body_text` as
  readable platform-neutral tokens such as `@display name`,
  `#channel-name`, `@everyone`, or platform-neutral fallbacks such as
  `@user`, `@role`, `#channel`, or `@entity`.
- Keep native platform mention syntax, such as CQ at codes and Discord
  mention tags, only in `message_envelope.raw_wire_text`.
- Do not emit legacy occurrence placeholders such as `@mentioned-user-1` in
  semantic fields. Brain-service intake validates semantic storage fields and
  rejects rows containing transport markers, occurrence placeholders, or
  platform-qualified fallback labels before persistence.
- Keep raw platform ids out of `message_envelope.body_text`. Typed mention
  identity belongs in `message_envelope.mentions` and reply metadata.
- Preserve the inbound platform message id when available.
- Treat an empty `messages` list as no outbound send.
- Send non-empty `messages` in order as separate normal chat messages.
- Honor `use_reply_feature` only for the first outbound message when the
  platform has a native reply mechanism.
- Own follow-up message delay and task lifecycle without blocking the adapter
  on the delay.
- Render `delivery_mentions` best-effort by replacing matching authored
  `@display_name` text inline in each logical message when present and
  feasible; otherwise send the original text unchanged.
- Post `/delivery_receipt` after each successful logical message send when
  `delivery_tracking_id` and that logical message's outbound platform id
  exist.

Brain service responsibilities:

- The brain service validates `ChatRequest` through Pydantic with
  `extra="forbid"`.
- The brain service can coalesce adjacent private follow-ups before frontline
  execution; group messages are retained as separate persisted fragments.
- The intake worker awaits only the serialized frontline call. Settlement
  timers and cognition execute in the separate settlement worker, so one
  waiting group turn does not block other intake.
- Frontline and settled calls share one FIFO relevance slot. Active settled
  relevance and cognition retain primary-interaction priority, while quiet
  timers hold no model or worker capacity.
- The settlement coordinator derives model-facing open and prelude slots from
  the same state used to apply the returned slot labels.
- The brain service owns foreground runtime-coordination admission for inbound
  chat turns and releases the foreground handle when the queued turn is
  processed, dropped, collapsed, rejected, or drained during shutdown.
- The brain service returns `ChatResponse` before all post-turn background work
  has necessarily completed.
- The brain service generates a non-empty `delivery_tracking_id` only when it
  returns user-visible messages and will persist an assistant conversation row.
- The brain service may run post-turn consolidation for selected surface
  outputs, action results, or private finalization even when `messages` is
  empty.
- Existing adapters that ignore `delivery_tracking_id` remain compatible; they
  simply do not enable future reply hydration by delivered outbound id.

### `POST /delivery_receipt`

Request model: `DeliveryReceiptRequest`.

Response model: `DeliveryReceiptResponse`.

Purpose:

- Let an adapter report the platform-generated outbound message id for one
  logical message in a previously returned normal `/chat` response.
- Enable later native replies that carry only a platform reply message id to be
  resolved against the assistant conversation row.

`DeliveryReceiptRequest` fields:

| Field | Required | Owner | Meaning |
| --- | --- | --- | --- |
| `platform` | yes | adapter | Same platform key used in the original `/chat` request. |
| `platform_channel_id` | no | adapter | Same channel id used in the original `/chat` request. |
| `delivery_tracking_id` | yes | brain then adapter | Value returned by `ChatResponse.delivery_tracking_id`. |
| `logical_message_index` | yes | adapter | Zero-based index of the `ChatResponse.messages` item whose platform id is being reported. |
| `platform_message_id` | yes | adapter | Outbound message id returned by the platform send API. |
| `delivered_at` | no | adapter/brain | Delivery timestamp. Empty string means the brain uses current UTC time. |
| `adapter` | no | adapter | Adapter implementation name, such as `napcat` or `discord`. |

`DeliveryReceiptResponse` fields:

| Field | Meaning |
| --- | --- |
| `status` | `updated` when a matching assistant row was updated; `not_found` when no row matched. |
| `updated` | Boolean mirror of whether the row was matched and updated. |

Adapter delivery receipt responsibilities:

- Send one receipt after each logical message platform send succeeds.
- Set `logical_message_index` to the index in `ChatResponse.messages`.
- Use the platform's durable outbound message id.
- Keep the user-visible platform send delivered when a receipt post fails.
- Retry `not_found` briefly because `/chat` can return before assistant-row
  persistence finishes.
- Stop retrying on HTTP transport errors or unexpected statuses after logging
  enough scope for diagnosis.

Current adapter policy:

| Adapter | Normal `/chat` receipt behavior |
| --- | --- |
| NapCat QQ | Reports `send_msg.data.message_id` for each logical outbound message after successful `send_msg`; retries `not_found` with short bounded delays. |
| Discord | Reports the first sent Discord `Message.id` for each logical outbound message after successful normal chat send; retries `not_found` with short bounded delays. |
| Debug | Omits receipts because it has no durable external platform message id. |

Only the first platform id for each logical message is reported. Adapter- or
platform-created chunks beyond that first id are transport artifacts and are
not separate reply-hydration targets in this contract.

Brain service delivery receipt responsibilities:

- The brain service updates assistant rows by generated
  `delivery_tracking_id`, `logical_message_index`, and platform. Non-empty
  channel scope is added as an optional disambiguator.
- Delivery receipts match generated tracking ids, logical message indexes, and
  platform scope.
- The receipt update leaves embeddings and RAG cache state unchanged.
- A `not_found` response is a retryable race signal for adapters.

### `POST /runtime/adapters/register`

Request model: `RuntimeAdapterRegistrationRequest`.

Response model: `RuntimeAdapterRegistrationResponse`.

Purpose:

- Register a cross-process adapter callback so trusted dispatcher or proactive
  delivery owners can send through that adapter.

Fields:

| Field | Required | Owner | Meaning |
| --- | --- | --- | --- |
| `platform` | yes | adapter | Platform key used by callback delivery tasks. |
| `callback_url` | yes | adapter | Base URL exposed by the adapter process. |
| `shared_secret` | no | adapter/operator | Bearer token expected by the adapter callback, if configured. |
| `timeout_seconds` | no | adapter/operator | Brain-side timeout for callback sends. |

The brain service stores this registration in the live adapter registry.

### `POST /runtime/adapters/heartbeat`

Request model: `RuntimeAdapterRegistrationRequest`.

Response model: `RuntimeAdapterRegistrationResponse`.

Purpose:

- Refresh the same runtime adapter callback registration so brain restarts and
  adapter restarts can self-heal.

The payload contract is identical to `/runtime/adapters/register`.

Adapters heartbeat periodically while running. The brain service treats
heartbeat as an idempotent re-registration.

### `POST /event`

Request model: `EventRequest`.

Purpose:

- Accept generic platform or operator events that are not normal chat turns.

Current behavior is intentionally minimal. Operational event types need defined
validation, ownership, and side effects.

## Reply Context Hydration

Adapters may provide reply target metadata in `message_envelope.reply`:

- `platform_message_id`
- `platform_user_id`
- `global_user_id`
- `display_name`
- `excerpt`
- `derivation`

The brain service first uses adapter-provided reply metadata. If the adapter
provides a reply platform message id but omits author or excerpt metadata, the
brain service can look up a delivered conversation row by exact
`platform`, `platform_channel_id`, and `platform_message_id`.

Hydration rules:

- Adapter-provided metadata wins over database fallback metadata.
- Database fallback uses exact platform/channel/message-id scope.
- Missing fallback rows are allowed and should degrade to the original adapter
  metadata.

## Persistence Timing

The normal `/chat` path records the incoming user row before frontline
execution. If that row is not committed, the request fails closed and no
visible reply is released. Discarded, listen-only, and private-coalesced
inputs retain their persisted source rows. A settled turn is not allowed to
run on any fragment whose source row was not committed.

For visible assistant output, the brain writes one assistant row per logical
`ChatResponse.messages` item before returning `ChatResponse` to the adapter.
Rows from the same cognition share the same `delivery_tracking_id` and
`llm_trace_id`, and each row stores its `logical_message_index`. Row-scoped
outbound mention metadata is persisted when a returned logical message contains
an exact authored `@display_name` token with a matching delivery candidate.
Visible assistant rows are derived from selected text surface outputs.
Background state updates such as conversation progress and consolidation may
still run after the response has been released.

When an episode has no visible text surface, the brain returns an empty
`messages` list and no delivery tracking id. That episode can still be
consolidated when private action results, calendar-triggered action results,
private surface outputs, or private finalization exist.

When L2d selects `accepted_task_request`, deterministic action execution first
creates or reuses an accepted-task lifecycle row. New accepted tasks are then
materialized into an internal `background_work_request`, queued durably before
selected L3 text runs, and projected back to L3 as semantic accepted-task state.
Status checks use `accepted_task_status_check` and never enqueue a worker job.
Completed accepted-task-backed jobs later return as
`accepted_task_result_ready` cognitive episodes. Background-work workers must
not call adapters, dispatcher delivery, or cognition directly.

Delivery receipt adapters may still need bounded `not_found` retry behavior
for transport timing and cross-process delivery, but a non-empty
`delivery_tracking_id` means the logical assistant rows for that response were
committed before the response was returned.

## Debug Modes

`DebugModesIn` currently exposes:

| Field | Meaning |
| --- | --- |
| `listen_only` | Run intake/listening behavior without user-visible response. |
| `think_only` | Run thinking but suppress returned messages. |
| `no_remember` | Skip consolidation/remembering side effects for the turn. |

Debug modes are explicit service controls. Adapters and debug clients set them
directly, and downstream stages receive them as structured state.

## Ownership Rules

Runtime adapters own:

- Platform event subscription and SDK/websocket lifecycle.
- Platform-specific parsing and outgoing rendering.
- Normalizing platform wire syntax into typed envelope fields.
- Replacing only platform mention tokens with readable mention tokens before
  `/chat`; adapters must not rewrite pronouns, aliases, or other authored
  text semantically.
- Calling `/chat` and rendering `ChatResponse.messages` as ordered normal chat
  sends.
- Extracting durable outbound platform message ids after successful logical
  message sends.
- Posting `/delivery_receipt` when the adapter supports durable outbound ids.
- Registering and heartbeating runtime callback URLs when dispatcher or
  proactive callback delivery is enabled.

The brain service owns:

- Pydantic API validation.
- Queueing and collapse policy.
- Global identity resolution.
- Reply context hydration from typed metadata and delivered conversation rows.
- Persona graph invocation.
- Assistant row persistence, logical message indexes, and delivery receipt
  updates.
- Runtime adapter registry integration for dispatcher/proactive callback
  dispatch.
- Health response composition.

The database package owns:

- MongoDB collection access.
- Delivery receipt row update mechanics.
- Conversation lookup by delivered platform message id.
- Index definitions required by service lookup paths.

## Compatibility Rules

- Adding optional response fields with safe defaults is compatible.
- Adding optional request fields is compatible only when old adapters can omit
  them without behavior loss.
- Adding required request fields to `/chat`, `/delivery_receipt`, or runtime
  adapter registration is breaking and requires coordinated adapter updates.
- Changing the meaning of `platform`, `platform_channel_id`,
  `platform_message_id`, or `delivery_tracking_id` is breaking.
- Changing delivery receipts beyond one platform id per logical message is
  breaking for reply hydration semantics unless this ICD and the DB contract
  are updated first.
- Existing adapters may ignore `delivery_tracking_id`; this preserves chat
  delivery but disables delivered-id reply fallback for their outbound rows.

## Failure Behavior

Invalid `/chat` payloads fail at the FastAPI/Pydantic boundary.

Brain service failures during graph execution should return an operational
fallback response or an empty response depending on where the failure occurs.
Adapters should log service failures and avoid sending partially trusted
content.

Receipt failure behavior:

- `updated`: adapter stops.
- `not_found`: adapter may retry within a bounded delay schedule.
- HTTP transport error: adapter logs and stops; user-visible send remains
  delivered.
- Unexpected receipt status: adapter logs and stops.

Runtime callback registration failures should be logged by adapters and retried
through heartbeat/startup behavior. Missing runtime adapters cause dispatcher
delivery validation to reject or fail callback sends according to dispatcher
policy.
