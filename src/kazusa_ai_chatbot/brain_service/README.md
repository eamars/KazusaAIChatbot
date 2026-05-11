# Brain Service Interface Control Document

## Document Control

- ICD id: `BS-ICD-001`
- Owning package: `kazusa_ai_chatbot.brain_service`
- Interface boundary: runtime adapters and debug clients -> brain service HTTP API
- Runtime consumers: platform adapters, debug adapter, scheduler dispatcher,
  service queue, persona graph, persistence, and health checks
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
- runtime adapter registration lets scheduler/proactive sends call adapter
  callback endpoints.

## Scope

This ICD covers:

- Public FastAPI endpoints owned by the brain service.
- Pydantic request and response models in `brain_service.contracts`.
- Adapter obligations before and after calling `/chat`.
- Delivery tracking and delivery receipt lifecycle.
- Runtime adapter registration and heartbeat protocol for scheduled delivery.
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

### Scheduler Dispatcher

The scheduler dispatcher is an internal caller that uses registered adapter
callbacks to send accepted future messages. Scheduler delivery and normal
`/chat` delivery receipts stay on separate service paths.

## Boundary Summary

```text
platform event
  -> adapter platform parser
  -> adapter-owned envelope normalizer
  -> POST /chat
  -> brain queue and persona graph
  -> ChatResponse(messages, use_reply_feature, delivery_tracking_id)
  -> adapter platform send
  -> platform returns outbound message id
  -> POST /delivery_receipt
  -> conversation row gains delivered platform message id
```

For scheduled or proactive delivery:

```text
adapter startup
  -> POST /runtime/adapters/register
  -> periodic POST /runtime/adapters/heartbeat
  -> brain scheduler selects registered adapter
  -> brain calls adapter callback /send_message
  -> adapter returns SendResult(message_id)
```

The two flows are related but separate. Normal `/chat` delivery receipts update
assistant conversation rows. Runtime callback sends return `SendResult` for
scheduler execution and are not backfilled into normal-chat rows by this ICD.

## Public Endpoints

### `GET /health`

Response model: `HealthResponse`.

Purpose:

- Report service readiness for the database, scheduler, and Cache2.
- Provide operational visibility without running the persona graph.

Adapters can use this endpoint for startup diagnostics. Chat availability is
reported by the health status, database status, and scheduler status fields.

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
| `channel_name` | no | adapter | Human-readable channel label. |
| `content_type` | no | adapter | High-level input type, currently usually `text`. |
| `message_envelope` | yes | adapter | Typed envelope defined by the message envelope ICD. |
| `timestamp` | no | adapter | Adapter event timestamp. Empty string means service receive time is used. |
| `debug_modes` | no | adapter/debug client | Debug toggles: `listen_only`, `think_only`, `no_remember`. |

`ChatResponse` fields:

| Field | Owner | Meaning |
| --- | --- | --- |
| `messages` | brain | Text messages the adapter should render to the platform. Empty means no user-visible reply. |
| `content_type` | brain | Outbound content type. Current normal value is `text`. |
| `attachments` | brain | Outbound attachments. Currently reserved for future use. |
| `use_reply_feature` | brain | Adapter should use native reply rendering for the first outbound message when possible. |
| `scheduled_followups` | brain | Count of scheduled follow-ups accepted during the turn. |
| `delivery_tracking_id` | brain | Brain-generated identifier for the assistant row that should receive a delivery receipt. Empty means no receipt should be posted. |

Adapter responsibilities:

- Send a valid typed `message_envelope`.
- Represent visible platform mentions in `message_envelope.body_text` as
  readable platform-neutral tokens such as `@display name`,
  `#channel-name`, `@everyone`, or occurrence fallbacks such as
  `@mentioned-user-1`.
- Keep native platform mention syntax, such as CQ at codes and Discord
  mention tags, only in `message_envelope.raw_wire_text`.
- Keep raw platform ids out of `message_envelope.body_text`; typed mention
  identity belongs in `message_envelope.mentions`.
- Preserve the inbound platform message id when available.
- Treat an empty `messages` list as no outbound send.
- Honor `use_reply_feature` for the first outbound message when the platform
  has a native reply mechanism.
- Post `/delivery_receipt` after a successful platform send when both
  `delivery_tracking_id` and an outbound platform message id exist.

Brain service responsibilities:

- The brain service validates `ChatRequest` through Pydantic with
  `extra="forbid"`.
- The brain service can collapse queued chat messages before graph execution.
- The brain service returns `ChatResponse` before all post-turn background work
  has necessarily completed.
- The brain service generates a non-empty `delivery_tracking_id` only when it
  returns user-visible messages and will persist an assistant conversation row.
- Existing adapters that ignore `delivery_tracking_id` remain compatible; they
  simply do not enable future reply hydration by delivered outbound id.

### `POST /delivery_receipt`

Request model: `DeliveryReceiptRequest`.

Response model: `DeliveryReceiptResponse`.

Purpose:

- Let an adapter report the platform-generated outbound message id for a
  previously returned normal `/chat` response.
- Enable later native replies that carry only a platform reply message id to be
  resolved against the assistant conversation row.

`DeliveryReceiptRequest` fields:

| Field | Required | Owner | Meaning |
| --- | --- | --- | --- |
| `platform` | yes | adapter | Same platform key used in the original `/chat` request. |
| `platform_channel_id` | no | adapter | Same channel id used in the original `/chat` request. |
| `delivery_tracking_id` | yes | brain then adapter | Value returned by `ChatResponse.delivery_tracking_id`. |
| `platform_message_id` | yes | adapter | Outbound message id returned by the platform send API. |
| `delivered_at` | no | adapter/brain | Delivery timestamp. Empty string means the brain uses current UTC time. |
| `adapter` | no | adapter | Adapter implementation name, such as `napcat` or `discord`. |

`DeliveryReceiptResponse` fields:

| Field | Meaning |
| --- | --- |
| `status` | `updated` when a matching assistant row was updated; `not_found` when no row matched. |
| `updated` | Boolean mirror of whether the row was matched and updated. |

Adapter delivery receipt responsibilities:

- Send the receipt after the platform send succeeds.
- Use the platform's durable outbound message id.
- Keep the user-visible platform send delivered when a receipt post fails.
- Retry `not_found` briefly because `/chat` can return before assistant-row
  persistence finishes.
- Stop retrying on HTTP transport errors or unexpected statuses after logging
  enough scope for diagnosis.

Current adapter policy:

| Adapter | Normal `/chat` receipt behavior |
| --- | --- |
| NapCat QQ | Reports `send_msg.data.message_id` after successful `send_msg`; retries `not_found` with short bounded delays. |
| Discord | Reports the first sent Discord `Message.id` after successful normal chat send; retries `not_found` with short bounded delays. |
| Debug | Omits receipts because it has no durable external platform message id. |

The first-message policy for Discord is a known reply-hydration limitation of
the current one-id receipt contract. Until multi-chunk receipts are wired,
native replies to non-first Discord chunks use adapter-provided metadata only.
Multi-id delivery receipts require an updated service and DB contract.

Brain service delivery receipt responsibilities:

- The brain service updates assistant rows by generated
  `delivery_tracking_id` and platform. Non-empty channel scope is added as an
  optional disambiguator.
- Delivery receipts match generated tracking ids and platform scope.
- The receipt update leaves embeddings and RAG cache state unchanged.
- A `not_found` response is a retryable race signal for adapters.

### `POST /runtime/adapters/register`

Request model: `RuntimeAdapterRegistrationRequest`.

Response model: `RuntimeAdapterRegistrationResponse`.

Purpose:

- Register a cross-process adapter callback so the brain scheduler can deliver
  accepted future messages through that adapter.

Fields:

| Field | Required | Owner | Meaning |
| --- | --- | --- | --- |
| `platform` | yes | adapter | Platform key used by scheduled tasks. |
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

The normal `/chat` path has two visible timings:

1. The brain returns `ChatResponse` to the adapter.
2. The brain persists assistant post-turn data and background state updates.

The response is intentionally returned before every background action has
necessarily completed. Delivery receipt adapters therefore need bounded
`not_found` retry behavior. The service side remains simple: it updates the row
if the row exists and returns `not_found` otherwise.

Callers treat `delivery_tracking_id` visibility as eventually consistent with
assistant-row persistence.

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
- Calling `/chat` and rendering `ChatResponse.messages`.
- Extracting durable outbound platform message ids after successful sends.
- Posting `/delivery_receipt` when the adapter supports durable outbound ids.
- Registering and heartbeating runtime callback URLs when scheduler delivery
  is enabled.

The brain service owns:

- Pydantic API validation.
- Queueing and collapse policy.
- Global identity resolution.
- Reply context hydration from typed metadata and delivered conversation rows.
- Persona graph invocation.
- Assistant row persistence and delivery receipt updates.
- Runtime adapter registry integration for scheduler dispatch.
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
- Changing delivery receipts from one platform id to multiple ids is breaking
  for reply hydration semantics unless this ICD and the DB contract are updated
  first.
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
through heartbeat/startup behavior. Missing runtime adapters cause scheduler
delivery validation to reject or fail scheduled sends according to dispatcher
policy.
