# Message Envelope Interface Control Document

## Document Control

- ICD id: `ME-ICD-001`
- Owning package: `kazusa_ai_chatbot.message_envelope`
- Interface boundary: runtime adapters -> brain service `/chat`
- Runtime consumers: service intake, queueing, conversation storage, RAG,
  cognition, dialog, and consolidation
- Extension owners: platform adapters and shared envelope protocol
  implementers

This document defines the formal contract between platform adapters and the
brain service. It is the source of truth for what data may cross the `/chat`
boundary and who is responsible for producing, validating, storing, and
consuming each field.

## Purpose

The message envelope prevents platform transport syntax from becoming chatbot
content. Platform-specific syntax includes mention tags, reply markers, CQ
codes, channel tags, role tags, emoji tags, adapter-only attachment payloads,
and any other wire representation that belongs to a chat platform rather than
to the user's authored message.

The brain service must receive semantic content and typed metadata:

- `body_text`: user-authored content only.
- `raw_wire_text`: original transport text for audit and replay only.
- `mentions`: typed mention records.
- `reply`: typed reply target, when present.
- `attachments`: normalized attachment references.
- `addressed_to_global_user_ids`: typed inbound addressee ids.
- `broadcast`: assistant-authored channel-wide output marker; inbound user
  envelopes must set it to `False`.

The local LLM path relies on this boundary. Downstream prompts and agents should
reason over semantic fields, not rediscover platform syntax from prose.

## Role Vocabulary

The word `role` is reserved for conversation author role after the envelope
enters the brain service. Other role-like concepts MUST use their own field
names so prompts and typed dicts do not force local LLMs or maintainers to infer
the namespace from shape alone.

| Namespace | Field / Type | Allowed Values | Owner | Meaning |
| --- | --- | --- | --- | --- |
| Conversation author | `ConversationMessageDoc.role`, `ConversationAuthorRole` | `user`, `assistant` | brain storage | Author of a stored conversation row. |
| Mention entity | `Mention.entity_kind`, `MentionEntityKind` | `bot`, `user`, `platform_role`, `channel`, `everyone`, `unknown` | adapter envelope | Kind of platform entity mentioned by the inbound message. |
| Referent grammar | `referents[].referent_role`, `ReferentRole` | `subject`, `object`, `time` | decontextualizer | Grammatical role of a phrase needing reference resolution. |
| Bot permission | `DispatchContext.bot_permission_role`, `BotPermissionRole` | dispatcher permission labels | dispatcher | Permission level used to expose or reject scheduled tool calls. |

`bot` means platform-account identity only. The active persona is the
`character`; generated stored conversation output is the `assistant`.

## Scope

This ICD covers:

- The `/chat` request shape accepted by the brain service.
- The `MessageEnvelope` shape embedded in every `/chat` request.
- Responsibilities of runtime adapters before calling `/chat`.
- Responsibilities of the brain service after receiving `/chat`.
- Extension points exported by `kazusa_ai_chatbot.message_envelope`.
- Storage and retrieval invariants for typed conversation rows.
- Failure behavior when the contract is violated.

This ICD does not cover:

- Outbound platform rendering details such as how an adapter sends messages
  back to a platform.
- Platform login, websocket lifecycle, heartbeat, or callback registration.
- Durable database migration procedures outside the typed fields described
  here.
- Character voice, dialog style, or model prompt content.

## Parties

### Runtime Adapter

A runtime adapter is the platform-facing process or module that receives
platform events and forwards normalized chat requests to the brain service.

Adapter examples include platform-specific runtime adapters, the debug adapter,
and any future adapter registered with the service. Each adapter owns its own
platform parser and normalizer implementation.

### Brain Service

The brain service is the platform-agnostic FastAPI service that accepts `/chat`,
queues work, builds graph state, runs the persona pipeline, persists
conversation rows, and returns text responses plus reply intent.

The brain service consumes only the typed envelope. It does not parse platform
wire syntax.

### Message Envelope Package

`kazusa_ai_chatbot.message_envelope` owns shared TypedDicts, Protocols,
registries, attachment policy, and platform-neutral resolver slots.

The package does not own concrete platform normalizers. Platform normalizers
belong in adapter modules.

## Normative Language

The words `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative:

- `MUST`: required for the interface to be valid.
- `MUST NOT`: forbidden by the interface.
- `SHOULD`: expected unless a documented local reason exists.
- `MAY`: allowed extension behavior.

## Boundary Summary

```text
platform event
  -> adapter platform parser
  -> adapter-owned EnvelopeNormalizer
  -> /chat request with MessageEnvelope
  -> brain service validation
  -> graph state derived from MessageEnvelope.body_text and typed fields
  -> conversation row with typed body_text, raw_wire_text, mentions, reply,
     attachments, addressed_to_global_user_ids, and broadcast
```

The adapter is the only party that understands the platform wire grammar. The
brain service is allowed to assume the `/chat` payload already satisfies this
ICD.

## `/chat` Request Contract

Every `/chat` request MUST provide the following top-level fields accepted by
`ChatRequest`:

| Field | Type | Owner | Meaning |
| --- | --- | --- | --- |
| `platform` | `str` | adapter | Stable platform key for the runtime adapter. |
| `platform_channel_id` | `str` | adapter | Platform channel id. Empty string for direct/private contexts when the platform has no channel id. |
| `channel_type` | `str` | adapter | Channel class. Current supported values are operationally `group` and `private`. |
| `platform_message_id` | `str` | adapter | Platform message id, if available. |
| `platform_user_id` | `str` | adapter | Platform id of the inbound message author. |
| `platform_bot_id` | `str` | adapter | Platform id of the active bot account, if available. |
| `display_name` | `str` | adapter | Display name of the inbound message author. |
| `channel_name` | `str` | adapter | Human-readable channel label, if available. |
| `content_type` | `str` | adapter | High-level input type such as `text`, `image`, or `mixed`. |
| `message_envelope` | `MessageEnvelopeIn` | adapter | The required typed envelope defined below. |
| `timestamp` | `str` | adapter | Event timestamp. Empty string means the service will use receive time. |
| `debug_modes` | `DebugModesIn` | caller | Optional debug controls. |

The brain service validates this model with extra fields forbidden. A request
that does not match this model is not a compatible `/chat` request.

## `MessageEnvelope` Contract

`MessageEnvelope` is dict-shaped for LangGraph state compatibility. It is
defined in `types.py` and mirrored by the service-side Pydantic input model.

| Field | Required | Type | Meaning |
| --- | --- | --- | --- |
| `body_text` | yes | `str` | User-authored text after removing platform transport syntax. |
| `raw_wire_text` | yes | `str` | Original platform text or closest textual replay form. Audit only. |
| `mentions` | yes | `list[Mention]` | Typed mentions found in the inbound message. |
| `reply` | no | `ReplyTarget` | Typed reply target, when the platform event or adapter derivation identifies one. |
| `attachments` | yes | `list[AttachmentRef]` | Normalized attachment references. |
| `addressed_to_global_user_ids` | yes | `list[str]` | Internal global ids addressed by this inbound row. |
| `broadcast` | yes | `bool` | True when the message is intended for the channel rather than specific addressees. |

### `body_text`

`body_text` MUST contain only authored semantic content. It MUST NOT contain:

- Platform mention tags.
- CQ codes.
- Native reply markers.
- Adapter reply boilerplate.
- Channel tags.
- Role tags.
- Custom emoji tags.
- Any synthetic marker inserted only to help platform transport.

`body_text` MAY be an empty string when the message is attachment-only or when
the authored content is only transport syntax.

The brain service derives `user_input`, RAG queries, cognition input, dialog
context, conversation progress input, and search text from `body_text` plus
attachment `description` text. The brain service MUST NOT derive semantic input
from `raw_wire_text`.

### `raw_wire_text`

`raw_wire_text` preserves the platform text for audit, debugging, and replay.
It MAY contain platform syntax. It MUST NOT be sent to local LLM prompts, RAG
search text, cache semantic keys, scoped interaction history, or cognition as
content.

### `mentions`

Each mention record SHOULD use this shape:

| Field | Type | Meaning |
| --- | --- | --- |
| `platform_user_id` | `str` | Platform id of the mentioned account or entity, when applicable. |
| `global_user_id` | `str` | Internal global id when resolution is known. |
| `display_name` | `str` | Display label if available. |
| `entity_kind` | `MentionEntityKind` | One of `bot`, `user`, `platform_role`, `channel`, `everyone`, or `unknown`. |
| `raw_text` | `str` | Original mention token or closest textual representation. |

Only `bot` and `user` mentions can contribute to
`addressed_to_global_user_ids`. Platform-role, channel, everyone, and unknown
mentions are metadata and MUST NOT be treated as resolved user addressees.
Consumers MUST treat `unknown` as ignore-on-input.

### `reply`

`reply` identifies the typed target of a reply. It SHOULD use this shape:

| Field | Type | Meaning |
| --- | --- | --- |
| `platform_message_id` | `str` | Platform id of the target message. |
| `platform_user_id` | `str` | Platform id of the target message author. |
| `global_user_id` | `str` | Internal global id of the target author, when known. |
| `display_name` | `str` | Display name of the target author, when available. |
| `excerpt` | `str` | Short excerpt of the target message, if available. |
| `derivation` | `ReplyDerivation` | `platform_native` or `leading_mention`. |

`platform_native` means the platform explicitly supplied a reply target.
`leading_mention` means the adapter deterministically derived a conversational
target from a leading mention according to that adapter's platform rules.

The brain service may project `reply` into a compact internal `reply_context`,
but `reply_context` is derived data. It is not the source of truth.

### `attachments`

Each attachment reference SHOULD use this shape:

| Field | Type | Meaning |
| --- | --- | --- |
| `media_type` | `str` | MIME type or platform-specific media type. |
| `url` | `str` | External retrieval URL, if retained. |
| `base64_data` | `str` | Inline binary payload, if retained. |
| `description` | `str` | Text description available to RAG and cognition. |
| `size_bytes` | `int` | Payload size when known. |
| `storage_shape` | `AttachmentStorageShape` | `inline`, `url_only`, or `drop`. |

Current RAG and cognition consumers read attachment `description` text. Binary
payloads are preserved for storage and future direct-modality consumers but are
not part of normal local LLM text prompts.

Attachment handlers MUST preserve description bytes exactly. Large binary
payloads MAY be stored as URL-only according to `attachment_policy.py`.

### `addressed_to_global_user_ids`

This field records the typed inbound addressee set.

For `channel_type="private"`:

- The adapter or envelope factory MUST address inbound user rows to the active
  character's global user id.
- `broadcast` MUST be `False`.

For group-like channels:

- Direct bot mentions SHOULD include the active character's global user id.
- Native replies to the bot SHOULD include the active character's global user id
  when the reply target is resolved.
- Direct mentions or replies to another resolved user SHOULD include that
  user's global id.
- Messages without a typed direct target SHOULD use an empty list and
  `broadcast=False` on inbound user envelopes.

The list MUST be deduplicated and deterministic.

### `broadcast`

`broadcast=True` is reserved for assistant-authored outbound rows that
intentionally address the channel rather than one participant.
`broadcast=False` is required for inbound user envelopes and normal
assistant-authored replies.

`broadcast` MUST NOT be used as a replacement for
`addressed_to_global_user_ids`. Consumers that need exact addressee identity
must read the list.

## Adapter Responsibilities

A runtime adapter MUST:

1. Parse platform wire events before calling `/chat`.
2. Build a complete `MessageEnvelope`.
3. Remove platform transport syntax from `body_text`.
4. Preserve original text in `raw_wire_text`.
5. Populate typed mentions and reply targets when available.
6. Resolve the active bot mention to the active character global user id.
7. Derive inbound `addressed_to_global_user_ids` deterministically.
8. Set inbound envelope `broadcast` to `False`.
9. Normalize attachments through registered `AttachmentHandler` implementations.
10. Send only the typed envelope and top-level `ChatRequest` fields to `/chat`.

An adapter MUST NOT:

- Send raw platform wire syntax as `body_text`.
- Depend on the brain service to strip platform syntax.
- Send legacy addressing booleans as source-of-truth fields.
- Put platform-specific normalizer code under `message_envelope`.
- Invent brain-side compatibility fields to preserve old behavior.
- Omit `message_envelope` from `/chat`.
- Send extra fields outside the `/chat` model and expect them to be ignored.

## Brain Service Responsibilities

The brain service MUST:

1. Require `message_envelope` on every `/chat` request.
2. Reject incompatible request shapes through model validation.
3. Derive graph `user_input` from `message_envelope.body_text`, plus collapsed
   body text when the queue explicitly coalesces compatible messages.
4. Derive internal reply context from `message_envelope.reply`.
5. Persist typed conversation rows with `body_text`, `raw_wire_text`,
   `mentions`, `reply`, `attachments`, `addressed_to_global_user_ids`, and
   `broadcast`.
6. Use `body_text` and attachment descriptions for embeddings and retrieval.
7. Use typed addressee fields when filtering recent interaction history.
8. Persist assistant rows with unrendered assistant `body_text` and typed
   outbound addressee fields.

The brain service MUST NOT:

- Parse CQ codes, native mention tags, role tags, channel tags, emoji tags, or
  reply boilerplate.
- Treat `raw_wire_text` as semantic user input.
- Reconstruct addressees from interleaving heuristics when typed fields are
  present.
- Provide a legacy fallback path for non-envelope `/chat` requests.
- Silently synthesize missing required envelope fields.
- Import concrete platform adapter normalizers.

The brain service MAY crash or return a validation error when an adapter sends a
non-compatible request. That failure is preferred over silently contaminating
memory, RAG, or cognition with transport syntax.

## Message Envelope Package Responsibilities

`kazusa_ai_chatbot.message_envelope` MUST provide:

- Public TypedDict contracts in `types.py`.
- Public Protocol extension points in `protocols.py`.
- Public registries in `registry.py`.
- Platform-neutral mention resolver slots in `resolvers.py`.
- Shared attachment policy and handlers for modality-level behavior.

`kazusa_ai_chatbot.message_envelope` MUST NOT provide:

- Concrete platform normalizer modules.
- Platform regexes for platform event parsing.
- Brain-side sanitizer helpers.
- Compatibility shims for legacy `/chat` requests.

Concrete normalizers live in adapter modules. Shared adapter-only construction
helpers may live under `src/adapters`, not under the brain package.

## Extension Points

### `EnvelopeNormalizer`

Implemented by each platform adapter. Converts one adapter request object into
a complete `MessageEnvelope`.

Normalizers accept:

- A platform request object.
- A `MentionResolver`.
- An attachment handler registry-like object.

Normalizers return:

- A complete `MessageEnvelope` with clean `body_text`.

### `MentionResolver`

Resolves raw mention fragments into internal identity fields. The default
resolver can resolve the active bot mention. Future profile-backed resolvers
may resolve user mentions without changing adapter parser logic.

### `AttachmentHandler`

Converts raw adapter attachment payloads into `AttachmentRef` records and
chooses storage shape. Handlers are modality-level extensions, not platform
normalizers.

### `NormalizerRegistry`

Maps platform keys to `EnvelopeNormalizer` implementations. Adapters register
their own normalizer with the service/runtime layer. Brain modules do not
retrieve concrete normalizers for content repair.

### `AttachmentHandlerRegistry`

Maps media-type prefixes to `AttachmentHandler` implementations. Shared
modality handlers can be reused by multiple adapters.

## Inbound Addressing Rules

Inbound addressing is computed before `/chat`:

```text
private message
  -> addressed_to_global_user_ids = [active character global id]
  -> broadcast = false

group message with typed bot mention or bot reply target
  -> addressed_to_global_user_ids includes active character global id
  -> broadcast = false

group message addressed to another resolved user
  -> addressed_to_global_user_ids includes that user's global id
  -> broadcast = false

group message with no typed direct target
  -> addressed_to_global_user_ids = []
  -> broadcast = true
```

Adapters SHOULD keep address derivation explainable and deterministic. If a
platform event cannot identify a target, the adapter should mark the message as
broadcast rather than ask the brain service to infer the addressee from prose.

## Outbound Addressing Rules

The brain pipeline produces assistant addressee fields separately from the
inbound envelope:

- `target_addressed_user_ids`: internal users the assistant response addresses.
- `target_broadcast`: whether the assistant response is channel-broadcast.

For normal non-broadcast replies, the dialog path defaults to the current
in-turn user when dialog is produced. Empty dialog produces an empty target
list. Explicit broadcast output must set `target_broadcast=True`.

Assistant rows persisted to conversation history MUST store:

- `body_text`: assistant-authored text before outbound platform rendering.
- `raw_wire_text`: rendered outbound wire text when available, otherwise the
  same text as `body_text`.
- `addressed_to_global_user_ids`: `target_addressed_user_ids`.
- `broadcast`: `target_broadcast`.

This symmetry is required so recent-history retrieval can distinguish which
assistant turns were addressed to which participant in group channels.

## Storage and Retrieval Invariants

New conversation rows MUST store typed envelope fields. Retrieval consumers MUST
read typed fields:

- Scoped interaction history filters assistant rows by
  `addressed_to_global_user_ids` and `broadcast`.
- RAG keyword and semantic search use `body_text`.
- Embedding source is `body_text` plus attachment `description` text.
- `raw_wire_text` is for audit and replay, not semantic search.

The current interface is strict. There is no alternate `/chat` path for legacy
content-only requests. If historical data exists, it must be migrated or handled
by explicit database migration tools before it enters strict consumers.

## Failure Policy

Invalid adapter payloads fail at the service boundary. Examples:

- Missing `message_envelope`.
- Missing required envelope fields.
- Extra fields in the `/chat` request.
- Non-list `mentions`, `attachments`, or `addressed_to_global_user_ids`.
- Non-boolean `broadcast`.

The brain service SHOULD log enough request scope to identify the adapter and
message id. It SHOULD NOT repair the envelope by parsing wire syntax.

Adapter-side parsing failures SHOULD be handled by the adapter. If the adapter
cannot safely normalize a platform event, it should drop, log, or dead-letter
that event according to adapter policy rather than send a partially compatible
request.

## Security and Privacy Notes

- `raw_wire_text` may contain platform identifiers or syntax. Treat it as audit
  data.
- `base64_data` may contain binary user content. Keep it out of normal prompt
  payloads unless a future direct-modality contract explicitly allows it.
- `display_name` is informational. Do not use display names as durable identity.
- Durable identity uses `global_user_id` when known.

## Versioning and Change Control

Changes to this ICD require corresponding updates to:

- `types.py`
- service input models
- adapter normalizers
- storage schema/tests
- boundary tests
- any consuming graph state TypedDicts

Adding a required envelope field is a breaking interface change and must update
all adapters in the same change set. The brain service does not maintain a
secondary compatibility path for missing required fields.

Adding an optional typed field is allowed when:

- The field has a clear owner.
- The field is not parsed from `body_text` by brain modules.
- Tests prove adapters can omit it safely or populate it consistently.

## Verification Checklist

Before merging changes that touch this interface, verify:

- `/chat` rejects extra top-level request fields.
- `/chat` requires `message_envelope`.
- Adapter tests prove platform wire markers are removed from `body_text`.
- Adapter tests prove typed mentions and replies are populated.
- Adapter tests prove inbound addressees and `broadcast` are deterministic.
- Brain static checks show no platform marker parsing under
  `src/kazusa_ai_chatbot`.
- Brain static checks show no brain-side sanitizer helpers.
- Conversation storage tests prove typed fields are persisted.
- Retrieval tests prove `body_text` is used for search and embeddings.
- Group-history tests prove assistant rows are filtered by typed addressee.
- Attachment tests prove descriptions are preserved and binary storage policy is
  respected.

## Import Rules

Allowed for brain modules:

- `from kazusa_ai_chatbot.message_envelope import MessageEnvelope`
- Other public TypedDicts or Protocols from `kazusa_ai_chatbot.message_envelope`
  when needed for type annotations.

Allowed for adapters:

- Public TypedDicts, Protocols, registries, and resolver slots from
  `kazusa_ai_chatbot.message_envelope`.
- Adapter-owned helper modules under `src/adapters`.

Forbidden:

- Brain modules importing concrete adapter normalizers.
- `message_envelope` importing concrete platform adapters.
- Platform normalizer modules under `kazusa_ai_chatbot.message_envelope`.
- Prompt or RAG modules parsing platform wire syntax.
