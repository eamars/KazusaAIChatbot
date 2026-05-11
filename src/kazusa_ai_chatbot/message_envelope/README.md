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

The top-level brain service HTTP request and response models are owned by the
[Brain Service ICD](../brain_service/README.md). This ICD owns the
`message_envelope` payload embedded in `/chat` and the typed-message invariants
that follow from it.

## Purpose

The message envelope prevents platform transport syntax from becoming chatbot
content. Platform-specific syntax includes mention tags, reply markers, CQ
codes, channel tags, role tags, emoji tags, adapter-only attachment payloads,
and any other wire representation that belongs to a chat platform rather than
to the user's authored message.

The brain service must receive semantic content and typed metadata:

- `body_text`: user-authored content plus readable visible mention tokens.
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
enters the brain service. Other role-like concepts use their own field names so
prompts and typed dicts keep the namespace explicit.

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

- The `message_envelope` payload embedded in every brain service `/chat`
  request.
- Responsibilities of runtime adapters before calling `/chat`.
- Responsibilities of the brain service after receiving `/chat`.
- Extension points exported by `kazusa_ai_chatbot.message_envelope`.
- Storage and retrieval invariants for typed conversation rows.
- Failure behavior when the contract is violated.

Related contracts are owned by their subsystem docs: the brain service ICD owns
top-level HTTP models and runtime adapter registration, adapters own platform
rendering and SDK lifecycles, the database ICD owns durable storage mechanics,
and dialog code owns character voice.

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

The brain service consumes the typed envelope.

### Message Envelope Package

`kazusa_ai_chatbot.message_envelope` owns shared TypedDicts, Protocols,
registries, attachment policy, and platform-neutral resolver slots.

Platform normalizers belong in adapter modules.

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

## `/chat` Envelope Boundary

The top-level `/chat` request model is `ChatRequest` and is defined by the
[Brain Service ICD](../brain_service/README.md).

For this ICD, the relevant service-level rule is narrower: every compatible
`/chat` request includes a `message_envelope` value that satisfies the contract
below. The brain service validates top-level request fields and extra field
rejection according to the brain service ICD.

## `MessageEnvelope` Contract

`MessageEnvelope` is dict-shaped for LangGraph state compatibility. It is
defined in `types.py` and mirrored by the service-side Pydantic input model.

| Field | Required | Type | Meaning |
| --- | --- | --- | --- |
| `body_text` | yes | `str` | User-authored text after replacing visible platform mentions with readable tokens and removing other platform transport syntax. |
| `raw_wire_text` | yes | `str` | Original platform text or closest textual replay form. Audit only. |
| `mentions` | yes | `list[Mention]` | Typed mentions found in the inbound message. |
| `reply` | no | `ReplyTarget` | Typed reply target, when the platform event or adapter derivation identifies one. |
| `attachments` | yes | `list[AttachmentRef]` | Normalized attachment references. |
| `addressed_to_global_user_ids` | yes | `list[str]` | Internal global ids addressed by this inbound row. |
| `broadcast` | yes | `bool` | True when the message is intended for the channel rather than specific addressees. |

### `body_text`

`body_text` contains authored semantic content. Visible platform mentions are
authored addressing content, so adapter normalization preserves them as
readable platform-neutral tokens:

| Mention kind | Readable token when label is known | Fallback token when label is unknown |
| --- | --- | --- |
| `bot` or `user` | `@display name` | `@mentioned-user-N` |
| `platform_role` | `@role name` | `@mentioned-role-N` |
| `channel` | `#channel-name` | `#mentioned-channel-N` |
| `everyone` | `@everyone`, `@here`, or `@all` | same raw broadcast word without platform wrapper |
| `unknown` | `@display name` | `@mentioned-entity-N` |

Adapter normalization removes or replaces platform transport syntax,
including:

- Platform mention tags, which must become readable tokens rather than being
  stripped.
- CQ codes.
- Native reply markers.
- Adapter reply boilerplate.
- Channel tags.
- Role tags.
- Custom emoji tags.
- Platform-only synthetic markers are represented through typed metadata.

`body_text` must not contain CQ codes, Discord mention tags, raw platform ids,
or platform names as lookup-failure stand-ins. Raw ids and raw tokens belong in
typed metadata and `raw_wire_text`. `body_text` can be an empty string when the
message is attachment-only or when the authored content is only non-mention
transport syntax.

The brain service derives `user_input`, RAG queries, cognition input, dialog
context, conversation progress input, and search text from `body_text` plus
attachment `description` text.

### `raw_wire_text`

`raw_wire_text` preserves the platform text for audit, debugging, and replay.
It can contain platform syntax and remains audit/replay data. Platform-native
mention syntax that is unsafe for the brain to interpret belongs here, not in
`body_text`.

### `mentions`

Each mention record uses this shape:

| Field | Type | Meaning |
| --- | --- | --- |
| `platform_user_id` | `str` | Platform id of the mentioned account or entity, when applicable. |
| `global_user_id` | `str` | Internal global id when resolution is known. |
| `display_name` | `str` | Display label if available. |
| `entity_kind` | `MentionEntityKind` | One of `bot`, `user`, `platform_role`, `channel`, `everyone`, or `unknown`. |
| `raw_text` | `str` | Original mention token or closest textual representation. |

Only `bot` and `user` mentions can contribute to
`addressed_to_global_user_ids`. Platform-role, channel, everyone, and unknown
mentions stay as metadata. Consumers ignore `unknown` mentions for addressed
user resolution.

Mention `display_name` values are labels, not semantic aliases. Adapters may
trim and collapse whitespace in labels, but they must not translate,
summarize, infer aliases, or rewrite ordinary authored words.

### `reply`

`reply` identifies the typed target of a reply. It uses this shape:

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

Each attachment reference uses this shape:

| Field | Type | Meaning |
| --- | --- | --- |
| `media_type` | `str` | MIME type or platform-specific media type. |
| `url` | `str` | External retrieval URL, if retained. |
| `base64_data` | `str` | Inline binary payload, if retained. |
| `description` | `str` | Text description available to RAG and cognition. |
| `size_bytes` | `int` | Payload size when known. |
| `storage_shape` | `AttachmentStorageShape` | `inline`, `url_only`, or `drop`. |

Current RAG and cognition consumers read attachment `description` text. Binary
payloads may be present in current-turn envelopes for direct modality work, but
durable storage may omit `base64_data` according to the deployment storage
policy.

Attachment handlers preserve description bytes exactly. Large binary
payloads can be stored as URL-only according to `attachment_policy.py`; inline
binary persistence is config-gated.

### `addressed_to_global_user_ids`

This field records the typed inbound addressee set.

For `channel_type="private"`:

- The adapter or envelope factory addresses inbound user rows to the active
  character's global user id.
- `broadcast` is `False`.

For group-like channels:

- Direct bot mentions include the active character's global user id.
- Native replies to the bot include the active character's global user id
  when the reply target is resolved.
- Direct mentions or replies to another resolved user include that
  user's global id.
- Messages without a typed direct target use an empty list and
  `broadcast=False` on inbound user envelopes.

The list is deduplicated and deterministic.

### `broadcast`

`broadcast=True` is reserved for assistant-authored outbound rows that
intentionally address the channel rather than one participant.
`broadcast=False` is required for inbound user envelopes and normal
assistant-authored replies.

Consumers that need exact addressee identity read
`addressed_to_global_user_ids`.

## Adapter Responsibilities

A runtime adapter:

1. Parses platform wire events before calling `/chat`.
2. Builds a complete `MessageEnvelope`.
3. Replaces visible platform mention syntax with readable mention tokens in
   `body_text`.
4. Preserves original text in `raw_wire_text`.
5. Populates typed mentions and reply targets when available.
6. Resolves the active bot mention to the active character global user id.
7. Derives inbound `addressed_to_global_user_ids` deterministically.
8. Sets inbound envelope `broadcast` to `False`.
9. Normalizes attachments through registered `AttachmentHandler`
   implementations.
10. Removes non-mention platform transport syntax from `body_text`.
11. Sends only the typed envelope and top-level `ChatRequest` fields to `/chat`.

## Brain Service Responsibilities

The brain service:

1. Requires `message_envelope` on every `/chat` request.
2. Rejects incompatible request shapes through model validation.
3. Derives graph `user_input` from `message_envelope.body_text`, plus collapsed
   body text when the queue explicitly coalesces compatible messages.
4. Derives internal reply context from `message_envelope.reply`.
5. Persists typed conversation rows with `body_text`, `raw_wire_text`,
   `mentions`, `reply`, `attachments`, `addressed_to_global_user_ids`, and
   `broadcast`.
6. Uses `body_text` and attachment descriptions for embeddings and retrieval.
7. Uses typed addressee fields when filtering recent interaction history.
8. Persists assistant rows with unrendered assistant `body_text` and typed
   outbound addressee fields.

The brain service returns validation errors for incompatible envelope shapes.

## Message Envelope Package Responsibilities

`kazusa_ai_chatbot.message_envelope` provides:

- Public TypedDict contracts in `types.py`.
- Public Protocol extension points in `protocols.py`.
- Public registries in `registry.py`.
- Platform-neutral mention resolver slots in `resolvers.py`.
- Shared attachment policy and handlers for modality-level behavior.

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
chooses storage shape. Handlers are modality-level extensions owned separately
from platform normalizers.

### `NormalizerRegistry`

Maps platform keys to `EnvelopeNormalizer` implementations. Adapters register
their own normalizer with the service/runtime layer.

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
  -> broadcast = false
```

Adapters keep address derivation explainable and deterministic. If a platform
event lacks a typed target, the adapter leaves the inbound addressee list empty
and keeps `broadcast=False`.

## Outbound Addressing Rules

The brain pipeline produces assistant addressee fields separately from the
inbound envelope:

- `target_addressed_user_ids`: internal users the assistant response addresses.
- `target_broadcast`: whether the assistant response is channel-broadcast.

For normal non-broadcast replies, the dialog path defaults to the current
in-turn user when dialog is produced. Empty dialog produces an empty target
list. Explicit broadcast output must set `target_broadcast=True`.

Assistant rows persisted to conversation history store:

- `body_text`: assistant-authored text before outbound platform rendering.
- `raw_wire_text`: rendered outbound wire text when available, otherwise the
  same text as `body_text`.
- `addressed_to_global_user_ids`: `target_addressed_user_ids`.
- `broadcast`: `target_broadcast`.

This symmetry is required so recent-history retrieval can distinguish which
assistant turns were addressed to which participant in group channels.

## Storage and Retrieval Invariants

New conversation rows store typed envelope fields. Retrieval consumers read
typed fields:

- Scoped interaction history filters assistant rows by
  `addressed_to_global_user_ids` and `broadcast`.
- RAG keyword and semantic search use `body_text`.
- Embedding source is `body_text` plus attachment `description` text.
- `raw_wire_text` is for audit and replay, not semantic search.

The current interface is strict and envelope-based. Historical content-only
data can be handled by explicit database migration tools before it enters
strict consumers.

## Failure Policy

Invalid adapter payloads fail at the service boundary. Examples:

- Missing `message_envelope`.
- Missing required envelope fields.
- Extra fields in the `/chat` request.
- Non-list `mentions`, `attachments`, or `addressed_to_global_user_ids`.
- Non-boolean `broadcast`.

The brain service logs enough request scope to identify the adapter and message
id.

Adapter-side parsing failures are handled by the adapter according to adapter
policy.

## Security and Privacy Notes

- `raw_wire_text` may contain platform identifiers or syntax. Treat it as audit
  data.
- `base64_data` may contain binary user content. Normal prompt payloads use
  attachment descriptions.
- `display_name` is informational.
- Durable identity uses `global_user_id` when known.

## Public Imports

Brain modules use:

- `from kazusa_ai_chatbot.message_envelope import MessageEnvelope`
- Other public TypedDicts or Protocols from `kazusa_ai_chatbot.message_envelope`
  when needed for type annotations.

Adapters use:

- Public TypedDicts, Protocols, registries, and resolver slots from
  `kazusa_ai_chatbot.message_envelope`.
- Adapter-owned helper modules under `src/adapters`.
