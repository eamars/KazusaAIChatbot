# Adapter Interface Control Document

## Adapter Responsibility Boundary

Adapters own platform transport, platform event parsing, platform-specific
wire normalization, adapter runtime registration, delivery receipts, and
platform rendering of returned text. The brain service owns character
semantics, queueing, RAG, cognition, dialog, persistence, scheduler behavior,
and service API validation.

Adapters must translate platform syntax into typed `MessageEnvelope` fields
before `/chat`. Platform-native wire syntax may remain in `raw_wire_text` for
audit and replay, but `body_text` must contain prompt-facing semantic text.

## Required Adapter Lifecycle

Each runtime adapter is responsible for a clear startup and shutdown lifecycle:

- Read adapter-specific configuration from arguments or environment variables.
- Connect to the platform transport.
- Discover the active bot account when the platform requires it.
- Register or refresh the runtime callback URL when callback delivery is
  enabled.
- Forward normalized inbound chat requests to the brain service.
- Deliver returned `ChatResponse.messages` to the platform when allowed.
  Each string is one logical outbound chat message, sent in order.
- Close platform clients, HTTP clients, runtime servers, and background tasks.

In normal local operation, `kazusa-control-console` starts and stops registered
adapter processes as child services. Direct `python -m adapters...` commands
remain development fallback paths, and the adapter-to-brain HTTP contract is
unchanged.

## Optional Runtime Send Interface

Adapters that support dispatcher or scheduled callback delivery expose a narrow
runtime send surface:

- `can_send_message(channel_id, channel_type) -> bool`
- `send_message(channel_id, text, reply_to_msg_id=None, channel_type=..., delivery_mentions=None) -> SendResult`

The runtime send interface is adapter-owned delivery validation. It must reject
unsupported or disallowed targets before native platform send.
This callback sends one requested message only; it does not expand a normal
`/chat` message sequence.
When `delivery_mentions` is present, adapters replace matching authored
`@display_name` text inline with native platform mention syntax when feasible;
invalid or incomplete candidates leave the original text unchanged.

## Normal Chat Response Rendering

For normal `/chat` responses, adapters render `ChatResponse.messages` as an
ordered sequence of platform sends:

- Send the first message immediately.
- Use native reply rendering only on the first message when
  `use_reply_feature` is true and the platform supports it.
- Send follow-up messages as normal chat messages after adapter-owned
  non-blocking delay tasks.
- Calculate follow-up delay from message text length using the adapter shared
  sequence helper, clamped to a small bounded range.
- Apply `delivery_mentions` to each logical message before platform-specific
  chunking or segment conversion.
- Post a delivery receipt for the first successfully sent platform message id
  when `delivery_tracking_id` is present.

## Message Envelope Contract

Adapters must create complete `MessageEnvelope` values for `/chat`:

- `body_text`: authored semantic text plus readable visible mention tokens.
- `raw_wire_text`: closest platform wire replay text.
- `mentions`: typed mention records.
- `reply`: typed reply target when available.
- `attachments`: normalized attachment references.
- `addressed_to_global_user_ids`: deterministic inbound addressees.
- `broadcast`: `False` for inbound user messages.

Adapters must keep raw platform ids, CQ markers, Discord mentions, and other
transport syntax out of `body_text` except when the syntax has been translated
into prompt-facing semantic text such as image descriptions.

## Runtime Registration Contract

Cross-process runtime adapters register their callback URL with the brain
service through `/runtime/adapters/register` and refresh it through
`/runtime/adapters/heartbeat`. Callback auth uses the configured shared secret
when present. Registration and heartbeat failures are logged and retried by
adapter startup or heartbeat behavior.

## Forbidden Adapter Behavior

Adapters must not own character judgment, cognition policy, prompt decisions,
RAG retrieval, persistence writes outside documented delivery receipts, or
brain-service fallbacks. Adapter code must not add response-path LLM calls,
runtime catalog downloads, database-managed platform syntax lookup, or fake
attachments for platform expressions.

## Testing Expectations

Adapter changes need deterministic tests for:

- Public module imports and documented `python -m` commands.
- Platform syntax projection into `MessageEnvelope.body_text`.
- Typed mentions, replies, attachments, addressees, and broadcast values.
- Runtime send capability and delivery behavior.
- Normal `/chat` ordered message sequence rendering, first-message reply
  behavior, follow-up delays, and per-message inline mention rendering.
- Boundary checks proving platform-specific syntax does not leak into brain
  service, cognition, RAG, dialog, persistence, or prompts.
