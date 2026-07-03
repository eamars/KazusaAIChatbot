# Dispatcher Package

`kazusa_ai_chatbot.dispatcher` owns deterministic adapter-facing delivery
primitives. It is not the calendar scheduler, not a cognition action planner,
not a consolidator extension, and not an LLM-facing tool-call router.

The package keeps:

- `AdapterRegistry`, `MessagingAdapter`, `RemoteHttpAdapter`, and
  `SendResult`: adapter-facing delivery boundary.
- `Task` and `DispatchContext`: validated send-handler payload shapes used by
  explicit dispatcher calls and legacy migration diagnostics.
- `ToolRegistry` and `ToolSpec`: deterministic handler registry primitives for
  dispatcher-owned tools.
- `build_send_message_tool()` and `handle_send_message(...)`: deterministic
  message delivery handler.

The removed process-local scheduler runtime is no longer part of the service
control plane. `PendingTaskIndex`, pending `scheduled_events` loading, and
startup-time scheduler task reconstruction are gone. The durable timing owner
is `kazusa_ai_chatbot.calendar_scheduler`.

## Runtime Boundary

```text
normal chat
  -> brain service selects visible text
  -> adapter sends returned ChatResponse messages
  -> adapter posts delivery receipt when it has a platform message id

explicit dispatcher send
  -> trusted owner builds Task + DispatchContext
  -> dispatcher validates target adapter capability
  -> handler writes the outbound row before delivery
  -> handler sends through AdapterRegistry
```

The calendar scheduler does not call `send_message` as a trigger kind. Future
or commitment due work becomes fresh cognition first; any later visible text
must be selected by the normal cognition/dialog path and delivered through the
appropriate adapter boundary.

Callers that can produce visible output from background context must pass the
generic runtime coordination checkpoints before invoking dispatcher delivery.
The dispatcher does not own stale-context cancellation policy; it executes a
validated send after the upstream caller has decided the pipeline is still
admissible.

## Task Shape

`Task` is the validated invocation shape for dispatcher-owned tools:

```python
Task(
    tool=str,
    args=dict,
    execute_at=datetime,
    tags=list[str],
)
```

`DispatchContext` carries source-side metadata needed by deterministic
handlers:

```python
DispatchContext(
    source_platform=str,
    source_channel_id=str,
    source_user_id=str,
    source_message_id=str,
    guild_id=str | None,
    bot_permission_role=str,
    now=datetime,
    source_channel_type=str,
)
```

`source_platform`, `source_channel_id`, and `source_channel_type` preserve the
trusted source context. Semantic decisions about whether a promise should be
scheduled or whether a character should speak belong upstream in cognition and
self-cognition, not in this package.

## `send_message`

The built-in dispatcher tool remains `send_message`:

```python
{
    "target_platform": str,
    "target_channel": str,
    "target_channel_type": "group | private",
    "text": str,
    "reply_to_msg_id": str | None,
    "delivery_mentions": list[dict] | omitted,
}
```

`delivery_mentions` is optional adapter-owned rendering metadata. The handler
validates only that the field is structurally usable, preserves it inside task
args, and passes it to the adapter unchanged. Mention feasibility, native
syntax, and fallback behavior belong to the platform adapter.

No current cognition or consolidation path creates delayed `send_message`
calendar work. Legacy pending direct-send rows from the old `scheduled_events`
control plane are cancelled by the approved migration instead of being
converted into calendar runs.

## Adapter Boundary

Adapters own delivery to external platforms. The dispatcher handler speaks
only to the `MessagingAdapter` protocol:

```python
async def send_message(
    channel_id: str,
    text: str,
    *,
    channel_type: str,
    reply_to_msg_id: str | None = None,
    delivery_mentions: Sequence[dict] | None = None,
) -> SendResult
```

Adapters may also expose `can_send_message(channel_id, channel_type=...)` so
the dispatcher can reject unavailable channels before write-ahead conversation
persistence. Remote adapters report this through `/send_message/capability`;
unavailable, unsupported, or not-configured targets fail closed. Platform
adapters own configured-channel permission checks, so capability means both
transport reachability and permission to deliver to that target.

`delivery_mentions` requests are best-effort. If an adapter cannot render one
or the request lacks the needed platform identity, it sends the original text.
Adapters replace matching authored `@display_name` text inline with native
platform mention syntax; there is no separate placement field.

`AdapterRegistry` maps platform keys to adapters. `RemoteHttpAdapter` bridges
dispatcher-owned delivery to an adapter-owned HTTP endpoint when the live
platform adapter runs in another process.

The HTTP contract for runtime adapter registration and heartbeat endpoints is
owned by the [Brain Service ICD](../brain_service/README.md).

## Ownership

LLMs own semantic decisions:

- whether a user-visible response should exist;
- whether a future cognition cycle should be requested;
- whether a memory lifecycle action should retire a commitment.

Deterministic code owns mechanics:

- runtime-coordination admission and cancellation before dispatcher handoff;
- adapter availability and channel capability checks at send time;
- write-ahead conversation persistence;
- delivery receipt updates;
- dispatcher status transitions and event-log metadata;
- calendar run creation, claim, lease, retry, and migration in the
  `calendar_scheduler` package.

Consolidation may persist and learn from prompt-safe episode traces, but it
must not schedule, dispatch, or author user-visible output.
