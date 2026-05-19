# Dispatcher Package

`kazusa_ai_chatbot.dispatcher` now contains deterministic delivery primitives
used by the scheduler and runtime adapters. It is not a cognition action
planner, not a consolidator extension, and not an LLM-facing tool-call router.

The package keeps:

- `Task`: validated scheduler invocation payload.
- `DispatchContext`: source-side delivery context persisted with scheduler
  rows.
- `ToolRegistry` and `ToolSpec`: deterministic registry for scheduler-owned
  handlers.
- `PendingTaskIndex`: in-process duplicate tracking for pending scheduler
  rows.
- `AdapterRegistry`, `MessagingAdapter`, `RemoteHttpAdapter`, and
  `SendResult`: adapter-facing delivery boundary.
- `build_send_message_tool()` and `handle_send_message(...)`: deterministic
  scheduled message delivery handler.

The removed runtime class cluster that generated, validated, and scheduled raw
LLM delivery calls is no longer part of this package. Consolidation does not
schedule user-visible text, and self-cognition does not hand prewritten text to
delivery.

## Runtime Lifecycle

```text
service startup
  -> build ToolRegistry
       register deterministic scheduler tools, including send_message
  -> build AdapterRegistry
       adapters may be registered at runtime for each platform
  -> rebuild PendingTaskIndex from pending scheduler rows
  -> configure scheduler with tools + adapters + pending index
  -> load pending scheduler events

scheduled time arrives
  -> scheduler rehydrates Task and DispatchContext
  -> tool handler runs
  -> handler records write-ahead outbound conversation row
  -> handler sends through the target platform adapter
  -> scheduler marks event completed or failed
```

The scheduler owns delayed execution. This package supplies the handler and
adapter abstractions the scheduler needs to execute rows that already exist.

## Scheduler Task Shape

`Task` is the validated invocation shape:

```python
Task(
    tool=str,
    args=dict,
    execute_at=datetime,
    tags=list[str],
)
```

`DispatchContext` carries source-side metadata needed at execution time:

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

Important meanings:

- `source_platform` identifies the platform associated with the scheduler row.
- `source_channel_id` and `source_channel_type` preserve source context.
- `bot_permission_role` is retained for scheduler row compatibility and future
  permission checks.
- `now` is the execution time used by deterministic handlers.

## `send_message`

The retained built-in scheduler tool is `send_message`.

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

`delivery_mentions` is optional adapter-owned rendering metadata. The scheduler
handler validates only that the field is structurally usable, preserves it
inside task args, and passes it to the adapter unchanged. Mention feasibility,
native syntax, and fallback behavior belong to the platform adapter.

No current cognition or consolidation path creates new `send_message` scheduler
rows. Existing pending rows can still be executed by the scheduler until a data
migration cancels them.

## Adapter Boundary

Adapters own delivery to external platforms. The scheduler handler speaks only
to the `MessagingAdapter` protocol:

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
Prefix is the only approved placement for the current contract.

`AdapterRegistry` maps platform keys to adapters. `RemoteHttpAdapter` bridges
scheduled delivery to an adapter-owned HTTP endpoint, which lets the brain
service execute scheduled work even when the live platform adapter runs in
another process.

The HTTP contract for runtime adapter registration and heartbeat endpoints is
owned by the [Brain Service ICD](../brain_service/README.md).

## Deduplication

`PendingTaskIndex` prevents duplicate pending tasks within the running process.

The deduplication signature is based on:

```python
{
    "tool": task.tool,
    "args": task.args,
    "execute_at": task.execute_at.isoformat(),
}
```

The index is rebuilt from pending scheduler rows at service startup.
Deduplication is structural. It does not decide whether two natural-language
promises mean the same thing; semantic decisions belong upstream in cognition.

## Ownership

LLMs own semantic decisions:

- whether a user-visible response should exist;
- whether a future cognition cycle should be scheduled;
- whether a memory lifecycle action should retire a commitment.

Deterministic code owns mechanics:

- scheduler row execution;
- adapter availability and channel capability checks at send time;
- write-ahead conversation persistence;
- delivery receipt updates;
- duplicate detection for pending scheduler rows;
- scheduler status transitions.

Consolidation may persist and learn from prompt-safe episode traces, but it
must not schedule, dispatch, or author user-visible output.
