# Dispatcher

`kazusa_ai_chatbot.dispatcher` is the task-dispatch layer for deferred tool execution.

It turns LLM-emitted raw tool calls into validated scheduled tasks, stores those tasks through the scheduler, and later delivers them through platform adapters. In the current runtime, this is mainly used for accepted future promises such as "remind me later" or "send a message to that group in one minute."

It is not a user-input classifier, not a commitment harvester, not a dialogue planner, and not a general agent loop. The dispatcher does not decide whether Kazusa accepted a request. That decision happens upstream in the consolidator's LLM output. The dispatcher only handles tool-call generation, validation, deduplication, scheduling, and delivery.

## Public Boundary

Production callers should use the package facade:

```python
from kazusa_ai_chatbot.dispatcher import (
    AdapterRegistry,
    DispatchContext,
    PendingTaskIndex,
    RawToolCall,
    RemoteHttpAdapter,
    TaskDispatcher,
    ToolCallEvaluator,
    ToolRegistry,
    build_send_message_tool,
)
```

The main runtime object is:

```python
dispatcher = TaskDispatcher(evaluator, pending_index)
```

The dispatcher accepts raw LLM tool calls:

```python
await dispatcher.dispatch(
    raw_calls=[
        RawToolCall(
            tool="send_message",
            args={
                "target_channel": "same",
                "text": "drink water.",
                "execute_at": "2026-04-22T18:16:28+00:00",
            },
        )
    ],
    ctx=dispatch_context,
    instruction="Kazusa should follow through on 1 accepted promise.",
)
```

It returns:

```python
{
    "scheduled": [(Task, event_id), ...],
    "rejected": [(RawToolCall, reason), ...],
}
```

## Runtime Lifecycle

```text
service startup
  -> build ToolRegistry
       register dispatchable tools, currently send_message
  -> build AdapterRegistry
       adapters may be registered at runtime for each platform
  -> rebuild PendingTaskIndex from pending scheduler rows
  -> build ToolCallEvaluator
  -> build TaskDispatcher
  -> configure consolidator with dispatcher + tool registry
  -> configure scheduler with tools + adapters + pending index
  -> load pending scheduler events

chat turn completes
  -> background consolidator writes durable memory/state
  -> accepted future_promises are preserved from LLM output
  -> task-dispatch LLM sees:
       finalized dialog,
       future_promises,
       source context,
       visible tool specs
  -> task-dispatch LLM emits RawToolCall rows
  -> TaskDispatcher validates and schedules them

scheduled time arrives
  -> scheduler rehydrates Task and DispatchContext
  -> tool handler runs
  -> handler sends through the target platform adapter
  -> scheduler marks event completed or failed
```

The dispatcher runs in the background consolidation path. It should not block response generation to the user.

## Design Intention

The dispatcher exists to separate semantic acceptance from operational execution.

- The consolidator LLM decides what promises or commitments Kazusa accepted.
- The task-dispatch LLM converts accepted future promises into candidate tool calls.
- The evaluator checks tool availability, schema shape, permissions, adapters, target defaults, and absolute time parsing.
- The dispatcher deduplicates and persists valid tasks.
- The scheduler owns delayed execution and status transitions.
- Platform adapters own actual delivery.

This keeps the local LLM's semantic work explicit while keeping execution mechanics deterministic and inspectable.

## Upstream Handoff

The dispatcher consumes `future_promises` that have already been emitted by the consolidator. It does not reinterpret arbitrary user input.

The task-dispatch LLM receives a structured payload containing:

- current UTC time,
- source platform and channel,
- source channel type,
- final dialog,
- decontextualized input,
- content anchors,
- normalized future promises,
- visible tool specifications.

It returns JSON shaped like:

```python
{
    "tool_calls": [
        {
            "tool": "send_message",
            "args": {
                "target_channel": "same",
                "text": "message to send later",
                "execute_at": "2026-04-22T18:16:28+00:00",
            },
        }
    ]
}
```

If the accepted commitment is an ongoing style, address, language, or formatting rule rather than a future action at a specific time, the task-dispatch LLM should return no tool calls. That distinction belongs in the LLM prompt and promise schema, not in keyword-based Python filters.

## Dispatch Context

`DispatchContext` carries source-side defaults and validation scope:

```python
DispatchContext(
    source_platform=str,
    source_channel_id=str,
    source_user_id=str,
    source_message_id=str,
    guild_id=str | None,
    bot_role=str,
    now=datetime,
)
```

Important meanings:

- `source_platform` is the default target platform when a tool call omits `target_platform`.
- `source_channel_id` is used when a tool call says `target_channel: "same"`.
- `bot_role` filters tools by permission before the LLM sees them and before evaluation accepts them.
- `now` is the frozen dispatch time used for immediate tasks and relative-promise normalization upstream.

The persisted scheduler event carries this context so scheduled execution can be rehydrated later.

## Tool Calls And Tasks

`RawToolCall` is untrusted LLM output:

```python
RawToolCall(tool=str, args=dict)
```

`Task` is the validated scheduled invocation:

```python
Task(
    tool=str,
    args=dict,
    execute_at=datetime,
    tags=list[str],
)
```

The evaluator transforms a raw call into a task only when:

- the tool exists and is visible in the current context,
- the arguments satisfy the tool schema,
- the target platform has a registered adapter,
- `execute_at`, when supplied, parses as an absolute ISO timestamp,
- default platform/channel substitution is structurally valid.

Rejected calls are returned with human-readable reasons and are not persisted.

## Tool Registry

`ToolRegistry` is the dispatcher capability roster. A tool specification defines:

- tool name,
- prompt-facing description,
- JSON-schema-like argument contract,
- async handler,
- optional minimum permission,
- optional source-platform allowlist.

The current built-in tool is `send_message`.

`send_message` schedules a platform message with:

```python
{
    "target_channel": str,
    "text": str,
    "target_platform": str | omitted,
    "execute_at": str | omitted,
    "reply_to_msg_id": str | None,
}
```

Additional tools should be added as explicit capabilities with narrow schemas. Do not make one generic "do anything" tool; dispatch remains safer when each tool has clear ownership and validation.

## Adapter Boundary

Adapters own delivery to external platforms.

The dispatcher and scheduler speak only to the `MessagingAdapter` protocol:

```python
async def send_message(
    channel_id: str,
    text: str,
    *,
    reply_to_msg_id: str | None = None,
) -> SendResult
```

`AdapterRegistry` maps platform keys to adapters. `RemoteHttpAdapter` bridges scheduled delivery to an adapter-owned HTTP endpoint, which lets the brain service schedule work even when the live platform adapter runs in another process.

If no adapter is registered for the target platform, evaluation rejects the call before scheduling.

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

The index is rebuilt from pending scheduler rows at service startup. The dispatcher also rejects duplicates that appear within the same batch of raw tool calls.

Deduplication is structural. It does not decide whether two natural-language promises mean the same thing; that semantic decision belongs upstream.

## Scheduler Integration

The dispatcher persists valid tasks as scheduler events. The scheduler:

- inserts the pending event into MongoDB,
- creates an in-process async timer,
- rehydrates the task when the time arrives,
- invokes the registered tool handler,
- marks the event as `completed` or `failed`,
- removes completed/cancelled tasks from the pending index.

Scheduler persistence is the durable boundary. If the service restarts, pending events can be loaded again from MongoDB and scheduled in memory.

## Semantic Ownership

LLMs own semantic decisions:

- whether a user-facing request became an accepted future promise,
- whether a promise means a delayed message or a long-lived behavioral rule,
- what message text should be sent later,
- how to map a named group/channel from the promise text into a target channel when the instruction explicitly provides one.

Deterministic code owns mechanics:

- visible tool filtering,
- schema validation,
- permission checks,
- adapter availability checks,
- timestamp parsing,
- source defaults such as `target_channel: "same"`,
- duplicate detection,
- scheduler persistence,
- delivery status updates.

Do not add keyword matching over user text, final dialog, or promise text to override the LLM's chosen channel. If the model emits the wrong kind of tool call, fix the prompt, schema, or live examples.

## Failure Behavior

The dispatcher is fail-closed:

- unavailable tools are rejected,
- invalid schemas are rejected,
- missing adapters are rejected,
- unparseable timestamps are rejected,
- duplicate pending tasks are rejected,
- scheduler write failures are reported as rejections.

Rejected dispatches are telemetry for logs and consolidation metadata. They do not affect the already-generated response.

## Test Coverage

Relevant tests:

- `tests/test_dispatcher.py`
- `tests/test_scheduler_future_promise.py`
- `tests/test_runtime_adapter_registration.py`
- dispatcher-related cases in persistence and scheduler tests

The live dispatcher tests are behavior evidence. They check that real LLM output can produce a valid `send_message` call for accepted future actions, avoid tool calls for ongoing style rules, target explicit group IDs from private chats, and normalize vague near-future promises into absolute execution times.
