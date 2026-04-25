# Task Dispatcher & Executable Future Promises

**Status:** Revised — 2026-04-25 (incorporates feedback)
**Scope:** Phase 1 completion + Phase 3 foundation

---

## Feedback (preserved as source-of-truth for this revision)

* Can the action be made implicit? For example, implicitly generate the tool call based on available tools, and save parameters based on the tool description?

* This is similar to how agentic AI models perform general-purpose tasks based on all available tools.

* Take the contemporary model: **SKILL + MCP**. SKILL is simplified into the character instruction; MCP is the tooling with arguments. The task dispatcher shall have two inputs:

  * Instruction
  * Available tools

* Coming out of the dispatcher the tool will be populated with parameters (similar to a lambda function, callable as a task in the context). A tool-call evaluator ensures all parameters are populated correctly. The example task structure:

  * `tool`: the name of the tool
  * `args`: All arguments passed into the tool during execution

* Then the task is either executed immediately, or at a future timeslot.

### Feedback on key development areas

* Show the **intake** of the task dispatcher module.
* Show the **output** of the task dispatcher module.
* Show the **control loop and timing diagram** — who triggers current and future tasks.
* How to handle **failure modes**.
* The character shall reply to the current message **before** executing the task. Today the architecture waits for consolidation before returning the generated message. Fix: as soon as the dialog is finalized, return the dialog to the adapter, then run consolidate → task dispatcher.
* Similar to the cache module, **track future tasks**.

### Technical guidelines

* One or multiple files for the task dispatcher module. If any single file exceeds 1000 lines, split.
* Do not include deferred (Tier-3) tasks in the plan.

### Round-2 feedback (this revision addresses)

* Specify exactly **what** is fed into the dispatcher from the consolidator and earlier steps.
* The MVP / sign-off scenario is: **character sends a new message in a specific chat group at a specific time (e.g. 10 minutes later)**. Everything else is gravy.
  * The character may reject (no tool call emitted) — that is fine.
  * The character may emit an incorrect tool call — that is fine; evaluator rejects it.
  * Messaging is **same-platform only** for now; cross-platform is structurally allowed but not exercised. No hard guard.
  * If the tool call does not specify a platform, default to the platform of the originating user message.
* The adapter must actually deliver the message. The adapter exposes an **MCP-alike interface** (a small set of typed async methods), not a full MCP server. Reference shape: `JulesLiu390/Amadeus-QQ-MCP/src/qq_agent_mcp/tools.py` `send_message`.

### Round-3 feedback (this revision addresses)

* `delay_minutes` is a bad design. The **UTC timestamp must be pre-calculated at the time of generation**. If the timestamp is absent, run immediately.
* Tool messages and future promises share **one mechanism** — every task goes through the scheduler regardless of whether `execute_at` is now or later. No `_execute_now` vs `_defer` branching.

### Round-4 feedback (this revision addresses)

* Time handling matches the existing `future_promise` path: parse `execute_at` with `datetime.fromisoformat`; on `ValueError`, **discard the task** (log + record in `rejected`). Do not clamp, do not fall back. Past timestamps are still valid input — the scheduler already fires them immediately via its `delay = max(0.0, ...)` path.
* Drop `PendingTaskIndex.cancel` from the MVP — no caller needs it yet.
* Permission-rejected tool calls stay silent from the character. No `recall_promise` follow-up. Log at debug level only.
* `ScheduledEventDoc` schema change is **big-bang**: replace the `payload`-shaped doc with a `tool` / `args` shape. No backwards compatibility, no `schema_version`, no compat shim. Existing dev DB events are wiped or ignored.
* Database inspection (2026-04-25) confirmed `scheduled_events` is empty (0 docs) and there are zero `commitment` rows in `user_profile_memories`. The `_handle_future_promise` → `update_commitment_status` chain has never produced or consumed data, so the side-effect is **dropped, not ported**. Details in the schema-replacement section.

---

## Problem Statement

The scheduler can persist and fire events, but there is no layer that translates cognition outputs into structured, routable actions. The consolidator emits vague "future promise" text; nothing converts that into a concrete `ScheduledEventDoc` that fires a real action.

Two gaps to close:

1. **No task dispatcher** — cognition nodes call `schedule_event()` directly with ad-hoc payloads.
2. **No tool vocabulary** — different delivery shapes (same channel, different channel, DM, reaction, media) are handled identically via an untyped `payload` dict, which the LLM has to guess at.

The original draft solved (2) by introducing a closed `ActionType` enum and per-type target dataclasses. The revised design instead treats actions as **registered tools with JSON-schema args**, mirroring how an agentic model already routes work — no enum, no per-action handler hierarchy.

---

## Design Model: Instruction + Tools → Populated Tool Call

The dispatcher follows the SKILL + MCP analogy:

- **SKILL** = simplified character instruction (what Kazusa intends to do, in natural language).
- **MCP** = tool surface — each tool declares a name, description, and JSON-schema for its arguments.

The dispatcher's job is to take the instruction together with the currently available tools, produce one or more populated tool calls (closures), validate them, and either execute now or schedule for later.

A populated tool call is the `Task`:

```python
@dataclass
class Task:
    tool: str                          # name of a registered ToolSpec
    args: dict                         # fully-populated kwargs, validated against args_schema
    execute_at: datetime               # absolute UTC; never relative. ctx.now if input was missing.
    source_event_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
```

`Task` is effectively `lambda: registry[tool].handler(**args)`. Once produced, generation and execution are decoupled.

`execute_at` is always populated by the time a `Task` exists — the evaluator fills in `ctx.now` if the raw tool call omitted it. Wire format (in `RawToolCall.args` and in `ScheduledEventDoc`) is an ISO-8601 UTC string; the in-memory form is a tz-aware `datetime`.

---

## MVP Sign-off Scenario

The end-to-end target for this refactor is a single, concrete capability:

> Kazusa, in response to a user message on platform P in channel C, decides to send a new message back to channel C **10 minutes later**. The user-visible reply happens immediately; the follow-up fires at T+10.

Walkthrough:

1. **t=0** — User sends a message on Discord, channel `#general`. Adapter ingests it as a `ChatEvent(platform="discord", channel_id="general", user_id="U1", ...)`.
2. **t=1** — L1/L2/L3 cognition produces dialog text. L3's decision includes a future-promise intent (e.g. `"check on user in 10m"`).
3. **t=2** — Dialog is returned to the adapter and the user sees Kazusa's reply. **User-facing latency stops here.**
4. **t=3** — Consolidator runs in the background. It builds a tool prompt from the dialog + L3 decision and the filtered tool registry (MVP: only `send_message`). The current UTC time is included in the prompt context so the LLM can produce absolute timestamps. The LLM emits one tool call:
   ```json
   {"tool": "send_message",
    "args": {"target_channel": "same",
             "text": "Hey, did things settle down?",
             "execute_at": "2026-04-25T14:32:17Z"}}
   ```
   (The consolidator may also normalize / re-clamp `execute_at` programmatically before handing the call to the dispatcher — see "Timestamp Generation" below.)
5. **t=4** — `ToolCallEvaluator` validates against `send_message`'s `args_schema`:
   * `target_channel == "same"` resolves to `ctx.source_channel_id = "general"`.
   * `target_platform` is omitted → defaults to `ctx.source_platform = "discord"`.
   * `execute_at` parses via `datetime.fromisoformat`. Unparseable → task discarded. Past timestamps are accepted as-is (the scheduler fires them immediately).
   Result: a populated `Task("send_message", {target_platform="discord", target_channel="general", text="...", reply_to_msg_id=None}, execute_at=2026-04-25T14:32:17Z)`.
6. **t=5** — Dispatcher writes a `ScheduledEventDoc` via `scheduler.schedule_event(...)` and records the `event_id` in `PendingTaskIndex`. **There is no separate immediate path.**
7. **t = execute_at** — `EventScheduler` fires (the existing `asyncio.sleep(delay)` mechanism handles `delay==0` and `delay>0` identically). Generic `_handle_task` rehydrates the `Task` and looks up `registry["send_message"].handler`.
8. **t = execute_at + ε** — Handler resolves the platform from `args["target_platform"]` (`"discord"`), grabs the Discord adapter from `AdapterRegistry`, and calls `adapter.send_message(channel_id="general", text="Hey, did things settle down?")`. The message lands.

**Acceptance criteria for the sign-off:**

* User reply latency at t=2 is unaffected by consolidation/dispatch.
* The scheduled message arrives in the correct channel on the correct platform at T+10 (±scheduler tick).
* If the consolidator emits zero tool calls, the system silently does nothing (rejection is fine).
* If the consolidator emits a malformed tool call, the evaluator rejects it, the user already got their reply, and nothing is scheduled (incorrect-call is fine).

Everything else (DMs, reactions, media, moderation) is the same shape with a different `ToolSpec` registered — none of it gates the sign-off.

---

## Timestamp Generation

`execute_at` is always a fully-resolved UTC instant by the time the dispatcher sees it. It is **never** a relative offset on the dispatcher's input boundary. Time handling mirrors the existing `future_promise` pipeline ([`persona_supervisor2_consolidator_persistence.py:283-291`](src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py#L283-L291)).

* **Consolidator** owns `execute_at` resolution. It injects the current UTC time into the LLM tool-prompt and asks the LLM to produce an absolute ISO-8601 UTC string. The consolidator does no programmatic relative-form conversion; whatever the LLM emits goes straight onto the `RawToolCall`.
  * LLM emits a parseable ISO-8601 UTC string → carried verbatim into `RawToolCall.args["execute_at"]`.
  * LLM omits the field → `RawToolCall.args` does not include `execute_at`.
* **Evaluator** does the only parse:
  * `execute_at` absent → `Task.execute_at = ctx.now` (= run immediately).
  * `execute_at` present and `datetime.fromisoformat(s)` succeeds → `Task.execute_at` is that value. Past timestamps are **not** clamped; the scheduler's `delay = max(0.0, (scheduled_at - now).total_seconds())` already fires them on the next loop turn, identical to how `future_promise` already behaves.
  * `execute_at` present and `datetime.fromisoformat(s)` raises `ValueError` → **discard the task**. Logged as a warning, recorded in `DispatchResult.rejected` with reason `"unparseable execute_at"`. No fallback, no retry.

This keeps the dispatcher contract pure (a `Task` always carries a real UTC instant) and keeps time-parsing semantics aligned with the only other code path that already does this (`_schedule_future_promises`).

---

## Module Intake (input contract)

The dispatcher does **not** invoke an LLM itself. The consolidator is the LLM-driving step; the dispatcher consumes the consolidator's already-emitted tool calls and turns them into executable / scheduled work.

### What the consolidator builds before calling `dispatch()`

The consolidator pipeline runs **after** the dialog has been delivered to the adapter (background task). It assembles:

1. **Instruction** — a short natural-language summary of intent, derived from L3 cognition + the finalized dialog. Example: `"Check in with U1 in ten minutes about whether things settled down."` Used both as the LLM system/user prompt and recorded in `Task.tags` for tracing.
2. **Tool registry view** — `ToolRegistry.filter(ctx)` returns the `ToolSpec`s that are valid in this context (platform-supported + permission-satisfied).
3. **Raw tool calls** — the LLM output produced from `(instruction, tools)`. Zero or more `RawToolCall(tool_name, args)`. Zero is a valid outcome — a "no-op" / rejection.
4. **Dispatch context** — the source-side defaults the evaluator uses to fill in omissions:
   ```python
   @dataclass
   class DispatchContext:
       source_platform: str        # e.g. "discord", "qq"
       source_channel_id: str      # the channel the user message came from
       source_user_id: str         # global_user_id of the speaker
       source_message_id: str      # the user's message id (for reply-to defaults)
       guild_id: Optional[str]     # for permission scoping
       bot_role: str               # "user" | "moderator" | "admin"
       now: datetime               # frozen at consolidator entry; used to clamp past execute_at and as the default when execute_at is omitted
   ```

### `dispatch()` signature

```python
async def dispatch(
    self,
    raw_calls: list[RawToolCall],
    ctx: DispatchContext,
    *,
    instruction: str = "",          # carried into Task.tags for tracing only
) -> DispatchResult:
    ...
```

The dispatcher's contract is: given raw tool calls and source-side context, produce validated `Task`s, route each to immediate execution or the scheduler, and report what happened.

`ToolSpec` mirrors an MCP tool definition:

```python
@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: dict                 # JSON-schema for arguments
    handler: Callable[[dict], Awaitable[None]]
    requires_permission: Optional[str] = None    # e.g. "mod"
    platforms: Optional[set[str]] = None         # None → all
```

`RawToolCall` is the unvalidated `(tool_name, args)` pair as emitted by the LLM. The evaluator turns it into a `Task` only after passing all checks.

---

## Module Output

`dispatch()` returns a `DispatchResult`:

```python
@dataclass
class DispatchResult:
    scheduled: list[tuple[Task, str]]        # (task, scheduler_event_id) — all validated tasks land here
    rejected: list[tuple[RawToolCall, str]]  # (raw call, rejection reason)
```

Every validated `Task` lands in `scheduled`, regardless of whether `execute_at` is now or in the future — the dispatcher always goes through `scheduler.schedule_event(...)`. Consumers that need to distinguish "fired immediately" from "deferred" can compare `task.execute_at` to `ctx.now` themselves; the dispatcher does not split them.

Downstream code (logging, observability, recall flows) consumes this. The originating cognition node never sees raw scheduler IDs — it sees the structured result.

---

## Tool-Call Evaluator

A gate between LLM tool-call generation and dispatch. Pure function, no IO.

```python
class ToolCallEvaluator:
    def __init__(self, registry: ToolRegistry): ...

    def evaluate(self, raw: RawToolCall, ctx: DispatchContext) -> EvalResult:
        """Validate one raw tool call against the registry. Checks:
          - tool name exists in the registry and is exposed for this ctx
          - all required args present (per args_schema)
          - arg types match (per args_schema)
          - requires_permission satisfied for ctx.platform / ctx.guild
          - execute_at parses; past timestamps tagged for immediate routing
        Returns EvalResult(ok, task_or_none, errors)."""
```

Failed evaluations never reach the scheduler. They surface in `DispatchResult.rejected` so the consolidator can either re-prompt the LLM or fall back to a `recall_promise` apology. Per the round-2 feedback, the MVP path treats both rejection (no tool call) and incorrect tool calls as acceptable — the user already has their reply; nothing else is required.

---

## Platform Defaulting & Cross-Platform Posture

* Tool args may include an optional `target_platform: str`. If omitted, the evaluator fills it from `ctx.source_platform`. This is the path the LLM is expected to take in the MVP.
* Tool args may include `target_channel`. The literal string `"same"` resolves to `ctx.source_channel_id`. Any other value passes through verbatim.
* If the LLM emits a `target_platform` that differs from `ctx.source_platform`, the evaluator **does not block it**. It logs a `cross_platform_dispatch` warning and routes to the requested platform's adapter. If that adapter is not registered, the call is rejected with `unknown_platform`.
* This preserves the cross-platform capability for later phases (identity aliasing, multi-platform broadcast) without requiring it now. No hard guard, no enforcement code that would have to be torn out later.

---

## Adapter Interface (MCP-alike)

The dispatcher needs a uniform way to actually deliver actions, regardless of which platform an adapter speaks. Rather than depend on a full MCP server / client pair, adapters expose an **MCP-alike `Protocol`** — a small, typed surface of async methods named after the tools they support.

Reference shape (from `JulesLiu390/Amadeus-QQ-MCP/src/qq_agent_mcp/tools.py`): a flat `send_message(target, text, ...)` async function with explicit kwargs and a structured return value. We adopt the shape, not the framework.

### The Protocol

```python
from typing import Protocol, Optional

class MessagingAdapter(Protocol):
    """MCP-alike interface implemented by every platform adapter.

    Each method corresponds to a tool the dispatcher may invoke. Methods that
    a given adapter cannot support raise NotImplementedError; the dispatcher
    surfaces this as a rejected tool call rather than crashing the scheduler."""

    platform: str        # e.g. "discord", "qq"

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to_msg_id: Optional[str] = None,
    ) -> "SendResult": ...

    # Future methods registered alongside the matching ToolSpec — none required for MVP:
    # async def send_dm(self, global_user_id: str, text: str) -> SendResult: ...
    # async def react_message(self, channel_id: str, message_id: str, emoji: str) -> None: ...
    # async def send_media(self, channel_id: str, media_url: str, text: str = "") -> SendResult: ...

@dataclass
class SendResult:
    platform: str
    channel_id: str
    message_id: str          # platform-assigned id of the message we just sent
    sent_at: datetime
```

### Why a `Protocol`, not a base class

* Adapters already exist (`DiscordAdapter`, `NapCatAdapter`); a `Protocol` lets them satisfy the interface structurally without inheritance churn.
* Static checkers / `isinstance` checks (when needed) work via `runtime_checkable`.
* Adding a new tool to the system is: register a `ToolSpec` + extend the `Protocol` + implement the method on each adapter that supports it. No central if/elif platform switch.

### Adapter registry & lookup

```python
class AdapterRegistry:
    def __init__(self) -> None:
        self._by_platform: dict[str, MessagingAdapter] = {}

    def register(self, adapter: MessagingAdapter) -> None:
        self._by_platform[adapter.platform] = adapter

    def get(self, platform: str) -> MessagingAdapter:
        if platform not in self._by_platform:
            raise UnknownPlatformError(platform)
        return self._by_platform[platform]
```

Wired in `service.py` lifespan: instantiate the existing Discord and NapCat adapters, register them, hand the registry to the dispatcher.

### `send_message` tool handler (MVP)

The single piece of glue between a tool name and an adapter call:

```python
async def handle_send_message(args: dict, ctx: DispatchContext, adapters: AdapterRegistry) -> None:
    platform   = args.get("target_platform") or ctx.source_platform
    channel_id = args["target_channel"]
    if channel_id == "same":
        channel_id = ctx.source_channel_id
    adapter = adapters.get(platform)            # may raise UnknownPlatformError
    await adapter.send_message(
        channel_id=channel_id,
        text=args["text"],
        reply_to_msg_id=args.get("reply_to_msg_id"),
    )
```

This handler is what `ToolSpec(name="send_message").handler` points at. Future tools (`send_dm`, `react_message`, ...) follow the same one-function-per-tool shape. There is no global tool dispatcher inside the adapter — the registry of `ToolSpec`s on the dispatcher side is the single source of truth.

### What stays out of scope

* No JSON-RPC transport, no MCP server process, no MCP client SDK dependency.
* No tool-discovery handshake — adapters statically declare which methods they implement; the registry is built at startup.
* If we later want to expose these tools to an external MCP-speaking agent, the `MessagingAdapter` Protocol is already shaped like MCP `tools/call`, so a thin adapter-server can be added without reworking handlers.

---

## Control Loop & Timing Diagram

Two invariants from feedback drive this diagram:

1. **Dialog returns to the adapter before consolidation/dispatch runs.** User-facing latency must not include consolidation or dispatch time.
2. **One mechanism for tool messages and future promises.** Every dispatched task is written to the scheduler with an `execute_at` and fired by the existing scheduler runtime — including "immediate" tasks (`execute_at = ctx.now`), which fire on the next event-loop turn because the scheduler's `asyncio.sleep(0)` returns immediately.

```
t=0  user message arrives at adapter
       │
t=1  L1 / L2 / L3 cognition produces dialog
       │
t=2  ┌──────────────────────────────────────────────┐
     │  Dialog → adapter → user sees reply          │  ← user-visible latency ends here
     └──────────────────────────────────────────────┘
       │
       │ (background, non-blocking; asyncio.create_task)
       ▼
t=3  Consolidator runs on the finalized dialog.
       │   - injects current UTC into the LLM prompt
       │   - LLM emits raw tool calls with absolute execute_at (or omitted)
       ▼
t=4  ToolCallEvaluator validates each raw tool call → Task
       │   - parses execute_at; missing → ctx.now; past → ctx.now (clamp)
       │   - resolves target_platform / target_channel defaults
       ▼
t=5  TaskDispatcher.dispatch(tasks)
       │   - dedupe vs PendingTaskIndex
       │   - for each Task: scheduler.schedule_event(ScheduledEventDoc(execute_at=...))
       │                    PendingTaskIndex.add(task, event_id)
       ▼
       (single path — no immediate vs deferred branch)
       │
       ▼
t = Task.execute_at
       │   EventScheduler's asyncio.sleep(delay) returns
       │   (delay==0 for "immediate" tasks; fires next event-loop turn)
       ▼
   generic _handle_task
       │   - rehydrates Task from ScheduledEventDoc
       │   - rebuilds DispatchContext from doc fields
       ▼
   registry[task.tool].handler(args=task.args, ctx=ctx, adapters=adapters)
       │
       ▼
   adapter.send_message(...) → message delivered
```

**Triggers:**

- **All tasks** — both "immediate" and "future" — are triggered by `EventScheduler` via the same `_handle_task`. The dispatcher only schedules; it never invokes handlers directly. This means crash resilience is identical for both kinds: a tool message generated just before a process crash is replayed on restart from MongoDB just like a future promise.
- The user reply (t=2) is unaffected — it has already been delivered before any task is scheduled.

---

## Pending Task Tracking

Because tool messages and future promises share one mechanism, the in-memory index covers **all unfired tasks**, not just future-dated ones. Mirrors the RAG cache pattern: in-memory index is the fast path, MongoDB (`ScheduledEventDoc`) is the source of truth.

```python
class PendingTaskIndex:
    """In-memory mirror of pending (unfired) scheduled tasks, immediate and future alike.

    Use cases:
      - Deduplication: drop a second dispatch with identical (tool, args, ~execute_at).
      - Observability: enumerate active pending tasks for status/diagnostics.

    Crash resilience: on startup, rebuild_from_db() repopulates the index from MongoDB.
    Lifecycle: entry added on schedule_event success; removed when _handle_task completes."""

    def add(self, task: Task, event_id: str) -> None: ...
    def remove(self, event_id: str) -> None: ...        # called on fire/complete/fail
    def find_by_target(self, key: str) -> list[tuple[Task, str]]: ...
    def rebuild_from_db(self, scheduler: EventScheduler) -> None: ...
```

Cancellation is intentionally not in the MVP — no caller needs it yet. When the first canceller appears (e.g. user-initiated retract), add `cancel(event_id)` then.

Index keys are derived from `(task.tool, hash(task.args))` so dedup works without parsing platform-specific args. MongoDB writes are async write-through; an in-memory entry is only added after the scheduler write succeeds, matching the cache module's invariant. Immediate tasks (`execute_at = ctx.now`) appear in the index for the brief window between scheduling and firing — useful for observability, cheap to maintain.

---

## ScheduledEventDoc Schema (big-bang replacement)

The existing `ScheduledEventDoc` carries an `event_type` discriminator and an untyped `payload` dict ([`db/schemas.py:289-301`](src/kazusa_ai_chatbot/db/schemas.py#L289-L301)). It is replaced with a `tool` / `args` shape that matches the dispatcher's `Task` directly.

**No backwards compatibility, no `schema_version`, no compat shim.** Per the round-4 feedback, we wipe / ignore existing dev-DB events and migrate every code path that writes scheduled events to the new shape in one PR.

### Database inspection — confirms big-bang is lossless

Inspected `kazusa_bot_core` on 2026-04-25 before drafting this section. The current state of the production-shaped collections:

| Collection                       | Count | Relevant content                                          |
| -------------------------------- | ----- | --------------------------------------------------------- |
| `scheduled_events`               | 0     | No `future_promise` (or any other event) has ever been persisted. |
| `user_profile_memories`          | 9     | All `diary_entry` (8) or `objective_fact` (1). Zero `commitment` rows; zero `status: "fulfilled"` rows. |
| `user_profiles.active_commitments` | 37 docs, **0** with non-empty array | The cognition-side filter at [`users.py:397`](src/kazusa_ai_chatbot/db/users.py#L397) (`status: "active"`) has nothing to filter today. |

Implication: the `_handle_future_promise` → `update_commitment_status(..., "fulfilled")` chain has produced and consumed **zero data** in this database. The "live behavior" of the existing future-promise pipeline is end-to-end empty. The big-bang migration therefore loses no data and no observable behavior.

### What is *not* ported

Per the inspection above, the commitment-fulfillment side-effect is **dropped, not ported**:

* No `commitment_id` field on the new `ScheduledEventDoc`.
* No call to `update_commitment_status` from `_handle_task`.
* `_handle_future_promise` is deleted outright with no replacement hook.

If commitment tracking later becomes a real feature (i.e. `user_profile_memories` starts accumulating commitment rows that L2/L3 actually reads), the cleanest re-introduction is either a separate `mark_commitment_fulfilled` tool the consolidator can emit alongside `send_message`, or a generic `post_success_hooks` field on `ScheduledEventDoc`. That decision belongs to the plan that introduces those rows — not this one.

### New shape

```python
class ScheduledEventDoc(TypedDict):
    event_id: str
    tool: str                       # ToolSpec name; replaces event_type
    args: dict                      # Task.args; replaces payload
    execute_at: str                 # ISO-8601 UTC; renamed from scheduled_at for consistency with Task
    status: str                     # "pending" | "completed" | "failed"
    created_at: str
    # Source-side context, needed to rebuild DispatchContext at fire time:
    source_platform: str
    source_channel_id: str
    source_user_id: str
    source_message_id: str
    guild_id: Optional[str]
    bot_role: str
```

### Changes from the old shape

| Old field           | New field          | Note                                                              |
| ------------------- | ------------------ | ----------------------------------------------------------------- |
| `event_type`        | `tool`             | Discriminator now matches a registered `ToolSpec` by name.        |
| `payload`           | `args`             | Validated against `ToolSpec.args_schema`.                         |
| `scheduled_at`      | `execute_at`       | Same ISO-8601 UTC semantics; renamed for consistency.             |
| `target_platform`   | `source_platform`  | Now describes the originating event, not the delivery target. Delivery target lives in `args`. |
| `target_channel_id` | `source_channel_id`| Same rename rationale.                                            |
| `target_global_user_id` | `source_user_id` | Same rename rationale.                                          |

### Code-path migration

* `_schedule_future_promises` ([`persona_supervisor2_consolidator_persistence.py:263`](src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py#L263)) is removed — its work is subsumed by the dispatcher writing `send_message` (or future `recall_promise`) tool calls.
* `_handle_future_promise` and other per-event_type handlers in `scheduler.py` are removed; replaced by the single generic `_handle_task` that looks up `registry[doc["tool"]].handler`.
* `db/bootstrap.py` index on `("status", 1), ("scheduled_at", 1)` is renamed to use `execute_at`.

---

## Failure Modes

| Failure                                       | Handling                                                                                  |
| --------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Unknown tool name                             | Evaluator rejects → `DispatchResult.rejected`; logged; consolidator may re-prompt the LLM. |
| Missing required arg                          | Evaluator rejects with the missing field name; never reaches dispatch.                    |
| Arg type mismatch                             | Evaluator rejects; logged with offending field and expected type.                         |
| Permission insufficient                       | Evaluator rejects → `DispatchResult.rejected`. **Silent from the character — no `recall_promise` follow-up. Debug-level log only.** |
| `execute_at` is in the past                   | Accepted as-is. Scheduler's `delay = max(0.0, ...)` fires it immediately. Same behavior as today's `future_promise`. |
| `execute_at` is unparseable (`ValueError`)    | Task discarded; reason `"unparseable execute_at"` recorded in `rejected`; warning log.    |
| `execute_at` is absent on input               | `ctx.now` substituted by the evaluator → fires on next event-loop turn.                   |
| Tool handler raises (immediate or future)     | Caught inside generic `_handle_task`; `ScheduledEventDoc` marked `failed`; logged.        |
| Tool handler raises at fire time              | Caught; `ScheduledEventDoc` marked `failed`; `PendingTaskIndex` entry removed.           |
| Scheduler write fails (Mongo unavailable)     | Returned in `rejected`; in-memory index NOT updated; cognition node sees the failure.    |
| Adapter delivery failure                      | Existing scheduler retry/backoff; after N retries, dead-letter to a `failed_actions` log. |
| Duplicate task in same dispatch batch         | Deduplicated pre-evaluation by `(tool, args_hash, ~execute_at)`.                          |
| Duplicate task across batches                 | Deduplicated against `PendingTaskIndex` before scheduling.                                |
| `target_platform` omitted                     | Filled from `ctx.source_platform`. No error.                                              |
| `target_platform` ≠ `ctx.source_platform`     | Allowed (no hard guard); logged as `cross_platform_dispatch`; routed to that adapter.     |
| `target_platform` not in `AdapterRegistry`    | Evaluator rejects with `UnknownPlatformError`; never dispatched.                          |
| Adapter raises `NotImplementedError` for tool | Caught at execute time; recorded in result; future tasks marked `failed` in DB.           |
| LLM emits zero tool calls (rejection)         | `dispatch()` is a no-op; returns empty `DispatchResult`. Acceptable per MVP scope.        |
| LLM emits malformed tool call                 | Evaluator rejects; user already has their reply; nothing scheduled. Acceptable per MVP.   |

---

## Tool Registry

Tools are registered once at service startup. Filtering happens at consolidator time so the LLM only sees tools that can actually run.

```python
registry = ToolRegistry()
registry.register(ToolSpec(name="send_message", ...))
registry.register(ToolSpec(name="send_dm", ...))
registry.register(ToolSpec(name="recall_promise", ...))
registry.register(ToolSpec(name="react_message", ...))
registry.register(ToolSpec(name="send_media", ...))

if bot_role >= "moderator":
    registry.register(ToolSpec(name="mute_user", requires_permission="mod"))
    registry.register(ToolSpec(name="ban_user", requires_permission="mod"))
```

The consolidator queries the registry with the current `(platform, guild, bot_role)` tuple and exposes only matching specs to the LLM. Adding a new action later = `registry.register(...)`, no dispatcher change.

---

## Tool Surface

The **MVP / sign-off** registers exactly one tool: `send_message`. All other tools listed are post-MVP and gated on adapter capability.

### MVP — required for sign-off

| Tool           | Required args                               | Optional args                                                                  | Default behavior                                                                                                                                                |
| -------------- | ------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `send_message` | `target_channel: "same"\|"<id>", text: str` | `target_platform: str`, `execute_at: ISO-8601 UTC string`, `reply_to_msg_id: str` | `target_platform` defaults to `ctx.source_platform`; `target_channel="same"` resolves to `ctx.source_channel_id`; missing `execute_at` means "run immediately". |

### Post-MVP (same shape, registered later)

| Tool             | Required args                                              | Description                           |
| ---------------- | ---------------------------------------------------------- | ------------------------------------- |
| `send_dm`        | `global_user_id: str, text: str`                           | Direct message a user.                |
| `recall_promise` | `promise_text: str`                                        | Apologize / withdraw a prior promise. |
| `react_message`  | `channel_id: str, message_id: str, emoji: str`             | Add a reaction to a message.          |
| `send_media`     | `target_channel: str, media_url: str, text: str = ""`      | Send an image / audio / file.         |

All post-MVP tools accept the same `target_platform` (optional, defaults to source) and `execute_at` (optional ISO-8601 UTC, missing = immediate) conventions. Tier-2 tools (`broadcast`, `edit_message`, `delete_message`, `mute_user`, `ban_user`) use the same shape; each is one `ToolSpec`, registered conditionally on adapter capability and bot permission.

---

## Module Layout

Per the technical guideline, the dispatcher is split into focused submodules from the start. No file approaches 1000 lines.

```
src/kazusa_ai_chatbot/dispatcher/
├── __init__.py
├── tool_spec.py        ToolSpec, ToolRegistry                  ~120 lines
├── task.py             Task, RawToolCall, DispatchResult,
│                       DispatchContext                          ~100 lines
├── evaluator.py        ToolCallEvaluator, EvalResult           ~150 lines
├── pending_index.py    PendingTaskIndex (in-mem + Mongo mirror) ~180 lines
├── adapter_iface.py    MessagingAdapter Protocol, SendResult,
│                       AdapterRegistry, UnknownPlatformError    ~80 lines
├── handlers.py         handle_send_message (MVP); future tool
│                       handlers added here                      ~60 lines
└── dispatcher.py       TaskDispatcher                          ~250 lines

src/kazusa_ai_chatbot/
├── scheduler.py                MOD  +30 lines  (generic _handle_task)
├── service.py                  MOD  ~20 lines  (instantiate registries + wire)
└── nodes/
    └── persona_supervisor2_consolidator_persistence.py
                                MOD  ~70 lines  (split: dialog→adapter, then dispatch)

src/adapters/
├── discord_adapter.py          MOD  ~30 lines  (implement MessagingAdapter.send_message)
└── napcat_adapter.py           MOD  ~30 lines  (implement MessagingAdapter.send_message)

tests/
├── test_tool_spec.py
├── test_evaluator.py
├── test_pending_index.py
├── test_adapter_registry.py
├── test_dispatcher.py
└── test_dispatcher_integration.py
```

Core total: ~920 lines of new code, ~180 of modifications.

---

## Implementation Steps (core only — Tier-3 deferred items excluded per guideline)

The steps below are ordered so that **the MVP sign-off scenario (delayed `send_message`) is reachable after S7**. Steps S1–S7 are the critical path; S8 is test polish.

#### S1. Dataclasses & registries (`tool_spec.py`, `task.py`, `adapter_iface.py`)

- `ToolSpec`, `ToolRegistry` with `register`, `get`, `filter(ctx)`.
- `Task`, `RawToolCall`, `DispatchResult`, `DispatchContext` with `to_scheduler_payload()` / `from_scheduler_payload()` round-trip.
- `MessagingAdapter` Protocol, `SendResult`, `AdapterRegistry`, `UnknownPlatformError`.

#### S2. ToolCallEvaluator (`evaluator.py`)

- Pure function; no IO.
- Validates tool exists, args satisfy `args_schema`, permissions satisfied.
- Parses `execute_at` with `datetime.fromisoformat`; absent → `ctx.now`; unparseable → discard task with reason `"unparseable execute_at"`.
- Applies the platform / channel defaulting rules: missing `target_platform` → `ctx.source_platform`; `target_channel == "same"` → `ctx.source_channel_id`.
- Returns `EvalResult(ok, task, errors)`.

#### S3. PendingTaskIndex (`pending_index.py`)

- In-memory dict keyed by `event_id`; secondary index by `(tool, args_hash)` for dedup.
- `add`, `remove`, `find_by_target`, `rebuild_from_db`. (No `cancel` in the MVP.)
- `remove(event_id)` is invoked from the generic `_handle_task` after a fire completes (success or failure).
- Mirrors the RAGCache pattern: in-memory primary, Mongo authoritative, async write-through.

#### S4. `send_message` handler + adapter implementations (`handlers.py`, `discord_adapter.py`, `napcat_adapter.py`)

- `handle_send_message(args, ctx, adapters)` per the spec in the Adapter Interface section.
- `DiscordAdapter.send_message(channel_id, text, reply_to_msg_id=None)` — wraps the existing `channel.send(...)` path; returns `SendResult`.
- `NapCatAdapter.send_message(channel_id, text, reply_to_msg_id=None)` — wraps `send_msg` (group_id route); returns `SendResult`.
- Both adapters expose a `platform` attribute matching their adapter id.

#### S5. TaskDispatcher (`dispatcher.py`)

- `dispatch(raw_calls, ctx, *, instruction="") → DispatchResult`.
- Pipeline: dedupe (within batch + against `PendingTaskIndex`) → evaluate → schedule.
- **Single routing path:** for every validated `Task`, the dispatcher writes a `ScheduledEventDoc` via `scheduler.schedule_event(...)` (using `Task.execute_at` if set, else `ctx.now`) and calls `PendingTaskIndex.add(...)`. There is no `_execute_now` path; the scheduler fires "immediate" tasks on the next event-loop turn because its `asyncio.sleep(0)` returns immediately.
- `DispatchResult.scheduled` collects all validated tasks; `DispatchResult.rejected` collects evaluator failures.

#### S6. ScheduledEventDoc big-bang migration

- Rewrite `ScheduledEventDoc` in `db/schemas.py` to the new `tool` / `args` / `execute_at` / `source_*` shape. Delete the old `event_type` / `payload` / `scheduled_at` / `target_*` fields outright.
- Update `db/bootstrap.py` to index `("status", 1), ("execute_at", 1)`.
- Delete `_handle_future_promise` and any other per-event_type handler from `scheduler.py`. Delete `_schedule_future_promises` from `persona_supervisor2_consolidator_persistence.py`.
- **Do not** port `update_commitment_status(..., "fulfilled")` into `_handle_task`. DB inspection (see schema-replacement section) confirmed there is no live data flowing through that side-effect; preserving it would be premature.
- Update `scheduler.schedule_event` and `_schedule_task` to read `execute_at` instead of `scheduled_at`.
- Wipe `scheduled_events` from the target DB on first run. Inspection found 0 rows already — effectively a no-op, but make it explicit so the migration is robust against any later-written stragglers.

#### S7. Scheduler integration + service / consolidator wiring

- `scheduler.py`: add one generic `_handle_task()` that re-hydrates `Task` from the doc, rebuilds `DispatchContext` from the doc's `source_*` fields, and invokes `registry[task.tool].handler(args=task.args, ctx=ctx, adapters=...)`. On completion (success or failure), calls `PendingTaskIndex.remove(event_id)`. All tools share this handler.
- `service.py` lifespan: instantiate `ToolRegistry` (register `send_message` only for MVP), `AdapterRegistry` (register Discord + NapCat), `ToolCallEvaluator`, `PendingTaskIndex` (rebuild from DB), `TaskDispatcher`. Inject all into the consolidator node and into the scheduler's handler closure.
- **Consolidator timing fix:** in the cognition graph, dialog is returned to the adapter as soon as L3 finalizes. Consolidation + dispatch runs as a background task (`asyncio.create_task`) so user-visible latency stops at the dialog. The background task swallows its own exceptions (logged) — a dispatch failure must never crash the cognition graph or surface to the user, since they already got their reply.
- The consolidator includes the current UTC time in the LLM tool-prompt so the LLM can produce absolute `execute_at` values.

**At the end of S7 the sign-off is achievable**: a user can chat with Kazusa, get an immediate reply, and receive a follow-up message in the same channel ten minutes later — both messages travel the same scheduler path.

#### S8. Tests

- `test_tool_spec.py` — registry register / filter / get.
- `test_evaluator.py` — required-field, type-mismatch, permission, past-execute_at, platform / channel defaulting.
- `test_pending_index.py` — add / remove / find / rebuild round-trip, dedup key correctness.
- `test_adapter_registry.py` — register / lookup / `UnknownPlatformError`.
- `test_dispatcher.py` — single scheduling path, dedup across batches, past / null / future `execute_at` all route through the scheduler, rejected-call accounting.
- `test_dispatcher_integration.py` — end-to-end with a stubbed `MessagingAdapter`; verifies dialog is delivered before dispatch completes, an `execute_at = now()` `send_message` fires on the next loop turn, and an `execute_at = now() + 10min` `send_message` fires at the right time (using a fake clock).

---

## Roadmap Alignment

| Existing TODO / Phase item                       | Implementing which steps                                |
| ------------------------------------------------ | ------------------------------------------------------- |
| **MVP sign-off: delayed `send_message`**         | S1–S7 (this is the gating deliverable)                  |
| Wire `future_promises` → Scheduled Events        | S1–S7                                                   |
| Adapter DM support                               | extend `MessagingAdapter` Protocol + `send_dm` ToolSpec |
| Autonomous Heartbeat (Phase 3)                   | S1–S7 + `broadcast` tool spec                           |
| Chaos & Event Engine (Phase 3)                   | S1–S7 + `send_media` tool spec                          |

---

## Out of Scope (deferred per technical guideline)

The following items previously listed in Tier-3 are **not** part of this plan. They fit the same `ToolSpec` shape and can be added later as registry entries without dispatcher refactoring:

- `mcp_tool_call` — full MCP integration with result-forwarding.
- `chain_action` with conditional follow-up — primary action whose completion schedules another with an LLM-evaluated condition.
- `register_watch` — short-lived listener for an expected user reply, with `on_timeout` action.
- Cross-platform DM via identity aliasing — resolves `global_user_id` to a specific platform adapter.

Each of these will be tracked in its own development plan once its prerequisites land.
