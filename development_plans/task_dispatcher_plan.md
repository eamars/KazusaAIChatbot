# Task Dispatcher & Executable Future Promises

**Status:** Draft — 2026-04-25
**Scope:** Phase 1 completion + Phase 3 foundation

---

## Problem Statement

The scheduler can persist and fire events, but there is no layer that translates cognition outputs into
structured, routable actions. The consolidator emits vague "future promise" text; nothing converts that
into a concrete `ScheduledEventDoc` that fires a real message to a real channel.

Two gaps to close:
1. **No task dispatcher** — cognition nodes call `schedule_event()` directly with ad-hoc payloads.
2. **No action vocabulary** — `send_message` to the same channel, a different channel, or a DM are
   all handled identically via an untyped `payload` dict. Cross-channel and cross-platform targeting
   is structurally unsupported.

---

## Architecture

```
Cognition (L2 / L3 / Consolidator)
        │
        │  List[ActionRequest]
        ▼
  ┌─────────────────────────────────────────┐
  │              TaskDispatcher             │
  │                                         │
  │  validate → deduplicate → route         │
  └──────┬──────────────────────────────────┘
         │
   execute_at?
   ┌─────┴──────┐
   │            │
  None      future time
   │            │
   ▼            ▼
ImmediateExec  Scheduler.schedule_event()
(delivery cb)  (ScheduledEventDoc → MongoDB)
                    │
               fires at T
                    │
               ActionHandler registry
                    │
               DeliveryCallback(platform, channel, text)
```

The `TaskDispatcher` is the single seam between "what should happen" and "how/when it happens."
Cognition nodes produce data; the dispatcher owns execution policy.

---

## Core Data Schema (`actions.py`)

```python
class ActionType(str, Enum):
    # ── Tier 1: MVP (implemented in core steps below) ──────────────────────
    SEND_MESSAGE    = "send_message"      # text → channel
    SEND_DM         = "send_dm"           # text → user DM
    RECALL_PROMISE  = "recall_promise"    # re-surface a commitment to the user

    # ── Tier 2: Near-term expansions (pick from menu below) ────────────────
    REACT_MESSAGE   = "react_message"     # emoji reaction on a message
    EDIT_MESSAGE    = "edit_message"      # edit a previously sent message
    SEND_MEDIA      = "send_media"        # image / audio / file attachment
    BROADCAST       = "broadcast"         # fan-out one message to N targets

    # ── Tier 3: Agentic expansions (further out) ───────────────────────────
    MCP_TOOL_CALL   = "mcp_tool_call"     # deferred MCP tool invocation
    CHAIN_ACTION    = "chain_action"      # action that schedules follow-up actions
    REGISTER_WATCH  = "register_watch"    # listen for a reply within a time window


@dataclass
class ActionTarget:
    platform: str                         # "discord", "napcat", "debug"
    channel_id: str                       # platform-native channel ID; "" for DM
    global_user_id: Optional[str] = None  # required for SEND_DM / BROADCAST


@dataclass
class ActionRequest:
    action_type: ActionType
    target: ActionTarget
    payload: dict
    execute_at: Optional[datetime] = None   # None → immediate
    source_event_id: Optional[str] = None   # originating chat event (tracing)
    follow_up: Optional["ActionRequest"] = None  # for CHAIN_ACTION
    tags: list[str] = field(default_factory=list)
```

**Payload schemas:**

| `action_type`     | Required payload keys              | Optional                        |
|-------------------|------------------------------------|---------------------------------|
| `send_message`    | `text: str`                        | `reply_to_msg_id: str`          |
| `send_dm`         | `text: str`                        | —                               |
| `recall_promise`  | `promise_text: str`                | `original_due_time: str`        |
| `react_message`   | `message_id: str`, `emoji: str`    | —                               |
| `edit_message`    | `message_id: str`, `new_text: str` | —                               |
| `send_media`      | `text: str`, `media_url: str`      | `media_type: str`               |
| `broadcast`       | `text: str`, `targets: list[dict]` | —                               |
| `mcp_tool_call`   | `tool_name: str`, `args: dict`     | `result_action: ActionRequest`  |
| `chain_action`    | (delegates to `follow_up` field)   | `condition: str`                |
| `register_watch`  | `expect_reply_within_minutes: int` | `on_timeout: ActionRequest`     |

---

## Dispatcher (`dispatcher.py`)

```python
class TaskDispatcher:
    def __init__(self, scheduler: EventScheduler, delivery_cb: DeliveryCallback): ...

    async def dispatch(self, requests: list[ActionRequest]) -> list[str]:
        """Validate, deduplicate, then route each request.
        Returns list of event_ids (deferred) or "immediate" per request."""

    async def _execute_now(self, req: ActionRequest) -> None:
        """Call delivery_cb directly for Tier-1 types; delegate to handler for others."""

    async def _defer(self, req: ActionRequest) -> str:
        """Serialize to ScheduledEventDoc, call scheduler.schedule_event()."""
```

Routing rules:
- `execute_at is None` → `_execute_now`
- `execute_at` in the past → `_execute_now` (never silently drop late promises)
- `execute_at` in the future → `_defer`

---

## LLM-Driven Action Generation

The consolidator LLM gains a `schedule_action` tool call:

```
schedule_action(
  action_type: "send_message" | "send_dm" | "recall_promise" | ...,
  target_channel: "same" | "<explicit id>",
  text: str,
  delay_minutes: int   # 0 = immediate
)
```

Consolidator maps `delay_minutes` → `execute_at = now() + timedelta(minutes=delay_minutes)`,
collects all tool calls → `list[ActionRequest]`, calls `dispatcher.dispatch(requests)`.

---

## Implementation Menu

Items are grouped by tier. The core steps must be done first; expansion items are independent
and can be picked in any order.

---

### CORE — Must implement first

#### C1. Action schema (`actions.py`) — ~80 lines
- `ActionType` enum (all tiers defined upfront as strings; handlers registered lazily).
- `ActionTarget`, `ActionRequest` dataclasses.
- `to_scheduler_payload()` / `from_scheduler_payload()` round-trip helpers.

#### C2. Task dispatcher (`dispatcher.py`) — ~120 lines
- `TaskDispatcher.__init__`, `dispatch()`, `_execute_now()`, `_defer()`.
- Platform validation before dispatch.
- Deduplication: drop duplicate `(action_type, target, payload)` within one `dispatch()` call.

#### C3. Scheduler handler (`scheduler.py` +30 lines)
- Add `_handle_action_request()`.
- Add `register_default_handlers()` helper to group all default registrations.

#### C4. Service wiring (`service.py` ~10 lines)
- Instantiate `TaskDispatcher` in lifespan; inject into consolidator node.

#### C5. Consolidator integration (~50 lines changed)
- Add `schedule_action` tool definition to consolidator prompt.
- Parse tool calls → `ActionRequest` list → `dispatcher.dispatch()`.
- Replace existing ad-hoc `schedule_event()` calls.

#### C6. Core tests (~150 lines)
- `test_actions.py`: serialization round-trip, late `execute_at` → immediate path.
- `test_dispatcher.py`: immediate fires delivery_cb; deferred calls `schedule_event`.
- `test_scheduler_action_handler.py`: handler deserializes and fires correctly.

---

### TIER 2 — Near-term expansions (pick any)

#### T2-A. `react_message` — ~60 lines
Add emoji reaction to an existing message.
- Payload: `message_id`, `emoji`.
- Discord adapter: `message.add_reaction(emoji)`.
- NapCat adapter: `set_msg_emoji_like` API.
- 1 new handler in `dispatcher.py`; 1 adapter method each.

#### T2-B. `edit_message` — ~60 lines
Edit a previously sent message (useful for correcting follow-ups or live countdowns).
- Payload: `message_id`, `new_text`.
- Discord: `message.edit(content=new_text)`.
- NapCat: not natively supported; log a warning and send a new message instead.
- Note: requires the bot to have stored the sent `message_id` from the delivery callback.

#### T2-C. `send_media` — ~80 lines
Send an image, audio clip, or file alongside optional caption text.
- Payload: `text`, `media_url`, `media_type` (`image`/`audio`/`file`).
- Discord: `channel.send(text, file=discord.File(...))`.
- NapCat: `send_msg` with CQ-code segment.
- Useful for Chaos Engine events (e.g., Kazusa shares a photo from "band practice").

#### T2-D. `broadcast` — ~70 lines
Fan-out one message to N `ActionTarget` objects in a single `ActionRequest`.
- Payload: `text`, `targets: list[dict]` (each is an `ActionTarget`-shaped dict).
- Dispatcher expands broadcast into N individual `send_message` requests and routes each.
- Use case: Autonomous Heartbeat sending the same "good morning" to multiple channels.

#### T2-E. Adapter DM support — ~40 lines per adapter
Required for `send_dm` to work end-to-end (currently unimplemented in adapters).
- Discord: fetch `User` object → `user.send(text)`.
- NapCat: `send_private_msg` API with `user_id`.
- Prerequisite for any DM-based future promise.

---

### TIER 3 — Agentic expansions (further out)

#### T3-A. `mcp_tool_call` — ~100 lines
Defer an MCP tool invocation to a future time.
- Payload: `tool_name`, `args`, optional `result_action` (what to do with the result).
- Handler calls `mcp_client.call_tool(tool_name, args)`, then optionally dispatches
  `result_action` with the tool output injected into its payload.
- Enables: "Search the web for X in 2 hours and send me what you find."

#### T3-B. `chain_action` + conditional follow-up — ~120 lines
An action that schedules another action after it completes, with an optional condition.
- `ActionRequest.follow_up` field carries the next `ActionRequest`.
- Handler fires the primary action, then calls `dispatcher.dispatch([follow_up])`.
- `condition: str` (natural language) is evaluated by a lightweight LLM call before
  the follow-up is dispatched ("only send the reminder if the user hasn't replied").
- Foundation for: nudge → firmer nudge → drop it escalation chains.

#### T3-C. `register_watch` — ~150 lines
After sending a message, register a short-lived listener for a user reply.
- Payload: `expect_reply_within_minutes`, optional `on_timeout: ActionRequest`.
- Implementation: store a `WatchDoc` in MongoDB with expiry. On each incoming message,
  `service.py` checks active watches for the channel/user; if matched, cancels the watch;
  if expired, fires `on_timeout`.
- Foundation for Phase 2 Empathic Accuracy: "I predicted you'd reply; did you?"

#### T3-D. Cross-platform DM — ~80 lines
Send a DM to a user identified by `global_user_id` on a platform different from the
originating one.
- Requires resolving `global_user_id` → `(platform, platform_user_id)` via `user_profiles`.
- Dispatcher looks up the user's platform accounts and selects the target adapter.
- Prerequisite: identity aliasing via `suspected_aliases` must be reliable.

---

## Roadmap Alignment

| Existing TODO / Phase item                  | Implementing which items |
|---------------------------------------------|--------------------------|
| Wire `future_promises` → Scheduled Events   | C1–C5                    |
| Adapter DM support                          | T2-E                     |
| Autonomous Heartbeat (Phase 3)              | C1–C5 + T2-D             |
| Chaos & Event Engine (Phase 3)              | C1–C5 + T2-C             |
| Empathic Accuracy Evaluator (Phase 2)       | T3-C                     |
| Shadow Prediction Branch (Phase 2)          | T3-A + T3-B              |
| Cross-platform identity (Phase 3)           | T3-D                     |

---

## Non-Goals (this plan)

- **Message queue (Redis/RabbitMQ):** asyncio + MongoDB sufficient until multi-instance.
- **Action retry policy:** scheduler status fields already track failures; retry logic is deferred.
- **Rate limiting / authorization:** out of scope for now.

---

## File Inventory (core only)

```
src/kazusa_ai_chatbot/
├── actions.py                  NEW  ~80 lines
├── dispatcher.py               NEW  ~120 lines
├── scheduler.py                MOD  +30 lines
├── service.py                  MOD  ~10 lines
├── nodes/
│   └── persona_supervisor2_consolidator_persistence.py
│                               MOD  ~50 lines
src/adapters/
├── discord_adapter.py          MOD  ~40 lines  (T2-E)
└── napcat_adapter.py           MOD  ~40 lines  (T2-E)
tests/
├── test_actions.py             NEW  ~60 lines
├── test_dispatcher.py          NEW  ~60 lines
└── test_scheduler_action_handler.py
                                NEW  ~30 lines
```

Core total: ~480 lines. Each Tier 2/3 item adds 60–150 lines independently.
