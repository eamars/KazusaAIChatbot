# global input queue plan

## Summary

- Goal: Add a global input queue so noisy message bursts are pruned before relevance/RAG, while preserving RAG-safe one-at-a-time processing.
- Plan class: medium
- Status: approved
- Overall cutover strategy: compatible
- Highest-risk areas: request lifecycle changes, dropped-message persistence, queue pruning correctness, adapter timeouts, and preserving the consolidation/RAG integrity gate.
- Acceptance criteria: `/chat` enqueues requests, prunes queued noise by the approved policy, saves dropped user messages without character replies, and never starts the next RAG pass before the previous consumed turn's writes/consolidation finish.

## Context

Current `/chat` processes one request through the service graph and uses the chat semaphore to prevent the next RAG pass from starting before previous post-response persistence/consolidation finishes.

The new architecture must move from direct request processing to queued input processing:

1. Input messages are queued.
2. When the waiting queue length is greater than 2, clear all messages that are neither tagged nor replies to the bot.
3. If still greater than 5, clear all messages that are not replies to the bot.
4. If still greater than 5, discard all waiting messages except the latest one.
5. Consume the oldest remaining message.

"Cleared/discarded" means the user message is saved to `conversation_history`, but the character does not respond. From the adapter perspective, the message is not replied to.

## Mandatory Rules

- Preserve existing `/chat` request and response schemas.
- Use a global process-local brain queue.
- Treat `mentioned_bot=True` as tagged.
- Treat adapter-supplied `reply_context.reply_to_current_bot=True` as replied.
- Do not derive queue `is_bot_reply` by querying conversation history or platform-specific records in the brain service.
- Do not treat replies to other users as protected replies.
- Dropped messages must be saved as user conversation rows.
- `/chat` must not save user messages before enqueueing; all user-message persistence is worker-owned.
- Dropped messages must be saved before the worker starts RAG/graph processing for the next surviving consumed item.
- Dropped messages must not run relevance, RAG, cognition, dialog, bot-save, conversation progress recording, or consolidation.
- Consumed messages must preserve the existing response path and post-response persistence/consolidation behavior.
- The worker must not consume the next queued item until the previous consumed item's bot-save/progress/consolidation path is complete.
- Do not change prompts, RAG agents, consolidator internals, `db_writer`, DB schemas, adapter contracts, Cache2 policy, or scheduler behavior.

## Must Do

- Add a global in-memory input queue owned by the brain service.
- Refactor `/chat` into an enqueue-and-await endpoint.
- Add a single worker that prunes waiting items and processes one surviving item at a time.
- Persist dropped user messages and return an empty `ChatResponse`.
- Preserve the RAG-safe invariant by awaiting post-response writes/consolidation before consuming the next queued item.
- Add deterministic tests for queue pruning, dropped-message persistence, worker ordering, and RAG-safe blocking.

## Deferred

- Mongo-backed queue or distributed lease for multi-worker/multi-replica deployment.
- Adapter-side batching or fire-and-forget behavior.
- New user-visible busy/drop messages.
- Any prompt, RAG, cognition, dialog, consolidator, scheduler, or DB schema changes.

## Cutover Policy

- `/chat` request/response API: compatible.
- Adapter behavior: compatible.
- Queueing behavior: bigbang within the brain process, because there must be exactly one active `/chat` ingress path after implementation.
- Database schema: compatible, no migration.

## Agent Autonomy Boundaries

- The implementation agent may choose local private helper names and queue storage mechanics only when they preserve this plan's contracts.
- The agent must not introduce alternate pruning rules, adapter API changes, DB schemas, distributed queue infrastructure, prompt rewrites, or unrelated cleanup.
- The agent must not process dropped messages through relevance/RAG/cognition/dialog.
- If single-process queueing is incompatible with the runtime, the agent must stop and report the blocker instead of inventing a distributed substitute.

## Target State

`/chat` becomes an enqueue-and-wait endpoint:

- It receives a `ChatRequest`.
- It prepares enough deterministic metadata for queue pruning and dropped-message persistence.
- It enqueues an input item and waits on that item's future.
- It returns the future's `ChatResponse`.
- It does not call `save_conversation`.

A single global worker owns graph execution:

- It prunes the waiting queue before each consume.
- It resolves dropped items with an empty response after saving their user messages.
- It consumes the oldest surviving message.
- It saves the consumed user message immediately before graph execution.
- It runs the existing graph for consumed messages.
- It resolves the consumed item's response future as soon as final dialog is ready.
- It then awaits post-response persistence/consolidation before consuming the next item.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Queue scope | Global process-local queue | Matches the requested global brain queue and current single-worker deployment. |
| Pruning timing | Before each consume, after enqueue | Keeps pruning deterministic and prevents stale queue growth. |
| Protected tagged message | `mentioned_bot=True` | Existing platform adapters already provide this structured field. |
| Protected reply message | Adapter-supplied `reply_context.reply_to_current_bot=True` | Adapters own platform reply interpretation; the brain queue consumes normalized metadata only. |
| Dropped message persistence | Save user row, no character response | User explicitly requested DB persistence and adapter-visible no-reply behavior. |
| Persistence ownership | Worker owns all user-message saves | Prevents queued future messages from appearing in history/RAG before queue pruning and ordering are resolved. |
| Ordering | Consume oldest surviving item | Matches the requested FIFO consume rule after pruning. |
| RAG safety | Worker waits for post-response writes before next consume | Prevents the next RAG pass from reading before prior DB writes/cache invalidation complete. |
| Multi-process support | Deferred | Current deployment is single uvicorn worker; multi-replica support needs DB-backed queue/lease. |

## Interface Contract

Add an internal queued item shape with these fields:

```python
{
    "sequence": int,
    "request": ChatRequest,
    "timestamp": str,
    "reply_context": ReplyContext,
    "global_user_id": str,
    "user_profile": dict,
    "bot_name": str,
    "future": asyncio.Future[ChatResponse],
}
```

Queue policy helpers must be deterministic:

```python
def _is_tagged(item) -> bool
def _is_bot_reply(item) -> bool
def _prune_waiting_queue(waiting_items: list[item]) -> tuple[list[item], list[item]]
```

`_is_bot_reply` must read only the normalized request metadata supplied by the adapter:

```python
item["request"].reply_context.reply_to_current_bot is True
```

It must not call `_hydrate_reply_context(...)`, query `conversation_history`, compare reply target IDs, or infer reply ownership from message text. `_prune_waiting_queue` returns `(survivors, dropped)` and applies the three pruning stages in order.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/service.py`
  - Add queue state, worker lifecycle, queue pruning, item completion, and dropped-message handling.
  - Refactor current `/chat` body into worker-owned processing without changing public API.

### Create

- `tests/test_service_input_queue.py`
  - Focused deterministic tests for pruning and worker behavior.

### Keep

- Adapters unchanged.
- RAG, cognition, dialog, consolidator, `db_writer`, scheduler, Cache2, and DB schemas unchanged.

## Implementation Order

- Add internal queue state and lifecycle fields in `service.py`: queue deque/list, sequence counter, worker task, worker wake event, shutdown flag.
- Add deterministic helpers for tagged/replied checks and pruning.
- Extract the existing request-preparation and user-message-save behavior into service-local internal routines or clearly separated blocks.
- Keep queue pruning on the adapter-supplied request shape; do not hydrate reply context before pruning.
- Change `/chat` to prepare metadata, enqueue, wake worker, await item future, and return its response.
- Remove user-message `save_conversation` from the endpoint path.
- Add the worker loop:
  - wait for queue items,
  - prune waiting items,
  - save dropped user messages and resolve them with empty `ChatResponse` before selecting the next consumed item,
  - pop the oldest survivor,
  - save the consumed user message,
  - process it through the existing graph path,
  - resolve its response future,
  - await bot-save/progress/consolidation before the next loop.
- Update lifespan startup/shutdown to start and cancel/drain the worker safely.
- Keep the current semaphore behavior only if still useful as a defensive guard inside the worker; the queue worker becomes the primary one-at-a-time gate.
- Add tests.

## Verification

### Tests

Run:

```powershell
pytest tests\test_service_input_queue.py -q
pytest tests\test_service_background_consolidation.py -q
pytest tests\test_service_health.py tests\test_persona_supervisor2.py tests\test_consolidator_efficiency.py -q
```

### Required Test Cases

- Queue length `3` with ordinary messages drops all non-tagged/non-bot-reply items.
- Queue length greater than `5` after first prune drops tagged-but-not-bot-reply items and keeps bot replies.
- Queue length still greater than `5` after second prune keeps only the latest waiting item.
- Dropped items call `save_conversation` and return empty `ChatResponse`.
- Dropped items are saved before the next surviving item's graph/RAG path starts.
- `/chat` enqueue path does not call `save_conversation` directly.
- Dropped items never invoke `_graph.ainvoke`.
- Worker consumes the oldest surviving queued item.
- A second graph invocation does not start until prior post-response consolidation finishes.
- `mentioned_bot=True` survives the first pruning stage.
- `reply_context.reply_to_current_bot=True` survives first and second pruning stages.
- A request with `reply_to_message_id` but missing/false `reply_context.reply_to_current_bot` is not protected by the reply rule.
- Replies to other users are not protected unless `mentioned_bot=True`.

## Acceptance Criteria

This plan is complete when:

- `/chat` uses the global queue instead of directly running the graph.
- Dropped messages are persisted as user rows and receive no character reply.
- The pruning policy exactly matches the requested three-stage behavior.
- RAG cannot start for the next consumed message until prior consumed-message post-response writes/consolidation finish.
- Existing adapters continue to work without API changes.
- All verification commands pass.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Adapter requests wait longer during bursts | Drop queued noise and keep the worker single-threaded | Queue tests and adapter smoke |
| Dropped messages pollute future RAG | This is accepted by user requirement because dropped messages must be saved | Dropped-message persistence test |
| In-process queue does not protect multi-worker deployment | Document single-worker assumption and defer DB-backed queue | Runtime/deployment review |
| Worker shutdown leaves pending HTTP calls | Resolve remaining futures with empty responses on shutdown | Shutdown-focused unit test if implemented |

## Rollback / Recovery

- Code rollback path: revert `service.py` queue integration and `tests/test_service_input_queue.py`.
- Data rollback path: none; no schema or migration is introduced.
- Irreversible operations: none beyond ordinary conversation-history user rows.
- Recovery verification: existing `/chat` direct path tests pass after rollback.

## Assumptions

- Queue ordering uses arrival order at the brain service, not platform timestamps.
- Current production remains one brain process / one uvicorn worker.
- Dropped user messages should be available to future history/RAG because they are saved to `conversation_history`.
