# message coalescing queue module plan

## Summary

- Goal: Move chat input queue bookkeeping out of `service.py` and add simple message coalescing for private follow-ups and addressed-start group follow-ups.
- Plan class: medium
- Status: completed
- Overall cutover strategy: compatible
- Highest-risk areas: queue ordering, user-message persistence, response futures, group-message directionality, and preserving the RAG-safe worker gate.
- Acceptance criteria: the service uses a dedicated queue module; private follow-ups and narrow addressed group follow-ups are collapsed inside the same global queue; every original message is saved before RAG; only one graph/RAG/dialog pass runs per collapsed turn.

## Context

The current global input queue lives inside `src/kazusa_ai_chatbot/service.py`.
It already provides one-at-a-time processing, group noise pruning, dropped
message persistence, and RAG-safe waiting for post-response writes before the
next turn enters graph/RAG.

This plan extends that architecture without adding a second lane. Message
coalescing should happen inside the same global queue worker before group-noise
pruning. The goal is to reduce repeated stale RAG passes when a user sends
rapid follow-ups that are really one thought.

## Database Research

Read-only research was run against
`kazusa_bot_core.conversation_history`. The detailed artifact is:

```text
test_artifacts/group_adjacent_collapse_stats_20260429.json
```

Definitions:

- group row: `channel_type == "group"`
- adjacent: consecutive rows in the same group channel sorted by timestamp
- same author: both adjacent rows are user rows with equal `platform_user_id`
- addressed: `mentioned_bot == true` or `reply_context.reply_to_current_bot == true`
- production-like QQ: `platform == "qq"` and channel id not starting with `pytest`

Production-like QQ group stats:

```text
group rows analyzed: 17,878
adjacent user-user pairs: 16,399
same-author adjacent pairs: 5,550
same-author rate: 33.84%
```

Same-author adjacent pair breakdown:

```text
plain -> plain        5,540  99.82%
plain -> addressed        5   0.09%
addressed -> plain        4   0.07%
addressed -> addressed    1   0.02%
```

For same-author pairs where the first message was addressed:

```text
addressed-start pairs: 5
addressed -> plain: 4
addressed -> addressed: 1
```

Candidate addressed-pair timing:

```text
count: 10
p50: 21.74s
p75: 38.8s
max: 75.86s
```

Conclusion:

- Same-author group adjacency is common.
- Almost all same-author group adjacency is `plain -> plain`, which can easily
  represent multi-direction group chat and should not be collapsed.
- Addressed-start same-author group follow-ups are rare but plausible collapse
  candidates.
- Group coalescing must be narrow: only collapse same-author adjacent group
  runs that start with an addressed message.

## Mandatory Rules

- Keep all chat messages inside the same global queue.
- Do not add a private lane, group lane, second worker, or concurrent processing path.
- Do not bypass the worker's RAG-safe post-response gate.
- Move queue bookkeeping from `service.py` into one dedicated module.
- No dedicated folder is needed.
- Keep the module simple; do not add wrappers around functions that do not need wrappers.
- This is a one-shot contract change. No compatibility shim, fallback field,
  alternate reply flag, or transitional old/new behavior is allowed.
- Coalesce only queued items that have not started graph processing.
- Preserve the earliest item queue position for every collapsed turn.
- Save every original collapsed message to `conversation_history` before graph/RAG.
- Use combined content in arrival order for graph input.
- Make `use_reply_feature` a monotonic permission latch: it starts `true`,
  and once any stage sets it `false`, no later stage may restore it to `true`.
- Collapsed turns must set `use_reply_feature=false` before graph/RAG so
  platform reply mode stays disabled through the whole pipeline.
- Return empty `ChatResponse` objects for earlier collapsed request futures.
- Return the actual dialog response through the surviving collapsed turn's future.
- Log collapsed platform message IDs and the surviving queued sequence.
- Do not change prompts, RAG agents, cognition, dialog, consolidator internals, `db_writer`, DB schemas, Cache2 policy, scheduler behavior, or adapter API shape.

## Must Do

- Create a dedicated module such as `src/kazusa_ai_chatbot/chat_input_queue.py`.
- Move queue dataclasses, queue state, pruning, coalescing, enqueue, and dequeue handoff into that module.
- Keep worker lifecycle, graph execution, persistence helpers, and response propagation in `service.py`.
- Add private-message coalescing.
- Add addressed-start group-message coalescing.
- Keep existing group-noise pruning behavior for non-coalesced group messages.
- Add deterministic tests for module behavior and service integration.

## Deferred

- Cross-process or multi-replica queue/coalescing.
- Adapter-side batching.
- Streaming or mutating the currently processing turn.
- User-visible "messages collapsed" response text.
- Collapsing `plain -> plain` group messages.
- Collapsing group runs that begin plain and later mention/reply to the bot.

## Cutover Policy

- `/chat` request/response API: compatible.
- Adapter behavior: compatible.
- Database schema: compatible, no migration.
- Runtime behavior: bigbang inside the brain process.
- Reply-flag behavior: bigbang contract change. The only supported behavior
  after implementation is the monotonic `use_reply_feature` latch feeding
  `ChatResponse.should_reply`.

## Agent Autonomy Boundaries

- The implementation agent may choose small private helper names inside
  `chat_input_queue.py` only when the helpers preserve this plan's contracts.
- The implementation agent must not introduce a class hierarchy, compatibility
  adapter, shim layer, old-contract fallback, or second reply flag.
- The implementation agent must not implement both old and new
  `use_reply_feature` semantics side by side.
- The implementation agent must not move prompt, RAG, cognition, dialog,
  consolidator, scheduler, Cache2, or database-schema behavior to satisfy this
  plan.
- If the monotonic latch cannot be implemented with the existing graph state
  reducer mechanism, the implementation agent must stop and report the blocker
  instead of inventing an alternate reply-mode pathway.

## Target State

The service owns HTTP, worker lifecycle, graph execution, persistence, and
response creation. The new queue module owns only queue bookkeeping:
accepting new messages, applying collapse/prune policy, and returning the next
dequeued handoff to the service.

Service worker loop:

```text
wait for queued items
coalesce queued private follow-ups
coalesce queued addressed-start group follow-ups
prune remaining group/noise messages
receive dropped/collapsed/next handoff from ChatInputQueue
save dropped and collapsed original user messages
consume oldest surviving queued item
call service-owned turn processor for one graph/RAG pass
return dialog for the surviving item
await post-response writes/consolidation/cache invalidation
repeat
```

Coalescing does not reorder work across scopes. A collapsed turn keeps the
queue position of the earliest item in its scope/run.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Module | `src/kazusa_ai_chatbot/chat_input_queue.py` | One dedicated file is enough; no folder needed. |
| Queue location | Same global queue | Preserves one-at-a-time RAG and persistence ordering. |
| Module style | Simple functions and dataclasses | Avoid wrapper objects unless they remove real complexity. |
| Coalescing timing | Before group/noise pruning | Normalize follow-ups before applying drop policy. |
| Private scope | `(platform, platform_channel_id, platform_user_id)` | Prevents combining different private users or channels. |
| Group scope | Same channel and same author, adjacent in queue order | Prevents mixing directions in group chat. |
| Group eligibility | First message in run must be addressed | Avoids collapsing ordinary multi-direction `plain -> plain` group chat. |
| Group follow-ups | May be addressed or plain | Real DB shows addressed-start follow-ups are usually plain. |
| Group time gap | Maximum 120 seconds between adjacent messages in a collapsed run | Covers observed addressed candidates while limiting stale merges. |
| Queue position | Earliest item in collapsed turn | Follow-ups do not jump ahead of older queued work. |
| Graph input | Combined content in arrival order | Lets the model answer the user's complete thought once. |
| Original rows | Save every original message | Preserves audit/history and satisfies persistence requirements. |
| Reply feature contract | Monotonic latch, default true | Any stage can veto platform reply mode; later stages cannot re-enable it. |
| Collapsed reply mode | Set `use_reply_feature=false` at graph input | A collapsed turn should not reply to one platform message because it represents multiple messages. |
| Earlier futures | Empty `ChatResponse` | Adapter receives no duplicate replies for collapsed follow-ups. |
| Surviving future | Actual dialog response | Exactly one character reply is produced for the collapsed turn. |

## Interface Contract

Add a dedicated internal queue module. Public API is service-internal only and
centered on one bookkeeping class:

```python
@dataclass
class QueuedChatItem:
    sequence: int
    request: ChatRequest
    timestamp: str
    future: asyncio.Future[ChatResponse]
    combined_content: str | None = None
    collapsed_items: list[QueuedChatItem] = field(default_factory=list)

@dataclass
class DequeuedChatTurn:
    next_item: QueuedChatItem | None
    dropped_items: list[QueuedChatItem]
    collapsed_items: list[tuple[QueuedChatItem, QueuedChatItem]]

class ChatInputQueue:
    async def enqueue(self, request: ChatRequest) -> ChatResponse: ...
    async def wait_for_next(self) -> DequeuedChatTurn: ...
    def complete(self, item: QueuedChatItem, response: ChatResponse) -> None: ...
    async def drain(self) -> list[QueuedChatItem]: ...
```

`chat_input_queue.py` must not receive platform events, start/stop workers,
save messages, call graph/RAG, run consolidation, or propagate responses to
adapters. It only resolves queue bookkeeping and futures through the common
`enqueue`, `wait_for_next`, `complete`, and `drain` interface.

## Reply Feature Latch

`use_reply_feature` must become a graph-state latch, not a plain last-writer
boolean. The default for a normal non-collapsed turn is `true`.

Variable origins and ownership:

- `use_reply_feature` originates in `IMProcessState` in
  `src/kazusa_ai_chatbot/state.py`. It is an internal graph-state permission
  latch for whether the platform reply feature is still allowed.
- `service.py` creates the first value in the graph initial state:
  `true` for normal turns and `false` for collapsed turns.
- Any graph node may update `use_reply_feature` to `false` to veto platform
  reply mode.
- No graph node may re-enable platform reply mode once the latch is `false`;
  the reducer must preserve `false`.
- `should_reply` originates only in the outbound `ChatResponse` API model in
  `service.py`. It is not graph state and must not become a second source of
  truth.
- `ChatResponse.should_reply` must be derived from the final graph state's
  `use_reply_feature` when dialog is actually being returned. There must be no
  separate collapse override, shim flag, or adapter-side inference to
  reinterpret it.

State contract:

```text
initial true + stage true  -> true
initial true + stage false -> false
current false + stage true -> false
current false + stage false -> false
```

Implementation target:

- Add a small boolean reducer in `state.py`, for example
  `def keep_false(current: bool, update: bool) -> bool`.
- Annotate `IMProcessState.use_reply_feature` with that reducer.
- Set `initial_state["use_reply_feature"] = True` for normal turns.
- Set `initial_state["use_reply_feature"] = False` for collapsed turns.
- Keep `ChatResponse.should_reply` sourced from final
  `result["use_reply_feature"]` and the existing "no dialog means no reply
  feature" response contract; do not add a separate collapse override.

This lets relevance, queue coalescing, debug modes, or future stages veto
platform reply mode while preserving the existing response field contract.

## Coalescing Rules

### Private Messages

Collapse queued private messages when all are:

- `request.channel_type == "private"`
- same `platform`
- same `platform_channel_id`
- same `platform_user_id`
- not yet processing

The collapsed private turn uses the earliest item as survivor. All later items
in that private scope become collapsed originals.

### Group Messages

Collapse a queued group run only when:

- all items are `request.channel_type == "group"`
- all items are in the same `platform` and `platform_channel_id`
- all items have the same `platform_user_id`
- items are adjacent in queue order with no other author or assistant turn between them
- the first item is addressed:
  - `mentioned_bot is True`, or
  - `reply_context.reply_to_current_bot is True`
- each adjacent gap is `<= 120` seconds using queued timestamps

Follow-up items in the run may be addressed or plain. Do not collapse group
runs that start plain, even if a later follow-up is addressed.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/chat_input_queue.py`
  - Queue item dataclass.
  - `ChatInputQueue` bookkeeping class.
  - Tagged/reply/private helpers.
  - Private coalescing.
  - Addressed-start group coalescing.
  - Existing group-noise pruning.
  - Enqueue, dequeue handoff, completion, and drain.

### Modify

- `src/kazusa_ai_chatbot/service.py`
  - Remove queue bookkeeping moved to the module.
  - Keep queue worker lifecycle and worker loop service-owned.
  - Keep graph/persistence/post-response behavior service-owned.
  - Adapt `/chat`, lifespan startup, and shutdown to call the queue class.
  - Initialize normal turns with `use_reply_feature=True`.
  - Initialize collapsed turns with `use_reply_feature=False`.

- `src/kazusa_ai_chatbot/state.py`
  - Add the monotonic false-preserving reducer for `use_reply_feature`.

### Tests

- Add or update queue module tests in `tests/test_service_input_queue.py` or a new focused `tests/test_chat_input_queue.py`.
- Keep service integration coverage in `tests/test_service_background_consolidation.py`.

## Implementation Order

1. Create `chat_input_queue.py` with `ChatInputQueue` and move existing queue dataclasses/helpers without behavior changes.
2. Wire service-owned `/chat`, worker lifecycle, and worker loop to the queue class and keep existing tests passing.
3. Add private coalescing helpers and tests.
4. Add addressed-start group coalescing helpers and tests.
5. Add combined-content handling for graph input and user-row persistence.
6. Add collapsed-original persistence and earlier future completion.
7. Add the `use_reply_feature` monotonic latch reducer and initialize
   collapsed turns with `use_reply_feature=False`.
8. Add collapse audit logs.
9. Run the verification suite.

## Progress Checklist

- [x] Create `src/kazusa_ai_chatbot/chat_input_queue.py`.
- [x] Move existing queue bookkeeping out of `service.py` without behavior drift.
- [x] Add private-message coalescing.
- [x] Add addressed-start group-message coalescing.
- [x] Persist every collapsed original before graph/RAG.
- [x] Add the monotonic `use_reply_feature` reducer in `state.py`.
- [x] Wire `ChatResponse.should_reply` from final `use_reply_feature` while preserving the no-dialog no-reply response contract.
- [x] Add queue module and service integration tests.
- [x] Run all verification commands.

## Verification

Run:

```powershell
pytest tests\test_service_input_queue.py -q
pytest tests\test_service_background_consolidation.py -q
pytest tests\test_service_health.py tests\test_persona_supervisor2.py tests\test_consolidator_efficiency.py -q
```

Implementation verification completed:

```powershell
python -m py_compile src\kazusa_ai_chatbot\chat_input_queue.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_state.py
pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_state.py -q
pytest tests\test_service_health.py tests\test_persona_supervisor2.py tests\test_consolidator_efficiency.py -q
```

Results:

- `py_compile`: passed.
- Focused queue/consolidation/state suite: `30 passed`.
- Requested regression suite: `8 passed`.

Required test cases:

- Existing group pruning behavior remains unchanged for non-coalesced messages.
- Multiple queued private messages in the same scope collapse into one surviving item.
- Private messages in different scopes do not collapse together.
- Private collapse preserves earliest queue position.
- Addressed-start same-author group follow-ups collapse when adjacent and within the time gap.
- `addressed -> plain` group follow-up collapses.
- `addressed -> addressed` group follow-up collapses.
- `plain -> plain` group messages do not collapse.
- `plain -> addressed` group messages do not collapse.
- Different-author group messages do not collapse.
- Same-author group messages separated by another user do not collapse.
- Same-author addressed group messages beyond the time gap do not collapse.
- Combined graph input uses message content in arrival order.
- Every original collapsed message is saved before graph/RAG starts.
- Earlier collapsed futures return empty `ChatResponse`.
- Surviving collapsed future receives the actual dialog response.
- Normal non-collapsed turns start with `use_reply_feature=True`.
- If any graph stage sets `use_reply_feature=False`, a later
  `use_reply_feature=True` update does not re-enable it.
- Collapsed turns produce final `use_reply_feature=False` without a late
  `ChatResponse.should_reply` override.
- Worker still waits for post-response writes/consolidation before consuming the next item.

## Acceptance Criteria

This plan is complete when:

- Queue bookkeeping is no longer embedded directly in `service.py`.
- The new queue module remains simple and does not introduce unnecessary wrappers.
- Private follow-ups coalesce inside the global queue.
- Narrow addressed-start group follow-ups coalesce inside the global queue.
- No private queued message is dropped by group/noise pruning.
- Plain-start group runs are not collapsed.
- All original collapsed messages are persisted before the collapsed RAG pass.
- Only one graph/RAG/dialog pass runs for a collapsed burst.
- `use_reply_feature` is a false-preserving graph-state latch.
- Collapsed turns disable platform reply mode by setting the latch false at
  graph input, not by overriding `ChatResponse.should_reply` at the end.
- Tests pass for module behavior, service integration, futures, ordering, and existing pruning.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Wrong group direction is merged | Only collapse same-author runs that start addressed | Group coalescing tests |
| Private bursts increase latency | Collapse queued follow-ups into one graph pass | Private worker tests |
| Lost audit trail | Save every original collapsed message | Persistence-order tests |
| RAG sees stale state | Keep existing worker post-response gate | Background consolidation test |
| Module boundary becomes overbuilt | Use simple functions/dataclasses; avoid wrappers unless needed | Code review |

## Rollback / Recovery

- Code rollback path: revert `chat_input_queue.py`, restore queue mechanics in `service.py`, and revert coalescing tests.
- Data rollback path: none; no schema or migration is introduced.
- Irreversible operations: none beyond ordinary conversation-history user rows.

## Assumptions

- `channel_type == "private"` is the service-level signal for private messages.
- Group-addressed means `mentioned_bot is True` or `reply_context.reply_to_current_bot is True`.
- Queue ordering uses brain-service arrival order and queued timestamps.
- The currently processing turn is not mutated by later follow-ups.
