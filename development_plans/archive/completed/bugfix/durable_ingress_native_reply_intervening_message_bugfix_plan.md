# durable ingress native-reply intervening-message bugfix plan

## Summary

- Goal: make the native-reply decision use durable inbound-message evidence
  instead of transient intake or turn-settlement state, and extend the same
  evidence model to background results that should reply to their original
  source message.
- Status: completed.
- Scope boundary: the /chat ingress receipt, the final /chat
  use_reply_feature calculation, and accepted-task/background-result
  delivery targeting. Conversation persistence, its query index, focused
  tests, and the relevant subsystem ICDs are included.
- Change direction: commit one canonical conversation_history receipt for
  every inbound /chat packet before queue admission, stamp it with a
  server-generated received_at, and query that history at response
  construction time. Background delivery carries the original source message
  ID separately from the synthetic tool-result identity and uses the same
  durable age/interleaving gate to populate the existing dispatcher reply
  target.
- Acceptance state: completed; implementation, independent review, parent
  remediation, and verification are recorded below.

The bug is caused by treating a work queue as a record of what the user sent.
The queue is a work-state projection: intake coalescing, pruning, shutdown
drain, and frontline discard can remove a packet from cognition while the
packet remains visible to the user. The queue therefore cannot answer whether
an inbound message was inserted between the response owner and the visible
response.

## Confirmed Decisions

- /chat counts every durable conversation_history row with role="user" in the
  same platform and channel, after the response-owner receipt and at or before
  the response cutoff. Author identity and later intake disposition do not
  filter this fact.
- conversation_history is the canonical durable receipt source. New inbound
  rows receive a server-generated UTC received_at before queue admission.
  The existing timestamp field keeps its current event/local-time meaning
  and is not used as the authoritative arrival clock.
- The /chat promotion keeps its existing group-channel and non-empty
  platform-message-ID guard. The change replaces only the queue-derived
  interleaving and process-local elapsed-time inputs.
- Self cognition remains unchanged. It does not reply to another user's
  message and does not participate in this native-reply decision.
- Background result delivery is included. When its deterministic gate
  qualifies, it replies to the original source message from
  BackgroundWorkJobDoc.source_message_id; it never uses the synthetic
  tool-result:<task_id> identity as the reply target.
- Background delivery applies the original-message target in the source
  channel when the source row, message ID, and adapter target are valid. It
  does not inherit the /chat group-only promotion guard merely because the
  result was produced by a background worker.
- The delay threshold is strict: elapsed_seconds > 120.0. Exactly 120.0
  seconds does not promote.
- A missing owner/source row, missing received_at, blank message ID, or
  unsupported target fails closed to the existing normal-send/base-flag
  behavior. No synthetic target or client timestamp is substituted.
- The graph's existing keep_true/base reply decision remains monotonic.
  Deterministic evidence may promote False to True, but cannot demote an
  existing True decision.
- No new LLM stage, prompt field, semantic relevance rule, public
  ChatResponse field, or reply_anchor_message_id contract is added.

## Scope And Change Direction

### Current evidence and failure boundary

The implementation review established these ownership facts:

| Path | Current source of truth | Failure for this bug |
|---|---|---|
| /chat admission | ChatInputQueue.enqueue() and later intake persistence | A queue-pruned, coalesced, drained, or frontline-discarded packet can be visible without being represented in the settlement fragments used by the final reply calculation. |
| /chat age | item.enqueue_monotonic | It measures queue-work age, not durable user-message receipt age, and cannot survive a queue lifecycle boundary. |
| conversation_history | save_user_message_from_item() and save_conversation() | It is the right durable surface, but current timestamp may be adapter-supplied local_timestamp rather than server arrival time. |
| Event log and protected LLM traces | operational diagnostics | They are best-effort/diagnostic records, not a complete control-plane receipt or response-order source. |
| Self cognition | self-cognition action candidate and worker | It has no user-message native-reply target and remains outside this fix. |
| Background result delivery | accepted-task result episode and dispatcher | It currently carries a synthetic tool-result platform ID and sends reply_to_msg_id=None, even though the job retains the original source message ID. |

The implementation must preserve the existing semantic disposition boundary:
frontline discard still stops cognition, but its durable inbound receipt
remains available for later delivery-order evidence.

### Durable ingress receipt

1. Normalize the inbound /chat request into the canonical conversation
   receipt before calling ChatInputQueue.enqueue(). The receipt must contain
   the platform, platform channel, user role, platform message ID when
   available, message body/envelope fields required by the existing history
   contract, and a server-generated received_at.
2. Use the canonical conversation_history collection for this receipt. If
   the existing full-history save path performs optional enrichment such as
   embedding generation after insertion, split that enrichment from the
   receipt commit so the receipt is committed before queue admission while
   remaining one canonical row rather than creating a shadow collection or a
   duplicate row.
3. Propagate the committed conversation row identity and its received_at
   through ChatRequest/QueuedChatItem and every coalesced fragment. Later
   frontline preparation, queue pruning, private-message collapse, and
   normal intake must update or consume that same row instead of inserting a
   second copy.
4. Preserve the existing visible-history behavior for packets that do not
   enter cognition. A queue-pruned, queue-drained, coalesced, or frontline
   discard packet still has a committed user row; source_episode_id remains
   an admission/cognition linkage and is not repurposed as a receipt status.
5. Add an idempotent conversation_history index supporting the exact
   interleaving lookup: platform, platform channel, role, and received_at.
   Keep existing timestamp indexes and assistant-delivery indexes intact.
6. Do not backfill legacy rows from timestamp. A client/local event time
   cannot prove server arrival order. New readers treat absent received_at as
   unavailable evidence, which preserves safe normal/base behavior for legacy
   rows.

received_at must use the project's canonical server UTC storage
representation with sufficient precision for range queries. It must never be
copied from ChatRequest.local_timestamp or any adapter-provided clock.

### Durable interleaving query

Add one deterministic database helper owned by the conversation-history
repository. Its contract is:

~~~text
has_inbound_after(
    platform,
    platform_channel_id,
    owner_received_at,
    response_cutoff_received_at,
) -> bool
~~~

The query filters only:

~~~text
platform == requested platform
platform_channel_id == requested channel
role == "user"
owner_received_at < received_at <= response_cutoff_received_at
~~~

It deliberately does not filter by author, source_episode_id, intake_action,
response_action, queue status, event-log status, or LLM trace state. The
upper cutoff is captured at the final response boundary so an inbound packet
received after that boundary cannot retroactively change the reply target. The
implementation must retain canonical timestamp precision and use a stable
insertion-order tie-break when equal server instants occur.

### /chat final reply decision

Replace the current settlement-fragment owner_mismatch and
item.enqueue_monotonic inputs in the final /chat response construction. The
target invariant is:

~~~text
base_reply = graph_result.use_reply_feature

owner = durable conversation row for the response-owner request
cutoff = server received-at captured at final response construction

intervening = has_inbound_after(
    owner.platform,
    owner.platform_channel_id,
    owner.received_at,
    cutoff,
)

delayed = cutoff - owner.received_at > 120.0

promotion = (
    request.channel_type == "group"
    and request.platform_message_id.strip() is non-empty
    and (intervening or delayed)
)

final_reply = bool(final_dialog) and (base_reply or promotion)
~~~

The existing response-owner/future routing remains responsible for deciding
which /chat request receives the response. The durable history query is only
the delivery-format fact that answers whether a later inbound message exists.
The query may be skipped when the graph base flag is already true, provided
the true value is preserved exactly; it must never be replaced with a queue
inspection.

### Background result delivery

1. Preserve the synthetic tool-result identity for the result episode,
   tracing, and accepted-task provenance.
2. Carry the job's original source_message_id as a distinct internal
   original-source field through result_source.py,
   build_tool_result_episode(), and
   _deliver_accepted_task_result_episode(). Do not overload the synthetic
   origin_metadata.platform_message_id.
3. At background delivery, resolve the original source row by exact
   platform, channel, and source message ID. Require a user-role row with a
   usable server received_at.
4. Capture a server delivery cutoff and apply the same durable interleaving
   query and strict delay rule:

~~~text
source_age = cutoff - source_row.received_at
intervening = has_inbound_after(
    source_row.platform,
    source_row.platform_channel_id,
    source_row.received_at,
    cutoff,
)

reply_target = source_message_id
    when (source_age > 120.0 or intervening)
    and the source ID/channel/adapter target is valid
    otherwise None
~~~

5. Pass reply_target through the existing dispatcher payload as
   reply_to_msg_id. Keep background workers free of direct adapter calls; the
   accepted-task result delivery boundary remains the owner of dispatch,
   validation, delivery receipts, and audit data.
6. Do not change cognition's semantic use_reply_feature contract for
   tool-result episodes. This is deterministic target selection at the
   delivery boundary, not an LLM decision.

### Self cognition

Leave self_cognition/runner.py, self_cognition/tracking.py, and
self_cognition/worker.py behavior unchanged. Existing optional action
candidate reply fields remain available for any separately authorized
self-cognition behavior, but this bugfix adds no user-message lookup or
original-message targeting to that path.

## Mandatory Skills

- development-plan: governs this draft's lifecycle, approval boundary,
  implementation handoff, verification, and closeout.
- local-llm-architecture: preserves deterministic ownership for receipt,
  timing, persistence, and delivery targeting while keeping semantic
  judgment in the existing LLM stages.
- py-style: applies to all Python schema, service, database, background
  delivery, and test changes.
- test-style-and-execution: applies to deterministic regression coverage,
  database-backed tests, and any live test execution.
- python-venv: applies to every Python compile, test, and database-test
  command.

No live database export is required for this draft. database-data-pull is
therefore not a mandatory implementation skill; a future implementation
should use it only if the user requests production-history evidence or a
read-only data pull.

## Mandatory Rules

- Read and follow development_plans/README.md, README.md, docs/HOWTO.md, and
  the directly involved subsystem READMEs before implementation.
- Use venv\Scripts\python.exe for Python checks and tests. Never read .env as
  part of this plan.
- Preserve unrelated worktree changes.
- Keep the archived
  relevance_native_reply_review_remediation_plan.md and
  relevance_native_reply_monotonic_delivery_plan.md unchanged.
- Use conversation_history.received_at as the only authoritative new
  server-arrival clock. Do not use queue depth, queue membership, settlement
  fragments, source_episode_id, event-log presence, LLM trace presence,
  item.enqueue_monotonic, or client local_timestamp to determine interleaving
  or the 120-second delay.
- Persist the receipt before queue admission and maintain exactly one
  canonical row for each inbound packet through coalescing, pruning,
  discard, and normal cognition paths.
- Keep RAG, cognition, dialog, persistence, consolidation, scheduler, and
  adapter ownership boundaries intact.
- Do not add pre-processing or post-processing that changes LLM relevance
  judgment. The deterministic history fact may change only delivery-format
  promotion after an existing response is available.
- Do not introduce compatibility aliases, parallel receipt stores, raw
  event-log fallbacks, or public reply-anchor fields.
- Preserve the strict >120.0 boundary, the existing group-only /chat
  promotion guard, and monotonic base-flag behavior.
- Preserve background worker ownership: workers produce result state; the
  accepted-task delivery path resolves the source and calls the dispatcher.
- When received_at is absent or the original source cannot be resolved, fail
  closed to the existing normal/base behavior and record enough deterministic
  diagnostic context for review.

## Must Do

1. Update the conversation-history schema contract with a server-only
   received_at field and document its relationship to the existing timestamp.
2. Implement the pre-queue /chat receipt boundary and propagate its row
   identity/receipt time through queue and coalescing structures.
3. Remove duplicate insertion from downstream normal-intake, queue-drop,
   queue-drain, and collapsed-private paths while preserving their completion
   and trace behavior.
4. Add the indexed durable interleaving helper and exact same-platform/channel
   role-user range query.
5. Replace the final /chat queue-derived native-reply promotion inputs with
   the durable owner age and cutoff query. Preserve empty-dialog suppression
   and the existing base True reducer behavior.
6. Carry the original background source message ID independently from the
   synthetic tool-result ID and use it for deterministic reply_to_msg_id
   selection when the durable gate qualifies.
7. Add deterministic regression coverage for:
   - receipt commit before queue admission;
   - a queue-pruned, queue-drained, or frontline-discarded packet remaining
     visible in conversation_history;
   - an intervening same-channel message from the same author;
   - an intervening same-channel message from a different author;
   - an intervening row whose intake path later discards or prunes it;
   - an unrelated platform/channel row not promoting the response;
   - a message received after the response cutoff not promoting the response;
   - strict <120, =120, and >120 durable age boundaries;
   - missing legacy received_at, missing owner row, blank message ID, empty
     dialog, and graph base True;
   - background source-ID propagation, original-source targeting, no
     synthetic tool-result targeting, missing-source fallback, and the same
     age/interleaving boundaries;
   - unchanged self-cognition delivery behavior.
8. Update the /chat, conversation-history, and background-work ICDs/READMEs
   to describe durable receipt ordering, disposition-independent evidence,
   source-message targeting, and the legacy-row fail-closed rule.
9. Record implementation evidence, focused test output, diff review, and
   final workspace status in the eventual execution record before lifecycle
   closeout.

## Deferred

- Changes to self-cognition behavior, self-cognition scheduling, or
  self-cognition prompts.
- Changes to the frontline relevance contract, settled relevance semantics,
  character stance, response-owner selection, turn coalescing policy, or
  response futures.
- Replacing the queue, redesigning shutdown, or making queue state durable.
- Cross-channel or cross-platform interleaving.
- Filtering intervening messages by author, mention, relevance, disposition,
  or whether they entered a cognition episode.
- Backfilling received_at from legacy timestamp values.
- New public response fields, adapter API changes, or a new reply-anchor
  semantic decision.
- Changing the background worker into an adapter caller.
- Delivery of a result after the original source row has been deleted or
  cannot be validated; those cases remain normal sends.
- Prompt changes, new LLM calls, model routing changes, or semantic
  post-processing.
- Unrelated queue, conversation-history, background-task, or dispatcher
  cleanup.

## Target State

The canonical live paths are:

~~~text
/chat request
  -> server UTC received_at
  -> one conversation_history user receipt
  -> queue admission/coalescing/pruning
  -> frontline/cognition when admitted
  -> final response cutoff
  -> durable conversation_history range query
  -> existing group native-reply promotion
  -> existing adapter delivery
~~~

~~~text
background job source_message_id
  -> synthetic tool-result episode identity
  -> original source ID carried as separate delivery metadata
  -> accepted-task result cognition/dialog
  -> delivery cutoff
  -> durable source-row age/interleaving query
  -> reply_to_msg_id = original source ID when qualified
  -> existing dispatcher and adapter delivery
~~~

For /chat, the durable fact is independent of packet fate:

~~~text
received user row after owner
  -> counts even when queue-pruned, queue-drained, coalesced,
     or frontline-discarded
  -> does not require cognition, source_episode_id, event log, or trace
~~~

For background delivery, the original message remains the only valid reply
target:

~~~text
original source message ID  -> valid reply target
tool-result:<task_id>       -> provenance/episode identity only
~~~

The durable row contract is:

~~~json
{
  "role": "user",
  "platform": "<server-normalized platform>",
  "platform_channel_id": "<server-normalized channel>",
  "platform_message_id": "<adapter ID when available>",
  "timestamp": "<existing event/local-time field>",
  "received_at": "<server-generated canonical UTC timestamp>"
}
~~~

Rows written before this change may lack received_at; they remain valid
historical records but are not authoritative evidence for new promotion.

## Change Surface

### Modify

- src/kazusa_ai_chatbot/service.py
  - pre-queue /chat admission;
  - final native-reply calculation;
  - accepted-task/background-result delivery target resolution.
- src/kazusa_ai_chatbot/chat_input_queue.py
  - receipt identity/time propagation through queued and coalesced items.
- src/kazusa_ai_chatbot/brain_service/contracts.py
  - canonical internal receipt metadata carried by the request/item contract.
- src/kazusa_ai_chatbot/brain_service/intake.py
  - consume/update the precommitted receipt without duplicate insertion.
- src/kazusa_ai_chatbot/db/schemas.py
  - ConversationMessageDoc.received_at contract and focused typed metadata.
- src/kazusa_ai_chatbot/db/conversation.py
  - receipt persistence/update support, source-row lookup, and the durable
    interleaving query.
- src/kazusa_ai_chatbot/db/bootstrap.py
  - idempotent conversation_history index for platform/channel/role/
    received_at.
- src/kazusa_ai_chatbot/background_work/result_source.py
  - preserve original source-message metadata separately from tool-result
    identity.
- src/kazusa_ai_chatbot/cognition_episode.py
  - carry original source metadata through the tool-result episode contract
    if the current episode shape is the boundary that loses it.
- src/kazusa_ai_chatbot/brain_service/README.md,
  src/kazusa_ai_chatbot/db/README.md, and
  src/kazusa_ai_chatbot/background_work/README.md
  - document durable receipt ordering and original-message delivery.
- docs/HOWTO.md
  - document the implementation/verification boundary if it describes
    /chat intake or background result delivery.
- tests/test_service_input_queue.py,
  tests/test_background_work_delivery.py,
  tests/test_save_conversation_invalidation.py, and the most relevant
  conversation-history/dispatcher tests
  - update existing expectations and add focused regression coverage.

### Create

- A focused test module for durable ingress/interleaving queries if the
  existing conversation-history tests cannot express the full contract.
- This active draft plan and its registry entry.

### Keep unchanged

- src/kazusa_ai_chatbot/self_cognition/runner.py,
  self_cognition/tracking.py, and self_cognition/worker.py.
- src/kazusa_ai_chatbot/brain_service/turn_settlement.py response-owner and
  future semantics.
- Frontline/settled relevance prompts and semantic contracts.
- Dispatcher and adapter method signatures; the existing reply_to_msg_id
  payload remains the delivery hook.
- Archived native-reply plans and all unrelated user worktree changes.

### Delete

- No collection, adapter, public schema, or production module is deleted.
- The queue-derived final-reply predicates are removed only after their
  durable replacements are in place; no fallback to the old predicates is
  retained.

## Agent Autonomy Boundaries

The implementation agent may choose internal helper names, the exact
conversation-row ID representation, the exact typed request/item field names,
and whether receipt enrichment uses an update or a single insert helper,
provided the one-row-before-queue invariant and query contract remain exact.
It may add idempotent indexes and focused test fixtures.

The implementation agent must stop for a plan amendment before changing any
of these decisions:

- the conversation_history collection as the canonical receipt source;
- server-generated received_at or the strict >120.0 threshold;
- same-platform/channel, role-user, disposition-independent interleaving;
- the existing group-only /chat promotion guard;
- original source-message targeting for background results;
- self-cognition exclusion;
- monotonic preservation of an existing graph True;
- fail-closed handling for legacy rows or missing source metadata;
- public schemas, prompts, semantic LLM ownership, or adapter contracts.

## Verification

Before implementation:

- Capture git status --short and preserve the existing modified README and
  untracked plan/test files.
- Read the repository README, docs/HOWTO.md, directly involved subsystem
  READMEs, source files, and current tests.

During implementation:

- Use venv\Scripts\python.exe.
- Run deterministic tests in batches and inspect failures.
- Use a live MongoDB test only when the environment explicitly provides one;
  run it as an explicitly identified live-DB case.
- Do not run live LLM calls: this change has no prompt or model behavior
  change.

Required evidence:

1. A persistence test proves the conversation_history receipt is committed
   before queue admission and that downstream intake does not duplicate it.
2. Intake tests prove a packet that never reaches cognition is still present
   with received_at, while source_episode_id remains absent.
3. Query tests prove author/disposition independence, same-channel scope,
   upper-cutoff exclusion, and legacy-row fail-closed behavior.
4. /chat tests prove the durable interleaving and durable age promote the
   valid group response, while exact 120.0, missing evidence, private scope,
   empty dialog, and unrelated channels preserve the existing result.
5. Background tests prove the original source ID reaches the dispatcher,
   tool-result:<task_id> is never passed as reply_to_msg_id, and <120, =120,
   >120, interleaving, missing row, and missing-ID cases follow the contract.
6. Self-cognition regression tests show no new user-message query or reply
   target.
7. Run the relevant deterministic pytest files, Python compilation/checks,
   and git diff --check; record exact commands and outcomes in the eventual
   execution record.
8. Review the final diff for duplicate persistence, accidental queue fallback,
   public contract changes, .env access, and unrelated edits.

## Acceptance Criteria

- Every new inbound /chat packet has one committed conversation_history user
  receipt with server-generated received_at before queue admission.
- Queue pruning, coalescing, shutdown drain, and frontline discard cannot
  erase the durable receipt or make it invisible to the interleaving query.
- The final /chat native-reply decision no longer relies on queue membership,
  settlement-fragment sequence, queue age, event-log presence, or trace
  presence.
- Any same-platform/channel inbound user row after the owner and before the
  cutoff promotes an eligible group /chat response, regardless of author or
  later packet disposition.
- A row after the cutoff does not promote the response.
- Durable owner age promotes only when it is strictly greater than 120.0
  seconds; exactly 120.0 does not.
- Existing graph use_reply_feature=True remains true when a visible dialog
  exists, and an empty dialog never produces a visible native reply.
- Background results that qualify use the original source message ID as
  reply_to_msg_id, including when the result took longer than 120 seconds.
- Synthetic tool-result IDs remain provenance identifiers and are never
  reply targets.
- Missing/legacy durable evidence produces the existing normal/base behavior
  without invented timestamps or message IDs.
- Self cognition has no behavior or contract change.
- The focused deterministic tests, any explicitly required live-DB test,
  Python checks, and diff review pass, and the completed plan is archived as a
  historical execution record after evidence is recorded below.

## Execution Record

- The user authorized execution after confirming the durable conversation
  history path for `/chat`, unchanged self cognition, and original-message
  targeting for background replies.
- A DeepSeek implementation handoff produced the shared-worktree
  implementation. Its bounded execution window ended before a final handoff;
  the parent retained the changes, completed the integration, and performed
  the final verification.
- A separate read-only DeepSeek review ran in parallel with parent review. It
  identified four issues: equal-timestamp owner anchoring, missing-owner
  fail-open behavior, listen-only trace attachment, and the enrichment-failure
  availability decision. The parent addressed all four. Parent review also
  removed a non-canonical trace-update fallback that used a legacy row alias.
- The implementation commits the durable user receipt before queue admission,
  binds `/chat` interleaving and age checks to the exact owner row and server
  cutoff, leaves self cognition unchanged, and carries the original
  background source message ID separately from the synthetic tool-result
  provenance ID. The exact owner row ID is an internal tie-break binding; the
  semantic filter remains same-platform/channel, role-user, disposition-
  independent ordering.
- Optional embedding enrichment remains best effort after the durable insert;
  enrichment failure is logged while cache invalidation still runs, preserving
  receipt availability and cache consistency.

### Verification evidence

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\background_work\result_source.py src\kazusa_ai_chatbot\brain_service\contracts.py src\kazusa_ai_chatbot\brain_service\intake.py src\kazusa_ai_chatbot\chat_input_queue.py src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\db\__init__.py src\kazusa_ai_chatbot\db\bootstrap.py src\kazusa_ai_chatbot\db\conversation.py src\kazusa_ai_chatbot\db\schemas.py src\kazusa_ai_chatbot\service.py tests\test_background_work_delivery.py tests\test_durable_ingress_interleaving.py tests\test_save_conversation_invalidation.py tests\test_service_background_consolidation.py tests\test_service_event_logging.py tests\test_service_input_queue.py`: passed.
- `venv\Scripts\python.exe -m pytest tests\test_background_work_delivery.py tests\test_durable_ingress_interleaving.py tests\test_save_conversation_invalidation.py tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_event_logging.py -q`: 145 passed.
- `git diff --check`: passed; no whitespace errors.
- `venv\Scripts\python.exe -m pytest tests\test_consolidation_origin_metadata.py -q`: 7 passed and 2 failed in the pre-existing `character_operational_work` fixture path at `src/kazusa_ai_chatbot/consolidation/core.py:168`. That file is outside this plan and unchanged.
- `venv\Scripts\python.exe -m pytest --collect-only -q`: collection remains blocked by four unrelated pre-existing test-environment/fixture errors (`experiments.cognition_core_v2_real_conversation_replay` missing, `asuna.json` profile key `name`, and two missing conversation-history fixture imports); 4,149 tests collected and 1,088 deselected before collection aborted.
- No live LLM or live-DB gate was run because this change does not alter prompts or models and no explicit live-DB environment was provided.

### Residuals

- The unrelated consolidation test failures and full-suite collection errors
  remain outside the scope of this completed bugfix. No in-scope focused test
  failures remain.
