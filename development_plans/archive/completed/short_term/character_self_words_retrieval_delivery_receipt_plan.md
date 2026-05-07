# character self words retrieval and delivery receipt plan

## Summary

- Goal: Let the active character reliably retrieve her own prior conversation words, and record delivered platform message IDs for assistant rows after adapter sends.
- Plan class: large
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`
- Overall cutover strategy: compatible behavior changes at RAG and API boundaries; no historical backfill or database migration.
- Bundling rationale: self-word retrieval and delivery receipts are kept in one plan because they share the same data flow on the inbound side. Native QQ replies to a prior assistant message arrive with only a `platform_message_id`; without a delivery-tracking record the brain cannot resolve that reply target, so the decontextualizer cannot expand "你不是说 X 吗？" with the correct prior assistant text, and RAG cannot reliably build active-character evidence. The two halves are sequenced as independent stages but ship together because reply-context quality is a precondition for self-word retrieval working end-to-end on QQ.
- Highest-risk areas: narrowing this to one incident phrase, making deterministic semantic classifiers over user text, breaking response latency, racing delivery receipts against assistant-row persistence, and leaking cross-channel/private history.
- Acceptance criteria: self-word requests retrieve active-character-authored evidence; active-turn self-hits stay excluded; delivered QQ assistant rows receive platform message IDs; all new behavior is covered by deterministic and live-LLM checks.

## Context

The incident showed a conversation-evidence self-hit: the user challenged the active character's prior wording, but RAG selected the user's current challenge as evidence instead of the active character's earlier message. A previous active-turn exclusion plan already guards current rows when `active_turn_platform_message_ids` is present.

The correct generalized capability is not "fix the TTC/CB1 wording." The target question is:

```text
Can the character look for her own words?
```

This includes user turns such as:

- "你刚才怎么说的？"
- "你之前是不是说过 X？"
- "你不是说 X 吗？"
- "你对那个项目前面怎么评价的？"
- "把你刚才那句原话找出来。"
- "What did you say earlier about X?"
- "You said X before, didn't you?"
- "Find me your exact words from earlier."

These are conversation-history evidence requests scoped to the active character as speaker. The capability covers both Chinese and English forms, both interrogative and imperative, and both topic recall and verbatim quote requests. They are not memory facts, not current episode recall, and not deterministic keyword cases over user input.

The following adjacent surface forms are explicitly OUT OF SCOPE for this plan and must NOT be routed to `speaker=active_character`:

- User asking about the user's own prior words: "我刚才说什么了？" / "What did I say earlier?" — these address the current user as speaker, not the active character.
- Shared-conversation references that ask about both sides: "我们之前讨论过这个吗？" / "Did we talk about this before?" — these are not author-scoped; the existing `speaker=any_speaker` route handles them today and this plan does not change that route.
- Third-party speaker references: "Mom said you mentioned X" — quoted reported speech embedded in the user message is not a self-word retrieval request.

Research performed before this draft:

- `git status --short`
  - Initial planning pass observed unrelated local work outside this plan; the reload audit below supersedes this for current worktree status.
- Direct deterministic probe of `conversation_evidence_agent._worker_context(...)`
  - Without `character_profile`, `speaker=active_character` removes current-user scope but cannot add active-character scope.
  - With `character_profile.global_user_id` and `character_profile.name`, it sets `global_user_id` to the character and `display_name` to the character.
- Baseline deterministic tests:
  - `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_excludes_active_turn_keyword_row tests\test_service_input_queue.py::test_worker_saves_collapsed_messages_before_graph -q`
  - Result: `2 passed`; current active-turn exclusion works in this checkout.
- Baseline adapter tests:
  - `venv\Scripts\python.exe -m pytest tests\test_runtime_adapter_registration.py::test_napcat_hydrates_reply_target_from_platform_get_msg tests\test_runtime_adapter_registration.py::test_napcat_handle_event_forwards_typed_bot_reply_metadata tests\test_runtime_adapter_registration.py::test_napcat_handle_event_sends_reply_as_message_segments tests\test_runtime_adapter_registration.py::test_napcat_runtime_send_message_uses_reply_segments -q`
  - Result: `4 passed`; inbound QQ reply hydration and outbound QQ send IDs already exist at the adapter, but normal `/chat` sends do not report delivery IDs back to the brain.

Reload audit performed against current codebase on 2026-05-07:

- `git status --short`
  - Result: only `development_plans/character_self_words_retrieval_delivery_receipt_plan.md` is untracked.
- Targeted baseline tests rerun:
  - `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_excludes_active_turn_keyword_row tests\test_service_input_queue.py::test_worker_saves_collapsed_messages_before_graph -q`
  - Result: `2 passed`; active-turn exclusion and collapsed-message persistence still pass in the current checkout.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - `initial_persona_state` includes full `character_profile`, but the `stage_1_research(...)` context passed to `call_rag_supervisor(...)` still omits any active-character identity field.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - The initializer prompt already lists `speaker=active_character` in the generic conversation-evidence scope and slot format.
  - It does not contain an explicit self-word or self-quote routing rule/example for "your prior wording", "you said", "你之前说", or "你不是说" style requests.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
  - `_worker_context(...)` already knows how to convert `speaker=active_character` into `global_user_id` and `display_name` when `context["character_profile"]` is present.
  - That path is currently underfed by production RAG context, so the existing code is not sufficient by itself.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
  - `ChatResponse` still has no `delivery_tracking_id`, and no delivery receipt request/response models exist.
- `src/kazusa_ai_chatbot/service.py`
  - `_hydrate_reply_context(...)` delegates only to intake hydration; there is no DB fallback by delivered platform message ID.
  - Route surface is still `/health`, runtime adapter registration/heartbeat, `/chat`, and `/event`; no `/delivery_receipt` endpoint exists.
  - Assistant persistence is called after queue completion and is not given a delivery tracking ID.
- `src/kazusa_ai_chatbot/db/conversation.py` and `src/kazusa_ai_chatbot/db/schemas.py`
  - No delivery receipt update helper, no platform-message-ID lookup helper, and no optional delivery metadata fields exist.
- `src/kazusa_ai_chatbot/db/bootstrap.py`
  - Bootstrap indexes do not include `platform_message_id`.
  - `db/script_operations.py` has a snapshot readability sort that mentions `platform_message_id`; that is not a query index and does not change this plan.
- `src/adapters/napcat_qq_adapter.py`
  - Normal `/chat` response handling calls `send_msg` and checks send status, but does not extract `data.message_id` or report it to the brain.
  - Runtime `send_message(...)` already extracts the send result message ID; keep that contract behavior-preserving.
- Test surface
  - Existing RAG integration tests do not assert that `stage_1_research(...)` passes active-character identity into RAG context.
  - Existing initializer tests assert only the generic `speaker=active_character` availability, not self-word routing.
  - No delivery receipt, delivery tracking, or reply-lookup tests exist.

Current architecture facts:

- `stage_1_research(...)` has access to `state["character_profile"]`, but its RAG context does not pass `character_profile` or a minimal active-character identity block.
- `conversation_evidence_agent` already recognizes `speaker=active_character`, so the target boundary exists but is not production-complete.
- Conversation workers accept `global_user_id` as the concrete speaker filter.
- The initializer prompt currently exposes generic `speaker=active_character` syntax but lacks explicit self-word routing guidance.
- `ChatResponse` does not include a delivery tracking identifier.
- Normal QQ `/chat` handling sends one combined response through `send_msg`; the NapCat response can carry a platform `message_id`, but the current normal chat path discards it.
- Assistant rows are saved by `brain_service.post_turn.save_assistant_message(...)` without `platform_message_id`.
- There is no `/delivery_receipt` endpoint.
- There is no brain-side conversation-history lookup by delivered platform message ID for reply-context hydration.
- Existing bootstrap-created `conversation_history` indexes do not include `platform_message_id`. The receipt update path queries by `delivery_tracking_id` and does not need a new index. The reply-context hydration fallback queries by (`platform`, `platform_channel_id`, `platform_message_id`) on every inbound reply with missing metadata; without an index this becomes a collection scan per message. This plan therefore adds one compound index `(platform, platform_channel_id, platform_message_id)` to `conversation_history` via `db/bootstrap.py`. The index is non-unique because historical rows have an empty `platform_message_id` and because adapters may legitimately reuse IDs across channels.

## Mandatory Skills

- `local-llm-architecture`: load before changing RAG prompt, routing, capability boundaries, LLM-facing context, or response-path call budget.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Resolving only the TTC/CB1 incident phrase is strictly forbidden.
- Do not add deterministic user-text classifiers, regex routers, or hardcoded "你不是说" branches in service, RAG orchestration, adapters, cognition, or dialog.
- Preserve the RAG2 responsibility boundary:
  - initializer/planner emits semantic slots,
  - dispatcher selects capability by prefix,
  - conversation evidence owns conversation-history retrieval,
  - deterministic code owns scope validation, identity handoff, active-turn exclusion, persistence update, and API validation.
- Do not make the initializer generate MongoDB fields, aggregation syntax, query documents, or tool parameters.
- Do not change relevance, cognition, dialog, consolidation, memory evidence, recall, or finalizer prompts unless a listed test proves that file is in the approved change surface.
- Do not remove or weaken active-turn exclusion.
- Do not search cross-channel or cross-platform self words. Self-word retrieval is always scoped to the current platform and channel.
- Do not backfill historical assistant rows.
- Do not add a new MongoDB collection for pending delivery receipts in this plan.
- Do not block normal chat delivery on adapter receipt success.
- Do not change scheduler/runtime proactive sends in this plan.
- Real LLM tests must be run one case at a time with logs inspected before moving to the next case.

## Must Do

- Generalize self-word retrieval through `Conversation-evidence` with `speaker=active_character`.
- Pass active-character identity into RAG runtime context so `speaker=active_character` can become a concrete `global_user_id` filter.
- Add deterministic tests proving active-character-scoped conversation evidence passes the character UUID/display name to the worker.
- Update initializer prompt guidance so active-character prior wording, claims, quotes, and self-authored statements produce `Conversation-evidence` slots scoped as `speaker=active_character`.
- Bump `INITIALIZER_PROMPT_VERSION` after the initializer prompt change.
- Add live LLM initializer tests for self-word retrieval and inspect their logs.
- Add a brain delivery-receipt API contract that lets adapters update an assistant conversation row with delivered platform message ID metadata.
- Add a local delivery tracking ID to chat responses and assistant conversation rows.
- Wire QQ normal `/chat` sends to call the brain delivery-receipt API after `send_msg` succeeds.
- Add a brain-side reply-target fallback lookup by delivered platform message ID for adapters that provide `reply_to_message_id` but not full reply metadata.
- Add deterministic service, DB, and QQ adapter tests for delivery receipts.
- Keep all changes compatible for existing adapters that ignore the new response field.

## Deferred

- Do not implement historical backfill for assistant `platform_message_id`.
- Do not add full multi-platform delivery receipt wiring for Discord normal `/chat` sends in this plan. The core API must be platform-neutral, but QQ wiring is the required adapter integration because the incident and verified message-id source are QQ/NapCat.
- Do not redesign `conversation_history` schema beyond adding optional fields listed in this plan.
- Do not create a separate self-quote RAG agent.
- Do not add over-retrieval or query retries solely to recover rows displaced by active-turn exclusion.
- Do not change vector-search index definitions.
- Do not change `conversation_history` embedding source semantics.
- Do not change how scheduled/proactive messages are persisted or delivered.
- Do not alter user-message persistence timing.

## Cutover Policy

| Area | Policy | Instruction |
|---|---|---|
| RAG context character identity | compatible | Add a `character_profile` dict containing exactly `global_user_id` and `name` to RAG context; do not pass any other persona fields through this path. |
| Conversation evidence active-character scope | compatible | Use the existing `speaker=active_character` contract; make it work reliably with concrete identity. |
| Initializer prompt strategy | compatible | Update prompt and bump initializer prompt cache version so stale strategies are not reused. |
| Active-turn exclusion | compatible | Preserve current behavior and add observability only. |
| ChatResponse schema | compatible | Add optional delivery-tracking field with default empty value; adapters that ignore it continue working. |
| Assistant row persistence | compatible | Add optional local delivery tracking and delivered platform ID fields; old rows remain valid. |
| Delivery receipt endpoint | compatible | Add a new endpoint; no existing caller changes are required except QQ adapter wiring. |
| Reply-context hydration fallback | compatible | If the adapter already provides full reply metadata, keep it; otherwise use DB lookup by platform message ID when available. |
| QQ adapter normal chat send | compatible | After successful `send_msg`, best-effort call `/delivery_receipt`; sending still succeeds if receipt reporting fails. |
| Database migration | compatible | No backfill, collection migration, or destructive DB change. |

## Agent Autonomy Boundaries

- The implementation agent may choose local helper names only when they preserve the contracts in this plan.
- The agent must not introduce alternate retrieval architecture, a new self-quote agent, a new pending-receipts collection, or new semantic code-side classification.
- The agent must treat edits outside the listed change surface as out of scope unless the plan is updated before implementation.
- If an existing helper already provides the needed persistence or adapter behavior, reuse or extend it instead of duplicating logic.
- If live LLM tests show the prompt route is still brittle, improve the initializer prompt within the approved self-word retrieval contract; do not add deterministic repair outside RAG.
- If delivery receipt races are observed in tests, solve them with the bounded adapter retry specified under "Implementation Order". Do not block user-visible send on receipt success and do not add a service-side idempotency store, pending-receipts collection, or queue.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

When a user asks about the active character's own prior words, RAG produces conversation evidence from assistant-authored rows:

```text
user asks: "你之前是不是说过 X？"
  -> decontextualizer preserves the self-word question
  -> initializer emits Conversation-evidence speaker=active_character
  -> conversation_evidence_agent converts active_character to character global_user_id
  -> keyword/search/filter worker searches only active-character rows in the current platform/channel by default
  -> active-turn user rows remain excluded
  -> cognition sees evidence containing the character's prior wording or an unresolved evidence state
```

When the QQ adapter delivers a normal chat response:

```text
brain returns ChatResponse(messages=[...], delivery_tracking_id="...")
  -> QQ adapter sends combined text via send_msg
  -> NapCat returns platform message_id
  -> QQ adapter POSTs /delivery_receipt with delivery_tracking_id and platform_message_id
  -> brain updates the assistant conversation_history row
  -> future native replies to that assistant message can be resolved by platform id
```

When a later inbound message carries only a native reply ID:

```text
adapter sends message_envelope.reply.platform_message_id
  -> brain reply-context hydration checks adapter-provided metadata first
  -> if author/excerpt metadata is missing, brain looks up conversation_history by platform/channel/message ID
  -> reply_context gains platform_user_id/display_name/excerpt when the delivered assistant row exists
  -> decontextualizer and RAG receive better reply-target context
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| General P1 capability | Use existing `Conversation-evidence` with `speaker=active_character` | The capability already owns conversation-history evidence and author scope; no new agent is needed. |
| Self-word semantics | Express in initializer prompt as active-character-authored prior wording/claims/statements | LLM owns semantic interpretation; deterministic code should not classify natural-language user text. |
| Character identity handoff | Pass active-character identity into RAG context | Conversation workers need a concrete `global_user_id` to scope rows. |
| Worker filtering | Keep worker argument generation local to existing workers | The initializer should not generate low-level query parameters. |
| Prompt cache | Bump `INITIALIZER_PROMPT_VERSION` | Existing durable initializer strategy cache could otherwise replay old `any_speaker` slots. |
| Delivery tracking cardinality | Track one normal chat response row per `/chat` response | Current QQ adapter sends one combined message, matching current assistant persistence which stores joined `final_dialog`. |
| Delivery receipt target | Update by generated `delivery_tracking_id` (`uuid.uuid4().hex`), not by body text or timestamp | Avoids ambiguous matching; receipt update remains O(1) by tracking ID. |
| Receipt race | Use best-effort adapter retry on `not_found` rather than blocking chat response on DB persistence | Preserves response latency and avoids adding a pending-receipts collection. |
| Reply metadata fallback | Use DB lookup only to fill missing reply metadata | Adapter-supplied platform metadata remains authoritative when present. |
| QQ adapter scope | Wire QQ normal chat delivery receipts now; leave Discord normal chat receipts deferred | QQ has verified one-message send ID in the incident path. Discord chunks normal chat output and needs a separate cardinality plan. |

## Contracts And Data Shapes

### RAG Context

Add one prompt-safe active-character identity field to RAG context with exactly this shape:

```python
"character_profile": {
    "global_user_id": str,
    "name": str,
}
```

These two keys are the entire `character_profile` payload for the RAG context. Do not expose boundary profile, mood, backstory, or other persona fields to RAG workers for this fix, even if they are available in `state["character_profile"]` upstream.

### Conversation Evidence Slot

Initializer output for self-word retrieval must use:

```text
Conversation-evidence: retrieve <prior active-character wording/topic/exact phrase> speaker=active_character
```

Use `speaker=active_character` for active-character-authored statements even when the user says "you" instead of the character name.

### Prompt-Rule Specification

The implementation agent must add one new self-word routing rule and one new example block to the initializer prompt at `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`. The rule must satisfy these contract points:

- It instructs the model that user requests asking about the active character's own prior wording, claims, quotes, opinions, or self-authored statements are conversation-history evidence requests.
- It directs those requests to a `Conversation-evidence:` slot scoped as `speaker=active_character`.
- It clarifies that the user's pronoun "you" (or "你") in such a question refers to the active character, even when the character name is not used.
- It explicitly excludes user-self questions ("我刚才说什么了？" / "What did I say earlier?") from this route.
- It does not add deterministic patterns or hardcoded surface forms; the rule is semantic.

The agent has discretion over the exact prose of the prompt rule, but the slot output for the following canonical inputs must match exactly when the rule is applied. These mappings are the contract; the prompt-render and live-LLM tests use them.

| Input (user message) | Required slot in initializer output |
|---|---|
| `你之前是不是说过那个项目要延期？` | `Conversation-evidence: retrieve prior active-character claim about the project being delayed speaker=active_character` |
| `你刚才怎么说的？` | `Conversation-evidence: retrieve active-character most recent prior utterance speaker=active_character` |
| `把你刚才那句原话找出来。` | `Conversation-evidence: retrieve active-character verbatim prior utterance speaker=active_character` |
| `What did you say earlier about the deadline?` | `Conversation-evidence: retrieve prior active-character statement about the deadline speaker=active_character` |
| `You said X before, didn't you?` | `Conversation-evidence: retrieve prior active-character claim about X speaker=active_character` |

The retrieval target wording (`retrieve <...>`) is paraphrasable, but every required slot must:

1. Begin with the literal token `Conversation-evidence:`.
2. Contain the literal substring `speaker=active_character`.
3. Reference the topic, phrase, or "prior utterance" semantically derived from the user input.
4. Not contain `speaker=any_speaker` or `speaker=current_user` for self-word inputs.

### Self-Word Test Corpus

Deterministic prompt-render tests and live-LLM initializer tests must both exercise the following corpus. The corpus is fixed by this plan; the implementation agent may not silently substitute different strings.

Positive cases (initializer must produce a `Conversation-evidence:` slot with `speaker=active_character` as the primary or only conversation-evidence slot):

1. `你之前是不是说过那个项目要延期？`
2. `你刚才怎么说的？`
3. `把你刚才那句原话找出来。`
4. `What did you say earlier about the deadline?`
5. `You said X before, didn't you?`

Negative cases (initializer must NOT produce `speaker=active_character` for these inputs):

6. `我刚才说什么了？` — user asking about user's own prior words; conversation-evidence may be emitted but with `speaker=current_user`, not `speaker=active_character`.
7. `我们之前讨论过这个吗？` — shared-conversation reference; existing `speaker=any_speaker` route applies and must not be replaced by `speaker=active_character`.

Deterministic prompt-render tests assert the parsed slot structure for cases 1–7 against a captured initializer output (cases 1–5 must contain `speaker=active_character`; cases 6–7 must not). Live-LLM tests cover at minimum cases 1, 3, 4 (positive) and case 6 (negative); the agent may add more live cases but must run each one at a time and inspect logs.

### ChatResponse

Extend `ChatResponse` with:

```python
delivery_tracking_id: str = ""
```

The value is non-empty only when `messages` is non-empty and a single assistant conversation row is expected to be persisted.

### Conversation History Row

Assistant rows may include:

```python
{
    "delivery_tracking_id": str,
    "platform_message_id": str,  # updated after adapter delivery when available
    "delivery_status": "pending" | "delivered",
    "delivered_at": str,
    "delivery_adapter": str,
}
```

`platform_message_id` remains optional for historical rows and for adapters not yet wired.

### Delivery Receipt API

Add a new request model:

```python
class DeliveryReceiptRequest(BaseModel):
    platform: str
    platform_channel_id: str = ""
    delivery_tracking_id: str
    platform_message_id: str
    delivered_at: str = ""
    adapter: str = ""
```

Add a response model:

```python
class DeliveryReceiptResponse(BaseModel):
    status: str  # "updated" | "not_found"
    updated: bool = False
```

Endpoint:

```text
POST /delivery_receipt
```

Behavior:

- Validate non-empty `platform`, `delivery_tracking_id`, and `platform_message_id`.
- Update only assistant rows matching `platform`, optional `platform_channel_id`, and `delivery_tracking_id`.
- Set `platform_message_id`, `delivery_status="delivered"`, `delivered_at`, and `delivery_adapter`.
- Return `not_found` without inserting a new row if no match exists.
- Do not recompute embedding; platform delivery metadata does not affect retrieval text.
- Do not emit Cache2 invalidation. Delivery metadata is not part of embedding text and does not affect retrieval results. Add a deterministic test asserting the receipt-update path does not call the embedding function and does not invoke any Cache2 invalidation helper.

### Reply Target Lookup

Add a DB helper:

```python
async def get_conversation_by_platform_message_id(
    *,
    platform: str,
    platform_channel_id: str,
    platform_message_id: str,
) -> ConversationMessageDoc | None:
    ...
```

Use it only as a hydration fallback when `reply_context.reply_to_message_id` is present but at least one of `reply_to_platform_user_id`, `reply_to_display_name`, or `reply_excerpt` is missing.

Do not use body text, timestamp proximity, display name, or fuzzy matching for this lookup.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Pass minimal active-character identity into `call_rag_supervisor(...)` context.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - Update initializer prompt self-word rule and examples.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - Bump `INITIALIZER_PROMPT_VERSION`.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
  - Keep existing active-character scoping; do not change the conversion logic in `_worker_context(...)`.
  - When `speaker=active_character` is requested but `character_profile` is absent or missing required keys, emit a single `WARN` log with the stable marker `conversation_evidence: speaker=active_character requested without character_profile` and proceed with current degrade-behavior. Do not raise.
  - Optionally emit a `DEBUG` log on successful resolution with the resolved `global_user_id`; do not log the user message or the slot text.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
  - Add `delivery_tracking_id` to `ChatResponse`.
  - Add delivery receipt request/response models.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
  - Save assistant rows with `delivery_tracking_id` and `delivery_status="pending"` when supplied.
- `src/kazusa_ai_chatbot/service.py`
  - Generate delivery tracking ID for non-empty responses.
  - Include it in `ChatResponse`.
  - Pass it into assistant persistence state.
  - Add `/delivery_receipt` endpoint.
  - Extend `_hydrate_reply_context(...)` to fill missing reply metadata from DB lookup by delivered platform message ID.
- `src/kazusa_ai_chatbot/db/conversation.py`
  - Add `apply_assistant_delivery_receipt(...)` helper.
  - Add `get_conversation_by_platform_message_id(...)` helper.
- `src/kazusa_ai_chatbot/db/schemas.py`
  - Add optional delivery metadata fields to `ConversationMessageDoc`.
- `src/kazusa_ai_chatbot/db/bootstrap.py`
  - Add a non-unique compound index on (`platform`, `platform_channel_id`, `platform_message_id`) for `conversation_history` so the reply-context fallback lookup is O(log n).
- `src/adapters/napcat_qq_adapter.py`
  - After normal chat `send_msg` succeeds and `delivery_tracking_id` is non-empty, POST a best-effort delivery receipt to the brain.
- Tests listed below.

### Keep

- Relevance behavior.
- Decontextualizer behavior.
- Cognition/dialog prompts.
- Conversation worker tool contracts.
- Existing active-turn exclusion behavior.
- Existing QQ runtime scheduler `send_message(...)` contract — unchanged. Do not have the scheduler path post delivery receipts in this plan.

### Create

- No new production module. All delivery-receipt and reply-lookup helpers are added to the existing files listed under "Modify".
- Update `src/kazusa_ai_chatbot/db/bootstrap.py` to ensure a non-unique compound index on (`platform`, `platform_channel_id`, `platform_message_id`) is created on `conversation_history` at startup.

## Implementation Order

1. Add failing deterministic RAG tests.
   - In `tests/test_rag_phase3_capability_agents.py`, add a test that `speaker=active_character` passes character `global_user_id` and display name to the selected worker.
   - In `tests/test_persona_supervisor2_rag2_integration.py` or adjacent integration test, add a patched `call_rag_supervisor` capture proving `stage_1_research` passes minimal active-character identity.
   - The current integration test already captures RAG context; extend it or add an adjacent test so the failing-before assertion is specifically the missing active-character identity, not a broad payload rewrite.
   - Run them and record expected failure before implementation.
2. Implement RAG identity handoff.
   - Pass `character_profile = {"global_user_id": ..., "name": ...}` from persona state into the RAG context dict; do not pass any other persona fields through this path.
   - Ensure `conversation_evidence_agent` keeps current behavior when identity is absent and emits a single `WARN` log line with the stable marker string `conversation_evidence: speaker=active_character requested without character_profile` so production gaps are observable.
3. Run deterministic RAG tests.
4. Add initializer prompt tests.
   - Update prompt-render tests in `tests/test_rag_initializer_cache2.py` to assert that the rendered initializer prompt contains the new self-word routing rule and an example whose input/output match a row from the Self-Word Test Corpus.
   - Add deterministic slot-parsing tests over corpus cases 1–7: positive cases (1–5) must produce a `Conversation-evidence:` slot containing `speaker=active_character`; negative cases (6–7) must not. These tests use a stubbed initializer that returns a captured fixture — they verify parsing and assertion logic, not the live LLM.
   - Add or update prompt version tests to expect the new prompt version.
5. Implement initializer prompt update and prompt version bump.
6. Run deterministic initializer tests.
7. Add live LLM initializer tests.
   - Add at minimum the four required cases from the Self-Word Test Corpus (cases 1, 3, 4, 6) as separate test functions in `tests/test_rag_phase3_initializer_live_llm.py`.
   - Run each live LLM test one at a time with `-q -s`; inspect trace output and verify the pass criteria in Verification > Live LLM Tests before moving to the next case.
8. Add failing delivery receipt and reply-lookup DB/service tests.
   - DB helper updates assistant row by `delivery_tracking_id`.
   - DB helper update path does not call the embedding function and does not invoke Cache2 invalidation.
   - DB helper reads a row by (`platform`, `platform_channel_id`, `platform_message_id`).
   - `db/bootstrap.py` creates the non-unique compound index `(platform, platform_channel_id, platform_message_id)` on `conversation_history` at startup.
   - `/delivery_receipt` returns `status="updated"` for a match and `status="not_found"` for a miss.
   - `ChatResponse` includes a non-empty `delivery_tracking_id` when `messages` is non-empty.
   - `ChatResponse` `delivery_tracking_id` is empty when `messages` is empty.
   - Assistant save receives and persists `delivery_tracking_id` and `delivery_status="pending"`.
   - Service reply-context hydration fills missing reply author/excerpt fields from a delivered assistant row but leaves adapter-supplied metadata untouched when present.
9. Implement delivery receipt models, DB helper, endpoint, service response wiring, and the `db/bootstrap.py` index addition.
10. Run delivery receipt DB/service tests.
11. Add failing QQ adapter receipt tests.
    - Extend `test_napcat_handle_event_sends_reply_as_message_segments` or add an adjacent test proving successful `send_msg` triggers a brain `/delivery_receipt` call with the correct `platform_message_id` and `delivery_tracking_id`.
    - Test: when the first receipt POST returns `status="not_found"`, the adapter retries and the second attempt's success (`status="updated"`) ends the retry loop. Total attempt count: 2.
    - Test: when all three attempts (initial + 2 retries at 250ms and 750ms) return `status="not_found"`, the adapter stops and emits a single WARN log; user-visible send still reports success.
    - Test: when the receipt POST raises a transport error or 5xx, the adapter does not retry and emits a single WARN log; user-visible send still reports success.
    - Test: no receipt call is made when `delivery_tracking_id` is empty.
    - Test: no receipt call is made when `send_msg` fails.
12. Implement QQ adapter receipt posting.
    - On HTTP success with `status="updated"`: stop.
    - On HTTP success with `status="not_found"`: retry at most 2 additional times with delays of 250ms and then 750ms. If the third attempt still returns `not_found`, log at `WARN` with the `delivery_tracking_id` and `platform_message_id` and stop.
    - On HTTP transport failure (timeout, connection error, 5xx): log at `WARN` and stop without retrying. Adapter send must remain successful from the user's perspective regardless of receipt outcome.
    - Do not retry when `delivery_tracking_id` is empty; skip the receipt call entirely.
13. Run QQ adapter tests.
14. Run full targeted verification suite in this plan.
15. Record execution evidence and update checklist stage sign-offs one at a time.

## Progress Checklist

- [x] Stage 1 - RAG active-character scope contract
  - Covers: deterministic tests and implementation for RAG context identity handoff.
  - Verify: targeted RAG deterministic tests pass.
  - Evidence: failing-before targeted tests failed on missing active-character identity/warn behavior; passing-after targeted tests `3 passed`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex / 2026-05-07`.
- [x] Stage 2 - Initializer self-word prompt contract
  - Covers: prompt tests, prompt update, cache version bump, and live LLM initializer tests.
  - Verify: deterministic prompt tests pass; each live LLM test is run individually and inspected.
  - Evidence: `INITIALIZER_PROMPT_VERSION = initializer_prompt:v17`; deterministic initializer tests `25 passed`; four live LLM initializer cases passed one at a time with traces inspected.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex / 2026-05-07`.
- [x] Stage 3 - Brain delivery receipt contract
  - Covers: models, response tracking ID, assistant persistence metadata, DB helper, reply-target lookup, endpoint.
  - Verify: DB/service delivery tests pass.
  - Evidence: targeted DB/service/post-turn tests `10 passed`; DB boundary tests `10 passed`.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex / 2026-05-07`.
- [x] Stage 4 - QQ adapter delivery receipt wiring
  - Covers: normal chat send receipt reporting and bounded retry behavior.
  - Verify: QQ adapter receipt tests pass.
  - Evidence: targeted QQ receipt tests `6 passed`; full adapter registration suite `18 passed`.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex / 2026-05-07`.
- [x] Stage 5 - Final verification and handoff
  - Covers: py_compile, targeted deterministic suite, live LLM trace review, and final execution record.
  - Verify: all commands in Verification pass or are explicitly blocked with reason.
  - Evidence: static compile passed; targeted deterministic suites passed; live traces inspected; manual smoke not run because no local QQ service/adapter smoke environment was started for this implementation pass.
  - Handoff: implementation complete.
  - Sign-off: `Codex / 2026-05-07`.

## Verification

### Static Checks

```powershell
venv\Scripts\python.exe -m py_compile `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py `
  src\kazusa_ai_chatbot\rag\cache2_policy.py `
  src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py `
  src\kazusa_ai_chatbot\brain_service\contracts.py `
  src\kazusa_ai_chatbot\brain_service\post_turn.py `
  src\kazusa_ai_chatbot\service.py `
  src\kazusa_ai_chatbot\db\conversation.py `
  src\kazusa_ai_chatbot\db\schemas.py `
  src\kazusa_ai_chatbot\db\bootstrap.py `
  src\adapters\napcat_qq_adapter.py
```

### Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py -q
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_initializer_cache2.py -q
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_rag2_integration.py -q
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py tests\test_save_conversation_invalidation.py -q
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_runtime_adapter_registration.py -q
```

### Live LLM Tests

Run each case one at a time with `-q -s` and inspect the trace before moving to the next case. Each case in the test corpus (Contracts > Self-Word Test Corpus) gets its own test function. The minimum required live-LLM cases are:

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_initializer_live_llm.py::test_initializer_live_self_words_chinese_claim_recall -q -s
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_initializer_live_llm.py::test_initializer_live_self_words_chinese_verbatim_quote -q -s
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_initializer_live_llm.py::test_initializer_live_self_words_english_topic_recall -q -s
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_initializer_live_llm.py::test_initializer_live_user_self_question_does_not_route_active_character -q -s
```

Mapping to the corpus: the four required live cases above cover corpus cases 1, 3, 4 (positive) and 6 (negative). The agent may add live coverage of cases 2, 5, and 7 if budget allows; each must be run individually.

Pass criteria (all must hold for every positive case):

- Parsed initializer output contains at least one slot whose first token is `Conversation-evidence:`.
- The first (or only) `Conversation-evidence:` slot contains the literal substring `speaker=active_character`.
- No `Conversation-evidence:` slot for a positive case contains `speaker=any_speaker` or `speaker=current_user`.
- No memory, recall, person-context, or web-search slot is emitted as a substitute for the conversation-evidence slot for a positive case.

Pass criteria for the negative case (corpus 6):

- The initializer output does NOT contain any slot with `speaker=active_character`.
- A `Conversation-evidence:` slot may be emitted, but if present it must use `speaker=current_user` or `speaker=any_speaker`.

If any pass criterion fails, the agent must adjust the initializer prompt within the contract in the Prompt-Rule Specification section and rerun. The agent may not weaken the pass criteria, may not add deterministic post-processing of initializer output, and may not skip cases.

### Manual Smoke

After implementation, run a local QQ/debug style smoke if the service and adapter are available:

- Send a message that causes Kazusa to reply.
- Confirm the assistant row initially has `delivery_tracking_id`.
- Confirm QQ send returns a platform message ID.
- Confirm `/delivery_receipt` updates the assistant row with `platform_message_id`.
- Ask a self-word question about the prior reply.
- Confirm RAG evidence cites the assistant row, not the user's current question.

## Acceptance Criteria

This plan is complete when:

- The active character can retrieve her own prior words through `Conversation-evidence` scoped as `speaker=active_character`.
- The incident class cannot self-hit the current user message when active-turn IDs are present.
- Active-character scoped retrieval applies a concrete character `global_user_id` worker filter.
- Initializer live LLM tests route self-word questions to `speaker=active_character`.
- The initializer prompt cache version is bumped.
- QQ normal chat sends report delivered platform message IDs back to the brain through `/delivery_receipt`.
- New assistant rows can be updated with delivered `platform_message_id` without historical backfill.
- Existing adapters that ignore `delivery_tracking_id` continue to work.
- All listed deterministic tests pass.
- Live LLM traces are inspected and judged acceptable.

## Data Migration

No migration is required.

Historical assistant rows without `platform_message_id`, `delivery_tracking_id`, or delivery metadata remain valid. This plan intentionally does not backfill them.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| The initializer still emits `any_speaker` for self-word requests | Explicit self-word prompt rule with canonical input→slot mappings, prompt-render tests over the full Self-Word Test Corpus (cases 1–7), prompt version bump, and live LLM tests over corpus cases 1, 3, 4, 6 | Live initializer tests inspect parsed slots against pass criteria |
| The initializer over-routes user-self questions to `speaker=active_character` | Negative cases (corpus 6, 7) included in deterministic and live LLM tests | Live test for case 6 asserts the absence of `speaker=active_character` |
| Character identity leaks too much persona context into RAG | Pass only `{"global_user_id", "name"}` in `character_profile` | Deterministic test asserts the RAG context dict's `character_profile` keys are exactly `{"global_user_id", "name"}` |
| Active-character scope silently degrades if identity is missing | Emit a `WARN` with the stable marker `conversation_evidence: speaker=active_character requested without character_profile`, and add an integration test proving production context supplies identity | RAG integration test plus a unit test that captures the WARN log when identity is absent |
| Delivery receipt races assistant-row persistence | Bounded QQ adapter retry: at most 2 retries at 250ms and 750ms after the initial attempt; user-visible send is never blocked | Adapter test with first `not_found`, second `updated`; second test where all three attempts return `not_found` and the adapter logs WARN |
| Delivery receipt updates wrong row | Update only by generated `delivery_tracking_id` (`uuid.uuid4().hex`) plus platform and optional `platform_channel_id` | DB helper tests |
| Reply fallback hydrates wrong row | Lookup only by exact (`platform`, `platform_channel_id`, `platform_message_id`); no body or timestamp fallback | DB lookup tests |
| Reply fallback degrades to a collection scan | Add a non-unique compound index `(platform, platform_channel_id, platform_message_id)` to `conversation_history` in `db/bootstrap.py` | Deterministic bootstrap test asserts the index is created at startup |
| Receipt metadata changes retrieval embeddings | Receipt update path must not call the embedding function nor any Cache2 invalidation helper | DB helper test asserts neither is called |
| Discord normal chat has multiple chunk IDs | Defer Discord wiring; keep core API platform-neutral | Deferred section and no Discord adapter changes |

## LLM Call And Context Budget

Before:

- RAG initializer: one response-path LLM call when initializer cache misses.
- Conversation evidence: existing worker selector/generator/judge calls as today.

After:

- No new response-path LLM calls.
- RAG initializer prompt gains a small self-word routing rule and one or two examples.
- RAG context gains only active-character identity (`global_user_id`, `name`), well below the default 50k-token cap.
- Delivery receipt path adds no LLM calls.

Latency impact:

- RAG identity handoff: negligible.
- Prompt text increase: negligible.
- QQ delivery receipt: post-send best-effort HTTP call from adapter to brain; it does not block brain response generation and should not block user-visible send success.

## Operational Steps

- Deploy brain and QQ adapter together for delivery receipts.
- No database backfill is required.
- No manual cache purge is required because `INITIALIZER_PROMPT_VERSION` changes the initializer strategy cache key.
- Monitor logs for:
  - active-character scope resolution in conversation evidence,
  - active-turn rows excluded,
  - delivery receipt `updated` vs `not_found`,
  - QQ adapter receipt failures.

## Execution Evidence

Pre-plan evidence:

- Direct probe of `_worker_context(...)`: active-character scope needs character identity in context.
- Active-turn exclusion targeted tests: `2 passed`.
- QQ reply/send baseline tests: `4 passed`.
- Reload audit on 2026-05-07: current code still lacks RAG identity handoff, self-word initializer routing guidance, delivery tracking contract, `/delivery_receipt`, DB receipt/update helpers, reply DB fallback, and QQ normal-chat receipt reporting.
- Reload targeted baseline tests on 2026-05-07: `2 passed`.

Implementation evidence must be appended here during execution:

- Failing-before deterministic tests:
  - Stage 1 targeted tests failed before implementation because production RAG context did not include active-character identity and the missing-identity warning did not exist.
  - Stage 2 targeted prompt tests failed before implementation because the prompt lacked the self-word routing rule/example and still used prompt cache version `v16`.
  - Stage 3 targeted DB/service tests failed before implementation because delivery receipt contracts, tracking IDs, DB helpers, reply fallback, and bootstrap index were absent.
  - Stage 4 targeted QQ adapter tests failed before implementation because successful normal-chat `send_msg` did not post `/delivery_receipt` and no receipt retry helper existed.
- Passing-after deterministic tests:
  - `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_active_character_scope_uses_character_identity tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_active_character_scope_warns_without_identity tests\test_persona_supervisor2_rag2_integration.py::test_stage_1_research_calls_rag2_and_projects_payload -q` -> `3 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_rag_initializer_cache2.py -q` -> `25 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py -q` -> `44 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_rag2_integration.py -q` -> `4 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py tests\test_save_conversation_invalidation.py tests\test_bot_side_addressing.py -q` -> `20 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_runtime_adapter_registration.py -q` -> `18 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_db.py::test_apply_assistant_delivery_receipt_updates_tracking_row tests\test_db.py::test_apply_assistant_delivery_receipt_has_no_embedding_or_cache_side_effects tests\test_db.py::test_get_conversation_by_platform_message_id_uses_exact_scope tests\test_db.py::test_db_bootstrap_creates_platform_message_lookup_index tests\test_db_public_boundary.py tests\test_script_db_boundary.py -q` -> `10 passed`.
- Live LLM trace paths and judgment:
  - `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__self_words_chinese_claim_recall.json`: slot `Conversation-evidence: retrieve prior active-character claim about the project being delayed speaker=active_character`; acceptable.
  - `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__self_words_chinese_verbatim_quote.json`: slot `Conversation-evidence: retrieve exact phrase of the active character's most recent message speaker=active_character`; acceptable.
  - `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__self_words_english_topic_recall.json`: slot `Conversation-evidence: retrieve prior active-character claim about the deadline speaker=active_character`; acceptable.
  - `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__user_self_question_not_active_character.json`: slot `Conversation-evidence: retrieve recent messages speaker=current_user`; acceptable negative case.
- Static check output:
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py src\kazusa_ai_chatbot\rag\cache2_policy.py src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py src\kazusa_ai_chatbot\brain_service\contracts.py src\kazusa_ai_chatbot\brain_service\post_turn.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\db\conversation.py src\kazusa_ai_chatbot\db\schemas.py src\kazusa_ai_chatbot\db\bootstrap.py src\adapters\napcat_qq_adapter.py tests\test_rag_phase3_initializer_live_llm.py` -> passed.
  - `git diff --check` -> no whitespace errors; Git reported only expected LF-to-CRLF working-copy warnings.
- Manual smoke result:
  - Not run in this implementation pass; no local QQ service/adapter smoke environment was started. Deterministic adapter tests cover successful receipt posting, retry-on-`not_found`, final `not_found` warning, transport failure, empty tracking ID skip, and send-failure skip.

## Glossary

- Self-word retrieval: retrieving the active character's own prior utterances from conversation history.
- Active-character scope: conversation evidence where the speaker is the active character, expressed as `speaker=active_character` and executed as a character `global_user_id` filter.
- Active-turn self-hit: retrieving the current user message as evidence for the same turn.
- Delivery tracking ID: brain-generated local identifier that lets an adapter report which assistant conversation row was delivered as which platform message.
- Delivery receipt: adapter-to-brain report containing the delivered platform message ID for a previously returned chat response.
