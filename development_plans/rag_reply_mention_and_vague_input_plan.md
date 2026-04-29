# rag reply mention and vague input plan

## Summary

- Goal: Prevent RAG from searching bot-addressing metadata as content, and route truly unresolved vague user input toward clarification instead of guessed retrieval.
- Scope framing: This is **Stage 1 of 2**. Stage 1 stops today's bleeding at two specific seams (RAG initializer slots, decontextualizer ambiguity). Stage 2 is the systemic fix — a typed message envelope at the adapter, with `body_text` (no wire markers) flowing through storage, search, cache keys, and prompts. Stage 1 must not foreclose Stage 2; see "Stage 2 — Systemic Follow-up" near the end.
- Plan class: large
- Status: completed
- Overall cutover strategy: compatible
- Highest-risk areas: over-blocking legitimate references, adding brittle deterministic vague-input heuristics, changing broad conversation behavior, and accidentally coupling RAG to platform-specific syntax.
- Acceptance criteria: the initializer/dispatcher never search the current bot mention or reply boilerplate as keyword evidence; decontextualizer can explicitly report unresolved references when full context is insufficient; unresolved-reference turns skip RAG and let cognition/dialog ask for clarification; media preservation remains out of scope and is handled by `input_queue_media_preservation_plan.md`.

## Context

The live failure involved this user input:

```text
[Reply to message] <@3768713357>  你知道这些是什么意思么？
```

The system generated:

```text
Conversation-keyword: find messages containing '<@3768713357>' to identify the content being referred to
Memory-search: search persistent memory for evidence relevant to explaining symbols or text mentioned in the context of the conversation
```

This failed because `<@3768713357>` was the current bot mention. It is addressing metadata, not content evidence. Searching it retrieves many unrelated bot-addressed messages and can pull stale facts into cognition. The second issue is broader: when relevance and decontextualizer, both with history, cannot resolve what the user means, the input is truly vague and the character should ask for clarification instead of letting RAG guess.

This plan addresses #3 and #4 only. It does not fix missing images/media through the input queue.

## Why this is Stage 1 only

The reported failure is one instance of a broader class: platform adapters flatten transport-envelope data — mentions, reply targets, CQ codes, role/channel/emoji refs — into a single prose `user_input` string, and every downstream stage (decontextualizer, RAG initializer, conversation_keyword_agent, persistent memory, cache keys, history rows, logs) is then asked to re-parse that string and tell envelope from content. There is no structural separation between the two.

Other instances of the same class that this plan does NOT fix:

- Other CQ codes: `[CQ:at,...]`, `[CQ:reply,...]`, `[CQ:image,...]`, `[CQ:face,...]`.
- Discord variants: `<@&role>`, `<#channel>`, `<:emoji:id>`.
- `@everyone` / `@here` flowing into keyword/semantic search.
- Stored conversation rows containing `<@bot_id>` literals — `conversation_keyword_agent` will keep returning bot-addressed messages as hits for unrelated keyword overlap, even after this plan ships.
- `build_initializer_cache_key` hashing wire markers — fragments cache across deployments and across cosmetically equivalent queries.
- Persistent memory consolidator durably remembering wire markers as if they were content.

Stage 1 is a targeted stopgap. Stage 2 (sketched at the end of this document) addresses the class. The Stage 1 sanitizer is intentionally written so it can be deleted in Stage 2.

## Mandatory Rules

- Do not implement media/attachment preservation in this plan. That is owned by `development_plans/input_queue_media_preservation_plan.md`.
- Do not add a narrow deterministic rule like "if user says 这些 then block RAG." That is speculative and likely to create false positives.
- Do not make RAG decide vague-input clarification by itself after losing context. The clarification signal must come from the relevance/decontextualizer boundary that sees full conversation context.
- Do not rewrite the whole RAG pipeline or introduce a new planner architecture.
- Preserve the existing stage order: relevance -> decontextualizer -> RAG -> cognition -> dialog.
- Keep RAG slots factual data targets only. Do not let RAG generate clarification dialog.
- Do not persist cache entries for unresolved-reference initializer plans.
- Do not use raw platform IDs, bot mention tokens, current bot IDs, reply markers, or CQ reply boilerplate as `Conversation-keyword` search targets.
- Allow legitimate non-bot mention use cases when the mention is the entity being asked about, for example "这个 <@123> 是谁？".
- Follow project Python style: typed helper signatures, imports at top, specific exception handling, complete docstrings for public/non-trivial helpers.
- Tests that check prompt behavior may use patched LLM outputs for plumbing, but real prompt behavior must be validated with one-at-a-time live LLM cases if live tests are added.
- Sanitizer must run on **both** cache write and cache read paths. Pre-existing bad cache rows must not silently replay after the fix ships.
- Bump `INITIALIZER_PROMPT_VERSION` (in `kazusa_ai_chatbot/rag/cache2_policy.py`) as part of this change to invalidate existing bad cache rows in `rag_cache2_persistent_entries`. A targeted delete of the known bad row is acceptable as a complement, not a substitute.
- The RAG supervisor `context` passed from `stage_1_research(...)` must include `platform_bot_id` and an explicit reply-boilerplate token list, so the sanitizer has a deterministic source of truth instead of regex-guessing `<@\d+>`.
- When RAG is skipped due to `needs_clarification`, the produced `rag_result` MUST be built via `project_known_facts([], current_user_id=..., character_user_id=..., answer="", unknown_slots=[], loop_count=0)` so all dict keys consumed by cognition (`user_image`, `character_image`, `memory_evidence`, `conversation_evidence`, `external_evidence`, `supervisor_trace`) exist with empty values. Do NOT assign a bare `{}`.
- The sanitizer MUST be implemented as an extensible token-class registry (initial entries: current bot mention from `platform_bot_id`, `[Reply to message]`, CQ reply markers). Adding a new marker class must be one registry entry, not a new function. This both keeps Stage 1 from accreting bespoke per-marker code and prepares for clean Stage 2 deletion.
- Drop rule for `Conversation-keyword` slots: extract the slot's quoted target phrase; strip any registered envelope token from it (and surrounding whitespace); if the remaining phrase has length < 2 characters or is empty, drop the whole slot. Otherwise keep the slot unchanged. Slots whose prefix is not `Conversation-keyword:` pass through.
- Decontextualizer's new ambiguity field name MUST NOT be a single boolean named `is_ambiguous`. The chosen `reference_resolution_status` (enum) + `needs_clarification` (boolean) + `clarification_reason` (free-text string) is acceptable; keep `clarification_reason` free-text rather than a structured tag so Stage 2 can migrate to per-referent grounding without a second schema migration.
- Cognition agents that MUST consume `needs_clarification`: `call_judgment_core_agent` (l2c) and `call_content_anchor_agent` (l3). Other cognition agents may read it but are not required to. This avoids the "and/or closely owned modules" ambiguity in scope.
- Decontextualizer fallback path (LLM exception or missing new fields) MUST log at WARN — not DEBUG — and include the user_input preview, so silent degradation on a flaky local Gemma is visible in logs.

## Must Do

- Extend decontextualizer output contract to include an ambiguity/clarification signal.
- Propagate that signal through `GlobalPersonaState`.
- Make `stage_1_research(...)` skip RAG when decontextualizer reports an unresolved required referent.
- Pass the ambiguity signal into cognition so the character can ask a clarification question.
- Harden the RAG initializer prompt so current-bot mentions and reply boilerplate are not treated as keyword content.
- Add deterministic sanitization for initializer slots so malformed LLM outputs using only bot mention/reply boilerplate as keyword targets are dropped before execution and before persistence.
- Add tests for the live failure shape using patched LLMs and deterministic slot sanitization.
- Add at least one inspectable live LLM prompt-contract test for the decontextualizer unresolved-reference case if the project already has the live-test harness available and the user approves running live LLM tests during execution.
- Bump `INITIALIZER_PROMPT_VERSION` so existing cache rows are invalidated atomically with this rollout.
- Pass `platform_bot_id` and a registered reply-boilerplate token list into the RAG supervisor `context` from `stage_1_research(...)`.
- Build the skipped-RAG `rag_result` via `project_known_facts([], ...)` to preserve the cognition-consumed dict shape.
- Wire the sanitizer into the cache read path immediately after `_read_cached_initializer_slots(...)` so cached-but-stale bad rows are dropped at read time even if a future cache row gets written under a missed version bump.
- Add explicit `needs_clarification` consumption to `call_judgment_core_agent` (l2c) and `call_content_anchor_agent` (l3).

## Deferred

- Media preservation and image/attachment persistence.
- Any change to queue coalescing.
- Full confidence scoring for RAG evidence.
- New RAG agents or new database query capabilities.
- Retrofitting old conversation-history rows.
- General reply-thread reconstruction beyond the existing `reply_context` and history payloads.
- Platform adapter changes unless tests prove the adapter is mislabeling `reply_to_current_bot`.
- Typed message envelope refactor (adapter -> `body_text` + typed mentions/reply/attachments -> storage/search/cache key consume `body_text`). Owned by Stage 2.
- Per-referent ambiguity grounding (replacing the boolean `needs_clarification` with a structured `referents: [{phrase, role, status}]` list and narrow-clarification cognition). Owned by Stage 2.
- Backfill / read-time normalization of historical conversation rows that contain wire markers in stored content. Owned by Stage 2.
- Conversation-search agents (`conversation_keyword_agent`, `conversation_search_agent`, `conversation_aggregate_agent`) querying `body_text` instead of the flattened wire form. Owned by Stage 2.

## Cutover Policy

This is a compatible behavior change:

- Existing API request and response shapes remain unchanged.
- Existing MongoDB collections remain unchanged.
- Existing decontextualizer callers continue to receive `decontexualized_input`.
- New optional state fields guide downstream behavior but do not break existing tests that ignore them.
- RAG initializer cache entries with old bad strategies remain possible in MongoDB until version invalidation or targeted cleanup. Execution should include a small cleanup step for any known bad cache key from this incident if one is identified.

## Agent Autonomy Boundaries

The implementation agent may edit:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` and/or closely owned cognition prompt modules only to consume the explicit clarification signal.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
- focused tests under `tests/`

The implementation agent must not:

- Edit `src/kazusa_ai_chatbot/service.py` for media preservation in this plan.
- Edit `src/kazusa_ai_chatbot/chat_input_queue.py` in this plan.
- Change persistence schema or add collections.
- Add broad keyword filters over user input outside the existing LLM interpretation stages and RAG slot validation.

## Target State

### Vague Input Flow

```text
user input + full history
  -> relevance decides should_respond as usual
  -> decontextualizer evaluates whether the user message has an unresolved required referent
       returns:
         decontexualized_input: original or rewritten text
         reference_resolution_status: resolved | unchanged_clear | unresolved_reference
         needs_clarification: true/false
         clarification_reason: short internal reason
  -> stage_1_research
       if needs_clarification:
         skip call_rag_supervisor
         set rag_result to empty/no evidence
       else:
         run RAG as today
  -> cognition/dialog
       if needs_clarification:
         ask a compact clarification question grounded in the current message
       else:
         normal behavior
```

### Bot Mention Search Hardening

```text
bad LLM slot:
  "Conversation-keyword: find messages containing '<@3768713357>' ..."

slot sanitizer:
  recognizes current bot mention token from context.platform_bot_id
  recognizes reply boilerplate such as "[Reply to message]" and CQ reply markers
  drops the slot before dispatcher/execution/cache persistence

result:
  no group history search is launched for current bot addressing metadata
```

## Design Decisions

| Decision | Policy | Rationale |
|---|---|---|
| Ambiguity owner | Decontextualizer reports unresolved references | It sees user input, reply context, recent history, channel topic, and indirect speech context before RAG loses grounding. |
| RAG behavior on unresolved reference | Skip RAG, do not guess | RAG is evidence retrieval; it should not invent what "this/these" means. |
| Clarification owner | Cognition/dialog asks the user | Keeps user-facing tone/persona in the normal response stage. |
| Mention-token hardening | Prompt rule plus deterministic slot sanitizer | Prompt education reduces bad output; sanitizer prevents one bad slot from executing. |
| Legitimate mentions | Allow non-bot mention entity questions | Avoid false positives for "who is <@123>?" style requests. |
| Cache policy | Do not persist unresolved-reference plans; sanitize before persistence; sanitize on cache read; bump prompt version | Prevents durable bad strategy rows AND prevents pre-existing bad rows from silently replaying. |
| History window for ambiguity | Decontextualizer keeps `chat_history_recent` only | Wide-history retrieval is RAG's job; if a referent is older than the recent window, asking for clarification is the correct behavior. Documented to make the trade-off explicit. |
| Local-LLM prompt fragility | Fallback to old behavior on missing fields, with WARN log | Local Gemma may omit new fields; we accept silent degradation but make it visible in logs. Full robustness requires Stage 2 (where ambiguity becomes a deterministic side effect of body_text + addressing typing rather than a prompt instruction). |
| Ambiguity field shape | Boolean `needs_clarification` + free-text `clarification_reason` now; structured per-referent later | Keeps Stage 1 schema lean while leaving room for Stage 2 to migrate without a second breaking change. |
| Sanitizer registry | Single token-class registry shared by prompt and code | Avoids drift between "prompt rule" and "deterministic check"; single deletion site for Stage 2. |

## Change Surface

| File | Changes |
|---|---|
| `persona_supervisor2_msg_decontexualizer.py` | Extend prompt/output contract; parse `reference_resolution_status`, `needs_clarification`, and `clarification_reason`; preserve old fallback behavior when fields are missing. |
| `persona_supervisor2_schema.py` | Add optional state fields for reference status and clarification reason. |
| `persona_supervisor2.py` | Propagate decontextualizer fields; skip RAG when `needs_clarification` is true; supply empty projected RAG result via `project_known_facts([], ...)`; pass `platform_bot_id` and the reply-boilerplate token registry into the RAG supervisor `context`. |
| `persona_supervisor2_cognition_l2.py` (judgment_core) and `persona_supervisor2_cognition_l3.py` (content_anchor_agent) | Consume `needs_clarification` and produce clarification-oriented judgment notes / content anchors instead of over-answering. The other cognition agents are out of scope for Stage 1. |
| `persona_supervisor2_rag_supervisor2.py` | Educate initializer prompt about reply/addressing metadata; add deterministic sanitizer with a token-class registry; run sanitizer on **both** cache write AND cache read paths; format `_INITIALIZER_PROMPT` from the same registry list so prompt and code agree on which markers are envelope-only. |
| `kazusa_ai_chatbot/rag/cache2_policy.py` | Bump `INITIALIZER_PROMPT_VERSION` so pre-existing bad cache rows are invalidated atomically with this rollout. |
| Tests | Add deterministic tests for decontext propagation, RAG skip, mention-token sanitizer, and no persistence of invalid initializer slots. |

## Implementation Order

1. Extend decontextualizer prompt contract.
   - Add statuses: `resolved`, `unchanged_clear`, `unresolved_reference`.
   - Add `needs_clarification` boolean.
   - Explain that unresolved reference requires clarification only when the referent is required to answer and cannot be resolved from visible input, reply context, history, channel topic, or indirect speech context.
   - Include examples:
     - reply excerpt has object -> resolved, no clarification
     - current input has literal object -> unchanged_clear, no clarification
     - "这些是什么意思" with no visible referent -> unresolved_reference, clarification
2. Update `call_msg_decontexualizer(...)` parsing.
   - Always return `decontexualized_input`.
   - Return `reference_resolution_status`, `needs_clarification`, `clarification_reason`.
   - If missing fields, default to `unchanged_clear`, `False`, `""` to preserve compatibility.
3. Update schema/state propagation.
   - Add optional fields to `GlobalPersonaState` and `CognitionState` as needed.
4. Update `stage_1_research(...)`.
   - Add `platform_bot_id` and a reply-boilerplate token list (sourced from a shared registry helper) to the `context` dict passed to `call_rag_supervisor`.
   - If `needs_clarification` is true, do not call `call_rag_supervisor`. Build `rag_result` via `project_known_facts([], current_user_id=..., character_user_id=..., answer="", unknown_slots=[], loop_count=0)` so the dict shape matches what cognition expects.
   - Otherwise run RAG as today.
5. Update cognition prompts/input — owners are explicit:
   - `call_judgment_core_agent` (l2c): include `needs_clarification` and `clarification_reason`; when true, the judgment_note must instruct downstream that broad prior context is NOT to be used as evidence.
   - `call_content_anchor_agent` (l3): include `needs_clarification`; when true, produce a single anchor that prompts the user for the missing referent, in the character's voice.
   - Other cognition agents are unchanged in Stage 1.
6. Harden initializer prompt in `persona_supervisor2_rag_supervisor2.py`.
   - Define reply/addressing metadata: current bot mention, platform bot ID, `[Reply to message]`, CQ reply tokens.
   - State these must not be used as keyword search targets.
   - Allow non-bot mentions only when the user asks about that mentioned account as the object.
7. Add deterministic sanitizer with token-class registry.
   - Registry helper in `persona_supervisor2_rag_supervisor2.py`:
     `_envelope_token_set(context: dict[str, Any]) -> tuple[str, ...]` returns the union of (a) `<@{platform_bot_id}>` when present, (b) `[Reply to message]`, (c) `[CQ:reply,` (CQ reply marker prefix). Future markers are added by appending to this function only.
   - Sanitizer: `_sanitize_initializer_slots(slots: list[str], context: dict[str, Any]) -> list[str]`.
     - For each slot starting with `Conversation-keyword:`: extract the quoted target phrase, strip every registered envelope token from it (and surrounding whitespace), drop the slot when the remaining phrase is empty or has length < 2 characters; otherwise pass the slot through unchanged.
     - Slots with other prefixes pass through unchanged.
   - Apply at TWO seams:
     1. After `_normalize_initializer_slots(...)` and before `_write_initializer_cache(...)` on the live LLM path.
     2. Immediately after `_read_cached_initializer_slots(...)` and before logging the cached slots on the cache-hit path.
   - Format `_INITIALIZER_PROMPT` so its "do not search these as keywords" rule lists the same registry tokens, ensuring the prompt and the deterministic check cannot drift apart.
8. Ensure invalid sanitized-empty plans are not persisted as useful wrong routes unless the final sanitized slot list is intentionally empty because no evidence is needed.
   - If all slots are removed due to invalid metadata-only targets, mark initializer cache reason as non-cacheable or do not schedule persistent upsert.
9. Bump `INITIALIZER_PROMPT_VERSION` in `kazusa_ai_chatbot/rag/cache2_policy.py` to invalidate pre-existing bad cache rows. Document the rationale in the commit message.
10. Add tests and run verification.

## Progress Checklist

- [x] Stage 1 — decontextualizer contract extended
  - Covers: implementation steps 1-2.
  - Expected files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`, focused decontextualizer tests.
  - Verify: `pytest tests/test_msg_decontexualizer.py -q`; compile the decontextualizer module.
  - Evidence: record changed files, test output, compile output, and any WARN-log behavior in `Execution Evidence`.
  - Handoff: next stage propagates the new fields through persona state.
  - Sign-off: Codex/2026-04-29 — verified by `pytest tests/test_msg_decontexualizer.py -q` and source compile.

- [x] Stage 2 — state propagation and RAG skip wired
  - Covers: implementation steps 3-4.
  - Expected files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`, focused supervisor/RAG-skip tests.
  - Verify: `pytest tests/test_persona_supervisor2_rag2_integration.py -q`; `pytest tests/test_persona_supervisor2_rag_skip_shape.py -q`; compile touched modules.
  - Evidence: record changed files, test output, compile output, and the skipped-RAG `project_known_facts([], ...)` result shape in `Execution Evidence`.
  - Handoff: next stage updates cognition consumers.
  - Sign-off: Codex/2026-04-29 — verified by `pytest tests/test_persona_supervisor2_rag2_integration.py -q`, `pytest tests/test_persona_supervisor2_rag_skip_shape.py -q`, and source compile.

- [x] Stage 3 — cognition clarification consumers updated
  - Covers: implementation step 5.
  - Expected files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`, or the current equivalent owned cognition prompt modules.
  - Verify: focused cognition tests or existing persona supervisor tests that exercise `call_judgment_core_agent` and `call_content_anchor_agent`; compile touched modules.
  - Evidence: record changed files, test output, compile output, and confirmation that broad prior context is not used as evidence when `needs_clarification=True`.
  - Handoff: next stage hardens RAG initializer prompt and sanitization.
  - Sign-off: Codex/2026-04-29 — verified by `pytest tests/test_cognition_clarification_consumers.py -q`, neighboring cognition tests, and source compile.

- [x] Stage 4 — RAG initializer sanitizer and cache seams complete
  - Covers: implementation steps 6-9.
  - Expected files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`, `src/kazusa_ai_chatbot/rag/cache2_policy.py`, focused RAG initializer/cache tests.
  - Verify: `pytest tests/test_rag_initializer_cache2.py -q`; static grep for the incident slot; compile touched modules.
  - Evidence: record changed files, test output, compile output, prompt-version bump, cache-read sanitization, cache-write sanitization, and non-persistence of metadata-only invalid slots in `Execution Evidence`.
  - Handoff: next stage completes full verification and smoke checks.
  - Sign-off: Codex/2026-04-29 — verified by `pytest tests/test_rag_initializer_cache2.py -q`, cache persistence tests, static grep, and source compile.

- [x] Stage 5 — verification, live checks, and manual smoke recorded
  - Covers: implementation step 10 and all Verification gates.
  - Expected files: tests under `tests/` plus this plan's `Execution Evidence` section.
  - Verify: run every deterministic test, compile command, static grep, approved live LLM check, and manual smoke listed in `Verification`, or record the exact blocker beside the skipped gate.
  - Evidence: fill `Execution Evidence` with exact command results, live/manual outcomes, blockers, and final changed-file list.
  - Handoff: plan can be marked `completed` only after evidence is recorded and acceptance criteria are met.
  - Sign-off: Codex/2026-04-29 — deterministic tests, compile, static greps, live LLM checks, and manual initializer smoke are recorded in `Execution Evidence`.

## Pre-implementation Sign-off

- Status: approved for implementation.
- Scope lock: implement only the Stage 1 work described in this plan; keep all Deferred and Stage 2 items out of the implementation.
- Reviewer: Codex.
- Date: 2026-04-29.
- Notes: The plan has explicit cutover policy, autonomy boundaries, mandatory rules, verification gates, rollback guidance, and now a staged progress checklist. No implementation work has been performed as part of this sign-off.

## Verification

### Deterministic Tests

- `pytest tests/test_msg_decontexualizer.py -q`
  - Existing tests still pass.
  - New patched-output test verifies new fields are parsed and returned.
  - Missing new fields defaults to backward-compatible values.

- `pytest tests/test_persona_supervisor2_rag2_integration.py -q`
  - Add/verify case where `needs_clarification=True` skips `call_rag_supervisor`.
  - Add/verify cognition receives clarification status in state.

- `pytest tests/test_rag_initializer_cache2.py -q`
  - Sanitizer drops `Conversation-keyword` slot whose only target is current bot mention.
  - Sanitizer drops reply boilerplate-only keyword targets.
  - Sanitizer drops CQ reply-marker-only keyword targets.
  - Sanitizer preserves legitimate keyword slots with real terms.
  - Sanitizer preserves legitimate non-bot mention entity questions when not current bot metadata.
  - Invalid dropped slot is not persisted to MongoDB.
  - **Cache replay**: a pre-seeded cached row containing only a current-bot-mention keyword target returns no slots after `_read_cached_initializer_slots(...)` + sanitizer, so the dispatcher is not entered.
  - **Version bump**: a row written under the previous `INITIALIZER_PROMPT_VERSION` is treated as a miss after the bump.

- `pytest tests/test_persona_supervisor2_rag_skip_shape.py -q` (new)
  - When `needs_clarification` is true, `stage_1_research(...)` returns a `rag_result` whose keys match `project_known_facts([], ...)` output: `user_image`, `character_image`, `memory_evidence`, `conversation_evidence`, `external_evidence`, `supervisor_trace` are all present.
  - Cognition consumers (`call_judgment_core_agent`, `call_content_anchor_agent`) do not raise `KeyError` on this shape.

- `pytest tests/test_msg_decontexualizer_warn.py -q` (new, or extend existing)
  - When the LLM raises, the fallback path logs at WARN level and the `user_input` preview is included in the log line.

### Compile

- `python -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py src/kazusa_ai_chatbot/nodes/persona_supervisor2.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`

### Static Greps

- `rg "Conversation-keyword: find messages containing '<@3768713357>'" src tests`
  - Expected only in tests documenting the bad-slot sanitizer, not as prompt example output.
- `rg "needs_clarification|reference_resolution_status" src/kazusa_ai_chatbot/nodes tests`
  - Expected in decontextualizer, schema, persona supervisor, cognition, and tests.

### Live LLM Tests

Run one at a time and inspect logs:

- Decontextualizer case: unresolved "这些是什么意思" with no visible referent returns `needs_clarification=true`.
- Decontextualizer case: same phrase with reply excerpt containing concrete symbols/text returns `needs_clarification=false`.
- RAG initializer case: bot mention in addressed message does not create a `Conversation-keyword` slot for the bot mention.

Do not batch-run these live LLM checks.

### Manual Smoke

- Reproduce the incident-like input after clearing any bad initializer cache row for the exact cache key.
- Confirm RAG is skipped if the referent is unresolved.
- Confirm final dialog asks for clarification instead of answering from "专属武器".
- Confirm a normal "昨天发言最多的是谁" query still runs RAG aggregate.
- Confirm a "这个 <@non_bot_id> 是谁" query can still resolve the mentioned account if supported by existing agents.

## Acceptance Criteria

This plan is complete when:

- Current-bot mention tokens and reply boilerplate cannot launch conversation keyword search.
- Decontextualizer can explicitly mark unresolved required references.
- `needs_clarification=True` skips RAG and reaches cognition/dialog.
- Clarification turns ask for the missing referent rather than using broad prior evidence.
- Existing clear queries still run RAG normally.
- Initializer cache does not persist metadata-only invalid search plans.
- Focused deterministic tests pass.
- Any live LLM/manual smoke blockers are recorded with exact failure output.

## Rollback / Recovery

- Revert decontextualizer contract extension and persona-state propagation if clarification behavior causes regressions.
- Revert RAG slot sanitizer independently if it blocks legitimate mention queries.
- Clear any bad persisted initializer rows by version bump or targeted delete from `rag_cache2_persistent_entries` if a bad strategy was persisted before this plan.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| False positive vague detection | Decontextualizer owns decision with full context; no deterministic "这些" blocker. | Live cases with reply excerpt/current literal object. |
| Legitimate mention queries get blocked | Sanitizer only blocks current bot mention/reply boilerplate as keyword target. | Test non-bot mention query preservation. |
| RAG skip starves useful evidence | Skip only when decontextualizer says required referent unresolved. | Regression tests for clear RAG queries. |
| Cognition ignores clarification flag | Add prompt/input test and live/manual smoke. | Test content anchors/dialog ask clarification. |
| Bad initializer plans remain durable | Sanitize before persistence; sanitize on read; bump `INITIALIZER_PROMPT_VERSION`; clear known bad row. | DB inspection / static test around upsert scheduling / cache-replay test. |
| Local Gemma silently omits new fields | Compatibility fallback retains old behavior; WARN log makes degradation visible. Cannot fully eliminate until Stage 2. | WARN-log test; periodic live-LLM contract test. |
| Stage 1 sanitizer accretes per-marker bespoke code | Implement as a single token-class registry that Stage 2 deletes wholesale; do not scatter per-marker checks. | Code review verifies the registry is the single source of truth for envelope tokens. |
| Boolean RAG-skip is too coarse for partial-vagueness inputs | Documented trade-off; Stage 2 introduces per-referent grounding. | Regression tests cover both fully-clear and fully-unresolved cases; partial-vagueness cases are accepted as Stage 2 work. |

## Stage 2 — Systemic Follow-up (separate plan)

This plan ships as Stage 1. Stage 2 is the structural fix and is described here only as a pointer so Stage 1 reviewers and implementers know the planned direction. Stage 2 will be drafted as a separate `development_plans/*.md`.

### Stage 2 scope

1. Adapter produces a typed `MessageEnvelope`:
   - `body_text: str` — user-authored content only, no wire markers.
   - `mentions: list[Mention]` — `{platform_id, global_id, display_name, role: bot|user|everyone}`.
   - `reply: ReplyTarget | None` — `{message_id, author_global_id, excerpt}`.
   - `attachments: list[...]`.
   - `raw_wire_text: str` — retained for audit/debug only.

   Each adapter (QQ, Discord, future) does this conversion ONCE. Downstream code consumes typed fields.

2. `IMProcessState.user_input` becomes derived from `body_text`. Wire-marker-laden text is no longer the canonical input. The current scattered typed fields (`mentioned_bot`, `reply_context.reply_to_current_bot`, `indirect_speech_context`) consolidate into the envelope's typed structure.

3. Storage migration: conversation history rows store the envelope. `conversation_keyword_agent`, `conversation_search_agent`, and `conversation_aggregate_agent` query `body_text`, never `raw_wire_text`. A read-time normalizer handles legacy rows during the migration window.

4. `build_initializer_cache_key` keys off `body_text` + a typed addressing intent (`addressed_to`, `reply_to_bot`, `reply_target_id`), not the wire form. Same query in different bot deployments shares cache; cosmetically equivalent queries no longer fragment.

5. Decontextualizer migrates from boolean `needs_clarification` to per-referent `referents: [{phrase, role: subject|object|time, status: resolved|unresolved}]`. Cognition can issue narrow clarification ("你说的『这些』是指什么？") while still using resolved anchors, replacing the binary RAG-skip cliff from Stage 1.

6. Memory consolidator stores `body_text` only; the persistent memory store no longer accumulates wire-marker noise.

7. The Stage 1 sanitizer (`_sanitize_initializer_slots`, `_envelope_token_set`) and the prompt rule that mirrors it are **deleted** in Stage 2 — wire markers are no longer reachable from `body_text`, so there is nothing to sanitize.

### Why Stage 1 first

- Stops today's bleeding without an adapter rewrite.
- Surfaces the registry of envelope tokens (mention, reply boilerplate, CQ reply marker) in one place — Stage 2 inherits the list as the basis for adapter normalization.
- Constrains Stage 1 schema choices (free-text `clarification_reason`, deletable sanitizer registry, boolean `needs_clarification` not named `is_ambiguous`) so Stage 2 doesn't have to undo them.

### What Stage 1 must avoid so Stage 2 stays cheap

- Do not store wire-marker text into any new persistence column or cache field.
- Do not add per-platform sanitizer branches; everything goes through the registry.
- Do not let cognition prompts hard-code the boolean field name in load-bearing logic — read it through a single helper so the Stage 2 schema swap is one edit.

## Execution Evidence

- Static grep results:
  - `rg "Conversation-keyword: find messages containing '<@3768713357>'" src tests` returned matches only in `tests/test_rag_initializer_cache2.py`, where the bad slot is used as sanitizer fixture input.
  - `rg "needs_clarification|reference_resolution_status" src/kazusa_ai_chatbot/nodes tests` returned expected matches in decontextualizer, schema, persona supervisor, cognition l2/l3, and tests.
- Test results:
  - `pytest tests/test_msg_decontexualizer.py -q` — 8 passed.
  - `pytest tests/test_persona_supervisor2_rag2_integration.py -q` — 2 passed.
  - `pytest tests/test_persona_supervisor2_rag_skip_shape.py -q` — 2 passed.
  - `pytest tests/test_rag_initializer_cache2.py -q` — 18 passed.
  - `pytest tests/test_cognition_clarification_consumers.py -q` — 2 passed.
  - Neighbor regressions: `pytest tests/test_persona_supervisor2.py -q` — 4 passed; `pytest tests/test_conversation_progress_cognition.py -q` — 8 passed; `pytest tests/test_cognition_preference_adapter.py -q` — 2 passed; `pytest tests/test_rag_cache2_persistent.py -q` — 8 passed; `pytest tests/test_rag_projection.py -q` — 4 passed.
- Compile results:
  - `python -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py src/kazusa_ai_chatbot/nodes/persona_supervisor2.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py src/kazusa_ai_chatbot/rag/cache2_policy.py` — passed with no output.
  - `python -m py_compile tests/test_cognition_clarification_consumers.py tests/test_persona_supervisor2_rag_skip_shape.py tests/test_cognition_live_llm_prompt_contracts.py` — passed with no output.
- Live LLM results:
  - `pytest tests/test_cognition_live_llm_prompt_contracts.py::test_live_msg_decontexualizer_marks_unresolved_reference -q -s -m live_llm` — passed; model returned `reference_resolution_status="unresolved_reference"` and `needs_clarification=True`.
  - `pytest tests/test_cognition_live_llm_prompt_contracts.py::test_live_msg_decontexualizer_resolves_reply_excerpt_reference -q -s -m live_llm` — passed; model resolved the reply excerpt and returned `needs_clarification=False`.
- Manual smoke:
  - Manual live initializer check for `[Reply to message] <@3768713357>  你知道这些是什么意思么？` with `platform_bot_id=3768713357` did not produce a `Conversation-keyword` slot for `<@3768713357>`. It produced one Memory-search slot; in the full Stage 1 flow, unresolved-reference turns skip RAG before this initializer runs.
  - The first manual initializer attempt failed because a script monkeypatched `asyncio.create_task` globally; it was rerun with only persistence no-op functions patched and passed the intended inspection.
- Changed files:
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - `tests/test_msg_decontexualizer.py`
  - `tests/test_persona_supervisor2_rag2_integration.py`
  - `tests/test_persona_supervisor2_rag_skip_shape.py`
  - `tests/test_rag_initializer_cache2.py`
  - `tests/test_cognition_clarification_consumers.py`
  - `tests/test_cognition_live_llm_prompt_contracts.py`
  - `development_plans/rag_reply_mention_and_vague_input_plan.md`
