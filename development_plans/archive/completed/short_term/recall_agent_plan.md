# recall agent plan

## Summary

- Goal: Add a RAG2 recall capability for active agreements, promises, plans, open loops, and current-episode state so the initializer no longer has to choose between conversation history, user memory units, and conversation progress for promise/episode recall.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`, `py-style`, `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: additive; introduce one new RAG2 helper agent and narrow prompt/projection changes without data migration or replacing existing conversation/history/memory agents.
- Highest-risk areas: broadening the initializer prompt too much, touching cognition layers unnecessarily, treating progress-derived recall as durable fact evidence, cache invalidation from prompt changes, and response-path latency from unnecessary history searches.
- Acceptance criteria: active agreement queries route to Recall; Recall reconciles progress, active commitments, pending scheduled events, and gated history evidence; cognition receives the direct recall answer through the existing `rag_result.answer` path without L2/L3 file edits; consolidator does not turn progress-only recall into durable character facts.

## Context

The observed failure was a user asking whether the character remembered today's agreement:

```text
早上好呀，还记得今天的约定么
```

RAG2 initialized this as:

```text
Conversation-keyword: find messages containing '约定' in the recent chat history
```

That route found the current user message, marked the slot resolved, and returned a self-hit rather than the actual appointment. The system had the correct state elsewhere:

- `conversation_progress` tracked the active episode and resolved/open threads.
- `user_memory_units.active_commitments` tracked ongoing accepted commitments.
- `conversation_history` contained literal proof of the original appointment.

The root problem is not missing data. It is that the initializer has no semantic route for "recall the current/past agreement." It must choose between overlapping physical stores, which is too fragile for a local planner.

Adjacent RAG improvements identified during planning, intentionally left for later plans:

- Real-time fact routing: live weather, temperature, schedules, prices, and current-status questions need a location/source-resolution path before any memory lookup.
- Conversation search quality: message search can gain topic, speaker, date window, and anti-self-hit controls so literal keyword search finds origin evidence instead of the current question.
- Memory source hygiene: shared world memory, user memory units, active commitments, and progress state need clearer source labels and authority guidance across RAG outputs.
- RAG regression coverage: routing fixtures should cover promises, exact quotes, world facts, user profile facts, character facts, live facts, and ambiguous follow-ups.
- Cache diagnostics: Cache2 traces can expose prompt-version, source-version, and volatile-source reasons more visibly when a route or result is stale.

## Mandatory Skills

Load these skills in this order before implementing, verifying, or handing off this plan:

- `development-plan-writing`: load before editing this plan, after automatic context compaction, and before every major checklist sign-off.
- `local-llm-architecture`: load before changing RAG2 routing, helper-agent boundaries, prompts, cognition inputs, consolidator prompt inputs, or any LLM call/context contract.
- `py-style`: load before editing Python production or test files.
- `test-style-and-execution`: load before adding, changing, or running deterministic, integration, live LLM, or prompt-render tests.
- `cjk-safety`: load before editing any Python prompt or source file containing Chinese/Japanese/Korean text.

## Mandatory Rules

- Follow the mandatory skills above. If compaction occurs, reread this entire plan and reload the skill that governs the next file or verification step before continuing.
- Preserve the current RAG2 architecture: initializer plans semantic slots, dispatcher selects a helper, helper owns low-level source logic, finalizer summarizes evidence.
- Do not add generic primary/secondary routing. The initializer must not hedge across multiple stores.
- Do not make the initializer produce database parameters, timestamps, query filters, or source precedence decisions.
- Add a first-class Recall slot only for agreement/promise/plan/open-loop/current-episode recall.
- Do not route exact quote, URL, filename, or "who said this exact phrase" requests to Recall. Those remain `Conversation-keyword` / `Conversation-filter`.
- Do not route world knowledge, durable character/world facts, live external facts, profile impressions, or relationship ranking to Recall.
- Keep `conversation_progress` invisible to relevance.
- Do not remove `conversation_progress` from cognition. Recall provides query-specific evidence; `conversation_progress` remains semantic flow state.
- Do not edit `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` or `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` in v1. Existing cognition code already receives public `rag_result` and already treats `rag_result.answer` as the priority direct answer path. If a focused test proves this contract is false, stop and revise this plan with user approval before touching cognition files.
- Do not add deterministic keyword classification over raw user text outside the LLM initializer. The Recall helper is limited to structural source metadata, enumerated `Recall:` slot modes, and the fixed collector gates in this plan.
- Disable Cache2 for the Recall helper in the first version because progress, active commitments, and scheduled events are volatile.
- Treat progress-derived recall as current operational evidence, not durable fact authority by default.
- Keep Recall database access read-only. Do not add write, update, delete, cancel, reschedule, migration, backfill, or collection-creation behavior.
- Use test-first implementation. Create or update the focused tests listed in `Implementation Order`, run them before production changes to capture the failure/baseline, then rerun the same tests after the implementation.
- Run live LLM tests one at a time and inspect each trace before starting the next live test.
- Do not claim prompt behavior is verified by mocked LLM tests. Mocked or deterministic tests can verify schema, rendering, routing parser contracts, and projection behavior; live LLM tests verify model routing behavior.
- If prompt strings with JSON examples are edited, run a runtime prompt-render check in addition to `py_compile`.
- Prompt edits must preserve trusted/untrusted boundaries: runtime user text remains `HumanMessage` content, prompt examples must not include unescaped braces that break `.format`/template rendering, and generic RAG2 routing rules must not hard-code one character as the only valid case.
- No deterministic keyword or regex classifier may reinterpret raw user text after the initializer. Deterministic code may only consume the enumerated `Recall:` mode, structured context fields, lifecycle status, timestamps, and source labels.
- Every LLM call touched by this plan must keep the same response-path call count unless this plan explicitly states otherwise in `LLM Call And Context Budget`.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.

## Must Do

- Add `recall_agent` as a RAG2 helper agent with the same public helper surface as existing agents: `run(task, context, max_attempts=1) -> dict`.
- Pass `conversation_progress` and `conversation_episode_state` into the RAG2 context from `persona_supervisor2.stage_1_research`.
- Add an initializer slot prefix:

```text
Recall: retrieve <active agreement / ongoing promise / current episode state / open loop> relevant to <topic>
```

- Add dispatcher support mapping `Recall:` slots to `recall_agent`.
- Make Recall collect candidates from:
  - already-loaded `conversation_progress` / `conversation_episode_state`,
  - active user memory units, especially `active_commitment`,
  - pending scheduled events for the same platform/channel/user,
  - conversation history only through the fixed `HistoryEvidenceCollector` gate.
- Implement deterministic source arbitration inside Recall:
  - current/today/now/一会儿/current episode queries prefer active `conversation_progress`,
  - durable ongoing promise queries prefer `user_memory_units.active_commitments`,
  - executable future action queries include pending `scheduled_events`,
  - exact wording/provenance queries prefer `conversation_history`,
  - conflict cases return the selected claim plus conflict notes.
- Project Recall results into a dedicated `rag_result.recall_evidence` field and keep `rag_result.answer` as the direct factual answer.
- Keep cognition L2/L3 files unchanged. Verify that existing cognition payload shaping preserves `rag_result.answer` and does not remove `rag_result.recall_evidence`.
- Update consolidator facts prompts to classify `rag_result.recall_evidence` as recall/provenance evidence. Progress-only recall must not create durable character facts unless supported by user current input, durable memory, conversation evidence, or external evidence.
- Add deterministic and live LLM tests covering routing, source arbitration, projection, cognition no-touch/pass-through expectations, and consolidator authority boundaries.

## Deferred

- Do not implement generic secondary/fallback route sets.
- Do not redesign Cache2.
- Do not persist Recall helper results.
- Do not create a new durable collection for recall.
- Do not migrate or rewrite historical `conversation_episode_state`, `user_memory_units`, `conversation_history`, or `scheduled_events`.
- Do not change conversation-progress recorder behavior except where tests reveal a structural payload incompatibility.
- Do not broaden Recall into generic memory search, profile lookup, relationship evaluation, or live external fact lookup.
- Do not remove existing conversation search/filter/keyword agents.

## Cutover Policy

The user-approved cutover strategy is additive with one prompt-cache bigbang.
Every changed area is classified exactly once:

| Area | Policy | Instruction |
|---|---|---|
| RAG2 `Recall:` route | `compatible` | Add the new slot prefix and helper mapping. Existing `Conversation-*`, `Memory-search`, `Profile`, `Relationship`, and `Web-search` routes continue to work. |
| Initializer prompt/cache | `bigbang` | Increment `INITIALIZER_PROMPT_VERSION` in the same change that edits the initializer prompt. Existing initializer cache entries are intentionally invalidated at deploy. |
| RAG2 context payload | `compatible` | Add `conversation_progress` and `conversation_episode_state` to the existing context dictionary. Existing context keys remain unchanged. |
| Recall helper | `compatible` | Add `recall_agent` as a new helper. It runs only when the dispatcher receives a `Recall:` slot. |
| Projection payload | `compatible` | Add `rag_result.recall_evidence` with an empty-list default. Existing projected keys remain unchanged. |
| Cognition layers | `compatible` | Keep L2/L3 cognition files unchanged. Existing public `rag_result.answer` carries the direct Recall answer into cognition. |
| Consolidator prompts | `compatible` | Add guidance for `recall_evidence` without removing existing evidence fields. |
| Database access | `compatible` | Add one read-only scheduled-event query helper over the existing collection. No schema migration, writes, or new collection are permitted. |
| Tests/docs | `compatible` | Add focused tests and README documentation without changing production behavior outside the approved route. |

Rollback is code-only:

1. Remove the `Recall:` initializer examples/rule and bump `INITIALIZER_PROMPT_VERSION` again.
2. Remove dispatcher/registry mapping for `recall_agent`.
3. Stop passing progress state into RAG2 context.
4. Leave any added projection field defaulting to an empty list until a later cleanup.
5. Run the focused RAG2 integration and prompt-routing tests listed in `Verification`.

## Agent Autonomy Boundaries

Implementation agents may only make these local implementation choices:

- Choose private helper function names inside `recall_agent.py`; public function,
  class, file, and payload names are fixed by this plan.
- Add private helpers only inside files listed in `Change Surface`. Each helper
  must serve one listed requirement and must not add a new source, route,
  persistence path, graph stage, cache namespace, or prompt stage.
- Add test fixtures inside the test files named in this plan. New production
  fixtures, seed data, or migrations are forbidden.
- Shall follow the mandatory skills listed in this plan when implementing new
  modules, updating existing modules, editing prompts, or writing tests.
- Choose local variable names and small private formatting details that preserve
  the contracts, public names, source precedence, and verification gates in this
  plan.

Implementation agents must not:

- Invent a broader "memory manager" subsystem.
- Create any new public module, package, graph node, DB collection, scheduler
  behavior, cache namespace, environment variable, CLI, or background task not
  listed in this plan.
- Add route fallback or self-repair loops to the RAG2 supervisor.
- Add source-precedence rules to the initializer beyond "this is a Recall slot."
- Make Recall run every source unconditionally.
- Treat source timestamp recency as the only authority rule.
- Change consolidation persistence schemas or scheduler semantics.
- Modify conversation-progress recorder behavior, conversation-progress stored
  schema, user-memory-unit schema, conversation-history schema, relevance
  behavior, dialog generator behavior, or generic RAG2 retry semantics.
- Modify cognition L2/L3 files in v1.
- Add deterministic keyword or regex classifiers over raw user input, final
  dialog, chat history, memory text, or progress text.
- Add any LLM call inside `recall_agent.py`.
- Add broad exception swallowing. Internal contract violations should fail tests
  or return the explicit `resolved=False` missing-scope result defined below.
- Use ad hoc shell/Python writes; edit files with `apply_patch` and use normal formatting/test commands.
- Mark a checklist item complete without recording the required evidence in
  `Execution Evidence`.

If this plan and code disagree, preserve the plan's stated intent and report the
discrepancy. If implementation reveals a missing architectural decision or a
required instruction is impossible, stop and update this plan with user approval
before coding past that point. Do not invent a substitute architecture.

## Specification Audit

This plan intentionally fixes the architecture before implementation. The active
implementation agent must treat these items as closed decisions:

- Public files to add: exactly `rag/recall_agent.py`,
  `db/scheduled_events.py`, `tests/test_rag_recall_agent.py`, and
  `tests/test_rag_recall_live_llm.py`.
- Public Recall class: `RecallAgent`.
- Public scheduled-event helper: `query_pending_scheduled_events`.
- New RAG2 slot prefix: exactly `Recall:`.
- Recall slot modes: exactly `active_episode_agreement`,
  `durable_commitment`, `episode_position`, and `exact_agreement_history`.
- Recall collectors: exactly `ProgressCollector`, `ActiveCommitmentCollector`,
  `ScheduledEventCollector`, and `HistoryEvidenceCollector`.
- Collector limits: progress 8 claims, active commitments 6 claims, scheduled
  events 10 claims, history query 20 rows with at most 5 copied claims.
- Cache policy: Recall has no Cache2 namespace and always reports
  `volatile_recall`.
- LLM policy: Recall performs zero internal LLM calls.
- Projection field: exactly `rag_result.recall_evidence`.
- No schema migration, new collection, new graph stage, new background task,
  new environment variable, generic fallback, or secondary route set is part of
  this plan.

If an implementation agent cannot complete the work within these fixed
decisions, the correct action is to stop and revise the plan with user approval.

## Target State

For the failed example, RAG2 must produce an equivalent trace:

```text
Initializer:
  ["Recall: retrieve active agreement relevant to today's appointment"]

Dispatcher:
  recall_agent

Recall selected result:
  primary_source=conversation_progress
  supporting_sources=["user_memory_units.active_commitments"]
  selected_summary="当前有效约定是：用户很快来千纱家接她，然后一起去坐过山车；抱抱和冰激凌是相关后续奖励/互动话题。"
```

The final cognition state receives:

- `rag_result.answer`: a direct factual answer to the user's recall question,
- `rag_result.recall_evidence`: structured provenance for the selected recall,
- `conversation_progress`: unchanged flow guidance for how to continue the episode.

Cognition L2/L3 prompt files are unchanged in v1. The direct answer path is the
existing `rag_result.answer` contract, and `recall_evidence` is added for
projection traceability and consolidator authority checks.

## Design Decisions

| Decision               | Choice                                                                                                                             | Rationale                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Recall as RAG2 helper  | Add `recall_agent` under `src/kazusa_ai_chatbot/rag/`                                                                              | Keeps RAG2 evidence retrieval ownership and avoids a new graph stage.                                     |
| Initializer role       | Emits one semantic `Recall:` slot                                                                                                  | Keeps local planner simple; source arbitration belongs to the specialist.                                 |
| Source collection      | Progress + active commitments + scheduled events; conversation history only under the fixed `HistoryEvidenceCollector` gates below | Avoids noisy transcript search and reduces latency.                                                       |
| Freshness model        | Source-specific lifecycle rules                                                                                                    | "Newest row wins" is wrong for promises; active progress and active commitments have different authority. |
| Cache policy           | No Cache2 for Recall v1                                                                                                            | Sources are volatile across turns and scheduler writes.                                                   |
| Projection             | Add `recall_evidence`                                                                                                              | Prevents progress-derived evidence from being confused with raw conversation evidence.                    |
| Cognition edits        | Do not edit L2/L3 cognition files in v1                                                                                            | Existing `_cognition_rag_result` copies public `rag_result`, and L3 already prioritizes `rag_result.answer`; avoiding prompt edits reduces blast radius. |
| Consolidator authority | Progress-only recall is operational, not durable                                                                                   | Prevents short-term episode state from becoming stable character lore.                                    |

## Contracts And Data Shapes

### New Module Interface

Public production entrypoints created by this plan:

```python
class RecallAgent(BaseRAGHelperAgent):
    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        raise NotImplementedError

async def query_pending_scheduled_events(
    *,
    platform: str,
    platform_channel_id: str,
    global_user_id: str,
    current_timestamp: str,
    limit: int = 10,
) -> list[ScheduledEventDoc]:
    raise NotImplementedError
```

Ownership boundary:

- `src/kazusa_ai_chatbot/rag/recall_agent.py` owns candidate collection, source arbitration, freshness explanation, conflict notes, and the helper return shape.
- `src/kazusa_ai_chatbot/db/scheduled_events.py` owns read-only scheduled-event retrieval for Recall.
- `persona_supervisor2_rag_supervisor2.py` owns routing and dispatcher registration.
- `persona_supervisor2_rag_projection.py` owns conversion from raw helper result to `rag_result.recall_evidence`.
- Cognition files consume the existing public `rag_result` and remain unchanged.
- Consolidator prompt files own durable-memory authority guidance only; they must not perform Recall source arbitration.

Allowed callers/importers:

- `persona_supervisor2_rag_supervisor2.py` may import and register `RecallAgent`.
- `recall_agent.py` may import `query_user_memory_units`, `get_conversation_history`, and `query_pending_scheduled_events`.
- Tests named in this plan may import `RecallAgent` and `query_pending_scheduled_events`.
- No other production module may import Recall private collectors or scheduled-event query internals.

Hidden internals:

- Collector functions/classes inside `recall_agent.py` are private implementation details.
- Freshness scoring helpers are private implementation details.
- Scheduled-event collection name, projection mechanics, and DB client access are hidden inside `db/scheduled_events.py`.

Test seams:

- Deterministic tests may monkeypatch DB helper symbols imported by `recall_agent.py`.
- Tests must exercise `RecallAgent.run(task, context, max_attempts=1)` as the public helper surface.
- Tests must not depend on private collector names except to assert that exactly the four collector categories listed in this plan are represented in behavior.

### Primary Helper Contract

Recall uses the standard RAG helper-agent surface:

```python
async def run(
    task: str,
    context: dict[str, Any],
    max_attempts: int = 1,
) -> dict[str, Any]:
    raise NotImplementedError
```

### Mandatory Context

These fields are required for every Recall invocation:

```python
{
    "platform": str,
    "platform_channel_id": str,
    "global_user_id": str,
    "current_timestamp": str,
}
```

If any of these fields are missing or blank, Recall must return `resolved=False`
with this error result shape. It must not widen DB queries across platforms,
channels, users, or time.

```json
{
  "resolved": false,
  "result": {
    "selected_summary": "",
    "recall_type": "unsupported",
    "primary_source": "",
    "supporting_sources": [],
    "freshness_basis": "missing mandatory recall context: <comma-separated field names>",
    "conflicts": [],
    "candidates": [],
    "error": "missing_mandatory_context"
  },
  "attempts": 1,
  "cache": {
    "enabled": false,
    "hit": false,
    "reason": "volatile_recall"
  }
}
```

### RAG2 Integration-Required Context

The production RAG2 integration must pass:

```python
{
    "conversation_progress": dict,
    "conversation_episode_state": dict | None,
}
```

`conversation_progress` is required in the RAG2 path because active-episode
recall is the core purpose of this capability. If `conversation_progress` is
absent, Recall must not answer as active episode recall. For
`active_episode_agreement` and `episode_position` modes, it must return
`resolved=False` unless `ActiveCommitmentCollector` finds active commitment
candidates. When active commitments are found, it must return
`recall_type="durable_commitment"`, `primary_source="user_memory_units"`, and a
`freshness_basis` that states active-episode state was unavailable. For
`durable_commitment` mode, active commitments remain a valid primary source.

`conversation_episode_state` can be `None`. When present, Recall uses
it for stronger freshness metadata such as `updated_at`, `expires_at`,
`turn_count`, and raw first-seen timestamps.

### Optional Context

These fields are non-required context fields. Recall must not require them:

```python
{
    "user_name": str,
    "original_query": str,
    "current_slot": str,
    "known_facts": list,
    "chat_history_recent": list,
    "chat_history_wide": list,
    "channel_type": str,
    "prompt_message_context": dict,
    "reply_context": dict,
    "indirect_speech_context": str,
}
```

Recall must not rely on non-required raw chat-history lists as its primary evidence
source. Literal conversation proof must come from bounded conversation-history
retrieval only when exact wording, provenance, or conflict resolution requires
it.

### Recall Slot Modes

The initializer must emit only these Recall mode names inside the `Recall:` slot:

```text
active_episode_agreement
durable_commitment
episode_position
exact_agreement_history
```

Accepted slot examples:

```text
Recall: retrieve active_episode_agreement relevant to today's appointment
Recall: retrieve durable_commitment relevant to ongoing promises
Recall: retrieve episode_position relevant to where the conversation left off
Recall: retrieve exact_agreement_history relevant to when the agreement was made
```

Any `Recall:` slot without one of these mode names must be handled as
`active_episode_agreement` unless the task text explicitly contains
`exact_agreement_history`. The helper must not invent additional modes.

### Fixed Collector Interfaces

Recall v1 has exactly four collectors. Implementation agents must not add,
remove, or rename collector categories without updating this plan.

1. `ProgressCollector`
   
   - Input: `context["conversation_progress"]` and
     `context.get("conversation_episode_state")`.
   - DB calls: none.
   - Output source label: `conversation_progress`.
   - Hard cap: at most 8 candidate claims.

2. `ActiveCommitmentCollector`
   
   - Input: `global_user_id`.
   - DB helper: `query_user_memory_units`.
   - Required call shape:

```python
await query_user_memory_units(
    global_user_id,
    unit_types=[UserMemoryUnitType.ACTIVE_COMMITMENT],
    statuses=[UserMemoryUnitStatus.ACTIVE],
    limit=6,
)
```

- Output source label: `user_memory_units`.
- Hard cap: at most 6 candidate claims.
3. `ScheduledEventCollector`
   
   - Input: `platform`, `platform_channel_id`, `global_user_id`,
     `current_timestamp`.
   - DB helper: `query_pending_scheduled_events` added by this plan.
   - Required limit: 10.
   - Output source label: `scheduled_events`.
   - Hard cap: at most 10 candidate claims.

4. `HistoryEvidenceCollector`
   
   - Input: `platform`, `platform_channel_id`, `global_user_id`.
   - DB helper: `get_conversation_history`.
   - Required call shape:

```python
await get_conversation_history(
    platform=platform,
    platform_channel_id=platform_channel_id,
    global_user_id=global_user_id,
    limit=20,
)
```

- Output source label: `conversation_history`.
- Hard cap: at most 5 candidate claims copied into the Recall result.
- Execution gate: do not run by default. Run only when the ranked candidates
  contain a conflict between `conversation_progress`, `user_memory_units`,
  or `scheduled_events`, or when the `Recall:` slot text contains
  `exact_agreement_history`. Exact quoted phrase, URL, filename, and
  "who said X" requests must not use Recall and therefore must not reach this
  collector.

Recall candidate shape:

```json
{
  "source": "conversation_progress | user_memory_units | scheduled_events | conversation_history",
  "claim": "short factual claim",
  "temporal_scope": "current_episode | durable_ongoing | pending_future_action | historical_proof",
  "lifecycle_status": "active | pending | recently_resolved | historical | completed | cancelled | stale",
  "evidence_time": "ISO timestamp or empty",
  "authority": "primary_for_current_episode | primary_for_durable_commitment | primary_for_exact_wording | supporting"
}
```

Recall result shape:

```json
{
  "selected_summary": "direct factual recall answer",
  "recall_type": "active_episode_agreement | durable_commitment | exact_history | mixed",
  "primary_source": "conversation_progress | user_memory_units | scheduled_events | conversation_history",
  "supporting_sources": ["source labels"],
  "freshness_basis": "short explanation",
  "conflicts": ["conflict notes; empty list when none"],
  "candidates": ["candidate records capped by the collector limits above"]
}
```

Helper return shape:

```json
{
  "resolved": true,
  "result": {
    "selected_summary": "direct factual recall answer",
    "recall_type": "active_episode_agreement | durable_commitment | exact_history | mixed",
    "primary_source": "conversation_progress | user_memory_units | scheduled_events | conversation_history",
    "supporting_sources": ["source labels"],
    "freshness_basis": "short explanation",
    "conflicts": [],
    "candidates": []
  },
  "attempts": 1,
  "cache": {
    "enabled": false,
    "hit": false,
    "reason": "volatile_recall"
  }
}
```

### Output Size Caps

Recall output must stay compact because it is passed through RAG2 finalization,
projected into the existing cognition state payload, and included in
consolidator prompts:

- `selected_summary`: maximum 600 characters.
- `freshness_basis`: maximum 400 characters.
- Each candidate `claim`: maximum 240 characters.
- Candidate count: maximum 29 records total, derived only from the fixed
  collector limits above.
- `conflicts`: maximum 5 strings, each maximum 240 characters.
- `supporting_sources`: source labels only, no copied evidence text.
- `rag_result.recall_evidence`: maximum 3 Recall entries per projected RAG2
  result. If more Recall slots exist, keep the first three dispatched Recall
  results in slot order and leave the full raw trace only in supervisor trace.

## Freshness And Authority Rules

The Recall helper must rank by query intent first, lifecycle status second, and timestamp third.

Current execution state:

```text
conversation_progress > scheduled_events > active_commitments > conversation_history
```

Durable ongoing agreement:

```text
active_commitments > scheduled_events > conversation_progress > conversation_history
```

Exact wording/provenance:

```text
conversation_history > active_commitments > conversation_progress
```

Hard source handling rules:

- Ignore progress as active evidence when `status` is inactive/empty, `continuity` is `sharp_transition`, or the loaded document is expired.
- Exclude memory units whose status is `completed`, `cancelled`, or `archived`. Recall v1 reads active commitments only; historical completed/cancelled commitment recall is out of scope for this plan.
- Treat pending scheduled events as strong evidence of an executable future obligation, not as social/relationship meaning.
- Treat conversation history as proof of what was said, not proof that a promise is still active.
- When newer active progress supersedes older appointment details, return the newer state and include the old detail only as a conflict/support note only if `HistoryEvidenceCollector` ran under its gate.

## LLM Call And Context Budget

Use `50k tokens` as the context cap. Estimates below use conservative character
counts because exact tokenizer counts are unavailable during planning.

| Stage | Before | After | Path | Model/helper | Context delta and cap policy | Latency/blocking | Verification |
|---|---:|---:|---|---|---|---|---|
| RAG2 initializer LLM | 1 call | 1 call | response path | existing initializer model | Adds concise `Recall:` route rules/examples and receives projected `conversation_progress` / `conversation_episode_state`. Added prompt/context must remain under 6,000 characters total. Volatile progress fields must not enter the initializer cache key. | No new call; same blocking point. | Prompt-render check, cache-version test, live route tests. |
| RAG2 dispatcher LLM | 1 call | 1 call | response path | existing dispatcher model | Adds one roster/mapping entry for `recall_agent`. No database parameters or source precedence text added to dispatcher output contract. | No new call; same blocking point. | Dispatcher mapping test and live RAG2 trace. |
| `recall_agent` | 0 calls | 0 calls | response path deterministic helper | Python helper | Uses structured context plus bounded DB reads. Output caps are defined in `Output Size Caps`. | Adds bounded DB latency only; history query runs only under its gate. | `tests/test_rag_recall_agent.py`. |
| RAG2 finalizer LLM | 1 call | 1 call | response path | existing finalizer model | Receives one compact Recall fact/result through existing known-facts flow. Recall evidence follows the caps above. | No new call; same blocking point. | Live RAG2 answer test and projection tests. |
| Cognition L2/L3 LLMs | existing calls | existing calls | response path | existing cognition models | No cognition prompt/file change. Existing payload shaping receives public `rag_result`, with `rag_result.answer` as the direct answer path. | No new call; same blocking points. | No-touch git check plus projection/pass-through test. |
| Dialog LLM | existing call | existing call | response path | existing dialog model | No direct prompt contract change in this plan. | No change. | Existing dialog/cognition smoke coverage. |
| Consolidator LLM | existing call | existing call | background | existing consolidator model | Receives evidence taxonomy guidance for `recall_evidence`. Progress-only recall is not durable fact authority by itself. | No response-path latency change. | Consolidator prompt test. |

The Recall helper must not run existing conversation keyword/search helper loops.
History evidence must use the fixed `get_conversation_history` call shape
defined above with `limit=20`. Do not add generator/judge loops inside Recall
v1.

The total response-path LLM call count for a Recall query remains unchanged
except for route selection causing an existing helper slot to execute the
deterministic Recall helper rather than a different helper. New response-path
LLM calls, larger context caps, or raw transcript dumps require a new approved
plan.

## Change Surface

Target ownership boundary: the Recall capability, consisting of the public
`RecallAgent` entrypoint and read-only scheduled-event support. Existing-module
edits below are integration adapters required to pass context, route the slot,
project the result, document the helper, or guard consolidation authority.

### Delete

- No files, modules, schemas, collections, routes, or tests are deleted by this plan.

### Create

- `src/kazusa_ai_chatbot/rag/recall_agent.py`
  - Owns candidate collection, freshness scoring, source arbitration, and result shaping.
- `src/kazusa_ai_chatbot/db/scheduled_events.py`
  - Adds a small read-only helper for pending scheduled events scoped by platform/channel/user.
- `tests/test_rag_recall_agent.py`
  - Deterministic source arbitration tests.
- `tests/test_rag_recall_live_llm.py`
  - One-by-one live initializer/RAG2 routing checks.

### Modify

- `src/kazusa_ai_chatbot/db/__init__.py`
  - Export the new scheduled-event query helper.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Add `conversation_progress` and `conversation_episode_state` to RAG2 context.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - Import/register `RecallAgent`.
  - Add `Recall:` slot prefix to initializer allowed prefixes and dispatcher mapping.
  - Add narrow initializer examples for active agreement and "where were we" recall.
  - Add dispatcher roster/mapping entry.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
  - Add `recall_evidence: []` to projected `rag_result`.
  - Project `recall_agent` raw result into `recall_evidence`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
  - Add `recall_evidence` to the evidence taxonomy.
  - State that progress-only recall is operational episode evidence and does not by itself authorize durable character facts.
- `src/kazusa_ai_chatbot/rag/README.md`
  - Document Recall helper responsibility and source precedence.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - Increment `INITIALIZER_PROMPT_VERSION` because initializer routing prompt changes invalidate old cached routes.

### Keep

- `conversation_progress` storage schema.
- `user_memory_units` storage schema.
- `conversation_history` schema.
- scheduler execution semantics.
- relevance agent.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`.
- dialog generator behavior.
- generic RAG2 fallback/retry semantics.

## Implementation Order

Rationale: prove the Recall module contract before wiring RAG2. Outside-module
edits happen only after module tests pass.

1. **Add module contract tests and capture pre-change failure**

   - Create `tests/test_rag_recall_agent.py`.
   - Cover missing mandatory scope, progress-first recall, commitment-first
     recall, pending scheduled events, history gate behavior, stale/sharp
     transition progress, completed/cancelled/archived memory exclusion,
     conflicts, and output caps.
   - Run `venv\Scripts\python.exe -m pytest tests\test_rag_recall_agent.py -q`
     before production edits and record the missing-module/helper failures in
     `Execution Evidence`.

2. **Add integration tests and capture current behavior**

   - Update `tests/test_persona_supervisor2_rag2_integration.py` for RAG2
     context plumbing.
   - Update `tests/test_rag_projection.py` for `recall_evidence` projection and
     cognition pass-through expectations.
   - Update `tests/test_rag_initializer_cache2.py` for `Recall:` prefix and
     initializer prompt version.
   - Create `tests/test_rag_recall_live_llm.py` with the live tests listed in
     `Verification`.
   - Run the practical deterministic integration tests before integration edits
     and record the expected missing-field/prefix failures.
   - Run the first live initializer test before prompt edits and record the
     current non-Recall route baseline.

3. **Implement module-owned support and Recall module**

   - Add `db/scheduled_events.py` with this exact public function:

```python
async def query_pending_scheduled_events(
    *,
    platform: str,
    platform_channel_id: str,
    global_user_id: str,
    current_timestamp: str,
    limit: int = 10,
) -> list[ScheduledEventDoc]:
    raise NotImplementedError
```

   - Query filter must be exactly:

```python
{
    "status": "pending",
    "source_platform": platform,
    "source_channel_id": platform_channel_id,
    "source_user_id": global_user_id,
    "execute_at": {"$gte": current_timestamp},
}
```

   - Sort by `execute_at` ascending, limit to `limit`, and project out `_id`.
   - Do not add write, cancel, reschedule, or scheduler execution helpers.
   - Implement `RecallAgent(BaseRAGHelperAgent)` with no cache namespace.
   - Add the four collectors defined in `Fixed Collector Interfaces`.
   - Implement deterministic candidate ranking and result shaping.
   - Keep `max_attempts` accepted for interface compatibility but unused beyond `attempts=1`.
   - Rerun `tests/test_rag_recall_agent.py`. If it fails because the module
     contract is incomplete, fix only module tests or module implementation to
     match this plan, then rerun before touching integration.

4. **Implement integration after module tests pass**

   - Add `conversation_progress` and `conversation_episode_state` to the
     `call_rag_supervisor` context in `persona_supervisor2.py`.
   - Register `recall_agent` and `Recall:` prefix.
   - Add initializer rule:

```text
Use Recall when the user asks what was agreed, promised, planned, left unresolved,
or where the current episode left off.
```

   - Add examples:
     - "还记得今天的约定么？" -> Recall.
     - "我们刚才说到哪儿了？" -> Recall.
     - "你答应过我什么来着？" -> Recall.
     - "我们是什么时候约好的？" -> Recall with `exact_agreement_history`.
     - "谁说过'约定就是约定'？" -> Conversation-keyword, not Recall.
   - Increment initializer prompt version.
   - Add `recall_evidence` to `project_known_facts`.
   - Do not edit cognition L2/L3 files.
   - Verify that the existing cognition public `rag_result` copy path does not
     strip `rag_result.answer` or `rag_result.recall_evidence`.
   - Update consolidator facts/evaluator prompts to include recall evidence authority rules.
   - Update RAG README helper list.
   - Rerun deterministic integration tests and prompt-render checks. If
     integration exposes a Recall module contract problem, return to steps 1 and 3
     before changing more integration code.

5. **Run live checks and final verification**

   - Run each live LLM test one at a time and inspect/save its trace before the
     next live test.
   - Run all deterministic, compile, static grep, and prompt-render checks in
     `Verification`.
   - Record each command and outcome in `Execution Evidence`.

## Progress Checklist

- [x] Stage 1 — module contract tests and pre-change failure captured.
  - Covers: implementation order step 1.
  - Files/modules: `tests/test_rag_recall_agent.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_rag_recall_agent.py -q` before production edits.
  - Evidence: record expected missing-module/helper failures in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-02` after verification and evidence are recorded.
- [x] Stage 2 — integration tests and current behavior captured.
  - Covers: implementation order step 2.
  - Files/modules: `tests/test_persona_supervisor2_rag2_integration.py`, `tests/test_rag_projection.py`, `tests/test_rag_initializer_cache2.py`, `tests/test_rag_recall_live_llm.py`.
  - Verify: run practical deterministic integration tests and the first live initializer test before integration edits.
  - Evidence: record expected missing-field/prefix failures and current non-Recall live route in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-02` after verification and evidence are recorded.
- [x] Stage 3 — Recall module and read-only support complete.
  - Covers: implementation order step 3.
  - Files/modules: `src/kazusa_ai_chatbot/rag/recall_agent.py`, `src/kazusa_ai_chatbot/db/scheduled_events.py`, `src/kazusa_ai_chatbot/db/__init__.py`, `tests/test_rag_recall_agent.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_rag_recall_agent.py -q` and `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\recall_agent.py src\kazusa_ai_chatbot\db\scheduled_events.py`.
  - Evidence: record module test output for all Recall source arbitration and output-cap cases in `Execution Evidence`.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-02` after verification and evidence are recorded.
- [x] Stage 4 — RAG2 integration and projection complete.
  - Covers: implementation order step 4.
  - Files/modules: `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`, `src/kazusa_ai_chatbot/rag/cache2_policy.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`, `src/kazusa_ai_chatbot/rag/README.md`.
  - Verify: deterministic integration/projection/cache tests, prompt-render checks, and cognition no-touch git check listed in `Verification`.
  - Evidence: record context payload, prompt version, route prefix, projection, consolidator prompt-render, README, and no-touch outputs in `Execution Evidence`.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-02` after verification and evidence are recorded.
- [x] Stage 5 — live checks and final verification complete.
  - Covers: implementation order step 5.
  - Files/modules: `tests/test_rag_recall_live_llm.py`, `test_artifacts/llm_traces/`, all changed production/test files.
  - Verify: run live tests one at a time, then all deterministic pytest, `py_compile`, static grep, and prompt-render commands listed in `Verification`.
  - Evidence: record each live trace path plus final command outputs and any skipped/blocked checks in `Execution Evidence`.
  - Handoff: implementation complete.
  - Sign-off: `Codex/2026-05-02` after verification and evidence are recorded.

## Verification

### Deterministic Tests

Run:

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_recall_agent.py -q
venv\Scripts\python.exe -m pytest tests\test_rag_projection.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_initializer_cache2.py -q
```

Required deterministic cases:

- Missing mandatory scope fields return `resolved=False` and do not query DB.
- RAG2 context plumbing passes `conversation_progress` and
  `conversation_episode_state`.
- Active progress answers "today's agreement" over older history.
- `continuity == "sharp_transition"` suppresses active progress.
- Missing `conversation_progress` prevents active-episode certainty but still
  permits durable commitment recall when active memory units exist.
- Active commitment answers durable "what did you promise" query.
- Completed/cancelled memory units are excluded.
- Pending scheduled event is included for executable future action.
- Exact wording request is not a Recall initializer route.
- `project_known_facts` places Recall result into `recall_evidence` and preserves `supervisor_trace`.
- Existing cognition payload shaping preserves public `rag_result.answer` and does not remove `rag_result.recall_evidence`.
- Consolidator prompt test proves `recall_evidence` is visible and progress-only recall is not listed as standalone durable character-fact authority.

### Real LLM Tests

Run one at a time and inspect logs before proceeding:

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_recall_live_llm.py::test_live_initializer_routes_active_agreement_to_recall -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_recall_live_llm.py::test_live_initializer_keeps_exact_phrase_on_conversation_keyword -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_recall_live_llm.py::test_live_rag2_recall_answers_today_agreement -q -s -m live_llm
```

Each live test must write a trace under `test_artifacts/llm_traces/` containing input, raw initializer output, parsed slots, dispatched agent, Recall result when applicable, final RAG2 answer, and agent judgment.

### Static And Prompt Checks

Run:

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\recall_agent.py src\kazusa_ai_chatbot\db\scheduled_events.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py
```

Also run a prompt-render check for edited prompts in:

- `persona_supervisor2_rag_supervisor2.py`
- `persona_supervisor2_consolidator_facts.py`

### Cognition No-Touch Check

Run:

```powershell
git diff --name-only -- src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py
```

This command must produce no output. If either cognition file appears, revert
only the implementation agent's cognition edits and rerun the focused tests. If
the implementation cannot pass without cognition edits, stop and revise this
plan with user approval.

### Static Grep Checks

Run:

```powershell
rg "Recall:" src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py src\kazusa_ai_chatbot\rag\README.md tests
rg "recall_evidence" src\kazusa_ai_chatbot tests
rg "RecallAgent|recall_agent" src\kazusa_ai_chatbot tests
rg "query_pending_scheduled_events" src\kazusa_ai_chatbot tests
```

Also run this negative check and record that it returns no production Recall
cache namespace:

```powershell
rg "recall.*cache|cache.*recall|volatile_recall" src\kazusa_ai_chatbot tests
```

The negative check is allowed to find the literal `volatile_recall` reason in
`recall_agent.py` and tests. It must not find a Cache2 namespace, persistent
cache key, or cache policy entry for Recall.

## Execution Evidence

Record execution evidence here during implementation. Do not tick checklist
items until the matching evidence is present.

- Stage 1 evidence: `venv\Scripts\python.exe -m pytest tests\test_rag_recall_agent.py -q` failed before production edits with `ImportError: cannot import name 'recall_agent' from 'kazusa_ai_chatbot.rag'`, proving the missing module entrypoint.
- Stage 2 evidence: `venv\Scripts\python.exe -m pytest tests\test_rag_projection.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_initializer_cache2.py -q` produced 6 expected/baseline failures and 20 passes before integration edits. New expected failures covered missing `recall_evidence`, missing RAG2 progress context, missing `Recall:` prompt route, and prompt-version mismatch. The same run also surfaced an existing adjacent prompt-version expectation: `test_initializer_prompt_version_bumps_to_v11_for_live_fact_contract` expects `initializer_prompt:v11` while code is `v7`.
- Stage 2 live evidence: `venv\Scripts\python.exe -m pytest tests\test_rag_recall_live_llm.py::test_live_initializer_routes_active_agreement_to_recall -q -s -m live_llm` failed as expected before prompt edits. The live initializer routed to `Memory-search: search persistent memory for evidence relevant to answering the question about 今天的约定`; trace written to `test_artifacts/llm_traces/rag_recall_live_llm__active_agreement.json`.
- Stage 3 evidence: `venv\Scripts\python.exe -m pytest tests\test_rag_recall_agent.py -q` passed 9 tests covering missing scope, progress-first recall, durable commitments, scheduled events, exact-history gate, stale progress, inactive memory exclusion, output caps, and scheduled-event query scoping.
- Stage 3 compile evidence: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\recall_agent.py src\kazusa_ai_chatbot\db\scheduled_events.py` passed.
- Stage 4 evidence: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\recall_agent.py src\kazusa_ai_chatbot\db\scheduled_events.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py tests\test_rag_recall_live_llm.py tests\test_consolidator_facts_rag2.py tests\test_rag_initializer_cache2.py` passed. Runtime prompt render passed for the RAG2 initializer, dispatcher, facts harvester, and fact harvester evaluator prompts.
- Stage 4 deterministic evidence: `venv\Scripts\python.exe -m pytest tests\test_rag_projection.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_initializer_cache2.py tests\test_consolidator_facts_rag2.py -q` passed 28 tests. This covered RAG2 progress context plumbing, `recall_evidence` projection, cognition public payload pass-through without L2/L3 edits, `Recall:` prompt route, `initializer_prompt:v12`, consolidator recall authority wording, and progress-only recall restrictions.
- Stage 4 module/regression evidence: `venv\Scripts\python.exe -m pytest tests\test_rag_recall_agent.py -q` passed 9 tests after integration.
- Stage 4 static evidence: `rg "Recall:" ...`, `rg "recall_evidence" ...`, `rg "RecallAgent|recall_agent" ...`, `rg "query_pending_scheduled_events" ...`, and `rg "recall.*cache|cache.*recall|volatile_recall" ...` returned the expected Recall route, projection, helper, scheduled-event helper, and no persistent Recall cache namespace beyond the allowed `volatile_recall` literal.
- Stage 5 live evidence: each real LLM test was run individually and inspected before the next run. `test_live_initializer_routes_active_agreement_to_recall` passed and wrote `test_artifacts/llm_traces/rag_recall_live_llm__active_agreement__20260502T045951245970Z.json`; the initializer emitted `Recall: retrieve active_episode_agreement relevant to today's agreement`. `test_live_initializer_keeps_exact_phrase_on_conversation_keyword` passed and wrote `test_artifacts/llm_traces/rag_recall_live_llm__exact_phrase.json`; the initializer emitted `Conversation-keyword: find messages containing '约定就是约定'` and no Recall slot. `test_live_rag2_recall_answers_today_agreement` passed and wrote `test_artifacts/llm_traces/rag_recall_live_llm__rag2_today_agreement.json`; the full RAG2 path used `recall_agent`, selected `conversation_progress`, and answered `记得，用户将在 9:30 接走该角色。`.
- Stage 5 final deterministic evidence: `venv\Scripts\python.exe -m pytest tests\test_rag_recall_agent.py -q` passed 9 tests. `venv\Scripts\python.exe -m pytest tests\test_rag_projection.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_initializer_cache2.py tests\test_consolidator_facts_rag2.py -q` passed 28 tests.
- Stage 5 final static evidence: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\recall_agent.py src\kazusa_ai_chatbot\db\scheduled_events.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py` passed. Runtime prompt render passed for the edited RAG2 and consolidator prompts. Cognition no-touch check produced no output. `git diff --check` reported only CRLF warnings, no whitespace errors.

## Acceptance Criteria

- `还记得今天的约定么` routes to `Recall:` under the real initializer.
- Exact quote/provenance queries still route to conversation keyword/filter agents.
- Recall returns one reconciled selected summary and provenance, not parallel unreconciled source dumps.
- Recall v1 performs no internal LLM calls and does not use Cache2.
- RAG2 finalizer can answer active agreement recall from Recall evidence.
- Cognition L2/L3 files remain unchanged, and cognition continues to receive the direct recall answer through existing `rag_result.answer`.
- Consolidator does not treat progress-only recall as durable character fact authority.
- Existing non-Recall RAG2 tests continue to pass.
- No relevance behavior changes.

## Risks

| Risk                                                        | Mitigation                                                                                                           |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Initializer starts routing broad memory questions to Recall | Keep examples narrow and add negative tests for exact phrases, world facts, profile impressions, and live facts.     |
| Recall duplicates `conversation_progress` inside cognition  | Do not edit cognition prompts; use existing `rag_result.answer` for the direct answer path and keep `conversation_progress` as flow guidance. |
| Progress-derived recall pollutes durable memory             | Update consolidator authority rules and tests.                                                                       |
| Scheduled-event query leaks across channels/users           | Add strict DB helper scoping tests.                                                                                  |
| History search makes Recall slow/noisy                      | Do not run history by default; keep it bounded and proof-only.                                                       |
| Cache stale results                                         | Disable Recall cache in v1.                                                                                          |

## Rollback / Recovery

Rollback is safe and code-only:

1. Remove `Recall:` routing examples from the initializer prompt and bump prompt version again.
2. Remove dispatcher mapping and registry entry for `recall_agent`.
3. Stop passing progress fields to RAG2 context.
4. Keep `rag_result.recall_evidence` as an empty default until a later cleanup.
5. Run RAG2 integration tests and confirm exact phrase/history/memory routes behave as before.

No database cleanup is required.

## Glossary

- Recall: RAG2 helper capability for active agreement, promise, plan, open-loop, and current-episode recall.
- Conversation progress: short-lived operational episode state loaded after relevance and before persona cognition.
- Active commitment: durable user memory unit representing a still-active accepted commitment or agreement.
- Scheduled event: pending executable future task stored in `scheduled_events`.
- Conversation history: literal persisted transcript evidence.
- Progress-only recall: recall evidence whose primary source is `conversation_progress` and not durable memory, transcript proof, or external evidence.
