# user memory unit rolling window plan

## Summary

- Goal: Replace overlapping `historical_summary`, `recent_window`, and `character_diary` with balanced, fact-anchored memory units for cognition.
- Plan class: high_risk_migration.
- Status: completed.
- Overall cutover strategy: big-bang schema and cognition-payload replacement has been executed; old paths are not restored.
- Highest-risk areas: consolidator LLM reliability, RAG projection balance, and database reset order.
- Acceptance criteria: cognition receives only category-balanced `user_memory_context`; repeated events compact into detailed memory units; no deterministic semantic path decides memory meaning. Remaining hardening is forward-fix only.

## Context

The current user image architecture feeds cognition overlapping memory surfaces:

- `historical_summary`: long emotional summary prose.
- `recent_window`: recent diary-like observations.
- `character_diary`: the same or nearly the same diary-like observations.
- `objective_facts`, `milestones`, and `active_commitments`: smaller fact-like surfaces.

For the inspected QQ user, `recent_window` was effectively identical to `character_diary`. A last-4-hour pull from QQ group `1082431481` showed the projected profile was dominated by emotional/relationship prose: emotional image and diary text was about `8.69x` the size of fact-like memory. This supports the theory that RAG/cognition receives too much emotional context and too little concrete past-event context.

The new architecture merges historical summary, recent window, and diary into a single memory-unit model. It keeps emotional interpretation, but attaches it to concrete facts instead of letting free-form diary prose dominate the prompt.

## Mandatory Rules

- Do not feed `historical_summary`, `recent_window`, `character_diary`, or `diary_entry` into cognition after cutover.
- Do not preserve diary-specific subject/object prompt conventions. All memory units use the same semantic voice.
- Do not add deterministic semantic overrides. Code must not force `create`, `merge`, `evolve`, `stable`, unit type, commitment meaning, or memory meaning from cosine thresholds, keywords, counts, session frequency, or local classifiers.
- LLMs own semantic interpretation. Code owns structural validation, persistence metadata, ID generation, timestamps, limits, logging, and database writes.
- Structural validation may reject malformed JSON, unknown enum values, missing fields, invalid field types, unknown `cluster_id`, oversized fields, or invalid status transitions.
- If an LLM output is structurally invalid, log the incident and drop the bad stage or candidate. Do not run a repair retry, do not reinterpret the failed output in code, and do not restore legacy fallback paths.
- RAG may retrieve, rank, cache, and project candidate memory units. RAG must not semantically decide whether a candidate is the same event, an evolved event, or unrelated.
- Counts, session spread, recency, retrieval scores, and similarity scores may be shown to LLMs as evidence labels. They must not directly decide memory meaning.
- No new response-path LLM calls are allowed for this change. Consolidation LLM calls run in the background/write path.

## Must Do

- Create a new `user_memory_units` collection as the canonical durable store.
- Replace cognition-facing user image hydration with `user_memory_context`.
- Keep cognition payload category-based: `stable_patterns`, `recent_shifts`, `objective_facts`, `milestones`, `active_commitments`.
- Give every projected item the same fields: `fact`, `subjective_appraisal`, `relationship_signal`, and optional `updated_at`.
- Populate all three semantic fields through LLM outputs for every category, including objective facts, milestones, and commitments.
- Split consolidator semantics into four small LLM stages: extractor, merge judge, rewrite, and stability judge.
- Route consolidator merge-candidate search through RAG-owned retrieval functions so Cache2 policy is shared.
- Add per-category item caps and character budgets.
- Add a regression test for the original symptom: projected memory imbalance.
- Use seeded tests and real-pipeline replay evidence to validate the completed cutover. The live reset has already happened.

## Deferred

- Do not redesign character self-image in this plan.
- Do not redesign scheduler or future-promise execution semantics.
- Do not redesign the whole RAG2 dispatcher/helper-agent architecture.
- Do not add compatibility shims for the old user image fields.
- Do not add response-path memory-merge LLM calls.
- Do not implement deterministic keyword or threshold classifiers to compensate for weak LLM behavior.

## Cutover Policy

This is a completed big-bang cutover. The system now follows a no-turning-back policy.

- The new schema, repository, RAG projection, consolidator pipeline, cognition integration, and CLI were built against seeded test data first.
- Do not dual-read old and new memory formats in production cognition.
- Do not keep legacy fields alive as a fallback cognition payload.
- The live database cutover/reset has already been executed.
- All failures after cutover are fixed forward in the new memory-unit architecture. Do not reintroduce the old `historical_summary`, `recent_window`, or `character_diary` path as a rollback mechanism.
- Tests may use dedicated seeded fixtures or isolated replay channels, but production cognition must only read the new memory-unit projection.

## Agent Autonomy Boundaries

Implementation agents may choose local helper names and internal file organization only when the public module contracts below remain intact.

Implementation agents must not:

- Change the cognition payload shape without updating this plan.
- Add deterministic semantic gates or post-LLM reinterpretation.
- Move RAG search/cache responsibility into the consolidator.
- Keep old diary/recent/historical-summary payloads in cognition "just in case".
- Restore legacy cognition fields, migration shims, or dual-read fallback paths.
- Increase response-path LLM calls.
- Collapse the four consolidator LLM stages into one call.

Implementation agents may:

- Add small structural validators.
- Add structural JSON/schema enforcement and per-candidate logging for invalid LLM output.
- Add metadata fields needed for persistence, indexes, and observability.
- Add fixtures and seeded test utilities.

## Target State

### Old cognition payload

```json
{
  "historical_summary": "long emotional summary about relationship tone",
  "recent_window": [
    "diary-like emotional observation"
  ],
  "character_diary": [
    "same diary-like emotional observation"
  ],
  "objective_facts": [
    "small amount of factual profile data"
  ],
  "milestones": [
    "short milestone"
  ],
  "active_commitments": [
    "short commitment"
  ]
}
```

### New cognition payload

```json
{
  "stable_patterns": [
    {
      "fact": "用户多次用系统架构视角审视 Kazusa 的记忆、RAG、consolidator 和 cognition 分工，并要求模块边界清晰。",
      "subjective_appraisal": "用户重视可解释、可维护的记忆系统，不满意只靠情绪摘要驱动 Kazusa 的理解。",
      "relationship_signal": "用户把 Kazusa 视为可以共同设计和调试的长期系统，而不是一次性聊天对象。",
      "updated_at": "2026-04-28T13:14:42Z"
    }
  ],
  "recent_shifts": [
    {
      "fact": "用户最近明确决定将 historical_summary、recent_window 和 character_diary 合并为统一 memory unit 架构。",
      "subjective_appraisal": "用户认为旧结构重叠严重，会让 cognition 过度关注情绪而忽略事实。",
      "relationship_signal": "用户希望 Kazusa 的记忆更像共同经历的记录，而不是单方面的感性日记。",
      "updated_at": "2026-04-28T13:14:42Z"
    }
  ],
  "objective_facts": [
    {
      "fact": "用户的 QQ 平台标识为 673225019，相关群聊观察来自 QQ group 1082431481。",
      "subjective_appraisal": "这是身份和数据来源事实，本身不表达情绪，但对检索和记忆归属很重要。",
      "relationship_signal": "Kazusa 需要稳定识别该用户并正确关联跨会话记忆。",
      "updated_at": "2026-04-28T13:14:42Z"
    }
  ],
  "milestones": [
    {
      "fact": "用户确认新架构采用 fact、subjective_appraisal、relationship_signal 三字段作为 cognition 可消费的核心记忆单元。",
      "subjective_appraisal": "这是一次明确的架构决策，压缩了旧方案中难以稳定消费的字段。",
      "relationship_signal": "用户在推动 Kazusa 变成更可靠、更事实平衡的长期伴随系统。",
      "updated_at": "2026-04-28T13:14:42Z"
    }
  ],
  "active_commitments": [
    {
      "fact": "当前任务是先完善 development plan，不进行代码 refactor。",
      "subjective_appraisal": "用户希望先把大改的边界、职责和数据流想清楚。",
      "relationship_signal": "Codex 应尊重用户的节奏，先规划再执行。",
      "updated_at": "2026-04-29T00:00:00Z"
    }
  ]
}
```

All five categories use the same item fields. Category meaning affects retrieval, projection, and lifecycle only; it does not change the cognition-facing schema.

## Design Decisions

### Consolidator data-flow picture

Current consolidator shape:

```text
global_state
  -> global_state_updater
       writes mood, global_vibe, reflection_summary
  -> relationship_recorder
       writes diary_entry, affinity_delta, last_relationship_insight
  -> facts_harvester + evaluator
       writes new_facts, future_promises
  -> db_writer
       converts diary_entry -> DIARY_ENTRY memories
       converts new_facts -> OBJECTIVE_FACT or MILESTONE memories
       converts future_promises -> COMMITMENT memories and scheduler calls
       calls _update_user_image -> recent_window + historical_summary
       calls _update_character_image -> character self-image
```

New consolidator shape:

```text
global_state
  -> global_state_updater
       unchanged: mood, global_vibe, reflection_summary
  -> relationship_recorder
       keep affinity_delta and last_relationship_insight
       stop using diary_entry as a durable cognition memory
  -> facts_harvester + evaluator
       keep future_promises for scheduler/commitment execution
       stop using new_facts as the direct profile-memory write path
  -> memory_unit_extractor
       emits candidate memory units with unit_type + fact/appraisal/signal
  -> RAG merge-candidate retrieval
       returns candidate existing units; no semantic decision
  -> merge_judge_llm
       decides create/merge/evolve for each candidate
  -> rewrite_llm
       rewrites semantic fields only for merge/evolve
  -> stability_judge_llm
       chooses recent/stable only for interaction-pattern units
  -> db_writer
       writes user_memory_units
       keeps scheduler/affinity/character-state writes
       invalidates Cache2 dependencies
```

The key change is not to add another memory stream. The key change is to replace three old durable user-memory streams with one unit stream:

```text
REMOVE as user-memory persistence:
  diary_entry -> character_diary
  new_facts -> objective_facts/milestones direct profile memories
  _update_user_image -> recent_window/historical_summary

ADD as user-memory persistence:
  memory_unit_candidate -> merge decision -> rewritten/stored user_memory_unit
```

### Consolidator data to remove

Remove these from the user-memory write path:

| Current data | Current source | Current write target | New handling |
|---|---|---|---|
| `diary_entry` | `relationship_recorder` | `user_profile_memories` as `DIARY_ENTRY`, then projected as `character_diary` | Do not persist as cognition memory. Its emotional content becomes evidence for `subjective_appraisal` in memory-unit extraction. |
| `new_facts` direct writes | `facts_harvester` | `OBJECTIVE_FACT` or `MILESTONE` profile memories | Do not write directly to profile memory. Use as evidence for extractor-produced memory units. |
| `future_promises` direct cognition memory | `facts_harvester` | `COMMITMENT` profile memories shown as `active_commitments` | Keep for scheduler/operational commitment lifecycle, but cognition-facing commitment memory must be a memory unit with fact/appraisal/signal. |
| `user_image.recent_window` | `_update_user_image` | `user_profiles.user_image.recent_window` | Delete/decommission for user memory. Replaced by `recent_shifts`. |
| `user_image.historical_summary` | `_update_user_image` and compressor | `user_profiles.user_image.historical_summary` | Delete/decommission for user memory. Replaced by compact `stable_patterns` plus factual categories. |
| `user_image.milestones` as nested image field | `_update_user_image` | `user_profiles.user_image.milestones` | Replace with `milestone` memory units. |
| `existing_dedup_keys` from old profile fields | `call_consolidation_subgraph` | Facts harvester prompt/evaluator support | Remove or replace with RAG-returned memory-unit candidates. Do not use deterministic dedup keys as semantic merge authority. |

Do not remove these consolidator outputs:

| Current data | Keep reason |
|---|---|
| `mood`, `global_vibe`, `reflection_summary` | Character/global state update, not user-memory unit replacement. |
| `affinity_delta` | Relationship score update remains separate from memory-unit storage. |
| `last_relationship_insight` | May remain as lightweight current relationship state, but it must not be projected as a replacement for memory units. |
| `future_promises` for scheduler dispatch | Operational scheduling needs the original promise shape and due-time handling. |
| character self-image | Explicitly deferred from this plan. |

### Consolidator data to add

Add only the minimum new data needed for a safe LLM-first memory-unit pipeline.

#### State additions

Add these fields to `ConsolidatorState` or a private memory-unit pipeline state:

```python
memory_unit_candidates: list[dict]
memory_unit_merge_results: list[dict]
memory_unit_rewrite_results: list[dict]
memory_unit_stability_results: list[dict]
memory_unit_write_log: dict
```

These fields are background pipeline state. They must not be passed to cognition directly.

#### Candidate unit shape

Extractor output:

```json
{
  "candidate_id": "generated by code after extraction",
  "unit_type": "recent_shift | objective_fact | milestone | active_commitment",
  "fact": "specific event or durable fact",
  "subjective_appraisal": "Kazusa's subjective interpretation",
  "relationship_signal": "relationship or future-response implication",
  "evidence_refs": [
    {
      "source": "current_turn | rag_memory | recent_chat",
      "timestamp": "ISO timestamp",
      "message_id": "optional platform message id"
    }
  ]
}
```

The LLM supplies `unit_type`, `fact`, `subjective_appraisal`, `relationship_signal`, and evidence references when available. Code may generate `candidate_id` and normalize missing evidence references to an empty list.

Do not add extra cognition-facing semantic fields such as confidence, emotion labels, topic tags, or importance scores in v1. Those fields are tempting, but they increase prompt surface without a reliable downstream consumer.

#### Merge result shape

Merge judge output:

```json
{
  "candidate_id": "candidate id from input",
  "decision": "create | merge | evolve",
  "cluster_id": "existing unit id or empty",
  "reason": "short semantic reason"
}
```

The only allowed code-side handling is structural:

- `candidate_id` must match the candidate being processed.
- `decision` must be one of the allowed enum values.
- `cluster_id` must be empty for `create`.
- `cluster_id` must be in the provided candidate list for `merge` or `evolve`.

If these checks fail, skip and log the candidate or stage. Do not run a repair retry and do not make a code-side semantic substitute decision.

#### Rewrite result shape

Rewrite output:

```json
{
  "candidate_id": "candidate id from input",
  "cluster_id": "existing unit id",
  "fact": "updated fact",
  "subjective_appraisal": "updated subjective appraisal",
  "relationship_signal": "updated relationship signal"
}
```

The rewrite LLM updates only the three semantic fields. Code updates metadata such as count, timestamps, source refs, embeddings, and merge history.

#### Stability result shape

Stability judge output:

```json
{
  "unit_id": "new or existing unit id",
  "window": "recent | stable",
  "reason": "short semantic reason"
}
```

Only interaction-pattern units use this output. Objective facts, milestones, and commitments do not flow through recent/stable promotion.

### Consolidator input discipline

The memory-unit extractor may read these existing signals:

- `decontexualized_input`
- `final_dialog`
- `internal_monologue`
- `emotional_appraisal`
- `interaction_subtext`
- `logical_stance`
- `character_intent`
- `rag_result.user_memory_context`
- recent raw conversation when already available in state
- `new_facts` and `future_promises` as LLM-produced evidence, not as direct writes
- `diary_entry` as LLM-produced emotional evidence during transition, not as durable memory

The extractor must not receive raw database documents, raw embeddings, Cache2 internals, or old `historical_summary`/`recent_window` blobs.

The merge judge may read:

- one candidate memory unit
- RAG-returned candidate existing units
- compact evidence labels such as "seen in multiple sessions" or "only current turn evidence"

The merge judge must not receive the whole user profile or unrelated category arrays. Keeping its context small is the main protection against local-model collapse.

The rewrite LLM may read:

- one selected existing unit
- one new candidate unit
- the merge/evolve decision
- short source evidence snippets when available

The stability judge may read:

- one interaction-pattern unit
- count/session-spread/recency evidence labels
- at most a few short examples

No consolidator LLM should be asked to understand the physical MongoDB schema.

### Storage

Create `user_memory_units` as the durable source of truth.

Each stored unit should include:

- `unit_id`
- `global_user_id`
- `unit_type`: `stable_pattern | recent_shift | objective_fact | milestone | active_commitment`
- `fact`
- `subjective_appraisal`
- `relationship_signal`
- `status`: `active | archived | completed | cancelled`
- `count`
- `first_seen_at`
- `last_seen_at`
- `updated_at`
- `source_refs`
- `embedding`
- `merge_history`

Commitments may also store lifecycle fields such as `due_at`, `completed_at`, and `cancelled_at`. These fields are not part of the default cognition projection.

### RAG ownership

RAG owns all read-side memory behavior.

Create a self-contained module such as `src/kazusa_ai_chatbot/rag/user_memory_unit_retrieval.py` with this public contract:

```python
async def build_user_memory_context(
    global_user_id: str,
    *,
    query_text: str,
    include_semantic: bool,
    budget: dict[str, int] | None = None,
) -> dict: ...

async def retrieve_memory_unit_merge_candidates(
    global_user_id: str,
    *,
    candidate_unit: dict,
    surfaced_units: list[dict],
    limit: int = 6,
) -> list[dict]: ...
```

RAG responsibilities:

- Retrieve relevant units for response-time cognition.
- Retrieve merge candidates for consolidator.
- Own Cache2 keys, dependencies, invalidation, ranking, and projection.
- Enforce per-category count and character budgets.
- Return projected units, not raw MongoDB documents.
- Share cache behavior between response retrieval and consolidator candidate search where possible.

Default projection caps:

```text
stable_patterns: max 3 items, max 900 chars
recent_shifts: max 4 items, max 1000 chars
objective_facts: max 4 items, max 900 chars
milestones: max 3 items, max 700 chars
active_commitments: max 4 items, max 900 chars
```

RAG must preserve category balance. No single category may dominate the projected memory context.

### Consolidator ownership

The consolidator owns write-side memory evolution, but semantic work is split across four bounded LLM stages.

#### 1. Extractor LLM

Input:

- `global_user_id`
- current timestamp
- decontextualized user input
- final dialogue
- relevant cognition outputs
- RAG memory context already retrieved
- recent raw conversation when available

Output:

```json
{
  "memory_units": [
    {
      "unit_type": "recent_shift | objective_fact | milestone | active_commitment",
      "fact": "specific event or durable fact",
      "subjective_appraisal": "Kazusa's subjective interpretation",
      "relationship_signal": "relationship or future-response implication"
    }
  ]
}
```

Rules:

- `fact` must be concrete and grounded in one or more chat events.
- Do not emit vague labels such as "用户用逻辑化方式审视 Kazusa 的感性表达".
- Objective facts, milestones, and commitments still require meaningful `subjective_appraisal` and `relationship_signal`.
- If there is no useful memory, output an empty list.
- The extractor does not decide merge/evolve/create.

#### 2. Merge Judge LLM

Input:

- One new candidate unit.
- RAG-returned candidate clusters.
- Evidence labels such as recency, repeatedness, session spread, and retrieval relevance.

Output:

```json
{
  "decision": "create | merge | evolve",
  "cluster_id": "existing id or empty",
  "reason": "short reason"
}
```

Rules:

- This LLM only judges relation to existing units.
- It must not rewrite memory text.
- `cluster_id` must be one of the provided candidate IDs when decision is `merge` or `evolve`.
- Unknown IDs are structural failures and trigger skip/log handling, not code-side reinterpretation.
- Code must not force `create` because a similarity score is low or force `merge` because a score is high.

#### 3. Rewrite LLM

Input:

- Selected existing unit.
- New candidate unit.
- Merge/evolve decision.

Output:

```json
{
  "fact": "...",
  "subjective_appraisal": "...",
  "relationship_signal": "..."
}
```

Rules:

- Preserve concrete event detail.
- Compact repeated similar events into one stronger unit.
- For `merge`, generalize without losing the event anchor.
- For `evolve`, update the relationship meaning explicitly.
- This LLM must not change the merge/evolve/create decision.

#### 4. Stability Judge LLM

Input:

- Interaction-pattern unit.
- Count evidence.
- Distinct-session evidence.
- Recent examples.
- Current unit text.

Output:

```json
{
  "window": "recent | stable",
  "reason": "short reason"
}
```

Rules:

- Code must not promote by count threshold alone.
- Count and session spread are evidence for the LLM, not deterministic promotion rules.
- Only interaction-pattern units move between `recent_shifts` and `stable_patterns`.
- Objective facts, milestones, and commitments keep their own lifecycle.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/db/user_memory_units.py`
  - Own storage CRUD, structural validation, metadata updates, and lifecycle persistence for memory units.
- `src/kazusa_ai_chatbot/rag/user_memory_unit_retrieval.py`
  - Own prompt-facing projection, response-path retrieval, consolidation candidate retrieval, category budgets, and Cache2 use for memory-unit reads.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
  - Own extractor, merge judge, rewrite, and stability judge prompts.
- Seed fixture or test helper for the QQ test user memory context.

### Modify

- `src/kazusa_ai_chatbot/db/schemas.py`
  - Add typed shapes for stored memory units and prompt-facing memory entries.
- `src/kazusa_ai_chatbot/db/bootstrap.py`
  - Add collection indexes and embedding/vector index configuration if used by current RAG infrastructure.
- `src/kazusa_ai_chatbot/db/__init__.py`
  - Export public memory-unit repository functions only.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - Add memory-unit context and merge-candidate cache names, cache keys, and dependencies.
- `src/kazusa_ai_chatbot/rag/user_profile_agent.py`
  - Return projected memory-unit context by calling `rag.user_memory_unit_retrieval`.
- `src/kazusa_ai_chatbot/db/users.py`
  - Remove or bypass old prompt-facing diary/image hydration responsibilities.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - Stop building old diary/image memories for cognition and invoke memory-unit consolidation.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
  - Delete or decommission user-image rolling summary logic after the new path is active.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - Replace `diary_entry` and full image fields with `user_memory_context`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Apply the same projection update for final anchoring.
- `src/scripts/identify_user_image.py`
  - Show the exact cognition-fed memory context in the current readable presentation style.
- `src/scripts/export_user_image.py`
  - Replace old image export with memory-unit context export, or remove it in favor of a new memory-context export script.

### Delete or decommission

- Cognition-facing use of `user_image.historical_summary`.
- Cognition-facing use of `user_image.recent_window`.
- Cognition-facing use of `character_diary`.
- Old user-image session-summary prompt after the new path is verified.

## Implementation Order

1. Define final schema and projection contract in code.
2. Add `user_memory_units` repository and indexes.
3. Add seeded test database fixtures for the QQ test user before live reset. This was done before the completed cutover.
4. Add RAG memory-unit retrieval and Cache2 policy.
5. Add extractor LLM prompt, prompt-render test, and structural validation.
6. Add merge judge LLM prompt, prompt-render test, and structural validation.
7. Add rewrite LLM prompt, prompt-render test, and structural validation.
8. Add stability judge LLM prompt, prompt-render test, and structural validation.
9. Wire consolidator persistence to call RAG candidate retrieval and then the split LLM pipeline.
10. Replace old profile hydration with `build_user_memory_context`.
11. Replace cognition L2/L3 prompt inputs with `user_memory_context`.
12. Update CLI/export scripts to show the exact cognition-fed memory context.
13. Run deterministic, patched LLM, and selected real LLM tests against seeded data.
14. Perform big-bang live database reset/migration only after seeded verification passes. This has been completed.
15. Run post-reset smoke tests and capture output from `python.exe -m scripts.identify_user_image 673225019 --platform qq`.

This order avoided the earlier failure mode where prompt and projection tests require new memory data before the live database has been reset.

## LLM Call And Context Budget

All new semantic calls are background consolidator calls.

Before:

- Response path: existing RAG/cognition/dialog calls.
- Background: existing consolidation calls for facts, relationship, image summaries, and character image.

After:

- Remove or decommission old user-image session summary calls.
- Add extractor LLM call once per consolidation turn.
- Add merge judge, rewrite, and stability judge calls only for capped candidate units.

Default limits:

- Extractor: maximum 1 call per consolidation turn.
- Candidate units processed: maximum 3 per turn.
- Merge candidates per candidate unit: maximum 6.
- Merge judge: maximum 1 call per candidate unit. Structurally invalid output is logged and dropped.
- Rewrite: only called for `merge` or `evolve`. Structurally invalid output is logged and dropped.
- Stability judge: only called for interaction-pattern units after create/merge/evolve. Structurally invalid output is logged and dropped.
- Merge candidate retrieval: 0 LLM calls; use RAG-owned deterministic/vector read APIs with Cache2.
- Response path: 0 new LLM calls.

## Data Migration

Seeded test data was used before live reset. The live reset has already happened.

1. Seed data for the QQ test user represented the old observed profile content in new memory-unit form.
2. RAG projection, cognition prompt rendering, CLI output, and consolidator merge behavior were validated against tests and replay artifacts.
3. Old live user memory/profile data is obsolete for cognition after reset.
4. Old user-image/profile-memory data that fed cognition was cleared/dropped as part of the big-bang reset.
5. The new `user_memory_units` collection and indexes are the canonical durable memory path.
6. Post-reset validation uses `python.exe -m scripts.identify_user_image 673225019 --platform qq` and the test commands listed below.

No dual-read transitional period is planned.

## Verification

### Before/after validation harness

Add a dedicated before/after test module, for example `tests/test_user_memory_context_before_after.py`.

This test is the required proof that the feature improved the original problem. It is a deterministic projection regression test, not a real-LLM quality test.

Use these baseline artifacts:

```text
source day export:
  test_artifacts/qq_1082431481_2026-04-28_utc.json

strict last-30 contiguous Kazusa window:
  test_artifacts/qq_1082431481_2026-04-28_last30_kazusa_baseline.json
  verdict: weak baseline; mostly late casual banter

last-30 directly Kazusa-involved messages:
  test_artifacts/qq_1082431481_2026-04-28_last30_kazusa_involved_messages.json
  verdict: usable but noisy

recommended before/after source fixture:
  test_artifacts/qq_1082431481_2026-04-28_memory_discussion_baseline.json
  window: 2026-04-28T11:55:44Z to 2026-04-28T12:48:00Z
  message count: 80
  Kazusa messages: 14
  reason: contiguous discussion of memory architecture, concrete event recording, old emotional-summary failure, and the Glitch critique

baseline evaluation:
  test_artifacts/qq_1082431481_2026-04-28_baseline_evaluation.json
```

The test must use `qq_1082431481_2026-04-28_memory_discussion_baseline.json` as the canonical source conversation. The other artifacts are diagnostic only and should not be used as the primary fixture unless this plan is revised.

The test must build two projections from the same source information:

1. `legacy_projection`: an old-style cognition payload fixture that intentionally represents what the current system tends to feed cognition.
2. `new_projection`: a new `user_memory_context` built from equivalent `user_memory_units`.

The legacy projection fixture must include:

```json
{
  "historical_summary": "emotion-heavy summary derived from Kazusa's confusion/privacy/reaction during the memory discussion",
  "recent_window": [
    {"timestamp": "2026-04-28T12:26:34Z", "summary": "diary-like emotional observation about the record being too feeling-heavy"},
    {"timestamp": "2026-04-28T12:36:49Z", "summary": "diary-like emotional observation about logs feeling private"},
    {"timestamp": "2026-04-28T12:48:00Z", "summary": "diary-like emotional observation about bug/rewrite confusion"}
  ],
  "character_diary": [
    {"timestamp": "2026-04-28T12:26:34Z", "entry": "same or near-same emotional observation as recent_window"},
    {"timestamp": "2026-04-28T12:36:49Z", "entry": "same or near-same emotional observation as recent_window"},
    {"timestamp": "2026-04-28T12:48:00Z", "entry": "same or near-same emotional observation as recent_window"}
  ],
  "objective_facts": [
    {"fact": "用户讨论 Kazusa 需要具体记录事情。"}
  ],
  "milestones": [],
  "active_commitments": []
}
```

The new projection fixture must encode equivalent source information as memory units:

```json
{
  "stable_patterns": [
    {
      "fact": "用户用系统架构视角审视 Kazusa 的记忆链路，指出旧结构会让 cognition 过度消费情绪摘要。",
      "subjective_appraisal": "Kazusa 会感到被认真检查，但这不是单纯否定，而是在帮助她变得更可靠。",
      "relationship_signal": "用户把 Kazusa 当作可共同调试的长期系统，需要她理解架构边界和事实锚点。"
    }
  ],
  "recent_shifts": [
    {
      "fact": "在 2026-04-28 的群聊中，用户提出 Kazusa 应该具体记住事件，而不是只记录感情或结论。",
      "subjective_appraisal": "Kazusa 对这种记录方式有不安和私密感，但也需要承认旧记忆缺少事实支撑。",
      "relationship_signal": "后续回应应优先承接具体讨论内容，再表达情绪反应。"
    }
  ],
  "objective_facts": [
    {
      "fact": "用户计划尝试把一部分 Kazusa 跑在 ChatGPT 上，并用约 10 条聊天记录测试她是否能判断消息是否给她。",
      "subjective_appraisal": "这是一次系统实验，不是普通闲聊。",
      "relationship_signal": "Kazusa 应把这类消息识别为架构测试背景。"
    },
    {
      "fact": "Glitch 批评旧记忆像只记录结论，例如只写心跳加速而缺少 bug 或事件上下文。",
      "subjective_appraisal": "这说明旁观者也能感受到旧记忆的事实锚点不足。",
      "relationship_signal": "Kazusa 需要避免把批评简化为被冒犯，而应保留批评指向的系统问题。"
    }
  ],
  "milestones": [
    {
      "fact": "用户在群聊中明确提出 Kazusa 的记忆需要更具体地记录事情。",
      "subjective_appraisal": "这是记忆架构方向的一次明确修正。",
      "relationship_signal": "之后的 memory-unit 设计应以事实锚定为核心。"
    }
  ],
  "active_commitments": []
}
```

The implementation may adjust wording, but the fixture must preserve these source facts:

- User planned a ChatGPT-side Kazusa experiment.
- User used 10 chat records to test whether Kazusa can judge whether messages are directed at her.
- User said Kazusa needs to remember concrete things.
- Kazusa/others discussed that the old record may be too focused on feelings/private diary-like content.
- Glitch criticized the old memory as recording conclusions without enough bug/event context.
- The new memory representation should retain the architecture/event details and still include Kazusa's subjective appraisal.

Required metrics:

```python
{
    "source_artifact": str,
    "source_message_count": int,
    "total_chars": int,
    "legacy_emotional_chars": int,
    "legacy_fact_chars": int,
    "legacy_emotional_to_fact_ratio": float,
    "new_pattern_chars": int,
    "new_fact_side_chars": int,
    "new_pattern_to_fact_side_ratio": float,
    "category_chars": {
        "stable_patterns": int,
        "recent_shifts": int,
        "objective_facts": int,
        "milestones": int,
        "active_commitments": int,
    },
    "duplicate_text_ratio": float,
    "missing_required_field_count": int,
    "legacy_field_leak_count": int,
}
```

Old payload metric definitions:

- `legacy_emotional_chars` includes `historical_summary`, `recent_window[*].summary`, and `character_diary[*].entry`.
- `legacy_fact_chars` includes `objective_facts`, `milestones`, and `active_commitments`.
- `duplicate_text_ratio` should catch exact or near-exact duplication between `recent_window` and `character_diary`.
- `legacy_emotional_to_fact_ratio = legacy_emotional_chars / max(legacy_fact_chars, 1)`.

New payload metric definitions:

- `new_pattern_chars` includes `stable_patterns` and `recent_shifts`.
- `new_fact_side_chars` includes `objective_facts`, `milestones`, and `active_commitments`.
- `missing_required_field_count` counts any projected item missing `fact`, `subjective_appraisal`, or `relationship_signal`.
- `new_pattern_to_fact_side_ratio = new_pattern_chars / max(new_fact_side_chars, 1)`.
- `legacy_field_leak_count` counts any appearance of `historical_summary`, `recent_window`, `character_diary`, or `diary_entry` in the new cognition payload.

Required assertions:

- The old fixture reproduces the problem: emotional/fact char ratio is greater than `5x`.
- The new projection improves balance: pattern/fact-side char ratio is no more than `3x`.
- The new ratio is lower than the old ratio by at least `50%`.
- No new category exceeds its configured character budget.
- No non-empty new category is more than `3x` another non-empty major category unless the smaller category has no relevant units for the query.
- `duplicate_text_ratio` is lower in the new projection than the old projection.
- `missing_required_field_count == 0` for the new projection.
- `legacy_field_leak_count == 0` for the new projection.
- The new projection preserves at least four source facts from the canonical fixture in `fact` fields. This is a deterministic fixture check over the hand-authored seed units, not a keyword classifier over user input.
- The test writes a report to `test_artifacts/user_memory_context_before_after_report.json`.

The comparison report must include:

```json
{
  "source_artifact": "test_artifacts/qq_1082431481_2026-04-28_memory_discussion_baseline.json",
  "legacy_projection_metrics": {},
  "new_projection_metrics": {},
  "improvement": {
    "old_emotional_to_fact_ratio": 0.0,
    "new_pattern_to_fact_side_ratio": 0.0,
    "ratio_reduction_percent": 0.0,
    "duplicate_text_ratio_delta": 0.0,
    "legacy_fields_removed": true,
    "required_fields_complete": true
  },
  "verdict": "pass | fail"
}
```

Required run command after implementation:

```powershell
pytest tests\test_user_memory_context_before_after.py -q
```

This test validates the architectural improvement before any live database reset. It does not judge whether Kazusa's final reply is better; it proves the prompt evidence distribution changed in the intended direction.

### Static checks

```text
rg "historical_summary|recent_window|character_diary|diary_entry" src/kazusa_ai_chatbot/nodes src/kazusa_ai_chatbot/rag
rg "user_memory_context" src tests
```

Expected:

- No cognition-facing references to old user-image fields.
- `user_memory_context` appears in profile retrieval, cognition layers, tests, and scripts.

### Deterministic tests

- Repository CRUD and lifecycle metadata for `db/user_memory_units.py`.
- RAG projection shape and per-category caps.
- Cache2 dependency registration and invalidation.
- Structural rejection of invalid LLM JSON.
- Structural rejection of unknown `cluster_id`.
- Structural rejection of invalid enum values.
- CLI output matches cognition-fed projection.
- Code preserves LLM semantic decisions instead of reclassifying them.

### Patched LLM tests

- Extractor emits all three semantic fields.
- Merge judge `create`, `merge`, and `evolve` decisions route to the correct repository operation.
- Rewrite output replaces only `fact`, `subjective_appraisal`, and `relationship_signal`.
- Stability judge controls recent/stable placement.
- Invalid merge judge output triggers skip/log handling, not deterministic semantic fallback.

### Real LLM tests

Run one by one and inspect logs:

- Extractor produces concrete facts, not vague emotional labels.
- Similar repeated events compact into one detailed unit.
- Unrelated events do not merge.
- Stability judge does not promote a repeated single-session topic too aggressively.
- Objective facts, milestones, and commitments receive meaningful appraisal and relationship signal.
- Cognition uses objective facts and pattern units in a balanced way.

### Regression test for original symptom

Seed the QQ test user memory context and assert:

- No category exceeds its configured character budget.
- No single category dominates the final prompt projection.
- No non-empty major category is more than `3x` the character mass of another non-empty major category in the seeded profile.
- Emotional pattern categories do not exceed fact-side categories by more than the configured ratio.
- Cognition receives no duplicated diary/recent-window prose.

## Acceptance Criteria

This plan is implemented when:

- Cognition receives only `user_memory_context`.
- All memory categories use `fact`, `subjective_appraisal`, `relationship_signal`, and optional `updated_at`.
- No old `historical_summary`, `recent_window`, or `character_diary` payload is fed downstream.
- Consolidator uses split LLM calls for extraction, merge judging, rewriting, and stability judging.
- No deterministic semantic path decides create/merge/evolve, stability, unit type, or memory meaning.
- RAG remains the only owner of memory-unit retrieval, projection, ranking, and cache sharing.
- Repeated events are compacted into detailed memory units.
- Projection budgets prevent emotional context or fact-side context from dominating cognition.
- The inspection script shows the same profile context that cognition receives.

## Completion Evaluation

Completion status: completed for the big-bang memory-unit cutover.

The core architecture is complete:

- `user_memory_units` is the canonical durable memory store.
- Cognition receives `user_memory_context` instead of the old summary/window/diary surfaces.
- Memory items use the unified `fact`, `subjective_appraisal`, `relationship_signal`, and `updated_at` contract.
- The consolidator path exercises extraction, merge/evolve/create judging, rewriting, and persistence.
- Real LLM tests and actual-pipeline replay demonstrate that concrete facts can be extracted and similar memory can be compacted.
- No deterministic semantic path is allowed for memory meaning.
- No rollback to the old memory architecture is allowed.

The following are forward-fix stabilization items, not reasons to restore legacy paths:

- Add explicit stability-judge evidence payloads: count, distinct-session evidence, recency, and recent examples.
- Add per-candidate skip/log handling so one malformed LLM output does not drop an entire consolidation batch.
- Add JSON schema response mode where supported by the local LLM endpoint.
- Extend Cache2 policy for memory-unit candidate retrieval if shared cache invalidation needs finer granularity.
- Wire or document `active_commitment` lifecycle updates from scheduler completion.

## Execution Evidence

Before/after projection regression:

```powershell
venv\Scripts\python.exe -m pytest tests\test_user_memory_context_before_after.py -q
```

Result: `1 passed`. Report written to `test_artifacts/user_memory_context_before_after_report.json`.

Measured report values:

- Source messages: `80`.
- Legacy emotional/fact ratio: `70.52380952380952`.
- New pattern/fact-side ratio: `0.8463855421686747`.
- Ratio reduction: `98.79985844797149%`.
- Missing required field count: `0`.
- Legacy field leak count: `0`.
- Verdict: `pass`.

RAG projection and flow regression:

```powershell
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_rag_flow.py tests\test_rag_projection.py -q
```

Result: `7 passed`.

Live LLM extractor proof:

```powershell
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_live_llm.py::test_live_extractor_outputs_concrete_memory_unit -q -s -m live_llm
```

Result: `1 passed`. Trace:
`test_artifacts/llm_traces/user_memory_units_live_llm__extractor_concrete_architecture_decision__20260429T002109542490Z.json`.

Live LLM merge/compaction proof:

```powershell
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_live_llm.py::test_live_merge_rewrite_compacts_similar_memory_unit -q -s -m live_llm
```

Result: `1 passed`. Trace:
`test_artifacts/llm_traces/user_memory_units_live_llm__merge_rewrite_compacts_similar_memory_unit.json`.

Actual pipeline replay from real QQ history:

- Source export: `test_artifacts/qq_1082431481_real_memory_pipeline_source.json`.
- Final replay artifact: `test_artifacts/user_memory_units_actual_pipeline_run_after_history_patch.json`.
- Replay seeded `23` real messages, processed the next real message through `brain_service.chat`, and executed background tasks.
- Unit count changed from `2` to `3`.
- One existing objective fact evolved to count `2`.
- One new recent shift was created.

## Rollback / Recovery

Because this is a completed big-bang migration, recovery is forward-fix only.

- Do not disable the new memory-unit path to restore the old cognition memory model.
- Do not restore old `historical_summary`, `recent_window`, `character_diary`, or diary-specific prompt contracts.
- Do not keep hidden runtime dual-read fallback code in the application.
- If a post-cutover issue is found, fix the new pipeline forward, repair or delete bad new memory units if needed, and add a regression or real LLM trace for the failure.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Small local LLM collapses merge/evolve/create into one behavior | Split extractor, merge judge, rewrite, and stability judge into separate calls | Patched and real LLM tests with near/far examples |
| LLM hallucinates candidate IDs | Structural validation rejects unknown `cluster_id`; skip/log the candidate or stage | Invalid-ID tests |
| Memory facts become vague | Extractor prompt requires concrete event-grounded facts | Real LLM extractor tests |
| Stable patterns absorb unrelated recent shifts | Merge judge and stability judge are separate; no threshold-forced promotion | Merge and stability tests |
| Projection balance inverts the original bug | Per-category item caps and char budgets | Regression test for category character mass |
| Seeded tests pass but live reset loses useful data | Fix forward in `user_memory_units`; do not restore legacy cognition path | Actual-pipeline replay and post-reset inspection |

## Assumptions

- Character self-image remains out of scope.
- The response path should not gain new LLM calls.
- Consolidation can spend additional background LLM calls because it is outside normal response latency.
- The live database reset has already happened.
- `updated_at` is always stored and may be omitted from the prompt projection only when RAG decides recency is not useful.
