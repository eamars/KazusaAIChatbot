# RAG1 / Cache1 Big-Bang Decommission Plan

## Context

The codebase carries two parallel RAG stacks:

- **RAG2** (source of truth, currently test-only): `nodes/persona_supervisor2_rag_supervisor2.py` exposes `call_rag_supervisor(...)`; `rag/cache2_runtime.py` + `rag/cache2_policy.py` + `rag/cache2_events.py` provide a session-LRU cache with dependency-event invalidation; eleven helper agents subclass `BaseRAGHelperAgent` in `rag/helper_agent.py`.
- **RAG1 / Cache1** (legacy, in production): `nodes/persona_supervisor2_rag.py` (1153 LOC) drives an embedding-similarity dispatcher; `rag/cache.py` (1099 LOC) implements an in-memory LRU with MongoDB write-through; collections `rag_cache_index` and `rag_metadata_index` persist entries and a per-user `rag_version` counter.

The two stacks have incompatible state shapes. RAG1 returns a structured `research_facts` dict with ~12 sub-keys plus a `research_metadata` block carrying `depth`, `cache_hit`, `depth_confidence`. RAG2 returns `{answer, known_facts[], unknown_slots[], loop_count}`. Cognition L2/L3, the facts-harvester, and the knowledge-distiller all read RAG1's shape directly.

The user has mandated:
1. RAG2 is the source of truth; consumers must be adjusted to it.
2. **No shims, no adapters, no compatibility layer.** Every legacy reference is rewritten or deleted.
3. RAG1 and Cache1 are deleted in full — code, tests, schemas, MongoDB collections.
4. The consolidator is reworked subject to project scope; recommendations welcome.
5. Cache invalidation follows `development_plans/rag_cache2_design.md` §"Consolidator → Cache 2 Integration Path".
6. Database deprecated collections are removed.
7. All checklist items mandatory; no deferred work.

Outcome: a single RAG path (RAG2), a single cache (Cache2 — session LRU), event-driven invalidation from `db_writer` and `save_conversation`, structured-plus-summarized payload to cognition (no raw blob bloat), LLM-driven dedup with DB-level idempotency in the consolidator, and a smaller, simpler codebase (~3,400 LOC removed; ~3 obsolete LangGraph files; 2 MongoDB collections dropped).

---

## Mandatory Refactor Rules

Non-negotiable. Apply to every change in this plan.

### R1 — Invoke relevant skills before every change

- **py-style** skill applies to every Python file written or modified. Six rules: imports at top (no inline except in test code), narrow try-except, specific exception types, complete docstrings, default values in one place, try-except only for external failures.
- **cjk-safety** skill must be invoked before writing or editing any `.py` file containing CJK string content (most prompt files in this codebase). Critical: use single-quoted string delimiters when content contains Chinese typographic quotes `"` `"`; never copy CJK content via the Write tool when a byte-copy script can be used; run `ast.parse` or `py_compile` after writing CJK Python files.
- These are auto-invoked per project memory. Verification (skill invocation visible in transcript before the edit) must happen on every code-touching turn.

### R2 — No deterministic filtering against user-derived data

User-derived data includes: user input, decontextualized input, chat history, dialog output, harvester output, evaluator output, RAG agent output, retrieval evidence, and any natural-language content that traces back to a user message.

- **Forbidden:** regex/Python filters that judge content (length thresholds, novelty heuristics, style-violation detectors, narrative-vs-attribute classifiers, future-time-token detectors, subject-inversion checkers, "trivial dialog" short-circuits, etc.).
- **Allowed:** structured-key set-membership lookups (e.g. `dedup_key in existing_keys`), schema validation that JSON parses, structural truncation by character count when the truncation is documented and uniform (not content-judgment), idempotent DB upsert by primary key.
- **Allowed:** passing structured hints into LLM prompts so the LLM enforces the rule under its own judgment (e.g. passing `existing_dedup_keys` as a hard exclusion list in the harvester prompt is fine; the LLM decides whether to honour it).
- LLM judgment is the only sanctioned filter for natural-language content. If the LLM is too expensive for a particular gating decision, the answer is to redesign the flow, not to add a regex.

This rule **SUPERSEDES** any earlier draft of this plan that referenced deterministic Python validators, trivial-dialog short-circuits, or post-LLM heuristic dedup.

### R3 — Trusted vs untrusted prompt boundary

- **System prompt** carries only trusted reference data: character profile fields (name, MBTI, mood, vibe, boundary parameters, linguistic texture, taboos), agent instructions, output schema, format examples. Character data is treated as trusted because it ships with the codebase.
- **User prompt** (HumanMessage) carries user input and anything derived from it: `decontextualized_input`, `chat_history`, `research_facts`/`rag_result`, `final_dialog`, `internal_monologue`, `content_anchors`, `interaction_subtext`, evaluator feedback, and all retrieval evidence.
- Never substitute user-derived content into a system-prompt `.format()` placeholder. Existing prompts in this codebase already follow this boundary; verify on every prompt edit.

### R4 — System prompt modifications require full audit

When any system prompt is modified — even a single line — the entire prompt is re-read with a system-engineer hat on. Audit checklist:

- Trusted/untrusted boundary preserved (R3).
- Instructions internally consistent — no contradictions introduced by the edit.
- Output schema (`# 输出格式` block) still matches what downstream parsers consume.
- Examples still illustrate the current rules (no stale rule references).
- No leftover references to deleted concepts: `depth`, `cache_hit`, `knowledge_base_results`, `input_context_results`, `external_rag_results`, `boundary_cache`, `rag_version`, `SHALLOW`/`DEEP`, the `RAG 元信息` block, `research_facts`, `research_metadata`.
- Field names referenced inside the prompt match the actual `rag_result` shape.
- Document the audit pass in the commit message ("audited <path> system prompt for X, Y, Z").

---

## Design Decisions (rationale before plan)

| Topic | Decision | Rationale |
|---|---|---|
| Knowledge-base distillation (`consolidator_knowledge.py`) | **DELETE** | Cache1 stored it with 30-day TTL; Cache2 is session-LRU only. No V2 consumer reads `knowledge_base`. Re-implementing as a persistent collection is net-new feature work outside the cutover scope. |
| Depth classification (`SHALLOW`/`DEEP` gating) | **DELETE** | V2 supervisor has no depth concept; loop-count cap (`_MAX_LOOP_COUNT`) already bounds work. Remove `rag/depth_classifier.py` + every reference. Consolidator no longer reads `metadata["depth"]`. |
| Affinity-gated external web search (`EXTERNAL_AFFINITY_SKIP_PERCENT = 40`) | **DELETE** | V1-only behaviour. V2 dispatcher selects `web_search_agent2` only when an unknown slot maps to it; no separate gate needed. |
| User-profile hydration in stage_1 | **DELETE** from supervisor; **RE-USE** in `user_profile_agent` | V2's `user_profile_agent` already returns the hydrated bundle. Consumers that previously read `state["user_profile"]` post-RAG must instead read it from the projected `rag_result`. |
| `rag_version` counter / `increment_rag_version` | **DELETE** | Cache2 uses event-driven invalidation; the counter is dead weight. |
| RAG2 output → cognition/consolidator payload | **HYBRID — structured raw for image bundles, summary-only for evidence agents** | Feeding raw `known_facts[]` blows token budget; feeding only `summary` strips L2/L3 prompts of the structured `user_image.milestones` etc. they reference. See §"Cognition Payload Shape". |
| `save_conversation` cache invalidation | **WRAP at db layer** | Today `save_conversation` emits no event — every conversation_*_agent cache entry goes stale on each new message. Wrap inside `db/conversation.py` so the contract holds at every call site. |
| Consolidator loop prevention (re-ingesting facts already in DB) | **TWO-LAYER: LLM harvester with `existing_dedup_keys` exclusion list → DB upsert by `dedup_key`** | Per R2, no deterministic filter on harvester output. The LLM receives the existing key set as a hard exclusion in its prompt and decides; DB upsert is the final idempotent guard. |
| Consolidator LLM efficiency | **PROCESS-LEVEL ONLY — parallel image updates, schema-driven evaluator skip** | Per R2, no content-based gating. Only safe optimisations: run independent updaters concurrently via `asyncio.gather`, and skip the evaluator when the harvester returns structurally empty lists (LLM-decided emptiness, not deterministic content filter). |

---

## Target Architecture (post-refactor)

```
service.py
  └─ persona_supervisor2 (StateGraph)
       ├─ stage_0_msg_decontexualizer
       ├─ stage_1_research  →  call_rag_supervisor (RAG2)
       │                         + project_known_facts(...)
       │                         emits state["rag_result"] (hybrid payload)
       ├─ stage_2_cognition (L1/L2/L3 — read state["rag_result"])
       └─ stage_3_action    (dialog_agent — reads stage_2 outputs)

  └─ Background: call_consolidation_subgraph
       ├─ existing_dedup_keys precomputed from user_profile (structured key set, not content)
       ├─ reflection (parallel) / facts_harvester (reads state["rag_result"] + existing_dedup_keys)
       ├─ LLM evaluator (skipped only when harvester returns empty lists)
       └─ db_writer
            ├─ MongoDB writes (steps 1, 2a, 2b, 3a, 3b, 4, 6, 7)
            ├─ user_image / character_image updates in parallel (asyncio.gather)
            └─ Emit CacheInvalidationEvent(s) to get_rag_cache2_runtime()
            (NO knowledge-base step; NO rag_version increment; NO deterministic content filters)

db/conversation.py::save_conversation
  └─ insert message
  └─ Emit CacheInvalidationEvent(source="conversation_history", ...) — db-layer wrap
```

**State key changes:**
- `research_facts` (dict) → **REMOVED**
- `research_metadata` (list[dict]) → **REMOVED**
- `rag_result` (dict) → **NEW** — hybrid payload, see §"Cognition Payload Shape"

---

## Cognition Payload Shape

Current consumers eat the entire `research_facts` blob with raw text fields:

- `cognition_l2.py:268` — passes the full `state["research_facts"]` dict (all 9+ keys, raw blobs).
- `cognition_l3.py:388-398` — explicitly enumerates and forwards 9 sub-keys including `objective_facts`, `user_image`, `character_image`, `input_context_results`, `external_rag_results`, `knowledge_base_results`, `third_party_profile_results`, `channel_recent_entity_results`, `entity_resolution_notes` — all raw text/dict blobs.
- `cognition_l3.py:533-537` (preference_adapter) — `objective_facts`, `user_image`, `character_image` raw blobs.
- `consolidator_facts.py:142, 257` — full `research_facts` dict for harvester + evaluator.

RAG2's native `known_facts[i]` shape is `{slot, agent, resolved, summary, raw_result, attempts, cache}`. The supervisor2 already runs a per-slot evaluator LLM that produces `summary` (a one-sentence digest of `raw_result`).

### Decision: hybrid by agent role, not uniform raw or uniform summary

Feeding **only `summary`** strips structured data that L2/L3 prompts reference by field name (`user_image.milestones`, `character_image.recent_observations`). Cognition would lose its grounding; LLM hallucinations would rise.

Feeding **all `raw_result`** drowns L2/L3 in token-heavy retrieval blobs (search hits with full message text, page snippets). Latency and cost blow up; the supervisor's evaluator effort is wasted because cognition re-reads the unfiltered raw data.

**Hybrid policy:**

| Agent class | What to forward to cognition |
|---|---|
| `user_profile_agent` (current user) | `raw_result` (structured user profile bundle — milestones, historical_summary, recent_observations) |
| `user_profile_agent` (character) | `raw_result` (structured character image bundle) |
| `user_lookup_agent`, `user_list_agent` | `summary` only (one-line "X resolved to UUID Y" / "[N users matched]") |
| `persistent_memory_search_agent`, `persistent_memory_keyword_agent` | `summary` for each hit + a single concatenated `evidence_text` string (top-K hit contents truncated to N chars) |
| `conversation_search_agent`, `conversation_filter_agent`, `conversation_keyword_agent`, `conversation_aggregate_agent` | `summary` only — supervisor2 already condenses these |
| `web_search_agent2` | `summary` + `evidence_text` (first 800 chars of fetched content per hit) |

**Implementation pattern.** New module `nodes/persona_supervisor2_rag_projection.py` exposes:

```python
def project_known_facts(
    known_facts: list[dict],
    *,
    current_user_id: str,
    character_user_id: str,
    evidence_char_limit: int = 800,
) -> dict:
    '''Project supervisor2 known_facts into a hybrid rag_result payload.

    Args:
        known_facts: Raw fact rows from call_rag_supervisor.
        current_user_id: Global user id of the speaker; used to identify
            which user_profile_agent result is the speaker vs. third party.
        character_user_id: Global user id of the character; used to route
            the character user_profile_agent result to character_image.
        evidence_char_limit: Max chars of raw content kept per evidence hit.

    Returns:
        rag_result dict consumed by stage_2_cognition and the consolidator.
    '''
```

The returned `rag_result` shape:

```python
{
    "answer": str,                         # supervisor.answer (one-line synthesis)
    "user_image": dict,                    # user_profile_agent raw for current user, or {}
    "character_image": dict,               # user_profile_agent raw for character, or {}
    "third_party_profiles": list[str],     # summaries from user_lookup / user_list
    "memory_evidence": list[dict],         # [{"summary": str, "content": str}, ...]
    "conversation_evidence": list[str],    # summary strings only
    "external_evidence": list[dict],       # [{"summary": str, "content": str, "url": str}, ...]
    "supervisor_trace": {
        "loop_count": int,
        "unknown_slots": list[str],
        "dispatched": list[dict],          # [{"slot": ..., "agent": ..., "resolved": ...}]
    },
}
```

Cognition prompts read structured `user_image` / `character_image` directly (matching the current shapes already in the L2/L3 prompts) and read flat lists for `*_evidence`. The biggest LLM cost saver: the consolidator facts harvester no longer reads raw search hits; it reads the same hybrid block.

---

## Consolidator Loop Prevention (re-ingestion of known facts)

Risk: consolidator writes facts to `user_profile_memories` → next turn RAG2 returns them → cognition references them → bot mentions them → consolidator extracts them again → tries to re-write.

Per R2, the defence is LLM-driven plus DB idempotency — no deterministic content filter on harvester output.

**Two-layer defence:**

1. **Layer 1 — LLM harvester with `existing_dedup_keys` exclusion list.** Compute `existing_dedup_keys: set[str]` once at consolidator entry from `state["user_profile"]` (union of objective_facts, active_commitments, milestones — a structured key set, not content). Pass into the harvester prompt as a hard exclusion list. The harvester is instructed: *"if a candidate fact maps to a `dedup_key` already in this list, do not emit it."* The LLM enforces. The "旧闻复读" rule in the evaluator prompt continues to provide a second LLM-level check, with the harvester now reading `rag_result.user_image` / `rag_result.character_image` / `rag_result.memory_evidence` for old-fact context.
2. **Layer 2 — `insert_profile_memories` upsert by `dedup_key`.** Final idempotency at the DB. If anything slips past the LLM, the upsert merges by key with no duplicate row.

The `existing_dedup_keys` set is passed as a JSON list in the user-prompt payload (R3 — it's derived from user data via the harvester's prior runs, so it belongs in the human message). The set membership comparison happens inside the LLM, not in Python.

The evaluator may be skipped when the harvester returns structurally empty `new_facts: []` AND `future_promises: []` — that is process control on LLM output shape (empty list), not a content judgment, and is allowed under R2.

---

## Cache Invalidation — Event Emission

### Helper agent dependency declarations (read-only audit)

| Agent | source | scope fields populated |
|---|---|---|
| `user_lookup_agent` | `user_profile` | platform, channel, global_user_id, display_name |
| `user_list_agent` (source=user_profiles) | `user_profile` | platform, channel |
| `user_list_agent` (source=conversation_participants) | `conversation_history` | platform, channel |
| `user_profile_agent` (current user) | `user_profile` | global_user_id |
| `user_profile_agent` (character) | `character_state` | (none) |
| `persistent_memory_keyword_agent`, `persistent_memory_search_agent` | `user_profile` | platform, channel, scope filters |
| `conversation_filter_agent`, `conversation_keyword_agent`, `conversation_search_agent` | `conversation_history` | platform, channel, range |

**No agent declares `source="user_image"`.** The design doc lists `user_image` as a future source. Today, it has zero consumers — emitting `user_image` events would scan and match nothing.

`upsert_user_image` writes to the `user_profiles` document (nested `user_image` field). `user_profile_agent` reads back from the same document and caches with `source="user_profile"` dependency. The correct event is **`user_profile`** scoped to that user.

`upsert_character_self_image` writes to `character_state` (nested self-image). `user_profile_agent` (character branch) caches with `source="character_state"`. The correct event is **`character_state`**.

### `db_writer` event emission (replaces `persona_supervisor2_consolidator_persistence.py` lines 624–650)

```python
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime

# ── Step 5: Cache2 invalidation events (after persistence) ──────
runtime = get_rag_cache2_runtime()
events: list[CacheInvalidationEvent] = []

if global_user_id and (
    write_log.get('user_profile_memories')
    or write_log.get('affinity')
    or write_log.get('relationship_insight')
    or write_log.get('user_image')
):
    events.append(CacheInvalidationEvent(
        source='user_profile',
        platform=state['platform'],
        platform_channel_id=state['platform_channel_id'],
        global_user_id=global_user_id,
        timestamp=timestamp,
        reason='consolidator: user_profile',
    ))

if write_log.get('character_state') or write_log.get('character_image'):
    events.append(CacheInvalidationEvent(
        source='character_state',
        reason='consolidator: character_state',
    ))

evicted_total = 0
for event in events:
    evicted_total += await runtime.invalidate(event)

cache_invalidated = [event.source for event in events]
metadata['cache_evicted_count'] = evicted_total
```

Notes:
- No `try/except` — Cache2 invalidation is in-memory; per `py-style` rule 6, try/except is reserved for external failures.
- Only `user_profile` and `character_state` sources are emitted. `user_image` is NEVER emitted (no agent depends on it today).
- Drop `AFFINITY_CACHE_NUKE_THRESHOLD` constant and its conditional `clear_all_user` branch — the wildcard scope on the `user_profile` event already invalidates every cached entry for that user.
- Drop `await increment_rag_version(...)` and `metadata['cache_invalidation_scope']` (replace with `cache_invalidated` populated above).

### `save_conversation` event emission (NEW — wrap at db layer)

`service.py:330` (user message) and `service.py:639` (bot reply) call `await save_conversation(...)`. Today no event fires — every cached entry from `conversation_filter/keyword/search/aggregate_agent` becomes stale instantly but stays served. Per `development_plans/rag_cache2_design.md`: *"`save_conversation`: invalidates overlapping conversation-history cache entries."*

Wrap inside `db/conversation.py::save_conversation` so the contract is enforced at the data layer rather than at every call site:

```python
# db/conversation.py
async def save_conversation(doc: ConversationMessageDoc) -> None:
    '''Persist one conversation message and invalidate matching cache entries.'''
    await _collection().insert_one(doc)
    from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
    from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
    await get_rag_cache2_runtime().invalidate(CacheInvalidationEvent(
        source='conversation_history',
        platform=doc['platform'],
        platform_channel_id=doc['platform_channel_id'],
        global_user_id=doc.get('global_user_id', ''),
        timestamp=doc.get('timestamp', ''),
        reason='save_conversation',
    ))
```

The event omits `display_name`, so it matches all entries scoped to the same `(platform, channel)` regardless of user-level filters. The `timestamp` field allows entries that cached a closed historical range strictly before this write to remain valid. Imports stay function-local to avoid circular import risk between `db.conversation` and `rag.cache2_*`.

The design doc's principle "Do not put invalidation calls inside DB-layer functions" was about the consolidator, where DB semantics weren't 1-to-1 with cache semantics. For `save_conversation` the contract IS 1-to-1 — every save invalidates — so wrapping is appropriate.

---

## Consolidator Efficiency Refactor

Current consolidator runs:
- 3 parallel LLM calls (`global_state_updater`, `relationship_recorder`, `facts_harvester`)
- Up to MAX_FACT_HARVESTER_RETRY × 2 LLM calls (harvester + evaluator loop)
- 1 LLM call for user_image merge (in `_update_user_image`)
- 1 LLM call for character_image merge (in `_update_character_image`)
- 1 LLM call for task dispatcher (only when promises exist)

Per R2 the efficiency work is restricted to process-level changes — no content-based gating. The following are the only sanctioned optimisations:

1. **Pass `existing_dedup_keys` to the harvester prompt** — the LLM uses the exclusion list to avoid re-emitting known facts. Reduces evaluator load only because the harvester emits less; no Python filter is added. Already covered in §"Consolidator Loop Prevention".
2. **Skip the evaluator when harvester returns structurally empty lists.** If `new_facts == [] and future_promises == []`, do not invoke the evaluator. This is shape-based (list-length-zero), not content-based, and complies with R2.
3. **Run `_update_user_image` and `_update_character_image` in parallel** via `asyncio.gather`. They are independent and operate on disjoint MongoDB documents. Process-level concurrency, no filter.
4. **Verify existing image-updater early-exit logic** is not removed in the refactor. Both `_update_user_image` and `_update_character_image` already short-circuit when their LLM-derived new content is empty — that LLM-driven decision must remain intact.

**Removed from earlier drafts of this plan (forbidden under R2):**
- Trivial-dialog short-circuit by character count or novel-noun-phrase check.
- Deterministic Python validator (~80 LOC) catching object inversion, narrative-style descriptions, future-time tokens, etc.
- Post-LLM dedup_key set-membership filter on harvester output.

(Future, NOT in this refactor: extract `_update_user_image` / `_update_character_image` into helper agents so they can also benefit from Cache2.)

---

## Files: Delete / Modify / Create

### Delete (entire file)

| Path | Reason |
|---|---|
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag.py` | RAG1 entry + helpers |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor.py` | RAG1 evaluator |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_executors.py` | RAG1 executor agents |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_resolution.py` | RAG1 resolution helpers |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_schema.py` | RAG1 state schema |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_knowledge.py` | KB distillation deleted |
| `src/kazusa_ai_chatbot/rag/cache.py` | Cache1 implementation |
| `src/kazusa_ai_chatbot/rag/depth_classifier.py` | V1-only construct |
| `src/kazusa_ai_chatbot/db/rag_cache.py` | Cache1 DB ops |
| `tests/test_rag_cache.py` | Cache1 unit tests |
| `tests/test_rag_live_llm.py` | RAG1 integration |
| `tests/test_persona_supervisor2_rag_and_l2.py` | RAG1+L2 path |
| `tests/test_persona_supervisor2_l3_and_consolidator.py` | Reads RAG1 shape; rewrite as new file |

### Modify

| Path | Change |
|---|---|
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | Replace stage_1 import; build `rag_result` from `call_rag_supervisor` + `project_known_facts(...)`; remove `research_facts`/`research_metadata` from state plumbing |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` (`GlobalPersonaState`) | Remove `research_facts`, `research_metadata`; add `rag_result: dict` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | Pass `rag_result` to L2/L3 (replace `research_facts` plumbing) |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` | Read `rag_result.user_image`, `rag_result.character_image` (raw structured); read `rag_result.memory_evidence`, `conversation_evidence`, `external_evidence`, `third_party_profiles` (summarised). Drop `input_context_results`, `external_rag_results`, `objective_facts`, `knowledge_base_results`, `channel_recent_entity_results`, `entity_resolution_notes` raw blobs. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` | Same shape changes for content_anchor (lines ~389-397) and preference_adapter (lines ~533-540): `research_facts.get(...)` → `rag_result.<field>` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py` | Forward `rag_result` instead of `research_facts`/`research_metadata`; precompute `existing_dedup_keys` from `state["user_profile"]` and add to ConsolidatorState |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py` | Replace `research_facts`/`metadata.depth/cache_hit` with `rag_result`; add `existing_dedup_keys: set[str]` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py` | Harvester reads `rag_result.user_image`, `character_image`, `memory_evidence` for old-fact comparison; receives `existing_dedup_keys` as a JSON list in the human-prompt payload (R3) and is instructed to honour it as a hard exclusion. Skip the evaluator only when the harvester returns structurally empty `new_facts` and `future_promises` lists. Drop "RAG 元信息" prompt block; replace with `supervisor_trace` (loop_count, unknown_slots). System prompt edits trigger R4 audit. **No deterministic content filter** added (R2). |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py` | Remove `_get_rag_cache`, `increment_rag_version`, `_update_knowledge_base` imports; remove Step 8; replace lines 624–650 with `CacheInvalidationEvent` emissions per §"Cache Invalidation — Event Emission" (only `user_profile` and `character_state` sources, NEVER `user_image`). Run `_update_user_image` and `_update_character_image` via `asyncio.gather` (process-level concurrency only — image updaters' existing internal LLM-driven early exits stay intact). Drop `try/except PyMongoError` around the new event emission (Cache2 is in-memory). |
| `src/kazusa_ai_chatbot/service.py` | Remove cache1 warm-start (~line 436) and shutdown (~line 469); no Cache2 startup needed (singleton lazy-inits). No cache invalidation wrapping at the `save_conversation` call sites — handled at the db layer. |
| `src/kazusa_ai_chatbot/db/conversation.py` | `save_conversation`: emit `CacheInvalidationEvent(source='conversation_history', platform, platform_channel_id, global_user_id, timestamp)` after successful insert. Use function-local imports of `cache2_events` and `cache2_runtime` to avoid circulars. |
| `src/kazusa_ai_chatbot/db/__init__.py` | Remove imports + `__all__` entries for `RagCacheIndexDoc`, `RagMetadataIndexDoc`, `clear_all_cache_for_user`, `find_cache_entries`, `get_rag_version`, `increment_rag_version`, `insert_cache_entry`, `soft_delete_cache_entries`; update docstring submodule map |
| `src/kazusa_ai_chatbot/db/schemas.py` | Lines 298–326 — delete `RagCacheIndexDoc`, `RagMetadataIndexDoc` |
| `src/kazusa_ai_chatbot/db/bootstrap.py` | Drop `rag_cache_index`/`rag_metadata_index` from `required_collections`; remove their index-creation blocks (~lines 122–141) and the vector-index entry (~line 147); add idempotent `drop_collection` for both legacy names; remove `enable_vector_index` call for `rag_cache_index` |
| `src/kazusa_ai_chatbot/rag/user_profile_agent.py` | Remove `from rag.depth_classifier import DEEP` (line 18); replace with literal value or own constant |
| `tests/test_user_profile_memories.py` | Audit; remove any `_get_rag_cache` / `increment_rag_version` references |
| `tests/test_e2e_live_llm.py` | Audit; replace cache1 assertions with `get_rag_cache2_runtime().get_stats()` |

### Create

| Path | Purpose |
|---|---|
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py` | `project_known_facts(known_facts, *, current_user_id, character_user_id, evidence_char_limit=800) -> dict` — projects RAG2 `known_facts[]` into the hybrid `rag_result` payload |
| `scripts/drop_legacy_rag_collections.py` | One-shot production cleanup: drops `rag_cache_index` + `rag_metadata_index`. Idempotent — silent if already gone. |
| `tests/test_rag_projection.py` | Unit tests for `project_known_facts` — one fixture per agent class, edge cases (empty, multi-user, character + current user mixed) |
| `tests/test_db_writer_cache2_invalidation.py` | Unit tests covering each `write_log` outcome → expected `CacheInvalidationEvent` emission and matching eviction |
| `tests/test_save_conversation_invalidation.py` | Unit tests for db-layer wrap: store conversation cache entries, call `save_conversation`, assert eviction |
| `tests/test_persona_supervisor2_rag2_integration.py` | Replaces `test_persona_supervisor2_rag_and_l2.py` — exercises stage_1_research with V2 supervisor end-to-end (mocked LLM) |
| `tests/test_consolidator_facts_rag2.py` | Replaces `test_persona_supervisor2_l3_and_consolidator.py` — exercises facts-harvester against new `rag_result` shape |
| `tests/test_consolidator_efficiency.py` | `existing_dedup_keys` correctly built from `user_profile`; harvester prompt receives the list; evaluator skipped when harvester returns empty; `_update_user_image` and `_update_character_image` execute concurrently. **No tests of deterministic content filters** (R2 — none exist in the implementation). |

### Keep (no change)

- `src/kazusa_ai_chatbot/rag/cache2_runtime.py`, `cache2_policy.py`, `cache2_events.py`, `helper_agent.py`
- All eleven RAG2 helper agents in `src/kazusa_ai_chatbot/rag/`
- `src/kazusa_ai_chatbot/rag/user_image_retriever_agent.py` (used by `user_profile_agent`)
- `tests/test_persona_supervisor2_rag_supervisor2_live.py` — primary RAG2 integration test
- `tests/test_rag_initializer_cache2.py`, all `tests/test_<helper_agent>_*.py`

---

## RAG2 Wiring in `persona_supervisor2.py`

The new `stage_1_research` node:

```python
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2 import call_rag_supervisor
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import project_known_facts


async def stage_1_research(state: GlobalPersonaState) -> dict:
    '''Run RAG2 progressive supervisor and emit projected rag_result.'''
    context = {
        'platform': state['platform'],
        'platform_channel_id': state['platform_channel_id'],
        'channel_type': state.get('channel_type', 'group'),
        'global_user_id': state['global_user_id'],
        'user_name': state['user_name'],
        'user_profile': state.get('user_profile', {}),
        'current_timestamp': state['timestamp'],
        'channel_topic': state.get('channel_topic', ''),
        'chat_history_recent': state.get('chat_history_recent', []),
        'chat_history_wide': state.get('chat_history_wide', []),
        'reply_context': state.get('reply_context', {}),
        'indirect_speech_context': state.get('indirect_speech_context', ''),
    }
    raw = await call_rag_supervisor(
        original_query=state['decontexualized_input'],
        character_name=state['character_profile'].get('name', ''),
        context=context,
    )
    rag_result = project_known_facts(
        raw.get('known_facts', []),
        current_user_id=state['global_user_id'],
        character_user_id=state['character_profile'].get('global_user_id', ''),
    )
    rag_result['answer'] = raw.get('answer', '')
    rag_result['supervisor_trace'] = {
        'loop_count': raw.get('loop_count', 0),
        'unknown_slots': raw.get('unknown_slots', []),
        'dispatched': [
            {'slot': f.get('slot'), 'agent': f.get('agent'), 'resolved': f.get('resolved')}
            for f in raw.get('known_facts', [])
        ],
    }
    return {'rag_result': rag_result}
```

Audit `call_rag_supervisor`'s actual signature/context contract during implementation; the `context` dict above must match what RAG2 helper agents read from `state['context']` (see `rag/user_profile_agent.py`, `rag/conversation_*_agent.py` for canonical reads).

---

## Database Migration

### `db/bootstrap.py` — new section after `existing = ...`

```python
# Drop legacy RAG1 collections (idempotent — safe across deploys)
for legacy in ('rag_cache_index', 'rag_metadata_index'):
    if legacy in existing:
        await db.drop_collection(legacy)
        logger.info("Dropped legacy collection '%s'", legacy)
        existing.discard(legacy)
```

Then remove the two collections from `required_collections`, their index-creation calls (lines ~122–141), and the `enable_vector_index('rag_cache_index', ...)` call (~line 147).

### `scripts/drop_legacy_rag_collections.py`

Standalone idempotent CLI (for ops to run once on production before redeploy):

```python
'''One-shot cleanup: drop legacy RAG1 MongoDB collections.

Run before deploying the RAG2-only build to ensure no stale collections persist.
Safe to run repeatedly.
'''

import asyncio
import logging

from kazusa_ai_chatbot.db._client import close_db, get_db

logger = logging.getLogger(__name__)


async def main() -> None:
    '''Drop rag_cache_index and rag_metadata_index if present.'''
    db = await get_db()
    existing = set(await db.list_collection_names())
    for name in ('rag_cache_index', 'rag_metadata_index'):
        if name in existing:
            await db.drop_collection(name)
            logger.info("Dropped collection '%s'", name)
        else:
            logger.info("Collection '%s' not present; skipping", name)
    await close_db()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

---

## Verification

### Static greps (must all return zero matches inside `src/`)

- `_get_rag_cache`
- `RAGCache\b`
- `from kazusa_ai_chatbot.rag.cache import`
- `from kazusa_ai_chatbot.nodes.persona_supervisor2_rag\b` (note: `_supervisor2` is allowed)
- `rag_cache_index`, `rag_metadata_index`
- `RagCacheIndexDoc`, `RagMetadataIndexDoc`
- `increment_rag_version`, `get_rag_version`, `insert_cache_entry`, `find_cache_entries`, `soft_delete_cache_entries`, `clear_all_cache_for_user`
- `InputDepthClassifier`, `from kazusa_ai_chatbot.rag.depth_classifier import`
- `research_facts`, `research_metadata` (state-key names)
- `_update_knowledge_base`, `knowledge_base_results`
- `EXTERNAL_AFFINITY_SKIP_PERCENT`, `AFFINITY_CACHE_NUKE_THRESHOLD`
- `source="user_image"` and `source='user_image'` (must not appear other than the `cache2_events.py` dataclass field literal)
- R2 negative-grep: search the consolidator package for `re\.match`, `re\.search`, character-count thresholds against `final_dialog`/`new_facts`/`future_promises`/`description` content, or any function whose name contains `is_trivial`, `validate_fact_format`, `looks_like_narrative` etc. — none should exist.

### Smoke tests

- `pytest tests/test_persona_supervisor2_rag_supervisor2_live.py` — passes
- `pytest tests/test_rag_initializer_cache2.py` — passes
- `pytest tests/test_rag_projection.py` (new) — passes
- `pytest tests/test_db_writer_cache2_invalidation.py` (new) — passes
- `pytest tests/test_save_conversation_invalidation.py` (new) — passes
- `pytest tests/test_persona_supervisor2_rag2_integration.py` (new) — passes
- `pytest tests/test_consolidator_facts_rag2.py` (new) — passes
- `pytest tests/test_consolidator_efficiency.py` (new) — passes
- Boot the service: `uvicorn kazusa_ai_chatbot.service:app --port 8000` — must start clean (no errors, no warnings about missing collections)
- Issue one `/chat` request — non-empty response, Cache2 stats in logs
- Send two `/chat` requests in the same channel → confirm `conversation_history` invalidation event in logs after each save
- Send a fact-bearing message twice in a row → confirm second turn's harvester emits empty `new_facts` (LLM honoured exclusion list) and evaluator is skipped

### DB verification

After clean boot:
```
> db.getCollectionNames()
# Must NOT include 'rag_cache_index' or 'rag_metadata_index'
```

---

## Implementation Order (avoid mid-refactor breakage)

1. Build `nodes/persona_supervisor2_rag_projection.py` and its unit tests first — it is the contract the rewrite hangs on.
2. State-shape rewrite (cognition L2/L3 + consolidator consumers updated to read `rag_result`).
3. Wire RAG2 entry into `persona_supervisor2.py` — V1 stack still on disk but unused.
4. Replace `db_writer` cache1 invalidation with Cache2 events; add `db/conversation.py` save_conversation wrap.
5. Apply consolidator efficiency refactor (Phase E): existing_dedup_keys precompute → harvester prompt exclusion list, evaluator skip on structurally empty harvester output, parallel image updates via asyncio.gather. (No deterministic content filter — R2.)
6. Delete `_update_knowledge_base` call + its module.
7. Remove cache1 warm-start in `service.py`.
8. Delete RAG1 modules (5 files).
9. Delete `rag/cache.py`, `rag/depth_classifier.py`, `db/rag_cache.py`.
10. Update `db/__init__.py`, `db/schemas.py`, `db/bootstrap.py`; add `scripts/drop_legacy_rag_collections.py`.
11. Test migration: delete obsolete tests, write new ones.
12. Run all greps + smoke tests.

Each step leaves the code in a runnable state for incremental commits. Step 2 is the largest and highest-risk; steps 8–10 are mechanical deletions.

---

## Implementation Checklist (ALL MANDATORY)

### Phase A — RAG2 projection module + consumer rewrite

- [x] Create `nodes/persona_supervisor2_rag_projection.py` with `project_known_facts(known_facts, *, current_user_id, character_user_id, evidence_char_limit=800) -> dict`. Must be unit-testable in isolation.
- [x] Update `nodes/persona_supervisor2_schema.py` — `GlobalPersonaState`: remove `research_facts`, `research_metadata`; add `rag_result: dict`
- [x] Update `nodes/persona_supervisor2_consolidator_schema.py` — `ConsolidatorState`: remove `research_facts`, `metadata.depth`/`cache_hit`; add `rag_result`, `existing_dedup_keys: set[str]`
- [x] Rewrite `nodes/persona_supervisor2_cognition.py` — pass `rag_result` to L2/L3
- [x] Rewrite `nodes/persona_supervisor2_cognition_l2.py`:
  - [x] Read `rag_result.user_image`, `rag_result.character_image` (raw structured) — placed in HumanMessage payload (R3)
  - [x] Read `rag_result.memory_evidence`, `rag_result.conversation_evidence`, `rag_result.external_evidence`, `rag_result.third_party_profiles` (summarised) — HumanMessage payload (R3)
  - [x] Drop `input_context_results`, `external_rag_results`, `objective_facts`, `knowledge_base_results`, `channel_recent_entity_results`, `entity_resolution_notes` raw blobs
  - [x] **R4 audit pass** on `_COGNITION_CONSCIOUSNESS_PROMPT`, `_BOUNDARY_CORE_PROMPT`, `_JUDGEMENT_CORE_PROMPT` after the input-format block edits
- [x] Rewrite `nodes/persona_supervisor2_cognition_l3.py` — same shape changes for `_CONTENT_ANCHOR_AGENT_PROMPT` (lines ~389-397) and `_PREFERENCE_ADAPTER_PROMPT` (lines ~533-540). **R4 audit pass** on each modified prompt.
- [x] Rewrite `nodes/persona_supervisor2_consolidator.py` — forward `rag_result`; precompute `existing_dedup_keys` from `state["user_profile"]`
- [x] Rewrite `nodes/persona_supervisor2_consolidator_facts.py`:
  - [x] Harvester reads `rag_result.user_image`, `character_image`, `memory_evidence` for old-fact comparison
  - [x] Receive `existing_dedup_keys` as a JSON list in the human-prompt payload (R3); system prompt instructs the LLM to honour the exclusion
  - [x] Drop "RAG 元信息" block; replace with `supervisor_trace` block (loop_count, unknown_slots)
  - [x] **R4 audit pass on each modified system prompt** — both `_FACTS_HARVESTER_PROMPT` and `_FACT_HARVESTER_EVALUATOR_PROMPT` get a full re-read; document the pass in the commit message
  - [x] Verify R3 boundary: only character-name and instructions in SystemMessage; all `decontexualized_input`, `final_dialog`, `rag_result`, `existing_dedup_keys`, `content_anchors`, `logical_stance`, `character_intent`, evaluator feedback go in HumanMessage

### Phase B — Wire RAG2 into supervisor

- [x] Edit `nodes/persona_supervisor2.py` — import `call_rag_supervisor` and `project_known_facts`; replace `stage_1_research` per §"RAG2 Wiring"
- [x] Verify `call_rag_supervisor` context-dict contract by reading helper agents in `rag/`

### Phase C — db_writer rework

- [x] Edit `nodes/persona_supervisor2_consolidator_persistence.py`:
  - [x] Remove imports: `_get_rag_cache` (line 49), `increment_rag_version` (line 24 group), `_update_knowledge_base` (line 37)
  - [x] Replace lines 624–650 with `CacheInvalidationEvent` emission block per §"Cache Invalidation — Event Emission" (only `user_profile` and `character_state` sources — NEVER `user_image`)
  - [x] Remove `AFFINITY_CACHE_NUKE_THRESHOLD` constant (line 54) and its branch
  - [x] Remove Step 8 (`_update_knowledge_base` call, ~lines 676–683); drop `kb_count`, `metadata['knowledge_base_entries_written']`
  - [x] Update `metadata` dict: replace `cache_invalidation_scope` with `cache_invalidated`; add `cache_evicted_count`
  - [x] Remove `try/except PyMongoError` around the new event emission (in-memory call)

### Phase D — `save_conversation` invalidation (db-layer wrap)

- [x] Edit `db/conversation.py::save_conversation`:
  - [x] Emit `CacheInvalidationEvent(source='conversation_history', platform, platform_channel_id, global_user_id, timestamp)` after successful insert
  - [x] Use function-local imports of `cache2_events` and `cache2_runtime` to avoid circulars
- [x] Verify all `save_conversation` callers — no service-side wrapping needed once db-layer wrap is in place

### Phase E — Consolidator efficiency refactor (R2-compliant)

- [x] Compute `existing_dedup_keys: set[str]` once at consolidator entry from `state["user_profile"]` (objective_facts ∪ active_commitments ∪ milestones). Structured key set only — no content extraction.
- [x] Pass `existing_dedup_keys` into the facts-harvester human prompt (R3 — derived from user data, belongs in HumanMessage) as a JSON list, with a system-prompt instruction telling the LLM to skip any candidate whose dedup_key is in the list. **R4 audit pass required** when the system prompt is touched.
- [x] Skip the evaluator only when the harvester returns structurally empty `new_facts == [] and future_promises == []`. No content-based pre-checks, no Python regex validator, no character-count heuristics.
- [x] Run `_update_user_image` and `_update_character_image` concurrently via `asyncio.gather`. Preserve each updater's existing LLM-driven early-exit behaviour.
- [x] Verify the harvester evaluator prompt's "旧闻复读" rule still references the correct field names after the `research_facts → rag_result` migration (`rag_result.user_image`, `rag_result.memory_evidence`). **R4 audit pass required.**

### Phase F — Service startup/shutdown

- [x] Edit `service.py` — remove cache1 warm-start (~line 436) and shutdown (~line 469); confirm no other Cache1 references in service module

### Phase G — Delete RAG1 modules

- [x] Delete `nodes/persona_supervisor2_rag.py`
- [x] Delete `nodes/persona_supervisor2_rag_supervisor.py`
- [x] Delete `nodes/persona_supervisor2_rag_executors.py`
- [x] Delete `nodes/persona_supervisor2_rag_resolution.py`
- [x] Delete `nodes/persona_supervisor2_rag_schema.py`
- [x] Delete `nodes/persona_supervisor2_consolidator_knowledge.py`
- [x] Delete `rag/cache.py`
- [x] Edit `rag/user_profile_agent.py` — remove `from kazusa_ai_chatbot.rag.depth_classifier import DEEP`; substitute literal or local constant
- [x] Delete `rag/depth_classifier.py`

### Phase H — Database

- [x] Edit `db/__init__.py` — remove imports + `__all__` for `RagCacheIndexDoc`, `RagMetadataIndexDoc`, `clear_all_cache_for_user`, `find_cache_entries`, `get_rag_version`, `increment_rag_version`, `insert_cache_entry`, `soft_delete_cache_entries`; update docstring submodule map (line 16)
- [x] Edit `db/schemas.py` — delete `RagCacheIndexDoc`, `RagMetadataIndexDoc` (lines 298–326)
- [x] Edit `db/bootstrap.py`:
  - [x] Add idempotent drop block for `rag_cache_index`, `rag_metadata_index`
  - [x] Remove the two collections from `required_collections`
  - [x] Remove their index-creation calls (~lines 122–141)
  - [x] Remove `enable_vector_index('rag_cache_index', ...)` (~line 147)
- [x] Delete `db/rag_cache.py`
- [x] Create `scripts/drop_legacy_rag_collections.py` per template above

### Phase I — Tests

- [x] Delete `tests/test_rag_cache.py`
- [x] Delete `tests/test_rag_live_llm.py`
- [x] Delete `tests/test_persona_supervisor2_rag_and_l2.py`
- [x] Delete `tests/test_persona_supervisor2_l3_and_consolidator.py`
- [x] Audit `tests/test_user_profile_memories.py`; remove cache1 references
- [x] Audit `tests/test_e2e_live_llm.py`; replace cache1 assertions with Cache2 stats
- [x] Create `tests/test_rag_projection.py` — one fixture per agent class; edge cases (empty, multi-user, character + current user mixed)
- [x] Create `tests/test_db_writer_cache2_invalidation.py` — one test per row of the event-emission table
- [x] Create `tests/test_save_conversation_invalidation.py` — store conversation cache entries, call `save_conversation`, assert eviction
- [x] Create `tests/test_persona_supervisor2_rag2_integration.py` — stage_1_research end-to-end with mocked LLM
- [x] Create `tests/test_consolidator_facts_rag2.py` — facts-harvester against new `rag_result` shape
- [x] Create `tests/test_consolidator_efficiency.py` — `existing_dedup_keys` set construction, harvester prompt payload includes the key list, evaluator skipped when harvester returns empty lists, image updaters run concurrently via `asyncio.gather`

### Phase J — Verification

- [x] Run all static greps from §"Verification — Static greps"; confirm zero matches in `src/` for forbidden RAG1 symbols. Legacy collection names remain only in the required bootstrap drop block, ops script, and drop tests.
- [x] `pytest tests/test_persona_supervisor2_rag_supervisor2_live.py` — passes
- [x] `pytest tests/test_rag_initializer_cache2.py` — passes
- [x] `pytest tests/test_rag_projection.py` — passes
- [x] `pytest tests/test_db_writer_cache2_invalidation.py` — passes
- [x] `pytest tests/test_save_conversation_invalidation.py` — passes
- [x] `pytest tests/test_persona_supervisor2_rag2_integration.py` — passes
- [x] `pytest tests/test_consolidator_facts_rag2.py` — passes
- [x] `pytest tests/test_consolidator_efficiency.py` — passes
- [x] Service boot: `uvicorn kazusa_ai_chatbot.service:app` — clean startup
- [x] Live `/chat` smoke: one request returns non-empty response, logs show Cache2 hit/miss stats
- [x] Live `/chat` smoke: two requests in the same channel → `conversation_history` invalidation event in logs after each save
- [x] Live `/chat` smoke: same fact-bearing message twice → second turn's harvester emits `new_facts: []` (LLM honoured `existing_dedup_keys`); evaluator is skipped; DB upsert is the only safety net exercised
- [x] Mongo: `db.getCollectionNames()` does NOT include `rag_cache_index` or `rag_metadata_index`

### Phase K — Documentation

- [x] Update README.md if it references RAG1, Cache1, `rag_cache_index`, depth classifier, or knowledge_base
- [x] Update `development_plans/rag_cache2_design.md` if any decision in this refactor diverges from the design (e.g., `user_image` source not emitted today; affinity-event always invalidates `user_profile`; `save_conversation` invalidation wrapped at db layer)

---

## Critical files referenced

- [src/kazusa_ai_chatbot/nodes/persona_supervisor2.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2.py)
- [src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py)
- [src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py)
- [src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py)
- [src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py)
- [src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py)
- [src/kazusa_ai_chatbot/rag/cache2_runtime.py](src/kazusa_ai_chatbot/rag/cache2_runtime.py)
- [src/kazusa_ai_chatbot/rag/cache2_events.py](src/kazusa_ai_chatbot/rag/cache2_events.py)
- [src/kazusa_ai_chatbot/db/conversation.py](src/kazusa_ai_chatbot/db/conversation.py)
- [src/kazusa_ai_chatbot/db/__init__.py](src/kazusa_ai_chatbot/db/__init__.py)
- [src/kazusa_ai_chatbot/db/schemas.py](src/kazusa_ai_chatbot/db/schemas.py)
- [src/kazusa_ai_chatbot/db/bootstrap.py](src/kazusa_ai_chatbot/db/bootstrap.py)
- [src/kazusa_ai_chatbot/service.py](src/kazusa_ai_chatbot/service.py)
- [development_plans/rag_cache2_design.md](development_plans/rag_cache2_design.md)
