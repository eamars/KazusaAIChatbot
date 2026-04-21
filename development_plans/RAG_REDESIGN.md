# RAG Architecture Redesign Plan

## Problem Statement

The current `user_fact_rag_dispatcher` generates search queries derived from the **input message**, causing the character's "knowledge" of the user to vary based on what the user is currently talking about. This produces inconsistent responses where the character may "know" different aspects of the user depending on the current topic.

Additionally:
- "Facts about the user's input" are currently mixed into User RAG instead of being handled by Internal RAG.
- There is no mechanism for the character to retrieve her own self-knowledge independent of user context.
- Internal RAG is too tightly user-anchored and lacks a long-lived knowledge accumulation layer.
- The consolidator (Stage 4) has no mechanism to update persistent image representations after a conversation.
- Images have no size control — without compaction they grow unboundedly; without milestone protection, important states fade.

---

## Desired Architecture: Three Stable Layers

| Layer | Source | Input-Dependence | Storage | Updated by |
|-------|--------|-----------------|---------|------------|
| **User Image** | How character sees this user | Independent | `user_profile.user_image` (persistent) | Consolidator (Stage 4) |
| **Character Self-Image** | How character sees herself | Independent | `character_state.self_image` (persistent) | Consolidator (Stage 4) |
| **Input-Context RAG** | Facts relevant to current topic | Input-correlated | RAG cache (short TTL) + `knowledge_base` (long TTL) | RAG subgraph (Stage 1) + Consolidator |

The first two layers are **stable anchors** stored directly in MongoDB profile documents — no cache TTL, no invalidation races. They are read directly from the profile at the start of each RAG pass and written back by the consolidator after each conversation where the image changed.

---

## Image Bookkeeping Rules

### Three-Tier Structure

Both `user_profile.user_image` and `character_state.self_image` are stored as structured documents, not plain strings:

```json
{
  "milestones": [
    {
      "event": "User requested English-only communication",
      "timestamp": "2026-01-15T10:00:00Z",
      "category": "preference"
    },
    {
      "event": "User and character entered a close relationship",
      "timestamp": "2026-02-01T15:30:00Z",
      "category": "relationship_state"
    }
  ],
  "recent_window": [
    { "timestamp": "2026-04-20T...", "summary": "Session narrative: ..." },
    { "timestamp": "2026-04-21T...", "summary": "Session narrative: ..." }
  ],
  "historical_summary": "Compressed narrative of the older relationship arc...",
  "meta": {
    "synthesis_count": 12,
    "last_updated": "2026-04-21T..."
  }
}
```

**Milestones** — Append-only, never compacted. These are relationship-defining states that must never fade regardless of how many conversations pass. Max ~80 words per entry; no cap on number of entries (milestones are rare by nature).

**Recent window** — Last N conversation summaries, kept at full narrative fidelity. Max size: 6 entries. When a 7th entry would be added, the oldest is merged into `historical_summary`.

**Historical summary** — Compressed narrative of older relationship history. Kept under a token budget (~600 tokens). When budget is exceeded, LLM compresses it in place, preserving distinctive traits and pivotal moments, discarding transient interactions.

### Token Budget

| Tier | Budget | Rationale |
|------|--------|-----------|
| Milestones | ~50 tokens/entry, no entry limit | Rare events; budget per-entry not total |
| Recent window | ~120 tokens/entry × 6 = ~720 tokens | Full fidelity, bounded window |
| Historical summary | max 600 tokens | Progressively compressed |
| **Total assembled image** | **~2000 tokens max** | Fits comfortably in dialog generator context |

### Rolling Mechanism

Each consolidator run that triggers synthesis follows this sequence:

```
1. Extract session delta: diary_entry + new_facts → session_summary (LLM, ~120 tokens)
2. Separate milestone facts from non-milestone facts (see Milestone Detection)
3. Append milestone facts to milestones[] list (bypass rolling entirely)
4. Append session_summary to recent_window[]
5. If len(recent_window) > MAX_WINDOW_SIZE (6):
       oldest_entry = recent_window.pop(0)
       historical_summary = LLM_merge(historical_summary, oldest_entry)
       if token_count(historical_summary) > 600:
           historical_summary = LLM_compress(historical_summary)
```

The LLM compressor is explicitly instructed: *"Preserve: distinctive personality traits, pivotal events, relationship arc direction. Discard: transient moods, casual topic mentions, repeated observations already captured."*

### Milestone Detection

Milestones are facts that define **persistent state** — they cannot fade or be overwritten by later conversations. Detected in `facts_harvester` by adding `is_milestone: bool` and `milestone_category: str` to each extracted fact.

**Milestone categories:**

| Category | Examples |
|----------|---------|
| `preference` | Language choice, communication style, explicit rules user set ("don't talk about X") |
| `relationship_state` | Relationship formation, major conflict, reconciliation, role definitions |
| `permission` | What user has explicitly allowed or forbidden the character to do/say |
| `revelation` | Character secrets or personal history shared with the user |

Milestone facts are **always prepended** to the assembled image text sent to the dialog generator, regardless of how much rolling/compaction has occurred. This ensures states like "user wants English" or "we are in a relationship" are always visible.

**Character image milestones** follow the same pattern, covering: core beliefs, relationship status, key self-revelations, domain expertise the character holds.

### Milestone Supersession

A milestone can be superseded (replaced) but never silently compacted. If a later fact explicitly contradicts an existing milestone (e.g., user switches back to Chinese after requesting English), the old milestone is marked `superseded_by: <new_milestone_id>` and the new one is appended. The history is preserved; only the active (non-superseded) milestones are shown to the dialog generator.

---

## Gap Analysis vs Current Design

| Aspect | Current | Desired | Gap |
|--------|---------|---------|-----|
| User RAG query source | Input-derived queries | Profile field read (`user_profile.user_image`) | Dispatcher → profile reader |
| User RAG storage | RAG cache (1h TTL) | Persistent MongoDB field, three-tier structure | New schema, remove cache type |
| User image size control | None | Rolling window + historical compaction + milestones | New bookkeeping logic |
| User image update | Manual cache invalidation | Consolidator writes on trigger | New consolidator step |
| Character Self-RAG | None | `character_state.self_image`, same three-tier structure | New profile field + consolidator step |
| Internal RAG focus | Entity-anchored, per-user strict | Input-topic primary, user secondary, cross-user allowed | Dispatcher refactor |
| Knowledge Base | None | `knowledge_base` RAG cache (30d) accumulated by consolidator | New cache type + consolidator step |
| `research_facts` schema | 3 keys | 5 keys reflecting the cleaner separation | Schema extension |
| Consolidator scope | Writes facts/diary/affinity | Also synthesizes images + knowledge | 3 new synthesis responsibilities |

---

## Phase 1: User Image — From RAG Cache to Persistent Profile Field

### 1.1 Read Path: Profile Field Instead of Dispatcher

The `user_fact_rag_dispatcher` is replaced by a **direct profile read**. At the start of the RAG subgraph, `user_profile.user_image` is read from MongoDB alongside the existing `user_profile` fetch.

No cache lookup, no dispatcher, no LLM call on the read path. The image is ready immediately as part of the profile load.

If `user_profile.user_image` is empty (new user), the dialog generator receives an empty/minimal image and the consolidator populates it after the first conversation.

### 1.2 Image Assembly for Dialog Generator

Before surfacing to the dialog generator, the three-tier structure is assembled into a flat text block:

```
[PERMANENT STATES]
- User requested English-only communication (2026-01-15)
- User and character entered a close relationship (2026-02-01)

[RELATIONSHIP HISTORY]
<historical_summary>

[RECENT INTERACTIONS]
<recent_window entries, newest last>
```

The affinity-based distortion (currently in `user_fact_rag_finalizer`) is applied to the assembled block, not to the stored image.

### 1.3 Write Path: Consolidator Updates the Image

**Trigger conditions** (any one sufficient):
1. `diary_entries` were written this session
2. `objective_facts` were written this session
3. `|processed_affinity_delta|` exceeds threshold
4. `last_relationship_insight` was updated

**Write target**: `user_profile.user_image` (the full three-tier document).

### 1.4 `research_facts` Key Rename

`user_rag_finalized` → `user_image`

---

## Phase 2: Character Self-Image — Same Mechanism as User Image

### 2.1 Read Path: `character_state.self_image` Field

Add `self_image` (three-tier document) to the `character_state` MongoDB document. Read at the start of the RAG subgraph — no dispatcher, no LLM call, no cache lookup.

### 2.2 Write Path: Consolidator Updates the Character Image

**Trigger conditions**:
1. `reflection_summary` was produced (`global_state_updater` ran)
2. Periodic refresh: every N conversations (prevents staleness even during quiet periods)

**Write target**: `character_state.self_image`.

Character image milestones include: core beliefs, relationship status with the user, knowledge domains the character holds, key self-revelations.

### 2.3 New `research_facts` Key: `character_image`

Surfaced to the dialog generator alongside `user_image`, assembled using the same three-tier format.

---

## Phase 3: Internal RAG Refactor + Knowledge Base

### 3.1 Input Focus Shift for `internal_rag_dispatcher`

**Changes:**
- Weight **input topic** as the primary search signal
- User context (`global_user_id`) becomes a secondary re-ranking filter, not a hard retrieval constraint
- Allow top-K results without `global_user_id` filter — cross-user facts about the same topic are eligible
- Re-rank: user-scoped results ranked above cross-user results of equal relevance

Rename to `input_context_rag_dispatcher`.

### 3.2 New Cache Type: `knowledge_base`

- TTL: `2592000s` (30 days)
- Scope: global (`global_user_id = ""`)
- Populated by the consolidator after each DEEP dispatch (Phase 5.3)

**Probe order in RAG subgraph:**
```
1. Check knowledge_base cache (topic match, global scope)
2. Check input_context cache (short-TTL, per-user)
3. Both miss → run full internal dispatch
```

### 3.3 Knowledge Accumulation

Written by the consolidator after the conversation (not during retrieval — read path stays clean). See Phase 5.3.

---

## Phase 4: Infrastructure Updates

### 4.1 New Cache Config Entry (`knowledge_base` only)

```python
# config.py
RAG_CACHE_KNOWLEDGE_BASE_TTL = int(os.getenv("RAG_CACHE_KNOWLEDGE_BASE_TTL", 2592000))

# rag/cache.py DEFAULT_TTL_SECONDS
"knowledge_base": 2592000,
```

### 4.2 Depth Classifier Update

- SHALLOW: profile reads for `user_image` + `character_image` always happen (part of profile load)
- DEEP: `input_context_rag` + `external_rag` + `knowledge_base` probe
- Revisit affinity < 400 DEEP override — recalibrate once stable images are always present

### 4.3 `clear_all_user` Update

- Remove: `user_image` (no longer in RAG cache)
- Add: `knowledge_base` partial clear (user-tagged entries only)

### 4.4 `research_facts` Schema After All Phases

```python
research_facts = {
    "user_image":             "...",  # assembled from user_profile.user_image tiers
    "character_image":        "...",  # assembled from character_state.self_image tiers
    "input_context_results":  "...",  # topic-correlated facts (was internal_rag_results)
    "knowledge_base_results": "...",  # accumulated cross-session knowledge
    "external_rag_results":   [...],  # unchanged
}
```

---

## Phase 5: Consolidator Updates

The consolidator gains three new synthesis responsibilities, added as steps inside `db_writer` or as dedicated pre-`db_writer` nodes.

### 5.1 `user_image_synthesizer`

**Trigger gate**: diary written OR objective facts written OR `|affinity_delta| > threshold`.

**Rolling update** (per bookkeeping rules):
1. Classify new facts as milestone vs. non-milestone
2. Append milestone facts to `milestones[]`
3. Build `session_summary` from non-milestone diary/facts
4. Append `session_summary` to `recent_window[]`
5. If `recent_window` exceeds max: merge oldest into `historical_summary`, compress if over budget

**Write**: `await upsert_user_image(global_user_id, updated_image_doc)`

### 5.2 `character_image_synthesizer`

**Trigger gate**: `reflection_summary` non-empty OR N conversations since last update.

**Rolling update**: same three-tier mechanism, sourcing from `mood`, `global_vibe`, `reflection_summary`, `character_profile`.

**Write**: `await upsert_character_self_image(updated_image_doc)`

### 5.3 `knowledge_base_updater`

**Trigger gate**: RAG metadata confirms DEEP dispatch ran this conversation.

**Action**:
1. Distill topic-level facts from `research_facts.input_context_results` + `external_rag_results`
2. Embed distilled topics
3. Store to `knowledge_base` RAG cache (global scope, 30d TTL)

### 5.4 `ConsolidatorState` Extension

```python
class ConsolidatorState(TypedDict):
    # ... existing fields ...
    research_facts: dict    # from GlobalPersonaState, for knowledge_base_updater
    research_metadata: list # to detect if DEEP dispatch ran
```

Seeded in `call_consolidation_subgraph`:
```python
"research_facts":    global_state.get("research_facts", {}),
"research_metadata": global_state.get("research_metadata", []),
```

---

## Implementation Plan

### Compatible Changes
*Additive — can be deployed incrementally without breaking existing behavior. Old code paths continue to work during rollout.*

| # | Change | Why compatible |
|---|--------|---------------|
| C1 | Add `user_image` document field to user profile schema (empty initially) | Additive field; old code ignores it |
| C2 | Add `self_image` document field to `character_state` schema | Additive field |
| C3 | Add `is_milestone` + `milestone_category` to `facts_harvester` output schema | New keys; evaluator and db_writer ignore unknown keys |
| C4 | Add `user_image_synthesizer` step to `db_writer` (writes to new field only) | New write; doesn't touch existing writes |
| C5 | Add `character_image_synthesizer` step to `db_writer` | Same |
| C6 | Add `knowledge_base` cache type to `config.py` + `rag/cache.py` | New cache type; old probe paths unaffected |
| C7 | Add `research_facts` + `research_metadata` to `ConsolidatorState` and seed them | New state keys; existing nodes ignore them |
| C8 | Add `knowledge_base_updater` step to `db_writer` | New write path; doesn't alter existing behavior |
| C9 | Read `user_image` + `character_image` from profile and surface as **additional** keys in `research_facts` | Dialog agent gets more data; `user_rag_finalized` still present |
| C10 | Add `knowledge_base` probe in RAG subgraph as pre-check before internal dispatch | Additive probe; dispatch still runs on miss |

### Breaking Changes
*Each item in this group changes existing behavior. All items within a group must deploy atomically — partial deployment causes regression.*

#### Group B1: User Image Switchover
*Deploy together. Old `user_rag_finalized` path removed; `user_image` from profile becomes the sole source.*

- Remove `user_fact_rag_dispatcher` and `user_fact_rag_finalizer` from RAG subgraph
- Remove `user_rag_finalized` key from `research_facts`
- Rename `user_image` (additional key from C9) to the primary key — dialog agent now reads `user_image` only
- Update dialog agent system prompt to reference `user_image` instead of `user_rag_finalized`
- **Prerequisite**: C1 + C4 have been deployed and `user_image` is populated for all active users (or fallback to empty image handled gracefully)

#### Group B2: Character Self-Image Activation
*Deploy together. Character image becomes part of the dialog generator's context.*

- `character_image` (from C9) promoted from additional key to required key in `research_facts`
- Update dialog agent system prompt to reference `character_image`
- **Prerequisite**: C2 + C5 deployed and `self_image` is populated

#### Group B3: Input-Context RAG Dispatcher Refactor
*Deploy together. Cross-user queries change retrieval behavior globally — cannot be partially rolled out.*

- Rename `internal_rag_dispatcher` → `input_context_rag_dispatcher`
- Modify retrieval to use input-topic primary signal + cross-user results
- Rename `internal_rag_results` → `input_context_results` in `research_facts`
- Update dialog agent system prompt to reference `input_context_results`

#### Group B4: Depth Classifier Recalibration
*Deploy alone after B1–B3 are stable. Changing routing affects all conversations.*

- Recalibrate SHALLOW/DEEP thresholds now that stable images remove the need for the affinity < 400 DEEP override
- Update depth classifier keyword centroids if needed

#### Group B5: Legacy Cleanup
*Deploy last. Removes deprecated code paths and cache types.*

- Remove `user_facts` legacy cache type (already marked deprecated)
- Update `clear_all_user` to remove old cache types, add `knowledge_base` partial clear
- Remove any remaining references to `user_rag_finalized`

---

## Future Work

- **Knowledge Graph**: Evolve `knowledge_base` into a structured entity/concept graph.
- **Image Versioning**: Keep snapshots of `user_image` and `character_state.self_image` to enable relationship arc analysis.
- **Milestone Review**: Periodic LLM pass to flag outdated milestones for user/operator review.
