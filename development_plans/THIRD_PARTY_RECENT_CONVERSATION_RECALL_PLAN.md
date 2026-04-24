# Unified RAG Solution Plan

## Problem Statement

The current Stage-3 input-context retrieval path is optimized for the **current user** and the **current channel window**.

It fails when the user asks, implicitly or indirectly, about **someone else** the character recently talked about.

Examples:

- "她刚刚说的那个你还记得吗？"
- "你前面不是还在跟别人聊这个吗？"
- "你最近提过他来着，后来怎样了？"
- "那个叫啾啾的，你之前怎么说她的？"

The root cause is that every retrieval decision is force-scoped to the current user:

- `target_user_name = current user`
- `target_global_user_id = current user`
- `target_platform_channel_id = current channel`

This is a **retrieval scoping problem**, not just a prompt problem.

---

## Implementation Status

| Phase | Description                                                                     | Status        |
| ----- | ------------------------------------------------------------------------------- | ------------- |
| 1     | Planner refactor — continuation resolver + enriched retrieval plan fields       | ⬜ NOT STARTED |
| 2     | Retriever refactor — plan-driven scopes, third-party backends, retrieval ledger | ⬜ NOT STARTED |
| 3     | Entity memory layer — durable entity/topic memory substrate                     | ⬜ DEFERRED    |
| 4     | Dialog surface upgrade — research_facts extension for third-party results       | ⬜ NOT STARTED |
| 5     | Bounded supervisor centralization — rag_supervisor_planner / evaluator          | ⬜ DEFERRED    |
| 6     | RAG module decomposition — split into focused submodules                        | ⬜ DEFERRED    |
| 7     | Evaluation — test coverage for third-party recall scenarios                     | ⬜ NOT STARTED |

---

## Engineering Recommendations

**Priority order for next implementation work:**

1. **Phase 1 first** — the enriched retrieval plan is the scaffolding everything else depends on. Without `retrieval_mode` and `entities` in the planner output, Phase 2 has nothing to execute against.

2. **Phase 4 alongside Phase 1** — extending `research_facts` is a low-cost dict change that should be done at the same time to avoid a second round of downstream prompt updates.

3. **Phase 2 next, prioritizing `CHANNEL_RECENT_ENTITY`** — this backend directly addresses the core third-party recall problem using existing conversation history, without requiring new storage.

4. **Defer Phase 3** — the `CHANNEL_RECENT_ENTITY` backend in Phase 2 covers most implicit recall cases without a new entity_memory collection. Introduce entity memory only if Phase 2 leaves clear gaps.

5. **Defer Phase 5** — the bounded supervisor is the most disruptive structural change. Let Phases 1-4 stabilize first to confirm the retrieval plan contract before centralizing orchestration.

6. **Phase 6 incrementally** — add tests as each phase ships rather than deferring all evaluation to the end.

---

## Unified Design Principle

RAG should be organized by **memory layer** and **subject identity**, not just by tool or query text.

Every retrieval decision should answer four questions:

1. **What subject is this about?** — current user / third-party user / non-user entity/topic
2. **What kind of memory is needed?** — stable profile/image / recent event / durable entity memory / external knowledge
3. **What scope is allowed?** — same channel recent window / same platform internal / global durable internal / external web
4. **Is the task resolved enough to retrieve yet?** — if not, continuation repair or clarification before broad retrieval

---

## Architecture Overview

### Target Pipeline

```text
message
  -> decontextualizer
  -> continuation_resolver                    ← Phase 1
  -> rag_planner                              ← Phase 1 (selects sources, not order)

  [Tier 0 — preloaded anchors, always available]
       current_user_image                     ← Phase 0 ✅
       character_image                        ← Phase 0 ✅

  [Tier 1 — parallel, no internal dependencies]
       entity_resolution                      ← Phase 1 (gates Tier 2)
       input_context_rag                      ← existing
       knowledge_base_rag                     ← existing

  [Tier 2 — parallel, gated on Tier 1 entity resolution]
       third_party_profile_rag                ← Phase 2, optional (needs resolved ID)
       channel_recent_entity_rag              ← Phase 2, optional (uses resolved ID for precision)
       [repeat per resolved entity if multiple subjects]

  [Tier 3 — sequential after all internal tiers complete]
       external_rag                           ← existing (informed by Tier 1+2 results)

  -> rag_evaluator                            ← Phase 5 (bounded, deferred)
  -> one repair pass if cascading dependency  ← Phase 5 (load-bearing, see below)
  -> rag_finalizer
```

### Unified RAG Layers

| Layer                              | Subject                                         | Purpose                                  | Storage / Source                      | Input Dependence        |
| ---------------------------------- | ----------------------------------------------- | ---------------------------------------- | ------------------------------------- | ----------------------- |
| **Current User Image**             | current speaker                                 | stable relationship / profile context    | `user_profile.user_image`             | independent             |
| **Character Image**                | character self                                  | stable self-knowledge / stance           | `character_state.self_image`          | independent             |
| **Input Context**                  | current task subject                            | recent conversational facts/events       | conversation search + internal memory | input-correlated        |
| **Entity Memory / Knowledge Base** | third-party users / recurring entities / topics | durable internal knowledge               | `entity_memory` + `knowledge_base`    | weakly input-correlated |
| **External Knowledge**             | public world/topic                              | outside knowledge not present internally | web / external retrieval              | input-correlated        |

The first two layers are **always-preloaded anchors** (Phase 0 complete). The other three are **planned retrieval sources** chosen per request.

---

## Retrieval Modes

The planner selects one or more retrieval modes per request.

**Note**: Such table needed to be included in the final code for the LLM planner to give better view of decision here. 

| Mode                      | Tier | Meaning                                                                       | Typical use                               |
| ------------------------- | ---- | ----------------------------------------------------------------------------- | ----------------------------------------- |
| `CURRENT_USER_STABLE`     | 0    | Use already loaded current-speaker anchors                                    | preferences, promises, relationship state |
| `THIRD_PARTY_PROFILE`     | 2    | Retrieve a known other user's profile/image                                   | "what do you remember about Jiujiu?"      |
| `CHANNEL_RECENT_ENTITY`   | 2    | Recall recent mentions/events about another person/entity in the same channel | "you were just talking about her"         |
| `GLOBAL_ENTITY_KNOWLEDGE` | 1    | Recall durable internal knowledge about a third party/entity/topic            | recurring named people, projects, groups  |
| `EXTERNAL_KNOWLEDGE`      | 3    | Retrieve outside-world knowledge                                              | factual web/domain questions              |
| `CASCADED`                | 1-3  | Activate multiple sources; compose via three-tier execution model             | any multi-source or group-chat scenario   |

Important: the **LLM planner** decides *which sources to activate*, not their execution order. Order is a structural property of tier membership — the planner never specifies it.

### CASCADED mode composition

In CASCADED mode, sources compose via the three-tier model. The planner output for CASCADED adds one optional field:

```json
{
  "retrieval_mode": "CASCADED",
  "active_sources": ["CHANNEL_RECENT_ENTITY", "THIRD_PARTY_PROFILE", "EXTERNAL_KNOWLEDGE"],
  "entities": [...],
  "external_task_hint": "search for X once internal recall completes"
}
```

`external_task_hint` lets the planner prime the Tier 3 search with predicted context. The executor uses actual Tier 1+2 results when they arrive, falling back to the hint only if internal sources return nothing.

**The planner does not specify order.** Tier membership is deterministic:

- Modes in Tier 1 always run before modes in Tier 2 — because Tier 2 needs resolved entity IDs from Tier 1's entity_resolution step.
- Modes in Tier 3 always run after all internal tiers — so external search is informed by what internal sources already found.

---

## Execution Order

Execution order is **not decided by the planner**. It is a fixed structural property of source type. This keeps the planner simple and the execution model testable without LLM involvement.

### Tier membership

| Tier | Sources                                                        | Runs when                         | Can run in parallel with               |
| ---- | -------------------------------------------------------------- | --------------------------------- | -------------------------------------- |
| 0    | `current_user_image`, `character_image`                        | Always, zero cost                 | Everything                             |
| 1    | `entity_resolution`, `input_context_rag`, `knowledge_base_rag` | Immediately after planner         | Each other                             |
| 2    | `third_party_profile_rag`, `channel_recent_entity_rag`         | After Tier 1 completes            | Each other; repeat per resolved entity |
| 3    | `external_rag`                                                 | After all internal tiers complete | Nothing                                |

### Why this ordering

**Tier 1 gates Tier 2.** `THIRD_PARTY_PROFILE` and `CHANNEL_RECENT_ENTITY` both need a resolved entity ID to be precise. Without it, profile retrieval has no target and channel search falls back to surface-form keyword matching. Entity resolution runs in Tier 1 so Tier 2 gets the confirmed ID before fetching.

**Tier 3 is always last.** External search is most useful when it knows what internal sources already found — it searches for what the character doesn't already remember. Running external in parallel with internal means the external dispatcher builds its query blind.

**Same source can activate multiple times in Tier 2.** In group-chat scenarios with multiple resolved entities (e.g., "what happened between them?"), `channel_recent_entity_rag` fires once per resolved subject, all in parallel within Tier 2.

### Dependency types

| Type                  | Description                                                                | Handled by                                       |
| --------------------- | -------------------------------------------------------------------------- | ------------------------------------------------ |
| Resolution dependency | Profile/channel needs entity ID before fetching                            | Tier 1 → Tier 2 gate                             |
| Scope dependency      | External needs internal context to build a targeted query                  | Tier 1+2 → Tier 3 gate                           |
| Validation dependency | Multiple internal sources cross-referenced                                 | Parallel within Tier 2, synthesis after          |
| Cascading dependency  | Result of retrieval reveals a new retrieval target unknowable at plan time | Phase 5 evaluator repair pass (one pass maximum) |

### Cascading dependencies and the Phase 5 repair pass

Some group-chat scenarios produce a target that is unknowable at plan time:

> "你之前说会帮她找的那个人，找到了吗？"
> *(That person you said you'd help her find — did you find them?)*

Tier 1+2 fetches the character's commitment to "her", which reveals the target person's name. Only then can `channel_recent_entity_rag` be re-run for that specific person. This cannot be expressed in the initial plan.

The Phase 5 evaluator handles exactly this: after the three-tier pass, it inspects the aggregated results and decides whether coverage is sufficient. If a cascading gap is found, it triggers **one additional sub-pass** — a fresh Tier 1-3 run scoped to the newly revealed target. The `max evaluator-triggered repair: 1` limit is **load-bearing** for this design — without it, cascading scenarios would chain into an unbounded loop.

---

## Phase 1 — Continuation Resolver + Planner Refactor

### What changes

Add a `continuation_resolver` step between the decontextualizer and RAG dispatch. This decides whether the input is retrieval-ready before retrieval fans out.

Responsibilities:

- resolve reply-only or continuation-like turns
- reconstruct omitted objects / tasks when evidence is available
- decide whether retrieval should operate on the raw decontextualized message or a repaired task packet

Output contract:

```json
{
  "needs_context_resolution": true,
  "resolved_task": "string",
  "known_slots": {},
  "missing_slots": [],
  "confidence": 0.0,
  "evidence": ["reply_target", "recent_turn"]
}
```

If confidence is too low, Stage 1 should prefer clarification over broad retrieval.

### Enriched planner output

Extend the Stage-1 planner to emit a richer retrieval plan alongside the existing output.

```json
{
  "retrieval_mode": "CURRENT_USER_STABLE | THIRD_PARTY_PROFILE | CHANNEL_RECENT_ENTITY | GLOBAL_ENTITY_KNOWLEDGE | EXTERNAL_KNOWLEDGE | CASCADED | NONE",
  "active_sources": ["CHANNEL_RECENT_ENTITY", "THIRD_PARTY_PROFILE", "EXTERNAL_KNOWLEDGE"],
  "task": "string",
  "entities": [
    {
      "surface_form": "啾啾",
      "entity_type": "person | group | topic | unknown",
      "resolution_confidence": 0.0,
      "resolved_global_user_id": "optional"
    }
  ],
  "subject": {
    "kind": "current_user | third_party_user | entity | topic | mixed",
    "primary_entity": "optional surface form or user id"
  },
  "time_scope": {
    "kind": "recent | explicit_range | none",
    "lookback_hours": 72
  },
  "search_scope": {
    "same_channel": true,
    "cross_channel": false,
    "current_user_only": false
  },
  "external_task_hint": "optional — primes Tier 3 search with predicted context; executor uses actual Tier 1+2 results when available",
  "repair_allowed": true,
  "clarify_allowed": true,
  "expected_response": "string"
}
```

This makes the retrieval target explicit and removes the assumption that every query is about the current user.

### Entity grounding

Add a lightweight entity grounding step after the planner.

Responsibilities:

- normalize entity mentions from `decontexualized_input`, `reply_context`, `chat_history_recent`
- attempt resolution against known participant display names in recent history, linked platform identities, optional alias tables

Outputs:

- resolved entity targets when possible
- unresolved but usable surface forms otherwise

Entity grounding **enriches** the retrieval plan but must not deterministically change the user's semantic intent.

Failed grounding affects source choice:

- unresolved third-party mention → prefer `CHANNEL_RECENT_ENTITY`
- resolved stable known user → `THIRD_PARTY_PROFILE` may be allowed
- unresolved world topic → prefer `GLOBAL_ENTITY_KNOWLEDGE` or `EXTERNAL_KNOWLEDGE`

---

## Phase 2 — Retriever Refactor

Replace the current hard-override approach with a plan-driven contract.

**Today:** `call_memory_retriever_agent_input_context_rag()` force-injects current-user scope.

**Desired:** the planner emits a retrieval plan; the retriever executes exactly that plan; deterministic code only injects structural bounds (platform, safe time ranges, max result counts).

### A. Current user self backend

Continue using existing tools:

- `search_user_facts`
- `character_diary`
- `objective_facts`
- active commitments

`CURRENT_USER_STABLE` is mostly a planning label for preloaded sources, not a signal to do extra DB work.

### B. Third-party profile backend

Use internal profile/image retrieval only when the planner concludes the response depends on remembered identity or enduring traits of a specific other user.

Trigger only when **all** hold:

- the message is substantially about a specific other person
- that person can be resolved with usable confidence
- event-only recent context is insufficient
- the response requires remembered profile-level information

Surfaces:

- `third_party_user_image[target_user_id]`
- selected persistent facts / relationship notes about that target

### C. Channel recent entity backend

Use conversation history as the primary source for third-party recall.

Query pattern:

- default scope: same platform + same channel + bounded recent window
- search by: resolved user ID when available, otherwise surface-form keyword search, fallback semantic search with entity-aware query text

This backend answers:

- "what was said about X recently?"
- "who were you talking about just now?"
- "what happened with that person from earlier?"

**This is the highest-priority backend to implement** — it requires no new storage and directly addresses the core problem.

### D. Global entity knowledge backend

Use durable entity summaries for recurring third parties.

Separate from raw user-profile memory. A distinct logical layer for:

- recurring named people
- recurring projects / places / groups
- long-lived relationship context not tied to one current speaker

Implementation options:

- extend `knowledge_base` with `subject_kind` tagging, or
- separate `entity_memory` collection (see Phase 3)

### E. External knowledge backend

Must remain history-blind and consume the **resolved task packet**, not raw continuation fragments.

### Request-local retrieval ledger

Add a request-local ledger to prevent duplicate subject fetches.

Ledger key: `(source_type, subject_key, retrieval_purpose)`

Examples:

- `("current_user_image", current_global_user_id, "profile_context")`
- `("third_party_profile", target_global_user_id, "relationship_context")`
- `("channel_recent_entity", "啾啾", "recent_event_recall")`
- `("knowledge_base", "glitch", "durable_entity_knowledge")`

Minimum rules:

- current-user preload registers itself immediately
- a third-party profile resolving to the current speaker must not be fetched again
- if a planner restates the same need, executor returns the already-loaded source summary instead of re-querying
- unresolved entities must not bounce repeatedly between profile retrieval and conversation search

---

## Phase 3 — Entity Memory Layer (Deferred)

**Defer until Phase 2 confirms a gap.** The `CHANNEL_RECENT_ENTITY` backend in Phase 2 covers most implicit recall cases using existing conversation history. Introduce entity memory only if Phase 2 leaves clear shortfalls.

If needed, add a new collection or cache-backed document type:

```json
{
  "entity_key": "normalized identifier",
  "display_names": ["啾啾", "Jiujiu"],
  "entity_type": "person",
  "resolved_global_user_id": "optional",
  "recent_mentions": [
    {
      "timestamp": "ISO-8601",
      "platform": "discord",
      "platform_channel_id": "...",
      "summary": "short factual summary of what was recently said/done"
    }
  ],
  "historical_summary": "compressed longer-term summary",
  "embedding": []
}
```

Population:

- on conversation write or consolidator pass
- extract salient third-party entities mentioned this turn
- append/update a bounded recent-mentions window
- periodically compress into historical summary

This is also the natural merge point for `knowledge_base` — merge at the storage-framework level (one substrate), keep distinct at the planner / research-facts / prompt level.

---

## Phase 4 — Dialog Surface Upgrade

Extend `research_facts` to carry separate retrieval result channels.

**Current shape** (research_facts today):

- `input_context_results`
- `external_rag_results`

**Target shape**:

```python
research_facts = {
    "user_image":                    "...",   # from Phase 0 preload
    "character_image":               "...",   # from Phase 0 preload
    "input_context_results":         "...",   # existing
    "third_party_profile_results":   "...",   # Phase 2
    "channel_recent_entity_results": "...",   # Phase 2
    "knowledge_base_results":        "...",   # existing, extended
    "external_rag_results":          [...],   # existing
    "entity_resolution_notes":       "...",   # Phase 1 grounding output
}
```

Downstream prompt updates to:

- explain source priority to cognition layers
- guide the character on how to weight stable anchors vs retrieved context
- prevent conflation of "what I remember durably" with "what was said recently about X"

---

## Phase 5 — Bounded Supervisor Centralization (Deferred)

**Defer until Phases 1-4 are stable.** This is the most disruptive structural change.

Replace pure split dispatch with a bounded `rag_supervisor_planner` / `rag_supervisor_evaluator` model.

### Recommended control limits

- max planning pass after continuation resolution: `1`
- max evaluator-triggered repair pass: `1` ← **load-bearing**: this is the only mechanism for cascading dependency scenarios (see Execution Order above); raising it to 2+ risks unbounded chaining
- max total new Stage-1 LLM rounds after decontextualizer: `2`
- no repeated ledger key in one request
- external retrieval may only consume a resolved task packet
- repair pass scope: the repair sub-pass runs a fresh Tier 1-3 cycle scoped only to the newly revealed target, not a full re-plan

### Why not keep pure split dispatchers?

Pure split dispatchers are now too narrow because the system needs top-level arbitration across:

- continuation repair
- third-party profile retrieval
- channel event recall
- durable internal knowledge
- external knowledge

### Why not use a free-form iterative supervisor?

Too expensive and too easy to turn into nested loops over planner → internal retriever → external retriever → evaluator. Wrong operational profile for local LLM deployment.

---

## Phase 6 — RAG Module Decomposition (Deferred)

**Defer until Phase 5 is complete.** The graph topology must be stable before splitting — Phase 5 is the most disruptive structural change and determines the final set of nodes. Splitting before Phase 5 would require restructuring again when the supervisor arrives.

**Trigger:** Do this even if Phase 5 remains deferred whenever `persona_supervisor2_rag.py` exceeds 5 LLM instances or ~800 lines (Phases 1–2 alone will push it past that threshold). Don't wait for Phase 5 if the file is already unmanageable.

### Why split

After Phase 5, the single RAG file will contain approximately 9 LLM instances:

| LLM                                    | Introduced in |
| -------------------------------------- | ------------- |
| `external_rag_dispatcher`              | existing      |
| `input_context_rag_dispatcher`         | existing      |
| `continuation_resolver`                | Phase 1       |
| `entity_grounder`                      | Phase 1       |
| `rag_planner`                          | Phase 1       |
| `third_party_profile_rag_dispatcher`   | Phase 2       |
| `channel_recent_entity_rag_dispatcher` | Phase 2       |
| `rag_supervisor_planner`               | Phase 5       |
| `rag_supervisor_evaluator`             | Phase 5       |

Plus all executor wrappers, state schemas, cache helpers, and image context builders. The file becomes unnavigable and mirrors the pre-refactor consolidator.

### Proposed split — follow the consolidator pattern

| Module                                  | Contents                                                                                                                                        |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `persona_supervisor2_rag.py`            | Slim orchestrator only: `call_rag_subgraph`, `_build_rag_graph`, `RAGState`                                                                     |
| `persona_supervisor2_rag_resolution.py` | Pre-retrieval layer: `continuation_resolver`, `entity_grounder`, enriched `rag_planner`                                                         |
| `persona_supervisor2_rag_executors.py`  | All dispatcher LLMs + executor wrappers (split further by source type if they diverge significantly)                                            |
| `persona_supervisor2_rag_supervisor.py` | `rag_supervisor_planner`, `rag_supervisor_evaluator`, repair-pass logic                                                                         |
| `rag/` subpackage                       | Push `_probe_cache`, `_probe_knowledge_base`, `_store_results_in_cache`, `_build_image_context` here — they are infrastructure, not graph nodes |

### Constraints

- This refactor is **behavior-neutral** — no prompt changes, no logic changes, only file boundaries move.
- Phase 7 evaluation tests should pass unchanged after the split.
- `RAGState` TypedDict stays in `persona_supervisor2_rag.py` (the orchestrator); submodules import it rather than each defining their own state.
- LLM singletons (e.g., `_external_rag_dispatcher_llm`) move with their owning node into the submodule — do not share LLM instances across module boundaries.

---

## Phase 7 — Evaluation

Add tests incrementally as each phase ships.

Test scenarios by phase:

**Phase 1:**

- Continuation-like messages that need pre-retrieval repair
- Planner correctly emits `CHANNEL_RECENT_ENTITY` for implicit third-party references
- Entity grounding resolves known display names from recent history

**Phase 2:**

- Implicit third-party recent mention recall (`CHANNEL_RECENT_ENTITY` backend)
- Explicit name recall resolving to `THIRD_PARTY_PROFILE`
- Mixed self + third-party recall
- Unresolved entity fallback to surface-form search
- No cross-user leakage when scope should stay local
- Duplicate-profile suppression via retrieval ledger
- External search receiving only resolved tasks, not raw continuation fragments

**Phase 3 (if implemented):**

- Entity memory populated correctly from consolidator pass
- Entity memory retrieved correctly for recurring people/entities

**Phase 5 (if implemented):**

- Bounded supervisor stays within max LLM round limits
- Evaluator triggers at most one repair pass

**Phase 6:**

- All nodes importable from their respective submodules without circular dependencies
- `call_rag_subgraph` behaviour identical before and after the split (no golden-path regression)

---

## Scope Policy

Consistent privacy / scope ladder for retrieval.

Recommended default order:

1. Already-loaded current-user / character anchors
2. `same_channel + recent window`
3. `same_platform + entity memory or third-party profile`
4. `cross-channel durable entity memory` only if the planner explicitly chooses it
5. External knowledge only when internal sources are insufficient for the resolved task

Do **not** jump from an implicit mention to unrestricted global raw-history search.

---

## Retrieval Execution Contract

**Today:** retrieval context is forcibly rewritten to current-user scope after the fact.

**Desired:**

- planners decide *what kind of source* is needed
- executors decide *how to query that source efficiently*
- evaluators decide *whether coverage is sufficient*
- no layer silently converts an event-recall need into a profile-recall need after the fact

Deterministic code may only inject **structural bounds**: platform, safe time ranges, max result counts. Semantic reinterpretation is not allowed.

---

## Open Design Decisions

### knowledge_base vs entity_memory

**Option A — Keep both concepts separate**

- `knowledge_base` = generic durable internal topic knowledge
- `entity_memory` = subject-aware third-party person/entity memory
- Pros: conceptually clean, easier policy separation
- Cons: more write/update logic, more retrieval arbitration

**Option B — Merge into one durable internal memory substrate**

- Unified schema with `subject_kind`, `subject_key`, `memory_scope`, `recent_window`, `historical_summary`, embeddings
- Pros: one storage/update framework, shared compaction logic, easier unified retrieval contracts
- Cons: policy mistakes can blur profile memory vs event memory if schema discipline is weak

**Recommendation:** merge at the storage-framework level; keep them distinct at the planner / research-facts / prompt level. One storage substrate, separate logical source types at retrieval time.
