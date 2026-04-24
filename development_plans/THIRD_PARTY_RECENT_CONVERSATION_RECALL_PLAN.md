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
| 1     | Planner refactor — continuation resolver + enriched retrieval plan fields       | ✅ DONE        |
| 2     | Retriever refactor — plan-driven scopes, third-party backends, retrieval ledger | ✅ DONE        |
| 3     | Entity memory layer — durable entity/topic memory substrate                     | Undone - Need rethink        |
| 4     | Dialog surface upgrade — research_facts extension for third-party results       | ✅ DONE        |
| 5     | Bounded supervisor centralization — rag_supervisor_planner / evaluator          | ✅ DONE        |
| 6     | RAG module decomposition — split into focused submodules                        | ✅ DONE        |
| 7     | Evaluation — test coverage for third-party recall scenarios                     | ✅ DONE        |
| 8     | Transparent cache layer — boundary cache between resolution and retrieval phases | ✅ DONE        |

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

  [Pre-Tier-1 — sequential]
       entity_grounder                        ← Phase 1 (resolves entities; output gates Tier 2)

  [Tier 1 — parallel, no internal dependencies]
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

- `entity_grounder` (pre-Tier-1) always completes before Tier 2 fires — because Tier 2 needs resolved entity IDs that `entity_grounder` produces.
- Modes in Tier 3 always run after all internal tiers — so external search is informed by what internal sources already found.

---

## Execution Order

Execution order is **not decided by the planner**. It is a fixed structural property of source type. This keeps the planner simple and the execution model testable without LLM involvement.

### Tier membership

| Tier  | Sources                                                | Runs when                              | Can run in parallel with               |
| ----- | ------------------------------------------------------ | -------------------------------------- | -------------------------------------- |
| 0     | `current_user_image`, `character_image`                | Always, zero cost                      | Everything                             |
| pre-1 | `entity_grounder`                                      | Sequential, after planner              | Nothing (gates Tier 2)                 |
| 1     | `input_context_rag`, `knowledge_base_rag`              | Parallel with entity_grounder          | Each other                             |
| 2     | `third_party_profile_rag`, `channel_recent_entity_rag` | After Tier 1 AND entity_grounder       | Each other; repeat per resolved entity |
| 3     | `external_rag`                                         | After all internal tiers complete      | Nothing                                |

### Why this ordering

**`entity_grounder` gates Tier 2.** `THIRD_PARTY_PROFILE` and `CHANNEL_RECENT_ENTITY` both need a resolved entity ID to be precise. Without it, profile retrieval has no target and channel search falls back to surface-form keyword matching. `entity_grounder` runs sequentially before Tier 1 so its output is already in state when Tier 1 starts. Tier 2 waits for both `entity_grounder` AND Tier 1 to complete before firing — it needs the resolved ID from the former and the internal context from the latter.

**Tier 3 is always last.** External search is most useful when it knows what internal sources already found — it searches for what the character doesn't already remember. Running external in parallel with internal means the external dispatcher builds its query blind.

**Same source can activate multiple times in Tier 2.** In group-chat scenarios with multiple resolved entities (e.g., "what happened between them?"), `channel_recent_entity_rag` fires once per resolved subject, all in parallel within Tier 2.

### Dependency types

| Type                  | Description                                                                | Handled by                                       |
| --------------------- | -------------------------------------------------------------------------- | ------------------------------------------------ |
| Resolution dependency | Profile/channel needs entity ID before fetching                            | pre-Tier-1 `entity_grounder` → Tier 2 gate      |
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

### Implementation Specifics

**Node identity and file location**

Three new nodes, all initially in `persona_supervisor2_rag.py` (move to `persona_supervisor2_rag_resolution.py` at Phase 6):

| Node | Type | Purpose |
|------|------|---------|
| `continuation_resolver` | LLM, single structured-output call | Decides whether input is retrieval-ready; produces resolved task |
| `rag_planner` | LLM, single structured-output call (supervisor) | Decides which sources activate and identifies surface-form entities |
| `entity_grounder` | Deterministic-first, LLM fallback only | Resolves surface forms to known identifiers |

**Why `rag_planner` is supervisor-type but not a full agentic loop**

`rag_planner` uses LLM intelligence to decide downstream routing — it is a supervisor in the LangGraph sense. However, it makes that decision in a **single structured-output call**, not an iterative tool-use loop. The iterative repair supervisor is Phase 5.

Keeping Phase 1 to two sequential LLM calls (`continuation_resolver` → `rag_planner`) is load-bearing for chat latency: every extra LLM round before retrieval begins costs 500ms–2s. `with_structured_output(RetrievalPlanSchema)` is the right implementation pattern — not a tool-calling agent.

**Entity grounder implementation order**

1. Deterministic pass: match surface forms against `chat_history_recent` participant display names
2. DB lookup: check platform-linked identities and alias tables
3. LLM pass (only if genuinely ambiguous after steps 1–2): small structured-output call to disambiguate

Steps 1–2 cover the common case (user names someone who appeared in recent history) with zero LLM cost.

**RAGState additions**

```python
"continuation_context": dict     # continuation_resolver output
"retrieval_plan": dict           # rag_planner structured output
"resolved_entities": list[dict]  # entity_grounder output: surface_form → resolved_global_user_id
"retrieval_ledger": dict         # key (source_type, subject_key, purpose) → cached result summary
```

**LangGraph topology change**

Current (inferred):
```
START → [parallel dispatchers] → finalizer → END
```

Target:
```
START → continuation_resolver → rag_planner → entity_grounder   ← sequential pre-Tier-1
  → [Tier 1: input_context_rag || knowledge_base_rag]           ← parallel (entity_grounder runs concurrently)
  → tier_2_gate  ← waits for entity_grounder AND Tier 1; conditional edge (deterministic, no LLM)
      "skip" → tier_3_gate
      "run"  → [Tier 2: third_party_profile_rag || channel_recent_entity_rag]  ← parallel, per entity via Send
  → tier_3_gate  ← conditional edge
      "skip" → rag_finalizer
      "run"  → external_rag
  → rag_finalizer → END
```

Gate functions check `state["retrieval_plan"]["active_sources"]` and `state["resolved_entities"]` — deterministic, no LLM.
Tier 2 dynamic fan-out (one invocation per resolved entity) uses LangGraph's `Send` API.

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

**DB query spec**

Primary query against `conversation_history` collection:

```python
{
    "platform": state["platform"],
    "platform_channel_id": state["platform_channel_id"],
    "timestamp": {"$gte": utcnow() - timedelta(hours=lookback_hours)},
    "$or": [
        {"global_user_id": entity["resolved_global_user_id"]},       # precise match when resolved
        {"content": {"$regex": entity["surface_form"], "$options": "i"}},  # surface-form fallback
    ]
}
# Sort: {timestamp: -1}   Limit: 20
```

Semantic fallback: if result count < 3, run embedding similarity search against `entity["surface_form"]` expanded with entity type context.

Result contract: returns `list[dict]` with `{timestamp, speaker, content}`; the executor formats to prose summary before inserting into `research_facts["channel_recent_entity_results"]`. Multiple entities each produce their own keyed entry.

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

Downstream prompt updates required in:

| File | Location | Change needed |
|------|----------|---------------|
| `persona_supervisor2_cognition_l3.py` | Lines 382–388, 526–529 | Add new keys to the `research_facts` dict passed to prompt: `third_party_profile_results`, `channel_recent_entity_results`, `entity_resolution_notes`. Update the prompt schema at lines 333 and 449 to explain source priority and weight stable anchors vs recent entity context. |
| `persona_supervisor2_cognition_l2.py` | Line 201 | Update prompt schema for `research_facts` to reference new keys. |

Prompt guidance additions:

- explain source priority (preloaded anchors > durable entity memory > recent channel context > external)
- distinguish "what I remember durably about X" (entity_memory / profile) from "what was said about X recently" (channel_recent_entity)
- prevent the character from treating a third-party recall result as if it were the current user's profile

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

Add tests incrementally as each phase ships. The integration scenarios below are LLM-in-the-loop tests — they exercise the real planner and retriever chain against a seeded conversation history fixture.

### Unit test checklist (per phase)

**Phase 1:** continuation-like messages repaired before retrieval; planner emits `CHANNEL_RECENT_ENTITY` for implicit third-party references; entity grounding resolves known display names from recent history.

**Phase 2:** implicit third-party recall via `CHANNEL_RECENT_ENTITY`; explicit name recall resolving to `THIRD_PARTY_PROFILE`; mixed self + third-party; unresolved entity surface-form fallback; no cross-user leakage; duplicate-profile suppression via retrieval ledger; external search receives only resolved task, not raw continuation fragment.

**Phase 3 (if implemented):** entity memory populated correctly on consolidator pass; retrieved correctly for recurring people/entities.

**Phase 5 (if implemented):** bounded supervisor stays within max LLM round limits; evaluator triggers at most one repair pass.

**Phase 6:** all nodes importable from their respective submodules without circular dependencies; `call_rag_subgraph` behaviour identical before and after the split.

---

### RAG Integration Test Scenarios

Each scenario: input → expected planner output → tier execution sequence → `research_facts` assertions.

---

**S1 — Explicit named third-party, entity resolves** *(CASCADED)*

Input: `"啾啾之前跟你说过什么有趣的？"`

```
Planner: CASCADED | active_sources=[CHANNEL_RECENT_ENTITY, THIRD_PARTY_PROFILE] | entities=[啾啾]

Tier 0: current_user_image, character_image
pre-1:  entity_grounder("啾啾") → resolved_id
Tier 1: input_context_rag  ||  knowledge_base_rag
Tier 2: channel_recent_entity_rag(subject_id)  ||  third_party_profile_rag(subject_id)
Tier 3: SKIP
```

Assert: `channel_recent_entity_results` non-empty; `entity_resolution_notes` contains resolved ID.

---

**S2 — Implicit pronoun, entity ambiguous** *(CASCADED)*

Input: `"她之前说的那个你还记得吗？"`

```
Planner: CASCADED | active_sources=[CHANNEL_RECENT_ENTITY] | entities=[她, resolution_confidence=0]

Tier 0: preloaded
pre-1:  entity_grounder("她") → UNRESOLVED
Tier 1: input_context_rag  ||  knowledge_base_rag
Tier 2: channel_recent_entity_rag(surface_form="她", semantic_fallback=true)
        third_party_profile_rag → SKIP (no resolved ID)
Tier 3: SKIP
```

Assert: Tier 2 fires with surface-form fallback; no profile retrieval attempted; `entity_resolution_notes` records unresolved.

---

**S3 — Self + third-party mixed** *(CASCADED)*

Input: `"我之前跟你说过的那个项目，啾啾知道吗？"`

```
Planner: CASCADED | active_sources=[CHANNEL_RECENT_ENTITY, THIRD_PARTY_PROFILE, GLOBAL_ENTITY_KNOWLEDGE]
         subject.kind=mixed | entities=[啾啾]

Tier 0: preloaded
pre-1:  entity_grounder("啾啾") → resolved_id
Tier 1: input_context_rag(current_user scope)  ||  knowledge_base_rag
Tier 2: channel_recent_entity_rag(subject_id)  ||  third_party_profile_rag(subject_id)
Tier 3: SKIP
```

Assert: `input_context_results` carries project context from current-user scope; `channel_recent_entity_results` carries 啾啾's activity — two distinct keys, not merged.

---

**S4 — Multi-entity group chat, `Send` fan-out** *(CASCADED)*

Input: `"小明和啾啾最近有没有一起找你聊过什么？"`

```
Planner: CASCADED | active_sources=[CHANNEL_RECENT_ENTITY, THIRD_PARTY_PROFILE]
         entities=[小明, 啾啾]

Tier 0: preloaded
pre-1:  entity_grounder("小明")→id_A, entity_grounder("啾啾")→id_B
Tier 1: input_context_rag  ||  knowledge_base_rag
Tier 2: channel_recent_entity_rag(id_A)  ||  channel_recent_entity_rag(id_B)   ← Send, per entity
        third_party_profile_rag(id_A)    ||  third_party_profile_rag(id_B)     ← Send, per entity
Tier 3: SKIP
```

Assert: exactly 4 Tier 2 calls fire; results keyed by subject; retrieval ledger prevents any subject fetched twice.

---

**S5 — Cascading dependency, repair pass** *(CASCADED + Phase 5)*

Input: `"你之前说会帮她找的那个人，找到了吗？"`

```
Planner: CASCADED | active_sources=[CHANNEL_RECENT_ENTITY] | entities=[她]

Tier 0: preloaded
pre-1:  entity_grounder("她") → id_her
Tier 1: input_context_rag → finds commitment mentioning "小李"  ||  knowledge_base_rag
Tier 2: channel_recent_entity_rag(id_her) → result reveals "小李" as search target
                                             ↑ cascading gap detected by Phase 5 evaluator
[Repair pass — one pass only]
pre-1': entity_grounder("小李") → id_xiaoli
Tier 2': channel_recent_entity_rag(id_xiaoli)
Tier 3: SKIP
```

Assert: two Tier 2 rounds fire; evaluator does NOT trigger a third pass; ledger contains entries for both `id_her` and `id_xiaoli`.

---

**S6 — Internal primes external search** *(CASCADED, scope dependency)*

Input: `"啾啾之前提到的那个游戏，你觉得好玩吗？"`

```
Planner: CASCADED | active_sources=[CHANNEL_RECENT_ENTITY, EXTERNAL_KNOWLEDGE]
         entities=[啾啾] | external_task_hint="search for the game Jiujiu mentioned recently"

Tier 0: preloaded
pre-1:  entity_grounder("啾啾") → id
Tier 1: input_context_rag  ||  knowledge_base_rag
Tier 2: channel_recent_entity_rag(id) → reveals game title "星露谷物语"
Tier 3: external_rag(query="星露谷物语 gameplay review")
        ↑ uses actual Tier 2 result; external_task_hint used only if Tier 2 returned empty
```

Assert: Tier 3 query contains game title from Tier 2 result; if Tier 2 mocked empty, Tier 3 falls back to `external_task_hint`.

---

**S7 — Bare continuation fragment** *(continuation_resolver gates retrieval)*

Input: `"那后来呢？"`

```
continuation_resolver: needs_context_resolution=true | resolved_task="..." | confidence=0.72

Tier 0: preloaded
pre-1:  entity_grounder → conditional on entities found by resolver
Tier 1: input_context_rag(query=resolved_task)  ← resolved task replaces raw input  ||  knowledge_base_rag
Tier 2: conditional on entity_grounder output
Tier 3: SKIP
```

Assert: `input_context_rag` is queried with resolved task string, not `"那后来呢？"`; `continuation_context.confidence` ≥ clarification threshold (0.7).

---

**S8 — Pure current-user, no retrieval** *(CURRENT_USER_STABLE, negative control)*

Input: `"你最近心情怎么样？"`

```
Planner: CURRENT_USER_STABLE | active_sources=[] | entities=[]

Tier 0: preloaded (sufficient)
Tier 1–3: SKIP
```

Assert: no Tier 1-3 calls fire; `channel_recent_entity_results` absent from `research_facts`.

---

**S9 — Stable profile recall, no recent context needed** *(THIRD_PARTY_PROFILE standalone)*

Input: `"你对啾啾的整体印象是什么？"`

The question is about enduring character traits, not recent events. Planner does NOT activate `CHANNEL_RECENT_ENTITY`.

```
Planner: THIRD_PARTY_PROFILE | active_sources=[THIRD_PARTY_PROFILE] | entities=[啾啾]

Tier 0: preloaded
pre-1:  entity_grounder("啾啾") → resolved_id
Tier 1: SKIP (no active Tier 1 sources)
Tier 2: third_party_profile_rag(subject_id=resolved_id)
        channel_recent_entity_rag → SKIP
Tier 3: SKIP
```

Assert: only `entity_grounder` fires pre-Tier-1 (Tier 1 skipped); `third_party_profile_results` populated; `channel_recent_entity_results` absent; if resolved ID matches current user, ledger suppresses the fetch.

---

**S10 — Pure external knowledge, Tier 1-2 skipped** *(EXTERNAL_KNOWLEDGE standalone)*

Input: `"你知道量子纠缠是什么意思吗？"`

No entities, no internal context required. The decontextualized input is already the full task.

```
Planner: EXTERNAL_KNOWLEDGE | active_sources=[EXTERNAL_KNOWLEDGE] | entities=[]

Tier 0: preloaded
Tier 1: SKIP (no entities, no active Tier 1 sources)
Tier 2: SKIP
Tier 3: external_rag(query="量子纠缠 quantum entanglement")
```

Assert: no Tier 1 or Tier 2 calls fire; `external_rag_results` populated; `channel_recent_entity_results` and `third_party_profile_results` absent.

---

### Cross-scenario invariants

| Invariant | How to assert |
|-----------|--------------|
| Tier 2 never starts before Tier 1 completes | mock Tier 1 with 100ms delay; assert Tier 2 start ≥ Tier 1 end |
| Tier 3 never starts before Tier 2 completes | same pattern |
| Retrieval ledger prevents duplicate subject fetch | S4: assert exactly 4 Tier 2 calls, not 8 |
| Repair pass capped at 1 | S5: mock Tier 2 to always return a new entity name; assert evaluator stops after one repair |
| External uses hint only when internal empty | S6: mock Tier 2 empty; assert Tier 3 query equals `external_task_hint` |
| Continuation resolver gates on resolved task | S7: assert `input_context_rag` never receives the raw input string |
| Standalone mode does not activate sibling sources | S9: assert `channel_recent_entity_rag` not called; S10: assert no Tier 1-2 calls |

---

## Phase 8 — Transparent Cache Layer (Deferred)

**Defer until Phase 6 is complete.** The two-phase split (resolution vs retrieval) that Phase 6 formalizes is the same boundary the cache uses. Implement Phase 8 immediately after Phase 6.

**Goal:** no RAG node imports or calls any cache function. Cache is purely an orchestration concern at the `call_rag` entry point and at the `external_rag` executor wrapper.

### Design

The new architecture naturally splits into two phases with different cost profiles:

| Phase | Nodes | Cost |
|-------|-------|------|
| Resolution | `continuation_resolver`, `rag_planner`, `entity_grounder` | Low — 2 LLM calls + deterministic lookup |
| Retrieval | Tier 1 + Tier 2 + Tier 3 + finalizer | High — DB queries, LLM calls, optional web fetch |

The cache sits at the boundary between them. Resolution always runs (it's cheap and produces the cache key). Retrieval only runs on a cache miss.

```python
async def call_rag(global_state: GlobalPersonaState) -> dict:
    # Phase A: always runs — produces the cache key
    resolution = await call_resolution_subgraph(global_state)

    # Boundary cache — neither subgraph sees this
    key = _build_cache_key(resolution)
    cached = await rag_cache.probe(key)
    if cached:
        return cached

    # Phase B: only on cache miss
    result = await call_retrieval_subgraph(global_state, resolution)
    await rag_cache.write(key, result)
    return result
```

### Cache key construction

The key must include all dimensions that make two queries semantically distinct:

```python
def _build_cache_key(resolution: dict) -> str:
    return hash_stable({
        "resolved_task":   resolution["continuation_context"]["resolved_task"],
        "entity_ids":      sorted(e["resolved_global_user_id"] or e["surface_form"]
                                  for e in resolution["resolved_entities"]),
        "active_sources":  sorted(resolution["retrieval_plan"]["active_sources"]),
        "lookback_hours":  resolution["retrieval_plan"]["time_scope"]["lookback_hours"],
    })
```

Query text alone is insufficient: the same text with different resolved entity IDs, different source sets, or different time windows must produce different cache entries.

### `external_rag` independent cache

Web calls are expensive and independently invalidated. Wrap the executor with a decorator — the node logic stays clean:

```python
# in rag/cache.py — infrastructure, not a RAG node
def cached_node(key_fn):
    def decorator(fn):
        async def wrapper(state):
            key = key_fn(state)
            hit = await rag_cache.probe(key)
            if hit:
                return hit
            result = await fn(state)
            await rag_cache.write(key, result)
            return result
        return wrapper
    return decorator

# in persona_supervisor2_rag_executors.py
@cached_node(key_fn=lambda state: state["retrieval_plan"].get("external_query", ""))
async def external_rag(state: RAGState) -> dict:
    # purely retrieval logic — no cache awareness
    ...
```

### Scoped cache invalidation

**Current behaviour:** `db_writer` invalidates the whole RAG cache namespace after every write. This becomes a problem when multiple source types exist — writing a third-party profile for user B must not evict a cached `CHANNEL_RECENT_ENTITY` result for user A.

**Target behaviour:** `db_writer` calls source-scoped invalidation based on what it actually wrote:

| What was written | Cache scope to invalidate |
|-----------------|--------------------------|
| Current user facts / diary | Full-plan entries where `entity_ids` includes current user as primary subject |
| Third-party profile for user X | `THIRD_PARTY_PROFILE` entries keyed on X's `resolved_global_user_id` |
| New conversation history for channel C | `CHANNEL_RECENT_ENTITY` entries keyed on channel C |
| `knowledge_base` entry for subject K | `GLOBAL_ENTITY_KNOWLEDGE` entries keyed on K's `subject_key` |

Implement as a `CacheInvalidationScope` dataclass produced by `db_writer` and consumed by a `cache_invalidator` helper in `rag/cache.py`. The consolidator never imports the cache directly — it produces a scope descriptor and the cache module acts on it.

### Dependencies

- Requires **Phase 6** (module decomposition) — `call_resolution_subgraph` and `call_retrieval_subgraph` must exist as distinct callable boundaries before the cache can sit between them.
- `cached_node` decorator lives in `rag/cache.py` (already planned as part of the `rag/` subpackage in Phase 6).
- Phase 8 is **behavior-neutral**: cache hits must produce results identical to cache misses. Phase 7 integration tests (S1–S10) must pass unchanged whether the cache is warm or cold.

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

## Resolved Design Decisions

### knowledge_base vs entity_memory — Option B chosen

**Decision:** one storage substrate, separate logical source types at retrieval time.

**Rationale:** both options require the LLM to make the same discrimination ("is this about a person or a topic?") at write time. In Option A the LLM routes to the wrong collection; in Option B it sets the wrong `subject_kind` tag. The error is equivalent in both cases, so the "blurred boundary" risk is not a differentiator. Option B removes the duplicate write path, compaction logic, embedding index, and cache invalidation that Option A would require for each substrate.

Option B is also more recoverable: a mislabelled `subject_kind` is corrected with a single update query; a wrong-collection write in Option A requires document migration.

**Constraint on the write path:** `subject_kind` must be validated before any write reaches the substrate. The LLM output must include an explicit `subject_kind` value from a closed enum (`topic`, `person`, `group`, `event`). A missing or unrecognised value must fail loudly — not default silently — so type errors surface immediately rather than accumulating as retrieval noise.

**Schema:**
```json
{
  "subject_kind": "topic | person | group | event",
  "subject_key": "normalized identifier",
  "memory_scope": "global | platform | channel",
  "recent_window": [{"timestamp": "...", "summary": "..."}],
  "historical_summary": "...",
  "embedding": []
}
```

At retrieval time, planners filter by `subject_kind` to get the correct logical view. The planner and `research_facts` surface treat `topic` results and `person` results as distinct source types — the unified substrate is invisible above the executor layer.
