# Continuation Resolution Before External RAG

## Goal

Resolve continuation-like messages before external retrieval, while deciding whether Stage 1 should remain a split dispatcher design or move to a centralized `rag_supervisor` model.

This update also resolves a missing policy question:

- whether “another user” should trigger third-party profile retrieval,
- where that retrieval belongs,
- and how to prevent duplicated `user_image` fetches.

---

## Hard Constraints

- **Do not feed raw conversation history into external RAG**
- **Do not introduce unbounded nested retrieval loops**
- **Optimize for local LLM latency and database pressure**

External RAG must remain history-blind. Any continuation repair must happen before external dispatch.

---

## Current Architecture Assessment

## Actual Current Flow

```text
message
  -> decontextualizer
  -> Stage 1 RAG subgraph
       - embed input
       - cache probe
       - depth classifier
       - input_context_rag dispatcher
       - external_rag dispatcher
       - cache writeback
  -> cognition
  -> dialog
  -> consolidation
```

## What Stage 1 Already Does Well

- `input_context_rag` and `external_rag` are already separated
- current-speaker `objective_facts`, `user_image`, and `character_image` are injected directly from state/profile instead of being re-fetched through the dispatcher
- cache and depth classification already suppress some unnecessary work
- `memory_retriever_agent` already has its own generator -> executor -> evaluator -> finalizer loop

## Current Weaknesses

- there is **no top-level arbiter** that reasons jointly across internal retrieval and external retrieval
- both dispatchers still reason primarily from the same `decontexualized_input`, so continuation fragments can still be under-resolved before dispatch
- the current design has **mixed evaluation layers**:
  - Stage 1 has top-level branching and thresholds
  - `memory_retriever_agent` has its own evaluator loop
  - external retrieval is comparatively shallow
- there is **no explicit third-party person-profile retrieval policy**
- there is **no retrieval ledger** preventing the system from asking for the same subject profile twice under different names

## Architectural Conclusion

The current design is **not wrong**. It is already partially decomposed and fairly efficient for local use.

The main architectural gap is **not** “lack of a supervisor” in the abstract.
The real gap is:

- continuation resolution is not explicit enough before dispatch,
- cross-source arbitration is weak,
- and identity-aware dedup is underspecified.

---

## The Core Problem to Fix First

Messages such as:

- `一定要算哦`
- `继续`
- `按刚才那个`
- reply-only follow-ups

are often linguistically complete but **task incomplete**.

If dispatch reasons from the fragment directly, external retrieval can be launched with a generic or wrong task.

Therefore the first required separation is still:

1. **internal continuation resolution**
2. **retrieval planning**
3. **external factual retrieval**

That principle remains valid even if a `rag_supervisor` is introduced.

---

## Architecture Options

## Option A — Keep Split Dispatchers + Add Continuation Resolver

```text
message -> decontextualizer -> continuation_resolver -> input_context_rag dispatcher + external_rag dispatcher
```

- **Pros**
  - lowest latency on local LLM
  - smallest refactor
  - preserves cache, depth classifier, and direct current-user profile preload
  - avoids adding another loop above `memory_retriever_agent`
- **Cons**
  - weak top-level arbitration across sources
  - evaluation remains fragmented
- **Feasibility**
  - **very high** and safest short-term

## Option B — Full Iterative `rag_supervisor`

```text
message -> decontextualizer -> continuation_resolver -> rag_supervisor loop -> retrievers -> rag_supervisor loop
```

- **Pros**
  - strongest global reasoning
  - one place for stop/continue policy and slot-gap analysis
- **Cons**
  - highest latency and token cost
  - most likely to recreate nested-loop degradation on local LLM
  - DB pressure rises sharply if retries are not tightly bounded
  - if sub-agents keep their own evaluators, this becomes loop-over-loops
- **Feasibility**
  - **technically feasible but operationally risky**

## Option C — Recommended Hybrid: Bounded `rag_supervisor`

```text
message -> decontextualizer -> continuation_resolver -> rag_supervisor_planner -> parallel retrievers -> rag_supervisor_evaluator -> optional one repair pass -> rag_finalizer
```

- **Pros**
  - explicit and smart top-level control
  - continuation repair happens before external search
  - much safer than a free-form iterative controller
- **Cons**
  - still more complex than Option A
  - requires stronger contracts from retrievers
- **Feasibility**
  - **good**, if hard budgets are enforced:
    - max planning pass: `1`
    - max repair pass: `1`
    - max total Stage 1 LLM rounds after decontextualizer: `2`
    - no repeated dedupe key in one request

## Evaluator Centralization Feasibility

Internal and external RAG **can** be decomposed if they become thin execution-oriented sources:

- internal RAG -> memory / conversation / persistent-memory executor
- external RAG -> web-search executor
- top-level evaluator -> decides sufficiency, missing slots, and whether a repair pass is allowed

This is feasible only if retrievers return compact structured results such as:

```python
{
    "source": "input_context_rag",
    "subject": "current_user | third_party_user | topic",
    "status": "complete | partial | empty | blocked",
    "coverage": ["answered_slot_a", "answered_slot_b"],
    "missing": ["slot_c"],
    "confidence": 0.0,
    "dedupe_key": "input_context:current_user:task_hash"
}
```

## Centralization Rule

- **one evaluator at the top level**
- **retrievers below it should be mostly execution-oriented**
- if a retriever keeps an internal loop, treat it as a bounded black box and do not let the supervisor loop aggressively on top of it

---

## Policy for User RAG vs Third-Party User Discussion

## Current User RAG

The current speaker's profile should remain **independent and preloaded**:

- `objective_facts`
- current speaker `user_image`
- `character_image`

This is cheap, stable, and already present in the current design.

## If the Conversation Is About Another User

If the conversation is about another user, then **sometimes yes**, the character should be able to fetch that other person's remembered profile or image.

But it should **not** be handled as the same thing as the current speaker preload.

## Recommended Rule

Treat this as a separate internal source:

- `third_party_user_image_rag`

This belongs to **internal RAG**, not external RAG, because the source of truth is still internal memory / profile data.

## When to Trigger Third-Party User Image Retrieval

Only trigger if all of the following hold:

- the message clearly refers to a specific other person or stable named entity
- that person matters to the response, not just a passing mention
- the answer depends on remembered traits, relationship history, or prior profile-level information
- the current speaker preload does not already satisfy the need

## When Not to Trigger It

Do not trigger it when:

- another user is only casually mentioned
- internal context retrieval already contains the needed facts
- the response only needs event history, not a profile
- the person cannot be resolved confidently to a known stored subject

---

## Dedup Policy for `user_image`

## Problem

Without explicit subject tracking, internal RAG may fetch a profile that is effectively already present, especially when:

- current speaker image is already injected,
- a dispatcher restates the same need in different wording,
- or a third-party entity aliases to the same person.

## Required Solution

Introduce a request-local retrieval ledger keyed by:

```python
(source_type, subject_id_or_name, retrieval_purpose)
```

## Minimum Rules

- current speaker preload registers a ledger entry immediately:
  - `("user_image", current_global_user_id, "profile_context")`
- internal RAG may not request the same key again in the same turn
- if a third-party entity resolves to the current speaker, do not fetch again
- if a third-party entity cannot be resolved to a stable user ID, downgrade to conversation/entity context search instead of profile fetch
- evaluator must see which subjects are already loaded before planning another retrieval

## Practical Interpretation

`user_image` should stop being treated as a generic blob and start being treated as:

- `current_user_image`
- `third_party_user_image[target_user_id]`

That identity separation is enough to prevent most duplicates.

---

## Proposed Bounded Stage 1 Design

## New Flow

```text
message
  -> decontextualizer
  -> continuation_detector
  -> conversation_context_resolver
  -> resolved_task_builder
  -> rag_supervisor_planner
  -> parallel source executors
       - current_user_profile preload (already loaded, no fetch)
       - input_context_rag
       - third_party_user_image_rag (optional)
       - external_rag (optional)
  -> rag_supervisor_evaluator
  -> optional one repair pass
  -> rag_finalizer
```

## Planning Priority

1. **resolve continuation first**
2. **use already-loaded current speaker facts first**
3. **try internal context retrieval before external when the task is under-specified**
4. **only call external when the resolved task explicitly needs outside knowledge**
5. **only fetch third-party profile when the response depends on that person's remembered identity or traits**

## Evaluator Responsibility

The central evaluator should answer only these questions:

- is the task now explicit enough?
- are the key slots filled?
- is external knowledge still necessary?
- is there unresolved identity ambiguity?
- should we stop, repair once, or clarify?

It should not rewrite retrieval strategy indefinitely.

---

## Continuation Resolver Requirements

## Trigger Signals

Mark `needs_context_resolution = True` if one or more apply:

- explicit reply target exists
- input is very short and imperative
- continuation markers appear:
  - `继续`
  - `就这个`
  - `按刚才那个`
  - `一定要算`
  - `查一下那个`
- core task slots are missing
- omitted referents exist:
  - `这个`
  - `那个`
  - pronouns
  - implicit objects

## Resolver Search Priority

1. explicit reply target
2. recent local turns
3. conversation search fallback
4. clarification if still ambiguous

## Output Contract

```python
{
    "needs_context_resolution": True,
    "resolved_user_intent": "currency_conversion",
    "task_summary": "继续完成上一条汇率换算任务",
    "known_slots": {"source_currency": "CNY", "target_currency": "JPY", "amount": 100},
    "missing_slots": [],
    "confidence": 0.92,
    "evidence": ["reply_target", "recent_turn"],
}
```

External dispatch must consume this packet, not the raw fragment.

---

## Performance Guardrails for Local LLM

- **no nested open-ended loops**
- **one top-level evaluator only**
- **maximum one repair pass**
- **parallelize retrieval executors, not planners**
- **short JSON-only contracts between stages**
- **do not pass large growing call histories unless absolutely necessary**
- **ledger-based dedupe for source/subject reuse**
- **cache remains mandatory**

If these guardrails cannot be enforced, do not build the supervisor.

---

## Implementation Phases

## Phase 1 — Continuation Resolution

- add `continuation_detector`
- add `conversation_context_resolver`
- add `resolved_task_builder`
- make external dispatch depend on resolved task packets

## Phase 2 — Retrieval Identity Model

- distinguish `current_user_image` from `third_party_user_image`
- add request-local retrieval ledger
- define subject resolution rules

## Phase 3 — Bounded Supervisor

- add `rag_supervisor_planner`
- simplify retrievers into execution-oriented source nodes
- add one central evaluator and one finalizer
- enforce one repair-pass maximum

## Phase 4 — Regression and Performance Tests

- continuation-task recovery
- ambiguity-to-clarification routing
- third-party profile retrieval policy
- duplicate `user_image` suppression
- local-model latency and DB load snapshots

---

## Acceptance Criteria

- continuation fragments no longer trigger generic external queries
- external RAG receives only explicit resolved tasks
- third-party profile retrieval is possible but never automatic on casual mention
- current speaker `user_image` is never redundantly re-fetched within the same turn
- central evaluation does not create nested uncontrolled loops
- Stage 1 remains practical for local LLM deployment

---

## Final Recommendation

Do **not** jump directly to a fully iterative multi-loop `rag_supervisor`.

The recommended architecture is:

- **keep current speaker profile preload independent**
- **add explicit continuation resolution before retrieval dispatch**
- **treat third-party user-image retrieval as a separate internal source**
- **centralize evaluation only if retrievers are simplified into bounded executors**
- **use a bounded hybrid supervisor, not an open-ended recursive one**

In short:

- **yes** to a `rag_supervisor` if it is explicit, bounded, and identity-aware
- **no** to nested free-form loops between internal RAG, external RAG, and user-image retrieval

That gives you the architectural clarity you want without reintroducing the performance collapse you already experienced.
