# self cognition tracking ICD

## Document Control

- ICD id: SC-TRACKING-ICD-001
- Status: reference draft
- Related plan:
  `development_plans/active/short_term/self_cognition_agency_loop_plan.md`
- Scope: boundary between the self-cognition module's tracking/dry-run system
  and existing Kazusa production owners.
- Runtime behavior change: none. This document defines the contract that a
  future approved implementation must follow.

## Controlling Boundary

Self-cognition owns its own tracking system.

Production code is not the tracking system. Production chat, progress,
consolidation, memory, scheduler, dispatcher, adapter, and reflection workflows
must not be modified or used as hidden persistence for idle cognition in this
draft.

Allowed production interaction:

- read evidence through agreed read interfaces;
- call shared RAG and cognition engines for reasoning;
- request outbound execution only through an explicit future handoff interface.

Forbidden production interaction:

- direct adapter send;
- direct scheduler insert;
- direct production conversation-progress write;
- direct production consolidation persistence;
- direct memory, relationship, affinity, image, or character-state mutation;
- normal `/chat` `final_dialog` output for idle cognition;
- semantic proactive permission policy outside cognition.

## Canonical Location Rule

This reference document is not allowed to remain the canonical runtime contract
after the production self-cognition module is implemented.

The implementation plan must move the canonical ICD content into the module
documentation, following the existing module pattern:

```text
src/kazusa_ai_chatbot/<module_name>/README.md
```

Canonical module location:

```text
src/kazusa_ai_chatbot/self_cognition/README.md
```

After that move, this reference document must remain only as a pointer to the
module documentation or be superseded according to the development-plans
registry.

## Logical Data Ownership

The self-cognition tracking system owns these logical records:

| Record | Purpose |
|---|---|
| `self_cognition_run` | One idle cognition execution and its route decision. |
| `self_cognition_trigger` | Why the run exists, including source refs and due state. |
| `self_cognition_evidence_ref` | Digest/reference to production evidence read by the run. |
| `self_cognition_route_effect` | The selected consumer route and allowed side effect. |
| `self_cognition_action_attempt` | Lifecycle and idempotency for an outward action. |
| `self_cognition_outbox_candidate` | Public message/tool candidate before handoff. |
| `self_cognition_growth_candidate` | Compact projected evidence for slower growth. |

These records may live in local dry-run files for this slice. A future
implementation may move them to durable storage, but they remain owned by
self-cognition.

## Read Interfaces

Self-cognition may read production state only through the following interface
classes. If an implementation needs a new read surface, this ICD must be
updated before code changes.

| Interface | Current source | Allowed use | Forbidden use |
|---|---|---|---|
| Conversation history read | existing conversation DB/query facade or diagnostic export | bounded recent visible dialog by platform/channel/user | treating internal thought as dialog |
| Conversation progress read | `conversation_progress.load_progress_context(...)` | current short-term thread state for L3 context | writing or incrementing visible turn state |
| Active commitment read | `query_user_memory_units(...)` or RAG recall projection with `active_commitment`/`active` filters | trigger selection and due-state framing | marking commitments complete directly |
| Pending scheduled event read | `query_pending_scheduled_events(...)` or RAG recall projection | duplicate and future-action awareness | inserting or updating scheduler rows |
| RAG2 read/retrieval | `call_rag_supervisor(...)` | memory, conversation, recall, live context, continuation, web-search evidence | durable memory writes |
| Cognition execution | `call_cognition_subgraph(...)` | L1/L2/L3 self-cognition reasoning | standalone self-cognition LLM prompt outside shared cognition |
| Promoted reflection read | existing promoted-context read path only | background mood/global context modifier | trigger source or direct growth write |
| Runtime state read | existing character/runtime state projection | current mood, time, adapter availability hints | adapter delivery or state mutation |

Read payloads must be bounded and source-labeled. The tracking record should
store stable references and digests first; full payload snapshots are
diagnostic artifacts, not the default persistence model.

## Route Contract

Each run selects exactly one primary route and may attach secondary candidate
effects.

| Route | Tracking write | Production write | Notes |
|---|---|---|---|
| `silent_no_write` | optional run trace | none | No later reflection impact. |
| `audit_only` | run and route trace | none | Evaluation only. |
| `progress_maintenance` | route effect candidate | none in this ICD | May later project into progress through a separate handoff. |
| `action_candidate` | action attempt and outbox candidate | none | Inspectable candidate, no delivery yet. |
| `scheduled_action` | action attempt state | future ICD handoff only | Scheduler execution is outside self-cognition. |
| `delivered_message` | delivery result ref | future ICD storage rule only | Transcript write requires a separate decision. |
| `growth_evidence` | compact growth candidate | none in this ICD | Raw internal monologue is forbidden. |

The route contract replaces the normal `/chat` side-effect switch. A non-empty
candidate message must not be treated as normal `final_dialog`.

## Action Attempt Lifecycle

Status values:

- `candidate`
- `held`
- `pending_handoff`
- `handoff_accepted`
- `scheduled`
- `sent`
- `failed`
- `cancelled`
- `duplicate_suppressed`
- `closed_no_action`

Required fields:

- `attempt_id`
- `run_id`
- `trigger_id`
- `source_kind`
- `source_id`
- `target_scope`
- `action_kind`
- `due_at`
- `idempotency_key`
- `status`
- `candidate_text_ref`
- `created_at`
- `updated_at`
- `retry_after`
- `handoff_ref`
- `delivery_ref`

Generated wording is not part of the idempotency identity.

## Repeat Suppression

A past-due, unfulfilled promise can qualify as a trigger on every idle tick.
That must not create a message every idle tick.

Idempotency key:

```text
source_kind + source_id + due_at + target_scope + action_kind
```

Required behavior:

- `future_due`: do not create an outbound send; record progress/scheduler
  maintenance only if useful.
- `due_now` or `past_due` with no matching action attempt: cognition may create
  one outbound candidate.
- matching `candidate`, `held`, `pending_handoff`, `handoff_accepted`, or
  `scheduled`: suppress duplicate send creation.
- matching `sent`: suppress duplicate send creation for the same due
  occurrence.
- matching `failed`: retry only through explicit retry state and backoff.
- matching `cancelled` or fulfilled source evidence: close the trigger and do
  not contact.
- new visible user evidence may create a new trigger occurrence with a new
  source identity or due time.

This is execution mechanics, not proactive permission. Cognition decides
whether contact is socially appropriate; tracking prevents repeated execution
of the same obligation.

## Handoff Interface

This ICD does not authorize live outbound execution. It defines the shape a
future handoff must use.

Ready outbox candidate fields:

- `outbox_id`
- `attempt_id`
- `target_platform`
- `target_channel`
- `target_channel_type`
- `text`
- `execute_at`
- `idempotency_key`
- `source_refs`
- `audit_reason`

Handoff rule:

```text
self-cognition tracking
-> ready outbox candidate
-> ICD handoff adapter
-> TaskDispatcher / scheduler
-> registered MessagingAdapter
```

The handoff adapter may reject an item for malformed target, missing adapter,
duplicate execution, runtime off-switch, blank text, or invalid execute time.
It must not reject or approve based on a semantic permission policy.

## Reflection And Growth Interface

Reflection may consume only compact tracking projections, not raw cognition
traces.

Allowed projection example:

```text
source: self_cognition_tracking
pattern: repeated held/contact decisions around soft commitments
scope: relationship-local or character-global, as explicitly selected
evidence_refs: run ids and source refs
suggested_learning: short reversible sentence
```

Forbidden projection content:

- raw internal monologue;
- private chain-like reasoning;
- unsourced user facts;
- generated promises not grounded in visible dialog or durable memory.

## Dry-Run Requirements

The module under `src/kazusa_ai_chatbot/self_cognition/` must model this ICD
even when it writes only local artifacts. The module and smoke commands must
not depend on `experiments/*`; that tree is removable.

Minimum artifacts:

- trigger record;
- run record;
- evidence refs;
- cognition input/output trace;
- route effect;
- action attempt when contact is considered;
- outbox candidate when public text is produced;
- loop trace explaining why the route was selected.

The dry-run path must not write production state or call production delivery
paths.

## Open Implementation Decisions

These items require later approved plans:

- durable storage collection names and indexes;
- exact read facade for bounded conversation history;
- exact dispatcher/scheduler handoff API;
- whether delivered proactive messages become normal conversation rows;
- whether progress-maintenance candidates are read directly by L3 or projected
  into conversation progress;
- growth-candidate promotion cadence and review rules.
