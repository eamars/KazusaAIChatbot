# Architectural Roadmap

Kazusa is now mostly past the original "extract the bot brain" milestone. The
current direction is a bounded, inspectable digital-character runtime: adapters
stay thin, live chat stays latency-bounded, LLM stages own semantic judgment,
and deterministic code owns validation, persistence, cache invalidation,
scheduling, and adapter delivery.

This file is a roadmap, not an execution-ready development plan. New multi-file
work should still get its own lowercase development plan before implementation.

## Status Legend

- [x] Roughly implemented: present in the current architecture and usable, even
  if it still needs hardening or narrower follow-up plans.
- [ ] Not ready: keep as future work unless a separate plan defines scope,
  contracts, cutover policy, and verification.

## Phase 1: Foundation And Architectural Decoupling

Goal: keep the brain independent from adapters and make platform, identity,
memory, retrieval, scheduling, and reflection explicit service boundaries.

- [x] **Standalone Brain Service:** `kazusa_ai_chatbot.service` is the FastAPI
  brain. Lifespan work covers DB bootstrap, character profile loading, graph
  compilation, MCP startup, scheduler setup, Cache2 hydration, pending scheduled
  event loading, and reflection worker startup.
- [x] **Typed Message Envelope:** Adapters normalize Discord/QQ/debug events
  into platform-neutral envelopes with body text, mentions, replies,
  addressees, and attachments. Brain code should reason over typed fields, not
  raw platform syntax.
- [x] **Global Identity Mapping:** `(platform, platform_user_id)` resolves to
  stable `global_user_id` rows in `user_profiles`, with profile and alias
  support for cross-platform continuity.
- [x] **Platform-Scoped RAG:** RAG 2 receives platform, channel, user, timestamp,
  active turn, and recent chat context. Conversation, user, relationship, recall,
  and Cache2 policies are scoped by platform/channel/user where relevant while
  still allowing global durable memory when the slot asks for it.
- [x] **Adapter Pattern:** Debug, Discord, and NapCat QQ adapters communicate
  with the brain through HTTP and register runtime delivery callbacks for
  scheduled messages.
- [x] **Scheduled Events:** `future_promises` are harvested after dialog,
  normalized, passed through dispatcher validation, persisted in
  `scheduled_events`, rehydrated on service startup, and delivered through
  registered adapters.
- [x] **DB Bootstrap:** Startup creates current collections and indexes, drops
  legacy RAG1 cache collections, and prepares scheduler/reflection storage.
- [x] **Structured Shared Memory:** The evolving `memory` path has active,
  superseded, expired, rejected, lineage, merge, source, authority, and evidence
  fields, with public APIs for insert, supersede, merge, and active lookup.
- [x] **Scoped User Memory Units:** User continuity now lives in
  `user_memory_units` as fact/appraisal/relationship-signal triples with
  merge/evolve/create handling and query-time candidate retrieval.
- [x] **Cognition Layer Modularization:** Cognition is split into L1
  subconscious, L2 consciousness/boundary/judgment, L3 contextual/style/content
  anchors/preference/visual, and L4 collection.
- [x] **Reflection Cycle:** Background hourly, daily, and global-promotion
  reflection exists outside the live response path. It stores inspectable run
  documents and can promote bounded lore or self-guidance through
  `memory_evolution` gates.
- [ ] **Memory Maintenance Sweep:** Add only if operational evidence shows stale
  active rows accumulating. The sweep must use existing memory-evolution APIs,
  avoid ad hoc Mongo writes, and treat query-time expiry filtering as the first
  line of defense.
- [ ] **RAG Scope Regression Suite:** Add deterministic and live-LLM cases that
  prove local channel evidence is preferred when appropriate, global durable
  memory remains available when explicitly needed, and private user continuity
  does not leak across users or channels.

## Phase 2: Psychological Modeling And Empathic Accuracy

Goal: improve social judgment without turning the live response path into an
unbounded simulation loop.

- [x] **Relationship Insight Baseline:** Affinity, relationship insight,
  relationship ranking evidence, and scoped user-memory triples provide the
  current relationship model.
- [x] **Dynamic Boundary Baseline:** Boundary profile, boundary core, judgment
  core, logical stance, and dialog evaluator already prevent simple "yes-man"
  behavior and preserve character boundaries.
- [ ] **Long-Horizon Relationship Model:** Future work should summarize stable
  relationship dynamics from user-memory units and reflection outputs, not add
  another live-response prompt that competes with cognition.
- [ ] **Response-Impact Prediction:** Keep this out of the normal chat path for
  now. If implemented, make it an opt-in evaluation or background diagnostic
  that compares predicted user reaction with later evidence and writes only
  through approved memory/reflection boundaries.
- [ ] **Empathic Accuracy Evaluation:** Build as an offline or post-turn quality
  signal with durable traces and human-inspectable criteria. Do not use it as a
  direct reward loop that rewrites personality or memory without gates.
- [ ] **Boundary Calibration Tests:** Add targeted real-LLM cases for excessive
  compliance, excessive defensiveness, unresolved ambiguity, intimacy requests,
  and authority/identity pressure.

## Phase 3: Autonomous Agency And Personality Evolution

Goal: support lifelike continuity and limited initiative without hidden,
random, or permissionless behavior.

- [x] **Delayed Reflection Loop:** The reflection worker provides the slow-wave
  background loop. It defers while primary interaction is busy and does not send
  messages on its own.
- [ ] **Permissioned Proactive Contact:** Do not add a free-running heartbeat
  that initiates chat from mood alone. Any proactive contact must go through
  explicit user acceptance, scheduler/dispatcher validation, quiet-hour policy,
  adapter availability checks, and audit logs.
- [ ] **World/Event Inputs:** Environmental events should be explicit curated or
  observed inputs with provenance. Avoid random shocks that silently rewrite
  mood, vibe, or lore.
- [ ] **Personality Drift Governance:** Do not inject random traits to fight
  predictability. Any personality shift should be slow, bounded, evidence-based,
  reversible, and represented through approved profile, reflection, or
  memory-evolution contracts.
- [ ] **Autonomy Safety Review:** Before adding self-initiated behavior, define
  permissions, rate limits, off-switches, adapter visibility, user-facing audit
  semantics, and rollback behavior in a separate plan.

## Technical Direction

| Area | Current Position | Direction |
| :--- | :--- | :--- |
| Message queue | Process-local queue is implemented and fits the single-brain deployment. | Consider Redis/RabbitMQ only for multi-process scale, durable queue replay, or cross-host adapters. Do not add it just for architecture symmetry. |
| User modeling | Relationship state, user-memory units, RAG evidence, cognition, and reflection already divide responsibilities. | Improve the existing boundaries before adding a separate user-modeling service. |
| Reflection | Background worker and promotion gates exist. | Keep reflection outside live chat; expose only promoted, compact context when enabled. |
| Prediction | Not in the response path. | Treat prediction as evaluation or background evidence until latency, privacy, and write-boundary rules are proven. |
| Personality shift | Character profile remains the stable identity anchor. | Allow only bounded, evidence-backed changes through explicit contracts; avoid random drift. |
| Reward signal | No direct reward loop. | Prefer inspectable metrics: continuity, boundary correctness, factual grounding, privacy safety, and accepted-commitment follow-through. |

## Near-Term Follow-Up Plans

- Write a RAG scope regression plan for platform/channel/user scoping and
  private-memory leakage tests.
- Write a memory maintenance sweep plan only if active stale rows become an
  operational problem.
- Write a proactive-contact safety plan before any autonomous message initiation
  beyond already accepted scheduled promises.
- Write a relationship-model hardening plan if `last_relationship_insight`
  remains too snapshot-like in live traces.
