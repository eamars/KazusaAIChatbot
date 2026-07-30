# Cognition Core V2 Short-Horizon Global State Composition Big-Bang Plan

## Summary

- Goal: close the V2 functional gaps formerly covered by mood, global vibe, and last relationship insight through native emotion-specific character carry-over, ordinary elapsed fading between sleep cycles, causal per-user relationship projection, and one role-based user/group interaction-style composition path.
- Plan class: `high_risk_migration`.
- Status: `draft`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `no-prepost-user-input`, `debug-llm`, `character-test`, `control-console-web-development`, `py-style`, `cjk-safety`, `test-style-and-execution`, and `python-venv`.
- Overall cutover strategy: a forward-only V2 big-bang contract replacement after the character identity-growth plan completes, with no legacy prose-state restoration, compatibility mapper, dual read, dual write, channel-situation store, historical replay, or database backfill.
- Highest-risk areas: preserving the frozen twenty-one-emotion semantics while extending character-scope elapsed evolution; deriving a global consequence without copying user-scoped relationship state; preventing the new operational route from competing with the router's four durable-task slots; keeping global state fresh across normal and accepted-task paths without leaking scoped detail; bounding every reachable input/output/security check without an uncaught pipeline failure; proving behavioral rather than prompt-only impact; preserving current-scene authority; and avoiding latency regression.
- Acceptance criteria: a zero-gap capability matrix and twenty-one-emotion scope matrix are signed; a normal offence can produce separately derived offender-specific relationship injury and privacy-safe character-global affect; ordinary elapsed evolution changes the effective global emotion before sleep; the next eligible cross-channel turn consumes its committed state version without treating another user as the offender; current history and conversation progress remain scene authority; one interaction-style snapshot serves relevance, cognition, and surface; every direct V2-path limit resolves by bounded reduction, owner-local replacement, or a typed terminal disposition; controlled and natural real-LLM evidence demonstrates emotion-specific causal behavior and fading; and authenticated browser sign-off shows full persisted/effective affect, scoped relationship state, exact consumed context, and bounded/degraded health.
- Execution authority: this draft authorizes documentation only. Production edits, database writes, live-service changes, and execution require plan approval plus an explicit implementation command.

## Context

The missing web labels exposed a pipeline gap rather than an isolated UI defect.
The baseline fields were compact prose summaries, but each represented a
different function:

| Baseline function | Current V2 capability | Gap |
|---|---|---|
| mood: character affect carried between responses | `CharacterCognitionStateV2.affect_activations` has causal emotion, intensity, phase, trend, and sleep recovery | character affect is omitted from ordinary user-turn cognition and relevance; accepted user turns do not update it |
| global vibe: slower cross-conversation posture | character drives, meaning, goals, threats, events, gaps, and affect already form a richer native state | the full operational state is neither projected nor composed into ordinary responses; non-sleep elapsed fading is absent |
| last relationship insight: compact reason behind the current relationship stance | `RelationshipStateV2` preserves axes and evidence, while user cognition state preserves relationship-rooted events and affect | production relevance reads a field that native V2 no longer stores; the V2 relationship prompt and console expose axes without bounded causal context |
| user/channel adaptation | user and group-channel interaction-style images already exist; the P0 context-reconnection fix restores bounded group engagement to eligible group self-cognition goal/action judgment | ordinary user relevance still lacks style, and production does not yet reuse one immutable snapshot across relevance, cognition, and surface |
| operator visibility | the console exposes native state tables | it does not show the effective global projection consumed by a turn or the causal relationship projection |

The target is functional supersession, not restoration of the strings `mood`, `global_vibe`, or `last_relationship_insight`. The concrete behavioral requirement is:

```text
user A offends Asuna
  -> A's user state retains relationship injury
  -> a separate source-free character update derives native global affect
  -> a later user-B turn composes elapsed-effective global affect with B's
     relationship and the current scene, without blaming B
```

The foreground user episode and post-turn character episode each keep one mutable scope and one `StateUpdateV2`; no user affect is copied. The target dependency chain is:

```text
prior receipt -> bounded barrier -> one global/user/style load
-> relevance with scene + causal relationship + global posture + style
-> progress remains scene authority -> Core V2 with identity + effective global
+ current-user relationship + style -> surface with same snapshot
-> consolidation target -> Core V2 state-only carry-over
-> native reducer -> one character update -> transactional commit/receipt
-> next eligible private/group turn consumes that version
```

This follows `development_plans/active/bugfix/cognition_core_v2_character_identity_growth_bigbang_plan.md`. That plan owns durable identity/self-image/growth/revisions/lineage; this plan owns transient operational state and scoped runtime composition. Execution is blocked until the prerequisite is `completed`, signed, and rebaselined.

### 2026-07-30 P0 Context-Reconnection Rebaseline

`development_plans/archive/completed/bugfix/cognition_core_v2_p0_context_reconnection_bugfix_plan.md`
restored the existing group-engagement producer to native V2 before this draft
executes. The post-P0 baseline has:

- one eligible cycle-zero group self-cognition load through
  `build_group_engagement_action_context(...)`;
- one exact `group_engagement_action_context` V2 carrier with bounded
  `engagement_guidelines` and `confidence`;
- goal cognition and action planning as its only cognition consumers;
- exact empty context and zero group-style reads for ineligible events;
- canonical self-cognition `content.semantic_text` as current-scene authority
  ahead of fallback `text`; and
- deterministic and live trace coverage for the producer-to-consumer edge.

This plan must adopt that carrier and those consumer boundaries. Its future
immutable `interaction_style_turn_snapshot.v1` replaces the direct connector
load and supplies the existing group projection; it does not add a second
group-engagement field, loader, vocabulary, or cognition consumer. Stage 1
must rebaseline against the completed and user-signed P0 plan before any
implementation.

## Mandatory Skills

- `development-plan`, `local-llm-architecture`, and `no-prepost-user-input`: lifecycle, architecture, semantic ownership, budgets, routing, and reliability.
- `debug-llm` and `character-test`: one-at-a-time live calls, protected traces, normal-entrypoint multi-turn tests, DB inspection, and readable reviews.
- `control-console-web-development`: contracts, authenticated browser validation, cache checks, and screenshots.
- `py-style`, `cjk-safety` where applicable, `test-style-and-execution`, and `python-venv`: all Python/test/environment work; commands use `venv\Scripts\python.exe`.

## Mandatory Rules

1. Production changes remain blocked while this plan is `draft`.
2. Execution remains blocked until the identity-growth plan and the P0
   context-reconnection plan are completed and signed off; reread both final
   target contracts before touching this plan.
3. Rebaseline every file, symbol, test, and UI panel against the post-identity HEAD. A material contract conflict stops execution and requires a plan update.
4. Use `venv\Scripts\python.exe`, `apply_patch`, and repo-relative quoted paths; check `git status --short`, root/subsystem docs, source, and tests before production edits; do not read `.env`.
5. `CharacterCognitionStateV2` in the singleton operational character document is the sole persisted short-horizon global-state authority.
6. Create no persisted `mood`, `global_vibe`, relationship-insight prose field, second character-state document, channel-situation document, or shadow state.
7. Latest identity is durable character authority; global operational state is transient posture; per-user cognition is relational authority; conversation progress is fast channel-scene authority; interaction-style images are learned expression/participation guidance.
8. Current message, reply/media evidence, bounded history, and conversation progress own current facts and topic. Global state and style cannot introduce a topic, event, promise, relationship fact, or reason to speak.
9. Private and group interactions may affect one global operational state only through a separately derived character-scope V2 update containing source-free native causal roots and native affect; no user activation row, relationship row, or relationship root is copied.
10. Cross-scope model context and public console payloads contain no user/channel identifier, quote, raw message, raw evidence ref, entity handle, event description, memory text, or private fact from global state.
11. Native relationship causal context stays with the current user. It never becomes global state or group-channel style.
12. The consolidation LLM owns routing only. Its existing `lane_tasks` list remains capped at four durable tasks; one independent optional `character_operational_state_task` slot cannot consume or be starved by those four slots.
13. A Core V2 state-only carry-over stage owns `no_change` versus `apply` and one trusted semantic appraisal. Deterministic code owns parsing, exact question/path/handle authority, scope, provenance, privacy, native reduction, limits, persistence, timeout, and telemetry.
14. Every raw carry-over response uses `parse_llm_json_output(..., deterministic_only=True)`. Oversized, malformed, or contract-invalid output triggers full same-owner replacement with the same semantic context and a bounded error code, for at most three total attempts; rejected output is never accumulated into the next prompt.
15. The operational stage adds no foreground call. Normal chat may add one routed background carry-over call. A delivered accepted-task result bypasses the durable router and invokes the same carry-over entrypoint directly, adding at most one background call and never entering durable consolidation.
16. One source episode can produce at most one operational receipt and one character-scoped `StateUpdateV2`. `source_episode_id` is the durable idempotency root across retries and restarts.
17. Every character cognition-state writer uses one canonical transactional compare-and-set owner with prior `updated_at`. A stale additive carry-over may reload, apply elapsed fading, and reapply the same validated proposal once; a second conflict records a typed failure without overwrite.
18. Register the durable pending receipt and process-local predecessor token before the corresponding normal or accepted-task response becomes externally observable. A predecessor ends as `committed`, `no_change`, `failed`, or `timed_out` before a later eligible turn reads global state.
19. Healthy completion is visible to the next private or group turn. Failure releases the turn with the last valid committed state and explicit degraded health. The process-local coordinator bounds waiting; the durable receipt provides restart idempotency.
20. Read-time character elapsed fading intentionally extends the frozen Stage 2 sleep-only character evolution clause. It changes an effective copy; the next semantic write or sleep recovery persists once from that evolved base.
21. Preserve all twenty-one emotion definitions, required-root guards, adjacent-emotion distinctions, begin/sustain/inactive thresholds, per-emotion decay rates, phase/trend rules, and retention thresholds. Sleep remains the stronger recovery path.
22. Emotion globality is activation-instance ownership. The carry-over lane may derive the eighteen emotions whose guards accept character roots. `love_attachment`, `jealousy`, and `loneliness` remain relationship-required user-state activations.
23. Closed cause classes are deterministic privacy-safe projection/UI labels only. They are absent from LLM appraisal input/output and persisted emotion authority.
24. Load user and group interaction-style documents once per logical turn, in parallel where both apply, and reuse one immutable snapshot for relevance, cognition, surface, telemetry, and console comparison. Populate the existing bounded `group_engagement_action_context` projection for goal/action consumers and retire its direct connector load in the same cutover.
25. Style guidance stays source/consumer labeled and cannot override identity, boundaries, current evidence, relationship state, stance, topic, or reason to speak.
26. Persist and expose every active/fading native affect row up to the frozen twenty-one-row bound. Prompt top-N limits never define state or console truth.
27. Apply the Universal Bounded Boundary Policy below to every character/byte/cardinality/deadline/security guard reachable through normal private/group chat and accepted-task-result V2 paths. No raw `ValueError`, `PromptBudgetError`, `CognitionContextLimitError`, oversized payload, provider-output overflow, or receipt error may escape and crash the queue, worker, graph, or delivery pipeline.
28. Existing caps stay fixed. Deterministic reduction preserves required semantic authority and removes optional context in declared stable order. Irreducible input yields zero model calls and a typed owner-local terminal result.
29. Oversized or invalid required model output uses bounded same-stage replacement. Attempt exhaustion yields a typed degraded/no-change/deny/omit/operational-failure result, with no unauthorized state, action, or delivery side effect.
30. Privacy, permission, persistence, optimistic-version, and required-schema violations fail closed; they are recorded with bounded metadata, release predecessor waiters, and retain the last valid state. Their semantic content is never truncated into validity.
31. Console and telemetry consume redacted bounded projections; protected traces retain only configured bounded head/tail diagnostics for rejected output.
32. Controlled state-seeding proof and natural forward-only service proof are separate gates. The natural proof uses a clean guarded database and normal paths, allows controlled clock advance, and directly edits none of the state, receipts, or cognition output.
33. Historical/current Asuna data is outside blocking acceptance; future read-only smoke requires separate explicit instruction.
34. Run live LLM cases one at a time, inspect each result, and author human-readable reviews from real inputs, outputs, traces, state, and dialog.
35. Authenticated browser sign-off against the natural proof data is blocking.
36. After context compaction and after each signed major stage, reread this complete plan.
37. Complete independent code review, parent remediation, verification reruns, and lifecycle evidence before merge or completion.

## Must Do

1. Produce a baseline-to-post-identity-V2 capability closure matrix covering every producer, persistence, fading, relevance, cognition, surface, self-cognition, style, ordering, privacy, telemetry, and UI dependency.
2. Make character cognition-state `updated_at` a strictly increasing version token and add one compare-and-set persistence entrypoint.
3. Add deterministic read-time character elapsed fading for event/threat/goal/gap salience and all native affect rows, while retaining existing unresolved floors, emotion-specific rates, lifecycle thresholds, and stronger sleep recovery.
4. Add a signed twenty-one-emotion ownership matrix proving the eighteen character-root-eligible emotions, the three relationship-required user-scoped emotions, dual-scope same-id isolation, and unchanged frozen formulas/guards.
5. Add one canonical privacy-safe full operational state view plus bounded consumer selections from effective character state.
6. Add a causal relationship projection from native current-user relationship state, relationship-rooted entities, evidence freshness, and relationship-rooted affect.
7. Replace settled relevance’s dead legacy reads with the two native projections.
8. Add a consolidation target that routes settled episodes into one Core V2 state-only character carry-over stage.
9. Apply accepted carry-over appraisals through the existing native V2 reducer to produce zero or one character-scoped `StateUpdateV2`, then commit through canonical optimistic persistence with episode idempotency.
10. Add bounded predecessor ordering across normal chat and accepted background-task result paths.
11. Load one user/group interaction-style snapshot before settled relevance
    and project it by consumer role, reusing the post-P0
    `group_engagement_action_context` contract for eligible group
    self-cognition.
12. Carry the same state/style snapshot through Cognition Core V2 and L3
    instead of reloading at the connector or surface.
13. Wire branch-relevant global affect/pressure only into relevant appraisal/goal branches and expression; wire relationship causes into relationship/social relevance and appraisal.
14. Preserve projected history and conversation progress as fast scene/topic authority, with explicit topic-pivot regression coverage.
15. Expose full persisted/effective global affect, bounded pressures, latest consumed subset, version timestamp/digests, predecessor health, causal relationship context, and consumer-specific style projections in the control console.
16. Prove offence-driven dual consequences, cross-user carry-over, relationship isolation, and emotion-specific elapsed fading with controlled A/B and normal-entrypoint real-LLM sequences.
17. Measure LLM call counts, DB reads, prompt sizes, barrier overhead, routed update latency, and no-pending-turn latency against the post-identity baseline.
18. Inventory every direct-path size/cardinality/deadline guard and prove its exact reduction, replacement, or typed terminal disposition, including frontline relevance, settled relevance, decontextualization, Core V2, dialog/surface, consolidation routing/carry-over, accepted-task post-turn, persistence, telemetry, and console rendering.

## Deferred

- Identity/self-image/growth/revision/lineage stay with the prerequisite plan.
- Current-message/reply/media/history/progress, RAG ranking, memory/residue/reflection algorithms, interaction-style producers/schema/cadence, relationship reducers, adapters, delivery, scheduler, authorization, and tools remain unchanged.
- No channel mood/topic/scene store, second residue lane, historical migration/replay/backfill/import, or current Asuna repair is added.
- No operator editor/slider/rollback, prose insight writer, feature flag, compatibility mode, or alternate V1 path is added.

## Cutover Policy

Overall strategy: post-identity forward-only V2 big-bang replacement.

| Area | Policy | Instruction |
|---|---|---|
| Global transient state | canonical V2 | existing character cognition singleton with monotonic `updated_at`; no prose summaries |
| Emotion ownership | native scoped derivation | derive character carry-over from character roots; never copy user activation or relationship rows |
| Character elapsed evolution | intentional V2 extension | apply frozen per-emotion rates between sleeps through a pure effective-state pass; retain stronger sleep recovery |
| Relationship continuity | native projection | axes plus relationship-rooted causal entities/affect/freshness; no stored insight string |
| Channel scene | keep | current projected history and conversation progress remain sole fast scene/topic context |
| Style composition | one snapshot | load user/group overlays before relevance and reuse stage projections |
| Update producer | post-turn Core V2 | normal router selects its reserved slot; accepted-task result invokes the same state-only owner directly; zero/one update and no foreground call |
| Ordering | bounded predecessor | register before response exposure, release on commit/no-change/typed failure |
| Persistence | optimistic big-bang | one canonical compare-and-set writer and episode idempotency |
| Prompts | bounded big-bang | replace legacy relevance keys and update all V2 consumers together |
| Console | public projection | show native source/effective/consumed state and no legacy labels |
| Database | clean post-identity target | preserve the native V2 state shape; no historical migration |
| Tests/docs | big-bang | replace dead assumptions and sign the complete dependency chain |

Cut over caller, callee, schemas, prompts, persistence, tests, console, and docs together. Runtime gets no legacy fallback; rejected seed names in `character_profile.py` and isolated cleanup/reporting in `db/script_operations.py` remain. Any policy-row change requires plan update and user approval.

## Target State

### Ownership And Composition

| Context | Owner | Consumer role | Forbidden authority |
|---|---|---|---|
| latest identity revision | identity-growth system | stable personality, values, boundaries, self-image | current topic or transient emotion |
| character operational state | Cognition Core V2 | cross-scope affect, pressure, readiness, recovery | private detail, user relationship fact, channel topic |
| current-user cognition state | Cognition Core V2 | relationship interpretation and person-specific affect/goals | other users or global posture |
| conversation progress | conversation-progress subsystem | current thread, phase, momentum, open loops, emotional trajectory | durable character identity or cross-channel state |
| current evidence/history | intake/RAG | facts, participants, current event, direct reasons to respond | durable state mutation |
| user style image | interaction-style owner | person-directed social/expression preference | facts, relationship stance, response reason |
| group style image | interaction-style owner | shared participation, engagement, and pacing | current topic or person-specific relationship |

One interaction may create a user-scope relationship/entity/affect update and a separately derived source-free character entity/affect update. Each operation has one mutable scope. Same-id activations remain valid because roots/owners differ; no row is copied.

Conflict precedence is current evidence for facts/topic; identity/standards for values/boundaries; current-user state for that relationship; progress for fast scene/open loops; global state for affect/pressure only; then group/user style for shared/person-directed expression.

### Baseline-To-V2 Capability Closure Gate

Create `test_artifacts/cognition_core_v2_short_horizon_state/baseline_to_v2_capability_closure.md` before deletion/cutover. Give every baseline occurrence one disposition:

- `superseded`: exact V2 producer, state, projection, consumer, and proof;
- `intentionally_removed`: approved reason and negative proof;
- `not_applicable`: exact evidence that the occurrence never owned runtime behavior.

Required rows cover global affect/posture production; persistence/restart; elapsed/sleep recovery; relationship causes; relevance/appraisal/goal/surface consumers; self-cognition and accepted-task writers; style composition; scene precedence; privacy; ordering; observability/UI; all twenty-one emotion ownership/decay exclusions; offence dual consequence/non-offender isolation; and call/prompt/DB/latency cost. Zero unexplained rows and parent sign-off are required before Stage 7.

## Design Decisions

### Functional Successors

- Mood is superseded by the complete effective character affect state, bounded
  by the frozen maximum of one activation per emotion id: emotion, qualitative
  intensity, phase, trend, native root kind, projection-only cause class, and
  freshness. Prompt consumers receive branch-relevant subsets; the console
  retains the complete redacted view.
- Global vibe is superseded by the composition of effective character affect
  and active operational pressures from goals, threats, events, gaps, drives,
  and meaning. No single summary string is persisted.
- Last relationship insight is superseded by current relationship axes plus up
  to two relationship-rooted causal entities, relationship affect, evidence
  freshness, and update freshness. No replacement prose field is persisted.

### Emotion Scope And Carry-Over

The twenty-one-emotion registry remains unchanged. This plan changes where a
new activation may be independently derived and when character activations
decay; it does not change an emotion formula.

| Carry-over class | Emotion ids | Rule |
|---|---|---|
| character-root eligible | `joy`, `fear`, `anger`, `sadness`, `disgust`, `surprise`, `compassion_empathy`, `gratitude`, `envy`, `pride`, `shame`, `guilt`, `embarrassment`, `curiosity`, `awe`, `nostalgia`, `relief`, `ennui_existential_angst` | The character carry-over episode may derive the emotion only when its existing native character roots and frozen guard pass. |
| relationship-required | `love_attachment`, `jealousy`, `loneliness` | The ordinary carry-over episode receives no `RelationshipStateV2`; these activations remain current-user state. A global anger, sadness, fear, or other consequence requires its own character root and formula. |

For an offence, native appraisal axes determine the emotional result:
unfairness/intentionality or boundary injury can support anger; loss or failed
recovery can support sadness; norm violation can support disgust; credible
threat can support fear; exposure and identity threat can support embarrassment
or shame only when their self-responsibility guards also pass. Multiple
activations may coexist when each frozen guard passes. The
carry-over LLM never outputs an emotion id; the native reducer derives every
activation from accepted typed causes.

### Global Operational Update

The existing router emits `consolidation_route_decision.v2`. Its durable
`lane_tasks` field retains the existing maximum of four. A separate nullable
`character_operational_state_task` field is a reserved operational slot, so a
full durable task list cannot starve global carry-over. The slot cites one to
four trusted `source_views.source_key` values and a reason of at most 160
characters. Reflection origins and episodes without a settled character
response receive `null`.

For normal chat, a non-null slot invokes
`run_character_carryover_cognition(...)`. A delivered accepted-task result
invokes that entrypoint directly from its result post-turn path: it has no
durable-memory routing question, and the carry-over stage itself can return
`no_change`. Neither path runs response goals, actions, surface, or dialog.
The direct path uses up to four ref-complete current-episode source views in stable order `episode_trace`, `internal_thought`, `assistant_final_dialog`; none means immediate `no_change` and zero calls.

The carry-over stage has its own required model route,
`COGNITION_LLM_CHARACTER_CARRYOVER`. It constructs exactly one trusted
`SemanticQuestionV2` with `question_id=question_kind="character_carryover"`.
The model returns exactly one `SemanticAppraisalResultV2`; one semantic
question never expands into multiple result rows. The validator binds the
question id, retained evidence handles, role handles, and exact delta paths.
One accepted result contains at most four propositions and four deltas.

Allowed operational effects:

- create or reinforce at most one abstract event, threat, or knowledge gap
  candidate rooted in retained episode evidence;
- update at most four existing native axes across that candidate, existing
  character goals/threats/events/gaps, drive pressure, or meaning;
- derive affect through the existing causal emotion reducer.

Forbidden effects:

- identity/profile/self-image changes;
- drive importance or standards changes;
- relationship-axis changes;
- relationship context or relationship-rooted activation creation;
- user/group style changes;
- source-specific text, identifiers, promises, facts, or quotes;
- absolute state replacement.

The proposal is additive and episode-rooted. On one compare-and-set conflict,
the persistence owner reloads the latest state, applies elapsed fading, and
reapplies the same validated proposal. It never asks deterministic code to
reinterpret the episode.

Trusted prompt handles are `self`, `unspecified_other`, and `group_context`; the latter two map only to canonical source-free `third_party/operational:unspecified_other` and `group/operational:group_context` role refs.
Exact candidate domains are:

- event: `responsibility`, `intentionality`, `harm`, `unfairness`, `exposure`,
  `repair_need`, `reparability`, `norm_violation`, `contamination_risk`,
  `identity_threat`, `outcome_impact`, `expectation_mismatch`,
  `comparison_gap`, `vastness`, `memory_warmth`, and `temporal_loss`;
- threat: `likelihood`, `expected_harm`, `uncertainty`, `controllability`,
  `coping_potential`, and `residual_pressure`;
- knowledge gap: `relevance`, `uncertainty`, `learnability`, `novelty`, and
  `model_accommodation`;
- existing character entities expose only their native axes; drives expose
  `pressure`; meaning exposes `purpose_coherence`, `agency`, and
  `identity_continuity`.

Every candidate path cites originating prompt evidence. Before trial reduction, deterministic privacy binding substitutes a canonical retained `EvidenceRefV2`: `source_kind="episode"`, opaque `source_id=source_episode_id`, original `occurred_at`, and `semantic_summary` equal only to a closed `character operational event|threat|knowledge gap` label. Candidate descriptions use the same closed label. Model explanation/reasons remain trace-only; user/channel IDs, relationship handles, quotes, and source descriptions are invalid. Native reducers alone derive goals/affect.

### Elapsed Fading

`apply_character_elapsed_decay(...)` uses elapsed UTC time and existing emotion
decay rates. It:

- decays goal/threat/event/gap salience at the existing four points/hour;
- preserves the current unresolved-pressure floor;
- applies each emotion's frozen rate/lifecycle/retention rule;
- leaves identity, standards, drive importance/pressure, and meaning unchanged;
- computes once from persisted `updated_at` to `effective_at`, retains that timestamp on the pure copy, writes nothing, and lets the next semantic/sleep commit persist once with a strictly newer timestamp.

Sleep remains the only stronger drive/residual recovery. Activations may become fading/disappear between sleeps; unresolved entities keep native floors, terminal entities follow native terminalization, and offender injury remains user-scoped.

### Relationship Causal Projection

`project_relationship_context(...)` accepts complete current-user cognition state and `effective_at`. It bands every relationship axis; selects only entities targeting that relationship; orders by active lifecycle, salience, recency; exposes at most two 160-character causal rows and two rooted affect rows plus relationship/evidence freshness; and omits IDs, raw evidence, scalars, and other users. Consumers reduce this canonical projection rather than reconstruct it.

### Projection-Only Cause Classes

Global cross-scope context contains no free-form source description. It maps
each affect row from its own primary root and each pressure row from its own
native row. Eligible affect phases are `active|fading`; eligible entity
statuses are goal `pursuing|blocked`, threat/event `active`, and knowledge gap
`open|reduced`.
Predicates use scalar threshold `>= 40` and this first-match precedence:

| Order | Cause class | Exact native predicate |
|---:|---|---|
| 1 | `safety_pressure` | root kind `threat` |
| 2 | `uncertainty_pressure` | root kind `knowledge_gap` |
| 3 | `meaning_pressure` | root kind `meaning` |
| 4 | `boundary_pressure` | goal kind `autonomy_boundary`, or linked event `max(unfairness,norm_violation,exposure,identity_threat) >= 40` |
| 5 | `repair_pressure` | goal kind `moral_repair`, or linked event `repair_need >= 40` and `max(harm,norm_violation) >= 40` |
| 6 | `safety_pressure` | goal kind `safety`, or linked event `harm >= 40` |
| 7 | `loss_pressure` | goal kind `loss_recovery`, or linked event `temporal_loss >= 40` |
| 8 | `uncertainty_pressure` | goal kind `epistemic_exploration` |
| 9 | `meaning_pressure` | goal kind `meaning_reconstruction`, or meaning `purpose_coherence < 40` or `agency < 40` |
| 10 | `competence_pressure` | goal kind `self_improvement`, or drive id `competence` with `pressure >= 40` |
| 11 | `connection_warmth` | linked event `memory_warmth >= 40`, or affect `emotion_id` is `joy`, `gratitude`, or `compassion_empathy` |
| 12 | `relationship_strain` | linked event `expectation_mismatch >= 40` with role-ref entity kind `third_party` or `group` |
| 13 | `goal_pressure` | any remaining eligible goal, or any remaining drive with `pressure >= 40` |
| 14 | `general_activation` | validated eligible root with no earlier match |

No description, explanation, source text, identifier, or model-authored class
participates. The class is derived after native reduction, cannot create,
suppress, merge, or rename emotion, and is absent from carry-over prompts and
persistence authority. Model contexts receive only the class plus native
emotion/pressure kind, qualitative band, lifecycle, trend, and freshness.

### Interaction-Style Composition

`build_interaction_style_context(...)` becomes the immutable turn snapshot;
group turns load user/group documents concurrently. Relevance receives up to
three engagement rows/source after grounded participation evidence. Cognition
gets up to two social and two engagement rows/source. For eligible group
self-cognition, its group engagement projection populates the existing exact
`group_engagement_action_context` consumed by goal cognition and action
planning. Surface gets existing bounded speech/social/pacing/engagement
projections. The connector's post-P0 direct group loader and the surface DB
reload are removed only when this one snapshot supplies those same consumers.

### Branch Routing

- `relationship_social` receives current-user causes, relevant global affect, and cognition style.
- `goal_threat_outcome` receives relevant global affect/pressures.
- `moral_identity` receives only boundary/repair global rows.
- `existential_drive` receives meaning/goal/competence pressure rows.
- `event_agency`/`epistemic` receive posture only for planner-routed matching roots.
- goal cognition gets relevant global/current-relationship projections.
- surface gets at most two affect rows selected by stance relevance, intensity, lifecycle, freshness; it cannot alter stance/facts.

### Ordering And Freshness

Evolve the existing clean-target `post_turn_lifecycle_record.v1` to mutable
`post_turn_lifecycle_record.v2` in the same collection. It embeds one
`character_operational_receipt.v1`; the unique `source_episode_id` index and
configured audit TTL define the durable idempotency horizon. No new collection
or backfill is added.

Before normal response completion or accepted-task adapter dispatch, atomically
insert/verify the lifecycle row with receipt `pending`, a 45-second lease, and
the current character-state version. A duplicate terminal episode reuses its
receipt; a duplicate with a live lease observes `in_progress` and does not run
a second model call. Startup changes expired `pending` receipts to `timed_out`.
A transaction commits state plus `committed` receipt together; `no_change|failed|timed_out` updates only the receipt. Initial claim failure creates an in-memory `durable=false, failed/persistence_failed` receipt, skips carry-over, allows the response, and releases waiters; transaction failure similarly leaves state unchanged.

The process-local coordinator supplies low-latency ordering:

1. `register_predecessor` assigns a monotonic sequence and future before
   exposure; at most 256 pending tokens exist.
2. Capacity overflow writes `failed/capacity_exceeded`, returns an already
   completed token, skips carry-over, and exposes degraded health.
3. `await_predecessors` captures the current sequence watermark and awaits all
   older futures concurrently under one absolute 45-second deadline.
4. Timeout terminalizes pending receipts, prevents their late commit, releases
   all waiters, and loads the last committed state.
5. Completed entries are evicted after captured waiters release; terminal
   receipt history remains durable until the configured audit TTL.

`updated_at` compare-and-set remains the persistence lock for consolidation,
`persona_supervisor2_cognition._commit_cognition_state`, and
`reflection_cycle.affect_settling.run_daily_affect_settling`.

### Universal Bounded Boundary Policy

This policy covers every size, byte, cardinality, deadline, and output-security
guard in the direct normal-chat and accepted-task-result V2 path and extends the frozen owner-local policy in `development_plans/archive/completed/bugfix/cognition_core_v2_prompt_budget_and_failure_containment_bugfix_plan.md`. Deterministic
code may reduce representation, validate authority, or select a typed
disposition; it does not infer semantic intent.

| Boundary | First disposition | Repeat disposition | Exhausted disposition |
|---|---|---|---|
| optional/supplemental input | remove lowest-priority rows in declared order, then middle-truncate eligible semantic text to its existing non-empty floor | none | owner-local typed skip/degraded result with zero model calls |
| frontline relevance input | remove bot continuity, preludes, excess open turns, then bound body while retaining both ends | none | authoritative input returns validated `start/input_limit`; non-authoritative input returns validated `discard/input_limit` |
| settled relevance input | remove fresh history, style guidelines, excess fragments, then scene/relationship supplements | none | non-authoritative returns `ignore/input_limit`; authoritative raises typed `SettledRelevanceContractError(input_limit)` for existing wait-or-operational-failure settlement |
| decontextualization/RAG projection input | trim optional history, media description, and lower-ranked evidence before required current-message fields | none | typed pre-state `input_limit`; decontextualization stops, or RAG returns an explicit unavailable evidence packet |
| Core V2 appraisal/goal/surface input | reuse canonical `prompt_budget` source-order reduction; preserve current evidence handles and required selections | none | optional appraisal slot is omitted with warning; required stage returns typed pre-commit operational failure |
| model output over stage cap | skip parse; retain only configured bounded head/tail trace | full same-owner replacement within that stage's existing attempt cap | typed contract/provider exhaustion; no partial state/action/delivery side effect |
| malformed/unknown/wrong-type model output | canonical parse then exact validator | full same-owner replacement where output is required | non-authoritative relevance becomes ignore/discard; router becomes empty plus `failed/route_invalid`; optional Core V2 slot is omitted; required cognition/dialog becomes typed operational failure; carry-over becomes degraded `no_change` |
| persistence/privacy/permission/CAS | reject candidate without semantic repair | one exact carry-over CAS reapply only | typed fail-closed receipt/result, last valid state, released barrier |
| telemetry/console payload | redact, cardinality-bound, and middle-truncate display-only text | none | explicit `unavailable|degraded` projection; endpoint/page remains renderable |

Every owning boundary records `normal|reduced|replaced|degraded|failed`,
attempt count, original/final character counts, and a bounded error code.
Provider context-length/HTTP-413 errors classify as `input_limit` and are never retried with unchanged input; existing transient-provider retry caps remain unchanged.
Rejected raw content, exception text, IDs, and secrets stay out of public
telemetry. Tests assert no raw limit exception reaches queue completion,
accepted-task delivery, graph API, or console endpoint.

### Console Observability

The Character page gains **Operational posture** after identity: persisted version/time; effective-at/fading status; all redacted persisted/effective affect rows (emotion, intensity, phase, trend, root, cause class, freshness); bounded pressures; separate consumed subset; latest run/version/digest/exact public context/status; and predecessor/reduction/replacement/failure health. User Relationship adds causes/affect/freshness; User/Group style labels `relevance|cognition|surface`.

The cognition graph stores exact public consumption under `l2.reasoning.detail.context_consumption`; the console renders it without reconstruction.

## Data Migration

No historical migration/backfill/baseline import is permitted. Native `cognition_state.v2` keeps strictly increasing `updated_at`. New episodes write lifecycle v2; existing lifecycle v1 audit rows remain operationally read-inert and expire under their TTL, with no conversion or inferred receipt. Startup fails closed if canonical native state is absent and never initializes from baseline fields.

## Contracts And Data Shapes

### Character Operational State View And Consumer Context

```text
character_operational_state_view.v1
  source_updated_at: UTC timestamp
  effective_at: UTC timestamp
  source_digest: sha256 digest
  view_digest: sha256 digest
  affect: list[max 21]
    emotion_id: existing EMOTION_IDS
    intensity: existing qualitative band
    phase: existing phase
    trend: existing trend
    primary_root_kind: goal | threat | event | knowledge_gap | drive | standard | meaning
    cause_class: closed enum above
    freshness: existing project_duration label
  pressures: list[max 8]
    pressure_kind: goal | threat | event | knowledge_gap | drive | meaning
    salience: existing qualitative band
    lifecycle: existing lifecycle label
    cause_class: closed enum above
    freshness: existing project_duration label

character_operational_context.v1
  source_updated_at: UTC timestamp
  effective_at: UTC timestamp
  view_digest: sha256 digest
  context_digest: sha256 digest
  consumer_role: settled_relevance | appraisal branch id | goal | surface
  affect: list[max 3]
    same row shape as state view
  pressures: list[max 4]
    same row shape as state view
```

`source_digest` hashes the canonical persisted operational state.
`view_digest` hashes the complete redacted state view excluding itself.
`context_digest` hashes the exact consumer selection excluding itself. Neither
digest is semantic evidence. All consumer contexts are selected from the
canonical state view; they never re-read or reinterpret persistent state.

### Relationship Operational Projection

```text
relationship_operational_context.v1
  axes: existing qualitative relationship bands
  causal_context: list[max 2]
    entity_kind: goal | threat | event | knowledge_gap
    semantic_summary: normalized, max 160 chars
    salience: qualitative band
    lifecycle: lifecycle label
    freshness: duration label
  affect: list[max 2]
    emotion_id
    intensity
    phase
    trend
    freshness
  relationship_freshness: duration label
  evidence_freshness: duration label | "无证据"
```

### Consolidation Route Decision

```text
consolidation_route_decision.v2
  lane_tasks: existing closed durable task rows, max 4
  character_operational_state_task: null | object
    reason: normalized non-empty string, max 160 chars
    source_keys: unique trusted source_view keys, 1..4
```

Both top-level keys are required/exact; top-level or durable-task failure keeps the existing typed empty-route result. The operational slot is independently available only for settled non-reflection character responses; its invalid source/cardinality/shape becomes `null` plus a typed error while valid durable tasks continue.
Operational source policy accepts only ref-complete `current_turn_user_message|assistant_final_dialog|internal_thought|episode_trace`; any RAG/reflection key rejects the operational task.
A validated `null` terminalizes `no_change`; malformed routing or a rejected non-null slot terminalizes `failed/route_invalid|source_policy_rejected`, never silent `no_change`.

### Character Carry-Over Decision

```text
character_carryover_decision.v1
  schema_version: "character_carryover_decision.v1"
  action: no_change | apply
  reason_code:
    no_lingering_effect | already_represented | transient_scene_only |
    unsupported | lingering_character_effect
  privacy_disposition: source_free | unsafe
  semantic_appraisal: null | one SemanticAppraisalResultV2

character_carryover_result.v2
  schema_version: "character_carryover_result.v2"
  decision: character_carryover_decision.v1
  state_update: null | one character-scoped StateUpdateV2
  disposition: no_change | apply | degraded
  error_code:
    null | input_limit | output_limit | contract_exhausted |
    provider_exhausted | privacy_rejected | deadline_exceeded |
    state_rejected
  attempt_count: integer 0..3
```

`no_change` requires `semantic_appraisal=null`. `apply` requires
`privacy_disposition=source_free`, exact question id, one to four propositions,
one to four deltas, at most one candidate handle, retained evidence, and at least one native state change
after trial reduction. Scope is bound to `character` outside model output. The
LLM emits neither cause class nor emotion id. Unsafe, empty-after-reduction,
over-limit, exhausted, or deadline results return `degraded/no_change` with no
state update. Free-form explanations remain protected and bounded.

```text
CharacterCarryoverServicesV1
  llm: LLMInvoker
  config: LLMCallConfig(route_name="COGNITION_LLM_CHARACTER_CARRYOVER")

run_character_carryover_cognition(
  *,
  source_episode_id: str,
  evidence: Sequence[CognitionEvidenceV2],
  base_state: Mapping[str, Any],
  effective_at: str,
  services: CharacterCarryoverServicesV1,
) -> CharacterCarryoverResultV2
```

### Interaction-Style Snapshot

```text
interaction_style_turn_snapshot.v1
  user_style:
    revision: int | 0
    status: active | empty | missing | failed
    overlay: existing sanitized overlay
  group_channel_style:
    revision: int | 0
    status: active | empty | missing | failed
    overlay: existing sanitized overlay | omitted for private
  relevance_projection: source-labeled engagement rows
  cognition_projection: source-labeled social/engagement rows
  surface_projection: source-labeled existing L3 rows
  snapshot_digest: sha256 digest
```

Prompt payloads omit document IDs, source reflection runs, platform/user/channel
IDs, and storage timestamps.

### Ordering Receipt

```text
character_operational_receipt.v1
  source_episode_id: protected internal field
  sequence: monotonic process-local integer
  status: pending | no_change | committed | failed | timed_out
  durable: bool
  base_updated_at: UTC timestamp
  committed_updated_at: UTC timestamp | none
  registered_at: UTC timestamp
  completed_at: UTC timestamp | none
  lease_owner: protected bounded process id
  lease_expires_at: UTC timestamp
  attempt_count: integer 0..3
  error_code:
    none | route_invalid | source_policy_rejected | capacity_exceeded | input_limit | output_limit |
    contract_exhausted | provider_exhausted | privacy_rejected |
    deadline_exceeded | state_rejected | transaction_unavailable | persistence_failed |
    version_conflict

post_turn_lifecycle_record.v2
  existing lifecycle_record_id/source_episode_id/delivery/action/status fields
  character_operational_receipt: character_operational_receipt.v1
  created_at: UTC timestamp
  purge_after: configured audit TTL timestamp

CharacterOperationalClaimV1
  disposition: claimed | in_progress | terminal
  receipt: character_operational_receipt.v1

PredecessorTokenV1
  source_episode_id: protected; sequence: process-local integer; registered_at: UTC

PredecessorBarrierResultV1
  status: healthy | degraded
  watermark: integer
  awaited_count: integer; timed_out_count: integer; wait_ms: non-negative integer
```

Public telemetry omits `source_episode_id` and exposes only status, version
timestamps, duration, and bounded error code.

### Public Interfaces

- `cognition_core_v2.state_reducers.apply_character_elapsed_decay(state, *, elapsed_seconds) -> dict`
- `cognition_core_v2.state_projection.project_character_operational_state(state, *, effective_at) -> character_operational_state_view.v1`
- `cognition_core_v2.state_projection.select_character_operational_context(state_view, *, consumer_role) -> character_operational_context.v1`
- `cognition_core_v2.state_projection.project_relationship_context(user_state, *, effective_at) -> dict`
- `cognition_core_v2.character_carryover.run_character_carryover_cognition(...) -> CharacterCarryoverResultV2`, with the exact signature above
- `db.character.compare_and_replace_character_cognition_state(*, expected_updated_at: str, replacement: Mapping[str, Any]) -> bool`; `db.post_turn_lifecycle` alone uses its private session-aware primitive inside the cross-collection transaction
- `db.interaction_style_images.build_interaction_style_context(...) -> interaction_style_turn_snapshot.v1`
- `db.post_turn_lifecycle.claim_character_operational_receipt(*, lifecycle_record: PostTurnLifecycleRecordV2, sequence: int, base_updated_at: str, registered_at: str, lease_owner: str, lease_expires_at: str) -> CharacterOperationalClaimV1`
- `db.post_turn_lifecycle.commit_character_operational_update(*, source_episode_id: str, lease_owner: str, expected_updated_at: str, replacement: Mapping[str, Any], completed_at: str) -> CharacterOperationalReceiptV1 | Literal["version_conflict"]`
- `db.post_turn_lifecycle.complete_character_operational_receipt(*, source_episode_id: str, lease_owner: str, status: Literal["no_change","failed","timed_out"], completed_at: str, error_code: str | None, attempt_count: int) -> CharacterOperationalReceiptV1`
- `db.post_turn_lifecycle.expire_character_operational_receipts(*, now: str) -> int`
- `db.post_turn_lifecycle.get_character_operational_receipt(source_episode_id: str) -> CharacterOperationalReceiptV1 | None`
- `consolidation.character_operational_state.run_character_operational_target(*, source_episode_id: str, sequence: int, evidence: Sequence[CognitionEvidenceV2], effective_at: str, services: CharacterCarryoverServicesV1) -> CharacterOperationalReceiptV1`
- `brain_service.character_state_ordering.register_predecessor(source_episode_id: str, *, registered_at: str) -> PredecessorTokenV1`
- `brain_service.character_state_ordering.capture_predecessor_watermark() -> int`
- `brain_service.character_state_ordering.complete_predecessor(token: PredecessorTokenV1, receipt: CharacterOperationalReceiptV1) -> None`
- `brain_service.character_state_ordering.await_predecessors(*, before_sequence: int, timeout_seconds: float = 45.0) -> PredecessorBarrierResultV1`

No second public reader/writer/projection path is permitted.

## LLM Call And Context Budget

- Foreground model-call count and all existing V2 route attempt caps remain
  unchanged. Added global/relationship/style fields fit inside each existing
  cap through the Universal Bounded Boundary Policy.
- Normal chat retains exactly one consolidation-router primary call. Router
  parsing is deterministic-only; invalid/oversized output produces a typed
  empty route and no hidden JSON-repair call. The reserved operational slot
  changes output shape, not router call count.
- Normal chat adds zero carry-over calls when the reserved slot is null and one
  primary background call when selected. Accepted-task result delivery adds no
  router call and zero or one direct background carry-over primary call.
- `COGNITION_LLM_CHARACTER_CARRYOVER` has required `BASE_URL|API_KEY|MODEL|MAX_COMPLETION_TOKENS|THINKING_ENABLED` values, appears in Brain model routes, and enforces an 8,192-token hard ceiling; the operator supplies values through normal deployment configuration before live gates, without plan-driven `.env` inspection/editing.
- Carry-over uses one primary plus at most two full replacements: three total
  calls, no JSON-repair calls, no accumulation of rejected output, and no
  retries after the absolute 45-second operational deadline.
- Static carry-over system text is at most 8,000 characters; the dynamic JSON
  packet is at most 8,000; total rendered input is at most 16,000. Raw model
  output above 8,000 characters is not parsed and triggers replacement.
- The dynamic packet retains all one-to-four selected current-episode evidence
  handles. It first removes lowest-priority pre-existing state rows in stable
  source order, then middle-truncates eligible evidence `semantic_text` from
  lowest priority to a 96-character non-empty floor. If required structure
  still exceeds 8,000, it makes zero calls and returns
  `degraded/input_limit`.
- Replacement input is the same fitted semantic packet plus only a closed
  error code. Protected rejected-output diagnostics retain at most 2,000 head
  and 2,000 tail characters with one marker; public traces retain none.
- The 45-second deadline includes normal routing (or direct accepted-task entry), calls, reduction, privacy validation,
  compare-and-set retry, transaction, and receipt completion.
- Full redacted state view: at most twenty-one affect and eight pressure rows;
  console/telemetry only and never copied wholesale into a model prompt.
- Each model-facing global context: at most three branch-relevant affect and
  four pressure rows, no free-form source text, and at most 1,200 characters.
- Relationship projection: at most two causal and two affect rows; serialized
  payload at most 900 characters.
- Relevance style: at most three engagement guidelines per source.
- Cognition style: at most two social and two engagement guidelines per source.
- Surface style: no larger than the existing L3 per-field caps.
- Appraisal routing follows the branch table above; the full global projection
  is not copied into every parallel branch.
- Prompt-size instrumentation records static, dynamic, total-input, raw-output,
  attempt, and reduction counts without raw private content.
- A captured long-context V2 case proves every direct path remains inside its
  cap or reaches the declared typed disposition without an uncaught exception.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/cognition_core_v2/character_carryover.py`: Core V2 state-only prompt, exact decision contract, bounded attempts, native trial reduction, and zero-or-one `StateUpdateV2` result.
- `src/kazusa_ai_chatbot/consolidation/character_operational_state.py`: target adapter, episode idempotency, optimistic commit orchestration, and receipts; it owns no emotion schema or semantic reinterpretation.
- `src/kazusa_ai_chatbot/brain_service/character_state_ordering.py`: bounded predecessor tokens and public receipts.
- `tests/test_cognition_core_v2_operational_projection.py`: elapsed fading, full-state view, consumer selection, projection-only cause mapping, relationship causes, caps, digests, and privacy.
- `tests/test_cognition_core_v2_emotion_scope_matrix.py`: all twenty-one ids, eighteen character-root-eligible cases, three relationship-required exclusions, same-id cross-scope isolation, unchanged formulas, and decay-rate parity.
- `tests/test_cognition_core_v2_character_carryover.py`: decision validation, no emotion/cause-class output, native derivation, offence distinctions, and zero-or-one character update.
- `tests/test_character_operational_state_consolidation.py`: four-plus-one cardinality, operational source/failure isolation, adapter ownership, idempotency, and CAS retry.
- `tests/test_character_operational_state_ordering.py`: normal, cross-worker, failure, timeout, and shutdown behavior.
- `tests/test_cognition_core_v2_universal_bounded_boundaries.py`: parameterized direct-path guard inventory, stable input reduction, oversized-output replacement, typed exhaustion, side-effect exclusion, and no escaped limit exception.
- `tests/test_short_horizon_state_composition_integration.py`: relevance/cognition/surface/style wiring and precedence.
- `tests/test_short_horizon_state_composition_live_llm.py`: controlled A/B live-model cases.
- `tests/test_short_horizon_state_composition_e2e_live_llm.py`: natural private/group causal sequences.
- `tests/control_console_e2e/test_short_horizon_state_visibility_e2e.py`: authenticated real-service browser gate.
- `tests/fixtures/cognition_core_v2_short_horizon_state_cases.json`: controlled inputs and behavior rubrics.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py`: pure read-time character fading and version-safe native reducer use.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`: canonical full global view, consumer selection, projection-only cause labels, and causal relationship projection.
- `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py`, `contracts.py`, `workspace.py`, `semantic_source_planner.py`, `semantic_appraisal.py`, `goal_cognition.py`, `prompt_budget.py`, `model_attempt_policy.py`, and `facade.py`: carry-over entrypoint, typed input, branch routing, bounded consumption, and owner-local terminal limits.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: state ownership, projection, fading, and branch contracts.
- `src/kazusa_ai_chatbot/config.py`, `src/control_console/brain_model_routes.py`, `tests/test_cognition_core_v2_stage_model_routing.py`, and `tests/test_control_console_brain_model_routes.py`: required carry-over model route and hard 8,192-token stage ceiling.
- `src/kazusa_ai_chatbot/db/character.py`: monotonic update timestamp, canonical compare-and-set writer, and private session-aware transaction primitive.
- `src/kazusa_ai_chatbot/db/schemas.py`, `db/post_turn_lifecycle.py`, `db/bootstrap.py`, and `db/__init__.py`: lifecycle v2 receipt claim/terminalization, unique episode idempotency, TTL, startup expiry, and atomic receipt/state transaction.
- `src/kazusa_ai_chatbot/db/interaction_style_images.py` and `db/__init__.py`: immutable all-stage snapshot, concurrent group loads, and stage projections.
- `src/kazusa_ai_chatbot/db/README.md`: replace stale runtime mood/vibe documentation.
- `src/kazusa_ai_chatbot/consolidation/target.py`, `schema.py`, `source_policy.py`, `lane_router.py`, `core.py`, `persistence.py`, `__init__.py`, and `README.md`: independent operational routing, ref-complete source validation, and sanitized metadata.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`, `turn_settlement.py`, `__init__.py`, and `README.md`: registration, bounded waiting, completion, and lifecycle receipt.
- `src/kazusa_ai_chatbot/service.py`: pre-relevance barrier/state/style load, dead legacy-read removal, pre-exposure normal/accepted-task registration, direct accepted-task carry-over, bounded operational failures, graph consumption projection, and snapshot reuse.
- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py`, `persona_relevance_agent.py`, and `relevance/README.md`: native context, tie-breaker rules, deterministic fitting, and declared authoritative/non-authoritative input-limit dispositions.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`, `persona_supervisor2.py`, `persona_supervisor2_cognition.py`, `persona_supervisor2_l3_surface.py`, `persona_supervisor2_msg_decontextualizer.py`, `persona_supervisor2_rag_dispatch.py`, `persona_supervisor2_rag_evaluator.py`, `dialog_agent.py`, and `nodes/README.md`: snapshot/CAS/surface cutover plus universal boundary normalization without changing semantic/attempt ownership.
- `src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py`: compare-and-set in `run_daily_affect_settling` without changing sleep semantics.
- `src/control_console/contracts.py`, `repository.py`, `app.py`, `redaction.py`, `static/index.html`, `static/console.js`, `static/console.css`, and `README.md`: redacted operational/relationship/style panels and latest-consumption rendering.
- `tests/test_post_turn_lifecycle_record.py`, `test_cognition_chain_connector_mapping.py`, `test_reflection_affect_settling.py`, `test_background_work_delivery.py`, `test_service_background_consolidation.py`, `test_frontline_relevance_agent.py`, `test_persona_relevance_agent.py`, `test_relevance_turn_settlement.py`, `test_msg_decontextualizer.py`, `test_persona_supervisor2_rag2_integration.py`, `test_rag_projection.py`, `test_dialog_agent.py`, `test_cognition_core_v2_prompt_budget_continuity.py`, `test_cognition_core_v2_model_retry_continuity.py`, `test_interaction_style_images.py`, `test_control_console_repository.py`, `test_control_console_cognition_graph.py`, and `test_control_console_web_surface.py`: exact existing focused/regression owners.
- `README.md` and `docs/HOWTO.md`: operational state ownership, diagnostics, and console interpretation.
- `development_plans/README.md`: lifecycle status only during approval, execution, and archive transitions.

### Delete

- No production file is deleted by this plan.
- Delete runtime reads, prompt keys, fixtures, and assertions whose only
  contract is `character_mood`, persisted mood/vibe, or
  `last_relationship_insight`.
- Delete the L3 independent interaction-style DB load after all consumers use
  the shared snapshot.

### Keep

- The prerequisite identity ledger, revision, proposal/review, and console lineage.
- Native relationship axes and foreground user-state reducer semantics.
- Conversation-progress and bounded-history producers.
- Interaction-style persistence, producer, validation, and ownership schemas.
- Existing emotion definitions, root guards, adjacent-emotion distinctions,
  thresholds, rates, causal reducer invariants, and stronger sleep recovery;
  only ordinary character-scope elapsed scheduling is extended.
- Baseline/historical cleanup references that are isolated from runtime.
- Adapters, RAG evidence ownership, dialog content ownership, actions, permissions, delivery, scheduler, reflection, and residue boundaries.

## Overdesign Guardrail

The minimal system is the existing character state, one Core V2 state-only carry-over, canonical global/relationship projections, one style snapshot, durable receipt plus bounded process ordering, and public observability.

Forbidden: prose state; channel mood/topic state; second global collection; event replay; foreground model calls; second emotion ontology/classifier/reducer/DAG; copied user/relationship state; style topic/relevance authority; per-consumer reload/projection; compatibility aliases/flags/fallback/dual writes; or operator editor. New helpers are limited to the exact carry-over, projection validation, transactional CAS, and predecessor owners above.

## Agent Autonomy Boundaries

Agents may choose private helper names, preserve-rubric fixture wording, and update a post-identity path proven to have moved without ownership change.

Stop for direction before adding persisted fields/collections; changing the emotion matrix/formulas/guards, cause enum/predicates, caps/deadline/rates/lifecycle/routing/precedence; changing identity, relationship reducer, progress, RAG semantics, reason-to-speak, action/scheduler/adapter/delivery semantics; adding calls/fallbacks/compatibility/flags/migration/current-Asuna access; broadening visibility; or editing outside the surface except a proven path move.

## Implementation Order

### Stage 1 — Post-Identity Rebaseline And Closure

1. Confirm the identity-growth and P0 context-reconnection prerequisite plans
   are completed and their execution evidence is signed.
2. Require a clean post-identity worktree, then record `git status --short`,
   HEAD, Python version, mandatory-skill reads, and relevant docs/source/tests.
3. Locate every character cognition-state reader/writer and every legacy runtime occurrence with `rg` and `git grep main`.
4. Create and sign the baseline-to-V2 capability closure artifact and the
   twenty-one-emotion scope/decay matrix.
5. Verify the post-identity forms of
   `persona_supervisor2_cognition._commit_cognition_state`,
   `affect_settling.run_daily_affect_settling`,
   `post_turn_lifecycle.py`, the accepted-task result path, and every exact
   focused test listed in this plan.

### Stage 2 — Parent-Owned Focused Test Contract

6. Create emotion-scope/projection tests for all twenty-one ids, versions,
   fading, exact cause predicates, relationship selection, privacy, caps, and
   digests; create the universal bounded-boundary inventory test.
7. Run each focused file and record the expected missing-symbol/contract failures.
8. Create persistence/ordering tests for lifecycle v2 claim, duplicate/live
   lease, terminal reuse, restart expiry, transaction atomicity, one CAS
   reapply, predecessor success/failure/timeout/capacity, and shutdown.
9. Run each focused file and record expected failures.
10. Start one production-code subagent with the approved plan, mandatory skills, focused tests, and production-only ownership.

### Stage 3 — Native State, Projection, And Persistence

11. Enforce strictly increasing character-state `updated_at` values.
12. Add character elapsed fading and canonical global/relationship projections.
13. Add lifecycle v2 receipt persistence/transactional commit; move exact cognition/affect-settling writers to compare-and-set.
14. Pass all Stage 2 state/projection/persistence tests before integration wiring.

### Stage 4 — Core V2 Carry-Over And Consolidation Adapter

15. Add the dedicated route, one-question/one-appraisal contract, fitting/replacement/privacy policy, and zero-or-one update.
16. Add the reserved router slot/adapter; invoke from selected normal routing or direct accepted-task post-turn.
17. Trial-reduce the validated proposal through native reducers, derive affect, and commit the validated `StateUpdateV2` through optimistic persistence.
18. Record sanitized route, no-change, commit, conflict, failure, and timeout metadata.
19. Pass route/carry-over tests, including four durable plus one operational task and accepted-task durable-consolidation exclusion.

### Stage 5 — Ordering And Shared Turn Snapshot

20. Add durable claim/process token before normal completion/accepted-task dispatch and barrier capture/wait before relevance.
21. Load effective character state and one interaction-style snapshot before settled relevance.
22. Reuse the same immutable state/style snapshot through cognition and surface.
23. Pass ordering/restart/capacity/read/style/failure/timeout/late-commit tests.

### Stage 6 — Relevance, Cognition, And Surface Composition

24. Replace dead relevance input with native projections and exact frontline/settled limit dispositions.
25. Add branch-specific operational/relationship/style context to V2 contracts and prompts.
26. Add goal-stage context and selected-stance-relevant surface affect.
27. Remove the surface DB reload and enforce topic/authority precedence.
28. Pass integration/universal tests across every declared boundary; prove no raw limit exception escapes.

### Stage 7 — Console And Public Telemetry

29. Add exact public context consumption to the cognition graph.
30. Add Character operational posture, User causal relationship, and consumer-labeled style projections.
31. Add redaction, missing-data, failure/degraded, version mismatch, and stale-service tests.
32. Pass repository, contract, web-surface, and fake-service browser tests.

### Stage 8 — Deterministic And Guarded-Database Verification

33. Run focused files, affected regression suites, static greps, compilation, and guarded clean-database persistence/restart/concurrency tests.
34. Record call/repair counts, reads, sizes, dispositions, barrier overhead, healthy/retry latency, and timeout release.
35. Fix only behavior within approved contracts and rerun affected gates.

### Stage 9 — One-At-A-Time Real-LLM Verification

36. Run each controlled emotion-specific A/B case separately and inspect protected traces.
37. Run the offence-by-A to later-user-B sequence, controlled-clock elapsed-fading sequence, apology/repair sequence, and private/group cross-scope sequences through normal entrypoints without direct state writes.
38. Run relationship-emotion isolation, privacy, stale-topic, irrelevant-state, and style-cannot-create-reason negative cases.
39. Parent authors the emotion-scope, controlled, and natural causal reviews.

### Stage 10 — Real-Service Browser Sign-Off

40. Start the real service and authenticated console against the same guarded data.
41. Validate Character, User, Group, Overview cognition graph, loading, empty, degraded, desktop, and narrow layouts.
42. Inspect network payloads and browser console; save screenshots and the browser review.

### Stage 11 — Review And Closeout

43. Run the full affected regression command and final static scans.
44. Start one independent review subagent with the plan, full diff, and evidence.
45. Parent fixes in-scope findings and reruns every affected gate.
46. Record review approval, residual risks, user sign-off, lifecycle status, and archive action.

## Execution Model

- Parent-led native subagent execution is mandatory; the parent owns tests, integration, all verification/evidence, remediation, and lifecycle.
- One production subagent owns only approved production files after Stage 2; one later independent review subagent implements no fixes.
- If subagent capability is unavailable, stop before execution unless the user explicitly authorizes fallback execution.

## Progress Checklist

- [ ] Stage 1 — prerequisite and closure matrices signed. Covers steps 1-5; verify status/HEAD/static inventory, zero unexplained closure rows, and all twenty-one scope/decay rows; retain rebaseline and both matrices; reread, hand off to Stage 2, and sign `<parent/date>`.
- [ ] Stage 2 — focused test contract established and production subagent started. Covers steps 6-10; verify named expected failures; retain commands/failures/subagent brief; reread, hand off to Stage 3, and sign `<parent/date>`.
- [ ] Stage 3 — native versioning, fading, projection, receipt transaction, and exact writer conversion pass. Covers steps 11-14; retain focused output; reread, hand off, and sign `<parent/date>`.
- [ ] Stage 4 — dedicated one-appraisal carry-over, reserved router slot, and direct accepted-task adapter pass. Covers steps 15-19; retain call/update/receipt/failure evidence; reread, hand off, and sign `<parent/date>`.
- [ ] Stage 5 — durable/process ordering and one-snapshot handoff pass. Covers steps 20-23; retain restart/capacity/timeout/read evidence; reread, hand off, and sign `<parent/date>`.
- [ ] Stage 6 — relevance/cognition/surface and universal bounded boundaries pass. Covers steps 24-28; retain exact dispositions, sizes, attempts, contexts, and no-escape proof; reread, hand off, and sign `<parent/date>`.
- [ ] Stage 7 — telemetry and console contracts pass. Covers steps 29-32; verify console unit/web/fake-service tests; retain redacted fixtures/render assertions; reread, hand off to Stage 8, and sign `<parent/date>`.
- [ ] Stage 8 — deterministic, guarded-DB, and performance gates pass. Covers steps 33-35; run every listed gate and retain output/performance review; reread, hand off to Stage 9, and sign `<parent/date>`.
- [ ] Stage 9 — one-at-a-time real-LLM gates pass. Covers steps 36-39; verify offence/emotion/elapsed/repair/cross-scope/negative rubrics; retain protected artifacts and readable reviews; reread, hand off to Stage 10, and sign `<parent/date>`.
- [ ] Stage 10 — authenticated browser gate passes. Covers steps 40-42; verify all target states/layouts/controls, redaction, screenshots, and zero browser errors; retain browser sign-off; reread, hand off to Stage 11, and sign `<parent/date>`.
- [ ] Stage 11 — independent review, remediation, and closeout pass. Covers steps 43-46; verify review approval/reruns/no unresolved findings; retain review/diff/lifecycle/user sign-off; archive only after evidence and sign `<parent/date>`.

## Verification

### Baseline Closure

```powershell
git grep -n -e "last_relationship_insight" -e "global_vibe" -e "character_mood" -e "\"mood\"" main -- src tests
rg -n "last_relationship_insight|global_vibe|character_mood|\bmood\b" src tests
rg -n "CharacterCognitionStateV2|replace_character_cognition_state|get_character_cognition_state" src tests
rg -n "build_interaction_style_context|build_group_engagement_action_context|build_user_engagement_relevance_context" src tests
```

Expected: every functional baseline row appears in the signed matrix; every
post-identity writer/consumer appears in the inventory.

### Static Gates

```powershell
rg -n "user_profile\.get\(\"last_relationship_insight\"|character_mood|state\.get\(\"mood\"|\"mood\"\s*:" src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/relevance src/kazusa_ai_chatbot/nodes src/kazusa_ai_chatbot/cognition_core_v2
rg -n "global_vibe|last_relationship_insight" src/kazusa_ai_chatbot src/control_console
rg -n "build_interaction_style_context" src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py
rg -n "character_operational_state_view|character_operational_context|character_carryover_decision|relationship_operational_context|interaction_style_turn_snapshot|character_operational_receipt" src tests
rg -n "COGNITION_LLM_CHARACTER_CARRYOVER|character_operational_state_task|consolidation_route_decision.v2" src tests
rg -n "semantic_appraisals.*max 4|character_operational_state.*lane_tasks" src tests
rg -n "replace_character_cognition_state" src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py
```

Expected:

- the first and third commands return zero matches; `rg` exit code `1` is expected;
- runtime matches for `global_vibe` and `last_relationship_insight` are zero;
- allowed non-runtime matches are the rejected seed list in
  `character_profile.py`, isolated historical cleanup in
  `db/script_operations.py`, and historical/active plan text;
- canonical route/contract matches occur only in declared owners/consumers;
- the sixth and seventh commands return zero matches, proving one appraisal,
  independent router cardinality, and canonical compare-and-set writers.

### Focused Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_operational_projection.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_emotion_scope_matrix.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_character_carryover.py -q
venv\Scripts\python.exe -m pytest tests/test_character_operational_state_consolidation.py -q
venv\Scripts\python.exe -m pytest tests/test_character_operational_state_ordering.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_universal_bounded_boundaries.py -q
venv\Scripts\python.exe -m pytest tests/test_short_horizon_state_composition_integration.py -q
venv\Scripts\python.exe -m pytest tests/test_post_turn_lifecycle_record.py tests/test_cognition_chain_connector_mapping.py tests/test_reflection_affect_settling.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_state.py tests/test_cognition_core_v2_projection.py tests/test_cognition_core_v2_prompt_budget_continuity.py tests/test_cognition_core_v2_model_retry_continuity.py -q
venv\Scripts\python.exe -m pytest tests/test_frontline_relevance_agent.py tests/test_persona_relevance_agent.py tests/test_relevance_turn_settlement.py -q
venv\Scripts\python.exe -m pytest tests/test_service_background_consolidation.py tests/test_background_work_delivery.py tests/test_interaction_style_images.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_stage_model_routing.py tests/test_control_console_brain_model_routes.py -q
venv\Scripts\python.exe -m pytest tests/test_control_console_repository.py tests/test_control_console_cognition_graph.py tests/test_control_console_web_surface.py -q
```

Expected: all pass. Each new test file is first run before implementation and
its expected failure is recorded.

### Guarded Database And Ordering

Use the test-style guarded live-database invocation, one file at a time.
Required proofs:

- clean character state has a valid native UTC `updated_at`;
- pre-exposure lifecycle v2 claim creates one pending receipt with a live lease;
- transactional commit advances `updated_at` exactly once and terminalizes the
  same receipt; injected failure between writes rolls back both;
- duplicate pending episode returns `in_progress` without a second LLM call;
- duplicate terminal episode returns the stored receipt without another call
  or write;
- startup terminalizes an expired lease as `timed_out`;
- one stale writer reloads/reapplies once and succeeds;
- second conflict fails without overwrite;
- missing transaction support returns `failed/transaction_unavailable`,
  releases the barrier, and preserves the prior state;
- restart loads the latest committed state version;
- every emotion score follows its frozen rate in the pure character effective
  copy, including representative 1/hour, 4/hour, and 12/hour cases;
- read-time fading changes phase/trend/retention without a read write or source
  version change;
- a later commit persists exactly once from the effective base without double
  decay;
- a user-A relationship activation and same-id character activation retain
  separate roots/owners through persistence and restart;
- four durable router tasks and one operational task coexist in one decision;
- accepted-task result skips durable consolidation, invokes the carry-over
  entrypoint at most once, and orders its receipt before incoming chat;
- coordinator capacity overflow records `failed/capacity_exceeded` and cannot
  late-commit;
- timeout releases by 45.5 seconds and exposes degraded health.

### Controlled Real-LLM A/B

Run every parameter ID separately:

```powershell
venv\Scripts\python.exe -m pytest "tests/test_short_horizon_state_composition_live_llm.py::test_offence_emotion_specific_counterfactual[anger_case]" -q -s
venv\Scripts\python.exe -m pytest "tests/test_short_horizon_state_composition_live_llm.py::test_offence_emotion_specific_counterfactual[sadness_case]" -q -s
venv\Scripts\python.exe -m pytest "tests/test_short_horizon_state_composition_live_llm.py::test_offence_emotion_specific_counterfactual[disgust_case]" -q -s
venv\Scripts\python.exe -m pytest "tests/test_short_horizon_state_composition_live_llm.py::test_elapsed_global_affect_counterfactual[case_01]" -q -s
venv\Scripts\python.exe -m pytest "tests/test_short_horizon_state_composition_live_llm.py::test_global_warmth_counterfactual[case_01]" -q -s
venv\Scripts\python.exe -m pytest "tests/test_short_horizon_state_composition_live_llm.py::test_relationship_cause_counterfactual[case_01]" -q -s
venv\Scripts\python.exe -m pytest "tests/test_short_horizon_state_composition_live_llm.py::test_style_scope_counterfactual[case_01]" -q -s
```

Repeat the elapsed, warmth, relationship, and style nodes with `case_02` and
`case_03`, inspecting each before the next run. All non-target input, identity
revision, current message/history, model assignment, generation settings, and
database seed remain fixed within a pair.

Rubric:

- anger, sadness, and disgust offence cases produce their respective native
  activation only when the frozen root guard passes; acceptance requires
  persisted reducer output, not prompt wording or a cause class;
- each offence activation changes a relevant appraisal, goal, stance, or
  expression in its emotion-appropriate direction without accusing the
  non-offending current user;
- controlled elapsed time lowers effective intensity/phase according to the
  frozen rate and changes later expression in at least two of three pairs,
  before any sleep pass;
- global warmth/curiosity changes openness or exploration in at least two of
  three pairs;
- causal relationship context changes relationship/social interpretation or
  goal selection in at least two of three pairs while axes remain identical;
- style changes participation/expression in at least two of three pairs but
  never creates a response reason, topic, fact, or relationship claim;
- no case is accepted solely because prompt text differs.

Create:

```text
test_artifacts/cognition_core_v2_short_horizon_state/controlled_ab_review.md
```

### Natural Cross-Scope Causal Proof

Run each sequence separately:

```powershell
venv\Scripts\python.exe -m pytest tests/test_short_horizon_state_composition_e2e_live_llm.py::test_offence_by_user_a_changes_next_user_b_turn -q -s
venv\Scripts\python.exe -m pytest tests/test_short_horizon_state_composition_e2e_live_llm.py::test_offence_global_affect_fades_before_sleep -q -s
venv\Scripts\python.exe -m pytest tests/test_short_horizon_state_composition_e2e_live_llm.py::test_apology_repairs_user_a_and_global_carryover -q -s
venv\Scripts\python.exe -m pytest tests/test_short_horizon_state_composition_e2e_live_llm.py::test_private_event_changes_next_group_turn -q -s
venv\Scripts\python.exe -m pytest tests/test_short_horizon_state_composition_e2e_live_llm.py::test_group_event_changes_next_private_turn -q -s
venv\Scripts\python.exe -m pytest tests/test_short_horizon_state_composition_e2e_live_llm.py::test_accepted_task_result_changes_next_turn -q -s
```

Each proof must join:

```text
normal chat intake
-> settled episode
-> consolidation route
-> Core V2 state-only carry-over decision
-> native root reduction and emotion derivation
-> committed operational state version
-> predecessor receipt
-> different-scope next turn
-> elapsed-effective state view and consumed context digest
-> relevant appraisal/goal
-> visible dialog
-> matching console projection
```

No direct operational-state write is allowed. Create:

```text
test_artifacts/cognition_core_v2_short_horizon_state/natural_causal_chain_review.md
```

### Negative Live-LLM Gates

Run one case at a time:

- rapid channel-topic pivot: conversation progress/current history follow the
  new topic while slow group style persists and global state introduces no old
  topic;
- irrelevant global activation: response content remains grounded in the
  current event;
- private-source global carry-over: no identity, quote, channel, event
  description, or private fact appears in the other scope;
- relationship-required activation isolation: `love_attachment`, `jealousy`,
  and `loneliness` are never created in character state by ordinary carry-over;
- style-only response pressure: an ungrounded group message remains ignored;
- relationship cause isolation: one user’s causal context never reaches
  another user.

All must pass; no majority threshold applies.

### Performance Gates

Create
`test_artifacts/cognition_core_v2_short_horizon_state/performance_review.md`
with post-identity baseline and final measurements.

Required:

- no new foreground model call;
- no carry-over call when the normal router slot is null or an accepted-task
  result lacks settled consolidatable character-response evidence;
- exactly one router primary call per eligible normal episode, zero JSON-repair
  calls, and at most one primary plus two carry-over replacements;
- at most one user and one group style-image read per logical group turn;
- one character-state read per logical turn after the predecessor barrier and
  no surface state/style reload;
- no surface style-image read;
- no-pending predecessor coordinator overhead below 5 ms at p95 over 1,000
  deterministic iterations;
- no-pending/no-operational-lane end-to-end p95 no more than 10% above the
  post-identity baseline over at least 20 guarded runs;
- healthy initial-attempt routed carry-over p95 at or below 20 seconds over at
  least 20 guarded configured-model runs;
- an immediate next-turn healthy barrier wait p95 at or below 20 seconds over
  the same runs;
- retry/provider-failure routes terminalize and release by 45.5 seconds;
- carry-over static/dynamic/total input is at most
  `8,000/8,000/16,000` characters and raw output at most 8,000 before parse;
- every unchanged stage stays inside its existing cap; over-cap fixtures reach
  the declared typed disposition with zero uncaught exceptions;
- no coordinator registry exceeds 256 pending entries;
- the captured long-context V2 reproduction passes with no call-count, prompt,
  crash, or latency regression outside these thresholds.

### Console Browser Gate

Use the in-app Browser when available; otherwise use project Playwright and
record the reason. Start the authenticated real console at its configured
loopback URL, normally `http://127.0.0.1:8765/`, against the same guarded data.

Verify:

1. Character Operational posture shows every persisted and elapsed-effective native affect row, its emotion id/phase/trend, and whether ordinary fading changed it before sleep.
2. The panel separately shows the exact bounded affect/pressure subset consumed by the latest turn, its run ID, source version, and context digest.
3. Persisted, effective, and consumed states plus version mismatch,
   reduced/replaced/degraded, input/output-limit, and timeout conditions render
   distinctly without exposing rejected raw content.
4. User A Relationship shows offender-specific axes, causal rows, affect, and freshness; user B's panel and payload contain none of A's relationship data.
5. Projection-only cause classes appear beside native emotion/root labels and never replace them.
6. User and Group style panels label relevance, cognition, and surface projections.
7. Private detail and internal IDs are absent from DOM and network payloads.
8. Loading, empty, failed, and stale-brain states are explicit.
9. Desktop and narrow layouts have no clipping, overlap, or unreachable content.
10. Brain model routes shows the required carry-over route and hard ceiling; every affected navigation/control is exercised.
11. Browser console and page errors are zero.

Save screenshots plus `test_artifacts/cognition_core_v2_short_horizon_state/console_browser_signoff.md`.
This gate is blocking.

### Full Regression

After every focused, database, live-LLM, performance, and browser gate passes,
run this exact affected suite:

```powershell
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_operational_projection.py tests/test_cognition_core_v2_emotion_scope_matrix.py tests/test_cognition_core_v2_character_carryover.py tests/test_character_operational_state_consolidation.py tests/test_character_operational_state_ordering.py tests/test_cognition_core_v2_universal_bounded_boundaries.py tests/test_short_horizon_state_composition_integration.py tests/test_cognition_core_v2_state.py tests/test_cognition_core_v2_projection.py tests/test_cognition_core_v2_emotion_lifecycle.py tests/test_cognition_core_v2_prompt_budget_continuity.py tests/test_cognition_core_v2_model_retry_continuity.py tests/test_cognition_core_v2_stage_model_routing.py tests/test_cognition_core_v2_integration.py tests/test_cognition_chain_connector_mapping.py tests/test_reflection_affect_settling.py tests/test_post_turn_lifecycle_record.py tests/test_frontline_relevance_agent.py tests/test_persona_relevance_agent.py tests/test_relevance_turn_settlement.py tests/test_msg_decontextualizer.py tests/test_persona_supervisor2_rag2_integration.py tests/test_rag_projection.py tests/test_dialog_agent.py tests/test_consolidation_lane_router_contract.py tests/test_consolidation_target_routing.py tests/test_consolidation_source_policy.py tests/test_service_background_consolidation.py tests/test_background_work_delivery.py tests/test_interaction_style_images.py tests/test_control_console_brain_model_routes.py tests/test_control_console_repository.py tests/test_control_console_cognition_graph.py tests/test_control_console_web_surface.py -q
```

Stage 1 requires a clean post-identity worktree before this plan starts.
Compile every tracked or untracked Python file changed by this plan:

```powershell
$changedPython = @(
  git diff --name-only --diff-filter=ACMRT HEAD -- '*.py'
  git ls-files --others --exclude-standard -- '*.py'
) | Sort-Object -Unique
if (-not $changedPython) { throw "No changed Python files found" }
foreach ($path in $changedPython) {
  & venv\Scripts\python.exe -m py_compile $path
  if ($LASTEXITCODE -ne 0) { throw "py_compile failed: $path" }
}
```

Expected: all pass with no deselected required case and no unreviewed warning.

## Independent Code Review

The independent reviewer receives the approved plan/prerequisite evidence, full diff/inventory, closure matrix, all test/browser/performance evidence, and controlled/natural causal reviews.

Review architecture/LLM ownership, state/privacy, receipts/transactions/CAS, ordering/timeouts, universal boundaries, budgets, scene precedence, DB reads, error visibility, console redaction, test quality, and plan compliance. Parent fixes only within the approved surface; contract/enum/limit/owner/scope changes stop for approval. Completion permits no blocker or high-severity finding.

## Acceptance Criteria

1. The prerequisite identity-growth plan is completed and signed.
2. Closure has zero unexplained rows; the signed emotion matrix proves eighteen character-root-eligible and three relationship-required ids with unchanged formulas/guards.
3. Character cognition state is the sole persisted short-horizon global state.
4. No runtime mood/vibe/last-insight fallback remains.
5. Router output supports four durable plus one independent operational task without starvation.
6. Carry-over uses one trusted question, zero/one appraisal, native emotion derivation, and no model-authored emotion/cause class.
7. Operational timestamps strictly increase; every writer uses the canonical optimistic owner.
8. Ordinary character elapsed fading uses frozen emotion-specific rates,
   persists once from the effective base without double decay, and composes
   with stronger sleep recovery.
9. Global state retains all active/fading native emotion rows; model contexts
   are bounded branch-relevant selections, causal, source-free, and private-safe.
10. Relationship projection contains axes plus bounded native
    causes/affect/freshness without a new prose field.
11. Relevance, V2 appraisal/goal, and text surface consume only their approved
    projections; current evidence/history/progress remain fact/topic authority.
12. One style snapshot is reused by every ordinary-turn consumer.
13. Receipt claim precedes normal completion/accepted-task dispatch; duplicate/restart/transaction tests prove one update/episode.
14. Healthy commit reaches the next eligible cross-channel turn; failure/timeout exposes degraded last-valid state.
15. Accepted-task result skips durable consolidation and invokes at most one
    directly owned operational carry-over.
16. Every inventoried direct-path guard reduces, replaces, or returns its
    declared typed terminal result; no raw size/limit/security exception
    crashes queue, graph, worker, delivery, persistence, or console.
17. Controlled real-LLM gates distinguish anger, sadness, and disgust and show
    emotion-specific elapsed behavioral change before sleep.
18. Normal-entrypoint offence, non-offender, elapsed-fading, apology/repair,
    and private/group cross-scope sequences pass without direct state writes.
19. Every relationship-emotion isolation, privacy, topic, and style negative passes.
20. Exact call, prompt/output, DB-read, barrier, active-path, and context-budget gates pass.
21. The authenticated browser gate matches full persisted/effective affect,
    scoped relationship data, exact consumed subset, and degraded/limit health.
22. Independent review is approved, all in-scope findings are remediated, and
    the user signs off the behavior, evidence, and web UI.

## Risks

| Risk | Mitigation and proof |
|---|---|
| global posture leaks a private event | source-free native roots plus projection-only cause classes; structural privacy tests and live cross-scope negative proof |
| generic cause classes collapse the twenty-one emotions | classes are post-reducer labels only; scope matrix, native-id assertions, and emotion-specific real-LLM cases |
| relationship emotion bleeds to another user | no relationship context in carry-over; fixed eighteen/three matrix and user-A/user-B isolation proof |
| state exists but model ignores it | branch-specific controlled A/B with appraisal/goal/dialog rubrics |
| global state injects stale topic | no descriptions in global projection; topic-pivot and irrelevant-state live negatives |
| relationship causes become another prose blob | derive bounded rows from native entities/affect; no persisted insight field |
| style starts deciding whether to speak | relevance prompt contract plus style-only ignore negative |
| operational task is starved by four durable tasks | independent nullable router slot and cardinality regression |
| one question produces ambiguous multi-row authority | exactly one appraisal result with trusted id/path/handle validation |
| duplicate/restart creates two updates | durable lifecycle receipt, unique episode index, lease, and atomic state/receipt transaction |
| size/security guard crashes a live path | universal disposition matrix, bounded trace, parameterized no-escape test, and service-level proof |
| post-turn update races next cognition | predecessor coordinator, 45-second deadline, monotonic state version, CAS tests |
| self-cognition overwrites chat effects | all writers use optimistic commit; one additive retry; concurrency test |
| character affect waits until sleep or decays twice | pure elapsed-effective copy, frozen rates, one later persistence timestamp, before-sleep and double-decay tests |
| new context worsens local-model limits | caps, branch routing, captured long-context test, prompt instrumentation |
| style loads increase latency | concurrent one-snapshot load, exact read-count and p95 gates |
| console displays reconstructed rather than consumed state | exact public projection embedded in cognition graph and matched by digest |
| scope overlaps identity-growth work | hard prerequisite, post-identity rebaseline, explicit ownership exclusions |

## Execution Evidence

- Prerequisite/rebaseline/closure and twenty-one-emotion matrices:
- Focused failures, production subagent, state/projection/persistence, Core V2 carry-over, consolidation adapter, ordering/style, and consumer integration:
- Static/compilation, guarded database, performance, controlled and natural real-LLM reviews, and negative gates:
- Browser sign-off, independent review/remediation, lifecycle closeout, and user sign-off:
