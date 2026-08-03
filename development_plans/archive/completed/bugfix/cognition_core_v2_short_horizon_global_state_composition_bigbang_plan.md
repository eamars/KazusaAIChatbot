# Cognition Core V2 Short-Horizon Global State Composition Big-Bang Plan

## Summary

- Goal: close the V2 functional gaps formerly covered by mood, global vibe, and last relationship insight through native emotion-specific character carry-over, ordinary elapsed fading between sleep cycles, causal per-user relationship projection, and one role-based user/group interaction-style composition path.
- Plan class: `high_risk_migration`.
- Status: `completed`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `no-prepost-user-input`, `debug-llm`, `character-test`, `control-console-web-development`, `py-style`, `cjk-safety`, `test-style-and-execution`, and `python-venv`.
- Overall cutover strategy: a forward-only V2 big-bang contract replacement after the character identity-growth plan completes, with no legacy prose-state restoration, compatibility mapper, dual read, dual write, channel-situation store, historical replay, or database backfill.
- Highest-risk areas: preserving the frozen twenty-one-emotion semantics while extending character-scope elapsed evolution; deriving a global consequence without copying user-scoped relationship state; preventing the new operational route from competing with the router's four durable-task slots; keeping global state fresh across normal and accepted-task paths without leaking scoped detail; bounding every reachable input/output/security check without an uncaught pipeline failure; proving behavioral rather than prompt-only impact; preserving current-scene authority; and avoiding latency regression.
- Acceptance criteria: a zero-gap capability matrix and twenty-one-emotion scope matrix are signed; a normal offence can produce separately derived offender-specific relationship injury and privacy-safe character-global affect; ordinary elapsed evolution changes the effective global emotion before sleep; the next eligible cross-channel turn consumes its committed state version without treating another user as the offender; current history and conversation progress remain scene authority; one interaction-style snapshot serves relevance, cognition, and surface; every direct V2-path limit resolves by bounded reduction, owner-local replacement, or a typed terminal disposition; controlled and natural real-LLM evidence demonstrates emotion-specific causal behavior and fading; baseline-parity proof shows persisted global affect changes at least one decision (appraisal/goal/stance) AND visible speech like the main-branch mood/global-vibe path did; and authenticated browser sign-off shows full persisted/effective affect, scoped relationship state, exact consumed context, and bounded/degraded health.
- Execution authority: the user approved this plan on 2026-08-02 with the explicit `Approve. Proceed` implementation command and approved Stage 11 closeout on 2026-08-03; this completed record is archived under `development_plans/archive/completed/bugfix/`.

## Context

Gap (user-confirmed criterion): the main branch persisted `mood`,
`global_vibe`, and `last_relationship_insight` and fed them into L1/L2/L2c2/L3
and surface with explicit instructions to color first reactions, appraisal and
goal judgment, tone (tense/relaxed/defensive/light), and closeness/aggression/
care. Cognition V2 dropped these from the production cognition chain; only
compatibility reads remain (`service.py` `character_mood`/
`last_relationship_insight`, `persona_relevance_agent.py` `character_mood`),
and the persisted native V2 affect is not yet consumed by relevance,
appraisal/goal, or surface. Global character dynamics therefore no longer
influence decisions or speech. This plan restores that influence through
native V2 affect/pressure projections and one state-only carry-over.

Required invariant: one user episode may update that user's relationship state
and separately derive source-free character affect. A later user or channel
may consume elapsed-effective character posture without receiving the earlier
user's facts, identity, or blame. Current message/history/progress remain fact
and topic authority. Implementation path:

```text
prior receipt -> bounded barrier -> one global/user/style load
-> relevance with scene + causal relationship + global posture + style
-> progress remains scene authority -> Core V2 with identity + effective global
+ current-user relationship + style -> surface with same snapshot
-> consolidation target -> Core V2 state-only carry-over
-> native reducer -> one character update -> transactional commit/receipt
-> next eligible private/group turn consumes that version
```

Stage 1 rebaselines the completed identity-growth and P0 context-reconnection
plans in `development_plans/archive/completed/bugfix/`; identity growth stays
durable authority and the shared snapshot replaces only P0's loader.

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

## Anti-Cheat Rules

1. Schema success, prompt differences, mocks, and direct seeds are supporting evidence only. Behavioral acceptance requires the production entrypoint, persisted native reduction, consumed-context digest, relevant appraisal/goal, and visible dialog.
2. Controlled A/B may seed only its declared pre-run variable. After entrypoint invocation, agents/tests may not create or repair operational state, receipts, cognition/context output, or dialog.
3. Natural proofs use a clean guarded database, normal service paths, and only the declared clock seam; direct reducer, writer, carry-over, or console-repository substitution fails the proof.
4. A/B pairs hold identity, message/history, database seed, model route/name/settings, prompt version, and code revision fixed; the review records the sole changed field.
5. Production/prompt code may not contain fixture identifiers, expected dialog/emotion, test-only branches, force-pass flags, hidden fallbacks, or environment bypasses.
6. The production subagent does not edit tests. Stage 2 freezes focused-test hashes. Later parent changes require the original assertion, plan-clause rationale, and retained pre/post output; weakening, deleting, skipping, xfail, or deselection fails the plan.
7. Retain raw successful and failed calls, parse results, state versions, receipts, digests, and dialogs before authoring reviews. Summaries may not rewrite or omit contradictory evidence.
8. Two-of-three applies only where stated. Structural, persistence, privacy, isolation, negative, browser, and regression cases pass individually.
9. Scripts/tests emit raw evidence; the parent authors readable reviews after inspection.
10. `anti_cheat_audit.md` records frozen/final test hashes, direct-write and bypass scans, A/B deltas, raw-evidence hashes, natural trace chains, and exceptions. Passing requires zero unexplained change or prohibited path.
11. Baseline-parity anti-cheat: a behavioral case passes only when the SAME persisted global state can be shown to change both a decision (appraisal/goal/stance) and the visible dialog, and the consumed-context digest contains the rows that produced that change. Prompt-only differences, a static/inserted mood line, hardcoded emotion or expected-dialog literals, direct state seeding before entrypoint invocation, mocked consumption, and surface-only rewrites that leave decisions unchanged are invalid.
12. No affect laundering: production code may not append global affect to dialog or surface without passing through the declared consumer projections and recorded context digest, and may not fabricate or duplicate affect rows to inflate influence. Full global state is never copied into a model prompt.
13. Baseline anchors are mandatory: each controlled and natural review records the main-branch anchors it parallels (prompt/source lines and, where available, the conversation-history example), the exact consumed global-state rows, and the decision+dialog deltas, so the reviewer can verify v2 restores the baseline influence direction without inventing facts, topics, or reasons to speak.
14. Carry-over role anti-cheat: the LLM selects only the declared source-free
    role handles. Production code may validate and map those handles, but may
    not infer actor/target roles from source text, fixture ids, expected
    emotion, or keywords, and may not hardcode every candidate as
    `actor=self`. Missing, unknown, duplicate, or conflicting role assignments
    follow the bounded same-owner replacement path.
15. A `state_rejected` proposal may consume only the existing three-attempt
    cap. It may not be converted to a commit, repaired by direct reducer/state
    writes, or made to pass by weakening native emotion guards, accepting zero
    deltas, or adding a route fallback.
16. Performance samples include every attempted run in stable order. Slow,
    failed, timed-out, or replaced runs may not be discarded from the raw
    ledger, thresholds may not be relaxed, and a route change invalidates the
    prior comparable sample set.

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
19. Produce the anti-cheat audit and final closure manifest defined below;
    completion cannot rely on checklist state or prose claims.
20. Prove baseline-parity behavior: at least one controlled and one natural
    real-LLM chain show persisted global affect changing both a decision
    (appraisal/goal/stance) and visible speech in the same influence direction
    as the main-branch mood/global-vibe path, with baseline anchors and
    consumed-context digests recorded.

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

| Contract | Exact fields and bounds |
|---|---|
| `character_operational_state_view.v1` | `source_updated_at`, `effective_at`, `source_digest`, `view_digest`; `affect` max 21 with existing emotion id/intensity/phase/trend, native root kind, closed cause class, freshness; `pressures` max 8 with native kind/salience/lifecycle, closed cause class, freshness |
| `character_operational_context.v1` | source/effective timestamps, `view_digest`, `context_digest`, `consumer_role=settled_relevance|appraisal branch|goal|surface`; same affect max 3 and pressure max 4 row shapes |

`source_digest` hashes persisted state, `view_digest` the complete redacted
view, and `context_digest` the exact selection, each excluding its own digest.
They are audit fields, not semantic evidence. Consumers select from the one
view and do not reload or reinterpret persistence.

### Relationship Operational Projection

`relationship_operational_context.v1` contains existing qualitative `axes`,
`causal_context` max 2 (`entity_kind`, normalized `semantic_summary` max 160,
salience, lifecycle, freshness), `affect` max 2 (emotion id, intensity, phase,
trend, freshness), relationship freshness, and evidence freshness or `无证据`.

### Consolidation Route Decision

`consolidation_route_decision.v2` has exact required keys `lane_tasks` (existing
closed durable rows, max 4) and nullable `character_operational_state_task`
with normalized reason max 160 and 1-4 unique trusted source keys.

Both top-level keys are required/exact; top-level or durable-task failure keeps the existing typed empty-route result. The operational slot is independently available only for settled non-reflection character responses; its invalid source/cardinality/shape becomes `null` plus a typed error while valid durable tasks continue.
Operational source policy accepts only ref-complete `current_turn_user_message|assistant_final_dialog|internal_thought|episode_trace`; any RAG/reflection key rejects the operational task.
A validated `null` terminalizes `no_change`; malformed routing or a rejected non-null slot terminalizes `failed/route_invalid|source_policy_rejected`, never silent `no_change`.

### Character Carry-Over Decision

| Contract | Exact fields |
|---|---|
| `character_carryover_decision.v1` | exact schema version; `action=no_change|apply`; `reason_code=no_lingering_effect|already_represented|transient_scene_only|unsupported|lingering_character_effect`; `privacy_disposition=source_free|unsafe`; nullable one `SemanticAppraisalResultV2` |
| `character_carryover_result.v2` | exact schema version; decision; nullable one character `StateUpdateV2`; `disposition=no_change|apply|degraded`; `error_code=null|input_limit|output_limit|contract_exhausted|provider_exhausted|privacy_rejected|deadline_exceeded|state_rejected`; attempts 0-3 |

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

`interaction_style_turn_snapshot.v1` contains user and optional group revision,
`active|empty|missing|failed` status, existing sanitized overlays, source-labeled
relevance/cognition/surface projections, and `snapshot_digest`. Private turns
omit group style. Prompt payloads omit document/reflection IDs, platform/user/
channel IDs, and storage timestamps.

### Ordering Receipt

| Contract | Exact fields |
|---|---|
| `character_operational_receipt.v1` | protected episode id; sequence; `pending|no_change|committed|failed|timed_out`; durable; base/committed/registered/completed/lease timestamps; bounded lease owner; attempts 0-3; the declared route/source/capacity/input/output/contract/provider/privacy/deadline/state/transaction/persistence/version error enum |
| `post_turn_lifecycle_record.v2` | existing lifecycle/delivery/action/status fields, embedded receipt, `created_at`, configured-TTL `purge_after` |
| `CharacterOperationalClaimV1` | `claimed|in_progress|terminal` plus receipt |
| `PredecessorTokenV1` | protected episode id, process sequence, registration time |
| `PredecessorBarrierResultV1` | `healthy|degraded`, watermark, awaited/timed-out counts, non-negative wait ms |

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

## Open-Gate Remediation Decision — 2026-08-03

| Gate | Architect decision | Execution boundary |
|---|---|---|
| Stage 8 performance | Retain every approved timing, size, call-count, and sample threshold. | The seven-minute Qwen 27B probe is a failed route candidate, not a reason to remove the gate. Complete the missing guarded samples on one approved configured route; any route change requires a fresh comparable sample set. |
| Stage 9 `state_rejected` | Correct the existing carry-over prompt/contract and native role adapter before changing model route or unrelated cognition. | The retained offence trace reached one valid `apply` decision and failed in `_reduce_apply_decision(...)` before CAS. The current adapter collapses all candidate roles to `actor=self`, while the approved contract already declares `self`, `unspecified_other`, and `group_context`. This is the smallest owning boundary. |
| Stage 10 browser | Use project Playwright because the in-app Browser reported `No browser is available`. | Start this workspace's console on the first free loopback port in `8770..8799`, supply the same guarded Stage 9 route/database settings through process environment, authenticate, and label the result `Playwright fallback validation`. |

Stage 9 remediation converges implementation to the already approved one-
question/one-appraisal contract:

- each `apply` proposition has exact keys `kind`, `semantic_value`,
  `evidence_handles`, `role_assignments`, and `deltas`; action, reason,
  semantic value, and roles have no parser defaults;
- `role_assignments` contains one to three unique exact
  `role|entity_handle` rows, using only
  `actor|experiencer|target|object` and
  `self|unspecified_other|group_context`; the LLM owns the semantic choice and
  deterministic code only binds the closed handles;
- an externally caused offence can therefore select
  `actor=unspecified_other,target=self`; self-caused events retain
  `actor=self`; no source identity or relationship handle enters global state;
- zero deltas are contract-invalid. A structurally valid proposal that fails
  native reduction or changes no native path triggers full same-owner
  replacement with closed error code `state_rejected` until the existing
  three-attempt cap, then returns degraded no-change;
- protected evidence records the parsed proposition, selected role handles,
  native candidate roles, changed paths or rejection class, attempt, and final
  receipt. Public evidence retains only bounded redacted handles/classes.

No route switch is authorized until these focused tests pass and the exact
offence trace is rerun. A later route change is allowed only when the corrected
contract still fails behavior and the replacement route independently passes
Stage 8 without changing caps, prompts, formulas, guards, or call counts.

## Implementation Order

### Stage 1 — Post-Identity Rebaseline And Closure

1. Confirm the identity-growth and P0 context-reconnection prerequisite plans
   are completed and their execution evidence is signed.
2. Require a clean post-identity worktree, then record `git status --short`,
   HEAD, Python version, mandatory-skill reads, and relevant docs/source/tests.
3. Locate every character cognition-state reader/writer and every legacy runtime occurrence with `rg` and `git grep main`.
4. Create and sign
   `test_artifacts/cognition_core_v2_short_horizon_state/rebaseline.md`,
   `baseline_to_v2_capability_closure.md`, and
   `emotion_scope_decay_matrix.md`.
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
9. Run each focused file, record expected failures, and record the SHA-256 hash
   of every focused test file in
   `test_artifacts/cognition_core_v2_short_horizon_state/focused_test_contract.json`.
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

### Stage 8 — Approved Remediation, Deterministic, And Performance Verification

33. Parent adds these exact focused failures to
    `tests/test_cognition_core_v2_character_carryover.py` before production
    correction:
    `test_external_offence_roles_reach_native_reducer`,
    `test_disgust_target_role_is_preserved`,
    `test_exact_carryover_schema_has_no_defaults`,
    `test_zero_delta_requests_replacement`, and
    `test_state_rejected_exhaustion_has_no_state_update`; record the failing
    output and updated focused-test hash in `anti_cheat_audit.md`.
34. In `cognition_core_v2/character_carryover.py`, implement only the exact
    role passthrough, exact-schema validation, non-zero-delta validation, and
    bounded native-rejection replacement defined in the Open-Gate Remediation
    Decision. Change no emotion formula/guard, route fallback, attempt cap,
    state writer, relevance path, dialog surface, or fixture-specific logic.
35. Run the focused carry-over file, the Stage 4 seven-file deterministic
    gate, affected regression suites, static greps, compilation, and guarded
    clean-database persistence/restart/concurrency tests. Record the accepted
    parsed roles, native roles, changed paths, receipts, hashes, and zero direct
    writes.
36. Complete every missing performance sample: call/repair counts, reads,
    prompt sizes, long-context disposition, no-pending E2E p95, healthy routed
    carry-over p95, immediate-next-turn p95, retry release, and timeout release.
    Preserve all raw runs and comparable environment metadata.

### Stage 9 — One-At-A-Time Real-LLM Verification

37. Rerun the three controlled offence cases first, separately, and inspect
    protected role/reduction traces after each; stop on any role, privacy,
    native-guard, or direct-write mismatch.
38. Run the remaining controlled emotion-specific A/B cases separately and inspect protected traces.
39. Run the offence-by-A to later-user-B sequence, controlled-clock elapsed-fading sequence, apology/repair sequence, and private/group cross-scope sequences through normal entrypoints without direct state writes.
40. Run relationship-emotion isolation, privacy, stale-topic, irrelevant-state, and style-cannot-create-reason negative cases.
41. Parent authors the controlled, natural, and negative reviews from retained
    raw evidence and adds pair-delta and evidence-hash rows to the anti-cheat
    audit.

### User-Accepted Stage 9 Mechanism Disposition — 2026-08-03

The user explicitly accepted Stage 9 based on the mechanism rather than the
weak local-model behavior. The bounded contracts, role handling, replacement
and exhaustion paths, native guards, privacy boundaries, persistence/CAS
behavior, ordering, and console mechanisms are the accepted basis. The live
model's lack of committed native affect, failed baseline-parity evidence, and
incomplete negative behavior cases remain recorded as non-passing evidence;
they are not reclassified as behavioral success or emotion-specific quality.
This closes the Stage 9 disposition for this execution without waiving the
final review, manifest, user sign-off, lifecycle, or archive requirements.

### Stage 10 — Real-Service Browser Sign-Off

42. Start the real service and this workspace's authenticated console against
    the same guarded data using the Stage 10 process-environment and port rule
    above; do not reuse another workspace's process or session.
43. Authenticate Playwright with the console's ephemeral operator token or an
    explicitly supplied token, redact the token from artifacts, then validate
    Character, User, Group, Overview cognition graph, loading, empty, degraded,
    desktop, and narrow layouts.
44. Inspect authenticated network payloads, browser console, and page errors;
    save screenshots and the browser review, then terminate only the processes
    and browser profile created for this gate.

### Stage 11 — Review And Closeout

45. Run the full affected regression command, final static scans, and complete
    `anti_cheat_audit.md`.
46. Start one independent review subagent with the plan, full diff, evidence,
    and anti-cheat audit; record its report in
    `test_artifacts/cognition_core_v2_short_horizon_state/independent_code_review.md`.
47. Parent fixes in-scope findings and reruns every affected gate.
48. Complete `final_closure_manifest.md`, present it with the readable behavior
    and browser evidence to the user, record explicit approval in
    `user_signoff.md`, then update lifecycle status and archive the plan.

## Execution Model

- Parent-led native subagent execution is mandatory; the parent owns tests, integration, all verification/evidence, remediation, and lifecycle.
- One production subagent receives the approved plan, frozen test hashes, and Anti-Cheat Rules, owns only approved production files after Stage 2, and cannot edit tests or evidence; one later independent review subagent independently checks the diff, hashes, raw evidence, and audit and implements no fixes.
- If subagent capability is unavailable, stop before execution unless the user explicitly authorizes fallback execution.

## Progress Checklist

- [x] Stage 1 — prerequisite and closure matrices signed. Covers steps 1-5; evidence: `test_artifacts/cognition_core_v2_short_horizon_state/rebaseline.md`, `baseline_to_v2_capability_closure.md`, and `emotion_scope_decay_matrix.md`; main-branch gap anchors recorded from `git grep main` (mood/global-vibe/relationship-insight producers and L1/L2/L2c2/L3/surface consumers) and read-only history probe `_history_probe.json`; parent `/root`, 2026-08-02; handoff to Stage 2 recorded.
- [x] Stage 2 — focused test contract established and production subagent started. Covers steps 6-10; evidence: `test_artifacts/cognition_core_v2_short_horizon_state/focused_test_contract.json` and `production_subagent_brief.md`; frozen-test commit `3be764f911bdd6d8faecee6ff499ed1a9c72f812`; parent `/root`, 2026-08-02; handoff to Stage 3 completed.
- [x] Stage 3 — native versioning, fading, projection, receipt transaction, and exact writer conversion pass. Covers steps 11-14; evidence: `test_artifacts/cognition_core_v2_short_horizon_state/deterministic_verification.md`, `guarded_database_and_ordering.md`, and the focused 44-test result; parent `/root`, 2026-08-03; handoff to Stage 4 completed.
- [x] Stage 4 — dedicated one-appraisal carry-over, reserved router slot, and direct accepted-task adapter pass. Covers steps 15-19; evidence: `test_artifacts/cognition_core_v2_short_horizon_state/deterministic_verification.md`, `full_regression.md`, `anti_cheat_audit.md`, the 24-test canonical lane suite, and retained live receipts; parent `/root`, 2026-08-03; handoff to Stage 5 completed.
- [x] Stage 5 — durable/process ordering and one-snapshot handoff pass. Covers steps 20-23; evidence: `test_artifacts/cognition_core_v2_short_horizon_state/guarded_database_and_ordering.md`, `deterministic_verification.md`, and the ordering, restart, capacity, timeout, database-read, and interaction-style test results; parent `/root`, 2026-08-03; handoff to Stage 6 completed.
- [x] Stage 6 — relevance/cognition/surface and universal bounded boundaries pass. Covers steps 24-28; evidence: `test_artifacts/cognition_core_v2_short_horizon_state/deterministic_verification.md`, `full_regression.md`, and the relevance, cognition, surface, integration, and universal-boundary test results; parent `/root`, 2026-08-03; handoff to Stage 7 completed.
- [x] Stage 7 — telemetry and console contracts pass. Covers steps 29-32; evidence: `test_artifacts/cognition_core_v2_short_horizon_state/deterministic_verification.md`, `full_regression.md`, and the repository, cognition-graph, contract, web-surface, redaction, and fake-service test results; parent `/root`, 2026-08-03; handoff to Stage 8 completed.
- [x] Stage 8 — approved carry-over remediation, deterministic, guarded-DB, and performance disposition accepted. Covers steps 33-36; the post-review role-conflict and semantic-value contract corrections are implemented and the final Stage 11 deterministic evidence is `531 passed, 4 deselected, 1 warning`, with focused carry-over `14 passed` and the final carry-over/operational/service set `54 passed`. The isolated no-pending predecessor probe records 1,000 iterations at 0.0011 ms p95. The user explicitly accepted the local-model p95/latency consequence; slow and incomplete samples remain recorded and are not numeric threshold passes.
- [x] Stage 9 — mechanism accepted with local-model behavioral drift. Covers steps 37-41; the post-fix anger, sadness, disgust, and natural A→B reruns remain retained failure evidence with no committed native affect or consumed global version. The user accepted this behavior based on the bounded mechanism; no behavioral success, baseline-parity success, or emotion-specific quality claim is made.
- [x] Stage 10 — authenticated Playwright fallback gate passes with an accepted guarded-database posture exception. Covers steps 42-44; the in-app Browser failure and project Playwright fallback are recorded. The current-workspace console on `8770` and brain on `8781` passed guarded authentication, route/API/network, redaction, explicit terminal-state, narrow-layout, screenshot, and zero-browser-error checks; the operational-posture data panel is explicitly unavailable because of the accepted reserved-database seed mismatch; evidence: `console_browser_signoff.md` and `playwright/console_browser_validation.summary.json`.
- [x] Stage 11 — independent review, anti-cheat audit, remediation, and closeout pass. Covers steps 45-48; the fresh review findings were remediated through the DeepSeek-owned final contract patch, the final deterministic rerun is `531 passed, 4 deselected, 1 warning` with focused `54 passed`, and the independent reviewer approved the final code/audit/evidence package. The four guarded live-DB smoke cases were rerun individually and are accepted as an external exception for the existing reserved-database seed hash mismatch; evidence and raw output are retained. The user approved the documented behavior, evidence, UI, and accepted exceptions; `user_signoff.md` records the approval, and the completed plan is archived.

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
rg -n "(^|[^A-Za-z0-9_])replace_character_cognition_state\s*\(" src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py
```

Expected:

- the first and third commands return zero matches; `rg` exit code `1` is expected;
- runtime matches for `global_vibe` and `last_relationship_insight` are zero;
- allowed non-runtime matches are the rejected seed list in
  `character_profile.py`, isolated historical cleanup in
  `db/script_operations.py`, and historical/active plan text;
- canonical route/contract matches occur only in declared owners/consumers;
- the sixth and seventh commands return zero matches, proving one appraisal,
  independent router cardinality, and that no direct legacy writer call
  remains; canonical `compare_and_replace_character_cognition_state(...)`
  calls are intentionally outside the direct-writer pattern.

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

Record commands, counts, failures, fixes, and final results in
`test_artifacts/cognition_core_v2_short_horizon_state/deterministic_verification.md`.

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

Record database name guard, transaction capability, commands, counts, state
versions, receipt transitions, restart results, and teardown scope in
`test_artifacts/cognition_core_v2_short_horizon_state/guarded_database_and_ordering.md`.

### Controlled Real-LLM A/B

Run every parameter ID separately:

```powershell
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" "tests/test_short_horizon_state_composition_live_llm.py::test_offence_emotion_specific_counterfactual[anger_case]" -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" "tests/test_short_horizon_state_composition_live_llm.py::test_offence_emotion_specific_counterfactual[sadness_case]" -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" "tests/test_short_horizon_state_composition_live_llm.py::test_offence_emotion_specific_counterfactual[disgust_case]" -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" "tests/test_short_horizon_state_composition_live_llm.py::test_elapsed_global_affect_counterfactual[case_01]" -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" "tests/test_short_horizon_state_composition_live_llm.py::test_global_warmth_counterfactual[case_01]" -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" "tests/test_short_horizon_state_composition_live_llm.py::test_relationship_cause_counterfactual[case_01]" -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" "tests/test_short_horizon_state_composition_live_llm.py::test_style_scope_counterfactual[case_01]" -q -s
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
- every accepted behavioral case shows the change in BOTH the decision
  (appraisal/goal/stance) and the visible dialog, with the consumed
  global-state rows and context digest recorded; a case that changes only
  prompt wording, only persisted state, or only surface wording fails;
- baseline parity: each accepted case records its main-branch anchor (the
  exact `git grep main` prompt/source line or conversation-history example it
  parallels) and the v2 influence direction must match the baseline's
  mood/global-vibe role (coloring appraisal/goal/tone/closeness without
  inventing facts, topics, or reasons to speak);
- no case is accepted solely because prompt text differs.

Create:

```text
test_artifacts/cognition_core_v2_short_horizon_state/controlled_ab_review.md
```

### Natural Cross-Scope Causal Proof

Run each sequence separately:

```powershell
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" tests/test_short_horizon_state_composition_e2e_live_llm.py::test_offence_by_user_a_changes_next_user_b_turn -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" tests/test_short_horizon_state_composition_e2e_live_llm.py::test_offence_global_affect_fades_before_sleep -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" tests/test_short_horizon_state_composition_e2e_live_llm.py::test_apology_repairs_user_a_and_global_carryover -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" tests/test_short_horizon_state_composition_e2e_live_llm.py::test_private_event_changes_next_group_turn -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" tests/test_short_horizon_state_composition_e2e_live_llm.py::test_group_event_changes_next_private_turn -q -s
venv\Scripts\python.exe -m pytest -m "live_llm and live_db" tests/test_short_horizon_state_composition_e2e_live_llm.py::test_accepted_task_result_changes_next_turn -q -s
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

Every chain must also record its main-branch baseline anchor and the exact
consumed global-state rows behind the decision and the dialog, proving the
dialog change is attributable to persisted global affect, not to prompt text
or a direct write. No direct operational-state write is allowed. Create:

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

Create
`test_artifacts/cognition_core_v2_short_horizon_state/negative_live_llm_review.md`
from the retained raw inputs, outputs, traces, state, and dialog.

### Performance Gates

Create
`test_artifacts/cognition_core_v2_short_horizon_state/performance_review.md`
with post-identity baseline and final measurements.

Every threshold below remains binding for an unwaived product-performance
claim. The failed seven-minute Qwen probe and the incomplete prior sample set
remain recorded as failures/incomplete evidence.

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

### User-Accepted Local-Model Performance Exception — 2026-08-03

The user explicitly accepted the observed p95/latency consequence of running
the configured local model and directed closure of the performance gate. The
recorded slow and incomplete samples remain preserved as evidence; they are
not reclassified as numeric threshold passes. This exception closes the
Stage 8 performance disposition for this execution and does not waive the
live behavioral, privacy, or final closeout gates.

### User-Accepted Stage 11 Guarded-Database Exception — 2026-08-03

The user explicitly accepted the reserved `_test_kazusa_live_llm` seed-integrity
issue encountered during the four individual guarded live-DB smoke reruns.
MongoDB was reachable, but the existing `character_state:{'_id': 'global'}`
document did not match the checked-in fixture hash, so each case failed before
cognition execution. The raw output remains retained in
`live_db_stage11_final_contract.log`; this is an external-state exception, not
a deterministic pass or a local-model behavior result. No destructive reset or
reseed was performed. This exception permits Stage 11 closeout to continue
with the live-DB gate recorded as `ACCEPTED EXCEPTION`, subject to independent
review and final user approval. The same accepted database state leaves the
browser Character operational-posture panel in an explicit terminal
`unavailable` state; the authenticated shell, API, redaction, layout, and
terminal-state mechanics remain evidenced, while persisted/effective posture
data availability is not claimed as a pass.

### Console Browser Gate

The in-app Browser failure is already recorded, so use project Playwright and
identify the evidence as fallback validation. Start only this workspace's
authenticated real console on the first free port in `8770..8799` at
`http://127.0.0.1:<port>/`, with the same guarded Stage 9 database and carry-
over route values supplied as process-environment overrides. Record the exact
workspace, port, URL, brain URL, database name, route/model, and session state;
redact credentials and the operator token. Evidence from the other workspace
listeners on 8765/8766 is invalid.

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

### Anti-Cheat Audit

Create and pass
`test_artifacts/cognition_core_v2_short_horizon_state/anti_cheat_audit.md`
before the full regression and independent review. The audit must contain:

- the Stage 1 HEAD and Stage 2 focused-test file hashes;
- final hashes and diffs for every focused test, with a plan-clause rationale
  and retained pre/post output for each parent-authored change;
- `git diff --check` and a changed-file inventory proving the implementation
  stayed inside `Change Surface`;
- source and test scans for skip, xfail, deselection, force-pass/test-only
  branches, fixture identifiers, expected dialog/emotion literals, direct
  operational-state writes, and direct receipt/context/dialog construction;
- for every controlled pair, the raw input/config hashes and a field-level
  delta showing exactly one declared independent variable;
- for every natural proof, correlated entrypoint, episode, receipt, state
  version, context digest, appraisal/goal, dialog, and console identifiers;
- SHA-256 hashes and paths for raw evidence used by each readable review;
- every warning, retry, replacement, failed call, excluded case, and manual
  intervention with its disposition.

Passing requires zero unexplained test-contract change, zero skipped/xfail/
deselected required case, zero prohibited production or direct-write path,
complete single-variable A/B deltas, complete natural trace chains, and exact
raw-evidence hashes. A missing row or artifact fails this gate. The independent
reviewer rechecks the audit against source, test diff, and raw evidence rather
than accepting the parent summary.

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

Expected: all pass with no deselected required case and no unreviewed warning; record the exact command, counts, warnings/dispositions, and output path in `test_artifacts/cognition_core_v2_short_horizon_state/full_regression.md`.

## Independent Code Review

The independent reviewer receives the approved plan/prerequisite evidence, full diff/inventory, closure matrix, all test/browser/performance evidence, and controlled/natural causal reviews.

Review architecture/LLM ownership, state/privacy, receipts/transactions/CAS, ordering/timeouts, universal boundaries, budgets, scene precedence, DB reads, error visibility, console redaction, test quality, anti-cheat evidence, and plan compliance. Parent fixes only within the approved surface; contract/enum/limit/owner/scope changes stop for approval. Completion permits no unresolved blocker/high finding and no unresolved medium finding involving correctness, behavior, privacy, persistence, ordering, performance, or evidence integrity. Low-severity residuals must be recorded in the closure manifest and explicitly accepted by the user.

## Final Sign-Off Contract

Final sign-off is an evidence review, not a checklist or test-count decision. The parent presents every artifact below from `test_artifacts/cognition_core_v2_short_horizon_state/`:

| Artifact | Required proof |
|---|---|
| `rebaseline.md` | prerequisite sign-offs, clean Stage 1 HEAD, environment/tool versions, source/test inventory |
| `baseline_to_v2_capability_closure.md` | every baseline producer/consumer row has one evidenced disposition; zero unexplained rows |
| `emotion_scope_decay_matrix.md` | all twenty-one ids, eighteen/three ownership split, roots, formulas, guards, rates, dual-scope isolation |
| `focused_test_contract.json` | Stage 2 paths and SHA-256 hashes before production implementation |
| `deterministic_verification.md` | red/green focused commands, static gates, compilation, integration and regression counts |
| `guarded_database_and_ordering.md` | database guard, receipts, transaction/CAS, idempotency, restart, fading, ordering and teardown proof |
| `performance_review.md` | baseline/final environment, samples, raw measurements, thresholds and dispositions |
| `controlled_ab_review.md` | raw-evidence links, single-variable pair deltas, native state and behavioral rubric results |
| `natural_causal_chain_review.md` | all six production-entrypoint chains from episode through persisted state to later dialog/console |
| `baseline_parity_behavior_review.md` | main-branch anchors (prompt/source lines plus conversation-history examples), v2 decision+dialog deltas, consumed-context digests, and a pass/fail verdict for the baseline-parity gate |
| `negative_live_llm_review.md` | every privacy, isolation, topic and style negative, all passing individually |
| `anti_cheat_audit.md` | frozen-test comparison, bypass/direct-write scans, pair controls, trace provenance and raw hashes |
| `console_browser_signoff.md` plus screenshots | all eleven browser checks on natural-proof data and zero console/page errors |
| `full_regression.md` | exact final command, selected/passed/skipped/warning counts and warning dispositions |
| `independent_code_review.md` | reviewer identity, diff baseline, findings, fixes, reruns, residuals and approval status |
| `final_closure_manifest.md` | one-to-one acceptance-criterion index and final lifecycle decision |
| `user_signoff.md` | exact user approval reference for behavior, evidence, UI and accepted low residuals |

`final_closure_manifest.md` must contain:

- plan path/status, Stage 1 and final HEADs, changed-file inventory, and every stage signature/date;
- model route/name/settings, prompt/code revision, guarded database, clock mode, host/runtime, and performance sample method;
- exactly one Acceptance Criterion 1-23 row with evidence path/hash, command or case id, observed result, pass/fail, reviewer, and date;
- every warning, replacement, failed attempt, exclusion, intervention, and residual risk with owner, relevance, disposition, and review status;
- independent approval, user-sign-off reference, lifecycle update, and archive destination.

Final passing rules:

The dated user-authorized Stage 8 performance disposition and Stage 9
mechanism disposition above are explicit closure-contract exceptions for this
execution. They permit Acceptance Criteria 14, 17, 18, 19, 20, and 23 to be
recorded as `ACCEPTED EXCEPTION` with their non-passing raw evidence retained;
they do not convert those observations into `PASS`, waive independent review,
or waive the user's final approval of the manifest, behavior evidence, and
browser evidence. The dated Stage 11 guarded-database disposition likewise
permits the four seed-blocked smoke cases to be recorded as an accepted
external-state exception with raw output retained; it is not a test pass.
For Acceptance Criterion 21, the authenticated browser mechanics may remain
directly evidenced, while the persisted/effective operational-posture data
availability is recorded as the same accepted external-state exception.
All other criteria remain subject to the rules below.

1. Every named artifact exists, is non-empty, and resolves to its cited raw hash; blanks, placeholders, missing rows, and prose-only evidence fail.
2. All eleven stages and twenty-three acceptance criteria pass; unchecked evidence links or missing signatures invalidate a checked box.
3. Every required case runs. Skip, xfail, deselection, deletion, or weakened coverage fails. A warning is reviewed only when source, owner, relevance, disposition, and reviewer agreement are recorded.
4. Every controlled, natural, negative, database, performance, browser, anti-cheat, and regression threshold passes; direct post-entrypoint mutation, prompt-only difference, mocked consumption, or rewritten raw output is invalid.
5. Baseline/final performance uses comparable recorded model/settings, host/runtime, database mode, warm-up, samples, and input; material mismatch requires rerun.
6. Independent review approves the exact final diff/audit after remediation and reruns; the severity rule above applies.
7. The baseline-parity gate passes: `baseline_parity_behavior_review.md` proves at least one controlled and one natural case where persisted global affect changed both a decision (appraisal/goal/stance) and visible speech, with baseline anchors and consumed-context digests; prompt-only, static-mood-line, hardcoded-emotion, direct-seed, and surface-only changes are invalid.
8. The user reviews the manifest, behavior reviews (including the baseline-parity review), low residuals, and browser evidence, then explicitly approves behavior, evidence, and UI. Earlier commands and silence are not sign-off; rejection reopens the earliest affected stage.
9. Only after rules 1-8 pass may the parent mark `completed`, update `development_plans/README.md`, and move the closed record to `development_plans/archive/completed/bugfix/`.

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
23. Baseline-parity behavior: controlled and natural real-LLM evidence proves
    persisted global affect changes at least one decision (appraisal/goal/
    stance) AND visible speech in the same influence direction as the main-
    branch mood/global-vibe path, with baseline anchors and consumed-context
    digests; prompt-only, static mood lines, hardcoded emotion/dialog, and
    surface-only rewrites are invalid.

## Risks

| Risk | Mitigation and proof |
|---|---|
| privacy or cross-user bleed | source-free global roots, scoped relationship projection, eighteen/three matrix, structural and live isolation negatives |
| emotion/topic/style authority drift | native ids remain authoritative; current scene retains topic/reason authority; controlled and negative live gates |
| duplicate, lost, or racing updates | durable receipt, transaction, monotonic CAS, predecessor deadline, restart/concurrency tests |
| ambiguous or unsafe model output | one trusted appraisal, exact validation, bounded replacement, typed failure and no-escape tests |
| double decay or sleep-only affect | pure elapsed-effective copy, frozen rates, once-only persistence and before-sleep proof |
| latency/context regression | fixed caps, branch selection, one snapshot, read/call counts and p95 performance gates |
| reconstructed console truth | exact consumed public projection and digest matched through browser/network evidence |
| shortcut or scope drift | frozen tests, anti-cheat audit, independent diff/raw-evidence review and prerequisite rebaseline |
| fake or prompt-only global-affect influence | baseline-parity behavior gate, decision+dialog deltas, consumed-context digests, baseline anchors, and anti-cheat rules 11-13 |

## Execution Evidence

Execution appended dated command, count, hash, result, reviewer, and stage-signature rows only to the exact artifacts in `Final Sign-Off Contract`. Completion resolved every Acceptance Criterion 1-23 row to those artifacts, preserved the accepted non-passing behavior and external-state exceptions, recorded approval in `user_signoff.md`, and archived the completed plan.
