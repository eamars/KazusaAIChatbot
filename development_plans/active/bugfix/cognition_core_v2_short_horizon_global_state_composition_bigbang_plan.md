# Cognition Core V2 Short-Horizon Global State Composition Big-Bang Plan

## Summary

- Goal: close the V2 functional gaps formerly covered by mood, global vibe, and last relationship insight through native emotion-specific character carry-over, ordinary elapsed fading between sleep cycles, causal per-user relationship projection, and one role-based user/group interaction-style composition path.
- Plan class: `high_risk_migration`.
- Status: `draft`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `no-prepost-user-input`, `debug-llm`, `character-test`, `control-console-web-development`, `py-style`, `cjk-safety`, `test-style-and-execution`, and `python-venv`.
- Overall cutover strategy: a forward-only V2 big-bang contract replacement after the character identity-growth plan completes, with no legacy prose-state restoration, compatibility mapper, dual read, dual write, channel-situation store, historical replay, or database backfill.
- Highest-risk areas: preserving the frozen twenty-one-emotion semantics while extending character-scope elapsed evolution; deriving a global consequence without copying user-scoped relationship state; keeping global operational state fresh across private/group boundaries without leaking scoped detail; proving that state changes cognition rather than merely appearing in a prompt; preserving current-scene authority; ordering post-turn global updates before the next eligible turn; avoiding prompt/latency regression; and showing full source/effective affect plus the actual consumed subset in the console.
- Acceptance criteria: a zero-gap capability matrix and twenty-one-emotion scope matrix are signed; a normal offence can produce separately derived offender-specific relationship injury and privacy-safe character-global affect; ordinary elapsed evolution changes the effective global emotion before sleep; the next eligible cross-channel turn consumes its committed state version without treating another user as the offender; current history and conversation progress remain scene authority; one interaction-style snapshot serves relevance, cognition, and surface; controlled and natural real-LLM evidence demonstrates emotion-specific causal behavior and fading; and authenticated browser sign-off shows full persisted/effective affect, scoped relationship state, and the exact latest consumed subset.
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
| user/channel adaptation | user and group-channel interaction-style images already exist | production loads them only at the final text surface; settled relevance and V2 goal judgment do not receive one shared snapshot |
| operator visibility | the console exposes native state tables | it does not show the effective global projection consumed by a turn or the causal relationship projection |

The target is functional supersession, not restoration of the strings
`mood`, `global_vibe`, or `last_relationship_insight`.

The concrete behavioral requirement is:

```text
user A offends the character
  -> user-A cognition retains relationship injury and A-specific consequences
  -> a separate character-scope carry-over episode derives native global affect
  -> a later turn with user B composes that elapsed-effective global affect
     with B's own relationship state and the current scene
  -> the response may remain angry, sad, disgusted, fearful, surprised, or
     otherwise affected for an emotion-specific duration without blaming B
```

This is not a user-affect copy. The foreground user episode and the post-turn
character carry-over episode each retain one mutable scope and one
`StateUpdateV2`. They represent different consequences of the same settled
interaction: person-specific relationship continuity and source-free
character-wide emotional carry-over.

The current dependency chain is:

```text
settled message/history -> relevance with blank legacy state
-> conversation progress as scene owner -> Core V2 with user state plus only
character drives/standards/meaning -> L3 independent style reload
-> post-turn without character update -> console stored state only
```

The target dependency chain is:

```text
prior receipt -> bounded barrier -> one global/user/style load
-> relevance with scene + causal relationship + global posture + style
-> conversation progress remains scene authority
-> Core V2 with latest identity + effective global + current-user relationship
+ stage style -> surface with selected stance + same snapshot
-> consolidation target -> Core V2 state-only carry-over
-> native reducer -> one character StateUpdateV2 -> optimistic commit/receipt
-> next eligible private/group turn consumes the committed version
```

This plan is a follow-on to
`development_plans/active/bugfix/cognition_core_v2_character_identity_growth_bigbang_plan.md`.
That plan owns durable identity, self-image, growth, latest identity revision,
and Character-page identity lineage. This plan owns transient operational
state and scoped runtime composition. Execution is blocked until the identity
plan is `completed`, its execution evidence is signed, and the post-identity
branch has been rebaselined.

## Mandatory Skills

- `development-plan`: approval, execution, review, lifecycle updates, evidence, and sign-off.
- `local-llm-architecture`: prompt ownership, projection budgets, routing, latency, and local-model reliability.
- `no-prepost-user-input`: the Core V2 carry-over stage owns semantic persistence judgment; deterministic code cannot infer it from words or rewrite its decision.
- `debug-llm`: controlled prompt comparisons, live calls, protected raw artifacts, and parent-authored readable reviews.
- `character-test`: adaptive multi-turn private/group tests through normal service entrypoints with trace and database inspection.
- `control-console-web-development`: console contracts, browser validation, cache checks, screenshots, and authenticated visual sign-off.
- `py-style`, followed by `cjk-safety` where applicable: all Python production and test edits.
- `test-style-and-execution`: focused, integration, real-LLM, guarded-database, and browser test changes and runs.
- `python-venv`: environment verification or dependency work; all Python commands use the project virtual environment.

## Mandatory Rules

1. Production changes remain blocked while this plan is `draft`.
2. Execution remains blocked until the identity-growth plan is completed and signed off; reread its final target contracts before touching this plan.
3. Rebaseline every file, symbol, test, and UI panel against the post-identity HEAD. A material contract conflict stops execution and requires a plan update.
4. Use `venv\Scripts\python.exe`, `apply_patch`, and repo-relative quoted paths; check `git status --short`, root/subsystem docs, source, and tests before production edits; do not read `.env`.
5. `CharacterCognitionStateV2` in the singleton operational character document is the sole persisted short-horizon global-state authority.
6. Create no persisted `mood`, `global_vibe`, relationship-insight prose field, second character-state document, channel-situation document, or shadow state.
7. Latest identity is durable character authority; global operational state is transient posture; per-user cognition is relational authority; conversation progress is fast channel-scene authority; interaction-style images are learned expression/participation guidance.
8. Current message, reply/media evidence, bounded history, and conversation progress own current facts and topic. Global state and style cannot introduce a topic, event, promise, relationship fact, or reason to speak.
9. Private and group interactions may affect one global operational state only through a separately derived character-scope V2 update containing source-free native causal roots and native affect; no user activation row, relationship row, or relationship root is copied.
10. Cross-scope model context and public console payloads contain no user/channel identifier, quote, raw message, raw evidence ref, entity handle, event description, memory text, or private fact from global state.
11. Native relationship causal context stays with the current user. It never becomes global state or group-channel style.
12. The consolidation LLM owns target routing only; its target adapter may orchestrate idempotency and commit. A Core V2 state-only carry-over stage owns `no_change` versus `apply`, semantic appraisal, and whether an interaction has a lingering character-wide effect. Deterministic code owns parsing, enums, schema, scope binding, source provenance, privacy shape, native reduction, caps, persistence, timeout, and telemetry.
13. Every raw operational-state LLM response uses `kazusa_ai_chatbot.utils.parse_llm_json_output(...)`; invalid keys, types, enums, or structure trigger stage-owned full replacement, capped at three attempts.
14. The new lane adds no foreground LLM call. Its Core V2 state-only stage runs only after the existing consolidation router selects it for a settled, consolidatable episode; it does not run the full response, goal, action, or surface DAG.
15. One accepted source episode can commit at most one separately scoped global operational update. The source episode ID is the idempotency root, and the carry-over stage emits either no update or exactly one character-scoped `StateUpdateV2`.
16. Every character cognition-state writer uses the canonical optimistic commit owner with the prior `updated_at` as its version token. A stale base may reload and reapply the same validated additive proposal once; a second conflict fails explicitly.
17. A predecessor update registered before a turn becomes eligible either commits, records `no_change`, or reaches a typed failure/timeout before that turn consumes global state.
18. Healthy predecessor completion is visible to the next private or group turn. Failure releases the turn with the last valid committed state plus an explicit degraded receipt; stale consumption is never silent.
19. Read-time character elapsed fading is an intentional extension of the frozen Stage 2 lifecycle clause that previously restricted character evolution to sleep. It changes the effective projection without writing on every read; the next semantic write or sleep recovery persists from the evolved base.
20. Preserve all twenty-one emotion definitions, required-root guards, adjacent-emotion distinctions, begin/sustain/inactive thresholds, per-emotion decay rates, phase/trend rules, and retention thresholds. Sleep recovery remains the stronger character recovery path. Ordinary elapsed fading cannot invent relief, change identity, erase unresolved pressure below existing floors, or reinterpret causes.
21. Emotion globality is activation-instance ownership, not an emotion-id classification. The carry-over lane may derive the eighteen emotions whose frozen guards can be satisfied from character-owned roots. `love_attachment`, `jealousy`, and `loneliness` remain user-scoped on this lane because their guards require relationship state; any source-free global consequence must be independently supported as another native character-rooted emotion.
22. Closed cause classes are deterministic privacy-safe projection and UI labels only. They cannot be LLM appraisal inputs, persisted emotion authority, required update-decision fields, or substitutes for native root kinds and axes.
23. Load user and group interaction-style documents once per logical turn, in parallel where both apply, and reuse one immutable snapshot for relevance, cognition, surface, telemetry, and console comparison.
24. Style guidance remains labeled by source and consumer. It affects participation or expression only within its role and cannot override identity, boundaries, current evidence, relationship state, or selected semantic stance.
25. Persist and expose all active/fading native affect rows up to the frozen twenty-one-row state bound. Consumer-specific prompt selections remain bounded and branch-relevant; a top-N prompt cap cannot define state truth or console truth.
26. Keep all prompt projections bounded. Do not raise an existing V2 context cap to make the new context fit.
27. The console consumes redacted public projections. Raw model prompts/outputs remain in protected traces.
28. Controlled state-seeding proof and natural forward-only service proof are separate gates. Passing either one never substitutes for the other.
29. The natural proof uses a clean guarded database and normal chat/background paths. It cannot directly edit character operational state, relationship state, interaction-style state, receipts, or cognition output; a controlled clock advance is allowed to prove elapsed evolution.
30. The historical/current Asuna database is outside blocking acceptance. Any future read-only smoke against it requires a separate explicit user instruction.
31. Run live LLM cases one at a time, inspect each result, and create human-readable reviews from real inputs, outputs, traces, state, and dialog.
32. Browser sign-off is a blocking acceptance gate and uses real persisted data from the natural live-LLM proof.
33. After automatic context compaction, reread this complete plan before continuing.
34. After each major checklist stage is signed, reread this complete plan before starting the next stage.
35. Complete independent code review, parent remediation, verification reruns, and lifecycle evidence before merge or completion.

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
11. Load one user/group interaction-style snapshot before settled relevance and project it by consumer role.
12. Carry the same state/style snapshot through Cognition Core V2 and L3 instead of reloading at the surface.
13. Wire branch-relevant global affect/pressure only into relevant appraisal/goal branches and expression; wire relationship causes into relationship/social relevance and appraisal.
14. Preserve projected history and conversation progress as fast scene/topic authority, with explicit topic-pivot regression coverage.
15. Expose full persisted/effective global affect, bounded pressures, latest consumed subset, version timestamp/digests, predecessor health, causal relationship context, and consumer-specific style projections in the control console.
16. Prove offence-driven dual consequences, cross-user carry-over, relationship isolation, and emotion-specific elapsed fading with controlled A/B and normal-entrypoint real-LLM sequences.
17. Measure LLM call counts, DB reads, prompt sizes, barrier overhead, routed update latency, and no-pending-turn latency against the post-identity baseline.

## Deferred

- Durable identity, self-image, growth, identity revisioning, and Character-page identity lineage remain owned by the prerequisite identity-growth plan.
- Current-message projection, reply/media normalization, conversation-history policy, conversation-progress algorithms, RAG ranking, memory evolution, residue, and reflection algorithms remain unchanged.
- No new short-lived channel topic, channel mood, scene snapshot, or separate residue lane is created; native character cognition state is the only global carry-over authority.
- User relationship axes and their foreground V2 reducers remain unchanged; this plan changes their projection and consumers.
- Interaction-style producers, reflection cadence, persisted overlay schema, and user/group ownership remain unchanged.
- Adapters, delivery, scheduler permissions, autonomous contact, action authorization, and tool execution remain unchanged.
- No historical database migration, replay, backfill, baseline-field import, or current Asuna repair is performed.
- No operator state editor, mood slider, manual relationship-insight writer, state rollback UI, feature flag, compatibility mode, or alternate V1 path is added.

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
| Update producer | post-turn routed Core V2 | consolidation selects the target; one state-only Core V2 carry-over stage produces zero or one character update; no foreground call |
| Ordering | bounded predecessor | register before response exposure, release on commit/no-change/typed failure |
| Persistence | optimistic big-bang | one canonical compare-and-set writer and episode idempotency |
| Prompts | bounded big-bang | replace legacy relevance keys and update all V2 consumers together |
| Console | public projection | show native source/effective/consumed state and no legacy labels |
| Database | clean post-identity target | preserve the native V2 state shape; no historical migration |
| Tests/docs | big-bang | replace dead assumptions and sign the complete dependency chain |

Cutover enforcement:

- Update caller, callee, schemas, prompts, persistence owner, tests, console, and docs in one execution.
- Keep no fallback reads from `mood`, `global_vibe`, or `last_relationship_insight`.
- Keep the rejected legacy seed names in `character_profile.py` only as explicit fail-closed validation.
- Keep historical cleanup/reporting references in `db/script_operations.py` only where they remain isolated from runtime.
- A change to any cutover row requires a plan update and user approval.

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

One interaction may therefore create two separately owned effects:

```text
foreground user-scope cognition
  -> current user's relationship axes/entities/affect

post-turn character-scope carry-over cognition
  -> source-free character entities/affect
```

Each operation has one mutable scope and one update. A repeated emotion id in both scopes is valid because its roots and owner differ. No activation, relationship entity, or evidence row is copied between scopes.

When contexts appear to conflict:

1. Current observed evidence wins factual and topic questions.
2. Latest identity and standards win character-value and boundary questions.
3. Current-user state wins relationship interpretation for that user.
4. Conversation progress wins fast channel phase and open-loop questions.
5. Global operational state colors appraisal, goal pressure, and expression without inventing content.
6. Group style governs shared participation/pacing; user style governs person-directed interaction; both remain subordinate to the first five owners.

### Baseline-To-V2 Capability Closure Gate

Create
`test_artifacts/cognition_core_v2_short_horizon_state/baseline_to_v2_capability_closure.md`
before production deletion or contract cutover. It must contain one row for
every baseline occurrence and one of:

- `superseded`: exact V2 producer, state, projection, consumer, and proof;
- `intentionally_removed`: approved reason and negative proof;
- `not_applicable`: exact evidence that the occurrence never owned runtime behavior.

Required capability rows:

1. global affect production after an interaction;
2. global atmosphere/posture production;
3. persistence and restart continuity;
4. ordinary elapsed fading and sleep recovery;
5. relationship-cause production and persistence ownership;
6. settled relevance consumption;
7. semantic-appraisal consumption;
8. goal/stance selection consumption;
9. text-surface expression;
10. self-cognition/accepted-task interaction with character state;
11. user/group interaction-style loading and composition;
12. current topic and scene precedence;
13. cross-scope privacy;
14. immediate-next-turn ordering;
15. operator observability and web UI;
16. all twenty-one emotions across user scope, character carry-over eligibility,
    decay, and relationship-required exclusions;
17. offence-driven dual consequence and non-offender isolation;
18. call, prompt, DB-read, and latency cost.

The matrix must have zero unexplained rows and parent sign-off before Stage 7.

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

The existing consolidation router gains
`character_operational_state` as a targetable lane. The router sees the settled
episode projection and decides whether to invoke the lane. The lane calls the
Core V2-owned `run_character_carryover_cognition(...)` state-only entrypoint.
That stage answers one semantic question: whether the settled interaction has
a source-free lingering effect on the character and, if so, which existing
native character roots/axes it supports. It returns `no_change` or one bounded
character-scope state update and does not run response goals, action planning,
surface planning, or dialog.

Allowed operational effects:

- create/reinforce one abstract current event;
- create/reinforce one threat, goal, or knowledge gap when supported;
- include up to four existing native semantic deltas inside accepted
  `SemanticAppraisalResultV2` rows for character drive pressure or meaning axes;
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

The Core V2 carry-over stage selects from prompt-local `self`,
`unspecified_other`, and `group_context` roles plus native event, threat, goal,
and knowledge-gap candidates. Accepted semantic appraisals retain the existing
typed native axes needed by the emotion formulas. Deterministic mapping gives
new entities a generic root-kind description solely to satisfy the native
entity shape; descriptions and projection-only cause classes never determine
emotion. Model-authored explanations and delta reasons are not persisted.
Opaque episode provenance remains internal for idempotency, while user/channel
IDs and relationship handles are invalid in a character-operational proposal.
The persisted state is source-free before later projection.

### Elapsed Fading

`apply_character_elapsed_decay(...)` uses elapsed UTC time and existing emotion
decay rates. It:

- decays character goal/threat/event/gap salience at the existing user-state
  rate of four points per hour;
- preserves the current unresolved-pressure floor;
- decays affect using each emotion definition;
- updates affect lifecycle and removes activations at the existing retention
  threshold;
- leaves identity, standards, drive importance, drive pressure, and meaning
  values unchanged;
- computes once from the persisted state's version timestamp to `effective_at`,
  retains that version timestamp on the pure copy, and performs no write;
- lets a later semantic commit or sleep pass persist from the effective copy
  with one new strictly increasing version timestamp, preventing double decay.

Sleep recovery remains the only stronger recovery pass for drive pressure and
residual pressure. This deliberately replaces the prior character-scope
sleep-only evolution rule while preserving the frozen emotion rates and
lifecycle. An activation can move from active to fading and disappear between
sleeps. Ongoing causal entities retain existing unresolved salience floors;
completed or repaired global entities follow native terminalization. Durable
injury toward the offender remains only in that user's relationship state.

### Relationship Causal Projection

`project_relationship_context(...)` accepts the complete current-user cognition
state and `effective_at`. It:

- projects all existing relationship axes to qualitative bands;
- selects active entities whose role refs target the current relationship;
- orders them by active lifecycle, salience, then recency;
- exposes at most two rows with kind, clipped semantic description, salience
  band, lifecycle, and freshness;
- exposes at most two affect activations rooted in those entities;
- exposes relationship and evidence freshness;
- omits IDs, raw evidence, scalar values, and other-user context.

The same canonical projection is reduced by consumer cap, not reconstructed by
relevance, cognition, or the console.

### Projection-Only Cause Classes

Global cross-scope context contains no free-form source description. It maps
typed native roots to this closed cause vocabulary:

| Cause class | Deterministic source |
|---|---|
| `connection_warmth` | generic social event with joy/gratitude/compassion activation or high memory warmth |
| `relationship_strain` | generic social event with source-free social harm, mismatch, or negative character-rooted activation |
| `boundary_pressure` | elevated unfairness, norm violation, exposure, identity threat, autonomy-boundary goal, or anger/disgust rooted there |
| `repair_pressure` | elevated repair need, guilt, shame, reparable harm, or moral-repair goal |
| `safety_pressure` | active threat, elevated harm, fear, or safety goal |
| `loss_pressure` | temporal loss, loss-recovery goal, sadness, or nostalgia rooted there |
| `goal_pressure` | active obstructed goal without a more specific class |
| `uncertainty_pressure` | active knowledge gap, surprise, or curiosity |
| `meaning_pressure` | meaning root, meaning-reconstruction goal, awe, ennui, or low coherence |
| `competence_pressure` | competence drive/goal obstruction or pride/shame rooted in competence |
| `general_activation` | validated active root with no more specific class |

Precedence is the table order except that a direct root kind
`threat`, `knowledge_gap`, or `meaning` selects its corresponding class first.
The class is derived only after native reduction. It is not present in the
carry-over LLM input or decision, is not persisted as affect authority, and
cannot create, suppress, merge, or rename an emotion. Model-facing contexts
receive the class, native emotion id or pressure kind, qualitative bands,
lifecycle, trend, and freshness.

### Interaction-Style Composition

`build_interaction_style_context(...)` becomes the canonical immutable turn
snapshot. In group turns it loads user and group documents concurrently.

- Relevance receives up to three engagement guidelines per source. They are
  tie-breakers after grounded participant, reply, mention, topic, and
  reason-to-speak evidence.
- Cognition receives up to two social and two engagement guidelines per source.
  User guidance informs person-directed handling; group guidance informs
  participation and shared pacing.
- Surface receives the current bounded speech, social, pacing, and engagement
  projections from the same snapshot.
- Action selection may consume group engagement from the snapshot where it
  already consumes that category.

The surface stops loading the database independently.

### Branch Routing

- `relationship_social` receives causal current-user relationship context,
  branch-relevant global affect, and cognition-stage social/engagement style.
- `goal_threat_outcome` receives branch-relevant global affect and operational
  pressures.
- `moral_identity` receives only boundary/repair global rows.
- `existential_drive` receives meaning/goal/competence pressure rows.
- `event_agency` and `epistemic` receive no global posture unless the existing
  semantic source planner routes a matching typed root.
- goal cognition receives the bounded branch-relevant global selection and
  current relationship projection.
- text surface receives up to two affect rows selected from the complete
  effective state by selected-stance relevance, then intensity, lifecycle, and
  freshness; it cannot change selected stance or content facts.

### Ordering And Freshness

Create one process-local character operational ordering coordinator:

1. Register an episode token before its accepted response becomes externally
   observable.
2. Complete the token after operational commit, `no_change`, typed failure, or
   timeout.
3. Before settled relevance loads character state, capture and await all older
   tokens.
4. Bound the complete Core V2 carry-over/update path to 45 seconds.
5. On timeout, record `timed_out`, release waiters, and consume the last valid
   state version with degraded health.
6. Use `updated_at` compare-and-set for self-cognition, sleep, and
   consolidation writers; process-local ordering is not the persistence lock.

Normal chat already serializes cognition/post-turn in the settlement worker;
the coordinator formalizes this guarantee and covers accepted background-task
results that can overlap chat.

### Console Observability

The Character page gains **Operational posture** after the identity panels. It
shows:

- persisted operational update time/version token;
- effective-at time and whether read-time fading changed the view;
- every redacted persisted/effective native affect row up to the twenty-one-row
  bound, including emotion id, intensity, phase, trend, root kind, cause class,
  and freshness;
- bounded effective pressure rows and the separate consumer-selected affect
  and pressure subset;
- latest turn’s consumed state version, context digest, exact public context,
  run ID, and consumption status;
- predecessor outcome and degraded/failure status.

The User Relationship panel adds causal rows, relationship affect, and
freshness. User and Group style panels label effective guidance by
`relevance`, `cognition`, and `surface`.

The brain cognition graph stores the exact public context under
`l2.reasoning.detail.context_consumption`. The console extracts that public
projection; it never reconstructs “what was probably consumed.”

## Data Migration

No historical migration, schema backfill, or baseline-field import is
permitted. The prerequisite identity plan creates the clean target. This plan
keeps the native `cognition_state.v2` shape and uses its existing `updated_at`
as a strictly increasing optimistic version token. Startup fails closed if the
post-identity target lacks canonical native state; no compatibility
initialization from baseline fields is allowed.

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
    primary_root_kind: goal | threat | event | knowledge_gap | meaning
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

### Character Carry-Over Decision

```text
character_carryover_decision.v1
  action: no_change | apply
  reason_code:
    no_lingering_effect | already_represented | transient_scene_only |
    unsupported | lingering_character_effect
  privacy_disposition: source_free | unsafe
  semantic_appraisals: existing bounded SemanticAppraisalResultV2 rows, max 4
```

`no_change` requires empty appraisals. `apply` requires
`privacy_disposition=source_free`, at least one supported appraisal,
and the settled episode evidence root. Scope is deterministically bound to
`character`; it is absent from model output. The LLM does not emit cause
classes or emotion ids. The Core V2 owner trial-reduces the proposal, derives
native affect, validates the replacement state, and returns a
`CharacterCarryoverResultV2` containing the decision plus either no update or
exactly one character-scoped `StateUpdateV2`. Free-form explanations remain
trace-only.

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
  base_updated_at: UTC timestamp
  committed_updated_at: UTC timestamp | none
  registered_at: UTC timestamp
  completed_at: UTC timestamp | none
  error_code: bounded enum | none
```

Public telemetry omits `source_episode_id` and exposes only status, version
timestamps, duration, and bounded error code.

### Public Interfaces

- `cognition_core_v2.state_reducers.apply_character_elapsed_decay(state, *, elapsed_seconds) -> dict`
- `cognition_core_v2.state_projection.project_character_operational_state(state, *, effective_at) -> character_operational_state_view.v1`
- `cognition_core_v2.state_projection.select_character_operational_context(state_view, *, consumer_role) -> character_operational_context.v1`
- `cognition_core_v2.state_projection.project_relationship_context(user_state, *, effective_at) -> dict`
- `cognition_core_v2.character_carryover.run_character_carryover_cognition(...) -> CharacterCarryoverResultV2`
- `db.character.compare_and_replace_character_cognition_state(*, expected_updated_at, replacement) -> bool`
- `db.interaction_style_images.build_interaction_style_context(...) -> interaction_style_turn_snapshot.v1`
- `consolidation.character_operational_state.commit_character_operational_update(...) -> receipt`
- `brain_service.character_state_ordering.register_predecessor(...)`
- `brain_service.character_state_ordering.complete_predecessor(...)`
- `brain_service.character_state_ordering.await_predecessors(...)`

No second public reader/writer/projection path is permitted.

## LLM Call And Context Budget

- Foreground call count: unchanged.
- Existing consolidation router call count: unchanged.
- Core V2 character carry-over stage: zero or one state-only call only when its
  consolidation target is selected;
  maximum three replacement attempts for invalid output, never semantic
  accumulation across attempts.
- Carry-over dynamic input: settled semantic episode evidence plus one bounded
  qualitative character-state projection; complete serialized packet at most
  8,000 characters. It contains no cause-class vocabulary, raw database shape,
  relationship state, user/channel identifier, or prompt-irrelevant context.
- Operational update deadline: 45 seconds including replacements, reducer, and
  commit.
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
- Existing stage context ceilings remain unchanged. Projection overflow is a
  typed construction failure, not silent clipping after prompt assembly.
- A captured long-context V2 case and prompt-size instrumentation must prove
  that the new context does not recreate or worsen context-limit failures.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/cognition_core_v2/character_carryover.py`: Core V2 state-only prompt, exact decision contract, bounded attempts, native trial reduction, and zero-or-one `StateUpdateV2` result.
- `src/kazusa_ai_chatbot/consolidation/character_operational_state.py`: target adapter, episode idempotency, optimistic commit orchestration, and receipts; it owns no emotion schema or semantic reinterpretation.
- `src/kazusa_ai_chatbot/brain_service/character_state_ordering.py`: bounded predecessor tokens and public receipts.
- `tests/test_cognition_core_v2_operational_projection.py`: elapsed fading, full-state view, consumer selection, projection-only cause mapping, relationship causes, caps, digests, and privacy.
- `tests/test_cognition_core_v2_emotion_scope_matrix.py`: all twenty-one ids, eighteen character-root-eligible cases, three relationship-required exclusions, same-id cross-scope isolation, unchanged formulas, and decay-rate parity.
- `tests/test_cognition_core_v2_character_carryover.py`: decision validation, no emotion/cause-class output, native derivation, offence distinctions, and zero-or-one character update.
- `tests/test_character_operational_state_consolidation.py`: router/adapter ownership, idempotency, optimistic commit, and compare-and-set retry.
- `tests/test_character_operational_state_ordering.py`: normal, cross-worker, failure, timeout, and shutdown behavior.
- `tests/test_short_horizon_state_composition_integration.py`: relevance/cognition/surface/style wiring and precedence.
- `tests/test_short_horizon_state_composition_live_llm.py`: controlled A/B live-model cases.
- `tests/test_short_horizon_state_composition_e2e_live_llm.py`: natural private/group causal sequences.
- `tests/control_console_e2e/test_short_horizon_state_visibility_e2e.py`: authenticated real-service browser gate.
- `tests/fixtures/cognition_core_v2_short_horizon_state_cases.json`: controlled inputs and behavior rubrics.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py`: pure read-time character fading and version-safe native reducer use.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`: canonical full global view, consumer selection, projection-only cause labels, and causal relationship projection.
- `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py`, `contracts.py`, `workspace.py`, `semantic_source_planner.py`, `semantic_appraisal.py`, `goal_cognition.py`, and `facade.py`: public carry-over entrypoint, typed input, branch routing, prompt variables, and bounded consumption.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: state ownership, projection, fading, and branch contracts.
- `src/kazusa_ai_chatbot/db/character.py` and `db/__init__.py`: monotonic update timestamp and canonical compare-and-set writer.
- `src/kazusa_ai_chatbot/db/interaction_style_images.py` and `db/__init__.py`: immutable all-stage snapshot, concurrent group loads, and stage projections.
- `src/kazusa_ai_chatbot/db/README.md`: replace stale runtime mood/vibe documentation.
- `src/kazusa_ai_chatbot/consolidation/target.py`, `schema.py`, `lane_router.py`, `core.py`, `persistence.py`, `__init__.py`, and `README.md`: routed operational lane and sanitized metadata.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`, `turn_settlement.py`, `__init__.py`, and `README.md`: registration, bounded waiting, completion, and lifecycle receipt.
- `src/kazusa_ai_chatbot/service.py`: pre-relevance state/style load, dead legacy-read removal, accepted-task ordering, graph consumption projection, and reuse downstream.
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py` and `relevance/README.md`: native context contract and tie-breaker rules.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`, `persona_supervisor2.py`, `persona_supervisor2_cognition.py`, `persona_supervisor2_l3_surface.py`, and `nodes/README.md`: snapshot handoff and removal of the surface DB reload.
- Post-identity self-cognition and sleep writer files identified at Stage 1: switch their character-state commits to compare-and-set without changing their semantic behavior.
- `src/control_console/contracts.py`, `repository.py`, `app.py`, `redaction.py`, `static/index.html`, `static/console.js`, `static/console.css`, and `README.md`: redacted operational/relationship/style panels and latest-consumption rendering.
- Focused existing tests named by Stage 1, including cognition-state, relevance,
  interaction-style, service background consolidation, self-cognition writer,
  control-console repository/web, and cognition-graph tests.
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

The actual problem is that an interaction can change only the current user's
state, so the character cannot remain emotion-specifically affected with later
users or decay naturally between sleep cycles; native relationship causes and
style composition are also disconnected. The minimal solution is one existing
character state, one Core V2-owned state-only carry-over stage, one canonical
global view with bounded consumer selections, one relationship projection, one
shared style snapshot, one bounded ordering coordinator, and public
observability.

Forbidden complexity:

- prose replacement fields;
- a channel mood/topic state;
- a second global-state collection;
- event sourcing or historical operational-state replay;
- a new foreground model stage;
- a second emotion ontology, generic mood classifier, or cause-class-driven
  emotion producer;
- copying user affect, relationship rows, or relationship-required emotions
  into character state;
- a full second response-cognition DAG for post-turn carry-over;
- a second relationship reducer;
- style-to-topic or style-to-relevance authority;
- per-consumer DB reloads or projection implementations;
- compatibility aliases, flags, fallback readers, or dual writes;
- an operator editor.

A new helper is permitted only for the Core V2 state-only carry-over boundary,
nontrivial projection/selection validation, compare-and-set ownership, or
predecessor coordination defined in this plan.
Evidence must show an existing owner cannot hold that responsibility before
adding any further module or field.

## Agent Autonomy Boundaries

Execution agents may:

- choose local private helper names inside the listed files when they implement
  the exact contracts and remove repeated structural validation;
- adjust test fixture wording while preserving the approved scenario,
  controlled variable, and rubric;
- update post-identity file paths in the Change Surface when Stage 1 proves a
  symbol moved without changing ownership.

Execution agents must stop and request direction before:

- adding a persisted semantic field or collection;
- changing the eighteen/three emotion scope matrix, any emotion formula/guard,
  cause enum, projection caps, 45-second deadline, fading rate, lifecycle
  threshold, branch routing, or composition precedence;
- changing identity, relationship reducer, conversation-progress, RAG,
  relevance reason-to-speak, action, scheduler, adapter, or delivery semantics;
- adding a model call, fallback path, compatibility layer, feature flag,
  historical migration, or current Asuna access;
- broadening private information visible cross-scope or in the console;
- editing outside the declared surface for a reason other than a proven
  post-identity path move.

## Implementation Order

### Stage 1 — Post-Identity Rebaseline And Closure

1. Confirm the prerequisite plan is completed and its execution evidence is signed.
2. Record `git status --short`, HEAD, Python version, mandatory-skill reads, and relevant README/source/test baselines.
3. Locate every character cognition-state reader/writer and every legacy runtime occurrence with `rg` and `git grep main`.
4. Create and sign the baseline-to-V2 capability closure artifact and the
   twenty-one-emotion scope/decay matrix.
5. Record the post-identity focused test list and exact self-cognition/sleep writer paths.

### Stage 2 — Parent-Owned Focused Test Contract

6. Create emotion-scope and projection tests for all twenty-one ids, versions,
   ordinary character fading, full-state versus consumer selection, cause
   mapping, causal relationship selection, privacy, caps, and digests.
7. Run each focused file and record the expected missing-symbol/contract failures.
8. Create persistence and ordering tests for optimistic commits, one retry, episode idempotency, predecessor success/failure/timeout, and shutdown.
9. Run each focused file and record expected failures.
10. Start one production-code subagent with the approved plan, mandatory skills, focused tests, and production-only ownership.

### Stage 3 — Native State, Projection, And Persistence

11. Enforce strictly increasing character-state `updated_at` values.
12. Add character elapsed fading and canonical global/relationship projections.
13. Add the `expected_updated_at` compare-and-set writer and update existing writers.
14. Pass all Stage 2 state/projection/persistence tests before integration wiring.

### Stage 4 — Core V2 Carry-Over And Consolidation Adapter

15. Add the Core V2 state-only decision schema, prompt, parser/replacement policy, privacy validation, and zero-or-one character update result.
16. Add the consolidation target adapter and invoke Core V2 only when the router selects it.
17. Trial-reduce the validated proposal through native reducers, derive affect, and commit the validated `StateUpdateV2` through optimistic persistence.
18. Record sanitized route, no-change, commit, conflict, failure, and timeout metadata.
19. Pass focused consolidation tests.

### Stage 5 — Ordering And Shared Turn Snapshot

20. Add predecessor registration/completion/waiting to normal chat and accepted-task result paths.
21. Load effective character state and one interaction-style snapshot before settled relevance.
22. Reuse the same immutable state/style snapshot through cognition and surface.
23. Pass ordering, DB-read-count, style-snapshot, and failure-path tests.

### Stage 6 — Relevance, Cognition, And Surface Composition

24. Replace dead relevance inputs with native global/relationship/style projections.
25. Add branch-specific operational/relationship/style context to V2 contracts and prompts.
26. Add goal-stage context and selected-stance-relevant surface affect.
27. Remove the surface DB reload and enforce topic/authority precedence.
28. Pass integration, prompt-size, context-limit, and negative-authority tests.

### Stage 7 — Console And Public Telemetry

29. Add exact public context consumption to the cognition graph.
30. Add Character operational posture, User causal relationship, and consumer-labeled style projections.
31. Add redaction, missing-data, failure/degraded, version mismatch, and stale-service tests.
32. Pass repository, contract, web-surface, and fake-service browser tests.

### Stage 8 — Deterministic And Guarded-Database Verification

33. Run focused files, affected regression suites, static greps, compilation, and guarded clean-database persistence/restart/concurrency tests.
34. Record call counts, DB reads, payload sizes, no-pending barrier overhead, routed update latency, and timeout release.
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

- Parent-led native subagent execution is mandatory.
- The parent owns tests, expected failures, integration wiring, verification,
  live runs, browser sign-off, evidence, review remediation, and lifecycle.
- One production-code subagent owns only production files in the approved
  Change Surface after Stage 2 establishes the focused contract.
- One independent review subagent runs after all implementation verification
  passes and does not implement fixes.
- If native subagent capability is unavailable, stop before execution and
  report the blocker unless the user explicitly authorizes fallback execution.

## Progress Checklist

- [ ] Stage 1 — prerequisite and closure matrices signed. Covers steps 1-5; verify status/HEAD/static inventory, zero unexplained closure rows, and all twenty-one scope/decay rows; retain rebaseline and both matrices; reread, hand off to Stage 2, and sign `<parent/date>`.
- [ ] Stage 2 — focused test contract established and production subagent started. Covers steps 6-10; verify named expected failures; retain commands/failures/subagent brief; reread, hand off to Stage 3, and sign `<parent/date>`.
- [ ] Stage 3 — native versioning, fading, projection, and persistence pass. Covers steps 11-14; verify focused state/projection/persistence tests; retain changed files/pass output; reread, hand off to Stage 4, and sign `<parent/date>`.
- [ ] Stage 4 — Core V2 carry-over and consolidation adapter pass. Covers steps 15-19; verify native derivation, zero-or-one update, sanitized events, idempotency, and conflicts; retain route/update/commit/failure evidence; reread, hand off to Stage 5, and sign `<parent/date>`.
- [ ] Stage 5 — ordering and one-snapshot handoff pass. Covers steps 20-23; verify ordering/timeout/read-count/style tests; retain sequence/timing/read evidence; reread, hand off to Stage 6, and sign `<parent/date>`.
- [ ] Stage 6 — relevance/cognition/surface composition passes. Covers steps 24-28; verify integration/authority/prompt-size/context-limit tests; retain consumer contexts/digests/regressions; reread, hand off to Stage 7, and sign `<parent/date>`.
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
rg -n "character_operational_state_view|character_operational_context|character_carryover_decision|relationship_operational_context|interaction_style_turn_snapshot" src tests
```

Expected:

- the first and third commands return zero matches; `rg` exit code `1` is expected;
- runtime matches for `global_vibe` and `last_relationship_insight` are zero;
- allowed non-runtime matches are the rejected seed list in
  `character_profile.py`, isolated historical cleanup in
  `db/script_operations.py`, and historical/active plan text;
- canonical new contract matches occur only in declared owners and consumers.

### Focused Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_operational_projection.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_emotion_scope_matrix.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_character_carryover.py -q
venv\Scripts\python.exe -m pytest tests/test_character_operational_state_consolidation.py -q
venv\Scripts\python.exe -m pytest tests/test_character_operational_state_ordering.py -q
venv\Scripts\python.exe -m pytest tests/test_short_horizon_state_composition_integration.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_state.py tests/test_cognition_core_v2_projection.py tests/test_interaction_style_images.py -q
venv\Scripts\python.exe -m pytest tests/test_persona_relevance_agent.py tests/test_relevance_turn_settlement.py tests/test_service_background_consolidation.py -q
venv\Scripts\python.exe -m pytest tests/test_control_console_repository.py tests/test_control_console_cognition_graph.py tests/test_control_console_web_surface.py -q
```

Expected: all pass. Each new test file is first run before implementation and
its expected failure is recorded.

### Guarded Database And Ordering

Use the test-style guarded live-database invocation, one file at a time.
Required proofs:

- clean character state has a valid native UTC `updated_at`;
- commit advances `updated_at` exactly once;
- duplicate episode is `no_change/already_committed`;
- one stale writer reloads/reapplies once and succeeds;
- second conflict fails without overwrite;
- restart loads the latest committed state version;
- every emotion score follows its frozen rate in the pure character effective
  copy, including representative 1/hour, 4/hour, and 12/hour cases;
- read-time fading changes phase/trend/retention without a read write or source
  version change;
- a later commit persists exactly once from the effective base without double
  decay;
- a user-A relationship activation and same-id character activation retain
  separate roots/owners through persistence and restart;
- accepted-task update and incoming chat consume state versions in order;
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
- no Core V2 carry-over call when the router omits the lane;
- at most one user and one group style-image read per logical group turn;
- no surface style-image read;
- no-pending predecessor coordinator overhead below 5 ms at p95 over 1,000
  deterministic iterations;
- no-pending/no-operational-lane end-to-end p95 no more than 10% above the
  post-identity baseline over at least 20 guarded runs;
- routed update completes or releases by the 45-second deadline;
- every prompt remains inside its existing cap;
- the captured long-context V2 reproduction does not regress.

### Console Browser Gate

Use the in-app Browser when available; otherwise use project Playwright and
record the reason. Start the authenticated real console at its configured
loopback URL, normally `http://127.0.0.1:8765/`, against the same guarded data.

Verify:

1. Character Operational posture shows every persisted and elapsed-effective native affect row, its emotion id/phase/trend, and whether ordinary fading changed it before sleep.
2. The panel separately shows the exact bounded affect/pressure subset consumed by the latest turn, its run ID, source version, and context digest.
3. Persisted, effective, and consumed states plus version mismatch/degraded/timeout conditions render distinctly.
4. User A Relationship shows offender-specific axes, causal rows, affect, and freshness; user B's panel and payload contain none of A's relationship data.
5. Projection-only cause classes appear beside native emotion/root labels and never replace them.
6. User and Group style panels label relevance, cognition, and surface projections.
7. Private detail and internal IDs are absent from DOM and network payloads.
8. Loading, empty, failed, and stale-brain states are explicit.
9. Desktop and narrow layouts have no clipping, overlap, or unreachable content.
10. Every affected navigation/control is exercised.
11. Browser console and page errors are zero.

Save screenshots plus `test_artifacts/cognition_core_v2_short_horizon_state/console_browser_signoff.md`.
This gate is blocking.

### Full Regression

After focused gates pass, derive the final affected-file test command from the
post-identity inventory and include at minimum cognition V2, relevance,
consolidation, service post-turn, self-cognition character-state writers,
interaction style, control-console contracts/repository/web, and control-console
E2E. Compile every changed Python file with:

```powershell
venv\Scripts\python.exe -m py_compile <each changed Python file>
```

Expected: all pass with no deselected required case and no unreviewed warning.

## Independent Code Review

The independent review subagent receives:

- this approved plan and prerequisite completion evidence;
- full implementation diff and changed-file inventory;
- closure matrix;
- focused/regression/database/live-LLM/browser/performance evidence;
- controlled and natural causal reviews.

It must review architecture ownership, LLM semantic boundaries, state/privacy
contracts, version/CAS correctness, ordering and timeout behavior, prompt
budgets, current-scene precedence, duplicate DB reads, error visibility,
console redaction, test quality, and exact plan compliance.

The parent may fix review findings only within the approved Change Surface.
A finding that changes a contract, enum, limit, owner, or scope stops closeout
for plan/user approval. Completion requires no unresolved blocker or
high-severity finding.

## Acceptance Criteria

1. The prerequisite identity-growth plan is completed and signed.
2. The capability matrix has zero unexplained baseline rows, and the signed
   twenty-one-emotion matrix proves eighteen character-root-eligible and three
   relationship-required ids without changing any formula or guard.
3. Character cognition state is the sole persisted short-horizon global state.
4. No runtime mood/vibe/last-insight fallback remains.
5. Operational update timestamps are strictly increasing and all character-state writes are optimistic and conflict-safe.
6. Ordinary character elapsed fading uses frozen emotion-specific rates,
   persists once from the effective base without double decay, and composes
   with stronger sleep recovery.
7. Global state retains all active/fading native emotion rows; model contexts
   are bounded branch-relevant selections, causal, source-free, and private-safe.
8. Relationship projection contains axes plus bounded native causes/affect/freshness without a new prose field.
9. Settled relevance consumes native global, relationship, and style context under reason-to-speak authority.
10. Relevant V2 appraisal/goal branches and text surface consume their approved projections.
11. Current evidence/history/conversation progress remain factual/topic authority.
12. One style snapshot is reused by every ordinary-turn consumer.
13. A healthy predecessor commit reaches the immediate next eligible cross-channel turn.
14. Failure/timeout consumes only the last valid state version and is visibly degraded.
15. Controlled real-LLM gates distinguish anger, sadness, and disgust and show
    emotion-specific elapsed behavioral change before sleep.
16. Normal-entrypoint offence, non-offender, elapsed-fading, apology/repair,
    and private/group cross-scope sequences pass without direct state writes.
17. Every relationship-emotion isolation, privacy, topic, and style negative passes.
18. Performance and context-budget gates pass.
19. The authenticated browser gate matches full persisted/effective affect,
    scoped relationship data, and the exact consumed subset.
20. Independent review is approved, all in-scope findings are remediated, and
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
