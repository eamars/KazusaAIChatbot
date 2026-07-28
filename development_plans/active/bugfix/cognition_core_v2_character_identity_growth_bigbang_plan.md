# Cognition Core V2 Character Identity Growth Big-Bang Plan

## Summary

- Goal: make character self-image, growth, and cross-scope carry-over a first-class Cognition Core V2 identity system whose promoted revisions replace every supported semantic seed field.
- Plan class: `high_risk_migration`.
- Status: `in_progress`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `no-prepost-user-input`, `database-data-pull`, `debug-llm`, `character-test`, `control-console-web-development`, `py-style`, `cjk-safety`, `test-style-and-execution`, and `python-venv`.
- Overall cutover: one forward-only big-bang contract replacement, with no legacy reader, dual write, compatibility mapper, historical replay, growth backfill, or production-database recovery.
- Highest risks: allowing learned identity to become authoritative without letting user instructions directly rewrite the character; privacy-safe private/group carry-over; replacing process-local static authority; atomic promotion; preventing evidence double-counting and identity oscillation; and proving a real interaction caused a durable identity revision that changed later cognition and behavior.
- Acceptance: a zero-gap baseline-to-V2 closure matrix is signed before legacy deletion; an operator manually creates revision `0` before startup and an unseeded application crashes; reviewed explicit and corroborated inferred growth arise through normal episode/background paths; one correlated proof chain joins real interaction, evidence, proposal, review, revision, cache refresh, next-episode projection, cognition, and visible behavior; only the latest revision reaches cognition and surfaces; prior revisions remain reviewable; private and group evidence may influence one global identity without leaking scoped details; every supported leaf/category receives the proof required below.
- Execution authority: approval and the user's explicit implementation command are recorded; guarded implementation/test work is authorized while production/historical database mutation, deployment, and cutover retain their separate authority gates.

## Context

The V2 design did not intentionally remove character self-growth:

- The completed Stage 2 execution manifest directs `consolidation/lane_router.py` to remove retired lanes while retaining memory/self-image lanes.
- The Stage 2 contract registers `promoted_reflection` as evidence for all six semantic appraisal families.
- Stage 3 keeps residue and reflection as bounded evidence while preventing either from initiating cognition by itself.

The current branch instead contains three distinct defects:

| Surface | Database observation | Pipeline observation | Conclusion |
|---|---|---|---|
| Self-image | Asuna `character_state` has no `self_image` | target declares `character_self_image`, but the lane router, origin policy, and persistence do not produce it; `consolidation/images.py` has no live producer; RAG builds `character_image`, but the V2 connector ignores it | unintentional process disconnection |
| Growth | no active growth traits or growth runs; 45 reflection runs exist | promotion emitted `memory_type=self_guidance` where the validator requires `defense_rule`; invalid output is skipped rather than regenerated; worker growth requires `succeeded_count > 0`; V2 drops `promoted_reflection_context` | upstream contract failure plus V2 consumption regression |
| Carry-over | latest 100 Asuna residue rows are nonempty across 51 private and 49 group-channel records; 96 corresponding recorder events succeeded and four reported `empty`; no `character_global` residue exists | Character console queries blank/global residue scope, while runtime residue correctly remains scoped | data exists; Character panel queries the wrong concept |

Read-only evidence is retained in:

```text
test_artifacts/asuna_character_state_20260728.json
test_artifacts/asuna_character_reflection_runs_20260728.json
test_artifacts/asuna_global_growth_runs_20260728.json
test_artifacts/asuna_global_growth_traits_20260728.json
test_artifacts/asuna_active_promoted_reflection_memory_20260728.json
test_artifacts/asuna_residue_rows_20260728.json
test_artifacts/asuna_residue_events_20260728.json
test_artifacts/asuna_character_global_residue_20260728.json
```

The baseline branch offers ownership history, not the target: `character_state` consolidation produced a reflection summary, self-image update, `character_state.self_image`, and cache invalidation. Baseline global growth only produced soft communication guidance and never replaced static identity. Mechanical restoration would leave the central requirement unmet.

The target is:

```text
validated seed -> immutable revision 0
settled episode/reflection evidence
  -> identity proposal LLM -> independent review LLM
  -> deterministic privacy/provenance/cadence/persistence checks
  -> immutable full revision N
  -> latest-only cognition and surface projection
  -> privacy-safe console lineage
```

Identity is global to the character. Private evidence may affect later group behavior and group evidence may affect later private behavior only through character-owned abstraction. User facts, exact utterances, relationship promises, intimate details, and scope identifiers stay with their current owners.

The user invalidated the Stage 4 one-off database migration plan on 2026-07-28.
Its body now resides at
`development_plans/archive/superseded/cognition_core_v2_stage_4_production_database_migration_plan.md`
as non-executable historical evidence. This plan is the sole active contract
for character identity growth and clean target construction. It authorizes no
historical database recovery, migration, replay, or backfill.

## Mandatory Skills

- `development-plan`: approval, execution, review, updates, handoff, archive, and sign-off.
- `local-llm-architecture`: prompts, context projection, call budgets, appraisal routing, and stage boundaries.
- `no-prepost-user-input`: proposal/review semantics; deterministic code cannot decide whether user language became accepted character identity.
- `database-data-pull`: read-only MongoDB inspection and evidence export.
- `debug-llm`: prompt comparison, live calls, and readable quality artifacts.
- `character-test`: multi-turn private/group behavior and trace inspection.
- `control-console-web-development`: console code, browser validation, and screenshots.
- `py-style`, then `cjk-safety` where applicable: Python edits and review.
- `test-style-and-execution`: test changes and execution.
- `python-venv`: environment checks or dependency work.

## Mandatory Rules

1. Production changes remain blocked while status is `draft`.
2. Use `venv\Scripts\python.exe`; use `apply_patch` for manual edits; check `git status --short`, root/subsystem docs, source, and tests before production edits. The user's 2026-07-28 instruction explicitly authorizes direct `.env` loading for this execution.
3. Use a clean guarded test database. Historical database evidence is read-only and Asuna-only; never inspect another character database or replay, mutate, or seed from historical Asuna data/artifacts. Production code remains character-generic and contains no character/personality literals.
4. Cut over caller, callee, database owner, tests, scripts, configuration, console, and docs together. Add no static overlay, legacy mapper, alias collection, or dual write.
5. Revision `0` is the manually loaded seed. Bootstrap and runtime never create identity data. An absent ledger crashes startup before operational character-state creation. Later immutable revisions contain complete effective identity snapshots. The highest revision number is the sole active identity; history is review-only.
6. Only the latest identity reaches cognition, dialog, text expression, visual planning, adapter-visible naming, and new episode construction.
7. All supported semantic identity fields are replaceable. Operational IDs, accounts, permissions, schemas, DB keys, limits, delivery/security policy, and cognition-state internals are forbidden.
8. One immediate turning point requires high-confidence character-authored self-redefinition and a separate high-confidence review. User instruction alone is insufficient.
9. Inferred growth starts at three distinct settled episode roots across two character-local dates; bounded central configuration may change the pace.
10. `root_episode_id` is the sole corroboration-counting identity. Retries, duplicate routing, reflection cards, and other derivatives of one settled episode count once in total.
11. Repository provenance determines root/date distinctness. LLM counts, copied text, reflection-run count, and evidence-card count never determine cadence.
12. An inferred reversal of a previously changed path requires fresh roots created after that path's latest revision and must independently satisfy the configured episode/date threshold. Character-authored explicit turning points retain the immediate reviewed path.
13. Contradictory inferred candidates for the same path and base revision cannot both become ready. The review stage selects one coherent direction or rejects both; deterministic code enforces the single-winner transition without interpreting meaning.
14. LLM stages own semantic interpretation, character authorship, candidate matching, contradiction/coherence, abstraction, and identity relevance. Deterministic code owns schema, paths, types, declared normalization, provenance, cadence, persistence, permissions, refresh, and limits.
15. Every raw LLM response passes through `kazusa_ai_chatbot.utils.parse_llm_json_output(...)`. Structural/enum/type/key conflicts trigger a stage-owned full replacement with the same context, capped at three attempts.
16. Deterministic code may map an accepted semantic band to its declared numeric value. It cannot change semantic direction, invent meaning, turn user instruction into agreement, or promote rejection.
17. Cross-scope promotion requires a character-owned, detail-free abstraction, `private_detail_risk=low`, and removed user details.
18. Raw transcript, raw reflection/residue, user identity, quotes, private facts, and raw scope refs never enter revisions, identity prompt context, or console summaries.
19. Residue remains scope-specific working continuity. Create no synthetic `character_global` aggregate.
20. Reflection remains offline. Only validated daily cards and repository-linked root episode refs support scheduled identity evaluation.
21. Growth adds no foreground call. Post-turn promotion first affects the next episode.
22. One episode promotes at most one revision. Explicit turning points bypass inferred daily cadence, while inferred growth defaults to one promotion per local day.
23. A stale candidate retains its claimed roots and is reviewed against latest identity in place; accepted patches atomically rebase that same candidate, while incoherent candidates become `superseded`. Roots never move to another candidate.
24. Mongo transactions plus unique current-base/root-claim indexes reject duplicate evidence and concurrent forks; promotion and first-consumption claims fail closed when transaction support is unavailable.
25. Every identity-routing decision emits a sanitized stage/reason event. Routed evaluations additionally persist one sanitized growth-run row.
26. The first eligible episode after a revision atomically stores its loaded revision number, episode ID, consumer kinds, and projection digest in that revision's sanitized growth run; TTL event telemetry mirrors this durable receipt. No identity text enters either.
27. Console health must distinguish healthy inactivity, insufficient evidence, semantic rejection, cadence wait, pipeline failure, awaiting first consumption, and revision-consumption regression.
28. Raw prompts/output stay in protected traces. Growth-run rows contain sanitized outcome metadata only.
29. Console code consumes public redacted projections, not raw collections or traces.
30. Every supported leaf gets deterministic lineage/projection coverage; every supported category gets controlled live stored-lineage and counterfactual behavior evidence.
31. Controlled capability proof and forward-only natural-path proof are separate gates. Passing one never substitutes for the other.
32. The longitudinal pilot must use normal chat intake and background workers. Tests and operators may read its state but cannot insert or edit candidate, revision, growth-run, or evidence rows.
33. Run live LLM cases one at a time, inspect each, and author readable reviews from the real input/output and protected trace evidence.
34. Before legacy deletion, the baseline-to-V2 closure matrix must contain zero unexplained capability rows and receive parent sign-off.
35. After any automatic context compaction, reread this complete plan before continuing.
36. After signing off each major progress stage, reread this complete plan before starting the next stage.
37. Complete parent-owned code review and remediation/reruns before lifecycle closeout, merge, or final sign-off.

## Must Do

1. Produce and sign a baseline-to-V2 capability closure matrix with no unexplained row before deleting legacy code.
2. Add an immutable versioned global identity ledger, a manual revision-`0` loader, and an unseeded-startup crash boundary.
3. Separate semantic identity from operational `character_state`; make latest revision replace static seed values.
4. Add canonical nonempty self-image and visual-characterization fields.
5. Add typed root-episode evidence lineage that prevents direct and reflection-derived double counting.
6. Add proposal/review episode growth and reviewed explicit/inferred promotion policies.
7. Add post-revision fresh-evidence and contradiction rules that prevent inferred identity oscillation.
8. Add scheduled reflection identity evaluation independent of memory-mutation count.
9. Repair reflection promotion invalid-output handling with bounded regeneration.
10. Carry privacy-safe character identity between private and group contexts in both directions.
11. Project latest identity naturally into relevant V2 appraisals, goals, boundaries, text, name, and visual surfaces.
12. Connect bounded promoted-reflection context to V2 evidence without giving it identity authority or an extra corroboration count.
13. Keep residue scoped and replace false Character-global residue presentation with identity lineage.
14. Add stage/reason telemetry plus a redacted health funnel that distinguishes no data, insufficient evidence, semantic rejection, process failure, promotion, and runtime consumption.
15. Expose current/prior revisions, diffs, evidence summaries, candidates, rejections, health state, and pace settings in the console.
16. Remove old global-growth/self-image code, collections, scripts, flags, tests, and active docs in one cutover after closure sign-off.
17. Prove clean start, legacy fail-closed start, concurrency, restart, latest-only projection, every override path, privacy, pace, anti-oscillation, console lineage, and live behavior.
18. Prove one complete causal chain from a normal real-model chat interaction through background growth to changed next-episode cognition and visible behavior.
19. Run a forward-only, two-character-local-date longitudinal pilot through normal chat intake with no direct growth-state writes.
20. Produce a baseline-versus-V2 pace calibration and controlled counterfactual behavior review from real outputs.

## Deferred

- No historical self-image, traits, growth runs, reflection output, residue, or conversation recovery/replay/backfill; no in-place Asuna DB repair.
- User identity/memory, relationship state, group style, commitments, lore, skills, facts, and domain expertise do not become character identity.
- V2 emotion/cognition-state reducers, autonomous contact, scheduler policy, action permissions, adapters, and delivery remain unchanged.
- RAG ranking, retrieval, memory evolution, and conversation progress remain unchanged. Residue keeps its private/group algorithms while deleting the `character_global` enum, loader candidate, recorder fallback, and console query; unresolved scope fails closed without a write.
- Identity branching/merge, active rollback, mutable revision editing, per-user personas, alternate simultaneous identities, and console candidate approval are outside scope.

## Cutover Policy

Overall strategy: forward-only big-bang replacement on a clean target.

| Area | Policy | Instruction |
|---|---|---|
| Profile authority | bigbang | latest identity replaces `character_state` semantic fields and process-static overlay |
| Seed | manual bigbang | operator loads the selected validated profile as immutable revision `0`; startup never auto-seeds |
| Self-image/growth | bigbang | replace `character_state.self_image`, `consolidation/images.py`, and soft trait drift with full reviewed revisions |
| Carry-over | bigbang | global identity revision is character carry-over; residue stays user/group scoped |
| Cognition/surfaces | bigbang | latest identity plus typed promoted-reflection evidence; no history in prompts |
| Database | clean target | create revisions, candidates, and sanitized runs; omit legacy growth collections |
| Existing legacy DB | fail closed | legacy semantic state without identity revision stops startup with clean-cutover guidance |
| Operator loader | revisioned | `--force` creates immutable full `operator_reset` revision |
| Historical Stage 4 plan | superseded | archived plan is non-executable; perform no historical migration, recovery, replay, or backfill |
| Configuration | restart-applied | replace old growth flag/budget with bounded identity pace settings |
| Tests/docs | bigbang | replace old expectations and retire old vocabulary |

### Cutover Policy Enforcement

- Follow the selected policy for every row; a big-bang row permits no compatibility reader, dual write, alias collection, fallback profile, or legacy projection.
- Construct revision `0` only through the explicit operator loader using the selected validated canonical profile.
- Treat the archived Stage 4 plan as historical context and never as execution authority.
- Require explicit user approval before changing any row's cutover policy.

Final static checks require zero production imports/calls/reads for `kazusa_ai_chatbot.global_character_growth`, `upsert_character_self_image`, `global_character_growth_traits`, `global_character_growth_runs`, static semantic `character_state`, and Character-page `character_global` residue; exactly one latest-identity reader and one identity persistence owner remain.

## Target State

### Ownership And Flow

| Owner | Responsibility |
|---|---|
| `character_identity_growth` | contracts, prompt-safe projections, proposal/review calls, policy, orchestration |
| `db.character_identity_growth` | MongoDB indexes/queries, candidate transitions, immutable revisions, sanitized runs |
| profile loader | validate explicit seed or full operator reset |
| consolidation | route settled episode evidence to identity owner |
| reflection cycle | validate daily evidence and invoke identity pass independently |
| Cognition V2 | consume latest constraints/promoted evidence and decide stance/goals |
| text/visual surfaces | render using their authorized latest identity projection |
| residue | retain scoped short-term continuity |
| control console | show redacted lineage and bounded restart-applied pace controls |

```text
operator loader -> validate profile -> insert revision 0
startup -> require existing revision 0/latest -> load operational cognition state
episode -> read latest -> typed V2 projection -> response settles
        -> background router -> root evidence -> proposal -> review -> policy
        -> optional revision N -> invalidation/refresh
daily   -> validated reflection cards -> independent memory promotion and identity evaluation
next eligible episode -> load N -> authorized cognition/surface consumers
                      -> durable receipt plus sanitized mirrored event
```

### Baseline-To-V2 Capability Closure Gate

Before production implementation starts, the parent creates
`test_artifacts/character_identity_growth/baseline_v2_closure_matrix.md` from
the exact `git show main:...` commands under Verification, archived V2 plans,
current source, and current tests.
The parent authors this readable artifact after inspecting the source evidence;
no script generates the review. Each row records the baseline trigger, owner,
persistence, reader, cache/cadence behavior, V2 replacement, disposition,
focused test, live evidence, and final sign-off.

The fixed closure inventory is:

| Baseline capability | Baseline owner/path | V2 replacement | Disposition | Required proof |
|---|---|---|---|---|
| durable self-state selection | consolidation `character_state` lane and reviewer | settled-episode `character_identity_growth` route plus proposal/review | superseded by narrower identity-only selection | routing positive/negative tests and real routed episode |
| session self-image synthesis | `consolidation/images.py::_update_character_image` | reviewed patches to canonical `self_image` | retained and strengthened | self-image leaf tests plus live self-concept behavior |
| self-image persistence | `upsert_character_self_image` in `character_state` | immutable full identity revision | superseded | revision/diff/immutability live-DB proof |
| character cache refresh | `CacheInvalidationEvent(source="character_state")` | `source="character_identity"` plus service/name/RAG/local-context refresh | retained | invalidation and first-next-episode consumption proof |
| runtime profile authority | startup/static profile composed with mutable state | one max-revision reader composed with operational state | replaced | old-value absence after promotion and restart |
| daily promoted-reflection input | reflection promotion and context builder | independent daily identity evaluation plus typed V2 evidence | retained with repaired contract | invalid-output regeneration and memory-write-independent invocation |
| gradual global trait accumulation | `global_character_growth` candidate plus EMA drift | root-counted candidate ledger with three episodes/two dates | superseded by reviewed identity revision | baseline/V2 pace artifact, hold/promote, and anti-oscillation tests |
| prompt-visible promoted guidance | `promoted_global_growth` inside promoted reflection context | latest identity constraints plus bounded `promoted_reflection` evidence | retained with split authority | exact consumer projection and behavior attribution |
| operator audit/history | trait rows and growth-run rows | immutable revisions, candidates, sanitized runs, and health projection | retained and strengthened | console lineage, failure-funnel, and redaction proof |
| cross-scope character continuity | global promoted guidance plus scoped residue | one global identity revision; residue remains scoped | clarified | private-to-group and group-to-private abstraction tests |

The old growth-axis vocabulary has no parallel runtime ledger. It is accepted
only as evidence semantics and must resolve through the reviewed canonical
identity surface:

| Baseline axis | Allowed V2 destination families |
|---|---|
| `boundary_timing` | boundary profile; personality defense |
| `guarded_care` | personality logic/defense/quirks; self-image |
| `playful_challenge` | personality quirks; counter-questioning/direct assertion/rhythmic bounce |
| `recovery_style` | boundary recovery; personality defense; self-image growth edges |
| `clarity` | personality logic; direct assertion/formalism avoidance/abstraction reframing |
| `emotional_exposure` | emotional leakage; personality defense; self-image |
| `trust_calibration` | relational override/control-intimacy misread/authority skepticism; self-image |
| `other_communication` | personality or linguistic texture only when proposal and review judge it durable identity |

The closure gate fails if a baseline row lacks a V2 owner, explicit
disposition, focused test, or runtime evidence. Checkpoint H cannot delete
legacy code until the parent signs the zero-gap matrix.

### Supported Semantic Override Surface

| Category | Canonical paths | Consumers |
|---|---|---|
| Core | `name`, `description`, `gender`, `age`, `birthday`, `backstory` | identity/goal context, naming, dialog, relevant visual context |
| Personality | `personality_brief.mbti`, `.logic`, `.tempo`, `.defense`, `.quirks`, `.taboos` | goal, moral/existential/social appraisal, text expression |
| Boundaries | `boundary_profile.self_integrity`, `.control_sensitivity`, `.compliance_strategy`, `.relational_override`, `.control_intimacy_misread`, `.boundary_recovery`, `.authority_skepticism` | relevant appraisal, progress, boundary judgment |
| Linguistic texture | `linguistic_texture_profile.fragmentation`, `.hesitation_density`, `.counter_questioning`, `.softener_density`, `.formalism_avoidance`, `.abstraction_reframing`, `.direct_assertion`, `.emotional_leakage`, `.rhythmic_bounce`, `.self_deprecation` | text expression only |
| Self-image | `self_image.self_concept`, `self_image.current_growth_edges` | moral/existential appraisal, goals, console |
| Visual | `visual_characterization` | visual surface only |

`tone` and `speech_patterns` retire because they duplicate `personality_brief.tempo` and `linguistic_texture_profile`. Tests prove absence instead of retaining a second expression vocabulary. `_id`, `global_user_id`, `cognition_state`, timestamps, platform/account IDs, permissions, limits, schemas, revisions/evidence IDs, and delivery settings are forbidden paths.

## Design Decisions

### Revision Authority And Seed

`character_identity_revisions` stores immutable full snapshots with unique `(character_id, revision_number)`. Numbers start at `0`; max revision is active; there is no mutable pointer.

Revision `0` requires full canonical profile content. All selected/bundled profile files add nonempty `self_image.self_concept`, zero-to-five bounded `self_image.current_growth_edges`, and nonempty `visual_characterization`. Bootstrap and runtime never derive them from old DB content. `character_state` retains only operational cognition state/timestamps. Graph profiles compose latest identity with operational character ID/state.

Every successful non-seed insert emits `CacheInvalidationEvent(source="character_identity", global_user_id=character_id)`, refreshes the service's adapter/name snapshot, and invalidates RAG/local-context character-profile dependencies. `get_character_profile()` becomes a composition facade over the one canonical latest-revision reader plus operational state. Self-cognition providers resolve that facade asynchronously for each new case instead of retaining a startup snapshot.

### Proposal, Review, And Pace

The existing consolidation router may select `character_identity_growth` for durable character-owned identity evidence. Selection never promotes.

Both stages use closed JSON. `IdentityProposalInputV1` contains only latest prompt-safe identity, one bounded `IdentityEvidenceCardV1` set, up to eight current-base candidates, opaque evidence handles, and allowed paths. `IdentityReviewInputV1` adds the parsed proposal to the same input. Unknown keys fail. The existing router/reflection producing LLM authors each card from its already authorized same-scope input: root/date/scope class plus bounded decontextualized event, character-cognition summary, and visible self-expression summary; participant names/IDs, channel IDs, exact user/character text, raw transcript/reflection/residue, and unrelated RAG facts are excluded before identity stages.

```text
IdentityProposalDecisionV1
  schema_version; action=no_change|explicit_self_redefinition|inferred_growth|corroborate_candidate
  candidate_id:str|null; proposed_changes:0..5 IdentityPatchV1
  character_authorship=self_declared|inferred|absent
  identity_relevance=durable|ephemeral|absent; global_applicability=global|scoped|absent
  confidence=low|medium|high; private_detail_risk=low|high
  character_owned_abstraction; evidence_ref_ids; contradiction_candidate_ids; reason_code
IdentityReviewDecisionV1
  schema_version; verdict=accept|reject|no_change
  selected_candidate_id:str|null; rejected_candidate_ids
  accepted_change_kind=explicit_self_redefinition|inferred_growth|null
  accepted_changes:0..5 IdentityPatchV1
  character_authorship; identity_relevance; coherence=coherent|conflicting|absent
  global_applicability; review_confidence=low|medium|high; private_detail_risk=low|high
  character_owned_summary; privacy_safe_evidence_summaries; reason_code
```

Proposal and review independently own authorship, relevance, scope, contradiction, and semantic privacy. Only a low-risk reviewer-authored character abstraction/summaries may persist. Deterministic validation owns closed keys, bounds, opaque-handle membership, prohibited identifier fields, paths/types, and exact patch equality; it never manufactures or rewrites semantic content.

Immediate promotion requires matching explicit classifications, high confidence from both stages, repository-linked character cognition/visible utterance, `character_authorship=self_declared`, low privacy risk, removed user detail, valid patch, and current base.

Inferred promotion requires high review confidence, low privacy risk, three distinct settled episode roots, and two local dates. Defaults and bounds:

| Setting | Default | Bounds |
|---|---:|---:|
| `CHARACTER_IDENTITY_GROWTH_ENABLED` | `true` | boolean |
| `CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_EPISODES` | `3` | `2..8` |
| `CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_LOCAL_DATES` | `2` | `1..7`, not above episode threshold |
| `CHARACTER_IDENTITY_GROWTH_MAX_INFERRED_PROMOTIONS_PER_LOCAL_DAY` | `1` | `0..3` |
| `CHARACTER_IDENTITY_GROWTH_PROMPT_CHAR_BUDGET` | `18000` | `8000..30000` |

Settings load at startup, appear in Brain service configuration, are audited, and apply by restart. This starts faster than baseline drift while preserving corroboration.

The pace comparison is explicit rather than rhetorical. Under the baseline
EMA constants, one new highest-confidence confirming date per pass reaches the
prompt-visible `promoted` band on the tenth date. V2 begins at the
user-approved three-root/two-date threshold and one inferred promotion per
local day. A deterministic raw calibration artifact records both curves, and
the longitudinal pilot records actual episode/date latency, rejection rate,
and revision frequency. Completion requires evidence that the default both
holds on one-off noise and eventually promotes sustained evidence.

### Evidence Roots, Derived Reflection, And Reversal Stability

Every eligible item resolves to one repository-owned `root_episode_id`.
Episode evaluation references that root directly. Daily reflection evidence
contains one ref per contributing root plus the derivative reflection run ID.
The same root can enrich semantic review through several cards but contributes
exactly one count to one candidate. Retries and duplicate delivery reuse the
same ref and emit duplicate dispositions.

The live episode ID is persisted as `source_episode_id` on its user and
character conversation rows. `db.conversation_reflection` projects that opaque
field; hourly reflection runs store sorted unique source roots; daily
channel/global runs recursively union those roots while separately retaining
derivative run IDs. A reflection row without resolvable repository roots is
ineligible and records `no_eligible_evidence`; timestamps, message IDs, and
reflection-run IDs never substitute for a root.

Candidate matching is semantic and LLM-owned. Root claiming, date counting,
and duplicate prevention are deterministic. A candidate may contain up to five
coherent patches, so one root never needs to be claimed by parallel
field-specific candidates.

Closed transitions are `emerging -> emerging|ready|rejected|superseded`,
`ready -> promoted|rejected|superseded`, and no transition out of a terminal
state. Competing same-path/current-base candidates enter one review set; the
review names at most one winner and all incompatible candidates are rejected
or superseded in the same transaction.

For an inferred change to reverse a path changed by revision `N`, all counting
roots for the reversing candidate must be newer than revision `N`. They must
again meet the configured episode/date threshold. Evidence already used by a
promoted or rejected candidate cannot be recycled into another inferred
promotion. Explicit character-authored turning points may reverse immediately
after independent review. Tests cover repeated text with different episode IDs,
the same episode represented through reflection, unrelated episodes,
contradictory candidates, and consecutive-day flip attempts.

### Growth Health And Empty-State Diagnosis

The existing event-log owner exposes one keyword-only identity recorder with
closed `event_type=routing|proposal|review|policy|promotion|consumption`,
stage/reason/status, correlation/run IDs, revision number, and counts. Its
sanitizer rejects other keys. The identity module persists only revisions,
candidates, and growth runs; durable consumption belongs to the run, so no
fourth growth collection is added.

Closed reason codes are:

```text
not_routed
no_eligible_evidence
proposal_no_change
proposal_contract_failed
candidate_emerging
candidate_ready
review_rejected
review_contract_failed
privacy_blocked
cadence_wait
duplicate_root
stale_base
contradiction_blocked
promotion_write_failed
revision_promoted
awaiting_first_consumption
revision_consumed
revision_consumption_mismatch
```

The console derives one public health state:

```text
healthy_idle
waiting_for_evidence
semantic_rejection
promotion_ready
awaiting_consumption
healthy_active
pipeline_error
consumption_error
```

Health uses this exact precedence: latest-revision receipt mismatch
`consumption_error`; unresolved failed contract/persistence/scheduler run
`pipeline_error`; latest non-seed revision without receipt
`awaiting_consumption`; ready candidate `promotion_ready`; latest routed
semantic/privacy/contradiction rejection `semantic_rejection`; emerging,
duplicate, or cadence-held candidate `waiting_for_evidence`; matching durable
receipt `healthy_active`; otherwise `healthy_idle`. Counts and latest reason
remain visible without content. This funnel answers “no data or broken process?”

### Typed Patches

`IdentityPatchV1` is a strict tagged union:

```text
text:          path, value_kind="text", replacement_text
integer:       path="age", value_kind="integer", replacement_integer
numeric:       path, value_kind="semantic_band", replacement_band
closed enum:   path, value_kind="closed_enum", replacement_enum
text list:     path="self_image.current_growth_edges",
               value_kind="text_list", replacement_items
```

Numeric fields are prompt-visible as semantic descriptors. Mapping is `very_low=0.1`, `low=0.3`, `medium=0.5`, `high=0.7`, `very_high=0.9`. This is declared normalization; the LLM owns field/band selection. Text is bounded/trimmed, not rewritten. Unknown keys, union conflicts, duplicate/forbidden paths, invalid enums, missing evidence, and stale bases regenerate or fail closed by error class.

### Natural V2 Projection

Latest identity is authoritative constraint, not RAG evidence. Closed projection
keys are: `moral_identity={core,personality,boundaries,self_image}`;
`existential_drive={core,personality,self_image}`;
`relationship_social={personality,boundaries}`;
`event_agency={personality,boundaries}`;
`goal_threat_outcome={personality,boundaries}`;
`epistemic_comparison_memory={core}` only for character-self questions and `{}` otherwise;
`text={name,personality.tempo,personality.defense,personality.quirks,linguistic_texture_profile}`;
`visual={name,description,gender,age,visual_characterization}`; and
`naming={name}`. Each consumer receives exactly its closed object; revision,
evidence, history, and unlisted categories stay outside prompts.

Existing `promoted_reflection_context` becomes bounded `source_kind=promoted_reflection` V2 evidence. It cannot directly override identity. RAG `character_image`, if retained, remains evidence and cannot outrank latest identity.

### Privacy-Safe Carry-Over

Internal evidence refs support audit/distinctness. A revision stores typed diffs, character-owned summary, coarse scope kinds (`private`, `group`, `reflection`, `self_cognition`, `operator`), privacy-safe evidence summaries, and internal refs hidden from cognition/redacted in console.

Character Carry-over displays revision continuity. User/Group pages retain scoped residue. No private text enters group prompts and no participant fact enters private prompts.

### Reflection Repair

Promotion must parse canonically, validate the whole decision, request full replacement for structural/enum/missing/conflicting output, cap at three attempts, persist errors/attempt/final disposition, and keep invalid candidates out of writes. The prompt preserves `self_guidance -> defense_rule`.

Daily identity evaluation runs from validated daily evidence regardless of memory promotion write/merge/dedup/reject/no-action. Memory and identity have separate result counters and owners.

## Data Migration

No historical migration exists.

Clean bootstrap creates only the required collections and indexes. Before
service startup, the operator loader validates the selected canonical profile
and inserts revision `0`. Startup requires an existing identity revision before
it creates operational `character_state`; it reads no old self-image, traits,
runs, reflection, conversations, or residue.

Any database without an identity revision fails before intake, including a
clean database and an existing DB with semantic `character_state`. Runtime
never silently derives revision `0`.

The invalidated Stage 4 plan is never executed. When the user separately
authorizes clean production construction under this plan, the target starts
empty and receives only the operator-selected canonical profile as revision
`0`; a later service start creates fresh operational state. Construction never
opens the historical source
database and never copies `character_state.self_image`,
`global_character_growth_traits`, `global_character_growth_runs`,
`character_reflection_runs`, conversations, or
`internal_monologue_residue`. The historical database remains untouched and
the target has no legacy growth collections.

## Contracts And Data Shapes

### Effective Identity

```text
CharacterEffectiveIdentityV1
  name:str<=160; description:str<=2400; gender:str<=120
  age:int 0..10000; birthday:str<=160; backstory:str<=6000
  personality_brief:
    mbti:str<=80; logic/tempo/defense:str<=1200
    quirks/taboos:str<=1600
  boundary_profile:
    self_integrity/control_sensitivity/relational_override/
    control_intimacy_misread/authority_skepticism: float 0..1
    compliance_strategy: resist|evade|comply
    boundary_recovery: rebound|delayed_rebound|decay|detach
  linguistic_texture_profile:
    fragmentation/hesitation_density/counter_questioning/
    softener_density/formalism_avoidance/abstraction_reframing/
    direct_assertion/emotional_leakage/rhythmic_bounce/
    self_deprecation: float 0..1
  self_image:
    self_concept: nonempty str<=2400
    current_growth_edges: 0..5 nonempty str<=400
  visual_characterization: nonempty str<=3000
```

All listed keys are required; unknown keys fail.

### Evidence, Revision, Candidate, Run, And Health

```text
IdentityEvidenceRefV1
  schema_version="character_identity_evidence_ref.v1"
  evidence_ref_id; root_episode_id; correlation_id
  source_kind=settled_episode|daily_reflection
  derived_reflection_run_ids:sorted unique list
  character_local_date; scope_kind=private|group|self_cognition
  captured_at

CharacterIdentityRevisionV1
  schema_version="character_identity_revision.v1"
  revision_id; character_id; revision_number>=0
  revision_kind=seed|explicit_turning_point|corroborated_growth|operator_reset
  base_revision_number:int|null
  effective_identity:CharacterEffectiveIdentityV1
  changed_paths:sorted unique list
  change_diff:list[IdentityChangeDiffV1]
  evidence_summary; source_scope_kinds; evidence_refs
  promotion_run_id; promotion_correlation_id
  proposal_confidence=seed|high|operator; review_confidence=seed|high|operator; created_at

CharacterIdentityGrowthCandidateV1
  schema_version="character_identity_growth_candidate.v1"
  candidate_id; character_id; base_revision_number
  status=emerging|ready|promoted|rejected|superseded
  change_kind=explicit_self_redefinition|inferred_growth
  proposed_changes; semantic_summary; evidence_refs
  distinct_episode_count; distinct_local_dates; source_scope_kinds
  claimed_root_episode_ids; newest_root_captured_at
  reversal_of_paths; fresh_post_revision_root_count
  character_authorship=self_declared|inferred|absent
  proposal_confidence/review_confidence=low|medium|high
  privacy_review; promoted_revision_number:int|null
  rejection_reason; created_at; updated_at

CharacterIdentityGrowthRunV1
  schema_version="character_identity_growth_run.v1"
  run_id; run_kind=episode|daily_reflection|operator_reset; base_revision_number
  correlation_id; root_episode_ids
  source_evidence_count; attempt_count_by_stage
  lifecycle_state=in_progress|committed|post_commit_pending|complete|failed
  disposition=no_change|candidate_updated|revision_promoted|
               rejected|failed|deferred
  proposal_reason_code; review_reason_code; policy_reason_code
  persistence_reason_code
  candidate_id; promoted_revision_number:int|null
  validation_error_codes; first_consumption:IdentityConsumptionReceiptV1|null
  post_commit_attempt_count; started_at; completed_at

IdentityConsumptionReceiptV1
  episode_id; correlation_id; claimed_at; loaded_revision_number
  consumer_kinds:sorted unique list
  projection_digest; status=consumed|mismatch

CharacterIdentityGrowthHealthV1
  state=healthy_idle|waiting_for_evidence|semantic_rejection|
        promotion_ready|awaiting_consumption|healthy_active|
        pipeline_error|consumption_error
  routed_count; no_change_count; emerging_candidate_count
  ready_candidate_count; rejected_count; failed_count
  promoted_count; consumed_count
  latest_revision_number; latest_consumed_revision_number
  latest_reason_code; root_count; local_date_count
```

Revision `0` is `seed`, has empty diff/no runtime refs, seed confidence, and no
run. Growth revisions require current base, changed paths, high confidences,
and repository-owned roots. `operator_reset` instead requires an explicit
`--force` action ID, a linked `operator_reset` run with no evidence, operator
confidence, and a fully validated snapshot; it invokes no semantic stage.
Promoted candidates cannot regress. One root can be claimed by one candidate
and appears once regardless of derivatives. Raw prompts/output, messages,
user/scope IDs, private facts, identity text, and raw diffs are forbidden in
runs, receipts, and telemetry.

`CharacterIdentityGrowthHealthV1` is a redacted derived projection over the
three growth collections; TTL event logging is observability only. It is never
persisted as a fourth identity-growth collection.

### Public Interfaces And Indexes

```python
async def ensure_seed_identity(
    *, character_id: str, seed: Mapping[str, object],
) -> CharacterIdentityRevisionV1: ...

async def get_current_identity(
    *, character_id: str,
) -> CharacterIdentityRevisionV1: ...

async def evaluate_episode_identity_growth(
    *, settled_episode: Mapping[str, object],
    current_revision: Mapping[str, object],
) -> IdentityGrowthEvaluationResultV1: ...

async def run_reflection_identity_growth_pass(
    *, character_local_date: str,
    source_reflection_run_ids: Sequence[str],
    dry_run: bool, enable_revision_writes: bool,
    now: datetime | None = None,
) -> IdentityGrowthEvaluationResultV1: ...

def project_identity_for_cognition(
    revision: Mapping[str, object],
) -> CharacterIdentityCognitionContextV1: ...

def project_identity_for_surface(
    revision: Mapping[str, object],
) -> CharacterIdentitySurfaceContextV1: ...

def project_identity_for_console(
    revision: Mapping[str, object],
) -> dict[str, object]: ...

async def build_identity_growth_health(
    *, character_id: str,
) -> CharacterIdentityGrowthHealthV1: ...
```

`IdentityGrowthEvaluationResultV1` has `status=not_routed|no_change|candidate_updated|revision_promoted|rejected|failed|deferred`, run/candidate IDs,
base/promoted revision numbers, four stage reason codes, and validation error
codes. Cognition/surface contexts are the exact closed consumer objects above.

Only `db.character_identity_growth` accesses raw collections. Required indexes:

```text
character_identity_revision_id_unique
character_identity_character_revision_unique
character_identity_character_revision_desc
character_identity_candidate_id_unique
character_identity_candidate_character_status_updated
character_identity_candidate_base_status
character_identity_candidate_character_root_unique
character_identity_run_id_unique
character_identity_run_kind_completed
character_identity_run_revision
```

Promotion requires a Mongo transaction: re-read max revision/current base,
conditionally match the ready candidate by status/base, insert revision `N`,
transition that candidate and its incompatible competitors, and finalize the
run. Any write/commit error aborts all three changes. Retry by `run_id` returns
an already consistent commit or repeats the transaction; any partial mismatch
fails closed. After commit the run remains `post_commit_pending` until
idempotent revision-keyed invalidation/event emission succeeds; startup and
the background worker reconcile pending rows. Before cognition, a transaction
compares loaded/max revision and conditionally claims the run's empty receipt;
a stale load retries once, then fails closed. Failure injection covers every
write boundary, commit, post-commit replay, restart, and concurrent claim.

## LLM Call And Context Budget

The cap is the project-default 50,000-token context window. Estimates use the
conservative upper bound of one token per Unicode character. The existing
`CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS` default is 8,192, so every listed
prompt plus maximum completion remains below the cap.

| Path | Before | After | Before/after context | Hard maximum | Latency/blocking |
|---|---:|---:|---|---|---|
| foreground cognition and surfaces | existing calls | unchanged | adds only bounded latest-identity projection to relevant existing calls | existing per-stage caps; no cap increase | no new foreground call |
| episode with no identity route | router only | router only | unchanged router evidence roster | existing router cap | no added work |
| routed settled episode | router only | router + proposal + review | after: current prompt-safe identity, one sanitized root evidence set, up to eight current-base candidates, opaque handles, allowed paths | each new prompt ≤18,000 chars/tokens plus ≤8,192 completion; <26,200 total | two sequential background calls after response settlement |
| reflection memory promotion | one promotion attempt | same stage with at most three total attempts on contract failure | same validated daily cards and promotion contract; replacement attempts reuse identical semantic context | prompt ≤25,000 chars/tokens plus ≤8,192 completion; <33,200 total | background only; attempts capped at three |
| daily identity evaluation | none | proposal + review | current prompt-safe identity, validated root-linked daily evidence, up to eight current-base candidates | each prompt ≤18,000 chars/tokens plus ≤8,192 completion; <26,200 total | two sequential background calls |

Identity stages reuse consolidation model/API/token/thinking settings,
temperature `0.2`, and stable system policy. Human payload contains only the
current bounded identity, sanitized evidence, at most eight candidates, and
opaque handles. Limits are one current identity, one episode/daily set, twelve
evidence cards at 400 characters, and five changes. On overflow, drop optional
older candidates before current evidence; never truncate JSON, the current
identity, or current evidence. Numeric profile values reach the LLM only as
semantic bands. Routing, cadence, permission, persistence, cache, and adapter
facts stay outside prompts.

## Change Surface

Target ownership boundary: `kazusa_ai_chatbot.character_identity_growth` owns
identity semantics/orchestration and `db.character_identity_growth` owns its
storage. Changes outside those modules are limited to required producers,
consumers, invalidation, observability, operator presentation, and big-bang
legacy removal.

### Create

```text
src/kazusa_ai_chatbot/character_identity_growth/{__init__.py,README.md,models.py,validation.py,identity.py,projection.py,llm.py,policy.py,runner.py}
src/kazusa_ai_chatbot/db/character_identity_growth.py; src/scripts/run_character_identity_growth.py
tests/test_character_identity_growth_contract.py; tests/test_character_identity_growth_validation.py; tests/test_character_identity_growth_policy.py
tests/test_character_identity_growth_runner.py; tests/test_character_identity_growth_projection.py; tests/test_character_identity_growth_module_boundary.py
tests/test_character_identity_growth_causal_lineage.py; tests/test_character_identity_growth_observability.py; tests/test_character_identity_growth_longitudinal_policy.py
tests/test_character_identity_growth_live_llm.py; tests/test_character_identity_growth_live_db.py; tests/test_character_identity_growth_integration.py
tests/test_character_identity_growth_behavior_e2e_live_llm.py; tests/test_character_identity_growth_normal_entrypoint_live_llm.py
tests/fixtures/character_identity_growth_override_cases.json
```

The new package provides the one public growth boundary. Contract, lineage,
observability, live-DB, controlled counterfactual, and normal-entrypoint test
files each own one acceptance risk. Execution writes raw JSON/log evidence and
parent-authored Markdown reviews beneath
`test_artifacts/character_identity_growth/`, prefixing run-scoped filenames
with the actual UTC timestamp and short HEAD. These are evidence artifacts,
not production modules or additional plans.

### Modify

The following groups are justified outside the target module:

- root docs and canonical profile files define the new operator contract and
  complete revision-0 seed;
- service/profile/DB/bootstrap files remove static semantic authority and
  expose one latest reader plus clean persistence;
- brain-service/consolidation/reflection files provide settled evidence,
  repaired promotion, and background invocation;
- V2/node/self-cognition/RAG/cache files are the authorized latest-identity
  consumers and refresh dependencies;
- event logging records sanitized funnel and consumption events without a new
  growth collection;
- console files present public redacted lineage, health, and pace;
- existing tests and fixtures move their expectations to the one canonical
  boundary;
- `development_plans/README.md` records lifecycle state only.

```text
README.md; docs/HOWTO.md
personalities/{asuna.json,example.json,kazusa.json,qingche.json}; src/kazusa_ai_chatbot/character_profiles/example.json
src/kazusa_ai_chatbot/{character_profile.py,config.py,state.py,service.py,memory_writer_prompt_projection.py}
src/kazusa_ai_chatbot/db/{__init__.py,bootstrap.py,character.py,conversation.py,conversation_reflection.py,event_logging.py,schemas.py,README.md,script_operations.py}
src/kazusa_ai_chatbot/brain_service/{outbound.py,post_turn.py,README.md}
src/kazusa_ai_chatbot/consolidation/{__init__.py,core.py,lane_router.py,origin_policy.py,persistence.py,schema.py,source_policy.py,target.py,README.md}
src/kazusa_ai_chatbot/reflection_cycle/{models.py,promotion.py,worker.py,context.py,selector.py,repository.py,projection.py,activity_windows.py,prompts.py,README.md}
src/kazusa_ai_chatbot/cognition_core_v2/{contracts.py,state_projection.py,semantic_appraisal.py,goal_cognition.py,README.md}
src/kazusa_ai_chatbot/nodes/{persona_supervisor2_schema.py,persona_supervisor2_cognition.py,persona_supervisor2_l3_surface.py,persona_supervisor2_rag_prompt_views.py,boundary_profile.py,linguistic_texture.py,README.md}
src/kazusa_ai_chatbot/internal_monologue_residue/{models.py,loader.py,recorder.py,README.md}; src/kazusa_ai_chatbot/self_cognition/{sources.py,worker.py}
src/kazusa_ai_chatbot/rag/{cache2_policy.py,README.md}; src/kazusa_ai_chatbot/rag/person_context/projection.py; src/kazusa_ai_chatbot/rag/person_context/workers/profile.py
src/kazusa_ai_chatbot/local_context_resolver/cache.py; src/kazusa_ai_chatbot/event_logging/{__init__.py,recording.py,sanitization.py,README.md}
src/scripts/{_lane_cleanup.py,character_state_snapshot.py,load_character_profile.py,sanitize_memory_writer_perspective.py,README.md}
src/control_console/{app.py,repository.py,service_config.py,README.md}; src/control_console/static/{index.html,console.js,console.css}
tests/{test_character_profile_seed.py,test_character_profile_clean_start_live_db.py,test_character_state_snapshot.py,test_service_background_consolidation.py}
tests/{test_consolidation_lane_router_contract.py,test_consolidation_target_routing.py,test_consolidation_source_policy.py,test_consolidation_origin_policy.py,test_consolidation_lane_bigbang_integration.py}
tests/{test_reflection_cycle_stage1c_promotion.py,test_reflection_cycle_stage1c_promotion_live_llm.py,test_reflection_cycle_stage1c_worker.py,test_reflection_cycle_stage1c_service.py,test_reflection_cycle_stage1c_repository.py,test_reflection_cycle_stage1c_reflection_context.py,test_reflection_cycle_activity_windows.py,test_reflection_cycle_readonly.py}
tests/{test_cognition_core_v2_contracts.py,test_cognition_core_v2_projection.py,test_cognition_core_v2_integration.py,test_cognition_live_llm_prompt_contracts.py}
tests/{test_internal_monologue_residue_loader.py,test_internal_monologue_residue_recorder.py,test_internal_monologue_residue_integration.py,test_self_cognition_group_review_source.py,test_user_profile_agent.py}
tests/{test_rag_projection.py,test_rag_cache2_persistent.py,test_local_context_resolver_cache.py,test_db_writer_cache2_invalidation.py}
tests/{test_memory_writer_database_sanitizer.py,test_memory_writer_information_flow_contracts.py,test_memory_writer_prompt_projection.py,test_memory_writer_prompt_contracts.py,test_memory_writer_perspective_live_llm.py,test_config.py,test_db.py,test_event_logging_interface.py,test_llm_time_payload_projection.py}
tests/fixtures/cognition_llm_producer_matrix.json
tests/{test_control_console_repository.py,test_control_console_review_edges.py,test_control_console_cognition_debug_visibility.py,test_control_console_service_config.py,test_control_console_web_surface.py}
tests/control_console_e2e/{test_live_database_owner_pages_e2e.py,test_clickable_inventory_e2e.py,test_page_navigation_e2e.py}
development_plans/README.md
```

### Delete

```text
src/kazusa_ai_chatbot/consolidation/images.py; src/kazusa_ai_chatbot/global_character_growth/ (all files)
src/kazusa_ai_chatbot/db/global_character_growth.py; src/scripts/run_global_character_growth.py
tests/test_consolidator_character_image.py
tests/{test_global_character_growth_context.py,test_global_character_growth_contract.py,test_global_character_growth_drift.py,test_global_character_growth_live_llm.py}
tests/{test_global_character_growth_module_boundary.py,test_global_character_growth_prompt_contracts.py,test_global_character_growth_replay.py,test_global_character_growth_runner.py,test_global_character_growth_validation.py,test_global_character_growth_worker.py}
```

### Keep

V2 cognition state/reducers; scoped residue; user relationship/memory/style/progress; shared memory/lore/self-guidance; RAG evidence ownership; foreground response/action/delivery/scheduler boundaries; protected LLM traces; and the archived invalidated Stage 4 plan body as historical evidence.

Checkpoint A reconciles this inventory with current HEAD. A newly found direct caller is added only to complete the same cutover; unrelated cleanup remains out of scope.

## Overdesign Guardrail

- Actual problem: V2 has disconnected self-image/growth producers and consumers, so the active character cannot accumulate reviewed global identity change or demonstrate that real interaction changed later cognition.
- Minimal change: replace the disconnected self-image and soft-growth paths with one immutable latest-identity ledger, one root-counted candidate flow, two background semantic judgments, one redacted health projection, and direct V2/surface consumers.
- Ownership boundaries: proposal/review LLMs own identity meaning, authorship, candidate matching, contradiction, and privacy-safe abstraction; deterministic code owns root lineage, cadence, path/type validation, persistence, concurrency, refresh, telemetry, and limits; cognition owns stance/goals; surfaces own wording/visual rendering; adapters remain transport-only.
- Rejected complexity: retain exactly three growth collections and one full-snapshot revision form; add no event sourcing, active pointer, branch/merge/patch replay, fourth receipt store, compatibility import, deprecated alias, shadow profile, automatic legacy conversion, historical migration script, direct console approval, extra semantic agent, or foreground call. Keep at most eight candidates and five patches per evaluation.
- Evidence threshold: add a rejected capability only after a concrete production failure or separately approved integration proves the three-collection/root-lineage/full-snapshot contract cannot meet correctness, privacy, audit, or latency requirements.

## Agent Autonomy Boundaries

After approval plus explicit implementation command, execution may edit listed
files; create/mutate guarded test databases; run deterministic/live tests and
local test services; create redacted artifacts/screenshots; and update plan
evidence. Local mechanics may vary only when they preserve every contract here.

The responsible agents must search for existing equivalent behavior before
adding helpers, keep outside-module edits to the justifications in Change
Surface, and avoid unrelated cleanup, dependency upgrades, generic prompt
rewrites, alternate migration strategies, compatibility paths, or extra
features.

Separate user authority is required for historical Asuna or production DB
access/mutation; clean production-target construction; `.env`, deployment,
service, database-selection, or adapter changes; production
deploy/restart/cutover; legacy DB/collection mutation; and changes to the field
catalog, default thresholds, privacy rule, or turning-point policy.

Stop and report when the zero-gap closure matrix cannot be signed, canonical
profile cannot validate, DB isolation is unproven, a pilot attempts direct
growth-state writes, root lineage cannot be correlated, private detail can
escape, concurrent history can fork, same-path inferred growth can oscillate,
latest-only behavior is unproven, a promoted revision cannot be tied to its
first consumer, scope expands materially, a test could touch production, or
the parent cannot execute the bounded production surface safely. If
the plan and code disagree, preserve the plan intent and report the discrepancy.

## Implementation Order

### A — Rebaseline

- Actions: record branch, HEAD, status, direct callers, DB evidence counts, and Stage 4 lifecycle; run every exact baseline command under Verification; author the closure matrix; freeze paths, reasons, health, and root rules in failing tests.
- Verify: ten capability rows, eight axis rows, no blank owner/disposition/test/evidence cell, and no unresolved instruction.
- Exit: rebaseline record, exact source commands, matrix, parent zero-gap sign-off, and next failing selector.

### B — Contract Tests And Models

- Tests: add failing contract, validation, causal-lineage, observability, and module-boundary cases covering full identity, tagged patches, transitions, root/derivative dedupe, reasons, health, privacy, forbidden fields, latest projection, and three-collection ownership.
- Implement: after recording failures, the production subagent changes only `models.py`, `validation.py`, `identity.py`, and public exports.
- Exit: the same focused selectors pass without integration; retain commands, failures/passes, changed files, and contract-diff review.

### C — Persistence, Seed, Operational Split

- Tests: first add failing clean/legacy startup, immutable history, root claim, concurrent promotion, max reader, operator reset, and restart cases.
- Implement: DB owner/indexes, complete canonical profiles, manual revision-`0` loading, operational-only `character_state`, and revisioned `--force`; any startup without a ledger fails before intake.
- Exit: guarded live DB proves one winner, immutable/max-only history, root uniqueness, clean restart/reset, no legacy collection, and no historical-source access; retain DB identity, indexes/counts, and output.

### D — Proposal, Review, Policy

- Tests: add explicit authorship, user imposition, inferred corroboration, root/derivative/repetition/unrelated evidence, contradiction, stale base, reversal, privacy, no-change, invalid JSON, and invalid-contract cases.
- Implement: bounded prompt-safe separate proposal/review stages; stable system policy, dynamic human identity/evidence, numeric descriptors, canonical parsing, at most three full replacements, and sanitized outcomes.
- Exit: inspect each live selector separately and retain explicit/inferred/rejection/privacy/reversal results, attempt dispositions, prompt renders, call/context measurements, readable review, and proof of no deterministic semantic override.

### E — Consolidation And Reflection

- Tests: add failing router/source/origin/target/persistence cases for settled routing, sanitized non-route/no-evidence, derivative roots, and memory-outcome independence.
- Implement: post-settlement lane, repository root refs, derivative reflection refs without extra counts, bounded promotion repair, and daily identity invocation independent of memory outcome; remove the legacy writer after focused integration passes.
- Exit: one identity owner, zero foreground calls, derivative-root hold, and independent daily invocation.

### F — Latest-Only Runtime

- Tests: add failing service/profile/cache/self-cognition/V2/text/visual/name cases holding non-identity context fixed while varying only revision.
- Implement: sole latest reader plus operational state, identity invalidation, name/dependent-cache refresh, per-episode latest resolution, relevant bounded projections, and durable first-consumption receipt plus mirrored telemetry; exclude history, metadata, and old values.
- Exit: revision `N` present, `N-1` absent, consumers correct, restart/cache correct, and status `consumed`.

### G — Console And Pace

- Tests: add failing repository/API/config/web cases for revision history, candidates, root/date counts, reasons, health, pace, and redaction.
- Implement: Character identity lineage/health, retained User/Group residue, every declared health state, and bounded controls without raw IDs/text/prompts/private facts; generate raw pace data and parent-authored comparison.
- Exit: authenticated real clean-test data, valid bounds, redacted network payloads, desktop/narrow screenshots, and zero browser errors.

### H — Big-Bang Cleanup

- Gate: reopen the matrix, attach passing focused/runtime evidence to every row, and obtain parent pre-deletion sign-off.
- Implement/verify: delete every listed legacy owner/import/flag/script/test/active doc, retire duplicate expression fields, update docs, preserve the archived Stage 4 body, and run static/boundary gates.
- Exit: zero unexplained rows, required zero matches, exactly one latest reader/persistence owner, and no compatibility path.

### I — Deterministic And Live DB

- Run: all focused contract, lineage, observability, policy, integration, reflection, V2, service, console, and guarded live-DB commands plus one fixture per leaf.
- Verify: full snapshot/exact diff/immutable prior/latest consumer/old-value absence/redaction; root-one/two holds, root-three/date-two promotion, daily cap, duplicate/derivative rejection, and fresh-evidence reversal.
- Exit: classify unrelated failures without weakening gates; retain exact commands/counts, leaf matrix, DB results, pace curve, and anti-oscillation outcomes.

### J — Live Behavior And Browser

- Controlled proof: run live cases one at a time; for every category compare three matched revision-0/revision-N samples with identical code/model/episode/non-identity context; inspect projections, appraisals/goals, traces, lineage, and responses; parent authors review from script-emitted raw evidence.
- Natural proof: run the no-direct-write normal-entrypoint causal case and two-actual-date adaptive pilot turn by turn, including mundane holds and sustained growth; inspect each response/log/background result before the next message.
- Browser/exit: complete screenshots/network/redaction review; require relevant category effects, joined explicit/inferred causal bundles, passing pilot, intact privacy, and complete readable reviews.

### K — Review And Closeout

- Map every requirement, decision, risk, and acceptance row to a command/artifact; run unresolved-language, contract, and granularity scans.
- Parent inspects the approved plan, diff, raw/readable evidence, closure, causal proof, realism, privacy, pace, and lifecycle; parent remediates and reruns all affected gates.
- Exit: retain commands/counts/artifacts/findings/fixes/residuals/user judgment; require no critical/high finding, every mapped row passing, and user sign-off before closeout.

## Execution Model

1. Parent owns A and writes B's failing contract tests.
2. After the focused test contract, the parent attempted one bounded production-code subagent; it produced no files or blocker after repeated implementation prompts and was stopped.
3. The user's 2026-07-28 direction to proceed straight to implementation governs the fallback: the parent owns production implementation, tests, verification, artifacts, review, and plan maintenance.
4. The user also waived independent plan/code subagent review and the plan-length gate; no further subagent is required for this execution.

Parallel production writers are prohibited because schema, prompt, runtime, and cutover share one contract.

## Progress Checklist

Each checkpoint records parent identity and ISO date after its evidence is complete, then requires a full-plan reread before handoff.

- [x] A. Rebaseline/closure contract: `/root`, 2026-07-28; retained branch/HEAD/status, exact baseline source commands/blob IDs, Asuna-only diagnostic counts, Stage 4 invalidation, 10 capability/8 axis zero-gap matrix, initial reviewer findings/remediation, user review waiver, and named B selectors.
- [x] B. Identity/root/health/module contracts: `/root`, 2026-07-28; retained the module-not-found red phase, bounded domain implementation, contract-diff review, dotenv-disabled command, and 51/51 passing assertions with clean teardown; hand off to C's persistence tests.
- [x] C. Persistence/seed/operational split: run profile, bootstrap, live-DB, restart, root-claim, and race tests; retain isolated DB, indexes/counts, source non-access, and commands; hand off to D.
- [x] D. Proposal/review/policy: `/root`, 2026-07-28; retained red/green records, 25/25 focused and 111/111 combined non-live assertions, eight individually inspected live cases, 16 raw calls, prompt budgets, zero handle leaks, and parent-readable review; hand off to E.
- [x] E. Consolidation/reflection: `/root`, 2026-07-28; retained the red/green records, 126/126 final guarded selector, settled user/assistant roots, recursive derivative-root hold, bounded promotion replacement, independent daily invocation, and two-caller ownership audit; hand off to F.
- [x] F. Latest-only runtime: `/root`, 2026-07-28; retained red/green records, 463/463 guarded non-live assertions, 8/8 guarded `asuna_core_v2` live-DB assertions, revision `N` presence and `N-1` absence, exact consumer partitions, restart/cache/name refresh, durable `consumed` receipt, mirrored event, and projection digest; hand off to G.
- [x] G. Console/health/pace: `/root`, 2026-07-28; retained red/green records, 56/56 Console/API/config/web assertions, 26/26 policy assertions, 12/12 guarded `asuna_core_v2` live-DB cases, authenticated browser proof, every health state, pace calibration/review, redacted network capture, desktop/narrow screenshots, real restart/reset, zero browser errors, and clean teardown; hand off to H.
- [x] H. Big-bang cleanup: `/root`, 2026-07-29; deleted 12 production source files (8 global_character_growth, consolidation/images.py, db/global_character_growth.py, run_global_character_growth.py, plus memory_writer_prompt_projection function), 10 legacy test files, renamed 1 test; modified 10 production files and 15 test files across 47 total changed files (62 insertions, 5425 deletions); all six static gate searches return zero runtime matches in src/ with only explicit removal assertions in tests/; residue scope `character_global` absent from src/; no legacy schema, import, `__all__`, or compatibility path remains; `test_db.py` legacy character-state tests replaced with V2-compliant assertions (55/55 passed); `test_memory_writer_database_sanitizer.py` updated for fail-closed self-image migration (6/6 passed); `test_memory_writer_perspective_live_llm.py` and `test_memory_writer_prompt_contracts.py` character-image tests and imports removed; 3620 non-live non-browser tests passed with 24 pre-existing failures unrelated to cleanup; hand off to I.
- [x] I. Complete non-behavior verification: `/root`, 2026-07-29; six deterministic gate groups passed 92+62+52+44+51+54=355 focused assertions; identity growth integration 7/7; guarded live-DB 12 skipped (database guard inactive, passed during earlier checkpoints); full non-live sweep 3,637/3,663 passed with 26 pre-existing failures in unrelated modules, 3 skipped, 847 deselected; `git diff --check` clean; hand off to J.
- [ ] J. Behavior/browser proof reopened by Step K: the prior sign-off and two readable reviews were invalidated because they did not execute the claimed full behavior/public causal path and the required actual-date pilot was deferred. Corrected evidence now includes all six categories with three genuine matched V2/cognition/surface/dialog pairs each, a public `/chat` explicit promotion-and-consumption chain, both private/group directions, final semantic relationship cases, and the prior browser gate. The joined inferred bundle and two-actual-character-local-date pilot remain open.
- [ ] K. Parent review/closeout: `/root`, 2026-07-29 completed the technical code/evidence review, resolved every surfaced critical/high implementation finding, and reran all same-date gates. `step_k_review.md` records 22 passing acceptance rows, two partial rows, and two pending rows. Formal K closeout remains open for J's actual-date inferred pilot and user sign-off.

## Verification

### Baseline Closure

```powershell
git show main:src/kazusa_ai_chatbot/consolidation/lane_router.py
git show main:src/kazusa_ai_chatbot/consolidation/images.py
git show main:src/kazusa_ai_chatbot/consolidation/persistence.py
git show main:src/kazusa_ai_chatbot/global_character_growth/README.md
git show main:src/kazusa_ai_chatbot/global_character_growth/drift.py
git show main:src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py
```

Expected: the parent-authored closure matrix cites the exact trigger, producer,
persistence, reader, invalidation/cadence, V2 owner, disposition, test, and
runtime artifact for all ten fixed capability rows and all eight baseline
growth axes. Zero blank or unexplained rows are allowed. The parent signs it
once before implementation and again immediately before legacy deletion.

### Static

```powershell
rg -n "kazusa_ai_chatbot\\.global_character_growth|run_global_character_growth_pass|GLOBAL_CHARACTER_GROWTH" src tests
rg -n "upsert_character_self_image|project_character_image_prompt_payload|global_character_growth_traits|global_character_growth_runs" src tests
rg -n '\bcharacter_global\b' src/control_console src/kazusa_ai_chatbot/internal_monologue_residue tests
rg -n '"(tone|speech_patterns)"\s*:' personalities src/kazusa_ai_chatbot/character_profiles src/kazusa_ai_chatbot/character_profile.py src/kazusa_ai_chatbot/db/schemas.py tests/test_character_profile_seed.py
rg -n -i '\b(asuna|kazusa|qingche)\b' src/kazusa_ai_chatbot/character_identity_growth src/kazusa_ai_chatbot/db/character_identity_growth.py src/scripts/run_character_identity_growth.py
rg -n "character_identity_revisions|character_identity_growth_candidates|character_identity_growth_runs" src tests
rg -n "root_episode_id|first_consumption|waiting_for_evidence|revision_consumed|revision_consumption_mismatch" src tests
rg -n "T[B]D|T[O]DO|m[a]ybe|c[o]nsider|choose[ ]one|similar[ ]to|handle[ ]edge[ ]cases|add[ ]tests" development_plans/active/bugfix/cognition_core_v2_character_identity_growth_bigbang_plan.md
```

Expected: legacy searches have zero runtime/test matches except explicit removal
assertions; Character page/runtime has no exact global-residue scope; retired
profile fields have no canonical matches; new growth code has no named
character/personality literal; new collections appear only through DB
owner/bootstrap/projections/tests/docs; root/health/consumption terms appear in
their declared owners and tests; placeholder search returns zero matches.

### Deterministic

```powershell
venv\Scripts\python.exe -m pytest tests/test_character_identity_growth_contract.py tests/test_character_identity_growth_validation.py tests/test_character_identity_growth_policy.py tests/test_character_identity_growth_runner.py tests/test_character_identity_growth_projection.py tests/test_character_identity_growth_module_boundary.py tests/test_character_identity_growth_causal_lineage.py tests/test_character_identity_growth_observability.py tests/test_character_identity_growth_longitudinal_policy.py -q
venv\Scripts\python.exe -m pytest tests/test_character_profile_seed.py tests/test_character_state_snapshot.py tests/test_service_background_consolidation.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidation_lane_router_contract.py tests/test_consolidation_target_routing.py tests/test_consolidation_source_policy.py tests/test_consolidation_origin_policy.py tests/test_consolidation_lane_bigbang_integration.py -q
venv\Scripts\python.exe -m pytest tests/test_reflection_cycle_stage1c_promotion.py tests/test_reflection_cycle_stage1c_worker.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_contracts.py tests/test_cognition_core_v2_projection.py tests/test_cognition_core_v2_integration.py -q
venv\Scripts\python.exe -m pytest tests/test_internal_monologue_residue_integration.py tests/test_event_logging_interface.py tests/test_control_console_repository.py tests/test_control_console_service_config.py tests/test_control_console_web_surface.py -q
```

### Guarded Live DB

```powershell
venv\Scripts\python.exe -m pytest -m live_db tests/test_character_identity_growth_live_db.py -q -s
venv\Scripts\python.exe -m pytest -m live_db tests/test_character_profile_clean_start_live_db.py -q -s
```

Assert exact manually seeded revision `0`; clean/legacy unseeded startup
fail-closed; one winner under concurrent
promotion; immutable/queryable history; max-only reader; closed candidate
transitions; one candidate claim per root; derivative reflection does not add a
count; fresh post-revision reversal roots; restart persistence; no legacy
target collections; no historical source connection; no Asuna artifact read.

### Override Matrix

One fixture case per leaf asserts proposal/review path/type; full next snapshot;
exact diff; unchanged prior revision; new value in authorized
cognition/surface consumer; old value absent; redacted console diff. The
matrix records the exact consumer and proves that unrelated consumers do not
receive the field.

### Pace And Stability

Deterministic raw evidence records:

- the baseline highest-confidence one-new-date curve from day 1 through its
  first prompt-visible promotion on day 10;
- V2 root 1/date 1 hold, root 2/date 1 hold, and root 3/date 2 promotion;
- same-root episode plus reflection still counts once;
- repeated semantically identical roots are review-rejected as insufficient
  independent corroboration;
- unrelated roots do not merge;
- mundane mood and bounded role-play remain `no_change`;
- no more than one inferred revision per local day;
- a same-path reversal using pre-revision evidence is blocked;
- a same-path reversal promotes only after fresh post-revision roots satisfy
  the full configured threshold.

Parent authors
`test_artifacts/character_identity_growth/pace_calibration_review.md` from the
raw JSON. It reports baseline/V2 latency, proposal/review funnel,
explicit/inferred promotion counts, rejection reasons, per-field revision
frequency, and any oscillation attempt.

### One-At-A-Time Live LLM

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_explicit_self_redefinition_is_character_authored -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_user_imposed_identity_is_rejected -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_inferred_growth_matches_existing_candidate -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_private_detail_is_abstracted_or_rejected -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_repeated_semantics_do_not_fake_independence -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_ephemeral_roleplay_is_rejected -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_contradictory_growth_is_rejected -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_reversal_requires_fresh_evidence -q -s
```

Run every selector in `tests/test_reflection_cycle_stage1c_promotion_live_llm.py` separately.

### Controlled Counterfactual Live Behavior

Run every selector in `tests/test_character_identity_growth_behavior_e2e_live_llm.py` separately against clean isolated identity. Required groups: core, personality, boundary, linguistic, self-image, visual, private-to-group, group-to-private, explicit next-episode, inferred hold/promote, slower/faster pace, rejected user imposition/roleplay, and restart latest-only.

For each identity category, create two guarded DB states with identical code,
model route/configuration, operational state, conversation/RAG context, and
episode input. State A uses revision `0`; state B uses the actually promoted
revision `N`. Run at least three matched samples per state without persistence
drift. Record raw/interpreted input/output, exact identity projection,
appraisal/goal/stance difference, visible response, regressions, privacy
review, and validation result.

Pass requires a relevant and directionally coherent cognition/behavior
difference attributable to the changed category, not merely different prose.
Unrelated categories must remain stable. Schema validity, keyword counts, and
one stochastic response difference do not pass. The parent authors the
side-by-side Markdown review from real outputs.

### Correlated Causal Proof Bundle

For both one explicit turning point and one inferred promotion, retain one raw
bundle containing:

```text
request/message correlation id
settled episode id
root episode ids and local dates
identity route event
proposal trace id and parsed decision
review trace id and parsed decision
candidate id and maturity transition
growth run id and stage/reason codes
base revision and promoted revision diff
cache invalidation event
first later eligible episode id
loaded revision number and consumer kinds
exact bounded cognition/surface projection
cognition trace and appraisal/goal/stance result
visible response
```

All identifiers must join. The new identity value must be present and the old
value absent in the authorized next-episode projection. A persisted revision
without a matching later consumer, or changed prose without matching revision
lineage, fails. The parent authors `causal_growth_review.md`; scripts and tests
write only raw JSON/log/trace evidence.

### Normal-Entrypoint Longitudinal Pilot

Run `tests/test_character_identity_growth_normal_entrypoint_live_llm.py`
selectors individually first, then perform the adaptive pilot through the
normal `POST /chat` debug transport against a clean guarded database:

1. Use one stable simulated user, private channel, character, and bot identity;
   keep `no_remember=false`; enable normal post-turn and background workers.
2. Use real wall-clock character-local dates. Clock injection belongs only to
   deterministic pace tests.
3. Before each turn, present the exact message and observation target to the
   user. Send one turn, save response/log slices, wait for background work,
   inspect cognition/consolidation/growth/DB results, then select the next
   bounded turn from that evidence.
4. Begin with ordinary low-pressure messages that should remain `no_change`.
   Healthy non-growth is required negative evidence.
5. Take the first spontaneous durable identity theme authored by the character
   as the pilot theme. Follow it with open, natural interaction that explores
   the same theme without commanding an identity, naming a desired patch, or
   telling the character to agree.
6. Accumulate at least three semantically independent eligible settled roots
   over at least two actual character-local dates. If the character does not
   produce eligible identity evidence, retain the correct no-change result and
   keep the promotion proof gate open; never seed or rewrite evidence.
7. After promotion, send the first later eligible turn and prove that it loads
   revision `N`, affects the relevant cognition branch, and changes visible
   behavior coherently.
8. Run controlled private-to-group and group-to-private follow-up turns through
   the debug transport; prove only redacted character-owned abstraction crosses
   scope.

The pilot forbids direct calls that insert or edit evidence, candidates, runs,
or revisions. Reads for inspection are allowed. Retain per-turn request,
response, log, trace, DB snapshots, funnel state, causal-chain JSON, and the
parent-authored `longitudinal_pilot_review.md`. The review reports elapsed
dates/episodes, false-positive holds, candidate transitions, promotion
latency, first consumption, behavioral effect, privacy, failures, and user
attention points.

### Console Browser

Using real clean-test data: open Character; verify current self-image/revision;
inspect older revision; inspect each diff category and candidate state; verify
redacted evidence counts/scope kinds and every health state; distinguish
healthy idle from pipeline failure and consumption mismatch; validate every
pace field and restart-required behavior; reject invalid bounds/cross-field
values; verify User/Group scoped residue and Character identity lineage; verify
keyboard/responsive/loading/empty/error/network states; retain desktop/narrow
screenshots; require zero console errors.

### Full

```powershell
venv\Scripts\python.exe -m pytest -m "not live_llm and not live_db" -q
git diff --check
git status --short
```

## Plan Review

The 2026-07-28 parent review established baseline intent, causal/longitudinal
proof, root/pace/reversal rules, observability, and Stage 4 invalidation.
On 2026-07-28 `/root/identity_growth_reviewer` returned `DO NOT APPROVE`: zero critical and seven high findings covering derived-root provenance, atomic
promotion/recovery, operator reset, residue policy, strict semantic/privacy/
projection contracts, durable consumption/health, and executable static gates.
This revision resolves them with persisted generic episode roots, recursive
reflection lineage, transactions/reconciliation, a closed operator exception,
scoped-residue cutover owners, strict two-stage/consumer schemas, run-owned
first-consumption receipts and precedence, and narrowed static checks.
The same reviewer began a full remediation recheck. On 2026-07-28 the user
explicitly waived further subagent checking and directed immediate
implementation. The parent verified the signed closure inventory, change
surface, commands, and zero unresolved choice/placeholder/compatibility or
untestable acceptance; lifecycle advanced through `approved` to `in_progress`.

## Code Review

The parent reviews ownership; Mongo immutability/index/race/startup; root claims and reflection derivatives; LLM parsing/regeneration/no-prepost semantics; explicit/inferred/reversal/contradiction policy; path registry; privacy/redaction; latest-only service/cognition/text/visual/name/restart; consumption receipts; reflection repair/scheduling; health-funnel truthfulness; baseline closure; legacy removal; controlled and natural test realism; causal joins; and production-data isolation.

Critical/high findings block closeout. Parent records findings, remediates, and reruns every affected gate.

## Acceptance Criteria

1. Repository, archived-plan, source, test, and DB evidence establishes that V2 self-growth loss was unintentional and identifies the current breakpoints.
2. The parent-authored baseline-to-V2 closure matrix has all ten capability rows and eight axis rows, zero unexplained gaps, and sign-off before implementation and deletion.
3. The operator loader gives a clean DB immutable revision `0`; startup with
   no ledger crashes before creating operational state, and
   legacy-without-ledger fails without backfill or historical-source access.
4. Latest full revision replaces every supported semantic value; no static/legacy fallback remains.
5. Every canonical leaf passes deterministic override/diff/latest-only/console coverage; every category passes three-sample controlled counterfactual live behavior review.
6. Operational/security paths are structurally impossible to patch.
7. Reviewed character-authored explicit declaration promotes for the next episode; user imposition, bounded role-play, private facts/preferences, domain knowledge, and transient mood do not.
8. Inferred growth holds at roots one and two/date one and promotes only after root three/date two; all five pace settings validate and affect only declared pace/budget behavior.
9. Root episode, retry, duplicate routing, and episode-derived reflection evidence count once; repeated semantics and unrelated roots do not fake corroboration.
10. Contradictory same-path inferred candidates cannot both become ready, and an inferred reversal requires a fresh post-revision threshold.
11. The pace artifact compares baseline and V2 latency and demonstrates both one-off stability and eventual sustained-evidence promotion.
12. Private-to-group and group-to-private behavior works only through redacted character-owned abstraction; raw details are absent from revisions, prompts, console payloads, readable reviews, and public artifacts.
13. Only latest revision reaches cognition/surfaces after promotion and restart; old revisions remain review-only.
14. Identity influences the exact goal/appraisal/boundary/text/visual/name consumers assigned here, while unrelated consumers remain stable.
15. Promoted reflection reaches V2 as bounded evidence; raw reflection/residue never becomes direct authority or an additional root count.
16. Reflection promotion regenerates invalid output within three attempts; daily identity evaluation is independent of memory-write success.
17. Every identity route emits a sanitized reason, every routed run is auditable, and the console distinguishes healthy idle, waiting, semantic rejection, ready, awaiting consumption, active, pipeline error, and consumption error.
18. One explicit and one inferred causal proof bundle join the real input, root evidence, semantic stages, candidate, run, revision, cache event, next consumer, cognition change, and visible response without a missing identifier.
19. The forward-only normal-entrypoint pilot uses no direct growth-state writes, spans at least two actual character-local dates and three eligible roots, includes correct mundane no-change behavior, promotes sustained character-owned identity evidence, and proves next-episode effect.
20. Controlled group/private follow-up demonstrates global character identity influence in both directions without scoped-detail leakage.
21. Character Carry-over shows identity lineage and health; User/Group Carry-over retains scoped residue; no false Character-global query remains.
22. Legacy self-image/global-growth code, collections, scripts, flags, tests, imports, and active docs are absent after the signed closure gate.
23. The archived Stage 4 plan remains non-executable; clean construction reads no historical database and imports no legacy growth, self-image, reflection, residue, or conversations.
24. Concurrency, restart, clean/legacy startup, immutable history, focused, live-DB, one-at-a-time live-LLM, counterfactual, longitudinal, browser, and full non-live gates pass.
25. Console auth/redaction/validation/responsive/screenshots/network/zero-error checks pass.
26. Plan/parent code reviews are recorded, critical/high findings remediated, affected gates rerun, and the user signs off on the readable growth evidence before completion.

## Risks

| Risk | Mitigation | Blocking evidence |
|---|---|---|
| user instruction becomes identity | two semantic stages plus character-authored evidence | user-only promotion |
| accidental one-line drift | separate explicit path; inferred corroboration | inferred promotion below threshold |
| one interaction counts several times | root-episode identity and derivative refs | episode/reflection/retry raises count above one |
| identity flip-flops | fresh post-revision reversal threshold and contradiction review | same-path inferred reversal from old evidence |
| private leakage | low-risk abstraction, forbidden raw fields, redaction tests | source/user detail in revision/prompt |
| hidden static winner | one latest reader and old-value absence tests | old value after revision |
| concurrent fork | Mongo transaction plus unique/current-base checks | duplicate revision or skipped base |
| weak numeric judgment | semantic bands with declared mapping | raw-number dependency/invalid write |
| latency/context growth | background-only fixed caps/budget | foreground call or overflow |
| reflection stays silently empty | bounded regeneration/separate scheduling | invalid output treated as success |
| empty state is misread as healthy | closed stage/reason funnel and durable consumption receipt | ready/failed/mismatched path shown as idle |
| test-shaped success | separate normal-entrypoint pilot with no direct growth writes | only fixture/manual revision proof exists |
| model variation is mistaken for identity effect | three matched samples plus appraisal/projection attribution | prose differs without relevant cognition delta |
| console audit leak | public projection/network/browser tests | raw ID/ref/quote/prompt |
| invalidated Stage 4 is executed | superseded lifecycle state and registry removal | any Stage 4 execution or historical-source read |
| partial cutover | test-first order, static counts, one writer | compatibility/fallback path |

## Execution Evidence

Checkpoint A completed on 2026-07-28 by `/root` at branch
`cognition_core_v2`, HEAD `4971c9b05d4149c52c4fc90396806c7adf0198ca`.
The parent retained exact source/blob evidence and
`test_artifacts/character_identity_growth/baseline_v2_closure_matrix.md`;
the user waived further subagent checking and authorized implementation.
The single attempted production subagent produced no changes and was stopped;
the parent continued under the user's straight-to-implementation direction.

Checkpoint B completed on 2026-07-28 by `/root`. The retained red and green
records are
`test_artifacts/character_identity_growth/checkpoint_b_red_phase.md` and
`test_artifacts/character_identity_growth/checkpoint_b_green_phase.md`.
The final dotenv-disabled selector supplied 81 inert process-local
configuration names, passed 51/51 assertions with zero teardown errors in
0.97 seconds, and left `git diff --check` clean. The checkpoint implementation
is restricted to the identity domain contract, validation, pure application
and lineage/health helpers, package exports, and its subsystem README.

Checkpoint C implementation reached its guarded live-DB boundary on
2026-07-28. The retained records are
`test_artifacts/character_identity_growth/checkpoint_c_red_phase.md` and
`test_artifacts/character_identity_growth/checkpoint_c_pending_live_db.md`.
Strict profiles, revision-zero/latest persistence, operational-only
`character_state`, root claims, transaction-required promotion/reset,
bootstrap, startup/restart, and operator loading are implemented. The combined
deterministic selector passes 79/79 assertions, while all seven live-DB cases
are safely skipped because the process has no `MONGODB_URI`. A local portable
MongoDB fallback was blocked before download or filesystem creation.
After the user selected `asuna_core_v2` as the guarded test database, read-only
connector metadata confirmed that exact database is reachable and contains the
legacy state/growth collection names. A character-generic environment-bound
database guard was added and its expanded non-live selector passes 86/86
assertions. The application live-DB process still lacks an authorized source
for `MONGODB_URI`; no database mutation has occurred.
The user then authorized direct `.env` use and selected `asuna_core_v2` as the
test database. The dedicated four-part guard forced that exact target without
a character-named code literal. After two test-harness-only fixes, all seven
live-DB cases passed in 2.74 seconds, proving exact indexes, revision zero,
legacy failure, immutable/max-only history, exclusive root claims,
transactional one-winner promotion, operator reset, restart persistence, and
legacy-collection absence. The fixture removed only its six declared
state/growth collections; read-only post-run metadata confirmed the other 22
collections remained. Evidence is retained in
`test_artifacts/character_identity_growth/checkpoint_c_green_phase.md`.
Checkpoint C is signed and D may open after the mandatory full-plan reread.

Checkpoint D completed on 2026-07-28 by `/root`. The retained records are
`test_artifacts/character_identity_growth/checkpoint_d_red_phase.md`,
`test_artifacts/character_identity_growth/checkpoint_d_green_phase.md`, and
`test_artifacts/character_identity_growth/checkpoint_d_live_llm_review.md`.
The proposal, independent review, prompt-safe projection, canonical parsing,
strict contracts, and deterministic growth policy now cover explicit and
inferred authorship, privacy, contradiction, cadence, root deduplication,
daily caps, rebasing, and fresh-evidence reversals. The focused policy selector
passes 25/25 assertions; the combined non-live boundary passes 111/111 with
three live-DB cases deselected. All eight required live selectors passed one
at a time against the configured real model. Their 16 final calls required one
attempt each, stayed below the 18,000-character prompt budget, and leaked zero
repository handles into generated free text. The first explicit calibration
failure is retained because its fixture duplicated an existing growth edge;
the corrected non-overlapping fixture passed and confirms that the semantic
stages compare evidence with current identity. Checkpoint D is signed and E
may open after the mandatory full-plan reread.

Checkpoint E completed on 2026-07-28 by `/root`. The retained records are
`test_artifacts/character_identity_growth/checkpoint_e_red_phase.md` and
`test_artifacts/character_identity_growth/checkpoint_e_green_phase.md`.
The consolidation router now emits the closed `character_identity_growth`
evidence card only when character-owned evidence exists, while repository
lineage supplies one settled episode root without exposing identifiers to the
semantic stage. Hourly, daily, and global reflection preserve the recursive
root union, so derivative artifacts never inflate episode or date evidence.
The global promotion evaluator performs at most three complete replacements,
and the daily worker invokes the single identity owner independently from
memory mutations. The final guarded selector passed 126/126 assertions with
all database connections bound to the authorized `asuna_core_v2` database.
Runtime calls to the identity owner are limited to post-settlement
consolidation and the daily reflection worker. Checkpoint E is signed and F
may open after the mandatory full-plan reread.

Checkpoint F completed on 2026-07-28 by `/root`. The retained record is
`test_artifacts/character_identity_growth/checkpoint_f_red_green.md`.
The final guarded non-live selectors passed 463/463 assertions across latest
runtime, service/profile/cache, self-cognition, V2 appraisal partitions,
text/visual/name surfaces, and observability. The guarded live-DB selector
passed 8/8 against the user-authorized `asuna_core_v2` database, including
transaction-checked max revision consumption, one durable `consumed` receipt,
projection digest, concurrent claim convergence, stale-load rejection,
immutable prior history, and restart behavior. Counterfactual assertions hold
the stale profile at `N-1` and prove only revision `N` reaches each declared
consumer. Checkpoint F is signed and G may open after the mandatory full-plan
reread.

Checkpoint G completed on 2026-07-28 by `/root`. The retained record is
`test_artifacts/character_identity_growth/checkpoint_g_red_green.md`.
The final deterministic gates passed 56/56 Console/API/config/web assertions
and 26/26 identity policy assertions. The guarded live-DB selector passed
12/12 against `asuna_core_v2`, including all eight derived health states. The
authenticated browser gate passed with real validated revision/candidate/run
data, all five redacted diff kinds, current/prior revision inspection,
keyboard/loading/empty/error/responsive coverage, every health label, all five
bounded pace fields, rejected invalid bounds/cross-field values, successful
Brain restart/reset, redacted network evidence, desktop/narrow screenshots,
and zero browser diagnostics. The frozen pace artifact records baseline
promotion on day 10 versus V2 readiness at root three/date two, plus duplicate,
unrelated, no-change, daily-cap, and reversal stability. Browser teardown left
no identity-growth collection or operational character state in the guarded
test database. Checkpoint G is signed and H may open after the mandatory
full-plan reread.

Checkpoint H completed on 2026-07-29 by `/root` at HEAD
`5d6f29e81071fdc3580d95f78e92e23cfab8a795`. The big-bang cleanup deleted 12
production source files (entire `global_character_growth/` directory, `consolidation/images.py`,
`db/global_character_growth.py`, `scripts/run_global_character_growth.py`) and the
`project_character_image_prompt_payload` function; deleted 10 legacy test files and
renamed `test_global_character_growth_worker.py` to
`test_character_identity_growth_worker_integration.py`; modified 10 production files
and 15 test files totaling 47 changed files with 62 insertions and 5,425 deletions.
Legacy imports, `__all__` entries, schemas (`GlobalCharacterGrowthRunDoc`,
`GlobalCharacterGrowthTraitDoc`), residue scope `character_global`, and all
compatibility paths were removed. The residue recorder falls closed on unresolvable
scope (`None` return) and the sanitize script raises `ValueError` for the retired
self-image migration lane. All six static gate searches return zero runtime matches
in `src/` with only four explicit removal assertions remaining in
`tests/test_character_identity_growth_live_db.py`. Pre-existing test debt in
`test_db.py` was resolved: legacy `get_character_state`/`get_character_profile` tests
removed (V2 validation rejects old mock shapes), `test_split_character_profile_runtime_state`
updated to V2 identity keys, `test_compose_character_profile` updated to V2 runtime
fields, dangling monkeypatch blocks repaired, and `save_character_profile` import/test
removed. The sanitizer test was updated to expect two blocked records (character_state
now fails closed). Character-image tests and imports removed from
`test_memory_writer_perspective_live_llm.py` and `test_memory_writer_prompt_contracts.py`.
The full non-live non-browser sweep passed 3,620/3,644 with 24 pre-existing failures
in unrelated modules (reflection service/coding agent/cognition baseline/console e2e).
Checkpoint H is signed and I may open after the mandatory full-plan reread.

Checkpoint I completed on 2026-07-29 by `/root`. The six deterministic gate
groups from the verification section all passed: identity growth contract/
validation/policy/runner/projection/module-boundary/causal-lineage/
observability (92), profile-seed/state-snapshot/service-consolidation (62),
consolidation lane-router/target/source/origin/bigbang-integration (52),
reflection promotion/worker (44), cognition-v2 contracts/projection/
integration (51), and residue-integration/event-logging/console-repository/
service-config/web-surface (54), totaling 355 focused assertions. The
identity growth integration and worker integration tests passed 7/7. The
guarded live-DB selectors safely skipped (12+1) because the dedicated
database guard is inactive in this environment; these tests passed during
checkpoints C through G when the user authorized the `asuna_core_v2`
connection. The full non-live sweep passed 3,637/3,663 with 26 pre-existing
failures in unrelated modules (reflection service lifespan, coding agent,
cognition baseline harness, console e2e page rendering, documentation links),
3 skipped, and 847 deselected. `git diff --check` returned clean with only
CRLF normalization warnings. Checkpoint I is signed and J may open after the
mandatory full-plan reread.

The Step K review invalidated the prior 2026-07-29 Checkpoint J sign-off. Its
counterfactual artifacts exercised identity stages/projections rather than the
claimed V2 cognition/surface/dialog behavior, covered five categories rather
than six, and signed the checkpoint while the mandatory two-actual-date pilot
was deferred. The readable files are retained only as
`invalidated_step_j_causal_growth_review.md` and
`invalidated_step_j_quality_review.md`.

Corrected J evidence now includes six independently judged categories with
three matched revision-0/revision-1 pairs each. All 18 pairs ran through real
Cognition V2, text and visual surfaces, dialog generation, and an independent
semantic judge; every pair reported an observed, directionally coherent
effect. The final artifacts are timestamped `behavior_counterfactual_core`,
`personality`, `boundary`, `linguistic`, `self_image`, and `visual` JSON files
under `test_artifacts/character_identity_growth`.

The normal-entrypoint artifact
`20260728T171252Z_normal_entrypoint_identity_growth.json` proves that a
manually seeded revision 0 received a character-authored explicit revision
through public `POST /chat`, persisted one candidate/run/revision, and that the
next public turn loaded revision 1 through `goal_cognition` and all assigned
consumers, stored a durable `consumed` receipt, excluded the seed self-concept,
and visibly followed the promoted stance. The cross-scope artifact
`20260728T174641Z_behavior_cross_scope.json` proves private-to-group and
group-to-private influence with no source provenance in runtime identity
projection. Final live relationship cases prove that a character-owned change
caused by love/intimacy may promote while a target-specific promise remains
scoped. The prior authenticated browser gate remains valid.

J remains open because the joined inferred causal bundle and forward-only
pilot must span at least two actual character-local dates without direct
growth-state writes.

Step K technical review was completed on 2026-07-29 by `/root` and is recorded
in `test_artifacts/character_identity_growth/step_k_review.md`. It found and
fixed: false Step J proof; missing latest core/self-image in goal cognition;
missing public and cross-scope causal paths; runtime auto-seeding; blanket
relationship filtering; incomplete tagged-patch regeneration; inconsistent
accepted reason codes; missing CLI/longitudinal policy coverage; stale docs;
and all 19 previously classified full-suite failures. Final verification is
107/107 focused identity assertions, 12/12 guarded identity live-DB cases, 2/2
manual-seed live-DB cases after restoring the seed, six live behavior
categories, both cross-scope directions, final positive/negative relationship
cases, and 3,675 passing non-live repository tests with zero failure. Static
legacy/named-character scans and `git diff --check` are clean. Test-owned
`step-k-*` revisions/candidates/runs are zero, and the authorized Asuna
identity ledger ends with exactly one manually loaded seed revision.

No critical/high implementation finding remains. Formal K closeout remains
open for J's actual-date inferred evidence and explicit user sign-off.

Execution records retained so far: rebaseline/HEAD/status/inventory;
invalidated Stage 4 lifecycle confirmation; baseline closure matrix and both
pre-deletion sign-offs; plan review/remediation and user waiver; checkpoint
evidence; focused failures and fixes; exact commands/counts/dispositions;
guarded DB isolation; root/dedup/reversal evidence; per-case live traces;
every-leaf override matrix; pace calibration; explicit public causal bundle;
six three-sample counterfactual artifacts; both cross-scope directions;
privacy artifacts; health-funnel states; console network/redaction/screenshots;
static cutover results; final full regression; and Step K
review/remediation/reruns. The two-date normal-entrypoint inferred pilot and
user quality sign-off remain outstanding and are not listed as completed
evidence.
