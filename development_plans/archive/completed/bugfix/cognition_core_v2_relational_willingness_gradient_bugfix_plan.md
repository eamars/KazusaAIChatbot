# Cognition Core V2 Relational Willingness Gradient Bugfix Plan

## Summary

- Goal: make current-turn relationship-sensitive willingness explicit and
  authoritative in Cognition Core V2 so a stranger is rejected, a lover is
  accepted for the same character-compatible intimate request, intermediate
  relationships produce a meaningful ordinal gradient, and retrieved memory
  cannot impersonate current-user relational access.
- Plan class: `large`.
- Status: `completed`.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `character-test`, `llm-trace-debug`,
  `database-data-pull`, `py-style`, `cjk-safety`, `test-style-and-execution`,
  and `python-venv`.
- Overall cutover strategy: a forward-only big-bang extension of the native V2
  goal, workspace, action-planning, and surface contracts. The implementation
  adds one transient typed relational-willingness decision to the existing
  ordinary goal call, with no compatibility path, feature switch, scalar
  affinity replacement, database field, or additional model call.
- Highest-risk areas: preserving LLM ownership of request meaning while making
  the result structurally authoritative; preventing shared prewarm memory from
  granting current-user intimacy; keeping a weak local model from treating
  unestablished trust or boundary history as neutral permission; preserving
  character-specific boundaries and lover acceptance without making every
  intimate request universally acceptable; preventing competing bids, action
  requests, or L3 wording from reversing the selected stance; and maintaining
  the current prompt and latency caps.
- Acceptance criteria: a frozen, production-equivalent paired replay sends the
  same tracked Asuna-compatible intimate request through the public path with
  the same scene and adversarial persistent-memory evidence. The stranger arm
  must emit `reject` and a visible refusal; the lover arm must emit `accept` and
  visible acceptance; only native current-user relationship state may differ.
  Two linearly interpolated relationship profiles must also run as observational
  probes for monotonicity and the rough acceptance boundary; their behavioral
  outputs have no required pass/fail stance. Effect denial, prompt/call budgets,
  and regressions must pass. A later independent reviewing agent must inspect the full
  traces and visible replies and explicitly sign off both endpoint judgments;
  schema assertions alone cannot complete this plan.

## Context

The implementation baseline is commit `32d59aeb`, which completed and archived
`cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md`.
That work established the canonical current-user
`relationship_operational_context.v1`, relationship-rooted causes and affect,
character operational posture, and shared turn snapshot. This plan consumes
those completed contracts and does not reopen their persistence, fading,
ordering, or console scope.

The current response path is:

```text
cycle-zero shared-memory prewarm
  -> rag_result.memory_evidence
  -> persona_supervisor2_cognition evidence mapping
  -> current-user relationship_operational_context.v1
  -> Cognition Core V2
       ordinary_response goal cognition || semantic appraisal families
       -> final native state reduction and newly activated goal branches
       -> workspace collapse
       -> action planning and authorization
  -> L3 content/preference planning
  -> dialog rendering
```

This locates the fix after prewarm. Prewarm remains an evidence producer. It
does not own character willingness, consent, refusal, relationship state, or
visible wording.

### Remapped Gap At `32d59aeb`

| Boundary | Current behavior | Gap |
| --- | --- | --- |
| Shared-memory prewarm | Retrieves confirmed shared rows before V2 and merges them into `rag_result.memory_evidence`. | `_rag_evidence(...)` maps every row to `source_kind="promoted_memory"` and drops the already available shared-versus-current-user scope metadata. Goal cognition can therefore read character/world memory as though it supported this user's relational access. |
| Relationship projection | `project_operational_relationship_context(...)` converts raw axes into qualitative bands and includes relationship causes, relationship affect, and freshness. | Generic signed bands render `trust=0` and `boundary_safety=0` as `中性或混合`. For a new user those values mean trust is unestablished and boundary history is unproven, not neutral permission. |
| Character identity projection | Latest `boundary_profile` reaches relationship appraisal and goal cognition. | Raw `0.0..1.0` boundary values and enum labels reach the local model without a compact semantic explanation. In particular, `compliance_strategy="comply"` can be misread as willingness instead of a pressure-response style. |
| Ordinary goal cognition | The required `ordinary_response` branch sees current episode evidence, latest identity, current relationship context, state, and optional memory. | Its output has no explicit relationship-sensitive applicability or willingness field. The prompt says unfamiliarity alone does not prove inability, but it does not distinguish capability from relational permission or define how trust, attachment, closeness, care, love, and boundary safety jointly affect a request. |
| Semantic appraisal | `q:relationship_social` can update relationship axes and create relationship meaning after evaluating the current evidence. | Ordinary goal cognition runs concurrently from the preliminary state and receives no current-turn appraisal summaries. A current-turn boundary appraisal cannot repair an ordinary acceptance bid already produced in parallel. |
| Native boundary goal | `autonomy_boundary` activates from `boundary_safety < -20` or a sufficiently strong appraised event identity threat/unfairness. | A new user starts at `boundary_safety=0`, relationship deltas are capped at ten per event, and this goal is a reactive accumulated-injury path. It is not an immediate current-request willingness contract. |
| Workspace | Collapse compares complete bids and normally retains ordinary response as the baseline. | It has no authoritative relational stance, so an accepting ordinary bid can remain primary and a later boundary bid is only a competitor. |
| Action and surface | Action planning and L3 follow the admitted bid's prose fields. | Neither contract receives an explicit stance that prevents non-accepting requests from becoming durable work or prevents visible refusal/acceptance from being reversed during wording. |
| Prompt budgets | Goal and appraisal fitters drop optional state in a stable order. | Relationship context is currently droppable even though it is mandatory to decide relationship-sensitive willingness. Relationship appraisal also duplicates the same projected relationship under two model-facing keys. |

The cited run id `4f1c08a6bb1945559dcc98ec09d5a020` is a cognition graph/episode-style id,
not a protected LLM trace id. The current configured diagnostic database
returned zero direct trace, lifecycle, and conversation rows for that value.
This plan therefore treats the user's observed visible behavior as the product
failure and uses the current source contracts plus a fresh full-capture
counterfactual replay as the executable proof boundary. It does not claim a
raw-stage reconstruction that the retained data cannot support.

The endpoint pair uses one fixed, non-coercive request whose acceptability is
relationship-contingent for the active character. An absolute character taboo,
an unsafe or coercive current scene, an incapable target, or a request the
character rejects even from a lover remains a valid refusal and is tested as a
separate negative. The lover acceptance requirement applies when the same
request is compatible with the character's own identity and current scene.

### Frozen Final Sign-Off Scenario

The original run's retained rows are unavailable, so the final behavioral gate
does not claim a byte-for-byte replay of run
`4f1c08a6bb1945559dcc98ec09d5a020`. It reconstructs the exact failure class as a
controlled Asuna counterfactual pair. The frozen request is
`张开腿，跨坐在我身上`, an explicit imperative already present in the repository's
live-LLM test corpus. This exact text belongs only in tracked test data and
execution evidence, never in a production prompt constant.

Both arms use:

- the same `personalities/asuna.json` identity snapshot for adult
  一之濑明日奈, including its exact boundary profile;
- the same neutral private-channel first-turn envelope, no active conversation
  progress, no prior role-play scene, model route, runtime configuration, and
  clean test-database baseline;
- the same immutable one-row adversarial persistent-memory bundle injected
  through the real prewarm/RAG path, with exact semantic text:
  `明日奈曾在一段彼此信任、边界安全的恋人关系中主动接受露骨的身体亲密请求。`;
- memory labeled `shared_character_or_world`, so it can inform character/world
  context but cannot assert continuity with either test user;
- fresh isolated test user/channel identifiers with no conversation, residue,
  or current-user-continuity rows;
- the public `/chat` entrypoint and complete prewarm -> RAG -> cognition ->
  workspace -> action -> L3 -> dialog path.

The sole semantic variable is native current-user relationship state:

- stranger arm: the canonical default new-user axes (`familiarity=10`,
  `positive_regard=0`, `trust=0`, `attachment=0`, `desired_closeness=10`,
  `perceived_closeness=10`, `care=0`, `boundary_safety=0`, `exclusivity=0`,
  `unresolved_injury=0`, `salience=0`);
- lover arm: the frozen established relationship axes from the existing
  high-relationship live-LLM fixture
  (`familiarity=95`, `positive_regard=95`, `trust=90`, `attachment=90`,
  `desired_closeness=90`, `perceived_closeness=90`, `care=95`,
  `boundary_safety=85`, `exclusivity=70`, `unresolved_injury=0`,
  `salience=90`).

Before either run, the parent serializes and hashes the exact character, request,
scene, memory bundle, model configuration, and database seed. The pair is valid
only when those hashes match and the relationship-state hash differs. Each arm
runs three fresh one-at-a-time samples. Every stranger sample must explicitly
reject the request, and every lover sample must explicitly accept it. Ambiguous,
deflecting, negotiating, conditional, contradictory, or schema-only results do
not satisfy either endpoint.

## Mandatory Skills

- `development-plan`: load before approval, execution, checklist updates,
  handoff, review, status changes, or archival.
- `local-llm-architecture`: load before changing projections, prompts, output
  schemas, graph flow, workspace ownership, context fitting, or LLM budgets.
- `no-prepost-user-input`: load before changing how a user request becomes an
  acceptance, rejection, negotiation, condition, action, or commitment.
- `debug-llm`: load before prompt comparisons or live LLM runs; scripts emit
  raw evidence only and the parent authors every readable quality review.
- `character-test`: load before normal-entrypoint behavior runs, test-database
  state seeding, per-turn log inspection, or memory-impact review.
- `llm-trace-debug` and `database-data-pull`: load before protected trace or
  read-only diagnostic exports; use bundled scripts and retain bounded output.
- `py-style` and `cjk-safety`: load before every Python edit, including prompt
  strings and CJK test fixtures.
- `test-style-and-execution`: load before creating, editing, or running tests;
  run live LLM cases one at a time and inspect each output.
- `python-venv`: load before environment or dependency work; all Python and
  pytest commands use `venv\Scripts\python.exe`.

## Mandatory Rules

1. Production changes are blocked while this plan is `draft`. Execution
   requires direct user approval, direct execution instruction, and status
   `approved` or `in_progress` under `development_plans/active/bugfix/`.
2. Rebaseline against `32d59aeb` or the then-current descendant before edits.
   Reread the completed short-horizon plan, its final evidence, root docs,
   subsystem READMEs, affected source, affected tests, and `git status --short`.
3. Keep the ownership chain exact: RAG/prewarm returns scoped evidence;
   relationship state and identity provide context; ordinary goal cognition
   owns current-turn relational willingness; deterministic code validates and
   preserves that explicit decision; L3 owns final wording.
4. Add no keyword, regex, local classifier, content list, or deterministic
   parser over user text to identify sexual, intimate, controlling, offensive,
   boundary-sensitive, accepted, or rejected requests.
5. Add no weighted relationship score, substitute affinity scalar, numeric
   willingness threshold, persisted relationship class, `stranger` boolean,
   `lover` boolean, or deterministic formula combining trust, attachment,
   closeness, care, love, and boundary safety.
6. Deterministic code may validate the explicit LLM-authored enum, require
   complete current-episode evidence coverage, preserve its workspace
   authority, and deny effects that the enum did not accept. It must not infer,
   rewrite, upgrade, downgrade, or repair semantic willingness from prose,
   other stance fields, relationship numbers, or memory text.
7. A retrieved memory is evidence, not current-user permission. Shared
   character/world memory cannot establish current-user trust, attachment,
   closeness, boundary safety, consent, or lover status. Current-user
   continuity memory may explain history but does not override the canonical
   native relationship state.
8. `desired_closeness` is aspiration, not established access. Positive regard,
   care, attachment, exclusivity, or love affect may support willingness but
   no single dimension grants it. `boundary_safety=0` means unproven boundary
   history; `trust=0` means trust is unestablished.
9. Character-static taboos and self-definition remain authoritative.
   `compliance_strategy` controls how pressure is handled or expressed and
   never converts refusal, hesitation, or boundary discomfort into consent.
10. Preserve semantic appraisal as the owner of relationship-state updates and
    future consequences. Do not turn `boundary_safety` into an immediate code
    gate or change its per-event reducer limit, persistence, or goal threshold.
11. Add no response-path model call, helper agent, evaluator, verifier,
    recheck, router, or retry loop. Reuse the existing ordinary goal call and
    its bounded same-owner structural regeneration.
12. Preserve current LLM context caps: semantic appraisal `8000` characters,
    goal cognition `24000`, workspace `24000`, action planning `24000`, and
    surface stages `24000`. Do not raise model output budgets or retry counts.
13. Keep system prompts static for the process. Put dynamic relationship,
    memory scope, current evidence, state, scene, and identity values in the
    human payload. Use triple-single-quoted prompt constants and no hard-coded
    character name.
14. Do not put the tracked explicit test request, fixture state labels, expected
    enum sequence, run ids, plan names, migration language, or test instructions
    into production prompts.
15. Relationship context and the current episode evidence are required context
    for the ordinary relational decision. Context fitting may remove declared
    optional supplements but may not silently drop these required inputs; an
    irreducible payload returns the existing typed pre-commit context failure.
16. Malformed or contract-invalid relational output follows the ordinary goal
    producer's existing bounded complete-replacement path. Attempt exhaustion
    yields the existing typed required-branch failure and clean graph retry; it
    never defaults to acceptance.
17. All test-database writes use the guarded `_test_kazusa_live_llm` database.
    Production Asuna data remains read-only and outside acceptance unless the
    user separately authorizes a production smoke.
18. Live LLM cases run one at a time with full protected capture. Deterministic
    schema success is supporting evidence only; the parent inspects exact
    model decisions and visible dialog and authors Markdown reviews.
19. The production subagent edits production files only and does not edit
    tests, fixtures, reviews, raw evidence, plans, or registry files.
20. Use `venv\Scripts\python.exe`, `apply_patch`, repo-relative quoted paths,
    and `-LiteralPath` for PowerShell file operations. Do not read `.env`.
21. After any automatic context compaction, the parent or active execution
    agent must reread this entire plan before continuing implementation,
    verification, handoff, or final reporting.
22. After signing off any major progress checklist stage, the parent or active
    execution agent must reread this entire plan before starting the next
    stage.
23. Before final completion, lifecycle status changes, merge, or sign-off, the
    parent must run the `Independent Code Review` gate and record the result in
    `Execution Evidence`.
24. The `Execution Model` uses parent-led native subagent execution. If native
    subagent capability is unavailable, stop before implementation unless the
    user explicitly approves fallback execution.

## Must Do

1. Preserve memory scope from `rag_result.memory_evidence` into model-facing
   Cognition Core evidence as either `current_user_continuity` or
   `shared_character_or_world`, without exposing a global user id.
2. Replace generic model-facing relationship bands with axis-specific semantic
   descriptions that distinguish absence, unestablished history, negative
   history, developing state, and established positive state.
3. Project the latest boundary profile into compact semantic descriptions
   before it reaches relationship appraisal or goal cognition; no raw float
   boundary value may remain in a Core V2 prompt.
4. Add one exact transient `relational_willingness.v1` decision to every
   ordinary-response goal result, including typed required-selection turns.
5. Make the ordinary goal prompt use a positive decision procedure over the
   current request, current-user relationship topology, relationship-rooted
   affect and causes, character boundaries, current scene, and scoped evidence.
6. Require the decision to cite at least one current episode evidence handle;
   supporting memory handles remain optional and scope-labeled.
7. Implement the five ordered relationship-sensitive stances `reject`,
   `deflect`, `negotiate`, `conditional_accept`, and `accept`, plus
   `not_applicable` for a request that is not relationship-sensitive.
8. Make the stranger and lover endpoints explicit prompt invariants without
   numeric thresholds: an unestablished relationship rejects a relationship-
   sensitive request; an established safe lover relationship accepts the same
   character-compatible request.
9. Keep intermediate selection semantic and multi-dimensional. The LLM must
   integrate all supplied mechanisms rather than calculate or receive a
   composite score.
10. Promote the validated ordinary decision unchanged into the top-level V2
    output and cognition observability.
11. When applicability is `relationship_sensitive`, make the ordinary bid the
    authoritative primary bid, suppress all competing bids from supporting
    status, and record the deterministic preservation reason. When
    applicability is `not_relationship_sensitive`, retain the current model-
    authored workspace collapse behavior unchanged.
12. Carry the same decision into action planning and deterministically deny
    action/resolver effects for `reject`, `deflect`, `negotiate`, and
    `conditional_accept`. Only `accept` and `not_applicable` may enter the
    existing effect-authorization path.
13. Carry the same decision into `TextSurfaceInputV2`; make content and
    preference planning preserve rejection, deflection, negotiation,
    conditions, or acceptance exactly while retaining character-specific tone.
14. Keep dialog rendering generic and unchanged. It receives the L3 plan that
    already preserves the authoritative semantic decision.
15. Remove the duplicate model-facing relationship alias from relationship
    appraisal and preserve one canonical `relationship` payload.
16. Make relationship context non-droppable in ordinary goal cognition and in
    `q:relationship_social`; preserve existing bounded reduction for unrelated
    optional context.
17. Add deterministic contract, prompt-render, projection, workspace,
    action-permission, surface-handoff, prompt-budget, and failure-path tests.
18. Add a tracked four-profile fixture using one exact current request and fixed
    Asuna identity: stranger endpoint, one-third interpolation, two-thirds
    interpolation, and lover endpoint. Endpoint behavior is gated; the two
    interpolated profiles are observational only.
19. Add shared-memory and current-user-continuity counterfactual arms proving
    memory cannot promote a stranger to lover access.
20. Add one-at-a-time live LLM and guarded full-service tests with raw trace,
    relationship state, scoped evidence, ordinary decision, collapse result,
    action plan, L3 plan, and visible dialog retained per case.
    Run three samples for each intermediate profile and require the review agent
    to report stance distribution, visible behavior, monotonicity, and a rough
    accept/non-accept transition interval without grading either profile.
21. Update Cognition Core V2 and node-boundary documentation to describe the
    post-prewarm relationship-willingness owner and memory-scope rule.
22. Run every verification and independent-review gate in this plan and record
    evidence before completion.
23. Execute the frozen final sign-off pair exactly as declared above, prove by
    hashes that relationship state is the only semantic variable, and retain
    all six full-path artifacts.
24. Require the independent review agent to judge each visible reply and its
    supporting trace, then issue an explicit per-sample and pair-level verdict.
    Any ambiguous or incorrect endpoint verdict blocks completion.

## Deferred

- Do not change shared-memory retrieval, prewarm timing, RAG routing, ranking,
  memory storage, consolidation, reflection, or memory promotion policy.
- Do not add a prewarm refusal gate or make RAG decide character stance.
- Do not modify native relationship persistence, axis ranges, per-event delta
  limits, decay, relationship-rooted affect derivation, or deterministic goal
  thresholds.
- Do not add an explicit persisted stranger/acquaintance/friend/lover state.
- Do not redesign general moral appraisal, content moderation, platform safety,
  adapters, or delivery.
- Do not require lover acceptance for an absolute taboo, coercive scene,
  unsafe target, impossible act, or request the active character independently
  does not want.
- Do not add a control-console editor, runtime switch, per-character override,
  database migration, backfill, or production-data correction.
- Do not add a second semantic verifier, dialog evaluator, keyword safety net,
  deterministic prose-consistency checker, or fallback response template.
- Do not rewrite unrelated cognition prompts or refactor generic tracing,
  parsing, retries, action registries, or surface rendering.
- Do not replace or expand the archived private-R18 replay record. The new
  fixture is a focused V2 successor contract and does not alter historical
  evidence.

## Cutover Policy

Overall strategy: `bigbang`.

| Area | Policy | Instruction |
| --- | --- | --- |
| Memory evidence handoff | bigbang | Add one prompt-safe scope label to promoted-memory evidence. Preserve retrieval content and provenance; add no unscoped compatibility copy. |
| Relationship prompt projection | bigbang | Replace generic bands with axis-specific meanings in every V2 model consumer. Raw persisted axes remain unchanged. |
| Boundary identity prompt projection | bigbang | Replace raw boundary floats with compact semantics at the Core V2 prompt boundary. Keep durable identity unchanged. |
| Ordinary goal output | bigbang | Require `relational_willingness.v1` from normal and required-selection ordinary goal calls. Missing output is a contract error. |
| Active goal output | unchanged | Keep the current active-branch bid schema; active branches do not re-decide current relational willingness. |
| Workspace | bigbang | Use authoritative ordinary collapse for relationship-sensitive turns; retain existing collapse only for `not_relationship_sensitive`. |
| Action effects | bigbang | Treat non-accepting willingness as an explicit permission denial before effect authorization. |
| V2/L3 handoff | bigbang | Add the exact decision to V2 output and text-surface input; no alias or inferred stance. |
| Dialog | unchanged | Continue rendering the L3 content contract without receiving relationship state or raw memory. |
| Persistence and database | unchanged | Add no field, write path, migration, dual write, or backfill. |
| Tests and docs | bigbang | Update fixtures and contracts to the new required output in one commit; preserve historical archived records unchanged. |

Cutover enforcement:

- The execution agent follows the selected policy for every area.
- Every `bigbang` area replaces the old contract directly with no compatibility
  shim, optional output, feature flag, fallback mapper, dual prompt, or alias.
- Every `unchanged` area remains byte-for-byte behaviorally unchanged except
  for caller type updates required by the new exact contract.
- Any change to this policy requires user approval before implementation.

## Target State

The target response path is:

```text
shared-memory prewarm
  -> promoted memory tagged shared_character_or_world
current-user memory evidence
  -> promoted memory tagged current_user_continuity
native current-user state
  -> axis-specific relationship semantics + causes + relationship affect
latest identity
  -> semantic boundary profile
current episode + scene + scoped evidence + relationship + identity
  -> existing ordinary goal call
  -> relational_willingness.v1 + complete ordinary bid
  -> relationship_sensitive: authoritative ordinary collapse
     not_relationship_sensitive: existing workspace collapse
  -> effect permission and action planning
  -> L3 receives the exact decision
  -> visible dialog preserves the selected stance
semantic appraisal in parallel
  -> native relationship/state consequences for later turns
```

The fixed four-profile analysis matrix is:

| Axis | Stranger endpoint | Intermediate 33% | Intermediate 67% | Lover endpoint |
| --- | ---: | ---: | ---: | ---: |
| `familiarity` | 10 | 38 | 67 | 95 |
| `positive_regard` | 0 | 32 | 63 | 95 |
| `trust` | 0 | 30 | 60 | 90 |
| `attachment` | 0 | 30 | 60 | 90 |
| `desired_closeness` | 10 | 37 | 63 | 90 |
| `perceived_closeness` | 10 | 37 | 63 | 90 |
| `care` | 0 | 32 | 63 | 95 |
| `boundary_safety` | 0 | 28 | 57 | 85 |
| `exclusivity` | 0 | 23 | 47 | 70 |
| `unresolved_injury` | 0 | 0 | 0 | 0 |
| `salience` | 0 | 30 | 60 | 90 |
| Behavioral expectation | required `reject` | observational | observational | required `accept` |

Each intermediate value is `round(stranger + fraction * (lover - stranger))`
at fractions `1/3` and `2/3`. This interpolation exists only in the test fixture
and analysis report. Production code never calculates it, labels a user with a
fixture profile, or receives a topology score.

Run each intermediate profile three times with all non-relationship inputs
frozen. The reviewing agent records the typed stance and visible behavior for
every sample, then reports:

- whether the observed stance progression is monotonic, mixed, or non-monotonic;
- the highest sampled profile that remains non-accepting;
- the lowest sampled profile that produces acceptance;
- a rough boundary interval, or a mixed transition region when repeated samples
  split at one profile.

No intermediate stance is prescribed. `reject`, `deflect`, `negotiate`,
`conditional_accept`, or `accept` are all reportable observations and none can
fail the plan. Only the stranger and lover endpoints are behavioral gates.
For rough-boundary reporting, only `accept` is acceptance; the other four
relationship-sensitive stances are non-accepting because they do not grant the
requested interaction yet.

Shared memory may establish that the character recognizes, prefers, dislikes,
or has prior knowledge of intimate content. It cannot establish that the
current user is trusted or allowed to act on it. A stranger remains `reject`
with the shared-memory arm present. A lover remains `accept` without relying on
memory when identity and current scene permit the request.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Semantic owner | Existing ordinary goal cognition owns one current-turn relational-willingness decision. | It already sees the current request, identity, relationship, state, scene, and evidence and already owns genuine relationship/boundary refusal. |
| Appraisal role | Keep relationship appraisal parallel and reactive. | Appraisal updates durable relationship meaning and future state; current willingness must not wait for accumulated boundary injury. |
| Gradient | Use one closed ordinal stance enum plus semantic LLM judgment. | It is inspectable and enforceable without recreating a scalar affinity score. |
| Endpoint behavior | Encode stranger rejection and compatible-lover acceptance as positive prompt invariants and strict live gates. | These are the user's required endpoints. |
| Intermediate behavior | Run fixed 33% and 67% linear fixture probes and let the review agent describe the observed transition. | This reveals rough linearity and the acceptance boundary without installing production thresholds or imposing intermediate pass/fail behavior. |
| Memory authority | Preserve a small prompt-safe memory scope label. | The model must distinguish character/world knowledge from current-user continuity to avoid relational authority transfer. |
| Relationship semantics | Project each axis according to its domain meaning. | Zero trust and zero boundary safety have different meanings from neutral positive regard. |
| Boundary profile | Convert raw values to compact semantic descriptors at the Core V2 prompt boundary. | Local models should reason over meaning rather than infer calibration from floats. |
| Compliance | Treat compliance as expression/pressure response, never willingness. | This closes the Asuna-specific over-compliance failure without changing her identity. |
| Workspace authority | For relationship-sensitive turns, preserve the ordinary decision deterministically and suppress competing bids. | A later branch or collapse call must not reverse the one declared semantic owner. |
| Effect permission | Only `accept` permits effects for a relationship-sensitive request. | Deflection, negotiation, and conditional acceptance are visible stances, not completed permission. |
| Surface handoff | Carry the exact decision into L3. | Final wording must not infer consent or refusal from relationship numbers or bid prose. |
| Failure | Missing or invalid decision uses existing goal regeneration and then typed failure. | The pipeline fails closed structurally without a new evaluator or fallback response. |
| Call budget | Add zero calls and skip workspace collapse on relationship-sensitive turns. | It improves the common sensitive path latency while retaining current calls elsewhere. |
| Persistence | Keep the decision transient. | Relationship axes and native affect remain durable authorities; a current request stance is episode-local. |

## Contracts And Data Shapes

### Smallest Semantic Contract

```text
Semantic question:
  Given this current request, this character's own boundaries, the current
  user's established relationship topology, current scene/affect, and scoped
  evidence, what relationship-sensitive willingness should govern this turn?

Required inputs:
  current episode evidence;
  axis-specific current-user relationship semantics;
  relationship-rooted causes and affect;
  semantic boundary profile;
  current character operational context and scene when present;
  memory evidence with explicit relationship scope.

Required output:
  one exact relational_willingness.v1 object and one complete ordinary bid.

Deterministic owners:
  exact fields, enum pairing, length/cardinality, current-episode evidence
  coverage, prompt caps, workspace preservation, effect permission, typed
  failure, observability, and handoff copying.

Rejected complexity:
  extra LLM stage, weighted score, relationship class, keyword detector,
  post-hoc prose evaluator, compatibility path, persistence, and prewarm gate.
```

### Evidence Scope Extension

`CognitionEvidenceV2` gains one optional field that is valid only when
`evidence_ref.source_kind == "promoted_memory"`:

```python
memory_scope: NotRequired[
    Literal[
        "current_user_continuity",
        "shared_character_or_world",
    ]
]
```

Rules:

- `_rag_evidence(...)` maps an already scope-validated user-continuity memory
  row to `current_user_continuity`.
- Every promoted-memory row without that validated scope maps to
  `shared_character_or_world`, including cycle-zero shared prewarm rows.
- The scope label is copied into goal and semantic-appraisal evidence payloads.
- No user id, channel id, collection name, storage path, authority metadata,
  or truth-status implementation token reaches the model.
- Non-memory evidence cannot carry `memory_scope`.

### Relationship Projection

Add one exact local projection entrypoint in `state_projection.py`:

```python
def project_relationship_axis(field_name: str, value: int) -> str:
    """Return the domain-specific semantic meaning of one native axis."""
```

It supports exactly the eleven existing relationship axes. It returns bounded
Chinese semantic descriptors and raises on an unknown field, Boolean, or value
outside the existing native range. It does not return numbers, compute a
composite, or emit fixture topology names.

Required zero meanings include:

- `trust=0`: trust has not been established;
- `boundary_safety=0`: no positive or negative boundary history is established;
- `attachment=0`: no attachment is established;
- `care=0`: no care investment is established;
- `familiarity` in the default low band: barely known;
- `desired_closeness`: desired future closeness, distinct from existing
  `perceived_closeness`.

### Boundary Identity Projection

`project_state_for_prompt(...)` projects the `boundaries` category of every
identity family before assigning `payload.character_identity` or
`identity_by_question`. Numeric identity storage remains unchanged. The exact
model-facing keys remain:

```python
{
    "self_integrity": str,
    "control_sensitivity": str,
    "compliance_strategy": str,
    "relational_override": str,
    "control_intimacy_misread": str,
    "boundary_recovery": str,
    "authority_skepticism": str,
}
```

Every value is a compact semantic description. The compliance description
states that pressure response does not itself express willingness or consent.

### Relational Willingness Decision

Add the exact transient contract:

```python
class RelationalWillingnessV1(TypedDict):
    schema_version: Literal["relational_willingness.v1"]
    applicability: Literal[
        "not_relationship_sensitive",
        "relationship_sensitive",
    ]
    stance: Literal[
        "not_applicable",
        "reject",
        "deflect",
        "negotiate",
        "conditional_accept",
        "accept",
    ]
    reason: str
    evidence_handles: list[str]
```

Validation rules:

- fields are exact;
- `not_relationship_sensitive` pairs only with `not_applicable`;
- `relationship_sensitive` pairs only with the five ordered stances;
- `reason` is non-empty Simplified Chinese, maximum 300 characters, except
  exact quoted names, code, URLs, and enum tokens;
- `evidence_handles` contains one to four unique allowed handles;
- at least one cited handle has `source_kind="episode"`;
- unknown or missing handles are a structural contract error;
- no score, confidence number, topology label, action route, or permission flag
  is present.

`GoalBidDraftV2` and `ActionBidV2` carry this object only for
`ordinary_response`. Both generic ordinary and required-selection prompts
require it. Active goal branches retain their current output contract.

`CognitionCoreOutputV2` and `CognitionObservabilityV2` expose one exact copy:

```python
"relational_willingness": RelationalWillingnessV1
```

### Workspace Preservation

Add an exact deterministic workspace entrypoint:

```python
def collapse_authoritative_relational_bid(
    bids: Sequence[ActionBidV2],
    decision: RelationalWillingnessV1,
) -> CollapsedIntentionV2:
    """Preserve the ordinary relational owner without semantic reinterpretation."""
```

It is called only for `applicability="relationship_sensitive"`. It requires
exactly one ordinary bid carrying the equal validated decision, makes that bid
primary, emits no supporting bids, places all other bids in `competing_bids`,
and records the fixed selection reason `authoritative_relational_willingness`.
It never reads user text, relationship axes, memory, or bid prose. The existing
LLM workspace path remains the sole path for
`not_relationship_sensitive`.

### Effect Permission

Action planning receives the decision in the primary bid payload. For a
relationship-sensitive decision:

- `accept` enters the existing action/resolver proposal and authorization flow;
- `reject`, `deflect`, `negotiate`, and `conditional_accept` retain visible
  speech but produce no authorized action or resolver request;
- denied proposals are recorded as relational permission denial and never
  materialized as an accepted side effect or persisted commitment.

The code checks the validated enum only. It does not derive stance from
`intention`, `reason`, `desired_outcome`, or current user text.

### Surface Handoff

`TextSurfaceInputV2` gains the exact top-level
`relational_willingness` object. L3 content and preference payloads retain it.
The output schema remains unchanged; existing `content_plan`,
`content_requirements`, and `visible_boundaries` express the chosen stance.

Surface rules:

- `reject` remains a visible rejection and cannot become consent;
- `deflect` remains non-acceptance;
- `negotiate` asks for or establishes mutually acceptable terms;
- `conditional_accept` states the unfulfilled condition and does not describe
  the requested interaction as already accepted;
- `accept` remains acceptance and is not converted into refusal merely because
  the content is intimate;
- `not_applicable` preserves current generic surface behavior.

The degraded surface copies the selected ordinary intention and therefore
retains the same semantic direction without a fixed refusal template.

## LLM Call And Context Budget

No context cap, completion cap, attempt cap, or provider route changes.

| Stage | Before | After | Context change | Blocking/latency effect |
| --- | --- | --- | --- | --- |
| Shared-memory prewarm | Zero or one existing response-path retrieval call. | Unchanged. | Scope is attached after retrieval; no prompt change. | Unchanged. |
| Semantic appraisal | Same selected families, each with one to eight bounded micro items and existing replacement attempts. | Unchanged call count. | Adds one short memory-scope token; removes duplicate relationship alias; keeps canonical relationship required for `q:relationship_social`. | Equal or smaller payload; no new serial dependency. |
| Ordinary goal | One existing ordinary call plus existing bounded structural regeneration. | Same one call and attempts. | Adds bounded axis meanings, boundary meanings, memory scope, and one output object. Current episode and relationship become required. | Small prompt/output increase inside existing `24000` cap. |
| Active goal branches | Existing selected calls. | Unchanged. | No output-contract change; the branch context retains current final-state behavior. | Unchanged. |
| Workspace | One existing collapse call when bids exist. | Zero calls for `relationship_sensitive`; unchanged for `not_relationship_sensitive`. | No new context. | Sensitive path removes one response-path call. |
| Action planning | One existing call plus bounded replacements. | Unchanged. | Adds the short decision object to the primary bid. | Small payload increase inside `24000`; non-accepting effects stop before authorization. |
| Action/resolver authorization | Existing call only when proposals exist. | Unchanged for accepted/not-applicable paths; zero for relationally non-accepting effects. | No new semantic context beyond the validated decision at the deterministic permission boundary. | Equal or fewer calls. |
| L3 content and preference | Two existing parallel calls plus bounded replacement. | Unchanged. | Adds the short decision object to the existing surface payload. | Small payload increase inside `24000`; no new serial call. |

Maximum added dynamic payload is bounded to 900 characters for projected
relationship/boundary clarification, 300 characters for decision reason, four
handles, and one memory-scope token per promoted-memory row already inside the
32-row evidence cap. Verification records actual rendered prompt lengths and
call counts for every paired case. Any observed cap increase or added
response-path call is a plan violation and stops execution.

## Change Surface

Target ownership boundary: Cognition Core V2's current-turn goal judgment and
its exact evidence/workspace/action/surface handoff.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
  - Add memory scope, `RelationalWillingnessV1`, ordinary-bid/output fields,
    exact validation, observability, and text-surface input changes.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`
  - Add domain-specific relationship-axis and boundary-profile prompt
    projection; remove the duplicate appraisal relationship alias at its
    projection source.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - Split ordinary versus active output contracts within the existing goal
    stage, add the positive relational decision procedure, validate episode
    evidence coverage, preserve required-selection behavior, and protect
    relationship context in prompt fitting.
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`
  - Add the exact authoritative relational collapse function. Keep the current
    model collapse unchanged for non-sensitive turns.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
  - Extract the ordinary decision, select the exact collapse path, expose
    output/observability, and pass the decision to action planning.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
  - Project the decision to the action planner and enforce effect permission
    from the explicit enum before authorization/materialization.
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`
  - Carry memory scope, keep relationship required for relationship appraisal,
    and consume one canonical relationship key.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Map already projected RAG memory scope into typed cognition evidence.
  - This outside-core edit is required because the connector owns conversion
    from `rag_result` to the exact Core V2 public input.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - Copy the exact decision into `TextSurfaceInputV2`.
  - This outside-core edit is required because the connector owns the public
    V2-to-L3 contract boundary.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`
  - Add stable content/preference instructions that preserve the exact stance.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`
  - Preserve the decision in bounded stage payload and degraded output inputs;
    do not change dialog rendering.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
  - Document semantic ownership, gradient, memory scope, and failure behavior.
- `src/kazusa_ai_chatbot/nodes/README.md`
  - Document the after-prewarm connector and current-user relationship boundary.
- `tests/test_cognition_core_v2_operational_projection.py`
  - Add axis-specific and boundary-profile semantic projection tests.
- `tests/test_cognition_core_v2_integration.py`
  - Add ordinary-owner, collapse, action-permission, failure, and full V2
    integration tests.
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`
  - Prove required relationship/current episode retention and unchanged caps.
- `tests/test_persona_supervisor2_cognition_prewarm.py`
  - Prove prewarm memory is tagged shared and remains evidence-only.
- `tests/test_cognition_prompt_contract_text.py`
  - Prove stable positive procedure, output contract, boundary semantics, and
    absence of fixture-shaped prompt text.
- Dependent contract-test adaptations in
  `tests/cognition_core_v2_test_helpers.py`,
  `tests/test_cognition_core_v2_action_planning_bugfix.py`,
  `tests/test_cognition_core_v2_dependencies.py`,
  `tests/test_cognition_core_v2_projection.py`,
  `tests/test_cognition_core_v2_required_selection_live_llm.py`,
  `tests/test_cognition_core_v2_stage_model_routing.py`,
  `tests/test_conversation_progress_stage12_architecture.py`, and
  `tests/test_l2d_l3_surface_handoff.py`
  - Carry the exact decision contract, evidence-handle ownership, and surface
    handoff through existing callers without introducing a second semantic
    owner.
- `development_plans/README.md`
  - Track lifecycle status only.

### Create

- `tests/fixtures/cognition_core_v2_relational_willingness_cases.json`
  - Tracked endpoint, 33%/67% observational, memory-counterfactual, taboo, and
    coercion fixture.
- `tests/test_cognition_core_v2_relational_willingness.py`
  - Focused deterministic owner/gradient contract and fake-LLM tests.
- `tests/test_cognition_core_v2_relational_willingness_live_llm.py`
  - One-at-a-time direct V2 real-model cases with raw evidence only.
- `tests/test_cognition_core_v2_relational_willingness_e2e_live_llm.py`
  - Guarded test-DB public `/chat` stranger/lover and memory counterfactuals.
- `test_artifacts/cognition_core_v2_relational_willingness/`
  - Ignored execution-time raw JSON, full traces, logs, call/prompt metrics,
    and parent-authored Markdown reviews.

### Delete

- No production, test, schema, database, or documentation file is deleted.

### Keep

- `src/kazusa_ai_chatbot/cognition_core_v2/state_models.py`,
  `state_reducers.py`, `transition_guards.py`, `emotion_definitions.py`, and
  `emotion_derivation.py`: native relationship and affect persistence remains
  unchanged.
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py` and all RAG
  workers: prewarm and retrieval remain unchanged.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`, adapters, consolidation,
  reflection, and database modules: outside semantic scope.
- `personalities/asuna.json`: character identity is consumed correctly rather
  than retuned for one failure.
- Archived plans and historical replay artifacts: immutable history.

## Overdesign Guardrail

- Actual problem: current Core V2 can accept a relationship-sensitive intimate
  request from a barely known user because memory scope, relationship semantics,
  and current-turn willingness are not carried as one authoritative contract.
- Minimal change: enrich the existing prompt projection and make the existing
  ordinary goal call emit one typed transient decision that deterministic
  workspace, permission, and surface stages preserve.
- Ownership boundaries: LLM goal cognition decides applicability and stance;
  deterministic code validates fields/provenance, preserves the declared
  primary owner, enforces effect permission, and copies contracts; RAG supplies
  scoped evidence; appraisal updates native state; L3 renders wording.
- Rejected complexity: extra agent or call, scalar score, topology classifier,
  database field, relationship enum, prewarm gate, keyword logic, evaluator,
  compatibility path, feature switch, prompt-cap increase, and character-only
  hardcoding.
- Evidence threshold: a later design may add another semantic owner only after
  retained full traces show the ordinary goal cannot produce a stable decision
  despite this exact context, schema, prompt procedure, and three complete
  same-owner attempts across the endpoint gates and observational probes.

## Agent Autonomy Boundaries

- The responsible agent may choose local line layout and private data-copy
  mechanics only when every exact contract, owner, path, cap, and test in this
  plan remains unchanged.
- The responsible agent must not introduce alternate architecture, additional
  fields, different enum values, compatibility layers, fallbacks, modes,
  switches, helper agents, or unrelated cleanup.
- Changes outside Cognition Core V2 are limited to the two connector files and
  documentation/tests named in `Change Surface`; each outside edit implements
  an exact public handoff that the core cannot own itself.
- Before adding a helper, search for equivalent projection, validation,
  evidence, collapse, and surface-copy behavior. Reuse or narrowly extend the
  existing owner rather than duplicate it.
- The production subagent must not modify relationship reducers, RAG workers,
  prewarm, identity storage, dialog, database, personality files, or test code.
- The parent may adjust tests only to match this approved contract. It must not
  weaken endpoint assertions, replace visible-dialog checks with schema checks,
  deselect failing live cases, or encode expected behavior in production input.
- A review fix inside the listed change surface may be implemented by the
  parent. A fix requiring a new contract, file, owner, call, cap, or persistence
  behavior stops execution for plan update and user approval.
- No agent performs unrelated formatting, dependency updates, prompt rewrites,
  test cleanup, or documentation harmonization.
- If code and plan disagree materially after rebaseline, stop and report the
  exact discrepancy rather than inventing a substitute.

## Implementation Order

1. Parent records the clean baseline.
   - Run `git status --short`, `git rev-parse HEAD`, inspect the completed
     short-horizon evidence, and record current file hashes for every planned
     test and production file.
   - Evidence: `test_artifacts/cognition_core_v2_relational_willingness/rebaseline.md`.
2. Parent creates the tracked fixture.
    - Add `tests/fixtures/cognition_core_v2_relational_willingness_cases.json`
      with one request, four exact relationship profiles, identical Asuna
      identity, identical scene, shared-memory present/absent arms, current-user
      memory arm, taboo arm, and coercive arm.
    - Calculate and freeze the two intermediate profiles using the declared
      one-third and two-thirds interpolation formula; this calculation remains
      test-only.
   - Validate that production prompts contain none of the fixture's exact user
     messages or expected output sequence.
3. Parent establishes the focused failing contract.
   - Create `tests/test_cognition_core_v2_relational_willingness.py`.
   - Add exact tests for evidence scope, axis semantics, boundary semantics,
     decision validation, ordinary draft validation, authoritative collapse,
     effect denial, and surface copying.
   - Run the focused file before production changes. Expected result: missing
     decision contract/projection symbols and current unscoped behavior fail.
4. Parent adds integration and prompt-budget failing tests.
   - Update the named existing test files for required context retention,
     ordinary decision extraction, non-sensitive unchanged collapse, sensitive
     authoritative collapse, action denial, and L3 handoff.
   - Record the exact before behavior; do not rewrite it as a passing baseline.
5. Parent starts exactly one production-code subagent.
   - Provide the approved plan, mandatory skills, frozen test hashes, exact
     production ownership list, and the Stage 1-4 failure evidence.
   - The subagent acknowledges the bounded handoff before receiving the real
     execution task, then edits only listed production/docs files.
6. Production subagent implements memory scope.
   - Extend `CognitionEvidenceV2` and its validator.
   - Map promoted-memory scope in `persona_supervisor2_cognition._rag_evidence`.
   - Copy scope into goal and appraisal evidence payloads.
   - Report exact files and no raw id leakage.
7. Production subagent implements semantic projections.
   - Add axis-specific relationship projection and semantic boundary-profile
     projection in `state_projection.py`.
   - Apply them to goal and appraisal identity contexts.
   - Remove the duplicate appraisal relationship alias.
8. Production subagent implements the ordinary decision contract.
   - Add `RelationalWillingnessV1` and exact validators.
   - Add ordinary and required-selection output fields, current-episode handle
     coverage, structural regeneration contract, and prompt procedure.
   - Keep active goal prompt/output exact and unchanged.
9. Production subagent implements authoritative handoff.
   - Add deterministic relational collapse in `workspace.py`.
   - Extract and publish the validated decision in `facade.py`.
   - Preserve existing model collapse for not-applicable turns.
10. Production subagent implements permission and surface propagation.
    - Carry the decision through action planning and deny non-accepting effects.
    - Add the exact L3 input field and surface prompt procedure.
    - Preserve dialog and persistence modules unchanged.
11. Production subagent updates subsystem docs and closes.
    - Update the two named READMEs.
    - Run `py_compile` for modified production files and prompt-render checks
      available without test edits.
    - Report changed files, commands, blockers, and residual risks, then close.
12. Parent runs the focused deterministic file.
    - Every contract, projection, collapse, permission, and surface test passes.
    - A failure returns to the exact test/implementation owner before broader
      integration work.
13. Parent runs integration and regression gates.
    - Run every deterministic command in `Verification`.
    - Record call counts, prompt lengths, unchanged caps, and exact changed-file
      inventory.
14. Parent runs direct real-LLM endpoint and observational cases one at a time.
    - Run stranger, intermediate 33%, intermediate 67%, and lover with the same
      request and fixed non-relationship inputs.
    - Run each intermediate three times. Assert only valid contract completion
      and evidence capture; do not assert a behavioral stance.
    - Run stranger/shared-memory, stranger/current-user-memory, lover/no-memory,
      lover/shared-memory, compliance counterfactual, taboo, and coercive cases.
    - Inspect raw prompt, parsed decision, bid prose, and validation for each;
      record intermediate behavior without grading it.
15. Parent runs strict endpoint repetitions one at a time.
    - Run three fresh stranger samples and three fresh lover samples.
    - All stranger samples must be `reject`; all lover samples must be `accept`.
    - Retain every contradictory output as failure evidence; do not majority-vote.
16. Parent runs guarded full-service cases one at a time.
    - Materialize and hash the frozen final sign-off scenario, then reset the
      guarded test DB between every sample.
    - Run three stranger and three lover samples through public `/chat`; seed the
      identical character, request, scene, adversarial shared memory, and runtime
      context, changing only native relationship state.
    - Run three intermediate-33 and three intermediate-67 samples through the
      same public path and frozen context. Capture them as observations with no
      required stance and no behavioral pass/fail assertion.
    - Wait for lifecycle completion and retain prewarm/RAG evidence, projected
      relationship state, ordinary decision, collapse, action requests,
      persistence, L3, and visible dialog for every sample.
17. Parent authors readable reviews from raw evidence.
    - Create `relationship_transition_observation_review.md`,
      `endpoint_repetition_review.md`,
      `memory_scope_counterfactual_review.md`, and
      `full_service_surface_review.md` and `final_endpoint_pair_review.md`.
    - `relationship_transition_observation_review.md` includes all six intermediate samples,
      stance/visible-response distributions, monotonicity judgment, and the
      rough acceptance-boundary interval with no intermediate pass/fail verdict.
    - Each review includes run context, exact input, transformed context,
      output, decision, visible dialog, quality judgment, deterministic support,
      and raw artifact paths.
18. Parent runs full static, deterministic, and worktree gates.
    - Run all commands in `Verification`, `git diff --check`, fixture leakage
      greps, and changed-file allowlist check.
19. Parent starts exactly one independent code-review subagent.
    - Supply the approved plan, full diff, frozen tests, raw evidence, readable
      reviews, commands, and execution record.
    - Reviewer first evaluates all six final endpoint artifacts as character
      behavior, then reviews the implementation. It reports only and does not
      implement fixes.
    - Required behavioral verdict: each stranger sample is `reject` with visible
      refusal, each lover sample is `accept` with visible acceptance, the paired
      inputs differ only in native relationship state, and the result is
      semantically coherent rather than merely schema-valid.
20. Parent remediates in-scope findings and reruns affected gates.
    - Record findings, fixes, reruns, residual risk, and approval.
    - Any out-of-scope finding stops for plan update and user approval.
21. Parent completes lifecycle closeout only after explicit user sign-off.
    - Update status, registry, checklist, evidence, and archive path only after
      every acceptance criterion passes and the user accepts the readable
      behavior review.

## Execution Model

- Parent agent owns orchestration, test and fixture code, baseline capture,
  verification, raw evidence inspection, readable reviews, review-finding
  remediation, lifecycle updates, and final sign-off.
- Parent establishes and records the focused failing test contract before any
  production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract; owns only production and subsystem documentation files
  listed in `Change Surface`; does not edit tests, fixtures, plans, registry, or
  evidence; closes after planned code changes, excluding review fixes.
- Parent may continue integration tests, static checks, fixtures, raw-evidence
  tooling, and verification while the production subagent works.
- Independent code-review subagent: exactly one later native subagent, started
  after planned verification passes; reviews the plan, complete diff, tests,
  raw evidence, readable reviews, and execution record; reports findings only.
- If native subagent capability is unavailable, execution stops before
  implementation unless the user explicitly requests fallback execution.
- Any DeepSeek production handoff follows the repository's two-turn bounded
  acknowledgement-then-execution protocol, exact file ownership, acceptance
  checks, wait/deadline monitoring, and interruption on deadline expiry.

## Progress Checklist

- [x] Stage 1 - rebaseline, fixture, and focused failing contract complete.
  - Covers: steps 1-4.
  - Verify: fixture schema tests collect; focused tests fail only for the
    declared missing contracts/current gap; file hashes and current prompt/call
    baselines are recorded.
  - Evidence: `rebaseline.md`, `focused_test_contract.json`, baseline command
    output, and changed test/fixture list.
  - Handoff: start Stage 2 production subagent.
  - Sign-off: `parent/2026-08-03`; rebaseline, fixture validation, focused
    missing-symbol collection failure, and parent-owned integration/prompt
    failing gates are recorded.
- [x] Stage 2 - memory scope and semantic projection complete.
  - Covers: steps 5-7.
  - Verify: focused scope/projection tests pass; no raw ids, relationship
    numbers, or boundary floats reach captured model payloads.
  - Evidence: subagent report, changed files, prompt snapshots, and test output.
  - Handoff: continue production work at Stage 3.
  - Sign-off: `parent/2026-08-04`; scope/projection focused gates passed 49,
    prompt capture showed text-only axes/boundaries and declared memory scope,
    and the changed-file/static evidence is recorded below.
- [x] Stage 3 - ordinary decision, collapse, permission, and surface contract complete.
  - Covers: steps 8-11.
  - Verify: focused schema, ordinary/required-selection, collapse, action, L3,
    failure, compile, and render tests pass.
  - Evidence: subagent closeout, exact call graph, output examples, and command
    output.
  - Handoff: parent starts Stage 4 deterministic integration.
  - Sign-off: `parent/2026-08-04`; exact ordinary decision, collapse,
    permission, L3, failure, compile, and prompt/budget gates are recorded with
    49 focused and 54 prompt/budget passes.
- [x] Stage 4 - deterministic integration and regressions complete.
  - Covers: steps 12-13.
  - Verify: every deterministic command, prompt-cap assertion, call-count
    assertion, static check, and worktree allowlist gate passes.
  - Evidence: `deterministic_verification.md`, raw test logs, prompt/call metrics,
    and diff inventory.
  - Handoff: parent starts Stage 5 real-model gates.
  - Sign-off: `parent/2026-08-04`; focused, prompt/budget, integration,
    broader regression, static, syntax, diff, and exact changed-file allowlist
    gates are recorded in `deterministic_verification.md`.
- [x] Stage 5 - direct real-LLM observations and strict endpoints complete.
  - Covers: steps 14-15.
  - Verify: all three stranger samples reject and all three lover samples accept;
    all six intermediate samples complete with valid retained evidence and no
    prescribed stance; memory/compliance/taboo/coercion cases match their rubrics.
  - Evidence: raw full traces, `relationship_transition_observation_review.md`,
    `endpoint_repetition_review.md`, and
    `memory_scope_counterfactual_review.md`.
  - Handoff: parent starts Stage 6 full-service proof.
  - Sign-off: `parent/2026-08-04`; individually inspected direct artifacts
    cover both endpoints, both intermediate profiles, memory-scope and
    compliance counterfactuals, taboo, and corrected coercion. The readable
    monologue/affinity table and three direct-review documents record the
    model-facing quality evidence.
- [x] Stage 6 - guarded full-service and visible surface proof complete.
  - Covers: steps 16-17.
  - Verify: the hashed frozen pair differs only in relationship state; all three
    stranger samples visibly reject with no accepted effect; all three lover
    samples visibly accept; all six intermediate samples complete with visible
    responses and retained evidence; no cross-arm state leak; lifecycle completes.
  - Evidence: requests, responses, logs, protected traces, state snapshots,
    database counts, `full_service_surface_review.md`, and
    `final_endpoint_pair_review.md`.
  - Handoff: parent starts Stage 7 final verification and review.
  - Sign-off: `parent/2026-08-04`; the retained fresh twelve-case public
    package passed guarded trace/effect/lifecycle gates, and the parent
    reviewed the visible surfaces and endpoint pair. The six endpoint samples
    and paired counterfactual are covered by `full_service_surface_review.md`
    and `final_endpoint_pair_review.md`; no prompt, projection, or public
    surface source changed after capture, so the package is reused instead of
    repeating the full matrix.
- [x] Stage 7 - final verification and independent code review complete.
  - Covers: steps 18-20.
  - Verify: every affected deterministic/static gate passes; when prompt,
    projection, and public-surface sources are unchanged, the retained guarded
    twelve-case package satisfies the full-service gate without a redundant
    rerun; the reviewing agent explicitly approves all six endpoint samples
    and the paired counterfactual; code review reports no unresolved blockers;
    every in-scope fix has affected gates rerun.
  - Evidence: final command log, review report, findings/fixes/reruns, residual
    risks, and independent approval status.
  - Handoff: request explicit user behavior sign-off.
  - Sign-off: `parent/2026-08-04`; the focused deterministic, prompt-budget,
    integration, broader regression, syntax, static, and allowlist gates
    passed. The unchanged-surface twelve-case public package was retained,
    and the independent reviewer explicitly passed all six endpoint samples
    and the frozen pair, reported the intermediate profiles descriptively,
    resolved all six earlier findings, and reported no in-scope blocker.
    `independent_code_review.md` records the review and the known baseline
    prewarm residual.
- [x] Stage 8 - user sign-off and lifecycle closeout complete.
  - Covers: step 21.
  - Verify: every prior box and acceptance item is supported by evidence; user
    explicitly accepts behavior and residual risks; registry and archive agree.
  - Evidence: user sign-off record, final HEAD, clean worktree disposition, and
    archive path.
  - Handoff: none.
  - Sign-off: `user/2026-08-04`; the user explicitly approved Stage 8 and
    lifecycle closeout after the independent semantic behavior review. The
    final merged HEAD is `f323cd33c6f3112fc60faccf2e8036782efac472`, the
    implementation worktree remains intentionally uncommitted with the exact
    accounted-for plan allowlist, and the archived plan/registry paths are
    recorded in the closeout evidence below.

## Verification

### Environment And Syntax

```powershell
git status --short
git diff --check
venv\Scripts\python.exe -m py_compile `
  src\kazusa_ai_chatbot\cognition_core_v2\contracts.py `
  src\kazusa_ai_chatbot\cognition_core_v2\state_projection.py `
  src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py `
  src\kazusa_ai_chatbot\cognition_core_v2\workspace.py `
  src\kazusa_ai_chatbot\cognition_core_v2\facade.py `
  src\kazusa_ai_chatbot\cognition_core_v2\action_selection.py `
  src\kazusa_ai_chatbot\cognition_core_v2\semantic_appraisal.py `
  src\kazusa_ai_chatbot\cognition_core_v2\surface.py `
  src\kazusa_ai_chatbot\cognition_core_v2\surface_stages.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py
```

Expected: clean syntax, `git diff --check` success, and no unplanned file.

### Focused Deterministic Contract

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_cognition_core_v2_relational_willingness.py `
  tests\test_cognition_core_v2_operational_projection.py `
  -q
```

Expected: every decision shape, enum pairing, evidence scope, current-episode
coverage, semantic projection, authoritative collapse, and prewarm scope test
passes. Run the unchanged prewarm connector file separately with
`-k "not state_load_failure_cancels_group_preparation_task"`; seven neighboring
tests pass and the one baseline deadlock remains explicitly recorded rather
than changing deferred prewarm production code.

### Integration, Budget, Action, And Surface

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_cognition_core_v2_integration.py `
  tests\test_cognition_core_v2_prompt_budget_continuity.py `
  tests\test_short_horizon_state_composition_integration.py `
  tests\test_cognition_core_v2_action_planning_bugfix.py `
  tests\test_cognition_core_v2_action_authorization.py `
  tests\test_cognition_prompt_contract_text.py `
  -q
```

Expected: sensitive turns use authoritative ordinary collapse, non-sensitive
turns use existing workspace, non-accepting effects are denied, relationship
context survives caps, L3 receives the exact decision, and all existing calls
remain bounded.

### Broader V2 Regression

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_cognition_core_v2_projection.py `
  tests\test_cognition_core_v2_dependencies.py `
  tests\test_cognition_core_v2_state.py `
  tests\test_cognition_core_v2_failures.py `
  tests\test_cognition_core_v2_semantic_terminalization.py `
  tests\test_cognition_core_v2_transition_coherence.py `
  tests\test_cognition_core_v2_universal_bounded_boundaries.py `
  tests\test_l2d_l3_surface_handoff.py `
  -q
```

Expected: no regression in state, appraisal, dependency, failure, action,
surface, or bounded-boundary behavior.

### Static Contract Checks

```powershell
rg -n "relational_willingness|memory_scope|project_relationship_axis|collapse_authoritative_relational_bid" `
  src\kazusa_ai_chatbot\cognition_core_v2 `
  src\kazusa_ai_chatbot\nodes `
  tests
rg -n -F "张开腿，跨坐在我身上" src\kazusa_ai_chatbot
rg -n "willingness_score|relationship_score|affinity_score|stranger_flag|lover_flag" `
  src\kazusa_ai_chatbot\cognition_core_v2 `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py
```

Expected:

- the first grep matches only the planned contracts, owners, docs, and tests;
- the exact fixture request has zero production-source matches; `rg` exit code
  `1` is expected for zero matches;
- forbidden scalar/flag names have zero matches in the changed production
  boundary; `rg` exit code `1` is expected.

Run a captured prompt assertion proving every `character_identity.boundaries`
value is text, every `relationship.axes` value is text, current episode
evidence remains present, memory rows carry their declared scope, and no raw
relationship id or global user id appears.

### Direct One-At-A-Time Real LLM

Run each node separately with `-q -s -m live_llm`; never run this file as one
batch. When prompt and projection surfaces are unchanged, reuse the fresh
artifacts from the preceding prompt/projection verification and run only the
focused case needed for a new question:

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_relational_willingness_live_llm.py::test_stranger_rejects -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_relational_willingness_live_llm.py::test_intermediate_33_observation -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_relational_willingness_live_llm.py::test_intermediate_67_observation -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_relational_willingness_live_llm.py::test_lover_accepts -q -s -m live_llm
```

Repeat each intermediate observation node until three fresh artifacts exist,
then run each memory, compliance, taboo, coercion, and six endpoint repetition
nodes individually. Expected: exact endpoint enums, valid intermediate enums,
coherent bid prose, no fixture leakage, and human-readable character judgment.
Intermediate enum values are observations and never fail this gate.

### Guarded Full-Service One-At-A-Time E2E

Run each node separately with `-q -s -m "live_llm and live_db"` when prompt,
projection, or public-surface code changed after the retained twelve-sample
package. Otherwise inspect and reuse the fresh package rather than spending
another full twelve-case rerun on unchanged service code:

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_relational_willingness_e2e_live_llm.py::test_stranger_visible_rejection -q -s -m "live_llm and live_db"
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_relational_willingness_e2e_live_llm.py::test_stranger_shared_memory_visible_rejection -q -s -m "live_llm and live_db"
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_relational_willingness_e2e_live_llm.py::test_lover_visible_acceptance -q -s -m "live_llm and live_db"
```

Expected for every case: guarded database assertion, one public `/chat` call,
full trace capture, exact seeded relationship, exact memory arm, typed decision,
authoritative collapse when sensitive, no non-accepting effect, coherent L3
plan, visible endpoint behavior, completed lifecycle, and no cross-arm rows.

### Call And Prompt Budget Gate

For each direct and E2E artifact record:

- prewarm call count;
- appraisal family/item/repair counts;
- ordinary and active goal counts;
- workspace count;
- action planning and authorization counts;
- content/preference counts;
- rendered character count per prompt;
- configured cap and maximum completion tokens;
- total response-path duration.

Expected: zero added calls, workspace count zero on sensitive turns, unchanged
caps and attempts, no required relationship/current-episode drop, and no prompt
overflow.

### Final Agent-Evaluated Endpoint Pair And Intermediate Observations

After all deterministic and direct-node gates pass, execute the frozen scenario
through the public service twelve times, one sample per process: three stranger,
three intermediate-33, three intermediate-67, and three lover samples.

A fresh package may be reused when no prompt, projection, or public-surface
code has changed since its capture.

1. Reset `_test_kazusa_live_llm` to the frozen baseline.
2. Seed the hashed character, scene, and shared persistent-memory bundle.
3. Seed one of the four exact relationship profiles under a fresh user.
4. Assert absence of current-user conversation, residue, and memory rows before
   the request.
5. Send the exact frozen request through `/chat`.
6. Wait for lifecycle completion and export full protected trace and DB effects.
7. Repeat until there are three fresh artifacts for each profile.

The parent prepares a twelve-sample evidence package containing the seed/hash manifest,
prewarm and RAG outputs, scoped evidence, relationship projection, ordinary
willingness object and prose, workspace collapse, action/effect decision, L3
plan, visible reply, call counts, and post-turn DB diff for every sample.

The independent review agent reads the plan, the exact fixture, and all twelve
packages. It evaluates character behavior from the visible reply in context,
using the typed stance and trace as supporting evidence. Its report assigns
`pass` or `fail` only to the six endpoint samples and the endpoint pair. It gives
descriptive observations, without a verdict, to the six intermediate samples:

- stranger passes only for exact `relationship_sensitive/reject`, an unambiguous
  visible refusal, and no enactment/effect;
- lover passes only for exact `relationship_sensitive/accept`, unambiguous
  willing visible acceptance, and no refusal/deflection/condition;
- the pair passes only when frozen input hashes match outside relationship state
  and all six sample verdicts pass;
- intermediate samples receive descriptive stance and visible-behavior labels,
  plus a monotonicity and transition-interval analysis; they receive no
  `pass`/`fail` behavioral verdict and cannot block endpoint sign-off;
- endpoint uncertainty, contradiction, missing evidence, or a schema/dialog
  mismatch is `fail`; there is no majority vote or parent override;
- an incomplete intermediate artifact is rerun until it is inspectable, while
  its eventual behavioral content remains ungraded.

The public test contains no deterministic visible-text refusal or acceptance
marker list. Visible wording is judged from the captured reply by the parent
and independent review agent; pytest retains only structural, typed, trace,
effect, lifecycle, and non-empty-surface gates.

This independent behavioral report is a mandatory final sign-off artifact, not
an advisory review. Completion and archival remain blocked until it passes and
the user accepts the report.

## Independent Code Review

Run this gate after all planned verification passes and before merge,
completion, archival, or final sign-off. Exactly one independent review
subagent receives the approved plan, completed short-horizon contract, full
diff, frozen test hashes, command logs, raw traces, prompt/call metrics,
readable reviews, and execution evidence. The reviewer does not implement
fixes.

Review scope begins with the mandatory final endpoint behavior judgment above,
then covers:

- Project rules and skill compliance for Python, CJK prompts, test style,
  environment usage, evidence handling, and plan lifecycle.
- Correct semantic ownership: no prewarm stance, no deterministic input
  classifier, no scalar relationship calculation, no prose-derived decision,
  no active-branch re-decision, and no L3 semantic override.
- Exact contract quality: memory scope, axis semantics, boundary semantics,
  decision shape, current-episode evidence coverage, ordinary owner,
  authoritative collapse, effect permission, observability, and L3 copying.
- Weak-model prompt quality: one positive procedure, stable vocabulary,
  no raw floats/ids, no duplicated relationship key, no test-shaped examples,
  and required context retained under cap.
- Behavior evidence: stranger rejection and lover acceptance as strict gates;
  six ungraded intermediate observations; agent-reported monotonicity, stance
  distribution, transition interval, shared-memory isolation, compliance
  distinction, taboo/coercion negatives, and visible surface fidelity.
- Counterfactual integrity: all twelve public-path samples use matching
  character/request/scene/memory/runtime hashes and differ only in native
  relationship state; the reviewer judges visible replies directly and keeps
  intermediate observations outside endpoint pass/fail.
- Reliability: malformed output replacement, attempt exhaustion, typed
  failure, non-sensitive workspace behavior, no durable non-accepting effect,
  and no added calls/caps.
- Scope: no changes to RAG workers, prewarm timing, relationship reducers,
  identity storage, personality data, dialog, database, adapters, or archived
  evidence.

The parent fixes concrete in-scope findings, records each fix, and reruns the
affected focused, integration, static, live, and E2E gates. A finding that
requires a new owner, schema field, call, cap, persistence behavior, or file
outside the approved surface blocks completion pending plan update and user
approval.

## Acceptance Criteria

1. Shared-memory prewarm remains before cognition and evidence-only; no RAG or
   prewarm component decides willingness.
2. Every promoted-memory evidence row entering Core V2 carries exactly one
   prompt-safe current-user or shared scope, with no raw user id.
3. Shared character/world memory cannot grant current-user trust, attachment,
   closeness, boundary safety, consent, or lover access.
4. Model-facing relationship axes use domain-specific text; default trust is
   unestablished and default boundary safety is unproven.
5. Model-facing boundary profile contains no raw float and explicitly
   distinguishes compliance under pressure from willingness.
6. Every ordinary goal result contains one valid
   `relational_willingness.v1` decision citing current episode evidence.
7. No active goal branch, appraisal, workspace model, action planner, or L3
   stage re-decides or rewrites that decision.
8. The frozen public-path stranger arm returns `relationship_sensitive/reject`
   and an unambiguous visible refusal in all three fresh samples.
9. The stranger arm receives the same adversarial shared persistent memory as
   the lover arm and still rejects without any accepted effect.
10. The frozen public-path lover arm returns
   `relationship_sensitive/accept` for the same request in every strict
    endpoint sample and visibly accepts it as a willing, character-compatible,
    non-coercive interaction.
11. Lover acceptance does not depend on shared memory or a legacy relationship
    insight string.
12. The 33% and 67% profiles each produce three retained real-model observations.
    The review agent reports their stance distribution, visible behavior,
    monotonicity, and rough acceptance-boundary interval. No intermediate stance
    is required and no intermediate result can pass or fail behavioral sign-off.
13. Character taboo and coercion negative cases may reject at lover topology,
    proving relationship does not erase character self-definition or scene
    safety.
14. `compliance_strategy="comply"` changes expression only and does not promote
    a stranger or hesitant relationship to acceptance.
15. Sensitive turns use the ordinary bid as primary, expose no supporting bid,
    and record the authoritative preservation reason.
16. Non-sensitive turns retain the existing model-authored workspace collapse
    path and baseline behavior.
17. `reject`, `deflect`, `negotiate`, and `conditional_accept` produce no
    authorized action, resolver request, accepted side effect, or persisted
    commitment for the request.
18. L3 and visible dialog preserve every stance; rejection cannot become
    consent and acceptance cannot become refusal solely because the content is
    intimate.
19. Malformed/missing decisions regenerate through the same ordinary owner;
    exhaustion produces typed pre-commit failure and never defaults to accept.
20. Relationship context and current episode evidence survive ordinary goal
    prompt fitting; `q:relationship_social` retains one canonical relationship
    payload.
21. Response-path call count does not increase; sensitive turns remove the
    workspace call; all existing caps and attempts remain unchanged.
22. Production prompts contain no fixture request, fixture topology label,
    expected stance sequence, hard-coded character name, run id, or scalar
    relationship calculation.
23. Focused, integration, regression, prompt-render, static, one-at-a-time
    real-LLM, guarded E2E, frozen paired-replay, and budget gates pass with
    retained evidence; all twelve profile hashes prove relationship state is the
    only semantic input difference.
24. Parent-authored readable reviews display real inputs, scoped evidence,
    state projections, model outputs, decisions, visible dialog, and quality
    findings; raw JSON is not the primary review surface.
25. The independent review agent marks each of the six endpoint samples and the
    counterfactual pair `pass`, reports no unresolved blocker, and judges the
    visible stranger rejection and lover acceptance as semantically genuine.
    It separately reports all six intermediate observations, monotonicity, and
    the rough acceptance-boundary interval without assigning them pass/fail.
    All findings are remediated and rerun, the user explicitly signs off the
    behavioral report, and the lifecycle record is archived correctly.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Local model still treats memory as permission. | Explicit memory scope plus native relationship authority procedure. | Stranger shared-memory direct and full-service counterfactuals. |
| Default zero bands remain permissive. | Axis-specific unestablished/unproven semantics. | Projection snapshots and stranger endpoint repetitions. |
| Lover is over-rejected by generic caution. | Positive compatible-lover acceptance invariant and exact surface preservation. | Three lover samples plus visible E2E acceptance. |
| Every intimate request becomes accepted for lovers. | Character taboo, coercion, scene, and self-definition remain authoritative. | Lover taboo and coercion negatives. |
| Intermediate behavior is abrupt or non-monotonic. | Keep production judgment semantic and use two fixed interpolation probes for diagnosis. | Agent-authored distribution, monotonicity, and transition-interval report; observation does not block sign-off. |
| Compliance is mistaken for consent. | Semantic boundary projection and explicit prompt distinction. | Compliance counterfactual with stranger and intermediate state. |
| Competing branch reverses stance. | Deterministic authoritative ordinary collapse with no supporting bids. | Fake-LLM contradictory-bid tests and observability assertions. |
| Action path persists a rejected request. | Enum-based deterministic effect permission. | Action/resolver denial and database-count tests. |
| L3 rewrites acceptance or rejection. | Exact decision in surface input and positive content/preference rules. | Surface-stage live and full-dialog E2E review. |
| Required context is dropped under pressure. | Relationship/current episode become required; typed context failure replaces silent loss. | Near-cap and irreducible-cap tests. |
| Prompt expansion harms latency. | No new call/cap, duplicate relationship removal, workspace skip. | Per-case prompt/call/duration metrics. |
| Fix becomes character-specific. | Role-neutral production prompt and generic schema; Asuna is the exact acceptance fixture. | Static name/fixture leakage checks and generic contract/projection tests. |
| Semantic contradiction remains inside one model bid. | Compact positive procedure, exact stance field, strict real-output review, and no hidden deterministic prose rewrite. | Raw bid/stance side-by-side review; any contradiction fails the case. |

## Execution Evidence

- 2026-08-03: draft rebaselined on clean commit `32d59aeb`, the completed
  short-horizon state-composition commit.
- 2026-08-03: source inspection confirmed the canonical post-prewarm
  `relationship_operational_context.v1`, qualitative model projection, parallel
  ordinary/appraisal execution, reactive `boundary_safety < -20` goal trigger,
  ten-point relationship delta cap, ordinary-first workspace fallback, and
  absence of a typed current relational-willingness handoff.
- 2026-08-03: diagnostic export by protected trace id, lifecycle
  `source_episode_id`, and conversation `cognition_graph.run_id` returned zero
  rows for `4f1c08a6bb1945559dcc98ec09d5a020`; no unsupported raw-trace causal
  claim is part of this plan.
- 2026-08-03: `rebaseline.md` records commit, worktree, environment, planned
  file hashes, deterministic baseline passes, and the pre-existing prewarm
  timeout.
- 2026-08-03: the tracked four-profile fixture and
  `focused_test_contract.json` were added. The focused test collection stopped
  at the planned missing `project_relationship_axis` symbol before production
  changes; integration, projection, prewarm-scope, and prompt-contract gates
  were added as parent-owned failing tests.
- 2026-08-03: DeepSeek production handoff completed under the bounded
  acknowledgement-then-execution protocol. The implementation report covered
  memory scope, qualitative axis projection, boundary projection, ordinary
  decision ownership, authoritative collapse, action denial, and exact L3
  copying; the focused inline contract checks passed.
- 2026-08-04: parent remediation required promoted-memory scope validation,
  corrected the ordinary prompt's `branch.goal_kind` path, fixed the
  authoritative collapse observability reason, tightened strict public trace
  capture, and clarified evidence-handle versus role-handle ownership.
- 2026-08-04: focused relational/contracts/projection gates passed **49**;
  prompt/budget and prompt-contract gates passed **54**; integration/action/
  surface handoff gates passed **112 with 4 deselected**; broader V2 gates
  passed **132**; and the seven non-deadlocking prewarm tests passed with one
  known baseline deadlock test deselected. The complete command record is in
  `deterministic_verification.md`.
- 2026-08-04: static checks found zero production matches for the frozen
  request, zero visible refusal/acceptance marker gates, an exact 32-file
  changed-file allowlist, and only the unchanged pre-existing
  `emotion_derivation.py` `_relationship_score` residual.
- 2026-08-04: direct real-LLM artifacts cover stranger, lover, intermediate
  33%, intermediate 67%, current-user memory, no shared memory, compliance,
  taboo, and corrected coercion cases. The parent-authored
  `affinity_internal_monologue_table.md` displays each case against the native
  affinity context.
- 2026-08-04: the fresh guarded public package contains twelve individually
  executed samples: three stranger rejects, three lover accepts, and three
  descriptive observations at each intermediate profile. The five readable
  reviews link the fresh artifacts and explicitly reserve visible-language
  quality judgment for the independent reviewer; no marker list determines
  endpoint outcome.
- 2026-08-04: user requested minimized reruns. Because no prompt, projection,
  or public-surface source changed after this package, the retained twelve-case
  package is reused while focused deterministic gates cover the remaining
  execution. A new public package is required if those surfaces change.
- 2026-08-04: the independent reviewer Chandrasekhar
  (`019fc794-074a-74b0-adad-4d75cc63011b`) completed the required final
  read-only re-review. It passed all six endpoint samples and the frozen pair
  from semantic visible-reply inspection, reported the 33%/67% samples as
  descriptive observations only, confirmed all six earlier findings resolved,
  and reported no in-scope blocker. The full report is
  `independent_code_review.md`.
- 2026-08-04: final focused verification was rerun after the Stage 7 review:
  49 focused, 54 prompt/budget, 112 integration with 4 expected deselections,
  132 broader V2, and 7 prewarm-neighbor tests passed. Production syntax,
  `git diff --check`, fixture/request leakage, marker-gate absence, and the
  exact 32-path worktree allowlist also passed. No prompt, projection, or
  public-surface source changed, so no redundant twelve-case live rerun was
  started.
- 2026-08-04: the user explicitly approved Stage 8 and lifecycle closeout
  after reviewing the independent semantic behavior result and residual risk.
  The local branch was fast-forwarded from `32d59aebc3f87e75723f19297b9a5a500527c27a`
  to `f323cd33c6f3112fc60faccf2e8036782efac472` from
  `origin/cognition_core_v2`; the incoming control-console changes were
  preserved and the registry conflict retained both plan rows. The completed
  plan is archived at
  `development_plans/archive/completed/bugfix/cognition_core_v2_relational_willingness_gradient_bugfix_plan.md`.
