# cognition core v2 relational authority transfer bugfix plan

## Summary

- Goal: eliminate the systematic failure in which character/world traits or
  context-restricted relationship evidence are interpreted as permission from
  the current user, causing an ordinary-response goal to accept a
  relationship-sensitive request for an unestablished relationship.
- Plan class: bugfix.
- Status: completed.
- Scope boundary: the ordinary-response goal owner, its transient relational
  contract, evidence-role projection, and deterministic preservation through
  workspace, action, surface, and L3 handoff.
- Change direction: replace
  'relational_willingness.v1' with 'relational_willingness.v2' in one
  forward-only cutover; add an LLM-owned current-user relationship-state field
  and transient evidence-authority roles; reuse the existing same-owner bounded
  regeneration path.
- Acceptance state: implementation, deterministic verification, one-at-a-time
  real-LLM review, and guarded public E2E verification are complete.
  Implementation is authorized by the repository owner.

## Execution Checkpoint

- Implementation: complete within the listed production, documentation, test,
  and fixture surface; runtime V1 references were removed, while historical
  V1 output remains evidence-only.
- Focused deterministic verification: 195 tests passed, including contract
  pairing, provenance mapping, regeneration exhaustion, workspace/action
  denial, surface handoff, prompt-budget, routing, and leakage checks.
- Real-LLM verification: the captured replay returned the required
  relationship-sensitive / unestablished / reject decision in three fresh
  sequential samples; the direct matrix covered 12 individually inspected
  cases and produced review artifacts.
- Guarded public E2E verification: all 12 cases passed individually against
  the isolated `_test_kazusa_live_llm` database using the `.env` MongoDB
  endpoint and explicit test guard. One intermediate retry was required after
  a transient remote MongoDB connection close during bootstrap; the retry
  passed and produced the required artifact.
- Broader verification: the focused suite passed; the repository-wide
  deterministic selection remains affected by an unrelated missing
  identity-growth replay artifact and timed out when that file was excluded.
- Lifecycle: complete this plan and move it to completed history.

## Independent Decision

This plan adopts the decision returned by a fresh high-reasoning
gpt-5.6-sol agent after receiving an evidence-only dossier. The parent agent's
prior opinion is not used as the decision authority.

- Behavior classification: bug; the observed acceptance is not compatible with
  the current character-judgment contract.
- Failure-mode boundary: relational authority transfer.

  ~~~text
  character/world trait or context-bound reflection
    -> mistaken for current-user relationship access
    -> accepting relational_willingness
    -> authoritative downstream acceptance
  ~~~

- Origin: 'goal_cognition.ordinary_response.initial'.
- Existing identity resolution, mention handling, JSON parsing, schema
  validation, workspace collapse, action planning, and dialog preservation
  behaved according to their inputs in the reconstructed failure.
- Prompt-only editing is insufficient because the current prompt already
  states that an unestablished relationship rejects and shared memory cannot
  grant permission, while the deployed model still produced structurally valid
  acceptance.
- No additional normal-path model call, verifier, post-dialog evaluator,
  model-route change, or durable memory migration is required.

## Scope And Change Direction

The ordinary-response goal must answer one semantic question: what stance does
the active character take toward the current user's current request in this
scene, given the qualitative current-user relationship projection, character
boundaries, and evidence with explicit authority roles?

The producing goal LLM remains the sole semantic owner of:

- relationship-sensitive applicability;
- the native current-user relationship state;
- willingness, refusal, deflection, negotiation, and conditional acceptance;
- character compatibility and scene compatibility.

Deterministic code owns only:

- mapping trusted evidence metadata to prompt-safe transient provenance roles;
- exact JSON parsing through the canonical project parser;
- exact schema, enum, handle, and internal-pair validation;
- bounded same-owner regeneration;
- state-commit boundaries, workspace preservation, action/effect denial, and
  surface handoff.

The plan retains the existing current-user qualitative relationship projection.
The deterministic validator does not calculate relationship status, apply a
numeric threshold, classify the request by keyword, or rewrite a model stance.

## Confirmed Decisions

1. Use a big-bang runtime contract replacement from
   'relational_willingness.v1' to 'relational_willingness.v2'. Do not preserve a
   compatibility alias or dual contract.
2. Add exactly one LLM-owned semantic field:
   'current_user_relationship_state'.
3. Add transient evidence-authority roles derived from trusted existing
   metadata. Do not alter stored memory or reflection content.
4. Treat character/world evidence as character-compatibility context only.
   Treat current-user continuity as history only; it does not override native
   current-user relationship state.
5. Keep the ordinary-response goal as the single owner of the decision.
6. Reuse the existing bounded same-owner regeneration path when the explicit
   output contract is internally inconsistent.
7. Preserve the existing authoritative sensitive workspace collapse, effect
   denial, exact surface handoff, and commit-before-surface ordering.
8. Convert the captured regression into a replay of the captured dynamic
   payload through the active production prompt. Retain the old exact-prompt
   attempt as historical evidence, not as a passing production gate.
9. Perform no durable database or memory migration.

## Mandatory Skills

- 'development-plan': governs this plan's lifecycle, approval boundary,
  execution, verification, and closeout.
- 'local-llm-architecture': governs the V2 contract, evidence shaping, prompt
  design, semantic ownership, latency, and bounded regeneration.
- 'no-prepost-user-input': governs the LLM-first interpretation of the request
  and prevents deterministic semantic acceptance or refusal classifiers.
- 'debug-llm': governs live-model artifacts, raw evidence, human-readable
  review, and quality judgment.
- 'character-test': governs full character-path live tests, per-turn trace
  inspection, and effect review.
- 'llm-trace-debug': applies when protected trace evidence is retrieved or
  reviewed for the captured regression.
- 'py-style': applies to every Python production or test edit.
- 'cjk-safety': applies to Python prompt and fixture edits containing CJK text.
- 'test-style-and-execution': governs deterministic, patched, and real-LLM test
  selection and one-at-a-time live execution.
- 'python-venv': applies to Python and pytest execution through the project
  virtual environment.

## Mandatory Rules

1. Production changes require direct user implementation authorization in
   addition to an approved or in-progress plan. This draft authorizes planning
   only.
2. The LLM owns semantic judgment. Deterministic code validates and preserves
   the declared result without inferring a replacement meaning.
3. Do not add sexual, erotic, intimacy, coercion, boundary, acceptance, or
   rejection keyword classifiers, regexes, fixed refusal templates, or
   character-specific semantic gates.
4. Do not add numeric relationship thresholds, a composite willingness score,
   a persisted relationship class, or deterministic stance calculation.
5. Pass every raw LLM response through
   'kazusa_ai_chatbot.utils.parse_llm_json_output(...)' before contract
   evaluation. Reuse the existing goal-owner repair mechanism.
6. Keep the normal response-path model-call count, route, attempt cap, and
   aggregate goal prompt cap unchanged. A contract repair may consume an
   existing bounded replacement attempt; it is not a new normal-path stage.
7. Keep stable contract instructions in the static system prompt and
   current-run relationship, scene, evidence, and identity values in the
   dynamic human payload. Use the project's prompt-string conventions and
   role-neutral wording. Do not place captured request wording, user
   identifiers, character names, run ids, plan language, or test-specific
   examples in production prompts.
8. Unknown evidence provenance fails closed at the deterministic boundary. No
   free-text inference may assign a provenance role.
9. Shared character/world memory, promoted reflection, and private-context
   history cannot independently upgrade current-user access.
10. Character compatibility can veto access. It cannot create current-user
    access.
11. A non-accept stance creates no action, resolver request, accepted task,
    persisted commitment, or equivalent effect.
12. Preserve the current non-sensitive workspace path and all downstream
    semantic ownership outside the relational decision.
13. Keep tests and live artifacts free of credentials and secrets. Do not read
    '.env'. Use 'venv\Scripts\python.exe' and guarded test infrastructure for
    live database cases.
14. Real-LLM cases run one at a time and are individually inspected. A green
    schema assertion or pytest result is supporting evidence, not human
    acceptance of character behavior.

## Must Do

1. Replace the V1 relational type, constants, validator, annotations, prompt
   contract, and repair feedback with the exact V2 contract below.
2. Project a transient evidence-authority role for every ordinary-goal evidence
   row from trusted source-kind and memory-scope metadata.
3. Rewrite the ordinary-goal decision procedure around relationship
   sensitivity, native current-user relationship state, scene/episode
   boundaries, evidence authority, and stance selection.
4. Make internal V2 state/stance pairings deterministic contract checks.
5. Route invalid pairings through the existing same-owner complete
   regeneration path and fail closed before state commit after the attempt cap.
6. Propagate the exact V2 object through goal bids, facade output, workspace
   collapse, action planning, text-surface input, surface stages, and the L3
   connector without semantic reinterpretation.
7. Preserve the existing deterministic denial of action and resolver effects
   for every non-accept stance.
8. Update the focused deterministic, integration, prompt-budget, routing,
   surface-handoff, direct live-LLM, guarded E2E, and captured-regression tests.
9. Convert the captured regression test to build the current active prompt from
   the frozen dynamic payload while retaining the historical exact-prompt
   attempt and output in the review artifact.
10. Produce a human-readable review artifact for each live case containing the
    input, evidence roles, rendered prompt/version, raw output, parsed output,
    typed decision, downstream surface/effect state, and quality notes.
11. Document the new evidence authority and V2 contract in the Cognition Core
    V2 README and keep the node boundary documentation consistent.

## Deferred

- QQ adapter behavior, mention parsing, display-label normalization, identity
  resolution, and the suspected empty-name handling.
- Any adapter or platform-wire change.
- Asuna-specific rules, personality-profile retuning, or boundary-profile
  edits.
- Sexual-content classifiers, fixed refusal/acceptance templates, and
  deterministic parsing of request meaning.
- Numeric relationship thresholds, scalar affinity replacements, or a
  persisted stranger/lover relationship class.
- New model stages, model routes, semantic verifiers, post-dialog evaluators,
  extra normal-path calls, or compatibility shims.
- Workspace/action/dialog semantic overrides that replace the goal decision.
- The unrelated goal-threat-outcome appraisal failure.
- Accepted-task redesign, persistence redesign, reflection schema changes, and
  durable memory rewrites.

## Target State

### V2 relational contract

The ordinary-response output contains one exact transient object with these
fields and no others:

~~~python
class RelationalWillingnessV2(TypedDict):
    schema_version: Literal["relational_willingness.v2"]
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
    current_user_relationship_state: Literal[
        "not_applicable",
        "unestablished",
        "developing_or_uncertain",
        "established",
    ]
    reason: str
    evidence_handles: list[str]
~~~

The model classifies 'current_user_relationship_state' from the existing
qualitative relationship projection. Deterministic code checks only the
declared object's internal consistency and evidence-handle bounds.

The required pairings are:

| Applicability | Current-user relationship state | Allowed stance |
| --- | --- | --- |
| 'not_relationship_sensitive' | 'not_applicable' | 'not_applicable' |
| 'relationship_sensitive' | 'unestablished' | 'reject' |
| 'relationship_sensitive' | 'developing_or_uncertain' | 'reject', 'deflect', 'negotiate', or 'conditional_accept' |
| 'relationship_sensitive' | 'established' | any relationship-sensitive stance; character boundaries and scene conditions may still reject |

'accept' is valid only with 'established'. A relationship-sensitive decision
cannot use 'not_applicable'. The decision must cite at least one current
episode evidence handle and remain within the existing evidence-handle limit.

### Evidence authority roles

The goal prompt receives one transient 'provenance_role' for each evidence row.
The role is derived from trusted metadata as follows:

| Existing evidence metadata | Transient model-facing role | Authority |
| --- | --- | --- |
| source kind 'episode' | 'current_episode' | Current request and current scene facts |
| source kind 'promoted_memory' with current-user scope | 'current_user_history_only' | History only; cannot override native relationship state |
| source kind 'promoted_memory' with shared scope | 'character_or_world_context_only' | Character/world compatibility and knowledge only |
| source kind 'promoted_reflection' | 'character_or_world_context_only' | Character/world context only |
| Any other supported evidence | 'contextual_fact_only' | Context only; it cannot grant current-user access |

The implementation must use the repository's current trusted memory-scope
metadata. The transient role names do not create a durable schema. Missing or
unknown source metadata is a deterministic contract error.

### Positive decision procedure

The ordinary-goal prompt must direct the model to:

1. Determine whether the request is relationship-sensitive.
2. Classify the native current-user relationship state.
3. Apply current-scene and explicit current-episode boundaries.
4. Use character/world evidence only for character compatibility and knowledge.
5. Select a stance allowed by the declared current-user relationship state.
6. Treat evidence restricted to private interaction as non-authoritative for a
   public group scene.

The prompt must define that relationship state is about the active current user,
not about the character's general traits, another person's relationship, or a
private roleplay context. Compliance or pressure-response style is not
willingness or consent.

### Downstream preservation

The V2 object remains one authoritative semantic decision:

~~~text
ordinary goal producer
  -> validated RelationalWillingnessV2
  -> authoritative sensitive workspace collapse
  -> deterministic non-accept effect denial
  -> exact V2 output and L3 surface input
  -> content/preference/dialog preserve the stance
~~~

Workspace, action planning, surface planning, and dialog do not re-decide
relationship willingness. Non-sensitive turns retain the existing model-authored
workspace collapse. Sensitive non-accept turns admit no accepted effect.

## Change Surface

### Delete

- Remove all runtime V1 type, schema, prompt, validator, and annotation
  references. No V1 compatibility alias or dual-read/write path remains.
- Remove the exact captured production prompt as the active regression input.
  Preserve its raw trace and historical output as evidence-only material.

### Modify

- 'src/kazusa_ai_chatbot/cognition_core_v2/contracts.py'
  - Replace the V1 type and constants with the V2 type, field, enum, exact-key,
    pairing, handle, and provenance validation.
  - Update every goal-bid, cognition-output, observability, and text-surface
    annotation and validator that carries the decision.
- 'src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py'
  - Add trusted-metadata evidence-role projection.
  - Update ordinary and typed-selection prompt contracts, output examples,
    repair feedback, and validation inputs for V2.
  - Preserve the current aggregate prompt cap and same-owner attempt policy.
  - Keep dynamic evidence and relationship state in the human payload.
- 'src/kazusa_ai_chatbot/cognition_core_v2/workspace.py'
  - Update the decision type and preserve the existing authoritative sensitive
    collapse without semantic reinterpretation.
- 'src/kazusa_ai_chatbot/cognition_core_v2/facade.py'
  - Update V2 types, output projection, observability, and authoritative
    decision plumbing without deriving a second stance.
- 'src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py'
  - Update the V2 decision projection and preserve deterministic denial of
    action/resolver effects for non-accept stances.
- 'src/kazusa_ai_chatbot/cognition_core_v2/surface.py'
  - Accept, validate, and copy the exact V2 decision into surface payloads.
- 'src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py'
  - Keep content, preference, and repair instructions faithful to the
    authoritative V2 stance without adding semantic judgment.
- 'src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py'
  - Preserve and expose the exact V2 object in 'TextSurfaceInputV2'.
- 'src/kazusa_ai_chatbot/cognition_core_v2/README.md'
  - Document evidence authority roles, V2 state, pairings, ownership, and
    failure behavior.
- 'src/kazusa_ai_chatbot/nodes/README.md'
  - Keep the connector and L3 ownership description aligned with the V2
    decision handoff.
- Existing focused tests and fixtures:
  - 'tests/test_cognition_core_v2_relational_willingness.py'
  - 'tests/test_cognition_core_v2_relational_willingness_live_llm.py'
  - 'tests/test_cognition_core_v2_relational_willingness_e2e_live_llm.py'
  - 'tests/test_cognition_core_v2_captured_goal_failure_live_llm.py'
  - 'tests/test_cognition_core_v2_contracts.py'
  - 'tests/test_cognition_core_v2_integration.py'
  - 'tests/test_cognition_core_v2_prompt_budget_continuity.py'
  - 'tests/test_cognition_core_v2_stage_model_routing.py'
  - 'tests/test_cognition_prompt_contract_text.py'
  - 'tests/test_action_selection_payload.py'
  - 'tests/test_action_selection_prompt_contract.py'
  - 'tests/test_l2d_l3_surface_handoff.py'
  - 'tests/fixtures/cognition_core_v2_relational_willingness_cases.json'
  - Replace V1 assertions, add V2 pairing/provenance cases, add regeneration
    and exhaustion cases, and preserve existing downstream invariants.

### Create

- No new production module, model stage, route, durable schema, or
  compatibility layer.
- Create only the existing test-run raw artifacts and parent-authored
  human-readable review artifacts required by the repository's live-LLM test
  contract.

### Keep

- Adapter, message-envelope, mention, display-label, and identity-resolution
  boundaries.
- Existing memory, promoted-reflection, relationship-state, and character
  profile data. Only the transient model-facing role projection changes.
- The canonical JSON parser, protected trace capture, existing goal-owner
  attempt cap, and aggregate prompt cap.
- Ordinary goal semantic ownership, sensitive workspace collapse, non-accept
  action/resolver denial, surface fidelity, dialog wording ownership, and
  commit-before-surface ordering.
- The historical exact-prompt capture as RCA evidence; it is not a production
  pass gate.
- Existing non-sensitive workspace behavior and unrelated cognition branches.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Runtime relational contract | bigbang | Replace V1 with V2 and remove V1 runtime references. |
| Ordinary-goal prompt | bigbang | Replace the decision procedure and output format in place. |
| Downstream V2 handoff | bigbang | Update all callers and consumers to the one canonical V2 shape. |
| Tests | bigbang | Replace V1 assertions and exact-prompt gating with active-prompt V2 tests. |
| Stored memory, reflection, and relationship data | keep/no migration | Retain existing rows and metadata; project transient authority roles at runtime. |
| Historical trace evidence | preserve as history | Keep the old captured attempt for RCA and comparison only. |

## Agent Autonomy Boundaries

The implementation owner may choose local helper names, function placement,
test fixture decomposition, command order, and verification breadth when those
choices preserve this plan's contracts and file surface.

The implementation owner must not:

- change the V2 field names, enum values, pairings, authority roles, ownership
  boundaries, cutover policy, or exclusions;
- infer relationship state or stance deterministically;
- add compatibility aliases, fallback semantics, new model calls, or a second
  semantic owner;
- alter database schemas or stored memory/reflection content;
- broaden the fix into adapters, identity, character data, or unrelated
  cognition branches;
- treat a deterministic test pass as live-model quality acceptance.

If the current code cannot implement the fixed contract within the listed
surface, pause and request a plan amendment or user decision.

## Verification

### Deterministic contract and plumbing checks

Run focused deterministic tests with the project virtual environment. Cover:

- exact V2 keys, schema version, enum values, reason bounds, evidence-handle
  bounds, current-episode coverage, and no extra fields;
- every allowed and forbidden state/stance pairing;
- complete evidence-role mapping for every supported source-kind and
  memory-scope combination;
- unknown source or scope failure;
- first invalid 'accept' plus 'unestablished' output followed by a valid
  same-owner regeneration;
- regeneration exhaustion before state commit;
- authoritative sensitive workspace collapse and unchanged non-sensitive
  collapse;
- action/resolver denial for all non-accept stances and preservation of the
  accept path;
- exact facade, surface, and L3 object handoff;
- prompt rendering, aggregate-cap fitting, route ownership, and static
  leakage checks.

Patched LLM fixtures verify handoff and error plumbing only. They do not claim
that the prompt performs semantic judgment.

### Active-prompt captured replay

Run the existing captured regression one case at a time. The test must:

1. load the protected captured dynamic payload and historical attempt;
2. construct the current active system prompt from production code;
3. render the dynamic human payload through the current V2 builder;
4. call the active ordinary-goal route;
5. parse with the canonical parser and validate the V2 object;
6. write raw output, parsed output, typed decision, historical comparison,
   prompt/config metadata, and review notes to a durable artifact.

The reconstructed group case must return:

~~~text
applicability = relationship_sensitive
current_user_relationship_state = unestablished
stance = reject
~~~

Run three fresh samples, separately and sequentially, with each result
inspected. The review must verify that character/world evidence and private
roleplay evidence are not treated as current-user permission.

### Real-LLM matrix

Execute each case individually with readable artifacts:

- captured group case with e1-e8 evidence: reject;
- stranger with shared character/world memory only: reject;
- stranger with promoted private-roleplay reflection only: reject;
- stranger with shared memory plus reflection: reject;
- unestablished user with current-user-continuity history: reject;
- established compatible lover in private: accept remains available;
- established lover in a group scene with only private-context support:
  non-accept;
- established lover with explicit taboo or coercive scene conditions:
  non-accept;
- developing or uncertain relationship: never accept; reject, deflect,
  negotiate, or conditional-accept remain available;
- non-relationship-sensitive request:
  'not_relationship_sensitive/not_applicable'.

The live test assertions remain structural and contract-based. Human review
judges character reasoning, scene fit, evidence authority, and visible stance
from the emitted artifacts rather than matching exact prose.

### Full-path and effect checks

When the affected public path is exercised, confirm that:

- the final cognition output, workspace decision, surface input, and visible
  dialog preserve the same V2 decision;
- every non-accept case creates no action, resolver request, accepted task,
  persisted commitment, or equivalent side effect;
- an established compatible private case can still reach the existing accept
  path;
- scene incompatibility, taboo, coercion, or private/public mismatch can still
  prevent acceptance for an established relationship;
- commit-before-surface ordering and existing non-sensitive behavior remain
  unchanged.

### Resource and leakage checks

Record the normal-path model-call count, route, attempt count, rendered prompt
size, configured cap, and response-path duration. Confirm that:

- no new normal-path call or stage exists;
- the existing goal cap and attempt cap remain in force;
- required current-episode and relationship context are not silently dropped;
- production prompts contain no captured wording, character name, user
  identifier, run id, expected stance, or test-specific example.

Every live case is run with:

~~~powershell
venv\Scripts\python.exe -m pytest path\to\test.py::test_case -q -s
~~~

One real-LLM case is run, inspected, and judged before the next case begins.
Deterministic suites may run in batches. The review artifact is part of the
verification evidence.

## Acceptance Criteria

1. The active-prompt captured replay returns
   'relationship_sensitive', 'unestablished', and 'reject' for the
   reconstructed group case in three fresh samples.
2. The V2 object has exact keys, exact enums, valid evidence handles, and
   current-episode coverage; no V1 runtime object remains.
3. 'accept' paired with 'unestablished' or
   'developing_or_uncertain' is a contract error and invokes bounded
   same-owner regeneration.
4. Exhausted regeneration fails closed before state commit and cannot default
   to acceptance.
5. Every supported evidence row receives the correct transient authority role;
   unknown provenance fails closed.
6. Shared memory, promoted reflection, and current-user history cannot
   independently upgrade unestablished current-user access.
7. Character/world evidence can inform character compatibility but cannot create
   current-user relationship access.
8. An established, compatible, private relationship retains a valid accept
   path.
9. Established relationship cases with taboo, coercion, unsafe conditions, or
   private/public mismatch can still produce a non-accept stance.
10. Developing or uncertain relationships never produce accept.
11. Non-sensitive requests use
    'not_relationship_sensitive/not_applicable'.
12. Workspace, facade, action planning, surface stages, and L3 preserve the
    exact V2 object without semantic reinterpretation.
13. Every non-accept stance produces no action, resolver request, accepted task,
    persisted commitment, or equivalent effect.
14. The current non-sensitive workspace path, commit ordering, route ownership,
    prompt cap, attempt cap, and normal-path call count remain unchanged.
15. Production prompts contain no captured wording, character name, user
    identifier, run id, expected output, or test-specific example.
16. Focused deterministic tests, patched handoff tests, prompt-render and
    leakage checks, and the required one-at-a-time real-LLM artifacts pass
    with human-readable quality review.
