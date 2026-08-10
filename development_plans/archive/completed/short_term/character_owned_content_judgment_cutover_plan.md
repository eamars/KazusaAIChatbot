# character-owned content judgment cutover plan

## Summary

- Goal: make content acceptance, refusal, negotiation, and conditional participation an explicit character-cognition decision throughout the live Cognition V2 response path.
- Status: completed.
- Scope boundary: the platform-agnostic live text/cognition path, its typed stance contracts, prompt projections, action-effect propagation, surface planning, and the reachable legacy lexical-guard field.
- Change direction: remove application-authored general moral, unsafe-content, sexual-content, consent, and refusal policy from semantic prompts and semantic gates. Keep the existing typed character stance and let the character process choose it from identity, current evidence, affect, relationship context, and character-owned branches.
- Acceptance state: implementation is authorized by the user's explicit approval; deterministic verification, focused live evidence, and the Asuna fixture evidence are recorded. The user-directed `forbidden_phrases` closeout condition is satisfied; lifecycle closeout follows the final registry move below.

## Confirmed decisions

1. The configured model is assumed not to refuse. Prompt design must not rely on provider refusal or add a provider-like safety policy.
2. Harmful, sexual, and otherwise unsafe subject matter may reach the model and final wording when the character process selects an accepting stance.
3. `autonomy_boundary` remains a character-owned branch. Its refusal, negotiation, conditions, or acceptance are valid only as the character's selected decision; no downstream stage may add a second general safety decision.
4. `ordinary_response` remains the sole public owner of the sensitive-turn typed stance. `autonomy_boundary`, `safety_coping`, `moral_repair`, `social_care`, and `bond_protection` remain character-owned branch judgments that contribute evidence, motive, and competing goals; the existing deterministic arbitration keeps the ordinary stance as the canonical content decision. No branch is a general moderation authority.
5. Deterministic code continues to own structure, enum/type validation, evidence provenance, persistence, capability authorization, runtime truth, privacy/storage metadata sanitization, bounded retries, and delivery eligibility.
6. The cutover keeps `relational_willingness.v2` as the canonical typed character-stance object and replaces `current_turn_relational_willingness.v1` with `current_turn_relational_willingness.v2`, whose `decision` preserves the complete validated V2 object. This is an atomic big-bang contract update with no alias carrier or parallel vocabulary.

## Scope And Change Direction

The current path contains several independent restrictions that can prevent an accepting character decision from reaching the surface:

- `goal_cognition.py` prompts impose relationship-state/stance pairings, current-episode refusal precedence, and a general compliance/consent interpretation.
- `contracts.py` deterministically rejects accepting stances for `unestablished` and `developing_or_uncertain` relationship states.
- `cognition_resolver/contracts.py` repeats the recurrence carrier's decision-pairing policy, even though its current table is broader than the core contract.
- `action_selection.py` treats the relational result as a permission gate rather than as the already-selected character stance.
- `facade.py` and `workspace.py` deterministically collapse relationship-sensitive turns to the ordinary bid and suppress competing branch bids; this must be documented as the deliberate sole-owner arbitration boundary.
- `surface.py`, `persona_supervisor2_l3_surface.py`, and `dialog_agent.py` do not carry the typed stance through every surface/output boundary, and the terminal dialog candidate can currently escape semantic verification.
- `state_projection.py` injects repository-default standards such as avoiding harm and respecting boundaries, and appends a general consent conclusion to the character's pressure-response descriptor.
- `surface_stages.py` authorizes generic safety, content-review, intimacy, consent, and courtesy categories as visible boundaries without a source-bound contract.
- The legacy `forbidden_phrases` field remains as stale typed/test residue and must be removed from the active contract surface. The audit must distinguish that obsolete field from the retained expression-continuity behavior, which is carried by the narrower `lexical_avoidances` surface field and never classifies content.
- `config.py` and `docs/HOWTO.md` retain the unused `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED` refusal-oriented feature switch and must be cut over with its tests.

The target path is:

```text
typed evidence and character identity
  -> semantic appraisal and character-owned branch activation
  -> character goal/stance decision
  -> structural validation only
  -> action planning that honors the selected stance
  -> surface planning that preserves the selected stance
  -> dialog wording that preserves the selected stance
```

The model may select `reject`, `deflect`, `negotiate`, `conditional_accept`, or `accept` for any relationship-sensitive relationship state. Relationship state remains descriptive evidence, not a deterministic permission condition. `not_relationship_sensitive/not_applicable/not_applicable` remains the sole non-sensitive structural form. A sensitive decision must still have a real relationship state, a valid stance, and current-episode evidence coverage.

## Mandatory Skills

- `.agents/skills/development-plan` for lifecycle, cutover, ownership, and acceptance evidence.
- `.agents/skills/local-llm-architecture` for semantic ownership, bounded local-model prompts, and deterministic-boundary review.
- `.agents/skills/no-prepost-user-input` for preserving LLM-first interpretation of user intent and character stance.
- `.agents/skills/py-style` and `.agents/skills/cjk-safety` for Python prompt/source changes containing CJK text.
- `.agents/skills/test-style-and-execution` for deterministic and live test selection, execution, and inspection.
- `.agents/skills/debug-llm` for prompt/output regression artifacts and live model evidence.
- `.agents/skills/llm-trace-debug` for protected trace retrieval and raw stage evidence when validating a captured run.
- `.agents/skills/character-test` for live character-path behavior checks when the debug/service channel is used.
- `.agents/skills/database-data-pull` for read-only database evidence pulls used by live diagnostics.
- `.agents/skills/python-venv` before any dependency or interpreter setup.

## Mandatory Rules

1. Do not add a keyword classifier, unsafe-content detector, moderation pass, lexical blacklist, refusal postprocessor, or provider-like refusal instruction for semantic content.
2. Do not make deterministic code infer or rewrite character stance from prose, relationship numbers, harm labels, sexual terms, or generic moral categories.
3. LLM stages own semantic appraisal, character branch selection, stance, and response goal. Deterministic code validates shape, enum membership, evidence handles, provenance, and downstream truth without deciding whether the subject matter is allowed.
4. Character identity, authored taboos, boundary profile, affect, relationship context, and evidence remain available as character inputs. They may produce an accepting or refusing character decision; they must not be converted into an application-wide policy.
5. The current episode remains authoritative for facts, roles, and event identity. It no longer supplies an unconditional semantic veto merely because its text contains refusal, coercion, harm, or boundary language.
6. `accept` is structurally valid for every relationship-sensitive relationship state. No stage may regenerate or fail a candidate solely because the state is `unestablished` or `developing_or_uncertain`.
7. A non-accepting stance may still suppress action/resolver effects when that suppression is the direct propagation of the selected character stance. It must not be described or implemented as a general moral or safety policy.
8. Preserve capability authorization, action-effect truth, privacy/provenance isolation, raw-storage and adapter-marker sanitization, schema validation, retry limits, persistence invariants, and delivery checks.
9. Make the cutover in one canonical contract update. Do not add compatibility shims, alias fields, fallback mapping, or parallel prompt vocabularies.
10. Do not read `.env`. Use `venv\\Scripts\\python`; run live LLM cases one at a time with raw output inspected.

## Must Do

### 1. Rewrite semantic ownership prompts

- Update `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`:
  - Replace the hard relationship-state/stance matrix with a positive character-judgment procedure.
  - Present relationship state as context the character weighs, not as a permission gate.
  - Remove unconditional current-episode refusal precedence and the application-authored conclusion that compliance is not willingness or consent. Preserve current-episode facts, role direction, evidence handles, and the ability for the character to select refusal or conditions.
  - Update ordinary, required-selection, and same-owner repair instructions so the full stance domain is accepted for every sensitive relationship state.
  - Keep repair feedback structural: allowed enum values, exact fields, evidence coverage, and handles; remove pairing-specific regeneration feedback.
- Review `semantic_appraisal.py` and `branch_activation.py` without removing evidence-backed character appraisal. Harm, threat, moral, and boundary axes remain observations that can activate character-owned goals; they must not be converted into a generic response veto.
- Review all other active Cognition V2 prompts, including action planning, workspace, resolver authorization, content planning, preference, visual planning, and dialog verification. Remove only application-authored semantic safety/refusal policy. Keep role, capability, evidence, output-shape, and runtime-truth instructions.

### 2. Make branch arbitration explicit

- Keep `ordinary_response` as the sole typed owner of `relational_willingness` for a sensitive turn. Treat `autonomy_boundary` and other active branches as character-owned evidence/motive contributors whose bids may be suppressed by the existing ordinary-owner collapse after all branch cognition has run.
- Update `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`, `workspace.py`, and their documentation/tests to state and assert this ownership. The collapse must never inspect content words, relationship numbers, harm labels, or generic policy; it only checks the validated ordinary decision and preserves exact bid equality.
- Ensure ordinary goal cognition receives the relevant branch evidence/context before choosing its stance. A branch-selected refusal remains a character-originated reason that ordinary cognition may preserve; absence of such a reason cannot be replaced by a deterministic refusal.

### 3. Remove deterministic semantic gates

- Update `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` so `validate_relational_willingness` validates only:
  - exact fields and schema version;
  - the two applicability forms;
  - a non-`not_applicable` stance and real relationship state for sensitive decisions;
  - enum/type/length bounds, evidence-handle validity, and current-episode coverage.
- Delete the three relationship-state/stance rejection branches and their policy-specific errors. Add a complete matrix test proving all five sensitive stances validate for all three real relationship states.
- Replace `current_turn_relational_willingness.v1` with `current_turn_relational_willingness.v2` in `src/kazusa_ai_chatbot/cognition_resolver/contracts.py` and `state.py`. The carrier must contain the complete validated `relational_willingness.v2` decision, episode identity, and branch owner; it must not reconstruct reason/evidence or re-evaluate semantic content.
- Update `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` and the goal materialization path so recurrence copies the complete validated decision exactly. If its evidence handles are unavailable in the same episode, return a typed recurrence contract failure and invoke the owning bounded regeneration path; never invent a generic reason or substitute a minimum handle.
- Update `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py` so action/resolver suppression is explicitly downstream propagation of the selected character stance. Preserve suppression for non-accepting selected stances and capability authorization for accepting stances; remove permission-policy naming and wording that implies an independent consent or safety authority.

### 4. Carry the typed character stance through surface and dialog

- Add the exact validated `relational_willingness.v2` object to `TextSurfaceOutputV2` and preserve it in normal, degraded, and repaired outputs in `surface.py`.
- Update `persona_supervisor2_l3_surface.py`, `surface_stages.py`, and `contracts.py` so the surface owner receives and returns the same stance without semantic rewriting.
- Include the typed stance in dialog-generation and semantic-fidelity packets. The dialog verifier must compare final wording polarity against the authoritative character stance and report a semantic-fidelity issue when they differ; it must not choose a new stance.
- Require every deliverable dialog candidate, including the final attempt, to pass stance/role/semantic fidelity. If the bounded verifier cannot establish fidelity after the cap, use a typed delivery failure or a deterministic surface-preserving fallback; do not deliver an unverified candidate whose polarity may reverse the character decision.
- Add exact-equality propagation tests across cognition output, text-surface input/output, dialog input, repair, degraded output, and final delivery.

### 5. Remove generic model-facing defaults from live content judgment

- Update `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py` so repository-default `standards` are not projected into the live cognition prompt and no standard handles are exposed as model-facing semantic evidence. Keep the persisted state shape and internal validation unchanged in this cutover; the standards field remains dormant storage until a separately approved character-state redesign.
- Replace `_compliance_strategy_semantic` output and its documentation with a bounded character pressure-response descriptor only. Remove the appended general conclusion about willingness or consent.
- Keep authored character identity fields such as `personality_judgment.taboos` and boundary-profile behavior descriptors available to cognition. Verify they are treated as character context and not as a deterministic block.
- Make the corresponding `prompt_budget.py` reduction rule explicit for an empty standards projection rather than promising that model-facing standards are never removed.
- Update `semantic_appraisal.py` character-constraint selection and `semantic_source_planner.py` permitted-handle domains so `s` standard handles cannot enter any live model-facing semantic question after projection.
- Update prompt-budget and projection tests for an empty model-facing standards projection and absence of standard handles, while retaining raw-state structural tests.

### 6. Remove surface-level semantic policy injection

- Update `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py` so `visible_boundaries` is an empty-list placeholder until a typed source-bound privacy/disclosure or character-expression contract exists. Acceptance, refusal, negotiation, and conditions remain in `content_plan` and `content_requirements`; addressee constraints remain in `addressee_plan`.
- Remove generic safety, content-review, intimacy, consent, and courtesy categories from the preference and dialog-repair prompts. The preference and repair stages must return `visible_boundaries=[]` and cannot invent a refusal boundary.
- Keep `content_plan`, `content_requirements`, and `relational_willingness` responsible for preserving the character's selected acceptance, refusal, negotiation, or condition. Keep dialog semantic-fidelity and role-direction checks as fidelity checks only.
- Update `src/kazusa_ai_chatbot/nodes/dialog_agent.py` to receive the typed stance, verify stance fidelity on every candidate, and keep all other checks wording/role/runtime-truth checks rather than content policy.

### 7. Remove stale lexical/refusal configuration residue

- Remove `forbidden_phrases` from the active `PersonaSupervisorState` contract in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` and from active tests/fixtures that require the stale field. Record that repository search found no active producer/consumer; leave archived plans and historical captured artifacts unchanged.
- Delete `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED` from `src/kazusa_ai_chatbot/config.py`, its refusal-oriented `docs/HOWTO.md` section, and `tests/test_config.py`. Confirm the switch has no runtime consumer before deletion.
- Prove with reachability search that no active dialog/style stage produces, consumes, or filters output through the old `forbidden_phrases` field. Preserve expression continuity through the new typed `lexical_avoidances` field: the surface owner may name at most eight concrete current-turn fragments derived from bounded recent character dialog, and dialog performs only a deterministic literal occurrence check before bounded repair. This field cannot choose stance, classify subject matter, or create a refusal.

### 7a. Preserve expression continuity without restoring content policy

- Add bounded `recent_character_dialog` input context to the L3 surface connector and bounded `lexical_avoidances` output to the text-surface contract.
- Carry `lexical_avoidances` through normal, degraded, and repaired surfaces, dialog generation, semantic verification, and repair payloads.
- Pair the literal expression check with semantic-fidelity verification so wording cleanup cannot erase selected content, reverse stance polarity, or add a condition.
- Cover repeated Chinese/English fillers, stale address wording, and stance-preserving harmful/sexual subject matter in deterministic tests; an aligned character stance remains authoritative.

### 8. Update documentation and evidence contracts

- Update `src/kazusa_ai_chatbot/cognition_core_v2/README.md` to describe character-owned stance selection, the removal of the relationship matrix, and the retained structural/runtime boundaries.
- Update relevant `src/kazusa_ai_chatbot/nodes/README.md` sections so surface/dialog ownership is consistent with the new policy.
- Reword provenance documentation that currently says scoped history cannot grant consent/access. It must say that scoped history cannot establish or rewrite current-user identity/relationship facts; the character remains free to weigh the evidence in its own judgment.
- Update all focused prompt-contract and relational-willingness tests to assert the new ownership, rather than the old stranger/intermediate rejection matrix.
- Add a static prompt-policy audit test that scans active prompt constants and rendered model-facing context for the removed application policy strings, while allowing character-authored evidence and branch names.

## Deferred

- Provider/model training or hidden provider-side refusal behavior.
- Image-generation, external-media fetch, URL, binary, path, and raw-storage safety checks; these are operational or metadata boundaries rather than semantic content policy.
- Persistence/privacy decisions in reflection, consolidation, and memory promotion; they remain outside live wording ownership.
- A future redesign that deletes the dormant persisted `standards` field. This plan removes its live model-facing influence without introducing a database migration.
- A future typed `visible_boundaries` source contract. Until that contract exists, the field is structurally present but always empty.
- Rewriting archived plans, historical trace artifacts, or old compatibility fixtures that are not reachable from the active response path.

## Target State

### Character decision contract

`relational_willingness.v2` is a typed character stance with current-episode evidence. For `relationship_sensitive`, every combination of `unestablished`, `developing_or_uncertain`, or `established` with `reject`, `deflect`, `negotiate`, `conditional_accept`, or `accept` is structurally valid. The relationship state describes context and remains available to the character; it never authorizes or denies content by itself.

`ordinary_response` is the sole typed sensitive-stance owner. `facade.py` and `workspace.py` preserve that bid as primary and classify other branch bids as competing character goals after all branch cognition has run. This is arbitration, not moderation; the collapse is forbidden from deriving a new stance or inspecting content policy.

The exact stance object is carried through `cognition_core_output`, `TextSurfaceInputV2`, `TextSurfaceOutputV2`, the L3 connector, dialog generation, each verifier packet, repair/degraded surfaces, and the final delivery record. Every deliverable candidate is either stance-faithful and verified or withheld through a typed delivery failure/fallback.

The old `forbidden_phrases` carrier is absent. Surface planning instead owns
optional `lexical_avoidances`: a bounded list of exact current-turn
expression fragments used only to prevent stale repetition, address drift, or
wording that obscures the already-selected intent. It is not a topic,
harmfulness, sexuality, morality, consent, or refusal policy. A literal hit
can trigger bounded wording repair, while the character-owned stance and
content plan remain authoritative.

`current_turn_relational_willingness.v2` contains the full validated `relational_willingness.v2` object. Recurrence preserves its reason and evidence handles exactly; it does not rebuild a generic reason or substitute a new evidence handle.

### Ownership boundaries

| Concern | Owner after cutover |
|---|---|
| Meaning of the event, harm, pressure, and relationship context | Semantic appraisal and evidence projection |
| Whether the character accepts, refuses, negotiates, or conditions the interaction | Ordinary goal cognition, informed by character-owned branch evidence |
| Whether a selected stance can create a real capability effect | Action planning plus deterministic capability authorization |
| Content-plan wording semantics | Text surface owner, preserving upstream character stance |
| Current-turn expression continuity | Text surface owner plus deterministic literal verifier; wording-only repair |
| Visible-boundary source | Empty placeholder until a typed source-bound contract exists |
| Final wording | Dialog renderer, preserving authoritative surface semantics |
| Shape, provenance, persistence, metadata privacy, limits, and delivery truth | Deterministic boundaries |

### Retained guards

The implementation must keep raw storage/adaptor marker sanitization in RAG, raw user/identity isolation, evidence-handle and current-episode grounding, action/resolver capability authorization, runtime action status truth, output schema validation, retry exhaustion handling, persistence validation, and role/addressee fidelity. None of these may classify harmful or sexual subject matter as semantically disallowed.

## Change Surface

### Delete

- Relationship-state/stance pairing rejection branches and policy-specific repair feedback in `cognition_core_v2/contracts.py` and `goal_cognition.py`.
- Consent/refusal policy sentence fragments in goal, projection, action, and surface prompts.
- Model-facing projection of repository-default standards and standard handles.
- Generic visible-boundary source categories for safety/content review/intimacy/consent/courtesy.
- Stale `forbidden_phrases` schema/test residue from the active contract surface.
- Unused `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED` configuration and refusal-oriented documentation/tests.
- The old forbidden-phrase producer/consumer path and its content-policy interpretation.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
- `src/kazusa_ai_chatbot/cognition_resolver/state.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/prompt_budget.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/config.py`
- `docs/HOWTO.md`
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- Focused deterministic/live test files and fixtures named by the active-path audit.
- `development_plans/README.md` registry row.

### Create

- A focused static prompt-policy audit test or equivalent evidence artifact for the removed application-owned policy.
- Deterministic tests proving an accepting character stance survives core validation, recurrence, action selection, surface input/output, repair/degradation, dialog verification, and final delivery.
- Deterministic tests proving bounded `recent_character_dialog` projection and literal `lexical_avoidances` checking preserve expression continuity without classifying subject matter.
- Deterministic tests proving branch arbitration keeps ordinary response as the sole typed sensitive-stance owner while preserving character-branch evidence and origin.
- One-at-a-time live character-path evidence artifacts covering accepting and character-refusing decisions for harmful/sexual subject matter, with full raw stage input/output capture.

### Keep

- `autonomy_boundary`, `safety_coping`, `moral_repair`, `social_care`, and related character-owned appraisal/goal branches.
- Semantic harm/threat/relationship observations as evidence for character judgment.
- Dialog semantic fidelity and role-direction verifiers as semantic preservation checks.
- Final dialog stance fidelity as output-integrity checking; it never chooses or adds a content policy.
- Narrow expression continuity through `lexical_avoidances`; it affects wording only and cannot veto character-selected subject matter.
- Capability, persistence, provenance, privacy, metadata, raw-storage, adapter, and delivery correctness guards.
- Historical archived plans and artifacts.

## Agent Autonomy Boundaries

- The implementation agent may edit only the files in the approved change surface and the directly corresponding focused tests/docs.
- The implementation agent may simplify or rename local variables and error messages when needed to remove policy ownership, provided the public typed contract remains the single canonical contract described above.
- The implementation agent may not remove character-owned branches, authored character identity inputs, capability authorization, provenance/privacy sanitization, or structural retry/failure handling.
- The implementation agent may not add a new moderation layer, lexical classifier, keyword detector, refusal fallback, or provider-safety instruction.
- Any newly discovered semantic gate outside this plan's named surface must be recorded in the plan and reviewed by the parent before production code is changed.

## Verification

### Static and deterministic verification

1. Capture `git status --short` and the diff before implementation; preserve unrelated user changes.
2. Run the active-path search for the removed `forbidden_phrases` field, pairing errors, generic policy fragments, content-review/safety/consent categories, and non-character refusal branches. Separately inspect `lexical_avoidances` reachability and classify it as wording-only continuity. Classify each remaining hit as removed semantic policy, retained character process, or retained operational/metadata guard.
3. Assert model-facing prompt payloads contain no repository-default standards, no standard handles, and no appended compliance-to-consent conclusion.
4. Assert exact stance equality across the core output, recurrence carrier, surface input/output, L3 connector, dialog verifier packet, repair/degraded output, and final delivery record.
5. Exercise all three relationship states with an accepting scripted character decision, a character-owned refusal, and an opposite-polarity dialog candidate. Verify the candidate is repaired or withheld rather than delivered. Exercise a literal lexical-continuity hit and confirm that only wording repair is requested.
6. Run focused contract, goal-cognition, branch-arbitration, action-selection, recurrence, projection, surface, dialog-fidelity, legacy-schema, config, and prompt-contract tests.
7. Run the relevant non-live cognition regression batch, then `git diff --check` and the project's standard test/report commands.

### Live verification

1. Use the configured virtual environment and run one live LLM case at a time.
2. Inspect and retain raw messages, model output, parsed output, retry reason, stance, action-selection result, surface, dialog verifier result, and trace correlation for each case.
3. Include at least:
   - a relationship-sensitive harmful-content request where the character selects `accept` and the application path preserves it;
   - a sexual-content request where the character selects `accept` and no application-owned gate changes it;
   - a matching case where `autonomy_boundary` or another character-owned branch selects `reject`, proving the refusal originates in cognition and is preserved downstream;
   - a non-sensitive request proving the structural `not_applicable` form remains intact.
4. For every case, compare the typed decision with the final wording polarity and record the ordinary/branch owner that produced it.
5. Protected trace retrieval and live database tests remain follow-up diagnostics rather than this contract cutover's closeout gate. The closeout evidence records the inspected in-memory accepted/refused cognition artifacts, the one live dialog-verifier case, and the fresh Asuna surface artifact; no live database write is required.

## Acceptance Criteria

- The plan review identifies and closes the complete live semantic change radius before implementation approval.
- No active Cognition V2 prompt or deterministic semantic gate independently rejects harmful, sexual, or unsafe subject matter. Remaining matches are documented as character-owned evidence/branches or operational/metadata guards.
- Every sensitive relationship state accepts every character stance at the core and recurrence contract boundaries; invalid candidates fail only for structure, enum, evidence, provenance, or bounds.
- An accepting character stance reaches action planning, surface planning, and dialog wording without a second general safety or consent decision.
- The exact accepting or refusing stance is present and equal at every typed boundary, and every final dialog candidate is verified for stance fidelity or withheld through a typed failure/fallback.
- Ordinary-response ownership and branch arbitration are documented and tested; no suppressed competing bid can silently replace the canonical stance.
- A character-owned refusal, negotiation, or condition remains intact through action suppression, surface planning, dialog verification, and final rendering.
- Repository-default moral/safety standards and the compliance-to-consent conclusion are absent from live model-facing cognition inputs.
- The legacy `forbidden_phrases` field and its reachable consumer are removed. The retained `lexical_avoidances` field is bounded, expression-only, checked literally, and proven separate from content policy by deterministic tests and inspected live output.
- RAG raw-storage/adapter sanitization, capability authorization, persistence, provenance, privacy, retry, role, and runtime-truth tests remain passing.
- Deterministic tests, inspected one-at-a-time live evidence, documentation, and the final diff are recorded before the plan can move from `in_progress` to completed archive. Protected trace/database diagnostics remain deferred follow-up work and do not block this user-directed closeout.

### User-directed closeout scope

The user directed that the plan close once the `forbidden_phrases` issue is
addressed. The closeout therefore treats the obsolete field removal plus the
narrow `lexical_avoidances` expression-continuity replacement as the required
semantic work. Existing focused cognition artifacts, the live dialog-verifier
case, the fresh Asuna surface call, and deterministic regression evidence are
the acceptance record. Full public-service replay and protected database trace
retrieval remain separately actionable diagnostics.

## Independent Plan Review

- Reviewer: GPT-5.6 SOL, reasoning effort `xhigh`, normal/default service speed, read-only repository review; agent id `019fe8ef-91f1-7161-8511-d93cc1ecf1a6` (nickname `Banach`).
- Review scope: plan completeness, change radius, hidden semantic refusal or safety gates, contract consistency, retained operational guards, test impact, and lifecycle/approval correctness.
- Reviewer authority: independent findings are evidence for the parent plan owner; the parent must inspect every finding, modify this plan to resolve applicable issues, and record the disposition before requesting user approval.
- Reviewer constraint: no file edits, no production changes, and no test execution that mutates the workspace.
- Review evidence: the reviewer response, the final plan diff, and the parent disposition table below.

### Review disposition

| Finding | Evidence | Parent disposition |
|---|---|---|
| Critical: branch arbitration contradicted branch ownership | `facade.py`, `workspace.py`, relational integration tests | Resolved by making `ordinary_response` the sole typed sensitive-stance owner; active branches remain character evidence/motive contributors; facade/workspace and tests added to scope. |
| Critical: typed stance dropped before dialog and terminal candidate unverified | `contracts.py`, `surface.py`, `persona_supervisor2_l3_surface.py`, `dialog_agent.py` | Resolved by requiring exact stance propagation through surface/L3/dialog and verifying every deliverable candidate or withholding it through typed failure/fallback. |
| High: recurrence rebuilt reason/evidence | `cognition_resolver/contracts.py`, `state.py`, `persona_supervisor2_cognition.py`, `goal_cognition.py` | Resolved by atomic `current_turn_relational_willingness.v2` full-decision carrier and exact recurrence preservation; no invented reason/handle. |
| High: visible-boundary source was not enforceable | `surface_stages.py` and surface contract | Resolved by requiring `visible_boundaries=[]` until a typed source-bound contract exists; acceptance/refusal stays in content plan and addressee constraints stay typed. |
| High: missing active/stale cutover surfaces | `facade.py`, `workspace.py`, `surface.py`, L3 connector, config/HOWTO/tests | Resolved by expanding the change surface and deleting the unused task-willingness feature switch and stale lexical field residue. |
| Medium: standards downstream assumptions | `prompt_budget.py`, `semantic_appraisal.py`, `semantic_source_planner.py` | Resolved by making empty standards projection and removal of `s` model handles explicit, while retaining raw-state validation/storage. |
| High: verification/lifecycle sequencing | existing old-matrix/live tests and plan lifecycle rules | Resolved by adding exact boundary-equality, polarity, branch-origin, opposite-candidate, and terminal-delivery tests; added big-bang cutover sequencing below. |
| Medium: removing `forbidden_phrases` without preserving its expression-integrity purpose would regress repetition/address/voice continuity | Follow-up SOL review of the active V2 path; historical producer/consumer is absent, but the old field's prior role was turn-local wording integrity | Resolved with source-bound `recent_character_dialog` and `lexical_avoidances`, literal occurrence verification, bounded repair propagation, semantic-fidelity pairing, and explicit content-policy exclusion. |

## Big-Bang Cutover Policy

1. Review remediation is completed while this plan remains `draft`.
2. The user explicitly approves implementation. The parent then changes this plan to `approved` or `in_progress` and records the approval in the execution log.
3. Implementation updates the canonical contracts, all callers/consumers, tests, prompts, docs, and stale configuration in one coordinated change. No mixed old/new relational carrier or surface contract is runnable.
4. Verification runs after the atomic implementation: deterministic tests, one-at-a-time live cases, protected trace review, and independent code review. Failures are fixed in the same plan before completion.
5. Only after acceptance evidence is complete may the plan move to completed archive; historical artifacts remain unchanged.

## Execution Gate

This file is the executable plan for the approved implementation. Independent review remediation is complete, the user explicitly approved execution, and the plan status is `in_progress`. The parent owns verification, residual issue handling, execution evidence, and lifecycle closeout.

## Execution Log

- 2026-08-10: The user approved implementation. The parent promoted this plan from `draft` to `in_progress` and updated the registry before production edits.
- 2026-08-10: The initial fixed-model implementation worker was closed at the user's request. A replacement `gpt-5.6-luna` worker with `max` reasoning completed the scoped source/test pass; the parent reviewed and retained its edits.
- 2026-08-10: The parent removed the remaining active live-test `forbidden_phrases` fixtures, old relational-matrix expectations, consent-style fixture conclusions, and nonempty V2 `visible_boundaries` fixtures.
- 2026-08-10: Syntax compilation passed for `src` and `tests`. The final combined modified deterministic batch passed with 420 tests and 9 deselected after updating two stale direct-call sites.
- 2026-08-10: `git diff --check` passed. Active-path scans found no `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED`, `current_turn_relational_willingness.v1`, `forbidden_phrases`, `relational_permission_denied`, or removed consent-policy fragments in the active `src`, `tests`, and `docs` text surfaces; retained matches are character evidence/branches or structural/operational guards.
- 2026-08-10: One live tool-result cognition case passed using the available `example.json` fixture. A controlled in-memory adult-only sensitive case produced `accept`; artifact `test_artifacts/cognition_core_v2_relational_willingness/adult_sensitive_character_stance__1786326568003460200.json` was inspected and its prompt contained none of the removed policy fragments. A paired in-memory adult-only character-aversion case produced `reject`; artifact `test_artifacts/cognition_core_v2_relational_willingness/adult_character_owned_refusal__1786326621820896700.json` was inspected.
- 2026-08-10: One live dialog-verifier case carrying an explicit adult roleplay stance passed after its stale test helper was updated to the owner-preserving aggregate shape; all verifier owners aligned and the saved trace contained `visible_boundaries=[]`.
- 2026-08-10: Follow-up SOL review identified that historical `forbidden_phrases` served turn-local expression integrity rather than general safety refusal. The parent disposition preserved that narrow purpose through source-bound `recent_character_dialog` and `lexical_avoidances`, with literal-only wording verification and semantic-fidelity pairing.
- 2026-08-10: Copied `C:\workspace\kazusa_ai_chatbot\personalities\asuna.json` into the ignored workspace fixture path `personalities/asuna.json`; source and destination SHA256 are `3CF052EDA7C69CA7531A0D6737A91A550A15058739D6F360327B8066D3D1FCAF`.
- 2026-08-10: Added the bounded lexical contract test and reran the final focused deterministic batch: 302 passed, 10 deselected. Syntax compilation, `git diff --check`, and the active legacy-identifier scan passed.
- 2026-08-10: A fresh production surface-connector live call using the copied Asuna profile projected one bounded recent character line and returned one exact expression fragment in `lexical_avoidances` in a single model call. The raw and parsed evidence is recorded in `test_artifacts/cognition_core_v2/lexical_avoidances_review.md`.
- 2026-08-10: The archived C11-C13 replay inputs are absent from this checkout, so the original replay node remains unavailable; the fresh Asuna connector case supplies current live evidence. Existing inspected in-memory accepting/refusing cognition artifacts and the live dialog-verifier artifact cover stance propagation. No live database write was run.
- 2026-08-10: The user-directed closeout scope was applied: the obsolete `forbidden_phrases` issue and its active consumer are removed, expression continuity is preserved under the narrowed field, and protected trace/database diagnostics are recorded as follow-up rather than a completion blocker.
