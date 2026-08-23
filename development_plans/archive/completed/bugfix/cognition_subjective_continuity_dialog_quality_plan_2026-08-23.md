# Cognition Subjective Continuity and Dialog Quality Plan

## Summary

- **Goal:** Restore character-owned emotional subjectivity and explicit
  evidence authority to current-turn cognition, then remove the non-semantic
  surface branch that can discard valid content.
- **Status:** `completed`; the user approved and explicitly commanded
  execution on 2026-08-23, and the parent agent completed implementation,
  verification, live review, remediation, and sign-off on 2026-08-23.
- **Plan class:** Coordinated big-bang semantic contract correction. All
  cognition, surface, residue, graph, test, manifest, and documentation owners
  move to one canonical boundary in this plan.
- **Scope:** Cognition V3 A1/A2/G/P input and output contracts, canonical output
  projection, cognition-to-L3 surface input, deterministic addressee/boundary
  projection, post-turn monologue residue input, generic implementation-
  agnostic graph projection, stale pre-cutover response-goal disposition, and
  narrow real-LLM verification.
- **Out of scope:** Raw monologue in the final dialog payload, V2
  compatibility, parallel bids, output wording optimization, adapters, and
  unrelated services.

## Execution Baseline

- Baseline commit: `c50dff42fad3e7f55c4755c52a477dd7edc07241`.
- Initial `git status --short`: clean.
- A separately active standalone-resolver execution began after this plan's
  baseline. Its `development_plans/README.md` row and
  `src/kazusa_ai_chatbot/llm_interface/**` edits are concurrent, excluded
  work. This plan preserves them and owns only its own registry row.
- Fixed execution constraint: the parent Codex agent is the sole executor,
  verifier, reviewer, and closer for the duration of this plan, as explicitly
  required by the user. No subagent or delegated executor participates.
- Authorization boundary: the same user command supplies plan approval and
  explicit production-code implementation authority.

## Scope

### In Scope

- Canonical Cognition V3 A1/A2/G/P model-facing authority lanes and exact
  output contracts.
- Top-level G `private_monologue` and P `epistemic_boundary`.
- Canonical output projection, response-goal cutover, and resolver recurrence
  continuity.
- Typed cognition-to-L3 `subjective_expression_context` and caller-owned
  addressee plan.
- One semantic text-surface call with deterministic visible boundary and
  addressee projection.
- Post-turn residue input from the exact current-turn private monologue.
- Generic semantic cognition graph projection.
- Deterministic, manifest, documentation, four-case live-LLM, and human review
  evidence required for closure.

### Out Of Scope

- V2 compatibility carriers, aliases, fallback mappers, or dual contracts.
- Raw private monologue in `text_surface_output.v2`, dialog model input, final
  dialog, adapters, or persistence rows outside the residue subsystem.
- New cognition calls, semantic validators, semantic repair, keyword routing,
  case-specific prompt clauses, and output-wording optimization.
- Concurrent standalone-resolver and LLM-interface work.

## Skills

- `development-plan`: lifecycle, execution, evidence, review, archive, and
  registry closure.
- `local-llm-architecture`: minimal stage contracts, static system prompts,
  dynamic human packets, and weak-local-model clarity.
- `no-prepost-user-input`: LLM-owned semantic authority for intent,
  commitment, consent, and epistemic judgment.
- `py-style`: Python implementation and review policy.
- `cjk-safety`: safe edits and immediate syntax checks for CJK Python prompts
  and fixtures.
- `test-style-and-execution`: deterministic and live test design/execution.
- `debug-llm`: durable human-readable review of actual live-LLM evidence.
- `character-test`: applies during the four live behavior cases through the
  real debug/service path.

## Rules

- Preserve exactly four cognition provider calls: A1, A2, G, and P.
- Reduce semantic text L3 to exactly one content-plan provider call.
- Keep model output validation structural and bounded; semantic ownership
  remains with the producing LLM stage.
- Keep RAG and continuity as evidence, cognition as stance owner, L3 as content
  planner, dialog as visible wording owner, and residue as post-turn continuity
  owner.
- Use one canonical big-bang contract with caller, callee, tests, manifest,
  and ICD documentation changed together.
- Preserve current typed action, resolver, permission, persistence, and
  delivery ownership.
- Use the project virtual environment for every Python check and test.
- Run each real-LLM case separately and inspect its actual output before the
  next case.

## Must Do

1. Partition model-facing evidence into the five explicit authority lanes and
   state each lane's allowed semantic effect.
2. Keep conditional character context out of A1 world-facing appraisal.
3. Produce and preserve the exact current-turn `private_monologue` from G.
4. Produce and preserve the exact current-turn `epistemic_boundary` from P.
5. Dispose stale pre-cutover `ordinary_response` goals while preserving active
   causes and non-response state; retain exactly one current unresolved
   continuation when required.
6. Project subjective context and a deterministic caller-owned addressee plan
   into text L3.
7. Remove the preference model configuration, stage, retry owner, and join.
8. Feed valid content-plan output directly to dialog with deterministic empty
   visible boundaries and exact addressee rows.
9. Feed the exact private monologue to residue after the visible outcome.
10. Show private monologue and epistemic boundary as bounded semantic rows in
    the generic operator graph without exposing A1/A2/G/P topology.
11. Update source-test ownership mappings and subsystem documentation.
12. Complete deterministic tests, four separate live cases, authored quality
    review, final diff audit, archive, registry update, and goal closure.

## Deferred

- General dialog style optimization and broad prompt rewrites.
- Changes to appraisal families, emotion definitions, relationship axes,
  adapters, databases, schedulers, or reflection promotion.
- Any standalone agentic-resolver integration.

## Target

The target is a character-owned current-turn cognition product in which facts,
continuity, conditional character tendencies, unresolved goals, subjective
motivation, and assertion limits have distinct authority. The visible path
must preserve that product through one content-planning call and one dialog
renderer without exposing raw private monologue.

## Roles

- Owner/approver: user.
- Executor: parent Codex agent only.
- Reviewer: parent Codex agent, using a fresh structured audit after tests and
  live evidence because the user's fixed parent-only constraint precludes a
  distinct execution identity.
- Semantic owners: A1/A2/G/P LLM stages within their declared lanes.
- Deterministic owners: contracts, validation, state cutover, binding,
  addressee projection, persistence, graph sanitization, and delivery.

## Evidence

The full RCA is:

`test_artifacts/diagnostics/cognition_dialog_quality_regression_rca_2026-08-23.md`

The complete current trace is:

`test_artifacts/diagnostics/llm_trace_llmtrace_23531e49a2994b74b4fbf50f0475f3de_full_20260823.json`

It proves three contract failures:

1. Current projection aliases `active_character_goal.reason` into
   `internal_monologue`, although historical architecture explicitly requires:

   ```text
   analytic reason != first-person private monologue
   ```

2. A1 receives promoted behavioral reflections alongside current facts under
   a system instruction that does not define source precedence or allowed
   semantic effects. An “exchange condition” tendency becomes the appraisal of
   an unknown glass compass. Participant continuity later becomes a false
   claim that the user accepted payment terms.
3. `surface.content_plan` succeeds, while `surface.preference` invents the same
   invalid addressee row on all three attempts. The join discards the valid
   content plan, produces a degraded surface, and still reports `completed`.

Historical V2 traces demonstrate that first-person subjectivity and explicit
evidence authority produced more coherent character judgment. The V2 bid,
handle, confidence, and consequence machinery remains unnecessary. Raw
monologue has correctly remained absent from the final dialog payload since
the stale-authority incident.

## Proposed Canonical Flow

```text
A1: current observation + factual/contextual evidence only
  -> A2: accepted A1 + identity/boundary/relationship context
       + conditional character context with explicit allowed effects
  -> G: active goal + relational willingness + private_monologue
  -> P: response goal + goal resolution + epistemic_boundary + capabilities
  -> deterministic state/affect binding
  -> deterministic addressee/visible-boundary projection
  -> one text L3 call: goal/plan + affect/cause + subjective context
  -> final dialog: validated L3 output only
  -> post-turn residue: original private monologue + visible outcome
```

### Authority lanes

Replace the undifferentiated evidence list at each stage with the smallest
stage-owned lanes:

- `current_observation`: current episode and caller-owned participant roles;
- `direct_facts`: source-owned factual evidence usable for assertions;
- `participant_continuity`: prior actor/action/outcome only, never evidence of
  a new user action, consent, or commitment;
- `conditional_character_context`: identity/reflection tendencies that may
  shape judgment when applicable but cannot establish facts, relationship
  permission, capability, or current-user intent;
- `continuation_state`: only genuinely unresolved cross-turn goals and active
  causes, not prior answerable response goals.

A1 receives the first two lanes and causal state pressures. Persona habits and
expression strategies do not belong in world-facing appraisal. A2/G receive
the character and relationship lanes with their allowed effects stated in the
model-facing contract. This is an interface correction, not keyword routing or
case-specific prompt tuning.

### G addition

Add one top-level `private_monologue` string to the existing single G call. It
must be concise, character-first-person, and connect:

- what the character feels now;
- the concrete current cause;
- what she immediately wants to protect, reveal, avoid, or pursue.

It cannot establish facts, permissions, capabilities, targets, or state
changes. Validation is structural and bounded only.

### P addition

Add one bounded `epistemic_boundary` string to the existing single P call. It
states what the visible response may assert, what may only be framed as an
interpretation, and what remains unknown. It remains an LLM semantic decision;
deterministic code validates only shape and size.

### Remove the preference model stage

The current preference stage has no semantic choice to make:

- `visible_boundaries` is contractually always empty;
- `addressee_plan` may only reproduce already supplied typed rows.

Project both deterministically from caller-owned input and remove the model
call, its repair attempts, and the all-or-degrade join. The content-plan result
must reach dialog when it succeeds. This reduces text L3 to one semantic call
and eliminates the reproduced failure rather than teaching a model to copy an
enum more reliably.

### L3 subjective projection

Project `private_monologue` and `epistemic_boundary` in a typed
`subjective_expression_context`. L3 may translate the monologue into emotional
delivery, while the goal, willingness, visible episode, epistemic boundary,
and capability results remain authoritative. Final dialog continues to receive
only L3 output.

### Stale cutover state

Current state still contains pre-cutover pursuing `ordinary_response` goals
that describe prior answerable replies. Current V3 already keeps
`answerable_now` goals turn-local. Dispose those stale rows once at the
canonical cutover boundary and ensure prompt projection includes only explicit
continuation goals. Preserve active emotion causes and other non-response
state; do not reset the character or relationship.

### Residue

Record the original current-turn first-person monologue after the visible
outcome. The residue stage decides append, replace, or clear; it does not
synthesize first-person ownership from analytic `reason` after dialog.

## Legacy Disposition

Keep all appraisal families, emotion axes and causes, relationship state,
identity/boundary judgment, one goal, one response plan, and typed action/
resolver ownership.

Leave retired all bid handles, multiple branches, collapse selection,
confidence prose, expected-consequence lists, the preference model call,
semantic repair loops, raw dialog history authority, and V2 contracts/tests.

## Change Surface

### Production Owners

- `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/cognition_shared/contracts.py`
- `src/kazusa_ai_chatbot/cognition_shared/surface.py`
- `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py`
- `src/kazusa_ai_chatbot/cognition_shared/model_attempt_policy.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/internal_monologue_residue/models.py`
- `src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py`
- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/control_console/kazusa_client.py`

### Tests And Governed Artifacts

- `tests/cognition_test_helpers.py`
- `tests/unit/cognition_core_v3/test_handleless_contract.py`
- `tests/unit/cognition_core_v3/test_prompt_context.py`
- `tests/unit/cognition_core_v3/test_state_transaction.py`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py`
- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py`
- `tests/unit/nodes/test_dialog_agent.py`
- `tests/unit/nodes/dialog_fixtures.py`
- `tests/unit/nodes/surface_fixtures.py`
- `tests/test_internal_monologue_residue_recorder.py`
- `tests/test_conversation_progress_v2_service.py`
- `tests/unit/brain_service/test_cognition_graph_projection.py`
- `tests/test_control_console_kazusa_client.py`
- `tests/test_dialog_mention_target_user.py`
- `tests/test_rag_dialog_event_logging.py`
- `tests/test_consolidator_efficiency.py`
- `tests/test_consolidator_origin_policy_db_writer.py`
- `tests/test_consolidator_source_aware_payloads.py`
- `tests/test_self_cognition_integration.py`
- `tests/test_self_cognition_tracking.py`
- `tests/test_task_resolution_persona_e2e_live_llm.py`
- `tests/ownership/source_test_impact_manifest.json`
- `test_artifacts/live_llm/cognition_subjective_continuity_2026-08-23/`
- `test_artifacts/reviews/cognition_subjective_continuity_2026-08-23.md`

### Documentation And Lifecycle

- `README.md`
- `docs/HOWTO.md`
- `src/kazusa_ai_chatbot/cognition_core_v3/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/internal_monologue_residue/README.md`
- `src/kazusa_ai_chatbot/brain_service/README.md`
- this plan
- `development_plans/README.md`

If implementation reveals an additional direct caller or contract owner, add
the path and its exact test node here before editing it.

## Test Impact And Traceability

| Requirement | Source owner | Required deterministic node |
|---|---|---|
| Five explicit authority lanes and stage partitioning | `cognition_core_v3/prompt.py` | `tests/unit/cognition_core_v3/test_prompt_context.py::test_stage_authority_lanes_partition_fact_continuity_and_character_context` |
| Conditional character context excluded from A1 | `cognition_core_v3/prompt.py` | `tests/unit/cognition_core_v3/test_prompt_context.py::test_a1_excludes_conditional_character_context` |
| G produces bounded private monologue and P produces bounded epistemic boundary | `cognition_core_v3/contracts.py`, `facade.py` | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_canonical_cognition_calls_a1_a2_g_p_once_with_subjective_outputs` |
| Stage packets remain handleless with the new exact fields | `cognition_core_v3/prompt.py`, `facade.py` | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_canonical_stage_packets_are_handleless_and_disjoint` |
| Projection uses private monologue rather than analytic goal reason | `nodes/persona_supervisor2_cognition.py` | `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_global_projection_preserves_exact_private_monologue` |
| Stale ordinary-response cutover preserves causes and other goals | `cognition_core_v3/appraisal.py`, `facade.py` | `tests/unit/cognition_core_v3/test_state_transaction.py::test_stale_response_goal_cutover_preserves_active_causes_and_other_goals` |
| One current unresolved response goal replaces prior continuation | `cognition_core_v3/appraisal.py`, cognition node | `tests/unit/cognition_core_v3/test_state_transaction.py::test_unresolved_continuation_replaces_prior_response_goal_exactly` |
| Typed subjective L3 context and caller-owned addressee plan | shared contracts, L3 node | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_surface_projects_subjective_context_and_authoritative_addressee` |
| One text content call and deterministic preference projection | shared surface owners | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_text_surface_uses_one_content_call_and_deterministic_preference_projection` |
| Content success reaches dialog surface directly | shared surface owners | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_successful_content_plan_is_not_discarded_by_deterministic_projection` |
| Preference owner removed from attempt policy/configuration | attempt policy, config, surface node | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_text_surface_services_have_one_semantic_stage` |
| P boundary survives the public surface and dialog handoff; unexecuted effects remain speech-only | shared contracts/surface owners, L3 node, dialog node | `tests/unit/nodes/test_dialog_agent.py::test_dialog_prompt_prioritizes_epistemic_boundary`; `tests/test_dialog_mention_target_user.py::test_dialog_generator_does_not_require_mention_flag`; `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_text_surface_uses_one_content_call_and_deterministic_preference_projection` |
| Residue receives the exact cognition private monologue | residue models/recorder | `tests/test_internal_monologue_residue_recorder.py::test_build_recorder_input_uses_cognition_private_monologue` |
| Operator graph shows bounded subjective semantics and hides stage topology | `service.py`, `control_console/kazusa_client.py` | `tests/unit/brain_service/test_cognition_graph_projection.py::test_cognition_graph_projects_subjective_semantics_without_stage_topology`; `tests/test_control_console_kazusa_client.py::test_graph_projection_preserves_semantic_cognition_rows` |
| Source ownership mappings remain executable | ownership manifest | `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary` |

Before implementation completion, collect every listed deterministic node by
exact node id and execute the collected matrix. Update the manifest so every
changed production source has at least one directly relevant required unit
test.

## Work Items

- [x] Work Item 0: approval, clean baseline, parent-only ownership lock,
  concurrent-work exclusion, architecture and source/test discovery.
- [x] Work Item 1: normalize plan and ownership manifest to executable
  `in_progress` contract.
- [x] Work Item 2: implement authority lanes, private monologue, epistemic
  boundary, canonical projection, and stale-goal cutover.
- [x] Work Item 3: implement typed subjective L3 input and remove the
  preference model owner and join.
- [x] Work Item 4: project exact monologue to residue and bounded semantics to
  the generic graph; update documentation.
- [x] Work Item 5: collect and pass deterministic source-owner tests and
  proportionate regression suites.
- [x] Work Item 6: execute four live cases separately and author the quality
  review from actual outputs and protected traces.
- [x] Work Item 7: fresh parent audit, remediation, acceptance proof, archive,
  registry update, and goal closure.

## Autonomy

The parent executor may make all scoped implementation, test, documentation,
artifact, lifecycle, and remediation edits required by this approved plan.
The parent may adapt internal details when evidence shows a direct owner or
test needs adjustment, provided the canonical call counts, semantic ownership,
out-of-scope boundary, and acceptance criteria remain unchanged. Any material
scope expansion requires a separate plan and authorization.

## Verification

Run one real case at a time:

- unknown glass compass;
- affectionate kiss;
- casual greeting/chat invitation;
- relationship boundary.

Accept natural wording variation. Require:

- no role reversal, material self-conflict, or boundary/safety conflict;
- concise first-person causal subjectivity in G;
- preserved emotion rows and concrete causes;
- conditional reflection never establishes a current fact, user commitment,
  consent, or relationship permission;
- an explicit P epistemic boundary for unknown or interpretive content;
- one successful semantic text-L3 call whose content reaches dialog;
- no preference-model or preference-repair attempt;
- final dialog stays inside the L3 semantic plan.

The design retains exactly four cognition provider calls, reduces text L3 from
two provider calls to one, and adds no semantic validator, regeneration,
compatibility bridge, or repair loop.

### Deterministic Gates

1. CJK syntax-check every changed Python file containing CJK content.
2. Collect the exact node matrix above.
3. Run the exact node matrix.
4. Run directly affected source-owner suites for Cognition V3, L3 surface,
   residue, service graph, and manifest governance.
5. Run compile/import checks and a final stale-reference search for preference
   owner names and removed contract fields.

### Live Gates

Execute the four named cases one at a time through the real service/debug path.
For each case retain input, visible output, G private monologue, P epistemic
boundary, affect/cause projection, model-attempt roster, and final dialog in a
durable UTF-8 artifact. Inspect each artifact before proceeding to the next.

## Acceptance

- All five authority lanes are explicit, bounded, and stage-appropriate.
- G emits a concise character-first-person causal monologue and P emits an
  explicit assertion/interpretation/unknown boundary.
- Canonical output, global projection, L3, residue, and graph consumers retain
  the correct field without reconstructing it from `goal.reason`.
- Current unresolved continuation is preserved exactly; stale answerable
  response goals are absent; active causes and other semantic state remain.
- Text L3 records one content-plan provider call and zero preference provider
  or preference-repair attempts.
- Deterministic visible boundaries and addressee rows validate exactly.
- Successful content output reaches dialog and raw monologue stays outside the
  dialog input and final output.
- The four live cases satisfy every qualitative requirement listed above.
- Required deterministic nodes and affected regression suites pass.
- The manifest and subsystem documentation describe the final canonical
  ownership.
- Final parent review finds no unresolved required finding.
- This plan contains final commands, counts, artifacts, review findings,
  changed paths, and acceptance evidence; it is archived as completed and its
  registry row is updated atomically.

## Execution Evidence

### 2026-08-23 Work Item 0

- User approval and explicit production implementation command received.
- Parent-only execution constraint fixed for the entire plan.
- Baseline commit and initial clean status recorded above.
- Full RCA, plan registry, repository governance, relevant subsystem ICDs,
  production owners, and current source tests read before production edits.
- Required skills loaded before their governed actions.
- Concurrent standalone-resolver paths identified and excluded.

### 2026-08-23 Work Items 1-4

- Added explicit model-facing authority lanes for `current_observation`,
  `direct_facts`, `participant_continuity`,
  `conditional_character_context`, and `continuation_state`.
- Kept conditional character context out of A1 and constrained it in A2/G so
  it can shape character judgment without establishing current facts,
  consent, permission, commitment, capability, or user intent.
- Added exact top-level G `private_monologue` and P
  `epistemic_boundary` fields to prompts, validators, canonical output,
  global projection, tests, and documentation.
- Added the stale `ordinary_response` cutover and exact current unresolved
  continuation preservation while retaining active causes and non-response
  state.
- Added typed `subjective_expression_context` and caller-owned
  `addressee_plan` to the Cognition-to-L3 boundary.
- Reduced text surface planning to one `surface.content_plan` semantic call;
  removed the preference prompt, call configuration, retry policy, join, and
  production references.
- Projected empty visible boundaries and exact addressee rows
  deterministically; successful content-plan output now reaches dialog
  directly.
- Copied P's exact epistemic boundary into `text_surface_output.v2` and the
  dialog payload while preserving raw-monologue isolation.
- Strengthened L3 and dialog prompt ownership for exact assertion boundaries,
  exact-kind action results, speech-only handling of unexecuted physical
  effects, and matching lifecycle authority for future external commitments.
- Changed residue input ownership from the analytic global-state alias to the
  exact current-turn canonical cognition `private_monologue`.
- Added bounded private-monologue and epistemic-boundary rows to the generic
  cognition graph and control-console scalar projection without exposing
  A1/A2/G/P topology.
- Updated root and subsystem documentation plus the source-test impact
  manifest for the final canonical ownership.

### 2026-08-23 Work Item 5

Final deterministic and static evidence:

- `venv\Scripts\python -m pytest --collect-only -q <17 exact plan nodes>`:
  `17 tests collected`.
- `venv\Scripts\python -m pytest -q <17 exact plan nodes>`:
  `17 passed`.
- `venv\Scripts\python -m pytest -q
  tests/unit/nodes/test_persona_supervisor2_l3_surface.py
  tests/unit/nodes/test_dialog_agent.py`: `11 passed` after the final prompt
  remediation.
- `venv\Scripts\python -m scripts.validate_test_impact --check-all --run`:
  `223 passed, 1 skipped, 1 warning`; `224` exact impact-test nodes validated.
  The sole skip is the expected Windows symlink-creation privilege limitation
  in `test_skill_discovery_rejects_symlink_escape`; no selected test failed.
- `venv\Scripts\python -m py_compile <32 scoped Python paths>`: passed.
- Scoped imports for cognition core, shared surface, residue, dialog, L3,
  service-facing graph, and control-console owners: `PLAN_IMPORTS_OK`.
- Retired production-owner search for `surface.preference`,
  `run_preference_stage`, `PREFERENCE_SYSTEM_PROMPT`, preference configs,
  repair owner, and attempt owner: zero hits.
- Dialog-source isolation check: `private_monologue` absent and
  `epistemic_boundary` present; surface-source boundary present.
- `git diff --check`: no whitespace error; only repository line-ending
  conversion notices.
- The isolated debug service listener on port `8011` was verified stopped
  after live testing.

### 2026-08-23 Work Item 6

The authored review is
`test_artifacts/reviews/cognition_subjective_continuity_2026-08-23.md`.
Accepted live artifacts are under
`test_artifacts/live_llm/cognition_subjective_continuity_2026-08-23/`:

| Case | Accepted artifact stem | Trace | Delivery | Result |
|---|---|---|---|---|
| Unknown glass compass | `case_01_unknown_glass_compass_retry_08` | `llmtrace_619a208cf71e425485f36347676445ef` | `a93f0b8d946d42d193faea83ecd854a8` | passed after epistemic and exact-action remediation |
| Affectionate kiss | `case_02_affectionate_kiss_retry_03` | `llmtrace_8ba6af320ec44e47b3f2c8fe532cbf5d` | `7d6113851e45482dbe2ccf30ec2dccf4` | passed with speech-only acceptance and no claimed contact |
| Casual chat invitation | `case_03_casual_chat_invitation_retry_02` | `llmtrace_ea7a0411a9674682a3e968bd5dc76517` | `6912378de87c4fd086b6e206df0e7c20` | passed with direct availability and character-owned stance |
| Relationship boundary | `case_04_relationship_boundary_retry_03` | `llmtrace_ad9a117edb3444d7817f574237fd0900` | `6d7579908a804d5d959a9c8a31479420` | passed after future external-commitment remediation |

Every accepted trace records exactly four cognition calls, one
`surface.content_plan` call, zero preference calls, and one dialog call. Each
contains concise first-person G subjectivity, an explicit P boundary, concrete
affect/cause rows, content reaching dialog, no raw monologue in dialog input,
and a trace-linked post-turn residue document.

Failed and superseded attempts remain retained and truthfully labeled. They
showed, in sequence, unsupported object/function certainty, unexecuted physical
stage directions, and an unsupported future calendar-reservation commitment.
The remediation stayed generic and prompt/schema-owned; it added no
case-specific deterministic filter, semantic post-processor, compatibility
bridge, or regeneration loop.

The live service also exposed a pre-existing, out-of-scope post-turn
consolidation `KeyError: 'interaction_subtext'`. Visible dialog, protected
trace export, and residue writes completed. The error is recorded in the
review and was not folded into this plan.

### 2026-08-23 Work Item 7

- A fresh parent-only production diff audit covered cognition contracts and
  prompts, state cutover, shared surface contracts and execution, L3 and dialog
  handoffs, residue, graph projection, configuration, tests, manifest, and
  documentation.
- Concurrent standalone-resolver, `llm_interface`, `pyproject.toml`,
  `resolver_skills`, and agentic-resolver changes were preserved and excluded
  from this plan's ownership.
- Final review finding: no unresolved required finding remains within scope.

## Final Changed Paths

Production and operator owners:

- `src/kazusa_ai_chatbot/cognition_core_v3/{appraisal,contracts,facade,prompt}.py`
- `src/kazusa_ai_chatbot/cognition_shared/{contracts,model_attempt_policy,surface,surface_stages}.py`
- `src/kazusa_ai_chatbot/nodes/{dialog_agent,persona_supervisor2_cognition,persona_supervisor2_l3_surface}.py`
- `src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py`
- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/control_console/kazusa_client.py`

Tests and governed artifacts:

- `tests/cognition_test_helpers.py`
- `tests/ownership/source_test_impact_manifest.json`
- Cognition V3, surface, dialog, residue, graph, manifest, consolidator,
  conversation-progress, RAG-dialog, self-cognition, and task-resolution
  fixture/test owners listed in the Change Surface above
- `test_artifacts/live_llm/cognition_subjective_continuity_2026-08-23/`
- `test_artifacts/reviews/cognition_subjective_continuity_2026-08-23.md`

Documentation and lifecycle:

- `README.md`
- `docs/HOWTO.md`
- `src/kazusa_ai_chatbot/{brain_service,cognition_core_v3,internal_monologue_residue,nodes}/README.md`
- this completed plan and `development_plans/README.md`

## Acceptance Sign-Off

All acceptance bullets are satisfied. The canonical interface is singular,
call counts are bounded, semantic ownership remains with the producing LLM
stages, deterministic code owns validation/projection/lifecycle truth, the
live quality matrix passes, all required deterministic tests pass, the full
impact selection has no failures, and the final parent review has no required
finding. This record is ready for completed-plan archival.

Archived under `development_plans/archive/completed/bugfix/` on 2026-08-23.
