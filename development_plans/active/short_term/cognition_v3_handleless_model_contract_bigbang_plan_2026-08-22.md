# Cognition V3 Handleless Model Contract Big-Bang Plan

## Summary

- **Goal:** Eliminate cognition generation failures caused by opaque handle,
  target-path, evidence-reference, and cross-stage contract matching.
- **Status:** `draft`; owner approval and a separate implementation command
  are required before production edits.
- **Scope boundary:** Cognition model-facing input/output contracts and their
  deterministic binding into the existing cognition state, goal, action, and
  resolver owners.
- **Change direction:** Keep internal references deterministic, remove them
  from model output, and bind every semantic result to a caller-owned task.
- **Cutover:** Big-bang replacement of the current handle-emitting cognition
  protocol. No compatibility schema, alias mapper, or old test migration.
- **Acceptance state:** A1, A2, G, optional W, and P retain their functional
  responsibilities while ordinary and captured-failure live cases complete
  on first-pass model generations without handle validation or repair.

## Evidence And Design Verdict

The primary defect is the cognition call design, not insufficient model
reasoning.

### Fresh real-LLM reproduction

One effect-free real-model run of `ordinary_neutral_response` used
`gemma-4-31b-isometry-fabled-persona-i1` and took 12 calls / 83.61 seconds.
The outer pytest and final output schema passed, but cognition contained these
generation failures:

| Owner | First result | Recovery result | Exact evidence |
| --- | --- | --- | --- |
| A1 `event_agency` | Rejected | Repeated and exhausted | `ce1` was permitted as subject/object and forbidden as a role-assignment entity; the model used it as both object representations |
| A2 `relationship_social` | Rejected | Repeated and exhausted | Grouped A2 used prior-transcript event handle `ce1`; singleton used evidence handle `e1`; neither belonged to the local relationship subject domain |
| P1 | Rejected | Rejected twice, then empty fallback | All three candidates included `self_cognition_response`; the shared system anchor advertised it while the local ordinary-turn schema forbade it |

The semantic content itself was coherent: the model correctly identified the
current user as intentionally moving the meeting and produced a valid
answer-now action plan. Rejection arose from identity vocabulary and field
ownership.

Raw evidence:

- `test_artifacts/cognition_core_v3/cogv3-cognition-root-cause-repro-20260822/raw_trials/ordinary_neutral_response__v3__trial-1__attempt-1.json`
- `test_artifacts/diagnostics/cognition_v3_real_llm_root_cause_reproduction_2026-08-22.md`

### Retained 72-run cohort

Direct recount of
`test_artifacts/cognition_core_v3/cogv3-g7-input-flow-final-20260822/raw_trials`
shows:

- 46 of 72 trials have appraisal-family exhaustion;
- 225 singleton appraisal recovery calls were added;
- 526 total model calls were made, with median 7 per trial;
- failures occur in every family: `relationship_social` 27,
  `existential_drive` 22, `event_agency` 15,
  `epistemic_comparison_memory` 8, `goal_threat_outcome` 6, and
  `moral_identity` 3.

A defect distributed across all six families and repeated after recovery is a
shared calling-routine defect. A stronger model may hide the defect more
often; it cannot make the contradictory contract correct.

## Confirmed Proposal Decisions

1. Internal IDs and references remain deterministic and private.
2. Opaque cognition handles such as `e1`, `ce1`, `ct1`, `ck1`, `ev1`, `g1`,
   `r1`, `b1`, `a1`, and storage IDs do not appear in model responses.
3. The model does not emit state target paths, evidence handles, object
   handles, or role-assignment entity handles.
4. Every semantic generation receives exactly one caller-owned bound focus.
   The caller
   already knows the task's internal matter, participants, evidence roots,
   writable axes, and downstream owner.
5. The model returns only semantic judgments: applicability, proposition
   meaning, qualitative axis change, reason, cause summary, willingness,
   goal, selection, or planning decision as owned by that stage.
6. Deterministic code attaches internal matter references, evidence roots,
   participant references, and exact state paths after the semantic product is
   accepted. This attachment is construction, not semantic inference.
7. A1 and A2 receive stage-local semantic workspaces. Accepted prior products
   are passed as compact typed data; raw assistant JSON is not conversational
   history for the next semantic owner.
8. The shared anchor contains only cognition invariants. Each stage receives
   one byte-stable stage-specific contract, so P1 ordinary and P1
   self-cognition variants never advertise each other's fields.
9. Current evidence produces one current-observation root. A1 decides whether
   supported event, threat, or knowledge-gap matters exist; code assigns
   internal references after that decision. The projection does not create
   three speculative entities per evidence row.
10. All six appraisal capacities, every cognition stage, every current state
    axis, relationship and affect projections, emotion root references and
    cause summaries, goal branches, workspace selection, action planning, and
    resolver planning remain available.
11. Semantic handle matching and target-path validation are removed from the
    LLM boundary because those values are absent from model output.
12. Canonical JSON parsing, small structural shape checks, state schema and
    transition checks, permissions, action/resolver availability, limits,
    persistence, authorization, and delivery safety remain deterministic.
13. Normal execution uses one generation per bound semantic focus. Axes owned
    by the same family/focus are generated together. Different focuses receive
    separate calls, so the model never has to echo an ID or preserve positional
    alignment. Grouped-to-singleton appraisal recovery and semantic
    regeneration are removed.
14. Provider or structurally unusable output fails closed before state commit.
    It does not receive a semantic repair loop.
15. Provider-native JSON-object mode may improve transport reliability, but it
    is not credited as a solution to identity or semantic contract failures.

## Why All Validators Are Not Removed

The current word `validator` combines unrelated responsibilities. This plan
separates them:

| Check class | Decision | Reason |
| --- | --- | --- |
| Model handle/evidence/path matching | Delete | The model no longer emits these fields, making this failure class unrepresentable |
| Post-generation semantic correctness scoring | Exclude | The producing LLM owns cognition semantics; no second semantic authority is added |
| JSON object and exact small stage shape | Keep | Prevent malformed transport from becoming runtime state |
| State axes, ranges, lifecycle, and replacement-state integrity | Keep | Protect persistent cognition state without judging meaning |
| Action/resolver availability, target authority, permission, and limits | Keep | Prevent cognition text from granting effects |
| Persistence, scheduling, and delivery checks | Keep | These are deterministic system boundaries outside semantic judgment |

Removing every check would allow an arbitrary P1 object to become an action or
persistent state mutation. The correct correction is to remove model-owned
identity matching, not effect safety.

## Target Model-Facing Flow

```text
typed episode + current state + retrieved evidence
    -> deterministic TurnWorkspace
       - current observation and semantic participant descriptions
       - unresolved current matters only
       - stage-relevant evidence with authority/freshness descriptions
       - active affects with concrete causes
    -> A1 world/causal appraisal for each selected bound focus
       - one call owns one focus and all applicable A1 axes
       - event agency, outcome, epistemic/memory judgments
    -> deterministic I1 binding and state reduction
       - attach internal matter/evidence/axis references
    -> A2 character/relationship appraisal for each selected bound focus
       - relationship/social, moral/identity, existential/drive judgments
    -> deterministic A2 binding and final appraisal reduction
    -> G goal and relational-willingness cognition
       - semantic target roles; internal refs attached by caller
    -> deterministic I2 selection
    -> optional W pairwise selection using first/second/combine only
    -> P action/resolver planning
       - semantic capability names, with ordinary and self-cognition contracts
         kept separate
    -> deterministic authorization and output assembly
```

The only positional choice is W's pairwise `first`, `second`, or `combine`
decision. It is not persisted, does not cross a stage boundary, and cannot be
confused with evidence, matter, participant, relationship, bid, action, or
resolver namespaces.

## Bound Appraisal Contract

For each appraisal family and focus, deterministic code prepares one
`BoundAppraisalTask`. Private task fields contain the internal subject, object,
matter, evidence roots, and writable axes. The model receives only the
semantic projection:

```json
{
  "focus": "the current user's proposed meeting-time change",
  "participants": {
    "actor": "the current user",
    "affected_person": "the active character"
  },
  "evidence": [
    "The current user says the meeting was moved from 10:00 to 11:00 and asks whether that works."
  ],
  "requested_judgments": [
    "intentionality",
    "responsibility"
  ]
}
```

The corresponding result contains no reference fields:

```json
{
  "judgments": {
    "intentionality": {
      "applicable": true,
      "assessment": "clearly intentional",
      "reason": "The user describes deliberately moving the meeting time."
    },
    "responsibility": {
      "applicable": true,
      "assessment": "directly responsible",
      "reason": "The user identifies themself as the person making the change."
    }
  }
}
```

The binder attaches this result to the task's private matter, evidence roots,
participants, and axis paths. Multiple focuses are invoked separately; output
never repeats, selects, or aligns IDs.

## Emotion Cause Preservation

Emotion cause remains a first-class cognition feature:

- the model returns a concrete `cause_summary` for applicable affect changes;
- the binder attaches the bound task's matter and evidence roots as
  `primary_root` and `root_refs`;
- `cause_status` remains deterministic lifecycle state;
- multiple conflicting active affects and their causes remain visible to G;
- relationship-root affect is projected with its concrete cause rather than a
  generic relationship-pressure sentence.

The model therefore explains the cause while code preserves exact provenance.

## Scope And Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_shared/contracts.py`
  - replace blanket source-kind-to-all-family visibility with the private
    bound-focus carrier used only inside cognition; preserve the public
    cognition input/output schemas.
- `src/kazusa_ai_chatbot/cognition_shared/state_projection.py`
  - keep private reference maps;
  - replace three-candidates-per-evidence model projection with one current
    observation root and semantic current-matter projection.
- `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py`
  - define private bound-task and typed stage-product carriers.
- `src/kazusa_ai_chatbot/cognition_core_v3/registry.py`
  - register stage-local contract owners without recovery-stage aliases.
- `src/kazusa_ai_chatbot/cognition_core_v3/anchor.py`
  - retain invariant cognition policy only; remove other-stage output fields.
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
  - replace handle, target-path, and evidence-echo schemas with stage-local
    semantic projections and handleless outputs.
- `src/kazusa_ai_chatbot/cognition_core_v3/semantic_source_planner.py`
  - prepare current bound focuses from typed provenance and unresolved matters;
    remove blanket six-family visibility and speculative candidate domains.
- `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py`
  - own handleless appraisal shape checks and deterministic task binding.
- `src/kazusa_ai_chatbot/cognition_core_v3/semantic_appraisal.py`
  - remove model-emitted handle/path validation and retain only bound-product
    materialization required by state reduction.
- `src/kazusa_ai_chatbot/cognition_core_v3/goal_cognition.py`
  - remove model-emitted evidence, persistent-goal, and participant handles;
    bind goal products to caller-owned sources and semantic target roles.
- `src/kazusa_ai_chatbot/cognition_core_v3/workspace.py`
  - use call-local positional selection without cross-stage bid handles.
- `src/kazusa_ai_chatbot/cognition_core_v3/action_selection.py`
  - use semantic capability names and disjoint ordinary/self-cognition P
    contracts; retain effect-boundary validation.
- `src/kazusa_ai_chatbot/cognition_core_v3/execution.py`
  - invoke stage-local prompts once and carry typed products without raw
    assistant history or semantic repair.
- `src/kazusa_ai_chatbot/cognition_core_v3/transcript.py`
  - retain trace/session facts only; remove raw accepted assistant JSON as the
    semantic handoff between different stage owners.
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
  - orchestrate the bound A1 -> I1 -> A2 -> G -> I2 -> optional W -> P flow.
- `src/kazusa_ai_chatbot/cognition_core_v3/session.py`
  - persist compact typed recurrence products rather than raw model messages.
- `src/kazusa_ai_chatbot/cognition_core_v3/diagnostics.py`
  - report first-pass stage disposition and eliminate accepted-degraded as a
    successful cognition label.
- `src/kazusa_ai_chatbot/cognition_core_v3/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `tests/ownership/source_test_impact_manifest.json`

### Delete Or Rewrite Instead Of Migrating

- tests whose sole contract is that model output emits `eN`, `ceN`, `ctN`,
  `ckN`, `evN`, `gN`, `r1`, `bN`, `aN`, target paths, or role-assignment
  handles;
- singleton appraisal recovery tests and prompt fixtures;
- raw cross-stage assistant-transcript assertions.

### Keep

- the public cognition input/output and replacement-state boundary;
- all state models, axes, affect roots, relationship state, goal state,
  reducers, authorization, persistence, resolver, and action execution
  boundaries unless an exact caller adaptation is required by this cutover;
- dialog, surface generation, RAG retrieval, adapters, database content,
  scheduler, and delivery behavior.

## Excluded Work

- Dialog or visible-output quality changes.
- Prompt examples tailored to either reproduction sentence.
- Model replacement or route tuning.
- Database cleanup or migration of existing semantic state.
- Upstream decontextualizer changes; any observed upstream failure is reported
  separately.
- Compatibility adapters for the removed handle-emitting protocol.
- Broad unit-test expansion, documentation tests, or rerunning previously
  passing Gate 7 cases.
- The approved native JSON-object transport plan's provider implementation.
  That plan executes first; this plan then consumes its completed transport
  baseline without treating it as semantic remediation.

## Execution Roles

### Parent architecture and acceptance owner

- **Responsibility:** freeze the contract, maintain plan lifecycle, inspect
  raw live evidence, review implementation scope, and decide acceptance.
- **Owned surface:** this plan, registry, architecture decisions, Luna handoff,
  diff review, and final disposition.
- **Authority:** analysis, plan amendments after owner decisions, read-only
  source/evidence inspection, and acceptance or bounded remediation requests.
- **Applicable skills:** `development-plan`, `local-llm-architecture`,
  `debug-llm`, and `test-style-and-execution` for evidence interpretation.
- **Capability floor:** full cognition architecture context, raw trace access,
  source review, and live-result interpretation.
- **Independence:** reviews the Luna implementation and verification evidence.
- **Acceptance output:** scoped diff decision and evidence-backed first-pass
  cognition verdict.
- **Gate:** starts after owner approval; closes only when every acceptance
  criterion is evidenced.

### Fixed implementation and verification executor

- **Fixed constraint:** reuse the existing `gpt-5.6-luna` worker with `max`
  reasoning on the standard-speed lane, as directed by the owner.
- **Responsibility:** implement the approved handleless cognition cutover,
  perform bounded remediation, run exact deterministic checks, and execute
  live cases one at a time.
- **Owned surface:** only the production, test, and cognition documentation
  paths listed in the approved change surface.
- **Authority:** production/test edits and scoped commands after explicit
  implementation authorization; no DB or external effects.
- **Applicable skills:** `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, and `debug-llm`.
- **Capability floor:** production Python refactor, cognition contract design,
  effect-free live-LLM execution, raw artifact inspection, and safe worktree
  preservation.
- **Independence:** implementation and verification reuse the same worker due
  the available two-slot runtime; parent retains acceptance authority.
- **Acceptance output:** scoped diff, exact deterministic results, two
  individually inspected real-LLM artifacts, and remediation handoff.
- **Gate:** receives work only after plan approval and explicit production
  implementation command; exits after parent accepts all evidence.

## Test Impact And Traceability

The final approved plan will update the ownership manifest. Verification stays
small and contract-focused.

| Source owner | Exact deterministic node | Supplemental live node | Regression prevented |
| --- | --- | --- | --- |
| Handleless stage projection and prompt | `tests/unit/cognition_core_v3/test_prompt.py::test_model_facing_stage_contracts_expose_no_opaque_handles_or_target_paths` | `tests/test_cognition_core_v3_candidate_live_llm.py::test_live_candidate_ordinary_neutral_response` | Opaque IDs, paths, or other-stage fields re-enter model contracts |
| Bound appraisal materialization and cause roots | `tests/unit/cognition_core_v3/test_appraisal.py::test_bound_appraisal_result_attaches_internal_matter_axes_and_causes_without_model_handles` | captured trace `llmtrace_0bae517c46d24c519181ddf185453146`, effect-free cognition replay | Appraisal semantics lose axes, matter ownership, or emotion causes |
| Stage-local execution handoff | `tests/unit/cognition_core_v3/test_execution.py::test_semantic_stages_receive_typed_products_without_raw_assistant_history` | `tests/test_cognition_core_v3_candidate_live_llm.py::test_live_candidate_ordinary_neutral_response` | Earlier output handles contaminate later stage choices |
| Facade ordering and first-pass disposition | `tests/unit/cognition_core_v3/test_facade.py::test_handleless_cold_chain_completes_each_stage_once_without_degraded_acceptance` | both listed live failure cases | Recovery or accepted-degraded returns as a normal path |
| P variants and effect safety | `tests/unit/cognition_core_v3/test_action_selection.py::test_ordinary_and_self_cognition_plans_use_disjoint_handleless_contracts` | ordinary neutral live case | Optional self-cognition fields leak into ordinary P1 or unsafe effects bypass checks |
| Private reference firewall | `tests/unit/cognition_core_v3/test_prompt.py::test_private_reference_map_never_crosses_model_boundary` | raw live artifact inspection | Storage or internal cognition identity leaks to the model |

Old handle-contract tests are deleted. They are not migrated into assertions
that preserve the removed design.

## Verification

1. Collect and run only the exact deterministic nodes in the traceability
   table plus source-impact nodes required for the actual changed production
   files.
2. Run the effect-free `ordinary_neutral_response` real-LLM case once and
   inspect its complete raw artifact.
3. Run one effect-free cognition-only replay of
   `llmtrace_0bae517c46d24c519181ddf185453146` and inspect its complete raw
   artifact.
4. Inspect raw model messages to prove opaque handles, target paths, raw prior
   assistant JSON, and irrelevant P fields are absent.
5. Inspect typed state output to prove all applicable axes and emotion causes
   bind to their internal roots.
6. Run `scripts.validate_test_impact --base-ref HEAD --run` for the actual
   production diff and `git diff --check`.

## Acceptance Criteria

1. Model-facing cognition requests and responses contain no internal or
   storage identity, evidence handle, matter handle, target path, bid handle,
   action handle, resolver handle, or cross-stage assistant JSON.
2. A1, A2, G, optional W, and P preserve their full semantic responsibilities.
3. All six appraisal families and all current state axes remain representable.
4. Affect outputs preserve concrete `cause_summary`, `primary_root`,
   `root_refs`, and `cause_status` through deterministic binding.
5. The ordinary live case completes with one A1, one A2, one G1a, and one P1
   generation, zero singleton calls, zero repair calls, zero family
   exhaustion, and no accepted-degraded disposition.
6. The captured production-failure replay completes every required cognition
   stage on first generation with no partial or degraded cognition result.
7. No pass depends on exact visible wording, dialog output, a semantic
   evaluator, a repair model, retry, validator weakening, or case-shaped
   prompt instructions.
8. Malformed transport, invalid persistent state, unauthorized effects,
   unavailable capabilities, invalid limits, or unsafe persistence still fail
   closed at their deterministic owners.
9. No compatibility layer or legacy handle-emitting model contract remains.
10. Existing passing Gate 7 cases are not rerun.

## Approval Boundary

This document is a proposal. Approval fixes the handleless model boundary,
validator separation, big-bang cutover, scope, and acceptance criteria. A
separate explicit implementation command is still required before any
production or test edit.
