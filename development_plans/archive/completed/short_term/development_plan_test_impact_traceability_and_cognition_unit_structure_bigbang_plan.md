# development plan test-impact traceability and cognition unit-structure big-bang

## Summary

- Goal: prevent a semantic change-radius miss like the archived character-owned
  content judgment cutover, where the plan named cognition-core production
  changes but did not name the unit tests that prove the changed contracts.
- Status: completed.
- Scope boundary: the development-plan skill contract, the cognition-core and
  cognition-resolver test ownership boundary, the deterministic test layout,
  and the repository checks that enforce source-to-test traceability.
- Change direction: make an exact source-to-test impact matrix mandatory in
  every executable plan, mirror deterministic unit tests under their owning
  source package, and fail the required test command when an impacted test
  node is missing or not collected.
- Acceptance state: execution complete; plan-scoped acceptance evidence is
  recorded below, with unrelated repository-wide residual risks explicitly
  retained as historical execution evidence.

## Failure Being Corrected

The archived
development_plans/archive/completed/short_term/character_owned_content_judgment_cutover_plan.md
identified the production change radius, but its test surface used category
phrases such as focused contract, projection, recurrence, and propagation
tests. It did not bind each changed owner and contract to an exact test node.

The missed unit coverage was observable in two places:

- The s standard-handle removal in
  cognition_core_v2/semantic_source_planner.py was not guarded by a direct
  assertion that moral_identity questions exclude standard handles.
- The current_turn_relational_willingness.v1 to .v2 carrier change in
  cognition_resolver/contracts.py did not have direct owner-boundary tests
  for complete V2 preservation and rejection of incomplete or V1 carriers.

The existing tests exercised adjacent behavior, static prompt text, or a
positive recurrence path. Those tests could pass while either missed
regression was reintroduced. The root cause is a planning and ownership
failure, not merely a missing assertion:

1. The plan inventoried files and behavior, but stopped before enumerating
   changed symbols and contract-breaking cases.
2. Flat, mixed test files made source ownership implicit and hid the absence
   of a direct unit test.
3. Pytest discovery had no authoritative source-to-test manifest and no gate
   requiring the corresponding nodes to be collected for a changed owner.
4. Live and integration evidence was allowed to stand in for deterministic
   unit ownership, even when the changed contract was a pure validator,
   projection, or carrier boundary.

## Scope And Change Direction

This is one atomic big-bang execution phase. The skill contract, manifest,
checker, canonical unit tree, migrated owner tests, documentation, and
verification evidence are implemented and verified as one coordinated change.
There is no transitional dual test ownership model and no staged migration.

The strict source-ownership boundary is the semantic cognition path that
exposed the failure:

- all non-__init__.py modules under
  src/kazusa_ai_chatbot/cognition_core_v2/;
- all non-__init__.py modules under
  src/kazusa_ai_chatbot/cognition_resolver/; and
- the direct node boundaries named by the cutover:
  src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py,
  src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py,
  src/kazusa_ai_chatbot/nodes/dialog_agent.py, and
  src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py.

The rule generalizes beyond those paths: any future executable plan that
changes a production Python symbol must include an exact source-to-test row,
even when the file is outside this initial manifest boundary.

No live runtime source behavior is changed by this plan. The work changes
planning policy, test ownership, test infrastructure, and documentation.

## Confirmed Decisions

- The source-to-test manifest is authoritative for the initial cognition
  boundary.
- Every mapped production module has at least one deterministic unit node that
  owns its direct contract. Integration and live-LLM nodes are supplemental
  evidence and cannot be the only mapped test.
- Canonical deterministic unit tests live under tests/unit/ and mirror the
  source package and module stem. Direct owner tests are moved into that tree
  and are not duplicated in the legacy flat files.
- A change-impact verifier resolves changed source paths to exact pytest node
  IDs, checks that those IDs are collected, and runs them through the standard
  project interpreter.
- The execution command sets an enforcement flag consumed by the shared
  pytest collection hook. A targeted run that omits a required node fails
  closed instead of silently claiming coverage.
- Existing live-LLM tests retain the project rule of one case at a time with
  inspected output. This plan adds no live-LLM closeout requirement because it
  changes no runtime cognition behavior.
- The plan registry status and the plan top matter remain synchronized. The
  document is archived only after acceptance evidence is recorded.

## Mandatory Skills

- development-plan: governs this plan's contract, lifecycle, exact change
  surface, impact matrix, and execution evidence.
- test-style-and-execution: governs deterministic unit tests, integration
  tests, collection checks, and the regular versus live test boundary.
- py-style: governs scripts/validate_test_impact.py, the pytest hook, and
  all new or moved Python tests.
- cjk-safety: applies to any Python test fixture or assertion that contains
  CJK text.

## Mandatory Rules

The implementation agent shall preserve these rules:

- Add Test Impact And Traceability to the required final-plan structure
  in the development-plan skill references.
- Every row in that section shall contain an exact production path, changed
  symbol or contract, semantic owner, one or more exact pytest node IDs, test
  mode, and the regression the node prevents. Directory-only entries,
  category-only test descriptions, and phrases such as relevant tests do
  not satisfy the rule.
- A semantic production change shall have a deterministic unit test owned by
  the changed source boundary. Integration, live, static-text, and snapshot
  tests supplement that unit test; they do not replace it.
- A plan that changes a caller, callee, carrier, projection, validator,
  reducer, or output boundary shall list both the direct owner test and every
  cross-boundary propagation test required by the change radius.
- The execution gate shall verify exact node collection before accepting test
  results. A passing broader suite does not waive a missing mapped node.
- Tests shall assert contract behavior, not only source text or fixture shape.
  For semantic ownership changes, each removed gate or newly authoritative
  carrier gets a regression assertion at its owning boundary.
- The implementation agent shall preserve the repository's LLM-first
  semantic ownership boundary. The new checker may validate test ownership and
  deterministic execution; it may not add runtime semantic gates, keyword
  classifiers, moderation, or response post-processing.
- The implementation agent shall preserve all pre-existing user changes in
  the dirty worktree and compare its execution diff with the captured
  pre-execution baseline.

## Must Do

1. Update the development-plan skill and references so exact test-impact
   traceability is a required part of every executable plan and of the
   execution gate.
2. Add an authoritative JSON manifest for the strict cognition source
   boundary. It shall contain one explicit entry per mapped production module,
   an owner, the contract/symbol covered, exact deterministic unit node IDs,
   and optional supplemental integration/live node IDs.
3. Add a Python verifier that validates manifest completeness, validates that
   every listed node exists, resolves the changed production paths for a
   supplied baseline, and runs the exact impacted deterministic nodes.
4. Add a shared pytest collection enforcement hook enabled by the standard
   impact-test command. The hook shall fail when a required mapped node is not
   present in the collected test set.
5. Create the canonical mirrored unit tree:

       tests/unit/cognition_core_v2/test_<source_module>.py
       tests/unit/cognition_resolver/test_<source_module>.py
       tests/unit/nodes/test_<source_module>.py

   Direct owner tests shall be moved from the flat files into these modules,
   with duplicate definitions removed from their old locations. Shared
   cross-owner behavior belongs under tests/integration/.
6. Add the contract-breaker unit tests listed in the impact matrix below,
   including the two cases missed by the archived cutover.
7. Update the cognition testing README, root testing guidance, and HOWTO
   command examples with the canonical tree and the mandatory impact-test
   command.
8. Register the verifier as a project script and add deterministic tests for
   the manifest, verifier failure modes, collection enforcement, and this
   plan's own required impact matrix.
9. Run the complete deterministic verification set, exact impacted nodes,
   collection checks, static scope checks, and diff hygiene checks. Record
   results before lifecycle closeout.

## Deferred

- Migration of unrelated flat tests outside the cognition-core, resolver, and
  named direct-node boundary.
- A repository-wide source manifest for packages that are outside the strict
  boundary in this plan.
- New coverage, mutation-testing, or third-party test-selection dependencies.
- Changes to live LLM prompts, model routing, database fixtures, adapters, or
  production runtime semantics.
- Automatic execution of live-LLM tests. Their existing one-case, inspected
  output contract remains in force when a later runtime plan touches them.

## Target State

### Plan contract

An executable plan contains a mandatory matrix with this shape:

| Changed source path | Symbol or contract | Semantic owner | Required deterministic pytest node(s) | Supplemental node(s) | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| exact repository-relative path | exact function/class/field/contract | owning stage/module | exact path::node IDs | exact integration/live IDs or none | unit, integration, or live_llm | observable failure |

The matrix is part of review and execution scope. It is not an appendix that
can be replaced by a test-category paragraph.

### Test ownership

The manifest uses this contract:

    {
      "schema_version": 1,
      "source_roots": [
        "src/kazusa_ai_chatbot/cognition_core_v2",
        "src/kazusa_ai_chatbot/cognition_resolver",
        "src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py",
        "src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py",
        "src/kazusa_ai_chatbot/nodes/dialog_agent.py",
        "src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py"
      ],
      "entries": [
        {
          "source": "exact/path.py",
          "owner": "exact semantic owner",
          "contract": "exact symbol or boundary",
          "required_unit_tests": [
            "tests/unit/...::test_exact_contract"
          ],
          "supplemental_tests": []
        }
      ]
    }

The real manifest shall use repository-relative POSIX paths, contain no
wildcard source entries, and contain no empty required_unit_tests list.
Every required node shall be deterministic and collected by the regular
pytest command. A new or changed source module in the strict boundary without
an explicit manifest entry fails the ownership check.

### Enforcement

scripts/validate_test_impact.py shall provide these behaviors:

- validate the manifest schema and strict-root completeness;
- resolve tracked and newly created production paths relative to a supplied
  baseline;
- report the exact mapped node IDs for every changed source path;
- invoke pytest --collect-only and fail if any required node is absent;
- run the exact deterministic node set when requested; and
- return a non-zero exit code for an unmapped source, stale node, empty unit
  mapping, missing collection, or failed test.

tests/conftest.py shall enforce the same collected-node invariant when
KAZUSA_TEST_IMPACT_REQUIRED=1. The hook shall use fixed subprocess argument
arrays, preserve normal pytest behavior when the flag is absent, and fail
closed when enforcement is requested but the baseline or manifest cannot be
loaded.

The project script and documented command shall be:

    venv\Scripts\python -m scripts.validate_test_impact --base-ref <recorded-baseline> --run

The implementation records the actual baseline value in execution evidence;
<recorded-baseline> is an execution input, not an architectural choice.

## Test Impact And Traceability

This plan applies the new rule to itself. The rows below are mandatory
contract-breaker tests for the failure radius that motivated the plan. The
same rows are represented in the manifest and are verified by the plan
contract test.

| Changed source path or governed artifact | Symbol or contract | Semantic owner | Required deterministic pytest node(s) | Supplemental node(s) | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py | permitted role handles for moral_identity | semantic source planner | tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_moral_identity_questions_exclude_standard_handles | none | unit | sN standard handles re-enter model-facing semantic questions |
| src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py | character-constraint question projection | semantic appraisal | tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_character_constraint_projection_excludes_standard_handles | none | unit | standard evidence is exposed through a different appraisal path |
| src/kazusa_ai_chatbot/cognition_core_v2/contracts.py | sensitive stance/state validation matrix | cognition contract validator | tests/unit/cognition_core_v2/test_contracts.py::test_sensitive_relational_willingness_accepts_all_real_states_and_stances | none | unit | a valid accepting or non-accepting character stance is rejected by state |
| src/kazusa_ai_chatbot/cognition_resolver/contracts.py | current_turn_relational_willingness.v2 complete carrier | resolver contract validator | tests/unit/cognition_resolver/test_contracts.py::test_current_turn_carrier_preserves_complete_v2_decision; tests/unit/cognition_resolver/test_contracts.py::test_current_turn_carrier_rejects_v1_or_incomplete_decision | none | unit | recurrence reconstructs, downgrades, or accepts an incomplete decision |
| src/kazusa_ai_chatbot/cognition_resolver/state.py | current-turn carrier state round trip | resolver state owner | tests/unit/cognition_resolver/test_state.py::test_current_turn_carrier_round_trips_without_semantic_reconstruction | none | unit | state persistence drops or rewrites the authoritative decision |
| src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py | stance-to-effect propagation | action selection owner | tests/unit/cognition_core_v2/test_action_selection.py::test_non_accepting_stance_suppresses_downstream_effects | none | unit | downstream suppression is removed or becomes an independent policy gate |
| src/kazusa_ai_chatbot/cognition_core_v2/workspace.py | ordinary response stance ownership | workspace/arbitration owner | tests/unit/cognition_core_v2/test_workspace.py::test_ordinary_response_remains_authoritative_stance_owner | none | unit | competing branches replace or reinterpret the selected typed stance |
| src/kazusa_ai_chatbot/cognition_core_v2/surface.py | surface output stance contract | surface owner | tests/unit/cognition_core_v2/test_surface.py::test_surface_output_preserves_relational_willingness_v2 | tests/integration/cognition_core_v2/test_relational_stance_preserves_polarity_through_surface_and_dialog | unit | the surface output drops or rewrites the selected polarity |
| src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py | L3 surface stance handoff | L3 surface owner | tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_surface_preserves_relational_willingness_v2 | tests/integration/cognition_core_v2/test_relational_stance_preserves_polarity_through_surface_and_dialog | unit | the L3 connector drops the authoritative stance |
| src/kazusa_ai_chatbot/nodes/dialog_agent.py | terminal candidate semantic fidelity | dialog verifier owner | tests/unit/nodes/test_dialog_agent.py::test_terminal_candidate_opposite_polarity_is_withheld | tests/integration/cognition_core_v2/test_terminal_dialog_candidate_opposite_polarity_is_withheld | unit | an unverified opposite-polarity final candidate is delivered |
| .agents/skills/development-plan/SKILL.md | mandatory exact impact matrix rule | development-plan skill | tests/test_development_plan_test_impact_contract.py::test_skill_requires_exact_test_impact_matrix | none | unit | future plans omit source-to-test ownership |
| .agents/skills/development-plan/references/plan_contract.md | required plan section and exact-node fields | plan contract reference | tests/test_development_plan_test_impact_contract.py::test_plan_contract_requires_traceability_fields | none | unit | a plan passes review with category-only test language |
| .agents/skills/development-plan/references/execution_gates.md | collection and changed-source verification gate | execution gate reference | tests/test_development_plan_test_impact_contract.py::test_execution_gates_require_changed_source_collection_check | none | unit | a broad passing suite masks a missing impacted node |
| tests/ownership/source_test_impact_manifest.json | manifest schema and strict-root completeness | test ownership registry | tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary; tests/test_test_impact_manifest.py::test_manifest_rejects_empty_unit_mapping | none | unit | a changed cognition module has no owning deterministic test |
| scripts/validate_test_impact.py | changed-path resolution and exact-node validation | test-impact verifier | tests/test_test_impact_manifest.py::test_unmapped_changed_source_fails_closed; tests/test_test_impact_manifest.py::test_stale_required_node_fails_closed | none | unit | an unmapped or stale test node is treated as covered |
| tests/conftest.py | enforcement-mode collection hook | pytest test boundary | tests/test_test_impact_manifest.py::test_required_node_collection_failure_is_reported | none | unit | a targeted pytest invocation silently omits the mapped node |
| pyproject.toml | standard impact-test entry point | repository test workflow | tests/test_test_impact_manifest.py::test_documented_impact_command_is_registered | none | unit | the verifier is not available through the project interpreter |
| README.md | root test workflow guidance | repository test workflow | tests/test_test_impact_manifest.py::test_root_documentation_describes_impact_command | none | unit | agents follow a test command that omits impact enforcement |
| docs/HOWTO.md | operator test workflow guidance | repository test workflow | tests/test_test_impact_manifest.py::test_howto_documents_impact_command | none | unit | execution handoffs omit the changed-source test command |
| src/kazusa_ai_chatbot/cognition_core_v2/README.md | cognition test ownership guidance | cognition test owner | tests/test_test_impact_manifest.py::test_cognition_readme_documents_mirrored_unit_tree | none | unit | cognition changes continue using ambiguous flat test categories |

Every future plan touching one of these contracts shall add or update rows
with the same precision. The execution agent may add rows for newly discovered
callers only through a plan amendment before changing their production owner.

## Change Surface

### Delete

- Duplicate direct-owner unit test definitions from the legacy flat cognition
  and resolver test files after their canonical copies are moved. The old
  files remain only for tests whose ownership is explicitly integration,
  live-LLM, or unrelated to the strict boundary.
- Any manifest entry that uses a wildcard source path, a category-only test
  description, or an empty deterministic unit list.

### Modify

- .agents/skills/development-plan/SKILL.md: make exact test-impact
  traceability and changed-node verification core planning rules.
- .agents/skills/development-plan/references/plan_contract.md: add the
  mandatory Test Impact And Traceability section and its required columns.
- .agents/skills/development-plan/references/execution_gates.md: require
  baseline-aware source closure, exact collection checks, and recorded node
  results before completion.
- tests/conftest.py: add enforcement-mode collection validation while
  preserving all existing fixtures and database guards.
- pyproject.toml: register the verifier as a project command.
- README.md, docs/HOWTO.md, and
  src/kazusa_ai_chatbot/cognition_core_v2/README.md: document the canonical
  unit layout and exact impact-test command.
- The directly affected legacy flat cognition/resolver test modules: extract
  owner tests into the canonical unit tree and retain only correctly classified
  shared tests.
- development_plans/README.md: register this document as an active draft.

### Create

- scripts/validate_test_impact.py: manifest loader, source-path resolver,
  exact-node collector, deterministic runner, and CLI entry point.
- tests/ownership/source_test_impact_manifest.json: explicit strict-boundary
  source ownership and exact node mapping.
- tests/test_test_impact_manifest.py: verifier, manifest, collection, and
  command-registration tests.
- tests/test_development_plan_test_impact_contract.py: skill/reference and
  self-plan contract tests.
- tests/unit/cognition_core_v2/, tests/unit/cognition_resolver/, and
  tests/unit/nodes/ mirrored test modules for every mapped non-package
  source module.
- tests/integration/cognition_core_v2/ propagation tests for the typed
  stance path and terminal candidate boundary.

### Keep

- All production runtime Python behavior and semantic ownership. This plan
  does not modify files under src/ except the cognition README named above.
- Existing fixtures, live-LLM cases, live-DB cases, and unrelated flat tests
  unless a direct owner test is explicitly moved by the manifest.
- Archived plans, including the failed cutover record, as historical evidence.
- User changes present before the execution baseline and all unrelated work in
  the dirty worktree.

## Agent Autonomy Boundaries

The implementation agent may choose import layout, fixture extraction,
helper placement, test parametrization, JSON formatting, subprocess argument
handling, and command ordering when those choices preserve this contract.

The implementation agent shall not:

- modify cognition runtime semantics or add a semantic compatibility layer;
- leave direct owner tests duplicated in both legacy and canonical locations;
- replace exact node IDs with test-file, marker, or category descriptions;
- map a semantic source only to a live, static-text, snapshot, or integration
  test;
- broaden the strict source boundary to unrelated packages;
- add dependencies or CI systems outside the named change surface; or
- archive the plan before all acceptance evidence is recorded.

If an existing test cannot be classified as unit, integration, or live-LLM
without changing its semantic contract, pause that local move and record the
conflict for a plan amendment. The runtime source remains unchanged while the
conflict is resolved.

## Verification

The execution owner captures the pre-execution git status --short and
changed-path baseline first. Existing user changes are excluded from the
execution diff and remain intact.

Run these checks with venv\Scripts\python:

1. Validate the plan/reference contract and manifest tests:

       venv\Scripts\python -m pytest tests/test_development_plan_test_impact_contract.py tests/test_test_impact_manifest.py -q

2. Collect every exact node in the impact matrix and fail on a missing node:

       venv\Scripts\python -m pytest --collect-only -q tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_moral_identity_questions_exclude_standard_handles tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_character_constraint_projection_excludes_standard_handles tests/unit/cognition_core_v2/test_contracts.py::test_sensitive_relational_willingness_accepts_all_real_states_and_stances tests/unit/cognition_resolver/test_contracts.py::test_current_turn_carrier_preserves_complete_v2_decision tests/unit/cognition_resolver/test_contracts.py::test_current_turn_carrier_rejects_v1_or_incomplete_decision tests/unit/cognition_resolver/test_state.py::test_current_turn_carrier_round_trips_without_semantic_reconstruction tests/unit/cognition_core_v2/test_action_selection.py::test_non_accepting_stance_suppresses_downstream_effects tests/unit/cognition_core_v2/test_workspace.py::test_ordinary_response_remains_authoritative_stance_owner tests/integration/cognition_core_v2/test_relational_stance_preserves_polarity_through_surface_and_dialog tests/integration/cognition_core_v2/test_terminal_dialog_candidate_opposite_polarity_is_withheld

3. Run the baseline-aware exact changed-source check and impacted deterministic
   nodes:

       venv\Scripts\python -m scripts.validate_test_impact --base-ref <recorded-baseline> --run

4. Run the regular deterministic regression suite with the enforcement flag
   enabled. The execution evidence records the exact command and result.

5. Run git diff --check, Python compilation for newly created Python files,
   and a changed-path audit proving that no runtime src/**/*.py file changed
   relative to the pre-execution baseline.

6. Inspect the manifest report. It must show one explicit deterministic owner
   node for every mapped cognition/resolver/direct-node source module, with no
   stale or uncollected node IDs.

Live LLM and live DB commands are outside this plan because the runtime
behavior is unchanged. Existing live tests remain available for later plans
and retain their one-case, inspected-output contract.

## Acceptance Criteria

- The development-plan skill and both named references require an exact
  Test Impact And Traceability section with source path, symbol/contract,
  owner, exact pytest node, mode, and regression fields.
- This plan's own matrix is complete, uses exact node IDs, and is validated by
  deterministic tests before the plan can be promoted.
- The strict cognition source manifest has an explicit entry for every mapped
  non-package production module and every entry has at least one deterministic
  unit node.
- The two missed regressions are directly tested: standard sN handles cannot
  enter moral_identity questions, and current-turn recurrence preserves a
  complete V2 decision while rejecting V1/incomplete carriers.
- The stance matrix, action-effect propagation, workspace ownership, surface
  propagation, and terminal candidate withholding each have the exact nodes
  named in the matrix.
- The verifier fails closed for an unmapped source, empty unit mapping, stale
  node, absent collected node, or failed impacted test.
- The standard impact-test command invokes the verifier and required pytest
  collection enforcement through the project virtual environment.
- Direct owner unit tests mirror the source package/module structure and are
  not duplicated in their legacy flat locations.
- Documentation explains when the impact command is mandatory and distinguishes
  unit, integration, and live-LLM evidence.
- The execution diff contains no unapproved production runtime change and
  preserves the pre-existing dirty worktree.
- git diff --check, the deterministic plan/manifest tests, exact matrix node
  collection, the baseline-aware impact command, and the regular deterministic
  regression suite all pass with evidence recorded.
- The registry row, plan top matter, checklist, and final lifecycle status are
  synchronized before archival.

## Progress Checklist

This checklist is executed as one phase; the rows are work items, not rollout
phases.

- [x] Capture baseline and confirm the exact owned file set.
- [x] Update skill/reference rules and add the self-plan contract tests.
- [x] Add manifest, verifier, project command, and collection enforcement.
- [x] Migrate/create the mirrored owner unit tests and propagation tests.
- [x] Update documentation and registry evidence.
- [x] Run every verification and acceptance gate; record results and residual
  risk.
- [x] Promote, complete, and archive only after the acceptance state is
  evidenced.

## Execution Evidence

### Baseline and owned scope

- Execution baseline: `2d7715edf49efcf80b0c270589024891aca436d7`.
- The pre-execution `git status --short --untracked-files=all` was clean.
- The owned set was the skill/reference contract, plan registry and document,
  test-impact verifier and manifest, pytest enforcement hook, mirrored
  cognition/resolver/node unit tests, cognition integration tests, and the
  three testing guidance documents. No pre-existing user change was present
  to exclude.

### Delivered change

- The development-plan skill, plan contract, and execution gates now require
  exact source-to-test traceability and exact-node collection evidence.
- `tests/ownership/source_test_impact_manifest.json` contains 37 explicit
  strict-boundary source entries, 48 deterministic owner nodes, and 3
  supplemental propagation nodes.
- `scripts/validate_test_impact.py`, the `validate-test-impact` project entry
  point, and the enforcement-mode `pytest` collection hook are installed.
- The canonical unit tree mirrors cognition-core, resolver, and direct node
  source ownership. Direct workspace, action-effect, and L3 tests were
  removed from the legacy flat relational-willingness module after their
  canonical owner tests were established. Canonical surface and dialog
  fixtures no longer depend on legacy flat test modules.
- The two missed regressions are directly covered: standard `sN` handles are
  excluded from `moral_identity` questions, and the resolver preserves a
  complete V2 current-turn decision while rejecting V1/incomplete carriers.

### Verification evidence

- Plan and manifest contract tests: `13 passed in 0.67s`.
- Exact manifest collection and run: `48 tests collected`; `48 passed in
  8.98s`.
- Plan-scoped canonical unit and cognition integration suite: `50 passed in
  15.00s`.
- Explicit propagation nodes: `2 passed in 6.85s`.
- Legacy relational-willingness regression file plus canonical moved-owner
  tests: `23 passed in 0.75s`.
- Baseline-aware command
  `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run`
  reported no changed strict-boundary source paths and exited successfully.
- Python compilation for all newly created/affected Python files passed;
  `git diff --check` passed; `git diff --name-only HEAD -- 'src/**/*.py'`
  was empty, proving no runtime Python source changed.

### Deviations and residual risk

- A repository-wide deterministic run exposed an unrelated flaky control
  console timing assertion (`Starting Brain service` versus the completed
  notice); the isolated rerun passed (`1 passed in 11.96s`).
- The non-E2E repository run reached 910 passing tests before stopping at the
  unrelated baseline harness case
  `test_neutral_case_expansion_has_no_asuna_source_contamination`, whose
  private `test_artifacts/chat_history_638473184_recent.json` fixture is
  absent. The plan-scoped suite is green, and no control-console or baseline
  harness source was changed. These repository-wide issues remain recorded
  for their owning plans rather than being folded into this change.
- No live-LLM test was run because this execution changes no runtime semantic
  behavior. The existing terminal-dialog test path emitted the repository's
  configured test-database connection log while its model calls were mocked;
  no database data was modified.

## Independent Plan Review

Before approval, review this plan for:

- exact source/test ownership rather than test-category wording;
- completeness of the cognition-core and resolver failure radius;
- absence of a runtime semantic change or compatibility layer;
- correctness of the one-phase cutover and exclusion boundary; and
- acceptance evidence that a broad passing suite cannot hide a missing owner
  test.

The reviewer has review authority only and does not edit files, execute the
plan, or convert this draft into authorization.
