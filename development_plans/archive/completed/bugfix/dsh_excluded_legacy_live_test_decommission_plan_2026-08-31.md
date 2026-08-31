# DSH Excluded Legacy Live-Test Decommission

## Summary

- Goal: remove the old DSH live component suites excluded from the approved
  Phase 3 trigger-source sign-off execution.
- Status: completed
- Scope boundary: test-only deletion plus removal of the stale current HOWTO
  command; production behavior and the approved sign-off matrix are unchanged.
- Change direction: retain the ten executed trigger-source E2E cases and the
  two executed dialog component cases as the current DSH live verification
  surface.
- Acceptance state: passed on 2026-08-31.

## Scope And Change Direction

The user explicitly directed that old DSH tests excluded from their approved
execution be removed. Delete the three legacy component suites that the Phase
3 sign-off reset classified as outside its execution command. Keep the generic
Stage 3 fresh-database suite because it is not a DSH resolver test.

## Mandatory Skills And Rules

- Apply `development-plan`, `test-style-and-execution`, and `py-style`.
- Preserve the existing worktree and all approved Phase 3 implementation work.
- Keep archived plans immutable as historical evidence.

## Must Do

- Delete `tests/test_agentic_resolver_live_llm.py`.
- Delete `tests/test_dsh_standard_profile_live_llm.py`.
- Delete `tests/test_dsh_brain_interaction_live_llm.py`.
- Remove the deleted standalone resolver command from `docs/HOWTO.md`.
- Verify that the ten trigger-source nodes and two dialog nodes still collect.
- Verify that no current non-archive file references a deleted suite.

## Deferred

- Production code, prompts, schemas, routes, persistence, and deployment.
- The non-DSH `tests/test_stage3_fresh_database_e2e_live_llm.py` suite.
- New replacement tests or changes to the approved twelve-node live surface.
- Live execution; this change removes explicitly unexecuted legacy suites.

## Target State

Current DSH live verification consists of the ten approved trigger-source
sign-off nodes and the two executed task-resolution dialog component nodes.
No old standalone, Standard-profile, or internal Brain-interaction live suite
remains collectable or advertised in current operator documentation.

## Execution Role

- Responsibility: delete the excluded legacy test surface and verify current
  collection.
- Owned surface: the three deleted test files, `docs/HOWTO.md`, this plan, and
  `development_plans/README.md`.
- Authority: test and documentation changes explicitly commanded by the user.
- Applicable skills: `development-plan`, `test-style-and-execution`,
  `py-style`.
- Capability floor: repository-aware test ownership review and pytest
  collection verification.
- Independence requirement: none; no production change or independent sign-off
  is required for this deletion-only override.
- Acceptance output: deletion diff, clean current-reference scan, and exact
  twelve-node collection result.
- Gate: the three excluded files exist at entry; all acceptance criteria pass
  at exit.

## Test Impact And Traceability

| Governed artifact | Changed contract | Owner | Deterministic nodes | Supplemental nodes | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| `tests/test_agentic_resolver_live_llm.py` | excluded standalone live diagnostics removed | test suite | none; test-only deletion | none | collection audit | obsolete standalone resolver tests remain mistaken for Phase 3 gates |
| `tests/test_dsh_standard_profile_live_llm.py` | excluded Standard-profile live diagnostics removed | test suite | none; test-only deletion | none | collection audit | prompt/tool choreography diagnostics encourage overfitting outside sign-off |
| `tests/test_dsh_brain_interaction_live_llm.py` | excluded internal-interaction live diagnostic removed | test suite | none; test-only deletion | none | collection audit | internal component behavior is mistaken for source-entry E2E evidence |
| `docs/HOWTO.md` | current live-test example references only retained tests | operator documentation | none | twelve retained live nodes | static scan plus collection | operators invoke a deleted legacy suite |

## Change Surface

### Delete

- `tests/test_agentic_resolver_live_llm.py`
- `tests/test_dsh_standard_profile_live_llm.py`
- `tests/test_dsh_brain_interaction_live_llm.py`

### Modify

- `docs/HOWTO.md`
- `development_plans/README.md`

### Keep

- the five `tests/test_dsh_*_e2e_live_llm.py` trigger-source files;
- `tests/test_task_resolution_dialog_live_llm.py`;
- `tests/test_stage3_fresh_database_e2e_live_llm.py`.

## Agent Autonomy Boundaries

The executor may choose collection and static-audit command mechanics. Any
production edit, replacement test, sign-off-oracle change, or non-DSH test
deletion requires a separate user decision.

## Verification

- Confirm the three legacy files are absent.
- Scan current non-archive files for their exact names.
- Collect the five trigger-source files and dialog component file with the
  `live_llm` marker and confirm exactly twelve nodes.
- Run `git diff --check`.

## Acceptance Criteria

1. All three excluded old DSH component suites are deleted.
2. Current documentation contains no command for a deleted suite.
3. The approved ten-case sign-off matrix and two dialog cases collect exactly.
4. The generic non-DSH Stage 3 suite remains unchanged.
5. No production file is changed by this decommission.

## Execution Evidence

- Deleted all seven excluded nodes by removing the three governed files.
- The current non-archive reference scan returned no deleted-suite match.
- The five trigger-source files and dialog component file collected exactly
  twelve live nodes in 1.13 seconds.
- Repository-wide live collection succeeded with 468 collected and 3,406
  deselected nodes; one unrelated manifest-dependent cohort remained skipped.
- `git diff --check` passed with line-ending warnings only.
- `tests/test_stage3_fresh_database_e2e_live_llm.py` remained unchanged.

## Outcome

The DSH live release surface now contains only the executed ten-case
trigger-source sign-off matrix and the two executed dialog component cases.
The excluded standalone resolver, Standard-profile, and internal Brain-
interaction suites are removed rather than retained as unexecuted tests.
