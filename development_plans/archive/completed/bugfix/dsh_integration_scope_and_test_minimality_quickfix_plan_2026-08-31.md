# DSH Integration Scope And Test Minimality Quickfix

- **Status:** completed
- **Created:** 2026-08-31
- **Baseline:** `4fab6a9af34f75804b4eab5ee42bddf0c28f0b8b`
- **Authorization:** the user explicitly requested the audit, this quickfix
  plan, and implementation of every surfaced issue.

## Goal

Keep the three DSH integrations and their current production behavior while
removing integration residue that does not belong to the product: tracked
runtime state, phase-labelled maintenance code, displaced or stale product
documentation, and tests that police deleted files, prompt wording,
documentation prose, planning manifests, or their own harness.

## Confirmed Findings

1. `.dsh-debug/` contains tracked media records and SQLite runtime state.
2. the retained drain CLI and DB helper expose temporary `Plan 3` names in
   Python source and in their public report vocabulary.
3. `README.md` still describes the standalone pre-cutover route, user relay,
   deleted model routes, the deleted coding agent, and deleted RAG2 execution.
   `README_CN.md`, `docs/HOWTO.md`, `docs/SUBAGENT_INTERFACES.md`, and
   `docs/architecture/cognition_contracts_design.md` lost unrelated product or
   architecture content when DSH documentation was written.
4. DSH integration commits added static deletion/decommission checks,
   documentation-text tests, prompt-literal tests, a cognition-producer
   inventory, removed-source manifest machinery, live-harness self-tests, and
   unrelated Asuna/RAG prompt fixtures. These protect an implementation batch
   rather than current behavior.

## Change Surface

### Runtime and repository hygiene

- ignore `.dsh-debug/` at the repository root and remove only its tracked
  runtime artifacts;
- rename `scripts/check_dsh_plan3_drain.py` to
  `scripts/check_dsh_legacy_drain.py`;
- rename `count_dsh_plan3_drain_rows` and the report schema to stable legacy
  drain vocabulary;
- replace plan-labelled docstrings in current Python owners with product
  terminology.

### Documentation

- retain the global product/onboarding structures in `README.md` and
  `README_CN.md`, and describe the current DSH-only task edge;
- restore non-DSH setup, adapter, HTTP API, data, and operations material in
  `docs/HOWTO.md`, then add the current DSH runbook;
- restore retained RAG3 and web-agent interfaces in
  `docs/SUBAGENT_INTERFACES.md` while leaving deleted executors out;
- restore the general cognition contracts in
  `docs/architecture/cognition_contracts_design.md` and replace only the task
  handoff section with the current DSH contract;
- remove temporary plan/evidence-ledger language from current DSH architecture
  and subsystem READMEs.

### Test minimality

Delete these process-only files and fixtures:

- `tests/test_agentic_resolver_decommission.py`;
- `tests/unit/task_resolution/test_decommission.py`;
- `tests/test_dsh_plan3_documentation.py`;
- `tests/test_cognition_llm_producer_matrix.py` and its JSON fixture;
- `tests/unit/test_config_dsh_cutover.py`;
- the Asuna private-affinity replay/live/harness files, the group-style live
  harness, and the Asuna revision live harness;
- `tests/test_rag_agent_package_prompt_stability.py` and its snapshot fixture.

Remove only the prompt-literal or harness-self-test nodes from the retained
cognition, L3, dialog, DSH-admission, and DSH E2E files. Keep typed schema,
authority, lifecycle, recovery, evidence, safety, and real behavior coverage.
Remove static assertions for deleted relay/provider/config/document paths from
mixed test files.

Restore `scripts/validate_test_impact.py` to active-source ownership only,
delete the `removed_sources` vocabulary, and remove mappings to deleted or
prompt-literal nodes from the ownership manifest. No replacement absence test
will be added.

## Exclusions

- no DSH profile, RPC, catalog, task, interaction, cognition, persistence, or
  delivery behavior redesign;
- no restoration of coding-agent, complex-task, RAG2, or legacy task executor
  code;
- no live database mutation, deployment, or environment-file inspection;
- no prompt tuning or deterministic post-processing of model semantics.

## Verification

The quickfix is complete when:

1. Git tracks no `.dsh-debug/` path and the root ignore rule covers it;
2. current Python under `src/` and `scripts/` contains no DSH plan label;
3. current product docs contain the preserved global sections, current DSH
   ownership, and no deleted coding/background-model route guidance;
4. removed-source, documentation-text, decommission, prompt-literal, and
   harness-self-test nodes no longer collect;
5. retained DSH contract, lifecycle, media-safety, interaction, cognition
   recurrence, and terminal-surface tests pass;
6. the impact validator, Python compilation, configured Ruff regression check,
   full non-live pytest suite, and `git diff --check` pass.

No new pytest node is created for the drain rename: the CLI receives a direct
`--help`/compile check, consistent with this plan's test-minimality objective.

## Evidence

- Baseline and integration boundaries:
  `4fab6a9af34f75804b4eab5ee42bddf0c28f0b8b` is the exact pre-DSH
  baseline; Stage 1 ends at `efcb83a7`, Stage 2 ends at `59357e59`, and
  Stage 3 ends at the audited `c001a541` workspace base.
- Repository hygiene: `git ls-files -- '.dsh-debug/**'` returned no paths;
  twelve tracked runtime media/SQLite artifacts were removed and the root
  ignore rule now owns future runtime state.
- Stable surface audit: the current product documentation/source scan found no
  DSH Plan 1/2/3 labels, obsolete drain names, user-relay wording, or removed
  background/coding route guidance outside historical development plans.
- Python: all changed Python compiled. Configured Ruff passed on the quickfix
  Python surface. The unchanged `script_operations.py` import-order and six
  `TRY004` diagnostics reproduce byte-for-byte from `HEAD`; the renamed drain
  helper introduced no new diagnostic.
- Impact ownership:
  `venv\Scripts\python scripts\validate_test_impact.py --check-all` validated
  all 482 exact active-source nodes in bounded Windows-safe batches.
- Focused behavior:
  the retained DSH resolver, interaction, gateway, task, worker, cognition,
  composition, and impact modules passed 120 tests.
- Sidecar: the pinned TypeScript build passed and Vitest passed all 100 tests
  across 14 files.
- Full regular suite:
  `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`
  completed with 3,364 passed, 4 skipped, and 507 live tests deselected.
- Final static checks: the renamed drain CLI `--help`, manifest JSON parse, and
  `git diff --check` passed.

## Outcome

The current product retains the complete DSH execution, authority, catalog,
interaction, persistence, cognition, recurrence, and delivery behavior. The
quickfix removes only integration residue: tracked runtime state, temporary
phase vocabulary, displaced documentation, process-only test machinery,
prompt/source/prose assertions, and unrelated live-test harness additions.
