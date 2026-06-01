# resolver_default_mainline_cutover_plan

## Summary

- Goal: Make the cognition resolver the default and only live persona workflow
  on `resolver-goal-poc`, remove the legacy mandatory RAG-first path, and
  prepare a clean merge from this branch into `main`.
- Plan class: high_risk_migration
- Status: in_progress; branch-side closure complete, Stage 8 remains gated by
  explicit user instruction.
- Mandatory skills: `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `debug-llm`, `development-plan`,
  `cjk-safety` for Python prompt edits.
- Overall cutover strategy: bigbang.
- Highest-risk areas: persona graph shape, demand-driven RAG behavior, HIL
  continuation, self-cognition resolver loops, prompt contract drift, bilingual
  documentation drift, and test deletion that accidentally removes useful RAG
  coverage.
- Acceptance criteria: resolver-only persona graph is the default and only
  runtime path; `COGNITION_RESOLVER_ENABLED` and legacy `stage_1_research`
  compatibility are removed; English and Chinese docs describe resolver as the
  baseline; deterministic and real LLM validation pass; extensive post-review
  regression passes; `main` merge waits for explicit user instruction.

## Context

This plan supersedes
`development_plans/archive/superseded/cognition_preserving_goal_resolver_production_plan.md`.

The superseded plan implemented the resolver behind
`COGNITION_RESOLVER_ENABLED=false` by default. That target was replaced by the
confirmed product decision:

- the resolver is now the baseline architecture on this branch;
- the old mandatory RAG-first live persona path is not preserved;
- no runtime backward compatibility or feature-flag fallback is required;
- RAG remains available only as a cognition-selected resolver capability;
- all semantic action and evidence decisions remain inside L1 -> L2 -> L2d.

Current branch state on `resolver-goal-poc`:

- the resolver-only live persona graph is implemented;
- `COGNITION_RESOLVER_ENABLED` and the legacy live `stage_1_research` graph
  node are removed from production code and live docs;
- RAG remains available through the resolver-selected
  `run_rag_evidence_for_persona_state(...)` helper;
- full deterministic regression passed;
- main merge remains blocked by Stage 8 until explicit user instruction.

The live graph is:

```text
stage_0_msg_decontexualizer
  -> stage_1_goal_resolver
  -> stage_2_memory_lifecycle
  -> stage_3_action / stage_3_no_response
```

## Mandatory Skills

- `local-llm-architecture`: load before graph, resolver, RAG, cognition,
  prompt, L3, or real LLM validation changes.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, deleting, or
  running tests.
- `debug-llm`: load before real LLM diagnostics and output-quality review.
- `cjk-safety`: load before editing Python files containing Chinese prompt
  strings.
- `development-plan`: load before updating this plan, lifecycle status,
  execution evidence, or merge sign-off.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan
  before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.
- Execution must use parent-led native subagent execution unless the user
  explicitly approves fallback execution for this plan.
- Preserve Kazusa as a cognition core system. Do not introduce a generic agent
  harness, direct tool router, or assistant-style planner outside the existing
  three-layer cognition.
- All semantic action direction remains in L1 -> L2 -> L2d. Deterministic code
  may validate, execute, limit, persist, and route observations; it must not
  decide when evidence is semantically needed.
- Remove runtime compatibility with the old mandatory RAG-first persona path.
  Do not preserve `COGNITION_RESOLVER_ENABLED`, hidden fallback branches,
  disabled-mode tests, or docs that instruct users to opt into the resolver.
- Keep existing RAG2 capability helpers that the resolver uses. Removing
  `stage_1_research` must not remove `run_rag_evidence_for_persona_state(...)`
  or the RAG cognitive episode adapter.
- No LLM stage may be asked to copy opaque ids. If a future stage needs to
  choose among items, use ordinal aliases and map back deterministically.
- Prompt edits must be generic and organically linked to the capability
  contract. Do not add lookup-table examples tuned to validation cases.
- Do not read `.env` unless the user explicitly asks for environment
  inspection. Validation may set environment variables inside subprocess tests.

## Must Do

- Remove `COGNITION_RESOLVER_ENABLED` from production config and docs.
- Remove the disabled-mode branch from `persona_supervisor2(...)`.
- Remove `stage_1_research(...)` as a live persona graph node.
- Keep resolver RAG execution through `run_rag_evidence_for_persona_state(...)`.
- Rewrite graph tests so the default persona graph always runs
  `stage_1_goal_resolver` and never calls `stage_1_research`.
- Rewrite or delete tests whose only purpose is preserving legacy mandatory
  RAG-first behavior.
- Preserve focused tests for RAG request construction, RAG projection, RAG event
  logging, and resolver capability RAG execution by moving assertions to the
  retained helper/capability boundary when needed.
- Update `README.md`, `README_CN.md`, `docs/HOWTO.md`,
  `src/kazusa_ai_chatbot/nodes/README.md`, and
  `src/kazusa_ai_chatbot/rag/README.md` to describe resolver-only live persona
  flow.
- Write `README_CN.md` in native, idiomatic Chinese. Do not translate the
  English README mechanically; update it as a first-class Chinese document with
  natural wording and the same architectural truth.
- Run deterministic and real LLM validation before merging to `main`.

## Deferred

- Removing RAG2 itself.
- Removing resolver max-cycle or capability-timeout controls.
- Adding a new RAG, new memory lookup system, direct DB tool, shell tool, or
  generic planner.
- Changing adapter delivery, scheduler permission, consolidation target
  routing, or MongoDB schema outside what is required by resolver-only graph
  cleanup.
- Adding progressive dialog or parallel speak/tool output.
- Keeping a runtime old-path escape hatch.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Live persona graph | bigbang | Always wire `stage_0_msg_decontexualizer -> stage_1_goal_resolver -> stage_2_memory_lifecycle`. Delete the disabled-mode graph branch. |
| Resolver enable flag | bigbang | Delete `COGNITION_RESOLVER_ENABLED` from config, docs, and tests. Do not invert it to default true; remove the choice. |
| Legacy mandatory RAG-first node | bigbang | Delete `stage_1_research(...)` as a persona graph node and remove direct graph tests for it. |
| RAG helper boundary | compatible | Keep `run_rag_evidence_for_persona_state(...)`, `build_text_chat_rag_request(...)`, RAG2 projection, and resolver `rag_evidence` capability behavior. These are not legacy; they are resolver capabilities. |
| Tests | bigbang | Delete disabled-path tests and rewrite retained RAG assertions around helper/capability entrypoints. |
| Docs | bigbang | Rewrite docs so resolver is the default architecture and old RAG-first is not documented as a runtime mode. |
| Merge to main | migration | First merge or rebase latest `main` into `resolver-goal-poc`, run verification, fix conflicts, then merge `resolver-goal-poc` to `main`. |

## Cutover Policy Enforcement

- The execution agent must follow the selected policy for each area.
- Do not choose a more conservative compatibility strategy by default.
- For bigbang areas, delete or rewrite legacy references instead of preserving
  them.
- For compatible areas, preserve only the surfaces explicitly listed
  above.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

Production persona turn:

```text
adapter/debug client
  -> brain service / queue / relevance
  -> stage_0_msg_decontexualizer
  -> stage_1_goal_resolver
       bounded loop:
         L1 -> L2 -> L2d
         optional cognition-selected resolver capability
         observation projected into next cognition cycle
  -> stage_2_memory_lifecycle
  -> stage_3_action or stage_3_no_response
  -> persistence / consolidation / scheduler
```

`stage_1_goal_resolver` is not a feature mode. It is the only live persona
workflow.

## Design Decisions

1. Delete the enable flag rather than defaulting it to true.

   The user explicitly rejected backward compatibility. Keeping the flag would
   leave the old path reachable and would make docs/tests continue treating the
   resolver as optional.

2. Delete only legacy live graph code, not resolver-owned RAG helpers.

   `stage_1_research(...)` is legacy because it forces RAG before cognition.
   `run_rag_evidence_for_persona_state(...)` is retained because it is the
   resolver capability implementation selected by L2d.

3. Rewrite tests by ownership boundary.

   Tests that prove the old graph order are deleted. Tests that prove RAG2
   projection, event logging, request shape, and resolver capability execution
   are kept or moved to the retained helper/capability boundary.

4. Treat real LLM validation as merge evidence, not prompt tuning.

   If a real LLM case chooses no RAG with a coherent reason, that can pass.
   If it repeats equivalent evidence requests or fails to answer a goal, record
   it and fix only contract bugs or generic prompt issues.

## Change Surface

Production code:

- Modify: `src/kazusa_ai_chatbot/config.py`
- Modify: `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- Modify only if direct references break:
  `src/kazusa_ai_chatbot/nodes/README.md`,
  `src/kazusa_ai_chatbot/rag/README.md`

Tests:

- Modify: `tests/test_config.py`
- Modify: `tests/test_cognition_resolver_persona_graph.py`
- Modify/delete legacy references in:
  - `tests/test_persona_supervisor2.py`
  - `tests/test_persona_supervisor2_rag2_integration.py`
  - `tests/test_persona_supervisor2_rag_skip_shape.py`
  - `tests/test_rag_dialog_event_logging.py`
  - `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  - `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`
- Keep and rerun:
  - `tests/test_cognition_resolver_loop.py`
  - `tests/test_cognition_resolver_persona_graph.py`
  - `tests/test_rag_cognitive_episode_adapter.py`
  - `tests/test_cognition_prompt_contract_text.py`
  - `tests/test_persona_supervisor2_action_initializer.py`

Docs and plans:

- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `docs/HOWTO.md`
- Modify: `src/kazusa_ai_chatbot/nodes/README.md`
- Modify: `src/kazusa_ai_chatbot/rag/README.md`
- Modify: `development_plans/README.md`
- Archive/supersede:
  `development_plans/archive/superseded/cognition_preserving_goal_resolver_production_plan.md`

## Overdesign Guardrail

- Do not add a new config flag to replace `COGNITION_RESOLVER_ENABLED`.
- Do not add shadow mode, dual-run comparison, or runtime fallback.
- Do not redesign resolver recurrence, RAG2, L2d schema, or pending HIL storage
  unless a test failure proves the current contract cannot support resolver-only
  cutover.
- Do not broaden cleanup into unrelated old plans, adapters, persistence, or
  scheduler code.

## Agent Autonomy Boundaries

- The execution agent may remove code and tests that are exclusively tied to
  the old mandatory RAG-first graph path.
- The execution agent may rewrite docs to remove optional-resolver language.
- The execution agent may update tests to prove resolver-only behavior.
- The execution agent must not remove retained RAG2 capability helpers, RAG
  projection, or resolver capability contracts.
- The execution agent must stop and ask for approval if cleanup requires
  changing database schemas, adapter delivery contracts, scheduler permission
  semantics, or L1/L2/L2d output schemas beyond prompt/doc wording.

## Implementation Order

1. Parent establishes graph/config test contract.
   - Update `tests/test_config.py` to assert `COGNITION_RESOLVER_ENABLED` is no
     longer exported and resolver cycle/timeout config remains.
   - Update `tests/test_cognition_resolver_persona_graph.py` so the default
     graph calls `stage_1_goal_resolver` and has no disabled branch.
   - Run both tests and record the expected failures before production edits.

2. Production-code subagent removes the old runtime branch.
   - Edit `src/kazusa_ai_chatbot/config.py` to delete
     `COGNITION_RESOLVER_ENABLED`.
   - Edit `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` to remove the
     import, delete `stage_1_research(...)`, always add
     `stage_1_goal_resolver`, and always edge from decontextualizer to resolver.
   - Do not change `run_rag_evidence_for_persona_state(...)`.

3. Parent rewrites legacy RAG-first tests.
   - For each failing test that imports or patches `stage_1_research`, decide
     from its assertion target:
     - graph-order assertion: delete or rewrite to resolver-only graph order;
     - RAG request/projection assertion: move to
       `run_rag_evidence_for_persona_state(...)` or resolver `rag_evidence`
       capability;
     - event logging assertion: move to the retained helper/capability path.
   - Do not keep a test that proves the disabled legacy graph path.

4. Parent updates docs.
   - Remove `COGNITION_RESOLVER_ENABLED` from sample env and text in
     `docs/HOWTO.md`.
   - Update `README.md` architecture overview so persona turn describes
     resolver-driven cognition and demand-driven RAG.
   - Update `README_CN.md` as a native Chinese document, not a literal
     sentence-by-sentence translation. It must describe the same resolver-only
     architecture and demand-driven RAG baseline in idiomatic Chinese.
   - Update `src/kazusa_ai_chatbot/nodes/README.md` and
     `src/kazusa_ai_chatbot/rag/README.md` so `stage_1_research` is not the
     normal live path.

5. Parent runs focused deterministic verification.
   - `venv\Scripts\python -m py_compile` over changed production and test files.
   - Focused pytest set listed in `Verification`.
   - `rg` checks proving `COGNITION_RESOLVER_ENABLED` and `stage_1_research`
     are absent from production/docs except archived superseded plans or
     historical test artifacts.

6. Parent runs real LLM validation.
   - Re-run existing resolver real DB comparison and self-cognition L2d
     diagnostics.
   - Inspect and summarize each case. Do not accept raw JSON alone as evidence.

7. Parent runs independent code review gate.
   - Review full diff against this plan, prompt rules, local LLM constraints,
     and deletion risk.
   - Fix findings within the approved change surface and rerun affected tests.

7.5. Parent runs extensive post-review regression.
   - Run the full deterministic pytest suite.
   - Run compile/import checks for changed production modules.
   - Re-run resolver-focused tests if full regression required any fix.
   - Record a merge-readiness report that lists all commands, failures, fixes,
     skipped tests, and residual risks.
   - Do not merge to `main` from this step.

8. Parent performs the explicit user-gated merge to `main`.
   - Do not start this step until the user explicitly instructs the agent to
     merge to `main` after reviewing the post-review regression evidence.
   - Confirm clean worktree.
   - Merge latest `main` into `resolver-goal-poc` if needed.
   - Rerun focused verification after conflict resolution.
   - Merge `resolver-goal-poc` into `main`.
   - Record merge commit and final verification in this plan.

## Execution Model

Execution must use parent-led native subagent execution:

- Parent owns test contract, docs, verification, execution evidence, lifecycle
  updates, merge coordination, and final sign-off.
- One production-code subagent owns production-code edits in
  `config.py` and `persona_supervisor2.py`.
- Parent owns test and docs cleanup while or after production-code edits.
- One independent code-review subagent reviews the full implementation diff.

If native subagent execution is unavailable, stop before production-code edits
unless the user explicitly approves fallback execution for this plan.

## Progress Checklist

- [x] Stage 1 - test contract established.
  - Covers implementation order step 1.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py -q`
  - Evidence: record expected failures before code edits in `Execution
    Evidence`.
  - Sign-off: parent/2026-06-01 after expected-failure evidence was recorded.

- [x] Stage 2 - resolver-only runtime graph implemented.
  - Covers implementation order step 2.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py -q`
  - Evidence: changed production files and pass/fail output.
  - Sign-off: parent/2026-06-01 after verification and evidence are recorded.

- [x] Stage 3 - legacy tests rewritten or deleted by ownership boundary.
  - Covers implementation order step 3.
  - Verify focused resolver/RAG/persona tests listed in `Verification`.
  - Evidence: list deleted tests, rewritten tests, retained coverage, and test
    output.
  - Sign-off: parent/2026-06-01 after verification and evidence are recorded.

- [x] Stage 4 - docs describe resolver-only baseline.
  - Covers implementation order step 4.
  - Verify:
    `rg -n "COGNITION_RESOLVER_ENABLED|stage_1_research|defaults to false|resolver enabled" README.md README_CN.md docs src\kazusa_ai_chatbot`
    returns no live-doc stale optional-resolver wording.
  - Evidence: grep output, changed doc list, and a short Chinese-language
    quality note confirming `README_CN.md` reads natively rather than like a
    literal translation.
  - Sign-off: parent/2026-06-01 after verification and evidence are recorded.

- [x] Stage 5 - deterministic verification passed.
  - Covers implementation order step 5.
  - Verify all commands in `Verification`.
  - Evidence: command output and any skipped-test rationale.
  - Sign-off: parent/2026-06-01 after verification and evidence are
    recorded.

- [x] Stage 6 - real LLM validation reviewed.
  - Covers implementation order step 6.
  - Verify diagnostics produce human-readable review artifacts.
  - Evidence: artifact paths, case summaries, pass/fail judgment, residual
    risks.
  - Sign-off: parent/2026-06-01 after verification and evidence are
    recorded.

- [x] Stage 7 - independent code review complete.
  - Covers implementation order step 7.
  - Verify review findings are recorded, fixed or explicitly accepted, and
    affected tests rerun.
  - Evidence: review summary and rerun commands.
  - Sign-off: parent/2026-06-01 after review evidence is recorded.

- [x] Stage 7.5 - extensive post-review regression complete.
  - Covers implementation order step 7.5.
  - Verify full deterministic regression and compile/import checks listed in
    `Verification`.
  - Evidence: merge-readiness report with command output, failures, fixes,
    skipped tests, and residual risks.
  - Sign-off: parent/2026-06-01 after regression evidence is recorded.

- [x] Stage 7.6 - final baseline-review blocker remediation complete.
  - Covers post-review blocker cleanup requested before plan closure.
  - Verify dry-run path removal, self-cognition resolver-only retrieval,
    no-op self-cognition pending hooks, WebAgent3 deterministic URL fallback
    removal, removal of deterministic L2d misplaced-field semantic repair,
    stale-reference scan, and full deterministic regression.
  - Evidence: remediation section in `Execution Evidence`.
  - Sign-off: parent/2026-06-01 after focused and full verification passed.

- [x] Stage 8 - branch merged to main after explicit user instruction.
  - Covers implementation order step 8.
  - Gate: do not execute until the user explicitly instructs merge to `main`
    after Stage 7.6 evidence is available.
  - Verify clean worktree and final post-merge focused verification.
  - Evidence: merge commit, final commands, branch status.
  - Sign-off: parent/2026-06-01 after merge evidence is recorded.

## Verification

Static checks:

```powershell
venv\Scripts\python -m py_compile `
  src\kazusa_ai_chatbot\config.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2.py
```

Focused deterministic tests:

```powershell
venv\Scripts\python -m pytest `
  tests\test_config.py `
  tests\test_cognition_resolver_persona_graph.py `
  tests\test_cognition_resolver_loop.py `
  tests\test_cognition_resolver_contracts.py `
  tests\test_cognition_resolver_l2d_contract.py `
  tests\test_rag_cognitive_episode_adapter.py `
  tests\test_persona_supervisor2_action_initializer.py `
  tests\test_cognition_prompt_contract_text.py `
  -q
```

Legacy-reference checks:

```powershell
rg -n "COGNITION_RESOLVER_ENABLED" src tests README.md README_CN.md docs
rg -n "stage_1_research" src\kazusa_ai_chatbot README.md README_CN.md docs
```

Expected result: no live production or live documentation references remain.
Historical references inside `development_plans/archive/`, old evidence files,
or intentionally retained test artifact text are acceptable only when excluded
from live checks.

Extensive post-review regression before merge-readiness:

```powershell
venv\Scripts\python -m compileall -q src tests
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest `
  tests\test_persona_supervisor2.py `
  tests\test_persona_supervisor2_rag2_integration.py `
  tests\test_dialog_agent.py `
  tests\test_l2d_l3_surface_handoff.py `
  tests\test_rag_projection.py `
  tests\test_web_agent3.py `
  -q
```

If full regression is blocked by an external dependency, database availability,
or a pre-existing flaky live test, record the exact failing test, error, and
reason. Stage 8 still requires explicit user instruction and must not proceed
unless the user accepts any residual regression risk.

Real LLM validation:

```powershell
venv\Scripts\python test_artifacts\cognition_resolver\real_db_comparison_20260601\run_real_db_comparison.py
venv\Scripts\python test_artifacts\cognition_resolver\real_db_comparison_20260601\diagnose_l2d_self_cognition.py
```

Run real LLM cases one at a time when diagnosing failures. Inspect raw LLM
outputs and write a human-readable review artifact before claiming pass.

## Independent Plan Review

Review performed before execution on 2026-06-01.

Findings and fixes:

- Finding: `README_CN.md` was not in the documentation change surface, so the
  Chinese README could remain stale after the English README changed.
  Fix: added `README_CN.md` to `Must Do`, `Change Surface`, implementation
  order, Stage 4 evidence, legacy-reference checks, and acceptance criteria.
- Finding: the previous Stage 8 wording allowed automatic merge to `main`.
  Fix: Stage 8 now requires explicit user instruction after post-review
  regression evidence is available.
- Finding: full regression was listed only as a generic pre-merge command and
  did not clearly occur after independent code review.
  Fix: added Stage 7.5 for extensive post-review regression and a
  merge-readiness report.
- Finding: an earlier cutover policy label used non-standard wording.
  Fix: the plan now uses only `bigbang`, `migration`, and `compatible`.

## Independent Code Review

The independent review must check:

- `COGNITION_RESOLVER_ENABLED` is fully removed from live production, tests, and
  docs.
- `persona_supervisor2(...)` has one live graph path and it always enters
  `stage_1_goal_resolver`.
- `stage_1_research(...)` and disabled-mode tests are removed.
- RAG2 capability helpers still support resolver-selected `rag_evidence`.
- Prompt edits, if any, are generic and not case-tuned lookup tables.
- No deterministic semantic rule was added to force RAG, speech, HIL, or
  no-speak.
- Tests were rewritten by ownership boundary rather than blindly deleted.
- Docs no longer describe resolver as optional or disabled by default.
- `README_CN.md` describes the resolver-only architecture in native Chinese and
  does not preserve stale mandatory RAG-first wording.
- Verification evidence supports merge to `main`.

Review findings must be fixed before merge unless the user explicitly accepts
the residual risk.

## Acceptance Criteria

- `COGNITION_RESOLVER_ENABLED` does not exist in live production code, live
  tests, or live docs.
- `persona_supervisor2(...)` always routes decontextualized input into
  `stage_1_goal_resolver`.
- No live graph branch calls mandatory pre-cognition `stage_1_research`.
- RAG still works through resolver-selected `rag_evidence`.
- HIL, approval, self-goal resolution, duplicate blocking, and max-cycle
  behavior still pass focused tests.
- Real LLM validation shows resolver-only behavior can:
  - answer simple inputs without unnecessary RAG;
  - call RAG when cognition needs evidence;
  - continue or block HIL coherently;
  - handle self-cognition with RAG or a reasonable no-RAG explanation.
- Documentation presents resolver as the baseline architecture.
- Full regression and independent review pass.
- Extensive post-review regression passes or residual risk is explicitly
  accepted by the user.
- `resolver-goal-poc` is merged into `main` only after explicit user
  instruction, with merge evidence recorded.

## Execution Evidence

### Stage 1 - Test Contract

- Changed files:
  - `tests/test_config.py`
  - `tests/test_cognition_resolver_persona_graph.py`
- Command:
  `venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py -q`
- Result: expected failure before production edits.
  - 41 passed, 4 failed.
  - `test_cognition_resolver_defaults_are_bounded` failed because
    `hasattr(config, 'COGNITION_RESOLVER_ENABLED')` still returned `True`.
  - `test_cognition_resolver_config_reads_remaining_environment` failed for
    the same exported flag while max-cycle and timeout env reads still worked.
  - `test_persona_graph_default_runs_goal_resolver_without_stage_1_rag` failed
    because the current graph called `legacy_rag` then `direct_cognition`
    instead of the resolver.
  - `test_persona_graph_exports_no_resolver_enable_or_stage_1_research` failed
    because the module still exports `COGNITION_RESOLVER_ENABLED`.

### Stage 2 - Resolver-Only Runtime Graph

- Production subagent: `019e808f-f81f-7ed1-968c-d3c95fdf9ecc` (`Anscombe`).
- Changed files:
  - `src/kazusa_ai_chatbot/config.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- Production changes:
  - Deleted `COGNITION_RESOLVER_ENABLED` from config.
  - Removed the import from `persona_supervisor2.py`.
  - Deleted `stage_1_research(...)` from the live persona module.
  - Rewired `persona_supervisor2(...)` to always use
    `stage_0_msg_decontexualizer -> stage_1_goal_resolver ->
    stage_2_memory_lifecycle`.
  - Preserved `run_rag_evidence_for_persona_state(...)` and the resolver
    `call_cognition_subgraph` handoff.
- Subagent commands:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`
    passed.
  - `venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py -q`
    passed: 45 passed in 4.45s.
  - `rg -n "COGNITION_RESOLVER_ENABLED|stage_1_research" src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`
    returned no matches.
  - `git diff --check -- ...` passed with only line-ending normalization
    warnings.
- Parent verification:
  - Reviewed production diff against Stage 2 scope.
  - Re-ran
    `venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py -q`.
  - Result: 45 passed in 4.43s.

### Stage 3 - Legacy Test Rewrite

- Changed files:
  - `tests/test_config.py`
  - `tests/test_cognition_resolver_persona_graph.py`
  - `tests/test_persona_supervisor2.py`
  - `tests/test_persona_supervisor2_rag2_integration.py`
  - `tests/test_persona_supervisor2_rag_skip_shape.py`
  - `tests/test_rag_dialog_event_logging.py`
  - `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  - `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`
  - `tests/test_l2d_l3_surface_handoff.py`
- Rewrite decisions:
  - Deleted disabled-path graph assertions.
  - Rewrote persona graph plumbing tests to patch
    `call_cognition_resolver_loop(...)`, preserving assertions about
    decontextualizer state, scoped history, selected text-surface routing,
    silence routing, consolidation snapshots, and cognitive episode handoff.
  - Moved RAG request/projection/skip-shape/event logging assertions from the
    deleted graph node to `run_rag_evidence_for_persona_state(...)`.
  - Kept direct cognition-subgraph tests where they test the cognition
    subgraph itself rather than the live persona graph entry path.
  - Removed literal deleted flag/node names from live tests; absence tests build
    the old symbol names from neutral string parts so static grep remains a
    useful decommission gate.
- Command:
  `venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py tests\test_persona_supervisor2.py tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py tests\test_rag_dialog_event_logging.py tests\test_multi_source_cognition_stage_00_regression_baseline.py tests\test_multi_source_cognition_stage_02_chat_episode_migration.py tests\test_l2d_l3_surface_handoff.py -q`
- Result: 114 passed in 8.00s.

### Stage 4 - Resolver-Only Documentation

- Changed files:
  - `README.md`
  - `README_CN.md`
  - `docs/HOWTO.md`
  - `src/kazusa_ai_chatbot/nodes/README.md`
  - `src/kazusa_ai_chatbot/rag/README.md`
- Documentation changes:
  - Removed the resolver enable flag from the HOWTO sample env and runtime
    description.
  - Rewrote the architecture overview so the live persona turn enters the
    cognition resolver after decontextualization.
  - Described RAG 2 as a demand-driven resolver capability selected by L2d,
    not a mandatory pre-cognition stage.
  - Updated node and RAG subsystem docs to point RAG coverage at
    `run_rag_evidence_for_persona_state(...)`.
  - Updated `README_CN.md` in native Chinese wording; it now uses phrasing such
    as `认知解析器`, `按需证据能力`, and `只有当认知选择需要时` instead of mechanically
    translating the English bullets.
- Command:
  `rg -n "COGNITION_RESOLVER_ENABLED|stage_1_research|defaults to false|resolver enabled" README.md README_CN.md docs src\kazusa_ai_chatbot`
- Result: no matches.

### Stage 5 - Deterministic Verification

- Static command:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`
- Static result: passed with exit code 0.
- Focused deterministic command:
  `venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_l2d_contract.py tests\test_rag_cognitive_episode_adapter.py tests\test_persona_supervisor2_action_initializer.py tests\test_cognition_prompt_contract_text.py -q`
- Focused deterministic result: 137 passed in 4.74s.
- Legacy-reference command:
  `rg -n "COGNITION_RESOLVER_ENABLED" src tests README.md README_CN.md docs`
- Legacy-reference result: no matches; exit code 1 is the expected `rg`
  no-match result.
- Legacy-node command:
  `rg -n "stage_1_research" src\kazusa_ai_chatbot README.md README_CN.md docs`
- Legacy-node result: no matches; exit code 1 is the expected `rg` no-match
  result.
- Skipped tests: none for this gate.

### Stage 6 - Real LLM Validation

- Main replay command:
  `venv\Scripts\python test_artifacts\cognition_resolver\real_db_comparison_20260601\run_real_db_comparison.py`
- Main replay result: exit code 0; 5 real DB replay cases completed with
  `status=ok`.
- Diagnostic command:
  `venv\Scripts\python test_artifacts\cognition_resolver\real_db_comparison_20260601\diagnose_l2d_self_cognition.py`
- Diagnostic result: exit code 0; R04 captured 3 L2d calls and R05 captured 3
  L2d calls.
- Harness check:
  `venv\Scripts\python -m py_compile test_artifacts\cognition_resolver\real_db_comparison_20260601\diagnose_l2d_self_cognition.py`
  passed after fixing local diagnostic trace collision in the ignored
  `test_artifacts` harness.
- Human-readable review artifact:
  `test_artifacts/cognition_resolver/real_db_comparison_20260601/real_db_side_by_side_review.md`
- Raw evidence artifacts:
  - `test_artifacts/cognition_resolver/real_db_comparison_20260601/real_db_comparison_results.json`
  - `test_artifacts/cognition_resolver/real_db_comparison_20260601/self_cognition_l2d_diagnostics.json`
  - `test_artifacts/cognition_resolver/real_db_comparison_20260601/R01_private_affection_resolver_trace.md`
  - `test_artifacts/cognition_resolver/real_db_comparison_20260601/R02_group_source_followup_resolver_trace.md`
  - `test_artifacts/cognition_resolver/real_db_comparison_20260601/R03_group_identity_challenge_resolver_trace.md`
  - `test_artifacts/cognition_resolver/real_db_comparison_20260601/R04_self_cognition_price_topic_resolver_trace.md`
  - `test_artifacts/cognition_resolver/real_db_comparison_20260601/R04_self_cognition_price_topic_diagnostic_resolver_trace.md`
  - `test_artifacts/cognition_resolver/real_db_comparison_20260601/R05_self_cognition_photo_topic_resolver_trace.md`
  - `test_artifacts/cognition_resolver/real_db_comparison_20260601/R05_self_cognition_photo_topic_diagnostic_resolver_trace.md`
- Case judgment:
  - R01 and R03 completed as one-cycle resolver paths with no unnecessary
    capability call; outputs remained in the expected character lane.
  - R02 selected `web_evidence`, returned through multiple resolver cycles,
    blocked a duplicate request, and produced a visible answer. Quality risk:
    final observations were timeout/duplicate failures, so partial web evidence
    was not retained as source-backed observation.
  - R04 selected `rag_evidence` for self-cognition and produced bounded trace
    evidence, but the answer drifted from the baseline useful price
    interjection; conversation-evidence projection treated useful nearby
    messages as unconfirmed.
  - R05 selected conservative no-speak/audit behavior with a coherent
    character reason to keep observing instead of forcing a public message.
- Residual risks:
  - `web_evidence` may exceed resolver capability timeout and lose partial
    evidence.
  - Conversation-evidence projection can be too conservative for self-cognition
    group context, especially when useful adjacency is present but not promoted
    to confirmed evidence.
  - L2d warnings appeared for invalid goal-progress shapes and pending
    resolution without active pending rows; they did not crash the run but
    should be inspected by the independent review gate.
- Pass judgment: mixed real-LLM pass for this cutover gate. The evidence proves
  resolver-only execution and bounded capability loops, with residual quality
  risks recorded for review. No prompt tuning or deterministic semantic
  override was added.

### Stage 7 - Independent Code Review

- Review subagent: `019e80ba-5c64-7ee3-b67d-f22b5f877f75` (`Beauvoir`).
- Review outcome before fix: not approved because
  `src/kazusa_ai_chatbot/rag/README.md` still said `RAG 2 runs before
  cognition in persona_supervisor2`.
- Fix:
  - Updated `src/kazusa_ai_chatbot/rag/README.md` so the integration section
    says RAG 2 is called from the cognition resolver only when L2d selects an
    evidence capability.
- Reviewer non-blocking notes:
  - A draft RAG3 plan still references `stage_1_research`; it is not live docs
    or active execution scope for this cutover.
  - Stage 6 remains a mixed real-LLM quality pass with recorded risks around
    `web_evidence` timeout, conservative self-cognition RAG projection, and
    L2d validation warnings.
  - Stage 7.5 full regression remains pending.
- Post-fix commands:
  - `rg -n "COGNITION_RESOLVER_ENABLED" src tests README.md README_CN.md docs`
    returned no matches; exit code 1 is expected for no matches.
  - `rg -n "stage_1_research" src\kazusa_ai_chatbot README.md README_CN.md docs`
    returned no matches; exit code 1 is expected for no matches.
  - `rg -n "RAG 2 runs before cognition|RAG runs before cognition|before cognition in `persona_supervisor2`|mandatory RAG-first|RAG-first|resolver enabled|defaults to false" README.md README_CN.md docs src\kazusa_ai_chatbot`
    returned no matches; exit code 1 is expected for no matches.
  - `git diff --check -- src\kazusa_ai_chatbot\rag\README.md development_plans\active\short_term\resolver_default_mainline_cutover_plan.md`
    passed with only CRLF normalization warnings.
- Stage 7 result: approved after the documented blocker was fixed. No prompt
  changes, deterministic semantic overrides, or compatibility fallback paths
  were added.

### Stage 7.5 - Extensive Post-Review Regression

- Diagnostic subagents:
  - `019e80c1-4563-7d63-8b7a-50daab6871e4` (`Einstein`) reviewed memory
    retrieval and user-memory evidence failures. Finding: stale tests and an
    adjacent prompt-projection id hygiene gap; no subagent edits.
  - `019e80c1-6e13-71b0-b13d-11622a4c943d` (`Noether`) reviewed prompt,
    global-growth, and prompt-fingerprint failures. Finding: stale fixtures,
    stale result shape, stale fingerprints, and one generic prompt boundary
    gap for promoted global growth; no subagent edits.
  - `019e80c1-975d-7870-aa2e-ff186236e4cf` (`Locke`) reviewed RAG
    continuation and initializer prompt failures. Finding: stale continuation
    shape and stale English-fragment assertions; no production changes needed
    for that group.
- First full regression command:
  `venv\Scripts\python -m pytest -q`
- First full regression result before cleanup:
  - 1814 passed, 23 failed, 269 deselected in 36.52s.
  - Failures were in stale graph/prompt fixture coverage and one adjacent
    projection hygiene area:
    `test_conversation_progress_cognition.py`,
    `test_global_character_growth_replay.py`,
    `test_memory_retrieval_tools.py`,
    `test_multi_source_cognition_stage_07_reflection_dry_run.py`,
    `test_multi_source_cognition_stage_08_internal_thought_dry_run.py`,
    `test_multi_source_cognition_stage_09_multimodal_input_sources.py`,
    `test_rag_continuation.py`, `test_rag_initializer_cache2.py`, and
    `test_user_memory_evidence_agent.py`.
- Cleanup changes:
  - Added id stripping for `platform_message_id`,
    `seed_conversation_row_id`, and `seed_platform_message_id` in
    `src/kazusa_ai_chatbot/rag/prompt_projection.py`, preserving internal
    stable ids while keeping raw/opaque ids out of LLM-facing projection.
  - Added a generic L2 prompt boundary for
    `promoted_reflection_context.promoted_global_growth`: use it only as
    global character-growth background, not as current-user fact, commitment,
    or chat evidence.
  - Added a generic L2c2 boundary-profile binding rule so relationship
    pressure calibrates distance and tone without manufacturing boundary
    violations or overriding current input.
  - Updated stale deterministic tests for current resolver/RAG result shapes,
    Chinese prompt contracts, local-time context fixtures, prompt fingerprints,
    and user-memory evidence summaries.
  - No lookup table, validation-case prompt tuning, deterministic semantic
    override, forced RAG, forced speech, or old-path compatibility fallback was
    added.
- Targeted failed-test rerun:
  `venv\Scripts\python -m pytest <23 failed tests from first run> -q`
- Targeted failed-test rerun result: 23 passed in 2.25s.
- Fresh compile/import command:
  `venv\Scripts\python -m compileall -q src tests`
- Fresh compile/import result: passed with exit code 0.
- Final full deterministic regression command:
  `venv\Scripts\python -m pytest -q`
- Final full deterministic regression result: 1837 passed, 269 deselected in
  40.69s.
- Resolver-focused post-fix command:
  `venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_persona_supervisor2_rag2_integration.py tests\test_dialog_agent.py tests\test_l2d_l3_surface_handoff.py tests\test_rag_projection.py tests\test_web_agent3.py -q`
- Resolver-focused post-fix result: 120 passed in 2.85s.
- Legacy-reference commands:
  - `rg -n "COGNITION_RESOLVER_ENABLED" src tests README.md README_CN.md docs`
  - `rg -n "stage_1_research" src\kazusa_ai_chatbot README.md README_CN.md docs`
- Legacy-reference result: both commands returned no matches; exit code 1 is
  expected for `rg` no-match.
- Diff hygiene command:
  `git diff --check`
- Diff hygiene result: passed with exit code 0; only Git CRLF-normalization
  warnings were emitted.
- Skipped tests: none beyond the suite's configured 269 deselected tests.
- Merge-readiness judgment: Stage 7.5 is complete for the branch. The branch is
  not merged to `main`; Stage 8 remains blocked until explicit user
  instruction.

### Main Baseline Comparison And Documentation Cleanup

- After Stage 7.5, the branch was compared against `main` using 20 recent
  conversation-derived cases: 15 group-chat cases and 5 private-chat cases.
- Human-readable comparison artifact:
  `test_artifacts/resolver_merge_eval_20260601/main_vs_resolver_comparison_20260601.md`
- Raw artifacts:
  - `test_artifacts/resolver_merge_eval_20260601/main_comparison_raw_results.json`
  - `test_artifacts/resolver_merge_eval_20260601/current_post_cleanup_full_raw_results.json`
- Observed result:
  - `main`: visible output 18/20, average duration 86.98 seconds, average
    visible output 63.1 characters.
  - `resolver-goal-poc`: visible output 16/20, average duration 68.03 seconds,
    average visible output 124.9 characters.
  - The resolver branch did not speak more often overall; when it spoke, output
    tended to be more complete because `resolver_goal_progress`, resolver
    observations, and final response requirements gave L3/dialog a clearer
    delivery contract.
- Quality criterion agreed during review: output passes when it makes sense as
  believable character behavior. It is a bug if an internal selected `speak`
  action produces no visible dialog.
- Documentation cleanup after this comparison:
  - `README.md` and `README_CN.md` now present only the current resolver-first
    architecture, with no optional-resolver or transition framing.
  - The `Architecture At A Glance` / `高层架构` sections were rewritten by an
    isolated subagent from source-code inspection and now use
    GitHub-renderable Mermaid diagrams.
  - A second isolated subagent reviewed the surrounding README sections for
    project-level consistency in both English and native Chinese.
  - `src/kazusa_ai_chatbot/self_cognition/README.md` now describes
    self-cognition as entering the bounded resolver, with RAG2 invoked only
    when L2d selects `rag_evidence`.
  - No production code changed during this documentation cleanup.

### Final Independent Baseline Review

- Review request:
  compare `resolver-goal-poc` at
  `103d7a0f1d0a41e086d42afd20d4a3ae4655690c` against `main` at
  `a311ae1f94a898d9fd62a53545310db75f9eed9a`.
- Reviewer: independent subagent `019e82c4-565b-7eb1-b3d9-0e993f1e680d`.
- Result: not closed; review found merge/closure blockers.
- Blocking findings:
  - `src/kazusa_ai_chatbot/self_cognition/runner.py` still has a
    deterministic pre-resolver RAG path for `RAG_BACKED_CASE_NAMES`; this
    conflicts with the requirement that self-cognition evidence lookup happen
    only when L2d selects `rag_evidence`.
  - Self-cognition dry-run/default resolver execution calls
    `call_cognition_resolver_loop(...)` without no-op pending hooks, so HIL or
    approval blocker selection can write durable pending rows during dry-run
    artifact generation.
- High/medium findings:
  - `src/kazusa_ai_chatbot/rag/web_agent3/agent.py` contains deterministic
    official-URL guessing and fallback reads in normal WebAgent3 execution.
    This pushes semantic source selection into deterministic code instead of a
    specialist agent/router contract.
  - The WebAgent3 official-URL fallback catches broad `Exception` inside
    non-boundary helper code, which can hide internal contract bugs as normal
    web-read failures.
- Review recommendation: do not merge or close this plan until the blockers
  are fixed or explicitly accepted as residual risk by the user.

### Final Review Remediation And Branch-Side Closure

- Remediation scope:
  - Removed the self-cognition file-output dry-run path completely:
    `src/scripts/run_self_cognition_dry_run.py`,
    `src/kazusa_ai_chatbot/self_cognition/artifacts.py`, and
    `tests/test_self_cognition_dry_run_cli.py`.
  - Removed `runner.run_self_cognition_case`,
    `runner.run_self_cognition_case_async`, dry-run artifact writing,
    dry-run event mirroring, worker output-root compatibility, and
    artifact-path worker result plumbing.
  - Removed deterministic pre-resolver self-cognition RAG. Source packets now
    enter cognition without preloaded RAG evidence; retrieval is counted only
    when the resolver records `rag_evidence` or `web_evidence` observations.
  - Added non-persistent self-cognition pending hooks so HIL/approval blocker
    selection can shape the in-memory resolver pass without writing durable
    pending ledger rows.
  - Removed WebAgent3 deterministic official-URL guessing and fallback reads.
    WebAgent3 still reads URLs selected by its router/subagents, but an empty
    search result is no longer converted by Python into guessed official URL
    reads.
  - Updated self-cognition, script, node, and root README wording so the
    current architecture is documented without the removed self-cognition
    dry-run artifact path.
- Independent review status after remediation:
  - The blocker on `RAG_BACKED_CASE_NAMES` is fixed; the symbol and
    deterministic self-cognition pre-RAG path are removed.
  - The blocker on self-cognition durable pending writes is fixed through
    non-persistent resolver pending hooks.
  - The WebAgent3 official-URL fallback and its broad exception path are
    removed.
  - Remaining `dry_run` references are unrelated maintenance/reflection
    paths, historical plan/test filenames, or the existing event-log schema
    field `dry_run=False` on production self-cognition worker telemetry.
- Supplemental final review:
  - Reviewer: independent subagent
    `019e82f0-a79b-72a3-a04e-307e12afc829` (`Linnaeus`).
  - Result: the dry-run/RAG/pending/WebAgent3 remediation checked out, but
    the review found one high-severity guardrail violation and one low-severity
    whitespace issue.
  - High finding: `persona_supervisor2_cognition_l2d.py` still performed
    deterministic semantic repair by moving misplaced resolver requests out of
    `action_requests` and misplaced terminal actions out of
    `resolver_capability_requests`. This violated the boundary that LLM stages
    own semantic action decisions while deterministic code validates contracts.
  - Low finding: the reference evidence report had a trailing blank line that
    failed `git diff --check main`.
  - Remediation: removed the L2d misplaced-field repair helpers and changed
    the corresponding tests to assert that contract-drift rows are dropped, not
    semantically reinterpreted by Python. Removed the trailing blank line from
    the evidence report.
- Verification commands:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\self_cognition\models.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\worker.py src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\reflection_cycle\worker.py`
    passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py`
    passed after removing deterministic misplaced-field repair.
  - `venv\Scripts\python -m pytest tests\test_persona_supervisor2_action_initializer.py::test_action_initializer_drops_misplaced_resolver_request tests\test_persona_supervisor2_action_initializer.py::test_action_initializer_drops_misplaced_terminal_action -q`
    passed: 2 passed in 1.83s.
  - `venv\Scripts\python -m pytest tests\test_persona_supervisor2_action_initializer.py tests\test_cognition_resolver_l2d_contract.py tests\test_l2d_l3_surface_handoff.py tests\test_cognition_resolver_loop.py tests\test_persona_supervisor2.py -q`
    passed: 81 passed in 2.52s.
  - `venv\Scripts\python -m pytest tests/test_self_cognition_tracking.py tests/test_self_cognition_framing.py tests/test_self_cognition_delivery_target.py tests/test_self_cognition_event_logging.py tests/test_self_cognition_integration.py -q`
    passed: 89 passed in 8.43s.
  - `venv\Scripts\python -m pytest tests/test_cognition_resolver_loop.py tests/test_rag_cognitive_episode_adapter.py tests/test_web_agent3.py tests/test_config.py tests/test_reflection_cycle_stage1c_service.py tests/test_reflection_cycle_stage1c_worker.py tests/test_runtime_adapter_registration.py tests/test_service_background_consolidation.py -q`
    passed: 227 passed in 12.20s.
  - `venv\Scripts\python -m pytest tests\test_internal_monologue_residue_prompt_boundaries.py::test_root_readmes_document_residue_architecture -q`
    passed after restoring the root README residue lane wording.
  - `venv\Scripts\python -m pytest -q` initially found one root README
    documentation-boundary failure; after the README fix the full deterministic
    suite passed: 1855 passed, 269 deselected in 30.49s.
  - Final full deterministic suite after supplemental review remediation:
    `venv\Scripts\python -m pytest -q` passed: 1855 passed, 269 deselected in
    31.34s.
  - `git diff --check main` and `git diff --check` passed; both emitted only
    LF/CRLF working-copy warnings.
  - Stale-reference scan for removed self-cognition dry-run symbols,
    deterministic self-cognition pre-RAG symbols, and WebAgent3 official-URL
    fallback symbols returned no matches in `src`, `tests`, root READMEs, or
    `docs`.
- Branch-side closure judgment:
  - The final independent-review blockers are remediated.
  - The plan is closed for branch-side implementation and verification.
- Stage 8 merge closure:
  - User explicitly instructed Stage 8 execution on 2026-06-01.
  - Committed final branch cleanup on `resolver-goal-poc`:
    `9e1767d Remove legacy self-cognition dry-run path`.
  - Fetched `origin`, switched to `main`, confirmed `main` was up to date
    with `origin/main`, then merged `resolver-goal-poc` with explicit merge
    commit `af2c3b3 Merge resolver-goal-poc`.
  - Post-merge compile command passed:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_resolver\loop.py src\kazusa_ai_chatbot\cognition_resolver\capabilities.py src\kazusa_ai_chatbot\cognition_resolver\contracts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\self_cognition\worker.py src\kazusa_ai_chatbot\rag\web_agent3\agent.py`.
  - Post-merge focused pytest command passed:
    `venv\Scripts\python -m pytest tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_l2d_contract.py tests\test_l2d_l3_surface_handoff.py tests\test_persona_supervisor2_action_initializer.py tests\test_persona_supervisor2.py tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_web_agent3.py -q`;
    result: 197 passed in 9.60s.
  - Stage 8 is complete locally on `main`.

## Risks

- Removing legacy tests may accidentally drop useful RAG projection coverage.
  Mitigation: move assertions to retained helper/capability tests before
  deleting old graph tests.
- Resolver-only path can increase latency when L2d selects evidence. Mitigation:
  keep max-cycle and capability-timeout controls.
- Local LLM may repeat semantically equivalent evidence requests. Mitigation:
  keep duplicate blocking and record real LLM traces; improve generic retry
  semantics only through a separate approved plan if needed.
- Main merge conflicts may arise if `main` changed persona graph or config.
  Mitigation: merge latest `main` into the branch before final verification.
