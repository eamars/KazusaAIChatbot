# resolver_default_mainline_cutover_plan

## Summary

- Goal: Make the cognition resolver the default and only live persona workflow
  on `resolver-goal-poc`, remove the legacy mandatory RAG-first path, and
  prepare a clean merge from this branch into `main`.
- Plan class: high_risk_migration
- Status: approved
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
`COGNITION_RESOLVER_ENABLED=false` by default. That is no longer the target.
The confirmed product decision is:

- the resolver is now the baseline architecture on this branch;
- the old mandatory RAG-first live persona path is not preserved;
- no runtime backward compatibility or feature-flag fallback is required;
- RAG remains available only as a cognition-selected resolver capability;
- all semantic action and evidence decisions remain inside L1 -> L2 -> L2d.

Current code still has the old compatibility split:

```text
if COGNITION_RESOLVER_ENABLED:
  stage_0_msg_decontexualizer -> stage_1_goal_resolver
else:
  stage_0_msg_decontexualizer -> stage_1_research -> stage_2_cognition
```

This plan removes that split and makes the live graph:

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

- [ ] Stage 1 - test contract established.
  - Covers implementation order step 1.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py -q`
  - Evidence: record expected failures before code edits in `Execution
    Evidence`.
  - Sign-off: parent/date after evidence is recorded.

- [ ] Stage 2 - resolver-only runtime graph implemented.
  - Covers implementation order step 2.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py -q`
  - Evidence: changed production files and pass/fail output.
  - Sign-off: parent/date after verification and evidence are recorded.

- [ ] Stage 3 - legacy tests rewritten or deleted by ownership boundary.
  - Covers implementation order step 3.
  - Verify focused resolver/RAG/persona tests listed in `Verification`.
  - Evidence: list deleted tests, rewritten tests, retained coverage, and test
    output.
  - Sign-off: parent/date after verification and evidence are recorded.

- [ ] Stage 4 - docs describe resolver-only baseline.
  - Covers implementation order step 4.
  - Verify:
    `rg -n "COGNITION_RESOLVER_ENABLED|stage_1_research|defaults to false|resolver enabled" README.md README_CN.md docs src\kazusa_ai_chatbot`
    returns no live-doc stale optional-resolver wording.
  - Evidence: grep output, changed doc list, and a short Chinese-language
    quality note confirming `README_CN.md` reads natively rather than like a
    literal translation.
  - Sign-off: parent/date after verification and evidence are recorded.

- [ ] Stage 5 - deterministic verification passed.
  - Covers implementation order step 5.
  - Verify all commands in `Verification`.
  - Evidence: command output and any skipped-test rationale.
  - Sign-off: parent/date after verification and evidence are recorded.

- [ ] Stage 6 - real LLM validation reviewed.
  - Covers implementation order step 6.
  - Verify diagnostics produce human-readable review artifacts.
  - Evidence: artifact paths, case summaries, pass/fail judgment, residual
    risks.
  - Sign-off: parent/date after verification and evidence are recorded.

- [ ] Stage 7 - independent code review complete.
  - Covers implementation order step 7.
  - Verify review findings are recorded, fixed or explicitly accepted, and
    affected tests rerun.
  - Evidence: review summary and rerun commands.
  - Sign-off: parent/date after review evidence is recorded.

- [ ] Stage 7.5 - extensive post-review regression complete.
  - Covers implementation order step 7.5.
  - Verify full deterministic regression and compile/import checks listed in
    `Verification`.
  - Evidence: merge-readiness report with command output, failures, fixes,
    skipped tests, and residual risks.
  - Sign-off: parent/date after regression evidence is recorded.

- [ ] Stage 8 - branch merged to main after explicit user instruction.
  - Covers implementation order step 8.
  - Gate: do not execute until the user explicitly instructs merge to `main`
    after Stage 7.5 evidence is available.
  - Verify clean worktree and final post-merge focused verification.
  - Evidence: merge commit, final commands, branch status.
  - Sign-off: parent/date after merge evidence is recorded.

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

Pending. Execution agents must record commands, outputs, changed files, real
LLM artifact paths, review findings, fixes, merge hash, and residual risks here.

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
