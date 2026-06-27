# coding agent phase2 code writing plan

## Summary

- Goal: implement Phase 2 as new-artifact `code_writing`: create new files,
  scripts, tests, docs, or small projects from scratch, then materialize them
  through the `code_patching` artifact boundary without mutating the caller
  workspace.
- Plan class: high_risk_migration. This replaces the previous Phase 2
  existing-repository modification attempt with a narrower new-artifact scope
  aligned to the coding-agent architecture.
- Status: in_progress
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `debug-llm`.
- Overall cutover strategy: bigbang inside Phase 2. Production code, prompts,
  tests, fixtures, and docs move to one new-artifact role contract with no
  compatibility layer for stale Phase 2 role shapes.
- Highest-risk areas: local-LLM overloading, vague artifact contracts,
  premature existing-source modification, weak role-level live LLM evidence,
  patch/file-tree assembly errors, unsafe workspace mutation, and context
  overflow under the 50k project cap.
- Acceptance criteria: every Phase 2 LLM role has a live LLM diagnostic suite
  generated from the five new-artifact hard gates; each role suite is run one
  case at a time and reviewed before any E2E attempt; unresolved role
  weaknesses block E2E; after role weaknesses are resolved, all five hard
  Phase 2 new-artifact gates pass by agent-authored review. The known
  validation-time generated-code execution issue is recorded as Phase 2.5
  security scope and does not block Phase 2 artifact-quality signoff.

## Context

The previous Phase 2 attempt combined new-code generation and existing-source
modification. Live E2E failures showed that this overloaded the local LLM with
source ownership, lifecycle preservation, current-file grounding, new-file
composition, patch materialization, and validation repair in one phase.

The architecture now separates the capabilities:

- `code_reading`: understand existing source.
- `code_writing`: create new artifacts from scratch.
- `code_modifying`: plan semantic changes to existing source files.
- `code_patching`: materialize selected artifacts into a patch or file tree.

The top-level coding supervisor owns the work ledger and can later interleave
reading, writing, modifying, patching, external evidence, and execution. Phase
2 implements only the new-artifact side of that architecture, plus the patching
boundary needed to return proposed artifacts. Existing-source modification is
assigned to a later `code_modifying` plan.

Phase 2 returns proposed artifacts only. It does not apply patches, run target
project commands, mutate real checkouts, or perform code execution.

The current validation-time generated-code execution issue is excluded from
Phase 2 pass/fail. It must be recorded as a Phase 2.5 security finding and
must not be used as permission to add new execution, mutation, or command
capabilities inside Phase 2.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing role boundaries, prompts,
  model calls, context packets, supervisor loops, or evaluator ownership.
- `no-prepost-user-input`: load before changing request interpretation,
  hard-gate loading, or any code that could keyword-route user input.
- `py-style`: load before editing Python production files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running live LLM cases or writing live LLM review
  artifacts.

## Mandatory Rules

- After automatic context compaction, the active agent must reread this entire
  plan before continuing implementation, verification, handoff, lifecycle
  updates, or final reporting.
- After signing off any major checklist stage, the active agent must reread
  this entire plan before starting the next stage.
- The user has explicitly prohibited Codex execution subagents for the current
  Phase 2 work. Execute this plan single-agent unless the user later changes
  that instruction.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the code-review gate defined by this plan and record the result in
  `Execution Evidence`.
- Real LLM tests are the primary quality gates. Deterministic tests support
  schema, parsing, filesystem safety, artifact validation, and wiring only.
- No E2E hard-gate run may start until every LLM role suite generated from all
  five hard gates has run one case at a time and produced an agent-authored
  review that identifies role weaknesses, pass behavior, and unresolved
  blockers.
- E2E remains blocked while any role-level live LLM review has an unresolved
  blocker weakness. Fix the smallest affected role contract first and rerun
  only the affected role cases before reassessing E2E readiness.
- Do not batch live LLM tests. Run one live LLM case, inspect its trace, write
  or update the review artifact, then decide the next run.
- Do not encode hard-gate keywords, repository names, expected files, expected
  functions, or expected answers in production code, runtime prompts,
  deterministic routing, deterministic pass/fail logic, or non-live tests.
- Hard-gate challenge text may appear only in the supporting gate document,
  allowed live-gate fixtures, raw traces, and agent-authored review artifacts.
- LLM stages own semantic judgment. Deterministic code owns validation,
  persistence, limits, path safety, prompt budget caps, patch parsing,
  sandboxing, public-output redaction, and file mechanics.
- Deterministic evaluators may reject missing required fields, invalid Python
  signatures, path leaks, prompt-budget overflow, unsafe paths, malformed
  artifacts, and impossible handoff shapes. They must not use user-input
  keyword matching to decide semantic pass/fail.
- The top-level coding supervisor owns cross-domain dispatch and loop memory.
  `code_writing` may return `need_external_evidence`; the supervisor performs
  that call and resumes writing with bounded evidence.
- `code_writing` must not modify existing source files in Phase 2. Requests
  that require existing-source semantic edits must be reported as requiring
  `code_modifying`.
- Programmers never receive peer programmer output. A programmer receives one
  local new-artifact contract and returns one implementation artifact for that
  contract.
- The replacement design must stay within the 50k context cap. Each role input
  must have a prompt-budget report and must fail closed before invoking the LLM
  when over the hard cap.

## Must Do

- Update the architecture, code-writing ICD, production contracts, prompts,
  fixtures, and tests to make Phase 2 `code_writing` a new-artifact capability.
- Keep the top-level supervisor as the only owner of interleaving. Workers
  return needs and artifacts; workers do not call other workers directly.
- Implement or keep a `code_patching` boundary sufficient to materialize
  new-file artifacts as a file tree or patch proposal.
- Remove stale Phase 2 role shapes that assign existing-source modification to
  `code_writing`.
- Define a Writing PM contract that emits new-artifact work items with purpose,
  file kind, imports, provided interfaces, consumed interfaces, required
  behavior, and artifact format.
- Define a Writing programmer contract that accepts exactly one new-artifact
  work item and returns exactly one fenced code or text artifact.
- Define a Patching worker contract that accepts selected new artifacts plus
  File Agent path reservations and returns one artifact package.
- Define an acceptance owner that preserves user-visible requirements before
  artifact decomposition.
- Define an alignment owner that compares generated artifacts against the
  preserved requirements before synthesis and feeds blocker feedback back to
  the Writing PM.
- Generate full role-level live LLM suites from every Phase 2 hard gate before
  any E2E run.
- Run role-level live LLM suites first, one case at a time, and write
  agent-authored review artifacts that highlight weaknesses.
- Block E2E until all role-level live LLM blocker weaknesses are resolved or
  explicitly accepted by the user.
- Preserve real-workspace immutability: Phase 2 proposes artifacts only.
- Preserve anti-cheat guardrails and run static checks before signoff.

## Deferred

- Do not implement `code_modifying` or existing-source semantic edits in Phase
  2.
- Do not implement Phase 5 code execution, command execution, package install,
  Docker execution, or target-project test execution.
- Do not apply patches to the real workspace or fetched checkout.
- Do not connect Phase 2 to Kazusa runtime, background work, L2d, action spec,
  adapters, dispatcher, scheduler, persistence, or consolidation.
- Do not add compatibility shims for old Phase 2 role names, old role
  contracts, or old programmer input shapes.
- Do not run E2E to validate small prompt or contract edits while role-level
  confidence remains below 90%.
- Do not remediate the known generated-code execution security gap in Phase 2;
  record it for Phase 2.5 and continue judging Phase 2 gates on
  new-artifact workflow and artifact quality.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Active Phase 2 plan | bigbang | Use this new-artifact plan as the only active Phase 2 execution contract. |
| Role contracts | bigbang | Replace stale existing-source modification contracts with new-artifact writing contracts. |
| Prompts | bigbang | Rewrite prompt wording to match the new-artifact role contracts. |
| Tests | bigbang | Replace stale role fixtures and E2E gates with new-artifact live LLM suites and bounded deterministic support checks. |
| E2E gates | gated | E2E runs only after role-level live LLM suites are reviewed and unresolved blockers are closed. |

## Cutover Policy Enforcement

- The active agent must follow the selected policy for each area.
- For bigbang areas, delete or rewrite stale references instead of preserving
  old call shapes.
- Any compatibility layer, alias module, fallback mapper, or old-role
  translation bridge requires explicit user approval before implementation.

## Target State

The Phase 2 direct flow is:

```text
Coding supervisor
-> optional Phase 0 fetch to establish source identity or workspace safety
-> acceptance owner preserves user-visible requirements
-> code_writing PM creates new-artifact plan
-> File Agent reserves and validates new artifact paths
-> handoff evaluator validates each new-artifact contract
-> Writing programmer writes one artifact per accepted contract
-> Writing PM reconciles generated artifacts
-> code_patching materializes selected artifacts
-> Structural validator checks artifact package
-> alignment owner checks generated artifacts against preserved requirements
-> local repair loop or synthesis
-> CodingPatchProposalResponse
```

The top-level supervisor owns the work ledger. The Writing PM owns the
new-artifact feature picture. The acceptance and alignment owners preserve and
check user-visible requirements. Each Writing programmer owns only the
implementation behind one accepted new-artifact contract. The Patching worker
owns file-tree or patch materialization.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Phase 2 capability | `code_writing` creates new artifacts only | This isolates the part local LLMs can learn first. |
| Existing-source edits | Assign to later `code_modifying` | Existing lifecycle preservation caused repeated Phase 2 failure. |
| Interleaving | Top-level supervisor owns the work ledger | Mixed future requests need reading, writing, modifying, patching, and evidence in one controlled loop. |
| Patching | Keep a separate `code_patching` boundary | Artifact materialization is file mechanics, not feature semantics. |
| Programmer context | One new-artifact contract per programmer | Prevents peer-output dependency and context overload. |
| Role testing | Derive every LLM role suite from every hard gate | Role weakness must be visible before expensive E2E attempts. |
| E2E policy | Block E2E until role suites are reviewed and blockers resolved | Prevents E2E thrashing as component validation. |

## Contracts And Data Shapes

### Supervisor Work Item

```python
{
    "work_item_id": str,
    "kind": "new_artifact",
    "user_goal": str,
    "evidence_summary": list[dict],
    "external_evidence": list[dict],
    "session_summary": dict,
}
```

### Writing PM Decision

```python
{
    "status": "need_programmers | need_external_evidence | sufficient | rejected",
    "feature_goal": str,
    "artifact_items": [
        {
            "artifact_id": str,
            "file_label": str,
            "file_kind": "source" | "test" | "docs" | "config" | "data",
            "content_format": "python" | "markdown" | "text" | "json" | "csv",
            "purpose": str,
            "imports": list[str],
            "provided_interfaces": list[dict],
            "consumed_interfaces": list[dict],
            "required_behavior": list[str],
        }
    ],
    "selected_artifacts": list[dict],
    "external_evidence_requests": list[dict],
    "limitations": list[str],
}
```

### Writing Programmer Contract

```python
{
    "artifact_id": str,
    "file_label": str,
    "file_kind": "source" | "test" | "docs" | "config" | "data",
    "content_format": "python" | "markdown" | "text" | "json" | "csv",
    "purpose": str,
    "imports": list[str],
    "provided_interfaces": list[dict],
    "consumed_interfaces": list[dict],
    "required_behavior": list[str],
}
```

The Writing programmer returns exactly one fenced code block or text block
containing the requested artifact. It does not return JSON, diffs, file paths,
patch anchors, command results, or peer-output commentary.

### Patching Input

```python
{
    "artifact_package_id": str,
    "artifacts": list[dict],
    "reserved_paths": list[dict],
    "max_artifact_chars": int,
}
```

### Patching Output

```python
{
    "status": "succeeded | failed | rejected",
    "artifact_package": dict,
    "created_files": list[str],
    "changed_files": list[str],
    "diagnostics": list[str],
}
```

For Phase 2, `changed_files` is empty unless the path describes generated
artifact metadata inside the managed proposal package.

### AI Review Packet And Phase 3 Feedback Shape

Each hard gate produces a review packet. Phase 2 records this feedback; Phase
3 may attach the same shape to background-work integration and result-ready
delivery.

```python
{
    "gate_id": str,
    "difficulty": "easy | low_medium | medium | medium_hard | hard",
    "input_request": str,
    "workflow_trace": {
        "supervisor_steps": list[dict],
        "worker_calls": list[dict],
        "role_outputs": list[dict],
        "validation_summaries": list[dict],
        "loop_limit_reached": bool,
    },
    "artifact_manifest": {
        "created_files": list[str],
        "changed_files": list[str],
        "artifact_count": int,
        "artifact_package_path": str,
    },
    "ai_review": {
        "status": "pass | fail",
        "pass_confidence": "high | medium | low",
        "workflow_correct": bool,
        "request_satisfied": bool,
        "artifact_completeness": "complete | partial | missing",
        "phase3_feedback": list[dict],
        "blocking_failures": list[str],
        "acceptable_variations": list[str],
    },
}
```

`phase3_feedback` is the planned feedback bridge for later integration. Each
entry names the failing stage and the kind of repair the supervisor should
schedule:

```python
{
    "stage": "supervisor | writing_pm | writing_programmer | patching_worker | structural_validator | synthesizer",
    "failure_type": "wrong_workflow | missing_artifact | incomplete_behavior | inconsistent_interfaces | unsafe_output | loop_limit | unclear_final_answer",
    "feedback": str,
    "suggested_next_step": "rerun_same_stage | ask_for_external_evidence | repair_artifact_contract | repair_generated_artifact | repair_patch_package | fail_request",
}
```

## LLM Call And Context Budget

Assume a 50k-token project cap. Use conservative character estimates when an
exact tokenizer is unavailable.

| Role | Route | Context inputs | Cap rule |
|---|---|---|---|
| Top-level coding supervisor | `CODING_AGENT_PM_LLM` | user goal, work ledger, compact evidence state, validation summary | fail closed before invoke if over hard cap |
| Writing PM | `CODING_AGENT_PM_LLM` | one new-artifact work item, external evidence, validation feedback | owns new-artifact decomposition |
| Acceptance owner | `CODING_AGENT_PM_LLM` | original user request | preserves user-visible requirements before decomposition |
| Alignment owner | `CODING_AGENT_PM_LLM` | original user request, preserved criteria, PM decision, generated artifacts, structural validation | checks final artifact/request alignment |
| Writing programmer | `CODING_AGENT_PROGRAMMER_LLM` | one accepted new-artifact contract | no peer output, no repository source dump |
| Patching worker | `CODING_AGENT_PROGRAMMER_LLM` | selected generated artifacts, reserved paths, artifact caps | no feature semantics beyond assembly |
| Synthesizer | `CODING_AGENT_PM_LLM` | selected artifacts, validation summary, public-safe limitations | no code repair |

Role-suite live LLM tests must record model route, model name, thinking state,
prompt version, input size, and output size.

## Change Surface

### Modify

- `development_plans/reference/designs/coding_agent_architecture.md`: align the
  reference architecture with supervisor-owned interleaving and the split
  between `code_writing`, `code_modifying`, and `code_patching`.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/README.md`: update the ICD
  so Phase 2 writing means new-artifact generation.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/models.py`: replace stale
  existing-source module contracts with new-artifact contracts.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/product_manager.py`: make
  the Writing PM emit `WritingPMDecision.artifact_items`.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/acceptance.py`: preserve
  acceptance criteria and judge generated artifact alignment through PM-route
  LLM calls.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/programmer.py`: align prompt
  and parser to the one-artifact contract.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/supervisor.py`: wire the
  new-artifact writing flow, File Agent reservation, patching boundary,
  validation, and bounded repair loop.
- `tests/test_coding_agent_phase2_new_artifact_contracts.py`: focused
  deterministic support checks for Writing PM parsing, one-artifact programmer
  parsing, new-file artifact package materialization, public redaction, and
  existing-source edit rejection.
- `tests/test_coding_agent_phase2_new_artifact_role_live_llm.py`: one
  real-LLM role case per test function for Writing PM, Writing programmer,
  Patching worker, and Synthesizer role suites. Supervisor chaining is reserved
  for E2E.
- `tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py`: one real-LLM
  E2E gate per test function through `coding_agent.propose_code_change(...)`.
- `test_artifacts/live_gate/`: replace stale existing-repo Phase 2 hard gates
  with the five new-artifact hard gates from
  `development_plans/reference/designs/coding_agent_phase2_new_artifact_gating_tests.md`.
- `test_artifacts/llm_reviews/`: write new live LLM role and E2E review
  artifacts.

### Keep

- Phase 0 fetching public contract.
- Phase 1 reading public contract.
- Existing local LLM route names:
  `CODING_AGENT_PM_LLM` and `CODING_AGENT_PROGRAMMER_LLM`.
- Real-workspace immutability.

### Deferred Change Surface

- `code_modifying` production package and prompts.
- Existing-source owner selection.
- Existing-file patch anchor selection beyond the generic patching boundary.
- Target-project command execution.
- Background-worker integration.

## Overdesign Guardrail

- Actual problem: Phase 2 failed because it asked local LLMs to create new code
  and safely modify complex existing repositories in the same phase.
- Minimal change: narrow Phase 2 to new-artifact writing and keep supervisor
  interleaving in the architecture for later mixed work.
- Ownership boundaries: supervisor owns the work ledger; Writing PM owns
  new-artifact decomposition; Writing programmer owns one artifact body;
  Patching worker owns materialization; deterministic code owns validation and
  path safety.
- Rejected complexity: existing-source semantic edits, source-owner PM,
  lifecycle-preserving modifications, execution feedback, patch apply, hidden
  compatibility mappers, and direct worker-to-worker calls.
- Evidence threshold: add `code_modifying` only after Phase 2 new-artifact
  gates pass and a later plan defines role-level live LLM tests for
  existing-source change contracts.

## Agent Autonomy Boundaries

- The active agent may choose local implementation mechanics only when they
  preserve the contracts in this plan.
- The active agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, or extra features.
- The active agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors unless explicitly
  listed in Must Do or Change Surface.
- If the plan and code disagree, the active agent must preserve the plan's
  stated intent and report the discrepancy.
- If a required instruction is impossible, the active agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

1. Reread this plan, the architecture reference, and the code-writing ICD.
2. Update the code-writing ICD and production data contracts to the
   new-artifact shapes.
3. Update Writing PM prompt and parser.
4. Update Writing programmer prompt and parser.
5. Update File Agent and patching boundary only as needed for new-file
   artifact reservation and materialization.
6. Update supervisor wiring so `code_writing` handles new artifacts and returns
   existing-source edit requests as requiring `code_modifying`.
7. Replace stale deterministic support tests.
8. Build role-level live LLM fixtures from the five hard gates.
9. Run role-level live LLM tests one case at a time and record reviews.
10. Run E2E hard gates only after role-level confidence is at least 90%.
11. Run static checks, deterministic support checks, anti-cheat greps, and
    independent code review.

## Execution Model

- Execute this plan single-agent because the user explicitly prohibited Codex
  execution subagents for the current Phase 2 work.
- The active agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- The active agent establishes focused deterministic support checks and
  role-level live LLM fixtures before production implementation.
- The active agent runs live LLM tests one case at a time and inspects output
  before the next live call.
- The active agent performs independent code review from a fresh-review
  posture after verification passes, unless the user later approves a separate
  reviewer.

## Progress Checklist

- [x] Stage 1 - architecture and plan alignment
  - Covers: architecture reference, this plan, registry consistency.
  - Verify: static grep for stale role wording in architecture and plan.
  - Evidence: record changed docs and grep results in `Execution Evidence`.
  - Sign-off: after verification and evidence are recorded.
- [x] Stage 2 - new-artifact contracts and ICD
  - Covers: `code_writing/README.md`, models, PM contract, programmer contract,
    patching boundary contract.
  - Verify: `venv\Scripts\python -m pytest -q tests/test_coding_agent_phase2_new_artifact_contracts.py`
    and `venv\Scripts\python -m compileall -q src/kazusa_ai_chatbot/coding_agent`.
  - Evidence: record changed files and test output.
  - Sign-off: after verification and evidence are recorded.
- [x] Stage 3 - prompt and supervisor wiring
  - Covers: Writing PM prompt, Writing programmer prompt, supervisor work flow,
    File Agent reservation, patching boundary.
  - Verify: rerun `venv\Scripts\python -m pytest -q tests/test_coding_agent_phase2_new_artifact_contracts.py`.
  - Evidence: record commands and outputs.
  - Sign-off: after verification and evidence are recorded.
- [x] Stage 4 - role-level live LLM suite
  - Covers: five hard-gate-derived role suites for Writing PM, Writing
    programmer, Patching worker, and Synthesizer.
  - Boundary: supervisor orchestration is not a role-level live LLM case
    because it chains multiple roles. Supervisor handoff remains covered by
    deterministic support checks and Stage 5 E2E.
  - Verify: run one live case at a time and write agent-authored review rows.
  - Evidence: raw traces and review artifact paths.
  - Sign-off: only when unresolved blocker weaknesses are closed.
  - Current state: expanded after E2E risk review; six new role cases were run
    one at a time and reviewed. One PM contract-shape failure was fixed with a
    general prompt rule, then rerun successfully. Current review:
    `test_artifacts/llm_reviews/coding_agent_phase2_new_artifact_role_review.md`.
- [ ] Stage 5 - E2E hard gates
  - Covers: five Phase 2 new-artifact hard gates through
    `coding_agent.propose_code_change(...)`.
  - Verify: run one E2E live gate at a time after Stage 4 reaches at least 90%
    confidence.
  - Evidence: raw traces, console logs, and agent-authored E2E review.
  - Sign-off: only when all five gates pass by agent-authored review.
- [ ] Stage 6 - final verification and independent code review
  - Covers: deterministic support tests, anti-cheat greps, stale-vocabulary
    greps, diff review, and final plan evidence.
  - Verify: all listed verification gates pass and review has no unresolved
    blockers.
  - Evidence: record findings, fixes, reruns, residual risks, and approval.
  - Sign-off: after review and evidence are recorded.

## Verification

### Static Greps

- `rg "Source Ownership PM|symbols_to_modify|existing_source_anchors|Module PM|Module programmer" src/kazusa_ai_chatbot/coding_agent tests test_artifacts/live_gate`
  - Expected: zero matches outside archived/superseded historical artifacts.
- `rg "code_modifying" src/kazusa_ai_chatbot/coding_agent tests`
  - Expected: matches only where Phase 2 returns or records that
    existing-source modification is a separate capability.
- Anti-cheat grep for hard-gate terms:
  - Expected: no matches in production code, runtime prompts, deterministic
    pass/fail logic, or non-live tests.
  - Allowed: the supporting gate document, live-gate fixtures, raw LLM traces,
    and LLM review artifacts.

### Deterministic Support Checks

- `venv\Scripts\python -m pytest -q tests/test_coding_agent_phase2_new_artifact_contracts.py`
  - Expected: passes after Stage 3.
  - Scope: Writing PM decision parsing, one-artifact programmer output parsing,
    new-file artifact package materialization, public response redaction, and
    existing-source edit rejection.
- `venv\Scripts\python -m compileall -q src/kazusa_ai_chatbot/coding_agent`
  - Expected: touched production modules import and compile cleanly.

### Real LLM Role Suites

- Run one live LLM case at a time with `-s`.
- Command shape:
  `venv\Scripts\python -m pytest -m live_llm tests/test_coding_agent_phase2_new_artifact_role_live_llm.py::<test_name> -q -s`
- Required roles: Writing PM, Writing programmer, Patching worker,
  Synthesizer. Supervisor chaining is reserved for E2E.
- Required coverage: every hard gate contributes role inputs for every
  participating LLM role.
- Required artifact: raw trace plus agent-authored review before the next live
  case starts.
- Test code and scripts may emit raw evidence only: JSON, logs, trace paths,
  parser results, validation summaries, and artifact manifests. The active
  agent must author the human-readable quality review after inspecting that
  evidence.
- Gate definitions and AI-review criteria are owned by
  `development_plans/reference/designs/coding_agent_phase2_new_artifact_gating_tests.md`.

### E2E Hard Gates

E2E is blocked until Stage 4 is signed off.

Run the five hard gates from
`development_plans/reference/designs/coding_agent_phase2_new_artifact_gating_tests.md`
one at a time through `coding_agent.propose_code_change(...)`.

Command shape:
`venv\Scripts\python -m pytest -m live_llm tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py::<test_name> -q -s`

Each gate must return proposed artifacts only and must not mutate the real
workspace.

Known validation-time execution of generated test artifacts is a Phase 2.5
security finding. During Phase 2 E2E review, record the finding but do not fail
the Phase 2 gate on that issue when the requested artifacts, workflow, and
public response are otherwise acceptable.

## Independent Code Review

Run this gate after all `Verification` commands and hard gates pass and before
final sign-off. Because the user currently prohibits Codex subagents, perform
the review from a single-agent independent-review posture unless the user later
approves a separate reviewer.

Review scope:

- Alignment with this plan, especially new-artifact-only Phase 2 scope,
  supervisor-owned interleaving, no existing-source semantic edits, role-suite
  gating, anti-cheat, and mutation boundary.
- Prompt and payload quality for local LLMs: concise wording, common language,
  no hidden architecture jargon, no gate-shaped examples, and stable output
  contracts.
- Code quality and design: role ownership, artifact boundaries, no
  compatibility shims, no peer subagent calls, and no hidden worker-to-worker
  loops.
- Test and evidence quality: every live LLM case is reviewed by an agent, not
  accepted by deterministic schema alone.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Independent Plan Review

Before implementation resumes from this plan, reread the architecture
reference, `code_writing` README, this plan, and recent Phase 2 E2E failure
records.

Review scope:

- The plan states one canonical Phase 2 architecture.
- Phase 2 is new-artifact writing only.
- The architecture still supports later supervisor interleaving between
  reading, writing, modifying, patching, external evidence, and execution.
- E2E is blocked until role-level live LLM diagnostics have highlighted and
  resolved weaknesses.
- The plan has no unresolved options, compatibility paths, or hidden future
  enhancements.

Record blockers and required edits before implementation resumes.

## Acceptance Criteria

This plan is complete when:

- The architecture and ICD use the new-artifact `code_writing` contract
  consistently.
- Existing-source semantic modification is not part of Phase 2 implementation
  or hard-gate signoff.
- Every participating LLM role has live LLM cases generated from every hard
  gate.
- Every role-suite live LLM case has a raw trace and agent-authored review.
- E2E is not run until role suites expose and resolve blocker weaknesses and
  the active agent records at least 90% readiness confidence from role
  evidence.
- All five hard E2E new-artifact gates pass by agent-authored review.
- Any observed generated-artifact execution is recorded as Phase 2.5 security
  scope and excluded from Phase 2 artifact-quality pass/fail.
- Deterministic support checks and anti-cheat greps pass.
- Proposed artifacts never mutate the caller's real workspace.
- Independent code review finds no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Writing PM emits vague artifact contracts | Require purpose, file kind, content format, imports, interfaces, and required behavior | Role-level Writing PM live LLM reviews |
| Programmer invents surrounding project context | Give one complete new-artifact contract and no peer output | Programmer role-suite reviews |
| Patching worker changes feature semantics | Patching input contains selected artifacts and path reservations only | Patching role-suite reviews and artifact validation |
| E2E thrashing resumes | Hard block E2E until role suites are reviewed and blockers resolved | Stage 4 readiness gate |
| Overfitting to hard gates | Anti-cheat grep and no gate-shaped prompt examples | Static grep plus review |
| Local LLM context overload | Prompt-budget reports and fail-closed caps | Trace context-budget fields |
| Deterministic validator becomes semantic keyword judge | Restrict deterministic checks to structure, paths, caps, syntax, and artifact shape | Plan review and code review |
| Generated-artifact execution appears in validation | Record as Phase 2.5 security scope and do not expand execution in Phase 2 | Phase 2 E2E review plus Phase 2.5 plan |

## Execution Evidence

- 2026-06-26: Phase 2 progress reset after user review.
  - All progress checklist stages are unchecked.
  - The five E2E hard gates were replaced with a difficulty gradient from
    easy single-file generation to hard multi-source project generation.
  - Previous Stage 1 sign-off was invalidated by this reset and must be rerun
    before implementation resumes.
- 2026-06-26: Created supporting AI-evaluated gate specification:
  `development_plans/reference/designs/coding_agent_phase2_new_artifact_gating_tests.md`.
  The plan now links to that file as the authoritative source for hard-gate
  definitions, test procedure, and AI-review pass criteria. The Phase 3
  feedback-loop payload shape is owned by this plan's `Contracts And Data
  Shapes` section.
- 2026-06-26: Independent plan review performed and surfaced documentation
  issues.
  - Fixed registry wording so the supporting gate document is a verification
    procedure and pass-criteria artifact, not a design artifact.
  - Fixed hard-gate anti-cheat allowance so challenge text is allowed in the
    supporting gate document, live-gate fixtures, raw traces, and review
    artifacts.
  - Fixed change-surface wording so `test_artifacts/live_gate/` derives gates
    from the supporting gate document instead of from this plan body.
  - Added exact planned test files and command shapes for deterministic support
    checks, role-level live LLM tests, and E2E live LLM gates.
  - Added the debug-LLM boundary: scripts may emit raw evidence only, and the
    active agent authors human-readable quality reviews after inspection.
- 2026-06-26: Implemented Stage 1 through Stage 3 for the renewed
  new-artifact Phase 2 design.
  - Updated architecture/plan alignment and removed stale existing-source
    modification role wiring from the active code-writing path.
  - Replaced legacy file/module/source-owner contracts with new-artifact
    contracts for Writing PM, Writing programmer, File Agent reservation,
    Patching worker, validator, synthesizer, and supervisor.
  - Added supervisor-owned validation repair loop.
  - Added File Agent path reservation for new artifacts, mechanical local
    import enrichment from reserved source paths, and patch materialization for
    new files only.
  - Updated programmer and PM prompts for new-artifact scope, validation
    feedback, Markdown artifact formatting, and test-contract alignment.
  - Verification:
    `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent`
    passed.
  - Verification:
    `venv\Scripts\python -m pytest -q tests\test_coding_agent_phase2_new_artifact_contracts.py`
    passed with 15 tests.
- 2026-06-26: Stage 4 role-level live LLM suite was corrected and initially
  signed off before E2E risk review.
  - Chained writing-supervisor live tests were removed from the Stage 4 role
    suite because they exercise multi-role orchestration instead of one role
    with ideal input.
  - Writing PM: 5/5 isolated cases passed AI role review.
  - Writing programmer: 9/9 isolated cases passed AI role review.
  - Patching worker: 5/5 isolated cases passed deterministic materialization
    checks.
  - Synthesizer: 5/5 isolated cases passed AI role review.
  - Review artifact:
    `test_artifacts/llm_reviews/coding_agent_phase2_new_artifact_role_review.md`.
- 2026-06-26: Stage 4 was expanded before E2E.
  - Added three PM repair role cases for validation-feedback repair:
    return-shape mismatch, missing required behavior, and missing consumed
    interface.
  - Added three programmer role cases for E2E-risk handoffs: Markdown-link
    tests, task-note renderer, and mocked inventory-fetch tests.
  - Verification:
    `venv\Scripts\python -m pytest --collect-only -q -m live_llm tests\test_coding_agent_phase2_new_artifact_role_live_llm.py`
    collected 30 role tests.
  - Deterministic support:
    `venv\Scripts\python -m pytest -q tests\test_coding_agent_phase2_new_artifact_contracts.py`
    passed with 15 tests.
  - Expanded live LLM execution:
    `test_live_writing_pm_repair_return_shape_mismatch` initially failed AI
    review because the Writing PM did not fully specify shared input-file and
    record-dictionary shapes after a return-shape repair.
  - Fix:
    `src/kazusa_ai_chatbot/coding_agent/code_writing/product_manager.py`
    now instructs the Writing PM and PM reviewer to define shared file,
    record, config, and command-line input shapes in source/test contracts.
    File parsing contracts must state line or record format, and returned
    record dictionaries must state required keys and value meaning.
  - Expanded role cases rerun one at a time:
    three PM repair cases and three Writing programmer cases passed AI role
    review at 100 confidence.
  - Current Stage 4 role-suite confidence is 100% from the isolated role
    evidence in
    `test_artifacts/llm_reviews/coding_agent_phase2_new_artifact_role_review.md`.
  - E2E was not run.
- 2026-06-26: Phase 2 pass wording updated for the security handoff.
  - Known validation-time generated-code execution is tracked by
    `development_plans/active/short_term/coding_agent_phase2_5_security_boundary_plan.md`.
  - Phase 2 E2E review now records the security finding without failing
    artifact-quality signoff solely on that issue.
- 2026-06-27: Added focused synthesizer live-role coverage for generated-test
  validation failure interpretation.
  - Initial role test design was corrected because the reviewer was too weak.
  - Tightened AI review reproduced the failure: the synthesizer claimed a
    generated implementation defect from a generated-test assertion.
  - Updated the synthesizer prompt so generated-test failures are reported as
    validation failures and not as proven implementation defects unless
    independent non-test evidence exists.
  - Verification:
    `venv\Scripts\python -m pytest -m live_llm tests/test_coding_agent_phase2_new_artifact_role_live_llm.py::test_live_synthesizer_generated_test_failure_interpretation -q -s`
    passed.
  - Collection:
    `venv\Scripts\python -m pytest --collect-only -q -m live_llm tests/test_coding_agent_phase2_new_artifact_role_live_llm.py`
    collected 31 live role tests.
  - Compile check:
    `venv\Scripts\python -m compileall -q src/kazusa_ai_chatbot/coding_agent/code_writing/synthesizer.py`
    passed.
- 2026-06-27: Added acceptance/alignment owner after Gate 04 artifact-alignment
  failure.
  - Reproduced the issue at role scope: the existing Writing PM role reviewer
    accepted the faulty Gate 04 PM decision because it checked internal
    source/test interface consistency but did not preserve the user-visible CLI
    requirement.
  - Implemented `code_writing.acceptance` with two PM-route LLM judgments:
    acceptance criteria preservation before PM decomposition and artifact
    alignment review after structural validation.
  - Wired alignment failure into the existing repair loop as feedback to the
    Writing PM.
  - Initial alignment prompt still failed by treating the missing CLI entry as
    a minor improvement. Prompt was tightened so requested user-facing access
    method is preserved and missing access path is a blocker.
  - Live LLM verification:
    `venv\Scripts\python -m pytest -m live_llm tests/test_coding_agent_phase2_new_artifact_role_live_llm.py::test_live_alignment_rejects_missing_cli_entrypoint -q -s`
    passed.
  - Deterministic support:
    `venv\Scripts\python -m pytest -q tests/test_coding_agent_phase2_new_artifact_contracts.py`
    passed with 16 tests.
  - Collection:
    `venv\Scripts\python -m pytest --collect-only -q -m live_llm tests/test_coding_agent_phase2_new_artifact_role_live_llm.py`
    collected 32 live role tests.
