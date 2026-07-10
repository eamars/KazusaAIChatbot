# coding_agent_assessment_gap_phase_b_plan

## Summary

- Goal: Close the assessment's remaining mechanical feedback gaps by giving
  every patch proposal a proposal-bound managed candidate, deterministic base
  verification, bounded execution repair, and refreshed validator feedback.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `py-style`,
  `test-style-and-execution`, `local-llm-architecture`,
  `no-prepost-user-input`
- Overall cutover strategy: Add one coding-agent-owned candidate-workspace and
  execution-plan contract for source-backed and source-free proposals, enable
  preapproval execution only through an explicit deployment flag, and replace
  transient repair prose with proposal-bound structured evidence.
- Highest-risk areas: operation routing rejecting preflight as unsupported,
  running generated code before approval, incorrectly broad verification scope,
  repair loops hiding environment blockers, and worker adapter ownership drift.
- Acceptance criteria: Gates 07, 08, and 09 pass as normal live LLM tests;
  source-backed and source-free candidate contracts, proposal-plan binding,
  Loop A feedback, repair budgets, failure bundles, and blocker tests pass.

## Context

The assessment correctly identifies the highest-leverage robustness gap:
proposals are delivered after static validation but before any managed apply or
execution evidence. Approval verification specs are selected in the background
worker from approval prose, defaulting to `python_compileall .`, instead of
being derived by the coding agent from changed files and repo tests. Repair sees
bounded redacted excerpts rather than structured failure evidence, and missing
dependencies are treated like repairable source failures.

This phase addresses the testable Loop B gaps:

- `test_live_gate_07_proposal_has_preapproval_preflight_evidence`
- `test_live_gate_08_vague_approval_runs_changed_file_tests`
- `test_live_gate_09_missing_dependency_becomes_typed_blocker`

The first real run of gate 07 did not reach proposal generation. The
background coding operation router rejected the task because the user requested
managed-copy preflight and focused tests, and the current router limits describe
all patch application and execution as unsupported. This phase therefore owns
the operation-router capability wording and tests, not only the later preflight
implementation.

The current code also exposes four plan-level gaps that must be closed here:

- `code_patching.apply.apply_approved_patch(...)` has one human-approval
  contract. Reusing or weakening it for preflight would collapse permission
  and containment into one boolean.
- `code_verifying` requires an explicit source-backed request, so the current
  draft's "any proposal" target cannot cover source-free patch proposals.
- execution specs are not bound to a source identity, proposal revision, or
  patch digest, so a revised or repaired proposal could reuse stale checks;
- `MAX_REPAIR_ATTEMPTS = 2` remains a fixed stage cap, and the existing repair
  feedback object still flattens failures rather than preserving individual
  failure signatures and trace frames.

Managed-copy containment is not an operating-system sandbox. The current
executor uses a scrubbed environment, an allowlisted argv, a managed working
directory, output caps, and timeouts, but executed Python can still access host
resources available to the Kazusa process. Enabling preapproval execution is
therefore an explicit deployment trust decision and must be recorded as such.

### Assessment Coverage And Handoff

| Assessment finding | Phase B disposition |
|---|---|
| Loop A anchor/contract feedback lacks refreshed context | Close in this phase without adding another retry tier. |
| Proposals lack real execution feedback before approval | Close through managed candidate preflight when explicitly enabled. |
| Approval checks are planned from approval prose | Close with a deterministic proposal-derived base plan. |
| Execution repair loses failure structure and stops after a fixed stage cap | Close with structured bundles and a bounded per-run budget. |
| Missing runtime/dependency is treated as source repair | Close with typed environment blockers before repair. |
| `create_child_pm` remains accepted but has no modifying-supervisor owner | Remove the dead status in this phase; retain the one-PM/one-programmer contract. |
| General read/search/edit/run JSON-action loop | Defer to a benchmark-gated follow-up plan after Phase C. |
| Delete/rename, arbitrary tools, installs, git publishing, private sources, other languages | Defer to a separate breadth plan. |

## Mandatory Skills

- `development-plan`: load before changing this plan or executing it.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before changing or running tests.
- `local-llm-architecture`: load before prompt, LLM contract, or agent workflow
  changes.
- `no-prepost-user-input`: load before changing how approval detail or an
  explicit user verification request becomes an execution spec.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan
  before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.
- Use parent-led native subagent execution unless the user explicitly approves
  fallback execution.
- Keep execution argv-only, allowlisted to `python_compileall` and `pytest`,
  bounded by existing timeouts and output caps.
- Preflight execution must use managed copies only and must never mutate the
  original checkout.
- Treat managed-copy execution as process containment, not a host sandbox.
  Keep preflight disabled by default and record the configured execution
  backend and opt-in state in run evidence whenever it is enabled.
- Human approval must still be required for approval-mode verification and any
  future target-side effect.
- Keep the background operation router semantic and process-stable. It may
  classify a managed-preflight coding request as modification work, but it must
  not receive or decide the deployment flag, approval state, execution
  feasibility, or permission.
- Bind every derived execution plan and every preflight result to one run id,
  source identity, proposal revision, and canonical patch-artifact digest.
  Reject stale bindings instead of silently regenerating or reusing them.
- Let an LLM specialist interpret an explicit user request for extra tests.
  Deterministic code may validate, contain, cap, and merge the returned specs;
  it must not keyword-classify free-form approval text into selectors.
- Keep the normal prompt contract under 50k tokens. Bound each failure bundle,
  the aggregate per-run feedback budget, refreshed file windows, and the
  number of repair calls before invoking a local model.
- Anti-cheat rule: execution specs must be derived from changed files, test
  files, and deterministic repository evidence. Do not satisfy gate 08 by
  keywording the test name, hardcoding fixture paths, or letting L2d/worker
  prose choose the primary verification scope.
- Anti-cheat rule: do not satisfy gate 07 by special-casing the words
  "preflight", "slug", or the fixture name. The route must accept any
  source-backed patch proposal that asks for managed-copy preflight when
  `CODING_AGENT_PREFLIGHT_EXECUTION` is enabled.

## Must Do

- Add `CODING_AGENT_PREFLIGHT_EXECUTION` with default disabled until this phase
  is explicitly enabled for live gates.
- Update the stable background coding operation router prompt and focused tests
  so a source-backed request whose semantic goal is a patch remains
  `code_modifying` even when it asks for managed-copy preflight. Evaluate
  `CODING_AGENT_PREFLIGHT_EXECUTION` only after routing in deterministic coding
  run code.
- Add focused routing tests proving
  `decide_background_coding_operation(...)` does not return `unsupported` for
  the gate-07 preflight shape, and still rejects
  arbitrary live execution, deployment, package installation, and original-tree
  mutation requests.
- Add an internal `materialize_managed_candidate(...)` boundary with a closed
  authorization purpose. `preapproval_preflight` is valid only when the
  deployment flag is enabled; `approved_verification` still requires the
  existing structured human approval. Keep `apply_approved_patch(...)` as the
  approval-enforcing public API and share only private copy/apply mechanics.
- Support two candidate baselines through that one internal boundary:
  `resolved_source` copies the resolved checkout, while `empty_source_free`
  materializes create-file proposal artifacts into an empty managed candidate.
  Do not add a second verifier or source-free-only executor.
- Add `proposal_revision` and a canonical SHA-256 digest over ordered patch
  artifacts to the coding-run ledger. Increment the revision and regenerate
  the digest after initial proposal, user revision, and execution repair.
- Derive the base execution plan inside the coding agent from the final changed
  files, repository evidence, conventional test companions, and generated
  source-free test files. Use exact changed Python paths for compile checks;
  add focused pytest selectors only for existing safe test files. Record
  `no_focused_test_discovered` when compile-only verification is the strongest
  available deterministic plan.
- Move the existing LLM execution-spec planner from the background adapter into
  `code_verifying.execution_planning` as an optional additive specialist. Call
  it only when the user explicitly requested extra verification; give it the
  semantic request plus bounded safe test candidates, validate its structured
  output, and merge valid extras without replacing the deterministic base plan.
- Run preflight apply and derived focused checks after static validation for
  `propose_patch` objectives when the config flag is enabled.
- For source-free preflight, treat generated test files as protected after the
  first candidate revision: execution repair may change generated source but
  may not weaken or rewrite the tests it is trying to satisfy. User-requested
  proposal revision remains the path for changing generated tests.
- Reuse the stored proposal-bound base plan for approval verification. Merge a
  validated additive user-extra plan only after the stored source identity,
  proposal revision, and patch digest still match.
- Replace repair feedback prose with one structured failure evidence bundle per
  failed spec. Persist public-safe bundle summaries as coding-run events and
  pass bounded full bundles only to the owning repair supervisor.
- Add a deterministic per-run repair budget: default four repair calls, hard
  maximum six, maximum 6,000 failure-bundle characters per attempt, and maximum
  24,000 failure-bundle characters across the run. Trusted direct callers may
  request a lower budget; L2d and the worker adapter may not raise it.
- Add deterministic no-progress and regression stop rules. Stop when the same
  normalized failure signature repeats after a repair or when a previously
  passing stored spec fails. Persist a typed blocker for either stop.
- Classify missing external dependencies and missing interpreters as typed
  environment blockers, and do not spend repair attempts on them.
- Run the environment blocker classifier before repair generation. A failure
  classified as `environment_dependency_missing` must set the durable run to
  `blocked`, append a typed blocker, and leave `repair_attempts` empty.
- Convert gates 07, 08, and 09 from known-gap `xfail(strict=True)` to normal
  live gates only after implementation passes them.
- Refresh Loop A feedback within the existing validation retry: include the
  exact validator diagnostic and bounded current file context around exact or
  nearest anchor candidates (at most 20 lines before and after). Do not add a
  new retry tier or deterministic patch correction.
- Remove `create_child_pm` from the `code_modifying` PM status, normalizer, and
  tests. The current modifying architecture has no child-PM execution owner;
  accepting the status and falling through to a generic blocker is contract
  drift, not graceful degradation.

## Deferred

- Do not add package installation, arbitrary shell, network checks, private
  source authentication, or non-Python executor plugins.
- Do not add the generic JSON-action loop.
- Do not implement `respond_to_blocker`, L2d affordance binding, source locking,
  approval evidence hardening, or the benchmark harness in this phase.
- Do not broaden preflight beyond managed copies and the existing two execution
  tools.
- Do not claim host filesystem, process, or network sandboxing. An isolated
  runner backend requires its own approved security plan.
- Do not add delete/rename, dependency installation, branch/commit/PR output,
  private-source authentication, repository indexing, or non-Python plugins.

## Cutover Policy

Overall strategy: bigbang for execution-plan ownership; additive and disabled
by default for preapproval execution.

| Area | Policy | Instruction |
|---|---|---|
| Base execution planning | bigbang | Route proposal, preflight, approval, and repair through one proposal-bound plan. Remove adapter-owned primary planning in the same change. |
| Optional extra verification | bigbang | Move semantic extraction to the coding-agent specialist and allow it to add validated specs only. |
| Managed candidate creation | bigbang | Use one internal baseline-aware materializer for source-backed and source-free proposals. Keep `apply_approved_patch(...)` approval-enforcing. |
| Preapproval execution | compatible | Add behind the default-disabled flag. Disabled runs keep review-only behavior and record that preflight was disabled. |
| Ledger schema | migration | Increment the run schema, write proposal revision/digest/plan fields for new and touched runs, and reject stale incomplete approval continuations. Do not infer missing bindings. |
| Tests | bigbang | Convert gates 07-09 only after deterministic contracts pass. |

The responsible execution agent must not preserve the adapter LLM planner as a
fallback primary path, create a second verifier for source-free work, or accept
unbound legacy approval verification. Any policy change requires user approval.

## Target State

When enabled, a source-backed or source-free patch proposal reaches
`awaiting_approval` only after the coding agent has materialized its exact
proposal revision in a managed candidate, run the proposal-derived base checks,
and repaired repairable failures within the stored budget. The final proposal,
candidate evidence, and execution plan share one source identity/revision/digest
binding. Approval revalidates that binding and reruns the same base plan, with
only validated user-requested extras added. Environment failures, no-progress,
and regressions surface as typed blockers rather than opaque source repairs.

When preflight is disabled, proposal generation remains review-only, the run
records `preflight_enabled=false`, and approval still uses the deterministic
proposal-derived plan. Phase C locking is the production-readiness gate before
operators enable longer preflight loops in a deployment with concurrent coding
workers.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Preflight safety | Use a separate managed-copy preflight authorization | Avoid confusing preflight with human approval. |
| Preflight routing | Classify the semantic patch task independently of the deployment flag | LLM routing must not decide operational feasibility or permission. |
| Candidate baseline | One materializer with `resolved_source` and `empty_source_free` | Loop B should apply to every proposal without duplicating verification engines. |
| Spec ownership | Coding agent derives specs from changed files | Verification scope belongs with the proposal being verified. |
| User extra checks | Coding-agent LLM specialist extracts additive specs; deterministic code validates and merges | Free-form user instruction interpretation stays LLM-owned while execution safety stays deterministic. |
| Plan binding | Bind plan to source identity, proposal revision, and artifact digest | Approval and repair must not execute stale specs against a different proposal. |
| Repair budget | Store a capped attempt and aggregate evidence budget per run | Effort scales within a hard bound instead of stopping at an opaque fixed stage cap. |
| Failure evidence | Structured bundle plus bounded excerpts | Small models need file, line, kind, and signature, not flattened prose. |
| Environment failures | Typed blocker with zero repair attempts | The agent cannot install dependencies today. |
| Host isolation | Record current backend as managed-copy process execution | The current executor does not provide OS sandboxing. |

## Contracts And Data Shapes

Preflight result:

```python
{
    "preflight_enabled": bool,
    "execution_backend": "managed_copy_process",
    "authorization_purpose": "preapproval_preflight",
    "run_id": str,
    "source_identity": dict[str, object],
    "proposal_revision": int,
    "patch_artifact_digest": str,
    "execution_plan_id": str,
    "status": "disabled | passed | blocked | failed",
    "apply_attempts": list[dict[str, object]],
    "execution_attempts": list[dict[str, object]],
    "repair_attempts": list[dict[str, object]],
    "blockers": list[dict[str, object]],
}
```

Derived execution plan:

```python
{
    "schema_version": "coding_execution_plan.v1",
    "plan_id": str,
    "origin": "changed_files",
    "baseline_kind": "resolved_source | empty_source_free",
    "run_id": str,
    "source_identity": dict[str, object],
    "proposal_revision": int,
    "patch_artifact_digest": str,
    "specs": [
        {
            "spec_id": str,
            "tool": "python_compileall",
            "paths": ["changed_python_file_or_package"],
        },
        {
            "spec_id": str,
            "tool": "pytest",
            "pytest_selectors": ["tests/test_x.py"],
        },
    ],
    "source_paths": list[str],
    "test_paths": list[str],
    "protected_test_paths": list[str],
    "limitations": list[str],
    "reason": str,
}
```

The initial successful proposal is revision `1`; each replacement proposal or
execution repair increments it before candidate materialization. Compute
`patch_artifact_digest` as SHA-256 over UTF-8 canonical JSON containing the
ordered artifact id, repo-relative path, and SHA-256 of each diff, sorted by
path then artifact id. For `empty_source_free`, use a synthetic public source
identity with `provider="source_free"` and
`current_commit="artifact-sha256:<patch_artifact_digest>"`.

Managed candidate authorization:

```python
{
    "purpose": "preapproval_preflight | approved_verification",
    "run_id": str,
    "proposal_revision": int,
    "patch_artifact_digest": str,
    "preflight_enabled": bool,
    "approval": dict[str, object] | None,
}
```

`preapproval_preflight` requires `preflight_enabled=True` and rejects any human
approval field. `approved_verification` requires the existing validated human
approval and does not accept the deployment flag as permission.

Failure evidence bundle:

```python
{
    "failure_id": str,
    "attempt_index": int,
    "spec_id": str,
    "tool": "pytest | python_compileall",
    "selector": str,
    "failure_kind": (
        "assertion | exception | import_error | compile_error | timeout | "
        "environment"
    ),
    "exception_type": str,
    "exception_message": str,
    "trace_frames": [
        {"path": str, "line": int, "function": str, "code_line": str},
    ],
    "failure_signature": str,
    "related_evidence_ids": list[str],
    "stdout_excerpt": str,
    "stderr_excerpt": str,
}
```

Repair budget:

```python
{
    "max_repair_calls": int,
    "max_bundle_chars_per_attempt": int,
    "max_total_bundle_chars": int,
    "repair_calls_used": int,
    "bundle_chars_used": int,
}
```

Defaults are `4`, `6000`, and `24000`; validation caps
`max_repair_calls <= 6`. Budget exhaustion creates a typed
`repair_budget_exhausted` blocker.

Environment blocker:

```python
{
    "code": "environment_dependency_missing",
    "blocker_kind": "environment",
    "message": str,
    "question": str,
    "options": list[str],
    "resume_target": "retry_verification",
    "details": {
        "missing_module": str,
        "tool": str,
        "spec_id": str,
        "selector": str,
        "execution_plan_id": str,
    },
    "created_at": str,
}
```

The external-module classifier may label a missing module as environment only
after comparing the missing root with bounded local import roots and the final
candidate manifest. Ambiguous local-vs-external imports remain source failures;
they must not be promoted to environment blockers by a name-only allowlist.

## LLM Call And Context Budget

Before this plan, approval may add one execution-spec planning call and repair
may call `propose_code_change(...)` up to the current fixed cap. After this
plan, the deterministic base plan adds zero LLM calls. An explicit user request
for extra verification adds at most one background PM-route extraction call;
it cannot replace the base plan. Preflight repair adds zero to four background
repair calls by default and never more than six.

Each repair call receives one bounded set of failure bundles, relevant source
windows, protected paths, the proposal revision/digest, and the prior patch
summary. Per-call prompt input remains below 50k tokens. Deterministic shaping
caps each bundle at 6,000 characters, all bundles for a run at 24,000
characters, trace frames at eight per failed spec, and refreshed anchor windows
at 41 lines. Oldest successful observations are dropped before current failed
frames or changed-file context.

Source-free preflight uses the same call budget. Repair routes back through
source-free writing with the pinned generated tests and current artifacts;
source-backed repair routes through modifying. Neither path restarts the
reading PM tree unless the existing repair contract explicitly requests missing
source evidence.

## Change Surface

### Delete

- Remove `EXECUTION_SPEC_PLANNER_PROMPT`, its stage LLM instance, and primary
  execution planning from
  `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`.

### Modify

- `src/kazusa_ai_chatbot/config.py`: add the default-disabled preflight flag
  and bounded repair-budget settings.
- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`: update operation-router
  semantics, add refreshed Loop A context, and retain mode/source identity on
  proposal responses.
- `src/kazusa_ai_chatbot/coding_agent/code_patching/models.py` and `apply.py`:
  add baseline-aware internal candidate materialization and closed purpose
  authorization while preserving `apply_approved_patch(...)` approval checks.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/models.py`: add execution
  plan, proposal binding, repair budget, failure bundle, and typed blocker
  shapes; allow the canonical source-free candidate mode.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py`: run
  derived specs for both baseline kinds, route repair by proposal mode,
  classify failures before repair, and enforce budget/stop rules.
- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`: connect proposal
  preflight for `propose_patch` objectives.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py`, `ledger.py`, and
  `supervisor.py`: increment the ledger schema; persist proposal revision,
  digest, execution plan, preflight policy/evidence, budget, and blockers; and
  revalidate bindings at approval.
- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`: stop owning
  execution planning; pass only the bounded semantic user-extra request to the
  coding-agent specialist.
- `src/kazusa_ai_chatbot/coding_agent/code_modifying/supervisor.py`: consume
  exact validator diagnostics and refreshed bounded target-file context inside
  the existing Loop A repair allowance.
- `src/kazusa_ai_chatbot/coding_agent/code_modifying/models.py`: remove the
  unhandled `create_child_pm` PM status and keep the closed one-PM/
  one-programmer lifecycle.
- Coding-agent, code-patching, code-verifying, coding-run, and background-work
  README files: document candidate modes, current process isolation, plan
  binding, repair budgets, and worker ownership.
- `tests/test_coding_agent_full_workflow_integration_live_llm.py`: remove
  `xfail` from gates 07, 08, and 09 after implementation.

### Create

- `src/kazusa_ai_chatbot/coding_agent/code_verifying/execution_planning.py`:
  deterministic base derivation, proposal binding/digest validation, bounded
  LLM extraction of additive user extras, and deterministic merge validation.
- `tests/test_coding_agent_phase_b_execution_planning.py`: base derivation,
  binding, additive extras, source-free candidates, and stale-plan rejection.
- `tests/test_coding_agent_phase_b_failure_feedback.py`: Loop A refreshed
  context, failure parsing, classifier boundaries, budgets, and stop rules.
- Add focused patch-apply and coding-run tests to existing Phase 5/9 contract
  files rather than creating duplicate ownership suites.

### Keep

- Existing executor allowlist and managed apply workspace containment.
- Existing `apply_approved_patch(...)` public approval contract, protected-path
  enforcement, redaction, async accepted-task lifecycle, and original-source
  immutability.

## Overdesign Guardrail

- Actual problem: The coding agent delivers proposals without proposal-bound
  mechanical evidence and later verifies them with approval-prose-planned
  checks and lossy repair feedback.
- Minimal change: Add one baseline-aware candidate boundary, one deterministic
  proposal-bound base plan, one capped feedback loop, and one optional additive
  selector specialist.
- Ownership boundaries: Coding agent owns proposal verification; worker adapter
  owns queue/runtime handoff; deterministic code owns execution safety and
  failure classification; LLM repair owns source-level semantic edits.
- Rejected complexity: arbitrary shell, installs, network controls falsely
  described as a sandbox, private credentials, generic action loop,
  delete/rename, git publishing, and language-plugin executor framework.
- Evidence threshold: Add rejected complexity only after gates 07-09 pass and a
  benchmark shows remaining failures caused by that missing breadth.

## Agent Autonomy Boundaries

- The responsible agent may add parser patterns only for failure shapes named
  in this plan and covered by deterministic positive and false-positive tests.
- Changes outside coding-agent, background-worker adapter, config, and tests
  require explicit justification in Execution Evidence.
- The responsible agent must not loosen path safety, output redaction, timeout,
  execution-tool, or protected-path rules.
- The responsible agent must not expose `preflight_enabled`, approval state,
  source paths, execution feasibility, or permission to either the operation
  router or additive-selector prompt.
- The responsible agent must not treat an LLM-selected extra spec as safe until
  deterministic path, tool, count, timeout, candidate-existence, and protected
  path validation passes.
- The responsible agent must not add a generic apply helper callable without a
  closed authorization purpose.
- If a proposed fix requires package installation or shell access, stop and
  report the blocker.

## Implementation Order

1. Parent establishes router and authorization baselines.
   - Add `test_background_operation_routes_managed_preflight_patch` to
     `tests/test_coding_agent_background_run_contracts.py` and negative cases
     for target mutation, deployment, installation, and arbitrary execution.
   - Add focused preflight-purpose tests to
     `tests/test_coding_agent_phase5_patch_apply_contracts.py` proving the
     preflight purpose rejects a disabled flag or human approval and that
     `apply_approved_patch(...)` still rejects missing approval.
   - Record the expected current router failure and missing internal candidate
     entrypoint before production implementation.
2. Parent establishes proposal-bound planning tests in
   `tests/test_coding_agent_phase_b_execution_planning.py`.
   - Cover source-backed changed-source/test mapping, source-free generated
     source/test mapping, compile-only limitation, additive extra extraction
     validation, protected paths, proposal digest changes, and stale binding
     rejection.
   - Expected baseline: the module and plan/digest fields are absent.
3. Parent establishes feedback tests in
   `tests/test_coding_agent_phase_b_failure_feedback.py`.
   - Cover assertion, exception, compile, timeout, missing interpreter, external
     missing module, ambiguous local module, repeated signature, regression,
     budget exhaustion, and refreshed anchor windows.
   - Expected baseline: current flattened feedback and fixed-attempt behavior
     do not satisfy the new contracts.
4. Parent starts exactly one production-code subagent with ownership limited to
   the Phase B production files in `Change Surface`, the three focused test
   contracts, and this approved plan.
5. Production subagent implements the internal candidate boundary first, then
   execution planning/digest binding, then verifier budgets/bundles, then run
   persistence and worker wiring. It removes adapter planning in the same
   cutover and reports all changed files and commands before closing.
6. Parent adds/updates integration tests while production implementation runs.
   - Extend Phase 9 contracts for initial proposal, revision, repair, disabled
     preflight, source-free preflight, approval reuse, and stale-plan rejection.
   - Extend background-run contracts for one optional user-extra specialist
     call and zero primary planner calls.
7. Parent runs the focused module tests. Any contract failure returns to steps
   1-5 before live integration changes.
8. Parent enables the preflight flag only inside gate 07, removes the three
   `xfail` markers after deterministic suites pass, and runs gates 07, 08, and
   09 one at a time with trace inspection.
9. Parent updates the coding-agent ICDs and records current process isolation,
   disabled-default behavior, exact call budgets, and Phase C locking handoff.
10. Parent runs the independent code-review gate, remediates only in-scope
    findings, reruns affected deterministic/live gates, and records evidence.

## Execution Model

- User explicitly authorized direct fallback execution without subagents on
  2026-07-10.
- Parent agent owns production changes, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes focused tests and expected baselines before
  production implementation starts.
- After planned verification passes, the parent conducts the Independent Code
  Review gate from a fresh-review posture and records findings and reruns.

## Progress Checklist

- [x] Stage 1 - focused contracts and baselines established.
  - Covers: steps 1-3.
  - Verify: the three focused deterministic files collect; record expected
    failures or missing symbols in `Execution Evidence`.
  - Handoff: start the production-code subagent at Stage 2.
  - Sign-off: `parent/2026-07-10`.
- [x] Stage 2 - candidate authorization and proposal binding complete.
  - Covers: steps 4-5 through candidate materialization and plan/digest
    binding.
  - Verify: Phase 5 apply contracts and Phase B execution-planning tests pass.
  - Evidence: changed production files, disabled-flag result, source-free and
    source-backed candidate results, and stale-binding rejection.
  - Handoff: continue Stage 3 with verifier feedback and persistence.
  - Sign-off: `parent/2026-07-10`.
- [x] Stage 3 - bounded repair and environment routing complete.
  - Covers: step 5 verifier work.
  - Verify: Phase B failure-feedback tests and Phase 8 verifier regressions
    pass.
  - Evidence: budgets, no-progress/regression blockers, Loop A window, and zero
    repair calls for environment blockers.
  - Handoff: parent completes worker/run integration at Stage 4.
  - Sign-off: `parent/2026-07-10`.
- [x] Stage 4 - run/worker integration and ICD cutover complete.
  - Covers: steps 6-7 and 9.
  - Verify: Phase 9 and background-run deterministic suites pass; static grep
    finds no adapter execution planner.
  - Evidence: proposal/plan binding through start, revise, repair, and approval;
    docs name process isolation and Phase C lock dependency.
  - Handoff: run live gates at Stage 5.
  - Sign-off: `parent/2026-07-10`.
- [x] Stage 5 - gates 07, 08, and 09 pass as normal live tests.
  - Covers: step 8.
  - Verify: run each live command separately and inspect its durable trace.
  - Evidence: route, candidate/preflight, focused-test derivation, blocker,
    privacy, and model-behavior judgment for each case.
  - Handoff: independent review at Stage 6.
  - Sign-off: `parent/2026-07-10`.
- [x] Stage 6 - independent code review complete.
  - Covers: step 10.
  - Verify: reviewer approval plus all remediation reruns.
  - Evidence: reviewer identity, diff reviewed, findings, fixes, reruns,
    residual risks, and approval status.
  - Handoff: update lifecycle only after this stage.
  - Sign-off: `parent/2026-07-10`.

## Verification

Run deterministic checks first:

```powershell
venv\Scripts\python -m pytest tests/test_coding_agent_phase_b_execution_planning.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_phase_b_failure_feedback.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_phase5_patch_apply_contracts.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_phase8_verify_repair_contracts.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_phase9_run_supervisor_contracts.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_background_run_contracts.py -q
venv\Scripts\python -m pytest --collect-only tests/test_coding_agent_full_workflow_integration_live_llm.py -q
```

Run static ownership and safety checks:

```powershell
rg -n "EXECUTION_SPEC_PLANNER_PROMPT|_plan_execution_specs" src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py
rg -n "shell=True|os\.system|subprocess\.(Popen|call)" src/kazusa_ai_chatbot/coding_agent
```

The first grep must return no matches; exit code 1 is expected. The second grep
must return no newly introduced execution paths. The existing argv runner may
continue using `subprocess.run` and must remain the only target-command runner.

Run live gates one at a time and inspect traces:

```powershell
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_07_proposal_has_preapproval_preflight_evidence -q -s -m live_llm
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_08_vague_approval_runs_changed_file_tests -q -s -m live_llm
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_09_missing_dependency_becomes_typed_blocker -q -s -m live_llm
```

Run and inspect one case before starting the next. Gate 07 must show a matching
proposal revision/digest/plan across candidate and run metadata. Gate 08 must
show the deterministically discovered focused test even when approval wording
is vague. Gate 09 must show `blocked`, an environment blocker, and zero repair
calls. All three must preserve public redaction and omit host roots.

## Independent Code Review

Run this gate after all Verification commands pass and before final sign-off.
The reviewer must inspect preflight safety, approval separation, execution-plan
ownership, source-free candidate behavior, proposal/digest binding, additive
selector interpretation, failure classification false positives, repair-loop
budgets, current process-isolation claims, and public trace sanitization. The
parent agent may fix findings only inside this plan's Change Surface, then
rerun affected verification commands and record the result.

## Acceptance Criteria

This plan is complete when:

- Proposal preflight evidence is present when the config flag is enabled.
- Both resolved-source and empty-source-free proposals use the same candidate
  and proposal-binding contract.
- Vague approval runs deterministic focused checks derived from changed files.
- A revised or repaired proposal cannot reuse a stale execution plan or
  candidate result.
- Explicit extra verification is LLM-interpreted inside the coding agent,
  deterministically validated, and additive to the base plan.
- Loop A feedback contains the exact validator message and bounded refreshed
  context without adding a retry tier.
- Missing dependency failures produce typed environment blockers and spend zero
  repair attempts.
- Default repair work is bounded to four calls and 24,000 failure-bundle
  characters, with hard maximum six calls and deterministic stop rules.
- Gates 07, 08, and 09 pass without `xfail`.
- Deterministic regression tests and independent code review pass.

## Execution Evidence

- 2026-07-10: user authorized direct fallback execution without subagents;
  plan and registry promoted to `in_progress`.
- Baseline: Phase B test modules were absent; existing Phase 5, 8, 9, and
  background-run contracts passed before the new contract tests were added.
- Deterministic verification: Phase B execution-planning and failure-feedback
  tests, Phase 5, Phase 8, Phase 9, background-run, existing-source planning,
  and code-modifying contract suites pass after the scoped changes.
- Static ownership check: the background adapter no longer contains
  `EXECUTION_SPEC_PLANNER_PROMPT` or `_plan_execution_specs`.
- Live gate 07: passed after private local-root recovery was added to the
  managed-candidate path; 204.06 seconds.
- Live gate 08: passed with vague approval selecting the proposal-derived
  focused slug test; 233.45 seconds.
- Live gate 09: inspected evidence showed the proposal correctly added the
  dependency import. The remaining fault was root-cause extraction: the
  verifier retained pytest's collection header instead of the later
  `ModuleNotFoundError`, and classified against the original source rather
  than the managed candidate. Focused regression coverage now preserves the
  missing-module root cause and passes the final candidate root to the
  classifier. The corrected one-at-a-time live rerun passed in 252.90 seconds,
  producing `blocked`, `environment_dependency_missing`, and zero repair
  attempts.
- Independent code review: direct-fallback fresh review inspected preflight
  safety, approval separation, execution-plan ownership, source-free behavior,
  binding, additive-selector validation, failure classification, repair
  bounds, process-isolation claims, and trace sanitization. It found a missing
  explicit output contract in the additive-selector prompt and an unnamed
  safe-test cap. Both were corrected; Phase B planning and failure-feedback
  tests passed again (6 passed). No residual release blocker remains.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Preapproval execution runs unsafe commands | Keep argv-only two-tool allowlist and managed copies | Gate 07 plus executor regression tests |
| Managed copy is mistaken for a host sandbox | Keep default disabled, record backend as process-only, and document operator trust | Config and metadata contract tests plus review |
| Source-free preflight lets repair weaken generated tests | Pin generated test paths after the first candidate revision | Source-free candidate and protected-path tests |
| Vague approval misses tests | Derive specs from changed source and test mapping | Gate 08 |
| Revised proposal reuses stale checks | Bind source identity, revision, and artifact digest | Stale-plan rejection tests |
| Additive selector specialist invents unsafe paths | Validate against bounded candidate test paths and merge additively | Execution-planning positive/negative tests |
| Missing dependency is misclassified | Deterministic classifier tests | Gate 09 |
| Repair loops regress passing checks | Regression stop rule | Focused repair-loop tests |
| Larger repair budget increases latency | Hard attempt/evidence caps and no-progress stop | Budget tests and trace call counts |
