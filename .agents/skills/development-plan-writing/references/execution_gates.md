# Execution Gates Reference

Read this reference for every final executable plan. It owns granular execution
steps, implementation order, progress checklist, verification, execution
evidence, handoff, and review-derived accuracy gates.

## Implementation Order

Implementation order must prevent avoidable breakage and agent improvisation.

Default to a module-first, test-first order. The plan should prove the module
contract before implementation, then prove integration after the module is
stable. For LLM, database, migration, or production-path changes, include the
focused module test and the relevant integration, real LLM, real database,
migration dry-run, or smoke test needed to cover the actual risk.

Every final plan must include this general sequence unless a step is truly
inapplicable:

1. Add or update module tests.
   - Name the exact test file, test function, fixture, or diagnostic that
     proves the target module contract.
   - Run the module test before implementation when applicable and record the
     expected failure, missing symbol, missing entrypoint, or baseline result.
2. Add or update integration tests.
   - Name the exact integration, real LLM, real database, migration, smoke, or
     cross-module test that proves callers can use the module correctly.
   - Run the integration test before implementation when practical and record
     the failure mode or current behavior.
3. Implement the module.
   - Keep the behavior inside the target module and approved public interface
     whenever possible.
   - Do not update existing outside modules or introduce new code paths,
     prompts, or variables without the strong justification required in
     `Change Surface`.
4. Run module tests.
   - Re-run the same module test from step 1.
   - Record the result in `Execution Evidence`.
5. Loop back to step 1 if needed.
   - If the module test fails or reveals an incomplete contract, update the
     module test or module implementation only to match the approved plan, then
     repeat steps 1, 3, and 4 before touching integration.
6. Implement integration.
   - Wire existing callers, adapters, prompts, database paths, or service
     entrypoints to the module only after module tests pass.
   - Keep integration edits inside the approved change surface.
7. Run integration tests.
   - Re-run the same integration test from step 2.
   - Then run any broader verification gates listed in `Verification`.
   - Record the result in `Execution Evidence`.
8. Loop back to step 1 if needed.
   - If integration exposes a module contract problem, return to module tests
     before changing integration again.
   - If integration exposes only wiring behavior, update integration and re-run
     the integration test.

For low-risk documentation-only plans, explicitly state that the
module/integration test-first sequence is not applicable and replace it with a
before/after review gate.

Additional ordering guidance:

- create the new contract first
- rewrite consumers next
- wire the new entrypoint
- delete legacy modules after references are gone
- run greps and smoke tests last

Include a short rationale when order matters:

```md
Build the projection module first because it becomes the contract used by
cognition, consolidation, and tests.
```

## Granular Execution Steps

Implementation stages must decompose into small, verifiable steps. A step is
too broad if an implementation agent could complete it in multiple incompatible
ways without violating the wording.

Default step shape:

- one action with one clear target
- exact file path, symbol, test, command, or artifact
- expected result or evidence before moving on
- no hidden design choice

For low-risk implementation work, aim for steps that take about 2-5 minutes:

```md
- [ ] Step 1 - add the failing serializer test
  - File: `tests/test_serializer.py`
  - Action: add `test_serializer_rejects_unknown_kind`.
  - Verify: `pytest tests/test_serializer.py::test_serializer_rejects_unknown_kind -q`
  - Expected before implementation: fails because `UnknownKindError` is not raised.
```

For high-risk architecture, prompt, migration, database, or production-path
work, do not force artificial 2-minute fragments. Use the smallest step that
preserves reviewability, rollback clarity, and a single decision boundary.

Use TDD triplets whenever code behavior changes:

1. write or update the focused failing test
2. run it and record the expected failure
3. implement the minimal change
4. rerun the same test and record the pass

Do not write broad steps such as:

- "implement the module"
- "add tests"
- "update callers"
- "handle errors"
- "wire everything together"

Replace them with named files, symbols, tests, commands, expected failures, and
expected passes.

Code snippets are optional. Include complete snippets only when the code is
stable, narrow, and unlikely to become stale before execution. For high-risk or
fast-moving code, prefer exact contracts, signatures, state shapes, invariants,
and verification commands over copy-paste implementation bodies.

## Progress Checklist

Every final plan must include a tickable progress checklist using Markdown
checkbox syntax (`- [ ]` or `1. [ ]`). These are the progress boxes
implementation agents update as each function, module, integration step, or
sign-off gate is completed. Small plans may have a short checklist, but they
still need one so progress and handoff state are visible.

Checkpoints exist so multiple agents can resume the work without rediscovering
state or guessing which partial changes are complete. They should be granular
enough that an agent can finish one checkpoint, verify it, mark it complete,
and hand off cleanly.

Each tickable checkpoint must describe:

- the stage, function, module, interface, integration, or sign-off gate being
  completed
- the files or modules expected to be touched
- the verification commands or static checks to run before ticking the box
- the evidence that must be recorded before ticking the box
- the next checkpoint or next implementation step
- the sign-off line the agent must complete after that checkpoint is done
- the granular execution steps inside the checkpoint, or a pointer to the
  numbered implementation steps it covers

```md
## Progress Checklist

- [ ] Stage 1 - module contract established
  - Covers: steps 1-3.
  - Verify: `python -m py_compile ...`; focused module tests pass.
  - Evidence: record changed files and test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - service integration complete
  - Covers: step 4.
  - Verify: service integration tests pass.
  - Evidence: record test output before moving on.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
```

Do not treat checked boxes as proof by themselves. An agent may tick a
checkpoint only after running its verification and recording the result in
`Execution Evidence` or a linked execution record. If verification is skipped
or blocked, leave the box unchecked and record why.

Agents must sign off stages one at a time, immediately after each stage is
genuinely complete. Do not pre-fill sign-offs, do not sign off future stages,
and do not batch multiple stage sign-offs at the end of a session. If handing
off mid-plan, leave all unfinished stages unchecked and unsigned, and add a
brief handoff note that points to the next unchecked stage.

After signing off any major checklist stage, the active agent must reread the
entire plan before starting the next stage. This reread requirement must be
written into the plan's `Mandatory Rules` and treated as part of each major
stage sign-off gate.

## Plan Self-Review

Before marking a plan approved or ready for implementation, perform and record a
brief self-review:

- **Coverage:** every `Must Do`, accepted design decision, and acceptance
  criterion maps to at least one implementation step or verification gate.
- **Placeholder scan:** no `TBD`, `TODO`, "similar to", "handle edge cases",
  "add tests", or open-ended implementation wording remains.
- **Contract consistency:** filenames, function names, state keys, schemas,
  prompt variables, test names, and command paths match across sections.
- **Granularity:** no checkpoint hides multiple unrelated edits, and every step
  has a target plus expected evidence.
- **Verification:** each important behavior change has a focused check and an
  expected result.

Fix issues inline before approval. If the self-review finds a missing decision,
return to discovery instead of guessing.

## Verification

Verification must be written as gates, not vague reminders. Include exact
commands or checks:

```md
## Verification

### Static Greps

- `rg "research_facts|research_metadata" src` returns no matches.

### Tests

- `pytest tests/test_rag_projection.py`
- `pytest tests/test_save_conversation_invalidation.py`

### Smoke

- Service boots without missing import errors.
- One chat request returns a non-empty response.

### Database

- `rag_cache_index` and `rag_metadata_index` are absent after migration.
```

State allowed exceptions directly beside each check.

For every static check, state the exact expected result:

- whether zero matches are expected and which nonzero exit code is acceptable
  for tools such as `rg`
- which matches are allowed, if any, and the file or literal context that makes
  them acceptable
- which matches are forbidden and what the agent must do if they appear

## Review-Derived Accuracy Gates

When internal or external review finds stale wording, expectation drift,
missing handoff artifacts, or avoidable implementation freedom, update the
active plan before approving or continuing execution. Completed plans may
receive only factual evidence corrections or links to a new follow-up plan.

Do not leave placeholder verification text such as "copy from the previous
stage later" once the previous stage has completed. Carry forward the actual
completed artifact names, commands, and evidence needed by the next agent.

Do not authorize private helpers, wrappers, aliases, or abstraction layers just
because they are convenient. A plan may permit a helper only when it removes
nontrivial repeated structural validation, performs local table lookup, or
matches an established project pattern. Raising-only helpers, pass-through
wrappers, feature flags, fallback paths, and alternate call sites must be
forbidden unless the plan explicitly justifies why they are part of the
contract.

## Execution Evidence

Do not treat checked boxes as proof. If the plan also records completion, add a
separate evidence section:

```md
## Execution Evidence

- Static grep results:
- Test results:
- Service boot:
- DB verification:
- Manual smoke:
```

Pre-execution plans should use unchecked checklist items. If a plan is
completed, either move checked items into an execution record or attach evidence
for the checks.

Never document scope creep in a completed plan. Once complete, the plan may
receive only factual corrections to execution evidence or links to a
new/superseding plan. If new requirements appear after completion, keep the
completed plan unchanged and create a separate follow-up plan.

## Execution Handoff

After finalizing a plan, state the intended execution mode:

- same-session inline execution
- sequential handoff to another agent
- parallel or subagent-driven execution, only when the user explicitly
  authorizes it and file ownership boundaries are disjoint

For handoff, name the next unchecked stage, required skills, files expected to
change next, and verification commands to run before sign-off.
