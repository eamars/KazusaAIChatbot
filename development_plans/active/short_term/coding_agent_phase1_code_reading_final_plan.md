# coding agent phase1 code reading final plan

## Summary

- Goal: deliver the Phase 1 read-only code-reading implementation with
  evidence-grounded answers through the standalone coding-agent interface.
- Plan class: large
- Status: in_progress
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `debug-llm`, `superpowers:test-driven-development`,
  `superpowers:subagent-driven-development`, and
  `superpowers:requesting-code-review`.
- Overall cutover strategy: bigbang inside the standalone `coding_agent`
  Phase 1 code-reading module; no compatibility path to the current
  planner, PM, synthesizer, or source-shaped test contracts.
- Highest-risk areas: live/local LLM quality, PM/programmer boundary drift,
  weak source grounding, broad repository context overflow with local LLMs,
  insufficient query taxonomy coverage, and accidental Phase 3 integration.
- Acceptance criteria: live/local LLM PM, programmer, synthesis, and
  end-to-end gates are run one case at a time, inspected, and recorded in
  debug-LLM review artifacts; deterministic tests pass as support checks; final
  answers are grounded in selected evidence; route configuration and docs are
  updated; independent code review approves the architecture, tests, live LLM
  evidence, and execution evidence.

## Context

Phase 0 code fetching is complete and remains the upstream contract for this
plan. Phase 1 must normalize the current code-reading work into a generic
PM/programmer architecture: the PM decomposes the user's question into bounded
reading work, programmer workers discover source facts from scoped evidence,
and synthesis explains only what the reports and selected evidence support.

This plan restarts Phase 1 as a final product-quality implementation. Staged
development does not lower the acceptance bar. The completed result must be a
generic read-only code-reading agent within the declared standalone Phase 1
scope, not a scaffold that defers required quality.

The superseded Phase 1 plan files were deleted from
`development_plans/active/short_term/` before this plan was written. They must
not be restored. The architecture reference now points only to this final
Phase 1 plan.

## Mandatory Skills

- `development-plan`: load before editing this plan, executing it, updating
  lifecycle records, or reviewing plan completion.
- `local-llm-architecture`: load before changing PM/programmer boundaries,
  prompt surfaces, LLM contracts, context budgets, or synthesis behavior.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before creating or inspecting live/local LLM diagnostic
  reports or answer-quality artifacts.
- `superpowers:test-driven-development`: load before production implementation;
  parent must establish the focused failing test contract first.
- `superpowers:subagent-driven-development`: load before execution; normal
  execution requires the parent plus exactly one production-code subagent and
  one independent code-review subagent.
- `superpowers:requesting-code-review`: load before the final independent
  review gate.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- Before approval or execution, the parent agent must run the plan's
  `Independent Plan Review` gate and resolve all blocker findings.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.
- Phase 1 remains standalone. Do not connect this work to Kazusa L2d,
  background-work jobs, service delivery, dispatcher, persistence, or adapter
  output.
- Phase 1 is read-only after fetching. Do not add code writing, patch apply,
  package installation, arbitrary shell execution, test execution against
  fetched repositories, or Docker execution.
- The PM may use generic reading task types and generic evidence slots only.
  It must not depend on source-specific constants, fixture names, or expected
  answer phrases to decide the reading plan.
- Programmer workers discover concrete identifiers from repository evidence.
  Final answers may mention concrete code terms only when those terms are
  traceable to evidence rows returned by programmer reports.
- `code_reading` uses one optional dedicated `CODING_AGENT_LLM` route for PM,
  programmer, and synthesis LLM calls. When the dedicated route is absent, it
  falls back to `BACKGROUND_WORK_LLM`. Partial `CODING_AGENT_LLM` configuration
  is invalid and must fail configuration validation instead of mixing values
  from different routes.
- Deterministic code owns path safety, file caps, allowed file types,
  workspace-root privacy, source-scope enforcement, and schema validation. LLM
  stages own semantic decomposition and explanation only.
- Tests must prove generic behavior through multiple independent fixtures or
  source scenarios. A test that only rewards one fixed source vocabulary is
  insufficient.
- Real LLM tests are the primary Phase 1 quality gate. Deterministic tests
  support schema, safety, limits, route configuration, and regression, but they
  must not be treated as proof that the code-reading agent is product-quality.
- Live/local LLM tests must run one case at a time with output inspected after
  each case. Batch-running real LLM cases for a green/red summary is forbidden.
- Final Phase 1 approval is blocked if the required live/local LLM route,
  workspace, or trace artifacts are unavailable. A skipped live LLM gate is
  acceptable only with explicit user approval recorded in `Execution Evidence`.
- Every live/local LLM gate must produce raw trace evidence and an
  agent-authored debug-LLM review artifact. The review artifact, not pytest
  status alone, is the evidence used for quality sign-off.

## Must Do

- Delete or rewrite the current Phase 1 implementation paths that embed
  source-specific answer logic in PM, planner, synthesizer, or tests.
- Keep Phase 0 fetching as the only source-resolution input to reading.
- Implement a generic PM/programmer reading architecture with strict context
  partitioning for local LLM constraints.
- Add broad test coverage for the query taxonomy and pass criteria in this
  plan, including independent source fixtures and negative cases.
- Add live/local LLM quality gates for PM decisions, programmer reports,
  synthesis, and end-to-end direct-interface reading. These gates are mandatory
  for Phase 1 sign-off.
- Add evidence-grounding validation so final concrete identifiers are traceable
  to programmer evidence rows.
- Update `src/kazusa_ai_chatbot/coding_agent/README.md` and
  `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md` to describe only
  the corrected final Phase 1 interface and limits.
- Add the optional `CODING_AGENT_LLM` route with fallback to
  `BACKGROUND_WORK_LLM`, including direct tests for absent, complete, and
  partial route configuration.
- Produce debug-LLM review artifacts for every required live/local LLM case,
  including raw inputs, raw outputs, parsed outputs, model route, evidence rows,
  quality notes, validation results, and pass/fail judgment.
- Run the independent plan review gate before approving or executing this
  plan.
- Run the full focused verification commands and record evidence before
  requesting independent code review.

## Deferred

- Do not integrate with L2d, action spec, background-work router, durable jobs,
  result-ready cognition, L3/dialog, dispatcher, adapters, or service startup.
- Do not implement `code_writing`, patch proposal, patch apply, or
  `code_executing`.
- Do not add compatibility shims for the failed Phase 1 PM/planner/synthesizer
  behavior.
- Do not rely on source-specific examples inside runtime prompts or planning
  constants to force a particular answer.
- Do not add broad web research or external-help calls as a normal reading
  path. Limited external help remains outside this Phase 1 implementation.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Superseded Phase 1 plans | bigbang | The superseded plan files are deleted and must not be restored. |
| Code-reading PM/planner/synthesizer | bigbang | Replace source-shaped logic with generic task taxonomy, evidence slots, programmer reports, and grounded synthesis. |
| Tests | bigbang | Rewrite weak source-shaped tests and add taxonomy, grounding, independent-fixture, and negative-case coverage. |
| LLM route | compatible | Add optional `CODING_AGENT_LLM_*` configuration. When absent, use `BACKGROUND_WORK_LLM_*`; when partially configured, fail fast. Do not add separate PM, programmer, or synthesizer routes in Phase 1. |
| Public standalone interface | compatible | Keep the existing direct `answer_code_question(...)` and `code_reading.run(...)` entrypoints. If the corrected contract cannot fit those entrypoints, stop and request a plan update before changing them. |
| Kazusa service integration | none | Leave core service, L2d, and background worker integration untouched. |

## Target State

The standalone coding agent can answer generic read-only code questions from a
successful Phase 0 repository or file source. The call path is:

```text
CodingAgentRequest
-> code_fetching.run(...)
-> CodeFetchingResult(status=succeeded)
-> code_reading.run(...)
-> reading supervisor
-> reading PM
-> repository map
-> bounded programmer assignments
-> programmer reports
-> PM sufficiency decision
-> grounded final answer
-> CodingAgentResponse
```

For non-trivial questions, the PM never receives the full repository or an
unbounded raw search dump. It creates generic assignments such as interface
reader, call-flow reader, state-lifecycle reader, dependency reader, docs
reader, or test reader. Programmer workers inspect bounded source slices and
return structured evidence. The PM synthesizes only from reports and selected
evidence rows.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| PM vocabulary | Use generic reading intent and required slots only | Keeps local LLM prompts reusable and separates decomposition from source-fact discovery. |
| Simplified IO | Use `PMInput`, `PMDecision`, `ProgrammerAssignment`, and `ProgrammerReport` with the minimal fields in this plan | Avoids fake structure and keeps local LLM context small. |
| Programmer evidence | Concrete identifiers originate only in programmer reports | Enforces product-manager/programmer memory ownership. |
| Answer synthesis | Ground final concrete code terms against selected evidence | Prevents hallucinated or memorized answer details. |
| LLM route | Add one optional `CODING_AGENT_LLM` route that falls back to `BACKGROUND_WORK_LLM` | Gives coding work a tunable model without creating unnecessary PM/programmer/synthesizer model split. |
| Test strategy | Make real LLM gates the primary product-quality proof; use deterministic tests for schema, safety, limits, route config, and regression | Phase 1 is an LLM code-reading agent, so prompt/agent behavior must be judged from real model outputs, not inferred from deterministic checks. |
| Phase 1 scope | Keep standalone direct interface only | Shrinks boundary and avoids premature Kazusa service integration. |
| Context scaling | Allow master-PM required flags and report compaction, but implement only the standalone hierarchy needed for Phase 1 | Preserves larger-project architecture without adding distributed orchestration not needed in Phase 1. |

## Contracts And Data Shapes

`CodeReadingRequest` remains the public reading input:

```python
{
    "question": str,
    "repository": CodeRepositoryRef,
    "source_scope": CodeSourceScope,
    "preferred_language": str | None,
    "max_answer_chars": int | None,
}
```

`PMInput` is the internal PM input:

```python
{
    "question": str,
    "repository_summary": dict,
    "source_scope": dict,
    "repo_map_summary": dict,
    "previous_reports": list[ProgrammerReport],
}
```

`PMDecision` is the only PM output shape:

```python
{
    "status": (
        "need_programmers"
        " | sufficient"
        " | needs_user_input"
        " | overloaded"
    ),
    "intent": str,
    "required_slots": list[str],
    "assignments": list[ProgrammerAssignment],
    "missing_slots": list[str],
}
```

Allowed generic `intent` values:

```text
architecture_overview
pipeline_or_data_flow
control_or_feedback_flow
api_or_interface_contract
symbol_behavior
state_lifecycle
dependency_usage
configuration_behavior
error_handling
test_coverage
docs_to_code_consistency
insufficient_evidence
unsupported_request
```

`ProgrammerAssignment` declares one bounded mission:

```python
{
    "assignment_id": str,
    "role": str,
    "scope": {
        "kind": "file | directory | symbol | search",
        "values": list[str],
    },
    "questions": list[str],
    "required_slots": list[str],
}
```

`ProgrammerReport` is the only programmer memory artifact passed back to PM:

```python
{
    "assignment_id": str,
    "status": "succeeded | blocked | no_evidence",
    "files_read": list[str],
    "facts": list[{
        "kind": str,
        "summary": str,
        "evidence_refs": list[str],
    }],
    "evidence": list[CodeEvidence],
    "open_questions": list[str],
}
```

Supervisor-owned deterministic limits:

```python
MAX_PROGRAMMERS_PER_WAVE = 3
MAX_PROGRAMMER_WAVES = 2
MAX_PROGRAMMER_REPORTS_PER_PM = 6
MAX_FILES_PER_PROGRAMMER = 6
MAX_EXCERPT_CHARS_PER_PROGRAMMER = 12000
```

The PM does not generate limits, forbidden-work policy, or report-shape names.
The supervisor validates assignment scope, applies limits, runs at most two
waves, and converts assignment or report overflow into `status="overloaded"`.
Because full master/subsystem PM fan-out is not implemented in Phase 1, an
overloaded result returns `needs_user_input` or a limitation instead of
pretending to answer a repository-scale question.

`CodeReadingResult.answer_text` must be synthesized from programmer reports and
evidence rows. It must not include an identifier, function, constant, file, or
module name unless the term appears in selected evidence or public repository
metadata.

## LLM Call And Context Budget

Phase 1 must remain usable with local LLMs. Use a conservative 50k-token
effective context cap for design and verification. Phase 1 adds one optional
route:

```text
CODING_AGENT_LLM_BASE_URL
CODING_AGENT_LLM_API_KEY
CODING_AGENT_LLM_MODEL
CODING_AGENT_LLM_MAX_COMPLETION_TOKENS
CODING_AGENT_LLM_THINKING_ENABLED
```

`CODING_AGENT_LLM_BASE_URL`, `CODING_AGENT_LLM_API_KEY`, and
`CODING_AGENT_LLM_MODEL` are optional as a complete group. If all three are
absent, the effective route uses `BACKGROUND_WORK_LLM_BASE_URL`,
`BACKGROUND_WORK_LLM_API_KEY`, and `BACKGROUND_WORK_LLM_MODEL`.
`CODING_AGENT_LLM_MAX_COMPLETION_TOKENS` and
`CODING_AGENT_LLM_THINKING_ENABLED` fall back to the background-work route's
corresponding values when omitted.

If any of the three required coding-agent route identity variables is present
without the other two, configuration must fail fast. Do not silently combine a
coding-agent base URL with a background-work model or API key.

Do not add `CODING_AGENT_PM_LLM`, `CODING_AGENT_PROGRAMMER_LLM`, or
`CODING_AGENT_SYNTHESIZER_LLM` in Phase 1. PM, programmer, and synthesis
boundaries are enforced by prompt/input contracts, context caps, real LLM
quality gates, and trace inspection, not by separate model routes.

The product-quality Phase 1 path must exercise real LLM calls through the
effective `CODING_AGENT_LLM` route. Deterministic code still owns validation,
path safety, caps, and public-safety enforcement, but deterministic-only
semantic decomposition or answer synthesis is not sufficient for Phase 1
acceptance.

| Stage | Normal call count | Maximum context input | Cap policy |
|---|---:|---|---|
| Reading PM decision | 1 `CODING_AGENT_LLM` call for non-trivial reading questions | `PMInput`, generic intent list, repository-map summary | No raw full files; no unbounded `rg` output. |
| Programmer worker | 1 `CODING_AGENT_LLM` call per bounded assignment | one `ProgrammerAssignment`, selected excerpts, repo-relative paths | Per-worker excerpts capped by supervisor limits. |
| PM synthesis | 1 `CODING_AGENT_LLM` call for supported answerable questions | programmer reports, selected evidence rows, limitations | No full source files; no raw checkout path. |

No implementation may use a single prompt or single unbounded context
containing the whole repository. If a question is trivial enough to answer
without an LLM call, it still must pass the required live/local LLM gate suite
before Phase 1 can be accepted as product-quality.

## Change Surface

### Delete

- No top-level production module deletion is required by this plan. Delete
  obsolete functions, constants, prompt text, or tests inside the files listed
  below only after replacement tests and callers prove they are unused.
- Superseded active Phase 1 plan files stay deleted and must not be restored.

### Modify

- `development_plans/README.md`: registry row for this final Phase 1 plan.
- `development_plans/reference/designs/coding_agent_architecture.md`: remove
  stale failed-plan links, old overdesigned PM/programmer IO, and missing
  Phase 2 or Phase 3 handoff notes.
- `README.md`: add `CODING_AGENT_LLM` to the route table as an optional coding
  work route with `BACKGROUND_WORK_LLM` fallback.
- `docs/HOWTO.md`: document optional `CODING_AGENT_LLM_*` environment
  variables, fallback behavior, and partial-config failure.
- `src/kazusa_ai_chatbot/coding_agent/README.md`: update direct interface and
  limitations after implementation.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md`: update PM,
  programmer, grounding, limits, and pass-criteria ICD.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/*.py`: replace the failed
  Phase 1 code-reading internals inside the code-reading ownership boundary.
- `src/kazusa_ai_chatbot/coding_agent/*.py`: update only direct supervisor or
  public model code needed to carry corrected reading traces and limitations.
- `src/kazusa_ai_chatbot/config.py`: add optional `CODING_AGENT_LLM_*`
  settings with fallback to `BACKGROUND_WORK_LLM_*` and fail-fast partial
  route validation.
- `src/kazusa_ai_chatbot/llm_interface/route_report.py`: include the effective
  coding-agent route in sanitized route diagnostics only if this is required
  by existing route-report tests or startup diagnostics.
- `tests/test_coding_agent_reading*.py`: replace weak tests and add broad
  taxonomy, grounding, independent-fixture, negative-case, and direct-interface
  coverage.
- `tests/test_coding_agent_live_llm.py`: update or replace with one test
  function per live/local LLM PM, programmer, synthesis, and end-to-end gate.
  Do not parameterize live LLM cases. Each case must write raw trace evidence
  for debug-LLM review.
- Tests covering configuration resolution for absent, complete, and partial
  `CODING_AGENT_LLM_*` environment variables.

### Create

- `tests/fixtures/coding_agent_reading/`: self-contained test repositories and
  independent source fixtures for taxonomy coverage.
- `tests/test_coding_agent_reading_acceptance.py`: neutral direct-interface
  acceptance coverage for representative supported reading scenarios.
- `test_artifacts/llm_traces/`: generated debug report location for live/local
  LLM diagnostics; generated artifacts remain ignored by git.
- `test_artifacts/llm_reviews/`: agent-authored debug-LLM review artifacts for
  the required live/local LLM gate suite; generated artifacts remain ignored by
  git unless the execution agent and user explicitly decide to commit a review
  summary.

### Keep

- Phase 0 `code_fetching` public contract and completed tests.
- Kazusa service, L2d, action spec, background-work router, dispatcher,
  adapters, database, and persistence code.

## Overdesign Guardrail

- Actual problem: Phase 1 code reading does not yet prove generic read-only
  code comprehension under the PM/programmer architecture.
- Minimal change: replace Phase 1 reading internals and tests with simplified
  PM/programmer IO, bounded programmer reports, evidence-grounded synthesis,
  live/local LLM quality gates, taxonomy coverage, and one optional
  coding-agent LLM route while keeping the standalone public interface.
- Ownership boundaries: Phase 0 fetches; Phase 1 reads; deterministic code
  validates paths, caps, and grounding; PM plans generic work; programmers
  discover source facts; PM synthesizes; Kazusa core service remains untouched.
- Rejected complexity: no service integration, no writing/execution agents, no
  compatibility path for failed Phase 1 internals, no broad web help, no
  source-specific runtime examples, no retry loops to repair weak answers.
- Evidence threshold: add more architecture only after this final Phase 1
  passes the live/local LLM gate suite, generic taxonomy tests, grounding
  checks, route-configuration checks, documentation review, and independent
  review.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside `coding_agent` and its
  tests as high-scrutiny changes. Existing outside modules may change only when
  this plan names them. If another outside change appears necessary, stop and
  request a plan update before editing it.
- The responsible agent may delete failed Phase 1 code paths and weak tests
  when replacement tests prove the corrected behavior.
- If equivalent path-safety, evidence, or model behavior already exists, reuse
  or move it instead of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Parent establishes the real LLM quality gate suite.
   - Create or update `tests/test_coding_agent_live_llm.py` with one test
     function per case. Do not parameterize these tests.
   - Use the exact case ids listed in `Primary Live/Local LLM Gate Suite`.
   - Each case must record raw trace evidence under
     `test_artifacts/llm_traces/` with input, effective route, model name,
     prompt or prompt-version identifier, raw model output, parsed output,
     selected evidence rows, final answer or limitation, and validation status.
   - Expected before implementation: fail, skip only because no live/local LLM
     route is configured, or produce a baseline trace showing the current
     implementation does not satisfy the case rubric. A skip caused by missing
     model configuration blocks Phase 1 approval until the user explicitly
     approves deferral.
2. Parent establishes simplified IO and supervisor-limit support tests.
   - Add tests for `PMInput`, `PMDecision`, `ProgrammerAssignment`, and
     `ProgrammerReport` validation.
   - Add tests for supervisor-owned fanout, wave, report, file, and excerpt
     limits.
   - Expected before implementation: fail because the simplified contracts or
     limit enforcement are missing or incomplete.
3. Parent establishes answer grounding and public-safety support tests.
   - Add tests proving concrete identifiers in answers are present in selected
     evidence rows or repository metadata.
   - Add tests proving public results do not expose absolute checkout paths,
     workspace roots, cache keys, secret-like files, `.env`, `.git`, binary
     payloads, or raw source dumps.
   - Expected before implementation: fail or expose missing grounding metadata.
4. Parent establishes taxonomy and direct-interface support tests.
   - Add fixture repositories covering all query categories listed in
     `Acceptance Criteria`.
   - Include at least two independent sources for common control-flow and
     data-flow categories.
   - Add direct-interface scenarios that exercise `answer_code_question(...)`
     through Phase 0 fetching and Phase 1 reading.
   - Expected before implementation: current reader fails several categories or
     cannot ground the direct-interface answer.
5. Parent establishes LLM-route support tests.
   - Add tests for absent, complete, and partial `CODING_AGENT_LLM_*`
     configuration, including fallback to `BACKGROUND_WORK_LLM_*`.
   - Expected before implementation: fail because the coding-agent route is not
     implemented or partial configuration does not fail fast.
6. Parent starts exactly one production-code subagent.
   - Subagent owns production code under `src/kazusa_ai_chatbot/coding_agent/`
     and the listed route-configuration files only.
   - Subagent must not edit tests unless parent explicitly directs a specific
     test-contract correction.
7. Production-code subagent replaces code-reading internals.
   - Implement `PMInput`, `PMDecision`, simplified `ProgrammerAssignment`,
     simplified `ProgrammerReport`, bounded assignment creation, programmer
     evidence extraction, report building, sufficiency checks, and grounded
     synthesis using the effective `CODING_AGENT_LLM` route for semantic PM,
     programmer, and synthesis stages.
   - Delete or rewrite source-shaped logic instead of preserving it.
8. Parent updates ICD documentation.
   - Rewrite coding-agent and code-reading READMEs to match the corrected
     interface, limits, traces, grounding checks, and LLM route fallback.
   - Update `README.md` and `docs/HOWTO.md` so the optional route is visible to
     operators before Phase 3 worker integration.
9. Parent runs deterministic support verification.
   - Run all commands listed in `Verification`.
   - Record outputs and limitations in `Execution Evidence`.
10. Parent runs the required live/local LLM gate suite.
   - Run one case at a time with `-q -s`.
   - Inspect each trace before running the next real LLM case.
   - Author debug-LLM review artifacts under `test_artifacts/llm_reviews/`
     from the raw traces. Scripts and tests may emit raw evidence only; the
     readable quality judgment must be written by the execution agent after
     inspecting real outputs.
   - Record pass/fail judgment, blockers, and residual quality risks in
     `Execution Evidence`.
11. Parent starts exactly one independent code-review subagent.
    - Reviewer inspects plan alignment, broad coverage, PM/programmer
      boundaries, grounding, live/local LLM review artifacts, and verification
      evidence.
12. Parent fixes review findings inside this plan's change surface.
    - Rerun affected verification before final sign-off.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs a specific correction;
  closes after planned production code changes are complete, excluding review
  fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - real LLM quality gate contract established
  - Covers: implementation step 1.
  - Verify: each named live/local LLM gate exists as a separate test function,
    writes raw trace evidence, and has a documented pass/fail rubric grounded
    in the Phase 1 code-reading contract.
  - Evidence: record baseline live/local LLM command output, trace paths, and
    current failure/skip/baseline behavior in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: complete for direction review.
- [x] Stage 2 - deterministic support test contract established
  - Covers: implementation steps 2-5.
  - Verify: focused IO, limit, grounding, public-safety, taxonomy,
    direct-interface, and LLM-route tests fail or expose the current baseline
    before production implementation.
  - Evidence: record fixture categories, commands, route cases, and baseline
    failures.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: complete for direction review.
- [x] Stage 3 - production code-reading architecture corrected
  - Covers: implementation steps 6-7.
  - Verify: focused IO, limit, grounding, public-safety, LLM-route, taxonomy,
    direct-interface, and initial live/local LLM smoke cases pass structurally.
  - Evidence: record changed production files and focused test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: complete for direction review.
- [x] Stage 4 - ICD documentation and route docs complete
  - Covers: implementation step 8.
  - Verify: README and HOWTO content match the implemented standalone
    interface, limits, public-safety rules, and LLM route fallback.
  - Evidence: record changed documentation files and doc-review notes.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: complete for direction review.
- [x] Stage 5 - deterministic support verification complete
  - Covers: implementation step 9.
  - Verify: static greps, focused tests, broader coding-agent tests, compile
    checks, and route-configuration checks pass.
  - Evidence: record all deterministic support verification outputs.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: complete for direction review.
- [x] Stage 6 - required live/local LLM gate suite complete
  - Covers: implementation step 10.
  - Verify: every required live/local LLM case is run individually, inspected
    before the next case, and recorded in an agent-authored debug-LLM review
    artifact.
  - Evidence: record commands, trace paths, review artifact paths, pass/fail
    judgments, blockers, and residual quality risks.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: complete for direction review.
- [ ] Stage 7 - independent code review complete
  - Covers: implementation steps 11-12.
  - Verify: independent review has no unresolved blockers and affected tests
    rerun after fixes.
  - Evidence: record reviewer findings, fixes, rerun commands, residual risks,
    and approval status.
  - Handoff: ready for lifecycle update after user approval.
  - Sign-off: pending.

## Verification

### Static Greps

- `rg -n "coding_agent_phase1_reading_plan|coding_agent_phase1_reading_pm_programmer_revision_plan" development_plans\README.md development_plans\reference README.md docs src tests`
  - Expected: no matches for stale failed-plan references.
- `rg -n "sufficiency_criteria|stop_conditions|expected_report_shape|master_pm_allowed|role_name|questions_to_answer|forbidden_work" src\kazusa_ai_chatbot\coding_agent\code_reading tests\test_coding_agent_reading*.py`
  - Expected: no matches, because Phase 1 uses the simplified IO contract.
- `rg -n "CODING_AGENT_PM_LLM|CODING_AGENT_PROGRAMMER_LLM|CODING_AGENT_SYNTHESIZER_LLM" src tests docs README.md`
  - Expected: no matches, because Phase 1 has exactly one optional coding-agent
    route.
- `rg -n "CODING_AGENT_LLM" README.md docs\HOWTO.md src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\coding_agent tests`
  - Expected: matches document and implement only the single optional
    `CODING_AGENT_LLM` route and its fallback to `BACKGROUND_WORK_LLM`.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_coding_agent_reading_pm_programmer.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_reading_acceptance.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_reading.py -q`
- The focused suite must include absent, complete, and partial
  `CODING_AGENT_LLM_*` route configuration cases.

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py tests\test_coding_agent_interface.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_pm_programmer.py tests\test_coding_agent_reading_acceptance.py -q`
  - Expected: deterministic support cases pass. Passing this command is not
    sufficient for Phase 1 sign-off without the real LLM gate suite below.

### Compile

- `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent`
  - Expected: all coding-agent Python files compile, including newly created
    `code_reading` modules.

### Primary Live/Local LLM Gate Suite

Run every live/local LLM case one at a time with this command shape:
`venv\Scripts\python -m pytest tests\test_coding_agent_live_llm.py::<case_id> -q -s`.
Inspect the raw trace and write or update the debug-LLM review artifact before
starting the next case. A missing live/local route, workspace, trace, or review
artifact blocks sign-off unless the user explicitly approves deferral in
`Execution Evidence`.

Required case ids:

```text
test_live_pm_decides_architecture_overview
test_live_pm_decides_pipeline_or_data_flow
test_live_pm_decides_symbol_behavior
test_live_pm_handles_ambiguous_or_too_broad_request
test_live_pm_rejects_unsupported_write_request
test_live_programmer_reads_file_scope
test_live_programmer_reads_directory_scope
test_live_programmer_reads_symbol_or_search_scope
test_live_programmer_reports_no_evidence
test_live_synthesizer_produces_grounded_answer
test_live_synthesizer_preserves_limitation
test_live_synthesizer_blocks_ungrounded_identifier
test_live_answer_code_question_pipeline_flow
test_live_answer_code_question_control_flow
```

Each case must have a durable trace and a review entry containing case id,
command, input source, effective route/model, relevant PM decision or
programmer report, selected evidence rows, raw and parsed outputs, final answer
or limitation, deterministic validation results, quality notes, and pass/fail
judgment.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the drafting
agent must reread the architecture reference, Phase 0 completed plan, this
plan, and relevant source/test context from a fresh-review posture.

Review scope:

- Architecture alignment: Phase 1 remains standalone, uses Phase 0 only for
  fetching, preserves the PM/programmer context-partitioning requirement, and
  does not touch Kazusa service integration.
- Generic reading coverage: the plan keeps source-specific fact discovery in
  programmer reports, requires independent fixtures, and requires evidence
  grounding for final concrete identifiers.
- Real LLM gate primacy: the plan treats PM, programmer, synthesis, and
  end-to-end live/local LLM cases as required Phase 1 quality gates, and does
  not allow deterministic tests to substitute for them.
- Simplified IO coverage: the plan uses the accepted minimal PM/programmer
  contracts and keeps assignment limits under deterministic supervisor
  ownership.
- LLM-route coverage: the plan adds only optional `CODING_AGENT_LLM` fallback
  to `BACKGROUND_WORK_LLM` and does not split PM/programmer/synthesizer models.
- Query coverage: the plan covers architecture, pipeline/data flow,
  control/feedback flow, API/interface contracts, symbol behavior, state
  lifecycle, dependency usage, configuration, error handling, tests,
  docs-code consistency, insufficient evidence, and unsupported requests.
- Instruction completeness: contracts, change surface, implementation order,
  verification commands, real LLM pass/fail rubrics, progress checklist,
  execution model, and acceptance criteria are specific enough for parent and
  subagent execution.
- Creativity suppression: no unresolved choices, hidden compatibility paths,
  broad helper freedom, source-shaped examples, half-done acceptance, or
  deferred-quality escape clauses remain.

Phase handoff review scope:

- Phase 1 output must remain directly mappable to Phase 3
  `BackgroundWorkResult` without exposing local roots, raw source, route
  secrets, job ids, leases, or adapter delivery fields.
- Phase 3 can pass the configured coding workspace root into the existing
  Phase 1 direct interface without parsing paths from user text.
- Phase 3 can rely on the same effective LLM route resolution; it does not need
  to introduce another coding-agent model route.
- Operator docs for `CODING_AGENT_LLM_*` must be complete before Phase 3 so
  worker deployment does not introduce undocumented configuration.
- Phase 2 code-writing can reuse the PM/programmer hierarchy, but cannot reuse
  Phase 1 reading result models as patch models without a separate writing
  plan.

Record blockers, non-blocking findings, required edits, and approval status.
Approve only when all blockers are resolved.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test,
  documentation, and command artifact.
- Code quality and design weaknesses, including PM/programmer ownership,
  hidden fallback paths, compatibility shims, prompt/context leaks, brittle
  fixtures, source-shaped planning, and avoidable blast radius.
- Live/local LLM evidence quality: every required real LLM case was run one at
  a time, inspected before the next case, and recorded with raw trace evidence
  plus an agent-authored debug-LLM review artifact.
- Source-independence quality: PM/planner/synthesis must keep decomposition
  generic; tests must include independent sources; answer concrete terms must
  be evidence-grounded.
- Simplified IO and route quality: implementation must use the accepted minimal
  PM/programmer IO and exactly one optional `CODING_AGENT_LLM` route with
  fallback to `BACKGROUND_WORK_LLM`.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including Phase 0 handoff, standalone Phase
  1 interface, README ICD updates, execution evidence, and path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface. If a finding requires a changed contract or edits
outside the approved boundary, stop and update the plan or request approval
before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The failed Phase 1 plan files are deleted and the registry and architecture
  reference point only to this final Phase 1 plan.
- `coding_agent_phase1_code_reading_final_plan.md` is the only active Phase 1
  code-reading plan.
- Phase 1 uses a PM/programmer structure where PM owns generic decomposition
  and programmers own source-specific evidence discovery.
- PM/programmer IO uses the simplified `PMInput`, `PMDecision`,
  `ProgrammerAssignment`, and `ProgrammerReport` contracts from this plan.
- Supervisor-owned limits cap reading fanout at 3 programmers per wave, 2
  waves, and 6 reports per PM. Larger requests return an overloaded limitation
  or `needs_user_input`; Phase 1 does not pretend to complete them.
- `CODING_AGENT_LLM_*` is optional. When absent, effective coding-agent LLM
  config falls back to `BACKGROUND_WORK_LLM_*`. Partial `CODING_AGENT_LLM_*`
  identity configuration fails fast.
- README and HOWTO document the optional coding-agent route and fallback.
- Phase 1 does not add separate PM, programmer, or synthesizer LLM routes.
- The product-quality path uses real LLM calls for PM decision, programmer
  report generation, and synthesis through the effective `CODING_AGENT_LLM`
  route.
- PM, planner, routing, assignment, and synthesis logic keep source-specific
  fact discovery inside programmer evidence and selected report data.
- Every concrete identifier in final answers is traceable to selected evidence
  rows or public repository metadata.
- The following query categories pass through deterministic tests:
  architecture overview, pipeline or data flow, control or feedback flow,
  API/interface contract, symbol behavior, state lifecycle, dependency usage,
  configuration behavior, error handling, test coverage, docs-to-code
  consistency, insufficient evidence, and unsupported request refusal.
- At least two independent fixtures or sources cover each common query
  category; control/feedback flow and pipeline/data-flow categories must not
  depend on one fixed source vocabulary.
- Direct-interface acceptance scenarios prove at least one control/feedback
  flow and one data/pipeline flow from source evidence.
- Required live/local LLM PM, programmer, synthesis, and end-to-end gates pass
  by agent judgment after one-at-a-time execution and trace inspection.
- Each live/local LLM gate has a durable raw trace and an agent-authored
  debug-LLM review artifact with pass/fail judgment. Pytest success alone is
  not sufficient.
- Focused tests, regression tests, static greps, and compile checks in
  `Verification` pass.
- Missing live/local LLM route, missing workspace, skipped real LLM gate, or
  missing debug-LLM review artifact blocks completion unless the user
  explicitly approves deferral in `Execution Evidence`.
- Independent code review finds no unresolved blockers.

## Execution Evidence

Record execution evidence here during implementation. Do not pre-fill checked
boxes or success claims before commands are run.

### 2026-06-20 execution start

- User explicitly approved execution with one production-code subagent pass,
  parent-owned test development and validation, parent-owned fixes after test
  failure, real LLM tests as the gating goal, no cheating/source-shaped
  shortcuts, and a stop before independent code review.
- Execution boundary: stop after implementation and validation evidence are
  ready for user direction review; do not start `Independent Code Review`.

### 2026-06-20 Stage 1 baseline

- Replaced `tests/test_coding_agent_live_llm.py` with the required one-case-per
  PM, programmer, synthesis, and direct-interface live/local LLM gate suite.
- Baseline command:
  `venv\Scripts\python -m pytest -o addopts='' -m live_llm tests\test_coding_agent_live_llm.py::test_live_pm_decides_architecture_overview -q -s`
- Baseline result: failed before any production implementation because current
  `code_reading.product_manager` does not expose the required LLM-backed
  `decide_reading_work` PM contract. This is the expected focused failure
  establishing the Stage 1 test contract.

### 2026-06-20 Stage 2 baseline

- Replaced the superseded PM/programmer test contract with simplified
  `PMInput`, `PMDecision`, `ProgrammerAssignment`, and `ProgrammerReport`
  support tests.
- Replaced broad old deterministic reading tests with public-safety,
  source-scope, repository-map, evidence-bound, route-resolution, and
  acceptance support checks.
- Removed the old signoff-query file that encoded the superseded source-shaped
  acceptance approach.
- Syntax check:
  `venv\Scripts\python -m py_compile tests\test_coding_agent_live_llm.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_pm_programmer.py tests\test_coding_agent_reading_acceptance.py`
  passed.
- Baseline command:
  `venv\Scripts\python -m pytest tests\test_coding_agent_reading_pm_programmer.py::test_contracts_define_simplified_pm_programmer_shapes -q`
- Baseline result: failed before production implementation because
  `code_reading.models` does not yet expose `PMInput` or `PMDecision`. This is
  the expected support-test failure for Stage 2.

### 2026-06-20 Stage 3 focused support result

- Production-code subagent `Hypatia` completed one production-code pass and
  reported changes under `src/kazusa_ai_chatbot/coding_agent/code_reading/`,
  `src/kazusa_ai_chatbot/config.py`, and
  `src/kazusa_ai_chatbot/llm_interface/route_report.py`.
- Parent integration added deterministic assignment validation before
  programmer evidence reads and added a pure route-resolution helper for
  absent, complete, and partial `CODING_AGENT_LLM_*` configuration tests.
- Focused support command:
  `venv\Scripts\python -m pytest tests\test_coding_agent_reading_pm_programmer.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_acceptance.py -q`
- Result: passed, 28 tests.

### 2026-06-20 deterministic verification

- Compile:
  `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent`
  passed.
- Static grep for stale failed-plan names returned no matches:
  `rg -n "coding_agent_phase1_reading_plan|coding_agent_phase1_reading_pm_programmer_revision_plan" development_plans\README.md development_plans\reference README.md docs src tests`
- Static grep for obsolete expanded Phase 1 IO terms returned no matches:
  `rg -n "sufficiency_criteria|stop_conditions|expected_report_shape|master_pm_allowed|role_name|questions_to_answer|forbidden_work" src\kazusa_ai_chatbot\coding_agent\code_reading tests -g 'test_coding_agent_reading*.py'`
- Static grep for forbidden split coding-agent LLM routes returned no matches:
  `rg -n "CODING_AGENT_PM_LLM|CODING_AGENT_PROGRAMMER_LLM|CODING_AGENT_SYNTHESIZER_LLM" src tests docs README.md`
- `CODING_AGENT_LLM` grep confirmed production, test, coding-agent README,
  route-report, README, and HOWTO references after documentation update.
- Regression command:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py tests\test_coding_agent_interface.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_pm_programmer.py tests\test_coding_agent_reading_acceptance.py -q`
- Regression result: passed, 56 tests.

### 2026-06-20 live/local LLM gate result

- Live/local route used: effective `CODING_AGENT_LLM` via
  `http://localhost:1234/v1`, model `qwen3.6-34b-80l-fable-5-heretic`.
- Required live/local LLM gates were run one case at a time with
  `venv\Scripts\python -m pytest -o addopts='' -m live_llm tests\test_coding_agent_live_llm.py::<case_id> -q -s`.
- Agent-authored review artifact:
  `test_artifacts/llm_reviews/coding_agent_phase1_live_llm_review.md`.
- The review records initial failures and fixes for unsupported-write PM
  intent, symbol representative-use evidence, over-strict dotted-identifier
  grounding, duplicate synthesis limitations, incomplete pipeline storage
  evidence, and control-flow PM/synthesis quality.
- After final production fixes, all required live/local LLM cases were rerun
  one at a time and passed:
  `test_live_pm_decides_architecture_overview`,
  `test_live_pm_decides_pipeline_or_data_flow`,
  `test_live_pm_decides_symbol_behavior`,
  `test_live_pm_handles_ambiguous_or_too_broad_request`,
  `test_live_pm_rejects_unsupported_write_request`,
  `test_live_programmer_reads_file_scope`,
  `test_live_programmer_reads_directory_scope`,
  `test_live_programmer_reads_symbol_or_search_scope`,
  `test_live_programmer_reports_no_evidence`,
  `test_live_synthesizer_produces_grounded_answer`,
  `test_live_synthesizer_preserves_limitation`,
  `test_live_synthesizer_blocks_ungrounded_identifier`,
  `test_live_answer_code_question_pipeline_flow`, and
  `test_live_answer_code_question_control_flow`.
- Latest passing trace paths are listed in the review artifact's
  `Final Current-Code Rerun` section.
- Independent code review has not been started, per the user's execution
  boundary.

### 2026-06-20 fresh pre-handoff verification

- After context compaction, reread this plan before final reporting and reran
  the handoff verification commands.
- Closed the production-code subagent `Hypatia`; its previous status reported
  production-code changes only and no test inspection.
- Fresh compile command:
  `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent`
  passed.
- Fresh deterministic regression command:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py tests\test_coding_agent_interface.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_pm_programmer.py tests\test_coding_agent_reading_acceptance.py -q`
  passed, 56 tests.
- Fresh live/local LLM gate rerun: all 14 required cases were rerun one case at
  a time with
  `venv\Scripts\python -m pytest -o addopts='' -m live_llm tests\test_coding_agent_live_llm.py::<case_id> -q -s`
  and passed. Latest trace paths are recorded in
  `test_artifacts/llm_reviews/coding_agent_phase1_live_llm_review.md`.
- Fresh end-to-end trace inspection: pipeline flow returned
  `status="succeeded"` with four evidence rows and a grounded flow through
  `decode_event`, `normalize_event`, `attach_site_context`, and
  `write_measurement`; control flow returned `status="succeeded"` with two
  evidence rows and grounded coverage of `regulate_cooling`,
  `sensor.read_temperature`, `FanController.update_speed`, `temperature_error`,
  `self.integral_error`, `derivative_error`, `fan_speed`, and
  `motor.set_speed_percent(fan_speed)`.
- Residual direction-review risk: the control-flow answer avoids the
  unsupported `PID` label but uses natural component descriptions for visible
  formula terms. This is recorded for independent code review rather than
  hidden as a completed final sign-off.
- Independent code review has still not been started, per the user's execution
  boundary.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Deterministic tests give false confidence | Make live/local LLM gates the primary quality gate and treat deterministic tests as support checks only | One-at-a-time real LLM gate suite and debug-LLM review artifacts |
| Live LLM pytest passes but output quality is weak | Require raw traces plus agent-authored quality review before sign-off | Debug-LLM review artifact with pass/fail judgment for every live case |
| Live LLM route or workspace is unavailable during execution | Treat missing route, workspace, traces, or review artifacts as blockers unless the user explicitly approves deferral | Execution Evidence records blocker or user-approved deferral |
| New tests cover too little source variety | Require independent fixtures or sources for common categories | Taxonomy tests and independent review |
| PM mixes decomposition with source fact discovery | Limit PM vocabulary to generic intent and required slots | PM/programmer contract tests and grounding tests |
| IO contract drifts back into fake structure | Keep simplified IO fields and supervisor-owned limits | Simplified-IO greps and model tests |
| LLM route creates config ambiguity | Treat partial `CODING_AGENT_LLM_*` identity config as invalid and fall back only when all identity fields are absent | Route config unit tests |
| Grounding check blocks legitimate prose | Ground only concrete code identifiers, files, symbols, constants, and module names | Grounding unit tests with natural-language explanations |
| Local LLM context overflows on larger repos | Keep PM context to repository summaries and programmer reports | Context-budget tests and trace inspection |
| Phase 1 drifts into service integration | Keep Change Surface and Deferred sections explicit | Independent review and git diff inspection |
