# coding agent phase2.5 security boundary plan

## Summary

- Goal: enforce the coding-agent agent-space security boundary before runtime
  integration, so generated artifacts cannot trigger real-world execution or
  mutation without a separately approved capability.
- Plan class: high_risk_migration
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  and `test-style-and-execution`.
- Overall cutover strategy: bigbang security cleanup for coding-agent
  validation and tool-call boundaries.
- Highest-risk areas: generated-code execution, raw command payloads,
  workspace mutation, hidden subprocess paths, and unclear tool ownership.
- Acceptance criteria: Phase 2.5 is complete when coding-agent validation and
  tests prove generated code, tests, commands, and scripts stay inert artifacts
  unless a later approved execution capability owns isolation and audit.

## Context

The coding-agent architecture separates agent-space artifact generation from
real-world execution. Phase 2 intentionally focuses on new-artifact proposals.
Execution, mutation, and patch application belong to later approved phases.

Phase 2.5 regulates the security boundary before Phase 3 integration. It owns
the cleanup of validation and test paths that could execute generated artifacts
or accept raw executable payloads.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing coding-agent role boundaries,
  tool capability contracts, or prompt-facing tool descriptions.
- `py-style`: load before editing Python production files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After automatic context compaction, the active agent must reread this entire
  plan before continuing implementation, verification, handoff, lifecycle
  updates, or final reporting.
- After signing off any major checklist stage, the active agent must reread
  this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the code-review gate defined by this plan and record the result in
  `Execution Evidence`.
- Agent stages may generate only structured tool-call intents, proposed
  artifacts, traces, and review records.
- Tool calls must target approved agent-space capabilities and must not accept
  raw executable payloads.
- Generated code, tests, commands, and scripts must remain inert artifacts.
- Real-world execution, package installation, network access, database writes,
  service startup, workspace mutation, patch application, and arbitrary shell
  access require a separate approved capability with isolation, permission, and
  audit.

## Must Do

- Audit coding-agent paths reachable from Phase 2 artifact generation,
  validation, review, and tests.
- Remove generated-artifact execution from Phase 2 validation paths.
- Replace execution-based generated-artifact checks with non-executing
  artifact inspection, structural validation, and AI-authored review.
- Define the coding-agent tool-call interaction guideline in the code-writing
  README or equivalent ICD.
- Add verification that coding-agent tests and validation paths do not execute
  generated code, generated tests, generated commands, or generated scripts.
- Keep Phase 2 E2E pass/fail focused on artifact quality and workflow
  correctness while this plan owns security remediation.

## Deferred

- Do not implement `code_executing`.
- Do not implement patch application to a real checkout.
- Do not add Docker execution, package installation, service startup, database
  writes, or network-backed execution.
- Do not broaden Phase 2 new-artifact scope.
- Do not add compatibility shims for old tool-call or validation shapes.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Validation boundary | bigbang | Remove execution of generated artifacts from coding-agent validation paths. |
| Tool-call contract | bigbang | Use one agent-space tool-call guideline with no raw executable payloads. |
| Tests | bigbang | Replace security-sensitive execution assumptions with non-executing checks and AI review evidence. |

## Target State

Coding-agent Phase 2 outputs are proposed artifacts and reviewable metadata.
Validation may inspect artifact shape, paths, caps, parseability, public-safe
metadata, and workflow handoff records. Validation does not run generated code
or generated tests.

Later execution-capable phases must introduce a dedicated capability with an
explicit interface, isolation boundary, permission model, and audit trail.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Agent-space boundary | Treat generated artifacts as inert data | Keeps LLM output separate from real-world effects. |
| Phase ownership | Phase 2.5 owns security remediation | Phase 2 remains focused on new-artifact generation quality. |
| Validation style | Prefer inspection and AI review over execution | Preserves real LLM gate intent without crossing the execution boundary. |
| Future execution | Require a separate approved capability | Prevents accidental shell or test execution through validation helpers. |

## Change Surface

### Modify

- `development_plans/reference/designs/coding_agent_architecture.md`: keep the
  security boundary as the architecture source of truth.
- `development_plans/active/short_term/coding_agent_phase2_code_writing_plan.md`:
  keep Phase 2 pass/fail wording aligned with this Phase 2.5 handoff.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/README.md`: document the
  code-writing agent-space tool-call boundary.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/patch_validation.py`:
  remove generated-code execution from validation.
- `tests/test_coding_agent_phase2_new_artifact_contracts.py`: add
  non-executing boundary checks.
- `tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py`: keep E2E
  evidence collection aligned with inert artifact review.

### Keep

- Phase 2 role prompts and new-artifact role contracts, except where wording
  exposes execution-capable tool semantics.
- Phase 2 hard-gate challenge definitions.

## Overdesign Guardrail

- Actual problem: coding-agent validation must not let generated artifacts
  create real-world effects before an approved execution capability exists.
- Minimal change: remove generated-artifact execution from Phase 2 validation
  and document one agent-space tool-call boundary.
- Ownership boundaries: LLM stages own semantic artifact generation;
  deterministic code owns structure, limits, path safety, metadata validation,
  and audit; execution-capable work belongs to a later approved capability.
- Rejected complexity: no sandbox runner, Docker runner, package installer,
  patch applier, service launcher, or command executor in this plan.
- Evidence threshold: execution support requires a separate approved plan with
  isolation, permission, audit, and dedicated tests.

## Agent Autonomy Boundaries

- The active agent may change only the files listed in `Change Surface` unless
  the user explicitly expands scope.
- The active agent must not add execution, mutation, package, service, network,
  database, or patch-apply capabilities.
- The active agent must not weaken Phase 2 anti-cheat or real LLM review rules.
- If a validation path cannot prove artifact quality without execution, the
  active agent must record the limitation and keep the generated artifact
  inert.

## Implementation Order

1. Reread this plan, the architecture reference, and the Phase 2 plan.
2. Audit coding-agent validation and E2E paths for generated-artifact
   execution.
3. Update ICD wording for the agent-space tool-call boundary.
4. Remove generated-artifact execution from validation.
5. Update deterministic support checks for non-executing validation.
6. Update live LLM E2E review procedure to record artifact paths for human and
   AI review without executing generated artifacts.
7. Run verification and record evidence.
8. Run independent code review.

## Execution Model

- Execute this plan single-agent unless the user later approves subagents.
- The active agent owns orchestration, tests, verification, execution evidence,
  review remediation, lifecycle updates, and final sign-off.
- Live LLM tests must run one case at a time with output inspected.
- Deterministic tests may verify security boundaries, structural validation,
  path safety, and absence of generated-artifact execution.

## Progress Checklist

- [ ] Stage 1 - audit and contract alignment
- [ ] Stage 2 - validation boundary cleanup
- [ ] Stage 3 - tests and review procedure update
- [ ] Stage 4 - verification
- [ ] Stage 5 - independent code review

## Verification

- `rg "subprocess|pytest|python -m|Start-Process|Invoke-Expression" src/kazusa_ai_chatbot/coding_agent tests`
  - Expected: no generated-artifact execution path remains in coding-agent
    validation or Phase 2 E2E tests. Approved unrelated matches must be
    documented in `Execution Evidence`.
- `venv\Scripts\python -m pytest -q tests/test_coding_agent_phase2_new_artifact_contracts.py`
- Run one Phase 2 E2E live LLM gate after Stage 4 evidence is complete and
  inspect the generated artifact paths without executing generated code.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Because the user currently prohibits Codex subagents, perform the review from
a single-agent independent-review posture unless the user later approves a
separate reviewer.

Review scope:

- Alignment with the architecture security boundary.
- Absence of generated-artifact execution in Phase 2 validation paths.
- Clear separation between agent-space artifacts and real-world effects.
- No new execution, mutation, install, network, database, service, or patch
  application capability.
- Test and evidence quality.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The architecture, ICD, code, and tests share one agent-space security
  boundary.
- Phase 2 validation does not execute generated code, generated tests,
  generated commands, or generated scripts.
- Coding-agent tests prove the non-executing boundary.
- Phase 2 E2E review artifacts identify generated artifact paths for
  inspection without generated-code execution.
- Independent code review finds no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Hidden execution remains in validation | Static grep plus focused review of validation call paths | Verification grep and code review |
| Artifact quality becomes harder to judge | Use AI-authored review plus direct artifact inspection | Live LLM review artifacts |
| Boundary wording drifts across docs | Keep architecture as source of truth and align README/plan wording | Plan review and grep |

## Execution Evidence

- Pending implementation.
