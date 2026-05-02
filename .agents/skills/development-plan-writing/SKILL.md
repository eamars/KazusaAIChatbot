---
name: development-plan-writing
description: Write, review, or improve execution-ready development plans for multi-file changes, migrations, decommissions, architecture changes, prompt/LLM pipeline changes, database changes, or risky refactors. Use this skill whenever the user asks for a development plan, implementation plan, refactor plan, migration plan, decommission plan, or asks you to evaluate whether a plan is clear enough for an AI agent to execute. The plan must serve both a human reader and an implementation agent, resolve decisions before finalizing, state cutover policy explicitly, and prevent agents from inventing scope.
---

# Development Plan Writing

Use this skill to create development plans that are pleasant for a human to read and precise enough for an AI implementation agent to follow without making architectural decisions on its own.

A final development plan is not a brainstorming document. It is an approved work contract.

Completed plans are closed records. New scope must go into a new or superseding plan, never appended to the completed one.

## Core Standard

Every final plan must satisfy two audiences:

- **Human owner:** can quickly understand the goal, risk, strategy, scope, and evidence for completion.
- **Implementation agent:** can follow exact technical instructions, file paths, contracts, order, forbidden changes, and verification gates without inventing missing decisions.

If those audiences conflict, split the writing into human-readable rationale first and agent-executable instructions second. Do not bury decisions only inside prose.

## Module-First Planning

When a plan introduces a new feature, function group, subsystem, service boundary, data pipeline, or behavior that can reasonably be isolated, aim to design it as a module.

This rule applies to new work. It does not force modularization for narrow bug fixes, refactors of existing code, or small improvements where the existing module already owns the behavior.

For every new module, define the interface before finalizing the plan:

- public functions, classes, methods, protocol, message shape, schema, CLI, endpoint, or event contract
- input and output data shapes
- ownership boundary: what the module owns and what existing code owns
- allowed callers/importers from the existing codebase
- hidden internals that callers must not reach into
- dependency injection or test seams
- integration points with existing modules
- focused tests that prove the module works independently
- integration tests that prove the existing code can call it correctly

The final plan must not leave the module interface for the implementation agent to invent. If the interface affects architecture or future maintainability, get explicit user agreement during discovery before marking the plan final.

## Filename Rule

Development plan filenames must be all lowercase.

Prefer snake_case:

```text
rag1_decommission_plan.md
cache2_invalidation_migration_plan.md
```

Do not use uppercase letters in plan filenames:

```text
RAG1_decommission_plan.md
RAG_SUPERVISOR2_PLAN.md
```

## Plan Lifecycle

Keep these stages separate:

- **Discovery / Drafting:** questions, options, code inspection, tradeoffs, and user confirmation are allowed here.
- **Final Development Plan:** no unresolved questions, no decision prompts, no alternatives left for the implementation agent.
- **Execution Record:** what was done, what passed, what failed, and evidence.

Do not mix final-plan content with unresolved design discussion.

Treat completed plans as closed historical records. After a plan reaches `completed`, do not add newly requested work, follow-up ideas, expanded scope, or changed requirements to that completed plan. New work belongs in a new plan, a superseding plan, or a separate follow-up plan with its own status and checklist.

## Required Top Matter

Start each plan with a compact human-readable summary:

```md
# <lowercase plan title>

## Summary

- Goal:
- Plan class:
- Status:
- Mandatory skills:
- Overall cutover strategy:
- Highest-risk areas:
- Acceptance criteria:
```

Use one of these status values:

```text
draft | approved | in_progress | completed | superseded
```

Use one of these plan classes:

```text
small | medium | large | high_risk_migration
```

## Length Budget

Choose the plan class before writing.

| Plan class | Typical scope | Target length | Maximum length |
|---|---|---:|---:|
| small | 1-2 files, low risk | 80-150 lines | 200 lines |
| medium | several files, no data migration | 150-300 lines | 400 lines |
| large | many files, contracts/prompts/tests | 300-600 lines | 800 lines |
| high_risk_migration | deletion, DB work, production behavior change | 500-900 lines | 1200 lines |

If a plan exceeds the target, compress repetition, move examples to appendices, or reference existing docs. Do not remove mandatory rules, scope, contracts, cutover policy, implementation order, or verification gates just to meet the length budget.

## Mandatory Sections

Every final plan must include these sections:

```md
## Summary
## Context
## Mandatory Skills
## Mandatory Rules
## Must Do
## Deferred
## Cutover Policy
## Agent Autonomy Boundaries
## Target State
## Design Decisions
## Change Surface
## Implementation Order
## Progress Checklist
## Verification
## Acceptance Criteria
```

Add these sections whenever relevant:

```md
## Data Migration
## Risks
## LLM Call And Context Budget
## Operational Steps
## Execution Evidence
## Glossary
```

## Mandatory Skills

Each plan must explicitly name every skill the implementation agent is required to load before making changes. Do not rely on the agent inferring mandatory skills from the repo, the task, memories, or surrounding conversation.

Use a short, concrete list:

```md
## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing prompt, graph, RAG, memory, cognition, dialog, evaluator, or background LLM behavior.
```

If no specialized skill is required, state that explicitly:

```md
## Mandatory Skills

- No specialized skill is required for this plan.
```

For plans that touch multiple domains, list the skills in the order the agent should load them and state which stage each skill governs. The plan must also copy the critical skill-derived rules into `Mandatory Rules`; naming a skill is not enough because implementation agents may lose context after compaction.

## Mandatory Rules

Each plan must explicitly state the project-specific rules the agent must follow. Do not rely on the agent loading skills, memories, or repo conventions efficiently.

Include rules such as:

- coding style rules
- test execution rules
- plan reread rules after automatic context compaction and major checklist sign-off
- prompt safety rules
- trusted vs untrusted prompt boundaries
- LLM call and context budget rules for prompt, agent, RAG, cognition, dialog, evaluator, or background LLM changes
- database safety rules
- forbidden filtering or validation patterns
- target-module boundary rules for changes outside the primary module
- required skill-derived practices, copied into the plan

Write the rules directly in the plan. It is acceptable and often desirable to duplicate important skill content here.

Every final plan must include these plan-continuity rules in `Mandatory Rules`:

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.

## No Unresolved Questions

Final plans must not contain unresolved questions or decision points.

Avoid these in final plans:

- `TBD`
- `maybe`
- `consider`
- `choose one`
- `option A / option B`
- `ask the user whether...`
- open-ended recommendations

Resolve uncertainty before finalizing the plan. If a decision depends on code inspection, inspect the code first. If a decision depends on user preference, ask the user during discovery, then encode the confirmed decision as an instruction.

Assumptions are allowed, but they must be fixed operating inputs:

```md
## Assumptions

- RAG2 is the only supported retrieval path after this refactor.
- Compatibility with the RAG1 state shape is intentionally not preserved.
- Legacy MongoDB collections may be dropped through the approved migration path.
```

Do not write assumptions as disguised questions.

For new modules, an unapproved or undefined public interface counts as an unresolved question. Do not finalize the plan until the module boundary and interface are accepted by the user.

## Must Do

The `Must Do` section defines non-negotiable scope.

Use directive language:

```md
## Must Do

- Replace all `research_facts` consumers with `rag_result`.
- Delete RAG1 modules listed in this plan.
- Add projection and invalidation tests listed in this plan.
- Run every verification command in the Verification section.
```

The implementation agent must not downgrade, reinterpret, or skip these items.

## Deferred

The `Deferred` section defines explicit non-scope.

Use directive language:

```md
## Deferred

- Do not redesign RAG2 helper-agent routing.
- Do not add persistent Cache2 storage.
- Do not create compatibility shims for the old RAG1 state shape.
- Do not refactor unrelated prompt architecture.
```

The implementation agent must not opportunistically do deferred work, even if it looks useful.

## Agent Autonomy Boundaries

Add a section that constrains implementation-agent judgment.

Recommended language:

```md
## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The agent must treat changes outside the target module as high-scrutiny changes. Updating an existing module outside the target module, or introducing a new code path, prompt, or variable, requires strong justification in the plan before implementation.
- The agent may remove code from the existing codebase with lighter justification when the removal is explicitly in scope and verified by references, greps, and tests.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors unless explicitly listed in Must Do.
- If the plan and code disagree, the agent must preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the agent must stop and report the blocker instead of inventing a substitute.
```

## Cutover Policy

Every plan that changes existing behavior must define cutover policy. For broad plans, use a policy matrix per area.

Each area must be marked exactly one of:

- `bigbang`
- `migration`
- `compatible`

The policy must be confirmed with the user before finalizing the plan.

For strategy definitions, a policy-matrix template, and enforcement language, read `references/cutover_policy.md`.

## Context And Target State

Describe the current state, target state, and why the change exists.

Good context includes:

- the user request or product/problem pressure driving the change
- concrete failure mode, missing capability, maintenance burden, risk, or workflow pain that makes the change necessary now
- old architecture and new architecture
- exact state/data shape changes
- production vs test-only status
- known consumers
- external systems affected
- why legacy behavior is being removed or preserved
- relevant previous attempts, completed plans, incidents, or code comments if they explain constraints
- adjacent improvement areas discovered during planning that are intentionally deferred

Target state should describe observable end behavior, not just files changed.

When listing adjacent improvement areas, keep them concise and non-authorizing. They are context for future planning, not permission for the implementation agent to expand scope:

```md
## Context

This change is driven by ...

Adjacent improvement areas intentionally left for later plans:

- ...
- ...
```

## Design Decisions

Use a decision table for meaningful choices:

```md
## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Cache invalidation | Use Cache2 dependency events | Cache1 version counters are removed. |
| State payload | Use hybrid `rag_result` | Structured image data is needed, raw search blobs are too large. |
```

Only include settled decisions. Do not list alternatives unless the alternative is clearly rejected and the rejection helps prevent future agent drift.

## Contracts And Data Shapes

For architecture or pipeline changes, define the new contracts explicitly:

- function signatures
- state keys
- payload shape
- input/output cardinality
- ownership boundaries
- refusal or failure conditions
- latency or call-count expectations, if relevant

Prefer precise examples:

```python
{
    "answer": str,
    "user_image": dict,
    "conversation_evidence": list[str],
    "supervisor_trace": {
        "loop_count": int,
        "unknown_slots": list[str],
    },
}
```

Define forbidden compatibility shapes when relevant:

```md
Do not preserve or recreate the legacy `research_facts` / `research_metadata` payload.
```

For new modules, include a dedicated interface contract. The interface can be code signatures, protocol messages, schemas, endpoint shapes, CLI commands, event payloads, or another concrete boundary appropriate to the codebase. The contract must be specific enough that existing code can integrate with the module without importing internals.

## LLM Call And Context Budget

If a plan adds, removes, rewires, or changes any prompt, model call, agent graph, evaluator, RAG/cognition/dialog stage, or background LLM job, include a before/after LLM budget before finalizing.

Use `50k tokens` as the default context-window cap unless the user explicitly sets another cap.

For each affected LLM call, state the before/after call count, whether it is response-path or background, the model/helper used if known, the context inputs before and after, the estimated maximum context use versus the cap, latency impact, blocking behavior, hard caps, truncation/drop policy, and verification tests. If exact tokenization is unavailable, use conservative character-based estimates and state the method. New response-path calls or cap increases require explicit user approval.

## Change Surface

Separate files into clear groups:

```md
## Change Surface

### Delete
### Modify
### Create
### Keep
```

For each path, explain why it is in that group. Use stable file paths and symbols. Line numbers may be included as hints, but never rely on line numbers alone.

When creating a new module, list the module's public entrypoint separately from its internals. Existing code should depend on the public entrypoint, not on private storage, prompt, cache, or helper files.

Every plan must name the target module or target ownership boundary. Any change outside that boundary must include strong justification in `Change Surface` or `Design Decisions`, especially when it updates an existing module or introduces a new code path, prompt, or variable. Strong justification means the plan explains why the target module cannot own the behavior, why the outside change is necessary for the approved contract, and how tests or greps will prove the change stayed bounded.

Code removal is less constrained by this rule when the removal is already in scope. For delete-only work, require enough evidence to show the deleted code is obsolete or unreferenced, but do not force the same level of justification required for adding or expanding behavior outside the target module.

## Implementation Order

Implementation order must prevent avoidable breakage and agent improvisation.

Default to a module-first, test-first order. The plan should prove the module contract before implementation, then prove integration after the module is stable. For LLM, database, migration, or production-path changes, include the focused module test and the relevant integration, real LLM, real database, migration dry-run, or smoke test needed to cover the actual risk.

Every final plan must include this general sequence unless a step is truly inapplicable:

1. Add or update module tests.
   - Name the exact test file, test function, fixture, or diagnostic that proves the target module contract.
   - Run the module test before implementation when applicable and record the expected failure, missing symbol, missing entrypoint, or baseline result.
2. Add or update integration tests.
   - Name the exact integration, real LLM, real database, migration, smoke, or cross-module test that proves callers can use the module correctly.
   - Run the integration test before implementation when practical and record the failure mode or current behavior.
3. Implement the module.
   - Keep the behavior inside the target module and approved public interface whenever possible.
   - Do not update existing outside modules or introduce new code paths, prompts, or variables without the strong justification required in `Change Surface`.
4. Run module tests.
   - Re-run the same module test from step 1.
   - Record the result in `Execution Evidence`.
5. Loop back to step 1 if needed.
   - If the module test fails or reveals an incomplete contract, update the module test or module implementation only to match the approved plan, then repeat steps 1, 3, and 4 before touching integration.
6. Implement integration.
   - Wire existing callers, adapters, prompts, database paths, or service entrypoints to the module only after module tests pass.
   - Keep integration edits inside the approved change surface.
7. Run integration tests.
   - Re-run the same integration test from step 2.
   - Then run any broader verification gates listed in `Verification`.
   - Record the result in `Execution Evidence`.
8. Loop back to step 1 if needed.
   - If integration exposes a module contract problem, return to module tests before changing integration again.
   - If integration exposes only wiring behavior, update integration and re-run the integration test.

For low-risk documentation-only plans, explicitly state that the module/integration test-first sequence is not applicable and replace it with a before/after review gate.

Additional ordering guidance:

- create the new contract first
- rewrite consumers next
- wire the new entrypoint
- delete legacy modules after references are gone
- run greps and smoke tests last

Include a short rationale when order matters:

```md
Build the projection module first because it becomes the contract used by cognition, consolidation, and tests.
```

## Progress Checklist

Every final plan must include a tickable progress checklist using Markdown checkbox syntax (`- [ ]` or `1. [ ]`). These are the progress boxes implementation agents update as each function, module, integration step, or sign-off gate is completed. Small plans may have a short checklist, but they still need one so progress and handoff state are visible.

Checkpoints exist so multiple agents can resume the work without rediscovering state or guessing which partial changes are complete. They should be granular enough that an agent can finish one checkpoint, verify it, mark it complete, and hand off cleanly.

Each tickable checkpoint must describe:

- the stage, function, module, interface, integration, or sign-off gate being completed
- the files or modules expected to be touched
- the verification commands or static checks to run before ticking the box
- the evidence that must be recorded before ticking the box
- the next checkpoint or next implementation step
- the sign-off line the agent must complete after that checkpoint is done

```md
## Progress Checklist

- [ ] Stage 1 — module contract established
  - Covers: steps 1-3.
  - Verify: `python -m py_compile ...`; focused module tests pass.
  - Evidence: record changed files and test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 — service integration complete
  - Covers: step 4.
  - Verify: service integration tests pass.
  - Evidence: record test output before moving on.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
```

Do not treat checked boxes as proof by themselves. An agent may tick a checkpoint only after running its verification and recording the result in `Execution Evidence` or a linked execution record. If verification is skipped or blocked, leave the box unchecked and record why.

Agents must sign off stages one at a time, immediately after each stage is genuinely complete. Do not pre-fill sign-offs, do not sign off future stages, and do not batch multiple stage sign-offs at the end of a session. If handing off mid-plan, leave all unfinished stages unchecked and unsigned, and add a brief handoff note that points to the next unchecked stage.

After signing off any major checklist stage, the active agent must reread the entire plan before starting the next stage. This reread requirement must be written into the plan's `Mandatory Rules` and treated as part of each major stage sign-off gate.

## Verification

Verification must be written as gates, not vague reminders.

Include exact commands or checks:

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

## Acceptance Criteria

Acceptance criteria describe the observable completed state.

Example:

```md
## Acceptance Criteria

This plan is complete when:

- There is exactly one active RAG path.
- Legacy RAG1 modules and Cache1 modules are deleted.
- No source file imports Cache1 or RAG1.
- New tests for projection and Cache2 invalidation pass.
- Legacy MongoDB collections are absent through the approved migration path.
```

## Risks

Use a compact risk table:

```md
## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Stale cache after conversation save | Emit Cache2 event in `save_conversation` | Cache invalidation test and live smoke |
```

## Execution Evidence

Do not treat checked boxes as proof. If the plan also records completion, add a separate evidence section:

```md
## Execution Evidence

- Static grep results:
- Test results:
- Service boot:
- DB verification:
- Manual smoke:
```

Pre-execution plans should use unchecked checklist items. If a plan is completed, either move checked items into an execution record or attach evidence for the checks.

Never document scope creep in a completed plan. Once complete, the plan may receive only factual corrections to execution evidence or links to a new/superseding plan. If new requirements appear after completion, keep the completed plan unchanged and create a separate follow-up plan.

## Writing Style

Write for a smart human and a literal implementation agent.

Use:

- direct instructions
- stable names and paths
- short rationale before long checklists
- explicit scope boundaries
- exact verification gates

Avoid:

- hidden decisions in prose
- ambiguous safety language
- optimistic phrases without tests
- stale line-number-only references
- long duplicated checklists when a table would be clearer
- recommendations that are not accepted instructions

The best development plans make the correct path easy and the wrong path obviously out of bounds.
