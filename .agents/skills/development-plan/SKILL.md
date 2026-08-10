---
name: development-plan
description: Use for creating, reviewing, approving, executing, verifying, signing off, or handing off development plans for implementation, refactor, migration, decommission, prompt or model behavior, database work, and other consequential changes.
---

# Development Plan

Use plans as explicit change contracts. A plan fixes the intended change,
ownership, boundaries, contracts, and acceptance state. The implementation
agent chooses local mechanics and an adequate verification radius within those
boundaries.

## Planning

1. Inspect the current state, relevant source and tests, existing plans, and
   the repository's plan registry when one exists.
2. Resolve user decisions before producing an executable plan.
3. State the change direction, target ownership boundary, affected contracts,
   exact change surface, exclusions, observable acceptance criteria, and an
   exact source-to-test impact matrix.
4. Name only the skills and references relevant to the planned work. Load
   detailed references when their subject applies.
5. Review the plan for scope, ownership, contract consistency, and unresolved
   decisions before approval.

Final plans are closed work contracts. They do not contain open questions,
unaccepted alternatives, or new scope discovered during execution.

## Execution

- Execute only a plan whose lifecycle state and registry location authorize
  execution.
- The parent or implementation owner maintains scope, work-item status,
  checkpoints, evidence, and any required lifecycle updates.
- Delegation is optional and follows the current harness and model-specific
  handoff rules. This skill does not prescribe a fixed agent count, model, or
  conversation sequence.
- Every handoff states the remaining scope, owned files or interfaces,
  applicable constraints, relevant skills, current verification state, and
  next checkpoint. The receiving agent ensures the relevant skills are in
  place before changing the owned surface.
- Select implementation mechanics and verification breadth according to the
  affected contract and risk. The plan's acceptance criteria remain fixed;
  execution may choose the smallest adequate checks and expand them when the
  change radius requires it.
- If the requested work, current code, or verification exposes a contract or
  scope change, pause and resolve it through a plan amendment or user
  decision before proceeding.
- Before implementation, capture the execution baseline and the explicitly
  owned file set. Preserve pre-existing worktree changes and compare the
  execution diff against that baseline.
- Before accepting verification for a production source change, resolve every
  changed source path to the exact deterministic pytest node IDs named by the
  plan. Confirm those nodes are collected and run. A passing broader suite
  does not replace a missing or uncollected mapped node.

## References

| Reference | Use |
|---|---|
| `references/plan_contract.md` | Required structure and content for final executable plans. |
| `references/cutover_policy.md` | Behavior replacement, compatibility, rollout, or migration decisions. |
| `references/execution_gates.md` | Ownership, handoff, checkpoints, verification, evidence, and review guidance during execution. |

## Core rules

- Serve the human owner and implementation agents with one unambiguous
  contract.
- Keep change direction, ownership boundaries, contracts, exclusions, and
  acceptance criteria explicit.
- Keep semantic or architectural decisions in the plan; do not leave them to
  implementation agents as open choices.
- Require a `Test Impact And Traceability` section in every executable plan.
  Each row names one exact repository-relative source or governed artifact
  path, the changed symbol or contract, its semantic owner, exact pytest node
  IDs, test mode, and the regression prevented. Directory-only entries,
  category-only descriptions, and phrases such as "relevant tests" are not
  traceability evidence.
- Require at least one deterministic unit node for every semantic production
  owner. Integration, live-LLM, static-text, snapshot, and collection tests
  supplement the owner unit test; they do not replace it.
- Require direct owner tests and cross-boundary propagation tests when a
  change touches a caller, callee, carrier, projection, validator, reducer,
  or output boundary.
- Keep implementation agents free to choose local mechanics, decomposition,
  command order, and verification breadth that preserve the contract.
- Do not authorize new architecture, compatibility layers, fallback paths,
  helper wrappers, extra features, or unrelated cleanup unless the plan
  explicitly includes and justifies them.
- Domain-specific limits and project rules belong in the plan only when they
  are relevant to its change surface. This skill supplies no universal
  project, model, token, or context values.

## Final-plan prohibitions

Final plans must not contain unresolved questions or decision points. Avoid
`TBD`, `maybe`, `consider`, `choose one`, `option A / option B`, and
open-ended recommendations. Assumptions must be fixed operating inputs, not
disguised questions.

Final executable plans must not use vague test instructions such as "focused
tests", "relevant tests", "regression coverage", or "run the affected suite"
without an adjacent exact source-to-test row and pytest node list.

## Style

Use direct instructions, stable names and paths, explicit scope boundaries,
and observable acceptance criteria. Keep procedural detail proportional to the
risk and variability of the work.
