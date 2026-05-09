---
name: development-plan-writing
description: Use when writing, reviewing, improving, or auditing development plans, implementation plans, refactor plans, migration plans, decommission plans, prompt or LLM pipeline change plans, database change plans, risky refactor plans, or plan-readiness reviews for AI-agent execution.
---

# Development Plan Writing

Create plans a human owner can read and an implementation agent can execute
without inventing architecture, scope, contracts, or verification.

A final development plan is an approved work contract. Discovery drafts may
contain questions and options; final plans must not. Completed plans are closed
records, so new scope belongs in a new or superseding plan.

## Workflow

1. Inspect relevant docs, source, tests, current git state, and existing plans.
2. In this repo, read `development_plans/README.md` before touching plans.
3. Resolve decisions with the user during discovery; encode the confirmed
   decision as instruction.
4. Choose plan class: `small`, `medium`, `large`, or `high_risk_migration`.
5. Load `plan_contract.md` and `execution_gates.md` for final executable
   plans; load `cutover_policy.md` when behavior changes.
6. Before approval, reread the plan against this skill and loaded references.

## References

| Read | When |
|---|---|
| `references/plan_contract.md` | Always for final executable plans. |
| `references/cutover_policy.md` | Behavior changes, legacy removal, data migration, or rollout strategy. |
| `references/execution_gates.md` | Always for final executable plans; covers granular steps, verification, evidence, handoff, and review gates. |

## Local Registry Rule

`development_plans/README.md` owns lifecycle policy:

- `long_term/`: roadmap only; promote before implementation.
- `active/short_term/`, `active/bugfix/`: execute only `approved` or
  `in_progress` plans.
- `archive/completed/`: historical records; do not append scope.
- `archive/superseded/`: do not execute.
- `reference/`: context only.
- `triage/`: blocked until classified and moved.

## Core Rules

- Satisfy two audiences: human owner first, implementation agent second.
- Name mandatory skills explicitly; copy critical skill-derived rules into the
  plan because context may compact.
- Include `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`,
  `Implementation Order`, `Progress Checklist`, `Verification`, and
  `Acceptance Criteria`.
- Break execution into granular, verifiable steps instead of broad work items.
- Define cutover policy as `bigbang`, `migration`, or `compatible` for every
  changed behavior area, and confirm it before finalizing.
- Name the target module or ownership boundary. Strongly justify any change
  outside it.
- For new modules, define public interface, data shapes, ownership, callers,
  hidden internals, test seams, integration points, and tests before finalizing.
- For prompt, graph, RAG, cognition, dialog, evaluator, background LLM,
  database, migration, or production-path changes, include the relevant budget
  and safety gates from the references.
- Do not authorize new architecture, compatibility layers, fallback paths,
  helper wrappers, extra features, or unrelated cleanup unless explicitly in
  scope and justified.

## Final-Plan Prohibitions

Final plans must not contain unresolved questions or decision points. Avoid
`TBD`, `maybe`, `consider`, `choose one`, `option A / option B`,
`ask the user whether...`, and open-ended recommendations.

Assumptions are allowed only when they are fixed operating inputs, not disguised
questions.

## Style

Use direct instructions, stable names and paths, explicit scope boundaries, and
exact verification gates. Avoid hidden decisions, ambiguous safety language,
stale line-number-only references, and unaccepted recommendations.
