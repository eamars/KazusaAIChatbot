---
name: development-plan-writing
description: Use when writing, reviewing, improving, or auditing development plans, implementation plans, refactor plans, migration plans, decommission plans, prompt or LLM pipeline change plans, database change plans, risky refactor plans, or plan-readiness reviews for AI-agent execution.
---

# Development Plan Writing

Create work-contract plans a human owner can approve and an implementation
agent can execute without inventing architecture, scope, contracts, or
verification. Drafts may contain questions; final and completed plans must not
collect new scope.

## Workflow

1. Inspect relevant docs, source, tests, current git state, and existing plans.
2. In this repo, read `development_plans/README.md` before touching plans.
3. Resolve decisions with the user during discovery; encode the confirmed
   decision as instruction.
4. Choose plan class: `small`, `medium`, `large`, or `high_risk_migration`.
5. Load `plan_contract.md` and `execution_gates.md` for final executable
   plans; load `cutover_policy.md` when behavior changes.
6. Before approval, reread the plan against this skill and loaded references.
7. When requested, run the optional independent plan review gate before
   approval, execution, or multi-stage handoff.
8. For executable plans, include a final independent code review gate.

## References

| Read | When |
|---|---|
| `references/plan_contract.md` | Always for final executable plans. |
| `references/cutover_policy.md` | Behavior changes, legacy removal, data migration, or rollout strategy. |
| `references/execution_gates.md` | Always for final executable plans; covers steps, verification, evidence, handoff, and review gates. |

## Local Registry Rule

`development_plans/README.md` owns lifecycle policy. Execute only `approved`
or `in_progress` plans under `active/`; use the registry for all other folders.

## Core Rules

- Serve two audiences: human owner first, implementation agent second.
- Name mandatory skills explicitly; copy critical skill-derived rules into the
  plan because context may compact.
- Include `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`,
  `Implementation Order`, `Progress Checklist`, `Verification`,
  `Independent Code Review`, and `Acceptance Criteria`.
- Break execution into granular, verifiable steps instead of broad work items.
- Define cutover policy as `bigbang`, `migration`, or `compatible` for every
  changed behavior area, and confirm it before finalizing.
- Name the target module or ownership boundary. Strongly justify any change
  outside it.
- For new modules, define public interface, data shapes, ownership, callers,
  internals, test seams, integration points, and tests before finalizing.
- For prompt, graph, RAG, cognition, dialog, evaluator, database, migration,
  or production-path changes, include relevant budget and safety gates.
- Do not authorize new architecture, compatibility layers, fallback paths,
  helper wrappers, extra features, or unrelated cleanup unless explicitly in
  scope and justified.
- Use the optional independent plan review gate when the user asks for plan
  approval, creativity-tightening, stage-boundary review, architecture
  alignment review, or handoff review.
- Require an independent code review stage after implementation verification
  and before final sign-off. The review must check style compliance, code
  quality, plan alignment, design weaknesses, regression risk, handoff
  artifacts, and stale or inaccurate verification gates. Keep the detailed
  review contract in the references.

## Final-Plan Prohibitions

Final plans must not contain unresolved questions or decision points. Avoid
`TBD`, `maybe`, `consider`, `choose one`, `option A / option B`, and open-ended
recommendations. Assumptions must be fixed operating inputs, not disguised
questions.

## Style

Use direct instructions, stable names and paths, explicit scope boundaries, and
exact verification gates. Avoid hidden decisions, ambiguous safety language,
stale line-number-only references, and unaccepted recommendations.
