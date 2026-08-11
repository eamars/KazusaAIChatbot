# Plan Contract Reference

Use this reference when writing or reviewing a final executable plan. The plan
describes the intended change, its boundaries, and the roles required to
deliver it. Runtime executor selection and handoff evidence belong in
`execution_gates.md`; local implementation mechanics remain an implementation
choice.

## Contents

- [Plan lifecycle](#plan-lifecycle)
- [Filename and top matter](#filename-and-top-matter)
- [Required plan content](#required-plan-content)
- [Execution roles](#execution-roles)
- [Scope and change direction](#scope-and-change-direction)
- [Mandatory skills and rules](#mandatory-skills-and-rules)
- [No unresolved decisions](#no-unresolved-decisions)
- [Must Do and Deferred](#must-do-and-deferred)
- [Test Impact And Traceability](#test-impact-and-traceability)
- [Target state and contracts](#target-state-and-contracts)
- [Change surface](#change-surface)
- [Agent autonomy boundaries](#agent-autonomy-boundaries)
- [Verification and acceptance](#verification-and-acceptance)
- [Progress, evidence, and handoff](#progress-evidence-and-handoff)
- [Independent reviews](#independent-reviews)

## Plan lifecycle

Keep these states distinct:

- **Discovery / drafting:** inspect the current system, resolve decisions, and
  compare alternatives with the user.
- **Final plan:** fixed scope, direction, contracts, ownership, and acceptance
  criteria with no unresolved decisions.
- **Execution record:** completed work, verification evidence, deviations,
  residual risk, and lifecycle closeout.

Treat a completed plan as closed history. New scope belongs in a new or
superseding plan.

## Filename and top matter

Use a lowercase, descriptive filename, preferably in `snake_case`.

Start with a compact summary:

```md
# plan title

## Summary

- Goal:
- Status: draft | approved | in_progress | completed | superseded
- Scope boundary:
- Change direction:
- Acceptance state:
```

## Required plan content

Every final executable plan must make these subjects explicit. They may be
combined into sections that fit the work:

```md
## Summary
## Scope And Change Direction
## Mandatory Skills
## Mandatory Rules
## Must Do
## Deferred
## Target State
## Execution Roles
## Test Impact And Traceability
## Change Surface
## Agent Autonomy Boundaries
## Verification
## Acceptance Criteria
```

Add these only when the work needs them:

```md
## Confirmed Decisions
## Cutover Policy
## Contracts And Data Shapes
## Runtime Or Resource Constraints
## Progress Checklist
## Execution Evidence
## Independent Plan Review
## Independent Code Review
## Execution Handoff
```

Do not add a section merely to satisfy a template. Do not add plan-size,
line-count, character-count, or universal context-budget requirements.

## Execution roles

Declare only the roles actually needed by the plan. A role is a stable
responsibility contract, not a permanent agent name, model, count, or
conversation choreography. Use a descriptive role identifier and define all
of these fields for each role:

- **Responsibility:** the semantic or operational outcome the role owns.
- **Owned surface:** exact repository-relative files, symbols, interfaces, or
  data boundaries the role may change or operate; identify any intentional
  overlap.
- **Authority:** decisions, edits, approvals, and external effects permitted
  to the role, including explicit limits.
- **Applicable skills:** skills and references that must be available before
  the role changes its governed surface.
- **Capability floor:** minimum reasoning, domain knowledge, context, tools,
  skill access, and verification ability required to perform the role safely.
- **Independence requirement:** the separation required from other roles or
  executors, or an explicit `none` when no separation is needed.
- **Acceptance output:** artifact, decision, test evidence, review result, or
  other observable output that proves the role's work is complete.
- **Gate:** entry conditions and exit conditions that must hold before the
  role starts and before its output is accepted.

Plans leave executor resolution to runtime. An exact executor, model, or
configuration may be listed only when the user or an approving plan explicitly
provides it. Mark that value as a **plan-scoped fixed execution constraint**
and state who may approve a change. Runtime selection must then honor the
binding; it cannot be replaced silently for price, availability, latency, or
convenience.

Do not encode a permanent agent roster, model roster, fixed agent count, or
mandatory dispatch choreography. State dependencies and gates only when they
are part of the change contract. A parent may split a role into narrower
runtime handoffs inside the declared owned surface when the role's acceptance
output, authority, and gates remain intact.

Keep review and remediation authority distinct. A review or sign-off role may
inspect the work, report findings, and decide pass or fail within its gate. A
remediation role applies corrections. The role that remediates cannot provide
the independent sign-off for those corrections; require a separate independent
review handoff when remediation occurs.

## Scope and change direction

State:

- the problem or current contract only to the extent needed to identify the
  change;
- the desired end state and behavior;
- the ownership boundary;
- the included and excluded work;
- dependencies on prior or later work;
- the decisions already confirmed by the user.

Keep rationale concise and subordinate to the change direction. A plan is an
implementation contract, not a design essay or historical record.

## Mandatory skills and rules

Name each skill that governs a planned file or work area and state when it
applies. Do not copy an entire skill into the plan. Include only additional
project or task rules that the implementation agent must preserve.

Relevant skills must be available to the responsible executor before the
executor changes the governed surface, including after a handoff. The plan
can name the skill; the execution record can state how it was made available.

## No unresolved decisions

Final plans must not leave implementation agents to choose architecture,
contracts, state shapes, ownership, compatibility behavior, migration policy,
or semantic authority.

Avoid:

- `TBD` or unresolved questions;
- open alternatives or option lists;
- vague instructions such as “handle edge cases” or “add suitable tests”;
- recommendations that have not been accepted as scope.

If discovery exposes a decision that affects scope or contracts, resolve it
before approval or keep the plan in draft.

## Must Do and Deferred

`Must Do` lists fixed, in-scope outcomes. `Deferred` lists explicitly excluded
work and prevents opportunistic expansion. Both sections use direct,
observable instructions.

## Test Impact And Traceability

Every executable plan that changes production behavior, a production contract,
a caller/callee boundary, or test infrastructure that enforces production
verification contains this section. Use one row per exact source or governed
artifact path.

Each row contains:

- the repository-relative path;
- the changed symbol, field, interface, or contract;
- the semantic owner;
- one or more exact deterministic pytest node IDs;
- supplemental integration or live node IDs, or `none`;
- the test mode; and
- the observable regression prevented.

An exact node ID includes the test file and test function, for example
`tests/unit/cognition_core_v2/test_contracts.py::test_exact_contract`.
Directory paths, test-file-only references, marker/category names, and phrases
such as “relevant tests” do not satisfy this section. Every semantic owner has
a deterministic unit node; integration and live evidence is supplemental. The
matrix is part of the executable scope and must be complete before approval.

## Target state and contracts

Describe the resulting ownership, interfaces, state shapes, data flow, or
behavior needed to implement and verify the change. Add concrete schemas,
signatures, examples, or invariants when they prevent interpretation drift.

Include runtime or resource constraints only when the affected system has a
real constraint that matters to this plan. Identify the source of the
constraint and the acceptance evidence; do not import generic defaults from
this skill.

## Change surface

Name the target ownership boundary and list the expected file or interface
surface. Group paths under:

```md
### Delete
### Modify
### Create
### Keep
```

For each path or symbol, state its purpose, expected direction of change, and
why it belongs in the scope. Explain any change outside the primary boundary.
Use stable paths and symbols; line numbers are optional navigation hints.

## Agent autonomy boundaries

State what the responsible executor may decide locally and what requires a
plan amendment or user decision.

The executor may choose local implementation mechanics that preserve the
stated contracts and change surface. The executor must not introduce new
architecture, alternate migration or compatibility behavior, fallback paths,
extra features, unrelated cleanup, or semantic decisions that the plan
assigns elsewhere.

If the plan and code disagree, or the stated contract cannot be implemented
within the boundary, the executor records the conflict and requests a decision
or plan amendment. It does not silently reinterpret the plan.

## Verification and acceptance

Define observable acceptance criteria and the evidence needed to establish
them. Identify the affected contracts and the kinds of checks that matter.
Executors choose exact commands and breadth unless a command or test is itself
part of the fixed contract.

Verification should cover the changed surface and expand to integration,
system, live-service, database, migration, or review checks when the actual
change and risk require them. A full-suite run is an available choice, not a
universal plan requirement.

## Progress, evidence, and handoff

Use a progress checklist when work has multiple work items, sessions, or
handoffs. Each item should identify its outcome, owner or surface, status, and
evidence or next checkpoint. Update the item when the task is complete; keep
the checkpoint process lightweight between items.

Use `Execution Evidence` for material commands, results, artifacts, decisions,
deviations, and residual risks. For each runtime handoff, record the resolved
executor, configuration, scope, and selection rationale as specified in
`execution_gates.md`.

Use `Execution Handoff` only when work actually changes agents or sessions. A
handoff names remaining scope, owned files or interfaces, relevant skills,
completed verification, known deviations, and the next checkpoint. It also
references the role contract and runtime resolution record.

## Independent reviews

Add an independent plan review when the user requests it or when plan
ambiguity, architectural risk, or a cross-stage dependency makes it useful.

Add an independent code review when the plan, repository policy, or change risk
requires a separate review. The review checks scope alignment, ownership,
contracts, implementation quality, verification evidence, and residual risk.
The plan states the review's authority, independence requirement, and evidence
required for closure. Remediation and final sign-off remain separate role
authorities.
