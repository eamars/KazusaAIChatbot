# Plan Contract Reference

Use this reference when writing or reviewing a final executable plan. The plan
describes the intended change and its boundaries; execution guidance belongs
in `execution_gates.md` and local mechanics remain an implementation choice.

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
# <lowercase plan title>

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

Relevant skills must be available to the responsible agent before the agent
changes the governed surface, including after a handoff. The plan can name the
skill; the execution record can state how it was made available.

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

State what the responsible agent may decide locally and what requires a plan
amendment or user decision.

The agent may choose local implementation mechanics that preserve the stated
contracts and change surface. The agent must not introduce new architecture,
alternate migration or compatibility behavior, fallback paths, extra features,
unrelated cleanup, or semantic decisions that the plan assigns elsewhere.

If the plan and code disagree, or the stated contract cannot be implemented
within the boundary, the agent records the conflict and requests a decision or
plan amendment. It does not silently reinterpret the plan.

## Verification and acceptance

Define observable acceptance criteria and the evidence needed to establish
them. Identify the affected contracts and the kinds of checks that matter.
Execution agents choose exact commands and breadth unless a command or test is
itself part of the fixed contract.

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
deviations, and residual risks. Record enough to support acceptance and a
future handoff, not every routine action.

Use `Execution Handoff` only when work actually changes agents or sessions. A
handoff names remaining scope, owned files or interfaces, relevant skills,
completed verification, known deviations, and the next checkpoint.

## Independent reviews

Add an independent plan review when the user requests it or when plan
ambiguity, architectural risk, or a cross-stage dependency makes it useful.

Add an independent code review when the plan, repository policy, or change risk
requires a separate review. The review checks scope alignment, ownership,
contracts, implementation quality, verification evidence, and residual risk.
The plan states the review's authority and the evidence required for closure
when this gate is used.
