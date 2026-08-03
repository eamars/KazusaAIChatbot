# Execution Guidance Reference

Use this reference to execute an approved plan while preserving its change
contract. It provides decision boundaries, not a mandatory command sequence or
agent choreography.

## Ownership

The plan identifies the parent or implementation owner, the change surface,
and any delegated ownership. The responsible agent owns the implementation
within that boundary. The parent or designated owner keeps the plan status,
checkpoints, evidence, and lifecycle record coherent.

Delegation is optional. When used, follow the current harness and model
handoff protocol. Keep delegated write scopes explicit and non-overlapping when
parallel work is used. A reviewer has review authority only unless the plan
explicitly grants a bounded remediation scope.

## Handoff

Before handing work over, record:

- the approved plan and current scope;
- completed and remaining work;
- owned files, interfaces, or data boundaries;
- applicable constraints and relevant skills;
- verification already run and its result;
- known deviations or blockers;
- the next checkpoint.

The receiving agent ensures the relevant skills and references are available
before changing the governed surface. It confirms the next checkpoint, works
within the stated boundary, and updates the checkpoint and evidence when the
task is complete.

## Change decisions during execution

The plan fixes intent, contracts, ownership, and exclusions. The agent may
choose local mechanics, decomposition, command order, and verification breadth
that preserve them.

When implementation reveals a scope, architecture, contract, or ownership
decision that the plan does not settle, pause that part of the work and request
a plan amendment or user decision. Record the issue and current state so work
can resume without rediscovery.

Do not add compatibility behavior, fallback paths, new abstractions, unrelated
cleanup, or extra features under the label of implementation detail.

## Verification selection

Choose the smallest verification set that establishes the affected contract,
then expand it when the change touches callers, integration boundaries,
persistence, external services, migrations, or other material risk. Use focused
tests for focused changes and broader checks when the change radius warrants
them. A full-suite run is appropriate when it provides evidence the affected
surface needs, not as an automatic repetition after every edit.

Record the checks actually run, their results, relevant artifacts, and any
residual risk. Do not claim acceptance from an unchecked box alone.

## Checkpoints and evidence

Update a plan checklist when a meaningful work item is complete. A checkpoint
should identify the completed outcome, changed surface, verification result,
remaining work, and next checkpoint when one exists.

Keep evidence proportional to the change. Preserve material test output,
review findings, migration or database results, live-service artifacts, and
scope decisions. Routine commands need no transcript unless their result is
part of acceptance or a later handoff.

## Plan review

Before approval or execution, review a draft when requested or when the plan
has unresolved scope, architecture, ownership, or handoff risk. Check that:

- the direction and target state are explicit;
- change ownership and exclusions are complete;
- contracts and acceptance criteria are testable;
- no unresolved decisions remain in the executable plan;
- the verification and handoff expectations fit the actual change.

Plan review does not authorize new scope. New work is added only through an
accepted plan amendment or a new plan.

## Code review

Use an independent code review when required by the plan, repository policy,
user direction, or the consequence of the change. Review the diff and material
evidence against:

- the approved change direction and surface;
- ownership boundaries and semantic authority;
- contracts, acceptance criteria, and exclusions;
- appropriate focused and broader verification;
- unplanned compatibility, fallback, abstraction, or cleanup;
- residual risk and handoff quality.

The responsible owner resolves findings inside the approved boundary and
reruns affected verification. Findings that require a new contract or scope
return to plan amendment before implementation.

## Completion

Close the plan only when its acceptance criteria are evidenced, material
review findings are resolved or explicitly accepted, checkpoints are current,
and the lifecycle record is updated where the repository uses one.

If work stops before completion, record the remaining scope, current state,
verification, blockers, and next checkpoint. Keep the plan in its appropriate
active state rather than representing unfinished work as complete.
