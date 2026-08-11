# Execution Guidance Reference

Use this reference to execute an approved plan while preserving its change
contract. It defines runtime executor resolution, decision boundaries, and
evidence requirements without prescribing a permanent roster or choreography.

## Contents

- [Ownership](#ownership)
- [Runtime-owned executor resolution](#runtime-owned-executor-resolution)
- [Eligibility gate](#eligibility-gate)
- [Selection factors and cost rule](#selection-factors-and-cost-rule)
- [Handoff record](#handoff-record)
- [Reassignment](#reassignment)
- [Verification selection](#verification-selection)
- [Checkpoints and evidence](#checkpoints-and-evidence)
- [Plan review](#plan-review)
- [Code review and sign-off](#code-review-and-sign-off)
- [Change decisions during execution](#change-decisions-during-execution)
- [Completion](#completion)

## Ownership

The plan identifies the parent or implementation owner, the change surface,
and the roles required for the work. The responsible executor owns the
implementation within the role's declared boundary. The parent or designated
owner keeps plan status, checkpoints, evidence, and lifecycle records coherent.

Before the first edit, capture the worktree status, changed-path baseline, and
explicitly owned file set. Pre-existing user or concurrent changes remain
outside the execution diff. Every changed production source path is matched to
the plan's exact source-to-test row before the work item is accepted.

## Runtime-owned executor resolution

Executor resolution is runtime-owned at every handoff. The plan supplies a
role contract; the parent supplies the current execution assignment. Resolve
the assignment from currently available project-native agents and models, with
their available tools, skills, context, verification access, configuration, and
resource state.

For each handoff, the parent should:

1. Read the role's responsibility, owned surface, authority, applicable
   skills, capability floor, independence requirement, acceptance output, and
   gate from the approved plan.
2. Establish the remaining scope, change radius, current verification state,
   and next checkpoint.
3. Identify currently available project-native executor candidates and their
   actual capability, context, tool, skill, verification, usage, latency, and
   budget conditions.
4. Remove candidates that fail any hard role, authority, capability, skill,
   tool, context, verification, or independence requirement.
5. Apply the selection factors and expected-total-cost rule below.
6. Record the resolution and send the handoff with the role contract and
   acceptance gate intact.

An exact executor, model, or configuration supplied by the user or an approved
plan is a **plan-scoped fixed execution constraint**. Validate that binding at
each applicable handoff and use it as an eligibility gate. Do not silently
substitute another executor, model, or configuration because of price,
availability, latency, or convenience. Changing a fixed constraint requires
the applicable user decision or plan amendment, and the decision belongs in
execution evidence.

The current harness and model handoff protocol owns model-specific mechanics:
invocation syntax, session setup, acknowledgement or execution turns, waiting,
interruption, retry, and transport details. This reference records the chosen
assignment and its rationale without defining those mechanics.

## Eligibility gate

An executor is eligible only when it can satisfy the complete role contract in
the current runtime. Check all of the following before handoff:

- semantic responsibility and task complexity are within the capability floor;
- the executor can access the declared owned surface and required context;
- required tools, skills, references, and verification paths are available;
- authority is sufficient for the intended edits, decisions, or effects;
- the executor can produce the declared acceptance output;
- the role's independence requirement is satisfied;
- production, security, data, and lifecycle constraints remain enforceable;
- any plan-scoped fixed execution constraint matches exactly.

Resource pressure may change which eligible executor receives a dynamic role,
or may support a narrower delegated slice within the role's owned surface. It
does not lower the capability floor, independence requirement, acceptance
output, or gate. A narrower slice must leave the remaining scope explicit and
must preserve complete acceptance evidence across the role's eventual work.

## Selection factors and cost rule

Among eligible dynamic candidates, assess the factors that affect delivery
quality and total resource use:

- task complexity;
- semantic ambiguity;
- change radius;
- production risk;
- context, tool, and skill needs;
- verification strength;
- expected supervision, retry, review, and rework;
- latency and availability;
- current usage and concurrency pressure; and
- budget or other resource limits.

Select the eligible executor with the lowest expected total execution cost,
not merely the lowest raw model price. A qualitative rationale is sufficient
when it explains the material tradeoffs. Treat expected total cost as the
combined burden of direct execution plus likely supervision, retries, review,
rework, delay, and resource contention. A lower-priced available executor may
be preferable under resource pressure when it still satisfies every hard
gate. A narrower delegated slice may likewise reduce expected cost when it
preserves the role contract and leaves an explicit checkpoint for the
remaining work.

Do not select a cheaper candidate whose weaker reasoning, missing tool or
skill access, inadequate context, weak verification, or likely rework makes
its expected total cost higher. Do not use a raw price comparison as the sole
rationale.

## Handoff record

Every handoff record contains enough information for the parent, recipient,
reviewer, and later audit to reconstruct the assignment:

- approved plan identifier and lifecycle state;
- role identifier and a reference to its complete role contract;
- remaining scope, exact owned files or interfaces, and any delegated slice;
- authority and independence requirements in force;
- resolved executor identity, including the runtime-native agent and model
  identifiers when available;
- resolved configuration, including non-secret route, tool, skill, context,
  and verification settings needed to reproduce the assignment;
- resolution mode: dynamic runtime selection or fixed execution constraint;
- selection rationale tied to capability eligibility, the listed factors, and
  expected total execution cost;
- baseline, completed verification, known deviations, and blockers;
- acceptance output, entry/exit gate, and next checkpoint; and
- any prior assignment and the reason for a runtime reassignment.

Record references or identifiers for secrets rather than credentials or secret
values. The record describes the effective configuration without expanding the
plan into model-specific handoff instructions.

## Reassignment

Runtime reassignment is execution mechanics when the role has dynamic
resolution. The parent may resolve a new eligible executor at a later
handoff—such as after a checkpoint, resource change, or failed attempt—while
the role responsibility, owned surface, authority, capability floor,
independence requirement, acceptance output, and gate remain unchanged.

Append a new handoff and evidence record for each reassignment. Preserve the
prior executor, configuration, scope, rationale, progress, verification, and
reason for the change. Reassignment does not erase a failed attempt or turn an
unmet gate into acceptance.

A fixed execution constraint cannot be reassigned silently. Obtain the
applicable user decision or plan amendment first, then record the changed
constraint and new resolution. If the fixed executor is unavailable, pause at
the gate and report the blocker.

## Verification selection

Choose the smallest verification set that establishes the affected contract,
then expand it when the change touches callers, integration boundaries,
persistence, external services, migrations, or other material risk. Use
focused tests for focused changes and broader checks when the change radius
warrants them. A full-suite run is evidence when it covers the affected
surface, not an automatic substitute for an exact mapped check.

For a source change, resolve each changed path through the authoritative
source-to-test manifest or equivalent plan matrix. Run pytest collection for
the exact mapped node IDs, fail on an unmapped path or stale node, and then run
those exact deterministic nodes. A broader passing suite cannot waive a
missing, deselected, or uncollected mapped node. Integration and live tests
remain additional evidence when the change radius crosses those boundaries.

## Checkpoints and evidence

Update a plan checklist when a meaningful work item is complete. A checkpoint
identifies the completed outcome, changed surface, verification result,
remaining work, and next checkpoint when one exists.

Keep evidence proportional to the change. Preserve material test output,
review findings, migration or database results, live-service artifacts, scope
decisions, and every executor-resolution record. `Execution Evidence` and each
`Execution Handoff` record the resolved executor, effective configuration,
scope, and selection rationale, along with the acceptance gate and checkpoint.
Routine commands need no transcript unless their result is part of acceptance
or a later handoff.

## Plan review

Before approval or execution, review a draft when requested or when the plan
has unresolved scope, architecture, ownership, or handoff risk. Check that:

- the direction and target state are explicit;
- change ownership, role contracts, and exclusions are complete;
- contracts and acceptance criteria are testable;
- no unresolved decisions remain in the executable plan;
- the verification and handoff expectations fit the actual change; and
- any fixed execution constraints identify their decision authority.

Plan review does not authorize new scope. New work is added only through an
accepted plan amendment or a new plan.

## Code review and sign-off

Use an independent code review when required by the plan, repository policy,
user direction, or the consequence of the change. Resolve the reviewer as a
separate eligible executor under the role's independence requirement. The
reviewer checks the diff and material evidence against:

- the approved change direction and surface;
- role ownership boundaries and semantic authority;
- contracts, acceptance criteria, and exclusions;
- appropriate focused and broader verification;
- unplanned compatibility, fallback, abstraction, or cleanup; and
- residual risk and handoff quality.

Review authority and remediation authority remain separate. The reviewer may
inspect, report findings, and pass or fail the review gate. A remediation role
implements corrections within its own declared authority. The remediation
executor does not independently sign off the corrected work; route it through
a separate independent review or sign-off handoff. Record findings,
remediation evidence, and final sign-off as distinct evidence.

## Change decisions during execution

The plan fixes intent, contracts, ownership, and exclusions. The executor may
choose local mechanics, decomposition, command order, and verification breadth
that preserve them.

When implementation reveals a scope, architecture, contract, ownership, or
fixed-execution decision that the plan does not settle, pause that part of the
work and request a plan amendment or user decision. Record the issue and
current state so work can resume without rediscovery.

Do not add compatibility behavior, fallback paths, new abstractions,
unrelated cleanup, or extra features under the label of implementation detail.

## Completion

Close the plan only when its acceptance criteria are evidenced, material
review findings are resolved or explicitly accepted, checkpoints are current,
all required independent sign-off is recorded, and the lifecycle record is
updated where the repository uses one.

If work stops before completion, record the remaining scope, current state,
verification, blockers, executor assignment, and next checkpoint. Keep the
plan in its appropriate active state rather than representing unfinished work
as complete.
