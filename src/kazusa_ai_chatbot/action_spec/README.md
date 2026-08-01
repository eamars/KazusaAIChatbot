# Action Spec ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.action_spec`
- Source of truth: registry, evaluator, execution contracts, and focused tests
- Document status: current v2 action contract

## Purpose

`action_spec` owns the deterministic envelope, validation, availability, and
execution trace for cognition-selected actions. It does not own generic
evidence gathering or worker selection.

## Boundary

Generic evidence work belongs to the resolver and task-resolution session.
Action specs retain only visible surfaces, memory lifecycle, future scheduling,
bound coding continuations, and scoped status checks.

## Public Interfaces

The registry supplies prompt-safe affordances, `ActionSpecEvaluator` validates
one materialized action, and trace execution applies only the reviewed handler
for its capability. Callers consume prompt-safe action results rather than
handler payloads or queue internals.

## Model-facing action roster

| Capability | Owner | Contract |
| --- | --- | --- |
| `speak` | L3 surfaces | Selects a visible surface intent; L3 owns final wording. |
| `memory_lifecycle_update` | memory lifecycle | Requests review of an existing commitment lifecycle. |
| `trigger_future_cognition` | scheduler | Requests a future cognition cycle without calling cognition directly. |
| `future_speak` | background work | Schedules a deterministic future-speak task from an exact trigger and semantic objective. |
| `accepted_coding_task_request` | background work | Continues one already-bound coding run through a closed lifecycle action. |
| `accepted_task_status_check` | accepted task | Reads current scoped accepted-task state without creating work. |

`apply_memory_lifecycle_update` is internal-only. `task_resolution_request` is
a resolver capability, not an action-spec capability: cognition uses it when
current evidence is insufficient, and deterministic resolver code owns its
inline budget and any durable promotion.

## Runtime Flow

Only `future_speak` and a bound coding continuation create action-originated
accepted-task state. Deterministic execution validates trusted user scope,
creates or reuses `accepted_task.v2`, then writes a reviewed
`background_work_job.v2`.

```text
future_speak
  -> accepted_task.v2
  -> requested_worker="future_speak"

accepted_coding_task_request
  -> accepted_task.v2
  -> requested_worker="task_orchestrator"
  -> frozen public coding-run continuation
```

There is no action-visible generic background-work request and no router that
chooses a worker after the live turn. Generic local, public, coding, and
text/computation evidence work enters through task resolution, whose bounded
orchestrator selects one specialist at a time.

## Input And Output Contracts

`accepted_coding_task_request` requires a prompt-safe
`coding_run:<run_id>` reference and one currently allowed closed decision:
`revise_proposal`, `summarize`, `status`, `approve_and_verify`,
`respond_to_blocker`, or `cancel`. Revision instructions and verification
requests stay structured; approval requires trusted current-turn evidence.

The action handler uses only the frozen public coding-run exports. It never
inspects coding-agent internals, determines a new coding task type, or grants
mutation authority. The task-orchestrator worker projects the public result
back through accepted-task result-ready state.

## Failure Behavior

Action prompts receive only semantic affordances, validated source references,
and prompt-safe results. Queue ids, leases, worker payloads, database details,
raw evidence, filesystem paths, credentials, and adapter identifiers remain
outside cognition and L3. Background workers never send adapter text directly;
normal result-ready cognition and delivery own visible follow-through.

## Testing Contract

Run action-spec evaluator, execution, cognition action-planning, bound coding,
future-speak, and background-work integration tests with the project virtual
environment. Prompt and capability roster changes also require static removal
checks for retired generic routes.

## Forbidden Paths

Do not use an action spec to choose a task-resolution specialist, worker,
timeout, checkpoint, repository path, tool argument, or final dialogue. Do not
introduce a generic background router or expose raw queue state to cognition or
L3.
