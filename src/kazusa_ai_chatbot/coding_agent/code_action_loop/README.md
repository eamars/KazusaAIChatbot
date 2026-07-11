# Coding Action Loop ICD

`code_action_loop` is the single model-facing execution engine for coding
objectives. One controller returns one strict `coding_action.v1` JSON object
per turn. Deterministic code validates and executes the action, persists its
observation, updates bounded context, and either requests another turn or
projects a terminal result.

## Boundary

- `prompts.py` owns the static controller role, action vocabulary, and output
  contract.
- `parser.py` validates the closed action and action-specific argument schema.
- `context.py` renders the bounded dynamic goal, capabilities, current
  candidate state, working note, failures, and prompt-safe observations.
- `actions.py` dispatches read, search, candidate edit, and approved semantic
  run actions. It does not accept commands.
- `state.py` owns the managed candidate, revision, candidate/overlay journal,
  recovery backups, and idempotent crash recovery.
- `supervisor.py` owns run/source locking, loop budgets, durable action and
  observation artifacts, finalization, typed blockers, and private lifecycle
  integration.

`read_only`, `propose_patch`, and `verify_repair` use the same protocol. Their
capability sets differ deterministically: read-only work receives no edit or
run capability, and `run` becomes available only with a trusted approved
execution context.

## Durability

Actions are appended before dispatch and observations are appended before the
next controller turn. Candidate mutations use
`prepared -> candidate_written -> overlay_written -> committed` journal phases.
Rollback metadata stores only affected-path identities, managed backup paths,
and prior overlay identities; it stores no inline previous content.

Approved run actions persist their structured execution identity before
execution and their terminal result before the loop observation. An orphaned
run reconstructs exactly one matching observation without executing again;
missing or mismatched evidence becomes a typed retry-loop blocker.

Proposal finalization binds the candidate revision and managed-tree digest to
the ordered canonical operation digest. Approval evidence is consumed once and
bound to that exact proposal. The apply package and each verification attempt
are persisted before their external effect, allowing an interrupted
continuation to rematerialize the same managed candidate and resume from the
first attempt without terminal evidence.

Action sequence and observation sequence are independent durable counters.
Recovery uses `action_sequence` to reconcile an action with its durable effect
observation, including recovered `finish` and `block` transitions. A failed
finish finalization adds a second failure observation for the same action
sequence before another controller turn.
Only the latest observed action may drive terminal recovery. A recovered
finalization failure is persisted as current failure evidence and returns to
the controller loop, matching the uninterrupted path. Reusing an operation id
requires the complete original mutation identity to match.

Every candidate read, edit, backup, rollback, and overlay reconstruction uses
the shared managed-path policy. A symlink at the managed root, an affected
path, or any existing ancestor is rejected before content is read or changed.

The engine is reached through the private coding-run evaluation boundary until
the comparison gate authorizes the one-way public coding-run cutover.
