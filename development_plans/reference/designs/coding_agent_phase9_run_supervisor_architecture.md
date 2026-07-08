# coding agent phase9 run supervisor architecture

## Summary

- Goal: define the Phase 9 directional architecture for a first-class coding
  run supervisor that coordinates coding-agent primitives across a durable,
  inspectable local workflow.
- Plan class: directional_architecture.
- Status: reference direction.
- Execution rule: reference only. Promote this document into
  `development_plans/active/short_term/` before implementation.
- Mandatory skills for future execution: `development-plan`,
  `local-llm-architecture`, `py-style`, `test-style-and-execution`, and
  `debug-llm`.
- Direction: move from independent direct APIs toward a deterministic run
  state machine while preserving the Kazusa split between manager, planner,
  programmer, patcher, applier, and executor.

## Context

By the end of Phase 8, the coding agent should have these capabilities:

- source resolution;
- read-only source evidence gathering;
- source-free artifact writing;
- existing-source modification planning;
- patch proposal materialization;
- approved apply into managed copies;
- bounded verification execution;
- capped verify-and-repair loops.

Those primitives are still called as separate direct APIs. Codex-like coding
capability also needs session continuity: a run should remember the goal,
source contract, evidence, proposed changes, approval state, apply workspace,
execution outcomes, repair attempts, blockers, and public trace. Phase 9 adds
that stateful orchestration layer without collapsing the role architecture
into one monolithic LLM call.

## Architecture Direction

Phase 9 introduces `coding_run`, a deterministic supervisor package:

```text
src/kazusa_ai_chatbot/coding_agent/coding_run/
  README.md
  models.py
  ledger.py
  supervisor.py
```

The run supervisor owns lifecycle state. Specialist modules keep their current
domain ownership:

- `code_fetching`: source contract.
- `code_reading`: read-only evidence.
- `code_writing`: source-free artifact generation.
- `code_modifying`: existing-source planning and programmer dispatch.
- `code_patching`: patch proposal and managed apply.
- `code_executing`: bounded command execution.
- `code_verifying`: verify-and-repair composition.

The run supervisor never becomes a general-purpose shell agent. It coordinates
closed operations and records decisions.

## Target State Machine

The run supervisor stores one canonical state machine:

```text
created
-> source_resolved
-> evidence_collected
-> plan_ready
-> proposal_ready
-> awaiting_approval
-> apply_ready
-> verification_ready
-> repair_ready
-> completed
```

Terminal states:

```text
completed
blocked
rejected
failed
cancelled
```

State transitions are deterministic and validated. LLM stages may decide
semantic plan content inside specialist boundaries, but deterministic code
decides whether a transition is legal.

## Public Directional API

Future executable planning should define these direct trusted APIs:

```python
from kazusa_ai_chatbot.coding_agent import start_coding_run
from kazusa_ai_chatbot.coding_agent import continue_coding_run
from kazusa_ai_chatbot.coding_agent import get_coding_run
```

`start_coding_run(...)` creates a run ledger and executes the first safe stage
based on the request shape.

`continue_coding_run(...)` advances a run with explicit structured input such
as approval, execution specs, repair approval, cancellation, or a narrowed
scope.

`get_coding_run(...)` returns public-safe run state, attempts, limitations, and
trace summaries.

## Ledger Direction

Initial Phase 9 storage should use managed workspace JSON ledgers under the
coding-agent workspace:

```text
<workspace_root>/coding_runs/<run_id>/run.json
<workspace_root>/coding_runs/<run_id>/events.jsonl
<workspace_root>/coding_runs/<run_id>/artifacts/
```

The public run id is opaque. Stored ledgers must avoid absolute original
source roots in public fields. Internal paths remain local to the managed
workspace and are loaded only by deterministic supervisor code.

MongoDB persistence is a later optional extension. The first executable Phase
9 plan should use workspace-local JSON to keep the blast radius small and to
match current coding-agent managed artifact storage.

## Run Ledger Shape

Directional `CodingRunLedger`:

```python
{
    "schema_version": 1,
    "run_id": str,
    "status": str,
    "goal": str,
    "created_at": str,
    "updated_at": str,
    "source": dict[str, object] | None,
    "repository": dict[str, object] | None,
    "source_scope": dict[str, object] | None,
    "evidence": list[dict[str, object]],
    "plans": list[dict[str, object]],
    "proposals": list[dict[str, object]],
    "approvals": list[dict[str, object]],
    "apply_attempts": list[dict[str, object]],
    "execution_attempts": list[dict[str, object]],
    "repair_attempts": list[dict[str, object]],
    "blockers": list[dict[str, object]],
    "limitations": list[str],
    "trace_summary": list[str],
}
```

Directional event shape:

```python
{
    "event_id": str,
    "run_id": str,
    "sequence": int,
    "event_type": str,
    "stage": str,
    "status": str,
    "summary": str,
    "public_payload": dict[str, object],
}
```

## Responsibilities

The run supervisor owns:

- run id creation;
- state transition validation;
- event appending;
- public state projection;
- specialist invocation order;
- approval state;
- execution-spec handoff;
- repair-attempt caps;
- cancellation;
- blocker reporting.

Specialists own:

- source resolution;
- evidence gathering;
- semantic change planning;
- artifact generation;
- patch mechanics;
- apply mechanics;
- command execution;
- verify-and-repair attempt internals.

Adapters and background workers remain thin. They may display run state or
submit structured continuation input, but the coding run supervisor owns the
coding lifecycle.

## Codex Capability Mapping

Codex behaves as a monolithic coding session with persistent memory in the
conversation, tool calls, file edits, tests, and iterative repair. Kazusa Phase
9 maps that behavior into deterministic local components:

| Codex behavior | Kazusa Phase 9 direction |
|---|---|
| Maintains task state in context | Workspace-local run ledger |
| Reads code before editing | `code_reading` evidence stage |
| Plans edits | `code_modifying` or `code_writing` PM stage |
| Edits files | Patch proposal plus approved managed apply |
| Runs tests | `code_executing` through structured specs |
| Repairs failures | `code_verifying` capped repair |
| Reports status | Public run projection and event summaries |
| Uses one strong model | Split local PM/programmer roles with deterministic caps |

## Local LLM Design Rules

- Keep each LLM call narrow and role-specific.
- Keep run state deterministic and outside prompts by default.
- Pass only the stage-specific projection needed by a specialist.
- Summarize prior stage outputs into bounded public-safe facts.
- Use explicit transition blockers instead of asking the LLM to manage global
  state.
- Prefer repeated small PM/programmer steps over one large prompt.
- Cap every loop and record cap exhaustion as a first-class blocker.

## Future Executable Phase 9 Scope

The eventual short-term Phase 9 plan should include:

- `coding_run` package models and ledger persistence;
- `start_coding_run(...)`, `continue_coding_run(...)`, and
  `get_coding_run(...)` public APIs;
- deterministic state transition tests;
- lifecycle event tests;
- public projection sanitization tests;
- continuation tests for approval, execution specs, repair attempts, scope
  narrowing, cancellation, and blockers;
- docs for direct callers and future control-console display;
- selected live LLM gates proving run-level orchestration remains grounded in
  specialist traces.

## Future Phase 9 Exclusions

Keep these outside the first executable Phase 9 plan:

- adapter auto-delivery;
- uncontrolled background run continuation;
- package installation;
- arbitrary shell command selection;
- repository mutation outside managed apply workspaces;
- MongoDB run persistence;
- web control-console UI;
- repository-scale reading fan-out, which belongs to Phase 10.

## Acceptance Direction

A future executable Phase 9 plan is ready for sign-off when:

- run ledgers survive process boundaries;
- every state transition is validated deterministically;
- public projections are sanitized;
- existing direct APIs still work independently;
- a run can progress from source-backed request to proposal, approval, apply,
  execution, repair, and completion;
- blocked states preserve enough information for a user or caller to continue;
- independent code review accepts the lifecycle boundary.

