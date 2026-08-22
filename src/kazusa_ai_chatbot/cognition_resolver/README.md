# Cognition Resolver ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.cognition_resolver`
- Source of truth: resolver contracts, loop, capability executor, and tests
- Document status: current canonical resolver boundary

## Purpose

The cognition resolver owns bounded capability execution and projects
source-owned observations into the caller's canonical cognition context.

## Boundary

The cognition resolver's live owner is
`nodes.persona_supervisor2.stage_1_goal_resolver`; scheduled owners use the
same capability boundary. It receives caller-owned semantic capability
context and never chooses an engine or mutates cognition state.

The resolver executes an approved semantic capability and returns typed
observations. It has no authority to mutate user or character cognition state,
select final wording, deliver adapter output, or reinterpret a capability
result. The cognition caller commits the final replacement state before L3,
action execution, dialog, or worker delivery can proceed.

## Runtime Flow

```text
canonical cognition output
  -> optional validated resolver capability request
  -> one bounded capability execution
  -> typed resolver observation
  -> caller-owned continuation or final response plan
```

Resolver capabilities return evidence only. They cannot write cognition state,
choose a character goal, rewrite response intent, or authorize delivery. The
caller owns any continuation context, permission check, persistence, and final
state commit. Resolver telemetry remains generic semantic status and excludes
prompts, private identifiers, and implementation details.

## Input And Output Contracts

Canonical requests contain only the semantic capability, goal, and trusted
caller context. Capability handlers may perform their bounded retrieval or
research operation and return a source-owned observation. The canonical
projection `project_resolver_observation_for_cognition(...)` produces:

- one typed resolver observation with `source_kind=resolver_observation`;
- complete source identity and UTC occurrence time;
- the exact source-owned semantic visibility map;
- a typed direct-fact list, empty unless a capability can prove the required
  fact provenance and targets.

An observation is evidence, never persona, intention, affect, relationship
state, or final stance. Resolver code cannot write `replacement_state`, choose
a goal branch, or rewrite an intention route.

Task and local-context execution require the validated cognition-owned
`GlobalPersonaState.cognition_scene_context` carrier. The generic loop validates
that carrier before invoking a task executor, so a missing or malformed handoff
is a typed boundary failure outside checkpoint replay. Task-resolution
advertisement uses the same deterministic readiness contract as execution;
targetless self-cognition with no executable identity therefore retains other
valid affordances while omitting task resolution.

Task-resolution requests use the caller-provided capability and continuation
context. The planner sees only the semantic capability roster and does not see
specialists, queue state, persistence mechanics, or private identifiers.

## Public Interfaces

| Entrypoint | Owner | Purpose |
| --- | --- | --- |
| `execute_resolver_capability_request(...)` | `capabilities.py` | Execute one bounded source capability. |
| `project_resolver_observation_for_cognition(...)` | `capabilities.py` | Convert a source result into typed semantic evidence/direct facts. |
| `build_resolver_telemetry_fields(...)` | `telemetry.py` | Emit bounded request/progress/status diagnostics without raw identifiers or state. |

Pending human clarification and approval records remain deterministic ledger
state owned by `pending.py`; they enter future cognition as typed evidence and
do not become resolver authority.

## Failure Behavior

Resolver diagnostics are bounded and semantic. Telemetry may contain
capability names, semantic goals, progress status, and a clipped progress
summary. It excludes raw prompt text, observation identifiers, evidence
handles, replacement state, state owner keys, private bids, handler parameters,
and platform identifiers.

Human-readable traces under `test_artifacts/cognition_resolver/` are diagnostic
artifacts only. They never become cognition input automatically.

Background handoff only reports a pending continuation after checkpoint,
accepted-task, pending-state, and queue promotion succeed. Any contract,
checkpoint, or enqueue failure stays in the typed resolver observation path and
does not authorize a visible completion claim.

## Testing Contract

Run cognition-resolver contract, loop, L2d, capability, task-resolution inline
promotion, and result-delivery integration suites with the project virtual
environment. Live LLM cases run one at a time with raw output inspected.

## Forbidden Paths

Do not choose a specialist, worker, timeout, checkpoint, delivery target, or
low-level tool parameter from cognition. Do not make resolver observations into
persona stance, mutate replacement state, send adapter text, or import
coding-agent internals.
