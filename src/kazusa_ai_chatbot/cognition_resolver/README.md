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

Visible resolver fallback actions carry caller-owned addressee provenance. When
the resolver creates a pending clarification or approval surface, converts a
user-input blocker, or closes a terminal blocker, its `speak` action includes
one `target` role for the resolved `GlobalPersonaState.global_user_id` and an
empty `evidence_handles` list. Delivery remains owned by the current channel;
the target role lets L3 project the current user as the direct recipient. A
missing or blank current-user identity is a resolver contract error and fails
closed before L3. Internal-thought and other non-user episodes retain their
private behavior and do not receive a fabricated visible fallback.

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

Required task evidence is represented by
`required_resolver_evidence_dependency.v2`, which contains only the accepted
request handle and referenced observation id. `required_task_observation(...)`
is the single semantic lookup owner. L3 derives evidence state, remaining
needs, continuation, excerpts, and prompt-safe handles from that observation;
the dependency never copies those mutable values.

The task-resolution service is the sole owner of its inline budget,
checkpoint, and deferred promotion. The generic resolver timeout continues to
bound other capabilities, while task resolution is awaited directly so the
loop cannot return with an unowned detached task still mutating durable state.

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

## Shared-Memory Prewarm Outcome

The first-cycle memory worker returns the typed
`SharedMemoryPrewarmOutcomeV1` carrier. The resolver owns validation and
projection of the bounded RAG shape; cognition owns semantic interpretation.
The fixed reason vocabulary is `worker_unresolved`,
`worker_contract_invalid`, `projection_failed`, `no_shared_memory`,
`worker_error`, `shared_memory_ready`, `shared_memory_merged`,
`empty_query_after_character_mention`, `not_first_cycle`, and
`unsupported_episode`. A skipped outcome is explicit and does not start a
worker. A ready outcome exposes retrieved evidence; merge produces the
`shared_memory_merged` disposition with truthful retrieved and merged counts.

The outcome is copied into the current graph-attempt checkpoint and then into
the Brain `cognition_run_observation.v1` `evidence.shared_memory_prewarm`
section. A retry starts with a cleared checkpoint, so stale prewarm evidence
cannot leak across graph attempts. Cancellation propagates without fabricating
a terminal outcome.

The validated carrier reports `latency_ms`, `retrieved_count`, and
`merged_count` as bounded diagnostic counts (the wire model names the latter
two `retrieved_shared_count` and `merged_shared_count`). The current graph attempt
owns the checkpoint: recording deep-copies the outcome, binds it to that
attempt, and clears it when a retry begins. The Brain publisher may
project the carrier into its observation section, while the resolver never
publishes raw worker values.

The prewarm-only request projection is structural rather than semantic query
interpretation. It copies the prompt message context and current-turn
histories, removes only a typed active-character mention whose bot identity
matches the active character global id, and excludes the exact active-turn
history rows. Participant mentions, other bots, unproven bot mentions, older
history, and longer literal tokens remain source content. The original state
and envelope remain unchanged, and a query emptied by the typed mention
becomes the explicit `empty_query_after_character_mention` skipped outcome.

Do not choose a specialist, worker, timeout, checkpoint, delivery target, or
low-level tool parameter from cognition. Do not make resolver observations into
persona stance, mutate replacement state, send adapter text, or import
coding-agent internals.
