# Cognition Resolver ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.cognition_resolver`
- Source of truth: resolver contracts, loop, capability executor, and tests
- Document status: current V2 resolver recurrence contract

## Purpose

The cognition resolver owns bounded recurrence around Cognition Core V2 and
projects source-owned observations into later cognition cycles.

## Boundary

The cognition resolver owns bounded recurrence around Cognition Core V2. Its
live owner is `nodes.persona_supervisor2.stage_1_goal_resolver`; the idle owner
is `self_cognition.runner._default_cognition_client`.

The resolver executes a cognition-selected capability and returns typed
observations. It has no authority to mutate user or character cognition state,
select final wording, deliver adapter output, or reinterpret a capability
result. Cognition Core V2 remains the semantic decision owner. The connector
commits the single final replacement state before L3, action execution, dialog,
or worker delivery can proceed.

## Runtime Flow

```text
CognitionCoreInputV2
  -> run_cognition(..., commit=False)
  -> ResolverCapabilityRequestV2[]
  -> one bounded capability execution
  -> typed resolver observation
  -> connector projects CognitionEvidenceV2
  -> next V2 cognition cycle
  -> terminal CognitionCoreOutputV2
  -> one state-scope commit
  -> L3/action/dialog or private terminal handling
```

`call_v2_resolver_loop(...)` owns the episode-local `ResolverWorkingStateV2`:

- `origin_scope`: `user` or `character`, fixed for the run;
- `cycle_index` and `max_cycles`: deterministic recurrence limits;
- `pending_requests`: exact `ResolverCapabilityRequestV2` rows;
- `observations`: prompt-safe capability outcomes;
- `cognition_output`: the latest complete V2 output;
- `terminal`: the loop terminal flag.

The loop carries the latest in-memory V2 output forward. It does not reload or
write cognition state between cycles. The caller commits only the final output.

### Persona parent-checkpoint guardrail

The live persona stage may bind one context-local
`CognitionRetryCoordinator` around its queued service graph. The canonical
connector owns the checkpoint: it resolves identity, reads mutable state,
joins cycle-zero shared-memory prewarm, builds one `CognitionCoreInputV2`, and
only then invokes the non-committing `run_cognition` child. A parent recovery
replays that child from independent copies of the same checkpoint with the
same services and one stable checkpoint digest. Preparation, capability
execution, pending writes, action execution, surface rendering, delivery, and
the final commit remain outside the replay.

The coordinator has one replay token and exactly two epochs. The first claim
belongs to either the existing service graph retry or the parent checkpoint;
the other owner cannot start a second replay. Parent recovery is limited to an
escaped pre-commit `CognitionExecutionError` whose code is exactly
`goal_bid_structure_exhausted` or `goal_bid_provider_exhausted`. The generic
resolver loop remains unchanged, and idle `self_cognition` calls do not bind
this guardrail. Direct connector, facade, goal-owner, and loop calls without a
bound coordinator retain their direct failure contracts.

Before the first V2 input is built, the persona connector may start the
shared-memory prewarm owned by `capabilities.py`. This happens only at resolver
cycle zero for `user_message` and `internal_thought` episodes. The task overlaps
identity and mutable-state preparation, then joins before evidence mapping.
Only confirmed shared `memory` rows are merged into
`rag_result.memory_evidence`; the prewarm leaves `rag_result.answer` empty,
excludes scoped user-memory units, and creates no resolver observation. Empty
or failed retrieval preserves the base RAG payload. Later cycles reuse the
merged state and do not repeat prewarm.

The prewarm task starts from `decontextualized_input` and removes each readable
`@display_name` token backed by a typed `bot` mention once. Plain character
names and `user`, `platform_role`, `channel`, `everyone`, and `unknown`
mentions remain part of the task. When only bot-addressing whitespace remains,
prewarm returns the canonical empty projected RAG payload before constructing
the request or starting the persistent-memory worker.

`CognitionCoreOutputV2.goal_resolution` remains the semantic owner’s answerability
decision. It answers whether the accepted user goal is sufficient to answer now;
it does not mirror a source-specific RAG `resolved` field. When the decision is
`answerable_now`, the deterministic loop suppresses any optional resolver request
and settles the episode without capability execution. The other typed decisions
retain their required-evidence, user-input, or blocked paths.

## Input And Output Contracts

V2 requests contain only `capability`, `semantic_goal`, and
`evidence_handles`. Capability handlers may perform their bounded retrieval or
research operation and return a source-owned observation. The canonical
projection `project_resolver_observation_for_cognition(...)` produces:

- one `CognitionEvidenceV2` with `source_kind=resolver_observation`;
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

`task_resolution_request` is the single generic evidence-work capability. The
model-owned `start_in_background` boolean selects its entry route. `true`
creates the checkpoint and enters the same accepted-task, pending-state, queue,
and idempotency boundary directly, without invoking an inline specialist. `false`
runs the session inline under the configured wall-clock budget; a deferred run
uses that same durable continuation boundary. A resolved or evidence-bearing
partial result becomes a prompt-safe observation for the next cognition cycle.
Deferred evidence is projected before continuation context, and an empty
deferred evidence set produces no fabricated partial knowledge. The planner
does not see specialists, queue state, or persistence mechanics.

## Public Interfaces

| Entrypoint | Owner | Purpose |
| --- | --- | --- |
| `call_v2_resolver_loop(...)` | `loop.py` | Run bounded V2 cognition/capability recurrence without intermediate state commits. |
| `execute_resolver_capability_request(...)` | `capabilities.py` | Execute one bounded source capability. |
| `project_resolver_observation_for_cognition(...)` | `capabilities.py` | Convert a source result into typed V2 evidence/direct facts. |
| `build_v2_resolver_telemetry_fields(...)` | `telemetry.py` | Emit bounded request/progress/status diagnostics without raw identifiers or state. |

Pending human clarification and approval records remain deterministic ledger
state owned by `pending.py`. When admitted to V2, they enter as typed evidence;
they do not restore the retired V1 cognition-chain contract.

## Failure Behavior

Resolver diagnostics are bounded and semantic. V2 telemetry may contain
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
