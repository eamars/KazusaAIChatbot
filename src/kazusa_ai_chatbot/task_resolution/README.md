# Task Resolution

## Document Control

- Owning package: `kazusa_ai_chatbot.task_resolution`
- Source of truth: contracts, state, orchestrator, service, and focused tests
- Document status: current inline-first task-resolution contract

## Purpose

`task_resolution` owns one inline-first, resumable semantic task session.
It accepts an authorized `task_resolution_request`, preserves a typed
checkpoint, and returns only a terminal prompt-safe result or a deferred
checkpoint for durable continuation.

## Boundary

- Cognition decides whether current evidence is sufficient and may emit one
  `task_resolution_request`.
- The task orchestrator chooses one next specialist, semantic subgoal, and
  validated coding objective mode.
- Specialists own their declared public IO and return typed evidence or a
  typed refusal.
- Deterministic code owns budgets, counters, retry eligibility, checkpoint
  validation, and durable handoff.
- Dialog owns final visible wording.

The package has exactly four initial specialists:
`local_context`, `public_research`, `coding`, and `text_computation`.

## Public Interfaces

```python
await resolve_task_inline(
    request,
    execution_context,
    inline_budget_seconds=30.0,
)

await resume_task_resolution(checkpoint, execution_context)
```

`TaskResolutionExecutionContextV1` contains only typed, prompt-safe context.
It excludes adapter objects, database handles, credentials, raw worker
payloads, and coding-agent internal state.

## Runtime Flow

- Four specialist dispatches maximum.
- Four raw orchestrator LLM calls maximum, including malformed structural
  candidates and their bounded replacements.
- Two route corrections maximum.
- Two invocations of one specialist for one task node maximum.
- An incompatible `(task_node_id, specialist)` pair cannot repeat.
- A second invocation requires a persisted retryable temporary operational
  failure or material later evidence that changes the handler input.
- `partial` requires at least one validated provenance-bearing evidence item.
- A deferred result carries the same nonterminal checkpoint and never resets
  counters after a process or lease retry.
- `pending_dispatch` records the exact selected specialist, subgoal, coding
  mode, and `selected` or `started` phase. Background work persists selection,
  then the started transition, before invoking a handler. A resumed started
  dispatch is settled unavailable rather than relaunched.
- Inline coding selections always return a deferred selected handover before a
  coding handler is imported or called.
- A resolved or evidence-bearing partial result with remaining needs creates
  deterministic dependency child nodes. The first dependency-ready node
  continues through the same bounded loop; exhausted continuation returns an
  evidence-bearing partial result.

The static orchestrator prompt receives a bounded human payload with task
state and evidence. It does not receive raw specialist payloads, worker
metadata, credentials, repository paths, or adapter identifiers.

## Failure Behavior

Specialists return typed incompatibility, availability, user-input, approval,
or failure outcomes. Deterministic checkpoint validation enforces dispatch and
route-correction limits; durable retries resume the same validated checkpoint.

## Testing Contract

Run task-resolution contract, state, orchestrator, specialist, inline-promotion,
and background-resume suites. Run live LLM cases one at a time and inspect their
saved artifacts before sign-off.

## Forbidden Paths

Do not add specialists, dynamic tool schemas, direct adapter delivery, raw
database access, arbitrary filesystem or shell access, or a second task-session
persistence collection. Do not inspect coding-agent internals; the coding
specialist uses only frozen public coding-run exports.
