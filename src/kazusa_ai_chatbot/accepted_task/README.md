# Accepted Task ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.accepted_task`
- Runtime role: user-facing delayed-work lifecycle
- Model-facing actions: `accepted_task_request`,
  `accepted_coding_task_request`, and `accepted_task_status_check`
- Internal executor: `kazusa_ai_chatbot.background_work`
- Related docs: [Action Spec](../action_spec/README.md),
  [Background Work](../background_work/README.md),
  [Brain Service ICD](../brain_service/README.md)

## Purpose

`accepted_task` owns the character-visible lifecycle for delayed work the
character has accepted. It sits above the internal `background_work` executor
so cognition can reason about ordinary task states instead of queue rows, job
ids, workers, leases, retry counters, or adapter callback details.

## Boundary

- LLM cognition decides whether a delayed user task is accepted or whether a
  progress/status answer is needed.
- Deterministic lifecycle code owns task identity, active duplicate rejection,
  state transitions, persistence, and source-policy validation.
- `background_work` remains the internal executor for queued work.
- Workers never send adapter messages directly. Completed accepted tasks return
  through source-bound cognition and normal dialog/delivery boundaries.
- L2d sees `accepted_task_request`, `accepted_coding_task_request`, and
  `accepted_task_status_check`. `background_work_request` remains an internal
  executable action produced by deterministic materialization.
- New accepted tasks may be executed by text-artifact workers or deterministic
  workers such as `future_speak`; cognition still reasons only over the
  accepted-task lifecycle.
- Durable coding tasks use a coding-specific action that exposes only closed
  semantic actions and prompt-safe `coding_run:<run_id>` references. The
  accepted-task layer still hides queue rows and worker internals.
- Coding worker completion may persist one optional `coding_run_context.v1`.
  It contains only the run ref, public status, objective summary, allowed next
  actions, one blocker question/options, follow-up state, and update time.

## Public Interfaces

Public callers interact with accepted-task state through the action-spec and
brain-service flow:

```text
L2d selected action
  -> accepted_task_request, accepted_coding_task_request, or accepted_task_status_check
  -> deterministic action-spec materialization
  -> accepted-task lifecycle row
  -> optional internal background_work_request or requested-worker handoff
  -> tool_result cognition episode
```

The prompt-visible accepted-task fields are semantic only:

```text
accepted_task_state, accepted_task_summary, wait_guidance, result_summary,
failure_summary
```

Do not project job ids, queue state, worker names, leases, retry counters,
adapter callback data, or database field names into cognition or dialog
prompts.

## Runtime Flow

New delayed work follows this order:

```text
claim accepted_task enqueueing row
  -> insert internal background_work job
  -> mark accepted_task pending with executor ref
  -> expose semantic acknowledgement/progress state to cognition
```

If the internal job insert fails, the accepted task moves to `enqueue_failed`
and the active identity is released. The character must not promise completion
for a task that has no durable executor row.

On result-ready turns, cognition consumes the current sanitized context. On a
later user turn, the accepted-task repository loads the newest open contexts
for the trusted requester and channel through its indexed public query, keeps
the newest row per run, and returns at most three contexts. Queue payloads,
worker metadata, approvals, execution details, and paths never enter this
lookup or the LLM prompt.

## Persistence

Active duplicate suppression applies while a task is in:

```text
enqueueing, pending, running, result_ready, failure_ready,
delivery_in_progress, delivery_retryable
```

Terminal states release the active identity:

```text
delivered, enqueue_failed, delivery_exhausted, cancelled, superseded
```

The active identity is built from trusted requester/channel scope plus the
structured semantic task seed/detail selected by cognition. The source message
id is provenance only and is excluded from identity so repeated turns can
resolve to the same active task.

## Result Delivery

Accepted-task-backed worker rows emit `tool_result` cognition
episodes when complete.

Dispatcher delivery synchronizes the accepted-task row to delivered or a
retryable delivery-failure state. Operators diagnose executor details through
internal job rows; the character receives only the accepted-task result or
failure summary.

## Failure Behavior

- Duplicate active work reuses or reports the active accepted-task state rather
  than enqueueing another job.
- Failed executor materialization moves the accepted-task lifecycle to
  `enqueue_failed`.
- Worker failure becomes a semantic accepted-task failure summary for later
  cognition.
- Delivery failure is tracked as retryable or exhausted delivery state; workers
  still must not send adapter messages directly.

## Testing Contract

Tests for this boundary should verify:

- accepted-task identity and duplicate suppression;
- materialization into `background_work_request`;
- materialization into validated requested-worker payloads for first-class
  deterministic worker handoffs;
- status-check behavior that does not enqueue new work;
- `tool_result` handoff into cognition;
- prompt projection that excludes queue, worker, lease, adapter, and database
  internals.

## Forbidden Paths

- Do not use deterministic keyword matching over raw user text to decide task
  identity.
- Do not expose worker names, job ids, leases, retry counters, or adapter ids
  to L2d or dialog prompts.
- Do not let workers send adapter messages directly.
- Do not bypass accepted-task lifecycle persistence for model-facing delayed
  user work.
