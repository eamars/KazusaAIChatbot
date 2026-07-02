# Accepted Task

`accepted_task` owns the user-facing lifecycle for delayed work the character
has accepted. It sits above the internal `background_work` executor so
cognition can reason about ordinary task states instead of queue rows, job ids,
workers, leases, or retry counters.

## Boundary

- LLM cognition decides whether a delayed user task is accepted or whether a
  progress/status answer is needed.
- Deterministic lifecycle code owns task identity, active duplicate rejection,
  state transitions, persistence, and source-policy validation.
- `background_work` remains the internal executor for queued work.
- Workers never send adapter messages directly. Completed accepted tasks return
  through source-bound cognition and dispatcher delivery.
- L2d sees `accepted_task_request` and `accepted_task_status_check`.
  `background_work_request` remains an internal executable action produced by
  deterministic materialization.
- New accepted tasks may be executed by text-artifact workers or by
  deterministic workers such as `future_speak`; cognition still reasons only
  over the accepted-task lifecycle.

## States

Active duplicate suppression applies while a task is in:

```text
enqueueing, pending, running, result_ready, failure_ready,
delivery_in_progress, delivery_retryable
```

Terminal states release the active identity:

```text
delivered, enqueue_failed, delivery_exhausted, cancelled, superseded
```

## Creation Order

New delayed work must follow this order:

```text
claim accepted_task enqueueing row
  -> insert internal background_work job
  -> mark accepted_task pending with executor ref
  -> expose semantic acknowledgement/progress state to cognition
```

If the internal job insert fails, the task moves to `enqueue_failed` and the
active identity is released. The character must not promise completion for a
task that has no durable executor row.

## Duplicate Policy

The active identity is built from trusted requester/channel scope plus the
structured semantic task seed/detail selected by cognition. The source message
id is provenance only and is excluded from identity so repeated turns can
resolve to the same active task.

No deterministic keyword matching over raw user text belongs in this module.

## Prompt Projection

Prompt-visible accepted-task fields are semantic only:

```text
accepted_task_state, accepted_task_summary, wait_guidance, result_summary,
failure_summary
```

Do not project job ids, queue state, worker names, leases, retry counters,
adapter callback data, or database field names into cognition or dialog
prompts.

## Result Delivery

Accepted-task-backed worker rows emit `accepted_task_result_ready` cognition
episodes when complete. Legacy worker rows without an accepted-task id may
still emit `background_work_result_ready` for compatibility.

Dispatcher delivery synchronizes the accepted-task row to delivered or a
retryable delivery-failure state. Operators diagnose executor details through
internal job rows; the character receives only the accepted-task result or
failure summary.
