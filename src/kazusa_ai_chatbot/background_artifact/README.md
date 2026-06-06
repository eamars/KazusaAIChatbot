# Background Artifact

## Document Control

This ICD describes the Stage 1 background artifact handoff boundary.

## Purpose

Background artifact jobs let cognition accept bounded text-only work that can
finish after the live response turn. The live turn may acknowledge a queued job,
but the finished artifact re-enters the character brain as a source-bound
cognitive episode.

## Scope

Stage 1 supports only text artifacts for `coding_snippet`, `text_rewrite`, and
`summary`. Jobs do not write files, run shell commands, install packages, mutate
repositories, download assets, perform web research, create images, handle
attachments, or stream chunked delivery.

## Parties

- L2d selects the semantic `background_artifact_request` capability.
- Action-spec execution validates and queues the request.
- `background_artifact.worker` claims jobs and produces text artifacts.
- `background_artifact.delivery` turns finished jobs into result-ready cognition
  sources and hands final delivery to the existing service boundary.
- `db.background_artifact_jobs` owns raw MongoDB access for job rows.

## Boundary Summary

Adapters remain thin. The worker never sends user-visible text directly. Result
delivery must go through cognition, dialog, and the dispatcher boundary owned by
the service layer. Calendar scheduler, proactive output, and future-cognition
polling are not job owners for this subsystem.

## Public Interface

- `BackgroundArtifactQueueRequest`
- `BackgroundArtifactQueueResult`
- `BackgroundArtifactRuntimeHandle`
- `enqueue_background_artifact_request(...)`
- `run_background_artifact_runtime_tick(...)`
- `start_background_artifact_runtime(...)`
- `stop_background_artifact_runtime(...)`

Runtime callers use these entrypoints and the public DB facade exports. Raw
MongoDB calls stay inside `kazusa_ai_chatbot.db.background_artifact_jobs`.

## Job Lifecycle

Jobs move through `queued`, `in_progress`, `completed`, `failed`,
`delivery_in_progress`, `delivered`, and `delivery_failed`. Queue acknowledgement
is allowed only after a durable queued row exists. Completed or failed jobs are
converted into `background_artifact_result_ready` cognitive episodes before any
visible result is delivered.

## LLM Input Contract

The artifact worker sees only semantic work fields:

- `work_kind`
- `objective`
- `input_summary`
- `max_output_chars`

The prompt and payload exclude adapter IDs, target channels, job leases, retry
state, database names, filesystem paths, credentials, and queue mechanics.

## Forbidden Paths

The feature must not use `calendar_scheduler`, `proactive_output`,
consolidator-generated promises, or `trigger_future_cognition` as the job owner
or polling loop. The worker must not import adapter delivery or dispatcher
modules, and it must not perform execution, filesystem, network research, image,
attachment, package-install, or repository mutation work.
