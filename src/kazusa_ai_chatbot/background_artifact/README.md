# Background Artifact Compatibility

## Document Control

This ICD describes the legacy background-artifact compatibility boundary. New
work uses `kazusa_ai_chatbot.background_work`.

## Purpose

Background artifact rows were the earlier text-only async job format. They are
kept so old queued, completed, or deliverable rows can still finish through the
existing cognition and dialog boundary. New live turns should create
accepted-task-backed internal background-work jobs instead.

## Compatibility Scope

The legacy package may process existing text-artifact rows and old
`BACKGROUND_ARTIFACT_*` configuration aliases. It is not the top-level runtime
contract for new work. L2d does not receive or emit `background_artifact_request`
for new turns.

## Current Runtime Boundary

New delayed work follows this path:

```text
L2d accepted_task_request
  -> accepted_task lifecycle validation and duplicate rejection
  -> internal background_work_request
  -> db.background_work_jobs queued row with accepted_task_id
  -> background_work.router route-only worker choice
  -> background_work.subagent.* worker execution
  -> accepted_task_result_ready cognition
  -> L3/dialog owned final visible wording
```

Workers never send adapter text directly, call cognition directly, write files,
run shell commands, install packages, mutate repositories, download assets,
perform web research, create images, handle attachments, or stream chunked
delivery.

## Legacy Public Interface

The old `BackgroundArtifact*` exports and `db.background_artifact_jobs` facade
remain for compatibility with existing rows. Runtime callers should prefer the
generic `BackgroundWork*` exports and `db.background_work_jobs` facade.
