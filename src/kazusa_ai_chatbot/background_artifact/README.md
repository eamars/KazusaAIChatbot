# Background Artifact Compatibility

## Document Control

This ICD describes the legacy background-artifact compatibility boundary. New
work uses `kazusa_ai_chatbot.background_work`.

## Purpose

Background artifact rows were the earlier text-only async job format. They are
kept so old queued, completed, or deliverable rows can still finish through the
existing cognition and dialog boundary. New live turns should queue
`background_work_request` jobs instead.

## Compatibility Scope

The legacy package may process existing text-artifact rows and old
`BACKGROUND_ARTIFACT_*` configuration aliases. It is not the top-level runtime
contract for new work. L2d does not receive or emit `background_artifact_request`
for new turns.

## Current Runtime Boundary

New background work follows this path:

```text
L2d background_work_request
  -> action-spec validation
  -> db.background_work_jobs queued row
  -> background_work.router route-only worker choice
  -> background_work.subagent.* worker execution
  -> background_work_result_ready cognition
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
