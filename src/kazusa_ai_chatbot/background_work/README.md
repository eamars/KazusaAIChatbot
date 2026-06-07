# Background Work

## Document Control

This ICD defines the generic background-work subsystem boundary for new
asynchronous jobs.

## Purpose

`background_work` owns generic asynchronous work accepted during a live
persona turn. The live path queues only a bounded task brief and prompt-safe
source scope. Worker routing, worker-local task classification, generation,
and result delivery happen after the live response path.

## Boundary

- L2d may request only generic `background_work_request` handoff.
- The background-work router emits only `action`, `worker`, and `reason`.
- Worker subagents own worker-local semantic parameters and artifacts.
- L3/dialog remain the only visible wording owners.
- Workers never send adapter text, call cognition directly, run shell or
  filesystem work, install packages, process attachments, or mutate
  persistence outside the public worker result contract.

## First Worker

`subagent.text_artifact` is the only enabled production worker. It has two
separate LLM stages:

1. Task router: chooses `coding_snippet`, `text_rewrite`, `summary`,
   `unsupported`, or `needs_user_input`.
2. Generator: produces one bounded text artifact or a failure summary.

The generic queue and router do not expose those worker-local task labels or
worker-facing task rewrites.

## Persistence

Raw MongoDB access lives in `kazusa_ai_chatbot.db.background_work_jobs`.
Callers use the public queue/runtime exports from `kazusa_ai_chatbot.background_work`.
Jobs move through queued, in-progress, completed or failed, delivery in
progress, delivered, and delivery failed states.

## Compatibility

`background_artifact` remains only as legacy text-artifact support for old
rows. New live-turn background requests use `background_work`.
