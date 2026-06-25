---
name: llm-trace-debug
description: Retrieve and review protected Kazusa LLM trace evidence for generated dialog, starting from visible dialog text, message identifiers, delivery tracking ids, or trace ids.
---

# LLM Trace Debug

Use this skill when investigating why a generated dialog was produced, when a
user gives visible dialog text and asks for the LLM handoff trail, or when you
need prompt/output evidence for a specific chat turn.

## What This Skill Retrieves

The sanitized event log is an audit index only. It does not contain raw prompts,
raw model outputs, parsed model output, final dialog text, evidence text, or
adapter wire payloads.

LLM decision evidence lives in the protected trace collections:

- `llm_trace_runs`
- `llm_trace_steps`

Conversation rows and audit rows link to the trace by `llm_trace_id`.

## Retrieval Workflow

1. If you have a trace id, export it directly:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace --trace-id <llm_trace_id>
   ```

2. If you only have visible dialog text, resolve the trace from
   `conversation_history`:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace --dialog-text "<visible dialog>"
   ```

3. If you have delivery or platform metadata, prefer the stable identifiers:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace --delivery-tracking-id <id>
   venv\Scripts\python -m scripts.export_llm_trace --platform-message-id <id>
   ```

4. To create compact review input for LLM debug review:

   ```powershell
   venv\Scripts\python -m scripts.export_dialog_trace_review_input --dialog-text "<visible dialog>"
   ```

## Review Procedure

Read the export in this order:

1. `conversation_history`: confirm visible input/output, timestamps, and
   `llm_trace_id`.
2. `event_log_events`: confirm pipeline stages, statuses, and sanitized timing.
3. `llm_trace_steps`: inspect stage order, prompt/output hashes, parse status,
   output state fields, and raw payloads if full capture was enabled.
4. Compare the final dialog against upstream parsed outputs and output state
   fields before making a causal claim.

## Capture Modes

`LLM_TRACE_CAPTURE_MODE=metadata` is the default. Metadata mode stores counts,
hashes, stage names, parser status, state handoff fields, timing, and model
metadata, but raw prompt/output fields are empty.

`LLM_TRACE_CAPTURE_MODE=full` stores raw prompt messages, raw response text, and
parsed output in protected trace collections.

`LLM_TRACE_CAPTURE_MODE=off` skips trace row writes. In that mode, use
`event_log_events` and `conversation_history` only.

## Retention

Logging retention is controlled by two shared settings:

- `AUDIT_LOG_TTL_DAYS` covers sanitized audit/event-log data.
- `DEBUG_LOG_TTL_DAYS` covers protected debug trace data.

Do not introduce per-collection TTL settings for trace debugging.
