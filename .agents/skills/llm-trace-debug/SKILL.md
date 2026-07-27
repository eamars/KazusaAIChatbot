---
name: llm-trace-debug
description: Retrieve and review protected Kazusa LLM trace evidence for generated dialog or pipeline failures, starting from full chat correlation identifiers such as chat:qq:ch_732699d6699040ae:1242015780, visible dialog text, platform message identifiers, delivery tracking ids, or trace ids.
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

Conversation rows and audit rows link to the trace by `llm_trace_id`. Runtime
failure events may instead use the service correlation id
`chat:<platform>:<hashed-channel-ref>:<platform-message-id>`.

## Retrieval Workflow

1. If the user supplies a full `chat:...` correlation id, use the bundled
   exporter first. Pass the identifier unchanged; its `ch_...` component is a
   hashed channel reference, not the platform channel id.

   ```powershell
   venv\Scripts\python .agents\skills\llm-trace-debug\scripts\export_by_correlation_id.py `
     "chat:qq:ch_732699d6699040ae:1242015780"
   ```

   The exporter performs the complete join in one operation:

   - exact `event_log_events.correlation_id` lookup;
   - platform and platform-message lookup in `conversation_history`;
   - `llm_trace_id` resolution;
   - protected trace-run, trace-step, trace-linked event, and conversation
     export.

   Use its single JSON output as the complete correlation-id review source.

2. If you have a trace id, export it directly:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace --trace-id <llm_trace_id>
   ```

3. If you only have visible dialog text, resolve the trace from
   `conversation_history`:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace --dialog-text "<visible dialog>"
   ```

4. If you have delivery or platform metadata, prefer the stable identifiers:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace --delivery-tracking-id <id>
   venv\Scripts\python -m scripts.export_llm_trace --platform-message-id <id>
   ```

5. To create compact review input for LLM debug review:

   ```powershell
   venv\Scripts\python -m scripts.export_dialog_trace_review_input --dialog-text "<visible dialog>"
   ```

## Review Procedure

Read the export in this order:

1. `correlation_event_log_events`: confirm runtime exception, stack
   fingerprint, pipeline stages, status, and final outcome when using a chat
   correlation id.
2. `conversation_history`: confirm visible input/output, timestamps, and
   `llm_trace_id`.
3. Each `llm_traces` entry: inspect its run, trace-linked events, and ordered
   steps. For direct trace exports, inspect the equivalent top-level fields.
4. `llm_trace_steps`: inspect stage order, prompt/output hashes, parse status,
   output state fields, and raw payloads if full capture was enabled.
5. Compare the failure event or final dialog against upstream parsed outputs
   and output state fields before making a causal claim.

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
