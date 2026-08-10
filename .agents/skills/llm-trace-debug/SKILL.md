---
name: llm-trace-debug
description: Retrieve and review protected Kazusa LLM trace evidence for generated dialog or pipeline failures, starting from full chat correlation identifiers such as chat:qq:ch_732699d6699040ae:1242015780, visible dialog text, platform message identifiers, delivery tracking ids, trace ids, or unclassified opaque identifiers.
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

## Identifier Classification

Classify the supplied value from its evidence and field provenance. Treat a
bare 32-character hexadecimal value, such as
`d63003933a924c03a258fcf9d891e6b5`, as unclassified: that shape can be a
`delivery_tracking_id`, `cognition_invocation_id`, or another UUID-derived
identifier. The value shape alone does not select an exporter flag.

Use these direct routes when the identifier type is established:

- `chat:...` is a service correlation id; use
  `export_by_correlation_id.py`.
- `llmtrace_...` or a value confirmed from an `llm_trace_id` field is a trace
  id; use `--trace-id`.
- A confirmed `delivery_tracking_id` uses `--delivery-tracking-id`.
- A confirmed `platform_message_id` uses `--platform-message-id`.
- A confirmed `cognition_invocation_id` is a selector after its parent trace is
  resolved; it is not an independent `export_llm_trace` lookup flag.

For a bare or otherwise unclassified value, perform exact candidate discovery
before exporting a trace. Use `scripts.export_collection` with a bounded limit
and separate output files. For example:

```powershell
$opaque_id = "d63003933a924c03a258fcf9d891e6b5"

venv\Scripts\python -m scripts.export_collection conversation_history `
  --filter ('{"$or":[{"delivery_tracking_id":"' + $opaque_id + '"},{"platform_message_id":"' + $opaque_id + '"},{"llm_trace_id":"' + $opaque_id + '"}]}') `
  --sort '{"timestamp":-1}' `
  --limit 100 `
  --output "test_artifacts\diagnostics\trace_id_discovery_conversation.json"

venv\Scripts\python -m scripts.export_collection llm_trace_runs `
  --filter ('{"trace_id":"' + $opaque_id + '"}') `
  --sort '{"started_at":-1}' `
  --limit 100 `
  --output "test_artifacts\diagnostics\trace_id_discovery_runs.json"

venv\Scripts\python -m scripts.export_collection llm_trace_steps `
  --filter ('{"$or":[{"trace_id":"' + $opaque_id + '"},{"cognition_invocation_id":"' + $opaque_id + '"}]}') `
  --sort '{"sequence":1,"created_at":1}' `
  --limit 500 `
  --exclude-field raw_messages `
  --exclude-field raw_response_text `
  --exclude-field parsed_output `
  --exclude-field capsule `
  --output "test_artifacts\diagnostics\trace_id_discovery_steps.json"
```

Read each export's `documents` array and collect exact matches plus their
`llm_trace_id` or `trace_id` values. Deduplicate the candidate trace ids.
Continue to the protected trace export only when the discovery produces one
consistent trace id and the candidate row has the expected identifier field.
Treat zero candidates as an unresolved identifier and multiple trace ids as an
ambiguous identifier; preserve the candidate exports and request the missing
identifier type or additional turn metadata. Never select a newest row merely
because the resolver sorts by time.

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

2. After identifier classification or exact candidate discovery, export the
   resolved trace id directly:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace --trace-id <llm_trace_id>
   ```

   A trace export is valid only when its `query.trace_id` equals the resolved
   id and at least one of `llm_trace_runs`, `llm_trace_steps`,
   `event_log_events`, or `conversation_history` contains evidence. File
   creation alone is not a successful lookup.

3. If you only have visible dialog text, resolve the trace from
   `conversation_history`:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace --dialog-text "<visible dialog>"
   ```

4. If you have delivery or platform metadata and the field type is known,
   prefer the stable identifiers:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace --delivery-tracking-id <id>
   venv\Scripts\python -m scripts.export_llm_trace --platform-message-id <id>
   ```

5. If a Cognition invocation id is known after trace resolution, select it in
   the protected export and verify the exact invocation appears in the
   exported `llm_trace_steps` or `cognition_failure_capsules`:

   ```powershell
   venv\Scripts\python -m scripts.export_llm_trace `
     --trace-id <llm_trace_id> `
     --cognition-invocation-id <cognition_invocation_id>
   ```

6. To create compact review input for LLM debug review:

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

## Control Console Trace-Correlation Handoff

For the agent use case `Look up the trace id of xxx`, `xxx` must be the exact
value copied from the Control Console Debug Chat metadata line `trace ...`.
Record the copied value and `source_surface=web_control_trace_id` before any
protected query. The line is the canonical browser trace surface; a graph
`run_id`, event `correlation_id`, delivery tracking id, action id, or bare
opaque value is not interchangeable with it.

Create the bounded identifier-only manifest first:

```powershell
venv\Scripts\python -m scripts.export_trace_correlation_manifest `
  --identifier <copied-value> `
  --source-surface web_control_trace_id `
  --output test_artifacts\diagnostics\trace_correlation_<name>.json
```

Inspect `parent_trace`, `identifiers`, `joins`, and `unresolved` in that order.
Continue to `scripts.export_llm_trace --trace-id <trace>` only when the parent
status is `confirmed`. Preserve `not_found`, `ambiguous`,
`not_available_from_web`, `not_captured`, `conflict`, and `not_available` as
terminal evidence; never select a newest candidate.

## Web Availability Matrix

| Identifier | Browser availability | Exact next route |
| --- | --- | --- |
| Debug Chat `trace_id` | rendered when authorized and retained | correlation manifest with `web_control_trace_id` |
| `delivery_tracking_id` | rendered separately from trace | `scripts.export_llm_trace --delivery-tracking-id <id>` after field confirmation |
| `platform_message_id` | API/request-only | `scripts.export_llm_trace --platform-message-id <id>` |
| `cognition_invocation_id` | protected-only | resolve parent, then the trace export selector |
| `global_user_id` | protected-only | correlation manifest with `protected_global_user_id` |
| action-attempt, background-job, accepted-task, calendar schedule/run ids | absent from current Console projection | correlation manifest with the matching protected source surface |
| graph `run_id` and event `correlation_id` | generic rendered surfaces, not trace ids | record `unknown`; manifest returns `not_available_from_web` until typed evidence exists |
| child/future execution trace ids | protected-only | inspect exact `joins.child_trace_runs` relations |

The manifest is an evidence handoff, not a raw trace export. Keep it under
`test_artifacts/diagnostics`, inspect one supplied anchor at a time, and keep
the separate protected trace export out of chat output.
