# LLM Tracing

This package owns the protected debug trace lane for model-stage prompts,
outputs, parsed results, and state handoff metadata.

`event_log_events` remains sanitized audit data. It may carry `llm_trace_id`
as a correlation label or reference, but it must not store raw prompt text,
raw model output, parsed model output, final dialog text, evidence text, or
adapter payloads.

Retention is governed by the shared logging retention settings:

- `AUDIT_LOG_TTL_DAYS` for sanitized audit/event-log rows.
- `DEBUG_LOG_TTL_DAYS` for debug trace rows.

`LLM_TRACE_CAPTURE_MODE` controls capture sensitivity:

- `off`: no trace-run or trace-step rows are written, but callers may still
  propagate `llm_trace_id`.
- `metadata`: default mode. Stores hashes, character counts, model/route
  metadata, parse status, state fields, and timing.
- `full`: stores raw prompt messages, raw response text, and parsed output in
  the protected trace collections.
