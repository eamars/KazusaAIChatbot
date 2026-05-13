# Self-Cognition Module

This module is the canonical ICD for `kazusa_ai_chatbot.self_cognition`.
It owns the source packet, route tracking, action-attempt, local ledger,
dispatcher-handoff, worker, and local artifact contracts for the idle
self-cognition agency loop.

## Boundary

The module supports two entry points:

- The dry-run command reads caller-supplied case files, optionally invokes the
  existing RAG2 supervisor once, invokes the existing shared L1/L2/L3 cognition
  graph, optionally invokes the existing dialog graph once to render a message
  after cognition selects outward contact without explicit candidate text, and
  writes local artifacts under the requested output directory.
- The opt-in service worker collects bounded visible/actionable source cases,
  invokes the same dry-run core, records a local action-attempt ledger, and may
  hand a cognition-selected `send_message` action candidate to the existing
  `TaskDispatcher`.

Production behavior is unchanged while `SELF_COGNITION_ENABLED=false`.
When it is enabled, the only allowed outward production side effect is the
normal dispatcher/scheduler handling of a non-duplicate action candidate.

The module does not call adapters directly, write `/chat` conversation rows,
run live-chat consolidation, update reflection state, update stable memory, or
update conversation progress/history. Dispatcher rejection is recorded locally
and must not be converted into an adapter send.

## Configuration

Central settings live in `kazusa_ai_chatbot.config`:

- `SELF_COGNITION_ENABLED`, default `false`.
- `SELF_COGNITION_WORKER_INTERVAL_SECONDS`, default `3600`.
- `SELF_COGNITION_MAX_CASES_PER_TICK`, default `3`.
- `SELF_COGNITION_TRACKING_DIR`, default `self_cognition_runs`.
- `SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT`, default `4000`.
- `SELF_COGNITION_RAG_EVIDENCE_CHAR_LIMIT`, default `4000`.
- Trigger-source enablement flags, all default `true`:
  `SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED`,
  `SELF_COGNITION_TRIGGER_CONVERSATION_PROGRESS_ENABLED`,
  `SELF_COGNITION_TRIGGER_RECENT_DIRECT_DIALOG_ENABLED`,
  `SELF_COGNITION_TRIGGER_PENDING_OUTBOX_ENABLED`,
  `SELF_COGNITION_TRIGGER_BOUNDED_TOPIC_FOLLOWUP_ENABLED`, and
  `SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED`.

Trigger flags control source collector eligibility only. They do not override
cognition's route or contact decision.

## Runtime Engine Budget

- Non-RAG cases may invoke the shared cognition graph once.
- RAG follow-up cases may invoke the RAG2 supervisor once before the shared
  cognition graph. Internal RAG2 helper calls remain governed by RAG2.
- If cognition selects outward contact but does not emit explicit
  `[ACTION_CANDIDATE]` text, the runner may invoke the existing dialog graph
  once to render the local action candidate text.
- Duplicate or held action attempts do not invoke dialog rendering because no
  send candidate can be created.

## Public Interface

- `sources.collect_self_cognition_cases(...)`
- `sources.collect_active_commitment_cases(...)`
- `tracking.build_idempotency_key(...)`
- `tracking.build_trigger_record(case)`
- `tracking.build_run_record(case, trigger_record, selected_route, budget)`
- `tracking.build_route_effect(run_record, route, consumer, effect_summary, next_topic=None)`
- `tracking.classify_route(case, cognition_output, action_attempt=None)`
- `tracking.build_action_attempt(case, trigger_record, existing_attempts)`
- `tracking.build_action_candidate(case, action_attempt, text)`
- `runner.run_self_cognition_case(case, output_dir, rag_client=None, cognition_client=None)`
- `artifacts.write_tracking_artifacts(output_dir, artifacts)`
- `artifacts.read_action_attempt_ledger(root_dir)`
- `artifacts.append_action_attempt_ledger(root_dir, attempt)`
- `handoff.build_raw_tool_call(action_candidate)`
- `handoff.dispatch_action_candidate(case, action_attempt, action_candidate, dispatcher, now)`
- `worker.run_self_cognition_worker_tick(...)`
- `worker.start_self_cognition_worker(...)`
- `worker.stop_self_cognition_worker(...)`

## Supported Cases

- `commitment_before_due`
- `commitment_past_due`
- `commitment_duplicate_tick`
- `private_no_action`
- `group_noise_rejected`
- `topic_rag_followup`

`commitment_past_due` and `commitment_duplicate_tick` do not force contact.
If shared cognition does not select outward contact, the route is recorded as
a silence, audit, or progress observation.

## Repeat Suppression

Action idempotency is based on source kind, source id, due time, target scope,
and action kind. Generated message text is not part of the identity.

Existing attempts with these statuses suppress a new send candidate for the
same idempotency key: `candidate`, `held`, `pending_handoff`,
`handoff_accepted`, `scheduled`, `sent`, and `duplicate_suppressed`.

The live worker stores suppression history in
`SELF_COGNITION_TRACKING_DIR/self_cognition_action_attempts.jsonl`. This file
is the module's own tracking system and is not a production database schema.

## Event Logging

The production worker mirrors sanitized trigger, run, route, action-attempt,
and dispatcher-result metadata through `kazusa_ai_chatbot.event_logging`.
This event-log mirror is the durable operator view for long-term production
counts and `/ops/self-cognition/stats`.

Local artifacts remain the canonical dry-run and debug output. Existing
artifact files and the local action-attempt ledger are not backfilled into
production MongoDB by this module. Event-log rows store ids, route names,
output modes, budget counters, dispatch status, and status labels; they must
not include source packet text, action candidate text, raw target channels, or
conversation bodies.

## Artifacts

The dry-run writer or live worker may produce:

- `self_cognition_trigger_record.json`
- `self_cognition_run_record.json`
- `self_cognition_rag_request.json`
- `self_cognition_rag_output.json`
- `self_cognition_cognition_input_after_rag.json`
- `self_cognition_cognition_output.json`
- `self_cognition_route_effect.json`
- `self_cognition_action_attempt.json`
- `self_cognition_action_candidate.json`
- `self_cognition_dispatch_result.json`
- `self_cognition_loop_trace.md`
- `self_cognition_action_attempts.jsonl`

Action candidates always use `dispatch_shape: "send_message"` and
`production_handoff: false`. In live mode, actual handoff state is represented
by `self_cognition_dispatch_result.json` and the local action-attempt ledger.

## Command

```powershell
venv\Scripts\python -m scripts.run_self_cognition_dry_run --case-file <path> --output-dir <path>
```

The command rejects missing files, malformed JSON, and unsupported case names
before creating the output directory.

## SC-TRACKING-ICD-001

The required local artifact shapes are:

```python
self_cognition_trigger_record = {
    "trigger_id": str,
    "trigger_kind": str,
    "target_scope": {
        "platform": str,
        "platform_channel_id": str,
        "channel_type": str,
        "user_id": str | None,
    },
    "source_refs": list[dict],
    "semantic_due_state": str | None,
    "actionability": str,
    "status": str,
}

self_cognition_run_record = {
    "run_id": str,
    "trigger_id": str,
    "idle_timestamp": str,
    "output_mode": "silent" | "preview" | "scheduled_action_request",
    "selected_route": str,
    "status": str,
    "evidence_refs": list[dict],
    "budget": {
        "rag_calls": int,
        "cognition_calls": int,
        "dialog_calls": int,
        "topic_limit": int,
    },
}

self_cognition_route_effect = {
    "run_id": str,
    "route": str,
    "consumer": str,
    "production_write": bool,
    "effect_summary": str,
    "next_topic": dict | None,
}

self_cognition_action_attempt = {
    "attempt_id": str,
    "run_id": str,
    "trigger_id": str,
    "source_kind": str,
    "source_id": str,
    "target_scope": dict,
    "action_kind": "send_message",
    "due_at": str | None,
    "idempotency_key": str,
    "status": (
        "candidate" | "held" | "pending_handoff" | "handoff_accepted"
        | "scheduled" | "sent" | "duplicate_suppressed"
        | "closed_no_action"
    ),
}

self_cognition_action_candidate = {
    "attempt_id": str,
    "target_platform": str,
    "target_channel": str,
    "target_channel_type": str,
    "text": str,
    "execute_at": str | None,
    "dispatch_shape": "send_message",
    "production_handoff": False,
}

self_cognition_dispatch_result = {
    "attempt_id": str,
    "idempotency_key": str,
    "production_handoff": bool,
    "status": "accepted" | "rejected" | "not_requested",
    "dispatcher_called": bool,
    "scheduled_event_ids": list[str],
    "rejections": list[str],
}
```
