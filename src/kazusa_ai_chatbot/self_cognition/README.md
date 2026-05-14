# Self-Cognition Module

This module is the canonical ICD for `kazusa_ai_chatbot.self_cognition`.
It owns the source packet, route tracking, action-attempt,
dispatcher-handoff, worker, production attempt persistence, and dry-run
artifact contracts for the idle self-cognition agency loop.

## Boundary

The module supports two entry points:

- The dry-run command reads caller-supplied case files, optionally invokes the
  existing RAG2 supervisor once, invokes the existing shared L1/L2/L3 cognition
  graph, optionally invokes the existing dialog graph once to render a message
  after cognition selects outward contact without explicit candidate text, and
  writes local artifacts under the requested output directory.
- The service worker collects bounded visible/actionable source cases,
  builds the same route records in memory, invokes the existing dialog graph
  for private finalization when consolidation is applied, calls the existing
  consolidator through the shared same-path entry, records sanitized event-log
  telemetry, persists action-attempt state through the DB facade, and may hand
  a cognition-selected `send_message` action candidate to the existing
  `TaskDispatcher`.

The production worker is enabled by default with `SELF_COGNITION_ENABLED=true`.
Set it to `false` to suppress self-cognition worker activation. The only
allowed outward production side effect is the normal dispatcher/scheduler
handling of a non-duplicate action candidate.

Self-cognition-created episodes set
`origin_metadata.debug_modes.no_visual_directives=true` by default, so the
shared L3 visual-directive LLM is skipped for self-cognition. These episodes do
not set `no_remember`. Production worker consolidation can update the existing
character-state, relationship, affinity, memory-unit, task-dispatch, and cache
lanes through the shared consolidator policy. It does not create a separate
self-cognition memory or progress store.

The module does not call adapters directly or write `/chat` conversation rows.
Private finalization exists only to feed the shared consolidator and optional
action-candidate rendering. Dispatcher rejection is recorded through event
logging and persisted action-attempt state; it must not be converted into an
adapter send.

## Configuration

Central settings live in `kazusa_ai_chatbot.config`:

- `SELF_COGNITION_ENABLED`, default `true`.
- `SELF_COGNITION_WORKER_INTERVAL_SECONDS`, default `3600`.
- `SELF_COGNITION_MAX_CASES_PER_TICK`, default `3`.
- `SELF_COGNITION_TRACKING_DIR`, default `self_cognition_runs`; used only by
  explicit dry-run/debug artifact writers, not by the production worker.
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
- When consolidation is applied, the runner may invoke the existing dialog
  graph once for private finalization even when no send candidate can be
  created.
- The production worker applies consolidation by default and keeps the existing
  `SELF_COGNITION_MAX_CASES_PER_TICK` case cap.

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
- `runner.build_self_cognition_case_artifacts(case, rag_client=None, cognition_client=None, dialog_client=None, consolidation_client=None, apply_consolidation=False)`
- `runner.build_self_cognition_case_artifacts_async(case, rag_client=None, cognition_client=None, dialog_client=None, consolidation_client=None, apply_consolidation=False)`
- `runner.run_self_cognition_case(case, output_dir, rag_client=None, cognition_client=None, dialog_client=None, consolidation_client=None, apply_consolidation=False)`
- `artifacts.write_tracking_artifacts(output_dir, artifacts)`
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

The live worker stores suppression history in the
`self_cognition_action_attempts` MongoDB collection through
`kazusa_ai_chatbot.db` helpers. Event logging mirrors sanitized run and
dispatch metadata for operators, but event logs are not used as production
control state.

## Event Logging

The production worker mirrors sanitized trigger, run, route, action-attempt,
consolidation-outcome, and dispatcher-result metadata through
`kazusa_ai_chatbot.event_logging`. This event-log mirror is the durable
operator view for long-term production counts and `/ops/self-cognition/stats`.

Dry-run artifacts remain the canonical debug output. The production worker
does not write artifact files. Event-log rows store ids, route names, output
modes, budget counters, dispatch status, consolidation write-success booleans,
scheduled-event counts, cache-eviction counts, origin labels, and status
labels; they must not include source packet text, private finalization text,
action candidate text, raw target channels, or conversation bodies.

## Artifacts

The dry-run writer may produce:

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
- `self_cognition_consolidation_outcome.json`
- `self_cognition_loop_trace.md`

Action candidates always use `dispatch_shape: "send_message"` and
`production_handoff: false` in dry-run artifacts. In live mode, actual handoff
state is represented by event logging, scheduler rows, and the
`self_cognition_action_attempts` collection.

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

self_cognition_consolidation_outcome = {
    "consolidation_called": bool,
    "write_success": dict[str, bool],
    "scheduled_event_count": int,
    "cache_evicted_count": int,
    "origin_trigger_source": "internal_thought",
    "origin_episode_id": str,
}
```
