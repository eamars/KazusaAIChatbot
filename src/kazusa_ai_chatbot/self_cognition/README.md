# Self-Cognition Module

This module is the canonical ICD for `kazusa_ai_chatbot.self_cognition`.
It owns source collection, trigger packet construction, route tracking,
action-attempt compatibility, worker, production attempt persistence, and
dry-run artifact contracts for the idle self-cognition agency loop.

## Boundary

The module supports two entry points:

- The dry-run command reads caller-supplied case files, optionally invokes the
  existing RAG2 supervisor once, invokes the existing shared L1/L2/L2d
  cognition graph, runs selected L3 text/dialog only when L2d selects a
  visible `speak` surface, and writes local artifacts under the requested
  output directory.
- The service worker collects bounded visible/actionable source cases,
  binds `SelfCognitionDeliveryTarget` before cognition, builds the same route
  records in memory, invokes the existing dialog graph only for selected
  visible `speak` rendering, calls the existing consolidator through the shared
  same-path entry, records sanitized event-log telemetry, persists
  action-attempt state through the DB facade, and dispatches selected `speak`
  through the runtime adapter bridge.

Self-cognition is an upstream trigger source for the shared persona path. It is
not a downstream action consumer, private cleanup channel, adapter sender, or
commitment-retirement executor. Once a case is collected, the rest of the
workflow follows the same cognition, action initialization, action evaluation,
surface handling, and consolidation boundaries as other cognitive episodes.

The current compatibility episode label in some prompt-selection and origin
metadata code is `internal_thought`; architecturally those cases are
self-cognition triggers and should not be treated as a separate action path.

The production worker is enabled by default with `SELF_COGNITION_ENABLED=true`.
Set it to `false` to suppress self-cognition worker activation. The only
allowed delayed production side effect is the normal future-cognition scheduler
path, where the later cognition cycle decides again whether any visible output
should exist.

Self-cognition-created episodes set
`origin_metadata.debug_modes.no_visual_directives=true` by default, so the
shared L3 visual-directive LLM is skipped for self-cognition. These episodes do
not set `no_remember`. Production worker consolidation can update the existing
character-state, relationship, affinity, memory-unit, and cache lanes through
the shared consolidator policy. It does not create a separate self-cognition
memory or progress store.

The module does not call platform adapters directly from graph code, route
direct `/chat` requests through the worker, or schedule prewritten user-visible
text. Production selected `speak` must attempt delivery after dialog rendering:
the dispatcher persists the assistant outbound row, looks up the registered
runtime adapter, sends to the bound target, and returns terminal delivery
metadata for action-attempt persistence.

Consolidation can run even when self-cognition selects no visible action.
Action results, episode-trace evidence, and an empty `final_dialog` are
sufficient to make the episode consolidatable. The consolidator consumes
prompt-safe evidence and does not execute actions, dispatch, schedule, or
trigger cognition.

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
- If L2d selects visible `speak`, the selected L3 text handler may invoke the
  existing dialog graph once to render text.
- When consolidation is applied without a selected visible `speak`, the runner
  reuses an empty `final_dialog` and does not invoke dialog.
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
- `tracking.build_action_candidate(case, action_attempt, text, mention_target_user=False)`
- `runner.build_self_cognition_case_artifacts(case, rag_client=None, cognition_client=None, dialog_client=None, consolidation_client=None, apply_consolidation=False)`
- `runner.build_self_cognition_case_artifacts_async(case, rag_client=None, cognition_client=None, dialog_client=None, consolidation_client=None, apply_consolidation=False)`
- `runner.run_self_cognition_case(case, output_dir, rag_client=None, cognition_client=None, dialog_client=None, consolidation_client=None, apply_consolidation=False)`
- `artifacts.write_tracking_artifacts(output_dir, artifacts)`
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

Past-due commitments do not disappear because of due age. Shared cognition may
select a private `memory_lifecycle_update` route, but that route is handed to
the memory lifecycle specialist before execution. A commitment can be retired
only after the specialist chooses a prompt-safe alias and deterministic code
resolves it into an executable `apply_memory_lifecycle_update` action for one
eligible `user_memory_units.active_commitment`. If the alias cannot be bound,
the lifecycle request is rejected before persistence.

## Repeat Suppression

Action idempotency is based on source kind, source id, due time, target scope,
and action kind. Generated message text is not part of the identity.

Existing attempts with these statuses suppress a new send candidate for the
same idempotency key: `candidate`, `held`, `pending_handoff`,
`handoff_accepted`, `scheduled`, `sent`, and `duplicate_suppressed`.

The live worker stores suppression and audit history in the
`self_cognition_action_attempts` MongoDB collection through
`kazusa_ai_chatbot.db` helpers. That collection now also backs generic
action-attempt metadata for the shared action-spec layer. Old send-message
attempt rows remain readable, and new rows must stay tolerant of the older
shape. Event logging mirrors sanitized trigger, run, route, action-attempt, and
consolidation metadata for operators, but event logs are not used as production
control state.

## Delivery Target Binding

Production source collectors attach either a bound
`SelfCognitionDeliveryTarget` or a target-binding failure before cognition.
The resolver prefers the latest known private channel on the same platform for
the semantic target user. If no known private channel exists, it uses the
self-cognition source channel when that source is a valid private or group
channel. Cases without a valid target are recorded as `target_binding_failed`
and stop before RAG, cognition, dialog, consolidation, and delivery.

`SelfCognitionDeliveryTarget` is deterministic runtime metadata. It must stay
out of source packets, RAG requests, cognition state, dialog state, prompts,
prompt anchors, prompt schemas, and model-facing artifacts.

## Delivery Mentions

Self-cognition may attach one platform-neutral `delivery_mentions` request to
a local action candidate artifact when the shared dialog graph
returns `mention_target_user=true` and the case has a semantic target user in
`target_scope.user_id`.

The target scope may carry delivery-only platform identity:

```python
{
    "platform": str,
    "platform_channel_id": str,
    "channel_type": str,
    "user_id": str | None,
    "platform_user_id": str | None,
    "display_name": str,
}
```

`platform_user_id` and `display_name` are not model context. They must not be
rendered into source packets, RAG requests, cognition state, dialog state, or
prompt text. They also do not participate in action-attempt idempotency.

Action candidates may carry:

```python
"delivery_mentions": [
    {
        "entity_kind": "user",
        "placement": "prefix",
        "platform_user_id": str | None,
        "global_user_id": str | None,
        "display_name": str,
        "requested_by": "dialog.mention_target_user",
    }
]
```

Self-cognition does not decide adapter capability, channel feasibility, native
mention syntax, or delivery. It only carries the dialog-owned semantic mention
request as local tracking metadata.

## Future Cognition Handoff

Self-cognition may produce local delivery-candidate artifacts for dry-run
inspection and duplicate suppression. In production worker ticks, selected
`speak` requires a bound delivery target, L3 text directives, and dialog in the
shared cognition path, then the worker hands the final text to dispatcher
delivery in the same tick. Dry-run candidate rendering remains inspection-only
and does not deliver.

`trigger_future_cognition` uses `scheduled_events` as an internal delayed
trigger source. The action handler records a private non-dispatcher slot; a
later self-cognition worker tick collects due slots as normal source cases and
marks the source slot completed after the normal runner path returns. The
model-facing source packet receives only semantic follow-up context, not
scheduler ids, action-attempt ids, continuation schema versions, or depth
limits.

## Event Logging

The production worker mirrors sanitized trigger, run, route, action-attempt,
and consolidation-outcome metadata through
`kazusa_ai_chatbot.event_logging`. This event-log mirror is the durable
operator view for long-term production counts and `/ops/self-cognition/stats`.

Dry-run artifacts remain the canonical debug output. The production worker
does not write artifact files. Event-log rows store ids, route names, output
modes, budget counters, consolidation write-success booleans,
scheduled-event counts, cache-eviction counts, origin labels, and status
labels; they must not include source packet text, action candidate text, raw
target channels, or conversation bodies.

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
- `self_cognition_consolidation_outcome.json`
- `self_cognition_loop_trace.md`

Legacy dry-run delivery candidates use `dispatch_shape: "send_message"` and
an inspection-only handoff marker in dry-run artifacts. In live mode,
production selected speech records a terminal action-attempt status such as
`sent`, `delivery_failed`, `held`, or `duplicate_suppressed`.

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
    "idle_timestamp_utc": str,
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
    "inspection_only": True,
    "delivery_mentions": list[dict],  # optional
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
