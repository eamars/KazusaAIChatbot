# Self-Cognition Module

This module is the canonical ICD for `kazusa_ai_chatbot.self_cognition`.
It owns source collection, trigger packet construction, route tracking,
action-attempt compatibility, worker execution, production attempt
persistence, and in-memory tracking records for the idle self-cognition agency
loop.

## Boundary

The module supports one production entry point: the service worker collects
bounded visible/actionable source cases, binds `SelfCognitionDeliveryTarget`
before cognition, builds route records in memory, invokes the existing dialog
graph only for selected visible `speak` rendering, calls the existing
consolidator through the shared same-path entry, records sanitized event-log
telemetry, persists action-attempt state through the DB facade, and dispatches
selected `speak` through the runtime adapter bridge.

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

Group chat review is hosted by the reflection-cycle worker, not by the
standalone self-cognition worker interval. Reflection phase slots select at
most one monitor-eligible group, build windows only for that selected group,
and pass at most one `group_chat_review` case into the normal self-cognition
runner. Self-cognition then owns routing, dialog rendering, attempt
persistence, consolidation, and source-bound delivery.

Self-cognition-created episodes set
`origin_metadata.debug_modes.no_visual_directives=true` by default, so the
shared L3 visual-directive LLM is skipped for self-cognition. These episodes do
not set `no_remember`. Production worker consolidation goes through the shared
target-aware consolidator policy. Real user-scoped cases may update existing
relationship, affinity, memory-unit, and cache lanes. Group-scoped or
targetless cases do not fabricate a user id and cannot reach user-profile
lanes. Self-cognition does not create a separate self-cognition memory or
progress store.

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
- `SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT`, default `4000`.
- `CHARACTER_SLEEP_LOCAL_PERIOD`, default `02:00-12:00` in
  `CHARACTER_TIME_ZONE`; empty disables sleep-period suppression.
- Trigger-source enablement flags, all default `true`:
  `SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED`,
  `SELF_COGNITION_TRIGGER_CONVERSATION_PROGRESS_ENABLED`,
  `SELF_COGNITION_TRIGGER_RECENT_DIRECT_DIALOG_ENABLED`,
  `SELF_COGNITION_TRIGGER_PENDING_OUTBOX_ENABLED`,
  `SELF_COGNITION_TRIGGER_BOUNDED_TOPIC_FOLLOWUP_ENABLED`, and
  `SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED`.

Trigger flags control source collector eligibility only. They do not override
cognition's route or contact decision.

The self-cognition module owns sleep-period trigger suppression through
`sleep_period.is_self_cognition_sleep_period(...)`. During the configured
local sleep period, production source selection skips active-commitment due
checks, and the reflection-cycle group review phase handler skips group
self-cognition before case collection. Due scheduled future-cognition slots
remain eligible. Reflection, consolidation, scheduler execution, dispatcher
validation, and adapter delivery are not paused by this predicate.

Daily affect settling is owned by `kazusa_ai_chatbot.reflection_cycle`, not by
this module. The service may inject
`should_pause_for_affect_settling` into `worker.start_self_cognition_worker` or
`worker.run_self_cognition_worker_tick`. The worker calls that probe after the
primary-interaction busy check and before source collection. When the probe
returns true, the tick records a deferred worker result with
`defer_reason="daily affect settling pending"` and does not collect source
cases. The self-cognition package must not import reflection-cycle internals.

## Runtime Engine Budget

- Every case enters the bounded cognition resolver and each resolver cycle
  invokes the shared L1/L2/L2d cognition graph.
- If L2d selects `local_context_recall`, the resolver invokes the existing RAG2
  supervisor as a capability observation and feeds the projected result into
  the next cognition cycle. Internal RAG2 helper calls remain governed by RAG2.
- If L2d selects `public_answer_research`, public/current/external answer
  investigation is handled by the complex task resolver through declared IO.
- If L2d selects neither recall nor public answer research, no deterministic
  self-cognition rule calls either capability on its behalf.
- If L2d selects visible `speak`, the selected L3 text handler may invoke the
  existing dialog graph once to render text.
- When consolidation is applied without a selected visible `speak`, the runner
  reuses an empty `final_dialog` and does not invoke dialog.
- The production worker applies consolidation by default and keeps the existing
  `SELF_COGNITION_MAX_CASES_PER_TICK` case cap. Reflection-attached group
  review uses the reflection phase invariant
  `REFLECTION_PHASE_GROUPS_PER_SLOT=1` instead of the standalone worker cap.

## Public Interface

- `sources.collect_self_cognition_cases(...)`
- `sources.collect_active_commitment_cases(...)`
- `sources.collect_group_chat_review_cases(...)`
- `tracking.build_idempotency_key(...)`
- `tracking.build_trigger_record(case)`
- `tracking.build_run_record(case, trigger_record, selected_route, budget)`
- `tracking.build_route_effect(run_record, route, consumer, effect_summary, next_topic=None)`
- `tracking.classify_route(case, cognition_output, action_attempt=None)`
- `tracking.build_action_attempt(case, trigger_record, existing_attempts)`
- `tracking.build_action_candidate(case, action_attempt, text)`
- `sleep_period.is_self_cognition_sleep_period(...)`
- `runner.build_self_cognition_case_artifacts(case, cognition_client=None, dialog_client=None, consolidation_client=None, apply_consolidation=False)`
- `runner.build_self_cognition_case_artifacts_async(case, cognition_client=None, dialog_client=None, consolidation_client=None, apply_consolidation=False)`
- `worker.run_self_cognition_worker_tick(...)`
- `worker.start_self_cognition_worker(...)`
- `worker.stop_self_cognition_worker(...)`

## Supported Cases

- `commitment_before_due`
- `commitment_past_due`
- `commitment_duplicate_tick`
- `private_no_action`
- `group_noise_rejected`
- `group_chat_review`
- `topic_rag_followup`

`group_chat_review` is built from reflection-cycle group activity windows. It
uses neutral chat-window data framing, bounded visible context, deterministic
semantic labels, and source-aligned delivery metadata.
For ambient group windows, the model-facing source packet says the character
has just noticed a group scene where she has not yet joined and nobody has
handed the topic to her. For directly addressed group windows, the source
packet says someone has pointed the topic at her. Empty windows are skipped
before cognition.
When the reflection input rows carry a usable sanitized group label,
group-review projection folds that label into the existing source-instruction
sentence, for example `下面是我在“动画讨论群”群聊里看到的现场观察资料。`.
The label is not rendered as a separate source-packet field and is not reused
as cognition-scene topic.

Ambient group review keeps `target_scope.user_id=None`: the semantic target is
the observed group scene, not a fabricated latest speaker. The delivery target
remains the source group channel. Before L2d action selection, targetless
group review may load bounded group-channel engagement guidance so the
character can judge whether the current scene gives enough reason to speak.
This guidance is evidence only; it is not a response-ratio control, a command
to speak, or a deterministic silence rule.

Group review also attaches bounded participant context during source
collection under `conversation_progress.participant_context`. This context is
source hydration, not a normal RAG-backed case: it deterministically selects
one primary social beat from the activity-window participant rows, hydrates
only that primary participant when a `global_user_id` is already present, and
caps visible samples plus nearby conversation evidence before the source packet
is rendered. Missing identity degrades to visible-only context; display-name
identity lookup, background participant profiles, web lookup, and full RAG
supervisor planning are not used.

Group review may also attach
`conversation_progress.group_scene_digest = {"digest": str}` with optional
`summary`. This is a single neutral observational string generated from the
selected activity window to help cognition read noisy group flow. The digest
preserves visible participant display names and visible assistant rows with
speaker attribution. It preserves quoted `你` and `我` as quoted row text and
does not resolve ambiguous second-person wording to the active character unless
the same row explicitly supports that address. It does not expose internal ids,
platform ids, message ids, URLs, or `participant_N` aliases. It is optional
source hydration only: it does not add a deterministic flow state, route
decision, response gate, speaker target, or action recommendation.

Group review may also attach
`conversation_progress.thread_reference_context` when a bounded visible row
contains second-person wording that is not directly addressed to the active
character. The object shape is:

```python
{
    "source": "group_review_thread_reference",
    "context_shape": "bounded_second_person_reference_warnings",
    "guidance": "二人称归属按同一行明确地址和可见线程读取；缺少同一行当前角色指向时，保留为侧线/未定对象。",
    "ambiguous_second_person_rows": [
        {
            "speaker": str,
            "sample": str,
            "referent_status": "ambiguous_or_side_thread",
            "basis": str,
        }
    ],
}
```

The context is bounded to three visible samples, contains no ids or delivery
metadata, and is rendered before the recent visible transcript. It is source
evidence for pronoun resolution, not a deterministic silence rule.

When these cases are consolidated, the deterministic target plan gives the
group channel its own target eligibility. Group-review participant presence is
not enough to create a user lane. The source label `self_cognition` is
provenance only and must never be used as `global_user_id`.

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
For `group_chat_review`, a prior `delivery_failed` attempt for the same
activity-window identity also suppresses another visible-speech attempt; this
prevents one group window from repeatedly trying the same selected speech.
Reflection also records every terminal group activity-window review in the
`self_cognition_group_review_windows` ledger by `source_id`. That ledger
suppresses repeated silent/audit-only review, records coalesced backlog
windows, and is separate from visible action-attempt idempotency.

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
The resolver binds to the concrete source channel when that source is a valid
private or group channel. Group sources target the same group channel; private
sources target the same private channel. A known private channel is used only
when it is already attached as the self-cognition source channel for the case.
The resolver does not perform private-channel lookup, private fallback, or
delivery retry. Adapter channel capability is checked immediately before
dispatcher write-ahead persistence.
Cases without a valid concrete source target are recorded as
`target_binding_failed` and stop before cognition, dialog, consolidation, and
delivery.

`SelfCognitionDeliveryTarget` is deterministic runtime metadata. It must stay
out of source packets, cognition state, dialog state, prompts, prompt anchors,
prompt schemas, and model-facing records.

## Delivery Mentions

Self-cognition may attach platform-neutral `delivery_mentions` render
candidates to a local action candidate record when the shared dialog graph
authors exact visible `@display_name` text and the case carries matching
delivery-only identity. Direct user-scoped cases can use `target_scope`;
group-review cases can use bounded `delivery_mention_users` collected from the
same activity window.

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
rendered into source packets, cognition state, dialog state, or prompt text.
They also do not participate in action-attempt idempotency.

Group-review cases may also carry a delivery-only list:

```python
"delivery_mention_users": [
    {
        "global_user_id": str,
        "platform_user_id": str,
        "display_name": str,
    }
]
```

This list is built from internal activity-window participant rows. It is not
rendered into source packets or prompt text; it is consumed only after dialog
authors exact visible tags.

Action candidates may carry:

```python
"delivery_mentions": [
    {
        "entity_kind": "user",
        "platform_user_id": str,
        "display_name": str,
    }
]
```

Self-cognition does not decide native mention syntax. Adapter-owned channel
capability and delivery are checked at the dispatcher boundary; self-cognition
only carries the minimal identity needed for adapters to replace authored
`@display_name` text with native platform mention syntax.

## Future Cognition Handoff

Self-cognition may produce local delivery-candidate records for duplicate
suppression and dispatcher handoff. In production worker ticks, selected
`speak` requires a bound delivery target, L3 text directives, and dialog in the
shared cognition path, then the worker hands the final text to dispatcher
delivery in the same tick.

`trigger_future_cognition` uses the durable calendar scheduler as an internal
delayed trigger source. The action handler records a private calendar
`future_cognition` slot; a later self-cognition worker tick claims due calendar
runs as normal source cases and marks the source slot completed or skipped
after the normal runner path returns. The model-facing source packet receives
only semantic follow-up context, not calendar ids, scheduler ids,
action-attempt ids, continuation schema versions, or depth limits.
Due-aware packets may render neutral due-state facts such as
`约定状态: 已过期` as source state. Actionability strings and route
instructions remain tracking metadata and must not be rendered to the
character.

User-facing delayed reminders use `future_speak`, which is accepted through
the accepted-task lifecycle before the internal worker schedules the
`future_cognition` slot. Self-cognition source cases and future-cognition due
cases may report or act on the scheduled semantic context, but they must not
create a duplicate accepted delayed task for the same active work.

## Event Logging

The production worker mirrors sanitized trigger, run, route, action-attempt,
and consolidation-outcome metadata through
`kazusa_ai_chatbot.event_logging`. This event-log mirror is the durable
operator view for long-term production counts and `/ops/self-cognition/stats`.

Event-log rows store ids, route names, output modes, budget counters,
consolidation write-success booleans, calendar source counts, cache-eviction
counts, origin labels, and status labels; they must not include source packet
text, action candidate text, raw target channels, or conversation bodies.

## Tracking Records

The in-memory runner may produce:

- `self_cognition_trigger_record.json`
- `self_cognition_run_record.json`
- `self_cognition_cognition_input.json`
- `self_cognition_cognition_output.json`
- `self_cognition_route_effect.json`
- `self_cognition_action_attempt.json`
- `self_cognition_action_candidate.json`
- `self_cognition_consolidation_outcome.json`
- `self_cognition_loop_trace.md`

Delivery candidates use `dispatch_shape: "send_message"`. Production selected
speech records a terminal action-attempt status such as `sent`,
`delivery_failed`, `held`, or `duplicate_suppressed`.

## SC-TRACKING-ICD-001

The required local tracking-record shapes are:

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
