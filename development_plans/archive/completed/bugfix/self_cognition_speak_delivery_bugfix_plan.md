# self-cognition speak delivery bugfix plan

## Summary

- Goal: make production self-cognition selected `speak` actions persist and dispatch through the runtime adapter bridge instead of terminating as private candidates.
- Plan class: high_risk_migration.
- Status: completed. Implementation, verification, and independent code review were completed on 2026-05-17.
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`, `py-style`, `test-style-and-execution`.
- Overall cutover strategy: bigbang for production selected-speak behavior; compatible for dry-run candidate inspection.
- Highest-risk areas: delivery target binding before cognition, autonomous `DispatchContext` construction, dispatcher handler return contract, stale architecture docs.
- Acceptance criteria: selected production `speak` cannot end as `candidate` or `dispatch_status=not_requested`; known private channel is preferred; otherwise the self-cognition source channel is used; no LLM prompt/model/schema changes are made.

## Context

The observed failure was a production self-cognition run where L2d selected a user-visible `speak` action, dialog rendered final text, and the worker recorded `dispatch_status=not_requested`. That was not a valid no-send decision. Under the current architecture, L2d decides whether to speak. Once it selects `speak`, deterministic code must bind a target, persist the outbound assistant row, dispatch through the registered runtime adapter, and record success or failure.

The old architecture treated self-cognition speak output as a private action candidate, and an earlier change decommissioned production delivery from this path. On 2026-05-17 the project owner directed that production self-cognition selected `speak` must attempt delivery, reversing that decommission. That decision is the controlling architecture authority for this plan and is recorded under `Architecture Decision Authority`. Dry-run candidate rendering remains available only as an inspection artifact.

This is not an LLM bug. The LLM selected `speak` and dialog generated appropriate message text. The gap is deterministic routing and delivery after the LLM decision.

## Architecture Decision Authority

- Decision date: 2026-05-17.
- Decision owner: project owner (this plan's requester).
- Decision: production self-cognition selected `speak` must resolve a delivery target before cognition and attempt delivery after dialog rendering. This reverses the earlier private-candidate decommission of production delivery from the self-cognition worker path.
- Scope of reversal: production worker runs only. Dry-run candidate rendering is unchanged and remains non-delivering.
- This decision is why the reference docs and the current self-cognition README are reclassified as superseded or rewritten. The supersession is authorized by this decision; it is not inferred by the implementation agent.
- The implementation agent must not re-litigate this decision. If the agent believes production delivery should not exist, it must stop and raise the conflict with the user instead of silently preserving the old private-candidate behavior.

## Mandatory Skills

- `development-plan-writing`: load before editing this plan or changing lifecycle status.
- `local-llm-architecture`: load before changing self-cognition graph, L2d/dialog boundaries, or background LLM pipeline wiring.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before continuing implementation, verification, handoff, status changes, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan before starting the next stage.
- Before completion, lifecycle status changes, merge, or sign-off, run the `Independent Code Review` gate and record the result in `Execution Evidence`.
- Do not change L2d prompts, dialog prompts, LLM schemas, model routing, or LLM call counts.
- Do not require the LLM to emit `target_id`.
- Do not treat missing LLM `target_id` as a suppress condition.
- Do not add deterministic semantic overrides after L2d. L2d still owns speak/no-speak.
- Deterministic code owns target binding, persistence, adapter lookup, adapter send, duplicate suppression, permission/rate-limit holds, delivery receipts, and audit.
- Production selected `speak` must not terminate as `candidate`, `private`, or `not_requested`.
- Self-cognition graph code must not call platform adapters directly.
- Direct `/chat` request/response behavior must not be routed through self-cognition worker code.
- Do not edit archived completed plans. They are historical records only.
- Do not remove unrelated active plan registry rows or untracked plans that predate this work.

## Must Do

- Bind a deterministic delivery target before production self-cognition cognition starts.
- Prefer known private channel on the same platform.
- Fall back to the self-cognition source channel when private channel is unavailable.
- Reject the case before cognition and record `target_binding_failed` when no valid target can be bound.
- Use the deterministic dispatcher send-message path for autonomous delivery.
- Persist an assistant outbound conversation row before adapter send.
- Update self-cognition action attempt status and event logs for sent, held, duplicate-suppressed, and delivery-failed outcomes.
- Mark incorrect legacy docs as superseded or rewrite them to point at the current canonical self-cognition README.
- Keep all LLM behavior unchanged.

## Deferred

No architecture decisions are deferred to the implementation agent.

The following work is explicitly out of scope:

- retry queues
- cached-text resend
- new permission model
- new proactive output transport
- prompt changes
- model changes
- LLM schema changes
- adapter API redesign
- database migration for existing conversation rows

## Cutover Policy

Overall strategy: bigbang for production selected-speak behavior.

| Area | Policy | Instruction |
|---|---|---|
| Production self-cognition selected `speak` | bigbang | Replace private-candidate terminal behavior with persistence plus runtime adapter handoff. No fallback to `dispatch_status=not_requested`. |
| Delivery target binding | bigbang | Require a binding result before cognition for all production source cases that can speak. Bound cases carry `delivery_target`; failed cases carry `target_binding_failure`, skip cognition, and record `target_binding_failed`. |
| Dry-run candidate rendering | compatible | Preserve explicit dry-run candidate artifacts and label them inspection-only. Dry-run remains non-delivering. |
| Dispatcher handler return value | compatible | Extend `handle_send_message(...)` to return delivery metadata while preserving callers that ignore the return value. |
| Existing action-attempt rows | compatible | Leave historical `candidate` rows unchanged. New production selected-speak rows must not terminate as `candidate`. |
| Documentation | bigbang | Current docs must describe speak-means-delivery. Incorrect reference docs must be rewritten or marked superseded. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative compatibility strategy by default.
- Bigbang areas must delete or rewrite old production behavior instead of preserving it behind a fallback.
- Compatible areas preserve only the explicitly listed surfaces.
- Any change to cutover policy requires user approval before implementation.

## Overdesign Guardrail

Actual problem: selected self-cognition speech is generated but not delivered.

Minimal change:

- bind target before cognition
- deliver selected speech through the existing dispatcher/runtime adapter bridge
- update statuses, events, docs, and tests

Ownership boundaries:

- LLM stages own semantic judgment and final wording.
- Self-cognition source collection owns pre-cognition target binding.
- Self-cognition worker owns pre-cognition rejection, duplicate/hold decisions, and selected-speak handoff.
- Dispatcher owns assistant-row persistence, adapter lookup, adapter send, and delivery receipt update.
- Adapters own platform-specific send behavior.
- Docs own supersession and current architecture statements.

Rejected complexity:

- no new retry queue
- no new outbox
- no direct adapter calls from self-cognition graph code
- no new LLM prompt/schema
- no platform-specific target inference
- no cached dialog resend
- no permission model redesign

Evidence required before adding flexibility:

- a deterministic test showing an existing required case cannot be delivered through the dispatcher bridge
- a code pointer proving the existing bridge cannot preserve assistant-row persistence or receipts
- user approval for expanding scope

## Agent Autonomy Boundaries

Implementation agents may choose local variable names and private helper layout only when the public contract in this plan remains unchanged.

Implementation agents must not decide:

- whether selected `speak` should deliver
- whether to fall back to source channel
- whether to use private channel when known
- whether target binding happens before or after cognition
- whether to change LLM prompts or schemas
- whether to treat candidate as a valid terminal production state
- whether to add retries or cached resend
- whether to keep stale docs without a supersession notice

If an implementation agent finds a required contract cannot be implemented, it must stop and report the blocker instead of inventing a new architecture.

## Target State

Production self-cognition has this flow:

```text
source collector
  -> bind SelfCognitionDeliveryTarget or produce target_binding_failed case
  -> target_binding_failed cases are recorded and stop here
  -> cognition/RAG/L2d
  -> dialog render
  -> action attempt pending_handoff
  -> dispatcher send_message bridge
  -> conversation row + adapter send + receipt
  -> action attempt sent/held/duplicate_suppressed/delivery_failed
  -> self-cognition event log
```

No production path leaves selected user-visible `speak` as a private candidate.

## Design Decisions

### Speak Means Delivery Attempt

For production worker runs:

- L2d no-speak means no delivery.
- L2d selected `speak` means deterministic code must attempt delivery.
- Missing LLM action `target_id` is not a delivery suppressor.
- Delivery failure is valid only when deterministic delivery requirements fail and an explicit failure status is recorded.

### Delivery Target Contract

Add this model to `src/kazusa_ai_chatbot/self_cognition/models.py`:

```python
class SelfCognitionDeliveryTarget(TypedDict):
    schema_version: Literal["self_cognition_delivery_target.v1"]
    platform: str
    platform_channel_id: str
    channel_type: Literal["private", "group"]
    target_global_user_id: str | None
    target_platform_user_id: str | None
    source_kind: Literal[
        "target_private_channel",
        "self_cognition_source_channel",
    ]
    source_ref: str
    source_platform_channel_id: str
    source_channel_type: Literal["private", "group"]
    source_message_id: str
    source_global_user_id: str | None
    source_platform_bot_id: str
    source_character_name: str
    guild_id: str | None
    bot_permission_role: str
    fallback_reason: Literal["", "private_channel_unavailable"]
```

Add `delivery_target: NotRequired[SelfCognitionDeliveryTarget]` to `SelfCognitionCase`.

`delivery_target` is deterministic runtime metadata. It is not part of the LLM output contract and must not be generated by the LLM.

### Target Resolution Order

For every production self-cognition case:

1. Query the latest known private channel for the target user on the same platform.
2. If a private channel exists, bind `source_kind="target_private_channel"`.
3. If no private channel exists, bind the self-cognition source channel with `source_kind="self_cognition_source_channel"` and `fallback_reason="private_channel_unavailable"`.
4. If no valid private or source channel exists, return `SelfCognitionTargetBindingFailure`; the worker records `target_binding_failed` and skips cognition.

"Known private channel" means an existing private-channel conversation row or adapter-owned identity mapping explicitly identifies a private channel. The implementation must not infer a private channel from a group `platform_user_id`.

Only `private` and `group` are valid delivery target channel types.

### Source Case Binding Matrix

| Source path | Current functions | Required target behavior |
|---|---|---|
| Scheduled future cognition | `collect_scheduled_future_cognition_cases(...)`, `_build_scheduled_future_cognition_case(...)` | Bind `delivery_target` from scheduled event source fields. If event source channel is invalid but a private target is known, normalize `source_channel_type` and `source_platform_channel_id` to that private target. If no known private channel exists, return a failed-binding case. |
| Active commitment due check | `collect_active_commitment_cases(...)`, `_build_active_commitment_case(...)` | Bind `delivery_target` from latest visible conversation row plus commitment `global_user_id`. Prefer private channel; otherwise latest row source channel. |
| Direct `collect_self_cognition_cases(...)` production aggregation | `collect_self_cognition_cases(...)` | Return bound cases and failed-binding cases; source-specific collectors own binding and failure payload creation. |
| Test seam `collect_cases_func` in `run_self_cognition_worker_tick(...)` | `worker.run_self_cognition_worker_tick(...)` | Treat injected production cases the same as normal cases. Missing `delivery_target` records `target_binding_failed` and skips cognition unless the test explicitly runs the dry-run runner outside the worker. |
| Dry-run case files and direct runner calls | `runner.build_self_cognition_case_artifacts*`, `runner.run_self_cognition_case*`, `scripts.run_self_cognition_dry_run` | Remain inspection-only. They may omit `delivery_target` and must not dispatch. |

### Target Binding Failure Contract

Resolver failure must be auditable and must not be silently dropped by collectors.

Add this model to `src/kazusa_ai_chatbot/self_cognition/models.py`:

```python
class SelfCognitionTargetBindingFailure(TypedDict):
    status: Literal["target_binding_failed"]
    reason: Literal[
        "missing_platform",
        "missing_target_user",
        "private_channel_unavailable_and_source_invalid",
        "private_channel_unavailable_and_source_missing",
    ]
    platform: str
    source_ref: str
    source_platform_channel_id: str
    source_channel_type: str
    target_global_user_id: str | None
    target_platform_user_id: str | None
```

Add these fields to `SelfCognitionCase`:

```python
target_binding_status: NotRequired[Literal["bound", "failed"]]
target_binding_failure: NotRequired[SelfCognitionTargetBindingFailure]
```

Failure owner:

- `resolve_self_cognition_delivery_target(...)` returns either `SelfCognitionDeliveryTarget` or `SelfCognitionTargetBindingFailure`.
- Source builders attach `target_binding_status="bound"` plus `delivery_target`, or `target_binding_status="failed"` plus `target_binding_failure`.
- `collect_self_cognition_cases(...)` returns bound and failed cases. It must not filter failed cases.
- `worker.run_self_cognition_worker_tick(...)` owns recording failed cases before cognition.

Worker failure behavior:

- If `target_binding_status == "failed"` or a production worker case has no `delivery_target`, the worker must not call RAG, cognition, dialog, consolidation, or delivery.
- It records a self-cognition event with:
  - `status="target_binding_failed"`
  - `selected_route="not_started"`
  - `output_mode="none"`
  - `dispatch_status="target_binding_failed"`
  - zero budget counters
  - `case_id`
  - `trigger_kind`
  - `target_binding_failure.reason`
  - `target_binding_failure.platform`
  - `target_binding_failure.source_ref`
  - `target_binding_failure.source_platform_channel_id`
  - `target_binding_failure.source_channel_type`
  - booleans for presence of target global and platform user ids
- It increments `skipped_count`.
- It records no action attempt.
- For scheduled future cognition cases, the worker must claim the scheduled event first, record `target_binding_failed`, and then mark the scheduled event completed. Invalid scheduled rows must not loop forever.
- For active commitment cases, the worker records the failure and does not mutate the memory unit.

### Private Channel Lookup Contract

Add this helper to `src/kazusa_ai_chatbot/db/conversation.py`:

```python
async def get_latest_private_channel_for_user(
    *,
    platform: str,
    global_user_id: str | None,
    platform_user_id: str | None,
) -> dict[str, Any] | None:
```

Query `conversation_history` for:

- same `platform`
- `channel_type == "private"`
- non-empty `platform_channel_id`
- `role == "user"`
- matching `global_user_id` or `platform_user_id`
- newest timestamp first

If both user ids are empty, return `None` without querying.

### Delivery Target Resolver Contract

Add this resolver to `src/kazusa_ai_chatbot/self_cognition/sources.py`:

```python
async def resolve_self_cognition_delivery_target(
    *,
    platform: str,
    source_platform_channel_id: str | None,
    source_channel_type: str | None,
    source_message_id: str | None,
    source_ref: str,
    source_global_user_id: str | None,
    source_platform_bot_id: str | None,
    source_character_name: str | None,
    guild_id: str | None,
    bot_permission_role: str | None,
    target_global_user_id: str | None,
    target_platform_user_id: str | None,
    get_latest_private_channel_func: Callable[..., Any] | None = None,
) -> models.SelfCognitionDeliveryTarget | models.SelfCognitionTargetBindingFailure:
```

Resolver output rules:

- `platform` is the source platform and target platform.
- `source_platform_channel_id` is always the original self-cognition source channel when known.
- `platform_channel_id` is the actual send destination.
- `source_message_id` defaults to `self_cognition:{source_ref}` when absent.
- `source_global_user_id` defaults to `target_global_user_id`.
- `bot_permission_role` defaults to `"user"`.
- `source_platform_bot_id` defaults to `""` when unavailable. The dispatcher then uses adapter bot identity fallback.
- `source_character_name` defaults to `"active character"` only if the character profile has no name.
- If source channel type is `private` or `group`, preserve it.
- If source channel type is invalid or empty and a private target is found, set `source_channel_type="private"` and `source_platform_channel_id` to the private target channel id.
- If source channel type is invalid or empty and no private target is found, return a target-binding failure.
- If `platform` is empty, return failure reason `missing_platform`.
- If both target user ids are empty, return failure reason `missing_target_user`.
- If private channel is unavailable and source channel id is empty, return failure reason `private_channel_unavailable_and_source_missing`.
- If private channel is unavailable and source channel type is not `private` or `group`, return failure reason `private_channel_unavailable_and_source_invalid`.

### DispatchContext Construction Contract

Add `src/kazusa_ai_chatbot/self_cognition/delivery.py` and implement `deliver_selected_speak(...)`.

Public signature:

```python
async def deliver_selected_speak(
    *,
    text: str,
    delivery_target: models.SelfCognitionDeliveryTarget,
    character_profile: Mapping[str, Any],
    adapter_registry: AdapterRegistry | None,
    now: datetime,
    reply_to_msg_id: str | None = None,
    delivery_mentions: list[dict[str, Any]] | None = None,
) -> SelfCognitionDeliveryResult:
```

Required status mapping:

- Empty `text` returns `status="delivery_failed"` with `failure_reason="empty_text"`.
- Missing `adapter_registry` returns `status="delivery_failed"` with `failure_reason="adapter_registry_unavailable"`.
- Successful `handle_send_message(...)` return must be a mapping containing non-empty `conversation_message_id` and `delivery_tracking_id`; it maps to `status="sent"`.
- `handle_send_message(...)` returning `None`, a non-mapping object, or a mapping missing `conversation_message_id` or `delivery_tracking_id` maps to `status="delivery_failed"` with `failure_reason="send_message_missing_delivery_metadata"`.
- `UnknownPlatformError` maps to `status="delivery_failed"` with `failure_reason="adapter_unavailable"`.
- `ConversationHistoryWriteError` maps to `status="delivery_failed"` with `failure_reason="conversation_history_write_failed"`.
- Any other exception from `handle_send_message(...)` maps to `status="delivery_failed"` with `failure_reason="adapter_send_failed:<ExceptionClassName>"`.
- Duplicate suppression and held outcomes come from the existing `tracking.py` action-attempt status derivation, not from new logic in `deliver_selected_speak(...)`. When that derivation yields `duplicate_suppressed` or `held`, the worker builds `SelfCognitionDeliveryResult` directly with that status and does not call the adapter.

`deliver_selected_speak(...)` must construct `DispatchContext` exactly as follows:

```python
ctx = DispatchContext(
    source_platform=delivery_target["platform"],
    source_channel_id=(
        delivery_target["source_platform_channel_id"]
        or delivery_target["platform_channel_id"]
    ),
    source_user_id=(
        delivery_target["target_global_user_id"]
        or delivery_target["source_global_user_id"]
        or ""
    ),
    source_message_id=delivery_target["source_message_id"],
    guild_id=delivery_target["guild_id"],
    bot_permission_role=delivery_target["bot_permission_role"] or "user",
    now=now,
    source_channel_type=(
        delivery_target["source_channel_type"]
        or delivery_target["channel_type"]
    ),
    source_platform_bot_id=delivery_target["source_platform_bot_id"],
    source_character_name=(
        delivery_target["source_character_name"]
        or str(character_profile.get("name") or "active character")
    ),
)
```

It must construct `send_message` args exactly as follows:

```python
args = {
    "target_platform": delivery_target["platform"],
    "target_channel": delivery_target["platform_channel_id"],
    "target_channel_type": delivery_target["channel_type"],
    "text": text,
    "execute_at": now.isoformat(),
    "reply_to_msg_id": reply_to_msg_id,
    "delivery_mentions": delivery_mentions or [],
}
```

Then it must call `handle_send_message(args, ctx, adapter_registry)`.

Private-target sends and group-fallback sends both set `ctx.source_user_id` to the semantic target user when available. The adapter destination still comes from `args["target_channel"]`.

### Delivery Result Contract

Add this return shape in `self_cognition/delivery.py`:

```python
class SelfCognitionDeliveryResult(TypedDict):
    status: Literal[
        "sent",
        "delivery_failed",
        "held",
        "duplicate_suppressed",
    ]
    conversation_message_id: str | None
    delivery_tracking_id: str | None
    adapter_message_id: str | None
    failure_reason: str | None
```

Extend `handle_send_message(...)` in `src/kazusa_ai_chatbot/dispatcher/handlers.py` to return delivery metadata:

```python
class SendMessageDispatchResult(TypedDict):
    conversation_message_id: str
    delivery_tracking_id: str
    adapter_message_id: str
```

Update `src/kazusa_ai_chatbot/dispatcher/tool_spec.py`:

```python
TaskHandler = Callable[
    [dict, "DispatchContext", "AdapterRegistry"],
    Awaitable[object | None],
]
```

Existing scheduler/tool callers remain compatible because they already await the handler and ignore the return value.

### Attempt Status Contract

Add `ACTION_ATTEMPT_STATUS_DELIVERY_FAILED = "delivery_failed"` to `models.py`.

Production selected-speak statuses:

- `pending_handoff`: in-memory transient marker only. Selected `speak` has final dialog and delivery is starting in the same worker tick. It is never persisted as a standalone action-attempt row. See `Worker Delivery Wiring Contract`.
- `sent`: adapter delivery succeeded. Delivery-receipt update is best-effort and does not affect this status.
- `delivery_failed`: a missing assistant-row persistence, missing adapter registry, missing adapter, or adapter send failure prevented delivery. A receipt-update failure after a successful adapter send does not produce `delivery_failed`; the dispatcher logs it as a warning.
- `duplicate_suppressed`: existing duplicate suppression blocked a repeat send.
- `held`: an existing deterministic suppression or hold rule in `tracking.py` blocked delivery before handoff. This plan adds no new permission model and no new rate limiter.

`candidate` remains valid only for dry-run inspection and pre-handoff internal artifacts. It is not a terminal production status for selected `speak`.

`delivery_failed` is not a duplicate-suppressing status. A future worker tick may re-evaluate the source case through L2d. This plan does not resend cached text.

### Worker Delivery Wiring Contract

Delivery is invoked from `worker.py`. `runner.py` renders dialog and returns artifacts only; it must not call adapters or `deliver_selected_speak(...)`. The runner may carry `delivery_target` through case and artifact data unchanged for the worker to read, but must not project it into any LLM-facing payload.

`run_self_cognition_worker_tick(...)` and `start_self_cognition_worker(...)` each add one keyword-only parameter:

```python
adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None
```

`start_self_cognition_worker(...)` threads the provider through `_self_cognition_worker_loop(...)` into the tick. `service.py` passes a provider that returns the process-local `AdapterRegistry`. A provider callable is required, not a direct registry value, because adapters register into the registry after worker startup.

Per tick the worker resolves the registry once by calling the provider. A `None` provider, or a provider returning `None`, is not a startup error: selected-speak delivery for that tick records `delivery_failed` with `failure_reason="adapter_registry_unavailable"` through `deliver_selected_speak(...)`.

The worker performs delivery synchronously inside the same tick that renders dialog. `pending_handoff` is an in-memory transient state only; the worker must not persist a standalone `pending_handoff` action-attempt row before calling `deliver_selected_speak(...)`. Only the terminal status (`sent`, `delivery_failed`, `duplicate_suppressed`, or `held`) is persisted. This prevents a crash between handoff and delivery from leaving a permanently suppressing `pending_handoff` row.

## LLM Call And Context Budget

Before this plan:

- production self-cognition can use existing RAG, cognition, dialog, and consolidation calls
- selected `speak` can render dialog but stop before delivery

After this plan:

- same RAG call budget
- same cognition call budget
- same dialog call budget
- same consolidation call behavior
- no new LLM stages
- no prompt or schema changes

Delivery target metadata is deterministic runtime data. It must never be projected into source packets, RAG requests, cognition state, dialog state, prompts, prompt anchors, prompt schemas, or LLM-facing artifacts. Existing prompt-safe source projection fields remain unchanged and must not be extended with delivery-only ids.

## Change Surface

### Docs To Update

- `src/kazusa_ai_chatbot/self_cognition/README.md`: rewrite as canonical speak-means-delivery architecture.
- `development_plans/reference/designs/cognition_contracts_design.md`: add selected self-cognition speech to the shared cognition/dialog/persistence/adapter contract.
- `src/kazusa_ai_chatbot/proactive_output/README.md`: add scope note that this test transport does not govern selected self-cognition speech.
- `development_plans/archive/superseded/self_cognition_tracking_icd.md`: add superseded banner.
- `development_plans/archive/superseded/self_cognition_reasoning_basis.md`: add superseded banner for private-candidate/no-production-delivery claims.
- `development_plans/archive/superseded/self_cognition_loop_architecture.md`: add superseded banner for no-send/private-candidate production statements.
- `development_plans/reference/designs/cognition_core_evolution_progression.md`: update the specific self-cognition decommission statements that say production delivery was removed from this path.

Do not edit archived completed plans. Do not edit `development_plans/reference/designs/action_spec_effector_expansion_architecture.md`; its delayed-contact language remains valid because it concerns prewritten delayed sends, not immediate selected self-cognition speech.

Legacy supersession banner must appear within the first 20 lines of each superseded reference document and must use this exact field set:

```markdown
> Superseded Architecture Document
>
> Status: superseded
> Superseded by plan: development_plans/active/bugfix/self_cognition_speak_delivery_bugfix_plan.md
> Canonical current doc: src/kazusa_ai_chatbot/self_cognition/README.md
> Supersession rule: private-candidate-only and no-production-delivery claims
> in this document are no longer architecture authority. Current production
> self-cognition selected `speak` must resolve a target before cognition and
> attempt delivery after dialog rendering.
```

When adding the banner, also replace stale related-plan pointers that imply the old private-candidate boundary is current. Keep historical rationale paragraphs only when the banner makes their superseded status explicit.

### Code To Update

- `src/kazusa_ai_chatbot/self_cognition/models.py`
- `src/kazusa_ai_chatbot/self_cognition/sources.py`
- `src/kazusa_ai_chatbot/self_cognition/worker.py`
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
- `src/kazusa_ai_chatbot/self_cognition/tracking.py`
- `src/kazusa_ai_chatbot/self_cognition/delivery.py`
- `src/kazusa_ai_chatbot/db/conversation.py`
- `src/kazusa_ai_chatbot/dispatcher/handlers.py`
- `src/kazusa_ai_chatbot/dispatcher/tool_spec.py`
- `src/kazusa_ai_chatbot/dispatcher/task.py` only if imports or docstrings need to expose the new autonomous context use
- `src/kazusa_ai_chatbot/service.py`

### Tests To Add Or Update

- `tests/test_self_cognition_delivery_target.py`
- `tests/test_self_cognition_integration.py`
- `tests/test_self_cognition_event_logging.py`
- `tests/test_self_cognition_tracking.py`
- `tests/test_dispatcher_send_message_result.py`
- `tests/test_delivery_mentions.py`
- `tests/test_runtime_adapter_registration.py`
- `tests/test_service_background_consolidation.py`
- `tests/test_self_cognition_architecture_docs.py`

## Implementation Order

1. Add documentation tests first.
   - File: `tests/test_self_cognition_architecture_docs.py`.
   - Add:
     - `test_legacy_private_candidate_docs_have_superseded_banner`
     - `test_canonical_self_cognition_readme_defines_delivery_target_before_cognition`
     - `test_canonical_docs_do_not_authorize_production_not_requested_for_speak`
   - Expected before implementation: fails on stale docs.

2. Add target binding unit tests.
   - File: `tests/test_self_cognition_delivery_target.py`.
   - Add:
     - `test_resolver_prefers_known_private_channel`
     - `test_resolver_falls_back_to_source_channel_when_private_missing`
     - `test_resolver_rejects_missing_private_and_source`
     - `test_resolver_does_not_infer_private_from_group_platform_user_id`
     - `test_resolver_rejects_invalid_source_channel_type`
     - `test_production_collectors_attach_delivery_target_before_cognition`
     - `test_collectors_return_failed_case_when_target_binding_fails`
     - `test_delivery_target_never_enters_llm_facing_payloads`
   - Expected before implementation: fails on missing resolver/model fields.

3. Add dispatcher return tests.
   - File: `tests/test_dispatcher_send_message_result.py`.
   - Add:
     - `test_handle_send_message_returns_delivery_metadata`
     - `test_task_handler_type_accepts_ignored_return_value`
   - Expected before implementation: fails because `handle_send_message(...)` returns `None` and `TaskHandler` is `Awaitable[None]`.

4. Add production speak-delivery integration tests.
   - File: `tests/test_self_cognition_integration.py`.
   - Add:
     - `test_worker_selected_speak_dispatches_to_private_channel`
     - `test_worker_selected_speak_falls_back_to_source_channel`
     - `test_worker_missing_delivery_target_blocks_before_dialog`
     - `test_worker_records_target_binding_failed_and_completes_scheduled_event`
     - `test_worker_selected_speak_never_records_not_requested`
     - `test_worker_no_speak_does_not_dispatch`
     - `test_worker_adapter_failure_marks_delivery_failed`
     - `test_worker_duplicate_suppression_marks_duplicate_suppressed`
     - `test_worker_persists_only_terminal_status_not_pending_handoff`
     - `test_worker_missing_adapter_registry_marks_delivery_failed`
   - Expected before implementation: selected-speak tests fail because current worker returns `not_requested`.

5. Add or update event/tracking tests.
   - Files: `tests/test_self_cognition_event_logging.py`, `tests/test_self_cognition_tracking.py`.
   - Add assertions for `target_binding_failed`, `pending_handoff`, `sent`, `delivery_failed`, and no terminal production `candidate`.

6. Add regression tests for related delivery surfaces.
   - `tests/test_delivery_mentions.py::test_self_cognition_delivery_preserves_mentions`
   - `tests/test_runtime_adapter_registration.py::test_self_cognition_uses_registered_runtime_adapter`
   - `tests/test_service_background_consolidation.py::test_self_cognition_worker_receives_adapter_registry_provider`

7. Update docs.
   - Apply the exact doc updates listed in `Change Surface`.
   - Run documentation tests and grep gates.

8. Implement target models and DB helper.
   - Update `models.py`.
   - Add `get_latest_private_channel_for_user(...)`.
   - Run target binding tests.

9. Implement source resolver and bind cases.
   - Update `sources.py`.
   - Bind scheduled future cognition and active commitment cases.
   - Validate injected worker cases before cognition.
   - Run target binding tests again.

10. Implement dispatcher result compatibility.
    - Update `dispatcher/handlers.py`.
    - Update `dispatcher/tool_spec.py`.
    - Run dispatcher result tests.

11. Implement self-cognition delivery module.
    - Add `self_cognition/delivery.py`.
    - Use exact `DispatchContext` and args contracts from this plan.
    - Run delivery module and dispatcher tests.

12. Wire production worker delivery.
    - Update `worker.py`, `runner.py`, `tracking.py`, and `service.py`.
    - Add the `adapter_registry_provider` parameter to `start_self_cognition_worker(...)` and `run_self_cognition_worker_tick(...)` and pass a provider from service startup, per `Worker Delivery Wiring Contract`.
    - Invoke `deliver_selected_speak(...)` from `worker.py`; persist only terminal attempt statuses.
    - Update attempt statuses and event logging.
    - Run self-cognition integration and event tests.

13. Run full required verification.

14. Run independent code review gate.

15. Fix critical and important review findings inside this plan's change surface and rerun affected verification.

## Progress Checklist

- [x] Stage 1 - documentation and target-binding tests added
  - Covers: implementation steps 1-2.
  - Verify: targeted tests fail for the expected stale-doc and missing-symbol reasons before implementation.
  - Evidence: record failing test names and failure reasons.
  - Sign-off: `Codex/2026-05-17`.

- [x] Stage 2 - dispatcher and production delivery tests added
  - Covers: implementation steps 3-6.
  - Verify: targeted tests fail for current `None` return and `not_requested` behavior.
  - Evidence: record failing test names and failure reasons.
  - Sign-off: `Codex/2026-05-17`.

- [x] Stage 3 - docs updated and supersession complete
  - Covers: implementation step 7.
  - Verify: docs tests pass and grep gates have only allowed matches.
  - Evidence: record changed docs and grep output summary.
  - Sign-off: `Codex/2026-05-17`.

- [x] Stage 4 - target model, DB helper, and source resolver complete
  - Covers: implementation steps 8-9.
  - Verify: `tests/test_self_cognition_delivery_target.py` passes.
  - Evidence: record pytest output.
  - Sign-off: `Codex/2026-05-17`.

- [x] Stage 5 - dispatcher return contract complete
  - Covers: implementation step 10.
  - Verify: `tests/test_dispatcher_send_message_result.py` passes.
  - Evidence: record pytest output.
  - Sign-off: `Codex/2026-05-17`.

- [x] Stage 6 - self-cognition delivery wiring complete
  - Covers: implementation steps 11-12.
  - Verify: self-cognition integration, event, and tracking tests pass.
  - Evidence: record pytest output.
  - Sign-off: `Codex/2026-05-17`.

- [x] Stage 7 - full verification complete
  - Covers: implementation step 13.
  - Verify: all commands in `Verification` pass or are recorded with explicit blockers.
  - Evidence: record command outputs.
  - Sign-off: `Codex/2026-05-17`.

- [x] Stage 8 - independent code review complete
  - Covers: implementation steps 14-15.
  - Verify: independent reviewer reports no unresolved critical or important findings.
  - Evidence: record reviewer id/name, findings, fixes, and rerun commands.
  - Sign-off: `Codex/2026-05-17`.

## Verification

Use `venv\Scripts\python.exe` for all Python commands.

### Required Focused Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_self_cognition_architecture_docs.py -q
venv\Scripts\python.exe -m pytest tests/test_self_cognition_delivery_target.py -q
venv\Scripts\python.exe -m pytest tests/test_dispatcher_send_message_result.py -q
venv\Scripts\python.exe -m pytest tests/test_self_cognition_integration.py -q
venv\Scripts\python.exe -m pytest tests/test_self_cognition_event_logging.py tests/test_self_cognition_tracking.py -q
```

### Required Regression Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_delivery_mentions.py tests/test_runtime_adapter_registration.py -q
venv\Scripts\python.exe -m pytest tests/test_service_background_consolidation.py -q
```

### Required Static Greps

Run:

```powershell
rg -n "production_handoff.*False|dispatch_status=.*not_requested|private tracking artifacts|private local candidate only; no production delivery|production delivery: none|removes production prewritten text delivery|production prewritten text delivery" src\kazusa_ai_chatbot\self_cognition development_plans\reference\designs src\kazusa_ai_chatbot\proactive_output
```

Allowed matches after implementation:

- dry-run-only docs that explicitly say dry-run candidate rendering is non-delivering
- legacy reference docs with the required superseded banner in the first 20 lines
- tests asserting old behavior is absent or superseded

No production code path may return `dispatch_status="not_requested"` after selected `speak`.

Run:

```powershell
rg -n "delivery_target" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\prompts src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\self_cognition\runner.py
```

Allowed matches after implementation:

- tests are not in this grep scope
- `runner.py` may carry `delivery_target` through case and artifact data only; it must not call adapters or `deliver_selected_speak(...)`
- no match may add `delivery_target` to source packets, RAG requests, cognition state, dialog state, prompt anchors, prompt schemas, or prompt text

### Required No-LLM-Change Grep

Run:

```powershell
git diff -- src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\prompts src\kazusa_ai_chatbot\self_cognition development_plans\active\bugfix\self_cognition_speak_delivery_bugfix_plan.md
```

Inspect the diff and record that no L2d prompt, dialog prompt, model routing, or LLM schema changed. Code changes in `self_cognition` are allowed only for deterministic routing, target binding, and delivery.

## Independent Plan Review

An independent subagent reviewed the first draft on 2026-05-17 and reported the draft was not ready because it lacked final-plan structure, left `DispatchContext` construction unresolved, did not enumerate every stale legacy doc, omitted `dispatcher/tool_spec.py`, and used broad test instructions.

A second independent subagent reviewed the revised draft on 2026-05-17 and reported remaining blockers: target-binding failures needed an owner and event contract, delivery target metadata exclusion from LLM-facing payloads was too weak, `source_platform_bot_id` defaulting was unspecified, `deliver_selected_speak(...)` needed an exact signature and status mapping, supersession banner fields were underspecified, invalid source channel type with private target was undecided, and the overdesign guardrail needed ownership boundaries before autonomy boundaries.

A third independent subagent reviewed the next revision on 2026-05-17 and found one remaining blocker: `deliver_selected_speak(...)` exposed an injected handler while another line required direct `handle_send_message(...)` use, and bad handler return values were not specified.

This revision resolves those findings by:

- adding required plan-contract sections
- defining exact `DispatchContext` construction
- enumerating legacy docs and grep gates
- adding `dispatcher/tool_spec.py` to the change surface
- naming exact test files and test function names
- defining target-binding failure models, owner, worker behavior, event payload, and scheduled-event completion
- forbidding `delivery_target` in all LLM-facing payloads and adding a grep gate
- specifying `source_platform_bot_id` defaulting
- defining `deliver_selected_speak(...)` signature and status mapping
- defining the exact supersession banner field set
- deciding invalid source channel behavior when private target wins
- moving `Overdesign Guardrail` before `Agent Autonomy Boundaries` and adding ownership boundaries
- removing the injected send handler from `deliver_selected_speak(...)`
- defining `None`, non-mapping, and incomplete handler returns as `send_message_missing_delivery_metadata`

A code-feasibility review on 2026-05-17 verified the referenced symbols against the codebase and found the delivery mechanism implementable, but flagged five issues: the architecture reversal lacked a recorded authority; `run_self_cognition_worker_tick(...)` would gain an unspecified public signature; `held` implied a non-existent new rate limiter; a persisted `pending_handoff` row could permanently suppress a case after a crash; and the `delivery_failed` wording implied delivery receipts were critical. This revision resolves those by adding `Architecture Decision Authority`, adding `Worker Delivery Wiring Contract` with the exact `adapter_registry_provider` signature, redefining `held` as existing tracking-derived state with no new rate limiter, making `pending_handoff` a non-persisted in-memory transient, and correcting the receipt wording.

The project owner approved this plan for execution on 2026-05-17 after a development-plan-guideline alignment check passed. The plan and its registry row are `approved`.

## Independent Code Review

After implementation and verification, request an independent code review by a separate subagent or human reviewer.

Review scope:

- full working tree diff for this plan
- plan alignment
- style compliance
- deterministic target binding
- dispatcher return compatibility
- selected-speak delivery behavior
- docs supersession accuracy
- test coverage and verification evidence

The plan cannot be marked completed while any critical or important review finding remains unresolved. Minor findings may be recorded as follow-up only when they do not affect behavior, tests, architecture, or documentation correctness.

## Acceptance Criteria

- Current docs state that production selected self-cognition `speak` must attempt delivery.
- Incorrect legacy reference docs are marked superseded or rewritten.
- Every production source case has either `delivery_target` or `target_binding_failure` before cognition.
- Missing target blocks the case before cognition and records `target_binding_failed`.
- Known private channel is preferred.
- Source channel is used when private channel is unavailable.
- Selected production `speak` persists an assistant outbound row.
- Selected production `speak` calls the registered runtime adapter bridge.
- Successful adapter send records `sent`.
- Adapter/persistence/registry failure records `delivery_failed`.
- Duplicate suppression records `duplicate_suppressed`.
- An existing tracking-derived suppression or hold records `held`.
- Selected production `speak` cannot end with terminal `candidate`.
- Selected production `speak` cannot end with `dispatch_status=not_requested`.
- Direct `/chat` behavior remains unchanged.
- Dry-run candidate rendering remains inspection-only and non-delivering.
- No L2d prompt, dialog prompt, model routing, LLM schema, or LLM call budget changes appear in the diff.
- Required deterministic tests pass.
- Independent code review has no unresolved critical or important findings.

## Risks

- Autonomous delivery uses dispatcher context that historically represented scheduled tool dispatch. The exact context construction in this plan prevents implementation-time interpretation.
- Returning metadata from `handle_send_message(...)` changes a typed public contract. The `TaskHandler` type update and compatibility tests prevent stale typing.
- Stale reference docs can mislead future agents. The docs tests and grep gates prevent private-candidate production claims from remaining authoritative.
- Adapter availability can vary at runtime. Missing adapter registry is recorded as `delivery_failed`; this plan does not add retry behavior.
- This plan reverses a prior decommission of production self-cognition delivery. `Architecture Decision Authority` records the owning decision; without that section the supersession of reference docs would be unauthorized. If the recorded decision is ever withdrawn, this plan must not proceed.
- A crash between handoff and delivery could strand a case. `Worker Delivery Wiring Contract` keeps `pending_handoff` an in-memory transient and persists only terminal statuses, so a stranded case is re-evaluated on the next tick instead of being permanently suppressed.

## Execution Evidence

Execution date: 2026-05-17.

Work distribution:

- Parent session owned all test development and validation.
- Production-code worker `019e3563-c9fd-7b71-9570-0f1ac1161c96`
  owned production code and documentation changes.
- Independent reviewers:
  - `019e3573-f03f-75a2-bdbb-9a16baa7226c` reported one critical and
    three important findings.
  - `019e3581-2ae9-7a72-ab09-0b3d2c038594` verified the critical finding
    was fixed and reported two remaining important findings.
  - `019e3588-7ca0-7441-a036-9e6b716bd7ba` verified the final two findings
    were closed and reported no critical, important, or minor issues.

RED evidence:

- `tests/test_self_cognition_integration.py::test_worker_empty_dialog_text_marks_delivery_failed`
  initially failed because selected speak with empty rendered text persisted
  `candidate` instead of terminal `delivery_failed`.
- `tests/test_self_cognition_event_logging.py::test_worker_records_target_binding_failure_without_dispatch_text`
  and
  `tests/test_self_cognition_event_logging.py::test_self_cognition_event_logger_records_target_binding_failure`
  initially failed because target-binding failure metadata was not passed or
  accepted by event logging.
- `tests/test_self_cognition_integration.py::test_worker_missing_delivery_target_blocks_without_adapter_provider`
  and
  `tests/test_self_cognition_integration.py::test_worker_tick_blocks_unbound_case_before_candidate_render`
  initially failed because worker ticks could process unbound cases when no
  adapter registry provider was supplied.
- `tests/test_self_cognition_architecture_docs.py::test_cognition_contracts_doc_names_selected_self_cognition_speak_delivery`
  and
  `tests/test_self_cognition_architecture_docs.py::test_proactive_output_doc_does_not_govern_self_cognition_speak`
  initially failed because the referenced docs did not name the selected
  self-cognition speak delivery boundary.
- `tests/test_self_cognition_event_logging.py::test_worker_synthesizes_missing_target_binding_failure_metadata`
  initially failed because worker-owned missing-target failures logged empty
  metadata.
- `tests/test_self_cognition_tracking.py::test_past_due_contact_decision_writes_action_attempt_and_candidate_without_handoff`
  initially failed because the dry-run route-effect text still said production
  records without delivery.

GREEN verification:

- `venv\Scripts\python.exe -m pytest tests/test_self_cognition_architecture_docs.py tests/test_self_cognition_delivery_target.py tests/test_dispatcher_send_message_result.py tests/test_self_cognition_integration.py tests/test_self_cognition_event_logging.py tests/test_self_cognition_tracking.py -q`
  passed: 87 passed.
- `venv\Scripts\python.exe -m pytest tests/test_delivery_mentions.py tests/test_runtime_adapter_registration.py tests/test_service_background_consolidation.py -q`
  passed: 70 passed.
- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\self_cognition\models.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\worker.py src\kazusa_ai_chatbot\self_cognition\delivery.py src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\event_logging\recording.py src\kazusa_ai_chatbot\dispatcher\handlers.py src\kazusa_ai_chatbot\dispatcher\tool_spec.py src\kazusa_ai_chatbot\db\conversation.py src\kazusa_ai_chatbot\service.py`
  passed.
- Stale no-delivery grep over `src\kazusa_ai_chatbot\self_cognition`,
  `development_plans\reference\designs`, and
  `src\kazusa_ai_chatbot\proactive_output` returned no matches for the blocked
  production no-delivery phrases, including `without delivery`.
- `delivery_target` LLM-facing grep over `src\kazusa_ai_chatbot\nodes`,
  `src\kazusa_ai_chatbot\self_cognition\projection.py`, and
  `src\kazusa_ai_chatbot\self_cognition\runner.py` returned no matches.
- Scoped diff inspection found no `nodes` prompt/model/schema changes; changes
  under `self_cognition` are deterministic routing, target binding, delivery,
  dry-run wording, and docs.
