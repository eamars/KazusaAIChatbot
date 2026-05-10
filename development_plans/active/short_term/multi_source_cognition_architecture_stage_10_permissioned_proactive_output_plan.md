# multi source cognition architecture stage 10 permissioned proactive output plan

## Summary

- Goal: Add a permissioned proactive-output contract that turns an already
  approved cognition preview into deterministic permission decisions, dry-run
  outbox records, and fake-adapter transport audit records.
- Plan class: high_risk_migration
- Status: completed
- Runtime behavior change: none. Stage 10 does not register a background
  worker, create a MongoDB collection, call the scheduler, call the dispatcher,
  or send through a real adapter.
- Database schema change: none.
- Prompt change: none.
- New LLM call: none.
- `/chat` latency impact: none. Stage 10 code is not imported by the live
  `/chat` response path.
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `database-data-pull`, `py-style`, and
  `test-style-and-execution`; also use `cjk-safety` before editing any Python
  file containing CJK prompt strings.
- Acceptance criteria: permission, preview, outbox, and audit contracts exist;
  deterministic denials cover every listed risk; dry-run records perform no
  adapter sends; fake-adapter transport writes sent audit metadata only for
  `ready` outbox records; Stage 00 and prior-stage gates pass.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: Stage 10 implementation is completed on branch
`stage-10-permissioned-proactive-output` with deterministic verification and
independent code review recorded below.

## Context

Earlier stages made multiple trigger and input sources enter cognition without
changing current `/chat` behavior. Stage 10 is the first proactive-output
boundary, so it deliberately stops before live contact. A cognition result can
only become a candidate preview. Deterministic policy decides whether that
preview can become an outbox item and whether a test-only fake adapter may mark
that outbox item as sent.

The top-level architecture requires explicit permission, dispatcher or
scheduler validation, adapter availability, and auditability before autonomous
contact. Stage 10 satisfies the permission, adapter-availability, idempotency,
and audit contracts. Real scheduler or dispatcher cutover remains forbidden in
this plan.

## Stage Handoff

### From Stage 09

Stage 10 carries forward these completed artifacts:

- branch: `stage-09-multimodal-cognitive-input-sources`;
- implementation commit: `566e6eb`;
- evidence and lifecycle commit on `main`: `9de93fe`;
- focused Stage 09 tests: `15 passed`;
- Stage 00 regression gate: `11 passed`;
- Stage 03 prompt-selection gate: `36 passed`;
- Stage 06 policy gate: `9 passed`;
- Stage 07 reflection dry-run gate: `14 passed`;
- Stage 08 internal-thought dry-run gate: `26 passed`;
- parent ledger row for `stage_09`: `completed`;
- registry row for Stage 09: `completed | completed`.

Stage 10 relies on Stage 09 only for source-expanded cognition contracts. It
does not read raw media, alter media descriptors, or modify cognition prompts.

### Runtime Inspection

The approved Change Surface below is based on code inspection and read-only
database export on 2026-05-10:

- Scheduler runtime: `src/kazusa_ai_chatbot/scheduler.py` persists tool events
  through `src/kazusa_ai_chatbot/db/scheduled_events.py`.
- Dispatcher runtime: `src/kazusa_ai_chatbot/dispatcher/dispatcher.py` writes
  scheduler events through `TaskDispatcher`.
- Existing send tool: `src/kazusa_ai_chatbot/dispatcher/handlers.py` uses
  `MessagingAdapter.send_message(...)`.
- Adapter protocol: `src/kazusa_ai_chatbot/dispatcher/adapter_iface.py`
  exposes `MessagingAdapter` and `SendResult`.
- Normal delivery receipts:
  `src/kazusa_ai_chatbot/db/conversation.py` updates assistant conversation
  rows through `apply_assistant_delivery_receipt(...)`.
- Normal `/chat` service path:
  `src/kazusa_ai_chatbot/service.py` registers adapters, saves assistant
  messages, and exposes `/delivery_receipt`.
- Read-only export artifacts, not committed:
  `test_artifacts\stage10_scheduled_events_sample.json` and
  `test_artifacts\stage10_delivery_rows_sample.json`.

Inspection result: the current database has `scheduled_events` and
`conversation_history` delivery fields, but no dedicated proactive outbox
collection. Stage 10 must not create one.

### To Later Work

Later plans may rely on:

- explicit proactive permission, preview, outbox, and send-audit record shapes;
- deterministic denial reason strings;
- fake-adapter transport tests proving the send boundary shape;
- no runtime registration, real DB write, or real adapter send in Stage 10.

Later plans must create a new approved plan before adding MongoDB outbox
persistence, scheduler integration, dispatcher integration, real adapter sends,
retry workers, permission-management UI, or normal conversation-history
insertion for proactive sent rows.

## Mandatory Rules

- Execute only from a feature branch forked from post-Stage-09 `main`.
- Keep edits inside the approved Change Surface.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces. Use single-quoted JSON arguments in PowerShell commands, for example
  `--sort '{"created_at":-1}'`.
- Do not send any proactive message through a real adapter.
- Do not call `scheduler.schedule_event`, `TaskDispatcher.dispatch`,
  `save_conversation`, or `apply_assistant_delivery_receipt` from Stage 10 code.
- Do not create or modify a MongoDB collection.
- Do not infer permission from user wording, keywords, `logical_stance`,
  `character_intent`, or any LLM field.
- Do not allow `user_message` trigger source through the proactive policy.
- Do not allow any output mode except `preview`.
- Do not send `internal_only` or `audit_only` preview content.
- Do not change normal `/chat` delivery, reply targeting, assistant
  persistence, consolidation behavior, RAG behavior, cognition prompts, media
  descriptor behavior, scheduler behavior, or dispatcher behavior.
- Do not add fallback sends, direct runtime adapter registration, retry loops,
  background workers, feature flags, or broad abstractions.
- Do not add raising-only helpers or pass-through wrappers.
- If implementation reveals an unlisted dependency, stop and update this plan
  before continuing.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any progress checklist stage, reread this entire plan
  before starting the next stage.

## Must Do

- Define exact TypedDict contracts for permission, quiet hours, preview,
  outbox, and send audit records.
- Define deterministic policy that returns the exact denial reasons listed in
  this plan.
- Add deterministic policy tests before implementation and observe the expected
  failure.
- Add dry-run outbox tests proving no adapter send occurs.
- Add fake-adapter transport tests proving only a `ready` outbox record can be
  marked sent.
- Reuse existing `MessagingAdapter` and `SendResult` only as a test transport
  protocol. Do not register the fake adapter with service runtime.
- Run static greps proving Stage 10 code does not call scheduler, dispatcher,
  normal conversation persistence, or delivery receipts.
- Run prior-stage regression gates and the full deterministic suite.
- Run the independent code review gate before lifecycle completion.

## Deferred

- Broad autonomous contact.
- Permission inference from chat content.
- MongoDB proactive outbox collection.
- Scheduler or dispatcher integration for proactive sends.
- Real adapter sends.
- Multi-recipient fanout.
- Media sends.
- LLM-based safety repair loops.
- Proactive conversation-history insertion.
- User-facing permission-management UI.
- Cross-platform transport abstraction redesign.

## Cutover Policy

Overall strategy: migration without runtime cutover.

| Area | Policy | Instruction |
|---|---|---|
| Permission records | migration | Add typed records and deterministic policy only. |
| Preview records | compatible | Keep previews separate from normal conversation history. |
| Outbox dry run | migration | Create in-memory/test outbox records only. |
| Fake transport | migration | Use only a local fake adapter in tests. |
| Real transport | deferred | No runtime send path in Stage 10. |
| `/chat` | compatible | No import or behavior change in normal response path. |

Rollback path: remove the `proactive_output` package and Stage 10 tests. There
are no database rows, scheduler events, runtime adapter registrations, or
conversation-history writes to roll back.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local variable names;
- local test helper names;
- assertion ordering;
- docstring wording that preserves the runtime meaning.

Not allowed:

- changing field names, status values, denial reason strings, function names,
  module names, test names, or Change Surface files;
- adding optional fields not listed below;
- changing denial order;
- adding database persistence;
- adding scheduler or dispatcher calls;
- adding runtime adapter registration;
- adding semantic keyword logic over user text;
- changing normal `/chat` behavior;
- adding speculative repository, protocol, strategy, config, retry, or worker
  abstractions.

## Target State

The Stage 10 test-only proactive path is:

```text
approved cognition preview fixture
-> deterministic proactive permission policy
-> proactive preview record
-> dry-run outbox record
-> fake-adapter send only for status "ready"
-> sent audit metadata on copied outbox record
```

The normal runtime path remains:

```text
adapter/debug client -> brain service -> queue/intake -> RAG -> cognition
-> dialog -> persistence/consolidation -> scheduler/reflection
```

No Stage 10 code is imported by the live runtime path.

## Contracts And Data Shapes

Create `src/kazusa_ai_chatbot/proactive_output/contracts.py` with these public
names only:

- `QuietHoursPolicy`
- `ProactivePermissionRecord`
- `ProactivePreviewRecord`
- `ProactivePolicyDecision`
- `ProactiveOutboxStatus`
- `ProactiveOutboxRecord`
- `ProactiveSendAuditRecord`
- `ProactiveOutboxStateError`

`QuietHoursPolicy` fields:

- `enabled: bool`
- `start_local_time: str`
- `end_local_time: str`

`ProactivePermissionRecord` fields:

- `permission_id: str`
- `platform: str`
- `platform_channel_id: str`
- `channel_type: str`
- `target_global_user_id: str`
- `target_platform_user_id: str`
- `allowed_trigger_sources: list[TriggerSource]`
- `allowed_output_modes: list[OutputMode]`
- `quiet_hours: QuietHoursPolicy`
- `expires_at: str`
- `enabled: bool`
- `created_at: str`
- `audit_reason: str`

`ProactivePreviewRecord` fields:

- `preview_id: str`
- `episode_id: str`
- `trigger_source: TriggerSource`
- `output_mode: OutputMode`
- `visibility: Visibility`
- `platform: str`
- `platform_channel_id: str`
- `channel_type: str`
- `target_global_user_id: str`
- `target_platform_user_id: str`
- `preview_text: str`
- `idempotency_key: str`
- `created_at: str`
- `audit_reason: str`

`ProactivePolicyDecision` fields:

- `allowed: bool`
- `reason: str`

`ProactiveOutboxStatus` values:

- `"dry_run"`
- `"ready"`
- `"sent"`
- `"denied"`
- `"failed"`
- `"cancelled"`

`ProactiveOutboxRecord` fields:

- `outbox_id: str`
- `preview_id: str`
- `permission_id: str`
- `idempotency_key: str`
- `platform: str`
- `platform_channel_id: str`
- `channel_type: str`
- `target_global_user_id: str`
- `target_platform_user_id: str`
- `preview_text: str`
- `status: ProactiveOutboxStatus`
- `created_at: str`
- `updated_at: str`
- `transport_attempt_count: int`
- `last_failure_reason: str`
- `sent_at: str`
- `platform_message_id: str`
- `delivery_adapter: str`
- `origin_kind: str`

`origin_kind` must be exactly `"proactive_preview"` before send and
`"proactive_sent"` after send.

`ProactiveSendAuditRecord` fields:

- `audit_id: str`
- `outbox_id: str`
- `event_type: str`
- `created_at: str`
- `reason: str`
- `platform_message_id: str`
- `delivery_adapter: str`

Create `src/kazusa_ai_chatbot/proactive_output/policy.py` with these public
names only:

- `PROACTIVE_ALLOWED_OUTPUT_MODE`
- `evaluate_proactive_permission`
- `is_local_time_in_quiet_hours`

`PROACTIVE_ALLOWED_OUTPUT_MODE` must be exactly `"preview"`.

`evaluate_proactive_permission(...)` signature:

```python
def evaluate_proactive_permission(
    *,
    preview: ProactivePreviewRecord,
    permission: ProactivePermissionRecord | None,
    existing_idempotency_keys: set[str],
    adapter_platforms: set[str],
    current_timestamp: str,
    current_local_time: str,
) -> ProactivePolicyDecision:
```

Denial checks must run in this exact order and return the exact reason string:

1. `permission is None` -> `"missing_permission"`
2. `permission["enabled"] is False` -> `"permission_disabled"`
3. `permission["expires_at"] <= current_timestamp` -> `"permission_expired"`
4. `preview["trigger_source"] == "user_message"` -> `"user_message_not_proactive"`
5. trigger source not allowed -> `"trigger_source_not_allowed"`
6. output mode not allowed or not `"preview"` -> `"unsafe_output_mode"`
7. visibility not `"model_visible"` -> `"content_not_public"`
8. platform/channel/channel type/target mismatch -> `"target_mismatch"`
9. current local time is inside quiet hours -> `"quiet_hours"`
10. preview platform absent from `adapter_platforms` -> `"adapter_unavailable"`
11. idempotency key already exists -> `"duplicate_idempotency_key"`
12. stripped preview text is empty -> `"empty_preview_text"`

The success decision is `{"allowed": True, "reason": "allowed"}`.

`is_local_time_in_quiet_hours(...)` signature:

```python
def is_local_time_in_quiet_hours(
    *,
    current_local_time: str,
    quiet_hours: QuietHoursPolicy,
) -> bool:
```

Rules: input times use `HH:MM`. If `quiet_hours["enabled"]` is false, return
false. If start equals end, the whole day is quiet. If start is before end, the
quiet window is `[start, end)`. If start is after end, the quiet window wraps
midnight and is `current >= start or current < end`. Invalid time strings must
raise `ValueError`.

Create `src/kazusa_ai_chatbot/proactive_output/outbox.py` with these public
names only:

- `build_proactive_preview_record`
- `build_proactive_outbox_record`
- `mark_proactive_outbox_denied`
- `send_ready_proactive_outbox`

`build_proactive_preview_record(...)` must accept keyword-only arguments named
exactly like `ProactivePreviewRecord` fields and return a
`ProactivePreviewRecord`.

`build_proactive_outbox_record(...)` signature:

```python
def build_proactive_outbox_record(
    *,
    outbox_id: str,
    preview: ProactivePreviewRecord,
    permission: ProactivePermissionRecord,
    status: ProactiveOutboxStatus,
    created_at: str,
) -> ProactiveOutboxRecord:
```

Allowed input status values are only `"dry_run"` and `"ready"`. Any other
status must raise `ProactiveOutboxStateError`.

`mark_proactive_outbox_denied(...)` signature:

```python
def mark_proactive_outbox_denied(
    *,
    outbox: ProactiveOutboxRecord,
    reason: str,
    updated_at: str,
) -> ProactiveOutboxRecord:
```

`send_ready_proactive_outbox(...)` signature:

```python
async def send_ready_proactive_outbox(
    *,
    outbox: ProactiveOutboxRecord,
    adapter: MessagingAdapter,
) -> ProactiveOutboxRecord:
```

It must raise `ProactiveOutboxStateError` unless
`outbox["status"] == "ready"`. For `"ready"`, it must call
`adapter.send_message(channel_id=..., text=..., channel_type=...,
reply_to_msg_id=None)`, then return a copied outbox with:

- `status` set to `"sent"`;
- `transport_attempt_count` incremented by 1;
- `sent_at` set to `SendResult.sent_at.isoformat()`;
- `platform_message_id` set to `SendResult.message_id`;
- `delivery_adapter` set to `SendResult.platform`;
- `origin_kind` set to `"proactive_sent"`;
- `updated_at` set to the same value as `sent_at`.

Do not catch adapter exceptions in Stage 10.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/proactive_output/__init__.py`
- `src/kazusa_ai_chatbot/proactive_output/contracts.py`
- `src/kazusa_ai_chatbot/proactive_output/policy.py`
- `src/kazusa_ai_chatbot/proactive_output/outbox.py`
- `tests/test_multi_source_cognition_stage_10_proactive_policy.py`
- `tests/test_multi_source_cognition_stage_10_proactive_outbox.py`

### Modify

- `development_plans/active/short_term/multi_source_cognition_architecture_stage_10_permissioned_proactive_output_plan.md`
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
- `development_plans/README.md`

### Keep

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/scheduler.py`
- `src/kazusa_ai_chatbot/db/scheduled_events.py`
- `src/kazusa_ai_chatbot/db/conversation.py`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/dispatcher/*.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2*.py`
- `src/kazusa_ai_chatbot/rag/*.py`
- `src/kazusa_ai_chatbot/reflection_cycle/*.py`
- `src/kazusa_ai_chatbot/internal_thought_cognition.py`

## Implementation Order

1. Reread this plan, Stage 09 `Execution Evidence`, the parent ledger row, and
   registry row.
2. Add `tests/test_multi_source_cognition_stage_10_proactive_policy.py` with:
   `test_missing_permission_is_denied`,
   `test_disabled_and_expired_permissions_are_denied`,
   `test_wrong_target_and_unapproved_trigger_are_denied`,
   `test_quiet_hours_denies_even_with_valid_permission`,
   `test_adapter_unavailable_and_duplicate_idempotency_are_denied`,
   `test_private_or_unsafe_preview_content_is_denied`, and
   `test_valid_permission_allows_preview`.
   Expected before implementation: import failure for missing package.
3. Implement `contracts.py` and `policy.py`.
4. Verify Stage 10 policy tests.
5. Reread this plan.
6. Add `tests/test_multi_source_cognition_stage_10_proactive_outbox.py` with:
   `test_preview_record_keeps_public_text_separate_from_outbox`,
   `test_dry_run_outbox_does_not_call_adapter`,
   `test_ready_outbox_fake_transport_marks_sent_with_audit_metadata`,
   `test_transport_refuses_dry_run_denied_or_sent_status`, and
   `test_denied_outbox_records_failure_reason_without_transport`.
   Expected before implementation: import failure for missing outbox module.
7. Implement `outbox.py` and `__init__.py`.
8. Verify Stage 10 outbox tests.
9. Run the full Verification section.
10. Run the Independent Code Review gate, remediate in-scope findings, rerun
    affected verification, then update lifecycle rows to completed.

## Progress Checklist

- [x] Stage 1 - prerequisite and runtime inspection complete.
  - Covers: approval prerequisites.
  - Verify: Stage 09 row is `completed`; scheduler, dispatcher, delivery, and
    conversation persistence contracts inspected.
  - Evidence/sign-off: `Codex / 2026-05-10`; Stage 09 branch
    `stage-09-multimodal-cognitive-input-sources`, commits `566e6eb` and
    `9de93fe`; read-only DB exports written under `test_artifacts`.
- [x] Stage 2 - executable contract approved.
  - Covers: exact files, data shapes, denial reasons, tests, and commands.
  - Verify: final executable wording is complete; registry and parent ledger mark Stage 10
    approved.
  - Evidence/sign-off: `Codex / 2026-05-10` after independent plan review.
- [x] Stage 3 - permission policy implemented.
  - Covers: Steps 2-4.
  - Verify: policy unit tests pass after expected import failure.
  - Evidence/sign-off: `Codex / 2026-05-10` after expected import failure for
    missing `kazusa_ai_chatbot.proactive_output` and green policy tests with
    `7 passed`.
  - Handoff: reread this plan, then start Stage 4.
- [x] Stage 4 - outbox dry-run and fake transport implemented.
  - Covers: Steps 5-8.
  - Verify: outbox tests pass after expected import failure; dry-run records do
    not call adapters.
  - Evidence/sign-off: `Codex / 2026-05-10` after expected import failure for
    missing `kazusa_ai_chatbot.proactive_output.outbox` and green outbox tests
    with `5 passed`.
  - Handoff: reread this plan, then start Stage 5.
- [x] Stage 5 - full verification and independent review complete.
  - Covers: Steps 9-10.
  - Verify: every Verification command passes and independent code review is
    recorded.
  - Evidence/sign-off: `Codex / 2026-05-10` after static compile, static
    greps, focused tests, prior-stage gates, full suite, and independent code
    review passed.
  - Handoff: no later stage in this parent plan.

## Verification

### Static Compile

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\proactive_output\__init__.py src\kazusa_ai_chatbot\proactive_output\contracts.py src\kazusa_ai_chatbot\proactive_output\policy.py src\kazusa_ai_chatbot\proactive_output\outbox.py tests\test_multi_source_cognition_stage_10_proactive_policy.py tests\test_multi_source_cognition_stage_10_proactive_outbox.py`

### Static Greps

- `rg -n "scheduler\\.schedule_event|TaskDispatcher|dispatcher\\.dispatch|save_conversation|apply_assistant_delivery_receipt|insert_scheduled_event|register_runtime_adapter|register_remote_runtime_adapter" src\kazusa_ai_chatbot\proactive_output tests\test_multi_source_cognition_stage_10_proactive_policy.py tests\test_multi_source_cognition_stage_10_proactive_outbox.py`

  Expected result: zero matches.

- `rg -n "proactive_output|proactive_preview|proactive_sent|Proactive" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\scheduler.py src\kazusa_ai_chatbot\db src\kazusa_ai_chatbot\dispatcher`

  Expected result: zero matches.

- `rg -n "send_message" src\kazusa_ai_chatbot\proactive_output tests\test_multi_source_cognition_stage_10_proactive_policy.py tests\test_multi_source_cognition_stage_10_proactive_outbox.py`

  Expected result: matches only in `outbox.py` `send_ready_proactive_outbox`
  and the fake-adapter outbox tests.

- `git diff --check`

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_10_proactive_policy.py`
- `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_10_proactive_outbox.py`

### Prior Stage Regression Gates

- `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`
- `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
- `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`
- `venv\Scripts\python.exe -m pytest tests\test_consolidation_origin_policy.py`
- `venv\Scripts\python.exe -m pytest tests\test_consolidator_origin_policy_db_writer.py`
- `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`
- `venv\Scripts\python.exe -m pytest`

### Independent Code Review

Review all Stage 10 code changes before sign-off:

- style compliance against `.agents/skills/py-style`;
- test compliance against `.agents/skills/test-style-and-execution`;
- plan alignment by exact Change Surface, public names, status values, denial
  reasons, denial order, and verification commands;
- design weaknesses, especially accidental live-send path, DB write,
  scheduler/dispatcher bypass, semantic permission inference, duplicate-send
  holes, and `/chat` regression risk.

Make in-scope fixes directly, rerun affected verification, then record the
review result in `Execution Evidence`.

## Acceptance Criteria

Stage 10 is complete when:

- permission, quiet-hour, preview, outbox, and audit contracts exist exactly as
  named;
- missing, disabled, expired, user-message, unapproved-trigger, unsafe-mode,
  private-content, wrong-target, quiet-hour, unsupported-adapter, duplicate,
  and empty-preview cases are denied deterministically;
- dry-run outbox creates no adapter send;
- approved fake-adapter transport writes auditable sent metadata;
- no Stage 10 code calls scheduler, dispatcher, normal conversation
  persistence, delivery receipts, DB writes, or runtime adapter registration;
- Stage 00 and prior-stage regression gates pass;
- full deterministic suite passes;
- independent code review approves the send boundary before merge.

## Independent Plan Review

Review on 2026-05-10:

- **Handoff:** Stage 09 evidence is now exact and carried forward.
- **Architecture alignment:** LLMs own preview wording outside this plan;
  deterministic Stage 10 code owns permission, target, quiet hours,
  idempotency, adapter availability, and audit metadata.
- **Boundary tightness:** no runtime import, DB collection, scheduler event,
  dispatcher call, or real adapter send is allowed.
- **Agent creativity suppression:** exact files, public names, fields, status
  values, denial reasons, denial order, tests, and verification commands are
  specified.
- **Approval:** approved for implementation from a post-Stage-09 feature
  branch.

## Plan Self-Review

- **Coverage:** parent Stage 10 scope maps to permission, preview, outbox, fake
  transport, audit metadata, rollback, and regression gates.
- **Completeness scan:** no planning-only instruction text remains.
- **Contract consistency:** the plan creates no autonomous contact by default
  and requires a later approval for real transport or persistence.
- **Granularity:** checkpoints separate policy, outbox/fake transport,
  verification, and independent review.
- **Verification:** denial coverage, no-send dry run, fake transport,
  static greps, prior-stage gates, full suite, and independent review are
  explicit.

## Execution Handoff

Execution mode: sequential implementation on a feature branch forked from
post-Stage-09 `main`.

Next action: create a Stage 10 feature branch, add the policy tests, observe
the expected import failure, implement the exact contracts, and continue
through the checklist.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Message sends without permission | Explicit permission record required | Policy denial tests |
| Wrong target receives message | Target validation before outbox/send | Policy tests and fake adapter audit |
| Duplicate proactive send | Idempotency key gate | Policy tests |
| Private thought leaks | Only `model_visible` `preview` content can send | Policy tests |
| Runtime contact accidentally enabled | No imports from service/scheduler/dispatcher; static greps | Static greps |
| `/chat` regresses | No normal response-path changes | Stage 00 and full suite |
| Audit loss after rollback | No DB writes in Stage 10 | Rollback policy |

## Completion Artifact Contract

When Stage 10 is complete, these artifacts must exist or be updated:

- `src/kazusa_ai_chatbot/proactive_output/contracts.py`
- `src/kazusa_ai_chatbot/proactive_output/policy.py`
- `src/kazusa_ai_chatbot/proactive_output/outbox.py`
- `tests/test_multi_source_cognition_stage_10_proactive_policy.py`
- `tests/test_multi_source_cognition_stage_10_proactive_outbox.py`
- parent ledger row for `stage_10` flipped to `completed`;
- registry row flipped to `completed | completed`;
- execution evidence in this plan naming branch, commit, checks, independent
  review, and sign-off.

The artifact must not include inferred permissions, broad autonomous contact,
normal `/chat` response-path changes, database writes, scheduler or dispatcher
calls, direct runtime adapter registration, or hidden transport sends.

## Execution Evidence

- Stage 09 evidence reread: branch
  `stage-09-multimodal-cognitive-input-sources`; implementation commit
  `566e6eb`; evidence/lifecycle commit `9de93fe`; focused Stage 09 tests
  `15 passed`; Stage 00 `11 passed`; Stage 03 `36 passed`; Stage 06 `9 passed`;
  Stage 07 `14 passed`; Stage 08 `26 passed`.
- Runtime inspection: inspected scheduler, dispatcher, adapter protocol,
  delivery receipt, conversation persistence, and service adapter registration
  files named in `Runtime Inspection`; read-only DB exports written to
  `test_artifacts\stage10_scheduled_events_sample.json` and
  `test_artifacts\stage10_delivery_rows_sample.json` and left uncommitted.
- Branch: `stage-10-permissioned-proactive-output`.
- Commit: implementation commit `680a6ca`.
- Static compile:
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\proactive_output\__init__.py src\kazusa_ai_chatbot\proactive_output\contracts.py src\kazusa_ai_chatbot\proactive_output\policy.py src\kazusa_ai_chatbot\proactive_output\outbox.py tests\test_multi_source_cognition_stage_10_proactive_policy.py tests\test_multi_source_cognition_stage_10_proactive_outbox.py`
  passed.
- Static greps:
  - scheduler/dispatcher/persistence forbidden-call grep: zero matches.
  - service/scheduler/db/dispatcher proactive runtime-import grep: zero
    matches.
  - `send_message` grep: only
    `src\kazusa_ai_chatbot\proactive_output\outbox.py` and
    `tests\test_multi_source_cognition_stage_10_proactive_outbox.py` matched.
  - `git diff --check`: passed.
- Focused tests:
  - policy test red: expected collection failure for missing
    `kazusa_ai_chatbot.proactive_output`.
  - outbox test red: expected collection failure for missing
    `kazusa_ai_chatbot.proactive_output.outbox`.
  - `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_10_proactive_policy.py`:
    `7 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_10_proactive_outbox.py`:
    `5 passed`.
- Prior stage regression gates:
  - `tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`:
    `15 passed`.
  - `tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`:
    `26 passed`.
  - `tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`:
    `14 passed`.
  - stale Stage 06 filename in this plan failed with file-not-found, then the
    plan was corrected to the real Stage 06 gates.
  - `tests\test_consolidation_origin_policy.py`: `5 passed`.
  - `tests\test_consolidator_origin_policy_db_writer.py`: `4 passed`.
  - `tests\test_multi_source_cognition_stage_00_regression_baseline.py`:
    `11 passed`.
- Full suite: `venv\Scripts\python.exe -m pytest` passed with `974 passed,
  217 deselected`.
- Independent code review: checked py-style, test style, plan alignment, and
  design risk. One issue found and fixed: explicit module public surfaces were
  added with `__all__`, and an unnecessary `Any` cast in the outbox test helper
  was removed. Reran compile, focused tests, and static greps after the fix.
- Completion diff review: Change Surface contains only the approved
  `proactive_output` package, two Stage 10 test files, this plan, the parent
  ledger, and the registry. No service, scheduler, DB, dispatcher, RAG,
  cognition prompt, or consolidation runtime files were modified.
- Lifecycle records: this plan status set to `completed`; parent ledger row set
  to `completed`; registry row set to `completed | completed`.
- Sign-off: `Codex / 2026-05-10`.
