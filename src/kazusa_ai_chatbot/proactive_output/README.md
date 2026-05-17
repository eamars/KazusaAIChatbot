# Proactive Output Interface Control Document

## Document Control

- ICD id: `PO-ICD-001`
- Owning package: `kazusa_ai_chatbot.proactive_output`
- Interface boundary: source-aware cognition preview records -> deterministic
  permission policy -> auditable outbox record -> test-only fake transport
- Runtime consumers: deterministic tests and future proactive-output planners
- Upstream owners: source-specific cognition entry points that may produce
  `output_mode="preview"` records after explicit approval
- Downstream owners: none in the current runtime. Scheduler, dispatcher,
  database persistence, and real adapter delivery are deliberately out of
  scope for this ICD version.

This ICD owns the proactive-output contract added by the multi-source
cognition architecture. It is intentionally narrower than the dispatcher
contract: dispatcher sends accepted user-facing commitments through scheduled
tasks, while proactive output models a permissioned future path for
non-user-message cognition previews.

## Purpose

Proactive output exists to prevent autonomous contact from becoming an implicit
side effect of reflection, internal thought, scheduler work, or model output.

The current module provides deterministic contracts for:

- explicit target permission;
- public preview text;
- policy allow or deny decisions;
- dry-run or ready outbox records;
- a fake transport boundary used only by tests.

No current production path imports this package to send a message. Stage 10
ends at contract, policy, outbox, and fake-transport verification.

This package does not govern selected self-cognition speech. A selected self-cognition `speak`
is immediate selected speech handled by the shared dialog, persistence, and
runtime adapter bridge path, not by the proactive output outbox or fake
transport contract.

## Public Interface

Current callers import concrete modules:

```python
from kazusa_ai_chatbot.proactive_output.contracts import (
    ProactiveOutboxRecord,
    ProactiveOutboxStateError,
    ProactiveOutboxStatus,
    ProactivePermissionRecord,
    ProactivePolicyDecision,
    ProactivePreviewRecord,
    ProactiveSendAuditRecord,
    QuietHoursPolicy,
)
from kazusa_ai_chatbot.proactive_output.outbox import (
    build_proactive_outbox_record,
    build_proactive_preview_record,
    mark_proactive_outbox_denied,
    send_ready_proactive_outbox,
)
from kazusa_ai_chatbot.proactive_output.policy import (
    PROACTIVE_ALLOWED_OUTPUT_MODE,
    evaluate_proactive_permission,
    is_local_time_in_quiet_hours,
)
```

The package `__init__.py` is not a public facade yet. A future runtime
integration must either add an explicit facade and update this ICD, or keep
using the module-level imports above.

## Record Contracts

`ProactivePermissionRecord` is the explicit target grant. It carries platform,
channel, target user, allowed trigger sources, allowed output modes, quiet
hours, expiry, enabled flag, and audit reason.

`ProactivePreviewRecord` is candidate outward text. It carries the source
episode id, trigger source, output mode, visibility, target fields, preview
text, idempotency key, timestamp, and audit reason.

`ProactivePolicyDecision` is deterministic:

```python
{
    "allowed": bool,
    "reason": str,
}
```

`ProactiveOutboxRecord` is an auditable local record. It contains target
fields copied from the preview, the permission id, idempotency key, preview
text, status, transport counters, delivery metadata, and `origin_kind`.

`ProactiveSendAuditRecord` is reserved for future audit persistence. This
module does not write audit rows.

## Policy Contract

`evaluate_proactive_permission(...)` is the required gate before an outbox
record can move from dry run to ready. It checks, in order:

- permission exists;
- permission is enabled;
- permission is not expired;
- preview is not sourced from `trigger_source="user_message"`;
- preview trigger source is allowed by the permission;
- preview output mode is `preview` and is allowed by the permission;
- preview visibility is `model_visible`;
- preview target exactly matches the permission target;
- current local time is outside quiet hours;
- an adapter platform is available;
- idempotency key is new;
- preview text is not blank.

The allow result is:

```python
{"allowed": True, "reason": "allowed"}
```

Every denial returns `allowed=False` with a stable reason string. Policy code
does not call LLMs, scheduler, dispatcher, database APIs, or adapters.

## Outbox Contract

`build_proactive_preview_record(...)` only builds a typed preview record from
caller-supplied values. It does not approve, schedule, persist, or send.

`build_proactive_outbox_record(...)` accepts only initial status `dry_run` or
`ready`. It also verifies that preview and permission target fields match
before constructing the record. A mismatch raises `ProactiveOutboxStateError`.

`mark_proactive_outbox_denied(...)` copies an outbox record into status
`denied` and records the failure reason. It does not call transport.

`send_ready_proactive_outbox(...)` is the test-only transport boundary. It
requires `status="ready"` and requires `adapter.platform` to match the outbox
platform before calling `adapter.send_message(...)`. On success it returns a
copied outbox record with status `sent`, incremented attempt count, delivery
metadata, and `origin_kind="proactive_sent"`.

Adapter exceptions are not caught by this module. A future runtime owner must
define retry, failure persistence, and backoff policy in a separate approved
plan before enabling real transport.

## State Model

Allowed outbox statuses:

```text
dry_run
ready
sent
denied
failed
cancelled
```

Current builders create only `dry_run` or `ready` records. Current helpers can
transition:

```text
ready -> sent
dry_run|ready|sent|failed|cancelled -> denied
```

No helper currently creates `failed` or `cancelled`; those are reserved typed
states for future runtime persistence.

## Forbidden Runtime Paths

This module must not:

- register a scheduler job;
- create scheduler-owned user-visible output;
- call normal conversation persistence;
- call delivery receipt persistence;
- register runtime adapters;
- import service startup code;
- turn reflection or internal thought into a fake user message.

The only permitted transport call in this ICD version is the test-only
`send_ready_proactive_outbox(...)` call to a supplied `MessagingAdapter`.

## Handoff Boundaries

Upstream cognition may produce a preview only when its source contract allows
`output_mode="preview"`. The preview is not deliverable until a matching
permission record exists and policy returns `allowed`.

Accepted future contact must go through a reviewed cognition or scheduler
owner. Proactive output is not a replacement for delayed-contact scheduling and
must not bypass permission or scheduler contracts.

Future production integration must define, in a new approved plan:

- where permission records are stored;
- how users grant, revoke, and inspect permission;
- which scheduler or queue owns delayed proactive sends;
- which audit collection stores preview, outbox, and send events;
- how retries, cancellation, and idempotency are enforced across restarts;
- how real adapters are selected and availability is checked at send time.

## Verification Contract

Any change to this module must run:

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\proactive_output\__init__.py src\kazusa_ai_chatbot\proactive_output\contracts.py src\kazusa_ai_chatbot\proactive_output\policy.py src\kazusa_ai_chatbot\proactive_output\outbox.py tests\test_multi_source_cognition_stage_10_proactive_policy.py tests\test_multi_source_cognition_stage_10_proactive_outbox.py
venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_10_proactive_policy.py tests\test_multi_source_cognition_stage_10_proactive_outbox.py
rg -n "scheduler\.schedule_event|dispatcher\.dispatch|save_conversation|apply_assistant_delivery_receipt|insert_scheduled_event|register_runtime_adapter|register_remote_runtime_adapter" src\kazusa_ai_chatbot\proactive_output tests\test_multi_source_cognition_stage_10_proactive_policy.py tests\test_multi_source_cognition_stage_10_proactive_outbox.py
rg -n "proactive_output|proactive_preview|proactive_sent|Proactive" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\scheduler.py src\kazusa_ai_chatbot\db src\kazusa_ai_chatbot\dispatcher
rg -n "send_message" src\kazusa_ai_chatbot\proactive_output tests\test_multi_source_cognition_stage_10_proactive_policy.py tests\test_multi_source_cognition_stage_10_proactive_outbox.py
git diff --check
```

Expected grep results:

- forbidden-call grep: zero matches;
- service/scheduler/db/dispatcher integration grep: zero matches;
- `send_message` grep: matches only this module's fake transport boundary and
  its tests.

Changes that add DB writes, scheduler integration, runtime adapter lookup, or
real transport are not documentation-only changes. They require a new approved
development plan and an update to this ICD.
