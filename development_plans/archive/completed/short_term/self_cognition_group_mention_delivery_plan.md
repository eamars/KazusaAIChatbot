# self cognition group mention delivery plan

## Summary

- Goal: Make proactive self-cognition group sends carry an optional
  platform-neutral target-user mention request derived from self-cognition
  target scope, while adapters decide whether and how a native mention can be
  rendered.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, `cjk-safety` when editing Python
  files containing CJK strings
- Overall cutover strategy: compatible
- Highest-risk areas: preserving the cognition/dialog/adapter ownership
  boundaries, avoiding new LLM prompt obligations, keeping native platform tag
  syntax out of brain-owned text, keeping outbound delivery backward
  compatible, and not making mention feasibility a brain-service concern.
- Acceptance criteria: implementation is complete only when self-cognition
  proactive group action candidates carry best-effort delivery mention
  metadata for their single semantic target user, dispatcher and scheduler
  paths preserve that metadata, adapters render or ignore it according to
  platform feasibility, existing sends without metadata remain unchanged, all
  verification gates pass, and independent code review approves the result.

## Context

The original symptom came from a proactive self-cognition message posted into a
group without a live inbound reply anchor. Participants could not tell which
user the character was addressing.

Code inspection corrected the design boundary:

- Current self-cognition does not support a real "group broadcast" outbound
  mode. The `group_noise_rejected` case is a silence/audit path and must not be
  used as a reason for this feature.
- Current live chat already has a relevance-layer `use_reply_feature` decision.
  Live-chat mention policy is not part of this plan.
- Current self-cognition action candidates are target-scoped `send_message`
  candidates. Active-commitment cases already know the semantic target user via
  `target_scope.user_id`, but they need platform-target metadata to request a
  native mention.
- Current dispatcher delivery carries target channel, channel type, text,
  optional platform, and optional reply id. It has no first-class outbound
  mention metadata contract.

The corrected product requirement is narrow:

```text
self-cognition proactive group send
  -> one resolved semantic target user
  -> optional delivery mention request
  -> adapter renders native mention if feasible, otherwise sends plain text
```

The brain service must not care whether a platform adapter supports native
mentions. Brain-owned code emits platform-neutral metadata only. Adapter-owned
code handles support, syntax, missing platform account ids, and no-op fallback.

## Mandatory Skills

- `development-plan-writing`: preserve this work contract and record execution
  evidence before lifecycle changes.
- `local-llm-architecture`: keep semantic judgment in the existing LLM stages,
  keep rendering deterministic, and do not add new LLM calls or prompt
  obligations.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files that contain CJK prompt text
  or adding CJK string literals to Python.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Do not change self-cognition channel selection, contact selection,
  duplicate suppression, scheduler validation, or permission policy.
- Do not change relevance prompts, cognition prompts, dialog prompts, dialog
  output schema, or live-chat reply behavior for this plan.
- Do not add `mention_target_user`, `{target_mention}`, XML/HTML tags,
  sentinel tokens, slot replacement, or any other LLM-authored mention marker.
- Do not add a new LLM call for tag insertion, tag repair, mention placement,
  capability detection, or delivery fallback.
- Self-cognition may request at most one target-user delivery mention for this
  plan, and only for proactive group sends with one semantic target user.
- Brain-owned code must not check whether QQ, Discord, debug, or any future
  adapter supports native mentions. Capability and feasibility are adapter
  concerns.
- Brain-owned code must not emit QQ `[CQ:at,...]`, Discord `<@...>`, readable
  `@name` prefixes, or any platform-native syntax into message text.
- Delivery-only `platform_user_id` and `display_name` metadata must not be
  added to LLM-visible self-cognition source packets, RAG requests, cognition
  state, dialog state, prompts, or rendered target-scope text.
- Adding delivery-only target metadata must not change action-attempt
  idempotency keys or duplicate suppression. Idempotency remains based on the
  existing stable target scope: platform, platform channel id, channel type,
  semantic user id, source id, due occurrence, and action kind.
- Dispatcher and scheduler code must treat `delivery_mentions` as optional
  metadata. Missing, empty, incomplete, unsupported, or ignored mentions must
  not block text delivery.
- Adapter code owns native rendering. If a requested mention is unsupported or
  lacks the platform identity needed by that adapter, the adapter sends the
  original text without a native mention and may log a non-content diagnostic.
- Prefix is the only approved placement. Adapters must not infer placement
  from natural language.
- Do not project adapter-rendered native mention syntax into conversation
  history. Existing semantic addressing such as `addressed_to_global_user_ids`
  remains the brain-side record of who the assistant addressed.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in
  `Execution Evidence`.

## Must Do

- Extend self-cognition target scope for active-commitment cases with optional
  `platform_user_id` and `display_name` from the selected latest conversation
  row.
- Define a platform-neutral `DeliveryMention` request shape for outbound
  sends.
- Add deterministic construction of zero or one `DeliveryMention` for
  self-cognition proactive group action candidates with a single semantic
  target user.
- Preserve existing duplicate-suppression identity when adding delivery-only
  target metadata.
- Propagate `delivery_mentions` from self-cognition action candidates through
  self-cognition handoff into dispatcher `send_message` args.
- Extend dispatcher validation, scheduler-compatible task args, handler
  delivery, remote adapter bridge, and adapter protocol to pass optional
  `delivery_mentions`.
- Implement adapter-side prefix rendering for supported QQ and Discord user
  mentions when the adapter has enough platform identity to render them.
- Implement adapter-side no-op behavior for unsupported or incomplete mention
  requests while preserving text delivery.
- Preserve existing sends that omit `delivery_mentions` as plain text sends.
- Preserve private-channel behavior by sending without delivery mention
  requests.
- Update self-cognition, dispatcher, adapter, and runtime-adapter docs so the
  ownership contract is explicit.
- Add deterministic unit and integration tests for self-cognition target
  metadata, candidate metadata construction, dispatcher pass-through, adapter
  rendering, adapter no-op behavior, and backward compatibility.

## Deferred

- Do not redesign self-cognition channel selection or proactive-contact
  policy.
- Do not add proactive-contact rate limits, quiet hours, cooldowns, or spam
  suppression.
- Do not add group broadcast as an outbound self-cognition capability.
- Do not add live-chat visible mention behavior.
- Do not add dialog-generated mention decisions.
- Do not add multi-target mentions.
- Do not add arbitrary placement, mid-text tags, suffix tags, natural-language
  placement inference, or slot replacement.
- Do not add channel mentions, role mentions, everyone/here mentions, or
  arbitrary platform mention syntax.
- Do not add adapter capability discovery to the brain service.
- Do not make target mentions required for delivery.
- Do not migrate existing scheduled events or historical conversation rows.
- Do not project rendered native mention syntax into conversation history.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Self-cognition target scope | compatible | Add optional platform-user metadata. Existing cases without the fields remain valid. |
| Self-cognition action candidate | compatible | Add optional `delivery_mentions`. Existing candidates without the field remain valid. |
| Dispatcher `send_message` args | compatible | Add optional `delivery_mentions`. Existing scheduled sends without the field still validate and deliver. |
| Scheduler documents | compatible | Preserve old scheduled-event rows. New rows may carry `delivery_mentions` only inside existing task args. No migration. |
| Adapter interface | compatible | Add optional `delivery_mentions` with a default empty value. Existing callers remain valid. |
| Runtime adapter HTTP payload | compatible | Add optional `delivery_mentions` to `/send_message` requests. Existing adapter callbacks without the field remain valid. |
| Conversation history | compatible | Keep assistant text prompt-safe and platform-neutral. Do not require history migration. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Do not choose a migration or big-bang rewrite for scheduler payloads.
- Any change to this policy requires user approval before implementation.

## Data Migration

No data migration is approved or required.

- Do not backfill historical scheduled events.
- Do not mutate historical conversation-history rows.
- Existing scheduled events and conversation rows without delivery mention
  metadata must continue to load and execute.

## Agent Autonomy Boundaries

- The agent may choose helper names and local function layout when the public
  contracts in this plan remain unchanged.
- The agent must not invent additional delivery modes, placement algorithms,
  prompt fields, prompt families, fallback LLM calls, feature flags,
  compatibility shims, or unrelated cleanup.
- The agent must treat changes outside self-cognition target projection,
  self-cognition tracking/handoff, dispatcher delivery, adapter delivery,
  runtime adapter bridge, focused docs, and focused tests as out of scope
  unless this plan names them.
- The agent may add small structural helpers for delivery mention construction,
  metadata normalization, and adapter prefix rendering.
- If existing helper behavior already satisfies schema validation or
  platform-neutral mention projection, reuse it instead of duplicating logic.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

For a self-cognition proactive group action with one semantic target user, the
candidate may carry delivery metadata like this:

```json
{
  "target_platform": "qq",
  "target_channel": "54369546",
  "target_channel_type": "group",
  "text": "About that harder challenge, it is time.",
  "dispatch_shape": "send_message",
  "delivery_mentions": [
    {
      "entity_kind": "user",
      "placement": "prefix",
      "platform_user_id": "673225019",
      "global_user_id": "256e8a10-c406-47e9-ac8f-efd270d18160",
      "display_name": "target display name",
      "requested_by": "self_cognition.target_scope"
    }
  ]
}
```

The dispatcher passes the metadata unchanged. The adapter receives original
text plus mention metadata:

```python
await adapter.send_message(
    channel_id=target_channel,
    text=text,
    channel_type=channel_type,
    reply_to_msg_id=reply_to_msg_id,
    delivery_mentions=delivery_mentions,
)
```

The QQ adapter renders a native prefix mention when feasible. The Discord
adapter renders a native prefix mention when feasible. Any adapter that cannot
render the request sends the original text unchanged. The brain service does
not branch on adapter support.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Scope | Limit this plan to self-cognition proactive group sends. | This fixes the observed failure without changing live chat or generic dialog behavior. |
| Group broadcast | Do not add or design around group broadcast. | Current self-cognition group-noise handling is a no-action path, not an outbound broadcast capability. |
| LLM ownership | Do not involve dialog, cognition, or relevance in mention generation. | The target user is already known from self-cognition target scope; adding a high-temperature dialog flag would increase fragility and duplicate existing routing decisions. |
| Brain/adapter boundary | Brain passes platform-neutral metadata; adapters render or no-op. | Platform support and native syntax are transport concerns. |
| Placement | Prefix only. | Prefix is deterministic and solves the clarity problem without token placement prompts. |
| Missing adapter support | Send plain text. | Mentions are clarity affordances, not delivery requirements. |
| Conversation history | Keep native mention syntax out of stored assistant text. | Future prompts should not learn adapter syntax. Semantic addressee metadata already records target ownership. |

## Contracts And Data Shapes

### Self-Cognition Target Scope

Extend self-cognition target scope with optional platform-user identity:

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

Rules:

- `user_id` remains the internal global user id and is the semantic target.
- `platform_user_id` is the best available platform account id from the latest
  selected conversation row.
- `display_name` is a diagnostic/readability label only.
- Missing `platform_user_id` must not prevent candidate construction or text
  delivery. The adapter decides whether the mention can be rendered.
- `platform_user_id` and `display_name` are delivery metadata, not model
  context. Do not render them into self-cognition source packets, RAG request
  context, cognition state, dialog state, or prompt-visible target-scope text.
- `platform_user_id` and `display_name` must not participate in
  duplicate-suppression idempotency keys.

### DeliveryMention

Use this platform-neutral shape wherever a send path needs an outbound
target-user tag request:

```python
DeliveryMention = {
    "entity_kind": "user",
    "placement": "prefix",
    "platform_user_id": str | None,
    "global_user_id": str | None,
    "display_name": str,
    "requested_by": str,
}
```

Rules:

- `entity_kind` must be `user` for this plan.
- `placement` must be `prefix` for this plan.
- `platform_user_id` is optional at the brain contract boundary because
  adapter feasibility is not a brain concern.
- `global_user_id` should be present when known and is used for audit and
  semantic addressing.
- `requested_by` must be `self_cognition.target_scope` for mentions produced
  by this plan.

### Self-Cognition Mention Construction

Construct zero or one `DeliveryMention` during action-candidate construction:

- Create one request when `target_channel_type == "group"` and
  `target_scope.user_id` identifies exactly one semantic target user.
- Do not create a request for private sends.
- Do not create a request for group-noise/audit-only cases, because they do
  not create action candidates.
- Do not inspect adapter type, adapter registration, adapter capability, or
  platform-specific syntax.
- Do not require `platform_user_id`; pass it when available.

### `send_message` Tool Args

Extend the existing schema with optional delivery metadata:

```json
{
  "delivery_mentions": [
    {
      "entity_kind": "user",
      "placement": "prefix",
      "platform_user_id": "platform account id when known",
      "global_user_id": "internal global user id when known",
      "display_name": "optional display label",
      "requested_by": "self_cognition.target_scope"
    }
  ]
}
```

Existing tool calls that omit `delivery_mentions` remain valid and send plain
text.

### Adapter Interface

Extend `MessagingAdapter.send_message(...)` with optional delivery mentions:

```python
async def send_message(
    channel_id: str,
    text: str,
    *,
    channel_type: str,
    reply_to_msg_id: str | None = None,
    delivery_mentions: Sequence[DeliveryMention] | None = None,
) -> SendResult:
    ...
```

Adapters receive text with no marker token. If a renderable prefix user
mention exists, they render it before the text using native platform syntax.
If no renderable mention exists, they send the original text unchanged.

## LLM Call And Context Budget

- No new LLM calls are allowed.
- No new model route is allowed.
- No prompt changes are allowed for relevance, cognition, dialog,
  consolidation, dispatcher, or self-cognition.
- No new context field is passed to any LLM for this feature.
- Normal response-path latency is unchanged.
- Background self-cognition LLM call count is unchanged.

## Change Surface

Expected code files:

- `src/kazusa_ai_chatbot/self_cognition/models.py`
- `src/kazusa_ai_chatbot/self_cognition/projection.py` only to preserve or
  verify that delivery-only metadata is not rendered into model-visible
  packets
- `src/kazusa_ai_chatbot/self_cognition/sources.py`
- `src/kazusa_ai_chatbot/self_cognition/runner.py` if target-scope projection
  or fixture normalization needs the new optional fields
- `src/kazusa_ai_chatbot/self_cognition/tracking.py`
- `src/kazusa_ai_chatbot/self_cognition/handoff.py`
- `src/kazusa_ai_chatbot/dispatcher/adapter_iface.py`
- `src/kazusa_ai_chatbot/dispatcher/evaluator.py`
- `src/kazusa_ai_chatbot/dispatcher/handlers.py`
- `src/kazusa_ai_chatbot/dispatcher/remote_adapter.py`
- `src/kazusa_ai_chatbot/dispatcher/task.py` only if scheduler rehydration
  needs typed helper support; prefer preserving `delivery_mentions` inside
  existing task args.
- `src/kazusa_ai_chatbot/brain_service/runtime_adapters.py` only if the
  runtime callback contract or request model needs the optional field.
- `src/adapters/napcat_qq_adapter.py`
- `src/adapters/discord_adapter.py`

Expected docs:

- `src/kazusa_ai_chatbot/self_cognition/README.md`
- `src/kazusa_ai_chatbot/dispatcher/README.md`
- `src/kazusa_ai_chatbot/brain_service/README.md` only if runtime adapter
  callback documentation needs updating.

Expected tests:

- `tests/test_delivery_mentions.py`
- `tests/test_self_cognition_tracking.py`
- `tests/test_self_cognition_integration.py`
- `tests/test_scheduler_future_promise.py`
- `tests/test_runtime_adapter_registration.py`

## Implementation Order

1. Add failing self-cognition metadata and candidate tests.
   - File: `tests/test_self_cognition_tracking.py`.
   - Cover active-commitment cases retaining target `platform_user_id` and
     display name from the selected latest conversation row.
   - Cover group action candidates carrying one prefix `delivery_mentions`
     request for the target user.
   - Cover duplicate-suppression idempotency keys staying unchanged when
     `platform_user_id` or `display_name` is present.
   - Cover self-cognition source-packet and RAG-request projection not
     rendering delivery-only metadata into LLM-visible context.
   - Cover private action candidates carrying no delivery mention request.
   - Cover group-noise rejected cases still creating no action candidate.
   - Verify before implementation:
     `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`.

2. Add failing delivery mention contract tests.
   - File: `tests/test_delivery_mentions.py`.
   - Cover construction with and without `platform_user_id`.
   - Cover no-op construction for private sends and missing semantic target.
   - Cover metadata shape, placement, and `requested_by`.
   - Verify before implementation:
     `venv\Scripts\python -m pytest tests\test_delivery_mentions.py -q`.

3. Add failing dispatcher and scheduler pass-through tests.
   - File: `tests/test_scheduler_future_promise.py`.
   - Cover `delivery_mentions` pass-through, omitted metadata, and empty-list
     metadata.
   - Cover handler passing metadata to the adapter without changing text.
   - Verify before implementation:
     `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`.

4. Add failing adapter rendering and no-op tests.
   - File: `tests/test_runtime_adapter_registration.py` or a focused adapter
     test file if existing adapter tests are a better local fit.
   - Cover QQ rendering of a prefix user mention as a native at segment before
     text when platform identity is present.
   - Cover Discord rendering of a prefix user mention as a native user mention
     before text when platform identity is present.
   - Cover adapter no-op behavior when the mention is unsupported or lacks
     required platform identity.
   - Verify before implementation:
     `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`.

5. Implement self-cognition target metadata and candidate construction.
   - Populate optional `platform_user_id` and `display_name` from the selected
     latest conversation row.
   - Add helper logic that constructs zero or one `DeliveryMention` for group
     self-cognition action candidates.
   - Keep delivery-only metadata out of LLM-visible projection and
     idempotency identity.
   - Thread the metadata into action candidates.
   - Re-run steps 1 and 2 tests and record evidence.

6. Implement self-cognition handoff and dispatcher pass-through.
   - Include `delivery_mentions` in raw `send_message` tool args when present.
   - Extend dispatcher schema/evaluation while preserving old args.
   - Pass optional metadata through handler and remote adapter payload.
   - Re-run step 3 tests and record evidence.

7. Implement adapter rendering.
   - Extend the adapter protocol with optional `delivery_mentions`.
   - Render supported prefix user mentions in QQ and Discord adapters.
   - No-op unsupported or incomplete requests without mutating text.
   - Re-run step 4 tests and record evidence.

8. Update docs.
   - Document that self-cognition may request a delivery mention for one
     target user in group proactive sends.
   - Document that adapters own feasibility and native rendering.
   - Document that brain-owned text remains platform-neutral.

9. Run verification gates.
   - Run focused tests first.
   - Run static greps.
   - Run broader affected regression tests.
   - Record all output in `Execution Evidence`.

10. Run independent code review.
    - Review the diff against this plan, style rules, adapter boundaries,
      scheduler compatibility, and verification accuracy.
    - Record findings and remediation in `Execution Evidence`.

## Progress Checklist

- [x] Stage 1 - self-cognition metadata and candidate tests written
  - Covers: implementation order step 1.
  - Verify: `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`.
  - Evidence: record expected failing tests or baseline result in
    `Execution Evidence`.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 2 - delivery mention contract tests written
  - Covers: implementation order step 2.
  - Verify: `venv\Scripts\python -m pytest tests\test_delivery_mentions.py -q`.
  - Evidence: record expected failing tests or baseline result in
    `Execution Evidence`.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 3 - dispatcher and scheduler pass-through tests written
  - Covers: implementation order step 3.
  - Verify: `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`.
  - Evidence: record expected failing tests or baseline result in
    `Execution Evidence`.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 4 - adapter rendering and no-op tests written
  - Covers: implementation order step 4.
  - Verify: `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`.
  - Evidence: record expected failing tests or baseline result in
    `Execution Evidence`.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 5 - self-cognition metadata and candidate implementation complete
  - Covers: implementation order step 5.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_delivery_mentions.py -q`.
  - Evidence: record changed files and test output in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 6 - dispatcher and runtime pass-through complete
  - Covers: implementation order step 6.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`.
  - Evidence: record changed files and test output in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 7 - adapter rendering complete
  - Covers: implementation order step 7.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`.
  - Evidence: record changed files and test output in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 8 - docs and full verification complete
  - Covers: implementation order steps 8 and 9.
  - Verify: all commands in `Verification`.
  - Evidence: record static grep results, compile output, and test output in
    `Execution Evidence`.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 9 - independent code review complete
  - Covers: implementation order step 10.
  - Verify: run the `Independent Code Review` gate.
  - Evidence: record review findings, fixes, rerun commands, residual risks,
    and approval status in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-14` after review approval and required reruns.

## Verification

### Static Greps

- `rg -n "target_mention|\\{target_mention\\}|<target|__TARGET_MENTION__|\\[\\[target_mention\\]\\]" src tests`
  - Expected after implementation: no production source matches. Test or
    experiment artifacts may mention historical token experiments only when
    explicitly scoped outside production behavior.
- `rg -n "mention_target_user" src tests`
  - Expected after implementation: no matches. This plan does not add a dialog
    mention flag.
- `rg -n "CQ:at|<@" src\kazusa_ai_chatbot src\adapters`
  - Expected after implementation: platform-native mention syntax appears only
    in adapter rendering or adapter normalization code and focused tests. It
    must not appear in cognition, dialog, dispatcher prompts,
    self-cognition tracking, or generic brain logic.
- `rg -n "delivery_mentions" src tests`
  - Expected after implementation: matches only in the approved change surface
    and focused tests.

### Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\models.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\self_cognition\tracking.py src\kazusa_ai_chatbot\self_cognition\handoff.py src\kazusa_ai_chatbot\dispatcher\adapter_iface.py src\kazusa_ai_chatbot\dispatcher\evaluator.py src\kazusa_ai_chatbot\dispatcher\handlers.py src\kazusa_ai_chatbot\dispatcher\remote_adapter.py src\kazusa_ai_chatbot\brain_service\runtime_adapters.py src\adapters\napcat_qq_adapter.py src\adapters\discord_adapter.py`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_delivery_mentions.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py -q`
- `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`
- `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_message_envelope.py tests\test_adapter_envelope_normalizers.py tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_self_cognition_event_logging.py -q`

## Independent Plan Review

Before approval or execution, review this plan for:

- Architecture alignment: self-cognition owns proactive target semantics,
  dispatcher owns metadata pass-through, adapters own native rendering and
  no-op feasibility.
- Scope correction: no group-broadcast capability is used as design
  justification.
- LLM discipline: no prompt changes, dialog flags, token placement, repair
  calls, or capability-detection LLM calls remain.
- Adapter boundary: brain-owned code does not check adapter support or emit
  adapter-native syntax.
- Compatibility: old action candidates, old scheduled sends, old adapter calls,
  and private sends remain valid.
- Verification completeness: tests cover group target metadata, private no-op,
  idempotency stability, LLM-visible projection non-leakage, missing platform
  id, dispatcher pass-through, QQ rendering, Discord rendering,
  unsupported/incomplete adapter no-op, and legacy no-metadata sends.

Blockers found during this review must be fixed in the plan before execution.

## Independent Code Review

Before completion, request or perform an independent code review that checks:

- Relevance, cognition, and dialog behavior were not changed for mention
  decisions.
- Group broadcast was not introduced.
- Brain-owned text contains no native platform mention syntax.
- Brain-owned code does not branch on adapter mention capability.
- Dispatcher and scheduler pass optional `delivery_mentions` without making
  them required.
- Delivery-only metadata does not enter LLM-visible self-cognition projection
  or idempotency keys.
- The adapter is the only layer that emits QQ or Discord native mention syntax.
- Prefix placement is deterministic; no adapter natural-language parsing or
  heuristic placement exists.
- Unsupported or incomplete mention requests do not block text delivery.
- Sends without `delivery_mentions` remain backward compatible.
- Scheduler rows without `delivery_mentions` still load and execute.
- Conversation-history rows do not store adapter-native mention syntax.
- Verification commands and static grep expectations are current.

The review must be recorded in `Execution Evidence` before this plan can move
to `completed`.

## Acceptance Criteria

- Self-cognition active-commitment target scope carries optional
  `platform_user_id` and `display_name` when available from the latest selected
  conversation row.
- Delivery-only target metadata does not change duplicate suppression and is
  not rendered into LLM-visible self-cognition context.
- Self-cognition proactive group action candidates for one semantic target user
  carry one prefix `delivery_mentions` request.
- Private action candidates and group-noise rejected cases do not carry
  delivery mention requests.
- Dispatcher, scheduler-compatible task args, remote adapter bridge, and
  adapter interface preserve optional `delivery_mentions`.
- QQ and Discord adapters render native prefix user mentions when feasible.
- Adapters send plain text when a mention request is unsupported or incomplete.
- Brain-owned code does not check adapter mention support.
- Brain-owned text and conversation-history rows do not contain adapter-native
  mention syntax.
- Sends without delivery mentions remain unchanged.
- Existing scheduled send tasks without `delivery_mentions` remain valid.
- All verification gates pass.
- Independent code review approves the final implementation.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Mention delivery expands into live-chat policy | Scope the plan to self-cognition action candidates only. | Static grep and code review confirm no relevance/dialog changes. |
| Brain starts depending on adapter capability | Forbid capability checks in brain-owned code. | Code review checks dispatcher/brain paths for support branching. |
| Delivery metadata leaks into model context | Mark delivery identity as non-model context and test projection omission. | Projection tests and static review. |
| Duplicate suppression changes unexpectedly | Keep delivery-only fields out of idempotency identity. | Idempotency stability tests. |
| Native mention syntax leaks into prompts/history | Keep syntax generation inside adapters and store brain text unchanged. | Static grep and conversation-history tests. |
| Incomplete platform identity blocks sends | Adapter no-ops unsupported/incomplete mentions and sends text. | Adapter no-op tests. |
| Scheduler compatibility regresses | Keep `delivery_mentions` optional inside existing task args. | Scheduler pass-through and legacy no-metadata tests. |

## Execution Evidence

Implementation has started.

- 2026-05-14 execution start:
  - Removed the obsolete `experiments/target_mention_token_reliability.py`
    script and the temporary `experiments/.gitignore` allow-list exception
    before implementation, per user request.
- 2026-05-14 Stage 1 evidence:
  - Added self-cognition tests for active-commitment target platform identity,
    group candidate `delivery_mentions`, private no-op, idempotency stability,
    and LLM-visible projection non-leakage.
  - Red verification:
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`
    failed as expected on missing `target_scope.platform_user_id` and missing
    action-candidate `delivery_mentions`.
  - Implemented optional self-cognition target metadata and deterministic
    group delivery-mention candidate construction.
  - Green verification:
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`
    passed with 26 tests.
- 2026-05-14 Stage 2 evidence:
  - Added `tests/test_delivery_mentions.py` for the public `DeliveryMention`
    shape, construction with missing `platform_user_id`, private no-op, and
    missing semantic-target no-op.
  - Red verification:
    `venv\Scripts\python -m pytest tests\test_delivery_mentions.py -q`
    failed as expected because `models.DeliveryMention` was missing.
  - Implemented `models.DeliveryMention`.
  - Green verification:
    `venv\Scripts\python -m pytest tests\test_delivery_mentions.py -q`
    passed with 4 tests.
- 2026-05-14 Stage 3 evidence:
  - Added dispatcher/scheduler tests for evaluator metadata preservation,
    omitted metadata, populated `delivery_mentions`, and empty-list
    `delivery_mentions` pass-through to the adapter without text mutation.
  - Added supporting self-cognition handoff coverage that action-candidate
    `delivery_mentions` reach the raw dispatcher args unchanged.
  - Red verification:
    `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`
    failed as expected because the dispatcher handler dropped populated and
    empty-list `delivery_mentions` before adapter delivery.
  - Red supporting verification:
    `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py::test_dispatch_action_candidate_preserves_delivery_mentions -q`
    failed as expected because `handoff.build_raw_tool_call` did not copy
    candidate `delivery_mentions`.
  - Implemented self-cognition handoff metadata preservation plus dispatcher
    schema, handler, adapter protocol, and remote adapter pass-through.
  - Green verification:
    `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`
    passed with 7 tests.
  - Green supporting verification:
    `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py::test_dispatch_action_candidate_preserves_delivery_mentions -q`
    passed with 1 test.
- 2026-05-14 Stage 4 evidence:
  - Added runtime adapter tests for remote proxy payload preservation, QQ
    prefix mention rendering, QQ incomplete-mention no-op, QQ callback
    endpoint metadata preservation, Discord prefix mention rendering, Discord
    incomplete-mention no-op, and Discord callback endpoint metadata
    preservation.
  - Red verification:
    `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`
    failed as expected on missing QQ/Discord `delivery_mentions` adapter
    signatures, missing callback request propagation, and missing native
    rendering.
  - Implemented QQ and Discord runtime request fields, endpoint pass-through,
    adapter optional `delivery_mentions` parameters, deterministic prefix
    rendering, and incomplete-mention no-op behavior.
  - Green verification:
    `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`
    passed with 36 tests.
- 2026-05-14 Stage 5 evidence:
  - Changed self-cognition files:
    `src/kazusa_ai_chatbot/self_cognition/models.py`,
    `src/kazusa_ai_chatbot/self_cognition/sources.py`,
    `src/kazusa_ai_chatbot/self_cognition/tracking.py`, and
    `src/kazusa_ai_chatbot/self_cognition/handoff.py`.
  - Green verification:
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_delivery_mentions.py -q`
    passed with 30 tests.
  - Supporting integration verification:
    `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py -q`
    passed with 13 tests after updating one stale expectation to include the
    new delivery-only target-scope fields.
- 2026-05-14 Stage 6 evidence:
  - Changed dispatcher/runtime pass-through files:
    `src/kazusa_ai_chatbot/dispatcher/adapter_iface.py`,
    `src/kazusa_ai_chatbot/dispatcher/handlers.py`, and
    `src/kazusa_ai_chatbot/dispatcher/remote_adapter.py`.
  - Green verification:
    `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`
    passed with 7 tests.
  - Runtime proxy payload preservation was covered by
    `tests/test_runtime_adapter_registration.py::test_remote_http_adapter_posts_send_message_payload`.
- 2026-05-14 Stage 7 evidence:
  - Changed adapter files:
    `src/adapters/napcat_qq_adapter.py` and
    `src/adapters/discord_adapter.py`.
  - QQ renders a native prefix `at` segment only for renderable group-send
    user mentions and otherwise sends the original text payload.
  - Discord renders a native prefix user mention only for renderable group-send
    user mentions and otherwise sends the original text.
  - Green verification:
    `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`
    passed with 36 tests.
- 2026-05-14 Stage 8 evidence:
  - Updated docs:
    `src/kazusa_ai_chatbot/self_cognition/README.md`,
    `src/kazusa_ai_chatbot/dispatcher/README.md`, and
    `src/kazusa_ai_chatbot/brain_service/README.md`.
  - Static grep:
    `rg -n "target_mention|\{target_mention\}|<target|__TARGET_MENTION__|\[\[target_mention\]\]" src tests`
    returned no matches.
  - Static grep:
    `rg -n "mention_target_user" src tests` returned no matches.
  - Static grep:
    `rg -n "CQ:at|<@" src\kazusa_ai_chatbot src\adapters`
    returned matches only in adapter-native normalization/rendering code:
    `src/adapters/discord_adapter.py` and
    `src/adapters/napcat_qq_adapter.py`.
  - Static grep:
    `rg -n "delivery_mentions" src tests` returned matches in the approved
    change surface and focused tests.
  - Compile:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\models.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\self_cognition\tracking.py src\kazusa_ai_chatbot\self_cognition\handoff.py src\kazusa_ai_chatbot\dispatcher\adapter_iface.py src\kazusa_ai_chatbot\dispatcher\evaluator.py src\kazusa_ai_chatbot\dispatcher\handlers.py src\kazusa_ai_chatbot\dispatcher\remote_adapter.py src\kazusa_ai_chatbot\brain_service\runtime_adapters.py src\adapters\napcat_qq_adapter.py src\adapters\discord_adapter.py`
    passed.
  - Focused test:
    `venv\Scripts\python -m pytest tests\test_delivery_mentions.py -q`
    passed with 4 tests.
  - Focused test:
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py -q`
    passed with 39 tests.
  - Focused test:
    `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`
    passed with 7 tests.
  - Focused test:
    `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`
    passed with 36 tests.
  - Regression test:
    `venv\Scripts\python -m pytest tests\test_message_envelope.py tests\test_adapter_envelope_normalizers.py tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_self_cognition_event_logging.py -q`
    passed with 60 tests.

- 2026-05-14 independent plan review for approval:
  - Reviewer mode: active agent fresh-review posture; no separate reviewer or
    subagent was used.
  - Inputs reviewed: plan contract, execution gates, cutover policy,
    development plan registry, local-LLM architecture rules, this plan, and
    targeted source/test checks for self-cognition, dispatcher, adapter, and
    runtime boundaries.
  - Blockers found before approval: delivery-only metadata needed explicit
    protection from LLM-visible projection, and duplicate-suppression
    idempotency stability needed explicit coverage.
  - Fixes made before approval: added mandatory rules, contract rules, change
    surface, implementation steps, review checks, acceptance criteria, and
    risks for projection non-leakage and idempotency stability.
  - Non-blocking findings: adapter rendering tests may live in
    `tests/test_runtime_adapter_registration.py` or a more focused adapter
    test file if existing local patterns make that cleaner.
  - Approval status: approved for execution.

- 2026-05-14 pre-execution plan review:
  - Required plan sections are present.
  - The accepted correction is recorded: this is a self-cognition targeted
    group-send delivery clarity fix, not a group-broadcast feature.
  - The previous dialog-generated `mention_target_user` design has been
    removed from the target architecture.
  - The brain/adapter ownership boundary is recorded: brain passes
    platform-neutral metadata and adapters handle native rendering or no-op
    behavior when a mention is not feasible.
  - No implementation has been performed from this draft.

- 2026-05-14 independent code review:
  - Reviewer mode: active-agent fresh-review posture; no separate reviewer or
    subagent was used.
  - Inputs reviewed: full implementation diff, this plan, style rules,
    dispatcher/scheduler compatibility, adapter boundaries, docs, static grep
    output, compile output, and focused/regression test output.
  - Finding 1: `handle_send_message` initially passed
    `delivery_mentions=None` to adapters even when the caller omitted the
    field. This broke existing test doubles and risked older adapter
    implementations that did not accept the new keyword.
  - Fix 1: changed the handler to include the `delivery_mentions` keyword only
    when the validated argument is a list. Omitted metadata now preserves the
    old call shape.
  - Finding 2: the self-cognition README briefly documented delivery-only
    platform identity as part of the stable tracking artifact `target_scope`.
    That conflicted with the implementation requirement that idempotency
    ignore delivery-only metadata.
  - Fix 2: corrected the README so delivery-only fields are documented only as
    source-case/action-candidate delivery metadata, not stable tracking
    identity.
  - Re-review status: approved. No unresolved findings remain.
  - Rerun verification after review fixes:
    `venv\Scripts\python -m pytest tests\test_dispatcher.py tests\test_dispatcher_event_logging.py -q`
    passed with 19 tests and 4 deselected.
  - Rerun verification after review fixes:
    `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`
    passed with 7 tests.
  - Rerun static verification after review fixes: compile passed, token greps
    returned no production matches, `mention_target_user` grep returned no
    matches, and native mention syntax remained limited to adapter files.

- 2026-05-14 final verification:
  - `venv\Scripts\python -m pytest tests\test_delivery_mentions.py -q`
    passed with 4 tests.
  - `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py -q`
    passed with 39 tests.
  - `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py -q`
    passed with 7 tests.
  - `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`
    passed with 36 tests.
  - `venv\Scripts\python -m pytest tests\test_message_envelope.py tests\test_adapter_envelope_normalizers.py tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_self_cognition_event_logging.py -q`
    passed with 60 tests.
  - `venv\Scripts\python -m pytest tests\test_dispatcher.py tests\test_dispatcher_event_logging.py -q`
    passed with 19 tests and 4 deselected.
  - Acceptance status: complete. The implementation keeps mention decisions out
    of dialog/relevance prompts, keeps adapter-native syntax inside adapters,
    preserves optional pass-through across dispatcher/scheduler/runtime
    bridges, and no-ops unsupported or incomplete adapter mention requests
    without blocking plain text delivery.
