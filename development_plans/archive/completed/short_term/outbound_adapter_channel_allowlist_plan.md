# outbound adapter channel allowlist plan

## Summary

- Goal: enforce adapter channel allowlists for brain-originated public platform
  sends while preserving private-message delivery.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, and `test-style-and-execution`
- Overall cutover strategy: bigbang for outbound public-channel permission;
  compatible for inbound listen-only behavior and existing HTTP wire shapes
- Highest-risk areas: changing inbound observation behavior, moving permission
  into brain/cognition code, returning success for rejected runtime sends,
  posting delivery receipts for suppressed chat responses, and blocking private
  messages by applying public-channel allowlists too broadly
- Acceptance criteria: Discord and NapCat block non-allowlisted public sends,
  private sends remain allowed, runtime sends reject with the existing 503 path,
  normal `/chat` responses to disallowed public targets are silently suppressed,
  no native platform send is observed for blocked paths, no brain-service files
  change, verification passes, and independent code review approves.

## Context

Current inbound behavior stays unchanged: listed Discord guild channels and QQ
groups are active, non-listed public targets are `listen_only`, and private
channels are active. The outbound gap is separate: runtime callback delivery
currently checks only platform reachability, so Discord can send to any
fetchable channel id and NapCat can send to any numeric group/private target.

The policy belongs in adapters because `channel_ids` is adapter deployment
configuration and platform delivery is adapter-owned. Brain service, cognition,
prompts, scheduler payloads, and dispatcher schemas stay unchanged.

## Adapter And ICD Surface

| Surface | Decision |
|---|---|
| `src/adapters/discord_adapter.py` | Change. Enforce configured guild-channel allowlist for outbound group/guild targets; preserve DMs. |
| `src/adapters/napcat_qq_adapter.py` | Change. Enforce configured group allowlist for outbound group targets; preserve private chats. |
| `src/adapters/debug_adapter.py` | Keep unchanged. It has no production platform allowlist surface in this plan. |
| `src/kazusa_ai_chatbot/dispatcher/remote_adapter.py` | Keep unchanged. It remains a platform-neutral proxy. |
| `/runtime/adapters/register` | No payload change. |
| `/send_message/capability` | No schema change. `available` now includes configured public-channel permission. |
| `/send_message` | No schema change. Disallowed public targets reject before native platform send. |
| `/chat` | No schema change. Inbound non-listed public messages may still be submitted as `listen_only`; adapter response rendering suppresses returned brain text before platform delivery when the public target is disallowed. |

Fixed outbound policy:

- Runtime callback capability: return `available=false` for disallowed public
  targets.
- Runtime callback send: `send_message(...)` raises `RuntimeError` before
  native platform send; the existing adapter endpoint returns HTTP 503.
- Normal `/chat` response delivery: silently suppress returned brain messages
  for the disallowed public target; do not call the native platform send API
  and do not post a delivery receipt.

## Mandatory Skills

- `development-plan-writing`: plan, registry, evidence, lifecycle changes.
- `local-llm-architecture`: ownership-boundary changes.
- `py-style`: Python edits.
- `test-style-and-execution`: test edits or execution.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Keep this plan outbound-only. Do not convert non-listed inbound public
  messages from `listen_only` to hard ignore.
- Do not change brain-service files/docs, prompts, RAG, cognition, dialog,
  scheduler payloads, persistence semantics, or runtime registration schemas.
- Public group/guild targets require membership in the adapter's configured
  `channel_ids` set. `channel_ids is None` and `channel_ids == []` both mean no
  public outbound channel is configured.
- Private targets are allowed regardless of `channel_ids`.
- `can_send_message(...)` must return `False` for disallowed public targets
  before platform fetch/API work.
- `send_message(...)` must raise `RuntimeError` before platform send for
  disallowed public targets.
- Normal `/chat` response rendering must not send returned brain messages to a
  disallowed public channel and must not post a delivery receipt.
- After context compaction or any stage sign-off, reread this entire plan
  before continuing.
- Before completion, lifecycle changes, merge, or sign-off, run
  `Independent Code Review` and record the result in `Execution Evidence`.

## Must Do

- Add adapter-owned allowlist checks for outbound Discord guild-channel sends.
- Add adapter-owned allowlist checks for outbound NapCat QQ group sends.
- Preserve private Discord DM and QQ private delivery.
- Apply the allowlist rule to runtime `can_send_message(...)`, runtime
  `send_message(...)`, and normal `/chat` response rendering.
- Keep dispatcher handler, remote adapter bridge, brain service, debug adapter,
  prompts, scheduler payloads, and persistence unchanged.
- Update `src/kazusa_ai_chatbot/dispatcher/README.md` and `docs/HOWTO.md`.
- Add deterministic tests for public rejection, private exemption, runtime
  `RuntimeError`, no native send, normal-response suppression, and no delivery
  receipt for suppressed normal responses.

## Deferred

- Do not reject inbound non-listed public-channel events before `/chat`.
- Do not remove or reinterpret `debug_modes.listen_only`.
- Do not add brain-service allowlists, database allowlists, config, runtime
  registration fields, scheduler fields, `/chat` fields, audit events,
  metrics, fallback paths, shims, shared helper modules, or broad refactors.
- Do not change delivery receipt semantics except that blocked/suppressed
  public sends must not be reported as successful platform deliveries.
- Do not change `delivery_mentions` behavior except where a send is blocked by
  the approved public-channel allowlist.

## Cutover Policy

Overall strategy: bigbang for outbound public-channel permission; compatible
for inbound observation and existing wire shapes.

| Area | Policy | Instruction |
|---|---|---|
| Discord runtime public sends | bigbang | Capability returns unavailable and send raises before native platform send. |
| NapCat runtime public sends | bigbang | Capability returns unavailable and send raises before native platform send. |
| Normal `/chat` response delivery | bigbang | Suppress platform send and delivery receipt for disallowed public targets. |
| Private sends | compatible | Preserve current private delivery; no `channel_ids` membership required. |
| Inbound `/chat` forwarding | compatible | Preserve current `listen_only` forwarding. |
| Runtime HTTP contract | compatible | Preserve request/response schemas. |
| Dispatcher | compatible | Preserve existing capability-first rejection path. |

## Cutover Policy Enforcement

- Do not preserve the old permissive public-channel send behavior behind a
  flag, fallback, compatibility path, or alternate adapter path.
- Compatible areas preserve only the surfaces explicitly listed above.
- Any cutover-policy change requires user approval before implementation.

## Overdesign Guardrail

- Actual problem: adapters can currently send brain-originated public messages
  to non-listed Discord guild channels or QQ groups.
- Minimal change: add adapter-local outbound allowlist checks that block
  non-listed public targets while preserving private sends.
- Ownership boundaries: adapters own configured channel permission and platform
  sends; dispatcher keeps existing capability-first rejection; brain service
  and LLM stages own none of this decision.
- Rejected complexity: no brain allowlist, new config format, database table,
  runtime registration field, scheduler migration, prompt/pipeline change,
  shared helper module, fallback path, or broad adapter refactor.
- Evidence threshold: centralize policy only after a future approved
  multi-adapter permission-management requirement or production incident proves
  adapter-local config is insufficient.

## Agent Autonomy Boundaries

- Add an adapter-local private method named `_outbound_channel_allowed(...)` in
  each changed platform adapter unless an existing same-adapter method already
  implements the exact target predicate.
- Do not create a shared module, shared helper, new package, or cross-adapter
  abstraction.
- Do not edit dispatcher, remote adapter, brain service, debug adapter, prompt,
  RAG, cognition, scheduler, or persistence code. If the plan and code conflict
  there, stop and report the blocker.
- Do not change inbound active/listen-only classification.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

All brain-originated platform sends pass this adapter-owned predicate before
platform delivery:

```text
channel_type == "private" -> allowed
channel_type == "group" and channel_id in configured channel_ids -> allowed
all other cases -> blocked before platform send
```

For Discord, `group` means guild channel and `private` means DM channel. For
NapCat QQ, `group` means group id and `private` means user/private chat id.

Dispatcher scheduled/proactive sends continue to use:

```text
dispatcher -> adapter.can_send_message(...)
  -> false for disallowed public target
  -> AdapterChannelUnavailableError
  -> no write-ahead assistant conversation row
```

If a runtime `send_message(...)` call reaches the adapter for a disallowed
public target, the adapter raises `RuntimeError` before native platform send.
Normal chat response rendering suppresses silently and posts no receipt.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Ownership | Enforce allowlists inside each platform adapter. | `channel_ids` is adapter deployment config and platform delivery is adapter-owned. |
| Brain service | Keep unchanged. | The user explicitly rejected brain-service change; no schema is needed. |
| Dispatcher | Keep unchanged. | It already fails closed when capability returns false before write-ahead persistence. |
| Private channels | Always allow private targets. | Private-message channels must not be affected. |
| `channel_ids=None` | Treat as no public outbound channel configured. | Matches existing listen-only public behavior when no active channels are configured. |
| Normal chat responses | Suppress silently. | There is no separate outbound request to return an error to; the platform must not observe a send. |

## Change Surface

### Modify

- `src/adapters/discord_adapter.py`: add `_outbound_channel_allowed(...)` and
  apply it before runtime capability fetch, runtime send, normal response send,
  and delivery receipt post.
- `src/adapters/napcat_qq_adapter.py`: add `_outbound_channel_allowed(...)` and
  apply it before runtime capability, runtime send, normal response send, and
  delivery receipt post.
- `tests/test_runtime_adapter_registration.py`: add focused tests for both
  adapters covering public rejection, private exemption, runtime `RuntimeError`,
  no native send, normal-response suppression, and no receipt.
- `src/kazusa_ai_chatbot/dispatcher/README.md`: clarify capability includes
  target permission, not only transport reachability.
- `docs/HOWTO.md`: document outbound allowlist behavior for Discord and NapCat.
- `development_plans/README.md`: keep the active plan row accurate.

### Keep

- `src/kazusa_ai_chatbot/dispatcher/handlers.py`
- `src/kazusa_ai_chatbot/dispatcher/remote_adapter.py`
- `src/kazusa_ai_chatbot/brain_service/**`
- `src/adapters/debug_adapter.py`
- `src/kazusa_ai_chatbot/nodes/**`
- `src/kazusa_ai_chatbot/rag/**`
- `src/kazusa_ai_chatbot/action_spec/**`

## Implementation Order

1. Add failing Discord runtime tests in
   `tests/test_runtime_adapter_registration.py` for unavailable capability,
   `RuntimeError`, no native send, and private exemption.
2. Implement Discord `_outbound_channel_allowed(...)` and apply it in runtime
   capability/send paths. Run the focused Discord tests.
3. Add failing NapCat runtime tests for unavailable capability, `RuntimeError`,
   no `send_msg`, and private exemption.
4. Implement NapCat `_outbound_channel_allowed(...)` and apply it in runtime
   capability/send paths. Run the focused NapCat tests.
5. Add and implement normal `/chat` response suppression tests for Discord and
   NapCat. Assert inbound non-listed public events still reach `/chat` as
   `listen_only`, no platform send occurs, and no delivery receipt is posted.
6. Update dispatcher README and HOWTO. Do not edit brain-service docs.
7. Run every command in `Verification`, then run `Independent Code Review`.

## Progress Checklist

- [x] Stage 1 - Discord runtime guard
  - Covers: steps 1-2.
  - Verify/evidence: focused Discord tests fail before and pass after.
  - Handoff: next agent starts Stage 2.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 2 - NapCat runtime guard
  - Covers: steps 3-4.
  - Verify/evidence: focused NapCat tests fail before and pass after.
  - Handoff: next agent starts Stage 3.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 3 - normal chat response guard
  - Covers: step 5.
  - Verify/evidence: suppression tests pass, preserve `listen_only`, and prove
    no receipt.
  - Handoff: next agent starts Stage 4.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 4 - docs
  - Covers: step 6.
  - Verify/evidence: non-brain docs mention outbound allowlist, private
    exemption, and unchanged inbound listen-only behavior.
  - Handoff: next agent starts Stage 5.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 5 - final verification
  - Covers: step 7 before review.
  - Verify/evidence: all `Verification` commands pass.
  - Handoff: next agent starts Stage 6.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 6 - independent code review
  - Verify/evidence: review findings, fixes, reruns, residual risks, and
    approval recorded; no blockers remain.
  - Handoff: plan can move toward completion after this stage only.
  - Sign-off: `Codex/2026-05-19`.

## Verification

- `rg -n "channel_ids" src\kazusa_ai_chatbot src\adapters`
  - Expected: new production allowlist logic appears only in platform adapters;
    no cognition, RAG, dialog, scheduler, or brain-service runtime logic.
- `rg -n "can_send_message|send_message/capability|AdapterChannelUnavailableError" src\kazusa_ai_chatbot\dispatcher src\kazusa_ai_chatbot\brain_service src\adapters`
  - Expected: dispatcher still performs capability-first rejection; runtime
    callback endpoints still delegate capability to adapters.
- `git diff --name-only -- src\kazusa_ai_chatbot\brain_service`
  - Expected: no output.

- `venv\Scripts\python -m py_compile src\adapters\discord_adapter.py src\adapters\napcat_qq_adapter.py`
- `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`
- `venv\Scripts\python -m pytest tests\test_dispatcher_send_message_result.py tests\test_dispatcher_event_logging.py tests\test_scheduler_future_promise.py -q`
- `venv\Scripts\python -m pytest tests\test_adapter_envelope_normalizers.py tests\test_service_input_queue.py -q`

## Independent Plan Review

- outbound-only scope, no inbound hard-ignore behavior;
- adapters own permission, dispatcher and brain service remain unchanged;
- private targets bypass public channel lists;
- no schema/config/runtime registration changes;
- runtime capability, runtime send, normal response send, and delivery receipt
  behavior are all specified;
- verification covers public rejection, private exemption, runtime
  `RuntimeError`, no platform send, no receipt, and unchanged `listen_only`.

### Review Result - 2026-05-19

- Reviewer mode: active agent fresh-review pass after reading the plan-writing
  skill, plan contract, execution gates, cutover policy, registry, source
  ownership boundaries, existing test filenames, placeholder grep, and user
  policy decisions.
- Blockers fixed: the earlier `large` classification was corrected because it
  was caused by plan verbosity, not implementation scope. The plan was
  compressed and reclassified as `medium`.
- Blockers fixed: `Overdesign Guardrail` is before `Agent Autonomy Boundaries`;
  helper naming is fixed; shared abstractions are forbidden; dispatcher and
  inbound classification are explicitly unchanged.
- Approval status: approved for execution.

## Independent Code Review

Before completion, an independent reviewer must inspect the full implementation
diff and record findings in `Execution Evidence`.

Review must check:

- adapter permission is enforced before platform fetch/API send;
- private sends are unaffected;
- dispatcher capability-first rejection remains before write-ahead persistence;
- no brain-service, cognition, RAG, prompt, scheduler, persistence, config,
  schema, migration, fallback, or shared helper surface is added;
- normal suppression preserves inbound `listen_only` and posts no receipt;
- docs, ICD surface, verification, and grep expectations match the diff.

Findings outside the approved change surface require a plan update or user
approval before code changes.

## Acceptance Criteria

- Discord and NapCat runtime capability return unavailable for non-listed public
  targets and available for private targets.
- Discord runtime send raises `RuntimeError` before native delivery for
  non-listed guild-channel targets and still sends private targets.
- NapCat runtime send raises `RuntimeError` before `send_msg` for non-listed
  group targets and still sends private targets.
- Normal `/chat` response rendering performs no platform send and posts no
  delivery receipt for non-listed public channels even if the brain returns
  messages.
- Inbound non-listed public-channel events still reach `/chat` as
  `listen_only`.
- Dispatcher scheduled/proactive sends to disallowed public targets fail
  through adapter capability before write-ahead assistant row persistence.
- Dispatcher README and HOWTO document the outbound allowlist contract and
  private exemption.
- All verification gates pass.
- Independent code review approves the final implementation.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Private sends blocked | Predicate allows `channel_type == "private"` before list checks. | Private capability and send tests for both adapters. |
| Inbound observation changes | Inbound classification is unchanged. | Normal-response tests assert `listen_only`. |
| Dispatcher persists before rejection | Capability returns unavailable for disallowed public targets. | Dispatcher regression tests and review. |
| Brain or LLM owns permission | Brain-service changes are forbidden. | Static grep and code review. |
| Runtime HTTP compatibility regresses | Preserve schemas and use existing 503 path. | Runtime adapter tests. |

## Execution Evidence

- 2026-05-19 independent plan review: no runtime implementation performed.
  Inputs included plan contract, execution gates, cutover policy, registry,
  source ownership boundaries, test-file existence, placeholder grep, and
  line-count/scope consistency.
- 2026-05-19 fixes before approval: plan compressed/reclassified as `medium`;
  guardrail ordering fixed; helper/dispatcher autonomy narrowed; inbound
  classification preservation clarified; registry status is `approved`.
- 2026-05-19 approval: independent plan review has no unresolved blockers.
- 2026-05-19 execution started: plan status changed to `in_progress`.
- 2026-05-19 Stage 1 red: `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py::test_discord_runtime_capability_rejects_unlisted_group tests\test_runtime_adapter_registration.py::test_discord_runtime_send_rejects_unlisted_group tests\test_runtime_adapter_registration.py::test_discord_runtime_allows_unlisted_private_target -q` failed as expected: unlisted group capability returned `True`, unlisted group send did not raise, private exemption passed.
- 2026-05-19 Stage 1 green: same command passed after `src\adapters\discord_adapter.py` added `_outbound_channel_allowed(...)` to runtime capability/send paths.
- 2026-05-19 Stage 2 red: `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py::test_napcat_runtime_capability_rejects_unlisted_group tests\test_runtime_adapter_registration.py::test_napcat_runtime_send_rejects_unlisted_group tests\test_runtime_adapter_registration.py::test_napcat_runtime_allows_unlisted_private_target -q` failed as expected: unlisted group capability returned `True`, unlisted group send did not raise, private exemption passed.
- 2026-05-19 Stage 2 green: same command passed after `src\adapters\napcat_qq_adapter.py` added `_outbound_channel_allowed(...)` to runtime capability/send paths.
- 2026-05-19 Stage 3 red: `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py::test_discord_suppresses_normal_response_for_unlisted_group tests\test_runtime_adapter_registration.py::test_napcat_suppresses_normal_response_for_unlisted_group -q` failed as expected: both adapters sent returned brain messages and posted delivery receipts for non-listed public targets.
- 2026-05-19 Stage 3 green: same command passed after both adapters suppressed normal `/chat` responses for disallowed public targets before platform send and delivery receipt post.
- 2026-05-19 Stage 4 docs: updated `docs\HOWTO.md` and `src\kazusa_ai_chatbot\dispatcher\README.md`. Verified with `rg -n "Outbound brain-originated|non-listed guild|non-listed groups|capability means both|not-configured targets|private chats remain|DMs remain" docs\HOWTO.md src\kazusa_ai_chatbot\dispatcher\README.md`.
- 2026-05-19 Stage 5 verification:
  - `rg -n "channel_ids" src\kazusa_ai_chatbot src\adapters` returned only adapter matches.
  - `rg -n "can_send_message|send_message/capability|AdapterChannelUnavailableError" src\kazusa_ai_chatbot\dispatcher src\kazusa_ai_chatbot\brain_service src\adapters` confirmed dispatcher and runtime adapter capability boundaries remained unchanged except adapter implementations/docs.
  - `git diff --name-only -- src\kazusa_ai_chatbot\brain_service` returned no output.
  - `venv\Scripts\python -m py_compile src\adapters\discord_adapter.py src\adapters\napcat_qq_adapter.py` passed.
  - `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q` passed: 50 passed.
  - `venv\Scripts\python -m pytest tests\test_dispatcher_send_message_result.py tests\test_dispatcher_event_logging.py tests\test_scheduler_future_promise.py -q` passed: 14 passed.
  - `venv\Scripts\python -m pytest tests\test_adapter_envelope_normalizers.py tests\test_service_input_queue.py -q` passed: 34 passed.
- 2026-05-19 Stage 6 independent code review: inspected the full diff,
  adapter inventory, dispatcher capability path, remote proxy, debug adapter,
  and brain-service diff. Findings: no blockers and no fixes required.
  Confirmed permission checks run before platform fetch/API send, private
  targets bypass public allowlists, normal suppression preserves inbound
  `listen_only` and posts no receipt, debug adapter remains out of scope, and
  no brain-service/runtime schema/config/shared-helper surface was added.
  Approval status: approved.
- 2026-05-19 lifecycle: execution evidence complete; plan moved to completed
  history under `development_plans/archive/completed/short_term/`.

## Plan Self-Review

- Coverage: every `Must Do` item maps to implementation order and verification.
- Minimality: only adapter-owned outbound permission plus focused docs/tests
  are in scope.
- Placeholder scan: no unresolved choices remain for reject vs discard, helper
  naming, shared abstraction, dispatcher edits, or inbound behavior.
- Contract consistency: `channel_ids`, `can_send_message(...)`,
  `send_message(...)`, `/send_message/capability`, private/group naming,
  `RuntimeError`, HTTP 503, and no-receipt behavior are consistent.
- Verification: focused tests cover public rejection, private exemption,
  runtime `RuntimeError`, normal-response suppression with no delivery receipt,
  compile, docs, static greps, and dispatcher regressions.
