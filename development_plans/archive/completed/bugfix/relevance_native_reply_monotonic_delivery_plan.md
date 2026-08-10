# relevance native reply monotonic delivery plan

## Summary

- Goal: make the final `/chat` native-reply decision monotonic while allowing
  deterministic group promotion for an assembled response or a response that
  has exceeded the two-minute delay threshold.
- Status: completed.
- Scope boundary: settled `/chat` response construction, its deterministic
  tests, and the related brain/relevance ICD wording.
- Change direction: replace the current obsolete-owner suppression at the
  final service boundary with an additive `True` promotion that reuses the
  existing graph latch and response-owner data.
- Acceptance state: implementation, verification, independent review, and
  lifecycle closeout are recorded below.

This plan supersedes the behavior described by the completed
`relevance_native_reply_anchor_guard_bugfix_plan.md`. The completed plan stays
archived as historical evidence and is not modified.

## Confirmed Decisions

- `use_reply_feature` remains a deterministic delivery flag. No new LLM call,
  prompt field, semantic anchor decision, or semantic check is added.
- The existing `keep_true` reducer remains the monotonic state mechanism.
  Once the graph state contains `True`, final visible response construction
  must not turn it back to `False`.
- New delivery conditions are additive. They can promote `False` to `True`
  and never demote `True`.
- The existing response-owner/effective-latest comparison is the mechanical
  intervening-fragment signal. No cross-author turn merge or semantic
  re-association is introduced.
- Promotion applies only to group `/chat` responses with a non-empty current
  response-owner platform message ID.
- Delay promotion uses the existing monotonic enqueue timestamp and the
  current monotonic clock at final response construction. The threshold is
  strictly greater than `120.0` seconds.
- If promotion inputs are unavailable, the existing latched flag passes
  through unchanged.
- Existing adapter behavior remains responsible for rendering the first
  native reply. No `reply_anchor_message_id` field is added to `ChatResponse`.
- Self-cognition and background result delivery remain outside this change.

## Scope And Change Direction

The graph already carries `use_reply_feature` through the `keep_true` reducer.
The service currently applies an additional conjunction requiring the response
owner to be the latest effective fragment. That conjunction can erase an
existing `True` decision. The service will instead preserve the graph latch
and add a deterministic group promotion when the existing owner/fragment
state or elapsed-time condition qualifies.

The implementation remains inside the brain-service delivery boundary:

```text
settled relevance Boolean
    -> existing keep_true graph latch
    -> deterministic group/owner/time promotion
    -> ChatResponse.use_reply_feature
    -> existing adapter first-message native reply
```

The semantic relevance contract, turn ownership, response futures, public
response schema, and adapter interfaces remain stable.

## Mandatory Skills

- `development-plan`: govern this plan's lifecycle, approval, execution,
  verification, review, and closeout.
- `local-llm-architecture`: preserve LLM/deterministic ownership boundaries;
  no operational delivery facts enter the relevance prompt.
- `py-style`: govern every production Python and test edit.
- `test-style-and-execution`: govern deterministic regression tests and their
  execution through `venv\Scripts\python`.

## Mandatory Rules

- Keep this plan in `draft` until the user approves it. Production changes
  require both an authorized lifecycle status and an explicit implementation
  command.
- Preserve the existing `keep_true` reducer in
  `src/kazusa_ai_chatbot/state.py`; do not create a second monotonic latch.
- Keep the LLM relevance output and graph state semantically unchanged.
- Keep all delivery promotion deterministic, bounded, and inspectable.
- Do not add compatibility shims, parallel reply vocabularies, feature flags,
  adapter fallbacks, or unrelated cleanup.
- Use the project virtual environment for Python and pytest commands.
- Do not read `.env`.
- Do not modify the archived predecessor plan.

## Must Do

1. Replace the final `use_reply_feature` calculation in
   `service._process_queued_chat_item` so an existing `True` survives owner
   mismatch and all new conditions only OR in `True`.
2. Reuse the existing response-owner sequence, latest settlement fragment,
   platform message ID, and monotonic enqueue timestamp.
3. Define the deterministic promotion contract exactly as:

   ```text
   base_reply = graph_result.use_reply_feature

   owner_mismatch = settlement_fragments exist and
                    latest_fragment.arrival_sequence != item.sequence

   promotion = channel_type == group
               and response_owner_platform_message_id is non-empty
               and (owner_mismatch or elapsed_seconds > 120.0)

   final_reply = base_reply or promotion
   ```

   `final_reply` is exposed as `True` only when a visible final dialog exists;
   an empty response has no outbound native reply to render.
4. Replace the obsolete-owner test expectation with the monotonic contract.
5. Add deterministic coverage for owner mismatch promotion, delay promotion,
   the strict 120-second boundary, existing `True` preservation, private
   scope, missing message ID, no-promotion conditions, and single-fragment
   behavior.
6. Update the brain-service and relevance README contracts to state that
   relevance supplies the base semantic request while deterministic service
   delivery may promote it and never erases `True`.
7. Run focused and affected verification, inspect the final diff, and record
   evidence in this plan before lifecycle closure.

## Deferred

- Adding `reply_anchor_message_id` or any other public `ChatResponse` field.
- Selecting an arbitrary historical message as a reply target.
- Cross-author group-turn joining, semantic retargeting, or LLM ownership
  checks for an interleaving message.
- Changes to `turn_settlement.py`, adapter implementations, dispatcher
  contracts, self-cognition candidates, background delivery, persistence, or
  control-console projections.
- Rewriting the relevance prompt, relevance schema, or live-LLM behavior.
- Platform capability discovery, message-deletion checks, or a new fallback
  delivery path.

## Target State

For a visible `/chat` response, the final flag satisfies this invariant:

```text
graph_result.use_reply_feature == True
    => ChatResponse.use_reply_feature == True
```

When the graph value is `False`, deterministic promotion may set the final
flag to `True` only for a group response with a valid response-owner platform
message ID and either an existing effective-latest mismatch or elapsed time
strictly greater than 120 seconds.

When the group scope, message ID, timing, or settlement information is absent,
the service does not invent a new value and returns the graph latch unchanged.

The adapter continues applying the flag only to the first outbound message,
as documented by the existing brain-service and adapter contracts.

## Contracts And Data Shapes

No public schema changes are permitted.

Existing internal inputs are authoritative:

| Input | Owner | Use |
| --- | --- | --- |
| `result["use_reply_feature"]` | graph/relevance latch | Base Boolean, already monotonic through `keep_true` |
| `settlement_fragments` | service/turn settlement | Existing latest-fragment comparison |
| `item.sequence` | service/response owner | Identifies the request receiving the response future |
| `item.request.platform_message_id` | adapter intake/service | Valid platform message support for new promotion |
| `item.enqueue_monotonic` and `time.monotonic()` | queue/service | Elapsed response-delay comparison |
| `final_dialog` | dialog/service | Whether a native reply can actually be rendered |

## Change Surface

### Delete

- No files or public interfaces are deleted.

### Modify

- `src/kazusa_ai_chatbot/service.py`: replace the non-monotonic final guard
  with the fixed additive calculation in the target-state contract.
- `tests/test_service_input_queue.py`: replace obsolete-owner suppression and
  add deterministic promotion/boundary regressions.
- `src/kazusa_ai_chatbot/brain_service/README.md`: document final service
  monotonicity and unchanged adapter ownership.
- `src/kazusa_ai_chatbot/relevance/README.md`: document the base semantic
  Boolean and deterministic additive delivery promotion.
- `development_plans/README.md`: register this draft under active bugfix
  plans.

### Create

- No new production or test modules.

### Keep

- `src/kazusa_ai_chatbot/state.py`: existing `keep_true` reducer and graph
  field annotation.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`: existing `ChatResponse`
  schema.
- `src/kazusa_ai_chatbot/brain_service/turn_settlement.py`: existing owner,
  fragment, and settlement lifecycle.
- Discord and NapCat adapter native-reply rendering.
- Self-cognition and background delivery behavior.
- `development_plans/archive/completed/bugfix/relevance_native_reply_anchor_guard_bugfix_plan.md`.

## Cutover Policy

Overall strategy: bigbang for the final service flag behavior and its tests;
compatible for the public response and adapter interfaces.

| Area | Policy | Instruction |
| --- | --- | --- |
| Service finalization | bigbang | Remove the obsolete-owner demotion directly. |
| Graph state | compatible | Retain the existing `keep_true` field and reducer. |
| Public API | compatible | Preserve `ChatResponse.use_reply_feature` without new fields. |
| Adapter delivery | compatible | Preserve first-message native-reply behavior. |
| Tests | bigbang | Replace assertions for removed demotion behavior. |
| Documentation | bigbang | Align active ICD wording with the new ownership contract. |

## Agent Autonomy Boundaries

The implementation owner may choose local helper names, expression layout,
test fixture setup, and command order while preserving this plan's formula,
scope, and file surface.

The implementation owner must not change semantic authority, add public fields,
move logic into adapters, join different-author turns, alter self/background
delivery, or introduce compatibility behavior. A conflict with the fixed
contract requires a plan amendment before proceeding.

## Verification

Use the project virtual environment.

Focused deterministic verification must cover:

- `tests/test_state.py`
- the native-reply and response-owner tests in
  `tests/test_service_input_queue.py`
- normal adapter first-message behavior in
  `tests/test_runtime_adapter_registration.py`

Affected regression verification must include the service background suite and
the relevant service/queue tests. Compile the changed production Python file,
run `git diff --check`, and finish with `git status --short`.

No live-LLM test is required because this plan changes no prompt, model route,
semantic schema, or LLM call count.

## Acceptance Criteria

The plan is complete when:

- `keep_true` remains the sole graph monotonic mechanism;
- a visible response with graph `use_reply_feature=True` returns `True` even
  when the response owner is older than the latest assembled fragment;
- group owner-mismatch and over-120-second cases promote `False` to `True`;
- the exact 120-second boundary does not promote;
- private, missing-ID, and unsupported-information cases preserve the base
  flag without a new override;
- single-fragment native reply behavior remains covered;
- adapter first-message rendering remains unchanged;
- self-cognition/background behavior remains unchanged;
- focused and affected tests pass;
- documentation, diff, status, and independent review evidence contain no
  unresolved in-scope finding.

## Progress Checklist

- [x] Draft plan reviewed against the archived predecessor and current code.
- [x] User approves this plan and explicitly authorizes implementation.
- [x] Service and test changes implemented within the locked surface.
- [x] ICD documentation updated.
- [x] Focused and affected verification passed.
- [x] Independent code review completed and findings resolved.
- [x] Plan lifecycle evidence recorded and status closed.

## Independent Plan Review

Before approval, the parent reviews this draft against the current
`state.py`, `service.py`, `turn_settlement.py`, adapter contracts, focused
tests, and the archived predecessor plan. The review must confirm that the
existing `keep_true` mechanism is reused, the final service boundary cannot
demote `True`, and no semantic, public-schema, adapter, or self/background
scope has entered the plan.

## Execution Evidence

- 2026-08-05: Draft created after repository inspection and user decisions.
- 2026-08-05: User authorized implementation; the plan moved to `in_progress`.
- 2026-08-05: DeepSeek implementation handoff completed within the bounded
  four-file ownership slice. The service reuses `keep_true`, preserves an
  existing `True`, and adds group owner-mismatch or strictly-over-120-second
  deterministic promotion; tests and ICD wording were updated.
- 2026-08-05: Parent verification passed with `venv\Scripts\python`:
  `py_compile` for the changed Python files; focused suite
  (`tests/test_state.py`, `tests/test_service_input_queue.py`, and
  `tests/test_runtime_adapter_registration.py`) with 156 passed; affected
  service/queue regression batch with 111 passed; and `git diff --check`
  clean apart from normal CRLF warnings.
- 2026-08-05: Parent independent code review checked the approved formula,
  monotonic graph reducer reuse, visible-dialog gating, public schema and
  adapter stability, and excluded self-cognition/background boundaries. No
  in-scope findings remained.
- 2026-08-05: Lifecycle record closed and moved to the completed archive.
