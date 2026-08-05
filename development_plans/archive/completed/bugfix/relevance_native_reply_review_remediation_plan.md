# relevance native reply review remediation plan

## Summary

- Goal: close the actionable findings from the post-implementation review of
  deterministic native-reply delivery.
- Status: completed.
- Scope boundary: native-reply response construction, its deterministic
  regressions, the directly affected HOWTO wording, and this lifecycle record.
- Change direction: preserve the approved monotonic promotion formula while
  treating whitespace-only message IDs as unavailable support and expanding
  coverage for visible-dialog and combined-gate behavior.
- Acceptance state: remediation, verification, parent review, and lifecycle
  closeout are recorded below; no semantic, public-schema, adapter, settlement,
  or proactive-delivery redesign entered scope.

## Confirmed Decisions

- No LLM, semantic, or cross-author ownership check is added.
- `keep_true` remains the sole graph monotonic mechanism.
- A visible response keeps an existing `True`; an empty dialog exposes no
  native-reply flag because there is no outbound message to anchor.
- A whitespace-only platform message ID is treated as unavailable support,
  just like an empty ID.
- The process-local monotonic clock and coordinator fragment ordering remain
  existing bounded invariants; no persistence or ordering redesign is added.

## Scope And Change Direction

The completed monotonic-delivery plan already implements the required service
formula. Review identified only low-severity coverage gaps, an invalid-ID
robustness edge, and a HOWTO sentence that does not distinguish the relevance
base Boolean from the final delivery flag. This follow-up closes those items
without changing semantic ownership or the public response contract.

## Mandatory Skills

- `development-plan`: govern this remediation plan and lifecycle.
- `local-llm-architecture`: preserve deterministic delivery ownership and keep
  operational facts out of semantic stages.
- `py-style`: govern the service and test edits.
- `test-style-and-execution`: govern deterministic regression additions and
  execution.

## Mandatory Rules

- Keep the archived
  `relevance_native_reply_monotonic_delivery_plan.md` unchanged.
- Preserve the existing `keep_true` reducer and public `ChatResponse` schema.
- Keep promotion deterministic and additive; do not add semantic checks,
  compatibility paths, new fields, or speculative helpers.
- Treat missing, empty, and whitespace-only platform message IDs as absent
  support and pass through the existing base flag.
- Preserve unrelated concurrent workspace changes.
- Use `venv\Scripts\python` for compilation and tests; do not read `.env`.

## Must Do

1. Make the native-reply promotion gate require a non-whitespace platform
   message ID while retaining the existing group, owner-mismatch, delay, and
   visible-dialog formula.
2. Add deterministic tests for an empty promotion-eligible dialog, an empty
   dialog with a latched `True`, combined owner-mismatch plus delay promotion,
   delayed promotion with missing support, and whitespace-only IDs.
3. Qualify the HOWTO wording so it describes the relevance base flag while
   acknowledging final deterministic delivery promotion.
4. Run focused and affected deterministic verification, inspect the diff, and
   record the review disposition and residual invariant evidence here.

## Deferred

- Persisting monotonic timestamps across process restarts.
- Replacing the coordinator's established fragment ordering invariant.
- Adding `reply_anchor_message_id` or changing any public schema.
- Changes to relevance prompts, LLM calls, turn settlement, adapters,
  self-cognition, background work, persistence, or dispatcher behavior.
- New architecture, configuration, feature flags, or compatibility shims.

## Target State

For a visible `/chat` response:

```text
base_reply = graph_result.use_reply_feature
owner_mismatch = settlement_fragments exist and
                 latest_fragment.arrival_sequence != item.sequence
promotion = channel_type == group
            and platform_message_id.strip() is non-empty
            and (owner_mismatch or elapsed_seconds > 120.0)
final_reply = bool(final_dialog) and (base_reply or promotion)
```

The graph latch remains monotonic. Unsupported message-ID input cannot create
new native-reply delivery, and no visible-dialog response can expose a flag
without an outbound message.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/service.py`: treat whitespace-only platform IDs as
  unavailable at the deterministic promotion boundary.
- `tests/test_service_input_queue.py`: cover the review-identified empty,
  combined-gate, delayed-missing-ID, and whitespace-ID cases.
- `docs/HOWTO.md`: distinguish the relevance base Boolean from final delivery
  promotion.
- `development_plans/README.md`: register and close this remediation record.

### Create

- `development_plans/archive/completed/bugfix/relevance_native_reply_review_remediation_plan.md`:
  this execution record.

### Keep

- `development_plans/archive/completed/bugfix/relevance_native_reply_monotonic_delivery_plan.md`.
- `src/kazusa_ai_chatbot/state.py`,
  `src/kazusa_ai_chatbot/brain_service/turn_settlement.py`,
  `src/kazusa_ai_chatbot/brain_service/contracts.py`, adapters, and proactive
  delivery paths.

## Agent Autonomy Boundaries

The implementation owner may choose local test fixture arrangement and exact
wording while preserving the fixed gate and scope. The owner must not change
semantic authority, public fields, response ownership, settlement ordering,
or proactive delivery. Any such change requires a new plan and user decision.

## Verification

Use `venv\Scripts\python` for:

- `py_compile` on changed Python files;
- focused state, native-reply, and adapter tests;
- affected service/queue regression tests;
- `git diff --check` and final workspace status.

No live-LLM test is required because this remediation changes no prompt, model
route, semantic schema, or LLM call.

## Acceptance Criteria

- Empty visible-dialog cases expose `False` even when promotion or the graph
  latch would otherwise be `True`.
- Owner-mismatch and delay conditions promote together when support is valid.
- Delayed responses with empty or whitespace-only message IDs preserve the
  base flag without promotion.
- Existing `True` remains monotonic for visible responses.
- HOWTO wording distinguishes base relevance from final delivery behavior.
- Focused and affected deterministic tests, compilation, and diff checks pass.
- No changes occur in the archived predecessor plan or excluded paths.

## Progress Checklist

- [x] DeepSeek read-only review completed and findings classified.
- [x] Service and deterministic test remediation implemented.
- [x] HOWTO wording updated.
- [x] Verification and parent review completed.
- [x] Lifecycle evidence recorded and plan archived.

## Execution Evidence

- 2026-08-05: DeepSeek read-only review completed with no confirmed defects;
  actionable findings were empty-dialog coverage, combined-gate coverage,
  whitespace-only ID robustness, and HOWTO wording ambiguity. Process-local
  clock and established fragment ordering were confirmed as bounded residual
  invariants.
- 2026-08-05: User directed remediation of all surfaced issues; plan entered
  `in_progress` without changing the archived predecessor.
- 2026-08-05: Service promotion now requires a non-whitespace platform message
  ID; deterministic tests cover both empty-dialog base values, combined
  owner-mismatch and delay promotion, delayed missing-ID preservation, and
  whitespace-only ID preservation. HOWTO wording distinguishes the base
  relevance Boolean from final delivery promotion.
- 2026-08-05: Verification passed with `venv\Scripts\python`: changed-file
  compilation; native-reply focused cases (11 passed); focused state/service/
  adapter suite (159 passed); affected service/queue regression batch (114
  passed); and `git diff --check` clean apart from normal CRLF warnings.
- 2026-08-05: Parent review confirmed the fixed gate, visible-dialog behavior,
  monotonic latch reuse, excluded boundaries, and residual process-local clock
  and fragment-order invariants. No unresolved in-scope findings remained.
- 2026-08-05: Lifecycle record closed and moved to the completed archive.
