# logical dialog message receipt plan

## Summary

- Goal: Persist each logical dialog message as its own assistant row while sharing the same cognition trace, and route delivery receipts by logical message index.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan`, `py-style`, `test-style-and-execution`, `local-llm-architecture`
- Overall cutover strategy: bigbang
- Highest-risk areas: delivery receipt API contract, assistant row persistence shape, past-dialog residual duplicate projection.
- Acceptance criteria: split dialog rows are persisted and receipted per logical message, residual lookup dedupes shared traces, focused tests pass.

## Context

Normal dialog now returns ordered `ChatResponse.messages`, but assistant
persistence still collapses the list into one newline-joined conversation row.
That prevents a reply to the second logical message from resolving to its exact
visible row and shared cognition trace.

The accepted design is intentionally small: multiple assistant rows may share
one `llm_trace_id`. No new source id, mapping table, platform-chunk contract,
prompt input, or LLM decision is added.

## Mandatory Skills

- `development-plan`: governs this execution record and verification evidence.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: preserve deterministic ownership and avoid prompt changes.

## Mandatory Rules

- Keep adapter/platform behavior out of brain prompts.
- Deterministic code owns persistence, receipt routing, delivery metadata, and residual dedupe.
- Do not add a new source id or platform chunk mapping.
- Do not change dialog, cognition, L2 prompts, RAG public evidence, consolidation, scheduler, or reflection behavior.
- Use focused deterministic tests for this plumbing change.
- After any automatic context compaction, reread this entire plan before continuing.
- After signing off any major checklist stage, reread this entire plan before starting the next stage.
- Run independent code review before completion unless the user explicitly continues the no-subagent fallback.

## Must Do

- Persist one assistant `conversation_history` row per non-empty `final_dialog` item.
- Store the same `delivery_tracking_id` and `llm_trace_id` on those rows.
- Store `logical_message_index` on split assistant rows.
- Add `logical_message_index` to normal chat delivery receipts and receipt matching.
- Post one receipt per logical message from normal chat adapters when a platform id is available.
- Group past-dialog cognition candidates by `llm_trace_id` before projection to avoid duplicate private residuals.
- Update the brain service, adapter, database, and past-dialog ICD text to match the new logical-message contract.

## Deferred

- Do not support adapter-created platform chunk ids beyond the first platform id for each logical message.
- Do not add source ids, mapping collections, alias tables, feature flags, or compatibility modes.
- Do not backfill historical joined assistant rows.
- Do not add live LLM tests; this change is deterministic plumbing.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Assistant persistence | bigbang | Replace newline-joined assistant-row persistence with one row per logical dialog item. |
| Delivery receipts | bigbang | Route receipts by `delivery_tracking_id` plus `logical_message_index`. |
| Historical rows | compatible | Existing joined rows remain readable; no backfill or dual write. |
| Platform chunks | deferred | Keep chunk-level reply mapping out of scope. |

## Target State

One cognition run can produce multiple logical messages. Each logical message is
persisted as one assistant row with exact `body_text`, shared `llm_trace_id`,
shared `delivery_tracking_id`, and its zero-based `logical_message_index`.
Adapters report receipts per logical message. Reply lookup by platform message
id finds the exact row, then residual lookup uses the shared trace.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Source pointer | Reuse `llm_trace_id` | It already names the cognition trace that produced all split messages. |
| Receipt routing | Add `logical_message_index` | Multiple rows share one tracking id before platform ids exist. |
| Residual projection | Dedupe by trace id | RAG may retrieve multiple rows from the same cognition. |
| Platform chunks | Defer | User explicitly does not value this edge case. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/brain_service/post_turn.py`: split assistant row writes.
- `src/kazusa_ai_chatbot/brain_service/outbound.py`: store `logical_message_index`.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`: add receipt index field.
- `src/kazusa_ai_chatbot/db/conversation.py`: match receipts by logical index.
- `src/kazusa_ai_chatbot/db/schemas.py`: document the new row field.
- `src/kazusa_ai_chatbot/service.py`: pass receipt index into DB helper.
- `src/adapters/delivery_receipts.py`: include receipt index in payloads.
- `src/adapters/discord_adapter.py`: post receipts for each logical message.
- `src/adapters/napcat_qq_adapter/ws_adapter.py`: post receipts for each logical message.
- `src/kazusa_ai_chatbot/past_dialog_cognition/runtime.py`: group residual projection by trace.
- ICD files for brain service, adapters, DB, and past-dialog cognition.
- Focused deterministic tests under `tests/`.

## Overdesign Guardrail

- Actual problem: split logical dialog messages cannot each receive a platform id and exact residual source row.
- Minimal change: split persistence rows, add logical index receipt routing, dedupe shared trace projection.
- Ownership boundaries: dialog owns ordered text; adapters own send timing; database owns row updates; residual code owns trace grouping.
- Rejected complexity: new source ids, mapping tables, chunk aliases, prompt inputs, LLM decisions, backfill, compatibility shims.
- Evidence threshold: add chunk-level mapping only after a real platform-chunk reply failure is accepted as product scope.

## Agent Autonomy Boundaries

- The implementation agent may adjust local signatures and tests only to satisfy this contract.
- The implementation agent must not introduce new architecture, alternate routing, prompt behavior, or speculative extension points.
- Changes outside the listed surface require stopping and reporting the reason.
- Existing uncommitted user work must be preserved.

## Implementation Order

1. Add or update focused tests for split persistence, receipt matching, adapter receipt payloads, and residual dedupe.
2. Implement the production changes in the listed modules.
3. Update ICD docs to match the contract.
4. Run focused tests, then adjacent deterministic tests.
5. Record evidence in this plan.

## Execution Model

- The user previously requested execution without subagents; this plan uses explicit fallback single-agent execution.
- The active agent owns tests, production code, verification, evidence, and final reporting.
- Independent code review is documented as skipped only if the no-subagent instruction remains active.

## Progress Checklist

- [x] Stage 1 - focused test contract
  - Files: `tests/test_bot_side_addressing.py`, `tests/test_db.py`, adapter receipt tests, past-dialog cognition tests.
  - Verify: targeted tests show old behavior gaps before production patch where practical.
  - Sign-off: Codex / 2026-07-01.
- [x] Stage 2 - production implementation
  - Files: production change surface listed above.
  - Verify: focused tests pass.
  - Sign-off: Codex / 2026-07-01.
- [x] Stage 3 - documentation and regression
  - Files: ICD docs listed above.
  - Verify: adjacent deterministic tests pass.
  - Sign-off: Codex / 2026-07-01.
- [x] Stage 4 - review and completion
  - Review: no-subagent fallback self-review unless user changes execution mode.
  - Verify: final git diff is scoped.
  - Sign-off: Codex / 2026-07-01.

## Verification

- `venv\Scripts\python -m pytest tests\test_bot_side_addressing.py tests\test_db.py tests\test_runtime_adapter_registration.py tests\test_past_dialog_cognition_context.py -q`
- `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py::test_delivery_receipt_endpoint_returns_updated_and_not_found -q`
- `venv\Scripts\python -m pytest tests\test_adapter_outbound_sequence.py -q`
- Static grep: `rg -n "first successfully sent platform message id|first logical outbound message|one-id receipt|joined" src\kazusa_ai_chatbot\brain_service\README.md src\adapters\README.md src\kazusa_ai_chatbot\past_dialog_cognition\README.md`

## Independent Code Review

Independent code review is normally required. Because the user requested
execution without subagents, the active agent performs a focused self-review of
the final diff against this plan and records residual risk in evidence.

## Acceptance Criteria

This plan is complete when split assistant rows preserve exact visible text,
delivery receipts update the intended logical row, RAG residual projection
dedupes repeated shared traces, adapters report logical-message receipts, and
the focused deterministic verification passes.

## Execution Evidence

- Baseline focused test run before production patch:
  `venv\Scripts\python -m pytest tests\test_bot_side_addressing.py tests\test_db.py::test_apply_assistant_delivery_receipt_updates_tracking_row tests\test_runtime_adapter_registration.py::test_napcat_handle_event_sends_brain_messages_as_sequence_with_first_reply_only tests\test_runtime_adapter_registration.py::test_discord_handle_message_sends_brain_messages_as_sequence_with_first_reply_only tests\test_past_dialog_cognition_context.py::test_candidates_sharing_trace_project_one_residual -q`
  failed as expected: missing `logical_message_index`, joined assistant row
  body text, old receipt signature, missing follow-up receipts, and duplicate
  shared-trace residual projection.
- Focused verification after implementation:
  `venv\Scripts\python -m pytest tests\test_bot_side_addressing.py tests\test_db.py tests\test_runtime_adapter_registration.py tests\test_past_dialog_cognition_context.py tests\test_service_background_consolidation.py::test_delivery_receipt_endpoint_returns_updated_and_not_found tests\test_adapter_outbound_sequence.py -q`
  passed with `155 passed, 13 deselected`.
- Endpoint check:
  `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py::test_delivery_receipt_endpoint_returns_updated_and_not_found -q`
  passed.
- Adapter delay helper check:
  `venv\Scripts\python -m pytest tests\test_adapter_outbound_sequence.py -q`
  passed with `4 passed`.
- Static grep:
  `rg -n "first successfully sent platform message id|first logical outbound message|one-id receipt|joined" src\kazusa_ai_chatbot\brain_service\README.md src\adapters\README.md src\kazusa_ai_chatbot\past_dialog_cognition\README.md`
  returned no matches.
- Standard non-live regression:
  `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`
  passed with `2561 passed, 2 skipped, 367 deselected`.
- Review: no-subagent fallback self-review caught and fixed one QQ follow-up
  receipt-ordering bug before final verification. A later self-review moved
  shared-trace dedupe into candidate validation so duplicate split rows do not
  consume the unique residual cap.
- Post-refinement past-dialog focused check:
  `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_context.py -q`
  passed with `7 passed`.
- Post-refinement standard non-live regression:
  `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`
  passed with `2561 passed, 2 skipped, 367 deselected`.
- Residual risk is limited to intentionally deferred platform chunk reply ids.
