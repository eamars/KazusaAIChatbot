# dialog mention target user plan

## Summary

- Goal: Move outbound user-mention judgment into the universal dialog LLM output as `mention_target_user: bool`, while deterministic delivery code applies reply override and adapter rendering.
- Plan class: medium
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `cjk-safety`, `test-style-and-execution`, `superpowers:test-driven-development`, `superpowers:verification-before-completion`
- Overall cutover strategy: bigbang for brain decision ownership, compatible for adapter metadata rendering.
- Highest-risk areas: local LLM prompt reliability, normal chat response contract, self-cognition handoff metadata, adapter delivery behavior.
- Acceptance criteria: dialog emits a validated boolean, reply override suppresses effective mentions, normal and self-cognition paths share the same semantic decision source, deterministic positive/negative tests pass, and real LLM positive/negative tests are run one by one and inspected.

## Context

The current self-cognition mention implementation builds `delivery_mentions`
deterministically from self-cognition target scope. That violates the desired
universal ownership boundary: normal chat and self-cognition should both let
the dialog stage decide whether the outgoing utterance needs explicit target
addressing.

The current dialog prompt only asks for `final_dialog`, and normal chat
responses do not expose `delivery_mentions`. The current adapter runtime path
can render delivery mentions, but normal `/chat` adapter responses still send
plain text or reply anchors only.

The target architecture is:

```text
normal chat or self-cognition
  -> cognition decides stance and content
  -> dialog generates final_dialog and mention_target_user
  -> deterministic delivery mapper:
       if use_reply_feature: no effective mention
       elif mention_target_user: build delivery_mentions
       else: no delivery_mentions
  -> adapter renders native mention if feasible, otherwise sends plain text
```

## Mandatory Skills

- `local-llm-architecture`: load before editing dialog prompts, LLM payloads, or graph state.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files that contain CJK prompt text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `superpowers:test-driven-development`: write failing deterministic tests before production code.
- `superpowers:verification-before-completion`: run fresh verification before reporting completion.

## Mandatory Rules

- Do not pass platform user ids, global user ids, native mention syntax, target tag availability, or insertion locations to the dialog LLM for mention rendering.
- The dialog LLM may only output `mention_target_user: bool` as semantic addressing intent.
- `use_reply_feature=True` must suppress effective delivery mentions after the dialog LLM returns.
- The adapter remains responsible for native rendering or no-op fallback.
- Deterministic code owns validation, platform metadata construction, and reply override.
- Adapters own channel/platform feasibility checks and no-op fallback when native mention is not possible.
- Prompt changes must keep explicit generation guidance plus `# Input Format` and `# Output Format`; the format must match the actual handler payload.
- Real LLM tests must be run one test at a time with trace output inspected before running the next real LLM case.
- Do not add extra LLM calls, retry loops, compatibility shims, feature flags, or alternate addressing agents.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the active agent must run the Independent Code Review gate and record the result in Execution Evidence.

## Must Do

- Rewrite the dialog generator prompt so it requires top-level `final_dialog` and `mention_target_user`.
- Add deterministic tests for positive mention generation, negative mention generation, malformed/missing flag repair, and reply override.
- Add normal chat response delivery metadata when effective mention is true.
- Add self-cognition action-candidate delivery metadata only when dialog output has `mention_target_user=True`.
- Update normal QQ and Discord adapter send paths to render `delivery_mentions` from `/chat` responses using existing adapter-owned rendering helpers.
- Add real LLM positive and negative tests for the dialog prompt contract and run them individually.
- Update relevant docs and plan registry entries.

## Deferred

- Do not add an addressing-only LLM.
- Do not make relevance decide mention behavior.
- Do not let dialog generate native mention syntax, platform ids, global ids, target insertion tokens, or tag placement.
- Do not redesign group broadcast semantics or multi-target cognition.
- Do not redesign the existing reply-feature relevance contract.
- Do not change scheduler dispatcher tool exposure beyond preserving current runtime-only `delivery_mentions` behavior.

## Overdesign Guardrail

- Actual problem: Unthreaded group sends from either normal chat or self-cognition can look unanchored when the dialog is personally aimed at the current user.
- Minimal change: Add one dialog-owned semantic boolean, `mention_target_user`, and deterministically suppress effective mention when `use_reply_feature=True`.
- Ownership boundaries: dialog LLM owns only semantic addressing intent; service/self-cognition mapping owns validation and delivery metadata construction; adapters own native mention rendering, channel feasibility, and no-op fallback.
- Rejected complexity: no `delivery_context` prompt input, no `channel_type` prompt input, no `use_reply_feature` prompt input, no `single_target_user` prompt input, no target insertion tokens, no mention placement choices, no adapter capability flags, no second mention LLM, no retry or repair loop.
- Evidence threshold: add any rejected complexity only after a failing deterministic or live LLM test demonstrates the minimal semantic flag plus deterministic reply override cannot satisfy a current approved behavior.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Brain mention decision | bigbang | Replace deterministic self-cognition mention decision with dialog `mention_target_user`. |
| Dialog prompt output shape | bigbang | Require `mention_target_user` with `final_dialog`; missing or invalid values normalize to false and log a contract warning. |
| Normal chat API | compatible | Add optional `delivery_mentions` to `ChatResponse`; clients that ignore it keep working. |
| Adapter rendering | compatible | Existing adapter send behavior remains; adapters render `delivery_mentions` only when present and feasible. |

## Agent Autonomy Boundaries

- Keep the change surface limited to dialog output, delivery metadata mapping, normal adapter response handling, self-cognition handoff mapping, tests, and docs.
- Do not add helper abstractions unless they remove real repeated validation or encode a stable domain concept.
- If a test exposes unrelated failures, report them instead of expanding scope.
- If the prompt cannot reliably produce `mention_target_user` in live tests, stop and report the evidence before inventing a fallback architecture.

## Target State

Dialog output has this contract:

```json
{
  "final_dialog": ["text fragment"],
  "mention_target_user": true
}
```

Effective delivery mention uses this deterministic rule:

```python
effective_mention_target_user = (
    bool(final_dialog)
    and bool(mention_target_user)
    and not use_reply_feature
)
```

When effective mention is true, deterministic code builds:

```json
[
  {
    "entity_kind": "user",
    "placement": "prefix",
    "platform_user_id": "<current platform user id>",
    "global_user_id": "<current global user id>",
    "display_name": "<current display name>",
    "requested_by": "dialog.mention_target_user"
  }
]
```

The dialog prompt does not receive delivery context, channel type, reply
feature state, target ids, adapter capability, or mention placement.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| LLM output field | Use `mention_target_user: bool`. | This is the smallest universal semantic contract and matches the user's requested interface. |
| User identity in prompt | Do not add target ids, mention-rendering names, channel type, reply state, or delivery context. Keep existing `user_name` only as ordinary dialog voice context. | Each cognition iteration has one current user; ids and delivery metadata only add leakage and ownership risk. |
| Reply conflict | Apply deterministic reply override after LLM output. | Reply anchoring is an upstream delivery decision and should win without relying on local LLM compliance. |
| Adapter fallback | Adapter no-ops when native mention is infeasible. | Adapter owns platform capability and rendering safety. |
| Real LLM validation | Use positive and negative live prompt tests. | The changed field is generated by a high-temperature local dialog LLM. |

## LLM Call And Context Budget

- Before: dialog generator has one response-path LLM call at temperature 0.65 and returns only `final_dialog`.
- After: same single response-path LLM call and same model configuration; output adds one boolean.
- Added context: none. The prompt contract changes output shape only.
- No new response-path LLM call is allowed.
- No retry loop beyond the existing dialog evaluator loop is added.
- Context budget risk is low; the prompt changes are structural output guidance, not large examples.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: prompt, payload, parsing, state output.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: propagate `mention_target_user`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: state typing for `mention_target_user`.
- `src/kazusa_ai_chatbot/state.py`: top-level state typing for `mention_target_user`.
- `src/kazusa_ai_chatbot/service.py`: compute effective delivery mentions for normal chat responses.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`: add optional `delivery_mentions` to `ChatResponse`.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`: preserve dialog mention decision in dialog state and action candidate.
- `src/kazusa_ai_chatbot/self_cognition/tracking.py`: stop deterministic self-cognition mention unless dialog requested it.
- `src/adapters/napcat_qq_adapter.py`: render response `delivery_mentions` for normal chat group sends.
- `src/adapters/discord_adapter.py`: render response `delivery_mentions` for normal chat group sends.
- Relevant docs under `src/kazusa_ai_chatbot/**/README.md`.
- Relevant deterministic and live LLM test files.

### Keep

- Existing dispatcher runtime `delivery_mentions` contract and adapter no-op safety.
- Existing relevance-owned `use_reply_feature` decision.
- Existing self-cognition target-scope platform metadata source.

## Implementation Order

1. Update the active plan and registry.
2. Add deterministic failing tests for dialog parser/prompt payload and service reply override.
3. Add deterministic failing tests for self-cognition action candidate using dialog `mention_target_user`.
4. Add deterministic failing tests for normal adapter response rendering from `delivery_mentions`.
5. Implement dialog prompt, parser, and state propagation.
6. Implement delivery mention mapping in normal service and self-cognition.
7. Implement normal adapter response rendering.
8. Add live LLM positive and negative dialog prompt tests with trace output.
9. Run focused deterministic tests.
10. Run live LLM tests one at a time and inspect trace output.
11. Run static checks and independent code review.

## Progress Checklist

- [x] Stage 1 - plan and registry updated
  - Verify: plan exists under `development_plans/active/short_term/` and registry lists it as `in_progress`.
  - Evidence: record `git diff -- development_plans` summary.
  - Sign-off: Codex / 2026-05-14 after confirming the active plan file exists and the registry row points to it.
- [x] Stage 2 - deterministic tests written and observed failing
  - Verify: focused pytest commands fail for missing `mention_target_user` or missing delivery metadata.
  - Evidence: record each failing command and failure reason.
  - Sign-off: Codex / 2026-05-14 after focused red tests failed for the missing dialog flag, missing `ChatResponse.delivery_mentions`, self-cognition still adding deterministic mentions, and adapters ignoring brain-provided mention metadata.
- [x] Stage 3 - production code implemented
  - Verify: focused deterministic tests from Stage 2 pass.
  - Evidence: record commands and changed files.
  - Sign-off: Codex / 2026-05-14 after focused dialog, service, self-cognition, delivery, and adapter tests passed with the minimal semantic-flag contract.
- [x] Stage 4 - live LLM prompt validation added and inspected
  - Verify: run positive and negative live LLM tests one at a time with `-q -s`.
  - Evidence: record trace paths and judgment for each case.
  - Sign-off: Codex / 2026-05-14 after both live LLM cases were run individually and their traces inspected.
- [x] Stage 5 - regression and static verification complete
  - Verify: run focused regression tests, `py_compile`, static greps, and `git diff --check`.
  - Evidence: record command outputs.
  - Sign-off: Codex / 2026-05-14 after deterministic suites, syntax checks, forbidden-token grep, approved-field grep, and diff whitespace check completed.
- [x] Stage 6 - independent code review complete
  - Verify: review full diff against this plan and project rules.
  - Evidence: record findings, fixes, rerun commands, and residual risks.
  - Sign-off: Codex / 2026-05-14 after fresh self-review, follow-up fixes,
    dispatcher/scheduler verification, and final static boundary checks.

## Verification

### Deterministic Tests

- `venv\Scripts\python -m pytest tests\test_dialog_mention_target_user.py -q`
- `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_delivery_mentions.py -q`
- `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`
- `venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_rag_dialog_event_logging.py tests\test_conversation_progress_history_policy.py -q`

### Real LLM Tests

Run each test individually and inspect the trace after each command:

- `venv\Scripts\python -m pytest -m live_llm tests\test_dialog_mention_target_user_live_llm.py::test_live_dialog_mentions_target_for_unanchored_group_self_cognition -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_dialog_mention_target_user_live_llm.py::test_live_dialog_does_not_mention_for_general_group_remark -q -s`

### Static Checks

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\brain_service\contracts.py src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\self_cognition\tracking.py src\adapters\napcat_qq_adapter.py src\adapters\discord_adapter.py`
- `rg -n "target_mention|\{target_mention\}|<target|__TARGET_MENTION__|\[\[target_mention\]\]" src tests` returns no matches; exit code 1 is acceptable.
- `rg -n "mention_target_user" src tests` returns only the approved schema, prompt, parser, mapping, docs, and tests.
- `git diff --check`

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
If no separate reviewer is available, the active agent must reread this plan,
inspect the full diff from a fresh-review posture, and record that no separate
reviewer was available.

Review scope:

- Project rules and style compliance for changed Python, tests, prompts, docs, and commands.
- Prompt contract clarity, CJK safety, JSON format consistency, and local LLM fragility.
- Alignment with `Must Do`, `Deferred`, `Change Surface`, delivery ownership, and reply override.
- Regression risk in normal chat, self-cognition, scheduler dispatcher, and adapter rendering.
- Live LLM trace quality and whether pass/fail assertions match the intended semantic contract.

Fix concrete findings directly only when the fix is inside this plan's change
surface. If a fix changes the contract or expands scope, stop and update the
plan before editing code.

## Acceptance Criteria

This plan is complete when:

- Dialog output includes a validated `mention_target_user` boolean.
- Missing or malformed `mention_target_user` normalizes to false with contract telemetry.
- Effective delivery mentions are suppressed whenever `use_reply_feature=True`.
- Normal chat and self-cognition both use dialog output as the semantic source of mention intent.
- The brain never asks the dialog LLM to generate native tags, ids, mention-rendering names, insertion tokens, or placement.
- Adapter rendering is best-effort and platform-owned.
- Deterministic tests, real LLM positive/negative tests, static checks, and independent code review pass or have documented blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Local dialog LLM omits or corrupts boolean at high temperature | Strict prompt, parser validation, missing/invalid normalizes false | Deterministic malformed-output tests and live LLM tests |
| Reply and mention both happen | Deterministic reply override after dialog | Service test where reply true suppresses delivery mentions |
| Normal adapter ignores delivery metadata | Wire existing rendering helpers into `/chat` response path | Adapter response-rendering tests |
| Self-cognition remains deterministic | Remove unconditional target-scope mention construction | Self-cognition false/true mention tests |

## Execution Evidence

- Stage 1: Created `development_plans/active/short_term/dialog_mention_target_user_plan.md` and updated `development_plans/README.md` active short-term registry row. `git diff -- development_plans\README.md development_plans\active\short_term\dialog_mention_target_user_plan.md` shows the new plan and registry entry; the README diff also includes pre-existing archive registry changes for unrelated completed plans.
- Stage 2: Added red tests in `tests/test_dialog_mention_target_user.py`, `tests/test_delivery_mentions.py`, `tests/test_self_cognition_tracking.py`, `tests/test_service_background_consolidation.py`, `tests/test_persona_supervisor2.py`, and `tests/test_runtime_adapter_registration.py`. Focused red commands failed as expected: `tests\test_dialog_mention_target_user.py -q` failed with missing `mention_target_user`; self-cognition focused tests failed because `build_action_candidate` lacked the keyword and old deterministic mention metadata was still present; service tests failed because `ChatResponse` had no `delivery_mentions`; adapter tests failed because normal response sends ignored `delivery_mentions`.
- Stage 3: Implemented dialog `mention_target_user` parsing/output, removed rejected `delivery_context` prompt payload, added normal `/chat` `delivery_mentions`, propagated self-cognition dialog mention intent into action candidates, and wired normal QQ/Discord adapter response rendering. Focused deterministic commands passed: `venv\Scripts\python -m pytest tests\test_dialog_mention_target_user.py -q` (3 passed); `venv\Scripts\python -m pytest tests\test_delivery_mentions.py -q` (5 passed); `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_group_action_candidate_omits_delivery_mention_without_dialog_flag tests\test_self_cognition_tracking.py::test_group_action_candidate_carries_dialog_delivery_mention_request tests\test_self_cognition_tracking.py::test_private_action_candidate_keeps_dialog_mention_request_for_adapter_noop tests\test_self_cognition_tracking.py::test_dialog_false_mention_flag_suppresses_group_action_delivery_mention tests\test_self_cognition_tracking.py::test_dialog_true_mention_flag_builds_group_action_delivery_mention -q` (5 passed); `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py::test_chat_response_adds_delivery_mentions_from_dialog_flag_without_channel_gate tests\test_service_background_consolidation.py::test_chat_response_reply_feature_suppresses_delivery_mentions -q` (2 passed).
- Stage 4: Live LLM positive case passed and trace was inspected: `venv\Scripts\python -m pytest -m live_llm tests\test_dialog_mention_target_user_live_llm.py::test_live_dialog_mentions_target_for_unanchored_group_self_cognition -q -s`; trace `test_artifacts\llm_traces\dialog_mention_target_user_live_llm__unanchored_group_self_cognition__20260514T112342274197Z.json` shows a direct overdue-challenge nudge with `mention_target_user=true` and no delivery context. Live LLM negative case passed and trace was inspected: `venv\Scripts\python -m pytest -m live_llm tests\test_dialog_mention_target_user_live_llm.py::test_live_dialog_does_not_mention_for_general_group_remark -q -s`; trace `test_artifacts\llm_traces\dialog_mention_target_user_live_llm__general_group_remark.json` shows a general group remark with `mention_target_user=false` and no delivery context.
- Stage 5: Regression and static gates passed: `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py -q` (21 passed); `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_delivery_mentions.py -q` (47 passed); `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q` (40 passed); `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py::test_evaluator_preserves_delivery_mentions_metadata tests\test_self_cognition_integration.py::test_dispatch_action_candidate_preserves_delivery_mentions -q` (2 passed); `venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_rag_dialog_event_logging.py tests\test_conversation_progress_history_policy.py -q` (18 passed); `venv\Scripts\python -m py_compile ...` over the plan Python files and live tests passed; forbidden mention-token grep returned no matches; `rg -n 'mention_target_user' src tests` returned only approved schema, prompt, parser, mapping, docs, and tests; `git diff --check` passed with CRLF warnings only.
- Stage 6: No separate reviewer was available in this session, so Codex
  performed the required fresh self-review against the full diff, this plan,
  and the project rules. Review finding 1: leftover `use_reply_feature` state
  was still being threaded into dialog-adjacent internal state after the prompt
  no longer used deterministic delivery inputs; fixed by removing those stale
  fields from persona and self-cognition dialog state setup. Review finding 2:
  a service test name/comment still implied a channel gate; fixed to document
  the adapter-owned no-op behavior. Review confirmed the dispatcher keeps
  `delivery_mentions` runtime-only by hiding it from the task-dispatcher
  LLM-visible args schema. Post-review reruns passed: `venv\Scripts\python -m
  pytest tests\test_persona_supervisor2.py tests\test_rag_dialog_event_logging.py
  tests\test_conversation_progress_history_policy.py -q` (18 passed);
  `venv\Scripts\python -m pytest
  tests\test_self_cognition_tracking.py::test_group_action_candidate_omits_delivery_mention_without_dialog_flag
  tests\test_self_cognition_tracking.py::test_group_action_candidate_carries_dialog_delivery_mention_request
  tests\test_service_background_consolidation.py::test_chat_response_adds_delivery_mentions_from_dialog_flag_without_channel_gate
  tests\test_service_background_consolidation.py::test_chat_response_reply_feature_suppresses_delivery_mentions -q`
  (4 passed); `venv\Scripts\python -m pytest tests\test_dispatcher.py
  tests\test_scheduler_future_promise.py -q` (20 passed, 4 deselected);
  `venv\Scripts\python -m py_compile ...` over changed source and test files
  passed; forbidden target-token grep returned no matches; dialog prompt/live
  LLM test grep found no deterministic delivery inputs; `git diff --check`
  passed with CRLF warnings only. Residual risk: the local high-temperature LLM
  can still choose the wrong boolean in novel phrasing, but malformed output
  fails closed to `false`, live positive/negative traces passed, and adapters
  own final delivery feasibility.
