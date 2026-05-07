# cognition silence short circuit and dialog evaluator quality plan

## Summary

- Goal: De-assert `should_respond` when cognition explicitly chooses silence, short-circuit dialog output, and tighten dialog evaluator quality criteria against cognition anchors.
- Plan class: medium.
- Status: completed
- Mandatory skills: `py-style`, `test-style-and-execution`, `local-llm-architecture`.
- Overall cutover strategy: compatible behavior change with no database migration, no environment flag, and no adapter change.
- Highest-risk areas: preserving the `should_respond` ownership boundary, avoiding a second relevance gate inside dialog, and changing evaluator criteria without changing the generator/evaluator conversation protocol.
- Acceptance criteria: silent cognition produces `should_respond=false` and no dialog call; evaluator checks final dialog against all relevant anchors and rejects unanchored concrete content while preserving the existing `feedback` / `should_stop` protocol.

## Context

The current live response path has a clear service-level response gate:

```text
relevance -> if should_respond=false, stop before persona
```

That gate exists in `src/kazusa_ai_chatbot/brain_service/graph.py`.
However, after relevance has allowed a turn into persona processing, the persona
graph currently runs:

```text
decontextualizer -> RAG -> cognition -> dialog
```

without a second route after cognition. This means cognition can semantically
decide that the character should remain silent, but dialog can still be called
unless dialog-local `expression_willingness == "silent"` happens to skip it.
That skip is too late and too local: response eligibility belongs in the
process state through `should_respond`, not as a hidden dialog behavior.

The dialog evaluator also has a narrower quality-contract gap. The current
prompt already references `[DECISION]`, `[AVOID_REPEAT]`, `[PROGRESSION]`, and
`[SCOPE]`; the weak point is specifically that topic-drift detection is centered
on `[FACT]` and `[ANSWER]`, and there is no general rule that rejects concrete
entities, numbers, measurements, dates, promises, or technical claims invented
by `final_dialog` when no anchor authorizes them. The motivating production
incident was a fabricated model-number/measurement style reply, but tests and
prompt edits must target the general unsupported-concrete-content class, not a
literal incident string.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing response-path graph behavior, cognition/dialog prompts, or evaluator behavior.

## Mandatory Rules

- Check `git status --short` before editing.
- Use `apply_patch` for manual edits.
- Use `venv\Scripts\python` for Python test execution.
- Do not read `.env`.
- Do not modify unrelated dirty files.
- Preserve `should_respond` as the response-chain gate. Dialog must not become a second relevance agent.
- Do not parse raw user text to decide silence. The silence decision must come from cognition-owned structured or semi-structured outputs.
- Do not change the conversational protocol between dialog generator and evaluator:
  - keep evaluator output as `{"feedback": str, "should_stop": bool}`;
  - keep generator receiving evaluator feedback through the existing messages path;
  - do not add `approved`, `retryable`, new evaluator schema fields, new retry-loop stages, or extra LLM calls.
- Update evaluator criteria only: the evaluator prompt must inspect final dialog quality against anchors, not against a hard-coded incident pattern.
- Do not add a new prompt-stage call in the live response path.
- Do not change RAG, memory, consolidation, adapters, queue pruning, delivery receipts, or scheduler behavior.
- Keep the existing dialog-local `expression_willingness == "silent"` skip as defensive depth. It must no longer be the primary silence mechanism, and tests must prove persona does not call dialog for cognition silence.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.

## Must Do

- Propagate the initial `should_respond` value into the persona state.
- Add `should_respond` to `GlobalPersonaState` unconditionally.
- Add a cognition-after route in `persona_supervisor2` so explicit cognition silence de-asserts `should_respond` and bypasses dialog.
- Add a deterministic no-response node in the persona graph. The node must return:
  - `should_respond: false`
  - `final_dialog: []`
  - `target_addressed_user_ids: []`
  - `target_broadcast: false`
- Ensure `persona_supervisor2` returns `should_respond=false` to the service graph when cognition short-circuits.
- Verify the service worker sees `should_respond=false` and `final_dialog=[]`, so assistant persistence, progress recording, and consolidation are skipped by existing final-dialog checks.
- Update the dialog evaluator prompt criteria to inspect final dialog against the complete anchor set:
  - `[DECISION]` as the required main response action;
  - `[FACT]` as the only source of required concrete factual claims;
  - `[ANSWER]` as the required answer when present;
  - `[SOCIAL]` as a soft expression posture unless it changes the decision;
  - `[AVOID_REPEAT]` and `[PROGRESSION]` as move-level continuity requirements;
  - `[SCOPE]` as length and coverage guidance.
- Add evaluator prompt rules that concrete entities, models, numbers, measurements, dates, promises, or technical claims in `final_dialog` must be supported by `[FACT]`, `[ANSWER]`, or another explicit anchor. Unsupported concrete claims are a quality failure.
- Add deterministic tests proving cognition silence skips dialog.
- Add evaluator prompt-contract tests proving the criteria are anchor-based and not incident-specific.

## Deferred

- Do not redesign the relevance prompt or relevance output schema.
- Do not redesign cognition L1/L2/L3 agent responsibilities.
- Do not add a new `[NO_RESPONSE]` anchor tag in this plan.
- Do not change the generator prompt except where a test shows existing wording directly contradicts the updated evaluator criteria.
- Do not change evaluator output schema, retry protocol, max retry count, or generator feedback message shape.
- Do not remove the existing dialog-agent `expression_willingness == "silent"` skip in this plan. It remains a defensive fallback for direct `dialog_agent` callers and unexpected graph regressions.
- Do not add keyword classification over user input.
- Do not change background consolidation semantics beyond relying on existing `final_dialog=[]` skip behavior.
- Do not add database migrations, environment variables, or feature flags.

## Cutover Policy

| Area                             | Policy     | Notes                                                                                                         |
| -------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------- |
| Cognition silence short-circuit  | compatible | Existing non-silent turns continue to dialog. Silent turns now terminate earlier with `should_respond=false`. |
| Dialog evaluator prompt criteria | compatible | Same LLM call, same payload shape, same output schema, stricter anchor-based inspection.                      |
| Tests                            | compatible | Add focused deterministic tests and one mandatory live LLM evaluator diagnostic.                             |
| Data and adapters                | compatible | No persistence schema, queue, delivery, or adapter contract changes.                                          |

## Agent Autonomy Boundaries

- The agent may choose local helper names only when they preserve the contracts in this plan.
- The agent must not introduce new architecture, alternate response gates, compatibility layers, fallback paths, or extra features.
- The agent must treat changes outside `persona_supervisor2`, `dialog_agent`, and their tests as high-scrutiny changes.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, broad prompt rewrites, or refactors.
- If existing code already has an equivalent helper for no-response state construction or response routing, reuse or lightly extract it instead of duplicating behavior.
- If the plan and code disagree, preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible without changing the generator/evaluator protocol, stop and report the blocker instead of inventing a substitute.

## Target State

The response path must behave as:

```text
service graph
  relevance
    should_respond=false -> END
    should_respond=true  -> persona graph

persona graph
  decontextualizer -> RAG -> cognition
    cognition says remain silent -> should_respond=false -> no-response output -> END
    cognition does not say silent -> dialog generator/evaluator -> END
```

Dialog evaluator must remain a quality inspector. It must not decide whether the
system should respond to the user. It only evaluates whether the generated
dialog faithfully renders the cognition anchors that reached dialog.

## Design Decisions

| Topic                       | Decision                                                                                                     | Rationale                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| Response gate ownership     | `should_respond` remains the process-level gate.                                                             | This matches the existing graph contract and keeps dialog from becoming relevance.                                                 |
| Cognition silence trigger   | Use cognition-owned output, with `expression_willingness == "silent"` as the canonical deterministic signal. | This is already structured and avoids parsing raw user input.                                                                      |
| Persona short-circuit shape | Route to a named deterministic no-response node rather than directly to `END`.                               | `persona_supervisor2` expects dialog-shaped fields; the no-response node preserves output shape and gives tests a stable boundary. |
| Dialog-local silence skip   | Keep the existing dialog-agent skip as defensive depth.                                                      | Direct calls to `dialog_agent` should still degrade safely, but normal persona flow must short-circuit before dialog.              |
| Evaluator contract          | Preserve `feedback` and `should_stop`.                                                                       | User explicitly requested no generator/evaluator conversational contract change.                                                   |
| Evaluator criteria          | Tighten topic-drift and anchor-fidelity checks while preserving existing evaluator responsibilities.         | The existing prompt already mentions several anchor types; the required fix is narrower than a full evaluator redesign.            |
| Unsupported facts           | Treat unsupported concrete facts as quality failures.                                                        | Dialog should not invent concrete entities, numbers, or measurements absent from anchors.                                          |

## Contracts And Data Shapes

### Cognition Silence Contract

The canonical silence signal for this plan is:

```python
state["action_directives"]["contextual_directives"]["expression_willingness"] == "silent"
```

The implementation may normalize whitespace and case. It must not inspect raw
user text. It must not treat `minimal`, `reluctant`, `avoidant`, or
`withholding` as silence.

When this signal is present after cognition, persona output must include:

```python
{
    "should_respond": False,
    "final_dialog": [],
    "target_addressed_user_ids": [],
    "target_broadcast": False,
}
```

The persona graph implementation must use this shape:

```text
stage_2_cognition
  -> stage_3_no_response, when cognition requests silence
  -> stage_3_action, otherwise
```

The no-response node must be named `stage_3_no_response` and must be a real
graph node so tests can assert the stable route behavior. Do not implement this
as an inline lambda that hides the state update.

### Evaluator Protocol Contract

The dialog generator/evaluator exchange remains unchanged:

```python
{
    "feedback": "Passed or concrete correction",
    "should_stop": True | False,
}
```

The evaluator human payload shape remains the existing payload. The agent may
reuse existing fields already present in the evaluator payload, including
`final_dialog`, `linguistic_directives`, `contextual_directives`,
`internal_monologue`, and `last_user_message`. Do not add required evaluator
payload fields in this plan.

### Anchor Quality Criteria

The evaluator prompt must define these checks:

- `[DECISION]`: `final_dialog` must execute the main response action.
- `[FACT]`: required concrete facts must appear naturally; concrete facts not in anchors must be rejected when they become the reply's substance.
- `[ANSWER]`: if present, the user-facing answer must be preserved.
- `[SOCIAL]`: expression posture is soft unless it changes or contradicts `[DECISION]`.
- `[AVOID_REPEAT]`: repeated response moves are failures when no `[PROGRESSION]` satisfies the required movement.
- `[PROGRESSION]`: final dialog must show the required movement when present.
- `[SCOPE]`: length and coverage are checked after anchor fidelity.

### Unsupported Concrete Content

For this plan, concrete content means any generated claim that names or specifies
an inspectable object or commitment rather than merely expressing tone. Examples
include model names, weapon/equipment identifiers, version strings, distances,
counts, prices, dates, times, locations, promises, scheduled actions, factual
properties, or technical specifications.

The evaluator prompt must contain a rule whose subject is unauthorized concrete
content and whose verdict is rejection. It is not enough for tests to grep for
one incident token. Prompt-contract tests must prove the rule covers at least
these general examples:

- a fabricated model/measurement style claim, such as a reply inventing a model
  identifier and range when anchors provide no such fact;
- a fabricated date/time/promise style claim, such as a reply inventing a
  scheduled follow-up when anchors provide no such commitment.

## LLM Call And Context Budget

| Call             | Before                    | After                                                        | Response path | Context impact               | Latency impact         |
| ---------------- | -------------------------:| ------------------------------------------------------------:| ------------- | ---------------------------- | ---------------------- |
| Relevance        | 1                         | 1                                                            | yes           | unchanged                    | unchanged              |
| Cognition stack  | unchanged                 | unchanged                                                    | yes           | unchanged                    | unchanged              |
| Dialog generator | 0-N depending on retry    | unchanged for non-silent turns; 0 for cognition-silent turns | yes           | unchanged payload            | lower for silent turns |
| Dialog evaluator | 0-N paired with generator | unchanged for non-silent turns; 0 for cognition-silent turns | yes           | prompt criteria changed only | lower for silent turns |

No new LLM calls are allowed. Evaluator prompt size may increase slightly for
anchor criteria, but must stay under the existing route context budget. Use
plain prompt-render checks and existing deterministic tests to verify formatting.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Add a post-cognition route.
  - Add a deterministic `stage_3_no_response` graph node.
  - Preserve and return `should_respond`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Add `should_respond` on `GlobalPersonaState` unconditionally.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Update only evaluator prompt criteria.
  - Keep the existing `conditional_skip_dialog_agent` behavior as defensive fallback.
  - Do not change evaluator output schema or generator/evaluator message protocol.
- `tests/test_persona_supervisor2.py`
  - Add deterministic cognition-silence short-circuit tests.
- `tests/test_service_background_consolidation.py`
  - Add or update a graph/service test proving `should_respond=false` and `final_dialog=[]` skip downstream output work after persona returns no response.
- `tests/test_dialog_agent.py`
  - Add evaluator prompt-contract assertions for anchor-based quality criteria.

### Create

- `tests/test_dialog_evaluator_live_llm_contract.py`
  - Add one mandatory live LLM diagnostic for anchor-based evaluator behavior.
  - The live case must directly exercise an unsupported model/range-style
    concrete claim and prove the evaluator returns retry feedback rather than
    `Passed`.

### Keep

- `src/kazusa_ai_chatbot/brain_service/graph.py`
  - Keep relevance-stage short-circuit unchanged.
- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - No changes.
- RAG, memory, consolidation, adapters, queue, scheduler, and database modules.

### Delete

- No files are deleted in this plan.

## Implementation Order

1. Add focused persona tests before implementation.
   - In `tests/test_persona_supervisor2.py`, add a test where mocked cognition returns `expression_willingness="silent"` and `dialog_agent` is an `AsyncMock`.
   - Expected baseline failure: dialog is called or output shape lacks `should_respond=false`.
2. Add service-level short-circuit verification.
   - In `tests/test_service_background_consolidation.py`, add or adapt a graph/service test proving persona-level no-response yields `should_respond=false`, `final_dialog=[]`, and no assistant output work.
3. Add evaluator prompt-contract tests.
   - In `tests/test_dialog_agent.py`, assert the evaluator prompt requires anchor-wide inspection and unsupported concrete facts to be rejected.
   - Assertions must be generic: do not assert only one historical incident string. At minimum, test language must cover unauthorized model/measurement content and unauthorized date/time/promise content as rejection-worthy.
4. Implement persona short-circuit.
   - Preserve initial `should_respond` in persona state.
   - Add `_cognition_requests_silence` as a local helper.
   - Add a named no-response graph node.
   - Route after cognition to either no-response or dialog.
   - Return `should_respond` from `persona_supervisor2`.
5. Update evaluator prompt criteria only.
   - Keep existing prompt rendering pattern.
   - Keep evaluator payload and output schema unchanged.
   - Add criteria for all anchor categories and unsupported concrete claims.
6. Run focused tests and prompt render checks.
7. Run broader deterministic response-path tests.
8. Run one live evaluator diagnostic for sign-off. If the live endpoint is
   unavailable, report the blocked sign-off instead of marking this plan
   complete.

## Progress Checklist

- [x] Stage 1 - tests define cognition silence short-circuit behavior

  - Covers: persona short-circuit tests and service-level no-response test.
  - Verify: run the new focused tests and record the expected baseline failure.
  - Evidence: record failing test names and failure reason in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-08`.

- [x] Stage 2 - evaluator prompt-contract tests added

  - Covers: anchor-wide evaluator criteria and unsupported concrete claim checks.
  - Verify: run `venv\Scripts\python -m pytest tests/test_dialog_agent.py -q`.
  - Evidence: record failing or passing result in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-08`.

- [x] Stage 3 - cognition silence route implemented

  - Covers: `persona_supervisor2.py` and schema typing update.
  - Verify: focused persona and service tests pass.
  - Evidence: record changed files and command output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-08`.

- [x] Stage 4 - evaluator criteria updated without protocol change

  - Covers: `_DIALOG_EVALUATOR_PROMPT` only unless a test proves a tiny adjacent edit is required.
  - Verify: dialog tests pass and grep confirms no new evaluator schema fields.
  - Evidence: record test and grep results.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-08`.

- [x] Stage 5 - full deterministic verification complete

  - Covers: all commands in `Verification`, including the mandatory live LLM
    evaluator diagnostic.
  - Verify: all deterministic checks pass.
  - Evidence: record command outputs and any skipped live tests.
  - Handoff: plan may move to completed execution record after user approval.
  - Sign-off: `Codex/2026-05-08`.

## Verification

### Static Checks

```powershell
rg "approved|retryable|fatal_errors" src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py
```

Allowed result: no new evaluator protocol fields introduced by this plan.
Existing unrelated matches must be recorded if present.

```powershell
rg "should_respond" src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py tests\test_persona_supervisor2.py
```

Expected result: persona now preserves and returns `should_respond`.

### Focused Tests

```powershell
venv\Scripts\python -m pytest tests/test_persona_supervisor2.py -q
venv\Scripts\python -m pytest tests/test_dialog_agent.py -q
```

### Service Graph / Background Skip Tests

```powershell
venv\Scripts\python -m pytest tests/test_service_background_consolidation.py::test_build_graph_skips_episode_state_loader_when_relevance_declines -q
```

Add the new persona-level no-response service test to this command list during
implementation and run it by exact test id.

### Prompt Rendering

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py
```

If evaluator prompt formatting uses `.format(...)`, add a runtime prompt-render
test or extend an existing prompt assertion test. `py_compile` alone is not
enough to catch broken prompt braces.

### Mandatory Live LLM Diagnostic

Run only one live case at a time and inspect output:

```powershell
venv\Scripts\python -m pytest tests/test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_unanchored_concrete_claim -q -s -m live_llm
```

This live test is required for sign-off. If the configured evaluator endpoint
is unavailable, the implementation may be locally complete but the plan is not
signed off.

## Risks

| Risk                                                          | Mitigation                                                                   | Verification                                                                   |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Silent cognition still reaches dialog                         | Add post-cognition route and assert `dialog_agent` is not awaited.           | Focused `test_persona_supervisor2.py` mock test.                               |
| Persona returns missing output fields after no-response route | Use deterministic no-response node with full output shape.                   | Persona test asserts `final_dialog`, targets, broadcast, and `should_respond`. |
| Dialog becomes an accidental relevance gate                   | Scope evaluator to quality against anchors only.                             | Prompt assertions and review of changed lines.                                 |
| Evaluator protocol accidentally changes                       | Static grep for new schema fields and tests using existing `should_stop`.    | Static checks and dialog tests.                                                |
| Prompt grows brittle for local LLM                            | Keep criteria short, anchor-labeled, and grounded in visible payload fields. | Prompt contract tests and mandatory live diagnostic.                           |

## Acceptance Criteria

This plan is complete when:

- A cognition result with `expression_willingness="silent"` de-asserts `should_respond`.
- The persona graph bypasses dialog for cognition silence.
- The service result for cognition silence contains `should_respond=false` and `final_dialog=[]`.
- Existing background progress/consolidation/persistence skip behavior applies because `final_dialog=[]`.
- The evaluator prompt checks final dialog quality against all relevant anchor categories.
- The evaluator rejects unsupported concrete claims as a general anchor-fidelity rule.
- The generator/evaluator conversational contract remains `feedback` plus `should_stop`.
- No relevance, RAG, memory, adapter, scheduler, queue, database, or delivery receipt behavior is changed.
- All deterministic verification commands pass or have recorded, approved exceptions.

## Execution Evidence

- Static grep results:
  - `rg "approved|retryable|fatal_errors" src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py`
    returned only existing mocked evaluator JSON fixtures containing
    `fatal_errors` in `tests/test_dialog_agent.py` and an unrelated
    `approved` word in a dialog-agent docstring. No new evaluator protocol
    fields were introduced.
  - `rg "should_respond" src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py tests\test_persona_supervisor2.py`
    confirms persona state typing, initial-state propagation, no-response
    output, persona return propagation, and focused assertions.
- Focused persona tests:
  - Baseline before implementation:
    `venv\Scripts\python -m pytest tests/test_persona_supervisor2.py::test_persona_supervisor2_silent_cognition_short_circuits_dialog -q`
    failed with `KeyError: 'should_respond'`.
  - After implementation:
    `venv\Scripts\python -m pytest tests/test_persona_supervisor2.py::test_persona_supervisor2_silent_cognition_short_circuits_dialog -q`
    passed.
  - Full focused file:
    `venv\Scripts\python -m pytest tests/test_persona_supervisor2.py -q`
    passed, `6 passed`.
- Dialog evaluator tests:
  - Baseline before implementation:
    `venv\Scripts\python -m pytest tests/test_dialog_agent.py::test_dialog_evaluator_prompt_checks_anchor_fidelity tests/test_dialog_agent.py::test_dialog_evaluator_prompt_rejects_unsupported_concrete_content -q`
    failed because the prompt lacked the anchor-wide inspection phrase and
    unsupported concrete-content rule.
  - After implementation:
    same focused command passed, `2 passed`.
  - Full focused file:
    `venv\Scripts\python -m pytest tests/test_dialog_agent.py -q`
    passed, `10 passed`.
  - Prompt architecture review pass:
    `_DIALOG_EVALUATOR_PROMPT` was reorganized for weak-model readability:
    role boundary first, explicit audit order second, anchor fidelity as the
    first hard gate, expression/structure checks after anchor fidelity, and
    soft style checks last. Duplicate fact/logic rules were folded into the
    anchor-fidelity section instead of appended as a separate late rule.
  - Added prompt-structure assertion:
    `tests/test_dialog_agent.py::test_dialog_evaluator_prompt_orders_hard_gates_before_style`.
  - After prompt architecture pass:
    `venv\Scripts\python -m pytest tests/test_dialog_agent.py -q`
    passed, `11 passed`.
- Service graph/background tests:
  - Baseline before persona implementation:
    `venv\Scripts\python -m pytest tests/test_service_background_consolidation.py::test_build_graph_preserves_persona_no_response tests/test_service_background_consolidation.py::test_chat_cognition_silence_skips_user_visible_work -q`
    passed, proving existing service behavior already skips visible work when
    the graph result contains `should_respond=false` and `final_dialog=[]`.
  - After implementation:
    `venv\Scripts\python -m pytest tests/test_service_background_consolidation.py::test_build_graph_skips_episode_state_loader_when_relevance_declines tests/test_service_background_consolidation.py::test_build_graph_preserves_persona_no_response tests/test_service_background_consolidation.py::test_chat_cognition_silence_skips_user_visible_work tests/test_service_background_consolidation.py::test_build_graph_preserves_consolidation_state_from_supervisor -q`
    passed, `4 passed`.
- Prompt render checks:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`
    passed.
  - After prompt architecture pass:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py tests\test_dialog_evaluator_live_llm_contract.py`
    passed.
  - `tests/test_dialog_agent.py` passed through prompt import and patched
    dialog-agent execution, covering evaluator prompt formatting in the
    affected deterministic suite.
- Mandatory live LLM diagnostic:
  - Added `tests/test_dialog_evaluator_live_llm_contract.py`.
  - Ran one case individually:
    `venv\Scripts\python -m pytest tests/test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_unanchored_concrete_claim -q -s -m live_llm`
    passed, `1 passed`.
  - Durable trace:
    `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__reject_unanchored_model_range.json`.
  - Inspected behavior: the live evaluator rejected
    `AG-12的有效射程按标定是460米。` with `should_stop=false` and feedback
    explaining that the reply violated `[DECISION]` and `[SCOPE]` because it
    provided concrete model/range facts when anchors required no technical
    details and no facts.
  - After prompt architecture pass, reran the same live case individually:
    `venv\Scripts\python -m pytest tests/test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_unanchored_concrete_claim -q -s -m live_llm`
    passed, `1 passed`.
  - New durable trace:
    `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__reject_unanchored_model_range__20260507T221136703817Z.json`.
  - Inspected behavior after prompt rewrite: the live evaluator again returned
    `should_stop=false`, with concise feedback that the dialog violated
    `[DECISION]` and `[SCOPE]` by providing a concrete model/range fact when
    anchors required no technical detail and no facts.
  - After fixing the prompt output-format example, reran the same live case:
    `venv\Scripts\python -m pytest tests/test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_unanchored_concrete_claim -q -s -m live_llm`
    passed, `1 passed`.
  - Final durable trace:
    `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__reject_unanchored_model_range__20260507T221311512319Z.json`.
  - Final inspected behavior: the live evaluator returned `should_stop=false`
    and rejected the reply for violating `[DECISION]` / `[SCOPE]`, explicitly
    calling out the unauthorized concrete model and parameter `AG-12/460米`.
