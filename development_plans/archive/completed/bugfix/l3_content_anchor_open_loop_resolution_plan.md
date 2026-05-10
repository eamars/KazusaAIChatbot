# l3 content anchor open loop resolution plan

## Summary

- Goal: Prevent L3 content anchors from treating a user's valid answer to an active quiz/open-loop question as unanswered, while preserving correct behavior for non-answers and wrong answers.
- Plan class: medium.
- Status: completed.
- Mandatory skills: `development-plan-writing`, `test-style-and-execution`, `local-llm-architecture`, `py-style`, `cjk-safety`.
- Overall cutover strategy: bigbang for the L3 content-anchor prompt contract; compatible for stored conversation progress and historical chat rows.
- Highest-risk areas: adding deterministic semantic repair, broadening the change outside L3, appending warning bullets instead of rewriting the prompt contract, under-testing wrong-answer behavior, and prematurely splitting content anchors into extra runtime LLM calls.
- Acceptance criteria: four focused live LLM failure-mode cases pass one by one; valid answers close or advance the open loop; non-answers and wrong answers keep the open loop unresolved; anchors remain content directives rather than full dialog; no production code outside the L3 content-anchor prompt is changed; the initial fix preserves one content-anchor LLM call.

## Context

The live QQ trace for platform user `673225019` showed a stuck-loop failure in a diving quiz episode.

Pipeline evidence:

- Relevance correctly decided the user was answering the bot's question.
- Conversation progress loaded the active episode and open loop.
- Decontextualizer preserved the answer text: `瓦尔萨尔瓦动作` and `捏鼻鼓气法`.
- RAG correctly reported no retrieval was needed.
- Cognition L3 content anchors reframed the user's answer as general knowledge display and produced progression guidance that continued asking the first question.
- Dialog followed the bad anchor and told the user that the first question had not been answered.
- Conversation progress recorder detected the loop after the bad response, which was too late to prevent the bad turn.

Root cause hypothesis:

```text
current user input -> L3 content-anchor interpretation -> dialog output
```

The failure is in the L3 content-anchor contract. It does not explicitly require the model to classify the current user input against `conversation_progress.open_loops` before emitting `[ANSWER]` and `[PROGRESSION]`. As a result, the model can use stale `current_blocker` or playful banter framing even when the current user input has already resolved the open loop.

## Architecture Probe Evidence

An approved diagnostic probe compared three candidate designs against the four known failure-mode cases:

- single rewritten content-anchor prompt
- hybrid open-loop classifier plus anchor generator
- split one-anchor-per-LLM generation following the dependency tree

Artifacts:

- Probe harness: `test_artifacts/diagnostics/content_anchor_split_feasibility_probe.py`
- Latest report: `test_artifacts/diagnostics/content_anchor_split_feasibility_report_trials3_v2.json`

Baseline evidence from the current production prompt, using prior live traces:

| Design | Cases | Pass Rate | Positive Pass Rate | Negative Pass Rate | Avg Latency | Calls / Case |
|---|---:|---:|---:|---:|---:|---:|
| Current production prompt | 4 | 50.0% | 50.0% | 50.0% | not measured in same harness | 1 |

Three-trial diagnostic results from `content_anchor_split_feasibility_report_trials3_v2.json`:

| Candidate | Cases | Pass Rate | Positive Pass Rate | Negative Pass Rate | Avg Latency | Max Latency | Calls / Case |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single rewritten prompt | 12 | 91.7% | 100.0% | 83.3% | 2.418s | 3.271s | 1 |
| Hybrid classifier plus anchor | 12 | 100.0% | 100.0% | 100.0% | 4.039s | 5.051s | 2 |
| Split one LLM per anchor | 12 | 91.7% | 100.0% | 83.3% | 4.141s | 5.836s | 6 |

Repeat probe after loading a different cognition LLM:

- Full three-trial run timed out after 20 minutes with no final report.
- Full one-trial run timed out after 15 minutes with no final report.
- The harness was extended with diagnostic-only architecture filtering and per-call timeout controls so the same cases could be run per candidate without touching production code.

One-trial repeated results:

| Candidate | Cases | Pass Rate | Positive Pass Rate | Negative Pass Rate | Avg Latency | Max Latency | Calls / Case | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Single rewritten prompt | 4 | 75.0% | 100.0% | 50.0% | 90.438s | 120.005s | 1 | Failed `wrong_guess` by per-call timeout, not by semantic false-accept output. |
| Hybrid classifier plus anchor | 4 | 100.0% | 100.0% | 100.0% | 158.743s | 179.369s | 2 | Semantically strongest but too slow for normal live chat. |
| Split one LLM per anchor | 4 | 75.0% | 100.0% | 50.0% | 284.030s | 300.028s | 6 | Failed `wrong_guess`; all six split-stage calls hit the 60s cap. |

Probe interpretation:

- The single rewritten prompt is the fastest candidate and already reaches 11/12 in the diagnostic harness. Its remaining automated failure was a conditional wording/scorer issue around a negative case, not a demonstrated need for more LLM stages.
- The hybrid candidate is the strongest measured fallback, reaching 12/12, but it doubles runtime LLM calls and adds roughly 67% average latency compared with the single rewritten prompt.
- The full split candidate does not outperform the hybrid or single candidate. It uses 6 LLM calls per case and produced an inter-anchor contradiction: the decision stage marked `unresolved_wrong_or_insufficient`, while the answer stage drifted into `这也能猜对？那是 Valsalva 法...`.
- The repeat run on a slower loaded LLM strengthened the latency conclusion. Even one content-anchor call averaged about 90s; two calls averaged about 159s; six split calls averaged about 284s under a stricter timeout cap. Under this model, adding runtime calls is operationally unacceptable for the live chat path.

Architecture conclusion:

- Initial implementation must use the single rewritten content-anchor prompt.
- Hybrid classifier plus anchor generation is an approval-gated fallback only if the single rewritten prompt fails expanded live validation after a full rewrite pass.
- Full one-anchor-per-LLM splitting is rejected for this bugfix because it adds call count, latency, graph complexity, and inter-stage contradiction risk without better measured reliability.

## Mandatory Skills

- `development-plan-writing`: load before editing, approving, executing, or updating this plan.
- `test-style-and-execution`: load before adding, changing, or running deterministic or live LLM tests.
- `local-llm-architecture`: load before changing any prompt, cognition, dialog, RAG, graph, evaluator, or LLM-stage behavior.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files that contain CJK prompt strings.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the active agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- Check `git status --short` before editing.
- Do not read `.env`.
- Do not make source, prompt, or test implementation changes until this draft plan is approved for execution by the user.
- On approval, update this plan's `Status` from `draft` to `approved` and update the registry row in `development_plans/README.md` before implementation starts.
- Use TDD: add focused tests first, run them before implementation, and record the baseline behavior.
- Run live LLM tests one case at a time with `-o addopts="" -m live_llm`, inspect each trace, and record the result before running the next case.
- Do not add deterministic keyword classifiers, regex answer detectors, post-hoc semantic validators, or production repair code over raw user input.
- LLM stages own semantic judgment. Deterministic code may only carry structured fields, validate output shape, persist state, and run tests.
- Keep the fix inside the L3 content-anchor stage unless the approved test setup requires fixture support in the live prompt-contract test file.
- Do not change relevance, decontextualizer, RAG, dialog, progress recorder, database state shape, adapters, or scheduler behavior.
- Do not split content anchors into multiple runtime LLM calls for the initial implementation.
- Do not implement the hybrid classifier plus anchor design unless the single-prompt rewrite fails expanded validation and the user approves a plan update.
- If the prompt is touched, rewrite the entire `_CONTENT_ANCHOR_AGENT_PROMPT` as a coherent, concise interpretation contract. Do not append isolated negative constraints, warning blocks, or plan-specific caveats.
- The rewritten prompt must keep anchors as content directives. It must not instruct the model to write full final dialog, a complete next quiz question body, markdown sections, or user-facing prose inside `[ANSWER]` or `[PROGRESSION]`.
- Any failed post-fix failure-mode test must trigger a full prompt rewrite pass, not a small appended warning.
- After editing `_CONTENT_ANCHOR_AGENT_PROMPT`, run both `py_compile` and a runtime `.format(character_name=...)` prompt-render check before live LLM validation.
- For Python files containing CJK strings, run `py_compile` after edits.

## Must Do

- Add at least two positive live LLM prompt-contract cases where the current user input resolves an active open loop.
- Add at least two negative live LLM prompt-contract cases where the current user input does not resolve the active open loop.
- Validate all four cases before implementation to nail down failure mode and blast radius.
- Rewrite the L3 content-anchor prompt around explicit current-turn interpretation:

```text
current input function
-> relation to active open loop
-> decision
-> fact
-> answer
-> progression
-> scope
```

- Preserve existing prompt responsibilities for stance binding, fact grounding, clarification override, active commitment temporal fields, and output labels.
- Preserve anchor-only behavior: `[ANSWER]` and `[PROGRESSION]` describe what the dialog should cover, not the final spoken response or full next-question content.
- Re-run the same failure-mode tests after the prompt rewrite.
- Run adjacent deterministic prompt-contract tests and syntax checks.

## Deferred

- Do not implement dialog fallback or evaluator repair for this bug.
- Do not modify conversation progress recorder behavior.
- Do not backfill or edit stored `conversation_episode_state`.
- Do not add a general quiz engine, scoring engine, or answer-key system.
- Do not solve all factual correctness for arbitrary quiz topics.
- Do not add a new LLM stage or new runtime LLM call.
- Do not implement full one-anchor-per-LLM splitting in this bugfix.
- Do not implement the hybrid classifier plus anchor design in the initial fix.
- Do not change user memory, consolidation, reflection, scheduler, or RAG contracts.

## Cutover Policy

Overall strategy: bigbang for the L3 content-anchor prompt contract.

| Area | Policy | Instruction |
|---|---|---|
| L3 content-anchor prompt | bigbang | Replace the existing prompt contract with a reorganized interpretation contract. No legacy prompt path or fallback. |
| Live LLM prompt-contract tests | bigbang | Add focused cases that must pass against the new contract. |
| Existing stored progress rows | compatible | Do not migrate or rewrite stored rows. The new prompt interprets current input against whatever progress projection is supplied. |
| Other cognition stages | compatible | Preserve existing behavior and prompts outside L3 content anchors. |
| Dialog generation | compatible | Dialog continues consuming content anchors; no dialog workaround. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not add compatibility shims, feature flags, fallback prompts, or dual prompt paths.
- Any change to this cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may edit `tests/test_cognition_live_llm_prompt_contracts.py` to add the approved failure-mode tests and required fixture support.
- The agent may edit `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` only to rewrite `_CONTENT_ANCHOR_AGENT_PROMPT`.
- The agent may update this plan's `Execution Evidence` while executing the approved plan.
- The agent must stop and ask for approval before touching any production file outside `persona_supervisor2_cognition_l3.py`.
- The agent must stop and ask for approval before changing test scope beyond the four failure-mode cases and adjacent prompt-contract checks.
- The agent must stop and ask for approval before adding any runtime LLM call, graph node, classifier stage, per-anchor generator, or split-anchor orchestration.
- The agent may read diagnostic artifacts under `test_artifacts/diagnostics`, but must not edit the diagnostic harness or reports during implementation unless the user approves a new investigation step.
- The agent must not commit, push, or open a PR unless explicitly requested.

## Target State

For active open-loop conversations, L3 content anchors first interpret the latest user input against the active loop. The current user turn has priority over stale blocker text in `conversation_progress`.

Expected behavior:

- If the user gives a correct or equivalent answer, `[ANSWER]` acknowledges or uses that answer, and `[PROGRESSION]` closes the old loop or moves to the natural next step.
- If the user gives a partial, wrong, or unsupported answer, `[ANSWER]` corrects, challenges, or asks for a more specific answer, and `[PROGRESSION]` keeps the loop unresolved.
- If the user only banters or compliments, `[SOCIAL]` may react to the banter, but `[ANSWER]` and `[PROGRESSION]` do not treat the question as answered.
- The prompt never instructs dialog to both accept an answer and demand the same answer again.
- The prompt keeps content anchors compact. It may say to advance to a second question, but it must not write the actual second question body inside the anchor.

## Design Decisions

- The fix belongs in L3 content-anchor prompt contract because the trace shows upstream interpretation and RAG were healthy, and dialog followed the L3 anchors.
- The implementation must remain LLM-first. A deterministic answer matcher would violate the architecture boundary and would not generalize to non-quiz open loops.
- Test coverage must include both positive and negative behavior because the fix could otherwise overcorrect and accept banter or wrong guesses as valid answers.
- The prompt rewrite must be structural. The model needs a clear interpretation order, not one more prohibition at the end of the existing prompt.
- The blast radius is bounded to L3 prompt behavior plus live prompt-contract tests.
- The approved architecture is single-prompt first. Diagnostic data showed the single rewritten prompt is the lowest-latency viable candidate, while hybrid is a fallback and full per-anchor splitting is not justified.
- The hybrid fallback is reserved for a future plan update because it changes runtime call count and latency. It must not be smuggled into this bugfix as a convenience if prompt tuning is hard.
- Full per-anchor splitting is specifically rejected here because the dependency tree is mostly sequential and the probe demonstrated answer-stage drift from an upstream unresolved decision.

## Contracts And Data Shapes

Input fields relied on by the prompt:

- `decontexualized_input`: current user input semantics. This is the freshest semantic evidence.
- `conversation_progress`: prior episode state, including `current_thread`, `current_blocker`, `open_loops`, `next_affordances`, and `progression_guidance`.
- `internal_monologue`: upstream cognition reasoning, including whether the user answer should be accepted, challenged, or treated as insufficient.
- `logical_stance` and `character_intent`: binding upstream stance and intent.
- `rag_result.answer`: direct fact summary when retrieval is relevant.
- `referents`: unresolved-reference gate.

Output contract:

- `content_anchors` remains a list of strings.
- `[DECISION]` remains first.
- `[SCOPE]` remains last.
- Allowed labels remain `[DECISION]`, `[FACT]`, `[ANSWER]`, `[SOCIAL]`, `[AVOID_REPEAT]`, `[PROGRESSION]`, and `[SCOPE]`.
- `[ANSWER]`, `[FACT]`, and `[PROGRESSION]` must not contradict one another.
- Anchors are not final dialog lines. They are compact instructions to the downstream dialog stage.

## LLM Call And Context Budget

- No new runtime LLM call is allowed in the initial implementation.
- No new graph node is allowed in the initial implementation.
- The existing L3 content-anchor call remains the only modified runtime prompt.
- The prompt rewrite must not significantly expand context. Remove or compress redundant prose while adding the explicit interpretation order.
- The live tests use existing test harness calls only.
- If the single-prompt approach cannot pass the approved live validation after a full rewrite pass, execution must stop and request approval for a plan update before using the measured hybrid fallback.

## Change Surface

Allowed files:

- `tests/test_cognition_live_llm_prompt_contracts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `development_plans/active/bugfix/l3_content_anchor_open_loop_resolution_plan.md`
- `development_plans/README.md`

Read-only evidence artifacts:

- `test_artifacts/diagnostics/content_anchor_split_feasibility_probe.py`
- `test_artifacts/diagnostics/content_anchor_split_feasibility_report*.json`

The `test_artifacts/diagnostics` files are investigation artifacts only. They are not production source and are not part of the runtime fix. They may be read for context but must not be edited during implementation. No other files are in scope.

## Implementation Order

1. Confirm clean starting state.
   - Command: `git status --short`
   - Expected: only approved plan, registry, and diagnostic artifact changes are present, or unrelated changes are documented and not touched.

2. Load required skills.
   - Read `py-style`, `test-style-and-execution`, `local-llm-architecture`, and `cjk-safety`.
   - Record skill loading in `Execution Evidence`.

3. Record plan approval and activate the plan.
   - File: `development_plans/active/bugfix/l3_content_anchor_open_loop_resolution_plan.md`
   - Action: change `Status: draft` to `Status: approved`.
   - File: `development_plans/README.md`
   - Action: change the registry row for this plan from `draft | deferred` to `approved | approved`.
   - Evidence: record the user's approval message and the status update in `Execution Evidence`.

4. Add live LLM fixture support in `tests/test_cognition_live_llm_prompt_contracts.py`.
   - Ensure direct `call_content_anchor_agent` test states include `cognitive_episode` and `referents`.
   - Use existing project cognitive episode builders and time-context helpers.
   - This is test harness support only.
   - The preferred implementation is to update the shared live `_make_state` helper once so existing direct content-anchor live tests and the new cases use the same valid state shape.
   - Do not alter existing case assertions except where required to keep them compatible with the current `call_content_anchor_agent` input contract.

5. Add two positive failure-mode tests.
   - Test name: `test_live_content_anchor_quiz_full_answer_closes_open_loop`.
   - Input: user gives `瓦尔萨尔瓦动作` / `捏鼻鼓气法` plus banter.
   - Expected post-fix: anchors acknowledge or preserve the answer and advance or close the first-question loop.
   - Test name: `test_live_content_anchor_quiz_concise_answer_closes_open_loop`.
   - Input: user says `瓦尔萨尔瓦，捏鼻鼓气法。轮到下一题了吧？`
   - Expected post-fix: anchors acknowledge correctness and move toward the next question.

6. Add two negative failure-mode tests.
   - Test name: `test_live_content_anchor_quiz_banter_without_answer_keeps_open_loop`.
   - Input: user compliments the character and says they have not thought of the answer.
   - Expected post-fix: anchors do not mark the first question as answered.
   - Test name: `test_live_content_anchor_quiz_wrong_answer_keeps_open_loop`.
   - Input: user gives a wrong or insufficient guess such as `是不是张嘴吞咽就好了？我猜的。`
   - Expected post-fix: anchors do not accept the answer as correct and do not advance to question two.

7. Run each new live LLM test before implementation.
   - Command pattern: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::<test_name> -q -s`
   - Expected: at least one positive and one negative case fail or expose the current behavior gap.
   - Record each output and trace review in `Execution Evidence`.

8. Rewrite `_CONTENT_ANCHOR_AGENT_PROMPT`.
   - File: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
   - Rewrite the full prompt into a concise ordered contract.
   - Required ordering: current input function, open-loop relation, decision, fact, answer, social posture, progression, scope.
   - Preserve existing output format and label set.
   - Do not append a new warning section to the old prompt.
   - Keep `[ANSWER]` and `[PROGRESSION]` as anchor directives. Do not embed full final dialog, markdown next-question blocks, or long user-facing prose.

9. Run syntax and prompt-render checks.
   - Command: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py`
   - Expected: pass.
   - Command: `venv\Scripts\python.exe -c "from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as m; m._CONTENT_ANCHOR_AGENT_PROMPT.format(character_name='测试角色'); print('OK')"`
   - Expected: prints `OK`.

10. Re-run the four new live LLM tests one by one.
   - Use the exact command pattern from step 7.
   - Expected: all four pass.
   - If any case fails, return to step 8 and rewrite the whole prompt again.
   - If a live case exceeds 5 minutes on the loaded model, stop after the case returns or times out externally, record the latency issue, and ask the user whether to continue with the current model before running more live cases.

11. Run adjacent deterministic tests.
    - Command: `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_cognition.py -q`
    - Expected: pass.

12. Run the focused live content-anchor regression checks already in the suite.
    - Run existing direct content-anchor live tests that cover birthday facts and direct conversation evidence one by one.
    - Expected: pass or documented skip only when the live endpoint is unavailable.

13. Run independent code review gate.
    - Review the implementation diff against this plan, architecture boundaries, prompt rewrite requirements, and test evidence.
    - Fix in-scope findings and repeat affected verification.

## Progress Checklist

- [x] Stage 1 - plan approved for execution.
  - Covers: implementation steps 1-3.
  - Verify: user explicitly approves execution; plan status and registry row are updated to `approved`.
  - Evidence: record approval, `git status --short`, and status-update diff in `Execution Evidence`.
  - Sign-off: `Codex / 2026-05-10` after approval and status updates are recorded.

- [x] Stage 2 - TDD failure-mode tests added.
  - Covers: implementation steps 4-6.
  - Verify: `git diff -- tests/test_cognition_live_llm_prompt_contracts.py` shows only planned test harness and four failure-mode cases.
  - Evidence: record changed test names and fixture support.
  - Sign-off: `Codex / 2026-05-10` after diff review.

- [x] Stage 3 - pre-fix behavior validated.
  - Covers: implementation step 7.
  - Verify: four live LLM tests run one by one and outputs inspected.
  - Evidence: record command, pass/fail, and diagnosis for each case.
  - Sign-off: `Codex / 2026-05-10` after all four baseline runs are recorded.

- [x] Stage 4 - full L3 prompt rewrite complete.
  - Covers: implementation steps 8-9.
  - Verify: prompt diff is a coherent rewrite, not appended constraints; `py_compile` and prompt-render check pass.
  - Evidence: record syntax-check output, prompt-render output, and prompt-diff review notes.
  - Sign-off: `Codex / 2026-05-10` after syntax, render, and diff review.

- [x] Stage 5 - post-fix failure-mode validation complete.
  - Covers: implementation step 10.
  - Verify: all four new live LLM tests pass one by one.
  - Evidence: record command and inspected trace summary for each test.
  - Sign-off: `Codex / 2026-05-10` after all four pass.

- [x] Stage 6 - adjacent regression validation complete.
  - Covers: implementation steps 11-12.
  - Verify: deterministic prompt-contract tests pass and selected existing live content-anchor checks pass or skip only due unavailable live endpoint.
  - Evidence: record commands and results.
  - Sign-off: `Codex / 2026-05-10` after verification.

- [x] Stage 7 - independent code review complete.
  - Covers: implementation step 13.
  - Verify: review finds no unresolved plan, architecture, prompt, style, or test-evidence issues.
  - Evidence: record findings, fixes, reruns, and approval status.
  - Sign-off: `Codex / 2026-05-10` after review approval.

## Verification

Focused live LLM tests:

```powershell
venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_full_answer_closes_open_loop -q -s
venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_concise_answer_closes_open_loop -q -s
venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_banter_without_answer_keeps_open_loop -q -s
venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_wrong_answer_keeps_open_loop -q -s
```

Syntax check:

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py
```

Prompt-render check:

```powershell
venv\Scripts\python.exe -c "from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as m; m._CONTENT_ANCHOR_AGENT_PROMPT.format(character_name='测试角色'); print('OK')"
```

Adjacent deterministic regression:

```powershell
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_cognition.py -q
```

Existing live content-anchor checks:

```powershell
venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_uses_character_public_facts_for_birthday_question -q -s
venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_does_not_leak_character_public_facts_on_unrelated_question -q -s
venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_answers_from_direct_conversation_evidence -q -s
```

Live LLM endpoint unavailability blocks execution sign-off. A skip caused by unavailable endpoint is not an acceptance pass for this plan.

## Independent Code Review

Before completion, perform a fresh review of the full diff.

Review scope:

- Confirm no files outside `Change Surface` were modified.
- Confirm no deterministic semantic answer parsing or post-hoc repair was added.
- Confirm `_CONTENT_ANCHOR_AGENT_PROMPT` was rewritten coherently rather than extended with an appended warning block.
- Confirm the prompt-render check passed and no `.format(...)` placeholder was broken.
- Confirm the prompt preserves existing output labels and JSON contract.
- Confirm anchors remain content directives and do not embed full final dialog, markdown next-question blocks, or long user-facing prose.
- Confirm all four failure-mode tests exist and cover two positive plus two negative cases.
- Confirm live LLM evidence was inspected one case at a time.
- Confirm adjacent deterministic tests and syntax checks were run.
- Confirm any post-fix failure triggered a full prompt rewrite pass.
- Confirm no runtime LLM call, graph node, classifier stage, or per-anchor split path was added.
- Confirm the architecture conclusion from `Architecture Probe Evidence` is followed: single-prompt first, hybrid only by approved plan update, full per-anchor split rejected.

Findings that require code outside the approved change surface must stop execution until the user approves a plan update.

## Acceptance Criteria

- The full-answer positive case acknowledges or preserves `瓦尔萨尔瓦` / `捏鼻鼓气` and does not keep asking the first question.
- The concise-answer positive case acknowledges correctness and advances naturally.
- The banter-without-answer negative case does not mark the first question as answered.
- The wrong-answer negative case does not mark the guess as correct and does not advance to question two.
- Existing content-anchor behavior for unrelated fact questions and direct evidence remains intact.
- Anchors remain compact content directives and do not contain full final dialog or full next-question bodies.
- No runtime LLM calls are added.
- The content-anchor runtime remains one LLM call.
- Hybrid classifier plus anchor generation is not implemented unless a later user-approved plan update supersedes this one.
- Full one-anchor-per-LLM generation is not implemented.
- No production code outside the L3 content-anchor prompt is changed.
- `Execution Evidence` contains commands, inspected outcomes, and review sign-off.

## Risks

- Live LLM nondeterminism can cause flaky tests. Mitigation: run one case at a time, inspect traces, and encode assertions around contract-level behavior rather than exact wording.
- Prompt rewrite can regress existing direct-evidence behavior. Mitigation: run existing birthday and direct conversation evidence prompt-contract checks.
- The fix can overcorrect and accept weak guesses. Mitigation: keep the wrong-answer negative case as a blocking gate.
- The fix can overcorrect and ignore playful banter. Mitigation: keep the banter-without-answer negative case and preserve `[SOCIAL]` behavior.
- Adding test fixture support can accidentally change unrelated live tests. Mitigation: scope fixture additions to required fields already expected by `call_content_anchor_agent`.
- A full per-anchor split can create inter-stage contradictions even when each small prompt seems simpler. Mitigation: do not implement split generation in this bugfix.
- The hybrid fallback has better measured reliability on the small probe but increases runtime call count and latency. Mitigation: keep it approval-gated behind a plan update.

## Execution Evidence

Status: completed on 2026-05-10 after implementation, validation, and independent review.

Execution start:

- Loaded mandatory execution skills: `development-plan-writing`, `test-style-and-execution`, `local-llm-architecture`, `py-style`, and `cjk-safety`.
- Starting state: `git status --short` showed `M development_plans/README.md` and `?? development_plans/active/bugfix/`.
- Approval: user instructed `Execute the plan now`.
- Lifecycle update: plan status changed from `draft` to `approved`; registry row changed from `draft | deferred` to `approved | approved`.

Stage 2 evidence:

- Updated `tests/test_cognition_live_llm_prompt_contracts.py` shared `_make_state` fixture to include `cognitive_episode` and `referents` for direct `call_content_anchor_agent` live tests.
- Added positive cases: `test_live_content_anchor_quiz_full_answer_closes_open_loop` and `test_live_content_anchor_quiz_concise_answer_closes_open_loop`.
- Added negative cases: `test_live_content_anchor_quiz_banter_without_answer_keeps_open_loop` and `test_live_content_anchor_quiz_wrong_answer_keeps_open_loop`.
- Syntax check passed: `venv\Scripts\python.exe -m py_compile tests\test_cognition_live_llm_prompt_contracts.py`.
- Diff review: changes were limited to planned fixture support and the four failure-mode live prompt-contract cases.

Stage 3 evidence:

- Pre-fix run: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_full_answer_closes_open_loop -q -s` passed in 6.74s. Inspected anchors accepted `瓦尔萨尔瓦` and advanced to the second challenge.
- Pre-fix run: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_concise_answer_closes_open_loop -q -s` passed in 4.66s. Inspected anchors accepted `瓦尔萨尔瓦法（捏鼻鼓气）` and moved to the next question.
- Pre-fix run: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_banter_without_answer_keeps_open_loop -q -s` initially failed because the test helper treated negated or future-conditional wording such as `不确认对方已答对` and `观察用户是否能给出正确回答` as acceptance. The helper was tightened, syntax check passed again, and the case reran successfully in 4.08s with anchors keeping the first-question loop unresolved.
- Pre-fix run: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_wrong_answer_keeps_open_loop -q -s` passed in 4.51s. Inspected anchors rejected `张嘴吞咽` as not the requested most common basic method and kept the first question active.
- Current loaded model did not reproduce the production failure on these four synthetic prompt-contract cases. The production QQ trace and recorded diagnostic baseline remain the reproduction evidence for the prompt-contract gap; the new tests define blast radius and guard against both non-acceptance and over-acceptance regressions.

Stage 4 evidence:

- Rewrote `_CONTENT_ANCHOR_AGENT_PROMPT` in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` around ordered current-turn interpretation: current input function, relation to active open loop, `[DECISION]`, `[FACT]`, `[ANSWER]`, `[SOCIAL]`, `[PROGRESSION]`/`[AVOID_REPEAT]`, and `[SCOPE]`.
- Preserved existing label schema, stance binding, fact grounding, clarification override, active commitment `due_state` handling, topic admission ownership, and output JSON contract.
- Kept the runtime architecture to one existing content-anchor LLM call; no graph node, classifier, repair path, or split-anchor call was added.
- Syntax check passed: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py`.
- Prompt render check passed: `venv\Scripts\python.exe -c "from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as m; m._CONTENT_ANCHOR_AGENT_PROMPT.format(character_name='测试角色'); print('OK')"` printed `OK`.
- Prompt token check: content prompt contains neither lowercase `dialog` nor the direct-evidence regression examples `充电线` / `HDMI`.
- Prompt diff review: changes reorganize the interpretation contract rather than adding a separate repair stage or production keyword detector.

Stage 5 evidence:

- Post-fix run: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_full_answer_closes_open_loop -q -s` passed in 6.81s. Inspected anchors confirmed `瓦尔萨尔瓦动作（捏鼻鼓气法）`, closed the first thread, and gave only a compact second-question direction.
- Post-fix run: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_concise_answer_closes_open_loop -q -s` passed in 4.98s. Inspected anchors confirmed `瓦尔萨尔瓦法（捏鼻鼓气）` and closed `第一题：耳压平衡基础方法`.
- Post-fix run: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_banter_without_answer_keeps_open_loop -q -s` passed in 4.51s. Inspected anchors kept the contest open and explicitly did not advance to the next question.
- Post-fix run: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_quiz_wrong_answer_keeps_open_loop -q -s` passed in 5.05s. Inspected anchors rejected `张嘴吞咽` as not the requested standard answer and kept the first question unresolved.

Stage 6 evidence:

- Adjacent deterministic regression passed: `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_cognition.py -q` returned 8 passed in 1.42s.
- Existing live content-anchor regression passed: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_uses_character_public_facts_for_birthday_question -q -s` passed in 6.07s and preserved `8月5日`.
- Existing live content-anchor regression passed: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_does_not_leak_character_public_facts_on_unrelated_question -q -s` passed in 4.74s and did not leak birthday or zodiac facts.
- Existing live content-anchor regression passed: `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_cognition_live_llm_prompt_contracts.py::test_live_content_anchor_answers_from_direct_conversation_evidence -q -s` passed in 6.08s and preserved `充电线`, `HDMI 线`, and用途不明线缆 details.

Stage 7 independent review:

- Change surface review passed: implementation changes are limited to `development_plans/README.md`, this plan record, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`, and `tests/test_cognition_live_llm_prompt_contracts.py`.
- Architecture review passed: no deterministic semantic answer parser, regex answer detector, post-hoc repair path, runtime classifier, graph node, fallback prompt, hybrid stage, or per-anchor split call was added.
- Runtime call review passed: the existing `_content_anchor_agent_llm` remains the only content-anchor LLM call, and no new `get_llm(...)` call was introduced in the changed region.
- Prompt review passed: the content-anchor prompt now makes current input function and active open-loop relation first-class interpretation steps while preserving allowed labels, JSON output, stance binding, fact grounding, clarification override, and active commitment `due_state` handling.
- Test review passed: four focused live cases cover two positive answer-resolves-open-loop paths and two negative non-answer/wrong-answer paths; existing direct content-anchor live regressions still pass.
- Evidence review passed: live LLM tests were run one case at a time and inspected; deterministic tests, syntax checks, prompt-render check, prompt token checks, and `git diff --check` passed.
- Residual note: the fresh synthetic pre-fix run did not reproduce a failing case on the currently loaded model; this is documented in Stage 3. The production QQ trace and prior diagnostic baseline remain the bug reproduction evidence, and post-fix validation now guards the intended contract.
- Review conclusion: no unresolved plan, architecture, prompt, style, or test-evidence issues remain inside the approved scope.

Lifecycle close:

- User requested plan completion after confirming the issue was fixed.
- Plan lifecycle updated from active approved bugfix to completed historical record at `development_plans/archive/completed/bugfix/l3_content_anchor_open_loop_resolution_plan.md`.

Architecture option probe:

- Harness: `test_artifacts/diagnostics/content_anchor_split_feasibility_probe.py`
- Syntax check: `venv\Scripts\python.exe -m py_compile .\test_artifacts\diagnostics\content_anchor_split_feasibility_probe.py` passed.
- Initial one-trial report: `test_artifacts/diagnostics/content_anchor_split_feasibility_report.json`.
- Three-trial report: `test_artifacts/diagnostics/content_anchor_split_feasibility_report_trials3.json`.
- Tightened three-trial report: `test_artifacts/diagnostics/content_anchor_split_feasibility_report_trials3_v2.json`.
- Latest measured result: single rewritten prompt 11/12 at 2.418s average latency with one call; hybrid classifier plus anchor 12/12 at 4.039s average latency with two calls; full per-anchor split 11/12 at 4.141s average latency with six calls.
- Repeat run after loading a different LLM:
  - `test_artifacts/diagnostics/content_anchor_split_feasibility_report_trials3_llm2.json` was attempted but timed out after 20 minutes and produced no final report.
  - `test_artifacts/diagnostics/content_anchor_split_feasibility_report_trial1_llm2.json` was attempted but timed out after 15 minutes and produced no final report.
  - The harness was extended with diagnostic-only `--approaches` and `--call-timeout-seconds` controls; syntax check passed after the extension.
  - `test_artifacts/diagnostics/content_anchor_split_feasibility_report_trial1_llm2_single.json`: single rewritten prompt 3/4 at 90.438s average latency with one call; failure was `wrong_guess` timing out at 120s.
  - `test_artifacts/diagnostics/content_anchor_split_feasibility_report_trial1_llm2_hybrid.json`: hybrid classifier plus anchor 4/4 at 158.743s average latency with two calls.
  - `test_artifacts/diagnostics/content_anchor_split_feasibility_report_trial1_llm2_split_timeout60.json`: full per-anchor split 3/4 at 284.030s average latency with six calls; `wrong_guess` failed because all six split calls hit the 60s cap.
- Conclusion recorded: implement the single rewritten prompt first; keep hybrid as an approval-gated fallback; reject full per-anchor splitting for this bugfix.
