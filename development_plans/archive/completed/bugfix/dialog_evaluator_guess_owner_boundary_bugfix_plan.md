# dialog evaluator guess owner boundary bugfix plan

## Summary

- Goal: make the dialog evaluator reject visible replies that turn a generic
  type/tag/condition guessing gate into guessing the character's own preference.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `test-style-and-execution`, `debug-llm`, `py-style`, `cjk-safety`
- Overall cutover strategy: bigbang prompt-contract tightening with no new
  message fields, no extra LLM calls, and no deterministic text filter.
- Highest-risk areas: over-rejecting valid playful gates, adding
  user-example-shaped prompt text, weakening existing evaluator checks, and
  expanding dialog ownership beyond content-anchor auditing.
- Acceptance criteria: production evaluator rejects the owner-flipped generic
  final dialog against real L3-style anchors, still accepts the owner-correct
  control, existing dialog prompt/payload tests pass, and live LLM regression
  traces are written and inspected.

## Context

The confirmed weakness is recorded in:

```text
test_artifacts/dialog_evaluator_owner_flip_poc_review.md
test_artifacts/dialog_evaluator_owner_flip_poc_results.json
```

The POC used real L3 anchors from:

```text
test_artifacts/l3_perspective_lock_poc_results.json
```

Those anchors authorize a recommendation gate:

```text
the other person should guess a type/tag or show sincerity as an unlock
condition for recommendation
```

Production evaluator passed this owner-flipped final dialog 5/5:

```text
不如先猜猜我会想看哪种类型？ / 猜中了再给清单。
```

The POC evaluator with a generic ownership hard gate rejected the same
owner-flipped dialog 5/5 and still passed this owner-correct control 5/5:

```text
不如再猜一个标签试试看？ / 答对了再给清单。
```

Upstream experiments did not confirm L2a, L2d, L3, or dialog generator as the
first-bad owner of the failure. The confirmed weakest point is that the dialog
evaluator does not stop the visible owner flip once it appears.

This plan captures that weakness in tests and tightens only the evaluator's
semantic audit contract.

## Mandatory Skills

- `development-plan`: load before editing, executing, reviewing, changing
  lifecycle status, or signing off this plan.
- `local-llm-architecture`: load before changing dialog prompts or response
  path LLM behavior.
- `test-style-and-execution`: load before adding, changing, or running
  deterministic or live LLM tests.
- `debug-llm`: load before running live LLM checks or writing readable quality
  review artifacts.
- `py-style`: load before editing Python production or test files.
- `cjk-safety`: load before editing Python files that contain CJK prompt or
  test strings.

## Mandatory Rules

- Production changes are allowed only while this plan status is `approved` or
  `in_progress`.
- Check `git status --short` before editing.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, and plan edits.
- Do not read `.env`.
- Do not modify production files outside `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
- Do not change RAG, L2 cognition, L2d, L3 content anchors, conversation
  progress, memory, scheduler, adapter delivery, persistence, event schemas,
  or queue behavior.
- Do not add a new LLM call, retry stage, schema field, structured handoff,
  feature flag, compatibility mode, fallback path, helper agent, or
  deterministic filter over user text or final dialog.
- Keep the LLM-to-LLM handoff unchanged. The evaluator continues to receive
  the existing single JSON human payload with `retry`, `final_dialog`,
  `linguistic_directives`, and `contextual_directives`.
- Implement the fix as stable evaluator system-prompt contract text only.
- The prompt text must be generic. It must not quote the real QQ user message
  or any user-specific incident string.
- Runtime prompt changes must be coherent. The agent must reread the whole
  evaluator prompt and ensure the role statement, pass conditions, hard gates,
  audit order, input format, and output format agree.
- Real LLM tests must run one at a time with `-s`, and each trace must be
  inspected before running the next live LLM case.
- Patched or deterministic tests may run in batches after the focused contract
  test is established.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Add a deterministic prompt-contract test proving the evaluator prompt
  contains a hard gate for type/tag/condition guessing ownership.
- Add a live LLM negative regression proving the evaluator rejects an
  owner-flipped generic final dialog against generic recommendation-gate
  anchors.
- Add a live LLM positive control proving the evaluator accepts an
  owner-correct generic final dialog against the same anchor family.
- Tighten `_DIALOG_EVALUATOR_PROMPT` so it rejects final dialog that turns a
  type/tag/condition unlock gate into guessing the character's own preference.
- Keep the fix in one evaluator prompt string. Do not add new model-facing
  fields.
- Preserve existing evaluator responsibilities: anchor fidelity, topic
  consistency, fact boundary, forbidden phrases, expression safety, and
  hesitation-density rule.
- Run focused deterministic and live LLM verification listed in this plan.
- Inspect live trace artifacts and record the observed model behavior in
  `Execution Evidence`.

## Deferred

- Do not change dialog generator behavior in this plan.
- Do not change L2a, L2d, or L3 prompt text.
- Do not change conversation-progress wording or open-loop recording.
- Do not add deterministic Chinese pronoun or keyword rules.
- Do not add examples copied from real user input to runtime prompts.
- Do not add a new structured field such as `guess_owner`,
  `preference_owner`, `requester`, or `beneficiary`.
- Do not update experiments except by adding execution evidence that points to
  already-created artifacts.
- Do not tune model routes, temperatures, retry count, or parser behavior.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Evaluator prompt | bigbang | Replace the old insufficient semantic audit with a stricter owner-preservation hard gate. Do not preserve an alternate evaluator mode. |
| Evaluator payload shape | compatible | Keep the existing top-level payload keys exactly as they are. |
| Dialog generator | compatible | Leave generator prompt, payload shape, and retry behavior unchanged. |
| Tests | bigbang | Add focused regression tests for the confirmed weakness; do not keep a passing expectation for owner-flipped text. |
| Other pipeline stages | compatible | Make no changes to RAG, cognition, L2d, L3, progress, memory, adapter, or persistence behavior. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- For `bigbang` areas, rewrite the behavior directly instead of adding
  compatibility aliases or feature flags.
- For `compatible` areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

The dialog evaluator keeps this semantic contract:

```text
content_anchors -> only source of required visible reply content
final_dialog -> candidate visible text to audit against those anchors
rhetorical_strategy / linguistic_style / contextual_directives -> expression
  constraints only
```

When `content_anchors` authorize a generic guessing gate:

```text
guess a type
guess a tag
show sincerity
pass an unlock condition
choose which category the user wants
```

the evaluator must reject final dialog that changes the target into the
character's own preference:

```text
guess what I want to read
guess what I like
guess my taste
guess my preference
```

The evaluator must still accept owner-correct playful gates:

```text
guess a type/tag as the condition for receiving recommendations
say which category you want before I recommend
show sincerity before I give the list
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Fix owner | Dialog evaluator prompt only. | The confirmed failure is that evaluator passes owner-flipped final text. |
| Message shape | Keep current evaluator human payload. | The user requested no new structured LLM-to-LLM handoff when one clear string suffices. |
| Prompt wording | Use generic semantic class wording. | Prevents overfitting to a real private example and keeps the prompt reusable. |
| Deterministic code | Do not add pronoun keyword filters. | Semantic judgment remains LLM-owned; deterministic code owns parsing, validation, and retry mechanics. |
| Generator | Leave unchanged. | Generator did not reproduce the owner flip in sampled runs; this plan fixes the confirmed guardrail. |
| Test strategy | Deterministic prompt contract plus live evaluator behavior. | Deterministic tests capture intended prompt text; live tests prove the model follows it. |

## Contracts And Data Shapes

No state or payload keys are added, removed, or renamed.

Evaluator human payload remains:

```python
{
    "retry": str,
    "final_dialog": list[str],
    "linguistic_directives": {
        "rhetorical_strategy": str,
        "linguistic_style": str,
        "accepted_user_preferences": list[str],
        "content_anchors": list[str],
        "forbidden_phrases": list[str],
    },
    "contextual_directives": {
        "social_distance": str,
        "emotional_intensity": str,
        "vibe_check": str,
        "relational_dynamic": str,
    },
}
```

Evaluator output remains:

```python
{
    "feedback": str,
    "should_stop": bool,
}
```

New semantic contract:

```text
If anchors only authorize guessing a type, tag, condition, category, or unlock
step, final_dialog must not make the guessing target the active character's
own taste, preference, or desired content unless anchors explicitly say so.
```

## LLM Call And Context Budget

| Call | Before | After |
|---|---|---|
| `dialog_generator` | Response-path call, unchanged. Receives existing generator prompt and human payload. | Unchanged. |
| `dialog_evaluator` | Response-path call, one call per dialog attempt, max `MAX_DIALOG_AGENT_RETRY`. Receives existing evaluator prompt and human payload. | Same call count, same payload, same retry cap; system prompt gains one short hard-gate rule. |

Context budget:

- No new response-path LLM call is added.
- No context cap is increased.
- The evaluator prompt grows by a small hard-gate paragraph, far below the
  default 50k-token cap.
- Latency impact is limited to a small prompt-length increase in an existing
  evaluator call.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Modify only `_DIALOG_EVALUATOR_PROMPT`.
  - Add a generic hard gate under `# Hard Gates`, near topic consistency and
    anchor fidelity.
  - Adjust `# Pass Condition` or `# Audit Order` only if needed to keep the
    evaluator prompt coherent.

- `tests/test_dialog_agent.py`
  - Add deterministic prompt-contract coverage for the new owner-preservation
    hard gate.

- `tests/test_dialog_evaluator_live_llm_contract.py`
  - Add two live evaluator cases: owner-correct positive control and
    owner-flipped negative regression.
  - Reuse existing live LLM endpoint skip and trace pattern.

- `tests/test_dialog_anchor_boundary_live_llm.py`
  - Repair the direct evaluator fixture by adding the required
    `dialog_usage_mode` state key so the adjacent stale-topic regression can
    reach the evaluator LLM.

### Keep

- `experiments/dialog_evaluator_owner_flip_poc.py`
  - Keep as supporting evidence only. Do not import experiment code into tests.

- `test_artifacts/dialog_evaluator_owner_flip_poc_review.md`
  - Keep as supporting evidence only.

### Do Not Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- RAG, memory, conversation-progress, adapter, service, scheduler, and
  persistence modules.

## Overdesign Guardrail

- Actual problem: the production dialog evaluator accepts owner-flipped visible
  text where a generic type/tag guessing gate becomes guessing the character's
  own preference.
- Minimal change: add a generic ownership hard gate to the existing evaluator
  prompt and capture it with deterministic and live LLM tests.
- Ownership boundaries: L3 owns content anchors; dialog generator renders
  visible text; dialog evaluator audits visible text against anchors; code owns
  parsing, retry mechanics, event logging, and delivery.
- Rejected complexity: new LLM fields, owner schemas, deterministic pronoun
  filters, new agents, extra LLM calls, generator changes, L2/L3 prompt
  changes, feature flags, fallback paths, and model-route changes.
- Evidence threshold: add structured owner fields or upstream prompt changes
  only after a separate experiment proves evaluator prompt reinforcement is
  insufficient or over-rejects valid outputs.

## Agent Autonomy Boundaries

- The responsible agent may choose exact Chinese wording inside the evaluator
  prompt only when it preserves the contract in this plan and remains generic.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, feature flags,
  helper agents, or extra features.
- The responsible agent must treat changes outside `dialog_agent.py` and the
  listed tests as out of scope unless an import or syntax failure is directly
  caused by approved edits.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- The responsible agent must not use real QQ message text as runtime prompt
  examples or test prompt examples.
- If the prompt edit cannot make the live evaluator reject the negative case
  without breaking the positive control, stop and report the blocker with live
  trace evidence.
- If the plan and code disagree, preserve the plan's stated ownership boundary
  and report the discrepancy.

## Implementation Order

1. Parent establishes the deterministic prompt test.
   - File: `tests/test_dialog_agent.py`
   - Add `test_dialog_evaluator_prompt_rejects_guess_owner_flip`.
   - The test asserts `_DIALOG_EVALUATOR_PROMPT` contains generic wording for
     `猜测对象`, `当前角色`, `偏好`, `我想看`, and `必须驳回`.
   - Run:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_evaluator_prompt_rejects_guess_owner_flip -q`
   - Expected before implementation: fail because the current prompt lacks the
     owner-preservation gate.

2. Parent establishes the live negative regression.
   - File: `tests/test_dialog_evaluator_live_llm_contract.py`
   - Add `test_live_dialog_evaluator_rejects_owner_flipped_guess_target`.
   - Use generic anchors:

```python
[
    '[DECISION] 接受推荐请求但设置获取门槛',
    '[ANSWER] 确认还有其他推荐，但要求对方先猜对一个类型或展示诚意作为交换条件',
    '[SOCIAL] 维持轻微挑衅和博弈感，不直接满足索取行为',
    '[PROGRESSION] 将话题引向“猜类型换推荐”的互动环节',
    '[SCOPE] ~40字，覆盖DECISION、ANSWER、SOCIAL、PROGRESSION',
]
```

   - Use invalid generic final dialog:

```python
[
    '不如先猜猜我会想看哪种类型？',
    '猜中了再给清单。',
]
```

   - Expected after final implementation: `result["should_stop"] is False`
     and evaluator feedback is not `Passed`.
   - Run before implementation:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_owner_flipped_guess_target -q -s`
   - Expected before implementation: fail with the current evaluator passing
     the owner-flipped dialog, matching the POC evidence.

3. Parent establishes the live positive control.
   - File: `tests/test_dialog_evaluator_live_llm_contract.py`
   - Add `test_live_dialog_evaluator_accepts_owner_preserving_guess_condition`.
   - Use the same anchor family as Step 2.
   - Use valid generic final dialog:

```python
[
    '不如再猜一个类型试试看？',
    '猜中了再给清单。',
]
```

   - Expected after final implementation: `result["should_stop"] is True`
     and evaluator feedback is `Passed`.
   - Run before implementation:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_accepts_owner_preserving_guess_condition -q -s`
   - Expected before implementation: pass or provide trace evidence that the
     control wording under-covers anchors. If it fails from under-coverage,
     revise only the control final dialog to cover the same anchors while
     preserving owner-correct guessing.

4. Production-code subagent edits only `_DIALOG_EVALUATOR_PROMPT`.
   - Add a hard gate under `# Hard Gates`:

```text
- 指代与动作所有权：如果 `content_anchors` 只要求对方猜类型、标签、条件、门槛或要看的类别，`final_dialog` 不得改成猜当前角色想看、喜欢、偏好或口味。除非 `content_anchors` 明确说明猜测对象是当前角色的偏好，否则含有“猜我”“我想看”“我喜欢”“我的口味/偏好”等等价表达的猜测句必须驳回，并在 `feedback` 中说明猜测对象或偏好所有者被改写。
```

   - Reread the whole evaluator prompt and adjust nearby wording only when
     required for coherence.
   - Do not edit `_DIALOG_GENERATOR_PROMPT`.

5. Parent reruns focused deterministic verification.
   - Run:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_evaluator_prompt_rejects_guess_owner_flip -q`
   - Expected after implementation: pass.

6. Parent reruns live LLM verification one case at a time.
   - Run and inspect:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_owner_flipped_guess_target -q -s`
   - Run and inspect:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_accepts_owner_preserving_guess_condition -q -s`
   - Record trace paths, raw feedback, and judgment in `Execution Evidence`.

7. Parent runs adjacent regression verification.
   - Run:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q`
   - Run one by one and inspect:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_unanchored_concrete_claim -q -s`
   - Run one by one and inspect:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_evaluator_rejects_stale_topic_dialog -q -s`
   - Run:
     `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py tests\test_dialog_evaluator_live_llm_contract.py`

8. Parent starts the independent code review gate after verification passes.
   - Review the plan, diff, live traces, deterministic test output, and prompt
     wording.
   - Fix only findings inside the approved change surface.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  does not edit tests unless the parent explicitly directs it; closes after
  planned production code changes are complete, excluding review fixes.
- Parent agent may continue live test traces, regression tests, static checks,
  and validation work while the production-code subagent edits production
  code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused test contract established
  - Covers: Implementation Order steps 1-3.
  - Files: `tests/test_dialog_agent.py`,
    `tests/test_dialog_evaluator_live_llm_contract.py`.
  - Verify: run the focused deterministic test and both live LLM cases one at
    a time with `-s`.
  - Evidence: record expected deterministic failure, expected live negative
    failure, positive-control baseline, and trace paths.
  - Sign-off: `Codex/2026-05-27` after evidence is recorded.

- [x] Stage 2 - evaluator prompt hard gate implemented
  - Covers: Implementation Order step 4.
  - File: `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
  - Verify: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py`.
  - Evidence: record changed prompt section and syntax result.
  - Sign-off: `Codex/2026-05-27` after verification and evidence are recorded.

- [x] Stage 3 - focused behavior verification passed
  - Covers: Implementation Order steps 5-6.
  - Verify: focused deterministic test passes; live negative rejects
    owner-flipped dialog; live positive accepts owner-correct dialog.
  - Evidence: record command output, trace paths, raw evaluator feedback, and
    human quality judgment.
  - Sign-off: `Codex/2026-05-27` after verification and evidence are recorded.

- [x] Stage 4 - adjacent regression verification passed
  - Covers: Implementation Order step 7.
  - Verify: adjacent deterministic dialog tests, existing live evaluator
    contract case, existing stale-topic evaluator case, and `py_compile` pass.
  - Evidence: record command output and live trace judgments.
  - Sign-off: `Codex/2026-05-27` after verification and evidence are recorded.

- [x] Stage 5 - independent code review complete
  - Covers: Implementation Order step 8.
  - Verify: independent code-review subagent reviews plan, diff, tests, traces,
    prompt wording, and execution evidence.
  - Evidence: record findings, fixes, rerun commands, residual risks, and
    approval status.
  - Sign-off: `Codex/2026-05-27` after review evidence is recorded.

## Execution Evidence

### Stage 1 - focused test contract

- `venv\Scripts\python.exe -m py_compile tests\test_dialog_agent.py tests\test_dialog_evaluator_live_llm_contract.py`
  passed after adding the deterministic and live test code.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_evaluator_prompt_rejects_guess_owner_flip -q`
  failed as expected because the current evaluator prompt does not contain
  `指代与动作所有权`.
- First live negative baseline run reached the evaluator fixture and failed
  before the LLM call because `dialog_usage_mode` was missing from the live
  evaluator state. The fixture was updated inside
  `tests/test_dialog_evaluator_live_llm_contract.py`.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_owner_flipped_guess_target -q -s -m live_llm`
  failed as expected against the current prompt: raw evaluator output returned
  `feedback=Passed` and `should_stop=true` for the owner-flipped generic
  dialog. Trace:
  `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__reject_owner_flipped_guess_target.json`.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_accepts_owner_preserving_guess_condition -q -s -m live_llm`
  passed against the current prompt: raw evaluator output returned
  `feedback=Passed` and `should_stop=true` for the owner-correct generic
  control. Trace:
  `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__accept_owner_preserving_guess_condition.json`.

### Stage 2 - evaluator prompt hard gate

- Production worker changed only
  `src\kazusa_ai_chatbot\nodes\dialog_agent.py`, inside
  `_DIALOG_EVALUATOR_PROMPT`.
- The evaluator prompt constant now uses a triple-single-quoted string and
  keeps `.format(...)` named placeholders for
  `{ltp_hesitation_density_rule}` and `{mbti_dialog_preference}`.
- Added a generic `指代与动作所有权` hard gate under `# 硬门槛`, plus matching
  pass-condition and audit-order wording. The prompt text uses generic
  semantic classes and does not embed real user message text.
- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py`
  passed in the production worker and passed again in parent verification.
- Independent review found no critical issues. Important findings were:
  update brittle prompt-heading tests to Chinese headings, record lifecycle
  evidence after the prompt edit, and keep the unrelated untracked
  `development_plans\active\bugfix\decontextualizer_unresolved_referent_retry_bugfix_plan.md`
  outside this bugfix scope.
- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py tests\test_dialog_evaluator_live_llm_contract.py`
  passed after the test expectation update.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_evaluator_prompt_rejects_guess_owner_flip tests\test_dialog_agent.py::test_dialog_evaluator_prompt_orders_hard_gates_before_style tests\test_dialog_agent.py::test_dialog_prompts_use_content_anchors_as_semantic_authority -q`
  passed after updating the deterministic tests to the localized evaluator
  headings and new topic-owner wording.

### Stage 3 - focused behavior verification

- `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_evaluator_prompt_rejects_guess_owner_flip -q`
  passed.
- First post-edit live negative run failed with the initial owner hard gate:
  evaluator still returned `feedback=Passed`, `should_stop=true`. Trace:
  `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__reject_owner_flipped_guess_target__20260527T113356365135Z.json`.
- A pronoun-baseline refinement was added to make explicit that
  `final_dialog` is spoken by the current character and that “我/我的” refers
  to the current character. Syntax and the focused deterministic prompt tests
  passed after this refinement.
- Second post-edit live negative run still failed after the pronoun baseline:
  evaluator again returned `feedback=Passed`, `should_stop=true`. Trace:
  `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__reject_owner_flipped_guess_target__20260527T113540123349Z.json`.
- A short generic `# 硬失败速查` checklist was added before the pass
  conditions. It states that type/tag/category/condition/unlock/sincerity
  guessing gates default to the addressed user and do not authorize guessing
  the current character's preference.
- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py tests\test_dialog_evaluator_live_llm_contract.py`
  passed after the final prompt refinement.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_evaluator_prompt_rejects_guess_owner_flip tests\test_dialog_agent.py::test_dialog_evaluator_prompt_orders_hard_gates_before_style tests\test_dialog_agent.py::test_dialog_prompts_use_content_anchors_as_semantic_authority -q`
  passed after the final prompt refinement.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_owner_flipped_guess_target -q -s -m live_llm`
  passed. Raw evaluator output returned `should_stop=false` and feedback
  identifying `猜测对象/偏好所有者被改写`. Trace:
  `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__reject_owner_flipped_guess_target__20260527T113644572964Z.json`.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_accepts_owner_preserving_guess_condition -q -s -m live_llm`
  passed. Raw evaluator output returned `feedback=Passed` and
  `should_stop=true`. Trace:
  `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__accept_owner_preserving_guess_condition__20260527T113654458303Z.json`.

### Stage 4 - adjacent regression verification

- `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q`
  passed: 21 tests passed.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_unanchored_concrete_claim -q -s -m live_llm`
  passed. Raw evaluator output returned `should_stop=false` and rejected the
  unsupported concrete range claim. Trace:
  `test_artifacts\llm_traces\dialog_evaluator_live_llm_contract__reject_unanchored_model_range.json`.
- Initial stale-topic adjacent live run failed before the LLM call because
  `tests\test_dialog_anchor_boundary_live_llm.py` built a direct evaluator
  state without required `dialog_usage_mode`. The plan change surface was
  expanded to allow this test-fixture repair only.
- `venv\Scripts\python.exe -m py_compile tests\test_dialog_anchor_boundary_live_llm.py`
  passed after adding `dialog_usage_mode` to that fixture.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_evaluator_rejects_stale_topic_dialog -q -s -m live_llm`
  passed. Raw evaluator feedback rejected the stale milk-tea topic as
  unauthorized by current anchors. Trace:
  `test_artifacts\llm_traces\dialog_anchor_boundary_live_llm__evaluator_accepts_stale_milk_tea_dialog__20260527T113834907876Z.json`.
- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py tests\test_dialog_evaluator_live_llm_contract.py tests\test_dialog_anchor_boundary_live_llm.py`
  passed.
- `git diff --name-only` showed only:
  `development_plans/README.md`,
  `src/kazusa_ai_chatbot/nodes/dialog_agent.py`,
  `tests/test_dialog_agent.py`,
  `tests/test_dialog_anchor_boundary_live_llm.py`, and
  `tests/test_dialog_evaluator_live_llm_contract.py`.
  `git status --short` also shows unrelated untracked
  `development_plans/active/bugfix/decontextualizer_unresolved_referent_retry_bugfix_plan.md`;
  it remains outside this bugfix scope.
- Human-readable live LLM review artifact:
  `test_artifacts/dialog_evaluator_owner_boundary_live_review_20260527.md`.

### Stage 5 - independent code review

- Final independent review found no blocking findings for the dialog evaluator
  owner-boundary fix.
- Reviewer confirmed the production change is scoped to
  `_DIALOG_EVALUATOR_PROMPT`, uses the required triple-single prompt constant,
  keeps the payload and output contract unchanged, uses ordinary Chinese
  prompt instructions, and does not add deterministic filters, fields, model
  calls, retries, helper paths, or private incident examples.
- Reviewer confirmed the `# 硬失败速查` section is acceptable generic prompt
  text and not user-specific.
- Reviewer confirmed the live tests remain structurally valid: added
  `dialog_usage_mode` fields are state requirements for direct evaluator
  calls, not model-facing payload changes.
- Low residual finding: `development_plans/README.md` also contains an
  unrelated decontextualizer draft row, and
  `development_plans/active/bugfix/decontextualizer_unresolved_referent_retry_bugfix_plan.md`
  remains an unrelated untracked file. It is outside this dialog evaluator
  bugfix scope and was not changed by this implementation.
- Residual risk accepted for this plan: normal live-LLM variance and possible
  over-rejection of more complex quoted-pronoun cases.

## Verification

### Focused Deterministic Test

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_evaluator_prompt_rejects_guess_owner_flip -q
```

Expected after implementation: pass.

### Focused Live LLM Tests

Run one at a time with `-s` and inspect the emitted trace before running the
next case:

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_owner_flipped_guess_target -q -s
```

Expected after implementation: evaluator rejects the owner-flipped final
dialog, feedback is not `Passed`, and `should_stop` is `False`.

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_accepts_owner_preserving_guess_condition -q -s
```

Expected after implementation: evaluator accepts the owner-correct final
dialog, feedback is `Passed`, and `should_stop` is `True`.

### Adjacent Deterministic Regression

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q
```

Expected after implementation: pass.

### Adjacent Live LLM Regression

Run one at a time with `-s` and inspect traces:

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_evaluator_live_llm_contract.py::test_live_dialog_evaluator_rejects_unanchored_concrete_claim -q -s
```

Expected after implementation: evaluator still rejects unsupported concrete
claims.

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_evaluator_rejects_stale_topic_dialog -q -s
```

Expected after implementation: evaluator still rejects stale-topic dialog
against current anchors.

### Syntax

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py tests\test_dialog_evaluator_live_llm_contract.py
```

Expected after implementation: exit code 0.

### Static Scope Check

```powershell
git diff --name-only
```

Expected after implementation: changed files are limited to:

```text
src/kazusa_ai_chatbot/nodes/dialog_agent.py
tests/test_dialog_agent.py
tests/test_dialog_anchor_boundary_live_llm.py
tests/test_dialog_evaluator_live_llm_contract.py
development_plans/active/bugfix/dialog_evaluator_guess_owner_boundary_bugfix_plan.md
development_plans/README.md
```

Additional generated live trace artifacts under `test_artifacts/` are allowed
if produced by the test harness.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/context leaks, brittle fixtures,
  and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including POC evidence, focused and
  regression tests, live trace inspection, and lifecycle evidence.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface. If a fix would cross the approved boundary or
alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The deterministic prompt-contract test proves `_DIALOG_EVALUATOR_PROMPT`
  contains a generic owner-preservation hard gate.
- The live negative regression shows production evaluator rejects the generic
  owner-flipped final dialog against generic recommendation-gate anchors.
- The live positive control shows production evaluator accepts the generic
  owner-correct final dialog against the same anchor family.
- Existing adjacent evaluator checks still pass.
- No new LLM call, field, helper agent, deterministic user-text filter, or
  compatibility path is added.
- No real user message is embedded into runtime prompt text.
- Independent code review is complete and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Evaluator over-rejects valid playful gates | Keep positive control with owner-correct guessing condition. | `test_live_dialog_evaluator_accepts_owner_preserving_guess_condition` |
| Prompt becomes user-example-shaped | Use generic semantic class wording only. | Independent code review inspects prompt text. |
| Existing evaluator checks weaken | Preserve existing hard gates and run adjacent live regressions. | Existing unanchored-claim and stale-topic live tests |
| Live LLM variance hides the issue | Run live cases one at a time and inspect trace artifacts. | Trace paths and raw feedback recorded in `Execution Evidence` |
