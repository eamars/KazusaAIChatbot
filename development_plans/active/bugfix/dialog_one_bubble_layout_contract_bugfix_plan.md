# dialog one bubble layout contract bugfix plan

## Summary

- Goal: align the dialog prompt contract with the actual one-bubble delivery
  path while preserving technical and code-block formatting.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: bigbang prompt wording replacement with the
  existing `final_dialog` JSON shape preserved.
- Highest-risk areas: accidentally implementing multi-message delivery,
  adding hard line budgets, breaking code-block formatting, letting the
  evaluator reject required code fences, and overfitting one group-chat
  incident instead of fixing the dialog output contract.
- Acceptance criteria: prompt-only production diff, deterministic prompt
  contract tests pass, and at least five one-at-a-time real LLM dialog tests
  produce inspected traces that satisfy the one-bubble layout contract.
- Current execution state: draft only; this plan does not authorize production
  prompt edits until the user explicitly approves implementation.

## Context

The current dialog generator prompt still describes output as if Kazusa were
sending several message fragments:

```text
模拟真人在聊天平台上打一段、发一段的节奏感
```

and later says each `final_dialog` element is a sendable line fragment. The
runtime does not deliver those fragments as independent platform messages. The
brain service, QQ adapter, and Discord adapter join the list with newline
characters and deliver one visible chat bubble.

The observed failure mode is a prompt-to-runtime contract mismatch, not an
adapter delivery requirement. The user explicitly rejected implementing a
multi-message send mechanism for this fix, especially because it does not fit
busy group-chat delivery. The user also rejected explicit line budgets because
technical answers, RCA notes, comparisons, and code answers need enough visible
space to complete the requested delivery.

The historical QQ Jigsaw sudoku-code reply from `2026-06-05 14:20 +12:00` is
a positive formatting reference. Kazusa used character voice before and after
the Python body, preserved code indentation in the stored `body_text`, and did
not inject persona wording inside the code body. That behavior is the target
for fixed-format blocks, with fenced code blocks preferred for safer rendering.

This plan is limited to the dialog prompt layer. The content-selection issue
from self-cognition and group broadcast routing is owned elsewhere.

## Mandatory Skills

- `development-plan`: load before plan lifecycle edits, execution, review, or
  lifecycle status updates.
- `local-llm-architecture`: load before changing prompt wording, LLM contracts,
  evaluator instructions, or live LLM verification design.
- `py-style`: load before editing `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
- `cjk-safety`: load before editing Python files that contain Chinese prompt
  strings; run syntax verification immediately after prompt edits.
- `test-style-and-execution`: load before adding, changing, or running
  deterministic or real LLM tests.
- `debug-llm`: load before adding live LLM trace artifacts, running real LLM
  cases, or judging LLM output quality.

## Mandatory Rules

- Do not touch production code unless this plan is `approved` or `in_progress`
  and the user has explicitly authorized implementation.
- The production change surface is prompt-only: edit only
  `_DIALOG_GENERATOR_PROMPT` and `_DIALOG_EVALUATOR_PROMPT` in
  `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
- Do not edit adapters, brain-service delivery, `post_turn`, queueing,
  persistence, JSON parsing, retry counts, event logging, normalizers, RAG,
  cognition, L3 surface generation, self-cognition, dispatcher, scheduler, or
  database code.
- Preserve the output JSON contract exactly: top-level object with
  `final_dialog: list[str]` and `mention_target_user: bool`.
- Treat `final_dialog` as one visible bubble after newline joining. Its
  elements are layout units inside the bubble, not separate platform sends.
- Do not add a hard maximum or minimum number of `final_dialog` elements,
  visible lines, sentences, characters, or paragraphs.
- Preserve anchor coverage over compactness. Technical deliveries must keep
  required facts, numbers, steps, risks, code, or examples.
- Do not implement multi-message send, batching, coalescing, post-generation
  formatter logic, adapter-specific branching, or deterministic line wrapping.
- Fixed-format blocks are protected islands. Code blocks, JSON examples, CLI
  output, logs, stack traces, config snippets, and tables required by anchors
  must preserve indentation, line order, blank lines, symbols, and fence
  markers.
- Character voice belongs outside fixed-format blocks. Do not insert filler,
  roleplay voice, comments, or extra prose inside code or fixed-format data
  unless the anchors explicitly require those exact lines.
- A fixed-format block must remain intact inside one `final_dialog` string.
  Prose before or after the block can be in the same string or adjacent layout
  strings, but code lines must not be split into separate semantic fragments.
- Decorative Markdown remains disallowed for ordinary prose. Fenced code blocks
  are allowed only when anchors require code or fixed-format output inside
  `final_dialog`; the model must still return JSON, not a top-level Markdown
  answer.
- Real LLM tests must run one test case at a time with `-q -s`, and each trace
  must be inspected before running the next real LLM case.
- Real LLM assertions must be structural and contract-focused. They must not
  fail solely because a response has more than a preferred number of lines.
- Use `venv\Scripts\python.exe` for Python commands. Do not read `.env`.
- Use `apply_patch` for manual edits.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.

## Must Do

- Replace multi-send wording in the dialog generator prompt with an explicit
  one-bubble layout contract.
- State that runtime joins `final_dialog` with newline characters into one
  visible bubble.
- Define layout guidance as semantic line grouping: casual replies can be
  compact, while technical deliveries can use multiple lines to group complete
  facts, steps, code, examples, or conclusions.
- Preserve the current semantic authority boundary: `content_anchors` remain
  the only source of facts, decisions, stance, and response actions.
- Preserve the current JSON output shape and mention-target semantics.
- Update the evaluator prompt so it audits one-bubble layout fitness without
  enforcing exact line counts.
- Update the evaluator prompt so it accepts fenced code blocks when anchors
  require code or fixed-format output, and rejects broken or voice-contaminated
  fixed-format blocks.
- Add deterministic prompt-text tests that prove the new prompt contract is
  present and the old multi-send wording is gone.
- Add at least five real LLM tests using L3-shaped `action_directives`:
  1. group casual direct reply;
  2. private conversation reply;
  3. group technical comparison with many numeric facts;
  4. historical Jigsaw-style Python code reply;
  5. fixed-format JSON input example reply.
- Run the real LLM tests one at a time and inspect each trace before judging
  the contract complete.

## Deferred

- No multi-message delivery mechanism.
- No adapter changes.
- No service response joining changes.
- No `final_dialog` schema change.
- No line-count, segment-count, or character-count budget.
- No deterministic output formatter, code-block parser, Markdown renderer, or
  post-generation repair pass.
- No cognition, L2d, L3 content-selection, self-cognition, RAG, memory,
  reflection, dispatcher, scheduler, or database changes.
- No content correctness fix for bad upstream anchors. If anchors contain
  wrong facts or incomplete code, this plan only preserves and renders the
  provided anchors; content ownership remains upstream.
- No model route, temperature, retry, timeout, or provider configuration
  changes.
- No documentation updates outside this plan and the development-plan registry.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Dialog generator prompt | bigbang | Replace multi-send and fragment-send wording with one-bubble layout wording. |
| Dialog evaluator prompt | bigbang | Add one-bubble layout and fixed-format-block audit rules without adding line budgets. |
| Output schema | compatible | Preserve `final_dialog: list[str]` and `mention_target_user: bool`. |
| Runtime delivery | no-op | Keep existing newline join behavior. |
| Code/fixed-format blocks | bigbang | Allow required fenced code blocks inside `final_dialog` while keeping top-level JSON output. |
| Tests | bigbang | Add focused deterministic prompt tests and five real LLM layout cases. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- Bigbang prompt areas must replace old multi-send wording directly instead
  of preserving it as an alternate path.
- Compatible schema areas preserve only the existing `final_dialog` and
  `mention_target_user` surfaces listed in this plan.
- No compatibility shim, adapter fallback, second delivery path, parser repair,
  or formatter may be added to preserve the old fragment-send interpretation.
- Any change to a cutover policy requires user approval before implementation.

## Target State

```text
L3 text surface action_directives
  -> dialog generator prompt renders one-bubble layout units
  -> dialog evaluator checks anchor fidelity plus one-bubble readability
  -> existing final_dialog list
  -> existing newline join
  -> one visible platform bubble with fixed-format blocks preserved
```

The target behavior keeps casual group replies compact, private replies
natural, technical replies complete, and code/fixed-format blocks intact.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Delivery model | One visible bubble | This matches the current service and adapter behavior. |
| `final_dialog` meaning | Layout units, not send events | Keeps the schema while removing false multi-send semantics. |
| Line control | No explicit budget | Technical deliveries need flexible length. |
| Layout guidance | Semantic grouping | Local LLMs need concrete grouping language without numeric caps. |
| Code blocks | Protected fixed-format island | Code readability depends on exact whitespace and line order. |
| Markdown | Fenced code only when required | Ordinary prose should not gain decorative Markdown. |
| Evaluator role | Audit layout and block preservation | The evaluator should catch broken formatting without becoming a formatter. |
| Runtime logic | Unchanged | Prompt-only scope avoids adapter and service blast radius. |
| Live tests | Full dialog with simulated L3 directives | This verifies the prompt against the shape L3 actually passes downstream. |

## Contracts And Data Shapes

Runtime output remains:

```python
{
    "final_dialog": list[str],
    "mention_target_user": bool,
}
```

The prompt-facing interpretation changes to:

```text
final_dialog:
  list of visible layout units that will be joined with "\n" into one bubble.
```

For ordinary prose, elements should be semantically grouped lines or
paragraphs:

```text
claim or stance
supporting fact group
boundary or caveat
short conclusion
```

For fixed-format blocks, the block stays whole inside one string:

````markdown
brief character-voice prose
```python
def example() -> None:
    print("kept")
```
brief closing prose
````

The model must not place roleplay filler, casual interjections, or explanatory
voice inside the fenced block unless those exact lines are part of the
content anchors.

## LLM Call And Context Budget

- Runtime LLM budget is unchanged.
- The dialog generator remains one `DIALOG_GENERATOR_LLM` call per attempt.
- The dialog evaluator remains one `DIALOG_EVALUATOR_LLM` call per attempt.
- Existing retry behavior remains unchanged.
- No new LLM routes, repair calls, summarizers, post-processors, or helper
  agents are added.
- The prompt receives the same state shape: `linguistic_directives`,
  `contextual_directives`, and `user_name`.
- The five real LLM tests use simulated L3 `action_directives` shaped like the
  selected text surface output. They do not run L1, L2, L2d, RAG, adapters, or
  database retrieval.
- Each real LLM test must write a durable trace under `test_artifacts` using
  the existing `tests.llm_trace.write_llm_trace` helper.
- Because the live test file exercises the full `dialog_agent` path, its
  endpoint skip helper must check both the `DIALOG_GENERATOR_LLM` and
  `DIALOG_EVALUATOR_LLM` route endpoints before running cases.
- Each real LLM test must build and trace `joined_dialog = "\n".join(final_dialog)`
  because that joined string is the visible one-bubble surface users see.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Edit `_DIALOG_GENERATOR_PROMPT` only for one-bubble layout, semantic line
    grouping, and fixed-format-block preservation.
  - Edit `_DIALOG_EVALUATOR_PROMPT` only for one-bubble layout audit,
    fixed-format-block acceptance, and broken-block rejection.

- `tests/test_dialog_agent.py`
  - Add deterministic prompt-text tests:
    - `test_dialog_generator_prompt_describes_one_bubble_layout_contract`
    - `test_dialog_prompts_preserve_fixed_format_blocks`
    - `test_dialog_evaluator_prompt_audits_layout_without_line_budget`
  - Update existing prompt-text assertions only when they assert the old
    multi-send or Markdown-ban wording.
  - Keep or add assertions that decorative Markdown and top-level Markdown
    answers remain disallowed, while required fenced code blocks inside JSON
    string values are allowed.

- `tests/test_dialog_one_bubble_layout_live_llm.py`
  - Create a focused real LLM test file with one test function per scenario.
  - Use simulated L3 `action_directives` and the full `dialog_agent` path.
  - Write one trace per case.
  - Write the trace before case-specific quality assertions so failed baseline
    or post-change cases still leave inspectable raw evidence.
  - Include these trace fields for every case: case id, model route/base URL
    for generator and evaluator, simulated L3 action directives, user name,
    contextual directives, raw result, `final_dialog`, `joined_dialog`,
    `mention_target_user`, structural validation results, and manual
    inspection notes placeholder.

- `development_plans/README.md`
  - Add this draft plan to the active bugfix registry.

### Keep

- No other production files.
- No adapter files.
- No brain-service files.
- No L3 surface-generation files.
- No database or migration files.
- No environment files.

## Overdesign Guardrail

The implementation must stop and return to the user if satisfying the live LLM
cases appears to require any of these changes:

- deterministic line wrapping;
- deterministic code-block parsing or repair;
- extra LLM repair calls;
- adapter-specific delivery logic;
- multiple platform sends;
- schema changes to `final_dialog`;
- L3, cognition, RAG, or self-cognition changes;
- hard-coded special handling for Jigsaw, QQ, or one historical incident.

The allowed response to a weak local model misunderstanding is clearer prompt
wording and focused live evidence, not new runtime machinery.

## Agent Autonomy Boundaries

- The parent agent owns tests, live LLM trace inspection, execution evidence,
  plan lifecycle updates, review feedback handling, and final sign-off.
- The production-code subagent owns only the approved prompt edits inside
  `_DIALOG_GENERATOR_PROMPT` and `_DIALOG_EVALUATOR_PROMPT`.
- The production-code subagent must report changed files, exact prompt sections
  changed, commands run, and residual risks before closing.
- If native subagent capability is unavailable at execution time, stop before
  production prompt edits unless the user explicitly approves fallback
  execution.
- Review fixes are allowed only inside the approved change surface. Any review
  finding that requires deterministic code or another subsystem must be
  reported as out of scope.

## Implementation Order

1. Confirm execution authorization.
   - Verify this plan status is `approved` or `in_progress`.
   - Verify the user has explicitly authorized prompt implementation.
   - Run: `git status --short`
   - Evidence: record status and dirty-worktree notes in `Execution Evidence`.

2. Load required skills and reread local contracts.
   - Load `development-plan`, `local-llm-architecture`, `py-style`,
     `cjk-safety`, `test-style-and-execution`, and `debug-llm`.
   - Reread `development_plans/README.md`.
   - Evidence: record the skill list and registry check in
     `Execution Evidence`.

3. Add deterministic prompt-contract tests before prompt edits.
   - File: `tests/test_dialog_agent.py`
   - Add `test_dialog_generator_prompt_describes_one_bubble_layout_contract`.
   - Add `test_dialog_prompts_preserve_fixed_format_blocks`.
   - Add `test_dialog_evaluator_prompt_audits_layout_without_line_budget`.
   - Verify before implementation:
     `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_generator_prompt_describes_one_bubble_layout_contract tests\test_dialog_agent.py::test_dialog_prompts_preserve_fixed_format_blocks tests\test_dialog_agent.py::test_dialog_evaluator_prompt_audits_layout_without_line_budget -q`
   - Expected before implementation: failures proving the old prompt lacks the
     new one-bubble and fixed-format-block wording, or still contains old
     multi-send wording.
   - Evidence: record the failing assertion summary.

4. Add live LLM one-bubble test scaffold before prompt edits.
   - File: `tests/test_dialog_one_bubble_layout_live_llm.py`
   - Reuse the live endpoint skip pattern from
     `tests/test_dialog_generator_live_llm_contract.py`.
   - Extend the skip helper to check both `DIALOG_GENERATOR_LLM_BASE_URL` and
     `DIALOG_EVALUATOR_LLM_BASE_URL`, because the full `dialog_agent` path uses
     both routes.
   - Use `tests.llm_trace.write_llm_trace`.
   - Build full dialog states with simulated L3 `action_directives`.
   - Do not parameterize the five real LLM cases.
   - Add a small helper that returns `joined_dialog = "\n".join(final_dialog)`
     after asserting `final_dialog` is a non-empty `list[str]`.
   - Add a small helper that writes the trace before any case-specific content
     assertion runs.
   - Evidence: record the new test names and trace payload fields.

5. Add live LLM case 1: group casual direct reply.
   - Test name: `test_live_dialog_one_bubble_group_casual_reply`
   - Simulated L3 context: QQ group, direct mention, light social distance,
     low emotional intensity, noisy but friendly group.
   - Content anchors:
     - `[DECISION] 直接回应 Jigsaw 的轻量协作问题`
     - `[ANSWER] 先按用途把东西分成充电、视频输出、待确认三类`
     - `[SOCIAL] 语气轻松，适合群里一眼读完`
     - `[SCOPE] 简短但完整`
   - Required inspection: one bubble reads like a direct group reply, does not
     look like separate sends, and does not add a new question.

6. Add live LLM case 2: private conversation reply.
   - Test name: `test_live_dialog_one_bubble_private_soft_reply`
   - Simulated L3 context: private chat, warmer relationship, user is anxious
     about whether a small plan is acceptable.
   - Content anchors:
     - `[DECISION] 接住用户的不确定感并给出明确结论`
     - `[ANSWER] 这个计划可以先按今晚版本执行，明早再复查`
     - `[SOCIAL] 私聊里可以比群聊多一点安抚，但不要撒娇过度`
     - `[SCOPE] 中短`
   - Required inspection: one bubble is natural for private chat and does not
     become a formal assistant checklist.

7. Add live LLM case 3: group technical comparison.
   - Test name: `test_live_dialog_one_bubble_group_technical_comparison`
   - Simulated L3 context: QQ group technical discussion, direct answer to a
     GPU comparison request.
   - Content anchors:
     - `[DECISION] 正面对比 GB300 和 Pro6000`
     - `[FACT] GB300: FP16 2250 TFLOPS, FP8 4500 TFLOPS, 288GB HBM3e, 12000 GB/s, TDP 1400W, FP32 90 TFLOPS`
     - `[FACT] Pro6000: FP16 125 TFLOPS, FP8 2000 TFLOPS, 96GB GDDR7, 约1792 GB/s, TDP 400W, FP32 125 TFLOPS`
     - `[ANSWER] GB300 更适合超大规模训练和推理；Pro6000 更适合工作站或较小规模推理`
     - `[SCOPE] 信息密度优先，允许多行完成对比`
   - Required inspection: all numeric facts appear, the reply groups facts
     semantically, and it does not use a hard-coded tiny response.

8. Add live LLM case 4: historical Jigsaw-style Python code reply.
   - Test name: `test_live_dialog_one_bubble_python_code_block_preserved`
   - Simulated L3 context: QQ group, Jigsaw asks Kazusa for a Python sudoku
     solver.
   - Content anchors:
     - `[DECISION] 交付 Python 数独求解器`
     - `[ANSWER] 使用回溯法，输入是 9 行字符串，0 表示空格`
     - `[ANSWER] 必须输出一个 fenced python code block`
     - `[FACT] 代码块内必须保留 def solve_sudoku(board):、find_empty、is_valid、solve，以及缩进`
     - `[SOCIAL] 角色语气只能放在代码块外`
     - `[SCOPE] 代码块优先保持格式，不压缩代码`
   - Required inspection: fenced Python block exists, indented lines remain
     visible, and no character voice appears inside the fence.

9. Add live LLM case 5: fixed-format JSON input example.
   - Test name: `test_live_dialog_one_bubble_json_example_preserved`
   - Simulated L3 context: QQ group follow-up asking for the sudoku input text
     format and one complete example.
   - Content anchors:
     - `[DECISION] 给出输入格式和完整例子`
     - `[ANSWER] JSON 顶层键是 puzzle`
     - `[ANSWER] puzzle 必须包含 9 行，每行 9 个数字，0 表示空位`
     - `[ANSWER] 必须输出完整 9x9 JSON fenced block`
     - `[SOCIAL] 可以在块外承认刚才说得不完整，但不要把歉意写进 JSON`
     - `[SCOPE] 完整例子优先，不能只给三行`
   - Required inspection: JSON block has 9 visible rows, no voice text appears
     inside the JSON, and the surrounding prose stays in one bubble.

10. Run one baseline live LLM case before prompt edits.
    - Command:
      `venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_python_code_block_preserved -q -s`
    - Expected before implementation: the case records current behavior and
      can fail structurally or qualitatively if the old prompt mishandles code
      fences or fragment semantics.
    - Evidence: record trace path and manual judgment. If the command fails,
      inspect whether the trace was written before failure; if no trace exists,
      fix only the test trace-ordering helper before continuing.

11. Start the production-code subagent for prompt-only edits.
    - Provide this plan, the deterministic failing tests, the baseline live
      trace, mandatory skills, and the exact production ownership boundary.
    - The subagent may edit only `_DIALOG_GENERATOR_PROMPT` and
      `_DIALOG_EVALUATOR_PROMPT`.
    - Evidence: record subagent summary and changed prompt sections.

12. Edit the dialog generator prompt.
    - Remove or replace the old multi-send phrases:
      `打一段、发一段`, `要发送的台词片段`, `换行节奏只能通过多个 final_dialog 元素表达`,
      and `使用 6-12 个短字符串片段`.
    - Add one-bubble layout wording:
      runtime joins `final_dialog` with newline characters into one visible
      bubble.
    - Add semantic grouping guidance without numeric caps.
    - Add fixed-format-block preservation and code-fence exception wording.
    - Preserve all existing semantic-anchor authority rules.

13. Edit the dialog evaluator prompt.
    - Add a hard gate for broken fixed-format blocks when anchors require code,
      JSON, logs, stack traces, configs, CLI output, or tables.
    - Add an allowance for fenced code blocks inside `final_dialog` when anchors
      require fixed-format output.
    - Add a soft layout check for one-bubble readability.
    - Explicitly state that the evaluator must not fail a reply solely because
      it uses several lines to complete a technical delivery.

14. Run CJK syntax verification.
    - Command:
      `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py`
    - Expected after implementation: pass.
    - Evidence: record command output.

15. Run deterministic prompt tests.
    - Command:
      `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_generator_prompt_describes_one_bubble_layout_contract tests\test_dialog_agent.py::test_dialog_prompts_preserve_fixed_format_blocks tests\test_dialog_agent.py::test_dialog_evaluator_prompt_audits_layout_without_line_budget -q`
    - Expected after implementation: pass.
    - Evidence: record command output.

16. Run the updated existing multi-part prompt contract test.
    - Command:
      `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_prompts_preserve_multi_part_deliverables -q`
    - Expected after implementation: pass with updated expectations that
      reject decorative Markdown but allow required fenced code blocks.
    - Evidence: record command output.

17. Run existing dialog deterministic tests.
    - Command:
      `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q`
    - Expected after implementation: pass.
    - Evidence: record command output.

18. Run real LLM case 1 and inspect trace.
    - Command:
      `venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_group_casual_reply -q -s`
    - Expected: structural assertions pass and manual inspection accepts the
      one-bubble group layout.
    - Evidence: record trace path and judgment.

19. Run real LLM case 2 and inspect trace.
    - Command:
      `venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_private_soft_reply -q -s`
    - Expected: structural assertions pass and manual inspection accepts the
      private-chat layout.
    - Evidence: record trace path and judgment.

20. Run real LLM case 3 and inspect trace.
    - Command:
      `venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_group_technical_comparison -q -s`
    - Expected: structural assertions pass, required GPU facts appear, and
      manual inspection accepts semantic grouping without a line budget.
    - Evidence: record trace path and judgment.

21. Run real LLM case 4 and inspect trace.
    - Command:
      `venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_python_code_block_preserved -q -s`
    - Expected: structural assertions pass, fenced Python formatting is
      preserved, and manual inspection finds no character voice inside code.
    - Evidence: record trace path and judgment.

22. Run real LLM case 5 and inspect trace.
    - Command:
      `venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_json_example_preserved -q -s`
    - Expected: structural assertions pass, the JSON example has nine visible
      rows, and manual inspection finds no character voice inside JSON.
    - Evidence: record trace path and judgment.

23. Run focused diff review.
    - Command:
      `git diff -- src/kazusa_ai_chatbot/nodes/dialog_agent.py tests/test_dialog_agent.py tests/test_dialog_one_bubble_layout_live_llm.py development_plans/README.md development_plans/active/bugfix/dialog_one_bubble_layout_contract_bugfix_plan.md`
    - Expected: production diff is prompt-only; test and plan diffs match this
      plan.
    - Evidence: record review notes.

24. Run independent code review.
    - Scope: prompt-only diff, deterministic prompt tests, live LLM test design,
      trace evidence, and absence of out-of-scope production changes.
    - Expected: no blocking findings, or findings remediated inside approved
      change surface.
    - Evidence: record review result and any remediation commands.

25. Complete lifecycle update after acceptance.
    - Update this plan's `Status` and `Execution Evidence` only after the user
      accepts the verification evidence.
    - Do not mark complete if any required live LLM trace is uninspected,
      skipped without reason, or qualitatively rejected.

## Execution Model

Execution is parent-led and uses native subagents after the parent establishes
the focused test contract. The normal order is:

1. Parent adds deterministic prompt tests and live LLM test scaffolding.
2. Parent records at least one baseline live trace before prompt edits.
3. Parent starts one production-code subagent for prompt-only edits.
4. Parent runs deterministic and real LLM verification one case at a time.
5. Parent starts one independent code-review subagent.
6. Parent remediates review findings only inside the approved change surface.
7. Parent records final evidence and asks the user for lifecycle sign-off.

If native subagent capability is unavailable, execution stops before production
prompt edits unless the user explicitly approves fallback execution.

## Progress Checklist

- [ ] Stage 1 - deterministic prompt contract established
  - Covers: implementation steps 1-3.
  - Files: `tests/test_dialog_agent.py`.
  - Verify: focused deterministic tests fail before prompt implementation.
  - Evidence: record failing assertion summary in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 2 - live LLM scenarios added and baseline recorded
  - Covers: implementation steps 4-10.
  - Files: `tests/test_dialog_one_bubble_layout_live_llm.py`.
  - Verify: one baseline live case runs with a durable trace.
  - Evidence: record trace path and manual judgment.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 3 - prompt-only implementation complete
  - Covers: implementation steps 11-14.
  - Files: `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
  - Verify: CJK syntax verification passes.
  - Evidence: record changed prompt sections and syntax-check output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 4 - deterministic verification complete
  - Covers: implementation steps 15-17.
  - Files: `tests/test_dialog_agent.py`.
  - Verify: focused prompt tests and full dialog-agent deterministic tests pass.
  - Evidence: record command outputs.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 5 - real LLM verification complete
  - Covers: implementation steps 18-22.
  - Files: `tests/test_dialog_one_bubble_layout_live_llm.py`,
    `test_artifacts/llm_traces/*`.
  - Verify: five live LLM cases are run one at a time and each trace is
    manually inspected.
  - Evidence: record trace paths and pass/fail judgment for every case.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 6 - review and lifecycle sign-off complete
  - Covers: implementation steps 23-25.
  - Files: this plan and `development_plans/README.md`.
  - Verify: focused diff review and independent code review complete.
  - Evidence: record review result, remediation, residual risk, and final
    user sign-off.
  - Handoff: none.
  - Sign-off: `<agent/date>` after evidence is recorded.

## Verification

Syntax:

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py
```

Focused deterministic prompt tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_generator_prompt_describes_one_bubble_layout_contract tests\test_dialog_agent.py::test_dialog_prompts_preserve_fixed_format_blocks tests\test_dialog_agent.py::test_dialog_evaluator_prompt_audits_layout_without_line_budget -q
```

Existing deterministic dialog tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py::test_dialog_prompts_preserve_multi_part_deliverables -q
venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q
```

Real LLM tests must run one at a time:

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_group_casual_reply -q -s
venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_private_soft_reply -q -s
venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_group_technical_comparison -q -s
venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_python_code_block_preserved -q -s
venv\Scripts\python.exe -m pytest tests\test_dialog_one_bubble_layout_live_llm.py::test_live_dialog_one_bubble_json_example_preserved -q -s
```

Every real LLM trace must be inspected before the next command runs. A passing
pytest result is insufficient without manual trace judgment.

## Independent Code Review

Run an independent review after deterministic and real LLM verification pass.
The review scope is:

- production diff is limited to `_DIALOG_GENERATOR_PROMPT` and
  `_DIALOG_EVALUATOR_PROMPT`;
- no adapter, service, parser, retry, route, model config, L3, cognition,
  self-cognition, RAG, persistence, dispatcher, scheduler, or database code was
  changed;
- prompt wording explicitly describes one-bubble layout;
- prompt wording does not impose hard line budgets;
- fixed-format block exception is clear and does not allow top-level Markdown
  output;
- deterministic prompt tests prove the intended contract;
- real LLM traces cover the five required scenarios and were inspected one at
  a time;
- review findings are remediated only inside the approved change surface.

Record the independent review result in `Execution Evidence` before lifecycle
completion.

## Acceptance Criteria

- The only production file changed is
  `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
- Inside that file, the only production changes are prompt-string edits to
  `_DIALOG_GENERATOR_PROMPT` and `_DIALOG_EVALUATOR_PROMPT`.
- The `final_dialog` and `mention_target_user` schema remains unchanged.
- The prompts no longer describe `final_dialog` elements as separate platform
  sends.
- The prompts explicitly state that `final_dialog` is joined into one visible
  bubble.
- The prompts use semantic grouping guidance without numeric line or segment
  budgets.
- The prompts protect code and fixed-format blocks from reflow, character
  voice injection, and splitting.
- Deterministic prompt tests pass.
- `tests/test_dialog_agent.py` passes.
- The five real LLM cases produce durable traces and pass both structural
  assertions and manual inspection.
- The technical comparison case preserves required numbers and conclusion.
- The Python code case preserves a fenced code block with visible indentation
  and no character voice inside the code.
- The JSON example case contains a complete 9x9 example and no character voice
  inside the JSON.
- Independent code review reports no unresolved blocking findings.

## Risks

- A local LLM can still over-fragment output despite prompt wording. The
  approved response is further prompt clarification inside this change surface,
  not deterministic formatting.
- Allowing fenced code inside `final_dialog` can conflict with the old Markdown
  ban. The prompt must distinguish top-level Markdown answers and decorative
  Markdown from required fenced code blocks inside JSON strings.
- Real LLM tests can pass structurally while still feeling awkward. Manual
  trace inspection is part of the acceptance gate.
- The evaluator can overcorrect style and damage technical completeness. Live
  tests must inspect final dialog after the evaluator path, not only raw
  generator output.
- Upstream bad anchors can still produce bad content. This plan only fixes
  dialog layout and fixed-format rendering.

## Execution Evidence

Record execution evidence here after implementation starts. Do not pre-fill
evidence during draft planning.
