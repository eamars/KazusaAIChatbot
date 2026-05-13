# task dispatcher json contract bugfix plan

## Summary

- Goal: (1) guarantee `parse_llm_json_output` always returns a `dict`, (2) remove
  the ambiguous task-dispatcher LLM no-op output contract, and (3) add a generic,
  opt-in `expected_output_format` string to the existing LLM JSON-repair fallback.
  The string must be the same output-format contract shown to the original LLM,
  so the repair LLM can reconstruct malformed or wrong-wrapper output against
  the caller's actual prompt contract without turning repair into semantic
  generation.
- Plan class: large
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, `superpowers:test-driven-development`,
  `superpowers:verification-before-completion`,
  `development-plan-writing`
- Overall cutover strategy: compatible for the parser API and return-contract
  hardening; bigbang for the task-dispatcher prompt no-op wording.
- Highest-risk areas: turning generic JSON repair into semantic generation;
  teaching the repair LLM to copy placeholder schema values as real data;
  confusing Python source escaping with the rendered prompt contract when handling
  JSON braces;
  rendering JSON-shaped repair prompt strings with brace-sensitive formatting APIs;
  weakening deterministic validation after parsing; changing the public return
  behavior of `parse_llm_json_output` in a way that surprises existing callers;
  changing unrelated consolidation behavior.
- Acceptance criteria: `parse_llm_json_output` always returns a `dict` (`{}` when
  every attempt fails); `_TASK_DISPATCHER_PROMPT` has one unambiguous no-op output
  shape, `{"tool_calls": []}`; `parse_llm_json_output` and the repair path it uses
  accept an optional, keyword-only `expected_output_format: str | None` and render
  it directly into the LLM-repair prompt template only when the deterministic path
  has already failed;
  `_generate_raw_tool_calls` passes the exact task-dispatcher output-format string
  that is also embedded in the normal task-dispatcher prompt; existing callers that
  omit the new argument keep their current call shape and behavior; no broad
  negative-path malformed-output test matrix is added; focused verification and
  one-by-one inspected live LLM evidence pass.

## Context

Production background consolidation failed in `db_writer` after the
task-dispatcher LLM returned valid JSON `[]`. The stack failed at
`_generate_raw_tool_calls(...): result.get("tool_calls", [])` because
`parse_llm_json_output("[]")` returned a Python list, and `.get` does not exist
on a list. `parse_llm_json_output` is annotated `-> dict` but does not currently
enforce that contract: `repair_json(..., return_objects=True)` and the LLM-repair
fallback can both legitimately return a list, `None`, or a scalar.

There are three independent problems:

1. The parser does not honor its own `-> dict` contract, so any caller that does
   `result.get(...)` can crash on a non-dict result.
2. The task-dispatcher prompt teaches two different no-op contracts:
   - repeated natural-language rules say to return an empty list or empty array
     when no reliable tool call exists (`返回空列表`, `输出空列表`, `返回空数组`);
   - the formal output schema expects an object wrapper containing `tool_calls`.
3. When the deterministic repair (`json_repair`) cannot recover usable JSON, the
   LLM-repair fallback has no information about the output-format contract that
   the original caller showed to the normal LLM, so it cannot reliably rebuild a
   truncated, garbled, or wrong-wrapper payload.

The corresponding LLM stage is `_task_dispatcher_llm` in
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`.
It uses the `CONSOLIDATION_LLM_*` route and converts accepted `future_promises`
into scheduler tool calls after the visible reply has already been produced.

This plan applies three mitigations:

1. Make `parse_llm_json_output` always return a `dict` — `{}` when every repair
   attempt fails. This closes the crash at every call site at once, regardless of
   which failure mode produced the non-dict result; it does not require
   enumerating failure modes.
2. Fix the task-dispatcher prompt so the no-op shape is explicit and
   non-negotiable: `{"tool_calls": []}`, never a top-level array.
3. Extend the existing LLM-repair fallback with an optional, keyword-only
   `expected_output_format` string. The string is the same target output contract
   used in the normal LLM prompt. It is purely a reconstruction hint for the
   escalation path - it never runs on a successful deterministic parse, and it
   never authorizes inventing data.

The deterministic `json_repair` call stays the first line of defense and keeps
handling ordinary malformed JSON (trailing commas, unclosed brackets, bad quotes,
markdown fences). The LLM-repair fallback remains the last resort for the nasty
failures `json_repair` cannot fix (severe truncation, garbled keys/values, wrong
top-level shape); the expected-output-format hint gives that last resort the same
output contract the normal LLM was supposed to follow.

## Mandatory Skills

- `local-llm-architecture`: load before changing prompt contracts, LLM JSON
  repair behavior, consolidation scheduling behavior, or parser-call boundaries.
  Keep local-LLM contracts short, explicit, and validated by deterministic code.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files that contain Chinese prompt
  strings. Run a syntax check after editing CJK-containing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
  Real LLM tests must run one case at a time and be inspected one case at a time.
- `superpowers:test-driven-development`: load before implementation. Use focused
  contract tests for prompt wording, the dict-return guarantee, and
  argument-plumbing behavior, but do not add a broad malformed-output negative-path
  test suite.
- `superpowers:verification-before-completion`: load before claiming the plan is
  complete or passing.
- `development-plan-writing`: load before updating this plan, lifecycle status,
  progress checklist, or execution evidence.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire
  plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run this plan's `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Do not execute this plan while `Status` is `draft`. Execution requires a status
  change to `approved` or `in_progress`.
- `parse_llm_json_output` must always return a `dict`. On total failure it returns
  `{}` and logs the failure; it must never return a list, `None`, or a scalar.
- Do not add a broad negative-path test matrix for malformed LLM outputs. A small
  number of focused contract tests (prompt wording, dict-return guarantee,
  expected-format plumbing) is allowed; an enumeration of arbitrary malformed JSON,
  markdown, wrapping, truncation, or type-mismatch cases is not.
- Do not make JSON repair synthesize semantic tool calls. Repair may fix syntax,
  fix the minimal wrapper shape, and reconstruct against a supplied
  expected-output-format string copied from the original prompt's output-format
  contract - but it must never invent domain rows, copy placeholder example values
  as real data, or fabricate values absent from the raw output.
- The expected-output-format hint is opt-in and escalation-only. Do not pass it on
  the deterministic success path, and do not add retry loops or additional LLM
  calls to the normal successful parse path.
- `expected_output_format` must be a string. It must come from the same output
  format text that the normal LLM sees in that stage's prompt. Do not pass an
  ad hoc dict contract or a separate reduced schema.
- The reusable expected-output-format string must contain only the target output
  contract, not surrounding prompt headings or generic instructions such as
  `# 输出格式` or `请只返回合法 JSON：`.
- The expected output format must be included in the JSON-repair LLM's prompt
  template, not sent as an input-message field alongside the broken JSON.
- The extracted task-dispatcher output-format constant must represent the rendered
  prompt contract as valid JSON-shaped text. If a prompt source is actually
  rendered with Python `.format(...)`, source literals must escape JSON braces with
  `{{ ... }}`; if the prompt source is passed straight to `SystemMessage`, source
  literals must use single `{ ... }` braces. `_TASK_DISPATCHER_PROMPT` is currently
  passed straight to `SystemMessage`, so its extracted output-format constant and
  visible prompt block must use single braces.
- Render `expected_output_format` into `_PARSE_JSON_WITH_LLM_PROMPT` by string
  concatenation or by `str.replace` on a non-brace sentinel such as
  `<<<EXPECTED_OUTPUT_FORMAT>>>`. Do not use `str.format`, `%` formatting, or
  f-string interpolation for JSON-containing prompt templates, because the
  inserted expected-format text contains literal `{`, `}`, and `:` characters.
- Use a Python-side branch for the repair prompt. When `expected_output_format` is
  `None`, use the base repair prompt with no empty "Expected output format" header.
  When it is supplied, use the variant that includes the expected-format block.
- Do not change scheduler dispatcher semantics, tool validation, adapter dispatch,
  promise harvesting, memory writes, affinity updates, character image updates,
  Cache2 invalidation, or conversation progress.
- Keep deterministic validation after parsing. `_generate_raw_tool_calls` keeps its
  per-row structural checks (drop non-dict rows, missing tool names, non-dict args,
  invalid `execute_at`, unusable `tool_calls`); the parser's dict guarantee does
  not replace those.
- Preserve backward compatibility for existing `parse_llm_json_output` and
  `parse_json_with_llm` callers that do not supply an expected output format.

## Must Do

- Make `parse_llm_json_output` return `{}` whenever every parse/repair attempt
  fails to produce a `dict` (replace the current log-and-return-non-dict tail with
  a log-and-normalize-to-`{}` tail).
- Replace every ambiguous task-dispatcher no-op instruction such as `返回空列表`,
  `输出空列表`, or `返回空数组` with an explicit `{"tool_calls": []}` instruction.
- Add a clear task-dispatcher prompt rule that the top-level output must always be
  a JSON object and must never be a top-level array.
- Extract the task-dispatcher target JSON contract into one string constant, such
  as `_TASK_DISPATCHER_OUTPUT_FORMAT`, as the rendered valid JSON-shaped output
  contract. Because `_TASK_DISPATCHER_PROMPT` is not `.format(...)`-rendered, embed
  that exact same single-brace string in `_TASK_DISPATCHER_PROMPT`.
- Add an optional, keyword-only `expected_output_format` argument to
  `parse_llm_json_output` and forward it to the LLM-repair fallback only.
- Add the same optional, keyword-only `expected_output_format` argument to
  `parse_json_with_llm`.
- Update dispatcher live-LLM test instrumentation so the inspected trace records
  raw model output, parsed output, whether LLM repair was invoked, and final raw
  calls.
- Update dispatcher live-LLM endpoint availability checking to probe
  `CONSOLIDATION_LLM_BASE_URL`, because `_task_dispatcher_llm` uses the
  `CONSOLIDATION_LLM_*` route.
- Update `_PARSE_JSON_WITH_LLM_PROMPT` so the repair LLM is told: the deterministic
  pass already handled ordinary syntax fixes, so it is being called for the hard
  residual cases; when `expected_output_format` is supplied, it is rendered into
  the repair prompt itself as the same output format text from the original LLM
  prompt; use it as the target format to reconstruct toward; preserve actual
  values from the raw output; never copy placeholder example values as data; and
  if the raw output and the expected format cannot be reconciled, return `{}`
  rather than fabricate structure.
- Implement the repair prompt rendering with concatenation or a non-brace sentinel
  replacement. Do not use brace-sensitive templating to render JSON-shaped output
  format text.
- Update `_generate_raw_tool_calls` to call `parse_llm_json_output` with the
  exact task-dispatcher output-format string used in `_TASK_DISPATCHER_PROMPT`.
  The parser's dict guarantee means no new type guard is needed at the call site;
  keep the existing `tool_calls` list check and per-row validation.
- Keep `_generate_raw_tool_calls` behavior as zero raw calls when no usable
  `tool_calls` list is available after parsing and structural validation.
- Inspect the task-dispatcher entry in `src/scripts/run_touched_llm_regression.py`
  against the current prompt/input contract. Update it when it disagrees with the
  current contract; otherwise record "no change required" in `Execution Evidence`.
- Inspect `src/kazusa_ai_chatbot/dispatcher/README.md` for task-dispatcher no-op
  wording. Update it when it does not explicitly say no tool calls means
  `{"tool_calls": []}`; otherwise record "no change required" in
  `Execution Evidence`.

## Deferred

- Do not redesign the scheduler dispatcher.
- Do not change the `future_promises` schema.
- Do not change `facts_harvester`, `fact_harvester_evaluator`, or memory-unit
  extraction prompts.
- Do not change global parser behavior for callers that do not opt in with an
  expected output format, beyond the dict-return guarantee, which only strengthens
  the already-documented `-> dict` contract.
- Do not introduce Pydantic models, dataclasses, or a generic schema-validation
  framework for all LLM outputs in this bugfix.
- Do not add a general malformed-output benchmark or fuzzing suite.
- Do not add code-side semantic fallback that converts arbitrary arrays into domain
  objects. Normalizing a non-dict parser result to `{}` is not semantic fallback.
- Do not wire the new expected-output-format argument into callers other than
  `_generate_raw_tool_calls` in this bugfix; other consumers can adopt it later.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| `parse_llm_json_output` return contract | compatible | Strengthen to always return a `dict` (`{}` on failure). This only enforces the existing `-> dict` annotation; any caller relying on a non-dict return is already broken. Do not change the return value for well-formed input. |
| Task-dispatcher prompt | bigbang | Replace ambiguous no-op wording directly. Future model output must use the object wrapper. Do not preserve prompt wording that permits top-level `[]`. |
| JSON parser / repair API | compatible | Add an optional keyword-only `expected_output_format` argument to `parse_llm_json_output` and `parse_json_with_llm`. Existing calls without the argument must keep their current call shape and behavior. |
| Task-dispatcher parser call | bigbang | Pass the expected format from `_generate_raw_tool_calls` immediately after the parser supports it. |
| Tests | compatible | Add focused prompt-contract, dict-return, and expected-format-plumbing checks only. Do not add a broad negative-path malformed-output matrix. |
| Runtime behavior | compatible | Well-formed object-shaped outputs continue to parse and dispatch as before. No-op outputs become zero raw calls. Non-dict parse results become `{}` then zero raw calls instead of a crash. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- Any change to a cutover policy requires user approval before implementation.
- Parser compatibility means preserving the existing function name and default
  calling behavior and honoring the documented `-> dict` return; it does not
  authorize broad schema repair for all callers.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve the
  contracts in this plan.
- The agent must not introduce new architecture, alternate parser layers,
  compatibility wrappers, prompt-repair agents, retry loops, or extra features.
- The agent must treat changes outside the named change surface as high-scrutiny
  changes and stop unless the change is required by this plan.
- The agent must not perform unrelated cleanup, formatting churn, dependency
  upgrades, prompt rewrites, or broad refactors.
- If the code and this plan disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of
  inventing a substitute.

## Target State

The task-dispatcher stage has one visible model contract:

```json
{
  "tool_calls": []
}
```

No-op cases, persistent style rules, uncertain future promises, and unsupported
tool requests all instruct the model to return that same object wrapper with an
empty `tool_calls` array. A top-level array is explicitly forbidden.

`parse_llm_json_output` always returns a `dict`. `{}` means "nothing usable could
be parsed"; callers can rely on `.get(...)` never raising.

The JSON-repair layer is layered, not semantic:

- `json_repair` is the first line and handles ordinary malformed JSON.
- The LLM-repair fallback is the last resort for failures `json_repair` cannot fix
  (severe truncation, garbled tokens, wrong top-level shape). A caller may
  optionally supply an expected output format; when supplied, the repair prompt
  uses it as a reconstruction contract so a wrong-wrapper or partially destroyed
  payload can be rebuilt to the intended shape — without inventing tool-call rows
  or copying placeholder values.

## Design Decisions

- Honor the parser's `-> dict` contract as the primary, mode-agnostic fix for the
  reported crash. It does not require enumerating failure modes; any non-dict result
  collapses to `{}`.
- Treat the task-dispatcher prompt cleanup as the root-cause fix for this bug.
- Treat the expected-output-format hint as a generic, reusable reconstruction aid
  for the LLM-repair escalation path; the task-dispatcher is its first consumer.
- Keep parser expected-output-format support opt-in and escalation-only.
- Represent the expected output format as the target output contract string used
  in the normal LLM prompt. The string must be the target output contract itself,
  without section headings or generic "return legal JSON" prose. The parser must
  not accept a separate dict contract for this argument in this bugfix.
- Normalize the task-dispatcher output-format block to the rendered valid JSON
  shape in both the prompt and the repair expected-format constant. For the current
  direct `SystemMessage` dispatcher prompt, that means single braces in source. In
  a future or different `.format(...)`-rendered prompt, doubled braces may appear
  only in the Python source template; the rendered prompt text and the
  `expected_output_format` passed to repair must still be single-brace valid JSON.
- The repair prompt must say the expected format is the original caller's output
  contract, not data to copy, and must instruct the model to preserve raw values
  when they can be mapped into that contract and return `{}` when they cannot.
- The expected output format is injected into `_PARSE_JSON_WITH_LLM_PROMPT`. The
  human/input message to the repair LLM carries the broken JSON text, not a JSON
  object containing an `expected_output_format` field.
- The repair prompt has two Python-selected variants: a base prompt when no
  expected format is supplied, and an expected-format prompt when it is supplied.
  The expected-format header must not appear empty in the no-format path.
- Failing closed to `{}` is a helper-wide safety tradeoff. Future consumers that
  need partial-data salvage should not opt into `expected_output_format` until they
  have a stage-specific contract for safe partial recovery.
- `_generate_raw_tool_calls` remains responsible for dropping non-dict rows, missing
  tool names, non-dict args, invalid `execute_at`, and unusable `tool_calls`. The
  parser's dict guarantee removes the need for a separate non-dict guard at that
  call site.

## Contracts And Data Shapes

Parser public call shape:

```python
parse_llm_json_output(
    raw_output: str,
    *,
    expected_output_format: str | None = None,
) -> dict  # always a dict; {} when every attempt fails
```

Repair helper call shape:

```python
parse_json_with_llm(
    broken_string: str,
    *,
    expected_output_format: str | None = None,
) -> dict
```

The dict-return guarantee is enforced at `parse_llm_json_output`.
`parse_json_with_llm` should also normalize its own result toward a dict to keep
its annotation honest, but `parse_llm_json_output` is the load-bearing guarantee
for callers.

Task-dispatcher output-format string:

```text
{
  "tool_calls": [
    {
      "tool": "工具名",
      "args": {
        "参数名": "参数值",
        "target_channel_type": "group | private，target_channel 不是 same 时必填"
      }
    }
  ]
}
```

This string intentionally uses single braces because it is the rendered target
contract. Python source templates may need doubled braces only when they are
actually rendered with `.format(...)`; `_TASK_DISPATCHER_PROMPT` is not.

No-op task-dispatcher output:

```json
{"tool_calls": []}
```

Repair prompt rendering when an expected format is supplied:

```text
<static _PARSE_JSON_WITH_LLM_PROMPT instructions>

Expected output format from the original prompt:
<<<EXPECTED_OUTPUT_FORMAT>>>

Only the broken JSON text is sent as the repair call's input message.
```

The implementation must render `<<<EXPECTED_OUTPUT_FORMAT>>>` with
`str.replace(...)` or build the prompt by concatenating the static text and the
expected-format string. It must not use `.format`, `%` formatting, or an f-string
template for JSON-containing prompt text.

Use the full task-dispatcher output-format string shown above for
`_generate_raw_tool_calls`. The same string must be embedded in the normal
task-dispatcher prompt and passed to `parse_llm_json_output`. Do not substitute a
separate schema, dict contract, reduced output-format string, or
runtime-data-bearing payload.

## LLM Call And Context Budget

- Normal valid JSON object output: zero additional LLM calls.
- Existing syntax-repair fallback path: unchanged maximum of one JSON-repair LLM
  call when `json_repair` cannot produce a dict.
- Wrong-wrapper, truncated, or garbled output with an expected format supplied:
  still at most one JSON-repair LLM call; the expected format is added to that same
  call, not a new one.
- Do not add retry loops.
- Do not add model calls to the live response path. This stage is post-turn
  background consolidation, but it still must stay bounded.

## Change Surface

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - `_TASK_DISPATCHER_PROMPT` (no-op wording, top-level-object rule, and
    normalization of the output-format block to rendered valid JSON-shaped text;
    because this prompt is not `.format(...)`-rendered, the source block uses
    single braces)
  - task-dispatcher output-format string constant reused by the prompt and parser
    call
  - `_generate_raw_tool_calls` parser invocation (pass `expected_output_format`);
    keep the existing `tool_calls` list check and per-row validation
- `src/kazusa_ai_chatbot/utils.py`
  - `parse_llm_json_output` (dict-return guarantee plus opt-in
    `expected_output_format`)
  - `parse_json_with_llm` (opt-in `expected_output_format`)
  - `_PARSE_JSON_WITH_LLM_PROMPT` (escalation framing and output-contract wording)
- `src/scripts/run_touched_llm_regression.py`
  - task-dispatcher regression case only, after explicit contract inspection
- `src/kazusa_ai_chatbot/dispatcher/README.md`
  - no-op tool-call output wording only, after explicit wording inspection
- Tests under `tests/`
  - `tests/test_dispatcher.py`
    - task-dispatcher prompt-contract checks
    - dispatcher-side production regression for raw top-level `[]`
    - `_generate_raw_tool_calls` parser-call plumbing check
    - live-LLM availability helper `_skip_if_llm_unavailable`
    - live trace writer `_write_live_dispatch_trace`
  - `tests/test_utils.py`
    - parser dict-return guarantee and expected-format repair prompt rendering
  - `tests/llm_trace.py`
    - existing trace sink used by live dispatcher traces; do not invent a new
      tracing framework unless this helper cannot satisfy the requirement
  - focused prompt-contract, dict-return, and expected-format-plumbing checks only
  - dispatcher live-LLM endpoint availability check and trace fields for
    `test_live_dispatcher_rejects_persistent_style_rule_as_tool_call`

## Implementation Order

1. Add a focused prompt-contract test or static check for `_TASK_DISPATCHER_PROMPT`.
   - Target: a deterministic test under `tests/`.
   - Expected before implementation: fails because the current prompt contains
     ambiguous no-op wording such as `返回空列表` or `返回空数组`, and because the
     current direct-to-`SystemMessage` output-format block uses doubled braces.
   - This is a prompt-contract test, not a malformed-output negative-path test.
2. Fix `_TASK_DISPATCHER_PROMPT`.
   - Replace ambiguous no-op instructions with `{"tool_calls": []}`.
   - Add a top-level-object-only rule.
   - Extract the pure target JSON contract into one string constant as rendered
     valid JSON-shaped text, and embed that same string in
     `_TASK_DISPATCHER_PROMPT`.
   - Because `_TASK_DISPATCHER_PROMPT` is not `.format(...)`-rendered, do not
     preserve doubled braces in its output-format block.
3. Run the focused prompt-contract test and record the pass.
4. Add a focused deterministic test that `parse_llm_json_output` returns `{}` when
   the underlying parse/repair yields a non-dict (for example input that reduces to
   a list, or a mocked LLM repair returning a list).
   - Expected before implementation: fails because the current tail returns the
     non-dict value.
5. Add a dispatcher-side deterministic production-regression test in
   `tests/test_dispatcher.py`.
   - Patch `_task_dispatcher_llm.ainvoke` to return the production-shaped raw
     output `[]`.
   - Call `_generate_raw_tool_calls(...)` with a minimal dispatch-generation state
     and dispatch context.
   - Assert the result is `[]` and no exception is raised.
   - Expected before implementation: fails with the reported
     `'list' object has no attribute 'get'` crash.
6. Make `parse_llm_json_output` normalize any non-dict result to `{}` (keep the
   existing failure log).
7. Add focused expected-format prompt-rendering coverage in `tests/test_utils.py`.
   - Verify that a supplied expected output format string is rendered into the
     LLM-repair prompt template through concatenation or non-brace sentinel
     replacement, not as an input-message field.
   - Verify that the repair call's input message contains only the broken JSON
     text.
   - Verify that the no-format path uses a base prompt without an empty "Expected
     output format" header.
   - Verify that the expected format is not used on the deterministic success path.
   - Verify that non-string expected output formats are not accepted by this public
     parser interface in this bugfix.
   - Do not add a malformed-output matrix.
8. Implement optional `expected_output_format` support in `parse_llm_json_output`
   and `parse_json_with_llm`, and update `_PARSE_JSON_WITH_LLM_PROMPT` with the
   escalation framing and output-contract wording.
   - Preserve existing caller compatibility.
   - Ensure the repair prompt treats the expected format as shape only.
   - Use a Python-side prompt variant branch for with-format vs no-format repair.
   - Use concatenation or non-brace sentinel replacement for the expected-format
     prompt variant.
9. Add a focused dispatcher plumbing test before wiring the call.
   - Assert `_generate_raw_tool_calls` calls `parse_llm_json_output` with
     `expected_output_format=_TASK_DISPATCHER_OUTPUT_FORMAT`.
   - Assert the expected-format value is exactly the rendered target JSON contract
     embedded in `_TASK_DISPATCHER_PROMPT`.
   - Expected before implementation: fails because the dispatcher call does not
     pass the new argument yet.
10. Wire `_generate_raw_tool_calls` to pass the task-dispatcher expected output
   format.
   - Verify through a focused mock that `_generate_raw_tool_calls` passes the exact
     same output-format string that is embedded in `_TASK_DISPATCHER_PROMPT`.
   - Keep per-row structural validation after parsing.
11. Inspect and reconcile documentation plus the touched LLM regression case.
   - Inspect `src/scripts/run_touched_llm_regression.py` task-dispatcher entry.
     Update it when it disagrees with the current prompt/input contract; otherwise
     record "no change required".
   - Inspect `src/kazusa_ai_chatbot/dispatcher/README.md` task-dispatcher no-op
     wording. Update it when it does not explicitly state `{"tool_calls": []}`;
     otherwise record "no change required".
12. Update dispatcher live-LLM test support in `tests/test_dispatcher.py`.
    - Change the dispatcher live-LLM availability check to use
      `CONSOLIDATION_LLM_BASE_URL`.
    - Ensure the live dispatcher trace for the no-op case records raw model output,
      parsed output, whether LLM repair was invoked, and final raw calls.
    - Write the trace via `tests.llm_trace.write_llm_trace`, which stores JSON
      artifacts under `test_artifacts/llm_traces/`; print or log the returned path
      during the `-s` live test run.
13. Run focused deterministic verification.
14. Run the task-dispatcher live LLM no-op case one at a time and inspect the
    trace.
15. Run independent code review and remediate findings inside the approved change
    surface.

## Progress Checklist

- [x] Stage 1 - task-dispatcher prompt contract established
  - Covers: implementation steps 1-3.
  - Verify: focused deterministic prompt-contract test passes and proves the
    task-dispatcher prompt embeds the same output-format string that will be passed
    to parser repair, and that the embedded output-format string is rendered valid
    JSON-shaped text. For this direct-to-`SystemMessage` prompt, the source string
    uses single braces rather than doubled braces.
  - Evidence: record before/after prompt-test result in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex / 2026-05-13` after verification and evidence are recorded.
- [x] Stage 2 - parser hardened (dict-return guarantee plus opt-in expected-format support)
  - Covers: implementation steps 4-10.
  - Verify: the dict-return test passes; expected-format plumbing tests pass;
    dispatcher production-regression test for raw top-level `[]` passes without
    raising;
    `_generate_raw_tool_calls` passes the exact task-dispatcher output-format
    string to `parse_llm_json_output`; existing `tests/test_utils.py` and
    `tests/test_dispatcher.py` non-`live_llm` cases still pass.
  - Evidence: record changed functions and test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex / 2026-05-13` after verification and evidence are recorded.
- [x] Stage 3 - docs, regression case, and focused plus live verification complete
  - Covers: implementation steps 11-14.
  - Verify: focused deterministic tests pass; dispatcher live-LLM availability
    check uses `CONSOLIDATION_LLM_BASE_URL`; live LLM task-dispatcher no-op case
    is run one at a time and inspected with raw model output, parsed output, repair
    invocation status, and raw calls recorded in a `tests.llm_trace.write_llm_trace`
    artifact under `test_artifacts/llm_traces/`.
  - Evidence: record commands, trace path, and manual judgment in
    `Execution Evidence`.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex / 2026-05-13` after verification and evidence are recorded.
- [x] Stage 4 - independent code review complete
  - Covers: implementation step 15.
  - Verify: review checks style, plan alignment, prompt contract consistency,
    parser scope, the dict-return guarantee, tests, and verification evidence.
  - Evidence: record findings, remediations, rerun commands, and approval status
    in `Execution Evidence`.
  - Handoff: plan may move to completed only after this stage is signed off.
  - Sign-off: `Codex / 2026-05-13` after review and required reruns are recorded.

## Verification

### Static Greps

- `rg "返回空列表|输出空列表|返回空数组" src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - Expected: no matches in `_TASK_DISPATCHER_PROMPT`.
- `rg "tool_calls" src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py src/kazusa_ai_chatbot/dispatcher/README.md src/scripts/run_touched_llm_regression.py`
  - Expected: task-dispatcher prompt, docs, and regression references agree that
    no-op output is `{"tool_calls": []}`.
- The focused prompt-contract test must inspect `_TASK_DISPATCHER_OUTPUT_FORMAT`
  directly and assert it contains rendered valid JSON-shaped text. Because
  `_TASK_DISPATCHER_PROMPT` is not `.format(...)`-rendered, that means single
  braces, not doubled-brace `{{ ... }}` source escapes.

### Syntax

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\utils.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py`
  - Expected: exits 0.
- After editing the CJK-containing prompt, also run `ast.parse` / `py_compile` per
  `cjk-safety`.

### Deterministic Tests

- `venv\Scripts\python -m pytest tests/test_utils.py -q`
  - Expected: exits 0, including the new dict-return and expected-format-plumbing
    cases. Expected-format prompt-rendering coverage proves the supplied format is
    inserted into the repair prompt template, not sent as an input-message field,
    and the no-format path has no empty expected-format header.
- `venv\Scripts\python -m pytest tests/test_dispatcher.py -q`
  - Expected: exits 0 for non-`live_llm` cases selected by default markers,
    including a focused check that the task-dispatcher parser call receives the
    exact output-format string embedded in the prompt, and a production-regression
    case where raw model output `[]` makes `_generate_raw_tool_calls(...)` return
    `[]` without raising.
- Run any new focused prompt-contract, dict-return, or argument-plumbing tests by
  exact node id before and after implementation.
  - Expected before implementation: prompt-contract test fails for ambiguous no-op
    wording; dict-return test fails because the current tail returns a non-dict.
  - Expected after implementation: focused tests pass.

### Live LLM

- `venv\Scripts\python -m pytest tests/test_dispatcher.py::test_live_dispatcher_rejects_persistent_style_rule_as_tool_call -q -s -m live_llm`
  - Run this case by itself and inspect the emitted trace.
  - Required precondition: the dispatcher live-LLM endpoint check probes
    `CONSOLIDATION_LLM_BASE_URL`, not `DIALOG_GENERATOR_LLM_BASE_URL`.
  - Required sink: `tests.llm_trace.write_llm_trace` writes a JSON artifact under
    `test_artifacts/llm_traces/`, and the test prints or logs the artifact path
    during the `-s` run.
  - Required trace fields: raw model output, parsed output, whether LLM repair was
    invoked, final raw calls, and manual judgment.
  - Expected: no exception; raw calls are empty; the trace proves whether the model
    followed the object wrapper contract or repair recovered to a dict and
    structural validation yielded zero raw calls.

## Independent Plan Review

- Review date: 2026-05-13.
- Reviewer mode: self-review from a fresh plan-readiness posture after user
  feedback was incorporated.
- References checked: `development-plan-writing`, `plan_contract.md`,
  `execution_gates.md`, and `cutover_policy.md`.
- Review result: approved for implementation.
- Blockers: none.
- Non-blocking notes: implementation remains bounded to the listed prompt,
  parser, dispatcher-test, touched-regression, and dispatcher README surfaces.
- Approval rationale: the plan has concrete contracts, exact change surface,
  test-first implementation order, deterministic reproduction of the production
  raw `[]` crash, live LLM trace requirements, cutover policy, and final
  independent code review gate. The earlier ambiguity around expected-format
  transport and Python brace escaping has been resolved in plan text.

## Independent Code Review

After verification and before marking this plan complete, run an independent code
review gate. The reviewer must check:

- `parse_llm_json_output` always returns a `dict` (`{}` on failure) and the failure
  log is preserved;
- prompt wording no longer teaches top-level arrays;
- `_TASK_DISPATCHER_OUTPUT_FORMAT` is rendered valid JSON-shaped text, and
  `_TASK_DISPATCHER_PROMPT` embeds that same string; because the dispatcher prompt
  is not `.format(...)`-rendered, the source text uses single braces;
- expected-output-format repair is opt-in, escalation-only, string-based, and uses
  the same output-format text shown to the original LLM;
- expected-format prompt rendering avoids `str.format`, `%` formatting, and
  f-string templates for JSON-containing text; the no-format path has no empty
  expected-format header;
- the repair prompt forbids copying placeholder values as data and prefers faithful
  raw values over fabricated structure;
- parser changes do not add broad semantic repair;
- `_generate_raw_tool_calls` still performs deterministic per-row structural
  validation;
- deterministic dispatcher coverage reproduces the production raw top-level `[]`
  case and proves `_generate_raw_tool_calls(...)` returns `[]` without raising;
- dispatcher live-LLM availability checking uses `CONSOLIDATION_LLM_BASE_URL`;
- dispatcher live-LLM trace evidence includes raw model output, parsed output,
  repair invocation status, final raw calls, manual judgment, and a
  `test_artifacts/llm_traces/` artifact path;
- no broad malformed-output negative-path test matrix was added;
- CJK prompt edits compile cleanly;
- verification evidence matches the actual changed files and commands.

If no separate reviewer is available, the active agent must reread this full plan
and perform a fresh self-review as the independent review substitute. Record
findings and remediation in `Execution Evidence`.

## Acceptance Criteria

- `parse_llm_json_output` always returns a `dict`; `{}` when every parse/repair
  attempt fails.
- Task-dispatcher no-op instructions consistently specify `{"tool_calls": []}`.
- The prompt explicitly forbids a top-level array output.
- The task-dispatcher expected-format constant and prompt output-format block use
  rendered valid JSON-shaped text. Because the dispatcher prompt is not
  `.format(...)`-rendered, its source text uses single braces, not doubled-brace
  escapes.
- `parse_llm_json_output` and `parse_json_with_llm` accept an optional, keyword-only
  `expected_output_format` without breaking existing callers, and it is used only
  on the LLM-repair escalation path.
- `_generate_raw_tool_calls` passes the task-dispatcher expected output format into
  parser repair and keeps deterministic per-row validation.
- Dispatcher live-LLM support probes `CONSOLIDATION_LLM_BASE_URL` and writes raw
  model output, parsed output, repair invocation status, final raw calls, and
  manual judgment into the no-op case trace.
- The JSON-repair prompt receives the same expected output-format string shown to
  the original LLM, uses it as the reconstruction contract, forbids copying
  placeholder values as data, and returns `{}` instead of fabricating structure
  when the raw output cannot be reconciled with the contract.
- The expected output-format string is injected into `_PARSE_JSON_WITH_LLM_PROMPT`;
  it is not sent as an input-message field.
- Expected-format prompt rendering uses concatenation or non-brace sentinel
  replacement, and the no-format repair path omits the expected-format block
  entirely.
- A deterministic dispatcher regression reproduces raw top-level `[]` from the LLM
  and proves `_generate_raw_tool_calls(...)` returns `[]` without raising.
- Existing object-shaped `{"tool_calls": [...]}` outputs still dispatch through the
  current validation path.
- No broad malformed-output negative-path matrix is added.
- Focused deterministic tests, syntax checks, and one-by-one inspected live LLM
  evidence pass.
- Independent code review is complete and recorded.

## Risks

- The JSON-repair LLM may still return the wrong top-level type. Mitigation:
  `parse_llm_json_output` normalizes any non-dict result to `{}`, so callers always
  get a dict; `_generate_raw_tool_calls` then yields zero raw calls.
- The expected-output-format text may be treated as data. Mitigation: the repair
  prompt explicitly says the expected output format is the original prompt
  contract, placeholder values must not be copied, raw values are the source of
  truth, and unreconcilable output must become `{}` rather than fabricated
  structure.
- Brace-sensitive prompt rendering may break when JSON-shaped expected-format text
  contains literal `{`, `}`, and `:` characters. Mitigation: render with
  concatenation or non-brace sentinel replacement, and cover both with-format and
  no-format prompt variants in deterministic tests.
- Prompt cleanup may not fully control the local model. Mitigation: pair the prompt
  fix with the parser dict guarantee, opt-in expected-format repair, and live LLM
  inspection.
- Strengthening the parser return contract could surprise a caller that relied on a
  non-dict return. Mitigation: the function is already annotated `-> dict`; review
  callers during implementation; the change only affects the failure tail, not the
  well-formed success path.
- Parser API changes could accidentally affect unrelated LLM stages. Mitigation:
  keyword-only optional argument, existing tests, escalation-only use, and no global
  behavior change for callers that omit the argument.

## Execution Evidence

- Status: completed.
- Implementation agent: Codex
- Evidence log:
  - Implementation completed on 2026-05-13.
  - RED prompt-contract test:
    `venv\Scripts\python -m pytest tests/test_dispatcher.py::test_task_dispatcher_prompt_uses_object_wrapper_contract -q`
    failed before implementation with missing `_TASK_DISPATCHER_OUTPUT_FORMAT`.
  - RED parser dict-return test:
    `venv\Scripts\python -m pytest tests/test_utils.py::test_parse_llm_json_output_returns_empty_dict_for_repaired_list -q`
    failed before implementation because `parse_llm_json_output("[]")` returned
    `[]`.
  - RED dispatcher production-regression test:
    `venv\Scripts\python -m pytest tests/test_dispatcher.py::test_generate_raw_tool_calls_returns_empty_for_raw_top_level_array -q`
    failed before implementation with `'list' object has no attribute 'get'`.
  - RED expected-format rendering test:
    `venv\Scripts\python -m pytest tests/test_utils.py::test_parse_json_with_llm_renders_expected_format_in_system_prompt -q`
    failed before implementation because `parse_json_with_llm` did not accept
    `expected_output_format`.
  - Focused post-implementation tests passed:
    `tests/test_dispatcher.py::test_task_dispatcher_prompt_uses_object_wrapper_contract`,
    `tests/test_utils.py::test_parse_llm_json_output_returns_empty_dict_for_repaired_list`,
    `tests/test_dispatcher.py::test_generate_raw_tool_calls_returns_empty_for_raw_top_level_array`,
    and
    `tests/test_utils.py::test_parse_json_with_llm_renders_expected_format_in_system_prompt`.
  - Additional focused expected-format and dispatcher plumbing tests passed:
    `tests/test_utils.py::test_parse_llm_json_output_rejects_non_string_expected_format`,
    `tests/test_utils.py::test_parse_llm_json_output_does_not_use_expected_format_on_success`,
    `tests/test_utils.py::test_parse_json_with_llm_omits_expected_format_header_when_absent`,
    and
    `tests/test_dispatcher.py::test_generate_raw_tool_calls_passes_task_dispatcher_expected_format`.
  - Static grep passed:
    `rg "返回空列表|输出空列表|返回空数组" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py`
    returned no matches.
  - Syntax checks passed:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\utils.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py tests\test_utils.py tests\test_dispatcher.py src\scripts\run_touched_llm_regression.py`.
  - CJK AST safety check passed for
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`,
    `tests/test_dispatcher.py`, and `tests/test_utils.py`.
  - Deterministic suites passed:
    `venv\Scripts\python -m pytest tests/test_utils.py -q` reported 12 passed;
    `venv\Scripts\python -m pytest tests/test_dispatcher.py -q` reported
    11 passed and 4 deselected.
  - Live LLM verification passed one case at a time:
    `venv\Scripts\python -m pytest tests/test_dispatcher.py::test_live_dispatcher_rejects_persistent_style_rule_as_tool_call -q -s -m live_llm`.
    Trace artifact:
    `test_artifacts/llm_traces/dispatcher_live_tool_call_generation__rejects_persistent_style_rule_as_tool_call.json`.
    Manual judgment: acceptable. Raw model output was fenced `{"tool_calls": []}`,
    parsed output was `{"tool_calls": []}`, LLM repair was not invoked, and final
    raw calls were empty.
  - Independent code review completed on 2026-05-13. Reviewed changed parser,
    task-dispatcher prompt/wiring, deterministic tests, dispatcher README, touched
    regression runner, trace artifact, and verification evidence against this plan.
    `git diff --check` reported no whitespace errors; it only printed existing
    Windows line-ending warnings. Review result: approved for completion.
  - Plan approved on 2026-05-13 after final readiness review before
    implementation began.
  - Independent plan review updated after user request; previous blockers addressed
    in plan text by requiring dispatcher trace observability, the correct
    `CONSOLIDATION_LLM_BASE_URL` live route check, and fixed task-dispatcher
    expected-format string selection.
  - User rejected the prior dict-based expected-format architecture. Plan revised
    so `expected_output_format` is a generic string copied from the normal LLM
    prompt's target output contract, with task dispatcher as the first required
    caller.
  - User clarified that the reusable task-dispatcher output-format string should
    not include `# 输出格式` or `请只返回合法 JSON：`, and that the repair LLM should
    receive the expected format through `_PARSE_JSON_WITH_LLM_PROMPT`, not as an
    input-message field.
  - User review identified required plan corrections: distinguish Python source
    brace escaping from rendered prompt JSON, use single braces for the current
    non-`.format(...)` dispatcher prompt, specify safe repair prompt rendering
    without brace-sensitive templating, add a deterministic dispatcher regression
    for raw top-level `[]`, and name the live trace sink and exact test paths.
