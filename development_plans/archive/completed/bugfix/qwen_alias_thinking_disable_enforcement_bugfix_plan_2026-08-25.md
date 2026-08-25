# Qwen Alias Thinking Disable Enforcement Bugfix Plan

## Summary

- Goal: Ensure every Qwen-family route configured with thinking disabled sends the existing provider-side `enable_thinking: false` control even when the configured model name is an opaque or unversioned alias.
- Status: completed
- Scope boundary: `kazusa_ai_chatbot.llm_interface` backend detection, its deterministic provider contract tests, and the subsystem ICD.
- Change direction: Broaden only disabled-Qwen enforcement from recognized Qwen3 names to the already-detected Qwen family; preserve narrow Qwen3 matching for enabled thinking.
- Acceptance state: Accepted after deterministic verification, one individually inspected real-LLM relevance gate, and parent review.

## Scope And Change Direction

The configured relevance model alias `qwen27b-5090` is detected as the Qwen family but does not match the Qwen3 thinking-model pattern. Consequently, `LLMThinkingConfig(enabled=False)` currently produces the generic `disabled` strategy and omits the provider-side disable control. The driver must instead reuse the existing explicit Qwen disabled strategy for every detected Qwen-family model while leaving enabled-thinking support restricted to the existing Qwen3 matcher.

This is a request-mapping correction. It does not change relevance prompts, completion budgets, semantic dispositions, cognition, dialog, adapters, database state, or model-route configuration.

## Mandatory Skills

- `development-plan`: governs this execution record and closure.
- `local-llm-architecture`: governs the provider ownership boundary and normal-path overhead.
- `py-style`: governs the Python production and test changes.
- `test-style-and-execution`: governs deterministic and real-LLM verification.

## Mandatory Rules

- Keep provider-specific request mapping inside `kazusa_ai_chatbot.llm_interface`.
- Keep `LLMThinkingConfig(enabled: bool)` unchanged.
- Reuse the existing explicit disabled-Qwen transport; add no compatibility shim, route flag, prompt instruction, retry, or model-specific alias list.
- Preserve strict Qwen3 recognition for enabled thinking.
- Use scenario-neutral model aliases in deterministic tests.
- Run the real-LLM gate one node at a time and inspect its retained trace before acceptance.
- Preserve all pre-existing worktree changes outside the owned surface.

## Must Do

- Return the existing explicit Qwen disabled strategy whenever `model_family == "qwen"` and route thinking is disabled.
- Preserve `ignored_unsupported_model` for enabled thinking on non-Qwen3 Qwen names.
- Prove that a generic Qwen-family alias receives `chat_template_kwargs.enable_thinking == false` through the provider constructor.
- Update the LLM interface ICD to distinguish family-wide disabled enforcement from narrow enabled-thinking support.

## Deferred

- Provider capability probing and new route configuration.
- Renaming the existing `qwen3_disabled` internal strategy.
- Completion-limit recovery, relevance fallback behavior, and trace exception capture.
- Prompt, schema, token-budget, cognition, dialog, adapter, or database changes.

## Target State

```text
Qwen family + thinking disabled
  -> thinking_strategy=qwen3_disabled
  -> extra_body.chat_template_kwargs.enable_thinking=false

Recognized Qwen3 + thinking enabled
  -> thinking_strategy=qwen3_enabled

Other Qwen + thinking enabled
  -> thinking_strategy=ignored_unsupported_model
```

The retained `qwen3_disabled` name is an internal transport identifier. This bugfix broadens when that existing transport is selected and avoids a parallel vocabulary or compatibility branch.

## Execution Roles

### Implementation And Verification

- Responsibility: Implement the family-wide disabled-Qwen mapping and produce deterministic plus real-LLM evidence.
- Owned surface: `src/kazusa_ai_chatbot/llm_interface/detection.py`, `tests/test_llm_interface_contracts.py`, `tests/test_llm_interface_openai_provider.py`, and `src/kazusa_ai_chatbot/llm_interface/README.md`.
- Authority: Modify only the declared mapping, tests, and documentation; run the mapped deterministic nodes and one existing relevance live-LLM node.
- Applicable skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`.
- Capability floor: Production Python contract work, OpenAI-compatible provider mapping, focused pytest verification, and real-LLM trace inspection.
- Independence requirement: none.
- Acceptance output: Scoped diff, passing exact deterministic nodes, and an inspected live trace showing a valid settled-relevance result without completion exhaustion.
- Gate: Start after baseline capture; finish when all mapped nodes pass and the live trace is judged acceptable.

### Parent Review And Closure

- Responsibility: Review scope, contract preservation, evidence, and lifecycle closure.
- Owned surface: Plan and registry lifecycle records plus read-only review of the implementation surface.
- Authority: Accept the implementation, request in-scope remediation, and close the plan.
- Applicable skills: `development-plan`, `local-llm-architecture`, `py-style`, `test-style-and-execution`.
- Capability floor: System architecture review and source-to-test traceability.
- Independence requirement: Separate from delegated implementation when delegation is used.
- Acceptance output: Review result, residual-risk statement, and completed lifecycle record.
- Gate: Start after implementation verification; finish after every acceptance criterion is evidenced.

## Test Impact And Traceability

| Source or artifact | Changed contract | Semantic owner | Exact deterministic pytest nodes | Supplemental live node | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/llm_interface/detection.py` | Disabled thinking selects explicit Qwen transport for any detected Qwen-family alias | LLM interface backend detection | `tests/test_llm_interface_contracts.py::test_describe_backend_enforces_disabled_thinking_for_qwen_alias` | `tests/test_relevance_turn_settlement_live_llm.py::test_live_multifragment_correction_uses_latest_intent` | deterministic unit plus real LLM | Opaque Qwen aliases silently inherit provider-default thinking |
| `tests/test_llm_interface_openai_provider.py` | The descriptor selected for a generic Qwen alias produces `enable_thinking: false` | OpenAI-compatible provider request mapping | `tests/test_llm_interface_openai_provider.py::test_provider_sends_qwen_disabled_payload_for_generic_alias` | none | deterministic integration | Detection and provider mapping pass independently while the composed request omits the disable field |
| `src/kazusa_ai_chatbot/llm_interface/README.md` | ICD documents family-wide disabled enforcement and narrow enabled support | LLM interface contract documentation | `tests/test_llm_interface_contracts.py::test_describe_backend_enforces_disabled_thinking_for_qwen_alias` | none | deterministic contract | Documentation continues to claim aliases may omit disabled enforcement |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/llm_interface/detection.py`: reorder the Qwen thinking strategy so disabled enforcement is family-wide and enabled support remains Qwen3-specific.
- `tests/test_llm_interface_contracts.py`: add a scenario-neutral generic Qwen alias descriptor regression.
- `tests/test_llm_interface_openai_provider.py`: add a composed provider-request regression for the generic alias.
- `src/kazusa_ai_chatbot/llm_interface/README.md`: document the corrected disabled/enabled distinction.
- `development_plans/README.md`: register and later close this plan.

### Create

- This plan while active; archive it after closure.

### Keep

- All prompts, public LLM config dataclasses, provider payload field names, relevance semantics, token caps, model routes, and unrelated worktree changes.

## Agent Autonomy Boundaries

The implementation role may choose local assertion structure and command order inside the declared surface. Any new strategy name, configuration field, provider probe, fallback, prompt change, or production file requires a plan amendment and user approval.

## Verification

1. Demonstrate the new descriptor test fails against the pre-fix implementation.
2. Run pytest collection for both exact deterministic node IDs.
3. Run both exact deterministic node IDs, then the adjacent LLM-interface contract and provider test files.
4. Run `tests/test_relevance_turn_settlement_live_llm.py::test_live_multifragment_correction_uses_latest_intent` once with the configured real relevance model; inspect the retained trace for a valid structured decision, configured alias, and absence of length exhaustion.
5. Run `git diff --check` on the scoped diff and perform parent code review.

## Acceptance Criteria

- A Qwen-family alias without a Qwen3 version marker receives explicit `enable_thinking: false` when thinking is disabled.
- Enabled thinking on that same alias remains unsupported.
- Recognized Qwen3 enabled and disabled behavior remains valid.
- Gemma and unknown-model mappings remain unchanged.
- Exact deterministic nodes and adjacent files pass.
- The single real-LLM relevance gate completes with an acceptable validated output and inspectable artifact.
- The scoped diff contains no prompt, semantic, token-budget, route, adapter, database, or unrelated changes.

## Progress Checklist

- [x] RCA and historical-design comparison complete.
- [x] Worktree baseline and owned surface recorded.
- [x] Pre-fix deterministic regression captured.
- [x] Production mapping and documentation implemented.
- [x] Deterministic verification passed.
- [x] Real-LLM gate run once and inspected.
- [x] Parent review passed and plan archived.

## Execution Evidence

- Baseline: pre-existing changes are limited to the separate agentic-resolver plan, architecture, README, and test surfaces shown by `git status --short`; they remain outside this plan.
- Runtime assignment: implementation and deterministic verification are assigned to the user-requested GPT-5.6 Luna max subagent already used for this incident. The narrow four-file surface and exact gates satisfy its production-code role while the parent retains architecture and closure authority.
- Pre-fix reproduction: both exact scenario-neutral regression nodes failed because a generic detected Qwen alias resolved to `disabled` instead of `qwen3_disabled`.
- Implementation: `detection._thinking_strategy(...)` now selects the existing explicit disabled-Qwen transport for every detected Qwen-family model when thinking is disabled. The existing Qwen3 matcher remains the enabled-thinking gate. The provider implementation required no change because its existing `qwen3_disabled` mapping already emits `chat_template_kwargs.enable_thinking=false`.
- Deterministic verification: both exact mapped nodes passed; the adjacent contract and provider files passed with `30 passed`; `git diff --check` passed.
- Real-LLM verification: `venv\Scripts\python -m pytest tests/test_relevance_turn_settlement_live_llm.py::test_live_multifragment_correction_uses_latest_intent -q -s -o addopts=` passed once in 4.03 seconds. The retained artifact `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__L09_latest_correction.json` records model `qwen27b-5090`, one complete 400-character structured response, 6,382 prompt characters, 2,901 ms model duration, and validated `response_action=proceed` without length exhaustion.
- Parent review: passed. The diff changes only the declared detection branch, scenario-neutral tests, and ICD wording. It adds no prompt content, model alias list, configuration, retry, semantic fallback, token-budget change, or unrelated cleanup.
- Residual risk: provider compliance with `chat_template_kwargs.enable_thinking=false` is evidenced by the composed request test and successful current-alias live response; the live artifact does not expose a provider `reasoning_tokens` field. The root driver omission is closed.
