# qwen thinking support plan

## Summary

- Goal: Add disabled-by-default provider-side thinking support for Qwen3-family models, including `qwen3.6-34b-80l-fable-5-heretic`, through the existing `LLInterface` boundary.
- Plan class: large.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`, `test-style-and-execution`, `debug-llm`.
- Overall cutover strategy: bigbang inside `kazusa_ai_chatbot.llm_interface`; no compatibility shim and no stage-level prompt changes.
- Highest-risk areas: preserving thinking-disabled default behavior for Qwen, avoiding prompt pollution, avoiding hidden reasoning leakage into caller-facing `LLMResponse.content`, and keeping diagnostics accurate.
- Acceptance criteria: deterministic tests prove Qwen3 enabled and disabled request mapping, unsupported Qwen variants remain unsupported, Qwen visible `<think>` spans are stripped, route diagnostics show `thinking_on` only for enabled supported thinking, docs reflect Qwen3 support and disabled-by-default behavior, focused verification commands pass, and the 20-case dialog L3 live LLM suite has captured Qwen thinking-on and thinking-off comparison results for final sign-off.

## Context

Kazusa already exposes route-level `*_THINKING_ENABLED` settings and `LLMThinkingConfig(enabled=...)`, but current backend detection only maps enabled thinking for Gemma 4. Qwen model names are detected as `model_family="qwen"` and currently receive `thinking_strategy="ignored_unsupported_model"` even when route thinking is enabled.

The existing LLM interface boundary is the correct ownership layer. Runtime stages build `SystemMessage` and `HumanMessage`, own route config, and call `LLInterface`; provider adapters own backend-specific request fields and response normalization. This plan must not modify cognition, RAG, dialog, consolidation, background-work prompts, adapter logic, or stage-owned `LLMCallConfig` construction.

The local service probe showed `qwen3.6-34b-80l-fable-5-heretic` loaded on `http://localhost:1234/v1` with `arch=qwen35`. The endpoint accepts Qwen-style request switches, but this specific served artifact returned `reasoning_tokens=0` for `chat_template_kwargs.enable_thinking=true`, top-level `enable_thinking=true`, and `/think` probes. A parser sanity probe that forced `<think>` output did return separate `reasoning_content` and nonzero `reasoning_tokens`, so response separation exists locally while activation is not proven for this artifact. Implementation must therefore be deterministic request-contract support, not a claim that every hosted Qwen artifact will produce reasoning tokens.

Execution update on 2026-06-19: further direct validation showed that the
LM Studio GGUF path for `qwen3.6-34b-80l-fable-5-heretic` activates actual
reasoning when the provider appends an assistant prefill containing
`<think>\n`. The request-level `chat_template_kwargs.enable_thinking=true`
remains part of the provider mapping, but the prefill is the validated local
activation mechanism for this model. Final live sign-off confirmed nonzero
reasoning tokens for every thinking-on dialog case and zero reasoning tokens
for every thinking-off dialog case.

Prior dialog thinking sign-off precedent lives in `development_plans/archive/completed/bugfix/dialog_evaluator_decommission_plan.md`. That plan used `tests/test_dialog_l3_surface_contract_live_llm.py`, `DIALOG_LIVE_THINKING=on`, and `DIALOG_LIVE_THINKING=off` for the Gemma-era captured banter target case. This Qwen plan expands the same existing live dialog harness to all 20 `test_live_dialog_l3_*` cases as final sign-off evidence.

## Mandatory Skills

- `development-plan`: load before executing, reviewing, approving, or signing off this plan.
- `local-llm-architecture`: load before changing `llm_interface` thinking strategy, provider request mapping, response normalization, diagnostics, or docs.
- `py-style`: load before editing Python source files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before any live/local LLM probe or quality review artifact.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution unless the user explicitly approves fallback execution.
- Do not read `.env`. Use explicit test configs and the existing project test environment helpers.
- Do not edit production code until the parent establishes focused failing or baseline tests for the `llm_interface` contract.
- Do not change runtime stage prompts, prompt wording, graph edges, route selection, cognition contracts, RAG behavior, dialog behavior, consolidation behavior, adapters, or environment variable names.
- Keep `LLMThinkingConfig` boolean. Do not add public thinking modes, budgets, preserve flags, provider kwargs, or route-specific model-family settings.
- Provider-specific thinking fields must terminate inside `kazusa_ai_chatbot.llm_interface.providers.openai_compatible`.
- Route modules must not construct `extra_body`, inspect model-family constants, or inject `/think` or `/no_think`.
- Qwen thinking must remain disabled by default. For Qwen3-family models, disabled route thinking must send an explicit Qwen chat-template disable payload so template defaults cannot silently enable reasoning.
- Do not add runtime provider capability probing. Model-name detection remains deterministic and diagnostic-only outside request mapping.
- Qwen3-family matching must use the normalized model name, not only `model_family`, because all Qwen model names currently collapse to `model_family="qwen"`.
- Real LLM sign-off runs must use `debug-llm` and `test-style-and-execution`: run one test case at a time, inspect its trace before moving to the next case, and author a human-readable comparison artifact from real traces.
- Do not make deterministic test success depend on a local model producing nonzero `reasoning_tokens` or longer wall-clock runtime; the final dialog sign-off must record both as secondary evidence, not as the sole pass/fail condition.

## Must Do

- Add Qwen3-family thinking strategies to backend detection while keeping non-Qwen3 Qwen models unsupported.
- Map Qwen3 enabled thinking to `extra_body.chat_template_kwargs.enable_thinking=True`.
- Map Qwen3 disabled thinking to `extra_body.chat_template_kwargs.enable_thinking=False` to enforce the project default.
- For Qwen3 enabled thinking, append the validated assistant prefill
  `<think>\n` inside the provider adapter without mutating caller-owned
  messages.
- Preserve existing Gemma 4 request mapping and `/think` prompt-trigger behavior.
- Preserve unsupported-model behavior for unknown, DeepSeek, OpenAI, and non-Qwen3 Qwen model names.
- Strip Qwen visible `<think>...</think>` spans from caller-facing `LLMResponse.content` when a provider leaks them into visible content.
- Keep raw provider responses available through `LLMResponse.raw_response`.
- Update deterministic tests for detection, provider mapping, response normalization, and route reporting.
- Update `README.md`, `docs/HOWTO.md`, and `src/kazusa_ai_chatbot/llm_interface/README.md` to document Qwen3 thinking support and disabled-by-default behavior.
- Run and inspect the 20-case `tests/test_dialog_l3_surface_contract_live_llm.py` dialog suite on `qwen3.6-34b-80l-fable-5-heretic` twice: once with `DIALOG_LIVE_THINKING=on` and once with `DIALOG_LIVE_THINKING=off`.
- Produce an agent-authored Markdown comparison artifact that captures every dialog case's thinking-on result, thinking-off result, trace paths, per-test wall-clock execution time, assessment outcome, reasoning-token evidence when available, and human quality judgment.

## Deferred

- Do not add `thinking_budget`, `reasoning_effort`, `preserve_thinking`, or other non-boolean route config.
- Do not add Qwen prompt-trigger injection with `/think` or `/no_think`.
- Do not add provider capability probing or per-model runtime discovery.
- Do not add streaming or tool-call thinking support.
- Do not expose `reasoning_content` as a first-class production field.
- Do not change defaults for any route's `*_THINKING_ENABLED` config.
- Do not tune which Kazusa runtime stages should enable thinking by default.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Backend detection | bigbang | Replace Qwen thinking classification from unsupported to explicit Qwen3 enabled/disabled strategies. |
| Provider request mapping | bigbang | Add Qwen3 request mapping inside the existing OpenAI-compatible provider. No alternate provider wrapper. |
| Runtime stage prompts | bigbang keep | Do not edit stage prompts or inject Qwen control tokens outside the provider adapter. |
| Public route config | bigbang keep | Keep the existing boolean `*_THINKING_ENABLED` route variables. |
| Response normalization | bigbang | Extend `LLMResponse.content` normalization to remove Qwen visible `<think>` spans. |
| Tests and docs | bigbang | Rewrite tests and docs that currently say Qwen thinking is unsupported. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, rewrite legacy expectations instead of preserving them through aliases or fallback paths.
- Any change to this cutover policy requires user approval before implementation.

## Target State

When a route points to a Qwen3-family model name such as `qwen3.6-34b-80l-fable-5-heretic` or `hiebo/Qwen3.6-34B-80L-Fable-5-Heretic`:

- `LLMThinkingConfig(enabled=True)` produces `BackendDescriptor.thinking_strategy == "qwen3_enabled"`.
- `LLMThinkingConfig(enabled=False)` produces `BackendDescriptor.thinking_strategy == "qwen3_disabled"`.
- The OpenAI-compatible provider constructs `ChatOpenAI(..., extra_body={"chat_template_kwargs": {"enable_thinking": True}})` when the strategy is enabled.
- The OpenAI-compatible provider constructs `ChatOpenAI(..., extra_body={"chat_template_kwargs": {"enable_thinking": False}})` when the strategy is the Qwen3 disabled strategy.
- Provider messages are copied for Qwen3 enabled thinking so the
  OpenAI-compatible provider can append an assistant prefill containing
  `<think>\n`. No `/think` or `/no_think` slash trigger is inserted, and
  caller-owned message objects are not mutated.
- Startup route diagnostics render `thinking_on` only for enabled supported thinking strategies, including `gemma4_enabled` and `qwen3_enabled`.
- Caller-facing `LLMResponse.content` never includes raw Qwen `<think>` spans if a provider leaks them into visible content.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Supported Qwen scope | Support numeric Qwen3-family model names, including `qwen3`, `qwen3-32b`, `qwen3.5-*`, `qwen3.6-*`, and repository-prefixed ids that normalize to those segments. Keep earlier and ambiguous Qwen names unsupported. | Qwen3 introduced the documented thinking controls; broad `qwen` support would overclaim. |
| Disable behavior | Send explicit `enable_thinking=False` for Qwen3 when route thinking is disabled. | The project default is disabled. Qwen template defaults vary by serving stack, so omission is not strong enough for Qwen. |
| Qwen activation trigger | Send the documented chat-template switch and append the validated `<think>\n` assistant prefill for Qwen3 enabled thinking. Do not inject `/think` or `/no_think` slash triggers. | Direct local validation proved the request-level hint alone did not activate reasoning on the hosted GGUF artifact, while assistant prefill did. Keeping this inside the provider preserves caller-owned prompt content and stage boundaries. |
| Public config | Keep `LLMThinkingConfig(enabled: bool)` unchanged. | Existing route config already expresses the user-facing contract. |
| Diagnostics label | Use `thinking_on` for enabled supported strategies only. | Operator reports should show active non-default features, not disabled enforcement mechanics. |
| Live evidence | The final sign-off uses the existing 20-case dialog L3 live suite on Qwen with thinking on and off. | The user requested the same with/without-thinking dialog evidence pattern used previously for Gemma, expanded to all 20 dialog live cases. |

## Contracts And Data Shapes

The public dataclasses remain unchanged:

```python
LLMThinkingConfig(enabled: bool = False)
LLMCallConfig(..., thinking: LLMThinkingConfig = LLMThinkingConfig())
BackendDescriptor(..., model_family: str, thinking_strategy: str, ...)
LLMResponse(content: str, backend: BackendDescriptor, raw_response: object | None, usage: Mapping[str, object])
```

Expected thinking strategies after implementation:

```text
disabled
gemma4_enabled
qwen3_enabled
qwen3_disabled
ignored_unsupported_model
```

The Qwen3 disabled strategy name is `qwen3_disabled`. Do not use a generic `disabled` strategy for Qwen3 route calls because the provider must send `enable_thinking=False`.

The Qwen3 enabled provider message mapping appends this copied assistant
message only when it is not already present:

```python
AIMessage(content="<think>\n")
```

The provider must not mutate caller-owned message objects.

Qwen3-family matching contract:

```python
def _is_qwen3_thinking_model(model: str) -> bool:
    """Return whether a Qwen model name is in the supported Qwen3 thinking scope."""
```

The helper must normalize the model name with `normalize_model_name(model)` and return:

- `True` for `qwen3` and normalized repository-prefixed ids that end in a `qwen3` segment, such as `Qwen/Qwen3`.
- `True` when a normalized `qwen3` segment is followed by `-` and a digit, such as `qwen3-32b` or `Qwen/Qwen3-32B`.
- `True` when a normalized `qwen3` segment is followed by `.` and a digit, such as `qwen3.6-34b-80l-fable-5-heretic` or `hiebo/Qwen3.6-34B-80L-Fable-5-Heretic`.
- `False` for `qwen2.5-*`, `qwen30-*`, `qwen3coder`, `qwen3-coder`, `qwen3-vl`, `qwen3vl`, repository-prefixed Qwen3-Coder or Qwen3-VL ids, and unknown Qwen variants.

`detect_backend_descriptor()` must pass the model name or normalized model name into `_thinking_strategy(...)`; `_thinking_strategy(...)` must not rely on `model_family` alone for Qwen3 support.

Qwen3 provider request mapping:

```python
extra_body = {
    "chat_template_kwargs": {
        "enable_thinking": True,
    },
}
```

Qwen3 disabled enforcement mapping:

```python
extra_body = {
    "chat_template_kwargs": {
        "enable_thinking": False,
    },
}
```

Qwen visible thought normalization:

```text
input:  "<think>private reasoning</think>{\"ok\": true}"
output: "{\"ok\": true}"
```

Unclosed Qwen visible thought normalization:

```text
input:  "visible prefix\n<think>private reasoning"
output: "visible prefix"
```

## LLM Call And Context Budget

Before:

- No additional LLM calls.
- Route-level thinking defaults to disabled.
- Gemma 4 enabled thinking adds provider request fields and one copied prompt trigger.
- Qwen route thinking requests are ignored as unsupported.

After:

- No additional LLM calls.
- No context-window cap changes; use the existing route `max_completion_tokens`.
- No new response-path blocking behavior beyond a route operator explicitly enabling Qwen3 thinking.
- Qwen3 disabled route calls add only a provider request flag and no prompt tokens.
- Qwen3 enabled route calls add the provider request flag plus a copied
  assistant prefill containing `<think>\n`, which adds a very small prompt-token
  increase and enables reasoning on the validated LM Studio GGUF path.
- If a Qwen provider emits hidden reasoning internally, latency and completion usage may rise only on routes where `*_THINKING_ENABLED=true`.
- Final live LLM sign-off adds 40 dialog-generator calls: 20 `tests/test_dialog_l3_surface_contract_live_llm.py` cases with Qwen thinking enabled and the same 20 cases with Qwen thinking disabled.
- Final live sign-off must be run one case at a time and inspected under `debug-llm` and `test-style-and-execution`; it is required evidence for plan completion.
- Local hosted artifacts may still report zero reasoning tokens even when the request mapping is correct. The sign-off artifact must record that behavior, not hide it.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/llm_interface/detection.py`: classify Qwen3-family model names into Qwen3 thinking strategies when route thinking is enabled or disabled.
- `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`: add Qwen3 `extra_body.chat_template_kwargs.enable_thinking` request mapping, explicit disabled mapping, and Qwen3 enabled assistant prefill while preserving caller message objects.
- `src/kazusa_ai_chatbot/llm_interface/contracts.py`: add Qwen visible `<think>` response stripping while preserving Gemma thought-channel stripping.
- `src/kazusa_ai_chatbot/llm_interface/route_report.py`: render `thinking_on` for `qwen3_enabled` as well as `gemma4_enabled`.
- `src/kazusa_ai_chatbot/llm_interface/README.md`: document Qwen3 support, Qwen disabled enforcement, and response normalization.
- `README.md`: update supported LLM/interface summary from Gemma-only thinking to Gemma 4 and Qwen3-family thinking.
- `docs/HOWTO.md`: update route variable notes to describe Qwen3 support and disabled-by-default semantics.
- `tests/test_llm_interface_contracts.py`: update detection and normalization tests.
- `tests/test_llm_interface_openai_provider.py`: update provider request mapping tests.
- `tests/test_llm_interface_route_report.py`: update optional feature tag tests.

### Create

- No new production module is expected.
- Create `test_artifacts/llm_reviews/qwen_dialog_l3_surface_thinking_comparison_<UTC>.md` during final sign-off. This is an agent-authored review artifact, not script-generated prose. It must include measured elapsed seconds for each of the 40 individual pytest invocations.

### Delete

- No files are deleted.

### Keep

- Keep `LLMCallConfig` and `LLMThinkingConfig` public fields unchanged.
- Keep existing route environment variables unchanged.
- Keep current Gemma 4 tests passing.
- Keep local hosted service setup outside source control and do not inspect `.env`.

## Overdesign Guardrail

- Actual problem: Qwen3.6 route thinking requests are currently ignored even though the project already has a route-level thinking toggle and Qwen3 documents an OpenAI-compatible thinking switch.
- Minimal change: Extend `llm_interface` model-family thinking strategy detection, OpenAI-compatible provider request mapping, response normalization, route diagnostics, and docs/tests.
- Ownership boundaries: Runtime modules own semantic prompts and route selection; `llm_interface` owns model-family detection, provider request fields, diagnostics, and response normalization; deterministic tests own request-contract proof; live probes only supply human-readable evidence.
- Rejected complexity: no new config fields, thinking budgets, preserve flags, stage prompt rewrites, Qwen slash-trigger injection, runtime probing, compatibility wrappers, provider fallback paths, route-level default changes, stage tuning, surfaced `reasoning_content`, or broad matching for ambiguous Qwen3-adjacent model names.
- Evidence threshold: Add rejected complexity later only after a separate approved plan cites a real model/server requirement that cannot be satisfied by the boolean Qwen3 request mapping and response normalization in this plan.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside `kazusa_ai_chatbot.llm_interface`, its deterministic tests, and its docs as out of scope unless the plan is updated and approved.
- The responsible agent may not change route defaults or enable thinking for any runtime stage.
- The responsible agent may not use the local live Qwen model to weaken deterministic request-contract tests.
- If the implementation discovers that Qwen3 disabled enforcement conflicts with a provider or test contract, stop and report the blocker instead of silently omitting the disable payload.
- If the plan and code disagree, preserve the plan's stated intent and report the discrepancy.

## Implementation Order

1. Parent adds or updates focused detection and normalization tests in `tests/test_llm_interface_contracts.py`.
   - Add a Qwen3 enabled detection test expecting `thinking_strategy == "qwen3_enabled"`.
   - Add a Qwen3 disabled detection test expecting `thinking_strategy == "qwen3_disabled"`.
   - Add supported Qwen3-name tests for `qwen3`, `qwen3-32b`, and `qwen3.6-34b-80l-fable-5-heretic`.
   - Add unsupported Qwen-name tests for `qwen2.5-32b`, `qwen30-32b`, `qwen3-coder`, and `qwen3vl`, expecting enabled thinking to remain `ignored_unsupported_model`.
   - Add complete and truncated Qwen `<think>` stripping tests.
   - Run: `venv\Scripts\python -m pytest tests\test_llm_interface_contracts.py -q`.
   - Expected before implementation: fail on Qwen3 strategy and Qwen stripping expectations.
2. Parent adds or updates provider mapping tests in `tests/test_llm_interface_openai_provider.py`.
   - Add a Qwen3 enabled backend descriptor fixture and assert `extra_body == {"chat_template_kwargs": {"enable_thinking": True}}`.
   - Add a Qwen3 disabled backend descriptor fixture and assert `extra_body == {"chat_template_kwargs": {"enable_thinking": False}}`.
   - Assert Qwen3 enabled provider messages append a copied `AIMessage(content="<think>\n")` prefill and do not mutate caller-owned message objects.
   - Run: `venv\Scripts\python -m pytest tests\test_llm_interface_openai_provider.py -q`.
   - Expected before implementation: fail because Qwen strategies are not mapped.
3. Parent adds or updates route-report tests in `tests/test_llm_interface_route_report.py`.
   - Add a `RouteDiagnostic` with `thinking_strategy="qwen3_enabled"` and assert `optional_feature == "thinking_on"`.
   - Add a `RouteDiagnostic` with Qwen disabled strategy and assert `optional_feature == "-"`.
   - Run: `venv\Scripts\python -m pytest tests\test_llm_interface_route_report.py -q`.
   - Expected before implementation: fail for `qwen3_enabled` optional tag.
4. Parent starts one production-code subagent with this approved plan, mandatory skills, failing test evidence, and a production boundary limited to `src/kazusa_ai_chatbot/llm_interface/*` plus docs.
5. Production-code subagent updates `detection.py`.
   - Keep `detect_model_family()` returning `model_family == "qwen"` for Qwen model names.
   - Add `_is_qwen3_thinking_model(model: str) -> bool` using the exact matching contract in `Contracts And Data Shapes`.
   - Update `detect_backend_descriptor()` and `_thinking_strategy(...)` so Qwen3 support has access to the model name or normalized model name.
   - Return `qwen3_enabled` when Qwen3-family thinking is enabled.
   - Return `qwen3_disabled` when Qwen3-family thinking is disabled.
   - Return `ignored_unsupported_model` for non-Qwen3 Qwen when thinking is enabled.
6. Production-code subagent updates `openai_compatible.py`.
   - Add Qwen3 enabled and disabled `extra_body.chat_template_kwargs.enable_thinking` mapping.
   - Add the Qwen3 enabled assistant prefill `<think>\n`, preserving caller-owned message objects and avoiding duplicate prefill if the last message already starts with `<think>`.
   - Preserve Gemma 4 `extra_body` and Gemma prompt-trigger behavior.
   - Ensure `_provider_messages()` rewrites only Gemma 4 thinking messages and Qwen3 enabled prefill messages.
   - Ensure `_chat_model_cache_key()` continues including `backend.thinking_strategy`, so enabled and disabled Qwen sessions do not share incompatible clients.
7. Production-code subagent updates `contracts.py`.
   - Add constants for Qwen `<think>` and `</think>` markers.
   - Extend `_normalize_response_content()` to call a Qwen stripping helper when `backend.model_family == "qwen"`.
   - Preserve existing Gemma stripping behavior exactly.
   - Keep `raw_response` untouched.
8. Production-code subagent updates `route_report.py`.
   - Include `qwen3_enabled` in optional feature rendering.
   - Do not render disabled Qwen enforcement as an optional feature.
9. Parent or production-code subagent updates docs.
   - Update `src/kazusa_ai_chatbot/llm_interface/README.md` first because it owns the runtime contract.
   - Update `README.md` and `docs/HOWTO.md` to match the ICD language.
   - State that Qwen3 support sends both request-level chat-template controls and, for validated LM Studio GGUF serving, the assistant prefill needed to activate reasoning.
10. Parent runs focused deterministic tests.
    - `venv\Scripts\python -m pytest tests\test_llm_interface_contracts.py tests\test_llm_interface_openai_provider.py tests\test_llm_interface_route_report.py -q`
11. Parent runs broader LLM-interface regression tests.
    - `venv\Scripts\python -m pytest tests\test_llm_interface_migration.py tests\test_llm_interface_reload.py tests\test_llm_interface_contracts.py tests\test_llm_interface_openai_provider.py tests\test_llm_interface_route_report.py -q`
12. Parent runs static greps.
    - `rg "ignored_unsupported_model" src tests docs README.md`
    - Expected: remaining matches must describe unsupported non-Qwen3 or unknown models, not Qwen3.6.
    - `rg "Gemma 4 thinking|Gemma-only|Gemma 4 thinking payload|qwen3_enabled|qwen3_disabled|enable_thinking" README.md docs\HOWTO.md src\kazusa_ai_chatbot\llm_interface tests`
    - Expected: docs and tests mention Qwen3 support consistently; no user-facing doc says thinking support is Gemma-only.
13. Optional parent live diagnostic under `debug-llm`.
    - Send one Qwen3.6 request through `LLInterface` with thinking enabled and one with thinking disabled.
    - Capture raw response content, `response_metadata.token_usage.completion_tokens_details.reasoning_tokens`, `additional_kwargs`, and normalized `LLMResponse.content`.
    - Record a human-readable review artifact. If reasoning tokens remain zero, record that the request mapping was sent but the local artifact did not activate reasoning.
    - Do not fail the plan solely because the local artifact reports zero reasoning tokens.
14. Parent starts one independent code-review subagent after planned deterministic verification passes, unless the user has explicitly required no subagent execution. If subagents are forbidden, record the fallback self-review and approved residual risk in `Execution Evidence`.
15. Parent remediates review findings only inside this plan's change surface and reruns affected deterministic verification.
16. Parent performs final Qwen dialog L3 with/without-thinking sign-off.
    - Use the existing `tests/test_dialog_l3_surface_contract_live_llm.py` harness introduced for dialog L3 surface contract review and previously used for Gemma-era thinking comparison in `development_plans/archive/completed/bugfix/dialog_evaluator_decommission_plan.md`.
    - Configure the dialog generator route explicitly for Qwen before each pytest subprocess:

```powershell
$env:DIALOG_GENERATOR_LLM_BASE_URL='http://localhost:1234/v1'
$env:DIALOG_GENERATOR_LLM_API_KEY='lm-studio'
$env:DIALOG_GENERATOR_LLM_MODEL='qwen3.6-34b-80l-fable-5-heretic'
Remove-Item Env:\DIALOG_LIVE_COLLECT_ONLY -ErrorAction SilentlyContinue
```

    - Run each of the following 20 pytest node ids one at a time with `$env:DIALOG_LIVE_THINKING='on'` and `$env:DIALOG_LIVE_PHASE='qwen_thinking_on_signoff'`.
    - After each run, inspect the emitted `test_artifacts/llm_traces/dialog_l3_surface_contract_live_llm__qwen_thinking_on_signoff__<case>__*.json` trace before running the next case.
    - Then run the same 20 pytest node ids one at a time with `$env:DIALOG_LIVE_THINKING='off'` and `$env:DIALOG_LIVE_PHASE='qwen_thinking_off_signoff'`.
    - After each off run, inspect its trace before running the next case.

```text
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_01_group_casual_reply
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_02_private_soft_reply
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_03_group_technical_comparison
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_04_python_code_block_preserved
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_05_json_example_preserved
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_06_magic_anchor_after_milk_tea_history
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_07_touch_refusal_boundary
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_08_spray_cooling_uncertainty
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_09_railway_correction_no_relationship_expansion
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_10_captured_banter_thinking_tail
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_11_casual_overloaded_plan
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_12_unknown_referent_clarification
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_13_insufficient_evidence_best_effort
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_14_future_schedule_no_commitment
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_15_background_work_ack
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_16_image_observation_uncertain
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_17_group_broadcast_public_conclusion
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_18_accepted_format_preference
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_19_third_party_status
tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_20_privacy_boundary
```

    - Command template for every individual case. Record `$elapsed.TotalSeconds` and `$LASTEXITCODE` before moving to the next case:

```powershell
$elapsed = Measure-Command { venv\Scripts\python.exe -m pytest <node-id> -q -s -m live_llm }
$exitCode = $LASTEXITCODE
$elapsed.TotalSeconds
if ($exitCode -ne 0) { exit $exitCode }
```

    - Do not batch these 40 real LLM runs into one pytest command.
    - If a case fails hard assertions, stop, inspect the trace, record the failure, and remediate only if the fix is inside this plan's approved change surface. If remediation would require prompt, dialog-agent, L3, cognition, or test-case contract changes, stop and request a plan update.
    - Create `test_artifacts/llm_reviews/qwen_dialog_l3_surface_thinking_comparison_<UTC>.md` after inspecting all 40 traces. The artifact must be authored by the agent from raw trace files and must include a side-by-side table for all 20 cases with: case id, on trace path, off trace path, on elapsed seconds, off elapsed seconds, elapsed-time delta, on final dialog summary, off final dialog summary, on assessment result, off assessment result, reasoning-token evidence if present, quality delta, regressions, and human attention points.
17. Parent records final sign-off evidence and completes lifecycle reconciliation after user approval.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure or baseline before production implementation starts.
- Production-code subagent: normally exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes. Execution on 2026-06-19 used user-requested single-agent fallback with no subagent.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: normally exactly one native subagent, started after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent; does not implement fixes. Execution on 2026-06-19 records a fallback parent self-review and user-approved residual risk instead.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused test contract established.
  - Covers: implementation steps 1-3.
  - Verify: run the three focused test files and record expected failures or baseline behavior in `Execution Evidence`.
  - Evidence: failing or baseline pytest output for detection, provider mapping, normalization, and route reporting.
  - Handoff: production-code subagent starts at Stage 2.
  - Sign-off: `Codex / 2026-06-19`; deterministic contract tests were updated and final passing evidence is recorded. Pre-implementation failing output was not preserved because execution pivoted through online/provider validation before code edits.
- [x] Stage 2 - production LLM-interface implementation complete.
  - Covers: implementation steps 4-8.
  - Verify: focused tests from Stage 1 pass.
  - Evidence: changed files, focused pytest output, and any blockers or deviations.
  - Handoff: documentation stage starts next.
  - Sign-off: `Codex / 2026-06-19`; `llm_interface` implementation completed with the validated Qwen3 assistant prefill added inside provider ownership.
- [x] Stage 3 - documentation updated.
  - Covers: implementation step 9.
  - Verify: static greps show no stale Gemma-only user-facing docs for thinking support.
  - Evidence: changed doc paths and grep output.
  - Handoff: regression verification starts next.
  - Sign-off: `Codex / 2026-06-19`; README, HOWTO, and LLM interface ICD now describe Gemma 4 plus Qwen3-family thinking support.
- [x] Stage 4 - deterministic verification complete.
  - Covers: implementation steps 10-12.
  - Verify: focused and broader LLM-interface tests pass; static greps match expected results.
  - Evidence: command output summary with exit status.
  - Handoff: optional live diagnostic or independent review starts next.
  - Sign-off: `Codex / 2026-06-19`; focused LLM-interface suite passed `38 passed`, static grep for stale Gemma-only wording returned no matches, and `git diff --check` exited 0 with line-ending warnings only.
- [x] Stage 5 - optional direct Qwen diagnostic complete or explicitly skipped.
  - Covers: implementation step 13.
  - Verify: if run, `debug-llm` review artifact exists and distinguishes request mapping from model behavior; if skipped, record why.
  - Evidence: artifact path or skip reason.
  - Handoff: independent code review starts next.
  - Sign-off: `Codex / 2026-06-19`; direct validation artifact exists at `test_artifacts/llm_reviews/qwen_thinking_validation_20260619T215322Z.md`, and the user separately confirmed the server log showed thinking on for `你是谁‘`.
- [x] Stage 6 - independent code review complete.
  - Covers: implementation steps 14-15.
  - Verify: review subagent inspected plan, diff, and evidence; parent reran affected tests after any fixes.
  - Evidence: review findings, fixes, rerun commands, residual risks, and approval status.
  - Handoff: final Qwen dialog sign-off starts next.
  - Sign-off: `Codex / 2026-06-19`; no subagent was invoked under the user's no-subagent execution constraint. Parent self-review found the original draft mismatch about Qwen message pass-through, corrected this plan to the validated assistant-prefill behavior, reran focused verification, and records the no-subagent review as an approved residual risk. Follow-up code review found and fixed a repository-prefixed Qwen3 model-id detection gap.
- [x] Stage 7 - final Qwen dialog L3 with/without-thinking sign-off complete.
  - Covers: implementation steps 16-17.
  - Verify: all 20 dialog L3 live cases were run individually with `DIALOG_LIVE_THINKING=on`, all 20 were run individually with `DIALOG_LIVE_THINKING=off`, each run's elapsed seconds were captured, traces were inspected one by one, and the comparison review artifact exists.
  - Evidence: 40 trace paths, 40 elapsed-time measurements, the review artifact path, per-case pass/fail and human quality judgment, and any approved residual risk.
  - Handoff: user approval for lifecycle completion or follow-up plan if sign-off exposes out-of-scope quality work.
  - Sign-off: `Codex / 2026-06-19`; final artifact exists at `test_artifacts/llm_reviews/qwen_dialog_l3_surface_thinking_comparison_20260619T215815Z.md`. User approved the result and observed that disabling thinking gives better dialog output, which matches the project default.

## Verification

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_llm_interface_contracts.py -q`
  - Expected after implementation: pass.
- `venv\Scripts\python -m pytest tests\test_llm_interface_openai_provider.py -q`
  - Expected after implementation: pass.
- `venv\Scripts\python -m pytest tests\test_llm_interface_route_report.py -q`
  - Expected after implementation: pass.

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_llm_interface_migration.py tests\test_llm_interface_reload.py tests\test_llm_interface_contracts.py tests\test_llm_interface_openai_provider.py tests\test_llm_interface_route_report.py -q`
  - Expected after implementation: pass.

### Static Greps

- `rg "ignored_unsupported_model" src tests docs README.md`
  - Expected after implementation: no match may state or test that Qwen3/Qwen3.6 thinking is unsupported.
  - Allowed matches: unsupported unknown models, non-Qwen3 Qwen models, and explicit unsupported fallback documentation.
- `rg "Gemma 4 thinking payload|Gemma-only|only emits a\\s+Gemma|qwen3_enabled|qwen3_disabled|enable_thinking" README.md docs\HOWTO.md src\kazusa_ai_chatbot\llm_interface tests`
  - Expected after implementation: docs describe Gemma 4 and Qwen3-family support consistently; tests cover Qwen3 strategies.

### Optional Direct Qwen Diagnostic

- Use `debug-llm` before running.
- Run one enabled and one disabled Qwen3.6 request through `LLInterface`.
- Expected evidence: human-readable review artifact with request config, backend descriptor, normalized content, raw metadata, reasoning token count, and note that local reasoning-token behavior is not deterministic acceptance.

### Final Qwen Dialog L3 Sign-Off

- Use `debug-llm` and `test-style-and-execution` before running.
- Configure `DIALOG_GENERATOR_LLM_*` for `qwen3.6-34b-80l-fable-5-heretic` on `http://localhost:1234/v1`.
- Run every exact pytest node id listed in implementation step 16 once with `DIALOG_LIVE_THINKING=on` and phase `qwen_thinking_on_signoff`.
- Run every exact pytest node id listed in implementation step 16 once with `DIALOG_LIVE_THINKING=off` and phase `qwen_thinking_off_signoff`.
- Each real LLM test must be run as a separate pytest command with `-q -s -m live_llm` wrapped by PowerShell `Measure-Command`; record elapsed seconds and inspect the trace before moving to the next case.
- Expected evidence: `test_artifacts/llm_reviews/qwen_dialog_l3_surface_thinking_comparison_<UTC>.md` with a side-by-side comparison of all 20 cases, links or paths to all 40 trace files, and 40 elapsed-time measurements.
- Completion rule: this gate passes only when every hard assertion passes or every failure has a recorded, user-approved residual risk, and the agent-authored review judges the Qwen thinking-on/off behavior acceptable for the dialog surface contract.

## Independent Plan Review

Review performed on 2026-06-19 before approval or execution. No separate subagent was used for this review pass because the current harness requires explicit subagent authorization for delegation; the drafting agent reread `development_plans/README.md`, the development-plan skill references, this plan, and the relevant `llm_interface` context from a fresh-review posture.

Review scope:

- Architecture alignment with the project boundary: adapter/debug client -> brain service -> queue/intake -> RAG -> cognition -> dialog -> persistence/consolidation -> scheduler/reflection.
- `llm_interface` ownership: runtime stages own prompts and route config; provider adapters own provider-specific request mapping and response normalization.
- Stage readiness: active short-term registry row exists, status remains `draft`, and implementation is blocked until user approval changes status to `approved` or execution starts under `in_progress`.
- Instruction completeness: contracts, change surface, implementation order, progress checklist, verification, evidence, and code-review gate are present.
- Creativity suppression: no prompt changes, compatibility shims, runtime probing, new public config fields, or route-default changes are authorized.

Findings and remediation:

| Finding | Severity | Remediation |
|---|---|---|
| Target State allowed `qwen3_disabled` or another equivalent disabled strategy, leaving a decision point for implementers. | Blocker | Fixed by requiring the stable `qwen3_disabled` strategy everywhere. |
| The plan said Qwen3 detection should use normalized model-name text but did not state how `_thinking_strategy()` would receive model text, while current code only passes `model_family`. | Blocker | Fixed by requiring `detect_backend_descriptor()` and `_thinking_strategy(...)` to pass model or normalized model text. |
| Qwen3-family matching was too broad and could accidentally include `qwen30`, `qwen3-coder`, or `qwen3vl`. | Blocker | Fixed by adding an exact `_is_qwen3_thinking_model(model: str) -> bool` contract and positive/negative tests. |
| The plan omitted the requested `Independent Plan Review` section. | Non-blocking after remediation | Fixed by adding this review record before the independent code-review gate. |

Post-review user-required sign-off update on 2026-06-19:

| Requirement | Remediation |
|---|---|
| Final sign-off must capture with-thinking and without-thinking results for the 20 dialog-agent real LLM cases on Qwen, following the prior Gemma-era pattern. | Added mandatory Stage 7, exact Qwen route setup, exact 20 pytest node ids, one-case-at-a-time execution rules, 40 required trace files, 40 elapsed-time measurements, and `test_artifacts/llm_reviews/qwen_dialog_l3_surface_thinking_comparison_<UTC>.md`. |

Review status: passed after the remediations above. Execution later proceeded
under direct user instruction, and this plan is now reconciled as completed.

## Independent Code Review

Run this gate after all required `Verification` commands pass and before final sign-off. The parent agent must create one independent code-review subagent through the current harness's native subagent capability. If native subagents are unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, documentation, and command artifact.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Code quality and design weaknesses, including provider ownership boundaries, hidden fallback paths, compatibility shims, prompt pollution, response-normalization leaks, brittle model-name detection, stale static grep expectations, and avoidable blast radius.
- Regression quality, including focused tests for enabled and disabled Qwen3 request mapping, unsupported Qwen variants, Gemma preservation, response normalization, diagnostics, docs, and any live diagnostic evidence.

The parent agent fixes concrete findings directly only when the fix is inside the approved change surface or this review gate explicitly allows review-only test or documentation corrections. If a fix would cross the approved boundary or alter the contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Qwen3-family enabled route thinking maps to the stable `qwen3_enabled` strategy.
- Qwen3-family disabled route thinking maps to an explicit provider disable strategy and sends `enable_thinking=False`.
- Non-Qwen3 Qwen thinking requests remain unsupported.
- Qwen provider messages are not rewritten with `/think` or `/no_think`.
- Gemma 4 enabled and disabled behavior remains unchanged.
- Caller-facing content strips visible Qwen `<think>` spans while preserving raw responses.
- Startup route diagnostics show `thinking_on` for `qwen3_enabled` and no feature tag for Qwen disabled enforcement.
- `README.md`, `docs/HOWTO.md`, and `src/kazusa_ai_chatbot/llm_interface/README.md` document the new behavior accurately.
- Focused tests, regression tests, static greps, and independent code review pass or are recorded with approved residual risk.
- The final Qwen dialog L3 sign-off artifact captures thinking-on and thinking-off results for all 20 dialog live cases and records an acceptable human quality judgment or user-approved residual risk for every case.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Local Qwen artifact accepts flags but still produces zero reasoning tokens | Treat implementation as deterministic request-contract support and make live diagnostic non-blocking evidence | Optional `debug-llm` review artifact records request mapping and raw metadata |
| Qwen template defaults enable thinking despite route config false | Send explicit Qwen3 `enable_thinking=False` when route thinking is disabled | Provider mapping test for Qwen disabled strategy |
| Visible Qwen `<think>` leaks into parser-visible JSON content | Strip complete and truncated Qwen thought spans during response normalization | `LLMResponse.from_raw` Qwen normalization tests |
| Broad Qwen matching overclaims support for older Qwen models | Detect Qwen3-family strategy only from normalized model names | Detection test for non-Qwen3 Qwen unsupported behavior |
| Ambiguous Qwen3-adjacent names are accidentally treated as thinking-capable | Match only `qwen3`, `qwen3-<digit>`, and `qwen3.<digit>` prefixes | Negative tests for `qwen30-32b`, `qwen3-coder`, and `qwen3vl` |
| Provider request mapping changes Gemma behavior | Keep Gemma strategy branches and message rewriting isolated | Existing Gemma provider tests plus focused regression command |
| Qwen thinking support changes provider request mapping but still hurts dialog quality | Require final 20-case dialog L3 live comparison with thinking on and off before completion | `qwen_dialog_l3_surface_thinking_comparison_<UTC>.md` review artifact |

## Execution Evidence

- Focused test baseline: deterministic tests were updated for Qwen3 detection,
  provider mapping, response normalization, and route reporting. A preserved
  pre-implementation failing pytest transcript is not available because the
  user required online/provider validation before code edits and execution then
  proceeded directly. Final focused evidence is recorded below.
- Production implementation: completed in `src/kazusa_ai_chatbot/llm_interface`.
  `detection.py` now returns `qwen3_enabled` and `qwen3_disabled` for supported
  Qwen3-family model names; `providers/openai_compatible.py` sends
  `chat_template_kwargs.enable_thinking` and appends the validated Qwen3
  `<think>\n` assistant prefill when enabled; `contracts.py` strips visible
  Qwen `<think>...</think>` spans; `route_report.py` renders `thinking_on` for
  Qwen3 enabled strategies.
- Documentation update: completed in `README.md`, `docs/HOWTO.md`, and
  `src/kazusa_ai_chatbot/llm_interface/README.md`. A stale Gemma-only wording
  scan with `rg "Gemma-only|limited to Gemma 4|only adds the\s+Gemma|only emits a\s+Gemma" README.md docs\HOWTO.md src\kazusa_ai_chatbot\llm_interface\README.md`
  returned no matches.
- Regression verification: final focused LLM-interface command
  `venv\Scripts\python.exe -m pytest tests\test_llm_interface_contracts.py tests\test_llm_interface_openai_provider.py tests\test_llm_interface_route_report.py tests\test_llm_interface_migration.py tests\test_llm_interface_reload.py -q`
  passed with `38 passed in 2.61s`.
- Static and compile verification: `venv\Scripts\python.exe -m py_compile tests\test_dialog_l3_surface_contract_live_llm.py`
  passed with no output. `git diff --check` exited 0 with Git CRLF warnings
  only.
- Optional live diagnostic: completed in
  `test_artifacts/llm_reviews/qwen_thinking_validation_20260619T215322Z.md`.
  Direct `LLInterface` validation showed disabled Qwen produced
  `reasoning_tokens=0` in `9.825s`, while enabled Qwen produced
  `reasoning_tokens=1310` in `18.055s`. The user then asked for a manual
  `你是谁‘` thinking-on probe and confirmed from the server log that thinking
  was on.
- Independent code review: no subagent was invoked under the user's no-subagent
  execution constraint. Parent fallback self-review found and fixed a plan
  mismatch: the draft still claimed Qwen messages would pass through unchanged,
  but validated LM Studio GGUF activation required an assistant prefill. The
  plan was corrected to match the actual implementation and final focused
  verification was rerun. Residual risk is recorded as user-approved for this
  plan lifecycle.
- Follow-up code review: parent review found one concrete issue in
  `src/kazusa_ai_chatbot/llm_interface/detection.py`: Qwen3 detection matched
  only normalized names starting with `qwen3`, so a configured model id such as
  `hiebo/Qwen3.6-34B-80L-Fable-5-Heretic` would not activate the Qwen3 thinking
  strategy even though the bare local alias did. The fix changed detection to a
  segment-based normalized-name regex, added positive coverage for
  `hiebo/Qwen3.6-34B-80L-Fable-5-Heretic`, `Qwen/Qwen3-32B`, and `Qwen/Qwen3`,
  and added negative coverage for repository-prefixed Qwen3-Coder and Qwen3-VL
  ids. Focused `tests\test_llm_interface_contracts.py -q` passed with
  `14 passed in 0.77s`; the broader LLM-interface regression command passed
  with `38 passed in 2.61s`; compile, stale Gemma-only wording grep, and
  `git diff --check` passed.
- Final Qwen dialog L3 sign-off: completed in
  `test_artifacts/llm_reviews/qwen_dialog_l3_surface_thinking_comparison_20260619T215815Z.md`
  with raw JSONL evidence at
  `test_artifacts/llm_traces/qwen_dialog_l3_surface_thinking_signoff_20260619T215815Z.jsonl`.
  All 40 live LLM pytest node runs were executed one case at a time and
  inspected. Thinking-on: 20/20 traces had nonzero reasoning tokens, total
  `40191`, average elapsed `29.480s`, assessment `14/20`. Thinking-off: 20/20
  traces had `reasoning_tokens=0`, average elapsed `4.620s`, assessment
  `14/20`. The user approved the result and observed that disabling thinking
  gives better dialog output; this supports the existing project default that
  thinking remains disabled by default.
