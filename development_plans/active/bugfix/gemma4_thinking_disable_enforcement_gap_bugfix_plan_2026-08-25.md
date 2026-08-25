# Gemma 4 Thinking Disable Enforcement Gap Bugfix Plan

## Summary

- Goal: Ensure every detected Gemma 4 route configured with thinking disabled sends the provider-side `enable_thinking: false` control.
- Status: in_progress
- Scope boundary: `kazusa_ai_chatbot.llm_interface` backend detection, OpenAI-compatible request mapping, deterministic provider tests, subsystem ICD, and one existing real-LLM relevance gate.
- Change direction: Add an explicit disabled-Gemma transport alongside the existing enabled-Gemma and disabled-Qwen transports.
- Acceptance state: Pending pre-fix deterministic reproduction, implementation, exact deterministic verification, one individually inspected real-LLM gate, and parent review.

## Scope And Change Direction

The production trace `llmtrace_b8e1e535c29d463380627781b7386f9e` failed in settled relevance with `openai.LengthFinishReasonError` after the configured Gemma 4 route exhausted its 512-token completion ceiling. The stage already owns `thinking=False`, but backend detection currently returns the generic `disabled` strategy for Gemma 4 and the provider therefore omits the explicit chat-template disable control. The active llama.cpp backend runs with automatic preserved reasoning, so omission leaves provider-default reasoning behavior in control of a latency-critical structured-output stage.

This plan corrects only provider request mapping. It preserves relevance prompts, completion budgets, semantic dispositions, turn settlement, cognition, dialog, adapters, persistence, route configuration, and failure policy.

## Mandatory Skills

- `development-plan`: governs this execution record and closure.
- `local-llm-architecture`: governs the minimal provider-boundary correction and rejected overhead.
- `py-style`: governs Python production and test changes.
- `test-style-and-execution`: governs deterministic and real-LLM verification.

## Mandatory Rules

- Keep provider-specific request mapping inside `kazusa_ai_chatbot.llm_interface`.
- Keep `LLMThinkingConfig(enabled: bool)` and all caller configs unchanged.
- Add no prompt wording, retry, fallback, route flag, token-budget increase, model alias list, or compatibility shim.
- Preserve Gemma 4 enabled-thinking trigger behavior and all Qwen behavior.
- Use scenario-neutral deterministic tests; test fixtures must not influence runtime prompts.
- Run the real-LLM gate one node at a time and inspect its durable artifact.
- Preserve every pre-existing worktree change outside the owned surface.

## Must Do

- Return an explicit disabled-Gemma strategy whenever `model_family == "gemma4"` and route thinking is disabled.
- Map that strategy to `chat_template_kwargs.enable_thinking == false` without adding a Gemma prompt trigger.
- Prove the composed detected-descriptor-to-provider request carries the false control.
- Document explicit disabled enforcement for both supported thinking families while retaining generic omission for unknown or unsupported families.

## Deferred

- Completion-limit recovery, semantic fallback, provider retry, and trace capture for provider exceptions.
- Relevance prompt, schema, token-budget, turn-settlement, cognition, dialog, adapter, database, or model-route changes.
- Provider capability probing and llama.cpp preset changes.
- Renaming the existing Qwen internal strategy.

## Target State

```text
Gemma 4 + thinking disabled
  -> thinking_strategy=gemma4_disabled
  -> extra_body.chat_template_kwargs.enable_thinking=false
  -> original caller messages remain unchanged

Gemma 4 + thinking enabled
  -> thinking_strategy=gemma4_enabled
  -> enable_thinking=true plus the existing copied /think trigger

Unknown family + thinking disabled
  -> thinking_strategy=disabled
  -> no provider-specific payload
```

The deterministic provider boundary owns enforcement. The semantic relevance stage continues to ask the same bounded question with the same inputs and output contract.

## Execution Roles

### Implementation And Verification

- Responsibility: Reproduce the missing Gemma disable mapping, implement the provider-boundary correction, and produce deterministic plus real-LLM evidence.
- Owned surface: `src/kazusa_ai_chatbot/llm_interface/detection.py`, `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`, `tests/test_llm_interface_contracts.py`, `tests/test_llm_interface_openai_provider.py`, and `src/kazusa_ai_chatbot/llm_interface/README.md`.
- Authority: Modify only the declared mapping, provider request field, tests, and ICD; run the mapped deterministic nodes, adjacent deterministic files, and one existing relevance live-LLM node.
- Applicable skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`.
- Capability floor: Production Python contract work, OpenAI-compatible request mapping, focused pytest verification, and real-LLM artifact inspection.
- Independence requirement: none.
- Acceptance output: Scoped diff, captured pre-fix failures, passing exact deterministic nodes, passing adjacent files, and an inspected Gemma live trace with a valid settled-relevance result below the ceiling.
- Gate: Start after baseline capture; finish when all mapped nodes pass and the live result is acceptable.
- Fixed execution constraint: The user-selected GPT-5.6 Luna subagent at max reasoning executes production code and tests. Only the user may change this constraint.

### Parent Architecture Review And Closure

- Responsibility: Maintain the plan, review scope and architecture, inspect evidence, and close the lifecycle record.
- Owned surface: This plan, its registry row, and read-only review of the implementation surface.
- Authority: Accept the implementation, request in-scope remediation, and archive the plan after all gates pass.
- Applicable skills: `development-plan`, `local-llm-architecture`, `py-style`, `test-style-and-execution`.
- Capability floor: System architecture review, source-to-test traceability, and live-output judgment.
- Independence requirement: Separate from delegated implementation.
- Acceptance output: Review result, residual-risk statement, and completed lifecycle record.
- Gate: Start after implementation verification; finish after every acceptance criterion is evidenced.

## Test Impact And Traceability

| Source or artifact | Changed contract | Semantic owner | Exact deterministic pytest nodes | Supplemental live node | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/llm_interface/detection.py` | Disabled Gemma 4 routes select an explicit disabled transport | LLM interface backend detection | `tests/test_llm_interface_contracts.py::test_describe_backend_enforces_disabled_thinking_for_gemma4` | `tests/test_relevance_turn_settlement_live_llm.py::test_live_multifragment_correction_uses_latest_intent` | deterministic unit plus real LLM | Gemma routes silently inherit backend-default reasoning |
| `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py` | The explicit disabled-Gemma strategy emits `enable_thinking: false` and preserves messages | OpenAI-compatible provider request mapping | `tests/test_llm_interface_openai_provider.py::test_provider_sends_gemma4_disabled_payload` | `tests/test_relevance_turn_settlement_live_llm.py::test_live_multifragment_correction_uses_latest_intent` | deterministic integration plus real LLM | Detection reports disabled while the composed provider request omits enforcement |
| `src/kazusa_ai_chatbot/llm_interface/README.md` | ICD distinguishes explicit supported-family disable controls from generic disabled omission | LLM interface contract documentation | `tests/test_llm_interface_contracts.py::test_describe_backend_enforces_disabled_thinking_for_gemma4` | none | deterministic contract | Documentation permits the production mapping gap to recur |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/llm_interface/detection.py`: add the explicit disabled-Gemma strategy selection.
- `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`: send the existing chat-template false control for disabled Gemma.
- `tests/test_llm_interface_contracts.py`: add the exact descriptor regression.
- `tests/test_llm_interface_openai_provider.py`: add the composed provider-request and message-preservation regression.
- `src/kazusa_ai_chatbot/llm_interface/README.md`: document the corrected contract.
- `development_plans/README.md`: register and later close this plan.

### Create

- This plan while active; archive it after closure.

### Keep

- Every prompt, stage config, completion cap, semantic contract, public dataclass, model route, persistence path, adapter path, and unrelated worktree change.

## Agent Autonomy Boundaries

The implementation role may choose local assertion placement and command order inside the declared surface. A new configuration field, prompt change, fallback, retry, token-budget change, provider probe, route change, strategy abstraction, or production file requires a plan amendment and user decision.

## Verification

1. Add the two exact scenario-neutral deterministic nodes and demonstrate both fail against the pre-fix implementation.
2. Run pytest collection for both exact node IDs.
3. Implement the explicit disabled-Gemma descriptor and provider mapping.
4. Run both exact deterministic nodes, then both adjacent LLM-interface test files.
5. Run `tests/test_relevance_turn_settlement_live_llm.py::test_live_multifragment_correction_uses_latest_intent` once against the configured real relevance route. Inspect its durable trace for model `gemma4-4090`, valid structured output, acceptable semantic behavior, and absence of length exhaustion.
6. Run `git diff --check` on the scoped diff and perform parent architecture review.

## Acceptance Criteria

- A detected Gemma 4 route with thinking disabled receives explicit `enable_thinking: false`.
- Disabled Gemma requests preserve caller-owned messages and add no `/think` trigger.
- Enabled Gemma behavior and all existing Qwen behavior remain valid.
- Unknown and unsupported families retain their existing omission behavior.
- Both exact deterministic nodes are collected and pass; both adjacent files pass.
- The single real-LLM relevance gate completes with an acceptable validated result and inspectable artifact.
- The scoped diff contains no prompt, token-budget, semantic, route, adapter, database, fallback, retry, or unrelated changes.

## Progress Checklist

- [x] Production trace and runtime exception verified.
- [x] Historical Qwen plan and current driver contract compared.
- [x] Worktree baseline and owned surface recorded.
- [x] Pre-fix deterministic regressions captured.
- [x] Production mapping and ICD implemented.
- [x] Exact and adjacent deterministic verification passed.
- [x] Real-LLM gate run once and inspected.
- [ ] Parent review passed and plan archived.

## Execution Evidence

- Failure anchor: QQ group `638473184`, platform message `264946698`, trace `llmtrace_b8e1e535c29d463380627781b7386f9e`, started `2026-08-25T16:33:27.399688+12:00`, completed failed at `2026-08-25T16:33:42.550065+12:00`, `final_dialog_count=0`.
- Runtime exception: `openai.LengthFinishReasonError` raised while parsing the `persona_relevance_agent` completion before cognition or trace-step persistence.
- Runtime route: startup diagnostics identify `RELEVANCE_AGENT_LLM` as `gemma4-4090` at `http://localhost:8080/v1`; llama.cpp reports automatic preserved reasoning.
- Baseline: the worktree contains separate agentic-resolver plan, architecture, README, and test changes. They remain outside this plan. This plan begins from commit `17a97490` for the completed Qwen fix.
- Architecture decision: enforce the caller's existing boolean at the deterministic provider boundary. Increasing the 512-token relevance ceiling, changing prompts, adding retries, or degrading semantic output would retain the request-contract defect and add latency or behavior drift.
- Runtime assignment: the implementation role is assigned to the user-selected GPT-5.6 Luna max subagent already used for the incident. Reusing its established provider-boundary context has the lowest expected total execution cost while satisfying the fixed execution constraint; the parent retains architecture and closure authority.
- Pre-fix reproduction: both exact deterministic nodes failed because disabled Gemma 4 resolved to the generic `disabled` strategy instead of `gemma4_disabled`.
- Implementation: detection now selects `gemma4_disabled` for detected Gemma 4 routes with thinking disabled. The OpenAI-compatible provider maps it to `chat_template_kwargs.enable_thinking=false` and preserves the original message sequence without a `/think` trigger.
- Deterministic verification: both exact nodes were collected and passed; the adjacent contract and provider files passed with `32 passed`; scoped `git diff --check` passed.
- Live-gate correction: the initially selected L21 node exposed an existing deterministic authoritative-participation shortcut, recorded no model response, and failed its unrelated native-reply-anchor assertion. It could not evidence provider transport. The plan mapping was corrected within the unchanged verification scope to L09, an existing settled-relevance case that invokes the configured model.
- Real-LLM verification: `tests/test_relevance_turn_settlement_live_llm.py::test_live_multifragment_correction_uses_latest_intent` passed once in 5.23 seconds. Artifact `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__L09_latest_correction__20260825T044924844198Z.json` records model `gemma4-4090`, 6,382 prompt characters, a complete 404-character structured response, 4,065 ms model duration, validated `response_action=proceed`, and no length exhaustion.
