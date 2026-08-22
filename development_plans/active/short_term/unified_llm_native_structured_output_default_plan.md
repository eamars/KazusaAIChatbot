# Unified LLM Native Structured Output Default Plan

## Summary

- **Goal:** Make provider-native JSON-object output the default for Kazusa LLM
  stages while preserving the existing parsers, evaluators, repair path, and
  the small set of intentionally free-form stages.
- **Status:** `approved`
- **Owner approval date:** 2026-08-22
- **Plan type:** Lightweight executable implementation contract
- **Execution authorization:** The owner approved this plan as executable.
  Production implementation starts on a separate explicit execution command.
- **Scope boundary:** One shared output-mode field, one provider mapping, three
  explicit free-form configurations, mechanical prompt cleanup, and small
  final verification.
- **Change direction:** Surgical compatible cutover. Native JSON-object mode is
  the default; recognized unsupported-feature responses receive one bounded
  retry through the existing text path.
- **Acceptance state:** The shared transport works, free-form stages remain
  free-form, existing parsing and repair still work, and the final spot checks
  pass.

## Confirmed Owner Decisions

1. Keep this fix lightweight and surgical.
2. Preserve existing runtime behavior and semantic ownership.
3. Use provider-native JSON-object output as the shared default instead of
   introducing per-stage schema classes or a new structured-output framework.
4. Keep existing parsers, validators, normalization, JSON repair, retries, and
   fail-closed behavior authoritative.
5. Limit prompt changes to serialization-only wording. Preserve field meaning,
   decision rules, grounding, refusal, silence, and character semantics.
6. Use final spot checks instead of baseline capture or exhaustive testing.
7. Keep unit coverage at the stable transport boundary. Prompt snapshots,
   per-prompt fixtures, and tests that freeze incidental wording are excluded.
8. Run only the exact small verification set listed in this plan.
9. Use one persistent `gpt-5.6-luna` coding subagent with `max` reasoning on
   the standard-speed runtime lane for this plan only.
10. Perform all documentation work as the final execution step. Complete code,
    verification, parent review, and remediation first. Then update the
    subsystem README, plan evidence and lifecycle, registry, and archive state
    in one documentation closeout pass.

## Mandatory Execution Order

Execute this plan in this order:

1. Implement production code, prompt changes, and the two transport checks.
2. Run the bounded verification and live spot checks.
3. Complete parent code review and all Luna remediation.
4. Freeze the accepted code diff.
5. Perform the final documentation closeout:
   - update `src/kazusa_ai_chatbot/llm_interface/README.md`;
   - record plan checklist and execution evidence;
   - mark the plan completed;
   - update the registry; and
   - archive the completed plan.

The plan, registry, subsystem README, and other documentation remain unchanged
during coding, verification, review, and remediation. A later code correction
returns execution to step 1, followed by a fresh final documentation closeout.

## Target State

```text
structured stage with default LLMCallConfig
    -> OpenAI-compatible provider requests {"type": "json_object"}
    -> existing LLMResponse.content
    -> existing parse_llm_json_output or local parser
    -> existing stage validator / normalization / repair / bounded failure

intentional free-form stage with output_mode="text"
    -> existing text request and response behavior
```

The native request guarantees an object-shaped JSON transport where the
configured endpoint supports it. Existing stage code continues to own field
shape, semantic validation, normalization, repair, and regeneration. This plan
does not introduce strict per-stage JSON Schema transport.

## Implementation Contract

### 1. Shared output mode

Add one immutable field to `LLMCallConfig`:

```python
output_mode: Literal["json_object", "text"] = "json_object"
```

`json_object` is the default because nearly every current Kazusa LLM stage
returns an object and already routes the result through an existing parser or
evaluator. `text` is the explicit mode for intentionally free-form stages.

Keep the `LLInterface.ainvoke(...)` and `LLInterface.invoke(...)` signatures
unchanged. Keep call sites unchanged unless they own a free-form config or
serialization-only prompt wording.

### 2. Provider mapping and fallback

The OpenAI-compatible provider maps `output_mode="json_object"` to the native
request format:

```json
{"response_format": {"type": "json_object"}}
```

The provider omits `response_format` for `output_mode="text"`.

When an endpoint returns a recognized unsupported-parameter or
unsupported-feature error naming `response_format` or JSON-object mode, the
provider retries that call once in text mode and logs the bounded fallback.
Schema, authentication, timeout, rate-limit, server, and unrelated request
errors retain their existing error behavior. LM Studio reload handling remains
unchanged.

Include `output_mode` in diagnostic and provider cache identity so a text model
instance and JSON-object model instance cannot share incompatible request
configuration.

### 3. Free-form exceptions

Set `output_mode="text"` only on these current free-form configurations:

- `src/kazusa_ai_chatbot/coding_agent/code_writing/programmer.py`
  - `_writing_programmer_llm_config`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
  - `_evaluator_summarizer_llm_config`
  - `_finalizer_llm_config`

The structured continuation assessor in the RAG evaluator keeps the default
`json_object` mode.

### 4. Prompt cleanup

For the prompt files listed under `Change Surface`, remove only generic
serialization enforcement such as:

- requests for JSON outside markdown fences;
- repeated `return only JSON` boilerplate;
- prose whose sole purpose is preventing surrounding explanations; and
- schema-shaped examples that merely demonstrate braces and quoting.

Keep semantic field names and concise field descriptions when the model needs
them to make the correct decision. Keep enum meanings, cardinality meaning,
grounding rules, evidence rules, and downstream semantic contracts. Keep
special repair and regeneration prompts intact because they own recovery.

## Fixed Execution Roles

### Parent architecture and control owner

- **Responsibility:** Maintain scope, resolve hard issues, approve prompt
  semantics, review the diff, and decide acceptance.
- **Owned surface:** This plan, registry lifecycle, architectural decisions,
  handoff records, review findings, and final sign-off.
- **Authority:** Read-only source inspection, plan amendments, scope decisions,
  and acceptance decisions.
- **Independence:** The parent performs review and sign-off while the coding
  executor performs implementation and remediation.
- **Acceptance output:** Reviewed diff, verification disposition, and final
  plan lifecycle record.

### Fixed coding executor

- **Executor:** One persistent project-native subagent.
- **Model:** `gpt-5.6-luna`.
- **Reasoning effort:** `max`.
- **Speed:** Standard-speed runtime lane.
- **Resolution mode:** Plan-scoped fixed execution constraint.
- **Responsibility:** Implement the complete bounded change, update the two
  transport tests, run the listed checks and spot checks, remediate parent
  review findings, and perform documentation closeout only after the parent
  accepts the code diff.
- **Authority:** Modify only the files in `Change Surface` and produce the
  listed evidence.
- **Required skills:** `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `debug-llm`, and `development-plan`.
- **Acceptance output:** Scoped diff, exact check results, live spot-check
  observations, and a concise handoff.

The same subagent identity is reused for implementation and remediation.
Availability or capability failure pauses execution for owner direction.
Additional coding and review subagents are outside this plan.

## Change Surface

### Core transport files

- `src/kazusa_ai_chatbot/llm_interface/contracts.py`
  - Add `output_mode` to `LLMCallConfig` with the JSON-object default.
- `src/kazusa_ai_chatbot/llm_interface/session.py`
  - Include `output_mode` in diagnostic identity.
- `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`
  - Map JSON-object and text modes, include mode in provider cache identity,
    and perform the one bounded unsupported-feature fallback.
- `src/kazusa_ai_chatbot/llm_interface/README.md`
  - In the final documentation closeout, document the default, text
    exceptions, and fallback boundary.

### Free-form configuration files

- `src/kazusa_ai_chatbot/coding_agent/code_writing/programmer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`

### Structured prompt-only files

Only serialization wording may change in these files:

- `src/kazusa_ai_chatbot/character_identity_growth/llm.py`
- `src/kazusa_ai_chatbot/coding_agent/code_action_loop/prompts.py`
- `src/kazusa_ai_chatbot/coding_agent/code_modifying/product_manager.py`
- `src/kazusa_ai_chatbot/coding_agent/code_writing/product_manager.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/anchor.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/authorization.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/goal_cognition.py`
- `src/kazusa_ai_chatbot/cognition_shared/character_carryover.py`
- `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/stages.py`
- `src/kazusa_ai_chatbot/consolidation/lane_router.py`
- `src/kazusa_ai_chatbot/consolidation/memory_units.py`
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
- `src/kazusa_ai_chatbot/local_context_resolver/stages.py`
- `src/kazusa_ai_chatbot/media_inspection/service.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence/selector.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence/workers/aggregate.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence/workers/filter.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence/workers/keyword.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence/workers/search.py`
- `src/kazusa_ai_chatbot/rag/live_context/selector.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence/selector.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence/workers/persistent_keyword.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence/workers/persistent_search.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence/workers/user_memory.py`
- `src/kazusa_ai_chatbot/rag/person_context/selector.py`
- `src/kazusa_ai_chatbot/rag/person_context/workers/list.py`
- `src/kazusa_ai_chatbot/rag/person_context/workers/lookup.py`
- `src/kazusa_ai_chatbot/rag/person_context/workers/relationship.py`
- `src/kazusa_ai_chatbot/rag/recall/review.py`
- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/web_search.py`
- `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py`
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`
- `src/kazusa_ai_chatbot/task_resolution/orchestrator.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/text_computation.py`

### Tests

- `tests/test_llm_interface_contracts.py`
- `tests/test_llm_interface_openai_provider.py`

### Keep unchanged

- `src/kazusa_ai_chatbot/llm_interface/interface.py`
- `src/kazusa_ai_chatbot/llm_interface/reload.py`
- `src/kazusa_ai_chatbot/utils.py`
- All stage parsers, evaluators, normalization, JSON repair, retry caps, and
  fail-closed dispositions.
- Route environment variables, model selection, persistence, adapters,
  scheduler, and delivery behavior.

## Test Impact And Traceability

This plan uses an owner-approved lightweight verification exception. Unit
coverage is limited to the durable transport boundary; prompt wording is
verified by final live spot checks rather than snapshots or per-prompt tests.

| Source or governed surface | Contract | Owner | Exact verification | Mode | Regression prevented |
|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/llm_interface/contracts.py` | JSON-object default and explicit text mode | Unified interface | `tests/test_llm_interface_contracts.py::test_call_config_defaults_to_json_object_output` | regular | Default or free-form mode drifts silently. |
| `src/kazusa_ai_chatbot/llm_interface/session.py`; `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py` | Provider mapping, cache separation, bounded unsupported fallback | OpenAI-compatible provider | `tests/test_llm_interface_openai_provider.py::test_provider_maps_json_object_text_and_unsupported_fallback` | regular | Native mode leaks into text calls or unsupported endpoints stop working. |
| Structured prompt-only files | Serialization wording changes while semantic decisions remain intact | Owning stage prompts | `tests/test_relevance_turn_settlement_live_llm.py::test_live_frontline_discards_clear_third_party_message`; `tests/test_dialog_agent_direct_live_llm.py::test_dialog_agent_direct_live_technical_numeric_comparison` | live-LLM spot check | Representative structured stages stop parsing or lose semantic behavior. |
| Free-form configurations | Explicit text mode preserves prose and artifact output | RAG finalizer and coding writer | `tests/test_persona_supervisor2_rag_supervisor2_live.py::test_rag_finalizer_live_preserves_visible_conversation_speaker` plus manual inspection of one writing-programmer artifact when that route is configured | live-LLM spot check | JSON mode wraps or suppresses intentionally free-form output. |

## Verification

Run only this bounded set after implementation:

1. `venv\Scripts\python -m pytest tests/test_llm_interface_contracts.py::test_call_config_defaults_to_json_object_output tests/test_llm_interface_openai_provider.py::test_provider_maps_json_object_text_and_unsupported_fallback -q`
2. Run these live cases one at a time and inspect their visible output:
   - `venv\Scripts\python -m pytest tests/test_relevance_turn_settlement_live_llm.py::test_live_frontline_discards_clear_third_party_message -q -s`
   - `venv\Scripts\python -m pytest tests/test_dialog_agent_direct_live_llm.py::test_dialog_agent_direct_live_technical_numeric_comparison -q -s`
   - `venv\Scripts\python -m pytest tests/test_persona_supervisor2_rag_supervisor2_live.py::test_rag_finalizer_live_preserves_visible_conversation_speaker -q -s`
3. When the coding-writer route is configured, run one ordinary writing request
   and inspect that it still returns one markdown-fenced artifact.
4. Run `git diff --check` on the owned files.

Baseline capture, full-suite execution, every-prompt replay, prompt snapshots,
and fixture-driven semantic redesign are excluded from this plan.

## Executor Autonomy Boundaries

The coding executor may choose local naming and small helper placement inside
the listed files while preserving the target contract. The change remains one
output-mode field, one provider mapping, three text exceptions, and mechanical
prompt cleanup.

Architecture changes, per-stage schema systems, feature flags, parallel APIs,
new provider abstractions, new retry loops, broad exception swallowing,
semantic prompt rewrites, and unrelated cleanup require a plan amendment.

If inspection finds another intentionally free-form production stage, the
executor reports its exact config and prompt to the parent. The parent may add
that exact config to the text-mode list when the existing prompt clearly owns
free-form output; broader ambiguity returns to the owner.

## Acceptance Criteria

1. `LLMCallConfig` defaults to native JSON-object mode.
2. The OpenAI-compatible provider sends the native JSON-object request for the
   default mode and omits it for explicit text mode.
3. A recognized unsupported native-output response receives one bounded text
   retry while unrelated errors retain existing behavior.
4. The coding writer, RAG evaluator summarizer, and RAG finalizer remain
   free-form.
5. Existing parsers, validators, JSON repair, retries, and failure dispositions
   remain functionally unchanged.
6. Prompt edits remove serialization-only wording while preserving semantic
   instructions.
7. The two transport checks pass.
8. The three live spot checks are run individually and their outputs remain
   usable and semantically appropriate.
9. The parent reviews the complete diff and records approval or sends bounded
   remediation to the fixed coding executor.

## Execution Checklist

- [ ] Capture the worktree baseline and exact owned files.
- [ ] Record the fixed Luna handoff.
- [ ] Implement the shared output mode and provider fallback.
- [ ] Mark the three free-form configurations as text.
- [ ] Apply mechanical serialization-only prompt cleanup.
- [ ] Add the two transport-boundary checks.
- [ ] Run the bounded verification set.
- [ ] Complete parent review and any Luna remediation.
- [ ] Freeze the accepted code diff.
- [ ] As the final step, update the subsystem README, record execution
      evidence, mark the plan completed, update the registry, and archive the
      plan.

## Current Handoff State

- **Plan status:** approved and executable.
- **Implementation:** ready for a separate explicit execution command.
- **Fixed executor:** one persistent `gpt-5.6-luna` subagent, `max` reasoning,
  standard-speed runtime lane.
- **Parent role:** architecture, hard-issue resolution, lifecycle, review, and
  acceptance.
