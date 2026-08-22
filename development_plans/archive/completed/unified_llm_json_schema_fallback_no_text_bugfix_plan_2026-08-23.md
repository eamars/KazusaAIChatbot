# Unified LLM JSON Schema Fallback Without Text Bugfix Plan

## Summary

- **Goal:** Replace the structured-output-to-text retry introduced by commit
  `863f607e` with one provider-native JSON Schema retry.
- **Status:** `completed`
- **Plan type:** Corrective lightweight bugfix contract
- **Scope boundary:** One OpenAI-compatible provider adapter, its focused
  transport tests, and final LLM-interface documentation.
- **Change direction:** Keep `json_object` as the structured primary request.
  Retry a recognized unsupported-`json_object` rejection once with the fixed
  generic JSON Schema. Propagate any fallback failure to the caller.
- **Acceptance state:** Parent reviewed and accepted the frozen code and test
  diff on 2026-08-23; verification evidence and lifecycle closeout are
  complete.

## Confirmed Decisions

1. Structured requests never enter plain-text transport as a recovery path.
2. A recognized endpoint rejection of `json_object` receives exactly one
   retry with the generic JSON Schema defined in this plan.
3. A rejected or failed JSON Schema retry raises through the existing caller
   boundary. The provider performs no third attempt.
4. Explicit `output_mode="text"` remains valid for the three intentionally
   free-form configurations established by the completed predecessor plan.
5. The change is surgical: provider transport only, focused contract tests,
   one final live spot check, and documentation closeout after code acceptance.
6. The coding executor is the same single Luna executor used for the
   predecessor work: `unified_structured_output_executor`, model
   `gpt-5.6-luna`, `max` reasoning, standard-speed runtime lane.

## Current Contract Audit

### Pipeline roles

- Semantic stages choose structured or intentionally free-form output through
  `LLMCallConfig.output_mode`.
- `OpenAICompatibleProvider` maps that public choice to provider request
  fields and owns the bounded transport retry.
- Existing stage parsers, validators, JSON repair, regeneration caps, and
  fail-closed dispositions own the returned content after transport succeeds.

### Observed failure

- Commit `863f607e` retries a recognized `json_object` rejection with
  `output_mode="text"`.
- The configured production-compatible endpoint rejects `json_object` and
  reports that `response_format.type` must be `json_schema` or `text`.
- A direct probe and a `ChatOpenAI` probe both accepted the generic JSON Schema
  payload fixed below and returned a JSON object.
- Production trace
  `test_artifacts/diagnostics/llm_trace_llmtrace_7ff0bfe648484f42ba963520eaa90052_20260822T141110Z.json`
  demonstrates why text recovery cannot satisfy the structured contract.

### Smallest current contract

```text
Semantic question: none; this is deterministic provider capability handling.
Inputs: public output_mode, provider response, and narrowly classified rejection.
Output: one structured provider response or the provider exception.
Deterministic owner: OpenAICompatibleProvider.
Rejected complexity: prompt changes, per-stage schemas, capability caches,
feature flags, extra probes, compatibility shims, and additional retries.
```

## Scope And Change Direction

Replace only the provider's current text retry. Preserve the public
`LLMCallConfig` values and caller behavior:

```text
output_mode="json_object"
    -> request {"type": "json_object"}
    -> success: return response
    -> recognized unsupported-json_object rejection:
         retry once with the fixed generic json_schema
         -> success: return response
         -> failure: propagate exception

output_mode="text"
    -> omit response_format
    -> return response or propagate exception
```

Unrelated primary errors propagate immediately. The JSON Schema request has no
recovery branch.

## Cutover Policy

Overall strategy: `bigbang`

| Area | Policy | Instruction |
|---|---|---|
| Structured provider recovery | bigbang | Replace the text retry directly with the fixed JSON Schema retry. |
| Explicit free-form calls | compatible | Preserve the three caller-selected text configurations exactly. |
| Public LLM config | compatible | Preserve `output_mode="json_object" | "text"` with the existing default. |
| Tests | bigbang | Replace the old text-fallback expectation with the new structured fallback and propagation contract. |

## Target State And Transport Contract

The fallback request uses this exact endpoint-validated payload:

```python
{
    "type": "json_schema",
    "json_schema": {
        "name": "kazusa_json_object",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": True,
        },
    },
}
```

This schema enforces an object transport while leaving field meaning and
validation with each existing stage. It does not create a stage schema system.
The provider cache identity distinguishes primary JSON-object, fallback JSON
Schema, and explicit text model construction so incompatible request settings
cannot share a cached model.

Official OpenAI documentation identifies `json_schema` as Structured Outputs,
requires a supplied schema, and prefers it over the older `json_object` mode:
<https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions/methods/create>.

## Mandatory Skills And Rules

- Apply `py-style` before changing the Python provider or test file.
- Apply `test-style-and-execution` before changing or running tests.
- Apply `local-llm-architecture` to preserve the deterministic transport
  boundary and minimal blast radius.
- Apply `debug-llm` for the single live structured-output spot check and its
  inspectable artifact.
- Apply `development-plan` for execution gates, evidence, and lifecycle
  closeout.
- Use `venv\Scripts\python` for Python and pytest commands.
- Preserve every pre-existing worktree change outside the owned files.

## Must Do

1. Replace the sync and async text retry with exactly one JSON Schema retry.
2. Reuse the existing narrow unsupported-`json_object` classification.
3. Make the fallback transport private to the provider; keep the public
   `LLMCallConfig.output_mode` contract unchanged.
4. Include the effective provider transport in provider-local cache identity.
5. Log that the bounded retry uses JSON Schema without logging secrets.
6. Propagate the fallback exception unchanged when the fallback fails.
7. Update only the focused provider tests named in the traceability matrix.
8. Run one final live spot check after deterministic verification and inspect
   its structured result.
9. Update the subsystem README and plan lifecycle records only after the
   parent accepts the code and verification evidence.

## Deferred And Kept Unchanged

- Keep all runtime prompts unchanged.
- Keep `LLMCallConfig`, `LLInterface`, session diagnostics, reload handling,
  response normalization, parsers, validators, JSON repair, stage retries,
  and failure dispositions unchanged.
- Keep the coding writer, RAG evaluator summarizer, and RAG finalizer in their
  existing explicit text mode.
- Keep per-stage JSON Schemas, schema registries, capability probing, endpoint
  capability caches, feature flags, additional fallback modes, and unrelated
  cleanup outside this plan.
- Keep the completed predecessor plan immutable as historical evidence.
- Keep concurrent worktree changes outside this plan.

## Execution Roles

### Parent architecture and acceptance owner

- **Responsibility:** Maintain the plan boundary, resolve hard issues, review
  the complete implementation diff and evidence, and control lifecycle state.
- **Owned surface:** This plan, registry entry, architecture decisions,
  handoff record, review findings, and final sign-off.
- **Authority:** Approve or reject implementation and request bounded
  remediation inside this plan. Production coding remains with the fixed
  executor.
- **Applicable skills:** `development-plan`, `local-llm-architecture`, and
  `debug-llm` for evidence review.
- **Capability floor:** System-level LLM transport architecture, Python diff
  review, and live-output evidence judgment.
- **Independence requirement:** Separate from the coding executor for final
  review and acceptance.
- **Acceptance output:** Written scope review, verification disposition, and
  lifecycle decision.
- **Gate:** Starts after owner approval; accepts only a scoped diff with every
  criterion evidenced.

### Fixed coding executor

- **Executor:** Reuse `unified_structured_output_executor`, the single
  project-native subagent used for the predecessor implementation.
- **Model:** `gpt-5.6-luna`.
- **Reasoning effort:** `max`.
- **Speed:** Standard-speed runtime lane.
- **Resolution mode:** Plan-scoped fixed execution constraint supplied by the
  owner. Only the owner may change it.
- **Responsibility:** Implement the complete provider correction, update the
  focused tests, run the bounded checks and final spot check, remediate parent
  findings, and perform documentation closeout after code acceptance.
- **Owned surface:** Only the files listed under `Change Surface`.
- **Authority:** Local implementation mechanics within the fixed transport
  contract; no architecture, prompt, or caller-contract changes.
- **Applicable skills:** `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `debug-llm`, and `development-plan`.
- **Capability floor:** Python async/sync provider code, OpenAI-compatible
  response formats, deterministic test design, and live LLM evidence review.
- **Independence requirement:** One coding executor; parent supplies separate
  review and acceptance.
- **Acceptance output:** Scoped diff, exact deterministic results, one
  inspected live artifact, and concise handoff.
- **Gate:** Starts only after plan approval and an explicit execution command;
  exits when the parent accepts all evidence.

Additional coding and review agents are outside this plan.

## Change Surface

### Modify during implementation

- `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`
  - Replace structured-to-text recovery with the fixed JSON Schema transport.
  - Separate cache identity by effective transport.
- `tests/test_llm_interface_openai_provider.py`
  - Replace the old text-fallback assertion and add fallback-failure
    propagation coverage.

### Modify during final documentation closeout

- `src/kazusa_ai_chatbot/llm_interface/README.md`
  - Document the JSON Schema retry and terminal propagation contract.
- `development_plans/README.md`
  - Maintain the plan registry and final lifecycle location.
- This plan file
  - Record execution evidence, status, and final handoff.

### Create

- `development_plans/active/bugfix/unified_llm_json_schema_fallback_no_text_bugfix_plan_2026-08-23.md`

### Keep

- Every production and test path not listed above.

## Test Impact And Traceability

| Source or governed artifact | Changed contract | Semantic owner | Exact deterministic pytest nodes | Supplemental live node | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py` | Recognized `json_object` rejection retries with the fixed generic JSON Schema; explicit text stays unformatted; fallback failure propagates | OpenAI-compatible provider | `tests/test_llm_interface_openai_provider.py::test_provider_retries_unsupported_json_object_with_json_schema`; `tests/test_llm_interface_openai_provider.py::test_provider_propagates_json_schema_fallback_failure` | `tests/test_dialog_agent_direct_live_llm.py::test_dialog_agent_direct_live_technical_numeric_comparison` | regular deterministic plus one real-LLM spot check | Structured calls silently degrade to text, reuse an incompatible cached model, or hide unsupported JSON Schema. |
| `src/kazusa_ai_chatbot/llm_interface/README.md` | Documented provider recovery matches accepted runtime behavior | LLM interface documentation | `tests/test_llm_interface_openai_provider.py::test_provider_retries_unsupported_json_object_with_json_schema` | none | documentation closeout backed by the owner test | Documentation continues to claim text recovery after the corrective cutover. |

## Verification

Run only this bounded set:

1. Confirm both exact deterministic nodes collect.
2. Run:
   `venv\Scripts\python -m pytest tests/test_llm_interface_openai_provider.py::test_provider_retries_unsupported_json_object_with_json_schema tests/test_llm_interface_openai_provider.py::test_provider_propagates_json_schema_fallback_failure -q`
3. Compile only the changed provider and provider-test files.
4. Run the final real-LLM spot check individually:
   `venv\Scripts\python -m pytest tests/test_dialog_agent_direct_live_llm.py::test_dialog_agent_direct_live_technical_numeric_comparison -q -s`
5. Inspect its durable trace and confirm that the configured endpoint follows
   `json_object` rejection -> JSON Schema success, returns parseable structured
   output, and produces usable `final_dialog` content. A test that stops before
   provider invocation does not satisfy this gate.
6. Run `git diff --check` on the owned files.

Baseline capture, full-suite execution, prompt snapshots, every-prompt replay,
and tests that freeze incidental generated wording are outside this plan.

## Agent Autonomy Boundaries

The fixed coding executor may choose private constant, helper, and parameter
names inside the provider while preserving the exact request sequence, schema,
cache separation, error propagation, and file boundary.

A required public contract change, prompt edit, caller edit, new schema shape,
new retry, extra file, or unavailable fixed executor pauses execution for an
owner decision and plan amendment. Test failures caused by unrelated worktree
changes are reported separately and remain outside remediation authority.

## Acceptance Criteria

1. Default structured calls first request `{"type": "json_object"}`.
2. Only a narrowly recognized unsupported-`json_object` rejection triggers one
   retry using the exact generic JSON Schema in this plan.
3. Structured recovery never constructs or invokes a text-mode model.
4. A JSON Schema fallback failure propagates to the caller immediately.
5. Unrelated primary provider failures propagate immediately.
6. Explicit `output_mode="text"` still omits `response_format` and preserves
   the three intentional free-form stages.
7. Provider cache identity separates JSON object, JSON Schema, and explicit
   text transports.
8. Both exact deterministic nodes pass.
9. The one live spot check is run individually, its trace is inspected, and
   it demonstrates real JSON Schema fallback with acceptable structured dialog.
10. The final diff contains only the listed files and preserves all concurrent
    worktree changes.
11. The parent reviews and accepts the code before documentation closeout and
    lifecycle completion.

## Progress Checklist

- [x] Record owner decisions and the endpoint-validated JSON Schema payload.
- [x] Define exact scope, roles, tests, and acceptance gates.
- [x] Receive owner approval and explicit execution command.
- [x] Capture the execution worktree baseline and fixed-executor handoff.
- [x] Implement the provider correction and focused tests.
- [x] Run bounded deterministic verification.
- [x] Run and inspect the single live spot check.
- [x] Complete parent review and bounded Luna remediation.
- [x] Freeze the accepted code diff.
- [x] Perform final documentation and lifecycle closeout.

## Execution Evidence

- The two exact deterministic nodes collected successfully: `2 collected in
  0.69s`.
- The two exact deterministic nodes passed together: `2 passed in 0.82s`.
- Compilation of the changed provider and provider-test files passed.
- The owned-path diff check passed; its only output was CRLF normalization
  notices.
- The exact named live pytest node was run individually. Its existing state
  fixture lacked the required `text_surface_output_v2`, so execution stopped
  before provider invocation and that command did not satisfy the live gate by
  itself. The live test and fixture remained unchanged under the strict change
  surface.
- A one-off in-memory invocation of the same technical numeric-comparison case
  supplied the current V2 surface contract. It exercised a real
  `json_object` rejection, exactly one JSON Schema retry, and a successful
  structured response.
- The inspected durable trace is
  `test_artifacts/llm_traces/dialog_agent_direct_live_llm__technical_numeric_comparison__20260822T215247245674Z.json`.
  Parsing and contract assessment passed with one visible dialog. The response
  preserved every supplied product name, number, unit, and use-case conclusion
  and introduced no automatic mention, Markdown table, unsupported fact, or
  exaggerated comparison.
- The parent reviewed and accepted the frozen implementation at provider hash
  `430e3c6d9bef3d1da20b8b81fea8dd796b1add48` and provider-test hash
  `f6b44a16f8476dbff0e5dc643f1f41b10ceb5f15`.

## Final Handoff

The OpenAI-compatible provider now preserves structured transport across its
single compatibility retry: `json_object` remains primary, a recognized
unsupported rejection retries once with the fixed generic JSON Schema, and
any fallback failure propagates. Explicit text remains caller-selected only,
and provider cache identity separates all three effective transports. The
named live-test fixture deviation is recorded above; the equivalent
current-contract invocation supplied the required real-provider evidence.
