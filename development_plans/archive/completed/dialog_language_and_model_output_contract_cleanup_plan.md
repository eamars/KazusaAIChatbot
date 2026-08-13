# dialog language and deterministic model-output contract cleanup

## Summary

- Goal: remove the final dialog renderer's hard-coded Chinese output directive while preserving Chinese requirements for cognition and internal reasoning, and remove deterministic metadata from model-facing output contracts.
- Status: completed
- Scope boundary: final visible-dialog rendering, Cognition V2 goal/action model contracts, conversation-progress recorder outputs, and local-context node artifacts.
- Change direction: make the dialog renderer defer to cognition-owned semantic context; make code-owned schema, identity, and boundary metadata deterministic before public validation and persistence.
- Acceptance state: implemented, verified, independently reviewed, and ready for archival.

## Confirmed Decisions

1. Chinese remains mandatory for COT/internal reasoning and Cognition V2 semantic output. This plan does not weaken cognition prompts, cognition validators, or internal-monologue language policy.
2. Only the final visible-dialog renderer loses the static `简体中文` output requirement. The main renderer and hard-failure repair renderer must both follow the upstream cognition/surface context without adding a new output-language field in this plan.
3. `final_dialog` remains the exact visible-output contract: one JSON object with only a non-empty `final_dialog` string list.
4. Public, storage, trace, and wire schemas retain their existing `schema_version` fields. The change removes schema metadata only from raw LLM-facing output contracts and binds it in deterministic code.
5. Known operation roles, resolver goal identity, recorder packet versions, and local-context producer metadata remain owned by their deterministic runtime boundaries. Semantic content remains model-owned.
6. The pre-existing untracked files observed during drafting remain outside this plan and must be preserved during execution.

## Scope And Change Direction

The target flow is:

```text
COT and Cognition V2 (Chinese contract retained)
  -> cognition-owned semantic surface
  -> dialog renderer without a forced output language
  -> final_dialog
  -> adapter delivery
```

For structured semantic producers, the target flow is:

```text
LLM semantic candidate
  -> canonical JSON parsing
  -> deterministic metadata and known-field binding
  -> existing public validator
  -> state, persistence, or downstream surface
```

The implementation removes the two hard-coded language sentences from
`dialog_agent.py`. It does not add a new language selector because the current
`TextSurfaceInputV2` contract has no explicit language field and the requested
change is to stop the final renderer from overriding cognition.

The implementation also removes model responsibility for the following
non-semantic fields:

- `relational_willingness.schema_version` in goal cognition.
- Known `selected_response_operation` role and selection fields that are copied from `required_selection_operations`.
- `resolver_goal_progress.schema_version` and `original_goal` in action selection.
- Scene and event recorder observation `schema_version` values.
- Local-context node artifact `schema_version`, `producer_node_id`, and `prompt_visible` metadata.

The model keeps semantic prose, semantic decisions, evidence references that it
must select, artifact semantic identity, artifact type classification, and
evidence projection content. Public validators continue to receive complete
typed objects after deterministic normalization.

## Mandatory Skills

- `development-plan` for lifecycle, ownership, exact traceability, and acceptance gates.
- `local-llm-architecture` for semantic ownership and model-facing contract boundaries.
- `debug-llm` for prompt changes and any real-LLM evidence artifacts.
- `test-style-and-execution` for deterministic and live test changes.
- `py-style` for every modified Python file.
- `cjk-safety` for Python files whose prompt strings retain or change CJK content.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification.
- Pass every raw LLM response through the canonical `parse_llm_json_output(...)` path before semantic evaluation.
- Keep cognition/COT Chinese constraints and the CJK relational-reason validator unchanged.
- Remove the legacy dialog language directive in one contract cutover; do not add a second prompt, alias, fallback, or compatibility vocabulary.
- Inject deterministic metadata before the existing public validator; do not weaken public validators to accept incomplete persisted or wire objects.
- Preserve the exact dialog, cognition, recorder, resolver, and local-context ownership boundaries.
- Run live LLM tests one case at a time, inspect each output, and emit the required debug artifact when live verification is used.
- Before implementation, record `git status --short`, the current commit, and the explicitly owned file set. Preserve the pre-existing untracked files observed during drafting.
- The production implementation gate was satisfied before execution: the plan was `in_progress` and the user explicitly authorized implementation.

## Runtime Or Resource Constraints

- The production-source implementation owner is one `deepseek_v4_flash_0731` subagent, resolved as the `deepseek-v4-flash` model with the runtime-provided high reasoning configuration.
- This is a plan-scoped fixed execution constraint supplied by the user. It applies only to production-source edits in this plan; changing it requires a user decision or plan amendment.
- The parent agent owns plan updates, test-file changes, test execution, evidence inspection, and final integration review. No second production-code subagent is used.

## Must Do

1. Remove the `新生成的对话使用简体中文` instruction from `_V2_DIALOG_GENERATOR_PROMPT` and `_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT` while preserving semantic fidelity, role direction, capability truth, source URL, code, enum-token, and exact JSON-output rules.
2. Add dialog prompt tests proving that the final renderer has no forced Chinese output directive while the cognition prompt tests still require Chinese semantic output.
3. Change goal-cognition model-facing contracts so `relational_willingness.schema_version` is code-injected and known operation carrier fields are reconstructed from the required operation. Preserve semantic operation text and any genuinely model-selected unknown endpoints.
4. Change action-selection model-facing contracts so resolver progress is a semantic delta only. An absent or empty current progress produces `null`; an existing progress carries `schema_version` and `original_goal` from validated runtime state while the model supplies only changed semantic fields.
5. Change scene and event recorder prompts to omit packet schema versions. The recorder attaches the canonical scene/event versions before the existing observation validators run.
6. Change the local-context node prompt to omit artifact schema, producer-node, and visibility metadata. The service binds those values from the active-node and visibility policy while retaining model-owned semantic artifact identity and evidence payload.
7. Update static, deterministic contract, propagation, and live-contract tests to prove both the reduced model-facing shapes and unchanged public shapes.
8. Review the final diff against this plan and record exact test collection and execution evidence before lifecycle approval.

## Execution Evidence

### Baseline and execution ownership

- Baseline commit: `ff85eae7be5509bd28752c0b549af5102ca92446`.
- Baseline status was captured before implementation. Existing dirty paths and
  untracked plans/fixtures/tests were preserved; the implementation scope was
  limited to the seven production owners named in this plan and their mapped
  tests.
- The production-source handoff used one `deepseek_v4_flash_0731` owner with
  the required acknowledgement turn followed by the execution turn. The
  bounded handoff reached its deadline after partial edits; those edits were
  reviewed, and the parent completed the remaining recorder and local-context
  production paths. No second production-code subagent was used.

### Deterministic verification

- `venv\\Scripts\\python -m scripts.validate_test_impact --base-ref HEAD --run`:
  39 exact impacted nodes collected and 39 passed in 11.42 seconds.
- The plan matrix command, including the four parametrized scene-rejection
  cases, collected and passed 28 tests in 0.92 seconds.
- The expanded cross-boundary regression batch collected and passed 256 tests
  in 1.53 seconds.
- Production-source `py_compile` passed for dialog, goal cognition, action
  selection, recorder, delta merge, local-context stages, and local-context
  service.
- `git diff --check` passed. Git reported only the repository's existing line
  ending conversion warnings.
- Static prompt audit passed for removal of the two final-dialog Chinese
  directives, removal of action/recorder/node model-facing metadata, removal
  of relational schema from the goal model contract, and retention of the
  Chinese cognition policy.
- The configured Ruff check was run across the changed production and test
  files. It reports existing repository style findings (including legacy
  import ordering and TRY004 rules in untouched surrounding code); no broad
  formatting rewrite was applied outside this plan's contract changes.

### Live LLM verification

- `test_live_dialog_generator_deepseek_returns_final_dialog_schema` was run
  individually with `-o addopts=""` and passed. Four cases returned parseable,
  non-empty `final_dialog` lists; the English upstream-surface case returned
  English visible wording.
- `test_live_dialog_generator_node_accepts_deepseek_output` was run
  individually with `-o addopts=""` and passed; the node preserved the native
  surface and returned a non-empty final dialog.
- Raw traces:
  `test_artifacts/llm_traces/dialog_generator_live_llm_contract__deepseek_final_dialog_schema__20260813T121359597177Z.json`
  and
  `test_artifacts/llm_traces/dialog_generator_live_llm_contract__node_deepseek_output.json`.
- Human-reviewed artifacts:
  `test_artifacts/llm_debug/dialog_generator_language_contract_review.md`
  and
  `test_artifacts/llm_debug/dialog_generator_node_language_contract_review.md`.

### Independent review

- Independent reviewer: `kazusa_plan_reviewer` (`Avicenna`).
- Initial review findings on raw metadata acceptance, language evidence, and
  missing ownership mapping were remediated with fail-closed rejection tests,
  an English upstream live case, and manifest entries.
- Final review verdict: `approve with follow-up`; the only follow-up was to
  record this execution evidence. No unresolved blocking implementation
  finding remains after the final source-artifact separation remediation.

## Deferred

- Adding an explicit `language` or `output_language` field to Cognition V2 or `TextSurfaceInputV2`.
- Removing Chinese requirements from semantic appraisal, goal cognition, action planning, surface planning, COT, conversation-progress prose, residue, consolidation, reflection, RAG, or other non-dialog stages.
- Removing schema metadata from public, storage, trace, resolver-state, surface-input, or surface-output contracts.
- Changing the adapter delivery contract or adding language conversion after `final_dialog`.
- Changing the separate coding-agent action-loop prompt contract.
- Changing artifact semantic IDs, artifact type semantics, evidence authority, persistence behavior, or resolver routing beyond deterministic metadata binding.
- Broad prompt wording cleanup unrelated to the fields listed in this plan.

## Target State

| Boundary | Model-owned content | Deterministic owner |
| --- | --- | --- |
| Final dialog renderer | Visible message wording and language selected by upstream cognition/context | Exact `final_dialog` JSON shape and delivery validation |
| Goal cognition | Goal semantics, relational stance, reasons, semantic operation, unresolved semantic endpoints | Relational schema version and known operation carrier fields |
| Action selection | Resolver progress semantic deltas | Progress schema version, original goal, and current-state merge |
| Conversation progress recorder | Scene facts, event observations, and semantic changes | Scene/event packet schema versions and delta/state materialization |
| Local-context node | Evidence summaries, projection content, semantic artifact identity/type | Artifact schema version, active producer node, and visibility policy |

The public result of each boundary remains structurally identical to the
current typed contract. Only the raw model-facing candidate is smaller and
semantic-focused.

## Cutover Policy

Overall strategy: bigbang for model-facing prompt contracts, compatible for
public typed boundaries.

| Area | Policy | Instruction |
| --- | --- | --- |
| Dialog renderer prompts | bigbang | Remove the forced language directive from both generation paths. |
| Cognition/action/recorder/local-context model outputs | bigbang | Stop requesting deterministic metadata and reject its model ownership in the new candidate shape. |
| Public and persisted contracts | compatible | Preserve existing schema fields and materialize them deterministically before validation or storage. |
| Tests | bigbang | Replace assertions that require model copies of deterministic fields with assertions for code-owned reconstruction. |

## Execution Roles

### implementation_owner

- Responsibility: implement the approved prompt and normalization changes and produce the mapped verification evidence.
- Owned surface: `src/kazusa_ai_chatbot/nodes/dialog_agent.py`, `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`, `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`, `src/kazusa_ai_chatbot/conversation_progress/recorder.py`, `src/kazusa_ai_chatbot/conversation_progress/delta_merge.py`, `src/kazusa_ai_chatbot/local_context_resolver/stages.py`, `src/kazusa_ai_chatbot/local_context_resolver/service.py`, and the test paths listed in the traceability matrix.
- Authority: may modify the owned production and test paths only after explicit implementation authorization and an approved or `in_progress` plan; may run deterministic and one-at-a-time live verification; may update execution evidence.
- Applicable skills: `development-plan`, `local-llm-architecture`, `debug-llm`, `test-style-and-execution`, `py-style`, and `cjk-safety`.
- Capability floor: able to trace prompt contracts through parsing, semantic evaluation, deterministic normalization, public validation, and final delivery across multiple subsystems.
- Independence requirement: separate from the code-review role.
- Acceptance output: source and test diff within the owned surface, exact mapped pytest results, prompt-contract evidence, and recorded deviations or residual risks.
- Gate: approved or `in_progress` plan, explicit user implementation authorization, execution baseline captured, and no unresolved plan amendment.

### independent_code_reviewer

- Responsibility: inspect the completed diff and evidence for scope, semantic ownership, contract preservation, and verification completeness.
- Owned surface: read-only review of the plan, changed files, mapped tests, and verification artifacts.
- Authority: may issue findings and pass or fail the review gate; may not remediate or authorize extra scope.
- Applicable skills: `development-plan`, `local-llm-architecture`, `debug-llm`, and `test-style-and-execution`.
- Capability floor: able to review cross-stage LLM prompt contracts and deterministic boundary normalization.
- Independence requirement: must be independent of the implementation owner.
- Acceptance output: written review verdict with every blocking finding resolved or explicitly accepted.
- Gate: implementation diff and mapped verification evidence are complete; exit requires separate remediation and re-review when findings are blocking.

## Test Impact And Traceability

| Repository path | Changed symbol or contract | Semantic owner | Exact deterministic pytest node IDs | Supplemental live node IDs | Test mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | `_V2_DIALOG_GENERATOR_PROMPT`, `_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT`, final-dialog boundary | Dialog renderer | `tests/test_dialog_agent.py::test_v2_prompt_describes_surface_renderer_boundary`; `tests/test_dialog_agent.py::test_dialog_generator_repairs_unresolved_context_once`; `tests/test_dialog_agent.py::test_dialog_agent_returns_final_dialog_and_target` | `tests/test_dialog_generator_live_llm_contract.py::test_live_dialog_generator_deepseek_returns_final_dialog_schema`; `tests/test_dialog_generator_live_llm_contract.py::test_live_dialog_generator_node_accepts_deepseek_output` | deterministic prompt/structure plus one-at-a-time live LLM | Prevents the final renderer from imposing Chinese after cognition and prevents accidental changes to final-dialog delivery. |
| `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` | goal output contract, relational willingness normalization, selected operation binding | Goal cognition | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_emits_selected_response_operation`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_output_contract_keeps_existing_schema`; `tests/test_cognition_core_v2_prompt_contract_guidance.py::test_goal_repair_feedback_preserves_cross_namespace_authority`; `tests/test_cognition_core_v2_relational_willingness.py::test_ordinary_goal_draft_carries_current_episode_decision` | none | deterministic fake-LLM contract and normalization | Prevents the model from copying schema and caller-known role metadata while preserving the public relational and operation contracts. |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py` | `ACTION_PLANNING_PROMPT`, `_validate_goal_progress_choice` | Action selection | `tests/unit/cognition_core_v2/test_action_selection.py::test_goal_progress_model_output_omits_protocol_metadata`; `tests/unit/cognition_core_v2/test_action_selection.py::test_goal_progress_binds_protocol_metadata_from_current_state`; `tests/unit/cognition_core_v2/test_action_selection.py::test_selected_intention_preserves_selected_response_operation` | none | deterministic contract and state-merge tests | Prevents model-authored schema/original-goal drift and preserves resolver progress continuity. |
| `src/kazusa_ai_chatbot/conversation_progress/recorder.py` | scene/event recorder prompts and parsed-candidate normalization | Conversation-progress recorder | `tests/test_conversation_progress_recorder.py::test_recorder_prompts_keep_protocol_metadata_code_owned`; `tests/test_conversation_progress_recorder.py::test_recorder_attaches_schema_versions_before_validation`; `tests/test_conversation_progress_v2_contract.py::test_exact_v2_packet_with_bson_expiry_is_accepted` | none | deterministic prompt and public-contract tests | Prevents recorder models from copying packet versions while preserving exact scene/event state contracts. |
| `src/kazusa_ai_chatbot/conversation_progress/delta_merge.py` | scene/event observation validation boundary | Conversation-progress state materializer | `tests/test_conversation_progress_stage12_architecture.py::test_scene_observation_and_event_batch_compose_without_semantic_repair`; `tests/test_conversation_progress_v2_contract.py::test_scene_observation_rejects_operational_or_future_fields` | none | deterministic reducer and contract tests | Prevents reduced model candidates from bypassing exact semantic-field validation or changing persistence ownership. |
| `src/kazusa_ai_chatbot/local_context_resolver/stages.py` | `_NODE_PROMPT` artifact output shape | Local-context node semantic producer | `tests/test_local_context_resolver_standalone.py::test_stage_prompts_keep_source_field_and_time_boundaries`; `tests/test_local_context_resolver_standalone.py::test_node_artifact_binds_code_owned_metadata` | none | deterministic prompt and normalization tests | Prevents the node model from copying schema, active-node identity, or visibility metadata. |
| `src/kazusa_ai_chatbot/local_context_resolver/service.py` | `_validated_artifact_for_node` deterministic binding | Local-context runtime boundary | `tests/test_local_context_resolver_standalone.py::test_node_artifact_binds_code_owned_metadata`; `tests/test_local_context_resolver_contracts.py::test_node_and_artifact_contracts_validate_source_owned_evidence` | none | deterministic boundary and public-contract tests | Preserves valid public artifacts after the model-facing shape is reduced. |
| `tests/test_cognition_prompt_contract_text.py` | cognition-language guardrail | Cognition prompt policy | `tests/test_cognition_prompt_contract_text.py::test_generated_semantic_prompts_preserve_language_policy` | none | deterministic static prompt | Prevents the dialog cleanup from accidentally removing the mandatory Chinese COT/cognition contract. |

## Change Surface

### Delete

None.

### Modify

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: remove only the two final-dialog Chinese output directives; preserve all semantic and structural instructions.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: reduce model-facing relational and selected-operation contracts; deterministically restore public fields.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`: reduce resolver-progress model output and bind protocol fields from validated current state.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`: remove recorder schema versions from prompts and attach them before existing validators.
- `src/kazusa_ai_chatbot/conversation_progress/delta_merge.py`: preserve exact public validation while accepting only the normalized recorder boundary shape.
- `src/kazusa_ai_chatbot/local_context_resolver/stages.py`: remove code-owned artifact metadata from the node prompt.
- `src/kazusa_ai_chatbot/local_context_resolver/service.py`: bind node artifact metadata from active-node and visibility policy before public validation.
- Existing test paths named in `Test Impact And Traceability`: replace metadata-copy assertions with deterministic-binding assertions and add the named owner tests.

### Create

No new production modules, compatibility modules, or migration scripts.

New test functions are added inside the existing owner test modules named in
the traceability matrix.

### Keep

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: public relational, cognition, and surface schemas, including the Chinese relational-reason validator.
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`: public resolver progress schema and validator.
- `src/kazusa_ai_chatbot/conversation_progress/models.py`: public scene/event/state shapes and schema versions.
- `src/kazusa_ai_chatbot/local_context_resolver/contracts.py`: public artifact and graph contracts.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: cognition-to-surface ownership and semantic projection.
- All COT/cognition Chinese prompt directives outside the final dialog renderer.
- Adapter delivery behavior after `final_dialog` is accepted.
- The separate coding-agent action-loop prompt and its schema contract.
- Pre-existing untracked worktree files outside this plan.

## Agent Autonomy Boundaries

The implementation owner may choose helper names, local normalization layout,
assertion organization, and command order within the listed files. The owner
must preserve the target-state ownership table, public schemas, exact dialog
output shape, Chinese cognition policy, and deferred scope.

The owner must request a plan amendment before changing public contracts,
adding an explicit language field, altering semantic artifact identity,
changing evidence authority, modifying adapter behavior, touching the coding
agent, or introducing compatibility/fallback paths. The owner must not treat
model-output tolerance for old metadata as compatibility behavior.

## Verification

1. Capture the execution baseline, including the current commit, `git status --short`, and the explicit owned file set. Exclude the pre-existing untracked plan and test artifacts from the execution diff.
2. Run every deterministic pytest node listed in the traceability matrix and confirm each node is collected and executed.
3. Run the listed live dialog nodes individually, inspect each parsed `final_dialog`, and create the required human-readable LLM debug artifact.
4. Run the configured lint/style checks for every modified Python file, including CJK safety review for changed prompt literals.
5. Perform a static prompt audit proving that the two dialog renderer prompts contain no forced Chinese output directive, while cognition/COT prompts retain their Chinese policy.
6. Verify that public schema versions remain present after deterministic normalization and that raw model candidates are not allowed to author the removed metadata.
7. Compare the final diff with the execution baseline and confirm no unrelated production, adapter, persistence, routing, or coding-agent changes are present.

## Acceptance Criteria

- The main dialog generator and hard-failure repair prompt contain no hard-coded final-output language requirement.
- Cognition and COT prompts continue to require Chinese semantic output, and the relational-reason validator remains unchanged.
- `final_dialog` remains a non-empty list of visible strings under the exact existing JSON shape.
- Goal-cognition model candidates omit relational schema metadata and caller-known operation carrier fields; public validated results retain them.
- Resolver-progress model candidates omit schema metadata and `original_goal`; validated state retains the canonical runtime values.
- Scene/event recorder candidates omit schema versions; the recorder adds canonical versions before validation.
- Local-context node candidates omit code-owned artifact metadata; the service restores valid public artifacts deterministically.
- Every mapped deterministic test node is collected and passes; every required live node is run individually with inspected evidence.
- The independent code review passes with no unresolved blocking finding.
- Execution began only after this plan was `in_progress` and the user explicitly authorized the production change; the completed evidence above satisfies the lifecycle gate.
