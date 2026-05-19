# generic cognition prompt migration plan

## Summary

- Goal: migrate the cognition chain to the corrected generic prompt family
  proven by the 2026-05-19 side-by-side real LLM comparison.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, and `cjk-safety`
- Overall cutover strategy: bigbang for cognition prompt text; compatible for
  existing cognition source selection, schemas, action-spec materialization,
  delivery, persistence, and source-packet projection
- Highest-risk areas: prompt logic-flow regression, source-mode leakage,
  metadata copying into generated fields, over-defensive boundary escalation,
  and vague L2d `detail` generation
- Acceptance criteria: production prompts preserve external-input behavior,
  improve self-cognition source framing, keep outputs in Chinese, avoid
  source-packet metadata copying, and pass deterministic plus real LLM gates

## Context

The supporting evidence is recorded in
`development_plans/reference/evidence/cognition_prompt_chain_side_by_side_comparison_20260519.md`.

Accepted conclusion: the corrected generic prompt family is the migration
target. It produces more natural character judgment than the current baseline
for self-cognition group windows and does not show external-input regression in
the real LLM controls.

## Mandatory Skills

- `development-plan-writing`: load before changing this plan, the registry,
  execution evidence, lifecycle status, or approval status.
- `local-llm-architecture`: load before changing cognition prompts, graph
  stage boundaries, prompt-facing state, or LLM context contracts.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files containing CJK prompt strings.

## Mandatory Rules

- Execute this plan only after status is changed to `approved` or
  `in_progress`.
- This project is a character brain, not a generic assistant chatbot. Do not
  add mechanical suppression, cooldowns, response-ratio clamps, keyword
  blockers, deterministic offense filters, or chatbot safety phrasing.
- LLM stages own semantic character judgment. Deterministic code owns schema,
  validation, prompt-safe projection, routing, persistence, and delivery.
- Any LLM prompt changed by this plan must be rewritten as a complete coherent
  prompt. Do not append warnings, patches, or isolated special-case blocks.
- Rewritten prompts must preserve this flow: role boundary, language policy,
  source-mode identification, current evidence interpretation, stage-specific
  task, stage-specific decision rules, output contract.
- Rewritten prompts must preserve existing output keys, schema shapes, enum
  values, JSON-only output requirements, and downstream action-spec contracts.
- Rewritten prompts must keep all newly generated free-text fields in
  Simplified Chinese except schema keys, enum values, IDs, URLs, code,
  commands, model labels, and source text that must remain in its original
  language.
- Self-cognition material must be framed as character-owned observation data,
  not a live user message. Prompts must not call it `用户输入`, `用户提供`, or a
  current group member utterance.
- Prompts must not copy source-packet headings, JSON, timestamps,
  semantic-label keys, transport summaries, or model-facing metadata into
  `internal_monologue`, `judgment_note`, social-context fields, action
  `detail`, or action `reason`.
- L2d `detail` must describe the concrete visible action target or private
  action target in the current scene. It must not generate final dialog text,
  quote source metadata, or ask to clarify a self-cognition transport summary.
- No new prompt variant is allowed. The existing prompt-selection contract
  remains unchanged.
- Source-packet projection is out of scope. Do not change private-chat source
  labels or group-review source-packet wording in this plan.
- Group engagement guidance remains evidence only. It can influence character
  judgment together with the observed scene; it must not command speech.
- Playful or noisy group mentions of the character are not automatic boundary
  attacks. Boundary escalation requires tone plus context that genuinely
  attacks identity, autonomy, dignity, or safety.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Real LLM tests must run one case at a time and be inspected one case at a
  time.
- Temporary experiment code created during execution must be deleted or
  converted into tests before final sign-off.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.

## Must Do

- Rewrite the full L1, L2a, L2b, L2c1, L2c2, and L2d system prompts using the
  corrected generic prompt direction.
- Preserve each stage's current callable, state fields, LLM route, output
  contract, and validator.
- Add positive source-mode constraints to each rewritten prompt:
  self-cognition data is character-owned observation data; external user input
  is current speech from the user.
- Add positive metadata-copy constraints to each rewritten prompt:
  generated free text summarizes the scene and decision, not the model-facing
  packet structure.
- Add positive L2d `detail` constraints:
  `detail` names the current visible reply target or private action target,
  not the final wording and not packet metadata.
- Keep L2d as the only owner of semantic action selection.
- Update prompt fingerprint expectations after rewriting prompts.
- Add deterministic prompt contract tests for source-mode wording,
  metadata-copy constraints, L2d `detail` constraints, enum preservation, and
  language policy preservation.
- Run real LLM verification for the self-cognition group sensitivity set and
  external-input controls after deterministic tests pass.

## Deferred

- Do not add a new prompt family for self-cognition.
- Do not change `persona_supervisor2_cognition_prompt_selection.py`.
- Do not change cognition state shape, `CognitiveEpisode`, action-spec models,
  action-spec materialization, dispatcher, adapters, scheduler, consolidation,
  persistence, or database schemas.
- Do not tune response-ratio parameters, group-noise thresholds, queue
  behavior, relevance gating, or source collection cadence.
- Do not migrate historical database rows.
- Do not change L3 text, dialog generation, visual directives, or dialog
  evaluator prompts.
- Do not edit `experiments/` as part of production migration.

## Cutover Policy

Overall strategy: bigbang for prompt text; compatible for existing runtime
contracts.

| Area                  | Policy     | Instruction                                                                                             |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| L1 prompt             | bigbang    | Rewrite `_COGNITION_SUBCONSCIOUS_PROMPT` as one coherent generic prompt.                                |
| L2a prompt            | bigbang    | Rewrite `_COGNITION_CONSCIOUSNESS_PROMPT` as one coherent generic prompt.                               |
| L2b prompt            | bigbang    | Rewrite `_BOUNDARY_CORE_PROMPT` as one coherent generic prompt.                                         |
| L2c1 prompt           | bigbang    | Rewrite `_JUDGEMENT_CORE_PROMPT` as one coherent generic prompt.                                        |
| L2c2 prompt           | bigbang    | Rewrite `_CONTEXTUAL_AGENT_PROMPT` as one coherent generic prompt.                                      |
| L2d prompt            | bigbang    | Rewrite `_ACTION_INITIALIZER_PROMPT` as one coherent generic prompt with positive `detail` constraints. |
| Prompt selection      | compatible | Preserve existing variant names and route selection.                                                    |
| Output contracts      | compatible | Preserve all existing JSON keys, enums, and validators.                                                 |
| Tests                 | bigbang    | Replace prompt fingerprints and add deterministic prompt contract checks.                               |
| Real LLM verification | compatible | Use existing live LLM routes and trace conventions.                                                     |

## Cutover Policy Enforcement

- Bigbang prompt areas must replace the old prompt text directly.
- Compatible areas must retain only the exact contracts listed above.
- Any change outside the listed change surface requires owner approval before
  editing.

## Agent Autonomy Boundaries

- The agent may choose local test helper names only.
- The agent must not create new prompt routes, source variants, fallback
  prompts, repair prompts, retry loops, or post-generation filters.
- The agent must not weaken language policy, enum policy, JSON-only output
  policy, or source-mode separation.
- The agent must not turn prompt constraints into silence rules. The desired
  behavior is grounded character judgment, not lower response rate.
- If live LLM verification shows a bad speak decision, update the relevant
  prompt logic flow inside the approved prompt surface. Do not add
  deterministic speak/silent filters.

## Target State

The cognition chain uses one source-aware generic prompt family. Each stage
first identifies whether the stimulus is external user speech or internal
self-cognition observation data, then performs the stage-specific cognition
task without confusing transport summaries for visible chat content.

L2d produces `speak` only when the current scene gives the character a grounded
reason to externalize a visible reply. It produces no action when the only
reason is private curiosity, observation, source-packet confusion, or waiting
for a better moment.

## Design Decisions

- Use one corrected generic prompt family for L1, L2a, L2b, L2c1, L2c2, and
  L2d.
- Do not create split prompts for self-cognition.
- Do not change prompt selection or cognition graph topology.
- Do not change source-packet projection in this plan.
- Keep group engagement guidance in the L2d action context as evidence.
- Treat `self_report_13` as a baseline overreaction, not a generic regression.
- Treat `self_report_09` as acceptable action shape with timezone-derived
  reason drift outside this prompt migration.

## LLM Call And Context Budget

- Number of LLM calls per live cognition path remains unchanged.
- Prompt-selection variants remain unchanged.
- No new RAG call, source loader, evaluator call, repair call, retry call, or
  follow-up LLM call is allowed.
- Rewritten prompts must be shorter than or equal to the current production
  prompt constant they replace.

## Change Surface

Modify only these production files:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2c2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`

Modify only these test files, or add the listed new test file:

- `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py`
- `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
- `tests/test_cognition_prompt_contract_text.py`

Modify only these documentation or plan files:

- `development_plans/active/bugfix/generic_cognition_prompt_migration_plan.md`
- `development_plans/reference/evidence/cognition_prompt_chain_side_by_side_comparison_20260519.md`
- `development_plans/README.md`

Forbidden files and directories:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
- `src/kazusa_ai_chatbot/self_cognition/projection.py`
- `src/kazusa_ai_chatbot/action_spec/**`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/reflection_cycle/**`
- `src/kazusa_ai_chatbot/db/**`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `experiments/**`

## Overdesign Guardrail

Actual problem: shared cognition prompts still bias internal self-cognition
observation data toward external user-message interpretation, and L2d `detail`
can become too creative or metadata-shaped.

Minimal change: rewrite the existing six cognition/action prompts as coherent
source-aware generic prompts and update prompt contract tests.

Ownership boundaries:

- L1 owns first affect only.
- L2a owns conscious scene interpretation and candidate stance/intent.
- L2b owns boundary appraisal.
- L2c1 owns final stance/intent synthesis.
- L2c2 owns social context appraisal.
- L2d owns semantic action selection.
- Deterministic code owns schemas, validation, materialization, persistence,
  and delivery.

Rejected complexity:

- no split prompt family
- no new source variants
- no deterministic speak/silent filter
- no response-ratio tuning
- no target inference
- no source-packet migration
- no additional LLM calls
- no post-processing repair for semantic decisions

Evidence required before expanding scope: a post-migration real LLM trace must
show repeated failure caused by a named forbidden area that cannot be fixed by
rewriting the approved prompt surface.

## Implementation Order

1. Reread this plan and record `git status --short`.
2. Add deterministic prompt contract tests in
   `tests/test_cognition_prompt_contract_text.py`.
3. Run the new deterministic prompt contract tests and record the expected
   pre-implementation failure.
4. Rewrite `_COGNITION_SUBCONSCIOUS_PROMPT` in
   `persona_supervisor2_cognition_l1.py`.
5. Rewrite `_COGNITION_CONSCIOUSNESS_PROMPT`, `_BOUNDARY_CORE_PROMPT`, and
   `_JUDGEMENT_CORE_PROMPT` in `persona_supervisor2_cognition_l2.py`.
6. Rewrite `_CONTEXTUAL_AGENT_PROMPT` in
   `persona_supervisor2_cognition_l2c2.py`.
7. Rewrite `_ACTION_INITIALIZER_PROMPT` in
   `persona_supervisor2_cognition_l2d.py`.
8. Update prompt fingerprints in
   `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py` and
   `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`.
9. Run deterministic tests.
10. Run real LLM external-input controls one test at a time.
11. Run self-cognition group sensitivity real LLM verification one case at a
    time and inspect full input/output traces.
12. Run the independent code review gate.
13. Record execution evidence and final status.

## Progress Checklist

- [x] Stage 1 - deterministic prompt contract tests added
  - Covers: steps 1-3.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py -q`
  - Evidence: recorded in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 2 - production prompts rewritten
  - Covers: steps 4-8.
  - Verify:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2c2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py`
  - Evidence: recorded in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 3 - deterministic tests pass
  - Covers: step 10.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py tests\test_multi_source_cognition_stage_07_reflection_dry_run.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_text_chat_and_reflection_prompt_fingerprints_remain_stable -q`
  - Evidence: recorded in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 4 - real LLM external controls pass
  - Covers: step 11.
  - Verify one case at a time:
    `venv\Scripts\python -m pytest tests\test_cognition_live_llm_prompt_contracts.py::test_live_cognition_stack_photo_request_chinese -q -s`
    `venv\Scripts\python -m pytest tests\test_cognition_live_llm_prompt_contracts.py::test_live_cognition_stack_boundary_command_repeated_fillers -q -s`
  - Evidence: recorded in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 5 - real LLM self-cognition sensitivity passes
  - Covers: step 12.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_group_response_sensitivity -q -s`
  - Evidence: recorded in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-19`.
- [x] Stage 6 - independent code review complete
  - Covers: steps 13-14.
  - Verify: review full diff against this plan, style, prompt flow, tests, and
    trace evidence.
  - Evidence: recorded in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-19`.

## Verification

Deterministic verification:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2c2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py
venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py -q
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py -q
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_text_chat_and_reflection_prompt_fingerprints_remain_stable -q
```

Real LLM verification:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_live_llm_prompt_contracts.py::test_live_cognition_stack_photo_request_chinese -q -s
venv\Scripts\python -m pytest tests\test_cognition_live_llm_prompt_contracts.py::test_live_cognition_stack_boundary_command_repeated_fillers -q -s
venv\Scripts\python -m pytest tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_group_response_sensitivity -q -s
```

Manual real LLM judgment must inspect trace files for:

- no English drift in generated free-text fields
- no self-cognition material described as a live user message
- no source-packet heading, JSON, timestamp, semantic-label key, or transport
  summary copied into generated fields
- `speak` reason grounded in visible chat content and character judgment
- `detail` names the scene action target rather than final wording or packet
  metadata
- external direct questions still speak
- external identity-boundary pressure still rejects

## Independent Plan Review

Review status: completed during drafting.

Review conclusions:

- Change surface is limited to the six production prompt constants, prompt
  contract tests, prompt fingerprints, and this plan registry.
- Prompt-selection, source-packet projection, action-spec materialization,
  service delivery, database, scheduler, L3, dialog, and experiments are
  explicitly forbidden.
- The plan contains no technical decision left for the implementation agent.
- The plan requires full prompt rewrites for any changed prompt and forbids
  appended special-case patches.
- The plan uses positive constraints instead of mechanical suppression.

## Independent Code Review

Before completion, an independent review must check:

- every changed prompt preserves the required logic flow
- all output keys, enums, JSON contracts, and validators remain compatible
- prompt text does not create a second hidden prompt family
- self-cognition source framing is not described as a user message
- L2d `detail` constraints are positive and scene-grounded
- tests cover prompt text contracts and updated fingerprints
- real LLM evidence was run one case at a time and inspected
- no forbidden files were edited

Findings must be fixed only inside the approved change surface. Any finding
requiring a broader change must stop implementation and request owner approval.

## Acceptance Criteria

- All six approved prompt constants are rewritten as complete coherent prompts.
- No new prompt variant, graph stage, LLM call, fallback path, or deterministic
  speak/silent filter is introduced.
- Deterministic verification passes.
- Real LLM external controls preserve appropriate visible action decisions.
- Real LLM self-cognition verification shows improved source framing and no
  repeated metadata-copy failure.
- `self_report_13`-style noisy group banter is not escalated into automatic
  boundary confrontation.
- `self_report_09`-style group commentary remains allowed when the character
  has a grounded reason to speak.
- Execution evidence records commands, trace paths, manual judgment, review
  findings, and residual risks.

## Execution Evidence

Implementation evidence as of 2026-05-19:

- Production prompt constants rewritten:
  `_COGNITION_SUBCONSCIOUS_PROMPT`,
  `_COGNITION_CONSCIOUSNESS_PROMPT`, `_BOUNDARY_CORE_PROMPT`,
  `_JUDGEMENT_CORE_PROMPT`, `_CONTEXTUAL_AGENT_PROMPT`, and
  `_ACTION_INITIALIZER_PROMPT`.
- The rewritten prompts preserve the existing prompt-selection contract and
  add explicit source handling for external speech, internal observation data,
  and reflection artifacts.
- Deterministic prompt contract tests were added in
  `tests/test_cognition_prompt_contract_text.py`.
- Prompt fingerprints were updated in
  `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py` and
  `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`.
- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2c2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py tests\test_cognition_prompt_contract_text.py tests\test_multi_source_cognition_stage_07_reflection_dry_run.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
  passed.
- `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py tests\test_multi_source_cognition_stage_07_reflection_dry_run.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_text_chat_and_reflection_prompt_fingerprints_remain_stable -q`
  passed: 23 tests passed.
- Real LLM external control
  `tests\test_cognition_live_llm_prompt_contracts.py::test_live_cognition_stack_photo_request_chinese`
  passed after the reflection-source fix; manual inspection showed
  `CONFIRM` / `PROVIDE`, Chinese generated free text, and no metadata-copy
  failure.
- Real LLM external control
  `tests\test_cognition_live_llm_prompt_contracts.py::test_live_cognition_stack_boundary_command_repeated_fillers`
  passed after the reflection-source fix; manual inspection showed explicit
  boundary refusal/rejection and no accepted coercive address preference.
- Real LLM self-cognition sensitivity batch
  `tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_group_response_sensitivity`
  passed with `SELF_COGNITION_SENSITIVITY_RUN_BATCH=1`.
- Latest self-cognition dataset trace:
  `test_artifacts/llm_traces/self_cognition_group_response_sensitivity_dataset__collected_group_windows__20260519T095623988287Z.json`.
- Latest self-cognition manual inspection: 20 raw group windows, 9 historical
  spoke labels, 0 current visible speak actions, 11 historical-label matches,
  and 0 generated-field hits for known source-packet metadata terms.
- Manual judgment: the self-cognition rerun showed no incorrect visible speak
  from self-cognition packet confusion. Silent mismatches were mainly windows
  where the replayed raw window already contained the character response or
  where the current topic had no grounded role for the character.
- Residual risk: the current prompt family is conservative on historical
  group-review replay. Further response-ratio tuning should use the existing
  live sensitivity test rather than adding deterministic speak/silent filters.
- Independent code review pass 1 found missing reflection-artifact source
  handling, missing execution evidence, and an incomplete change-surface list.
  The prompt source handling, deterministic reflection contract coverage,
  prompt fingerprints, change surface, and execution evidence were updated.
- Independent code review pass 2 found no issues. The reviewer confirmed
  reflection source handling, execution evidence, change-surface scope,
  source-mode separation, metadata-copy constraints, L2d `detail` constraints,
  enum/schema compatibility, and absence of new prompt variants or
  deterministic speak/silent filters.
- Final deterministic review commands passed:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2c2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py tests\test_cognition_prompt_contract_text.py tests\test_multi_source_cognition_stage_07_reflection_dry_run.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
  and
  `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py tests\test_multi_source_cognition_stage_07_reflection_dry_run.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_text_chat_and_reflection_prompt_fingerprints_remain_stable -q`.
- 2026-05-19: owner accepted the current quality, requested closure, and
  directed registry cleanup. The plan is closed as a completed historical
  record.
