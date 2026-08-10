# cognition core v2 generation-contract prompt projection bugfix plan

## Summary

- Goal: reduce recurring parseable Cognition Core V2 contract failures by making the local-model-facing semantic appraisal and goal prompts natural, ordered, and internally consistent while preserving deterministic semantic ownership.
- Status: completed
- Plan class: high-risk prompt and contract-boundary bugfix
- Scope boundary: semantic appraisal and goal cognition prompt contracts, their model-facing payload ordering and repair instructions, focused deterministic tests, controlled replay evidence, and targeted live-LLM verification.
- Change direction: prioritize the GPT-5.6-sol assessment. Treat the primary failure as a generation-contract interaction; address contract communicability first, prompt design second, and prompt-owner projection clarity third. Keep the validator strict and unchanged.
- Acceptance state: sign-off requires a frozen affected-test manifest with an expected-outcome pass ratio of at least 95%, all critical contract and regression gates passing, and a parent-authored evidence review.

## Confirmed Decisions

- The GPT-5.6-sol independent assessment is the primary prioritization input.
- The DeepSeek assessment remains supporting evidence.
- The user authorizes rewriting the related prompts into natural, logical, organic language for a weaker local LLM.
- The later implementation handoff used the DeepSeek subagent under its bounded protocol. The handoff did not complete, so the parent implemented the approved scope and will execute and inspect the tests.
- The user explicitly authorized execution after the drafting phase. This document is the completed execution record for the approved scope.
- The final native read-only plan review returned GO with no concrete blocking findings (reviewer `019fcd51-5007-7fc1-9976-2138966e8273`).
- The runtime output schema, enum values, handle domains, path ownership, deterministic validation, and state-transition authority remain canonical unless a separately approved amendment changes them.

## Evidence And Prioritization

The evidence packet contains 375 failed nested attempts classified as contract_error after parsing, with recurring failures around evidence handles, candidate-origin citations, semantic paths, terminal transitions, and exact fields. Historical and current semantic prompts differ, and some historical inputs do not satisfy the current memory_scope validator. The deterministic failure matrix has 22 passing tests, while 5 of 12 current live cases reproduced their named family.

GPT-5.6-sol ranked the generation-contract interaction at high confidence, contract complexity and communicability next, prompt effects next, projection effects as a lower-confidence contributor, validator defect as low confidence, and retry behavior as an amplifier rather than the origin. This plan follows that order.

## Scope And Change Direction

The implementation will make the existing semantic contracts easier for a weaker model to execute without weakening them.

The semantic appraisal prompt will use one positive procedure: understand the single question, inspect only the supplied evidence, choose at most one proposition and one delta, map every structured handle to its permitted domain, attach required origin evidence, and perform a final JSON check.

The goal prompts will use one positive procedure: understand the current event and authoritative role direction, decide one complete character-owned goal, cite only permitted evidence, fill every required field, apply the relational-willingness pairing, and perform a final JSON check.

The prompt language will remain Simplified Chinese because that is the current model-facing language. The rewrite will improve semantic flow and reduce negative-constraint accumulation; it will not add character voice, implementation history, plan language, hidden stage terminology, or new semantic fields.

Model-facing payload assembly inside the two prompt-owner modules will place the existing evidence rows, handle-domain declarations, candidate-origin map, permitted path domains, required fields, and repair feedback in a stable order. No alias namespace, compatibility field, fallback mapper, or new LLM stage will be introduced.

## Cutover Policy

Overall strategy: bigbang for the runtime prompt behavior and prompt-owner payload presentation. The old runtime prompts are replaced in place after the verification gate. Captured historical prompts remain available only inside the read-only replay harness and never run in production.

There will be no runtime feature flag, dual prompt path, compatibility shim, alternate vocabulary, or semantic validator relaxation.

## Mandatory Skills

Execution must load and apply the following before touching the governed surface:

- development-plan, including references/plan_contract.md, references/execution_gates.md, and references/cutover_policy.md;
- local-llm-architecture before editing any prompt, payload projection, model call, or retry context;
- debug-llm before running live LLM cases or authoring the human-readable review;
- py-style before editing Python;
- cjk-safety before editing Python prompt text containing Chinese;
- test-style-and-execution before adding, changing, or running tests;
- python-venv before running Python or pytest;
- no-prepost-user-input before changing prompts that shape semantic interpretation of user-directed content.

## Mandatory Rules

- Use venv\Scripts\python.exe for every Python and pytest invocation.
- Read the plan, repository guidance, and governed subsystem documentation again at the start of execution and after any context compaction.
- Apply the canonical kazusa_ai_chatbot.utils.parse_llm_json_output entry point before contract evaluation. Do not add a parser or repair model.
- Keep semantic judgment in the LLM and validation, handle authority, path ownership, retry counting, persistence, and disposition in deterministic code.
- Keep every prompt constant in a triple-single-quoted string. Keep the system prompt static and place per-run facts in the human message.
- Do not introduce plan names, migration language, stage implementation names, compatibility notes, or test-shaped fixture instructions into runtime prompts.
- Do not add aliases, compatibility layers, default semantic values, keyword routing, verifier LLM calls, or extra healthy-path model calls.
- Do not change contracts.py or transition_guards.py as part of this plan. A demonstrated false rejection requires a plan amendment.
- Run deterministic tests in batches. Run live LLM cases one at a time and inspect each artifact before starting the next case.
- Preserve unrelated worktree changes and never read .env.

## Must Do

### 1. Freeze baseline and evidence

- Record git status, current prompt hashes and lengths, current deterministic results, current live-case dispositions, and the two known pre-existing prompt-length assertion failures from the completed robustness plan.
- Create a frozen sign-off manifest before the first post-change test. The manifest records each selected test case, its expected outcome, category, source file, and whether it is deterministic, a negative contract control, or a live model case.
- Perform a read-only lexical audit of the preserved model-facing messages for ce-number, ct-number, ev-number, r-number.exclusivity, relationship.r-number, and active_events.ev-number patterns. Store raw counts as structured evidence and author the readable interpretation manually.

### 2. Run the separating replay

Use preserved capsules and a fixed source snapshot for a test-only 2×2×2 replay:

- captured historical prompt wording versus current prompt wording;
- captured historical model-facing projection versus current prompt-owner projection;
- captured repair feedback versus the current compact contract-feedback presentation.

Hold model build, sampling settings, source state, attempt cap, and strict validator semantics constant. Run without retries first, then apply one identical repair turn to failed outputs. Record the exact rejected rule, output family, and per-attempt trajectory. This diagnostic phase does not authorize a production contract change.

### 3. Rewrite semantic appraisal prompts

Modify semantic_appraisal.py only within the existing semantic appraisal prompt and repair-message ownership:

- rewrite SEMANTIC_APPRAISAL_PROMPT into short ordered sections for purpose, evidence authority, handle legend, decision procedure, output shape, and final check;
- make the evidence-handle, role-handle, entity-handle, candidate-origin, and delta-path boundaries explicit in one compact model-facing explanation;
- explain candidate-origin citation as a positive mapping step: every structured candidate reference brings its mapped evidence handle into that same object;
- explain delta paths as exact selections from the supplied permitted path table, without inviting the model to construct foreign paths;
- preserve one micro-appraisal item, proposition-or-null, delta-or-null, exact top-level keys, all existing enum values, numeric bounds, text bounds, and Simplified Chinese free-text rules;
- rewrite the appraisal repair message so it identifies the one failed rule, the allowed values, and the required complete object in a natural sequence without restating unrelated constraints.

### 4. Rewrite goal cognition prompts

Modify goal_cognition.py only within the existing goal prompt and repair-message ownership:

- rewrite GOAL_COGNITION_PROMPT, REQUIRED_SELECTION_GOAL_PROMPT, GENERIC_GOAL_REPAIR_INSTRUCTIONS, and SELECTION_GOAL_REPAIR_INSTRUCTIONS into a positive, stepwise decision procedure;
- keep current event role direction, authoritative evidence, character identity, boundary, relationship, and current-episode authority explicit but concise;
- state the separation between role handles and evidence handles once, close to the output fields that use them;
- state the relational-willingness pairing as a small ordered rule table or equivalently clear prose, including current-episode evidence coverage;
- require a complete output object on repair and treat invalid_draft as data rather than instructions;
- preserve all existing top-level keys, relationship schema, enum values, evidence limits, target-role limits, text limits, and non-routing ownership;
- remove repeated or contradictory negative lists while retaining every rule needed by deterministic validation.

### 5. Add focused deterministic coverage

Parent-owned tests will cover the changed prompt-owner boundary without changing production authority:

- prompt constants use the required static shape and render without unresolved placeholders;
- model-facing payloads expose evidence and role domains without cross-namespace aliases;
- candidate-origin mappings are adjacent to the permitted candidate handles and remain complete;
- delta paths are selected from the permitted path domain rather than invented from unrelated state paths;
- goal repair feedback contains the exact allowed evidence, role, required-field, current-episode, and validation-error values;
- output keys, enum values, bounds, and strict contract behavior remain unchanged;
- the existing 20-family failure matrix remains represented as deterministic negative controls.

New deterministic coverage belongs in tests/test_cognition_core_v2_prompt_contract_guidance.py. Existing contract, projection, retry-continuity, semantic-terminalization, transition-coherence, prompt-budget, and trace-failure matrix tests remain in the sign-off suite.

### 6. Run targeted live verification

The parent will run and inspect, one case at a time:

- tests/test_cognition_core_v2_trace_failure_modes_live_llm.py;
- tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py;
- tests/test_cognition_core_v2_goal_capability_live_llm.py;
- tests/test_cognition_core_v2_required_selection_live_llm.py;
- tests/test_cognition_core_v2_relational_willingness_live_llm.py.

Each live case must retain its raw model output, parsed output, validation result, failure family, and human-readable quality note. A case is passed only when its actual outcome matches the frozen expected outcome; an expected negative control passes when it produces the declared deterministic rejection.

### 7. Produce review and sign off

The parent will author a Markdown review from the real replay and live outputs. The review will include run context, input and output summaries, decision behavior, quality notes, validation results, raw evidence paths, failed-case analysis, and residual risk.

The parent will review the DeepSeek diff against this plan, verify the change surface, run the complete frozen sign-off suite, and record the lifecycle decision. DeepSeek does not sign off its own implementation.

## Deferred

- Changes to contracts.py, transition_guards.py, state_models.py, or deterministic validator permissiveness.
- Changes to retry counts, retry ownership, JSON repair, evaluator stages, or failure dispositions.
- A new projection namespace, compatibility alias, fallback mapper, verifier model, or healthy-path model call.
- Structural redesign of state_projection.py. The plan only changes prompt-owner payload presentation in semantic_appraisal.py and goal_cognition.py.
- Workspace collapse, action planning, authorization, surface, dialog, persistence, scheduler, adapter, database, and reflection prompts.
- Historical database migration, production trace rewriting, and broad prompt changes outside the two named prompt owners.
- Reclassification of an expected failure as a pass without matching the frozen expected outcome.
- Unrelated cleanup or formatting changes.

## Target State

The local model receives a short, coherent explanation of the semantic task before the JSON contract. Each structured field has one visible authority and one handle domain. Candidate-origin evidence and allowed path construction are explained where the model uses them. Repair feedback asks for a complete regeneration against explicit allowed values. The model remains responsible for semantic judgment; deterministic code remains responsible for validation and state safety.

The runtime output schemas and deterministic state transitions remain unchanged. The old prompts are not retained as a production fallback.

## Contracts And Data Shapes

Semantic appraisal remains a single-item result with exactly question_id, proposition, and delta. Proposition and delta remain one object or null. Evidence handles remain question.evidence_handles. Role/entity handles remain question.permitted_role_handles. Candidate-origin evidence remains the supplied mapping. Delta paths remain exact members of question.permitted_delta_path_domains.

Goal cognition remains a complete candidate with its existing fields, exact evidence and target-role arrays, expected consequences, confidence, and relational_willingness contract where required. No field is renamed, added, defaulted, or semantically rewritten.

## Change Surface

### Delete

- None.

### Modify

- src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py: rewrite the semantic appraisal system prompt, repair instructions, and local payload ordering while preserving the existing output and validation contract.
- src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py: rewrite generic and required-selection goal prompts, repair instructions, and local repair payload ordering while preserving the existing goal contract.
- tests/test_cognition_core_v2_prompt_contract_guidance.py: add parent-owned deterministic prompt/projection guidance coverage.
- tests/test_cognition_prompt_contract_text.py: include the existing prompt-contract text suite in the frozen manifest and final deterministic run.
- tests/test_cognition_core_v2_trace_failure_modes_live_llm.py: pass the historical canonical repair domains to the internal semantic-owner test call after the owner contract gained explicit repair projection input.
- tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py: classify valid current-prompt results as passing outcomes instead of treating non-exhaustion as an expected failure.
- tests/test_cognition_core_v2_required_selection_live_llm.py: normalize missing archived fixture references to the available canonical reproduction fixtures and valid evidence-source vocabulary so the live contract cases reach the model boundary.
- development_plans/README.md: register this plan under active bugfix plans.

### Create

- development_plans/archive/completed/bugfix/cognition_core_v2_generation_contract_prompt_projection_bugfix_plan.md: completed execution record for this scope.
- test_artifacts/diagnostics/cognition_core_v2_prompt_contract_projection_signoff_manifest_2026-08-05.json: frozen raw sign-off case inventory and expected outcomes.
- test_artifacts/diagnostics/cognition_core_v2_prompt_contract_projection_review_2026-08-05.md: parent-authored human-readable review.
- test_artifacts/diagnostics/cognition_core_v2_prompt_contract_projection_replay_2026-08-05.json: raw 2×2×2 replay and lexical-audit evidence.

### Plan Amendment

The frozen manifest was amended before final classification to include the existing prompt-contract text suite (16 cases) and the seven parent-owned prompt-guidance cases required by the acceptance contract. The amended denominator is 239 cases: 169 original deterministic cases, 16 prompt-contract text cases, 7 prompt-guidance cases, and 47 live model cases. After the first taboo live run exposed a current-episode boundary-priority gap, the goal prompt gained an explicit boundary-precedence sentence and that case was rerun; the manifest records the iteration and final classification.

### Keep

- src/kazusa_ai_chatbot/cognition_core_v2/contracts.py: strict semantic contract and deterministic validation authority.
- src/kazusa_ai_chatbot/cognition_core_v2/transition_guards.py: state-transition ownership and terminal-state enforcement.
- src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py: existing canonical state projection and private handle bindings.
- src/kazusa_ai_chatbot/cognition_core_v2/workspace.py, action planning, authorization, surface, dialog, persistence, scheduler, adapters, and database code.
- Existing failure-matrix and live-LLM tests as regression evidence; do not delete, skip, or weaken cases.

## Agent Autonomy Boundaries

DeepSeek execution ownership is limited to:

- src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py;
- src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py.

DeepSeek may choose local wording, section order, and helper-local payload ordering within those files. It may not change output schemas, enum values, validation rules, state reducers, transition guards, retry counts, model routes, or any file outside the owned set.

The parent owns the baseline, replay evidence, test-file changes, test execution, raw-artifact inspection, review artifact, and sign-off. The parent may amend the plan before execution when evidence requires a contract or ownership decision; DeepSeek may not silently expand scope.

The later DeepSeek handoff must use the required acknowledgement-only first turn and a separate real execution turn. Before handoff, the parent records git status and the exact owned file set. The parent uses a hard wait deadline and interrupts the agent on expiry.

## Verification

Verification must include:

- pre-change baseline and frozen sign-off manifest;
- read-only lexical audit and 2×2×2 replay evidence;
- Python compile checks for edited Python files containing Chinese prompt text;
- focused deterministic prompt, contract, projection, retry, terminalization, transition, budget, and failure-matrix tests;
- targeted live cases run individually with raw output inspection;
- broader non-live cognition regression coverage proportional to the final diff;
- parent review of prompt naturalness, logical ordering, namespace clarity, and preservation of character-brain ownership;
- diff/scope review confirming no forbidden compatibility or validator changes.

## Acceptance Criteria

### Frozen test ratio

Before the first post-change test, freeze the sign-off manifest and its denominator. The denominator is the individual test-case count across the named deterministic suites, the new prompt-guidance suite, and the named live cases. The manifest cannot be reduced, skipped, xfailed, or reclassified after seeing post-change results without a plan amendment.

A case passes only when its actual result matches its manifest expectation. That includes valid output for positive cases and the exact declared contract rejection for negative controls. The sign-off pass ratio is:

passed expected-outcome cases / executed manifest cases

The ratio must be at least 95% using the unrounded value.

### Critical gates

- 100% of deterministic contract, projection, prompt-guidance, and targeted regression cases pass.
- All existing failure families remain covered by negative controls, and no new failure family appears.
- Targeted positive live cases do not reproduce the named contract failure after the prompt cutover, except where the frozen manifest explicitly defines a negative control.
- No production change occurs outside the two owned prompt-owner files.
- No output field, enum, handle domain, path ownership rule, state-transition rule, retry cap, parser, or route changes.
- Every live result has an inspected raw artifact and parent-authored quality note.
- The parent-authored Markdown review records the final ratio, all failures, causal interpretation, residual risk, and exact evidence paths.
- The independent code review and plan checklist are complete before the plan can move from in_progress to completed.

## Progress Checklist

- [x] Baseline, prompt hashes, current test results, and amended frozen sign-off manifest recorded.
- [x] Lexical audit and 2×2×2 diagnostic replay recorded.
- [x] DeepSeek execution handoff attempted under the bounded protocol; parent fallback implementation completed within the owned prompt-owner files.
- [x] Prompt and payload changes reviewed for naturalness, ordering, and contract fidelity.
- [x] Focused deterministic tests pass.
- [x] Targeted live cases run individually and inspected.
- [x] Aggregate sign-off ratio is at least 95%; critical gates pass.
- [x] Parent review, residual-risk record, independent GO review, and lifecycle closeout completed.

## Post-Closeout Real-Run Regression Evidence

This evidence note records a production-shaped real-LLM failure discovered
after closeout. It does not change the completed plan's scope, frozen test
denominator, acceptance decision, or implementation contract.

- Case: `relationship_social_attachment_range_after_prompt_fix`.
- Run: delivery tracking id `f48a049ff4c4421d8b73ed8ac96ffed1`, trace id
  `llmtrace_9f325fea52674dcf913ccf6f9ca755de`.
- Input: `不行不行，不能天天放任性欲支配大脑`.
- Observed failure: `semantic_appraisal.q:relationship_social` exhausted its
  bounded attempts because `attachment is outside its range`.
- Companion observations: `q:goal_threat_outcome` also hit a terminal-event
  transition contract failure, and the ordinary-response goal required one
  repair attempt. The run completed with six visible dialog messages and a
  partial-failure capsule.
- Classification: regression case requiring live coverage and human quality
  review; this run is not a passing positive case.
- Raw evidence: `test_artifacts/diagnostics/db_llm_trace_delivery_f48a049ff4c4421d8b73ed8ac96ffed1.json`;
  readable review:
  `test_artifacts/diagnostics/llm_debug_review_delivery_f48a049ff4c4421d8b73ed8ac96ffed1.md`.
