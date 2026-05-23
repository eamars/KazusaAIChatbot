# rag2 cognition-ready evidence plan

## Summary

- Goal: Update production RAG2 so memory, recall, and conversation evidence
  reaches cognition as source-backed, formatted, prompt-safe facts instead of
  uneven free-form summaries or raw transcript fragments.
- Plan class: large
- Status: approved
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: bigbang for prompt-facing RAG evidence formatting;
  compatible for the existing `rag_result` top-level keys and supervisor trace;
  no database, index, collection, cache, routing, or graph migration.
- Highest-risk areas: breaking the RAG-to-cognition evidence contract,
  exposing adapter-wire CQ syntax, leaking raw UTC or microsecond timestamps
  into prompts, duplicating prompt-safety helpers, importing experiment code,
  adding response-path LLM cost, hiding useful evidence in trace-only fields,
  and overfitting production output to the experiment harness.
- Acceptance criteria: production RAG2 emits cognition-ready formatted
  evidence over the same core memory/recall/conversation sources, preserves
  existing downstream `rag_result` consumers, passes focused deterministic
  checks, passes a readable `debug-llm` agent review on the experiment case
  set, contains no imports from `experiments/`, and removes the experiment code
  after production verification.

## Context

The completed recall-quality experiment compared current RAG2 against an
experiment-only improved path over real database cases. The decisive run is:

```text
test_artifacts/rag2_recall_quality/contextual_alt10_improved_v2_blind_review.md
```

Mapped result:

```text
current RAG2 baseline: 0 cases preferred
formatted improved search V2: 10 cases preferred
tie: 0
unclear: 0
```

The user also reviewed the formatted outputs directly and preferred the
structured improved output shape over current RAG's free-form and sometimes
Markdown-like synthesis.

This plan adopts the experiment conclusion into production RAG2. It does not
adopt DAG, graph RAG, conversation graph, or a new retrieval architecture. The
production fix is to improve the existing RAG2 helper-agent and projection
surface so cognition receives direct facts, uncertainty, and supporting
evidence without needing to infer meaning from raw rows.

Current production boundary:

```text
stage_1_research
  -> call_rag_supervisor
  -> rag_initializer
  -> rag_dispatcher
  -> top-level capability worker
  -> rag_evaluator
  -> rag_finalizer
  -> project_known_facts
  -> state["rag_result"]
  -> stage_2_cognition
```

Relevant production modules:

- `src/kazusa_ai_chatbot/rag/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/persistent_memory_search_agent.py`
- `src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/recall_agent.py`
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py`
- `src/kazusa_ai_chatbot/rag/prompt_projection.py`
- `src/kazusa_ai_chatbot/time_boundary.py`
- `src/kazusa_ai_chatbot/utils.py`
- `src/kazusa_ai_chatbot/message_envelope/prompt_projection.py`

Reference experiment code, design only:

- `experiments/rag2_recall_quality/improved_search_runner.py`
- `experiments/rag2_recall_quality/format_improved_outputs.py`
- `experiments/rag2_recall_quality/contextual_alt_case_matrix_builder.py`
- `experiments/rag2_recall_quality/blind_io_randomizer.py`
- `experiments/rag2_recall_quality/README.md`

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing RAG worker prompts, evaluator
  prompt shape, finalizer prompt shape, `rag_result` shape, or cognition
  handoff behavior.
- `debug-llm`: load before running live RAG2 comparisons, running live LLM
  tests, comparing outputs, or writing human-readable review artifacts.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files or tests that contain CJK
  string literals.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before continuing
  implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- Do not execute production code changes from this plan while `Status` is
  `draft`. Implementation requires user approval and status `approved` or
  `in_progress`.
- Do not import from `experiments.*`, `experiments/`, or
  `test_artifacts/` in production source, tests, or docs. The experiment code is
  a design guide only.
- The implementation agent must inspect and reuse existing production
  projection and prompt-safety helpers before creating new logic. This is
  required for prompt-injection safety and consistent timestamp handling.
- Reuse or extend production code in these areas where possible:
  - `kazusa_ai_chatbot.rag.prompt_projection` for LLM-facing recursive
    projection, raw-key stripping, and local-time conversion.
  - `kazusa_ai_chatbot.time_boundary` for storage UTC parsing and configured
    local-time projection. If second precision is required, add a production
    helper that reuses this boundary instead of slicing timestamp strings.
  - `kazusa_ai_chatbot.utils.trim_history_dict` and related production history
    projection behavior for `<image>...</image>` semantics. If a reusable image
    block renderer is needed, expose or relocate it in production code rather
    than copying private experiment code.
  - `kazusa_ai_chatbot.message_envelope.prompt_projection` for prompt-safe
    current-message and reply-attachment summaries.
  - `kazusa_ai_chatbot.rag.memory_retrieval_tools.conversation_message_payload`
    as the production conversation-row projection boundary after it is
    hardened.
- RAG returns evidence. Cognition decides stance, boundaries, character
  judgment, and response goals. Dialog renders visible wording.
- Preserve the existing `rag_result` top-level contract unless a change is
  explicitly documented and tests prove all consumers still work.
- Do not add a new response-path LLM call. Improve existing worker output,
  evaluator/finalizer prompting, and deterministic projection instead.
- Any runtime LLM prompt change must follow the local-LLM prompt rule:
  - write down the smallest semantic contract before editing the prompt;
  - keep stable role, policy, output shape, and decision procedure in the
    system message;
  - keep only current-run input, retrieved evidence, known facts, and tool
    results in the human message;
  - use plain domain language and avoid development-process terms such as
    "V2", "experiment", "productionization", "plan", "checkpoint", or
    "migration";
  - do not ask the model to generate backend parameters, database filters,
    search limits, graph traversal instructions, cache policy, or route
    mechanics;
  - do not send raw operational data and expect the model to infer meaning when
    deterministic code can project it into semantic descriptors first.
- Do not change the RAG initializer into a backend-query generator. It remains
  semantic-only.
- Do not add or change MongoDB databases, collections, indexes, caches, or
  schema migrations for this plan.
- Do not change graph/DAG/conversation-graph behavior, recent-history window
  ownership, conversation progress, dialog persona wording, or model routing.
- Prompt-facing evidence must not expose raw adapter wire syntax such as
  `[CQ:...]`, `raw_wire_text`, binary/base64 data, raw attachment URLs, raw
  storage ids, embedding arrays, or raw source rows.
- Prompt-facing evidence timestamps must use configured local wall-clock time.
  Raw UTC timestamps may remain only in trace/debug/source-ref material.
- Image-only and image-follow-up evidence must be represented with
  `<image>...</image>` blocks using escaped description text.
- Source refs and raw rows belong in `supervisor_trace`, worker payloads, or
  debug artifacts. They must not become the primary cognition evidence text.

## Must Do

- Read the current RAG, cognition, prompt-safety, and time-boundary contracts
  before implementation.
- Map every current `rag_result` consumer that reads `answer`,
  `memory_evidence`, `recall_evidence`, `conversation_evidence`,
  `external_evidence`, `user_image`, `user_memory_unit_candidates`,
  `character_image`, and `supervisor_trace`.
- Add or extend a production RAG evidence-formatting boundary under
  `src/kazusa_ai_chatbot/rag/`. Preferred shape is a small production module
  such as `evidence_formatting.py` unless inspection shows an existing module is
  the clearer owner.
- Harden conversation-row projection in production so RAG worker outputs use
  typed body text, prompt-safe reply excerpts, bounded attachment descriptions,
  escaped image blocks, local timestamps, and no raw adapter-wire syntax.
- Update memory evidence output so the public summary is answer-oriented:
  direct conclusion first, then supporting facts, uncertainty, and source
  boundaries.
- Update scoped user-memory retrieval so multiple literal anchors can
  contribute rows until the bounded evidence limit is reached, matching the
  useful experiment behavior that fixed the missed precision-priority memory.
- Update conversation evidence output so it separates direct evidence rows from
  nearby context, keeps speaker/time/message flow readable, and avoids
  summarizing unrelated neighboring group messages as if they are the same
  topic.
- Update recall evidence output so active commitments, completed commitments,
  no-evidence cases, and nearby commitments are formatted in the same
  cognition-ready style.
- Update `persona_supervisor2_rag_evaluator.py` and finalizer prompting only as
  needed to preserve structured, source-backed summaries. Do not add a new
  finalizer stage.
- Before changing any evaluator, finalizer, or helper-agent prompt, record the
  smallest semantic contract in execution evidence:
  - semantic question;
  - required inputs;
  - required output fields;
  - deterministic owners;
  - rejected prompt complexity.
- Update `persona_supervisor2_rag_projection.py` so existing public
  `rag_result` fields contain formatted evidence while preserving current
  field types.
- Keep raw worker payloads, raw source ids, and source rows available only in
  trace/debug surfaces needed for inspection.
- Update `src/kazusa_ai_chatbot/rag/README.md` and, if the handoff text changes,
  `src/kazusa_ai_chatbot/nodes/README.md`.
- Add focused deterministic tests for projection safety, timestamp projection,
  formatted evidence shape, existing consumer compatibility, and no experiment
  imports.
- Run live/debug LLM comparison on the same experiment case set before removing
  the experiment code. The human-readable report must include input and output
  for review and follow `debug-llm` guidance.
- After production implementation, verification, and independent code review,
  remove `experiments/rag2_recall_quality/`. Keep the experiment decision in
  this plan, the experiment plan, and the registry.

## Deferred

- Do not implement DAG, graph RAG, GraphRAG, `conversation_graph`, or a new
  memory graph in this plan.
- Do not replace the recent-history window.
- Do not change `conversation_progress`, internal monologue residue,
  reflection promotion, self-cognition, or dialog policy.
- Do not redesign RAG initializer routing or add RAG3 routing.
- Do not add new DB collections, indexes, cache tables, backfills, or data
  migrations.
- Do not create a new experiment package. Use the existing experiment package
  only as pre-cleanup design reference.
- Do not add deterministic relevance scoring that overrules the LLM evaluator's
  semantic judgment.
- Do not preserve experiment code after production sign-off.

## Cutover Policy

Overall strategy: bigbang for prompt-facing formatted evidence, compatible for
existing public `rag_result` keys and trace surfaces.

| Area                                 | Policy                     | Instruction                                                                                   |
| ------------------------------------ | -------------------------- | --------------------------------------------------------------------------------------------- |
| Prompt-facing RAG evidence text      | bigbang                    | Replace uneven free-form evidence summaries with formatted cognition-ready evidence directly. |
| Existing `rag_result` top-level keys | compatible                 | Preserve keys and field types used by cognition and consolidation.                            |
| RAG worker raw payloads              | compatible                 | Keep raw/debug data in trace and worker payloads only.                                        |
| RAG initializer and dispatcher       | compatible                 | Preserve semantic slots and capability routing.                                               |
| Database and cache                   | compatible                 | No new DB objects, migrations, or index changes.                                              |
| Experiment code                      | bigbang after verification | Remove `experiments/rag2_recall_quality/` after production implementation and review pass.    |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- If an area is `bigbang`, replace old prompt-facing behavior instead of
  keeping a flag, fallback, or dual prompt surface.
- If an area is `compatible`, preserve only the compatibility surface listed in
  this plan.
- Any change to cutover policy requires user approval before implementation.

## Target State

Production RAG2 still returns the same high-level `rag_result` shape:

```python
{
    "answer": str,
    "user_image": dict,
    "user_memory_unit_candidates": list[dict],
    "character_image": dict,
    "third_party_profiles": list[str],
    "memory_evidence": list[dict],
    "recall_evidence": list[dict],
    "conversation_evidence": list[str],
    "external_evidence": list[dict],
    "supervisor_trace": dict,
}
```

The difference is the quality of prompt-facing evidence. Cognition should see:

```text
Conclusion: direct factual answer or no-evidence result.
Evidence summary:
- Speaker/source at local time: full readable support.
- Relevant nearby context only when it changes the interpretation.
Uncertainty: what was not proven, or why the evidence is only nearby.
```

Memory evidence remains a list of dictionaries, but `summary` and `content`
become cognition-ready text instead of row dumps:

```python
{
    "summary": "Conclusion: ...",
    "content": "Evidence summary:\n- ...\nUncertainty: ...",
    "source_system": "user_memory_units",
    "scope_type": "user_continuity",
}
```

Conversation evidence remains `list[str]`, but each string is a formatted
evidence block rather than a raw paragraph or unstructured final answer.

Recall evidence remains `list[dict]`, but the prompt-facing fields inside each
dict use the same direct-answer plus evidence-summary shape.

## Design Decisions

| Topic                | Decision                                                           | Rationale                                                                                                      |
| -------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Production direction | Adopt the formatted improved V2 output behavior into RAG2          | The same-input blind-style review preferred formatted improved V2 in 10/10 real cases.                         |
| Architecture         | Improve existing RAG2 workers, evaluator/finalizer, and projection | The experiment showed a strong output-shape win without requiring DAG, graph, or a new retriever.              |
| Evidence owner       | RAG formats source-backed facts; cognition interprets them         | This preserves the existing RAG/cognition ownership boundary.                                                  |
| Output surface       | Preserve current `rag_result` top-level keys                       | The highest-risk consumer impact is cognition and consolidation expecting existing keys.                       |
| Experiment code      | Reference only, no imports                                         | Experiment modules are ignored, unreviewed production-wise, and must not become runtime dependencies.          |
| Prompt safety        | Reuse production projection helpers first                          | Prompt-safe CQ stripping, image blocks, raw-key stripping, and time projection already have production owners. |
| LLM budget           | No new response-path LLM call                                      | The fix should improve existing worker/evaluator/finalizer output rather than increasing latency and cost.     |
| Timestamp policy     | Local prompt-facing time, raw UTC trace-only                       | Existing time-boundary work already treats local time as the LLM-facing contract.                              |
| Cleanup              | Remove experiment code after production sign-off                   | The experiment was a proof path, not a long-term production dependency.                                        |

## Contracts And Data Shapes

### Formatted Evidence Text

All public evidence blocks should follow this semantic order:

```text
Conclusion: <direct fact, answer, or no-evidence finding>
Evidence summary:
- <source label, speaker when available, local timestamp when available, full readable evidence>
- <supporting or disconfirming evidence>
Uncertainty: <missing proof, conflict, or "none" when clear>
```

The implementation may tune labels and wording, but it must keep the same
meaning order: answer first, evidence second, uncertainty third.

### Source References

Source refs may include row ids, message ids, collection names, raw UTC
timestamps, and selected raw rows only in trace/debug payloads. They must not
be required by cognition to understand the evidence.

### Existing Consumers

Before changing projection behavior, inspect at least:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/consolidation/`
- `tests/test_rag_projection.py`
- `tests/test_consolidator_facts_rag2.py`
- `tests/test_user_memory_units_rag_flow.py`

## LLM Budget

- Current RAG2 already uses initializer, helper selector/generator/judge where
  applicable, evaluator, and finalizer calls.
- This plan must not add another live response-path LLM stage.
- Prompt changes may make existing evaluator/finalizer output more structured.
- Prompt changes must follow the local-LLM prompt rule: system message for
  stable contract, human message for current-run facts, plain semantic
  vocabulary, no implementation-history language, and no backend search
  mechanics.
- Deterministic formatting may assemble existing source fields into formatted
  blocks after the existing LLM stages.
- Any proposed extra LLM call must stop for user approval with cost, latency,
  and failure-mode justification.

## Change Surface

Expected source changes:

- `src/kazusa_ai_chatbot/rag/evidence_formatting.py` or an equivalent existing
  production module selected after code inspection.
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py`
- `src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`
- `src/kazusa_ai_chatbot/rag/recall_agent.py`
- `src/kazusa_ai_chatbot/rag/prompt_projection.py`
- `src/kazusa_ai_chatbot/time_boundary.py` if second-precision local
  formatting needs a public helper.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/rag/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md` if the cognition handoff text
  changes.

Expected tests:

- `tests/test_rag_evidence_formatting.py`
- `tests/test_rag_prompt_evidence_safety.py`
- `tests/test_memory_retrieval_tools.py`
- `tests/test_rag_projection.py`
- `tests/test_llm_time_payload_projection.py`
- `tests/test_rag_finalizer_time_context.py`
- `tests/test_rag_phase3_capability_agents.py`
- `tests/test_user_memory_evidence_agent.py`
- `tests/test_rag_search_body_text.py`

Expected validation helper:

- `scripts/validate_rag2_public_evidence_artifacts.py`

Expected cleanup:

- Remove `experiments/rag2_recall_quality/` after production implementation,
  debug review, deterministic verification, and independent code review pass.

## Overdesign Guardrail

This is a productionization plan for a proven experiment result. The execution
agent must not turn it into a new retrieval architecture project.

Do not add:

- graph storage;
- graph traversal;
- new embeddings;
- new database indexes;
- new memory collections;
- route planner changes;
- prompt-router redesign;
- extra response-path LLM calls;
- shadow RAG pipelines;
- broad refactors unrelated to formatted evidence quality.

## Agent Autonomy Boundaries

The execution agent may choose exact helper names, function names, and internal
module placement if they stay within the source boundaries above and preserve
the contract.

The execution agent must ask for approval before:

- changing `rag_result` top-level keys or field types;
- adding a new LLM call;
- changing RAG initializer or dispatcher semantics;
- adding a database/index/cache migration;
- changing cognition prompt ownership;
- keeping experiment code after production sign-off;
- adopting graph/DAG behavior.

## Implementation Order

1. Parent establishes the test contract before production edits.
   
   - File: `tests/test_rag_evidence_formatting.py`
   - Add:
     - `test_format_evidence_block_orders_conclusion_evidence_uncertainty`
     - `test_format_evidence_block_uses_empty_uncertainty_when_clear`
     - `test_format_evidence_block_does_not_emit_blank_sections`
     - `test_format_storage_utc_for_llm_seconds_projects_configured_local_time`
     - `test_format_storage_utc_for_llm_seconds_rejects_ambiguous_time`
   - Expected first run: fails because
     `kazusa_ai_chatbot.rag.evidence_formatting` and/or
     `format_storage_utc_for_llm_seconds` do not exist.
   - Command:
     `venv\Scripts\python.exe -m pytest tests/test_rag_evidence_formatting.py -q`
   - Evidence: record failing symbols or missing-entrypoint output before
     production implementation starts.

2. Parent adds prompt-safety and projection tests before production edits.
   
   - File: `tests/test_rag_prompt_evidence_safety.py`
   - Add:
     - `test_public_rag_result_evidence_allows_trace_ids_only`
     - `test_public_rag_result_evidence_rejects_raw_cq_wire_text_urls_ids_and_embeddings`
     - `test_public_rag_result_evidence_allows_source_refs_inside_supervisor_trace`
   - Public fields under test: `answer`, `memory_evidence`,
     `recall_evidence`, `conversation_evidence`, and `external_evidence`.
     `supervisor_trace` is the only allowed trace/source-ref exception.
   - Expected first run: fails because no shared production safety assertion or
     formatter exists yet.
   - Command:
     `venv\Scripts\python.exe -m pytest tests/test_rag_prompt_evidence_safety.py -q`
   - Evidence: record the failing public-field leak behavior.

3. Parent adds conversation-row projection tests before production edits.
   
   - File: `tests/test_memory_retrieval_tools.py`
   - Add:
     - `test_conversation_message_payload_projects_image_blocks_from_attachments`
     - `test_conversation_message_payload_projects_reply_image_blocks`
     - `test_conversation_message_payload_uses_local_second_precision_timestamp`
     - `test_conversation_message_payload_drops_raw_attachment_url_and_storage_ids`
     - `test_conversation_message_payload_strips_or_avoids_cq_wire_syntax`
   - Expected first run: at least URL/id/time/image-block/CQ tests fail against
     current `conversation_message_payload`.
   - Command:
     `venv\Scripts\python.exe -m pytest tests/test_memory_retrieval_tools.py -q`
   - Evidence: record which projection regressions are present.

4. Parent adds user-memory and projection contract tests before production
   edits.
   
   - File: `tests/test_user_memory_evidence_agent.py`
   - Add:
     `test_user_memory_evidence_agent_collects_multiple_literal_anchor_hits`.
   - File: `tests/test_rag_projection.py`
   - Add:
     - `test_project_known_facts_projects_formatted_memory_evidence`
     - `test_project_known_facts_projects_formatted_conversation_evidence`
     - `test_project_known_facts_projects_formatted_recall_evidence`
     - `test_project_known_facts_keeps_raw_refs_trace_only`
   - Expected first run: fails because current output is not formatted and
     scoped user-memory literal search stops at the first successful anchor.
   - Commands:
     `venv\Scripts\python.exe -m pytest tests/test_user_memory_evidence_agent.py tests/test_rag_projection.py -q`
   - Evidence: record expected failures and baseline output snippets.

5. Parent starts the production-code subagent.
   
   - Preconditions: steps 1-4 have failing or baseline test evidence recorded.
   - Subagent ownership: production source files only, limited to the expected
     source change surface in this plan. The subagent must not edit tests,
     experiment code, test artifacts, registry files, or this plan unless the
     parent explicitly hands over a plan-evidence update.
   - Required instructions to subagent:
     - read this plan, RAG README, nodes README, the completed prompt-safe
       history plan at
       `development_plans/archive/completed/bugfix/history_media_projection_image_boundary_plan.md`,
       and the experiment decision;
     - reuse production prompt-safe helpers where possible;
     - do not import from `experiments/`;
     - do not add new LLM calls;
     - do not change DB/index/cache/schema/routing;
     - do not revert unrelated user or agent edits.
   - Expected subagent output: changed production files, commands run,
     blockers, and residual risk.

6. Production-code subagent implements the source changes.
   
   - Add `src/kazusa_ai_chatbot/rag/evidence_formatting.py` with the formatter
     and prompt-safety functions proven by step 1 and step 2 tests.
   - Add `format_storage_utc_for_llm_seconds` to
     `src/kazusa_ai_chatbot/time_boundary.py` if second precision is needed.
   - Harden `conversation_message_payload` and its helper projections.
   - Update user-memory, memory, recall, conversation, evaluator/finalizer, and
     projection modules inside the approved change surface only.
   - For any prompt touched in evaluator/finalizer/helper agents, include a
     short prompt-contract note in execution evidence and remove runtime prompt
     language that describes the experiment, plan, migration, or development
     process.
   - Preserve public `rag_result` top-level keys and field types.

7. Parent runs focused module tests and loops until stable.
   
   - Commands:
     - `venv\Scripts\python.exe -m pytest tests/test_rag_evidence_formatting.py -q`
     - `venv\Scripts\python.exe -m pytest tests/test_rag_prompt_evidence_safety.py -q`
     - `venv\Scripts\python.exe -m pytest tests/test_memory_retrieval_tools.py -q`
     - `venv\Scripts\python.exe -m pytest tests/test_user_memory_evidence_agent.py tests/test_rag_projection.py -q`
   - If any focused test fails, fix inside the approved source surface or
     update the test only when the contract was wrong. Record the reason.

8. Parent adds and runs integration validation.
   
   - Update only if needed:
     - `tests/test_llm_time_payload_projection.py`
     - `tests/test_rag_finalizer_time_context.py`
     - `tests/test_rag_phase3_capability_agents.py`
     - `tests/test_rag_search_body_text.py`
   - Command:
     `venv\Scripts\python.exe -m pytest tests/test_llm_time_payload_projection.py tests/test_rag_finalizer_time_context.py tests/test_rag_phase3_capability_agents.py tests/test_rag_search_body_text.py -q`
   - Evidence: record pass/fail and any consumer contract issue discovered.

9. Parent adds artifact validation helper.
   
   - File: `scripts/validate_rag2_public_evidence_artifacts.py`
   - Behavior: read baseline artifacts, inspect only
     `projected_rag_result.answer`, `memory_evidence`, `recall_evidence`,
     `conversation_evidence`, and `external_evidence`, and fail on public-field
     leaks:
     - `[CQ:`
     - `raw_wire_text`
     - `base64_data`
     - `embedding`
     - `_id`
     - `conversation_row_id`
     - `platform_message_id`
     - raw attachment `url`
     - raw ISO UTC with microseconds
     - raw source row dictionaries
   - Allowed exception: `projected_rag_result.supervisor_trace` and raw
     `rag_supervisor_result` may contain ids and source refs for debug.
   - Command:
     `venv\Scripts\python.exe -m py_compile scripts\validate_rag2_public_evidence_artifacts.py`

10. Parent updates docs.
    
    - Files:
      - `src/kazusa_ai_chatbot/rag/README.md`
      - `src/kazusa_ai_chatbot/nodes/README.md` only if handoff text changed.
    - Required doc update: formatted evidence contract, prompt-facing safety
      boundary, and trace-only source refs.

11. Parent runs debug-LLM production validation before experiment cleanup.
    
    - Input artifact:
      `test_artifacts/rag2_recall_quality/inputs/contextual_case_matrix_alt10.jsonl`
    - Production RAG invocation harness:
      `venv\Scripts\python.exe -m experiments.rag2_recall_quality.baseline_runner --cases test_artifacts/rag2_recall_quality/inputs/contextual_case_matrix_alt10.jsonl --limit 10`
    - Raw production output artifacts:
      `test_artifacts/rag2_recall_quality/baseline/rag2-alt-*.json`
    - Public-evidence validation command:
      `venv\Scripts\python.exe -m scripts.validate_rag2_public_evidence_artifacts --input-dir test_artifacts/rag2_recall_quality/baseline --case-prefix rag2-alt- --expected-count 10`
    - Human-readable review output:
      `test_artifacts/rag2_recall_quality/production_cognition_ready_evidence_review.md`
    - The review must include, for every case:
      - full input;
      - decontextualized RAG input;
      - production `projected_rag_result`;
      - comparison note against the prior improved V2 output where useful;
      - agent judgment on cognition-readiness;
      - prompt-safety validation result.

12. Parent runs full verification and independent code review.
    
    - Run the full verification section below.
    - Start the independent code-review subagent only after deterministic and
      debug-LLM verification pass.
    - Fix review findings inside the approved change surface or request user
      approval if the review requires scope expansion.

13. Parent removes experiment code after review pass.
    
    - Command:
      `Remove-Item -Recurse -Force -LiteralPath 'experiments/rag2_recall_quality'`
    - Deletion proof:
      `Test-Path -LiteralPath 'experiments/rag2_recall_quality'`
    - Expected result: `False`.
    - Rerun import boundary and focused deterministic tests after cleanup.

## Execution Model

- Parent-led execution using exactly two execution subagents in sequence after
  plan approval:
  1. one production-code subagent after the parent establishes the focused test
     contract in steps 1-4;
  2. one independent code-review subagent after planned implementation and
     verification pass.
- The parent owns orchestration, tests, integration, docs, debug-LLM review,
  cleanup, execution evidence, lifecycle updates, and final sign-off.
- The production-code subagent owns production source changes only. It must not
  edit tests, experiment files, ignored artifacts, or development plans unless
  the parent explicitly changes ownership.
- The independent code-review subagent owns review only. It must not implement
  fixes.
- If native subagent capability is unavailable during implementation, stop and
  report the blocker. Do not silently execute as a single-agent production
  change unless the user explicitly approves fallback execution.

## Progress Checklist

- [ ] Stage 0 - plan approved for implementation
  - Covers: approval only.
  - Verify: plan `Status` is `approved` or `in_progress`.
  - Evidence: record user approval and initial `git status --short`.
  - Handoff: start Stage 1.
  - Sign-off: `<agent/date>`.
- [ ] Stage 1 - focused test contract established
  - Covers: implementation steps 1-4.
  - Verify: run the four focused pytest commands listed in steps 1-4.
  - Evidence: record expected failures, missing symbols, and baseline snippets.
  - Handoff: start production-code subagent.
  - Sign-off: `<agent/date>`.
- [ ] Stage 2 - production-code subagent completed source changes
  - Covers: implementation steps 5-6.
  - Verify: subagent reports changed production files and commands run.
  - Evidence: record subagent id, changed files, blockers, and residual risks.
  - Handoff: parent runs focused tests.
  - Sign-off: `<agent/date>`.
- [ ] Stage 3 - focused module tests pass
  - Covers: implementation step 7.
  - Verify: focused formatter, prompt-safety, conversation projection,
    user-memory, and RAG projection tests pass.
  - Evidence: record command output and any loop-back fixes.
  - Handoff: run integration tests.
  - Sign-off: `<agent/date>`.
- [ ] Stage 4 - integration tests and docs complete
  - Covers: implementation steps 8-10.
  - Verify: integration pytest command and `py_compile` for the artifact
    validation helper pass.
  - Evidence: record docs changed and validation-helper compile output.
  - Handoff: run debug-LLM production validation.
  - Sign-off: `<agent/date>`.
- [ ] Stage 5 - debug-LLM production validation complete
  - Covers: implementation step 11.
  - Verify: baseline runner writes 10 `rag2-alt-*` artifacts, artifact public
    evidence validator reports expected count 10, and
    `production_cognition_ready_evidence_review.md` exists.
  - Evidence: record commands, report path, and agent review conclusion.
  - Handoff: run full verification and independent code review.
  - Sign-off: `<agent/date>`.
- [ ] Stage 6 - independent code review complete and remediated
  - Covers: implementation step 12.
  - Verify: independent reviewer reports no unresolved blockers or major
    findings, or accepted residual risk is recorded with user approval when
    needed.
  - Evidence: record review id, findings, fixes, and rerun commands.
  - Handoff: remove experiment code.
  - Sign-off: `<agent/date>`.
- [ ] Stage 7 - experiment code removed and final checks pass
  - Covers: implementation step 13.
  - Verify: `Test-Path -LiteralPath 'experiments/rag2_recall_quality'` returns
    `False`, import grep returns no production/test imports, and focused tests
    still pass.
  - Evidence: record deletion proof and final `git status --short`.
  - Handoff: final user report.
  - Sign-off: `<agent/date>`.

## Verification

Use `venv\Scripts\python.exe` for all Python commands.

Focused deterministic checks:

```powershell
venv\Scripts\python.exe -m pytest `
  tests/test_rag_evidence_formatting.py `
  tests/test_rag_prompt_evidence_safety.py `
  tests/test_memory_retrieval_tools.py `
  tests/test_rag_projection.py `
  tests/test_llm_time_payload_projection.py `
  tests/test_rag_finalizer_time_context.py `
  tests/test_rag_phase3_capability_agents.py `
  tests/test_user_memory_evidence_agent.py `
  tests/test_rag_search_body_text.py
```

Artifact validator compile:

```powershell
venv\Scripts\python.exe -m py_compile scripts\validate_rag2_public_evidence_artifacts.py
```

Import boundary check:

```powershell
rg -n "experiments\\.|from experiments|import experiments" src tests scripts
```

Experiment cleanup proof:

```powershell
Test-Path -LiteralPath 'experiments/rag2_recall_quality'
```

Expected result after cleanup: `False`.

Prompt-facing leak checks must cover public `projected_rag_result` fields only,
excluding `supervisor_trace` and raw debug artifacts:

- `[CQ:`
- `raw_wire_text`
- `base64_data`
- `embedding`
- `_id`
- `conversation_row_id`
- `platform_message_id`
- raw attachment `url`
- raw source row dictionaries
- raw ISO UTC with microseconds in public evidence
- image-only blank evidence rows

Live/debug LLM verification:

```powershell
venv\Scripts\python.exe -m experiments.rag2_recall_quality.baseline_runner --cases test_artifacts/rag2_recall_quality/inputs/contextual_case_matrix_alt10.jsonl --limit 10
venv\Scripts\python.exe -m scripts.validate_rag2_public_evidence_artifacts --input-dir test_artifacts/rag2_recall_quality/baseline --case-prefix rag2-alt- --expected-count 10
```

Expected raw artifacts:

```text
test_artifacts/rag2_recall_quality/baseline/rag2-alt-001.json
...
test_artifacts/rag2_recall_quality/baseline/rag2-alt-010.json
```

Required human-readable report:

```text
test_artifacts/rag2_recall_quality/production_cognition_ready_evidence_review.md
```

The report must include each input, decontextualized RAG input, current
production `projected_rag_result`, relevant prior improved V2 output when
useful, agent quality judgment, raw artifact paths, and whether cognition can
consume the facts without reconstructing the answer from raw rows.

## Independent Plan Review

Before this plan can be approved for implementation, run an independent plan
review with a fresh agent. The reviewer must check:

- architecture alignment with RAG/cognition ownership;
- whether implementation steps are production-executable;
- whether prompt-injection and prompt-safe projection requirements are clear;
- whether experiment code is reference-only and forbidden as an import;
- whether cleanup is enforceable;
- whether tests and debug-LLM validation are sufficient;
- whether any step accidentally broadens scope into DAG, graph RAG, routing, or
  database migration.

Record the review result in `Execution Evidence` before requesting approval.

## Independent Code Review

Before completion, run an independent code review with a fresh agent. The
reviewer must read this plan, inspect the final diff, and focus on:

- prompt-safety regressions;
- raw CQ, raw UTC, raw id, raw URL, binary, or embedding leaks;
- broken RAG-to-cognition contract;
- new production imports from `experiments`;
- accidental new LLM calls;
- behavior that hides useful evidence from cognition;
- tests that assert implementation details without checking consumer behavior.

Record the review result and fix status in `Execution Evidence`.

## Acceptance Criteria

- Production RAG2 uses formatted cognition-ready evidence for memory,
  user-memory, recall, and conversation evidence.
- Existing `rag_result` top-level keys and field types remain compatible with
  cognition and consolidation consumers.
- Same-source real-data cases used by the experiment produce readable,
  source-backed production output that an agent review judges at least as useful
  as current RAG2 and consistent with the V2 experiment decision.
- Prompt-facing evidence contains no raw CQ syntax, raw storage ids,
  `raw_wire_text`, binary/base64 data, raw attachment URLs, embedding arrays, or
  raw ISO UTC microsecond timestamps.
- Image evidence is represented with escaped `<image>...</image>` blocks.
- No new response-path LLM call is added.
- No new database, collection, index, cache, or migration is added.
- Production source and tests do not import from `experiments/`.
- `experiments/rag2_recall_quality/` is removed after production verification
  and review pass.
- RAG and handoff docs describe the formatted evidence contract.

## Execution Evidence

### 2026-05-23 draft

- User requested a new production executable plan based on the RAG2 recall
  quality experiment decision.
- User added two constraints:
  - implementation must reuse existing production code where possible to keep
    prompt-injection and projection safety;
  - production code may refer to experiment design, but must not import from
    `experiments/`.
- Current draft status means this plan is not yet approved for implementation.

### 2026-05-23 independent plan review

- Independent reviewer `019e545d-fc63-7ec2-912a-67c8dce9409d` reported:
  - BLOCKER: execution model and implementation steps were not granular enough
    for the development-plan gates.
  - MAJOR: prompt-safety verification missed ids, URLs, embeddings, source
    rows, and trace-only exceptions.
  - MAJOR: experiment cleanup lacked a mechanical deletion proof even though
    `experiments/rag2_recall_quality/` is git-ignored.
  - MAJOR: debug-LLM validation did not name the exact input artifact, harness,
    raw output path, or review report path.
- Remediation applied in this draft:
  - replaced broad execution steps with parent-owned test-contract-first
    stages;
  - required exactly two execution subagents after approval;
  - added public-field prompt leak tests and a production artifact validator;
  - added exact debug-LLM input, harness, raw artifact, and Markdown report
    paths;
  - added `Test-Path` deletion proof for the ignored experiment directory.
- The same reviewer re-checked the remediated plan and reported no remaining
  blockers or major findings for the reviewed items.

### 2026-05-23 prompt-rule clarification

- User added that LLM prompt work must follow the LLM prompt rule.
- Added mandatory prompt-change rules:
  - smallest semantic contract before prompt edits;
  - stable contract in system message;
  - current-run facts in human message;
  - no backend/search mechanics in prompts;
  - no development-process or experiment language in runtime prompts;
  - deterministic projection of raw operational data before model input.
