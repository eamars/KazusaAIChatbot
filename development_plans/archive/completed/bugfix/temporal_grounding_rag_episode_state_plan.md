# temporal grounding rag episode state plan

## Summary

- Goal: Prevent producer and writer stages from emitting unresolved relative temporal language into active operational state or producer-owned active evidence, without adding a central temporal-grounding module, deterministic semantic parser, or downstream consumer safeguard.
- Plan class: large
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `cjk-safety`, `test-style-and-execution`.
- Overall cutover strategy: bigbang for stage prompt/output contracts; compatible for historical stored rows; no data migration.
- Highest-risk areas: moving semantic judgment into deterministic code, over-suppressing valid historical quotes at producer time, appending prompt fragments that break prompt flow, accidentally adding consumer safeguards, and touching RAG/cognition modules outside the producer boundary.
- Acceptance criteria: each producer or writer stage that emits time-bearing active state owns its temporal contract; unresolved temporal leakage is omitted or marked inactive before persistence/projection; deterministic code only detects missing or unreadable structured fields, persists/carries fields, and routes existing output.

## Context

The QQ private-chat investigation for platform user `673225019` showed a reinforcement loop around `考核` and `奖励`. The root path was not just dialog repetition:

```text
conversation_progress -> recall_agent -> RAG finalizer -> cognition anchors -> dialog -> progress recorder
```

The risk is temporal leakage. Text such as `明天考核`, `今晚奖励`, `下次`, `later tonight`, or `next Friday` can be correct in its source turn but wrong when later treated as current. The problem is broader than `明天`: any relative phrase that describes a date, time, due date, interval, recurrence, or future event can leak if a stage promotes it without temporal ownership.

Live-LLM diagnostics in `tests/test_temporal_relative_terms_live_llm.py` validated the current failure mode:

- The progress recorder preserved `今晚`, `明天`, `下周`, and `一会儿` in operational episode state with no absolute dates.
- Downstream RAG finalizer diagnostics showed that leaked producer text can be repeated later.
- L3 content-anchor diagnostics showed downstream cognition can be affected by upstream leakage.

Those downstream observations explain the risk, but they are not part of the fix boundary. The approved fix is producer-side only.

The revised architecture is stage-local:

```text
producer stage owns temporal grounding for the facts it emits
RAG evidence producers own temporal grounding for summaries they emit
RAG and cognition consumers remain unchanged and receive no new safeguards
deterministic code detects missing structured fields and carries source-owned labels only
```

This matches a producer-owned NLU pattern: date/time is resolved at the entity, slot, memory, or evidence stage that understands the utterance. Downstream fulfillment remains simple and does not become a second semantic correction layer. It does not use a global cleanup pass over arbitrary prose.

## Mandatory Skills

- `local-llm-architecture`: load before changing recorder, consolidation, producer prompt, helper-agent, or evidence-producer contracts.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files containing Chinese/Japanese text or prompts.
- `test-style-and-execution`: load before adding, changing, or running deterministic, patched, or live LLM tests.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Check `git status --short` before editing.
- Do not read or edit `.env`.
- Do not add a dedicated temporal-grounding module.
- Do not add deterministic natural-language parsing, keyword filtering, regex-based semantic discard, or global cleanup over arbitrary prose.
- Do not implement topic suppression for `考核`, `奖励`, or any other topic word. The issue is temporal validity, not topic validity.
- Do not add a new response-path LLM call.
- Treat the fix as best-attempt LLM behavior. If a producer LLM fails to follow the temporal contract, do not add deterministic natural-language catchers or downstream consumer safeguards to repair the mistake. Record the failure through tests/logs and improve the owning prompt contract instead.
- Do not change whether a user promise, agreement, or commitment is semantically accepted. The existing LLM stage that owns that semantic judgment remains the owner.
- Deterministic code may only determine whether expected structured data is present and machine-readable enough for persistence, projection, and routing. If a field is absent, malformed, or unreadable, deterministic code may treat that field as missing or unusable.
- Deterministic code must not validate the semantic quality, correctness, freshness, completeness, or adequacy of LLM-produced content. It must not decide whether the LLM grounded a promise well, chose the right absolute date, or wrote a high-quality fact. This remains best-attempt prompt behavior validated by real LLM tests.
- Do not add new deterministic lifecycle inference for this fix. Deterministic code may carry source-owned `due_at`, `due_state`, `temporal_resolution`, `active_eligible`, and `discard_reason` when present, but must not compute or repair them from prose.
- Do not add temporal safeguards to RAG finalizers, cognition modules, dialog modules, or other downstream consumers. This plan fixes leakage only where active state or evidence is produced.
- This is primarily a prompt-contract change. Add deterministic code only when needed to carry structured fields already produced by the owning LLM stage or to detect that required structured data is missing.
- Any prompt change must restructure the entire affected prompt as a coherent stage contract, not just the nearest section. Do not paste, copy, or append a standalone temporal-safety block, common instruction block, warning, bullet, or afterthought into an existing prompt.
- Temporal grounding instructions must be naturally integrated into the prompt's role, input interpretation, generation procedure, output format, and stage-specific examples or redlines where those concepts already belong. A reviewer must be able to read the prompt top to bottom without seeing plan-specific wording or a bolted-on policy fragment.
- Before editing any prompt, inspect the full prompt and handler payload together. The edited prompt must keep `# Generation Procedure`, `# Input Format`, and `# Output Format` internally consistent with the actual JSON payload and parsed output.
- Do not touch files, prompts, validators, schemas, or tests that are not necessary for the stage-local contract being changed.
- Keep implementation additions minimal and robust: prefer updating the existing owning prompt and existing structured payload over adding new helpers, schema fields, or routing paths.
- Real LLM tests are the gating item. The implementation cannot be signed off, marked complete, or reported as accepted unless the required live LLM tests pass one by one and the traces are inspected.
- Every prompt edited by this plan must have an explicit row in `Prompt Validation Matrix` and a named real LLM test before the edit is accepted. The four existing diagnostics are not enough by themselves if additional prompts are changed.
- If implementation discovers another prompt must be edited, stop and update `Prompt Validation Matrix`, add the real LLM gate, and rerun the relevant plan review before changing that prompt.
- Raw `conversation_history.body_text` remains exact evidence and must not be rewritten.
- Historical quotes may preserve relative wording only when clearly labeled as quotes/proof, not active facts.
- Any active operational or producer-owned active evidence fact containing unresolved relative time must be omitted or marked inactive by the producing LLM stage before persistence or projection.
- For Python files containing CJK strings, use single-quoted literals where CJK quote corruption is possible and run `py_compile` after edits.
- Real LLM tests must be run one case at a time and inspected one case at a time.

## Must Do

- Update the progress recorder prompt/output contract so it must either emit absolute calendar wording for time-bearing operational fields or omit the unsafe item from operational state.
- Update the memory-unit/promise writer prompts so accepted time-bearing commitments include structured `due_at` when a due date exists, preserve source wording separately where useful, and mark unresolved timing as not eligible for active commitment use.
- Update facts harvester/evaluator prompts only if they emit or approve time-bearing future promises as active facts.
- Update conversation evidence summarization only where it produces a summary from raw history; raw history remains raw, but produced summaries distinguish historical quote/proof from active promise.
- Convert the live diagnostic tests into post-fix live contract checks after implementation.
- Maintain the `Prompt Validation Matrix` so every edited prompt has a direct live LLM validation gate. If a prompt-bearing file is inspected but left unchanged, record that no-edit decision in `Execution Evidence`.

## Deferred

- Do not backfill or rewrite historical `conversation_history`, `user_memory_units`, or `conversation_episode_state` collections.
- Do not add Duckling, dateparser, Recognizers-Text, or any new parser dependency in this plan.
- Do not implement per-user timezone inference.
- Do not redesign scheduler firing, reminder dispatch, or reflection promotion.
- Do not remove raw quotes from history evidence.
- Do not decommission existing `due_at` / `due_state` user-memory projection.
- Do not add consumer-side temporal safeguards in RAG finalizer, RAG evaluator, cognition L3, dialog, or other downstream modules.
- Do not solve every natural-language time expression. Ambiguous timing must be marked unresolved or omitted by the owning producer LLM stage when active use would be unsafe.

## Cutover Policy

| Area                                    | Policy                       | Instruction                                                                                                   |
| --------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------- |
| New `conversation_episode_state` writes | bigbang                      | Recorder-owned output contract changes immediately; unresolved time-bearing operational items are omitted.    |
| Existing episode-state rows             | compatible                   | No migration and no consumer repair. Old relative wording can remain until naturally overwritten or aged out. |
| Memory/promise writes                   | bigbang                      | Time-bearing commitments must include structured due metadata or be marked unresolved/not active-eligible.    |
| Recall active agreement output          | compatible                   | No new Recall safeguard in this plan. Recall consumes whatever producer-owned records provide.                |
| Conversation history evidence           | compatible                   | Preserve raw text; summaries distinguish historical quotes from active facts.                                 |
| RAG finalizer and cognition consumers   | compatible                   | No prompt or guard changes. Downstream modules are outside this producer-side plan.                           |
| Live diagnostic tests                   | bigbang after implementation | Convert from diagnostic traces to contract assertions.                                                        |

## Agent Autonomy Boundaries

- The implementation agent may edit prompt contracts and validators inside the listed stages only.
- The agent must not introduce a central module, shared semantic parser, feature flag, migration, or extra LLM stage.
- The agent must not add consumer-side safeguards to RAG finalizer, RAG evaluator, cognition L3, dialog, or related downstream consumers.
- The agent must not perform unrelated prompt rewrites, formatting churn, dependency upgrades, or broad refactors.
- The agent must not touch code where no update is required to satisfy the listed prompt-contract changes.
- The agent must treat prompt edits as first-class design work: rewrite the entire affected prompt into a coherent ordered procedure and stage contract, not by appending isolated caveats or copying a shared temporal block.
- The agent may not add new deterministic ranking, filtering, or semantic quality checks as a substitute for producer prompt behavior.
- If the plan and code disagree, preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

Operational episode state uses source-stage-owned grounded wording:

```json
{
  "current_thread": "2026-05-08 晚上游戏安排和 2026-05-09 香料考核奖励",
  "open_loops": [
    "2026-05-09 香料考核决定特别称呼奖励"
  ],
  "progression_guidance": "不要把已过期的考核条件继续当作未来压力"
}
```

If the recorder cannot resolve or safely restate the timing, it omits the item from operational fields. It does not write:

```text
明天香料考核决定特别称呼奖励
```

Memory/promise output separates fact, source wording, and temporal status:

```json
{
  "fact": "2026-05-09 香料考核决定特别称呼奖励",
  "source_phrase": "明天香料考核",
  "source_time": "2026-05-08 21:47",
  "due_at": "2026-05-09 00:00",
  "temporal_resolution": "resolved",
  "active_eligible": true
}
```

Conversation evidence summaries can preserve raw quote text while labeling it as historical:

```json
{
  "summary": "用户在 2026-05-08 21:46 说过“明天早上考核通过的话，下次给特别称呼”。这是历史原文证据，不是当前活跃承诺。",
  "active_eligible": false
}
```

## Design Decisions

| Topic                   | Decision                                     | Rationale                                                                                                                                 |
| ----------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Temporal ownership      | Stage-local LLM ownership                    | The stage seeing the source context understands whether text is promise, quote, plan, or style.                                           |
| Central module          | Rejected                                     | A shared parser would become a hidden semantic authority and drift from stage intent.                                                     |
| Deterministic role      | Missing-data and plumbing only               | Code detects absent/unreadable structured fields, stores/carries fields, and routes source-owned labels without judging semantic quality. |
| Implementation emphasis | Prompt-focused, minimal code                 | The failure is prompt contract behavior; code changes only carry LLM-owned structured output or detect missing fields.                    |
| Leakage handling        | Producer-only omit or inactive marking       | Unsafe active facts must not be produced as active state; downstream consumers are not changed in this plan.                              |
| LLM failure handling    | Best attempt, no deterministic repair        | If the LLM fails the contract, improve the owning prompt and live test coverage rather than adding prose catchers.                        |
| Raw history             | Preserve exactly                             | History is audit evidence and can contain relative wording as quote/proof.                                                                |
| Unsupported timing      | Producer marks inactive or omits active fact | Guessing dates is worse than losing an active hint.                                                                                       |
| RAG/cognition consumers | Out of scope                                 | No finalizer, evaluator, L3, or dialog safeguard is added by this plan.                                                                   |
| Topic repetition        | Do not suppress topics                       | `考核` can remain valid if temporally grounded.                                                                                             |

## Contracts And Data Shapes

### Shared Shape, Not Shared Parser

Stages may use a common structural vocabulary in their own outputs:

```json
{
  "temporal_resolution": "resolved | unresolved | timeless | historical_quote",
  "active_eligible": true,
  "source_phrase": "optional raw phrase",
  "source_time": "local YYYY-MM-DD HH:MM when known",
  "due_at": "local YYYY-MM-DD HH:MM when known",
  "discard_reason": "required when active_eligible=false for temporal reasons"
}
```

This is a schema vocabulary, not a central parser. Each producer stage fills it using its own prompt and visible context. You shall NOT inject common wordings into the LLM prompt. All changes must be in place.

Deterministic code may:

- detect that required structured fields are absent;
- treat malformed or unreadable machine fields as missing/unusable for persistence or projection;
- preserve and pass through producer-owned fields such as `due_at`, `due_state`, `temporal_resolution`, `active_eligible`, and `discard_reason`.

Deterministic code must not:

- scan arbitrary prose for relative words to decide discard;
- rewrite natural-language facts;
- infer a due date from `source_phrase`;
- change `active_eligible` based on keyword matching.
- judge whether an LLM-chosen absolute date is correct;
- judge whether LLM-produced wording is high quality, sufficiently grounded, or semantically complete;
- compute a new `due_state`, `temporal_resolution`, or `active_eligible` value from natural-language text.
- add downstream filtering, ranking, or guard behavior based on temporal semantics.

### Conversation Progress Recorder Contract

Modify `src/kazusa_ai_chatbot/conversation_progress/recorder.py`:

- The recorder must treat `conversation_episode_state` as operational state, not raw evidence.
- For time-bearing scalar/list fields, it must output absolute date/time wording when it can resolve the timing from `current_turn_timestamp`, `prior_episode_state`, or visible history.
- If timing cannot be resolved, it must omit the item from active fields.
- It must not copy a prior item containing unresolved relative time into the new state unless it rewrites it into absolute wording.

Validation remains missing-data oriented: required keys exist, required fields are machine-readable enough for storage/projection, and unreadable fields are treated as absent. Validation does not keyword-scan for relative terms or judge semantic quality.

### Memory And Promise Writer Contract

Modify the memory-unit and future-promise writer prompts in the consolidation path:

- Accepted commitments with due dates must include `due_at` and `temporal_resolution="resolved"`.
- If a promise-like fact contains relative timing but the writer cannot resolve it, it must emit `temporal_resolution="unresolved"`, `active_eligible=false`, and `discard_reason`.
- Raw source wording may be preserved in `source_phrase`; the operational `fact` must not rely on unresolved relative timing.

Persistence may store these structured fields when available. If schema changes are too broad, the plan may store the structured fields in existing prompt-facing metadata only. Do not add consumer filtering to compensate for missing fields.

### Conversation Evidence Contract

Modify `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py` and its summarizer path:

- Raw history rows remain quote/proof evidence.
- The conversation-evidence summarizer must label time-bearing quote summaries as historical unless it can safely ground them from visible source time.
- Historical quote summaries must not set `active_eligible=true`.
- If conversation evidence produces an active agreement summary and the timing is unresolved, it must emit the summary as unresolved or historical-only. This is producer behavior for the summary itself, not a downstream consumer safeguard.

## Change Surface

### Modify Only If Required

- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`: recorder prompt/output contract for absolute-or-omit operational state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`: commitment writer temporal fields and unresolved handling.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`: future-promise extraction prompt alignment if it emits time-bearing promises.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`: historical quote vs active evidence contract.
- `tests/test_temporal_relative_terms_live_llm.py`: keep producer-side live diagnostics and convert relevant cases into post-fix producer contract checks.
- Focused deterministic tests for producer output shape and persistence/projection plumbing only.

The implementation agent must inspect each listed file before editing. If a file already satisfies the revised contract or is not needed for the minimal fix, leave it untouched and record that decision in execution evidence.

### Keep

- No `src/kazusa_ai_chatbot/temporal_grounding.py`.
- No deterministic natural-language temporal parser.
- Raw `conversation_history.body_text`.
- Existing `time_context.py` public contracts.
- Existing scheduler and dispatcher behavior.
- `src/kazusa_ai_chatbot/rag/recall_agent.py`: no new temporal selection or safeguard behavior.
- `src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py`: no new consumer-side inference or filtering behavior.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_prompt_views.py`: no new consumer-side temporal metadata requirement unless a producer already emits it and existing projection needs a trivial pass-through.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`: no finalizer, summarizer, evaluator, or continuation-assessor temporal guard changes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`: no cognition or content-anchor temporal guard changes.

## Prompt Validation Matrix

This matrix is mandatory. A prompt may be edited only if it is listed here with a live LLM gate, or the agent first updates this matrix and adds the matching test. If a prompt-bearing file is inspected and the prompt is not edited, no live LLM test is required for that unchanged prompt, but the no-edit decision must be recorded in `Execution Evidence`.

| File                                                                           | Prompt(s)                          | Contract involvement                                                                                                 | Required real LLM gate                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------ | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/kazusa_ai_chatbot/conversation_progress/recorder.py`                      | `_RECORDER_PROMPT`                 | Required edit. Owns operational `conversation_episode_state` and must emit absolute-or-omit time-bearing state.      | Convert `tests/test_temporal_relative_terms_live_llm.py::test_live_recorder_diagnostic_relative_episode_state` into `test_live_recorder_contract_absolute_or_omit_episode_state`.                                                          |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py` | `_EXTRACTOR_PROMPT`                | Required inspect and likely edit. Owns accepted commitment extraction and `due_at` / `temporal_resolution` emission. | Keep/run `tests/test_user_memory_units_live_llm.py::test_live_extractor_anchors_tomorrow_commitment_due_at`; add `test_live_extractor_marks_unresolved_relative_commitment_inactive` if unresolved active-eligibility behavior is changed. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py` | `_REWRITE_PROMPT`                  | Required inspect; edit only if rewrite can reintroduce relative due wording or drop structured due metadata.         | If edited, add `tests/test_user_memory_units_live_llm.py::test_live_rewrite_preserves_due_at_and_does_not_reintroduce_relative_due_text`.                                                                                                  |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py` | `_STABILITY_PROMPT`                | Required inspect; edit only if stability judgment can promote unresolved or past-due relative commitments.           | If edited, add `tests/test_user_memory_units_live_llm.py::test_live_stability_respects_temporal_resolution_and_due_state`.                                                                                                                 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py` | `_MERGE_JUDGE_PROMPT`              | Inspect only. Edit only if merge decision loses temporal metadata or causes stale relative promise carry-forward.    | If edited, add `tests/test_user_memory_units_live_llm.py::test_live_merge_judge_respects_structured_temporal_metadata`.                                                                                                                    |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`        | `_FACTS_HARVESTER_PROMPT`          | Required inspect; edit if it emits time-bearing future promises or active facts.                                     | If edited, add `tests/test_temporal_relative_terms_live_llm.py::test_live_facts_harvester_contract_relative_due_terms`.                                                                                                                    |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`        | `_FACT_HARVESTER_EVALUATOR_PROMPT` | Required inspect; edit if evaluator can approve unresolved relative future promises.                                 | If edited, add `tests/test_temporal_relative_terms_live_llm.py::test_live_facts_evaluator_contract_relative_due_terms`.                                                                                                                    |
| `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`                     | `_SELECTOR_PROMPT`                 | Required inspect; edit if selected evidence summary can present historical relative quotes as active promises.       | If edited, add `tests/test_temporal_relative_terms_live_llm.py::test_live_conversation_evidence_contract_historical_quote_not_active_promise`.                                                                                             |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`             | all prompts                        | Out of scope. RAG evaluator/finalizer/summarizer prompts are downstream consumers for this issue.                    | Do not edit. If touched, stop and create a separate consumer-side plan.                                                                                                                                                                    |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`              | all prompts                        | Out of scope. Cognition L3 is a downstream consumer for this issue.                                                  | Do not edit. If touched, stop and create a separate consumer-side plan.                                                                                                                                                                    |

## LLM Call And Context Budget

- No new response-path LLM calls.
- No new background LLM calls.
- Existing LLM calls receive clearer output contracts and, where already present, structured fields.
- RAG finalizer, cognition, and dialog calls are unchanged.
- Expected latency impact is negligible because call count does not change.
- If producer prompt context budget is tight, preserve visible source time, current turn timestamp, source label, and short source excerpt; drop verbose unrelated history first.

## Implementation Order

1. Reconcile the target files against `Prompt Validation Matrix`. Before editing any prompt, confirm the prompt is listed with a named real LLM gate.
2. Add or update the live LLM test for the prompt being changed. Run it once before the prompt edit when practical and record the baseline failure or diagnostic behavior.
3. Inspect the full prompt and handler payload for the target stage before editing.
4. Rewrite the entire affected prompt into a coherent generation flow and stage contract; do not append isolated warnings, paste copied blocks, or preserve duplicated plan wording.
5. Manually review the prompt diff before running tests. If the edit reads like an appended block rather than a natural rewrite, revise the prompt first.
6. Run a prompt-render or syntax check for the edited stage.
7. Run that prompt's named live LLM test before moving to the next prompt. If it fails the temporal contract, revise the owning prompt rather than adding deterministic repair code.
8. Repeat steps 1-7 only for stages that must change.
9. Add deterministic tests only for producer output shape, missing-field handling, persistence, and projection plumbing. These tests must not assert keyword-scanned semantic discard or consumer-side filtering.
10. Convert producer-side live temporal diagnostics from diagnostic assertions to post-fix contract assertions.
11. Run focused deterministic tests.
12. Run every required live LLM test for every edited prompt individually and inspect trace files.
13. Record execution evidence only after implementation is actually performed and all edited-prompt live LLM gates pass.

## Progress Checklist

- [x] Stage 1 - stage-local contracts updated
  - Covers: recorder, memory/promise writer, facts harvester, and conversation-evidence producer prompt edits.
  - Verify: `Prompt Validation Matrix` reconciled, prompt render/syntax checks, focused payload tests, and the named live LLM gate for each edited prompt.
  - Evidence: record changed prompts, prompts inspected but not edited, render checks, and live LLM trace paths in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-08` after verification and plan reread.
- [x] Stage 2 - structured temporal fields preserved
  - Covers: producer output shape, persistence fields, and projection plumbing needed by the producer-owned contracts.
  - Verify: focused deterministic tests for field presence, missing-field behavior, and persistence/projection shape.
  - Evidence: record test output before moving on.
  - Sign-off: `Codex/2026-05-08` after verification and plan reread.
- [x] Stage 3 - consumer boundary verification
  - Covers: proving no RAG finalizer, RAG evaluator, cognition L3, dialog, or Recall temporal safeguard was added.
  - Verify: focused git diff review and static grep for modified consumer prompt files.
  - Evidence: record the unchanged consumer files in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-08` after verification and plan reread.
- [x] Stage 4 - live LLM verification
  - Covers: one-by-one live temporal contract checks for every prompt edited by this plan.
  - Verify: run each required live LLM test named in `Prompt Validation Matrix` individually with `-m live_llm -q -s`.
  - Evidence: record trace paths and manual judgment for each edited prompt. This stage is mandatory and cannot be skipped.
  - Sign-off: `Codex/2026-05-08` after verification and plan reread.

## Verification

### Static

- `venv\Scripts\python.exe -m py_compile tests\test_temporal_relative_terms_live_llm.py`
- Prompt render checks for every edited prompt file.
- `rg "temporal_grounding|ground_temporal_text|sanitize_operational_text" src tests` returns no production implementation references. Test names may mention temporal diagnostics only.
- `git diff --name-only` must not include `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` or `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`.
- `git diff -- src/kazusa_ai_chatbot/rag/recall_agent.py src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_prompt_views.py` must be empty unless the only change is a trivial producer-field pass-through explicitly recorded in `Execution Evidence`.

### Deterministic Tests

Run focused tests after implementation:

```powershell
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py tests\test_conversation_progress_flow.py -q
venv\Scripts\python.exe -m pytest tests\test_consolidator_facts_rag2.py tests\test_user_memory_units_rag_flow.py -q
```

Deterministic tests should prove:

- producer-owned structured fields are present when emitted;
- missing or unreadable producer fields are treated as missing;
- persistence/projection shape remains machine-readable;
- no deterministic test relies on keyword scanning to classify natural-language temporal leakage.

Deterministic tests are not sufficient for sign-off. They only prove plumbing and validation. Live LLM behavior is the acceptance gate for this plan.

### Live LLM Tests

Run one at a time. The required final set is determined by `Prompt Validation Matrix`: every edited producer prompt must have its named test run and inspected. Consumer-side RAG finalizer and cognition tests are not part of this plan.

Minimum required commands for prompts already expected to change:

```powershell
venv\Scripts\python.exe -m pytest tests\test_temporal_relative_terms_live_llm.py::test_live_recorder_contract_absolute_or_omit_episode_state -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_live_llm.py::test_live_extractor_anchors_tomorrow_commitment_due_at -m live_llm -q -s
```

Additional required commands when those prompts are edited:

```powershell
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_live_llm.py::test_live_extractor_marks_unresolved_relative_commitment_inactive -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_live_llm.py::test_live_rewrite_preserves_due_at_and_does_not_reintroduce_relative_due_text -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_live_llm.py::test_live_stability_respects_temporal_resolution_and_due_state -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_live_llm.py::test_live_merge_judge_respects_structured_temporal_metadata -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_temporal_relative_terms_live_llm.py::test_live_facts_harvester_contract_relative_due_terms -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_temporal_relative_terms_live_llm.py::test_live_facts_evaluator_contract_relative_due_terms -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_temporal_relative_terms_live_llm.py::test_live_conversation_evidence_contract_historical_quote_not_active_promise -m live_llm -q -s
```

Post-fix expected behavior:

- Recorder emits absolute-or-omitted operational state for time-bearing items.
- Memory/promise writer emits resolved due metadata or inactive/unresolved producer-owned records.
- Facts harvester and conversation-evidence producer prompts do not emit unresolved relative timing as active current/future facts.
- No consumer prompt is changed to repair or filter temporal leakage.

## Acceptance Criteria

This plan is complete when:

- No dedicated temporal-grounding module or deterministic natural-language parser exists.
- Prompt edits are integrated as full-prompt rewrites with coherent stage flow, not appended as isolated caveats, pasted common blocks, or copied plan text.
- Producer-side LLM contracts produce structured temporal fields, mark unresolved time-bearing facts inactive, or omit unsafe operational facts before persistence/projection.
- RAG finalizer, RAG evaluator, cognition L3, dialog, and Recall do not receive new temporal safeguards.
- Deterministic tests pass for producer output shape and persistence/projection plumbing.
- Every edited prompt's real LLM test listed in `Prompt Validation Matrix` passes individually and traces show no unsafe active promotion of unresolved relative temporal terms.
- No new response-path LLM call was added.
- No untouched stage or unrelated file was modified without a recorded necessity.

## Risks

| Risk                                               | Mitigation                                                                              | Verification                       |
| -------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------- |
| Producer still leaks relative wording              | No consumer repair is added; revise the owning producer prompt and rerun its live test. | Producer live LLM tests.           |
| Deterministic code quietly becomes semantic parser | Explicit forbidden rules and greps for central parser names.                            | Static grep plus review.           |
| Prompt caveats get ignored                         | Rewrite entire affected prompts as ordered stage contracts instead of appending warnings or pasted blocks. | Prompt review plus live LLM tests. |
| Live LLM remains noncompliant                      | Do not sign off; revise owning producer prompt and rerun the specific live test.        | Stage 4 gate.                      |
| Valid history is lost                              | Raw history remains quote/proof; only producer summaries avoid active promotion.        | Conversation evidence tests.       |
| Old leaked rows remain retrievable                 | Accepted residual risk under producer-only boundary; no migration or consumer repair.   | Execution evidence notes scope.    |

## Execution Evidence

Pre-implementation diagnostic evidence from 2026-05-08:

- `venv\Scripts\python.exe -m py_compile tests\test_temporal_relative_terms_live_llm.py`: passed.
- `test_live_recorder_diagnostic_relative_episode_state`: passed structurally; trace `test_artifacts/llm_traces/temporal_relative_terms_live_llm__recorder_episode_state_relative_terms.json`; diagnostic `appears_temporally_unsafe=True` with hits `今晚`, `明天`, `下周`, `一会儿`.
- Earlier drafting also produced downstream RAG finalizer and L3 traces showing propagation risk. Those traces are retained only as incident context, not implementation gates; active tests for this producer-side plan no longer import or call those modules.

Implementation evidence from 2026-05-08:

- Edited `_RECORDER_PROMPT` in `src/kazusa_ai_chatbot/conversation_progress/recorder.py` as a full Chinese stage contract. The prompt now treats `conversation_episode_state` as active operational state, defaults stale time-bearing open loops to omission, and uses direct `# 输入格式` / `# 输出格式` JSON sections matching the style of `_COGNITION_SUBCONSCIOUS_PROMPT`.
- Recorder deterministic plumbing change was limited to carrying existing structured prior timestamps (`created_at`, `updated_at`, `expires_at`) into the prior-state payload; no prose parsing, keyword repair, or semantic validation was added.
- Edited `_EXTRACTOR_PROMPT` in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py` as a full Chinese extractor contract. It now asks the producer to anchor due dates when resolvable and omit unsafe unresolved active commitments.
- Inspected `_REWRITE_PROMPT`, `_STABILITY_PROMPT`, and `_MERGE_JUDGE_PROMPT` in `persona_supervisor2_consolidator_memory_units.py`; no edit was required because this change did not alter their structured merge/stability contracts.
- Edited `_FACTS_HARVESTER_PROMPT` and `_FACT_HARVESTER_EVALUATOR_PROMPT` in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py` as full Chinese producer/evaluator contracts. Future promises must carry concrete `due_time` when time-bearing, and evaluator feedback rejects unresolved relative-time promise candidates.
- Inspected `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`; `_SELECTOR_PROMPT` only selects a worker and does not promote historical text into active summaries, so it was left unchanged and no conversation-evidence live gate was required.
- Prompt render check passed for `recorder`, `memory_extractor`, `facts_harvester`, and `facts_evaluator`; recorder no longer uses the extra `human payload 是以下 JSON` line.
- Static boundary checks passed: `rg "temporal_grounding|ground_temporal_text|sanitize_operational_text" src tests` returned no matches; `git diff` for `recall_agent.py`, `user_memory_evidence_agent.py`, `persona_supervisor2_rag_prompt_views.py`, `persona_supervisor2_cognition_l3.py`, and `persona_supervisor2_rag_evaluator.py` was empty.
- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py tests\test_consolidator_facts_rag2.py tests\test_conversation_progress_flow.py tests\test_conversation_progress_recorder.py tests\test_user_memory_units_live_llm.py tests\test_user_memory_units_rag_flow.py tests\test_temporal_relative_terms_live_llm.py`: passed.
- `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py tests\test_conversation_progress_flow.py -q`: 15 passed.
- `venv\Scripts\python.exe -m pytest tests\test_consolidator_facts_rag2.py tests\test_user_memory_units_rag_flow.py -q`: 15 passed.
- Live recorder gate passed: `tests\test_temporal_relative_terms_live_llm.py::test_live_recorder_contract_absolute_or_omit_episode_state`; trace `test_artifacts/llm_traces/temporal_relative_terms_live_llm__recorder_contract_absolute_or_omit_episode_state__20260508T115009263315Z.json`; diagnostic had no relative hits and the stale old exam/game loops were omitted from `open_loops`.
- Live memory extractor resolved-date gate passed: `tests\test_user_memory_units_live_llm.py::test_live_extractor_anchors_tomorrow_commitment_due_at`; trace `test_artifacts/llm_traces/user_memory_units_live_llm__extractor_anchors_tomorrow_commitment_due_at__20260508T115026957409Z.json`; active commitment used semantic date `2026-05-07` and persisted `due_at=2026-05-06T12:00:00+00:00`.
- Live memory extractor unresolved-date gate passed: `tests\test_user_memory_units_live_llm.py::test_live_extractor_marks_unresolved_relative_commitment_inactive`; trace `test_artifacts/llm_traces/user_memory_units_live_llm__extractor_marks_unresolved_relative_commitment_inactive.json`; LLM output was `{"memory_units": []}`.
- Live facts harvester gate passed: `tests\test_temporal_relative_terms_live_llm.py::test_live_facts_harvester_contract_relative_due_terms`; trace `test_artifacts/llm_traces/temporal_relative_terms_live_llm__facts_harvester_contract_relative_due_terms.json`; promise action had no relative-time hit and `due_time` was `2026-05-09 00:00`.
- Live facts evaluator gate passed: `tests\test_temporal_relative_terms_live_llm.py::test_live_facts_evaluator_contract_relative_due_terms`; trace `test_artifacts/llm_traces/temporal_relative_terms_live_llm__facts_evaluator_contract_relative_due_terms.json`; evaluator returned `should_stop=False` with feedback for the unresolved relative-time candidate.
- No new response-path LLM call, central temporal module, deterministic natural-language parser, consumer-side temporal safeguard, or topic suppression behavior was added.

Reviewer-feedback refinement from 2026-05-09:

- Replaced the recorder prompt's standalone explicit relative-term banlist with field-level guidance and redline examples. The revised wording explains that free-text operational fields must be self-contained when read days later, should remove date-deictic prefixes when time is only context, and must use `YYYY-MM-DD` / `YYYY-MM-DD HH:MM` when the date affects a commitment.
- Added explicit recorder examples for `current_thread` and `avoid_reopening`, because live testing showed the local model would otherwise still preserve `今天` or `下周二` in active operational fields.
- Prompt render check passed and confirmed the old phrase `以下词或结构不得出现在任何自由文本输出中` is absent.
- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\conversation_progress\recorder.py tests\test_temporal_relative_terms_live_llm.py`: passed.
- `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py tests\test_conversation_progress_flow.py -q`: 15 passed.
- Live recorder gate passed after refinement: `tests\test_temporal_relative_terms_live_llm.py::test_live_recorder_contract_absolute_or_omit_episode_state`; trace `test_artifacts/llm_traces/temporal_relative_terms_live_llm__recorder_contract_absolute_or_omit_episode_state__20260508T133034315771Z.json`; diagnostic had no relative hits, stale time-bearing items were omitted or compressed to no-date operational labels, and no deterministic repair was added.
