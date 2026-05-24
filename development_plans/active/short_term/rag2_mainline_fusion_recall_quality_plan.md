# rag2 mainline fusion recall quality plan

## Summary

- Goal: Fuse the useful current-branch private-memory recall behavior into the
  cleaner mainline RAG2 behavior, then fix shared RAG2 quality gaps without
  choosing one branch wholesale.
- Plan class: large
- Status: in_progress
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `no-prepost-user-input`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: bigbang for RAG2 prompt-facing language and
  evidence-format wording; compatible for existing `rag_result` top-level keys
  and supervisor trace; no database, index, cache, routing, graph, or model
  migration.
- Highest-risk areas: losing the current branch's private-memory recall gain,
  regressing mainline's cleaner group-chat synthesis, overfitting to the 20
  experiment cases, adding deterministic user-keyword matching, degrading
  retrieval through English translation, leaking raw ids or raw UTC timestamps,
  overvaluing source provenance instead of cognition usability, and changing
  cognition/dialog ownership.
- Acceptance criteria: the fused implementation preserves mainline output
  cleanliness, preserves the current branch's private user-memory recall win,
  promotes direct candidate evidence when it answers the requested slot,
  produces Chinese prompt-facing RAG evidence consistently, evaluates quality
  from the cognition consumer's point of view, passes focused deterministic
  checks, and passes `debug-llm` review on the fixed 20-case branch-vs-mainline
  suite plus a fresh holdout set.

## Context

The branch-vs-mainline RAG2 review used 20 fixed real database-backed inputs:

```text
experiments/rag2_branch_quality/reports/branch_vs_mainline_quality_review_20260524.md
```

Raw artifacts:

```text
test_artifacts/rag2_branch_quality/current_branch_20260524/
test_artifacts/rag2_branch_quality/mainline_20260524/
```

Observed result:

| Area | Current branch | Mainline | Fusion decision |
|---|---|---|---|
| Private durable user memory | Better in specific cases, especially the user's cat memory and some continuity projection | Can miss user-specific facts and retrieve generic guidance memory | Preserve the current-branch private-memory recall behavior |
| Group-chat synthesis | Usable but sometimes noisier | Cleaner chronology, speaker attribution, and Chinese output | Keep mainline synthesis as the baseline |
| Negative-result answers | Often trace-like | More readable and directly consumable | Keep mainline-style negative-result formatting |
| Candidate promotion | Fails when exact answer remains an unconfirmed candidate | Same failure | Fix generically in evaluator/finalizer contract |
| Technical memory recall | Misses 4090/NVFP4 and Gemma 4 quantization | Same failure | Treat as shared coverage gap |
| Media recall | Misses metal embedded nut product image | Same failure | Treat as shared media-search gap |
| Output language | Mixed English and Chinese | Mostly Chinese | Make prompt-facing RAG output Chinese-first |

The earlier production RAG2 cognition-ready evidence work improved the
evidence contract, but the new experiment shows that branch-level changes must
be fused carefully. The current branch is not a general replacement for
mainline. The value is narrower: better scoped private-memory recall and
continuity projection. Mainline remains better for final evidence readability
and group-chat attribution.

The user has explicitly changed the language decision: previous discussion
favored English for internal RAG wording, but this plan supersedes that
decision. Prompt-facing RAG language must be Chinese to avoid translation drift
between Chinese chat data, retrieval queries, evidence summaries, and cognition
consumption.

The user has also clarified the evaluation basis: source/provenance returned by
RAG is useful for multi-turn RAG search, gap analysis, trace inspection, and
debugging, but it must not be treated as a primary quality metric for the
cognition-facing handoff. The quality question starts from the consumer:
whether cognition receives the facts, uncertainty, and concise context needed
to reason and generate. Source detail is supporting evidence only.

Relevant architecture documents:

- `src/kazusa_ai_chatbot/rag/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `development_plans/archive/completed/short_term/rag2_cognition_ready_evidence_plan.md`
- `development_plans/reference/evidence/rag2_recall_quality_experiment_plan.md`

## Mandatory Skills

- `development-plan`: load before approving, executing, reviewing, updating, or
  signing off this plan.
- `local-llm-architecture`: load before changing RAG prompts, evaluator
  behavior, finalizer behavior, evidence projection, or cognition handoff.
- `debug-llm`: load before running live RAG2 comparisons, inspecting LLM output,
  or writing human-readable quality reports.
- `no-prepost-user-input`: load before adding any matching, filtering,
  promotion, or gating logic that could interpret user content.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files that contain Chinese prompt
  strings or Chinese test fixtures.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before continuing
  implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- Do not execute this plan while `Status` is `draft`. Implementation requires
  user approval and status `approved` or `in_progress`.
- Do not choose either branch wholesale. Use mainline behavior as the synthesis
  and formatting baseline, and port only the current-branch behavior that is
  proven to improve private durable user-memory recall.
- Do not import from `experiments.*`, `experiments/`, or `test_artifacts/` in
  production code or production tests. Experiment files are evidence and design
  references only.
- Do not add a new response-path LLM call. Use existing RAG2 stages and keep
  the current loop budget unless this plan is updated and approved.
- Do not change MongoDB databases, collections, indexes, vector indexes, Cache2
  schema, model routes, or adapter behavior.
- Do not change cognition, dialog, conversation progress, conversation graph,
  DAG, RAG3 routing, or consolidation behavior in this plan.
- RAG returns evidence. Cognition decides stance, boundaries, character
  judgment, and response goals. Dialog owns final visible wording.
- Evaluate RAG output from the cognition consumer's point of view. Do not score
  an output higher merely because it includes source names, source ids, source
  collections, or provenance text. Score it higher only when the output gives
  cognition a more accurate, complete, compact, uncertainty-aware factual basis.
- The initializer remains semantic-only. It must not generate backend query
  parameters, MongoDB filters, index names, cache policy, or route mechanics.
- Stable code-level route identifiers must remain ASCII/English when they are
  deterministic contracts, such as `Memory-evidence:` or
  `Conversation-evidence:`. The semantic task text after the route token, all
  LLM-facing prompt instructions, generated search/refined queries, worker
  summaries, evaluator summaries, finalizer answers, and cognition-facing
  evidence text must be Chinese-first.
- Do not translate Chinese user input or Chinese evidence into English as an
  intermediate retrieval language. Preserve Chinese anchors and Chinese
  phrasing for search quality unless the source evidence itself is English.
- Do not add deterministic keyword anchors derived from user input. Deterministic
  code may validate stable non-user contracts such as route prefixes, enum
  values, schema keys, and source-type labels.
- Candidate promotion must be semantic and evidence-based. If a retrieved
  candidate directly answers the slot, the evaluator may promote it to
  confirmed evidence. Deterministic code may validate and sanitize the promoted
  shape, but must not decide semantic relevance from user-keyword matching.
- Prompt-facing evidence must not expose raw adapter wire syntax such as
  `[CQ:...]`, raw source rows, raw storage ids, raw global user ids, embeddings,
  binary/base64 data, raw attachment URLs, raw UTC timestamps, or microsecond
  timestamps.
- Source/provenance detail belongs in trace/debug fields or short supporting
  evidence, not in the primary answer criteria. Public RAG text may include a
  short speaker/time cue only when it helps cognition understand who said what
  or when something happened.
- Prompt-facing timestamps must use configured local wall-clock time with
  second precision at most.
- Image evidence must use prompt-safe image descriptions or escaped
  `<image>...</image>` blocks where the current production projection requires
  that shape.

## Must Do

- Read the branch-vs-mainline review and raw artifacts before implementation.
- Diff the current branch against mainline and isolate the exact production
  changes that plausibly caused the private-memory recall win. Record the
  selected changes and rejected changes in execution evidence.
- Preserve mainline-style answer synthesis and group-chat formatting unless a
  targeted change is required by this plan.
- Port or recreate only the current-branch private-memory recall improvement
  that helps scoped user continuity, user-memory-unit retrieval, or memory
  projection.
- Convert RAG2 prompt-facing language to Chinese:
  - initializer prompt instructions and examples;
  - worker prompts where they produce search/refined queries or summaries;
  - evaluator and continuation prompt instructions;
  - finalizer prompt instructions;
  - public evidence labels and public answer wording.
- Keep deterministic route tokens stable. Localize the semantic task text after
  the route token and all prompt-facing language.
- Replace English evidence labels such as `Conclusion`, `Evidence summary`, and
  `Uncertainty` in prompt-facing public evidence with Chinese cognition-facing
  labels: `结论`, `上下文`, and `不确定性`.
- Add a generic candidate-promotion path for cases where candidate evidence
  directly answers the current slot but the helper result is marked unresolved.
  The `耗材厂` case must be covered by this generic behavior without
  hardcoding its terms.
- Preserve useful negative-result behavior: when no direct evidence is found,
  report the no-evidence conclusion and nearby commitments/facts separately.
- Preserve or improve source sanitation: no raw message ids, raw global user
  ids, raw CQ syntax, or raw UTC timestamps in `rag_result.answer`,
  `memory_evidence`, `recall_evidence`, or `conversation_evidence`.
- Update the experiment quality matrix so source/provenance presence is not a
  scored dimension. The scored dimensions are cognition usability, coverage,
  factual accuracy, attribution where needed for meaning, uncertainty handling,
  compactness, and prompt safety.
- Update `src/kazusa_ai_chatbot/rag/README.md` and, if the cognition handoff
  wording changes, `src/kazusa_ai_chatbot/nodes/README.md`.
- Re-run the fixed 20-case branch-vs-mainline suite and a fresh 20-case holdout
  through real RAG2 after implementation. Write a `debug-llm` human-readable
  review that shows inputs and outputs.
- During implementation, run focused real LLM branch-vs-mainline probes when a
  targeted change is ready for feedback. Run each live case individually,
  inspect the output before continuing, and use the result to guide the next
  narrow edit. Do not batch-run live probes for a simple pass/fail signal.

## Deferred

- RAG3 router/interpreter work.
- Graph RAG, DAG, or conversation graph work.
- New databases, collections, indexes, vector-search changes, or memory schema
  migrations.
- Conversation-progress redesign.
- Dialog style or persona voice changes outside RAG evidence wording.
- New LLM stages, extra response-path calls, fallback retries, or compatibility
  shims not named in this plan.
- Deterministic semantic matching over user input.

## Cutover Policy

Overall strategy: bigbang for prompt-facing RAG2 wording and prompt language;
compatible for `rag_result` keys and trace fields.

| Area | Policy | Instruction |
|---|---|---|
| RAG2 prompt-facing language | bigbang | Convert RAG2 LLM-facing instructions, examples, summaries, refined queries, and public evidence labels to Chinese-first wording. Do not keep parallel English prompt variants. |
| Code-level route tokens | compatible | Keep existing route-prefix contracts. The semantic task text after route tokens must be Chinese-first. |
| `rag_result` top-level keys | compatible | Preserve existing public keys and field types for cognition and consolidation consumers. |
| Supervisor trace | compatible | Preserve trace/debug fields for inspection. Raw ids may remain trace-only. |
| Retrieval architecture | compatible | Use existing RAG2 initializer, dispatcher, specialist agents, evaluator, finalizer, and projection boundaries. |
| Experiment code | bigbang | Do not import experiment code. After production verification, remove obsolete experiment code only if the user approves cleanup scope. |

## Cutover Policy Enforcement

- The execution agent must follow the selected policy for each area.
- If an area is `bigbang`, rewrite the old prompt-facing behavior instead of
  preserving parallel variants.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  in this plan.
- Any change to cutover policy requires user approval before implementation.

## Target State

RAG2 keeps the existing architectural flow:

```text
initializer -> dispatcher -> specialist agent -> evaluator -> finalizer
  -> project_known_facts -> state["rag_result"] -> cognition
```

The fused target behavior is:

- Mainline remains the baseline for readable synthesis, group attribution,
  local chronology, and cognition-facing output shape.
- Current-branch private-memory recall improvements are preserved for scoped
  user continuity and user profile facts.
- RAG2 prompt-facing evidence is Chinese-first from retrieval planning through
  cognition-facing projection.
- Candidate evidence that directly answers a slot can become confirmed
  evidence through the evaluator contract.
- Nearby evidence remains available as nearby/unconfirmed evidence instead of
  being mixed into direct answers.
- Public RAG evidence is prompt-safe and cognition-ready:

```text
结论：直接事实答案，或明确说明没有确认事实。
上下文：
- 只保留 cognition 需要理解事实的说话人、时间或场景；不要堆叠来源名。
不确定性：剩余不确定性、冲突、或“无”。
```

## Design Decisions

- Use mainline as the behavioral baseline for final answer readability and
  group-chat synthesis.
- Port only proven current-branch private-memory recall improvements.
- Supersede the earlier English-internal-RAG decision. Chinese is the default
  LLM-facing RAG language because the source chat data, user input, retrieved
  evidence, and cognition consumer are primarily Chinese.
- Keep code identifiers stable where they are deterministic contracts. This
  avoids a broad dispatcher migration while still removing English translation
  from semantic retrieval content.
- Fix candidate promotion at the evaluator/finalizer contract level, not with
  deterministic user-keyword matching.
- Keep no-evidence cases honest. Do not convert nearby evidence into direct
  facts unless the evaluator can semantically justify the promotion.

## Contracts And Data Shapes

The public `rag_result` keys remain compatible:

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
    "supervisor_trace": {
        "loop_count": int,
        "unknown_slots": list[str],
        "dispatched": list[dict],
    },
}
```

Prompt-facing evidence text inside those fields must use Chinese labels and
Chinese summaries. Trace payloads may keep internal identifiers and raw refs
needed for debugging, but they must not be copied into public evidence text.
The cognition-facing fields must not reward or emphasize source provenance for
its own sake. Include source-like details only when they are part of the fact's
meaning, such as speaker attribution in group chat, event time, or whether
evidence is confirmed versus nearby/unconfirmed.

Candidate-promotion records, if represented explicitly, must be trace-visible
and public-evidence-safe:

```python
{
    "resolved": True,
    "promotion_source": "candidate_evidence",
    "promotion_reason": "候选证据直接回答当前槽位。",
    "public_summary": "结论：...\n上下文：...\n不确定性：...",
}
```

This shape is illustrative. Implementation must reuse the existing production
fact-row and projection contracts where possible instead of adding unnecessary
new public fields.

## LLM Call And Context Budget

- Do not add a new RAG response-path LLM call.
- Preserve the existing RAG2 loop cap and continuation caps unless the user
  approves a separate budget change.
- Chinese prompt conversion must not increase prompt length materially. Replace
  English instructions in place rather than appending Chinese duplicates.
- Candidate promotion must run inside existing evaluator/finalizer calls or
  existing deterministic validation.

## Change Surface

Expected production inspection and change surface:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/recall_agent.py`
- `src/kazusa_ai_chatbot/rag/prompt_projection.py`
- `src/kazusa_ai_chatbot/rag/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md` only if handoff wording changes.

Expected test and experiment surface:

- Focused production tests under `tests/` for prompt rendering, projection,
  candidate promotion, source sanitation, and private-memory recall behavior.
- Experiment-review matrix updates so the human-readable comparison evaluates
  cognition usability rather than provenance richness.
- Existing experiment runner under `experiments/rag2_branch_quality/` as
  comparison harness only.
- New `debug-llm` review artifact under an experiment report or
  `test_artifacts/` path after real LLM validation.

## Overdesign Guardrail

- Do not add graph search, RAG3 routing, a new memory index, a new media index,
  a new finalizer stage, or a second answer synthesizer.
- Do not make the initializer understand backend storage or direct database
  filters.
- Do not add compatibility prompt variants in both English and Chinese.
- Do not solve missed technical-memory cases by hardcoding technology terms.
- Do not solve the `耗材厂` case by hardcoding brand names or that phrase.
- Do not add new public `rag_result` fields unless existing fields cannot
  safely express the required evidence.

## Agent Autonomy Boundaries

- The execution agent may inspect branch diffs and choose the smallest code
  changes that preserve the current-branch private-memory win.
- The execution agent may rewrite RAG2 prompts into Chinese within the approved
  prompt surfaces.
- The execution agent may add focused tests and experiment report artifacts.
- The execution agent must not change the cutover policy, public state shape,
  database schema, model routing, graph architecture, or cognition/dialog
  prompts without user approval.
- If implementation reveals that preserving the private-memory win requires a
  broader memory schema or retrieval-index change, stop and report the blocker
  instead of expanding scope.

## Implementation Order

1. Parent rereads this plan, `rag/README.md`, `nodes/README.md`, and the
   branch-vs-mainline report.
2. Parent inspects the current branch versus mainline diff and records which
   current-branch changes are selected for fusion and which are rejected.
3. Parent adds focused tests for:
   - Chinese prompt-facing evidence labels;
   - no raw ids, raw CQ syntax, or raw UTC timestamps in public evidence;
   - candidate evidence promotion from an exact-answer candidate;
   - preservation of private user-memory recall behavior.
4. Parent runs the focused tests before implementation and records baseline
   failures or current behavior.
5. Parent starts one production-code subagent with this approved plan, the
   focused test contract, and the exact production change surface.
6. Production-code subagent implements the selected private-memory fusion,
   Chinese prompt conversion, candidate-promotion contract, and public evidence
   sanitation changes.
7. Parent updates integration tests or harness checks for `rag_result`
   compatibility and cognition-facing projection.
8. Parent runs focused tests, integration tests, static checks, and py_compile
   for touched Python files.
9. Parent runs focused real LLM branch-vs-mainline probes for the changed
   behavior as implementation feedback. Preferred probes include the cat/profile
   memory case, the `耗材厂` candidate-promotion case, and one mainline-clean
   group attribution case.
10. Parent runs the fixed 20-case real LLM suite and writes a `debug-llm`
   branch-quality report showing inputs and outputs.
11. Parent selects a fresh 20-case holdout from real DB-backed data, runs it
    one live LLM case at a time, and writes a second `debug-llm` report.
12. Parent starts an independent code-review subagent after verification
    passes.
13. Parent remediates review findings within this plan's change surface,
    reruns affected verification, records execution evidence, and updates plan
    lifecycle only after user-approved completion.

## Execution Model

- Execution is parent-led and uses native subagents.
- Parent owns test contract, integration verification, live LLM test execution,
  report writing, plan lifecycle updates, and final sign-off.
- Production-code subagent owns only production source changes listed in
  `Change Surface`.
- Independent code-review subagent owns review only and must not implement
  fixes.
- If native subagents are unavailable during execution, stop and report the
  blocker unless the user explicitly approves fallback execution.

## Progress Checklist

- [x] Stage 1 - branch/mainline diff audit recorded.
- [ ] Stage 2 - focused tests added and baseline behavior recorded.
- [x] Stage 3 - production-code subagent completes approved fusion changes.
- [x] Stage 4 - integration compatibility checks pass.
- [x] Stage 5 - focused branch-vs-mainline live LLM probes completed during
      implementation feedback.
- [x] Stage 6 - fixed 20-case `debug-llm` review completed.
- [ ] Stage 7 - fresh 20-case holdout `debug-llm` review completed.
- [ ] Stage 8 - independent code review completed and findings resolved.
- [ ] Stage 9 - execution evidence recorded and lifecycle status updated.

## Verification

Required deterministic verification:

```powershell
venv\Scripts\python.exe -m py_compile <touched Python files>
venv\Scripts\python.exe -m pytest <focused RAG2 tests> -q
```

Required live LLM verification:

- During implementation, run focused real LLM probes between the current branch
  and mainline when a targeted behavior changes. Use the same real input for
  both sides, inspect output one case at a time, and record a short
  `debug-llm` note or execution-evidence summary.
- Run the 20 fixed cases from
  `experiments/rag2_branch_quality/cases/rag2_branch_quality_cases_20260524.jsonl`
  one case at a time.
- Produce a human-readable `debug-llm` report showing each input and output.
- Run a fresh 20-case holdout set selected from real database-backed inputs.
- Produce a second human-readable `debug-llm` report showing each input and
  output.
- In both reports, do not count source/provenance richness as a positive quality
  metric. Treat source details as raw/supporting evidence and judge the primary
  output by cognition usability.

Quality gates:

- The cat/profile case must preserve the current branch's successful recall of
  the user's cat details.
- The group exact-phrase and Ab screenshot cases must preserve mainline's
  cleaner attribution and no message-id leakage.
- The `耗材厂` case must promote the candidate evidence into the answer through
  generic candidate-promotion behavior.
- Public RAG evidence must be Chinese-first and must not include raw ids, raw
  CQ syntax, raw UTC timestamps, or debug-only source rows.
- No new response-path LLM calls are introduced.
- The review conclusion must state whether cognition can consume the RAG output
  directly to form stance, judgment, and response goals without reading trace
  fields.

## Independent Plan Review

Before approval or execution, run an independent plan review focused on:

- whether the plan truly fuses branch strengths instead of picking a branch;
- whether Chinese prompt-facing language is specified without breaking route
  contracts;
- whether candidate promotion avoids deterministic user-keyword matching;
- whether quality scoring starts from the cognition consumer rather than source
  provenance richness;
- whether the change surface is small enough for production safety;
- whether verification proves both fixed 20-case behavior and fresh holdout
  behavior.

Record the review outcome in this plan before changing `Status` to `approved`
or `in_progress`.

## Independent Code Review

After planned implementation and verification pass, run an independent code
review over the full diff. The review must check:

- compliance with this plan and mandatory skills;
- prompt safety and CJK string safety;
- no imports from `experiments/` or `test_artifacts/`;
- no deterministic semantic matching over user input;
- no raw ids, raw CQ syntax, or raw UTC timestamps in public evidence;
- no cognition/dialog ownership drift;
- no evaluation claim that treats source/provenance richness as a primary RAG
  quality win;
- real LLM review artifacts match the claimed quality result.

Record findings, fixes, rerun commands, and residual risk in execution
evidence before sign-off.

## Acceptance Criteria

- Mainline-style group synthesis and answer cleanliness are preserved.
- Current-branch private durable user-memory recall improvement is preserved.
- Prompt-facing RAG language is Chinese-first across planning, retrieval
  summaries, evaluator/finalizer output, and cognition-facing evidence.
- Quality evaluation is cognition-consumer-first. Source/provenance detail is
  available for trace/debug and gap analysis, but is not a scored quality
  dimension for the RAG-to-cognition handoff.
- Candidate evidence that directly answers a slot is promoted generically.
- Existing `rag_result` top-level keys and field types remain compatible.
- No raw ids, raw CQ syntax, raw UTC timestamps, or debug source rows appear in
  public RAG evidence.
- Fixed 20-case real LLM review and fresh 20-case holdout review are completed
  with readable inputs and outputs.
- Independent code review is completed with findings resolved or explicitly
  accepted by the user.

## Execution Evidence

- 2026-05-24: User approved the plan and requested execution. Status changed
  to `in_progress`.
- 2026-05-24: User added an execution requirement to run focused real LLM
  branch-vs-mainline probes during implementation for realtime feedback where
  practical.
- 2026-05-24: Branch/mainline diff audit started. Selected for fusion:
  scoped `user_memory_units` recall that combines slot/current/original query,
  prompt-safe evidence formatting, trace-only source refs, and generic
  unresolved-candidate handling. Rejected as-is: English-first RAG2 prompt
  wording, English public evidence labels, source-heavy public evidence,
  strict deterministic coverage as the only promotion gate, and any new
  user-input keyword matching.
- 2026-05-24: Production-code subagent rewrote RAG2 prompt-bearing files into
  Chinese-first prompt-facing language while preserving stable JSON keys,
  route tokens, source text, URLs, filenames, and code/model labels. Follow-up
  local patches fixed the remaining continuation prompt headings, one recall
  fallback status string, and the RAG evidence documentation contract.
- 2026-05-24: Focused deterministic verification passed:
  `venv\Scripts\python.exe -m pytest tests/test_rag_evidence_formatting.py
  tests/test_rag_projection.py tests/test_rag_prompt_evidence_safety.py
  tests/test_rag_finalizer_time_context.py -q` returned `45 passed`.
  `py_compile` passed for touched production and focused test Python files.
  `rg` found no stale English RAG2 marker phrases such as `Use English`,
  `# Input Format`, `# Output Format`, `Return valid JSON only`,
  `This RAG run found`, `Checked sources`, or `Nearby but unconfirmed`.
- 2026-05-24: Focused real LLM branch-vs-mainline probes completed one case at
  a time for `rag2-branch-quality-005`, `rag2-branch-quality-017`, and
  `rag2-branch-quality-019`. Review artifact:
  `experiments/rag2_branch_quality/reports/fusion_current_vs_mainline_probe_20260524.md`.
  Raw current-branch artifacts:
  `test_artifacts/rag2_branch_quality/fusion_current_probe_20260524/`.
  Raw mainline artifacts:
  `test_artifacts/rag2_branch_quality/fusion_mainline_probe_20260524/`.
  Result: current branch clearly improves scoped user-memory recall in case
  005, answers case 017 with fewer loops than mainline, and preserves answer
  correctness for case 019 while showing a remaining over-broad attachment
  planning weakness.
- 2026-05-24: Independent prompt-surface review found three issues: public
  evidence redaction still used English `[source id omitted]`, several
  evaluator prompt example values were English placeholder prose, and the
  dispatcher/initializer input examples used invalid JSON prose values. Fixed
  all three by using Chinese public redaction labels, Chinese placeholder
  values, and valid JSON examples. Fresh verification after the fixes:
  `venv\Scripts\python.exe -m pytest tests/test_rag_evidence_formatting.py
  tests/test_rag_projection.py tests/test_rag_prompt_evidence_safety.py
  tests/test_rag_finalizer_time_context.py -q` returned `45 passed`;
  `py_compile` passed for the edited prompt/formatting files; `git diff
  --check` returned exit code 0.
- 2026-05-24: Parent reviewed touched LLM prompt surfaces again before fixed
  20-case validation. Additional prompt/contract fixes removed dispatcher
  internal-worker detail, made evaluator/finalizer source cues conditional,
  prevented readable message-id leakage, replaced placeholder initializer
  schema examples, and tightened raw-id/source-id handling in initializer,
  evaluator, and finalizer prompts.
- 2026-05-24: During live fixed-case validation, case 011 exposed initializer
  placeholder-slot copying and case 016/018 exposed raw UUID/source-id and
  invalid person-slot dependencies in prompt-facing initializer output. Fixed
  by prompt tightening plus structural normalization over stable slot grammar:
  raw source ids are sanitized from initializer slots and
  `speaker=person resolved in slot N` is dropped when slot N cannot resolve a
  person. This validation does not use deterministic keyword matching over
  user input.
- 2026-05-24: Deterministic verification after the additional fixes passed:
  `venv\Scripts\python.exe -m pytest tests/test_rag_evidence_formatting.py
  tests/test_rag_projection.py tests/test_rag_prompt_evidence_safety.py
  tests/test_rag_finalizer_time_context.py -q` returned `47 passed`;
  `venv\Scripts\python.exe -m pytest
  tests/test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_drops_invalid_person_slot_reference
  tests/test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_keeps_valid_person_slot_reference
  tests/test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_removes_source_ids
  tests/test_rag_prompt_evidence_safety.py tests/test_rag_evidence_formatting.py
  -q` returned `19 passed`; `py_compile` passed for edited prompt/source/test
  files.
- 2026-05-24: Fixed 20-case live RAG2 validation completed one case at a time
  against real DB-backed data. Human-readable `debug-llm` review artifact:
  `experiments/rag2_branch_quality/reports/fixed20_current_rag2_review_20260524.md`.
  Raw artifacts:
  `test_artifacts/rag2_branch_quality/fixed20_current_20260524/`. Structured
  raw summary:
  `test_artifacts/rag2_branch_quality/fixed20_current_20260524/_case_summary.json`.
  Result from cognition-consumer view: most private-memory and group-chat
  cases are usable, candidate promotion works for the `耗材厂`/X2D-style cases,
  and public message/source-id safety is improved. Remaining high-risk gaps:
  technical conversation recall (`Gemma 4` quantization), coined phrase recall
  (`星尘薄荷魔法`), media product-image recall (`金属预埋螺母`), and one official
  identity-memory lookup (`千纱生日和星座`).
- 2026-05-24: Final local verification for Stage 6 passed:
  `venv\Scripts\python.exe -m py_compile
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_initializer.py
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_evaluator.py
  src\kazusa_ai_chatbot\rag\evidence_formatting.py
  tests\test_persona_supervisor2_rag2_integration.py
  tests\test_rag_evidence_formatting.py
  tests\test_rag_prompt_evidence_safety.py` exited 0;
  `venv\Scripts\python.exe -m pytest tests\test_rag_evidence_formatting.py
  tests\test_rag_projection.py tests\test_rag_prompt_evidence_safety.py
  tests\test_rag_finalizer_time_context.py
  tests\test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_drops_invalid_person_slot_reference
  tests\test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_keeps_valid_person_slot_reference
  tests\test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_removes_source_ids
  -q` returned `50 passed`; `git diff --check` exited 0 with only CRLF
  normalization warnings.
- 2026-05-24: Failed fixed-run cases were compared against mainline using the
  same real inputs. Review artifact:
  `experiments/rag2_branch_quality/reports/failed_cases_current_vs_mainline_20260524.md`.
  Extracted evidence:
  `test_artifacts/rag2_branch_quality/fixed20_current_20260524/_failed_vs_mainline_summary.json`.
  Result: mainline clearly wins on `rag2-branch-quality-020` because it
  retrieves the durable birthday/zodiac memory in one loop; mainline is also
  cleaner and cheaper on `rag2-branch-quality-007`; both branches fail
  `rag2-branch-quality-004` and `rag2-branch-quality-018`, while current is
  safer than mainline on 018 because it no longer exposes a raw
  `global_user_id`. Next development priority is to restore mainline's durable
  character/world memory attribute recall while preserving current-branch
  private user-memory recall, then address exact/coined phrase load, media
  product-image recall, and technical conversation recall.
