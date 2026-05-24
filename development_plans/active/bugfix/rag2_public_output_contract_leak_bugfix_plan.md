# rag2 public output contract leak bugfix plan

## Summary

- Goal: remove RAG2 implementation/process wording from cognition-facing
  `rag_result` fields while preserving trace diagnostics and the existing RAG2
  retrieval architecture.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: bigbang for public output wording; compatible for
  `rag_result` top-level shape and `supervisor_trace`.
- Highest-risk areas: hiding useful negative evidence, over-filtering quoted
  user/source text, breaking cognition consumers that prioritize
  `rag_result.answer`, and leaving old prompt fixtures that normalize the leak.
- Acceptance criteria: public RAG2 output no longer contains generated phrases
  such as `本次 RAG`, `槽位`, `已检查来源`, capability prefixes, agent names, or
  backend field names; diagnostic routing details remain available only under
  trace/raw-result material; focused deterministic and live/readable RAG checks
  pass.

## Context

The audit record is:

```text
experiments/rag2_branch_quality/reports/rag2_llm_output_contract_audit_20260524.md
```

The confirmed defect is not model-specific. The public leak is encoded in the
RAG2 finalizer prompt and deterministic fallback:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
  instructs the finalizer to output `本次 RAG 没有需要检索的外部/内部事实。`
  when `known_facts` is empty.
- `_unresolved_finalizer_answer()` emits
  `本次 RAG 没有找到已确认事实。 已检查来源：...` when all facts are
  unresolved.
- `_unresolved_summary()` stores public-ish summaries containing full slot
  text such as `槽位：Conversation-evidence: ...` and backend field names such
  as `conversation_evidence`.
- The continuation assessor prompt permits `promotion_summary` wording like
  `候选证据直接回答了当前槽位...`; promoted summaries can become public
  `conversation_evidence` or `memory_evidence`.
- `project_known_facts()` sanitizes raw storage and adapter markers, but it
  does not reject semantic process leakage.

Downstream impact is high because cognition receives `rag_result` directly:

- L2 consciousness reads `_cognition_rag_result(state["rag_result"])`.
- L3 content anchors prioritize `rag_result.answer` when it directly answers
  the current input.
- L2d action initialization copies `rag_result.answer` into `rag_answer`.

RAG must return evidence. Cognition decides stance, boundaries, character
judgment, and response goals. This bugfix restores that ownership boundary.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing RAG prompts, evaluator
  summaries, finalizer behavior, projection shape, or cognition handoff.
- `debug-llm`: load before running live RAG2 checks, comparing LLM output, or
  writing human-readable review artifacts.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files that contain CJK prompt or
  test strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Preserve the RAG2 architecture:
  `initializer -> dispatcher -> helper agent -> evaluator -> finalizer ->
  projection -> cognition`.
- Do not change the RAG initializer, retrieval routing, search algorithms,
  worker selection, vector/BM25 behavior, Cache2 policy, database schema,
  model routing, conversation graph, conversation progress, cognition prompts,
  dialog prompts, scheduler, adapters, or persistence.
- Do not add a new response-path LLM call, retry loop, fallback prompt,
  compatibility shim, feature flag, alternate RAG path, or experiment import.
- Runtime prompt changes must follow the local-LLM prompt rule:
  stable role/policy/output contract stays in the system prompt; human payload
  contains only current-run input and retrieved evidence; prompts use plain
  domain language; prompts must not introduce development-process terms.
- Prompt-facing generated RAG wording must be Chinese-first. Keep source text,
  display names, URLs, filenames, code/model labels, and quoted content in
  their original language.
- Public `rag_result.answer`, public evidence summaries, generated conclusions,
  and generated uncertainty text must not expose generated implementation
  terms: `本次 RAG`, `RAG2`, `槽位`, `当前槽位`, `已检查来源`,
  `known_facts`, `unknown_slots`, `_agent`, capability prefixes, route names,
  or backend field names.
- Deterministic leak checks must target generated RAG control/evidence text
  only. They must not scan raw user input or reject quoted source-message text
  solely because a user discussed terms such as "RAG" or a model name.
- Diagnostic details such as slot strings, agent names, route mismatch,
  missing context, source refs, and unresolved raw payloads must remain in
  `known_facts`, raw helper results, logs, or `supervisor_trace`, not in the
  public evidence text consumed by cognition.
- Do not remove useful source evidence from `memory_evidence`,
  `conversation_evidence`, `recall_evidence`, or `external_evidence`; rewrite
  generated summaries so cognition sees facts and uncertainty instead of
  retrieval diagnostics.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Replace the empty-`known_facts` finalizer behavior with a cognition-safe
  empty answer path. Empty `known_facts` means there is no RAG evidence to add,
  not a fact sentence for cognition to repeat.
- Replace all-unresolved finalizer output with a cognition-safe negative
  evidence summary such as `未找到能确认当前问题的证据。`
- Keep nearby but unconfirmed candidates only when they add useful context, and
  format them without source-agent labels, route names, slot text, or backend
  field names.
- Rewrite unresolved evaluator summaries so `known_facts[*].summary` remains
  human-readable but does not dump slot text, route names, or storage key names.
- Tighten the continuation assessor prompt so `promotion_summary` must be a
  direct fact sentence suitable for public evidence, not a review statement
  about candidate adequacy or a slot.
- Add prompt-safety validation for generated public RAG text while preserving
  quoted source text and `supervisor_trace`.
- Update deterministic tests that currently assert the leaked phrases.
- Update cognition prompt test fixtures that hardcode leaked RAG wording.
- Update RAG documentation to state that public evidence must not name RAG,
  slots, routes, agents, backend keys, or source-check diagnostics.
- Run focused deterministic tests and a small readable real RAG/debug-LLM
  validation set against saved or live real cases.

## Deferred

- Do not improve retrieval accuracy, ranking, synthesis quality, RAG
  initializer routing, conversation search, memory search, recall search,
  Cache2, graph/DAG, or model selection in this plan.
- Do not change public `rag_result` top-level keys.
- Do not remove `supervisor_trace`.
- Do not add or migrate MongoDB collections, indexes, or cache documents.
- Do not modify experiment code except to reference the audit artifact.
- Do not change downstream cognition behavior to compensate for bad RAG text;
  RAG must provide prompt-safe evidence before cognition consumes it.

## Cutover Policy

Overall strategy: bigbang for public RAG wording; compatible for state shape.

| Area | Policy | Instruction |
|---|---|---|
| Empty `known_facts` public answer | bigbang | Return an empty `final_answer` instead of `本次 RAG...`; do not preserve the old phrase. |
| All-unresolved public answer | bigbang | Replace `本次 RAG 没有找到已确认事实。 已检查来源：...` with cognition-safe negative evidence wording. |
| Evaluator unresolved summaries | bigbang | Replace slot-dump summaries with public-safe unresolved summaries; keep slot and raw diagnostics in existing structured fields. |
| Continuation promotion summaries | bigbang | Require promoted summaries to be direct fact sentences; reject process wording in generated summaries. |
| Public `rag_result` shape | compatible | Preserve existing top-level keys and evidence collection types. |
| `supervisor_trace` diagnostics | compatible | Preserve slot strings, agent names, source refs, continuation metadata, and unresolved diagnostics for debug use. |
| Tests | bigbang | Rewrite tests and fixtures that expect leaked public phrases. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- For bigbang areas, rewrite old behavior directly instead of adding
  compatibility branches.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

When no retrieval is needed:

```python
{
    "answer": "",
    "known_facts": [],
    "unknown_slots": [],
    "loop_count": 0,
}
```

When retrieval ran but no fact was confirmed:

```text
rag_result.answer = "未找到能确认当前问题的证据。"
```

When nearby candidates are useful but unconfirmed:

```text
rag_result.answer = "未找到能确认当前问题的证据；只找到附近但未确认的候选：<short candidate facts>。"
```

Public evidence blocks contain direct facts:

```text
结论：小钳子回复 Kuroneeeko 说 "a1比0.2大好大一圈"，随后建议 "所以买X2D"。
上下文：
- 命中消息：...
不确定性：无
```

Public evidence blocks do not contain process wording:

```text
结论：候选证据直接回答了当前槽位...
```

Diagnostic details remain available under `supervisor_trace` and raw
`known_facts` fields for debugging and future multi-turn RAG, but cognition
does not receive them as primary facts.

## Design Decisions

- The fix belongs in RAG2 output production, not cognition. Downstream agents
  should not learn to ignore leaked RAG diagnostics.
- Empty `known_facts` should produce an empty public answer because there is no
  retrieval evidence to add. This avoids teaching cognition that a lack of RAG
  work is a fact about the user question.
- All-unresolved retrieval can still produce a concise negative evidence
  statement because cognition benefits from knowing that retrieval was tried
  and did not confirm the requested fact.
- `supervisor_trace` remains the diagnostic channel for slots, agent names,
  source refs, and unresolved causes.
- Prompt-safety checks should validate generated public text at generation
  boundaries. They must not reject raw quoted source content that happens to
  discuss RAG, slots, route labels, or model names.
- Public generated summaries should preserve uncertainty, not hide it. The
  target is "no implementation language", not "always positive evidence".

## Contracts And Data Shapes

Public `rag_result` top-level shape remains:

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

Generated public text fields must obey:

```text
Allowed:
- direct facts;
- concise no-evidence statements;
- concise uncertainty;
- source speaker/time/content where useful;
- source quotes in original language.

Forbidden:
- generated references to RAG, RAG2, slots, current slot, known_facts,
  unknown_slots, agent names, capability prefixes, route names, backend field
  names, source-check diagnostics, raw storage IDs, raw UTC, adapter wire
  syntax, and binary/raw attachment payloads.
```

Trace/raw fields may continue to contain implementation diagnostics:

```python
rag_result["supervisor_trace"]["dispatched"][*]["slot"]
rag_result["supervisor_trace"]["dispatched"][*]["agent"]
known_facts[*]["slot"]
known_facts[*]["agent"]
known_facts[*]["raw_result"]
```

## LLM Call And Context Budget

- Do not add new LLM calls.
- Empty `known_facts` should reduce cost by bypassing the finalizer LLM.
- Prompt changes must shorten or clarify existing prompts; do not add long
  examples or broad lookup tables.
- The finalizer and continuation assessor remain on existing RAG LLM routes.

## Change Surface

Expected production files:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
  - Rewrite empty/all-unresolved finalizer behavior.
  - Rewrite unresolved summaries.
  - Tighten finalizer and continuation prompts.
- `src/kazusa_ai_chatbot/rag/evidence_formatting.py`
  - Add generated-public-text leak validation helpers or extend existing
    validation without scanning quoted source text broadly.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
  - Apply generated-text validation at projection boundaries where needed.
  - Keep source refs and trace diagnostics unchanged.
- `src/kazusa_ai_chatbot/rag/README.md`
  - Document public evidence versus trace diagnostics.

Expected test files:

- `tests/test_rag_finalizer_time_context.py`
  - Replace tests asserting leaked phrases.
  - Add tests for empty `known_facts`, all-unresolved output, unresolved
    summaries, and continuation promotion wording.
- `tests/test_rag_projection.py`
  - Add projection tests proving public fields reject generated process leaks
    while `supervisor_trace` may retain diagnostic data.
- `tests/test_cognition_live_llm_prompt_contracts.py`
  - Replace leaked RAG fixture text with an empty or cognition-safe answer.

Allowed documentation or evidence files:

- `development_plans/active/bugfix/rag2_public_output_contract_leak_bugfix_plan.md`
- `development_plans/README.md`
- A debug-LLM review artifact under an existing `experiments/.../reports/`
  directory after live/readable validation.

## Overdesign Guardrail

- Do not solve retrieval quality, synthesis completeness, or search coverage
  in this plan.
- Do not introduce a new public evidence schema.
- Do not add a generic post-hoc scrubber that blindly removes terms from user
  quotes or source messages.
- Do not make cognition responsible for filtering RAG implementation details.
- Do not add compatibility for old leaked phrases.
- Do not add broad deterministic routing or keyword matching over user input.

## Agent Autonomy Boundaries

- The parent agent owns the focused test contract, verification, readable
  debug-LLM review, plan updates, and final sign-off.
- The production-code subagent owns production code changes only within the
  listed change surface.
- The review subagent owns independent code review only and must not implement
  fixes.
- The execution agent may add small private helper functions inside the listed
  production files when they reduce duplication and preserve the approved
  contract.
- The execution agent must stop and request plan approval before touching
  RAG initializer, search agents, cognition prompts, dialog, database schema,
  cache policy, model routing, adapters, or experiment code.

## Implementation Order

1. Parent establishes focused deterministic tests.
   - Update `tests/test_rag_finalizer_time_context.py` first.
   - Expected pre-implementation result: tests fail because existing output
     contains `本次 RAG`, `已检查来源`, or `槽位`.
2. Parent starts one production-code subagent with this approved plan,
   mandatory skills, focused failing tests, and the listed change surface.
3. Parent adds projection and cognition-fixture tests while the production
   subagent edits production code.
   - Update `tests/test_rag_projection.py`.
   - Update `tests/test_cognition_live_llm_prompt_contracts.py`.
4. Production-code subagent implements the RAG2 public-output cleanup.
   - Keep implementation inside evaluator, evidence formatting, projection,
     and RAG README.
5. Parent runs focused deterministic tests and records output.
6. Parent runs a small readable RAG/debug-LLM validation using real saved or
   live cases:
   - one empty/no-retrieval case;
   - one all-unresolved conversation case;
   - one all-unresolved memory case with nearby candidates;
   - one resolved promoted conversation case.
7. Parent writes a human-readable debug-LLM review artifact showing input,
   public `rag_result` output, and whether downstream-facing text is
   cognition-ready.
8. Parent starts one independent code-review subagent after verification
   passes.
9. Parent remediates approved review findings within this plan's change
   surface, reruns affected verification, and records results.

## Execution Model

- Execution requires parent-led native subagent execution.
- Do not execute this plan until the user approves it and the status is changed
  to `approved` or `in_progress`.
- Normal execution uses exactly two subagents:
  1. one production-code subagent after the parent establishes failing focused
     tests;
  2. one independent code-review subagent after implementation verification.
- If native subagent capability is unavailable, stop before execution and
  report the blocker. Fallback single-agent execution requires explicit user
  approval.

## Progress Checklist

- [ ] Stage 1 - deterministic finalizer/evaluator contract tests established
  - Covers: implementation order step 1.
  - Files: `tests/test_rag_finalizer_time_context.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests/test_rag_finalizer_time_context.py -q`.
  - Evidence: record expected pre-implementation failures in `Execution Evidence`.
  - Handoff: next agent starts Stage 2.
  - Sign-off: pending.

- [ ] Stage 2 - production public-output cleanup implemented
  - Covers: implementation order steps 2 and 4.
  - Files: evaluator, evidence formatting, projection, RAG README.
  - Verify: focused tests from Stage 1 pass.
  - Evidence: record changed files and focused test output in `Execution Evidence`.
  - Handoff: next agent starts Stage 3.
  - Sign-off: pending.

- [ ] Stage 3 - projection and cognition-fixture tests updated
  - Covers: implementation order step 3.
  - Files: `tests/test_rag_projection.py`,
    `tests/test_cognition_live_llm_prompt_contracts.py`.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_projection.py tests/test_cognition_live_llm_prompt_contracts.py -q`.
  - Evidence: record pass/fail output in `Execution Evidence`.
  - Handoff: next agent starts Stage 4.
  - Sign-off: pending.

- [ ] Stage 4 - real-output readable validation complete
  - Covers: implementation order steps 6 and 7.
  - Files: new debug-LLM report under `experiments/.../reports/`.
  - Verify: inspect the report against the audit's failure cases.
  - Evidence: link the report and summarize human-readable quality findings in
    `Execution Evidence`.
  - Handoff: next agent starts Stage 5.
  - Sign-off: pending.

- [ ] Stage 5 - independent code review complete
  - Covers: implementation order steps 8 and 9.
  - Files: full implementation diff.
  - Verify: review subagent reports no blocking findings, or all blocking
    findings are remediated and affected checks rerun.
  - Evidence: record review findings, fixes, rerun commands, and residual risk
    in `Execution Evidence`.
  - Handoff: lifecycle update after user sign-off.
  - Sign-off: pending.

## Verification

Required deterministic checks:

```powershell
venv\Scripts\python.exe -m pytest tests/test_rag_finalizer_time_context.py -q
venv\Scripts\python.exe -m pytest tests/test_rag_projection.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_live_llm_prompt_contracts.py -q
```

Required static checks:

```powershell
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/rag/evidence_formatting.py
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py
```

Required readable validation:

- Produce a debug-LLM review artifact with real RAG output for the four case
  categories listed in `Implementation Order`.
- The report must show the input, public `rag_result.answer`, public evidence
  fields, and a quality note for downstream cognition consumption.
- The report must not require the user to inspect raw JSON as the primary
  review surface.

Forbidden verification shortcut:

- Do not claim success from schema validity, command success, or lack of
  exceptions alone. Inspect the actual public output text.

## Independent Plan Review

Before moving this plan from `draft` to `approved`, run a focused plan review
against:

- the audit finding;
- RAG2 README public evidence contract;
- development-plan required sections;
- local-LLM architecture boundary;
- no-pre/post user-input guardrail for deterministic matching.

Record plan review findings in `Execution Evidence` before approval.

## Independent Code Review

After implementation verification passes, run one independent code-review
subagent. The review scope is:

- public RAG output contract compliance;
- prompt wording safety and minimality;
- no new LLM calls;
- no changes outside the approved change surface;
- no broad deterministic matching over user input or quoted source text;
- deterministic tests and readable debug-LLM evidence quality;
- preservation of `supervisor_trace` diagnostics.

The review subagent must not implement fixes. Parent handles remediation only
inside this plan's approved change surface.

## Acceptance Criteria

- Empty `known_facts` no longer produces `本次 RAG...` in `rag_result.answer`.
- All-unresolved RAG output no longer exposes `RAG`, `槽位`, `已检查来源`,
  capability prefixes, agent names, or backend field names in public fields.
- Resolved promoted evidence uses direct fact wording rather than
  `候选证据直接回答当前槽位`.
- Public generated RAG summaries and uncertainty text are Chinese-first,
  concise, and suitable for cognition consumption.
- `supervisor_trace` still contains diagnostic slots, agents, source refs, and
  continuation metadata.
- Focused deterministic tests pass.
- A readable debug-LLM review artifact shows real outputs before final sign-off.
- No production source imports from `experiments/` or `test_artifacts/`.

## Risks

- A strict semantic leak guard can falsely reject legitimate source quotes if
  it scans raw user/source content. The implementation must guard generated
  text boundaries rather than broad source text.
- Empty `rag_result.answer` can expose assumptions in downstream tests that
  treated no-RAG as a fact sentence. Those tests must be updated to match the
  RAG-to-cognition contract.
- Negative evidence still has value. The implementation must not drop all
  unresolved information; it must present only the user-relevant uncertainty.

## Execution Evidence

Status: not started.

- Plan review: pending.
- Focused deterministic tests before implementation: pending.
- Production implementation evidence: pending.
- Deterministic verification after implementation: pending.
- Readable real-output debug-LLM validation: pending.
- Independent code review: pending.
