# rag conversation evidence current episode boundary bugfix plan

## Summary

- Goal: Fix the immediate RAG2 failure where active-character prior-word evidence
  containing current-episode wording is rejected as Recall, and prevent
  unresolved RAG output from being presented as proof that no record exists.
- Plan class: medium
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: compatible
- Highest-risk areas: weakening the Recall boundary, changing generic prompts
  instead of the failing contract, hiding unresolved evidence behind confident
  prose, and broadening the bugfix into RAG 2.1 architecture work.
- Acceptance criteria: the incident-shaped active-character self-word lookup
  reaches conversation evidence, active agreement recall still rejects to
  Recall, unresolved-only RAG finalization stays uncertainty-preserving, and no
  existing database data is changed.

## Context

The incident sequence confirmed by database rows is:

- The active character asked the user to explain the definition: "你先解释一下呗，我等你的定义呢".
- The user later said: "刚刚不是你想了解什么是病娇嘛".
- The initializer produced a `Conversation-evidence` slot asking for current
  episode evidence where the active character expressed interest in learning
  about yandere.
- `conversation_evidence_agent` rejected the slot with
  `missing_context=["incompatible_intent:Recall"]`.
- The RAG finalizer projected that failure as if no active-character record
  existed, and cognition received the false negative.

This is a boundary bug. Prior work already added active-character self-word
retrieval and Recall/current-episode routing. The missing case is the overlap:
conversation transcript evidence can mention the current episode without being
episode-state Recall.

This bugfix is intentionally smaller than the RAG 2.1 contract plan. It fixes
the live failure mode without changing historical data, without rewriting the
initializer architecture, and without adding LLM prompt-negative constraints.

## Mandatory Skills

- `local-llm-architecture`: load before changing RAG routing, capability
  admission, prompt-facing evidence, or response-path call budgets.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python tests or source containing Chinese
  or Japanese strings.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the plan's `Independent Code Review` gate and record
  the result in execution evidence.
- Do not mutate existing MongoDB data, conversation history rows,
  conversation-progress state, user memories, or cached incident artifacts.
- Do not add negative prompt constraints to fix this failure mode.
- Do not change adapter behavior, relevance, decontextualization, cognition,
  consolidation, reflection, or memory contents in this bugfix.
- Do not add deterministic user-text routing outside the RAG capability agent.
- Keep Recall authoritative for active agreements, ongoing promises, current
  plans, open loops, and episode-position questions.
- Keep Conversation-evidence authoritative for prior messages, speaker
  attribution, exact wording, URLs, filenames, recent topics, counts, and
  message statistics.
- Preserve the existing response-path LLM call count. This bugfix must not add
  a new response-path LLM call.

## Must Do

- Add a focused failing capability-agent test for an incident-shaped
  `Conversation-evidence` slot that mentions the current episode and
  `speaker=active_character`.
- Add a regression test proving active agreement or episode-position Recall
  still rejects from `ConversationEvidenceAgent` as
  `incompatible_intent:Recall`.
- Replace the broad `if "active agreement" in normalized or "current episode" in normalized`
  rejection in `ConversationEvidenceAgent._deterministic_plan(...)` with a
  positive Recall-owned task detector.
- Ensure active-character prior-word, prior-claim, quote, definition-request,
  explanation-request, and self-authored-statement tasks remain Conversation
  evidence even when the slot text mentions current episode or just-now
  context.
- Add deterministic unresolved-summary tests for incompatible-route and
  missing-context results.
- Change unresolved-only RAG finalization so it returns deterministic
  uncertainty-preserving summaries instead of asking the finalizer LLM to turn
  unresolved facts into prose.
- Run focused deterministic tests and a prompt-render or initializer test only
  if implementation touches initializer-facing code.

## Deferred

- Do not implement RAG 2.1 structured slot records in this bugfix.
- Do not introduce an evidence authority envelope in this bugfix.
- Do not repair or rewrite the already persisted incident conversation-progress
  state.
- Do not change Cache2 strategy schema, persistent cache documents, or cache
  invalidation policy.
- Do not perform live database writes, backfills, migrations, or cleanup.

## Cutover Policy

Overall strategy: compatible

| Area | Policy | Instruction |
|---|---|---|
| Conversation-evidence admission | compatible | Narrow only the Recall-owned rejection logic. Existing conversation search, filter, aggregate, active-turn exclusion, and speaker-scope behavior stay intact. |
| Recall boundary | compatible | Preserve existing Recall ownership for agreements, commitments, plans, open loops, and episode-position state. |
| RAG finalizer unresolved-only path | compatible | Add deterministic handling only for all-unresolved known facts. Resolved or mixed resolved/unresolved finalization keeps the existing finalizer path unless tests prove a narrower deterministic branch is required. |
| Prompts | compatible | No prompt-negative fix is approved. Prompt edits are out of scope unless a focused test proves a non-negative prompt-render correction is required by the code change. |
| Data | compatible | Existing MongoDB rows and cache data remain unchanged. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a broader strategy by default.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local helper names only when the helper is private to the
  approved change surface and preserves this plan's contracts.
- The agent must not introduce new RAG architecture, a new retrieval agent, a
  dispatcher reroute, a retry loop, or a compatibility shim.
- The agent must not update old completed plans except by reading them as
  historical context.
- If an existing helper already detects missing-context or incompatible-route
  status, reuse it instead of duplicating logic.
- If implementation shows that a required behavior belongs in the RAG 2.1
  plan, stop and report the boundary instead of expanding this bugfix.

## Target State

For an incident-shaped slot:

```text
Conversation-evidence: retrieve messages from the current episode where the
active character asked for a definition or explanation of yandere
speaker=active_character
```

`ConversationEvidenceAgent` treats the request as transcript evidence, applies
active-character speaker scope, selects the existing conversation search or
filter path, and exposes found assistant-authored rows through the normal
projection payload.

For a Recall-owned slot:

```text
Conversation-evidence: retrieve active agreement for today's appointment
```

`ConversationEvidenceAgent` still returns unresolved with
`missing_context=["incompatible_intent:Recall"]` and does not call a
conversation worker.

For unresolved-only RAG finalization, the downstream answer states that the
specific retrieval path did not produce usable evidence. It does not claim that
no such message, interaction, or record exists globally.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Failure owner | Fix `ConversationEvidenceAgent` admission and RAG unresolved finalization | The DB rows exist and routing/evidence projection failed before cognition. |
| Recall detection | Use positive Recall-owned task concepts instead of raw `current episode` substring | Current episode can describe transcript scope as well as episode state. |
| Prompt strategy | Avoid prompt-negative constraints | The user explicitly rejected this class of fix for the incident. |
| Data strategy | Leave existing data unchanged | The user explicitly disallowed existing data changes. |
| Scope split | Put authority envelopes and structured slot records in the RAG 2.1 plan | Those changes are architectural improvements, not required for the immediate bugfix. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
  - Replace the broad deterministic Recall rejection in `_deterministic_plan`.
  - Keep worker selection and `_worker_context` behavior otherwise intact.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
  - Improve deterministic unresolved summaries for incompatible-route and
    missing-context results.
  - Add deterministic finalization only when every known fact is unresolved.
- `tests/test_rag_phase3_capability_agents.py`
  - Add incident-shaped Conversation-evidence admission tests and preserve the
    active-agreement rejection test.
- `tests/test_rag_finalizer_time_context.py`
  - Add deterministic tests for incompatible-route summaries,
    missing-context summaries, and unresolved-only finalizer behavior.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
  - Keep unchanged unless a focused regression proves its existing slot wording
    cannot reach the fixed capability agent.
- `src/kazusa_ai_chatbot/conversation_progress/`
  - Keep unchanged and do not rewrite persisted progress data.

## LLM Call And Context Budget

- Before: initializer may call one planner LLM; dispatcher is deterministic for
  recognized prefixes; capability agents may use existing worker calls;
  evaluator finalizer calls one summarizer/finalizer LLM.
- After: no new response-path LLM calls.
- Unresolved-only finalization should reduce LLM use by skipping the finalizer
  LLM when all facts are unresolved.
- Context-window cap: use the project default of 50k tokens. This plan does not
  increase any prompt budget.
- Verification: focused tests must prove deterministic unresolved-only
  finalization runs without invoking the finalizer LLM.

## Implementation Order

1. Add failing tests in `tests/test_rag_phase3_capability_agents.py`.
   - Add `test_conversation_evidence_current_episode_active_character_definition_request_uses_conversation_worker`.
   - Add or update a Recall-boundary test proving active agreement still
     returns `incompatible_intent:Recall`.
   - Run the two tests and record the current failure.
2. Update `ConversationEvidenceAgent._deterministic_plan(...)`.
   - Add a private helper such as `_recall_owned_task_reason(task_body: str) -> str`.
   - The helper returns `"Recall"` only for positive Recall-owned concepts:
     active agreement, active promise, active commitment, ongoing commitment,
     current plan, open loop, unresolved loop, next step, episode position, or
     where the episode left off.
   - It does not reject self-word or transcript-proof terms such as prior
     wording, prior claim, asked for a definition, asked for an explanation,
     quote, exact phrase, speaker, message, or self-authored statement.
3. Rerun the focused capability-agent tests.
4. Add evaluator tests for unresolved status.
   - Cover `missing_context=["incompatible_intent:Recall"]`.
   - Cover plain missing context such as `["person_ref"]`.
   - Cover all-unresolved known facts reaching `rag_finalizer`.
   - File: `tests/test_rag_finalizer_time_context.py`.
5. Update `persona_supervisor2_rag_evaluator.py`.
   - Make `_unresolved_summary(...)` describe incompatible route as route
     incompatibility, not as absent records.
   - Add a deterministic unresolved-only finalizer branch before the LLM
     finalizer call.
6. Run focused evaluator tests.
7. Run the regression set listed in `Verification`.
8. Run the independent code review gate and record findings before completion.

## Progress Checklist

- [x] Stage 1 - capability-agent regression tests added
  - Covers: implementation order step 1.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_current_episode_active_character_definition_request_uses_conversation_worker tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_rejects_active_agreement_intent -q`
  - Evidence: record expected failure before implementation.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-14`.
- [x] Stage 2 - conversation-evidence admission fixed
  - Covers: implementation order steps 2-3.
  - Verify: rerun the Stage 1 command and record pass.
  - Evidence: record changed files and test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-14`.
- [x] Stage 3 - unresolved finalization tests added
  - Covers: implementation order step 4.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_rag_finalizer_time_context.py::test_unresolved_summary_describes_incompatible_route tests\test_rag_finalizer_time_context.py::test_unresolved_summary_describes_missing_context tests\test_rag_finalizer_time_context.py::test_rag_finalizer_all_unresolved_uses_deterministic_summary -q`
  - Evidence: record expected failure before evaluator implementation.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-14`.
- [x] Stage 4 - unresolved finalization fixed
  - Covers: implementation order steps 5-7.
  - Verify: run all focused tests and the regression commands in
    `Verification`.
  - Evidence: record changed files and command output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-14`.
- [x] Stage 5 - independent code review complete
  - Covers: implementation order step 8.
  - Verify: review full diff against this plan, project style, and regression
    risk; rerun affected tests after any review fix.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, residual
    risks, and approval status.
  - Handoff: plan can be marked complete only after this stage is signed off.
  - Sign-off: `Codex/2026-05-14`.

## Verification

### Static Greps

- `rg -n "\"current episode\"|active agreement" src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py`
  - Expected: no broad conditional that rejects every task containing
    `"current episode"`; Recall-owned terms may appear inside the new helper or
    tests.
- `rg -n "negative constraint|forbidden_phrases" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag`
  - Expected: if this command returns matches, inspect `git diff` and verify
    this bugfix added no prompt-negative rule. Existing unrelated prompt text
    may remain.

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_current_episode_active_character_definition_request_uses_conversation_worker tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_rejects_active_agreement_intent -q`
- `venv\Scripts\python.exe -m pytest tests\test_rag_finalizer_time_context.py::test_unresolved_summary_describes_incompatible_route tests\test_rag_finalizer_time_context.py::test_unresolved_summary_describes_missing_context tests\test_rag_finalizer_time_context.py::test_rag_finalizer_all_unresolved_uses_deterministic_summary -q`

### Regression Tests

- `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_rag_projection.py -q`
- If initializer code is changed:
  `venv\Scripts\python.exe -m pytest tests\test_rag_initializer_cache2.py -q`

### Data Safety

- `git diff --stat`
  - Expected: source and test files only, plus this plan if it remains in the
    implementation branch. No generated DB dumps or modified fixture exports
    unless explicitly created by tests.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including prior adjacent plans, focused and
  regression tests, execution evidence, next-stage handoff notes, and
  path-safe commands.

Fix concrete findings directly only when the fix is inside the approved change
surface. If a fix would cross the approved boundary or alter the contract, stop
and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
execution evidence.

## Execution Evidence

### 2026-05-14 - Codex

- Scope changed:
  `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`,
  `tests/test_rag_phase3_capability_agents.py`, and
  `tests/test_rag_finalizer_time_context.py`.
- Stage 1 evidence: before the capability-agent fix, the incident-shaped
  active-character/current-episode test failed with
  `missing_context=["incompatible_intent:Recall"]`; the existing active
  agreement rejection still passed.
- Stage 2 evidence: `ConversationEvidenceAgent` now rejects only positive
  Recall-owned task concepts. Focused boundary tests pass, including
  active-character definition-request evidence, active agreement Recall
  rejection, and episode-state Recall rejection.
- Stage 3 evidence: before the evaluator fix, incompatible-route and
  missing-context summaries used generic absence wording, and all-unresolved
  finalization invoked the finalizer LLM.
- Stage 4 evidence: unresolved summaries now distinguish incompatible route
  and missing context. All-explicit-unresolved facts use a deterministic
  uncertainty-preserving finalizer branch; malformed or legacy fact records
  still use the existing LLM finalizer path.
- Verification commands:
  - `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_current_episode_active_character_definition_request_uses_conversation_worker tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_rejects_active_agreement_intent tests\test_rag_phase3_capability_agents.py::test_conversation_evidence_rejects_episode_state_intent -q`:
    `3 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_rag_finalizer_time_context.py -q`:
    `6 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py -q`:
    `55 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_rag_projection.py -q`:
    `13 passed`.
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_evaluator.py tests\test_rag_phase3_capability_agents.py tests\test_rag_finalizer_time_context.py`:
    passed.
  - `git diff --check`: passed with line-ending warnings only.
  - `rg -n '"current episode"|active agreement' src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py`:
    only the `active agreement` Recall marker remains.
  - `git diff -U0 -- src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag | rg -n "negative constraint|forbidden_phrases"`:
    no bugfix prompt-negative diff matches.
- Initializer regression command was not run because initializer-facing code was
  unchanged.
- Data safety: no migrations, backfills, cache edits, conversation-progress
  rewrites, memory edits, or incident-history edits were performed. After one
  pre-fix failing finalizer run exercised the existing telemetry path, finalizer
  tests were patched to no-op event logging.
- Independent review gate: no separate reviewer was used because the active
  execution request did not authorize a new subagent for this code review. The
  active agent reread the plan and reviewed the full diff against scope,
  routing boundaries, prompt constraints, data safety, and regression risk.
  Finding fixed: the deterministic all-unresolved finalizer branch now requires
  `resolved is False` explicitly, preserving the existing LLM finalizer path
  for malformed or legacy fact records. Approval status: accepted.

## Acceptance Criteria

This plan is complete when:

- The incident-shaped active-character self-word/current-episode
  Conversation-evidence test passes.
- Active agreement and episode-state Recall ownership remains protected.
- Unresolved-only RAG finalization does not assert that no record exists.
- No LLM prompt-negative constraint was added for this incident.
- No existing database data was changed.
- Focused and regression tests listed in `Verification` pass or any blocked
  command is recorded with a concrete reason.
- Independent code review is complete and recorded.
