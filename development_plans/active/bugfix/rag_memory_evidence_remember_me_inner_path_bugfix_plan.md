# rag memory evidence remember-me inner path bugfix plan

## Summary

- Goal: route "do you remember me?" and equivalent current-user prior-interaction
  memory requests to scoped `user_memory_evidence_agent` instead of shared
  `persistent_memory_search_agent`.
- Plan class: large
- Status: draft
- Mandatory skills: `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: bigbang for routing and prompt strategy cache;
  compatible for public RAG result shape.
- Highest-risk areas: prompt drift in the RAG initializer, current-user scope
  leakage into shared memory search, stale Cache2 initializer entries, and
  accidental changes to cognition/dialog behavior.
- Acceptance criteria: a scoped current-user recognition query produces a
  `Memory-evidence:` slot whose inner memory worker is
  `user_memory_evidence_agent`; shared character/world memory remains routed to
  `persistent_memory_search_agent`.

## Context

The observed production trace for `@杏山千纱 你还记得我吗` showed the outer RAG
path behaving correctly:

```text
initializer -> Memory-evidence:
dispatcher  -> memory_evidence_agent
```

The failure occurred inside `memory_evidence_agent`. The generated slot asked
for durable evidence about the current user's identity and past interactions,
but the deterministic selector did not classify "past interactions" as scoped
current-user continuity. It fell through to `persistent_memory_search_agent`,
which searches shared/global memory rows and returned unrelated character-world
facts. The evaluator rejected those candidates, and cognition safely avoided
hallucinating recognition.

The previous completed scoped-memory plan already established that
`Memory-evidence:` is an umbrella capability and that
`user_memory_evidence_agent` owns current-user private continuity. This bugfix
does not reopen that architecture. It narrows the missing recognition and
prior-interaction wording path.

## Mandatory Skills

- `local-llm-architecture`: load before changing RAG prompts, worker selection,
  prompt-versioning, or helper-agent contracts.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files that contain CJK prompt or
  test strings.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Preserve the RAG2 contract:
  `initializer -> semantic slot`, `dispatcher -> top-level capability`,
  `memory_evidence_agent -> internal worker`, deterministic code validates
  scope and execution.
- Do not add a new top-level RAG prefix.
- Do not make the initializer emit worker names, MongoDB collection names,
  index names, or low-level search parameters.
- Do not add response-path LLM calls, retry loops, fallback prompts, feature
  flags, compatibility shims, or alternate RAG paths.
- Do not change cognition, dialog, evaluator finalizer, persistence,
  scheduler, adapter delivery, or conversation evidence behavior.
- Prompt changes must be written as an organic rewrite of the affected
  initializer prompt flow. Do not append an isolated bullet or one-off example
  that contradicts the surrounding rule order. The rewritten prompt must read
  coherently from evidence-dependency gate, through source ownership, through
  conflict resolution, slot format, examples, and generation procedure.
- Any initializer prompt behavior change must bump
  `INITIALIZER_PROMPT_VERSION` and update tests that pin the version.
- Do not hard-code a concrete character name in reusable prompt rules or
  examples. Use role-neutral wording such as "the active character" unless the
  runtime `{character_name}` variable is required by the existing prompt.
- Current-user prior-interaction and recognition requests must route to scoped
  user memory only when the route text also has current-user scope from the
  slot or runtime context. Do not classify arbitrary third-party history as
  current-user continuity.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Add a deterministic capability-agent regression test for the exact RCA slot:
  `Memory-evidence: retrieve durable evidence about the current user's identity
  and past interactions with active character`.
- Make `memory_evidence_agent` route current-user recognition, prior shared
  interaction, and interaction-history wording to `user_memory_evidence_agent`.
- Preserve `persistent_memory_search_agent` routing for official character
  facts, shared world facts, common-sense topics, exact memory identifiers,
  tags, proper nouns, home/address/location facts, and global memory rows.
- Rewrite the RAG initializer prompt flow organically so "do you remember me?",
  "你还记得我吗", and equivalent current-user recognition questions generate a
  scoped current-user continuity `Memory-evidence:` slot.
- Bump `INITIALIZER_PROMPT_VERSION` from `initializer_prompt:v17` to
  `initializer_prompt:v18`.
- Add deterministic prompt contract tests covering the rewritten prompt and
  version bump.
- Add one live-LLM initializer regression for `@<active character> 你还记得我吗`
  that expects a `Memory-evidence:` slot with current-user scoped continuity or
  prior shared interaction wording.
- Update RAG documentation to state that current-user recognition and prior
  shared interaction memory belong to scoped user memory under
  `Memory-evidence:`.

## Deferred

- Do not change `user_memory_evidence_agent` retrieval algorithms, vector
  thresholds, lexical extraction, or recency fallback.
- Do not change `persistent_memory_search_agent`.
- Do not add or migrate MongoDB collections.
- Do not tune embeddings or RAG search limits.
- Do not change `Person-context`, `Recall`, `Conversation-evidence`, cognition,
  dialog, consolidation, scheduler, or adapters.
- Do not add a `User-memory-evidence:` prefix.
- Do not implement global lore promotion or channel/group continuity.
- Do not solve unrelated "remember what I just said" conversation-history
  cases in this plan; those remain `Conversation-evidence:` when the request is
  about recent or exact chat content.

## Cutover Policy

Overall strategy: bigbang for the changed behavior.

| Area | Policy | Instruction |
|---|---|---|
| `memory_evidence_agent` current-user recognition routing | bigbang | Replace the fall-through shared-memory route with scoped user-memory routing for current-user recognition and prior-interaction slot text. |
| Initializer prompt wording | bigbang | Rewrite the prompt flow coherently and bump the initializer prompt version; do not preserve stale cached strategy compatibility. |
| Public RAG slot prefix | compatible | Keep `Memory-evidence:` as the top-level prefix. |
| Public RAG projection shape | compatible | Preserve existing `rag_result` fields and scoped metadata behavior. |
| Shared memory retrieval | compatible | Preserve existing shared/world/official memory routing. |
| Tests | bigbang | Update expected prompt version and add focused regressions for this bug. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- For bigbang areas, rewrite the old behavior directly instead of adding a
  compatibility branch.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The target ownership boundary is RAG2 memory-evidence worker selection plus
  the RAG initializer prompt contract.
- The agent may choose local helper names only when they preserve the contracts
  in this plan.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, or extra features.
- The agent must treat changes outside the files listed in `Change Surface` as
  out of scope unless this plan is updated first.
- The initializer prompt rewrite may reorganize the existing prompt text only
  to improve logical flow around existing source ownership. It must not change
  unrelated routing semantics.
- The agent must not satisfy the prompt requirement by adding only a single
  pattern-gallery example. The rule flow and generation procedure must also
  align with the new scoped-continuity intent.
- If existing tests encode the old fall-through behavior, update them only to
  match this approved contract and record the old expectation in
  `Execution Evidence`.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

For current-user recognition queries, the response-time path is:

```text
user asks "do you remember me?"
  -> RAG initializer emits one Memory-evidence slot for current-user scoped
     continuity / prior shared interactions
  -> dispatcher routes by prefix to memory_evidence_agent
  -> memory_evidence_agent selects user_memory_evidence_agent
  -> user_memory_evidence_agent searches user_memory_units scoped by
     context["global_user_id"]
  -> projection preserves scoped memory rows and user_memory_unit_candidates
  -> cognition decides whether recognition is supported by evidence
  -> dialog either acknowledges known continuity or honestly asks for hints
```

The specific RCA slot must route as:

```python
{
    "primary_worker": "user_memory_evidence_agent",
    "source_policy": "scoped current-user continuity evidence",
}
```

Official/shared memory remains unchanged:

```text
Memory-evidence: retrieve durable evidence about the active character's official address
  -> memory_evidence_agent
  -> persistent_memory_search_agent
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Fix boundary | Fix inside `memory_evidence_agent` worker selection and initializer prompt contract. | Dispatcher already chooses the correct top-level capability; cognition/dialog already handle uncertainty safely. |
| Prompt strategy | Rewrite the relevant initializer prompt flow organically. | The user requires prompt logic changes to read as a coherent route policy, not as a bolted-on special case. |
| Prompt cache | Bump `INITIALIZER_PROMPT_VERSION` to `initializer_prompt:v18`. | Existing cached initializer strategies are tied to old routing guidance. |
| Top-level prefix | Keep `Memory-evidence:`. | Current-user private continuity is already owned by the memory evidence capability. |
| Worker path | Use deterministic route markers before selector LLM. | This keeps the common path bounded and inspectable for local models. |
| Data source | Use scoped `user_memory_units` through existing `user_memory_evidence_agent`. | The question is about current-user continuity, not shared character/world memory. |
| Downstream behavior | Leave cognition/dialog unchanged. | They correctly avoided hallucinated recognition when evidence was missing. |

## Contracts And Data Shapes

### MemoryEvidenceAgent Selection Contract

`_deterministic_plan(task, context)` must route to
`user_memory_evidence_agent` when all of these are true:

- route text includes current-user scope from `task`, `context["original_query"]`,
  or `context["current_slot"]`;
- route text asks about recognition, remembering the current user, prior shared
  interactions, past interactions, shared history, user-specific continuity, or
  equivalent CJK wording.

Required English route concepts:

```text
past interaction
past interactions
prior interaction
prior interactions
interaction history
prior shared interaction
prior shared interactions
shared history
remember the current user
recognize the current user
remember me
```

Required CJK route concepts:

```text
过往互动
历史互动
之前的互动
以前的互动
共同经历
记得我
还记得我
认识我
记得当前用户
认识当前用户
```

The route must remain `persistent_memory_search_agent` for shared durable facts
without current-user scope.

### Initializer Prompt Contract

The rewritten initializer prompt must preserve the existing visible input and
output formats:

```python
{
    "original_query": "user's question",
    "context": "auxiliary info",
}
```

```python
{
    "unknown_slots": ["slot 1", "slot 2", "..."],
}
```

The prompt must state, in the normal rule flow, that:

- current-user profile or durable person context remains `Person-context:`;
- recent or exact chat content remains `Conversation-evidence:`;
- active agreements remain `Recall:`;
- current-user private continuity, recognition, accepted preferences,
  user-specific lore, and prior shared interactions belong to
  `Memory-evidence:`;
- a "do you remember me?" query needs evidence unless the surrounding context
  already makes it a routine no-retrieval greeting, and its evidence target is
  current-user scoped continuity rather than shared character/world memory.

The pattern gallery must include one boundary anchor equivalent to:

```text
Query: "<character mention>你还记得我吗"
  -> The current user asks whether the active character has durable continuity
     about this user. Use scoped memory evidence, not shared world memory and
     not Person-context.
  ["Memory-evidence: retrieve current-user private continuity and prior shared interactions with the active character"]
```

The generation procedure must explicitly apply the current-user scoped memory
rule before generic durable memory defaults and after live/recall/person/history
source checks.

## LLM Call And Context Budget

Before:

- Response path uses the existing RAG initializer LLM call, deterministic
  dispatcher, memory capability selection, optional worker retrieval, evaluator,
  cognition, and dialog.
- `memory_evidence_agent` may use its selector LLM only when deterministic
  routing does not return a plan.

After:

- No new response-path LLM calls.
- No new background LLM calls.
- Initializer prompt content changes and version becomes
  `initializer_prompt:v18`.
- Current-user recognition slots must use deterministic memory-worker routing,
  so selector LLM usage does not increase for this bug path.
- Prompt rewrite must keep the initializer under the existing local-model
  context budget. The implementation agent must record rendered prompt
  character length before and after in `Execution Evidence`; any increase above
  15 percent requires user approval before implementation continues.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
  - Extend deterministic scoped-user topic recognition for current-user
    recognition and prior shared interactions.
  - Update selector prompt wording only when needed to keep it consistent with
    the deterministic contract.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
  - Organically rewrite the initializer prompt flow around current-user scoped
    continuity.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - Bump `INITIALIZER_PROMPT_VERSION` to `initializer_prompt:v18`.
- `src/kazusa_ai_chatbot/rag/README.md`
  - Document that current-user recognition and prior shared interactions are
    scoped user-memory evidence.
- `tests/test_rag_phase3_capability_agents.py`
  - Add deterministic worker-selection regression for the RCA slot and a
    shared-memory non-regression.
- `tests/test_rag_initializer_cache2.py`
  - Update initializer prompt-version expectation and add deterministic prompt
    contract assertions for the rewritten remember-me route.
- `tests/test_rag_phase3_initializer_live_llm.py`
  - Add one `live_llm` initializer regression for `你还记得我吗`.

### Create

- No new source modules.
- No new database collections.

### Keep

- Keep `user_memory_evidence_agent.py` retrieval behavior unchanged.
- Keep `persistent_memory_search_agent.py` unchanged.
- Keep `persona_supervisor2_rag_dispatch.py` unchanged unless a test reveals a
  pure version or documentation mismatch; dispatcher routing behavior is not
  the bug.
- Keep cognition and dialog unchanged.

## Implementation Order

1. Add capability-agent tests in
   `tests/test_rag_phase3_capability_agents.py`.
   - Add `test_memory_evidence_remember_me_slot_uses_scoped_worker`.
   - Use a fake `user_memory_agent` that returns one scoped memory row.
   - Use a fake `search_agent` that would fail the test if called.
   - Input slot:
     `Memory-evidence: retrieve durable evidence about the current user's identity and past interactions with active character`.
   - Expected before implementation: fails because `primary_worker` is
     `persistent_memory_search_agent` or because the forbidden fake search
     worker is called.
2. Add shared-memory non-regression in
   `tests/test_rag_phase3_capability_agents.py`.
   - Add `test_memory_evidence_official_character_fact_stays_shared_memory`.
   - Input slot:
     `Memory-evidence: retrieve durable evidence about the active character's official address`.
   - Expected after implementation: `primary_worker` remains
     `persistent_memory_search_agent`.
3. Add prompt contract tests in `tests/test_rag_initializer_cache2.py`.
   - Update `test_initializer_prompt_version_bumped_for_capability_cutover` to
     expect `initializer_prompt:v18`.
   - Add `test_initializer_prompt_documents_current_user_recognition_memory_route`.
   - The prompt contract test must render `_INITIALIZER_PROMPT.format(...)` and
     assert the rendered prompt contains `你还记得我吗`,
     `current-user private continuity`, `prior shared interactions`, and
     `Memory-evidence: retrieve current-user private continuity`.
4. Add the live initializer regression in
   `tests/test_rag_phase3_initializer_live_llm.py`.
   - Add `test_live_initializer_routes_remember_me_to_scoped_memory`.
   - Query: `@<active character> 你还记得我吗`.
   - Expected prefixes: `["Memory-evidence:"]`.
   - Forbidden prefixes: `["Person-context:", "Conversation-evidence:"]`.
   - Required slot fragments: `["current-user", "continuity"]` or
     `["prior shared interaction"]`; use the existing test helper style for
     required fragments.
   - Run this live test one case at a time with `-m live_llm -q -s`.
5. Implement deterministic worker routing in
   `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`.
   - Add the route concepts from `Contracts And Data Shapes` to the scoped
     current-user topic recognition path.
   - Keep the existing `has_scoped_user_scope and has_scoped_user_topic` shape.
   - Do not inspect adapter raw syntax or mutate user text.
6. Rewrite the initializer prompt in
   `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`.
   - Read the full prompt before editing.
   - Integrate current-user recognition into the evidence-dependency gate,
     memory evidence rule, context pre-check, conflict resolution, slot format
     guidance, pattern gallery, and generation procedure.
   - Remove or revise wording that would make current-user recognition look
     like generic durable shared memory or profile lookup.
7. Bump `INITIALIZER_PROMPT_VERSION` in
   `src/kazusa_ai_chatbot/rag/cache2_policy.py`.
8. Update `src/kazusa_ai_chatbot/rag/README.md` with the narrowed scoped
   recognition behavior.
9. Run prompt render checks and focused tests listed in `Verification`.
10. Run the independent code review gate and remediate findings inside the
    approved change surface.

## Progress Checklist

- [ ] Stage 1 - Failing tests added
  - Covers: implementation steps 1 through 4.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py::test_memory_evidence_remember_me_slot_uses_scoped_worker -q`
    fails for the current inner worker path.
  - Evidence: record the failing assertion or forbidden-worker call in
    `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - MemoryEvidenceAgent inner path fixed
  - Covers: implementation step 5.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py::test_memory_evidence_remember_me_slot_uses_scoped_worker tests/test_rag_phase3_capability_agents.py::test_memory_evidence_official_character_fact_stays_shared_memory -q`.
  - Evidence: record worker names from passing tests in `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - Initializer prompt rewritten and versioned
  - Covers: implementation steps 6 and 7.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_initializer_cache2.py::test_initializer_prompt_version_bumped_for_capability_cutover tests/test_rag_initializer_cache2.py::test_initializer_prompt_documents_current_user_recognition_memory_route -q`.
  - Evidence: record rendered prompt length before/after and prompt-version
    assertion result in `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - Documentation and focused regression complete
  - Covers: implementation step 8 and deterministic focused tests.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_initializer_cache2.py -q`.
  - Evidence: record test output in `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - Live initializer smoke complete
  - Covers: implementation step 4 after prompt rewrite.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_remember_me_to_scoped_memory -m live_llm -q -s`.
  - Evidence: record the emitted slot and test result in `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - Independent code review complete
  - Covers: `Independent Code Review`.
  - Verify: review full diff against this plan, then rerun affected focused
    tests after any review fixes.
  - Evidence: record review mode, findings, fixes, rerun commands, residual
    risks, and approval status in `Execution Evidence`.
  - Handoff: plan may be marked completed only after this checkpoint passes.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg "initializer_prompt:v17" src tests`
  - Expected after implementation: no matches.
  - `rg` exit code 1 is acceptable when there are zero matches.
- `rg "initializer_prompt:v18" src tests`
  - Expected after implementation: matches in `cache2_policy.py` and relevant
    tests only.
- `rg "User-memory-evidence|user_memory_evidence:" src tests`
  - Expected after implementation: no new top-level prefix usage. Existing
    Python class or module names are allowed only when they are
    `user_memory_evidence_agent`.

### Prompt Render

- Run:
  `venv\Scripts\python.exe -c "from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_initializer import _INITIALIZER_PROMPT; rendered=_INITIALIZER_PROMPT.format(character_name='<active character>'); print(len(rendered)); assert '你还记得我吗' in rendered; assert 'Memory-evidence: retrieve current-user private continuity' in rendered"`
  - Expected after implementation: command exits 0 and prints the rendered
    prompt length.

### Python Compile

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\memory_evidence_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_initializer.py src\kazusa_ai_chatbot\rag\cache2_policy.py`
  - Expected after implementation: exits 0.

### Focused Deterministic Tests

- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py::test_memory_evidence_remember_me_slot_uses_scoped_worker tests/test_rag_phase3_capability_agents.py::test_memory_evidence_official_character_fact_stays_shared_memory -q`
  - Expected after implementation: both pass.
- `venv\Scripts\python.exe -m pytest tests/test_rag_initializer_cache2.py::test_initializer_prompt_version_bumped_for_capability_cutover tests/test_rag_initializer_cache2.py::test_initializer_prompt_documents_current_user_recognition_memory_route -q`
  - Expected after implementation: both pass.

### Focused Regression Batch

- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_initializer_cache2.py tests/test_user_memory_evidence_agent.py -q`
  - Expected after implementation: all pass.

### Live LLM Gate

- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_remember_me_to_scoped_memory -m live_llm -q -s`
  - Expected after implementation: one case passes, and the inspected output
    shows a `Memory-evidence:` slot for current-user scoped continuity or prior
    shared interactions.
  - This live test must be run one case at a time with output inspected.

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
- Prompt quality: the initializer prompt must read as a coherent source-routing
  policy, not as isolated special-case text.
- Regression and handoff quality, including focused tests, live-LLM evidence,
  static checks, execution evidence, and any residual risk.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture/documentation
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The RCA slot routes to `user_memory_evidence_agent`.
- Shared character/world memory slots still route to
  `persistent_memory_search_agent`.
- The initializer prompt has been organically rewritten to include
  current-user recognition and prior shared interactions in the scoped memory
  rule flow.
- `INITIALIZER_PROMPT_VERSION` is `initializer_prompt:v18`.
- Deterministic focused tests, prompt render checks, compile checks, and the
  listed regression batch pass.
- The live-LLM initializer gate passes with inspected output.
- RAG documentation describes the scoped recognition behavior.
- Independent code review has passed and its result is recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Prompt rewrite changes unrelated source routing | Keep the rewrite limited to current-user recognition flow and run existing initializer prompt tests. | `tests/test_rag_initializer_cache2.py` and live initializer regression. |
| Current-user recognition overroutes third-party history to scoped memory | Require current-user scope plus recognition/prior-interaction topic markers. | Capability-agent shared-memory non-regression test. |
| Stale initializer Cache2 strategies preserve old wording | Bump `INITIALIZER_PROMPT_VERSION` to `initializer_prompt:v18`. | Static grep and version test. |
| Local LLM prompt becomes longer or less coherent | Require full prompt render check and rendered length evidence; cap increase over 15 percent without approval. | Prompt render command and execution evidence. |
| Downstream model hallucinates recognition | Leave cognition/dialog unchanged; RAG only supplies scoped evidence or unresolved state. | Existing cognition/dialog behavior plus end-to-end log inspection when live tested. |

## Execution Evidence

- Pre-implementation failing tests:
- Prompt rendered length before/after:
- Static grep results:
- Compile results:
- Focused deterministic test results:
- Focused regression batch results:
- Live LLM gate result and inspected slot:
- Documentation update:
- Independent code review result:
