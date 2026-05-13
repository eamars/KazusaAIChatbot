# rag memory evidence remember-me inner path bugfix plan

## Summary

- Goal: route "do you remember me?" and equivalent current-user prior-interaction
  memory requests to scoped `user_memory_evidence_agent` instead of shared
  `persistent_memory_search_agent`.
- Plan class: large
- Status: completed
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
- Prompt changes must be the minimum source-ownership clarification needed for
  the bug boundary. Do not turn `_INITIALIZER_PROMPT` into a lookup table based
  on the observed failure mode. Avoid adding failure-mode examples when the
  same behavior can be captured by the existing route rules.
- Any initializer prompt edit requires a full review of the entire rendered
  prompt before the edit is accepted. The implementation agent must inspect the
  prompt from Rule 0 through Output format, build a route-contract checklist
  covering every top-level source family, and record the checklist result in
  `Execution Evidence`.
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
- Apply the minimum RAG initializer prompt clarification so current-user
  recognition and equivalent current-user prior-interaction questions generate
  a scoped current-user continuity `Memory-evidence:` slot.
- Bump `INITIALIZER_PROMPT_VERSION` from `initializer_prompt:v17` to
  `initializer_prompt:v18`.
- Add deterministic prompt contract tests covering the rewritten prompt and
  version bump.
- Add one live-LLM initializer regression for `@<active character> 你还记得我吗`
  that expects a `Memory-evidence:` slot with current-user scoped continuity or
  prior shared interaction wording.
- Add live-LLM initializer negative regressions for adjacent routes so the
  prompt rewrite proves it did not steal active agreements, recent chat recall,
  active-character self-word recall, current-user URL recall, profile/person
  reads, or shared official memory.
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
| Initializer prompt wording | bigbang | Apply a minimal source-ownership clarification and bump the initializer prompt version; do not add lookup-table examples or preserve stale cached strategy compatibility. |
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
- The initializer prompt change must stay limited to existing source ownership
  rules. It must not change unrelated routing semantics.
- The agent must not satisfy the prompt requirement by adding a pattern-gallery
  lookup-table entry for the observed failure mode.
- The agent must not accept the prompt rewrite until the rendered full prompt
  has been reviewed as one document. The review must check that each rule,
  conflict-resolution item, slot-format row, pattern-gallery case, input
  format, generation-procedure step, and output format remains consistent.
- The agent must not rely only on deterministic prompt-text assertions for this
  prompt change. Real LLM positive and negative initializer tests are required
  and must be run one case at a time with output inspected.
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
- current-user private continuity, recognition, durable user memory facts,
  accepted preferences, and prior shared interactions belong to
  `Memory-evidence:`;
- current-user recognition and prior-interaction queries target scoped
  current-user continuity.

The prompt must not add a pattern-gallery lookup entry for the observed
failure mode. The generation procedure should remain unchanged unless the
route contract cannot be made clear through the existing source ownership
rules.

The implementation must produce and record a full-prompt review checklist with
these source-family expectations:

| Source family | Expected prompt contract |
|---|---|
| No retrieval | greetings, thanks, welcome-back, praise, and routine social acknowledgement stay empty unless durable evidence is required. |
| Live context | current time/date/weekday and current external facts stay `Live-context:`. |
| Recall | active agreements, promises, plans, open loops, and current episode state stay `Recall:`. |
| Person context | profile, impression, compatibility, relationship, ranking, user-list, and active-character self-profile stay `Person-context:`. |
| Conversation evidence | exact phrases, quoted messages, URLs, recent/fuzzy chat topics, speaker provenance, user-self recent words, and active-character self-words stay `Conversation-evidence:`. |
| Memory evidence scoped user continuity | current-user recognition, durable user memory facts, accepted preferences, and prior shared interactions use `Memory-evidence:` with scoped current-user continuity wording. |
| Memory evidence shared durable facts | official character/world facts, common-sense facts, stable home/address/location facts, object/place/concept knowledge, tags, and memory identifiers stay `Memory-evidence:` without implying current-user scope. |
| Web evidence | public web page/topic reads that are not current/live stay `Web-evidence:`. |

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
- Real LLM validation after the prompt rewrite must cover the positive and
  negative initializer matrix in `Verification`. These tests use the existing
  initializer LLM call only; they do not add runtime LLM calls to production.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
  - Extend deterministic scoped-user topic recognition for current-user
    recognition and prior shared interactions.
  - Update selector prompt wording only when needed to keep it consistent with
    the deterministic contract.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
  - Add minimum source-ownership clarification around current-user scoped
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
  - Add one `live_llm` positive initializer regression for `你还记得我吗`.
  - Add or extend live-LLM negative initializer regressions for adjacent
    source families affected by the full-prompt rewrite.

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
     assert the rendered prompt contains `current-user private continuity`,
     `prior shared interactions`, and
     `Memory-evidence: retrieve current-user private continuity`.
4. Add the live initializer positive and negative regressions in
   `tests/test_rag_phase3_initializer_live_llm.py`.
   - Add `test_live_initializer_routes_remember_me_to_scoped_memory`.
   - Query: `@<active character> 你还记得我吗`.
   - Expected prefixes: `["Memory-evidence:"]`.
   - Forbidden prefixes: `["Person-context:", "Conversation-evidence:"]`.
   - Required slot fragments: `["current-user", "continuity"]` or
     `["prior shared interaction"]`; use the existing test helper style for
     required fragments.
   - Add `test_live_initializer_remember_today_agreement_stays_recall`.
     Query: `早上好呀，还记得今天的约定么`.
     Expected prefixes: `["Recall:"]`; forbidden prefixes:
     `["Memory-evidence:", "Conversation-evidence:", "Person-context:"]`.
   - Add `test_live_initializer_remember_recent_user_words_stays_conversation`.
     Query: `你还记得我刚刚说那堆充电线里大概有哪些吗？`.
     Expected prefixes: `["Conversation-evidence:"]`; required slot fragments:
     `["speaker=current_user"]`; forbidden prefixes:
     `["Memory-evidence:", "Person-context:", "Recall:"]`.
   - Add `test_live_initializer_active_character_self_words_stays_conversation`.
     Query: `你之前是不是说过那个项目要延期？`.
     Expected prefixes: `["Conversation-evidence:"]`; required slot fragments:
     `["speaker=active_character"]`; forbidden prefixes:
     `["Memory-evidence:", "Person-context:", "Recall:"]`.
   - Add `test_live_initializer_current_user_url_recall_stays_conversation`.
     Query: `我上次发的那个链接里有什么信息？`.
     Expected prefixes: `["Conversation-evidence:", "Web-evidence:"]`;
     required slot fragments: `["speaker=current_user"]`; forbidden prefixes:
     `["Memory-evidence:", "Person-context:", "Recall:"]`.
   - Add `test_live_initializer_named_person_impression_stays_person_context`.
     Query: `<character mention>你觉得小明这个人怎么样`.
     Expected prefixes: `["Person-context:"]`; forbidden prefixes:
     `["Memory-evidence:", "Conversation-evidence:", "Recall:"]`.
   - Add `test_live_initializer_official_address_stays_shared_memory_slot`.
     Query: `你家的官方地址是什么？`.
     Expected prefixes: `["Memory-evidence:"]`; required slot fragments:
     `["official address"]` or `["官方地址"]`; forbidden slot fragments:
     `["current-user", "private continuity", "prior shared interaction"]`.
   - Run each live test one case at a time with `-m live_llm -q -s`, inspect
     each emitted slot, and record the observed route in `Execution Evidence`.
5. Implement deterministic worker routing in
   `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`.
   - Add the route concepts from `Contracts And Data Shapes` to the scoped
     current-user topic recognition path.
   - Keep the existing `has_scoped_user_scope and has_scoped_user_topic` shape.
   - Do not inspect adapter raw syntax or mutate user text.
6. Apply the minimum initializer prompt clarification in
   `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`.
   - Read the full prompt before editing.
   - Build the full-prompt route-contract checklist listed in
     `Initializer Prompt Contract`.
   - Integrate current-user recognition only into the existing memory evidence
     rule, context pre-check, conflict resolution, and slot format guidance.
   - Do not add a failure-mode pattern-gallery entry or rewrite the generation
     procedure unless tests prove it is necessary.
7. Bump `INITIALIZER_PROMPT_VERSION` in
   `src/kazusa_ai_chatbot/rag/cache2_policy.py`.
8. Update `src/kazusa_ai_chatbot/rag/README.md` with the narrowed scoped
   recognition behavior.
9. Run prompt render checks and focused tests listed in `Verification`.
10. Run the independent code review gate and remediate findings inside the
    approved change surface.

## Progress Checklist

- [x] Stage 1 - Failing tests added
  - Covers: implementation steps 1 through 4.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py::test_memory_evidence_remember_me_slot_uses_scoped_worker -q`
    fails for the current inner worker path.
  - Evidence: record the failing assertion or forbidden-worker call in
    `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-13` after verification and evidence are recorded.
- [x] Stage 2 - MemoryEvidenceAgent inner path fixed
  - Covers: implementation step 5.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py::test_memory_evidence_remember_me_slot_uses_scoped_worker tests/test_rag_phase3_capability_agents.py::test_memory_evidence_official_character_fact_stays_shared_memory -q`.
  - Evidence: record worker names from passing tests in `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `Codex/2026-05-13` after verification and evidence are recorded.
- [x] Stage 3 - Initializer prompt rewritten and versioned
  - Covers: implementation steps 6 and 7.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_initializer_cache2.py::test_initializer_prompt_version_bumped_for_capability_cutover tests/test_rag_initializer_cache2.py::test_initializer_prompt_documents_current_user_recognition_memory_route -q`.
  - Evidence: record rendered prompt length before/after, full-prompt
    route-contract checklist result, and prompt-version assertion result in
    `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `Codex/2026-05-13` after verification and evidence are recorded.
- [x] Stage 4 - Documentation and focused regression complete
  - Covers: implementation step 8 and deterministic focused tests.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_initializer_cache2.py -q`.
  - Evidence: record test output in `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `Codex/2026-05-13` after verification and evidence are recorded.
- [x] Stage 5 - Live initializer smoke complete
  - Covers: implementation step 4 after prompt rewrite.
  - Verify:
    run every live LLM test listed in `Verification` one case at a time with
    `-m live_llm -q -s`.
  - Evidence: record each emitted slot, positive/negative classification, and
    test result in `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 6.
  - Sign-off: `Codex/2026-05-13` after verification and evidence are recorded.
- [x] Stage 6 - Independent code review complete
  - Covers: `Independent Code Review`.
  - Verify: review full diff against this plan, then rerun affected focused
    tests after any review fixes.
  - Evidence: record review mode, findings, fixes, rerun commands, residual
    risks, and approval status in `Execution Evidence`.
  - Handoff: plan may be marked completed only after this checkpoint passes.
  - Sign-off: `Codex/2026-05-13` after verification and evidence are recorded.

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
  `venv\Scripts\python.exe -c "from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_initializer import _INITIALIZER_PROMPT; rendered=_INITIALIZER_PROMPT.format(character_name='<active character>'); print(len(rendered)); assert 'Memory-evidence: retrieve current-user private continuity' in rendered; assert 'prior shared interactions' in rendered"`
  - Expected after implementation: command exits 0 and prints the rendered
    prompt length.
- Full-prompt review:
  - Expected after implementation: `Execution Evidence` contains a route
    checklist result for every source family listed in `Initializer Prompt
    Contract`, and no checklist item is marked blocked or unreviewed.

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

### Live LLM Positive Gate

- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_remember_me_to_scoped_memory -m live_llm -q -s`
  - Expected after implementation: one case passes, and the inspected output
    shows a `Memory-evidence:` slot for current-user scoped continuity or prior
    shared interactions.
  - This live test must be run one case at a time with output inspected.

### Live LLM Negative Gates

Run each command separately with output inspected:

- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_remember_today_agreement_stays_recall -m live_llm -q -s`
  - Expected after implementation: `Recall:` is emitted; no
    `Memory-evidence:`, `Conversation-evidence:`, or `Person-context:` slot is
    emitted for the agreement.
- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_remember_recent_user_words_stays_conversation -m live_llm -q -s`
  - Expected after implementation: `Conversation-evidence:` with
    `speaker=current_user`; no scoped memory slot.
- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_active_character_self_words_stays_conversation -m live_llm -q -s`
  - Expected after implementation: `Conversation-evidence:` with
    `speaker=active_character`; no scoped memory slot.
- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_current_user_url_recall_stays_conversation -m live_llm -q -s`
  - Expected after implementation: `Conversation-evidence:` for the current
    user's URL plus `Web-evidence:` for content retrieval; no scoped memory
    slot.
- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_named_person_impression_stays_person_context -m live_llm -q -s`
  - Expected after implementation: `Person-context:`; no scoped memory slot.
- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_official_address_stays_shared_memory_slot -m live_llm -q -s`
  - Expected after implementation: `Memory-evidence:` for the active
    character's official address; slot text must not contain current-user
    private-continuity wording.

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
- Prompt validation quality: the reviewer must inspect the full-prompt
  route-contract checklist and the real LLM positive/negative test traces.
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
- The live-LLM positive and negative initializer gates pass with inspected
  output recorded for every case.
- RAG documentation describes the scoped recognition behavior.
- Independent code review has passed and its result is recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Prompt change expands into lookup-table routing | Keep the initializer edit to minimum source-ownership clarification and validate with real LLM positive/negative gates. | Prompt diff review, `tests/test_rag_initializer_cache2.py`, and live initializer regression. |
| Current-user recognition overroutes third-party history to scoped memory | Require current-user scope plus recognition/prior-interaction topic markers. | Capability-agent shared-memory non-regression test. |
| Stale initializer Cache2 strategies preserve old wording | Bump `INITIALIZER_PROMPT_VERSION` to `initializer_prompt:v18`. | Static grep and version test. |
| Local LLM prompt becomes longer or less coherent | Require full prompt render check and rendered length evidence; cap increase over 15 percent without approval. | Prompt render command and execution evidence. |
| Downstream model hallucinates recognition | Leave cognition/dialog unchanged; RAG only supplies scoped evidence or unresolved state. | Existing cognition/dialog behavior plus end-to-end log inspection when live tested. |

## Execution Evidence

- Pre-implementation failing tests:
  - `venv\Scripts\python.exe -m py_compile tests\test_rag_phase3_capability_agents.py tests\test_rag_initializer_cache2.py tests\test_rag_phase3_initializer_live_llm.py` passed after adding red tests.
  - `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py::test_memory_evidence_remember_me_slot_uses_scoped_worker -q` failed as expected before implementation. Observed `primary_worker=persistent_memory_search_agent`, `missing_context=['memory_evidence']`, and assertion `result["resolved"] is True` failed.
- Prompt rendered length before/after:
  - Stage 3 before edit: 18167 characters.
  - Stage 3 initial prompt rewrite: 20672 characters, a 13.79 percent
    increase. This was below the 15 percent threshold but was superseded by
    the user's post-review requirement to keep the prompt change minimal.
  - Post-review minimal prompt clarification: 18630 characters, a 2.55 percent
    increase from baseline. The final prompt diff adds no pattern-gallery
    lookup entry and leaves the generation procedure unchanged.
- Static grep results:
  - Stage 6: `rg "initializer_prompt:v17" src tests` returned exit code 1
    with no matches, as expected.
  - Stage 6: `rg "initializer_prompt:v18" src tests` matched only
    `src\kazusa_ai_chatbot\rag\cache2_policy.py` and
    `tests\test_rag_initializer_cache2.py`.
  - Stage 6: `rg "User-memory-evidence|user_memory_evidence:" src tests`
    returned exit code 1 with no matches, so no new top-level prefix was
    introduced.
  - Post-minimal-prompt rerun repeated these static checks with the same
    results.
- Compile results:
  - Stage 3: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_initializer.py src\kazusa_ai_chatbot\rag\cache2_policy.py tests\test_rag_initializer_cache2.py` passed.
  - Stage 6: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\memory_evidence_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_initializer.py src\kazusa_ai_chatbot\rag\cache2_policy.py` passed.
  - Post-minimal-prompt rerun: the same compile command passed.
- Focused deterministic test results:
  - Stage 2: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\memory_evidence_agent.py` passed.
  - Stage 2: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py::test_memory_evidence_remember_me_slot_uses_scoped_worker tests/test_rag_phase3_capability_agents.py::test_memory_evidence_official_character_fact_stays_shared_memory -q` passed. Observed `primary_worker=user_memory_evidence_agent` for the RCA slot and `primary_worker=persistent_memory_search_agent` for the official-address slot.
  - Stage 3 prompt render: `venv\Scripts\python.exe -c "from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_initializer import _INITIALIZER_PROMPT; rendered=_INITIALIZER_PROMPT.format(character_name='<active character>'); print(len(rendered)); assert 'Memory-evidence: retrieve current-user private continuity' in rendered; assert 'prior shared interactions' in rendered"` passed after prompt minimization and printed `18630`.
  - Stage 3 focused prompt tests: `venv\Scripts\python.exe -m pytest tests/test_rag_initializer_cache2.py::test_initializer_prompt_version_bumped_for_capability_cutover tests/test_rag_initializer_cache2.py::test_initializer_prompt_documents_current_user_recognition_memory_route -q` passed; version assertion observed `initializer_prompt:v18`.
- Focused regression batch results:
  - Stage 4: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_initializer_cache2.py -q` passed after documentation update and test-name cleanup; observed `79 passed`.
  - Stage 6: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_initializer_cache2.py tests/test_user_memory_evidence_agent.py -q` passed before and after the review-only plan fix; final observed result was `86 passed`.
  - Post-minimal-prompt rerun of the same focused regression batch passed with
    `86 passed`.
- Full-prompt route-contract checklist:
  - Stage 3 full rendered prompt reviewed from Rule 0 through Output format
    after the final minimal prompt clarification.
  - No retrieval: greetings, thanks, welcome-back, praise, and routine social
    acknowledgement remain empty unless durable evidence is required.
  - Live context: current time/date/weekday and current external facts remain
    `Live-context:`.
  - Recall: active agreements, promises, plans, open loops, and current
    episode state remain `Recall:`.
  - Person context: profile, impression, compatibility, relationship,
    ranking, user-list, and active-character self-profile remain
    `Person-context:`.
  - Conversation evidence: exact phrases, quoted messages, URLs,
    recent/fuzzy chat topics, speaker provenance, user-self recent words, and
    active-character self-words remain `Conversation-evidence:`.
  - Memory evidence scoped user continuity: current-user private continuity,
    recognition, accepted preferences, durable user memory facts, and prior
    shared interactions use `Memory-evidence:` with scoped current-user
    continuity wording.
  - Memory evidence shared durable facts: official character/world facts,
    common-sense facts, stable home/address/location facts, object/place/
    concept knowledge, tags, and memory identifiers remain `Memory-evidence:`
    without current-user continuity wording.
  - Web evidence: public web page/topic reads that are not current/live remain
    `Web-evidence:`.
  - Checklist result: pass; no source family blocked or unreviewed.
  - Prompt shape result: pass; no observed-failure pattern-gallery example was
    added, and the generation procedure remains unchanged.
- Live LLM positive gate result and inspected slot:
  - Stage 5 positive: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_remember_me_to_scoped_memory -m live_llm -q -s` passed.
    Inspected slot: `Memory-evidence: retrieve current-user private continuity and prior shared interactions with the active character`.
    Trace: `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__remember_me_to_scoped_memory.json`.
  - Post-minimal-prompt rerun passed. Inspected slot:
    `Memory-evidence: retrieve current-user private continuity and prior shared interactions with the active character`.
    Trace: `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__remember_me_to_scoped_memory__20260513T100537098848Z.json`.
- Live LLM negative gate results and inspected slots:
  - Stage 5 negative agreement: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_remember_today_agreement_stays_recall -m live_llm -q -s` passed.
    Inspected slot: `Recall: retrieve active_episode_agreement relevant to today's agreement`.
    Trace: `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__remember_today_agreement_stays_recall.json`.
  - Stage 5 negative recent current-user words: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_remember_recent_user_words_stays_conversation -m live_llm -q -s` passed.
    Inspected slot: `Conversation-evidence: retrieve recent messages from current user mentioning charging cables or contents of a pile of cables speaker=current_user`.
    Trace: `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__remember_recent_user_words_stays_conversation.json`.
  - Stage 5 negative active-character self-words: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_active_character_self_words_stays_conversation -m live_llm -q -s` passed.
    Inspected slot: `Conversation-evidence: retrieve prior active-character claim about the project being delayed speaker=active_character`.
    Trace: `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__active_character_self_words_stays_conversation.json`.
  - Stage 5 negative current-user URL chain: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_current_user_url_recall_stays_conversation -m live_llm -q -s` passed.
    Inspected slots: `Conversation-evidence: retrieve messages containing a URL speaker=current_user`; `Web-evidence: retrieve public web content for the URL found in slot 1`.
    Trace: `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__current_user_url_recall_stays_conversation.json`.
  - Stage 5 negative named-person impression: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_named_person_impression_stays_person_context -m live_llm -q -s` passed.
    Inspected slot: `Person-context: retrieve profile/impression for display name 小明`.
    Trace: `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__named_person_impression_stays_person_context.json`.
  - Stage 5 negative official address: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_official_address_stays_shared_memory_slot -m live_llm -q -s` passed.
    Inspected slot: `Memory-evidence: retrieve durable evidence about the active character's official address`.
    Trace: `test_artifacts\llm_traces\rag_phase3_initializer_live_llm__official_address_stays_shared_memory_slot.json`.
  - Post-minimal-prompt rerun of all negative gates passed one case at a time.
    Inspected slots remained in their expected families:
    `Recall: retrieve active_episode_agreement relevant to today's agreement`;
    `Conversation-evidence: retrieve recent messages from current user about charging cables speaker=current_user`;
    `Conversation-evidence: retrieve prior active-character claim about the project being delayed speaker=active_character`;
    `Conversation-evidence: retrieve messages containing a URL speaker=current_user`;
    `Web-evidence: retrieve public web content for the URL found in slot 1`;
    `Person-context: retrieve profile/impression for display name 小明`;
    `Memory-evidence: retrieve durable evidence about the active character's official address`.
- Documentation update:
  - Stage 4: `src/kazusa_ai_chatbot/rag/README.md` now states that
    current-user recognition, accepted preferences, user-specific lore, and
    prior shared interactions are scoped `user_memory_units` evidence under
    `Memory-evidence:` and are handled by `user_memory_evidence_agent`.
- Independent code review result:
  - Review mode: same-agent fresh-review posture; no separate reviewer was
    available because no explicit subagent delegation was requested.
  - Inputs reviewed: full plan after Stage 5, `git status --short`,
    `git diff --stat`, `git diff --check`, source/test/doc/plan diffs,
    full-prompt route checklist, and all live LLM inspected slots.
  - `git diff --check` result: no whitespace errors; only Windows
    line-ending warnings.
  - Finding 1: the historical `Independent Plan Review` section still said
    execution was blocked after the user had requested execution.
  - Fix 1: updated that section to distinguish the earlier draft-time review
    from the current `in_progress` execution status.
  - Follow-up source finding: deterministic memory-worker selection let a
    remember-me `original_query` satisfy scoped-user scope and topic checks for
    every `Memory-evidence:` slot in the same request. A separate official
    address slot could therefore route to `user_memory_evidence_agent`.
  - Fix 2: added
    `test_memory_evidence_shared_fact_ignores_remember_me_query_scope` and
    changed scoped-user topic detection to use current-slot text only. The
    original query can still provide supporting private scoped-continuity
    context, but it no longer turns unrelated shared-memory slots into
    remember-me lookups.
  - Post-fix targeted rerun:
    `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py::test_memory_evidence_remember_me_slot_uses_scoped_worker tests\test_rag_phase3_capability_agents.py::test_memory_evidence_official_character_fact_stays_shared_memory tests\test_rag_phase3_capability_agents.py::test_memory_evidence_shared_fact_ignores_remember_me_query_scope tests\test_rag_phase3_capability_agents.py::test_memory_evidence_old_setting_slot_uses_scoped_context -q`
    passed with `4 passed`.
  - Post-fix compile rerun:
    `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\memory_evidence_agent.py tests\test_rag_phase3_capability_agents.py`
    passed.
  - Post-fix focused regression batch:
    `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py tests\test_rag_initializer_cache2.py tests\test_user_memory_evidence_agent.py -q`
    passed with `87 passed`.
  - Residual risk: initializer behavior is still model-dependent for unseen
    phrasings, but the required real LLM positive and adjacent negative gates
    passed. Production logs should continue to be monitored for new phrasing
    gaps.
  - Approval status: passed after the follow-up routing-leak fix.
  - Post-review user correction: the user rejected negative-constraint and
    lookup-table prompt expansion. The initializer prompt was reduced to
    minimum source-ownership clarification in Rule 6, Rule 8, conflict
    resolution, and slot format only. Deterministic and live LLM gates were
    rerun after this reduction and passed.

## Independent Plan Review

- Review mode: same-agent fresh review requested by user on 2026-05-13.
- Inputs reviewed: this plan, `development_plans/README.md`,
  `development-plan-writing` references, `local-llm-architecture`, current
  `persona_supervisor2_rag_initializer.py`, and existing live initializer tests.
- Blocker found: the draft required only one real LLM positive initializer test
  for `你还记得我吗`; it did not require real LLM negative tests for adjacent
  source families, which is insufficient for a full prompt rewrite.
- Fix applied: added a required full-prompt route-contract checklist and a
  positive/negative live LLM validation matrix covering Recall,
  Conversation-evidence, Person-context, scoped Memory-evidence, and shared
  Memory-evidence routes.
- Non-blocking finding at plan-review time: the plan was `draft` and blocked
  until approved, which matched the registry lifecycle rule.
- Execution status update: after the user requested execution on 2026-05-13,
  the plan and registry were promoted to `in_progress`.
- Approval status: plan review passed after the fixes above; execution is now
  governed by the staged checklist and verification gates in this document.
