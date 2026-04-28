# native shape boundary hardening plan

## Summary

- Goal: Remove the identified bandage-style shape repairs in risks 1-7 with targeted bug fixes and unit tests, without adding new architecture.
- Plan class: medium
- Status: draft
- Overall cutover strategy: compatible
- Highest-risk areas: durable memory writes, RAG projection into cognition, and RAG helper argument validators.
- Acceptance criteria: the listed sites stop stringifying or reparsing corrupted native-shape payloads; malformed corner cases are covered by unit tests; JSON repair for raw LLM output remains unchanged.

## Context

This is a bug-fix plan, not a redesign. The problem is a repeated bad pattern:

```text
wrong native shape enters a workflow
  -> local code applies str(...), JSON-string reparsing, or empty fallback
  -> upstream contract failure is hidden
  -> corrupted text or misleading state reaches later stages
```

The fix is to remove that pattern from the specific risk sites. Do not add a shared validator module. Do not add a new abstraction layer. Do not broaden this to unrelated cleanup. Per later user direction, the repeated string-only helper may live as a small shared `text_or_empty(...)` utility in the existing utils module; this is not a new validation layer.

Risk 8, `parse_llm_json_output(...)`, is explicitly not a bug here. General JSON repair is required for raw LLM output. The bug is using local repair after JSON has already been parsed into workflow data.

## Mandatory Rules

- Do not add a new module, framework, shared validation package, schema system, or broad abstraction.
- The only shared helper allowed by user direction is `text_or_empty(...)` in the existing utils module.
- Do not remove or weaken `parse_llm_json_output(...)`, `repair_json`, or the JSON repair LLM fallback.
- Do not parse stringified dict/list payloads inside domain logic to recover state.
- Do not add content-pattern detectors such as `is_bad_string`, repr detectors, or natural-language string classifiers.
- Do not use `str(dict)`, `str(list)`, or `json.dumps(..., default=str)` to make workflow data fit.
- Normalize only at the local input/output boundary of the function being fixed. Inside the process, use typed local values.
- Use direct `isinstance(...)` checks and local helper functions only where they keep the fix smaller than repeated code.
- Execute stage by stage. No big batch edits across multiple risk areas.
- Each stage must have a corrupted-shape unit test and must be signed off before moving to the next stage.

## Must Do

- Risk 1: stop `user_profile_agent` from parsing stringified JSON in `known_facts`.
- Risk 2: stop diary normalization from converting non-string list items with `str(item)`.
- Risk 3: stop cognition L3 accepted preferences from converting non-string items with `str(item)`.
- Risk 4: stop consolidator memory persistence from converting diary/fact fields with `str(...)`.
- Risk 5: reduce in-process `_normalize_text`, `_as_dict`, and `_as_list` use in RAG projection by validating each fact row once locally.
- Risk 6: stop RAG helper arg normalizers from coercing dict/list LLM output fields into string tool args.
- Risk 7: stop RAG supervisor initializer/dispatcher normalization from stringifying slots, `agent_name`, or `task`.
- For each risk, add a unit test proving corrupted data entering that workflow does not become repr text or executable state.
- Preserve existing good-path behavior with positive tests.

## Deferred

- Do not create `boundaries/native_shape.py`.
- Do not redesign RAG2, Cache2, cognition, profile memory, relationship insight, or conversation progress.
- Do not change prompts except a tiny output-shape clarification if a failing test proves the model contract is ambiguous.
- Do not migrate database data.
- Do not refactor unrelated `str(...)` usage in adapters, logs, scripts, tests, cache keys, or CLI tools.
- Do not change general LLM JSON repair.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| JSON repair | compatible | Leave `utils.py` JSON repair unchanged. |
| Risk sites 1-7 | compatible | Replace local bandage coercion with local shape checks. |
| Public APIs | compatible | Keep existing graph, helper-agent, and DB interfaces. |
| Bad input behavior | compatible | Invalid malformed payloads are dropped, omitted, or marked unresolved according to the existing local failure style. |
| DB data | compatible | No migration. New writes must not create repr text. |

## Agent Autonomy Boundaries

- The agent may use the existing shared `text_or_empty(...)` helper for string-only boundary checks.
- The agent must not add a shared module or a general validation abstraction.
- The agent must not batch multiple checkpoints together.
- The agent must not invent new behavior when malformed input is found; use the existing local failure style for that stage.
- The agent must record test evidence before moving to the next checkpoint.

## Target State

Bad shape is handled locally and plainly:

```text
dict where string expected -> skipped, invalid args, or unresolved result
list item where string expected -> skipped, invalid args, or unresolved result
stringified JSON where native dict/list expected -> not reparsed
```

Good shape still works:

```text
list[str] -> stripped strings
dict raw_result -> projected facts
valid RAG tool args -> same DB/tool calls
raw malformed LLM JSON text -> still repaired by parse_llm_json_output(...)
```

## Change Surface

Modify only as needed:

- `src/kazusa_ai_chatbot/rag/user_profile_agent.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- RAG helper arg normalizers under `src/kazusa_ai_chatbot/rag/`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
- `src/kazusa_ai_chatbot/utils.py` for the shared `text_or_empty(...)` helper only
- Focused tests under `tests/`

Keep:

- `src/kazusa_ai_chatbot/utils.py`
- DB schemas and collections
- public graph/helper-agent function signatures

## LLM Call And Context Budget

No new LLM calls.

Context cap remains 50k tokens. This plan should not meaningfully increase prompt size. Any prompt wording change must be a small output-shape clarification, not a prompt rewrite.

## Implementation Order

- [x] Checkpoint 1 — risk 1, user profile known-facts recovery.
  - Add a unit test where `known_facts` contains a stringified JSON object with `global_user_id`; it must not resolve.
  - Add a positive unit test where native dict/list known facts still resolve.
  - Remove stringified JSON parsing from `user_profile_agent`.
  - Run only the risk-1 tests, then local user-profile-agent tests.
  - Record sign-off before Checkpoint 2.

- [x] Checkpoint 2 — risk 2, diary entry normalization.
  - Add a unit test where diary payload contains a dict/list item; it must not become repr text.
  - Add a positive unit test for valid string and `list[str]` diary payloads.
  - Replace `str(item)` conversion with string-only handling.
  - Run only the diary schema tests, then local consolidator schema tests.
  - Record sign-off before Checkpoint 3.

- [x] Checkpoint 3 — risk 3, accepted user preferences.
  - Add a unit test where `accepted_user_preferences` contains dict/list items; they must not become repr text.
  - Add a positive unit test for valid `list[str]`.
  - Replace item stringification with string-only filtering or local invalid-output handling.
  - Run only cognition L3 preference tests.
  - Record sign-off before Checkpoint 4.

- [x] Checkpoint 4 — risk 4, durable memory doc building.
  - Add unit tests where diary/fact/milestone fields contain dict/list values; they must not be persisted as repr text.
  - Add positive unit tests for valid diary/fact/milestone docs.
  - Remove `str(...)` coercion from durable memory content fields.
  - Run only consolidator persistence tests.
  - Record sign-off before Checkpoint 5.

- [x] Checkpoint 5 — risk 5, RAG projection.
  - Add unit tests where fact rows have dict/list `slot`, `agent`, `summary`, or wrong-shape `raw_result`; they must not be stringified into cognition payload.
  - Add positive tests for normal user profile, memory evidence, conversation evidence, and web evidence projection.
  - Validate each fact row once near the top of the loop and remove repeated in-process normalization where touched.
  - Run only RAG projection tests.
  - Record sign-off before Checkpoint 6.

- [x] Checkpoint 6 — risk 6, RAG helper args.
  - Work helper family by helper family, not all at once.
  - Add unit tests where string fields receive dict/list values and must not become repr strings in tool args.
  - Add positive tests for valid args.
  - Replace `str(raw_args.get(...))` with string-only checks.
  - Run the touched helper's tests before moving to the next helper.
  - Record sign-off before Checkpoint 7.

- [x] Checkpoint 7 — risk 7, RAG supervisor initializer/dispatcher.
  - Add unit tests where initializer slots contain dict/list values; they must not become repr slot strings.
  - Add unit tests where dispatcher `agent_name` or `task` is dict/list; they must not become executable work.
  - Add positive tests for valid slots and dispatch payloads.
  - Remove slot/task/agent stringification.
  - Run RAG supervisor2 boundary tests.
  - Record sign-off before Checkpoint 8.

- [x] Checkpoint 8 — final static review and regression.
  - Run targeted static greps for the risk sites.
  - Run all touched test files.
  - Run `tests/test_conversation_progress_flow.py` to make sure the earlier fix remains stable.
  - Record final evidence.

## Verification

### Stage-Gated Protocol

For every checkpoint:

1. Add the corrupted-shape unit test.
2. Run the checkpoint test and record the failure when practical.
3. Make only that checkpoint's code change.
4. Re-run the checkpoint test until it passes.
5. Run the checkpoint's local regression tests.
6. Record changed files and commands.
7. Move to the next checkpoint only after sign-off.

### Static Greps

Run these after each relevant checkpoint, scoped to the touched file:

- `rg "str\\(item\\)|str\\(slot\\)|str\\(raw_args|get\\([^\\n]*\\)\\.strip\\(\\)|str\\(entry\\.get|str\\(fact\\.get" <touched-file>`
- `rg "parse_llm_json_output\\(stripped\\)|startswith\\(\\\"\\{\\\"\\)|startswith\\(\\\"\\[\\\"\\)" src/kazusa_ai_chatbot/rag/user_profile_agent.py`
- `rg "json\\.dumps\\([^\\n]*default=str" <touched-file>` for prompt-facing files

Allowed hits must be explained in execution evidence. Log/cache/debug-only serialization can remain.

## Acceptance Criteria

- Risks 1-7 no longer hide bad native shapes with stringification or local reparsing.
- Every checkpoint has a corrupted-shape unit test and a positive good-path test.
- Each checkpoint is signed off before the next checkpoint is edited.
- No new shared module or architecture is added.
- JSON repair remains unchanged.
- No public API or DB schema break is introduced.

## Execution Evidence

To be filled during implementation:

- Checkpoint 1:
  - Corrupted-shape unit test: `tests/test_user_profile_agent.py::test_extract_global_user_id_does_not_parse_stringified_known_facts`.
  - Pre-fix result: failed; `_extract_global_user_id_from_known_facts(...)` parsed the stringified JSON and returned `stringified-user-id`.
  - Post-fix result: passed.
  - Local regression result: `venv\Scripts\python.exe -m pytest tests\test_user_profile_agent.py -q` passed, 5 tests.
  - Static grep result: no `parse_llm_json_output(stripped)`, `startswith("{")`, `startswith("[")`, or `str(value.get("global_user_id"...))` recovery remains in `user_profile_agent.py`.
  - Changed files: `src/kazusa_ai_chatbot/rag/user_profile_agent.py`, `tests/test_user_profile_agent.py`.
- Checkpoint 2:
  - Corrupted-shape unit test: `tests/test_persona_supervisor2_schema.py::test_normalize_diary_entries_does_not_stringify_container_items`.
  - Pre-fix result: failed; dict/list diary items became Python repr strings.
  - Post-fix result: passed.
  - Local regression result: `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_schema.py -q` passed, 9 tests.
  - Static grep result: no `str(item)` remains in `persona_supervisor2_consolidator_schema.py`.
  - Changed files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`, `tests/test_persona_supervisor2_schema.py`.
- Checkpoint 3:
  - Corrupted-shape unit test: `tests/test_cognition_preference_adapter.py::test_preference_adapter_does_not_stringify_container_items`.
  - Pre-fix result: failed; dict/list preference items became Python repr strings.
  - Post-fix result: passed.
  - Local regression result: `venv\Scripts\python.exe -m pytest tests\test_cognition_preference_adapter.py -q` passed, 2 tests.
  - Static grep result: no `accepted_user_preferences.*str(` or `str(item)` remains in `persona_supervisor2_cognition_l3.py`.
  - Changed files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`, `tests/test_cognition_preference_adapter.py`.
- Checkpoint 4:
  - Corrupted-shape unit test: `tests/test_user_profile_memories.py::test_build_memory_docs_does_not_stringify_malformed_content_fields`.
  - Pre-fix result: failed; malformed diary/fact/commitment fields became durable repr-text content.
  - Post-fix result: passed.
  - Local regression result: `venv\Scripts\python.exe -m pytest tests\test_user_profile_memories.py::test_build_memory_docs_does_not_stringify_malformed_content_fields tests\test_user_profile_memories.py::test_build_memory_docs_stores_milestone_facts_only_once tests\test_db_writer_cache2_invalidation.py -q` passed, 4 tests.
  - Static grep result: no `str(...)` coercion over entry/fact/commitment/promise/raw_fact remains in `persona_supervisor2_consolidator_persistence.py`.
  - Changed files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`, `tests/test_user_profile_memories.py`.
- Checkpoint 5:
  - Corrupted-shape unit test: `tests/test_rag_projection.py::test_project_known_facts_does_not_stringify_malformed_fact_values`.
  - Pre-fix result: failed; malformed slot/summary/content/raw_result values became repr text in projected cognition payload.
  - Post-fix result: passed.
  - Local regression result: `venv\Scripts\python.exe -m pytest tests\test_rag_projection.py tests\test_persona_supervisor2_rag2_integration.py -q` passed, 5 tests.
  - Static grep result: no `_normalize_text` or stringification over slot/agent/summary/raw_result/content remains in `persona_supervisor2_rag_projection.py`.
  - Changed files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`, `tests/test_rag_projection.py`.
- Checkpoint 6:
  - Corrupted-shape unit tests: `tests/test_rag_helper_arg_boundaries.py` conversation, persistent-memory, user-list, and relationship arg-boundary tests.
  - Pre-fix result: conversation and persistent-memory malformed-field tests failed with repr strings in normalized args; user-list failed on malformed `display_name_value`; relationship already returned invalid for malformed enum values but still used local stringification internally.
  - Post-fix result: passed.
  - Local regression result: `venv\Scripts\python.exe -m pytest tests\test_rag_helper_arg_boundaries.py -q` passed, 12 tests.
  - Static grep result: RAG helper `_normalize_args(...)` and `_normalize_relationship_args(...)` no longer stringify `raw_args` fields or `raw_val` values. Remaining hits in touched files are judge feedback or runtime context handling, not the helper arg normalizers covered by this checkpoint.
  - Changed files: `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`, `src/kazusa_ai_chatbot/rag/conversation_keyword_agent.py`, `src/kazusa_ai_chatbot/rag/conversation_filter_agent.py`, `src/kazusa_ai_chatbot/rag/conversation_aggregate_agent.py`, `src/kazusa_ai_chatbot/rag/persistent_memory_search_agent.py`, `src/kazusa_ai_chatbot/rag/persistent_memory_keyword_agent.py`, `src/kazusa_ai_chatbot/rag/user_list_agent.py`, `src/kazusa_ai_chatbot/rag/relationship_agent.py`, `tests/test_rag_helper_arg_boundaries.py`.
- Checkpoint 7:
  - Corrupted-shape unit tests: `tests/test_rag_initializer_cache2.py::test_normalize_initializer_slots_does_not_stringify_container_items` and `tests/test_rag_initializer_cache2.py::test_normalize_dispatch_does_not_stringify_task_or_agent`.
  - Pre-fix result: failed; container slots became repr slot strings, and dict `task` became repr executable work.
  - Post-fix result: passed.
  - Local regression result: `venv\Scripts\python.exe -m pytest tests\test_rag_initializer_cache2.py -q` passed, 6 tests; `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_rag2_integration.py -q` passed, 1 test.
  - Static grep result: no `_text_or_empty(...)` duplicates remain; no `str(slot)`, `str(raw_dispatch...)`, `str(...agent_name...)`, or `str(...task...)` dispatcher coercion remains in `persona_supervisor2_rag_supervisor2.py`.
  - Shared helper adjustment: by user direction, repeated local `_text_or_empty(...)` helpers were consolidated into `src/kazusa_ai_chatbot/utils.py::text_or_empty(...)`.
  - Changed files: `src/kazusa_ai_chatbot/utils.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`, `tests/test_rag_initializer_cache2.py`, plus prior touched boundary files updated to import `text_or_empty(...)`.
- Checkpoint 8:
  - Static grep result: no matches for duplicated `_text_or_empty(...)`, user-profile stringified-JSON recovery, targeted `str(slot)`, targeted dispatcher `str(raw_dispatch...)`, `str(entry.get...)`, `str(fact.get...)`, or `json.dumps(..., default=str)` risk-site patterns in the scoped files.
  - Shared-helper grep result: `text_or_empty(...)` is defined once in `src/kazusa_ai_chatbot/utils.py` and imported by the touched boundary files that need string-only boundary handling.
  - Compile result: `venv\Scripts\python.exe -m py_compile ...` over touched source and test files passed.
  - Regression result: `venv\Scripts\python.exe -m pytest tests\test_user_profile_agent.py tests\test_persona_supervisor2_schema.py tests\test_cognition_preference_adapter.py tests\test_user_profile_memories.py tests\test_db_writer_cache2_invalidation.py tests\test_rag_projection.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_helper_arg_boundaries.py tests\test_rag_initializer_cache2.py tests\test_conversation_progress_flow.py -q` passed, 70 tests.
- Final judgement:
  - Risks 1-7 are addressed by targeted boundary bug fixes and corrupted-shape unit tests.
  - Raw LLM JSON repair remains unchanged.
  - Malformed parsed workflow data is no longer locally repaired by reparsing stringified JSON or by converting containers into repr text at the risk sites.
  - The only new shared code is the user-requested `text_or_empty(...)` helper in the existing utils module; no new validation module or architecture was added.
