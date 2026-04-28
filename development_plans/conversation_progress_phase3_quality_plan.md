# conversation progress phase3 quality plan

## Summary

- Goal: Trace and fix conversation-progress entry corruption at the source so stage payloads remain native Python objects or JSON, and new stored/projection entries keep the agreed native string-or-entry-object shapes.
- Plan class: medium
- Status: approved
- Overall cutover strategy: compatible
- Highest-risk areas: applying a bandage normalizer instead of fixing the source boundary, accidentally adding semantic filtering while fixing structural data shape, breaking old episode documents, and expanding Phase 3 into behavior redesign.
- Acceptance criteria: recorder prior-state payloads, recorder validation, repository writes, and prompt projection preserve native data-shape boundaries and cannot create or persist Python repr strings by coercing nested containers; existing public APIs and response behavior remain otherwise unchanged.

## Context

The QQ group `1082431481` audit found a concrete bug in `conversation_episode_state`: several entry-list fields contain stringified nested dictionaries instead of clean semantic text. Examples look like:

```python
{"text": "{'text': '{\\'text\\': ...}", "first_seen_at": "..."}
```

This is a data-shape bug, not a new conversation-behavior design problem. Phase 3 is therefore scoped to source tracing, structural cleanup, and regression protection only.

This plan intentionally excludes dynamic feedback, dynamic refusal, group-topic tracking, and broad conversation-behavior tuning. Those require a separate design discussion because they interact with user profile, user memories, relationship state, and roadmap-level feedback loops.

The architectural rule for this phase is the Phase 1 stage-boundary rule: data passed between stages must be either native Python objects with well-defined shape or valid JSON strings at LLM/message boundaries. Python repr strings of containers, such as `"{'text': ...}"`, are not an allowed intermediate format.

## Mandatory Rules

- Preserve the existing public facade: `load_progress_context(...)` and `record_turn_progress(...)`.
- Do not add a new public module, endpoint, graph stage, response-path LLM call, persistence collection, or schema field.
- Relevance remains independent. Do not pass `conversation_progress` or stored episode state into relevance.
- Do not add deterministic keyword matching, regex classifiers, or semantic gates over user or assistant natural-language text.
- The primary fix must trace and stop the source corruption. Do not rely on a broad normalizer that makes malformed data appear acceptable.
- Structural validation is allowed: type checks, text/list shape checks, timestamp handling, TTL handling, length caps after string validation, and guarded writes.
- The fix must not reinterpret user meaning. It may only enforce native shape at explicit boundaries.
- Do not create Python repr strings by coercing dict/list containers into text fields.
- Do not change Content Anchor behavior, Dialog Agent behavior, boundary-pressure policy, user-profile behavior, user-memory behavior, relationship insight behavior, or group-topic behavior in this phase.
- Keep malformed legacy documents readable; do not perform a bulk data migration.
- Python edits must follow project style, with focused helpers, typed contracts, and tests for every structural boundary.

## Must Do

- Add a source-trace test proving the corruption path: stored entry dict -> recorder prior state -> recorder returns dict/list item -> `_string_list`/`cap_text` turns it into `str(dict)`.
- Add a recorder-facing prior-state projection so the recorder receives clean text lists for prior entries, not raw stored entry dictionaries.
- Harden recorder-output validation so entry-list fields must be lists of strings. Dict/list items must fail validation instead of being converted with `str(...)`.
- Harden string-list validation for `assistant_moves`, `overused_moves`, and `next_affordances` so dict/list items fail validation instead of being converted with `str(...)`.
- Replace broad object-to-string entry/list persistence with strict string-only persistence.
- Add projection-time shape validation so legacy non-native stored values do not reach prompts.
- Add tests proving malformed nested recorder outputs are rejected before persistence and cannot store `str(dict)`.
- Add tests proving malformed legacy non-native shapes are suppressed before prompt projection.
- Add a focused audit check against the saved QQ-style malformed sample pattern.

## Deferred

- Do not create dynamic feedback or empathic-accuracy behavior in this plan.
- Do not implement dynamic refusal, push-pull, or boundary-pressure behavior changes.
- Do not strengthen Content Anchor anti-repeat behavior in this plan.
- Do not create group-level conversation-progress state.
- Do not redesign relevance, `channel_topic`, RAG2, Cache2, user profile, user memories, relationship insight, or the cognition graph.
- Do not migrate all historical episode documents.
- Do not rewrite Kazusa's general personality, style, or dialog generator prompt.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Public facade | compatible | Keep `load_progress_context(...)` and `record_turn_progress(...)` unchanged. |
| Stored document shape | compatible | Keep existing fields. Clean new writes; tolerate legacy malformed values. |
| Recorder prior-state payload | compatible | Convert storage-shaped entry objects into recorder-facing text lists before JSON serialization. |
| Recorder validation | compatible | Reject malformed output shape instead of coercing containers to strings. |
| Projection | compatible | Suppress malformed legacy entries before prompt injection. |
| Cognition/Dialog | unchanged | Do not change response-planning or final-wording behavior. |
| Relevance/group topic | unchanged | Do not add progress input or group-topic state. |

## Agent Autonomy Boundaries

- The agent may choose helper names and small mechanics only when they preserve the existing public contracts.
- The agent must not introduce a new enum, schema field, module boundary, persistence model, prompt-behavior feature, or feedback system.
- The agent must not solve boundary-pressure handling or anti-repeat behavior in this phase.
- The agent must not parse user meaning from stored text.
- If a legacy entry/list value is not in the native storage shape, drop it from prompt projection instead of trying to infer its meaning.
- The agent must trace every removed coercion to the source boundary it protected. Do not leave unexplained `str(...)` coercions on user/assistant/LLM natural-language fields.

## Target State

Newly recorded episode entries are clean:

```python
{"text": "user is asking about Kazusa's exclusive weapon", "first_seen_at": "2026-04-28T07:38:28Z"}
```

They must not look like:

```python
{"text": "{'text': '{\\'text\\': ...}", "first_seen_at": "..."}
```

Legacy malformed documents may remain in the database. This phase does not infer whether an already-stored string is semantically corrupted; that requires explicit data cleanup or migration. Projection only rejects non-native shapes such as dict/list values in string fields.

The recorder must receive prior entry state in a native, text-only shape:

```python
{
    "prior_episode_state": {
        "user_state_updates": ["clean text"],
        "open_loops": ["clean text"],
        "resolved_threads": ["clean text"],
        "avoid_reopening": ["clean text"],
        "assistant_moves": ["clean speech-act label"],
        "overused_moves": ["clean speech-act label"],
        "...": "other scalar fields"
    }
}
```

It must not receive prior entry lists in storage shape when those lists are expected to be copied back as text:

```python
{
    "user_state_updates": [
        {"text": "clean text", "first_seen_at": "2026-04-28T07:38:28Z"}
    ]
}
```

## Source Trace And Bandage Audit

The source corruption path is:

```text
Mongo stored state
  -> prior_episode_state contains entry objects: {"text": str, "first_seen_at": str}
  -> record_with_llm sends raw prior_episode_state to recorder prompt
  -> recorder prompt says "copy its text exactly"
  -> LLM sometimes returns the entire entry object or an already stringified entry object
  -> _string_list uses str(item), converting dict/list to Python repr text
  -> repository cap_text accepts object and uses str(value)
  -> stored state now contains "{'text': ...}" as text
  -> next recorder call receives the corrupted string and nests it again
```

Bandage-like methods or call sites to fix or constrain:

| Location | Current pattern | Risk | Phase 3 source fix |
|---|---|---|---|
| `record_with_llm(...)` | Sends raw `prior_episode_state` including entry dictionaries | LLM copies storage shape instead of text shape | Send recorder-facing prior state with entry lists as strings. |
| `_string_list(...)` | `str(item).strip()` for every list item | Converts dict/list into Python repr strings | Replace with strict string-only validation. |
| `_validated_label(...)` | `str(value).strip()` | Masks non-string enum values | Require string labels before enum validation. |
| Scalar recorder fields | `str(payload.get(...))` | Masks object/list/dict payloads in text fields | Require strings for text fields; allow empty default only when field is absent. |
| `cap_text(value: object, ...)` | Converts arbitrary objects with `str(value)` | Makes every caller a possible repr-string source | Use only after string validation, or split into string-only cap helper. |
| Repository `_cap_strings(...)` | Filters with `str(value).strip()` and calls `cap_text` | Can persist dict/list repr strings in move/affordance lists | Accept only string list items. |
| `preserve_first_seen_entries(...)` | Type says `list[str]` but trusts caller and calls `cap_text(raw_text)` | Can persist malformed recorder output if validation fails upstream | Assert/validate `raw_text` is string; fail/drop non-string values. |
| `prior_first_seen` map | `str(entry["text"])` and `str(entry["first_seen_at"])` | Can preserve malformed legacy keys | Use only entries whose `text` and `first_seen_at` are strings. |
| Projection `_project_entries(...)` | `cap_text(entry["text"])` and `str(entry.get(...))` | Can expose malformed legacy shapes to prompts | Project only entries whose `text` and timestamp are strings and whose timestamp is parseable. |
| Projection string-list fields | `cap_text(item)` and `if str(item).strip()` | Can expose legacy dict/list values | Project only string items. |
| Runtime telemetry | `str(document["continuity"])`, `str(document["status"])` | Low risk but masks invalid stored enum shape | Keep only if source validation exists; otherwise validate before telemetry when touched. |

Allowed conversions:

- JSON serialization/deserialization at the LLM boundary.
- String coercion for non-natural-language metadata only when the source type is already constrained or the conversion is explicitly metadata formatting.
- Capping/trimming after a field has been validated as a string.

Forbidden conversions:

- `str(dict)` or `str(list)` for any user/assistant/LLM natural-language field.
- Python repr strings created by `str(dict)` or `str(list)` in storage, prompt, or inter-stage payloads.

## System Contract Audit

| Stage | Current role | Phase 3 change |
|---|---|---|
| Relevance | Decide whether Kazusa should respond and provide existing group/channel context | No change. |
| Conversation Progress Loader | Load per-user/channel short-term state after relevance | No lifecycle change. |
| Projection | Convert stored state into prompt-facing progress | Add structural quarantine for malformed legacy shapes. |
| Recorder | Background LLM that updates short-term progress after final dialog | Receive a recorder-facing clean prior state and reject malformed output shape. |
| Persistence | Store short-term state with TTL and turn count | Persist only validated native strings in entry/list fields and guard stale writes as before. |
| Content Anchor/Dialog | Plan and voice the response | No change in this phase. |

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Entry corruption | Fix the source boundary first | The issue is created before persistence; projection quarantine alone is a bandage. |
| Recorder prior state | Send text-only entry lists to the recorder | Prevents the LLM from copying storage-entry objects back into output fields. |
| Recorder output validation | Reject non-string list items | Stops Python repr strings at the ingestion boundary. |
| Persistence validation | Store only strings for text/list fields | Prevents later code from converting containers into text. |
| Legacy malformed shapes | Drop non-native values from prompt projection | Bad old shapes should not poison prompts, and bulk migration is unnecessary. Already-stored strings are not classified by content. |
| Public interface | No public API or schema-field changes | Phase 3 is bug hardening of existing contracts. |
| Behavior tuning | Deferred | Dynamic feedback/refusal overlaps user profile and memory design and needs separate discussion. |
| Group topic | No change | Group topic remains owned by relevance/channel context, not per-user progress bugfixing. |

## Contracts And Data Shapes

No public interface changes.

Existing production callers continue to use:

```python
async def load_progress_context(
    *,
    scope: ConversationProgressScope,
    current_timestamp: str,
) -> ConversationProgressLoadResult: ...

async def record_turn_progress(
    *,
    record_input: ConversationProgressRecordInput,
) -> ConversationProgressRecordResult: ...
```

Internal source-fix contracts:

```python
def build_recorder_prior_state(
    prior_episode_state: ConversationEpisodeStateDoc | None,
) -> dict:
    """Return a recorder-facing prior state with entry lists as plain strings."""

def require_string_list(value: object, field_name: str) -> list[str]:
    """Return stripped string items or raise ValueError for non-string items."""

def project_clean_entries(
    entries: list[ConversationEpisodeEntryDoc],
    *,
    current_timestamp: str,
    limit: int,
) -> list[ConversationProgressEntry]:
    """Project only native string entry text into prompt-facing entries."""
```

These helpers are structural only. They must not decide what the text means.

Entry-list fields covered:

```text
user_state_updates
open_loops
resolved_threads
avoid_reopening
```

String-list fields covered:

```text
assistant_moves
overused_moves
next_affordances
```

String-list fields must reject dict/list values instead of storing `str(value)`. They must not classify natural-language string content.

## LLM Call And Context Budget

Context cap: 50k tokens.

| LLM call | Before | After | Path | Context impact |
|---|---:|---:|---|---|
| Relevance | unchanged | unchanged | response path | No progress input. |
| Recorder | 1 background call after responsive turn | 1 background call after responsive turn | background | Prior state shape becomes cleaner/smaller; no new inputs. |
| Content Anchor | unchanged | unchanged | response path | No prompt-behavior changes in this phase. |
| Dialog Agent | unchanged | unchanged | response path | No change. |

No new response-path LLM calls are allowed. The projected `conversation_progress` budget remains `<= 5000` chars.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Build recorder-facing prior state with text-only entry lists before JSON serialization.
  - Tighten structural validation so entry-list fields and string-list fields cannot become stringified dict/list values.
  - Clarify recorder prompt output shape for entry lists only.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py`
  - Store only validated native string entries and reject/drop malformed non-native values.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py`
  - Suppress malformed legacy entries before prompt projection.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py`
  - Narrow text-capping helpers so entry/list call sites cannot pass arbitrary objects.
- Tests under `tests/`
  - Add focused source-trace, validator, repository/build, and projection tests for malformed entry shapes.

### Keep

- Existing public facade.
- Existing DB collection and schema fields.
- Existing relevance inputs and outputs.
- Existing group/channel topic ownership.
- Existing Content Anchor and Dialog Agent behavior.
- Existing background recorder scheduling.

### Delete

- Nothing.

## Implementation Order

- [x] Checkpoint 1 — source trace and recorder-facing prior state.
  - Covers: reproducing the corruption path and adding a recorder-facing prior-state projection with entry lists as strings.
  - Verify: test proves raw stored entry dictionaries are not sent to the recorder prompt payload.
  - Evidence: record test output in `Execution Evidence`.
  - Next: Checkpoint 2.

- [x] Checkpoint 2 — strict recorder output validation.
  - Covers: string-only validation for entry-list, move-list, affordance-list, enum, and scalar text fields.
  - Verify: tests prove recorder output with dict/list items raises `ValueError` and cannot reach persistence.
  - Evidence: record focused test names and results.
  - Next: Checkpoint 3.

- [x] Checkpoint 3 — clean new writes.
  - Covers: repository document building and list capping without `str(dict)` style coercion.
  - Verify: tests prove non-string entries/lists cannot be persisted and native string entries preserve `first_seen_at`.
  - Evidence: record focused test names and results.
  - Next: Checkpoint 4.

- [x] Checkpoint 4 — protect projection from legacy malformed shapes.
  - Covers: malformed existing stored docs do not inject nested dict strings into prompts.
  - Verify: projection test with malformed non-native shapes passes; payload remains within budget.
  - Evidence: record projection output summary.
  - Next: Checkpoint 5.

- [x] Checkpoint 5 — static boundary checks.
  - Covers: no relevance, group-topic, Content Anchor, Dialog Agent, user-profile, or memory behavior change.
  - Verify: static greps and focused diff review.
  - Evidence: record grep results.
  - Next: Checkpoint 6.

- [x] Checkpoint 6 — bugfix sign-off.
  - Covers: deterministic tests and artifact inspection.
  - Verify: run all commands in `Verification`.
  - Evidence: save before/after artifact checks without overwriting Phase 2 traces and record final judgement.

## Verification

### Static Greps

- `rg "conversation_progress|conversation_episode_state" src/kazusa_ai_chatbot/nodes/relevance_agent.py` returns no matches.
- `rg "group_progress|group_episode|channel_episode" src/kazusa_ai_chatbot` returns no new production group-progress state symbols.
- `rg "future_promises|user_profile|user_profile_memories|relationship_insight|last_relationship_insight" src/kazusa_ai_chatbot/conversation_progress src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` shows no Phase 3 coupling to profile/memory/relationship systems.
- `rg "str\\(.*entry|str\\(.*item|str\\(.*payload|cap_text\\(.*item|cap_text\\(.*raw" src/kazusa_ai_chatbot/conversation_progress` must not show entry/list persistence converting structured values with `str(...)` or arbitrary-object capping.
- `rg "is_container_repr_text|is_clean_text" src/kazusa_ai_chatbot/conversation_progress` must return no matches; Phase 3 must fix source shape, not classify bad-looking strings.

### Tests

- `venv\Scripts\python.exe -m pytest tests/test_conversation_progress_flow.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_conversation_progress_runtime.py tests/test_conversation_progress_module_boundary.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_service_background_consolidation.py -q`

### Artifact Checks

- Export relevant `conversation_episode_state` rows after implementation.
- Confirm new/updated documents do not contain nested `"{'text':"` style corruption in entry fields.
- Confirm prompt-facing projection omits malformed non-native shapes. Already-stored strings are not classified by content.

## Acceptance Criteria

This plan is complete when:

- Recorder prior-state payload uses native text-only lists for entry fields.
- Recorder validation rejects non-string list items and non-string text fields before persistence.
- Repository writes cannot persist nested dict strings in entry/list fields.
- Legacy non-native entry/list shapes are suppressed before prompt projection.
- Existing public load/record API remains unchanged.
- Relevance remains free of conversation-progress input.
- No group-topic tracking state or group-topic persistence is introduced.
- No Content Anchor/Dialog behavior redesign is included.
- No dynamic feedback, dynamic refusal, user-profile, user-memory, or relationship-insight coupling is introduced.
- No new response-path LLM calls are introduced and prompt-facing progress remains within the existing cap.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Fix becomes a bandage normalizer | Trace and fix recorder prior-state/output source before projection quarantine | Source-trace tests |
| Dropping malformed legacy shapes loses useful context | Accept loss for non-native legacy shapes; do not infer semantics | Projection tests |
| Serialized container strings are unsafe to parse broadly | Do not parse or classify Python repr strings in module logic; stop creating them at source | Source-boundary tests |
| Scope expands into behavior tuning | Explicit Deferred and static/diff review | Checkpoint 5 |
| Prompt change increases latency/context | No response-path changes; same payload cap | LLM budget evidence |

## Execution Evidence

- Checkpoint 1: implemented `build_recorder_prior_state(...)`; test `test_recorder_prior_state_exposes_entry_text_lists_only` proves stored entry dictionaries are converted to native text lists before the recorder LLM payload.
- Checkpoint 2: tightened recorder validation; test `test_recorder_validator_rejects_container_items_before_persistence` proves dict/list items fail before persistence.
- Checkpoint 3: repository entry/list writes are string-only; test `test_repository_rejects_non_string_new_entries` proves non-string entry values cannot be persisted.
- Checkpoint 4: projection suppresses malformed legacy shapes; test `test_projection_suppresses_malformed_legacy_shapes` proves non-native values do not reach prompt-facing progress.
- Checkpoint 5: static boundary checks completed. Relevance has no `conversation_progress` or `conversation_episode_state` matches; no new `group_progress`, `group_episode`, or `channel_episode` production symbols; no new conversation-progress coupling to user profile, user memory, or relationship insight systems.
- Checkpoint 6: deterministic test gate and artifact inspection completed without overwriting Phase 2 traces.
- Static grep results:
  - `rg "conversation_progress|conversation_episode_state" src/kazusa_ai_chatbot/nodes/relevance_agent.py`: no matches.
  - `rg "group_progress|group_episode|channel_episode" src/kazusa_ai_chatbot`: no matches.
  - `rg "future_promises|user_profile|user_profile_memories|relationship_insight|last_relationship_insight" src/kazusa_ai_chatbot/conversation_progress src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`: matches only the pre-existing cognition L3 profile/relationship code, not `conversation_progress`.
  - `rg "str\\(.*entry|str\\(.*item|str\\(.*payload|cap_text\\(.*item|cap_text\\(.*raw" src/kazusa_ai_chatbot/conversation_progress`: no matches.
  - `rg "is_container_repr_text|is_clean_text" src/kazusa_ai_chatbot/conversation_progress`: no matches.
- Test results:
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\conversation_progress\policy.py src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\conversation_progress\repository.py src\kazusa_ai_chatbot\conversation_progress\projection.py src\kazusa_ai_chatbot\conversation_progress\runtime.py tests\test_conversation_progress_flow.py`: passed.
  - `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_flow.py -q`: 11 passed after removing content-pattern detection.
  - `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_runtime.py tests\test_conversation_progress_module_boundary.py -q`: 4 passed.
  - `venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py -q`: 4 passed.
- Artifact review:
  - Read-only export: `test_artifacts\conversation_progress_phase3_episode_state_after_20260428.json`, 8 live `conversation_episode_state` documents for QQ channel `1082431481`.
  - Legacy malformed summary: `test_artifacts\conversation_progress_phase3_episode_state_after_20260428_malformed_summary.json`, 23 malformed legacy items still present in live rows. This is expected because Phase 3 does not migrate historical data.
  - Earlier content-pattern projection summary is superseded by the source-boundary correction. Existing already-stringified legacy rows remain data-cleanup candidates; the module no longer tries to classify them by content.
- Final judgement: source corruption is blocked at recorder input/output and persistence boundaries; malformed non-native legacy shapes are ignored, while already-stored strings are left to explicit cleanup rather than heuristic classification.

## Glossary

- Recorder-facing prior state: a native Python object prepared for the recorder LLM where copyable entry fields are plain strings, not storage entry dictionaries.
- Malformed legacy shape: an existing stored entry/list value whose Python type violates the storage contract, such as a dict/list where a string is required.
- Dynamic feedback: future roadmap-level design that compares character output with later user input; explicitly out of scope for this bugfix phase.
