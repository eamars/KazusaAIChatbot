# character local time context plan

## Summary

- Goal: Ensure every LLM-visible time string is converted to character-local, timezone-unaware text, while every LLM-produced time intended for storage is converted back to UTC.
- Plan class: large.
- Status: draft.
- Mandatory skills: `py-style`, `cjk-safety`, `test-style-and-execution`, `local-llm-architecture`, `no-prepost-user-input`.
- Overall cutover strategy: bigbang for prompt-facing time presentation; compatible for UTC database storage; no historical data migration.
- Highest-risk areas: prompt payload consistency, conversation history timestamps, RAG helper result timestamps, and future-promise `due_time` normalization.
- Acceptance criteria: no UTC or timezone-aware timestamp is fed to any LLM payload; newly stored/scheduled LLM-derived times are UTC; existing database rows remain unchanged.

## Context

The service currently stores conversation and scheduler timestamps as UTC, which is correct for persistence. The problem is that the same UTC ISO strings are also sent into LLM prompts as `timestamp`, `current_timestamp`, chat-history row timestamps, RAG evidence timestamps, and consolidation reference times.

This causes natural-language time references such as `今天`, `明天`, `tomorrow`, `this afternoon`, and `一会儿` to be interpreted against UTC rather than the character's lived local clock. In the inspected QQ private chat for user `673225019`, the user and character were effectively speaking in a timezone about 12 hours away from UTC. Some stored active commitments ended up with natural dates that matched the local conversation, while scheduler events could still be created around UTC/past instants.

The confirmed product rule is:

- The LLM must not reason about timezones, offsets, UTC conversion, or IANA timezone names.
- Python must convert every time string before it reaches an LLM.
- Python must convert every LLM-produced time that affects storage or scheduling back to UTC.
- Old database data must not be converted. The rule applies only to new prompt payloads and new writes.

This plan focuses on the timezone/local-time boundary only. Scheduler date-only policy is deferred except where timezone conversion requires field-level UTC normalization.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python prompt files that contain Chinese text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing prompt, graph, RAG, cognition, dialog, evaluator, or background LLM behavior.
- `no-prepost-user-input`: load before changing future-promise, memory, or commitment persistence paths.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Do not make the LLM timezone-aware. Do not feed timezone names, UTC offsets, or offset-bearing ISO strings to LLM prompts unless a test proves the string is intentionally not model-facing.
- Every LLM-facing time string must be character-local and timezone-unaware, formatted consistently by Python.
- Conversation history timestamps, RAG conversation evidence timestamps, memory evidence timestamps, current time payloads, scheduler prompt payloads, and consolidation timestamps are all in scope when they are sent to an LLM.
- Database rows remain UTC. Do not rewrite, backfill, migrate, or normalize historical rows.
- LLM semantic decisions remain LLM-owned. Python may convert structured time fields to and from UTC, but must not keyword-match user text or override whether a promise, instruction, or commitment was accepted.
- Code may structurally validate known time fields such as `due_time`, `due_at`, `execute_at`, `from_timestamp`, and `to_timestamp`. It must not scan arbitrary natural language and invent times.
- Do not add response-path LLM calls.
- Do not solve the date-only scheduler behavior in this plan. Keep the implementation focused on timezone presentation and UTC conversion.
- For Python prompt files containing Chinese text, follow `cjk-safety` and avoid unsafe quote edits.

## Must Do

- Add a dedicated character-local time context module.
- Add character timezone configuration with a fixed MVP default of `Pacific/Auckland`.
- Convert the incoming UTC turn timestamp into a prompt-facing local, timezone-unaware time context before any LLM call sees it.
- Sanitize all prompt-facing chat-history rows so their `timestamp` fields are local and timezone-unaware.
- Sanitize all prompt-facing RAG/helper evidence rows containing timestamps before they are sent to an LLM.
- Replace LLM payload keys that currently expose raw UTC `timestamp` or `current_timestamp` with sanitized local values, while preserving UTC values only for deterministic code paths.
- Convert structured LLM-produced time fields back to UTC before storage, scheduler dispatch, or persistence metadata.
- Add deterministic tests that prove the boundary behavior without live LLM calls.
- Add prompt-render/static tests that fail if obvious UTC ISO strings leak into LLM payloads.

## Deferred

- Do not implement per-user timezone selection.
- Do not implement channel/group timezone selection.
- Do not add a user-facing timezone setup flow.
- Do not infer timezone from user text, platform locale, location, IP, weather city, or chat content.
- Do not migrate or rewrite old database data.
- Do not change scheduler date-only behavior beyond preserving UTC conversion for exact times.
- Do not redesign the promise schema beyond the fields needed to distinguish local prompt time from UTC storage time.
- Do not refactor unrelated RAG, cognition, dialog, scheduler, or memory architecture.

## Cutover Policy

Overall strategy: bigbang for LLM-facing time strings; compatible for UTC storage; no data migration.

| Area | Policy | Instruction |
|---|---|---|
| LLM-facing current time | bigbang | Replace raw UTC `timestamp` / `current_timestamp` in model payloads with character-local timezone-unaware strings in one cutover. |
| LLM-facing conversation history | bigbang | All history rows passed to prompts must expose sanitized local timestamps only. Do not preserve raw UTC aliases in prompt payloads. |
| LLM-facing RAG/helper evidence | bigbang | Any timestamp in evidence sent to LLMs must be local and timezone-unaware. Deterministic DB query code may keep UTC internally. |
| LLM output persistence | bigbang | Structured time fields produced by LLMs must be converted to UTC before storage/scheduling. |
| Database historical rows | compatible | Existing UTC rows remain unchanged and readable. No migration or backfill. |
| Public API request timestamp | compatible | Adapters may continue sending UTC or timezone-aware request timestamps. Service code normalizes before LLM use. |
| Scheduler date-only semantics | compatible | Existing behavior is not redesigned in this plan, except exact local times converted to UTC before scheduling. |
| Tests | bigbang | New tests assert the new prompt-facing time contract directly. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a compatibility shim for LLM-facing raw UTC timestamps.
- If an area is `bigbang`, rewrite the prompt payload path directly instead of preserving raw timestamp aliases for the model.
- If an area is `compatible`, preserve only the compatibility surface listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local helper names only when the public contracts in this plan remain intact.
- The agent must not introduce alternate timezone sources, user timezone inference, compatibility fallbacks, or extra features.
- The agent must not perform unrelated cleanup, dependency upgrades, prompt rewrites, or broad refactors.
- Changes outside the listed change surface require strong justification and must preserve this plan's intent.
- If equivalent time-formatting behavior already exists, move or wrap it into the new module instead of duplicating it.
- If the plan and code disagree, preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

All persisted timestamps remain UTC:

```json
{
  "timestamp": "2026-05-03T00:01:25.283613+00:00",
  "execute_at": "2026-05-03T02:00:00+00:00"
}
```

All LLM-visible time context is character-local and timezone-unaware:

```json
{
  "current_local_datetime": "2026-05-03 12:01",
  "current_local_weekday": "Sunday"
}
```

Do not add any other derived date or time aliases to the model-facing `time_context`. The LLM can derive those from `current_local_datetime` and `current_local_weekday`; Python remains responsible for converting the raw UTC turn timestamp into that local context.

LLM-visible history rows keep the existing shape but sanitized time values:

```json
{
  "role": "user",
  "display_name": "蚝爹油",
  "body_text": "你也记得一会儿多穿点别感冒了。我啥时候来接你呢？",
  "timestamp": "2026-05-03 12:00"
}
```

No LLM payload should contain:

```text
2026-05-03T00:00:03.036410+00:00
2026-05-03T00:00:03Z
+12:00
Pacific/Auckland
UTC
```

Exceptions are allowed only in non-model-facing logs/tests or when a test explicitly proves the value is not sent to an LLM.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Timezone source | Character timezone only for MVP | User timezone is unavailable and inference would be unreliable. |
| Config default | `CHARACTER_TIME_ZONE=Pacific/Auckland` | Matches the observed deployment need and supports DST via IANA rules. |
| LLM awareness | LLM receives local naive strings only | Reduces local-model burden and avoids offset math mistakes. |
| Storage | UTC remains canonical | Existing database design is already mostly UTC and operationally correct. |
| Old data | No migration | The new rule applies to new prompt payloads and new writes only. |
| History timestamps | Sanitize at projection boundary | Prevents raw UTC leaks across many prompts with one shared mechanism. |
| LLM output conversion | Convert known structured time fields by schema | Preserves LLM semantic decisions while making persistence deterministic. |
| Scheduler date-only issue | Deferred | User approved the policy direction but asked to focus on timezone first. |

## Contracts And Data Shapes

### New Module

Create `src/kazusa_ai_chatbot/time_context.py`.

Public interface:

```python
class TimeContextDoc(TypedDict):
    current_local_datetime: str
    current_local_weekday: str


def build_character_time_context(timestamp: str | None) -> TimeContextDoc: ...


def sanitize_timestamp_for_llm(timestamp: str | None) -> str: ...


def sanitize_history_for_llm(rows: list[dict]) -> list[dict]: ...


def sanitize_prompt_time_fields_for_llm(value: object) -> object: ...


def local_llm_time_to_utc_iso(value: str) -> str: ...


def local_date_bounds_to_utc_iso(local_date: str) -> tuple[str, str]: ...
```

Ownership:

- `time_context.py` owns timezone loading, local formatting, parsing local LLM time strings, and UTC conversion.
- Existing service, RAG, cognition, consolidation, and dispatcher code call the public functions.
- Callers must not import private timezone internals or call `zoneinfo.ZoneInfo` directly for prompt-facing time conversion.
- Raw UTC turn timestamps remain in the existing deterministic state key `timestamp`. They are not part of `TimeContextDoc` and must not be copied into prompt payloads.

Formatting contract:

- `TimeContextDoc.current_local_datetime`: `YYYY-MM-DD HH:MM`.
- `TimeContextDoc.current_local_weekday`: exact English full weekday from Python, for example `Sunday`; do not localize, translate, abbreviate, or include timezone.
- Sanitized prompt-facing timestamp values: `YYYY-MM-DD HH:MM`.
- Sanitized prompt-facing date-only values, when the source field is already date-only: `YYYY-MM-DD`.
- Output UTC storage format: timezone-aware ISO string with `+00:00`.

`sanitize_timestamp_for_llm` contract:

- Input `None` or empty string returns an empty string.
- Input offset-aware ISO 8601, including `Z`, is parsed as an instant, converted to the configured character timezone, and returned as `YYYY-MM-DD HH:MM`.
- Input already matching `YYYY-MM-DD HH:MM` is treated as already sanitized and returned unchanged.
- Input already matching `YYYY-MM-DD` is treated as an already sanitized local date and returned unchanged.
- Input naive ISO strings with `T`, offset-free seconds, slash dates, natural language, or any non-ISO/non-canonical format is invalid for this helper. Log a warning and return an empty string. Do not pass through inconsistent time text.

`sanitize_history_for_llm` contract:

- Return a deep-copied list; never mutate the caller's rows.
- Sanitize the exact key `timestamp` on each row by calling `sanitize_timestamp_for_llm`.
- Preserve all non-time fields exactly, including `body_text`, `raw_wire_text`, `display_name`, `role`, `reply_context`, and addressing fields.
- If a row has no `timestamp`, preserve the missing key as missing. Do not add synthetic time.

`sanitize_prompt_time_fields_for_llm` contract:

- Recursively traverse dictionaries and lists intended for model-facing payloads.
- Sanitize values for these exact keys only: `timestamp`, `created_at`, `updated_at`, `first_seen_at`, `last_seen_at`, `last_timestamp`, `from_timestamp`, `to_timestamp`, `expiry_timestamp`, `execute_at`, `due_at`, `due_time`, `completed_at`, `cancelled_at`, `current_timestamp`, `current_turn_timestamp`, `existing_updated_at`, `existing_last_seen_at`.
- Do not inspect or rewrite arbitrary string values, message bodies, summaries, facts, actions, or natural-language text.
- Do not add derived date aliases or raw UTC reference fields while sanitizing.

`local_llm_time_to_utc_iso` contract:

- Accept exactly `YYYY-MM-DD HH:MM`.
- Reject date-only strings, time-only strings, strings with seconds, strings containing `T`, timezone offsets, timezone names, `UTC`, `Z`, or natural language.
- Raise `ValueError` on rejected input. The caller must handle it according to that stage's existing structural failure policy.
- Interpret the accepted value as character-local time and return UTC ISO with `+00:00`.

`local_date_bounds_to_utc_iso` contract:

- Accept exactly `YYYY-MM-DD`.
- Return the UTC bounds for local start-of-day inclusive to next local start-of-day exclusive.
- Use this helper for date-window DB queries such as local `today` and `yesterday`.

Failure behavior:

- Invalid inbound UTC timestamp falls back to current UTC and logs a warning.
- Invalid LLM-produced local time raises `ValueError` at the structural conversion boundary.
- Conversion failures must be logged and the malformed field skipped or rejected according to the existing stage's structural failure behavior. Do not invent a replacement time.

### State Contract

Add `time_context: TimeContextDoc` to:

- `IMProcessState`
- `GlobalPersonaState`
- `CognitionState`
- `ConsolidatorState`

The raw `timestamp` may remain in state for deterministic code and storage, but it must not be inserted directly into LLM payload dictionaries.

The service must build `time_context` once from the queued turn timestamp and snapshot it into the graph state. Background consolidation must receive that same snapshot. Do not rebuild `time_context` at consolidation time, because asynchronous background execution may occur after the user-visible turn.

### Prompt Payload Contract

Any LLM handler that currently sends:

```json
{
  "timestamp": "UTC ISO string",
  "current_timestamp": "UTC ISO string"
}
```

must instead send:

```json
{
  "time_context": {
    "current_local_datetime": "2026-05-03 12:01",
    "current_local_weekday": "Sunday"
  }
}
```

Do not include raw UTC timestamps, IANA timezone names, offsets, or derived date aliases in this payload.

If a prompt's wording needs "system time", describe it as the active character's current local date/time, not UTC.

### Structured Time Output Contract

Known LLM output fields that represent absolute or relative-to-current local times must be converted before storage:

- `future_promises[*].due_time`
- `memory_units[*].due_at`, if added or emitted
- dispatcher raw tool-call `args.execute_at`, if the LLM has been changed to receive local prompt time

The implementation must preserve the model's semantic fields and only convert the structured time field value.

Only call `local_llm_time_to_utc_iso` for exact local datetime fields matching `YYYY-MM-DD HH:MM`. Date-only fields must stay date-only and must not be converted into a scheduled instant in this plan. Natural-language phrases such as `明天下午两点` must be resolved by the LLM into structured local fields first or treated as unresolved/invalid by the receiving stage.

## LLM Call And Context Budget

No new LLM calls are allowed.

Before:

- Existing response-path calls receive raw UTC timestamps in multiple payloads.
- Existing background consolidation and dispatcher calls receive raw UTC timestamps in multiple payloads.

After:

- Existing calls receive a compact local `time_context` block and sanitized history/evidence timestamps.
- Character count impact is neutral or slightly lower because long ISO strings with offsets are replaced by shorter local strings.
- Response-path latency must not increase except for deterministic Python formatting.

Context cap: use the existing project cap. This plan does not increase context limits or add retry loops.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/time_context.py`: public timezone conversion and prompt-time sanitization module.
- `tests/test_time_context.py`: deterministic tests for local formatting and UTC conversion.
- `tests/test_llm_time_payload_sanitization.py`: deterministic/patched tests that assert no LLM payload contains raw UTC timestamp strings.

### Modify

- `src/kazusa_ai_chatbot/config.py`: add `CHARACTER_TIME_ZONE` defaulting to `Pacific/Auckland`.
- `src/kazusa_ai_chatbot/state.py`: add `time_context` to `IMProcessState`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: add `time_context` to persona and cognition states.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`: add `time_context` to consolidator state.
- `src/kazusa_ai_chatbot/service.py`: build `time_context` once per queued turn and pass it into graph state; keep raw `timestamp` for storage.
- `src/kazusa_ai_chatbot/utils.py`: keep `trim_history_dict` as the raw compact history projection, and add or update callers so model-facing history always passes through the canonical `time_context.sanitize_history_for_llm` helper. Do not create a second history timestamp sanitizer.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: pass sanitized history and `time_context` through RAG, cognition, dialog, and consolidation state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`: include `time_context`; ensure `chat_history` timestamps are sanitized.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`: replace raw time payloads, if present, with `time_context`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`: replace raw time payloads and prompt wording with local time context.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`: replace raw time payloads and ensure content-anchor prompt uses local time context.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: ensure tone/history payload timestamps are sanitized if included.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`: pass `time_context` into background consolidation state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`: replace system-time wording and convert outgoing `future_promises[*].due_time` to UTC after parsing.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`: sanitize input history timestamps and convert any emitted structured due time to UTC before persistence.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`: use local prompt time for dispatcher LLM payloads; ensure accepted exact times are UTC before scheduler dispatch.
- `src/kazusa_ai_chatbot/rag/conversation_aggregate_agent.py`: compute `today` and `yesterday` bounds from local character dates using local `00:00` start-of-day converted to UTC; keep `recent` as a rolling duration ending at the turn timestamp unless an existing test proves a date-bound behavior is required.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`: sanitize evidence strings and structured timestamps before model-facing summary/finalizer payloads.
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py`: add sanitized projection helpers for any returned rows that are later sent to LLMs.
- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`: review because it receives history with timestamps; only sanitize the LLM payload after deterministic relevance windowing has used raw UTC.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`: review initializer, dispatcher, summarizer, and finalizer payloads for timestamp leaks.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`: review model payloads for current-turn timestamps.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`: review model payloads for session timestamps; do not migrate old image data.
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`: review generator/evaluator payloads for `from_timestamp` and `to_timestamp`; DB queries remain UTC.
- `src/kazusa_ai_chatbot/rag/conversation_keyword_agent.py`: review generator/evaluator payloads for `from_timestamp` and `to_timestamp`; DB queries remain UTC.
- `src/kazusa_ai_chatbot/rag/conversation_filter_agent.py`: review generator/evaluator payloads and examples for raw UTC timestamps.
- `src/kazusa_ai_chatbot/rag/web_search_agent.py`: replace prompt-facing current-time references with sanitized local time context while preserving external API behavior.
- `src/kazusa_ai_chatbot/rag/recall_agent.py`: keep structural expiry/activity checks on UTC internally, but sanitize any evidence timestamps before LLM-facing output.

### Static-Grep Findings Policy

If static grep finds a timestamp-bearing LLM handler not listed above:

- Record the file path, matched symbol, and whether the timestamp is model-facing in `Execution Evidence`.
- Do not modify the unlisted file unless the agent adds a short justification under `Execution Evidence` explaining why it is required by this plan's "any LLM-visible time" rule.
- If the unlisted file would require new architecture, new tools, or broad prompt redesign, stop and request a plan update instead of implementing.

### Keep

- MongoDB stored conversation timestamps remain UTC.
- Scheduler `scheduled_events.execute_at` remains UTC.
- Request/adapter timestamp contract remains accepted as UTC or timezone-aware input.
- Existing export scripts may keep raw UTC because they are diagnostic artifacts, not LLM prompt payloads.

## Implementation Order

1. Add deterministic tests for the new time-context module.
   - Expected initial result: missing module/import failure.
2. Add payload-sanitization tests around representative LLM payload builders.
   - Cover decontextualizer, cognition content anchors, facts harvester, dispatcher payload, and RAG aggregate bounds.
   - Expected initial result: raw UTC strings are present.
3. Implement `time_context.py` and config.
4. Run `pytest tests/test_time_context.py -q`.
5. Add `time_context` to state schemas and service initialization.
6. Sanitize shared prompt-facing conversation history projection.
7. Wire sanitized `time_context` through persona, cognition, dialog, consolidation, and dispatcher payloads.
8. Update RAG/helper evidence formatting and local-date query bound conversion.
9. Add LLM output UTC conversion for structured time fields.
10. Run focused tests and static greps.
11. Run a narrow service/pipeline smoke test or existing deterministic graph tests.
12. Record verification output in `Execution Evidence`.

Build the module first because it is the contract every prompt and persistence boundary must share.

## Progress Checklist

- [ ] Stage 1 - time context module contract and tests
  - Covers: `time_context.py`, `tests/test_time_context.py`.
  - Verify: `pytest tests/test_time_context.py -q`.
  - Evidence: record test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 2 - state and service wiring
  - Covers: `config.py`, `state.py`, persona/consolidator schemas, `service.py`.
  - Verify: targeted import/compile checks and focused state tests.
  - Evidence: record changed files and command output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 3 - prompt-facing history and RAG timestamp sanitization
  - Covers: `utils.py`, RAG helper/evidence modules, decontextualizer payloads.
  - Verify: payload-sanitization tests and static grep for raw UTC leaks.
  - Evidence: record grep and test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 4 - cognition/consolidation/dispatcher payload conversion
  - Covers: cognition L1/L2/L3, dialog, facts harvester, memory units, persistence dispatcher payloads.
  - Verify: prompt-render/payload tests pass; no raw UTC strings in captured LLM payloads.
  - Evidence: record test output and representative captured payload snippets.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 5 - LLM output UTC conversion
  - Covers: structured `due_time`, `due_at`, and `execute_at` conversion boundaries.
  - Verify: deterministic tests for local output to UTC storage fields.
  - Evidence: record test output.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 6 - final verification and smoke
  - Covers: all touched modules.
  - Verify: all commands in `Verification`.
  - Evidence: record all command results and any skipped checks with reasons.
  - Handoff: plan ready for completion report.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static Greps

Run:

```powershell
rg "current_timestamp|timestamp" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag
rg "UTC|\\+00:00|Z\"" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag
rg "time_context" src tests
```

Expected:

- Remaining raw `timestamp` references in LLM modules are either deterministic-only or pass through the sanitization module before prompt construction.
- No prompt payload builder directly inserts raw UTC timestamps into `HumanMessage`.
- `time_context` appears in state schemas, service wiring, prompt payload tests, and affected prompt handlers.

### Deterministic Tests

Run:

```powershell
pytest tests\test_time_context.py -q
pytest tests\test_llm_time_payload_sanitization.py -q
```

Required cases:

- `2026-05-03T00:00:03+00:00` renders as `2026-05-03 12:00` for `Pacific/Auckland`.
- History row timestamps are replaced with local naive strings.
- `sanitize_timestamp_for_llm(None)` and empty input return an empty string.
- `sanitize_timestamp_for_llm("2026-05-03 12:00")` returns the same canonical local string.
- `sanitize_timestamp_for_llm("2026-05-03")` returns the same canonical local date string.
- `sanitize_timestamp_for_llm("2026-05-03T12:00:00")` logs/handles invalid naive ISO input and returns an empty string.
- Local naive `2026-05-03 14:00` converts to `2026-05-03T02:00:00+00:00`.
- `local_llm_time_to_utc_iso("14:00")`, `local_llm_time_to_utc_iso("2026-05-03")`, `local_llm_time_to_utc_iso("明天下午两点")`, and offset-bearing inputs raise `ValueError`.
- Local date bounds for `2026-05-03` convert local `2026-05-03 00:00` through local `2026-05-04 00:00` into UTC start/end ISO strings.
- `sanitize_prompt_time_fields_for_llm` recursively sanitizes only approved time keys and leaves `body_text`, `fact`, `action`, and other natural-language fields unchanged.
- Captured representative LLM payloads contain no `+00:00`, `Z`, `UTC`, or timezone name.
- RAG "today" query bounds are computed from the character-local date and converted to UTC before DB query.

### Existing Focused Tests

Run focused tests covering touched areas:

```powershell
pytest tests\test_dispatcher.py -q
pytest tests\test_rag_phase3_capability_agents.py -q
pytest tests\test_user_memory_units_rag_flow.py -q
```

If unrelated failures appear, record them and do not expand scope unless they block the timezone change.

### Compile Check

Run:

```powershell
python -m compileall src\kazusa_ai_chatbot
```

Use the project venv if the local shell requires it.

## Acceptance Criteria

This plan is complete when:

- Every model-facing current-time payload uses `time_context` with local timezone-unaware strings.
- Every model-facing conversation-history timestamp is local and timezone-unaware.
- Every model-facing RAG/helper evidence timestamp is local and timezone-unaware.
- No LLM payload includes raw UTC ISO strings, timezone offsets, timezone names, or `UTC` labels unless a test proves the value is not model-facing.
- New structured LLM time outputs are converted to UTC before storage or scheduling.
- Existing historical database rows are not migrated or rewritten.
- Deterministic and focused integration tests listed in `Verification` pass or have documented unrelated failures.

## Data Migration

No data migration is allowed.

Existing rows in these collections remain unchanged:

- `conversation_history`
- `user_memory_units`
- `memory`
- `scheduled_events`
- `conversation_episode_state`
- any cache or profile collections containing timestamps

The new behavior applies only to:

- new prompt payloads built after implementation
- new LLM-derived time fields written after implementation
- new scheduler events created after implementation

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| A raw UTC timestamp leaks through a rarely used prompt | Add captured payload tests and static greps over all LLM modules | `test_llm_time_payload_sanitization.py` and greps |
| Deterministic DB code accidentally receives local naive timestamps | Keep raw `timestamp` in internal state; sanitize only prompt-facing projections | RAG aggregate bounds test and focused RAG tests |
| LLM output conversion changes semantic acceptance | Convert only schema-defined time fields; do not alter action/target/commitment text | Unit tests assert non-time fields pass through |
| Prompt files with Chinese text get quote corruption | Load `cjk-safety`; run compile and prompt-render tests | `compileall` and payload tests |
| Scheduler date-only bug remains | Explicitly deferred and documented | Future plan needed for date-only scheduling policy |

## Execution Evidence

- Static grep results:
- Deterministic test results:
- Focused integration test results:
- Compile result:
- Representative sanitized payload:
- Known skipped checks or unrelated failures:

## Glossary

- Character-local time: the active character's configured local clock, currently `Pacific/Auckland` in Python configuration.
- Timezone-unaware prompt time: local time rendered without offset, timezone name, or UTC marker, such as `2026-05-03 12:01`.
- UTC storage time: timezone-aware ISO string normalized to UTC, such as `2026-05-03T02:00:00+00:00`.
- Model-facing payload: any data structure serialized into a `HumanMessage`, `SystemMessage`, prompt string, or LLM repair/evaluator/generator call.
