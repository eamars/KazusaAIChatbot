# live context runtime facts plan

## Summary

- Goal: Make RAG2 answer present-tense runtime questions such as current time, date, and weekday through the existing `Live-context:` capability, without adding a new top-level RAG capability or exposing timezone mechanics to the LLM.
- Plan class: large
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: compatible extension of `Live-context:`; no database migration; no new graph stage; initializer cache version bump for prompt contract changes.
- Highest-risk areas: broadening `Live-context` into a generic context agent, confusing character-local time with user-local or explicit-place time, stale initializer cache entries routing current-time questions as missing-location external facts, leaking UTC/timezone strings back into model-facing payloads, and adding new provider branches before the existing external-live path has an explicit top-down source-class structure.
- Acceptance criteria: `LiveContextAgent` is first refactored into explicit top-down source branches without behavior change; current time/date/weekday questions then route to `Live-context:`, runtime-backed facts answer from deterministic current-turn state, external live facts still use the existing external lookup path, and no durable memory/history/profile/promise source is read for runtime facts.

## Context

The character-local time work already introduced compact LLM-facing time context:

```json
{
  "current_local_datetime": "2026-05-03 14:53",
  "current_local_weekday": "Sunday"
}
```

The current RAG2 behavior still has a routing gap. A user asking `现在几点` can be initialized as:

```text
Live-context: answer current time for unknown location
```

`live_context_agent` then treats this like a location-scoped external live fact and returns missing `location`. That is wrong for character-local current time, because Python already has the character-local current time in the turn state.

This plan keeps the Phase 3 top-level capability set intact. It does not add `Runtime-context:`. Instead, it refines `Live-context:` so it owns present-tense facts from two source classes:

- current-turn runtime facts already available in process state,
- external live facts that require bounded lookup providers.

`runtime_snapshot` is an internal Python term only. The LLM must not need to know or emit that label.

## Mandatory Skills

- `local-llm-architecture`: load before changing RAG2 initializer prompt, capability boundaries, dispatcher contracts, or helper-agent behavior.
- `py-style`: load before editing Python production or test files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files that contain Chinese examples or prompt text.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Do not add a new top-level `Runtime-context:` capability, dispatcher prefix, graph stage, cache namespace, database collection, background task, or scheduler behavior.
- Keep the existing RAG2 stage contract: initializer emits semantic slots; dispatcher routes by prefix; `LiveContextAgent` owns low-level source selection; deterministic code validates and executes.
- Do not make the initializer emit Python source names such as `runtime_snapshot`, `runtime_snapshot_provider`, `time_context`, `current_timestamp`, or `UTC`.
- Do not expose UTC timestamps, timezone names, timezone offsets, or IANA timezone IDs to any LLM payload for this feature.
- Do not make the LLM timezone-aware. The LLM sees only sanitized character-local strings such as `current_local_datetime` and `current_local_weekday`.
- Do not add deterministic keyword or regex classification over raw user text outside the initializer or specialist LLM extractors. Deterministic parsing of the structured `Live-context:` slot text inside `LiveContextAgent` is allowed.
- Before adding runtime-backed fact types, runtime provider logic, or new live-provider behavior, refactor `LiveContextAgent` into explicit top-down source-class branches while preserving current external live behavior.
- The branch refactor must be behavior-preserving. Existing explicit-location, character-location, user-location, missing-location, and opening-status tests must pass before runtime facts are implemented.
- Any helper extracted during the branch refactor must isolate existing non-trivial behavior or source-class routing. Do not add thin wrappers around single field access or simple expressions.
- `LiveContextAgent` must not read durable memory, chat history, promises, profiles, relationship state, or conversation progress for runtime-backed facts.
- `LiveContextAgent` may continue using existing memory/conversation workers only for target/scope lookup for external live facts, such as character-local weather or user-local temperature.
- Current time/date/weekday must not require `location`.
- `LiveContextAgent` must treat legacy or drifted structured slots such as `Live-context: answer current time for unknown location` as runtime-backed `current_time`, not as a missing-location external lookup. This is defensive compatibility for old cache traces or local-LLM drift; the initializer prompt must still stop teaching that slot form.
- Character-local current time/date/weekday, user-local current time/date/weekday, and explicit-place current time/date/weekday are different scopes. Do not collapse them into one runtime answer.
- Explicit-place current time/date/weekday, such as "Auckland now" or "Beijing today", is an external live lookup when the place is explicit. It must not use the character-local runtime provider.
- Current user local time/date/weekday may only be answered from a future sanitized `user_time_context` already present in runtime state. If absent, return unresolved with `missing_context=["user_time_context"]`, not `location`.
- User-local time is out of scope unless a sanitized user-local time context already exists in runtime state. Do not infer user timezone from text, location, locale, platform, IP, weather city, or chat history.
- Top-level capability agents remain uncached in v1. The runtime-backed branch must report the existing no-cache reason.
- Any initializer prompt change must bump `INITIALIZER_PROMPT_VERSION`.
- Any dispatcher-visible roster/prefix change would require an agent registry version bump, but this plan does not add a new prefix or top-level agent.
- Runtime prompt-render checks are required for every changed prompt string that is formatted with `.format(...)`, f-strings, or JSON examples.
- Preserve INFO observability for `live_context_agent output` with `resolved`, `primary_worker`, `missing_context`, `selected_summary`, and cache reason.

## Must Do

- First, refactor `src/kazusa_ai_chatbot/rag/live_context_agent.py` so `LiveContextAgent.run(...)` makes a top-down source-class decision before target resolution:
  - `runtime_snapshot` branch for active-character runtime facts, added later in this plan,
  - `external_live_lookup` branch for the existing target-resolution and web-delegation behavior.
- During the refactor stage, do not add new runtime fact types, runtime provider behavior, prompt examples, or initializer version changes.
- Rename or clarify the existing selector prompt/handler contract as external-live selection only. The current selector must not become responsible for runtime-backed facts.
- Preserve the existing external live behavior through the refactor: explicit target goes directly to web, active character location uses memory only for target/scope, current user location uses recent conversation only for target/scope, and unknown external live facts return missing `location` or `target`.
- Refine the `Live-context:` prompt contract so current time, date, and weekday are explicitly live-context facts.
- Update initializer examples so current-time questions emit one of the approved runtime-backed live slots:
  - `Live-context: answer active character current local time`
  - `Live-context: answer active character current local date`
  - `Live-context: answer active character current local weekday`
- Extend `LiveContextAgent` structured slot parsing with runtime-backed fact types:
  - `current_time`
  - `current_date`
  - `current_weekday`
- Add a deterministic runtime facts provider inside `live_context_agent.py`.
- Answer runtime-backed facts from `context["time_context"]` only.
- Add defensive handling so legacy `current time/date/weekday for unknown location/target` slots answer from runtime state instead of returning `missing_context=["location"]`.
- Keep explicit-place current time/date/weekday on the external live lookup path.
- Return unresolved `user_time_context` for current-user local time/date/weekday when no user-local context exists.
- Return standard top-level capability output shape, with `primary_worker` set to `runtime_context_provider`.
- Keep external live facts on the existing web/target-resolution path.
- Add deterministic tests for current time/date/weekday and for the no-location requirement.
- Add initializer prompt/cache tests covering the new examples and version bump.

## Deferred

- Do not implement per-user timezone support.
- Do not implement user timezone inference.
- Do not add a user-facing timezone setting flow.
- Do not add a separate `Runtime-context:` top-level capability.
- Do not expand runtime-backed facts beyond current time/date/weekday in this plan.
- Do not change scheduler date-only behavior.
- Do not modify promise, memory, profile, relationship, conversation history, or conversation progress schemas.
- Do not redesign cognition projection unless existing `rag_result.answer` and existing live-context projection prove insufficient in tests.

## Cutover Policy

Overall strategy: compatible extension of `Live-context:`.

| Area | Policy | Instruction |
|---|---|---|
| Top-level RAG capability set | compatible | Keep `Live-context:` as the only top-level capability for present-tense facts. |
| Current time/date/weekday routing | bigbang | After the prompt version bump, initializer output must use runtime-backed `Live-context:` slots, not missing-location external slots. |
| External live facts | compatible | Weather, temperature, opening status, schedule, price, exchange rate, and current public status keep the existing external live lookup behavior. |
| LLM-facing time strings | bigbang | Runtime-backed answers use only `current_local_datetime` and `current_local_weekday` derived by Python. |
| Database data | compatible | No data migration or historical rewrite. |
| Initializer cache | bigbang | Bump initializer prompt version so stale current-time routing does not persist. |

## Agent Autonomy Boundaries

- The agent may choose private helper function names, but the public behavior and output contracts in this plan are fixed.
- The agent must not add a new top-level capability or dispatcher prefix.
- The agent must not broaden `LiveContextAgent` into a generic system-state, memory, profile, promise, or conversation agent.
- If `context["time_context"]` is missing or malformed, return unresolved with `missing_context=["time_context"]`; do not fall back to UTC, `datetime.now()`, DB, memory, or web.
- If tests show cognition cannot consume the runtime-backed answer through existing RAG projection, stop and update this plan before editing cognition.
- If static grep finds additional current-time prompt examples, update them only when they are part of RAG2 initializer/live-context contract; do not refactor unrelated prompts.

## Target State

For character-local time:

```text
User: 现在几点？
Initializer: Live-context: answer active character current local time
Dispatcher: live_context_agent
LiveContextAgent source: runtime_context_provider
Selected summary: 当前本地时间是 2026-05-03 14:53，星期日。
```

For character-local weekday:

```text
User: 今天星期几？
Initializer: Live-context: answer active character current local weekday
Dispatcher: live_context_agent
LiveContextAgent source: runtime_context_provider
Selected summary: 当前本地日期是 2026-05-03，星期日。
```

For current user local time without configured user time context:

```text
User: 我这边现在几点？
Initializer: Live-context: answer current user local time if configured
Dispatcher: live_context_agent
LiveContextAgent source: runtime_context_provider
Result: unresolved with missing_context=["user_time_context"]
```

For explicit-place current time:

```text
User: 奥克兰现在几点？
Initializer: Live-context: answer current time for explicit location Auckland
Dispatcher: live_context_agent
LiveContextAgent source: external live lookup
Result: resolved from external evidence or unresolved with provider failure
```

For external live facts:

```text
User: 现在多少度？
Initializer: Live-context: answer current temperature for unknown location
Dispatcher: live_context_agent
LiveContextAgent source: external live lookup
Result: unresolved with missing_context=["location"]
```

For a stale or drifted current-time slot:

```text
Slot: Live-context: answer current time for unknown location
Dispatcher: live_context_agent
LiveContextAgent source: runtime_context_provider
Result: resolved from context["time_context"]; no location is requested
```

No LLM-facing payload for this feature may contain:

```text
runtime_snapshot
UTC
Pacific/Auckland
+12:00
2026-05-03T02:53:00+00:00
```

`runtime_context_provider` may appear in deterministic result metadata, logs,
and raw capability results. It must not be taught as initializer vocabulary,
slot text, prompt instruction text, or a model-selected route.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Top-level capability | Use existing `Live-context:` | Preserves the completed Phase 3 top-level capability set. |
| Runtime source name | Keep `runtime_snapshot` Python-only | Local LLM should reason over facts, not source mechanics. |
| Model-facing fields | Use existing `time_context` fields only | Keeps prompt time compact and timezone-unaware. |
| Runtime-backed scope | Active character current time/date/weekday only | Solves the observed bug without creating a generic context agent or answering user/place time incorrectly. |
| Branching strategy | Refactor to top-down source-class branching before adding providers | Keeps future live-provider additions from accumulating inside target-resolution conditionals. |
| Missing state behavior | Return unresolved `time_context` | Avoids accidental UTC fallback or machine-local clock drift. |
| User-local time | Deferred unless already present | User timezone is not reliably available and must not be inferred. |
| Explicit-place time | External live lookup | Mapping arbitrary place names to timezones is not owned by the LLM or runtime provider. |
| External live facts | Keep existing path | Prevents regression for weather, temperature, opening status, prices, schedules, and live public status. |

## Contract

### Source Classes

`Live-context:` owns present-tense facts for the current response from exactly these source classes:

| Source class | Meaning | IO allowed |
|---|---|---|
| `runtime_snapshot` | Internal Python-only name for current-turn facts already available in process state | No |
| `external_live_lookup` | External present-tense facts fetched for this turn through bounded providers | Yes |

The LLM is not taught these source class names. They are implementation and logging concepts.

### Internal Branch Contract

`LiveContextAgent.run(...)` must follow this high-level control flow after the
behavior-preserving refactor:

```text
parse structured Live-context slot into a normalized plan
choose source_class
if source_class == runtime_snapshot:
    resolve from sanitized current-turn runtime state
if source_class == external_live_lookup:
    resolve target/scope if needed
    delegate live value lookup to the external provider path
```

The external branch owns the current target-resolution behavior. Runtime-backed
facts must not enter the external target-resolution branch.

### Runtime-Backed Slots

The initializer may emit only these runtime-backed slot forms:

```text
Live-context: answer active character current local time
Live-context: answer active character current local date
Live-context: answer active character current local weekday
```

The initializer must not emit:

```text
Live-context: answer current time for unknown location
Live-context: answer current date for unknown target
Live-context: answer runtime_snapshot
Runtime-context: answer current local time
```

If a stale cache entry, older test fixture, or local-LLM drift still supplies a
structured slot containing `current time`, `current date`, or `current weekday`
with `unknown location` or `unknown target`, `LiveContextAgent` must normalize it
to the matching runtime-backed fact type before target/location planning.

### Time Scope Matrix

| User asks about | Example | Approved source |
|---|---|---|
| Bare current time/date/weekday | `现在几点？`, `今天星期几？` | active character runtime `time_context` |
| Character-side current time/date/weekday | `你那边现在几点？` | active character runtime `time_context` |
| User-side current time/date/weekday | `我这边现在几点？` | future `user_time_context` only; unresolved if absent |
| Explicit place current time/date/weekday | `奥克兰现在几点？` | external live lookup |
| Relative date for history query | `我昨天说了什么？` | `Conversation-evidence` using local date bounds |
| Active agreement or promise time | `今天的约定是什么？`, `啥时候来接你？` | `Recall`, not `Live-context` |
| Scheduled future action creation | `一分钟之后发消息` | consolidation/dispatcher scheduler path, not RAG2 live context |

Do not infer user timezone from location, weather city, locale, or chat history.
If a user asks for their own local time and no `user_time_context` exists, the
system may ask for clarification in the final response, but RAG must not return
`missing_context=["location"]` for that case.

### Runtime Provider Input

`LiveContextAgent` reads:

```python
context["time_context"]["current_local_datetime"]
context["time_context"]["current_local_weekday"]
```

Accepted input shape:

```python
{
    "current_local_datetime": "YYYY-MM-DD HH:MM",
    "current_local_weekday": "Sunday",
}
```

The provider must reject missing, non-string, empty, timezone-bearing, or offset-bearing values as malformed runtime state.

### Runtime Provider Output

Runtime-backed answers use the existing top-level capability result shape:

```python
{
    "resolved": True,
    "result": {
        "capability": "live_context",
        "primary_worker": "runtime_context_provider",
        "supporting_workers": [],
        "source_policy": "current-turn runtime state",
        "selected_summary": "当前本地时间是 2026-05-03 14:53，星期日。",
        "resolved_refs": [],
        "projection_payload": {
            "external_text": "当前本地时间是 2026-05-03 14:53，星期日。",
            "url": "",
        },
        "worker_payloads": {
            "runtime_context_provider": {
                "current_local_datetime": "2026-05-03 14:53",
                "current_local_weekday": "Sunday",
            },
        },
        "evidence": ["当前本地时间是 2026-05-03 14:53，星期日。"],
        "missing_context": [],
        "conflicts": [],
    },
    "attempts": 1,
    "cache": {
        "enabled": False,
        "hit": False,
        "cache_name": "",
        "reason": "capability_orchestrator_uncached",
    },
}
```

The selected summary may use Chinese wording because it is direct evidence for the Chinese conversation. It must not contain timezone names, offsets, or UTC.

### External Live Fact Boundary

These remain external live facts and must not be answered by the runtime provider:

```text
current time/date/weekday for an explicit third-party place
weather
temperature
opening status
schedule
price
exchange rate
current event status
latest public status
availability
```

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - Update initializer prompt wording and examples so current time/date/weekday route to `Live-context:` runtime-backed slots.
  - Keep weather/temperature/opening examples on the external live path.
- `src/kazusa_ai_chatbot/rag/live_context_agent.py`
  - First perform a behavior-preserving refactor of `LiveContextAgent.run(...)` into explicit top-down source-class branches.
  - Keep the existing external live selector prompt adjacent to its LLM instance and handler, and clarify that it selects external live target/source only.
  - Extend fact type extraction for `current_time`, `current_date`, and `current_weekday`.
  - Add deterministic runtime-backed handling before target/location planning.
  - Keep existing external live fact behavior unchanged.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - Bump `INITIALIZER_PROMPT_VERSION`.
- `tests/test_rag_phase3_capability_agents.py`
  - Add deterministic tests for runtime-backed live facts.
- `tests/test_rag_initializer_cache2.py`
  - Update prompt version assertion and initializer prompt contract assertions.
- `tests/test_rag_phase3_initializer_live_llm.py`
  - Add or update live LLM initializer cases for current time/date/weekday if this suite is available in the test environment.
- `tests/test_llm_time_payload_projection.py`
  - Add a focused assertion that runtime-backed live-context output contains no UTC, offset, timezone name, or raw ISO timestamp.

### Keep

- `character_local_time_context_plan.md` remains the source of truth for timezone conversion and UTC storage policy.
- Existing `time_context.py` remains the source of truth for building and formatting character-local time.
- Existing `Live-context:` dispatcher prefix remains unchanged.
- Existing `web_search_agent2` behavior remains unchanged for external live facts.
- Existing RAG projection may continue mapping `live_context_agent` output into `external_evidence` for compatibility, unless tests prove a blocker.

## Implementation Order

1. Run focused baseline tests for current external live behavior.
2. Refactor `live_context_agent.py` into explicit top-down source-class branches without adding runtime facts or prompt changes.
3. Rerun focused external live tests. Do not continue until behavior is unchanged.
4. Add failing deterministic tests for `LiveContextAgent` current time/date/weekday from `context["time_context"]`.
5. Add prompt-contract tests asserting initializer examples include current local time/date/weekday and do not include missing-location current-time examples.
6. Update `live_context_agent.py` runtime-backed fact extraction and provider handling.
7. Update initializer prompt examples and bump `INITIALIZER_PROMPT_VERSION`.
8. Run prompt-render checks for the changed initializer prompt.
9. Run focused deterministic tests.
10. Run live initializer tests one at a time if configured.
11. Record verification results in `Execution Evidence`.

## Progress Checklist

- [x] Stage 1 - external live baseline
  - Covers: existing `LiveContextAgent` explicit-location, character-location, user-location, missing-location, and opening-status behavior.
  - Verify: focused external live tests pass before refactor.
  - Evidence: record baseline command output.

- [x] Stage 2 - behavior-preserving source-branch refactor
  - Covers: `src/kazusa_ai_chatbot/rag/live_context_agent.py`.
  - Verify: same focused external live tests pass after refactor.
  - Evidence: record changed structure and no-behavior-change test output.
  - Gate: do not add runtime fact types or runtime provider behavior until this stage passes.

- [x] Stage 3 - deterministic live-context runtime tests
  - Covers: `tests/test_rag_phase3_capability_agents.py`.
  - Verify: tests fail before implementation for current time/date/weekday.
  - Evidence: record initial failure.

- [x] Stage 4 - live-context runtime provider implementation
  - Covers: `src/kazusa_ai_chatbot/rag/live_context_agent.py`.
  - Verify: deterministic runtime tests pass.
  - Evidence: record changed behavior and test output.

- [x] Stage 5 - initializer contract update
  - Covers: `persona_supervisor2_rag_supervisor2.py`, `cache2_policy.py`, initializer tests.
  - Verify: prompt contract tests and prompt-render checks pass.
  - Evidence: record new prompt version and rendered prompt check.

- [x] Stage 6 - projection and leak checks
  - Covers: live-context output projection and LLM-facing payload checks.
  - Verify: no UTC/timezone leak tests pass.
  - Evidence: record payload test output.

- [x] Stage 7 - final focused regression
  - Covers: relevant RAG2 tests.
  - Verify: all commands in `Verification`.
  - Evidence: record final command output and skipped live tests, if any.

## Verification

### Deterministic Tests

Run:

```powershell
pytest tests\test_rag_phase3_capability_agents.py -q
pytest tests\test_rag_initializer_cache2.py -q
pytest tests\test_llm_time_payload_projection.py -q
```

Required cases:

- Before runtime fact implementation, the source-branch refactor preserves explicit location, active character location, current user location, missing user location, and opening status behavior.
- `Live-context: answer active character current local time` resolves from `context["time_context"]`.
- `Live-context: answer active character current local date` resolves from `context["time_context"]`.
- `Live-context: answer active character current local weekday` resolves from `context["time_context"]`.
- Runtime-backed facts do not call `web_agent`, `memory_search_agent`, or `conversation_search_agent`.
- Runtime-backed facts do not require `location`.
- Missing `time_context` returns unresolved with `missing_context=["time_context"]`.
- `Live-context: answer current time for unknown location` resolves from `context["time_context"]` and does not return missing `location`.
- `Live-context: answer current user local time if configured` returns unresolved with `missing_context=["user_time_context"]` when no user-local context exists.
- `Live-context: answer current time for explicit location Auckland` does not use the runtime provider.
- External current temperature with unknown location still returns missing `location`.
- Current temperature for active character location still uses memory only for target/scope and web for the live value.
- Initializer prompt includes current local time/date/weekday examples.
- Initializer prompt no longer teaches `current time for unknown location`.
- `INITIALIZER_PROMPT_VERSION` is bumped from the previous value.

### Prompt Render

Run the repo's existing initializer prompt-render test path, or add a focused test that imports and renders the initializer prompt.

Expected:

- Prompt renders without `.format(...)` or brace errors.
- Prompt includes `Live-context: answer active character current local time`.
- Prompt does not include `Runtime-context:`.
- Prompt does not include `runtime_snapshot`.

### Optional Live LLM Tests

Run one at a time only if the live LLM test environment is configured:

```powershell
pytest tests\test_rag_phase3_initializer_live_llm.py -q -k current_time
pytest tests\test_rag_phase3_initializer_live_llm.py -q -k current_date
pytest tests\test_rag_phase3_initializer_live_llm.py -q -k current_weekday
```

Inspect each trace before proceeding to the next one.

### Static Greps

Run:

```powershell
rg "Runtime-context|runtime_snapshot|current time for unknown location" src tests
rg "current_local_datetime|current_local_weekday" src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\nodes tests
```

Expected:

- No production prompt teaches `Runtime-context:`.
- No LLM-facing prompt teaches `runtime_snapshot`.
- `current time for unknown location` does not remain in initializer examples.
- `current_local_datetime` and `current_local_weekday` appear only in deterministic runtime/time-context contracts and tests, not as timezone math instructions to the LLM.

## Acceptance Criteria

This plan is complete when:

- Current time/date/weekday queries can be answered through `Live-context:` without requiring location.
- Runtime-backed live facts use only `context["time_context"]`.
- Runtime-backed live facts make zero DB, web, memory, conversation, profile, promise, or relationship reads.
- External live facts continue to use the existing external live lookup path.
- Initializer cache version is bumped so stale current-time routing does not survive.
- No new top-level RAG capability, dispatcher prefix, graph node, cache namespace, or database schema is introduced.
- Focused deterministic tests pass.
- Live LLM initializer tests pass or are explicitly skipped with environment reason.

## Data Migration

No data migration is allowed.

Existing rows in conversation, memory, profile, promise, scheduler, progress, and cache collections remain unchanged.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| `Live-context` becomes a generic context agent | Limit runtime-backed scope to current time/date/weekday | Contract tests and static grep |
| Current-time queries still route as missing-location external facts | Add initializer examples and bump prompt version | Initializer tests and live traces |
| Runtime provider leaks UTC or timezone strings | Read only sanitized `time_context` fields | Payload leak tests |
| External weather/temperature behavior regresses | Keep external fact types on existing path | Existing capability-agent tests |
| Missing `time_context` gets silently replaced by machine time | Explicit unresolved result | Missing-context test |

## Execution Evidence

- Stage 1 and Stage 2 were verified in-session with the focused external-live cases and re-confirmed in the final regression by passing explicit-location, active-character-location, current-user-location, missing-location, and opening-status `LiveContextAgent` tests.
- Stage 3 added deterministic runtime-backed `LiveContextAgent` tests before implementation; Stage 4 verification then passed with the focused runtime set: `7 passed, 24 deselected`.
- Stage 5 prompt contract and prompt-render path passed with `pytest tests\test_rag_initializer_cache2.py -q -k "initializer_prompt"`: `5 passed, 16 deselected`.
- Stage 5 implementation state matches the contract: initializer prompt teaches active-character current local time/date/weekday and current-user local time if configured; `INITIALIZER_PROMPT_VERSION` is `initializer_prompt:v15`.
- Stage 6 leak and projection checks passed with `pytest tests\test_llm_time_payload_projection.py -q`: `16 passed`.
- Stage 6 includes a focused runtime-backed live-context payload check asserting no UTC, offset, timezone name, or raw ISO timestamp leaks through the runtime provider result.
- Stage 7 deterministic final regression passed with `pytest tests\test_rag_phase3_capability_agents.py -q tests\test_rag_initializer_cache2.py -q tests\test_llm_time_payload_projection.py -q`: `68 passed`.
- Stage 7 static greps were checked. No production prompt teaches `Runtime-context:`. `current time for unknown location` remains only in defensive tests/compatibility handling, not initializer teaching. `runtime_snapshot` remains only as internal Python/source-class terminology, not initializer vocabulary.
- Optional live LLM initializer route checks were run one by one with the live marker enabled and all passed:
  - `test_live_initializer_routes_current_time_to_runtime_live_context`
  - `test_live_initializer_routes_current_date_to_runtime_live_context`
  - `test_live_initializer_routes_current_weekday_to_runtime_live_context`

## Glossary

- Runtime fact: a present-tense fact already available in current process state before RAG runs.
- `runtime_snapshot`: internal Python/source-class term for runtime facts; not LLM vocabulary.
- Runtime provider: deterministic code inside `LiveContextAgent` that reads sanitized current-turn state.
- External live lookup: bounded provider path for outside-world present-tense facts such as weather, temperature, opening status, prices, schedules, and public status.
- Character-local time: the active character's configured local clock, formatted without timezone name, offset, or UTC marker.
