# character reflection cycle stage 1a plan

## Summary

- Goal: Build and evaluate a read-only Character Reflection Cycle over monitored channels, including message-bearing hourly reflection and daily synthesis prompt behavior, without MongoDB writes or production behavior changes.
- Plan class: large
- Status: completed
- Mandatory skills: `local-llm-architecture`, `development-plan-writing`, `database-data-pull`, `py-style`, `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: compatible. This stage adds read-only code, local artifacts, interface documentation, and tests only; it does not create production workers, DB write paths, memory rows, indexes, or prompt-facing production context.
- Highest-risk areas: local LLM output quality, prompt/context size, read-query performance on current indexes, private data exposure in local artifacts, hidden DB coupling, and prompt contract drift from existing cognition style.
- Acceptance criteria: deterministic tests pass; monitored-channel selection uses last character reply time; every message-bearing hour in monitored channels is evaluated; live LLM checks are run one at a time after the final prompt contract and inspected; a real monitored-channel artifact is generated and accepted before Stage 1c uses it.

## Context

Stage 1a exists to answer one primary question before production integration:

```text
Can hourly reflection produce useful, scoped, privacy-aware analysis over real recent monitored-channel messages with the configured LLM?
```

This stage is intentionally read-only. It may read from MongoDB through the approved DB interface and write local artifacts under `test_artifacts/`, but it must not write to MongoDB.

Daily synthesis is included as a secondary chain sanity check. It runs per channel over compact `active_hour_slots` derived from hourly reflection outputs produced in the same evaluation run. It is not a lore-promotion approval signal and must not read raw transcripts or full hourly reflection objects directly.

The implementation now has an explicit module interface document:

```text
src/kazusa_ai_chatbot/reflection_cycle/README.md
```

That ICD is part of the Stage 1a contract. It defines public entry points, allowed dependencies, DB read shape, prompt payload contracts, attachment policy, and forbidden coupling.

## Mandatory Skills

- `local-llm-architecture`: load before writing reflection prompts, prompt payloads, LLM JSON contracts, or context budgets.
- `development-plan-writing`: load before modifying this plan.
- `database-data-pull`: load before pulling or exporting live MongoDB diagnostic samples.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files containing Chinese prompt text.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major checklist stage, the active agent must reread this entire plan before starting the next stage.
- Stage 1a must not write to MongoDB.
- Stage 1a must not create MongoDB collections or indexes.
- Stage 1a must not modify `memory`, `user_memory_units`, `character_state`, `conversation_episode_state`, `scheduled_events`, or `character_reflection_runs`.
- Stage 1a must not create, update, or override `user_image`, `user_memory_units`, user profiles, consolidator outputs, lore, or persistent character memory. Reflection participant observations are character-training evaluation evidence only; user image remains owned by the consolidator.
- Stage 1a must not start a service worker.
- Stage 1a must not change `/chat`, cognition, RAG retrieval, dispatcher, scheduler, or persistent-memory search behavior.
- Stage 1a must not import or depend on `memory_evolution`.
- Stage 1a may write local JSON artifacts only under caller-provided `output_dir`; the current default manual path is `test_artifacts/reflection_cycle_readonly/`.
- Stage 1a must convert raw numeric metrics into descriptive labels before sending them to the LLM.
- Private and group channel scopes must stay separate in prompts and output artifacts.
- Monitored-channel selection is deterministic: include a channel when the latest character message for that `platform`, `platform_channel_id`, and `channel_type` falls inside the monitor eligibility window. The default eligibility window is the last 24 hours relative to the run end time.
- Stage 1a must not create or maintain a counter, state variable, or dedicated collection just to decide monitoring eligibility. It must query conversation history through `db.conversation_reflection` for the latest character message time.
- Once a channel is monitored, every hour in the evaluation window must be considered. An hourly reflection slot may be skipped only when that hour contains no `assistant` or `user` messages at all.
- Hours with user messages but no character reply are still message-bearing hours and must be evaluated for observation, progress, and response-quality implications.
- Selector code must not execute MongoDB commands directly. MongoDB access belongs only in `kazusa_ai_chatbot.db.conversation_reflection`.
- Message reads must use an allowlist projection. They must not load embeddings, raw wire text, attachment binary payloads, or arbitrary attachment metadata.
- Prompt payload JSON keys must remain stable English machine-facing keys, matching existing cognition/consolidator style.
- Chinese belongs in prompt prose, review questions, descriptive label values, validation warnings, and generated free-text fields.
- Do not repeat language-policy instructions inside every schema sample value.
- Hourly prompt payloads must be capped at `READONLY_REFLECTION_HOURLY_PROMPT_MAX_CHARS=8000`.
- Daily synthesis prompt payloads must be capped at `READONLY_REFLECTION_DAILY_PROMPT_MAX_CHARS=25000`.
- Daily synthesis must consume compact `active_hour_slots` for one channel only. It must not consume raw transcript rows or full hourly reflection objects.
- Daily synthesis must preserve `active_hour_summaries.hour` by exact copy from `active_hour_slots.hour`; it must not convert time zones or rewrite hour formats.
- Real LLM tests must be run one by one and inspected one by one.
- Stage 1c is blocked until Stage 1a real LLM output from the final prompt contract is approved.

## Must Do

- Add a read-only `reflection_cycle` module with public package exports:

```python
async def collect_reflection_inputs(
    *,
    lookback_hours: int = 24,
    now: datetime | None = None,
) -> ReflectionInputSet: ...

async def run_readonly_reflection_evaluation(
    *,
    lookback_hours: int = 24,
    now: datetime | None = None,
    output_dir: str,
    use_real_llm: bool,
) -> ReflectionEvaluationResult: ...
```

- Add a DB interface module for read-only reflection queries.
- Add monitored-channel selection from `conversation_history` through that DB interface only.
- Add hour-slot construction that evaluates every message-bearing hour for monitored channels and skips only hours with no messages.
- Add message allowlist projection using only prompt-needed fields and `attachments.description`.
- Add semantic metric labels for message volume, assistant participation, participant diversity, and window span.
- Add hourly reflection prompt with a narrow observational schema.
- Add daily synthesis prompt that runs only after hourly outputs are available, consumes compact per-channel `active_hour_slots`, and only writes local evaluation artifacts.
- Add centralized language policy in each prompt and align section style with existing cognition/consolidator prompts.
- Add local artifact output containing selected monitored channels, message-bearing hourly slots, fallback status, query diagnostics, prompt previews, raw/parsed LLM output, validation warnings, and manual review notes.
- Add deterministic tests for monitored-channel selector, message-bearing hourly slot rules, DB interface boundary, DB allowlist, prompt payload budget, prompt contract, validators, and local-artifact-only runtime behavior.
- Add real LLM tests for one private hourly slot, one group hourly slot, and one per-channel daily synthesis case.
- Add CLI:

```powershell
venv\Scripts\python.exe -m scripts.run_reflection_cycle_readonly --lookback-hours 24 --output-dir test_artifacts\reflection_cycle_readonly --real-llm
```

- Add a reflection-cycle ICD:

```text
src/kazusa_ai_chatbot/reflection_cycle/README.md
```

## Deferred

- MongoDB writes.
- MongoDB index creation.
- `character_reflection_runs` collection.
- Memory evolution.
- Lore promotion.
- Prompt-facing reflection context in production cognition.
- Background worker.
- Autonomous messages.
- Stage 1b memory search/seeding changes.
- Stage 1c integration.

## Cutover Policy

| Area | Policy | Notes |
|---|---|---|
| Production service | compatible | No service wiring in Stage 1a. |
| MongoDB | compatible | Read-only. No collections, indexes, or documents created. |
| Reflection artifacts | compatible | Local files under caller-provided `output_dir` only. |
| LLM usage | compatible | Evaluation-only calls; no response-path calls. |
| Interfaces | compatible | New module and DB interface only; existing callers are unchanged. |

## Agent Autonomy Boundaries

- The agent may adjust private helper names, but the public package exports and read-only boundary are fixed.
- The agent must not add a convenience DB persistence path.
- The agent must not fix DB performance by adding indexes in Stage 1a.
- If recent monitored-channel data is insufficient, record that as evidence and use the bounded fallback only when `fallback_used` and `fallback_reason` are explicit.
- Every artifact must include `fallback_used` and `fallback_reason`.
- If real LLM output is poor, do not tune integration scope; revise Stage 1a prompt/evaluation only.
- If an interface dependency is unclear, update `reflection_cycle/README.md` before changing code.

## Target State

```text
CLI or test
  -> read recent conversation data through db.conversation_reflection
  -> select monitored channels by latest character message time
  -> split monitored channels into message-bearing hour slots
  -> project hourly slot messages and semantic labels
  -> run or skip read-only hourly reflection prompts per hour slot
  -> validate hourly structured outputs
  -> run or skip per-channel daily synthesis over compact active-hour slots
  -> validate daily synthesis output
  -> write local artifact
  -> human reviews and approves/rejects Stage 1a output
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| DB writes | Forbidden | Stage 1a is approval/evaluation only. |
| DB access | Dedicated `db.conversation_reflection` interface | Keeps selector from crossing into storage execution. |
| Message projection | Allowlist DB projection plus prompt projection | Avoids accidental raw field exposure and keeps attachment policy explicit. |
| Attachment handling | Use only bounded `attachments.description` | Gives reflection image/file context without binary or raw metadata. |
| Monitor selector index | Not added | Performance is measured here; index work belongs in Stage 1c if needed. |
| Channel eligibility | Latest character message in last 24 hours | Matches the production monitoring rule without counters or a dedicated monitoring collection. |
| Hour slot rule | Evaluate every message-bearing hour in monitored channels | User-only hours still matter for observation and progress; skip only hours with no messages. |
| Scope unit | Platform/channel/channel_type channel plus UTC hour slot | Matches current storage shape while preserving hourly cadence and private/group separation. |
| Prompt payload keys | English machine-facing keys | Matches existing cognition/consolidator style and validator contracts. |
| Prompt language | Chinese prose and generated free-text policy | Meets language requirement without coupling schema keys to language. |
| Artifacts | Local files | Allows inspection without DB side effects. |
| Real LLM | Required before Stage 1c | The final reflection prompt must be proven with actual model behavior. |
| Daily synthesis | Included as chain sanity check | It validates compact hourly-slot-to-daily composition, not lore promotion. |
| Hourly schema | Narrow observational schema | Produces a sharper approval signal than evaluating forward-looking fields. |
| Memory evolution | Excluded | Keeps Stage 1a independent from Stage 1b. |

## Data Contracts

The authoritative interface contract is `src/kazusa_ai_chatbot/reflection_cycle/README.md`.

### Channel And Hour-Slot Model

```python
@dataclass
class ReflectionScopeInput:
    scope_ref: str
    platform: str
    platform_channel_id: str
    channel_type: str
    assistant_message_count: int
    user_message_count: int
    total_message_count: int
    first_timestamp: str
    last_timestamp: str
    messages: list[dict[str, Any]]
```

Monitored-channel rule:

- Same `platform`, `platform_channel_id`, and `channel_type` has a latest character message timestamp inside the monitor eligibility window.
- The default monitor eligibility window is the last 24 hours relative to the run end time.
- Monitoring eligibility is computed by querying `conversation_history` through `db.conversation_reflection`; no counter, state variable, or dedicated monitoring collection is introduced.
- `scope_ref` for the channel is a stable non-identifying hash of `platform`, `platform_channel_id`, and `channel_type`.
- Hourly `scope_ref` values add the UTC hour-start suffix to the channel `scope_ref`.
- System channels are not specially selected; role filtering restricts rows to `assistant` and `user`, and normal channel-type labels are carried through.

Hour-slot rule:

- Once a channel is monitored, every hour in the evaluation window is considered.
- An hourly slot is skipped only when that hour has no `assistant` or `user` messages.
- A user-only hour is still evaluated. The hourly prompt must treat missing character reply as evidence about conversation progress or missed response opportunity, not as a reason to drop the slot.
- An assistant-only hour is also evaluated when message rows exist, because it may contain character-spoken lore or setting signals.

### DB Message Allowlist

```python
{
    "_id": 0,
    "platform": 1,
    "platform_channel_id": 1,
    "channel_type": 1,
    "role": 1,
    "platform_user_id": 1,
    "global_user_id": 1,
    "display_name": 1,
    "body_text": 1,
    "timestamp": 1,
    "attachments.description": 1,
}
```

### Budgets

```python
READONLY_REFLECTION_MAX_SCOPES = 25
READONLY_REFLECTION_MAX_MESSAGES_PER_SCOPE = 120
READONLY_REFLECTION_MAX_MESSAGE_CHARS = 280
READONLY_REFLECTION_HOURLY_PROMPT_MAX_CHARS = 8000
READONLY_REFLECTION_DAILY_PROMPT_MAX_CHARS = 25000
READONLY_REFLECTION_ARTIFACT_PROMPT_PREVIEW_CHARS = 2000
READONLY_REFLECTION_MONITOR_ELIGIBILITY_HOURS = 24
READONLY_REFLECTION_FALLBACK_LOOKBACK_HOURS = 168
READONLY_REFLECTION_DAILY_SLOT_TEXT_CHARS = 180
```

Budgets are character-count caps. If a payload exceeds budget, the projection drops oldest message rows first, then trims individual message text.

### Hourly Output

```python
{
    "topic_summary": str,
    "participant_observations": [
        {
            "participant_ref": str,
            "observation": str,
            "evidence_strength": "low|medium|high",
        }
    ],
    "conversation_quality_feedback": list[str],
    "privacy_notes": list[str],
    "confidence": "low|medium|high",
}
```

### Daily Output

```python
{
    "day_summary": str,
    "active_hour_summaries": [
        {
            "hour": str,
            "summary": str,
        }
    ],
    "cross_hour_topics": list[str],
    "conversation_quality_patterns": list[str],
    "privacy_risks": list[str],
    "synthesis_limitations": list[str],
    "confidence": "low|medium|high",
}
```

Stage 1a output is not a storage contract. Stage 1c may reuse the validated core fields, but Stage 1a itself persists only local artifacts. Stage 1c must add lore-candidate and promotion fields in its own plan/tests if needed; Stage 1a does not evaluate those forward-looking fields.

Daily `active_hour_summaries.hour` values must exactly match input `active_hour_slots.hour` values. The validator records a warning if the model rewrites or converts an hour label.

## Change Surface

### Created

- `src/kazusa_ai_chatbot/db/conversation_reflection.py`
- `src/kazusa_ai_chatbot/reflection_cycle/__init__.py`
- `src/kazusa_ai_chatbot/reflection_cycle/models.py`
- `src/kazusa_ai_chatbot/reflection_cycle/selector.py`
- `src/kazusa_ai_chatbot/reflection_cycle/projection.py`
- `src/kazusa_ai_chatbot/reflection_cycle/prompts.py`
- `src/kazusa_ai_chatbot/reflection_cycle/runtime.py`
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
- `src/scripts/run_reflection_cycle_readonly.py`
- `tests/test_reflection_cycle_readonly.py`
- `tests/test_reflection_cycle_prompt_contracts.py`
- `tests/test_reflection_cycle_live_llm.py`

### Modified

- `src/kazusa_ai_chatbot/db/__init__.py`: exports the reflection DB read interfaces.

### Kept

- `service.py`
- `/chat` request handling
- cognition and dialog nodes
- RAG retrieval agents
- dispatcher and scheduler
- memory and user-memory writers
- MongoDB bootstrap and collection schemas

## Implementation Order

1. Add deterministic tests for monitored-channel selector, DB interface, no-write guard, message-bearing hour-slot rule, fallback artifact fields, and prompt payload budgets.
2. Implement DB read interface, read-only selector, and prompt projection.
3. Implement hourly prompt payloads and validators.
4. Implement compact per-channel daily synthesis prompt payloads and validators.
5. Implement read-only runtime and CLI artifact writer.
6. Add reflection-cycle ICD.
7. Run deterministic tests.
8. Run real LLM private hourly-slot test and inspect trace.
9. Run real LLM group hourly-slot test and inspect trace.
10. Run real LLM daily synthesis test and inspect trace.
11. Run real monitored-channel artifact generation and record approval or rejection.

## Progress Checklist

- [x] Stage 1a interface boundary documented.
  - Covers: `src/kazusa_ai_chatbot/reflection_cycle/README.md`.
  - Verify: `git diff --check -- src\kazusa_ai_chatbot\reflection_cycle\README.md`.
  - Evidence: interface document updated to monitored-channel selection, message-bearing hour slots, DB dependency boundary, and consolidator/user-image non-write boundary.
  - Sign-off: Codex / 2026-05-04 / completed.
- [x] Read-only DB interface and monitored-channel selector implemented.
  - Covers: `db/conversation_reflection.py`, `reflection_cycle/selector.py`.
  - Verify: deterministic tests below.
  - Evidence: `pytest tests\test_reflection_cycle_readonly.py tests\test_reflection_cycle_prompt_contracts.py -q` passed 15 tests. `test_monitored_channel_selection_uses_latest_character_message`, `test_db_interface_lists_monitored_channel_rows_readonly`, and `test_selector_source_has_no_direct_mongo_execution` prove the selector uses the DB interface and latest character message rows, not counters or direct MongoDB calls.
  - Sign-off: Codex / 2026-05-04 / completed.
- [x] Prompt projection and prompt contracts implemented.
  - Covers: `projection.py`, `prompts.py`, prompt contract tests.
  - Verify: deterministic tests below.
  - Evidence: `pytest tests\test_reflection_cycle_readonly.py tests\test_reflection_cycle_prompt_contracts.py -q` passed 15 tests. Prompt tests prove English machine-facing keys, centralized Chinese language policy, no raw transcripts in daily synthesis, daily prompt budget, and exact-hour validation for daily summaries.
  - Sign-off: Codex / 2026-05-04 / completed.
- [x] CLI artifact writer implemented.
  - Covers: `runtime.py`, `src/scripts/run_reflection_cycle_readonly.py`.
  - Verify: deterministic local-artifact test.
  - Evidence: deterministic runtime tests passed; prompt-only 48-hour artifact selected 2 channels, produced 34 hourly slots, 2 daily syntheses, no fallback, and no prompt warnings.
  - Sign-off: Codex / 2026-05-04 / completed.
- [x] Private hourly real LLM case run and inspected after final prompt contract.
  - Verify: `pytest tests\test_reflection_cycle_live_llm.py::test_live_readonly_private_scope_reflection -q -s -m live_llm`.
  - Evidence: passed through the project venv. Latest trace: `test_artifacts/llm_traces/reflection_cycle_readonly_live_llm__scope_live_private__20260504T120249009274Z.json`. Parsed output had required hourly fields and no validation warnings.
  - Sign-off: Codex / 2026-05-04 / completed.
- [x] Group hourly real LLM case run and inspected after final prompt contract.
  - Verify: `pytest tests\test_reflection_cycle_live_llm.py::test_live_readonly_group_scope_reflection -q -s -m live_llm`.
  - Evidence: passed through the project venv. Latest trace: `test_artifacts/llm_traces/reflection_cycle_readonly_live_llm__scope_live_group__20260504T120301064979Z.json`. Parsed output had required hourly fields and no validation warnings.
  - Sign-off: Codex / 2026-05-04 / completed.
- [x] Daily synthesis real LLM case run and inspected after final prompt contract.
  - Verify: `pytest tests\test_reflection_cycle_live_llm.py::test_live_readonly_daily_synthesis -q -s -m live_llm`.
  - Evidence: passed through the project venv. Latest trace: `test_artifacts/llm_traces/reflection_cycle_readonly_live_llm__daily_synthesis__20260504T120311747265Z.json`. Parsed output had required daily fields, exact copied hour values in raw JSON, and no validation warnings.
  - Sign-off: Codex / 2026-05-04 / completed.
- [x] Real monitored-channel artifact generated and accepted after final prompt contract.
  - Verify: `venv\Scripts\python.exe -m scripts.run_reflection_cycle_readonly --lookback-hours 48 --output-dir test_artifacts\reflection_cycle_stage1a_monitored_real --real-llm`.
  - Evidence: `test_artifacts/reflection_cycle_stage1a_monitored_real/readonly_reflection_evaluation_20260504T115909145767Z.json` selected private QQ `673225019` and group QQ `54369546`, produced 34 hourly reflections and 2 daily syntheses, used no fallback, and produced zero prompt or schema warnings. Markdown review table: `test_artifacts/reflection_cycle_stage1a_monitored_real/stage1a_signoff_report.md`.
  - Sign-off: Codex / 2026-05-04 / completed. Stage 1a is signed off.

## Verification

Deterministic tests:

```powershell
pytest tests\test_reflection_cycle_readonly.py tests\test_reflection_cycle_prompt_contracts.py -q
```

Real LLM tests must be run one at a time:

```powershell
pytest tests\test_reflection_cycle_live_llm.py::test_live_readonly_private_scope_reflection -q -s -m live_llm
pytest tests\test_reflection_cycle_live_llm.py::test_live_readonly_group_scope_reflection -q -s -m live_llm
pytest tests\test_reflection_cycle_live_llm.py::test_live_readonly_daily_synthesis -q -s -m live_llm
```

Manual evaluation:

```powershell
venv\Scripts\python.exe -m scripts.run_reflection_cycle_readonly --lookback-hours 48 --output-dir test_artifacts\reflection_cycle_stage1a_monitored_real --real-llm
```

Static checks:

```powershell
rg "get_db|\.aggregate\(|\.find\(|\.command\(" src\kazusa_ai_chatbot\reflection_cycle\selector.py
rg '"评估模式"|"范围元数据"|"对话"|"评审问题"|"小时反思"' src\kazusa_ai_chatbot\reflection_cycle
```

Allowed static-check exceptions:

- Chinese-key grep may match negative assertions in tests only.
- Artifact keys such as `hourly_reflections` in `runtime.py` are local artifact shape, not model-facing prompt payload.

## Acceptance Criteria

- No MongoDB write method is called by Stage 1a code.
- Selector uses only `db.conversation_reflection` interfaces for MongoDB access.
- Message reads use an allowlist projection.
- Monitored channels are selected by latest character message timestamp in the monitor eligibility window.
- No counter, state variable, or dedicated monitoring collection is used for channel eligibility.
- Every message-bearing hour in monitored channels is evaluated; only hours with no `assistant` or `user` messages are skipped.
- Prompt payloads fit configured budgets.
- Private and group channel scopes are isolated.
- Hourly output uses the narrow observational schema only.
- Daily synthesis consumes compact `active_hour_slots` only.
- Local artifacts contain query diagnostics, selected monitored channels, message-bearing hourly slots, fallback fields, raw/parsed LLM output, and validation warnings.
- Real LLM outputs from the final prompt contract are manually inspected and accepted before Stage 1c starts.

## Validation Against Current Implementation

| Area | Expected by plan | Current implementation | Status |
|---|---|---|---|
| Public module | `reflection_cycle` package exports collection and runtime functions | Implemented in `__init__.py` | Pass |
| DB boundary | Selector calls DB interface only | Implemented and covered by source/static test | Pass |
| MongoDB writes | None | No reflection write path exists | Pass |
| DB message shape | Allowlist projection | Implemented with `attachments.description` only | Pass |
| Interface doc | Required | `reflection_cycle/README.md` added | Pass |
| Prompt keys | English schema keys | Implemented after correction | Pass |
| Prompt language | Chinese generated free text via central policy | Implemented and tested | Pass |
| Monitor selector | Latest character message in last 24h | Implemented through `list_recent_character_message_channels`; no counter or dedicated monitoring collection | Pass |
| Hour-slot skip rule | Skip only hours with no messages | Implemented; 48-hour real artifact includes user-only group hours and no empty hours | Pass |
| Daily raw transcript boundary | Daily consumes compact active-hour slots only | Implemented and tested; daily prompt receives no raw transcripts or full hourly objects | Pass |
| Daily hour identity | Daily output copies input hour labels exactly | Implemented and tested with exact-hour validator | Pass |
| Consolidator boundary | Reflection does not update user image or user memory | Implemented as read-only artifact path and documented in ICD | Pass |
| Deterministic tests | Focused tests pass | `pytest tests\test_reflection_cycle_readonly.py tests\test_reflection_cycle_prompt_contracts.py -q` passed 15 tests | Pass |
| Live LLM approval | Required before 1c | Private hourly, group hourly, and daily synthesis live tests passed one by one after final prompt contract | Pass |
| Monitored-channel artifact approval | Required before 1c | 48-hour real artifact generated and inspected; no prompt/schema warnings | Pass |

## Execution Evidence

- Syntax check:
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\reflection_cycle\prompts.py src\kazusa_ai_chatbot\reflection_cycle\selector.py src\kazusa_ai_chatbot\reflection_cycle\runtime.py src\kazusa_ai_chatbot\reflection_cycle\projection.py src\kazusa_ai_chatbot\db\conversation_reflection.py src\scripts\run_reflection_cycle_readonly.py` passed.
  - `python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\prompts.py src\kazusa_ai_chatbot\reflection_cycle\projection.py tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_cycle_live_llm.py` also passed immediately after the exact-hour validation change.
- Deterministic test results:
  - `venv\Scripts\pytest.exe tests\test_reflection_cycle_readonly.py tests\test_reflection_cycle_prompt_contracts.py -q` passed 15 tests.
  - Covered selector DB boundary, latest-character monitored-channel selection, no direct MongoDB selector calls, message allowlist, local-artifact-only runtime behavior, message-bearing hourly slot expansion, prompt budgets, centralized Chinese language policy, daily compact-slot contract, and exact-hour daily validation.
- Static checks:
  - `git diff --check -- ...` passed for the Stage 1a plan, reflection-cycle package, DB interface, CLI, and tests. Git reported only the existing LF-to-CRLF working-copy warning for the plan file.
  - Selector grep for `get_db`, `.aggregate(`, `.find(`, and `.command(` returned no matches.
  - Chinese prompt-key grep returned matches only in negative test assertions, not in runtime prompt payloads.
- Interface doc:
  - `src/kazusa_ai_chatbot/reflection_cycle/README.md` documents the DB dependency boundary, monitored-channel rule, message-bearing hourly rule, attachment allowlist, prompt contracts, artifact contract, and consolidator/user-image non-write boundary.
- Real LLM private case artifact:
  - `venv\Scripts\pytest.exe tests\test_reflection_cycle_live_llm.py::test_live_readonly_private_scope_reflection -q -s -m live_llm` passed.
  - Trace: `test_artifacts/llm_traces/reflection_cycle_readonly_live_llm__scope_live_private__20260504T120249009274Z.json`.
  - Inspection: required hourly schema fields present; no validation warnings; generated free text in Chinese.
- Real LLM group case artifact:
  - `venv\Scripts\pytest.exe tests\test_reflection_cycle_live_llm.py::test_live_readonly_group_scope_reflection -q -s -m live_llm` passed.
  - Trace: `test_artifacts/llm_traces/reflection_cycle_readonly_live_llm__scope_live_group__20260504T120301064979Z.json`.
  - Inspection: required hourly schema fields present; no validation warnings; participant refs stayed abstract.
- Real LLM daily synthesis artifact:
  - `venv\Scripts\pytest.exe tests\test_reflection_cycle_live_llm.py::test_live_readonly_daily_synthesis -q -s -m live_llm` passed.
  - Trace: `test_artifacts/llm_traces/reflection_cycle_readonly_live_llm__daily_synthesis__20260504T120311747265Z.json`.
  - Inspection: required daily schema fields present; no validation warnings; raw JSON preserved exact input `hour` labels.
- Monitored-channel performance summary:
  - Prompt-only 48-hour preflight: `venv\Scripts\python.exe -m scripts.run_reflection_cycle_readonly --lookback-hours 48 --output-dir test_artifacts\reflection_cycle_stage1a_monitored_prompt_only` selected 2 channels, produced 33 hourly prompt-only reflections and 2 daily prompt-only syntheses, used no fallback, and produced no prompt warnings. This was a budget preflight, not the acceptance artifact.
  - Real 48-hour run: `venv\Scripts\python.exe -m scripts.run_reflection_cycle_readonly --lookback-hours 48 --output-dir test_artifacts\reflection_cycle_stage1a_monitored_real --real-llm` selected 2 channels, produced 34 real hourly reflections and 2 real daily syntheses, and used no fallback.
  - Selected channels: private QQ `673225019` with 107 source messages and 18 hourly slots; group QQ `54369546` with 88 source messages and 16 hourly slots.
  - Prompt budget evidence: hourly prompts min 2561 chars, max 6147 chars, average 3487.5 chars; daily prompts 8960 chars and 8691 chars; all under configured budgets.
  - Warning evidence: 0 hourly prompt/schema warnings and 0 daily prompt/schema warnings.
  - Full real artifact: `test_artifacts/reflection_cycle_stage1a_monitored_real/readonly_reflection_evaluation_20260504T115909145767Z.json`.
  - Markdown hourly review table: `test_artifacts/reflection_cycle_stage1a_monitored_real/stage1a_signoff_report.md`.
- Stage 1a approval decision:
  - Approved/signed off for Stage 1a read-only evaluation. Stage 1c may consume this completed plan as evidence, but Stage 1c must still implement any write-capable memory/lore integration under its own plan and tests.
