# character reflection cycle stage 1a plan

## Summary

- Goal: Build and evaluate a read-only Character Reflection Cycle over the last 24 hours of real conversation data, including hourly reflection and daily synthesis prompt behavior, without writing anything to MongoDB or changing production behavior.
- Plan class: medium
- Status: draft
- Mandatory skills: `local-llm-architecture`, `development-plan-writing`, `database-data-pull`, `py-style`, `test-style-and-execution`
- Overall cutover strategy: compatible. This stage adds read-only code, local artifacts, and tests only; it does not create collections, indexes, service workers, memory rows, or prompt-facing production context.
- Highest-risk areas: local LLM output quality, prompt/context size, read-query performance on current indexes, private data in local artifacts, and accidentally introducing DB writes before approval.
- Acceptance criteria: the read-only cycle runs against last-24h data, produces local hourly and daily-synthesis evaluation artifacts, passes deterministic tests, passes real LLM tests that are inspected one by one, and receives explicit approval before Stage 1c uses it.

## Context

Stage 1a exists to answer one primary question before any production integration:

```text
Can hourly reflection produce useful, scoped, privacy-aware analysis over real last-24h conversation data with the real configured LLM?
```

This stage is intentionally read-only. It may read from MongoDB and write local files under `test_artifacts/`, but it must not write to MongoDB.

Stage 1a also includes a daily synthesis prompt, but only as a secondary chain sanity check. It runs over the hourly reflection outputs produced in the same 24h evaluation window. It is not a lore-promotion approval signal and must not read raw transcripts directly.

Daily synthesis is justified in 1a because Stage 1c will eventually chain hourly documents into a daily run. This stage should prove that the prompt contract can combine hourly outputs without expanding context or mixing private/group scopes. The approval-critical signal remains the hourly reflection output.

Prior data validation showed that a last-24h sample had many channels but only two assistant-active channels. The selector must reproduce that kind of behavior with a deterministic rule: a scope is assistant-active only when the window contains at least one assistant message and at least one user message for the same platform/channel scope. Inactive channels are skipped, and private/group scopes stay separate.

## Mandatory Skills

- `local-llm-architecture`: load before writing reflection prompts, prompt payloads, LLM JSON contracts, or context budgets.
- `development-plan-writing`: load before modifying this plan.
- `database-data-pull`: load before pulling or exporting live MongoDB diagnostic samples.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major checklist stage, the active agent must reread this entire plan before starting the next stage.
- Stage 1a must not write to MongoDB.
- Stage 1a must not create MongoDB collections or indexes.
- Stage 1a must not modify `memory`, `user_memory_units`, `character_state`, `conversation_episode_state`, `scheduled_events`, or `character_reflection_runs`.
- Stage 1a must not start a service worker.
- Stage 1a must not change `/chat`, cognition, RAG retrieval, dispatcher, scheduler, or persistent-memory search behavior.
- Stage 1a must not import or depend on `memory_evolution`.
- Stage 1a may write local JSON/Markdown artifacts only under `test_artifacts/reflection_cycle_stage1a/`.
- Stage 1a must convert raw numeric metrics into descriptive labels before sending them to the LLM.
- Private and group scopes must stay separate in prompts and output artifacts.
- Assistant-active scope is deterministic: include a scope only when `assistant_message_count >= 1` and `user_message_count >= 1` inside the requested lookback window.
- Hourly prompt payloads must be capped at `STAGE1A_HOURLY_PROMPT_MAX_CHARS=8000`.
- Daily synthesis prompt payloads must be capped at `STAGE1A_DAILY_SYNTHESIS_PROMPT_MAX_CHARS=10000`.
- Daily synthesis must consume parsed hourly outputs and scope metadata only. It must not consume raw transcript rows.
- Real LLM tests must be run one by one and inspected one by one.
- Stage 1c is blocked until Stage 1a real LLM output is approved.

## Must Do

- Add a read-only `reflection_cycle` module with public functions:

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

- Add read-only active-scope selection from `conversation_history`.
- Add scoped message projection using `body_text`, excluding embeddings and `raw_wire_text`.
- Add semantic metric labels for activity, participation, speaker diversity, assistant follow-up signal, privacy scope, and context size.
- Add hourly reflection prompt with a narrow observational schema.
- Add daily synthesis prompt that runs only after hourly outputs are available and only for local evaluation artifacts.
- Add local artifact output containing:
  - selected scopes
  - assistant-active rule counters
  - fallback used yes/no
  - fallback reason
  - query timing and explain summary when available
  - prompt payload previews
  - raw LLM output
  - parsed LLM output
  - validation warnings
  - manual review notes placeholder
- Add deterministic tests for selector, projection, validators, prompt payload budget, and no-DB-write guard.
- Add real LLM tests for one private hourly scope, one group hourly scope, and one daily synthesis case when enough recent data exists.
- Add CLI:

```text
python src\scripts\run_reflection_cycle_readonly.py --lookback-hours 24 --output-dir test_artifacts\reflection_cycle_stage1a --real-llm
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
| Reflection artifacts | compatible | Local files under `test_artifacts/` only. |
| LLM usage | compatible | Evaluation-only calls; no response-path calls. |

## Agent Autonomy Boundaries

- The agent may adjust private helper names, but the public functions and read-only boundary are fixed.
- The agent must not add a convenience DB persistence path.
- The agent must not fix DB performance by adding indexes in Stage 1a.
- If last-24h data is insufficient for a private or group case, record that as evidence and use the nearest recent assistant-active scope only after stating the fallback in the artifact.
- Every artifact must include `fallback_used` and `fallback_reason` so reviewers cannot miss fallback evaluation.
- If real LLM output is poor, do not tune integration scope; revise Stage 1a prompt/evaluation only.

## Target State

```text
CLI or test
  -> read last-24h conversation data
  -> select assistant-active scopes
  -> project scoped messages and semantic metrics
  -> run read-only hourly reflection prompts
  -> validate hourly structured outputs
  -> run daily synthesis over parsed hourly outputs
  -> validate daily synthesis output
  -> write local artifacts
  -> human reviews and approves/rejects Stage 1a output
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| DB writes | Forbidden | Stage 1a is approval/evaluation only. |
| Scope selector index | Not added | Performance is measured here; production index work belongs in Stage 1c. |
| Artifacts | Local files | Allows inspection without DB side effects. |
| Real LLM | Required before Stage 1c | The reflection prompt must be proven with actual model behavior. |
| Daily synthesis | Included as chain sanity check | It validates hourly-to-daily composition, not lore promotion. |
| Hourly schema | Narrow observational schema | Produces a sharper approval signal than evaluating forward-looking fields. |
| Memory evolution | Excluded | Keeps Stage 1a independent from Stage 1b. |

## Data Contracts

### Scope

```python
class ReflectionScope(TypedDict):
    scope_key: str
    scope_type: Literal["private_user", "group_public"]
    platform: str
    platform_channel_id: str
    channel_type: Literal["private", "group"]
    global_user_id: NotRequired[str]
```

Assistant-active rule:

- Group scope: same `platform`, `platform_channel_id`, and `channel_type="group"` has `assistant_message_count >= 1` and `user_message_count >= 1` inside `[window_start, window_end)`.
- Private scope: same `platform`, `platform_channel_id`, and `channel_type="private"` has `assistant_message_count >= 1` and `user_message_count >= 1` inside `[window_start, window_end)`. The private `global_user_id` is taken from the user rows in that private channel.
- If a private channel has multiple user ids in the window, emit one private scope per `global_user_id` only when that user's `user_message_count >= 1` and the channel's `assistant_message_count >= 1`.
- System channels are ignored.

### Budgets

```python
STAGE1A_MAX_SCOPES = 25
STAGE1A_MAX_MESSAGES_PER_SCOPE = 120
STAGE1A_MAX_MESSAGE_CHARS = 280
STAGE1A_HOURLY_PROMPT_MAX_CHARS = 8000
STAGE1A_DAILY_SYNTHESIS_PROMPT_MAX_CHARS = 10000
STAGE1A_ARTIFACT_PROMPT_PREVIEW_CHARS = 2000
```

Budgets are character-count caps because this stage must be deterministic across local LLM providers. If a payload exceeds budget, drop oldest message rows first, then trim individual message excerpts to `STAGE1A_MAX_MESSAGE_CHARS`. The artifact must record every truncation.

### Outputs

```python
class ReadonlyHourlyReflectionOutput(TypedDict, total=False):
    topic_summary: list[str]
    participant_observations: list[str]
    conversation_quality_feedback: list[str]
    privacy_notes: list[str]
    confidence: Literal["low", "medium", "high"]
```

```python
class ReadonlyDailySynthesisOutput(TypedDict, total=False):
    scope_summaries: list[str]
    cross_scope_topics: list[str]
    conversation_quality_patterns: list[str]
    privacy_risks: list[str]
    synthesis_limitations: list[str]
    confidence: Literal["low", "medium", "high"]
```

```python
class ReflectionEvaluationArtifact(TypedDict, total=False):
    requested_window_start: str
    requested_window_end: str
    actual_window_start: str
    actual_window_end: str
    fallback_used: bool
    fallback_reason: str
    selected_scopes: list[ReflectionScope]
    assistant_active_rule_counters: dict
    query_timing_ms: dict
    prompt_budget: dict
    truncation_log: list[str]
    hourly_outputs: list[ReadonlyHourlyReflectionOutput]
    daily_synthesis_output: ReadonlyDailySynthesisOutput
    validation_warnings: list[str]
    manual_review_notes: str
```

Stage 1a output is not a storage contract. Stage 1c may reuse the validated core fields, but Stage 1a itself persists only local artifacts. Stage 1c must add lore-candidate and promotion fields in its own plan/tests if needed; Stage 1a does not evaluate those forward-looking fields.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/reflection_cycle/__init__.py`
- `src/kazusa_ai_chatbot/reflection_cycle/models.py`
- `src/kazusa_ai_chatbot/reflection_cycle/selector.py`
- `src/kazusa_ai_chatbot/reflection_cycle/projection.py`
- `src/kazusa_ai_chatbot/reflection_cycle/prompts.py`
- `src/kazusa_ai_chatbot/reflection_cycle/readonly_runtime.py`
- `src/scripts/run_reflection_cycle_readonly.py`
- `tests/test_reflection_cycle_stage1a_readonly.py`
- `tests/test_reflection_cycle_stage1a_prompt_contracts.py`
- `tests/test_reflection_cycle_stage1a_live_llm.py`

### Keep

- `service.py`
- `db/bootstrap.py`
- `db/memory.py`
- RAG persistent-memory agents
- Dispatcher and scheduler
- MongoDB collection schemas

## Implementation Order

1. Add deterministic tests for selector, projection, no-write guard, assistant-active scope rule, fallback artifact fields, and prompt payload budgets.
2. Implement read-only selector and projection.
3. Implement hourly prompt payloads and validators.
4. Implement read-only runtime and CLI artifact writer.
5. Run deterministic tests.
6. Run one real LLM private hourly-scope test and inspect artifact.
7. Run one real LLM group hourly-scope test and inspect artifact.
8. Run one real LLM daily synthesis test over the parsed hourly outputs and inspect artifact.
9. Record Stage 1a approval or rejection in the artifact summary.

## Progress Checklist

- [ ] Read-only selector/projection tests added and failing for missing symbols.
- [ ] Read-only selector/projection implemented.
- [ ] Hourly prompt payload and validators implemented.
- [ ] Daily synthesis prompt payload and validators implemented.
- [ ] CLI artifact writer implemented.
- [ ] Deterministic tests pass.
- [ ] Private hourly real LLM case run and inspected.
- [ ] Group hourly real LLM case run and inspected.
- [ ] Daily synthesis real LLM case run and inspected.
- [ ] Stage 1a approval artifact recorded.

## Verification

```powershell
pytest tests\test_reflection_cycle_stage1a_readonly.py -q
pytest tests\test_reflection_cycle_stage1a_prompt_contracts.py -q
```

Real LLM tests must be run one at a time:

```powershell
pytest tests\test_reflection_cycle_stage1a_live_llm.py::test_live_stage1a_private_scope_reflection -q -s
pytest tests\test_reflection_cycle_stage1a_live_llm.py::test_live_stage1a_group_scope_reflection -q -s
pytest tests\test_reflection_cycle_stage1a_live_llm.py::test_live_stage1a_daily_synthesis -q -s
```

Manual evaluation:

```powershell
python src\scripts\run_reflection_cycle_readonly.py --lookback-hours 24 --output-dir test_artifacts\reflection_cycle_stage1a --real-llm
```

## Acceptance Criteria

- No MongoDB write method is called by Stage 1a code.
- Last-24h assistant-active scopes are selected and inactive channels are skipped.
- Assistant-active selection uses the reviewed `assistant_message_count >= 1` and `user_message_count >= 1` rule.
- Prompt payloads fit configured budgets.
- Private and group scopes are isolated.
- Hourly output uses the narrow observational schema only.
- Daily synthesis consumes parsed hourly outputs only.
- Local artifacts contain query timing, selected scopes, fallback fields, raw/parsed LLM output, and validation warnings.
- Real LLM outputs are manually inspected and accepted before Stage 1c starts.

## Execution Evidence

- Deterministic test results:
- Real LLM private case artifact:
- Real LLM group case artifact:
- Real LLM daily synthesis artifact:
- Last-24h performance summary:
- Stage 1a approval decision:
