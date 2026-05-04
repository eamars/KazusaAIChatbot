# reflection memory integration stage 1c plan

## Summary

- Goal: Integrate the approved Stage 1a read-only reflection cycle with the completed Stage 1b evolving memory subsystem so daily reflection can persist 0 or 1 high-signal sanitized global lore mutation.
- Plan class: large
- Status: draft
- Mandatory skills: `local-llm-architecture`, `development-plan-writing`, `memory-knowledge-maintenance`, `no-prepost-user-input`, `py-style`, `test-style-and-execution`, `database-data-pull`
- Overall cutover strategy: bigbang integration after entry criteria. Stage 1c adds production reflection persistence, monitored-channel indexing, worker wiring, and daily lore promotion in one approved integration path rather than preserving a parallel candidate-only production mode.
- Highest-risk areas: promoting poor reflection output, private leakage, live cognition latency, memory poisoning, stale Cache2, and bypassing the Stage 1a approval gate.
- Acceptance criteria: Stage 1c starts only after Stage 1a real LLM approval and Stage 1b completion; production reflection writes to `character_reflection_runs`; daily promotion writes through Stage 1b memory APIs only; private/user-specific details are rejected; live chat remains priority; autonomous messaging remains out of scope.

## Context

Stage 1c is the integration stage. It is not where reflection quality is first proven and not where the memory search schema is built.

Inputs:

- Stage 1a provides approved read-only selector/projection logic, narrow hourly observational prompts, daily synthesis chain sanity checks, prompt budgets, and real LLM evaluation artifacts.
- Stage 1b provides evolving memory APIs, active-only search, seed tooling, and Cache2 memory invalidation.

Stage 1c ties them together for production use.

## Mandatory Skills

- `local-llm-architecture`: load before changing production reflection prompts, service worker scheduling, or prompt-facing context.
- `development-plan-writing`: load before modifying this plan.
- `memory-knowledge-maintenance`: load before writing reflection-inferred global memory rows.
- `no-prepost-user-input`: load before changing character-agreement or lore-admission logic.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before validating production-like runs against live data.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major checklist stage, the active agent must reread this entire plan before starting the next stage.
- Do not start Stage 1c until Stage 1a has explicit approval evidence from real LLM monitored-channel evaluation.
- Do not start Stage 1c until Stage 1b focused tests pass and memory reset/reseed behavior is complete.
- Stage 1c must not change Stage 1b memory schema except through a separate approved plan.
- Stage 1c must not change Stage 1a approved core observational fields or prompt budgets except through a separate approved plan.
- Stage 1c may add lore-candidate and lore-promotion fields as additive integration work with its own tests; those fields are not treated as already approved by Stage 1a.
- Hourly reflection may write reflection-run documents but must not write global memory.
- Daily reflection may promote at most 1 global lore mutation per character-local day.
- Daily promotion must use Stage 1b memory APIs only.
- Private evidence must be sanitized before global lore promotion.
- User-specific facts, preferences, relationships, health, and commitments must not be stored in global memory.
- Character-spoken or character-agreed evidence is required for lore promotion.
- Boundary core assessment is required for lore promotion.
- Live chat has priority; reflection LLM calls must defer while chat is queued or processing.
- Stage 1c must not send autonomous messages.
- Stage 1c must not call `/chat`.

## Must Do

- Add production `character_reflection_runs` persistence.
- Add monitored-channel production index support on `conversation_history` for latest character-message lookup:

```text
conv_role_ts_platform_channel:
  [("role", 1), ("timestamp", -1), ("platform", 1), ("platform_channel_id", 1)]
```

- Promote Stage 1a read-only monitored-channel selector, message-bearing hour-slot projection, and approved core reflection contracts into a production runtime that can write reflection-run documents.
- Add Stage 1c-specific lore-candidate fields to the production hourly output contract.
- Add daily lore-promotion decision and validation.
- Use Stage 1b `memory_evolution` APIs for insert/supersede/merge.
- Add `REFLECTION_CYCLE_ENABLED`, `REFLECTION_CONTEXT_ENABLED`, and `REFLECTION_LORE_PROMOTION_ENABLED` flags.
- Add service worker lifecycle and idle checks.
- Add prompt-facing reflection context only behind `REFLECTION_CONTEXT_ENABLED`.
- Add CLI for production dry-run/write modes:

```powershell
python src\scripts\run_reflection_cycle.py hourly --dry-run
python src\scripts\run_reflection_cycle.py daily --dry-run
python src\scripts\run_reflection_cycle.py lore-promotion --dry-run
```

- Add tests for reflection persistence, worker idle behavior, daily promotion validation, privacy rejection, boundary rejection, memory API calls, and Cache2 invalidation after promotion.

## Deferred

- Autonomous messages and proactive cognition.
- Per-message reflection.
- Memory schema redesign beyond Stage 1b.
- New seed data or external facts.
- User-memory writes.
- Character-state writes.
- Scheduler event creation.
- Dispatcher send-message changes.

## Cutover Policy

| Area | Policy | Notes |
|---|---|---|
| Reflection persistence | bigbang | Production reflection uses DB run documents, not local artifacts. |
| Memory writes | bigbang | Daily promotion uses Stage 1b memory APIs; no candidate-only production write path. |
| Monitored-channel index | bigbang | Add production index support for latest character-message lookup. |
| Service worker | compatible | Off by default via feature flag. |
| Prompt-facing context | compatible | Off by default via feature flag. |
| Autonomous messaging | compatible | Not implemented. |

## Agent Autonomy Boundaries

- The agent must verify Stage 1a and Stage 1b evidence before coding.
- The agent may adapt Stage 1a function names only if public Stage 1c contracts remain clear.
- The agent must not add a second memory writer.
- The agent must not write directly to `db.memory` except through Stage 1b APIs.
- The agent must not broaden lore promotion to more than 1 item per day.
- If Stage 1a output is not approved, stop; do not implement Stage 1c.

## Target State

```text
hourly worker
  -> select monitored channels by latest character message time
  -> build message-bearing hourly slots
  -> reflection LLM
  -> character_reflection_runs hourly_slot docs

daily worker
  -> daily reflection over per-channel hourly_slot docs
  -> 0 or 1 lore promotion decision
  -> Stage 1b memory_evolution API
  -> memory Cache2 invalidation

normal chat
  -> optional bounded reflection context
  -> existing cognition/dialog path
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Entry gate | Requires 1a approval and 1b completion | Prevents integrating unproven reflection or incomplete memory APIs. |
| Production reflection storage | `character_reflection_runs` | Gives audit trail for hourly slots and daily decisions. |
| 1a core fields | Preserve approved observational fields | Keeps 1c from changing evaluation criteria after approval. |
| Lore fields | Additive in 1c | Lore-candidate fields are integration-specific and need 1c tests. |
| Hourly memory writes | Forbidden | Keeps short-window noise out of lore. |
| Daily promotion | Max 1 | Keeps lore high-signal. |
| Memory writer | Stage 1b API only | Keeps storage rules centralized. |
| Worker | Feature-flagged | Protects live chat while enabling controlled rollout. |
| Monitoring eligibility | Latest character message in last 24 hours | No counter, state variable, or dedicated monitoring collection is needed. |
| Hourly production slot | Every message-bearing hour in monitored channels | User-only and assistant-only hours still provide reflection evidence; skip only empty hours. |

## Data Contracts

```python
class CharacterReflectionRunDoc(TypedDict, total=False):
    run_id: str
    run_kind: Literal["hourly_slot", "daily_global"]
    status: Literal["succeeded", "failed", "skipped", "dry_run"]
    scope: dict
    window_start: str
    window_end: str
    hour_start: str
    hour_end: str
    character_local_date: str
    source_message_refs: list[dict]
    source_reflection_run_ids: list[str]
    output: dict
    lore_promotion_decision: dict
    validation_warnings: list[str]
    error: str
```

```python
class LorePromotionDecision(TypedDict, total=False):
    decision: Literal["promote_new", "supersede", "merge", "reject", "no_action"]
    selected_candidate_id: str
    sanitized_memory_name: str
    sanitized_content: str
    memory_type: str
    authority: str
    signal_strength: Literal["high"]
    character_agreement: Literal["spoken", "agreed"]
    boundary_assessment: dict
    privacy_review: dict
    evidence_refs: list[dict]
```

Promotion requires:

- high signal
- character-spoken or character-agreed evidence
- acceptable boundary assessment
- user details removed
- private-detail risk not high
- empty `source_global_user_id` in memory write

## Change Surface

### Create

- `src/kazusa_ai_chatbot/reflection_cycle/repository.py`
- `src/kazusa_ai_chatbot/reflection_cycle/runtime.py`
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
- `src/kazusa_ai_chatbot/reflection_cycle/lore_promotion.py`
- `src/scripts/run_reflection_cycle.py`
- `tests/test_reflection_cycle_stage1c_repository.py`
- `tests/test_reflection_cycle_stage1c_worker.py`
- `tests/test_reflection_cycle_stage1c_lore_promotion.py`
- `tests/test_reflection_cycle_stage1c_integration.py`

### Modify

- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/db/bootstrap.py`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- Minimal cognition planning prompt module that consumes reflection context.

### Keep

- Stage 1b memory schema and APIs except normal imports.
- Dispatcher send-message path.
- `/chat` request handling.
- User-memory consolidator writers.

## Implementation Order

1. Verify Stage 1a approval artifact and Stage 1b completion evidence.
2. Add tests for reflection repository, promotion validation, and worker idle behavior.
3. Add `character_reflection_runs` schema/indexes and repository.
4. Add production runtime using Stage 1a approved monitored-channel selector, hour-slot projection, and core prompts.
5. Add Stage 1c lore-candidate fields and lore-promotion integration using Stage 1b memory APIs.
6. Add service worker and feature flags.
7. Add prompt-facing reflection context behind flag.
8. Run focused tests.
9. Run dry-run against monitored-channel data.
10. Enable writes only after dry-run evidence is reviewed.

## Progress Checklist

- [ ] Stage 1a approval and Stage 1b completion evidence verified.
- [ ] Repository/promotion/worker tests added.
- [ ] Reflection DB repository implemented.
- [ ] Production runtime implemented.
- [ ] Lore promotion integrated through Stage 1b APIs.
- [ ] Worker and flags implemented.
- [ ] Prompt-facing context implemented behind flag.
- [ ] Focused tests pass.
- [ ] Last-24h dry-run evidence recorded.

## Verification

```powershell
pytest tests\test_reflection_cycle_stage1c_repository.py -q
pytest tests\test_reflection_cycle_stage1c_worker.py -q
pytest tests\test_reflection_cycle_stage1c_lore_promotion.py -q
pytest tests\test_reflection_cycle_stage1c_integration.py -q
```

```powershell
python src\scripts\run_reflection_cycle.py hourly --dry-run
python src\scripts\run_reflection_cycle.py daily --dry-run
python src\scripts\run_reflection_cycle.py lore-promotion --dry-run
```

Run MongoDB explain for the monitored-channel selector and confirm the production index is used.

## Acceptance Criteria

- Stage 1c refuses to proceed without recorded Stage 1a approval and Stage 1b completion evidence.
- Monitored-channel selection uses latest character message time and does not rely on counters or a dedicated monitoring collection.
- Hourly production reflection writes only `character_reflection_runs`.
- Hourly production reflection evaluates every message-bearing hour in monitored channels and skips only empty hours.
- Daily production reflection writes at most one lore mutation through Stage 1b APIs.
- Private/user-specific details are rejected before global memory writes.
- Character agreement and boundary assessment are required.
- Reflection worker is disabled by default and defers while chat is busy.
- Prompt-facing reflection context is disabled by default and bounded when enabled.
- No autonomous messages are sent.

## Validation Against Current Implementation

This stage has not been implemented in the current workspace and remains
blocked.

| Entry Gate Or Component | Expected by plan | Current implementation | Status |
|---|---|---|---|
| Stage 1a approval | Required before coding | Stage 1a plan now requires monitored-channel selection and message-bearing hour slots; implementation and live approval must be refreshed | Blocked |
| Stage 1b completion | Required before coding | `memory_evolution` package and reset CLI do not exist | Blocked |
| `character_reflection_runs` repository | Required | `reflection_cycle/repository.py` does not exist | Not started |
| Production worker | Required | `reflection_cycle/worker.py` does not exist | Not started |
| Lore promotion | Required | `reflection_cycle/lore_promotion.py` does not exist | Not started |
| Production CLI | Required | `src/scripts/run_reflection_cycle.py` does not exist | Not started |
| Feature flags | Required | `REFLECTION_CYCLE_ENABLED`, `REFLECTION_CONTEXT_ENABLED`, and `REFLECTION_LORE_PROMOTION_ENABLED` are not implemented | Not started |
| Autonomous messages | Deferred | No autonomous message path was added | Boundary preserved |
| `/chat` coupling | Forbidden | Stage 1a did not modify `/chat`; Stage 1c must preserve this unless a future Stage 2 plan changes it | Boundary preserved |

Current Stage 1a decisions that Stage 1c must respect:

- Stage 1a public entry points are exported from `kazusa_ai_chatbot.reflection_cycle`.
- Stage 1a selector must read MongoDB only through `db.conversation_reflection`.
- Stage 1a monitored-channel selection must be based on latest character message time, not assistant/user activity counters.
- Stage 1a hourly slots must include every message-bearing hour in monitored channels and skip only empty hours.
- Stage 1a prompt payload keys are stable English machine-facing keys.
- Stage 1a generated free-text language is controlled by a centralized Chinese language-policy block.
- Stage 1a DB message reads use an allowlist projection and may include only bounded attachment descriptions.
- Stage 1a daily synthesis receives compact active-hour slots, not raw transcripts or full hourly reflection objects.
- Stage 1a output remains a local-artifact evaluation contract, not a production storage contract.

Stage 1c remains unsigned. The first Stage 1c checklist item cannot be ticked
until Stage 1a live approval and Stage 1b completion evidence are both recorded.

## Execution Evidence

- Stage 1a approval artifact:
- Stage 1b completion evidence:
- Focused test results:
- Monitored-channel dry-run summary:
- MongoDB explain summary:
- Lore-promotion dry-run summary:
- Any deviations:
