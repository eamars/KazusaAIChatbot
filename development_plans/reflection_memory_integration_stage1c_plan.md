# reflection memory integration stage 1c plan

## Summary

- Goal: Integrate the approved Stage 1a read-only reflection cycle with the completed Stage 1b evolving memory subsystem so per-channel daily reflection can feed a separately approved global promotion step for two durable lanes: global lore and character self-guidance.
- Plan class: large
- Status: draft
- Mandatory skills: `local-llm-architecture`, `development-plan-writing`, `memory-knowledge-maintenance`, `no-prepost-user-input`, `py-style`, `test-style-and-execution`, `database-data-pull`
- Overall cutover strategy: bigbang integration after entry criteria. Stage 1c adds production reflection persistence, monitored-channel indexing, worker wiring, per-channel daily synthesis storage, and two-lane global daily promotion in one approved integration path rather than preserving a parallel candidate-only production mode.
- Highest-risk areas: promoting poor reflection output, private leakage, live cognition latency, memory poisoning, retrieval pollution from behavioral advice, stale Cache2, and bypassing the Stage 1a approval gate.
- Acceptance criteria: Stage 1c starts only after Stage 1a real LLM approval and Stage 1b completion; production reflection writes to `character_reflection_runs`; the new global promotion prompt has its own one-by-one real LLM approval evidence before memory writes are enabled; promotion writes through Stage 1b memory APIs only; private/user-specific details are rejected; live chat remains priority; raw hourly summaries are never consumed directly by cognition; autonomous messaging remains out of scope.

## Context

Stage 1c is the integration stage. It is not where reflection quality is first proven and not where the memory search schema is built.

Inputs:

- Stage 1a provides approved read-only selector/projection logic, narrow hourly observational prompts, daily synthesis chain sanity checks, prompt budgets, and real LLM evaluation artifacts.
- Stage 1b provides evolving memory APIs, active-only search, seed tooling, and Cache2 memory invalidation.

Stage 1c ties them together for production use. Hourly reflection outputs remain
intermediate evidence. They are stored for audit and per-channel daily
synthesis, but they must not be injected into normal chat context directly.
Durable character learning happens only after a separate global promotion
prompt reviews the character-local day's per-channel daily syntheses and
chooses sanitized output for one of the approved lanes.

Stage 1a approves only these model-facing contracts:

- hourly observational reflection,
- per-channel daily synthesis over compact active-hour slots.

Stage 1a does not approve lore or self-guidance promotion. Promotion requires a
new Stage 1c prompt, separate validators, and one-by-one real LLM inspection
before any production memory write can be enabled.

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
- Stage 1c must preserve the approved Stage 1a hourly and per-channel daily output schemas. Do not add promotion fields to those prompts without a separate approval pass.
- Stage 1c promotion decisions belong to a new global promotion prompt. That prompt is not approved by Stage 1a and must pass its own one-by-one real LLM inspection gate before memory writes are enabled.
- Hourly reflection may write reflection-run documents but must not write global memory.
- Per-channel daily reflection may write `daily_channel` reflection-run documents but must not write global memory.
- Global daily promotion may promote at most 1 global lore mutation and at most 1 character self-guidance mutation per character-local day after reviewing all eligible per-channel daily syntheses for that day.
- Daily promotion must use Stage 1b memory APIs only.
- Daily promotion must use Stage 1b `find_active_memory_units` score tuples for duplicate detection before insert, supersede, or merge decisions.
- Daily promotion must construct `EvolvingMemoryDoc` documents directly. It must not call `build_memory_doc`, `save_memory`, or the legacy memory writer path.
- Daily promotion must treat `RuntimeError("memory write or reset is already running")` from Stage 1b APIs as a deferred/skipped promotion for the next worker cycle, not as a fatal worker failure.
- `decision="reject"` means rejecting the current promotion candidate only. Stage 1c must not retire, reject, or deactivate an existing active memory unit without a replacement because Stage 1b intentionally exposes no "mark existing rejected" API.
- Production reflection must disable the Stage 1a evaluation fallback window. If no channel has a latest character message inside the monitor window, the production worker idles.
- Production run documents must persist the raw scope triple `platform`, `platform_channel_id`, and `channel_type` in addition to any hashed `scope_ref`.
- `source_message_refs` and `evidence_refs` must be threaded from the collected input rows and repository metadata, not reconstructed from LLM output.
- Private evidence must be sanitized before any durable promotion.
- User-specific facts, preferences, relationships, health, and commitments must not be stored in global memory.
- Character-spoken or character-agreed evidence is required for lore promotion.
- Self-guidance promotion must describe the active character's future response behavior, not facts about a user.
- Boundary core assessment is required for lore promotion.
- Boundary core assessment is required for self-guidance promotion when the guidance could affect character identity, attachment, intimacy, safety, or setting boundaries.
- Raw hourly summaries and raw hourly reflection documents must not be consumed directly by live cognition or prompt-facing context.
- Prompt-facing reflection context may include only promoted lore and promoted self-guidance; it must be behind `REFLECTION_CONTEXT_ENABLED`.
- Live chat has priority; reflection LLM calls must defer while chat is queued or processing.
- Production reflection workers must isolate failures per hourly slot, per daily channel, and per global promotion run. A failed LLM call, parse, validation, or repository write for one slot must create a failed or skipped run document and must not stop unrelated slots in the same worker pass.
- Production reflection workers may retry an external LLM invocation failure at most once. Parsed JSON validation warnings are not retry triggers; they must be recorded on the run document.
- Every reflection run document must include `prompt_version`, `git_sha`, and `attempt_count` so later memory writes can be audited against the producing code and prompt contract.
- Stage 1c must not send autonomous messages.
- Stage 1c must not call `/chat`.

## Must Do

- Add production `character_reflection_runs` persistence.
- Add prompt/build provenance to all production reflection run documents.
- Persist typed scope metadata with raw `platform`, `platform_channel_id`, and `channel_type` fields on every run document.
- Persist `source_message_refs` from repository/input metadata alongside prompt-safe projections; do not ask the LLM to produce database join keys.
- Use composite source message refs from the Stage 1a input rows: `platform`, `platform_channel_id`, `channel_type`, `timestamp`, and `role`. Leave `conversation_history_id` absent unless a future persistence-only DB projection explicitly adds it.
- Add monitored-channel production index support on `conversation_history` for latest character-message lookup:

```text
conv_role_ts_platform_channel:
  [("role", 1), ("timestamp", -1), ("platform", 1), ("platform_channel_id", 1)]
```

- Promote Stage 1a read-only monitored-channel selector, message-bearing hour-slot projection, and approved core reflection contracts into a production runtime that can write reflection-run documents.
- Preserve Stage 1a hourly and per-channel daily prompt schemas for production reflection storage.
- Add per-channel daily synthesis persistence as `daily_channel` run documents.
- Add a separate global daily promotion prompt that consumes compact `daily_channel` outputs across all monitored channels for one character-local day.
- Add global two-lane promotion decisions and validation:
  - `lore`: character/world/self facts that can become global persistent lore.
  - `self_guidance`: durable character response-behavior lessons that can guide future cognition without storing user-specific facts.
- Add a one-by-one real LLM approval mini-gate for the new global promotion prompt before enabling memory writes.
- Use Stage 1b `memory_evolution` APIs for insert/supersede/merge.
- Use Stage 1b `find_active_memory_units` semantic scores to decide whether validated candidates are similar enough to supersede or merge instead of inserting a new lineage.
- Add `REFLECTION_CYCLE_ENABLED`, `REFLECTION_CONTEXT_ENABLED`, `REFLECTION_LORE_PROMOTION_ENABLED`, and `REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED` flags.
- Add service worker lifecycle and idle checks.
- Add prompt-facing promoted reflection context only behind `REFLECTION_CONTEXT_ENABLED`.
- Add CLI for production dry-run/write modes:

```powershell
python src\scripts\run_reflection_cycle.py hourly --dry-run
python src\scripts\run_reflection_cycle.py daily --dry-run
python src\scripts\run_reflection_cycle.py promote --dry-run
```

- Add tests for reflection persistence, worker idle behavior, daily two-lane promotion validation, privacy rejection, boundary rejection, memory API calls, prompt-facing context filtering, and Cache2 invalidation after promotion.

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
| Memory writes | bigbang | Daily two-lane promotion uses Stage 1b memory APIs; no candidate-only production write path. |
| Monitored-channel index | bigbang | Add production index support for latest character-message lookup. |
| Service worker | compatible | Off by default via feature flag. |
| Prompt-facing context | compatible | Off by default via feature flag; raw hourly summaries are never prompt-facing context. |
| Autonomous messaging | compatible | Not implemented. |

## Agent Autonomy Boundaries

- The agent must verify Stage 1a and Stage 1b evidence before coding.
- The agent may adapt Stage 1a function names only if public Stage 1c contracts remain clear.
- The agent must not add a second memory writer.
- The agent must not write directly to `db.memory` except through Stage 1b APIs.
- The agent must not broaden lore promotion beyond 1 item per character-local day.
- The agent must not broaden self-guidance promotion beyond 1 item per character-local day.
- The agent must not implement raw hourly reflection retrieval in live cognition, even behind a feature flag.
- If Stage 1a output is not approved, stop; do not implement Stage 1c.

## Target State

```text
hourly worker
  -> select monitored channels by latest character message time
  -> idle when no monitor-active channel exists
  -> build message-bearing hourly slots
  -> reflection LLM
  -> character_reflection_runs hourly_slot docs

per-channel daily worker
  -> daily reflection over per-channel hourly_slot docs
  -> character_reflection_runs daily_channel docs

global promotion worker
  -> compact all daily_channel docs for one character-local day
  -> global promotion LLM
  -> character_reflection_runs daily_global_promotion doc
  -> lore lane: 0 or 1 promotion decision
  -> self_guidance lane: 0 or 1 promotion decision
  -> Stage 1b memory_evolution API
  -> memory Cache2 invalidation

normal chat
  -> optional bounded promoted reflection context
  -> existing cognition/dialog path
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Entry gate | Requires 1a approval and 1b completion | Prevents integrating unproven reflection or incomplete memory APIs. |
| Production reflection storage | `character_reflection_runs` | Gives audit trail for hourly slots and daily decisions. |
| 1a core fields | Preserve approved observational fields | Keeps 1c from changing evaluation criteria after approval. |
| Promotion prompt | New global Stage 1c prompt | 1a did not approve promotion fields, character agreement, boundary assessment, privacy review, or evidence refs. |
| Promotion prompt approval | One-by-one real LLM mini-gate before writes | Prevents treating 1a daily synthesis approval as memory-write approval. |
| Daily synthesis granularity | Per-channel `daily_channel` | Matches Stage 1a and preserves reviewed behavior. |
| Promotion granularity | One global promotion prompt per character-local day | Enforces max 1 lore and max 1 self-guidance mutation across all channels. |
| Hourly memory writes | Forbidden | Keeps short-window noise out of lore. |
| Daily lore promotion | Max 1 per character-local day | Keeps lore high-signal. |
| Daily self-guidance promotion | Max 1 per character-local day | Lets behavior improve without storing user image or raw hourly summaries. |
| Raw hourly context | Forbidden for live cognition | Hourly records are evidence for daily synthesis, not prompt-facing memory. |
| Memory writer | Stage 1b API only | Keeps storage rules centralized. |
| Similarity decision input | Use Stage 1b score tuples | Promotion must not re-fetch embeddings or silently insert every candidate as a new lineage when similar active memory exists. |
| Lock contention | Defer promotion to next cycle | Stage 1b fail-fast locking protects reset/runtime writes; worker should record skipped/deferred run state and continue unrelated work. |
| Legacy memory path | Forbidden for 1c writers | 1c must construct `EvolvingMemoryDoc` directly so authority, evidence, privacy, lineage, and source-kind stay explicit. |
| Candidate rejection | Reject only the candidate | Stage 1b has no API to retire existing active lore without replacement, and reflection must not deactivate durable lore solely from a reject decision. |
| Worker | Feature-flagged | Protects live chat while enabling controlled rollout. |
| Monitoring eligibility | Latest character message in last 24 hours | No counter, state variable, or dedicated monitoring collection is needed. |
| Hourly production slot | Every message-bearing hour in monitored channels | User-only and assistant-only hours still provide reflection evidence; skip only empty hours. |
| Production fallback | Disabled | The 168-hour fallback is useful for read-only evaluation only; production should idle when no monitor-active channel exists. |
| Evidence refs | Repository/input-derived, not LLM-derived | Prompt projection deidentifies data; database join keys must be threaded beside the LLM payload. |

## Data Contracts

```python
class CharacterReflectionRunDoc(TypedDict, total=False):
    run_id: str
    run_kind: Literal[
        "hourly_slot",
        "daily_channel",
        "daily_global_promotion",
    ]
    status: Literal["succeeded", "failed", "skipped", "dry_run"]
    prompt_version: str
    git_sha: str
    attempt_count: int
    scope: ReflectionScopeDoc
    window_start: str
    window_end: str
    hour_start: str
    hour_end: str
    character_local_date: str
    source_message_refs: list[dict]
    source_reflection_run_ids: list[str]
    output: dict
    promotion_decisions: list[dict]
    validation_warnings: list[str]
    error: str
```

```python
class ReflectionScopeDoc(TypedDict):
    scope_ref: str
    platform: str
    platform_channel_id: str
    channel_type: str
```

```python
class ReflectionMessageRef(TypedDict, total=False):
    conversation_history_id: str
    platform: str
    platform_channel_id: str
    channel_type: str
    role: Literal["user", "assistant"]
    timestamp: str
```

```python
class ReflectionPromotionDecision(TypedDict, total=False):
    lane: Literal["lore", "self_guidance"]
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

Run-kind ownership:

- `hourly_slot` stores approved hourly observational output and
  `source_message_refs` for the source messages in that hour.
- `daily_channel` stores approved per-channel daily synthesis output and
  `source_reflection_run_ids` for the channel's hourly slots.
- `daily_global_promotion` stores the new Stage 1c global promotion prompt
  output and `source_reflection_run_ids` for all included `daily_channel` docs.

`ReflectionMessageRef` values are produced by repository/input plumbing. They
are not model-facing prompt fields and must not be reconstructed from
`topic_summary`, `day_summary`, or any other LLM output.
Because Stage 1a prompt-facing message projection does not include MongoDB
`_id`, Stage 1c must use the composite fields already present in
`ReflectionMessageRef`. `conversation_history_id` is optional and must remain
absent unless a future DB interface adds a persistence-only projection that is
kept out of the LLM prompt payload.

Global promotion prompt input shape:

```python
class GlobalPromotionPromptPayload(TypedDict):
    evaluation_mode: Literal["daily_global_promotion"]
    character_local_date: str
    channel_daily_syntheses: list[dict]
    evidence_cards: list[dict]
    promotion_limits: dict
    review_questions: list[str]
```

Rules:

- `channel_daily_syntheses` contains compact `daily_channel` parsed outputs,
  run ids, channel type, confidence, validation warning labels, and source
  dates. It must not contain raw transcripts.
- `evidence_cards` contains bounded, sanitized evidence prepared by
  deterministic code from source run metadata. It may include short
  active-character utterance snippets when needed to prove `spoken` or
  `agreed`, but must not include user identity or private user details.
- The prompt may output `ReflectionPromotionDecision` rows only. Deterministic
  validators decide whether any row can call Stage 1b memory APIs.

Lane rules:

- `lore` writes use `source_kind="reflection_inferred"`, empty `source_global_user_id`, and `memory_type="fact"`.
- `self_guidance` writes use `source_kind="reflection_inferred"`, empty `source_global_user_id`, and `memory_type="defense_rule"`.
- The explicit lane is stored in `CharacterReflectionRunDoc.promotion_decisions`. The persisted Stage 1b memory row is lane-addressable by `source_kind` plus `memory_type`; do not add a new Stage 1b memory schema field in Stage 1c.
- `lore` content must describe character/world/self facts, not user facts.
- `self_guidance` content must describe future character response behavior, not the user's profile, preferences, commitments, or relationship state.

Promotion requires for both lanes:

- high signal
- evidence refs pointing to hourly or daily reflection run documents
- user details removed
- private-detail risk not high
- empty `source_global_user_id` in memory write

Before any memory write, deterministic promotion code must call
`find_active_memory_units` with a semantic query built from the sanitized
candidate and `exclude_memory_unit_ids` when replacing or re-checking a known
unit. The returned `(score, memory_doc)` pairs are the only similarity signal
approved for deciding:

- `promote_new`: no active match meets the merge/supersede threshold,
- `supersede`: one active same-lineage match meets the threshold and the new
  content should replace it,
- `merge`: multiple active matches meet the threshold and the new content
  should merge them,
- `reject`: the candidate is rejected and no existing memory unit is mutated.

Do not call lower-level DB helpers for similarity search. Do not fetch stored
embeddings to compute cosine similarity in Python. Do not default to
`promote_new` when the scored search fails; record a skipped/deferred promotion
run instead.

Additional `lore` requirements:

- character-spoken or character-agreed evidence
- acceptable boundary assessment

Additional `self_guidance` requirements:

- behavior guidance is phrased as an instruction to the active character's future responses
- no deterministic keyword gate may rewrite user statements into guidance; the LLM must emit the lane decision directly
- boundary assessment is acceptable when guidance affects character identity, intimacy, attachment, safety, or setting behavior

### Prompt-Facing Reflection Context

`REFLECTION_CONTEXT_ENABLED` may expose only promoted outputs to normal chat.
The context payload must be compact and must not include raw hourly summaries,
raw transcripts, full reflection-run documents, or unpromoted candidates.

Allowed context shape:

```python
class PromotedReflectionContext(TypedDict, total=False):
    promoted_lore: list[dict]
    promoted_self_guidance: list[dict]
    source_dates: list[str]
    retrieval_notes: list[str]
```

Rules:

- `promoted_lore` contains promoted Stage 1b memory rows with `source_kind="reflection_inferred"` and `memory_type="fact"` only.
- `promoted_self_guidance` contains promoted Stage 1b memory rows with `source_kind="reflection_inferred"` and `memory_type="defense_rule"` only.
- Each list must be capped before entering cognition. Default cap is 3 rows per lane.
- The projection must summarize row meaning for the local LLM; do not pass raw database metadata except stable source dates and short evidence notes.
- If no promoted rows are relevant, omit the reflection context entirely.

## LLM Call And Context Budget

| Path | Calls | Blocking | Context budget | Notes |
|---|---:|---|---:|---|
| Hourly reflection worker | 1 LLM call per message-bearing hour | Background only | 8000 chars per hour prompt | Uses Stage 1a approved hourly cap and schema. |
| Per-channel daily synthesis worker | 1 LLM call per monitored channel/day | Background only | 25000 chars per daily prompt | Uses Stage 1a approved per-channel daily cap and schema. |
| Global promotion worker | 1 LLM call per character-local day | Background only | 25000 chars per promotion prompt | New Stage 1c prompt consuming compact `daily_channel` docs and bounded evidence cards; requires one-by-one real LLM approval before writes. |
| Promotion persistence | 0 LLM calls | Background only | N/A | Deterministic validation and Stage 1b API writes only; no semantic reinterpretation of lane decisions. |
| Normal chat reflection context | 0 new LLM calls | Response path only when flag enabled | Max 3 promoted lore rows and 3 promoted self-guidance rows | Adds bounded promoted memory context to existing cognition; raw hourly data is forbidden. |

No Stage 1c reflection LLM call is allowed on the live response critical path.
The worker must defer while chat is queued or processing.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/reflection_cycle/repository.py`
- `src/kazusa_ai_chatbot/reflection_cycle/runtime.py`
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
- `src/scripts/run_reflection_cycle.py`
- `tests/test_reflection_cycle_stage1c_repository.py`
- `tests/test_reflection_cycle_stage1c_worker.py`
- `tests/test_reflection_cycle_stage1c_promotion.py`
- `tests/test_reflection_cycle_stage1c_promotion_live_llm.py`
- `tests/test_reflection_cycle_stage1c_reflection_context.py`
- `tests/test_reflection_cycle_stage1c_integration.py`

### Modify

- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/db/bootstrap.py`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- Minimal cognition planning prompt module that consumes promoted reflection context.

### Keep

- Stage 1b memory schema and APIs except normal imports.
- Dispatcher send-message path.
- `/chat` request handling.
- User-memory consolidator writers.

## Implementation Order

1. Verify Stage 1a approval artifact and Stage 1b completion evidence.
2. Add tests for reflection repository, promotion validation, and worker idle behavior.
3. Add `character_reflection_runs` schema/indexes and repository.
4. Add production runtime using Stage 1a approved monitored-channel selector, hour-slot projection, per-channel daily projection, and core prompts. Keep the 168-hour fallback disabled in production mode.
5. Add per-slot worker error isolation with bounded external-LLM retry and failed-run persistence.
6. Add global promotion prompt, promotion validators, and dry-run artifact output. Do not enable memory writes yet.
7. Run the global promotion prompt real LLM mini-gate one case at a time and record inspection evidence.
8. Add two-lane promotion integration using Stage 1b memory APIs.
   - Include tests for scored similarity decisions and lock-contention deferral.
9. Add service worker and feature flags.
10. Add prompt-facing promoted reflection context behind flag; raw hourly summaries must not be retrievable by live cognition.
11. Run focused tests.
12. Run dry-run against monitored-channel data.
13. Enable writes only after dry-run evidence and promotion prompt approval are reviewed.

## Progress Checklist

- [ ] Stage 1a approval and Stage 1b completion evidence verified.
- [ ] Repository/promotion/worker tests added.
- [ ] Reflection DB repository implemented.
- [ ] Production runtime implemented.
- [ ] Per-slot worker failure isolation implemented and verified.
- [ ] Global promotion prompt implemented in dry-run mode and real LLM mini-gate approved.
- [ ] Two-lane promotion integrated through Stage 1b APIs.
- [ ] Worker and flags implemented.
- [ ] Prompt-facing promoted context implemented behind flag.
- [ ] Focused tests pass.
- [ ] Last-24h dry-run evidence recorded.

## Verification

```powershell
pytest tests\test_reflection_cycle_stage1c_repository.py -q
pytest tests\test_reflection_cycle_stage1c_worker.py -q
pytest tests\test_reflection_cycle_stage1c_promotion.py -q
pytest tests\test_reflection_cycle_stage1c_reflection_context.py -q
pytest tests\test_reflection_cycle_stage1c_integration.py -q
```

Run real LLM promotion-prompt tests one by one and inspect each artifact before
continuing:

```powershell
pytest tests\test_reflection_cycle_stage1c_promotion_live_llm.py::test_global_promotion_live_normal_case -q -s
pytest tests\test_reflection_cycle_stage1c_promotion_live_llm.py::test_global_promotion_live_privacy_rejection_case -q -s
pytest tests\test_reflection_cycle_stage1c_promotion_live_llm.py::test_global_promotion_live_no_signal_case -q -s
```

```powershell
python src\scripts\run_reflection_cycle.py hourly --dry-run
python src\scripts\run_reflection_cycle.py daily --dry-run
python src\scripts\run_reflection_cycle.py promote --dry-run
```

Run MongoDB explain for the monitored-channel selector and confirm the production index is used.

Run a source grep proving Stage 1c writers do not use the legacy memory path:

```powershell
rg "build_memory_doc|save_memory" src\kazusa_ai_chatbot\reflection_cycle src\scripts\run_reflection_cycle.py
```

Allowed result: no matches in Stage 1c production writer code.

## Operational Steps

Existing Atlas vector search indexes cannot be edited in place. The current
Stage 1b runtime search path does not require scalar vector filter fields
because it post-filters after vector scoring. If Stage 1c or a future
optimization reintroduces `$vectorSearch.filter`, deployments that already
have `memory_vector_index` must plan a manual Atlas drop-and-recreate operation
for that index. Do not treat the bootstrap helper as proof that an existing
Atlas index definition was modified.

Before enabling promotion writes:

- inspect the existing `memory_vector_index` definition,
- do not require scalar filter paths unless the implementation uses
  `$vectorSearch.filter`,
- recreate the index with the needed filter paths if vector prefilters are
  reintroduced,
- run a semantic `find_active_memory_units` dry-run and confirm scores are
  present,
- keep promotion writes disabled if vector scores are unavailable.

## Acceptance Criteria

- Stage 1c refuses to proceed without recorded Stage 1a approval and Stage 1b completion evidence.
- Monitored-channel selection uses latest character message time and does not rely on counters or a dedicated monitoring collection.
- Production monitoring disables the read-only fallback window and idles when no monitor-active channel exists.
- Hourly production reflection writes only `character_reflection_runs`.
- Hourly production reflection evaluates every message-bearing hour in monitored channels and skips only empty hours.
- Per-channel daily production reflection writes `daily_channel` run documents and does not write memory.
- Global promotion writes `daily_global_promotion` run documents and is the only reflection step allowed to produce promotion decisions.
- The global promotion prompt has recorded one-by-one real LLM approval evidence before memory writes are enabled.
- Hourly, daily-channel, and global-promotion production run documents include prompt/build provenance.
- Run documents persist raw scope triples in addition to hashed scope refs.
- Evidence refs and source message refs are repository/input-derived, not LLM-derived.
- Source message refs use composite message identity fields unless a future DB interface explicitly provides `conversation_history_id`.
- Promotion duplicate detection uses Stage 1b similarity scores before choosing `promote_new`, `supersede`, or `merge`.
- Memory write lock contention records a skipped/deferred promotion and does not fail the whole worker pass.
- Stage 1c writers construct `EvolvingMemoryDoc` directly and do not call `build_memory_doc` or `save_memory`.
- `decision="reject"` never mutates an existing active memory unit.
- One failed hourly slot, daily channel, or global promotion run records a failed/skipped run document and does not stop unrelated work in the same worker pass.
- Global promotion writes at most one lore mutation and at most one self-guidance mutation through Stage 1b APIs per character-local day.
- Private/user-specific details are rejected before all global memory writes.
- Lore promotion requires character agreement and boundary assessment.
- Self-guidance promotion requires behavior-only content and boundary assessment when it can affect character identity, intimacy, attachment, safety, or setting behavior.
- Reflection worker is disabled by default and defers while chat is busy.
- Prompt-facing reflection context is disabled by default and bounded when enabled.
- Prompt-facing reflection context never includes raw hourly summaries or raw `hourly_slot` run outputs.
- No autonomous messages are sent.

## Validation Against Current Implementation

This stage has not been implemented in the current workspace and remains
blocked.

| Entry Gate Or Component | Expected by plan | Current implementation | Status |
|---|---|---|---|
| Stage 1a approval | Required before coding | Stage 1a plan now requires monitored-channel selection and message-bearing hour slots; implementation and live approval must be refreshed | Blocked |
| Stage 1b completion | Required before coding | `memory_evolution` package, reset CLI, active-only retrieval, scored semantic search, and completion evidence exist in the current workspace | Ready after 1b review |
| `character_reflection_runs` repository | Required | `reflection_cycle/repository.py` does not exist | Not started |
| Production worker | Required | `reflection_cycle/worker.py` does not exist | Not started |
| Two-lane promotion | Required | `reflection_cycle/promotion.py` does not exist | Not started |
| Global promotion prompt approval | Required | No promotion prompt or real LLM mini-gate exists | Not started |
| Per-slot error isolation | Required | No production worker exists; read-only evaluation does not provide retry semantics | Not started |
| Prompt-facing promoted context | Required | raw-hourly-excluding reflection context module does not exist | Not started |
| Production CLI | Required | `src/scripts/run_reflection_cycle.py` does not exist | Not started |
| Feature flags | Required | `REFLECTION_CYCLE_ENABLED`, `REFLECTION_CONTEXT_ENABLED`, `REFLECTION_LORE_PROMOTION_ENABLED`, and `REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED` are not implemented | Not started |
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
- Stage 1a daily synthesis is per-channel and must remain separate from global promotion.
- Stage 1a does not approve promotion fields, boundary assessment, privacy review, character-agreement assessment, or evidence refs.
- Stage 1a's 168-hour fallback is evaluation-only and must be disabled for production write-capable runs.
- Stage 1a output remains a local-artifact evaluation contract, not a production storage contract.
- Stage 1a hourly summaries are evidence for Stage 1c daily reflection only; they must not become direct cognition context.

Stage 1c remains unsigned. The first Stage 1c checklist item cannot be ticked
until Stage 1a live approval and Stage 1b completion evidence are both recorded.

## Execution Evidence

- Stage 1a approval artifact:
- Stage 1b completion evidence:
- Focused test results:
- Monitored-channel dry-run summary:
- MongoDB explain summary:
- Global promotion prompt real LLM approval:
- Promotion dry-run summary:
- Prompt-facing context filtering evidence:
- Any deviations:
