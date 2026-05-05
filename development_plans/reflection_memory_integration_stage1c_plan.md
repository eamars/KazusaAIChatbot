# reflection memory integration stage 1c plan

## Summary

- Goal: Integrate the approved Stage 1a read-only reflection cycle with the completed Stage 1b evolving memory subsystem so per-channel daily reflection can feed a separately approved global promotion step for two durable lanes: global lore and character self-guidance.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `local-llm-architecture`, `development-plan-writing`, `memory-knowledge-maintenance`, `no-prepost-user-input`, `py-style`, `cjk-safety`, `test-style-and-execution`, `database-data-pull`
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

## Entry Gate Handoff

Stage 1c may start only from these completed interfaces:

| Source stage | Required handoff artifact | Stage 1c instruction |
|---|---|---|
| Stage 1a | `development_plans/character_reflection_cycle_stage1a_plan.md` with completed real LLM approval and monitored-channel artifact evidence | Reuse only the public `kazusa_ai_chatbot.reflection_cycle` entry points and Stage 1a prompt schemas. Do not change the approved hourly or per-channel daily model-facing contracts. |
| Stage 1a ICD | `src/kazusa_ai_chatbot/reflection_cycle/README.md` | Extend the ICD for production write-capable Stage 1c interfaces before wiring service or memory integration. Do not bypass its DB-read boundary. |
| Stage 1b | `development_plans/memory_evolution_stage1b_plan.md` with completion evidence | Write memory only through `kazusa_ai_chatbot.memory_evolution` public APIs and use returned score tuples for similarity decisions. |
| Stage 1b ICD | `src/kazusa_ai_chatbot/memory_evolution/README.md` | Treat this as the memory interface contract. Stage 1c must not import `kazusa_ai_chatbot.db.memory_evolution` or operate on `db.memory` directly. |

The implementation agent must copy the exact Stage 1a approval artifact path
and Stage 1b focused test command/result into `Execution Evidence` before
coding. If either artifact is missing or contradicted by current code, stop and
report the blocker instead of implementing Stage 1c.

## Mandatory Skills

- `local-llm-architecture`: load before changing production reflection prompts, service worker scheduling, or prompt-facing context.
- `development-plan-writing`: load before modifying this plan.
- `memory-knowledge-maintenance`: load before writing reflection-inferred global memory rows.
- `no-prepost-user-input`: load before changing character-agreement or lore-admission logic.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python prompt files that contain Chinese instructions or examples.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before validating production-like runs against live data.

## Mandatory Rules

This section defines execution invariants and prohibitions. These are not a
task list. If a rule implies code or tests are needed, the corresponding
deliverable appears in `Must Do`, `Implementation Order`, or `Verification`.

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major checklist stage, the active agent must reread this entire plan before starting the next stage.
- Do not start Stage 1c until Stage 1a has explicit approval evidence from real LLM monitored-channel evaluation.
- Do not start Stage 1c until Stage 1b focused tests pass and memory reset/reseed behavior is complete.
- Stage 1c must not change Stage 1b memory schema except through a separate approved plan.
- Stage 1c must not change Stage 1a approved core observational fields or prompt budgets except through a separate approved plan.
- Stage 1c must preserve the approved Stage 1a hourly and per-channel daily output schemas. Do not add promotion fields to those prompts without a separate approval pass.
- Stage 1c promotion decisions belong to a new global promotion prompt. That prompt is not approved by Stage 1a and must pass its own one-by-one real LLM inspection gate before memory writes are enabled.
- The global promotion prompt, prompt version, LLM instance, handler, payload builder, validators, lane constants, and similarity resolver belong in `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`. Do not add the global promotion prompt to Stage 1a `prompts.py`.
- The global promotion prompt must reuse Stage 1a's language policy: JSON keys and enum values stay English; newly generated free-text fields use Simplified Chinese; source snippets remain in their original language when precision matters.
- All Stage 1c reflection LLM calls must use the existing consolidation LLM
  route from `config.py`: `CONSOLIDATION_LLM_BASE_URL`,
  `CONSOLIDATION_LLM_API_KEY`, and `CONSOLIDATION_LLM_MODEL`. Do not add
  `REFLECTION_LLM_BASE_URL` or another reflection-specific LLM route in Stage
  1c.
- Hourly reflection may write reflection-run documents but must not write global memory.
- Per-channel daily reflection may write `daily_channel` reflection-run documents but must not write global memory.
- Global daily promotion may promote at most 1 global lore mutation and at most 1 character self-guidance mutation per character-local day after reviewing all eligible per-channel daily syntheses for that day.
- Daily promotion must use Stage 1b memory APIs only.
- Daily promotion must use Stage 1b `find_active_memory_units` score tuples for duplicate detection before insert, supersede, or merge decisions.
- Daily promotion must construct `EvolvingMemoryDoc` documents directly. It must not call `build_memory_doc`, `save_memory`, or the legacy memory writer path.
- Daily promotion must use `MemoryAuthority.REFLECTION_PROMOTED`, `MemorySourceKind.REFLECTION_INFERRED`, `MemoryStatus.ACTIVE`, and an empty `source_global_user_id` for every Stage 1c memory write.
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
- Reflection is lower priority than `/chat`, assistant-message persistence, conversation progress recording, post-dialog consolidation, and scheduled-event dispatch. If any primary interaction or consolidation work is busy, the reflection worker must skip or defer its current tick before starting a new LLM call or memory write.
- The reflection worker must be started by default from FastAPI lifespan wiring
  in `service.py`. It starts unless `REFLECTION_CYCLE_DISABLED=true`. It must
  not use the `scheduled_events` collection, FastAPI `BackgroundTasks`,
  `/chat`, or dispatcher send-message paths.
- The service layer may pass a busy-probe callback into the worker. The worker must not import `service.py` or reach into `_chat_input_queue` directly.
- `kazusa_ai_chatbot.reflection_cycle` must not call `get_db`, Motor/PyMongo collection methods, raw MongoDB aggregation/find/update/delete operations, or embedding adapters directly. MongoDB access belongs in `kazusa_ai_chatbot.db.conversation_reflection` for conversation reads and `kazusa_ai_chatbot.db.reflection_cycle` for reflection-run persistence.
- Production reflection workers must isolate failures per hourly slot, per daily channel, and per global promotion run. A failed LLM call, parse, validation, or repository write for one slot must create a failed or skipped run document and must not stop unrelated slots in the same worker pass.
- Production reflection workers may retry an external LLM invocation failure at most once. Parsed JSON validation warnings are not retry triggers; they must be recorded on the run document.
- Any reflection run document written by Stage 1c must include `prompt_version` and
  `attempt_count` so later memory writes can be audited against the prompt
  contract and retry path.
- Any reflection run document written by Stage 1c must use `run_id` as MongoDB `_id` and must also store `run_id` as a normal field for projection/debug readability.
- Reflection logs must follow the logging contract in this plan: `INFO` is for
  operator-critical lifecycle and promotion outcomes; `DEBUG` is for
  supporting evidence, scores, confidence, validation warnings, and prompt
  budget details. Neither level may log raw transcripts, raw hourly outputs,
  user identifiers, or unsanitized private details.
- Stage 1c must not send autonomous messages.
- Stage 1c must not call `/chat`.

## Must Do

This section defines required deliverables. It should name the modules,
interfaces, flags, prompts, indexes, tests, and CLI work that must exist by the
end of Stage 1c. Behavior limits and prohibitions live in `Mandatory Rules`.

- Add production `character_reflection_runs` persistence.
- Add `kazusa_ai_chatbot.db.reflection_cycle` as the only DB-interface module for `character_reflection_runs` reads/writes and reflection-run indexes.
- Update `src/kazusa_ai_chatbot/reflection_cycle/README.md` into the Stage 1c production ICD, including public write-capable interfaces, DB boundaries, service scheduling, feature flags, memory API dependency, prompt-facing context rules, and forbidden imports.
- Add Stage 1c public facades exported from `kazusa_ai_chatbot.reflection_cycle`; service, CLI, and tests must use these facades instead of importing worker/repository internals.
- Add prompt version and attempt metadata to all production reflection run
  documents.
- Add prompt version constants:
  - Hourly and daily production run documents use the existing Stage 1a `READONLY_REFLECTION_PROMPT_VERSION` for their unchanged prompts.
  - Global promotion uses `GLOBAL_PROMOTION_PROMPT_VERSION = "reflection_global_promotion_v1"` in `promotion.py`.
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
- Add the exact global promotion prompt skeleton, typed payload builders, and prompt-render test described in this plan.
- Instantiate the global promotion LLM in `promotion.py` with `get_llm(...)`
  using `CONSOLIDATION_LLM_BASE_URL`, `CONSOLIDATION_LLM_API_KEY`, and
  `CONSOLIDATION_LLM_MODEL`, matching Stage 1a hourly and daily reflection.
- Add global two-lane promotion decisions and validation:
  - `lore`: character/world/self facts that can become global persistent lore.
  - `self_guidance`: durable character response-behavior lessons that can guide future cognition without storing user-specific facts.
- Add a one-by-one real LLM approval mini-gate for the new global promotion prompt before enabling memory writes.
- Use Stage 1b `memory_evolution` APIs for insert/supersede/merge.
- Use Stage 1b `find_active_memory_units` semantic scores to decide whether validated candidates are similar enough to supersede or merge instead of inserting a new lineage.
- Add `REFLECTION_CYCLE_DISABLED`, `REFLECTION_CONTEXT_ENABLED`, `REFLECTION_LORE_PROMOTION_ENABLED`, and `REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED` flags.
- Add `REFLECTION_WORKER_INTERVAL_SECONDS`, `REFLECTION_HOURLY_SLOTS_PER_TICK`, `REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME`, and `REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME` config values with the defaults specified in this plan.
- Add FastAPI lifespan worker lifecycle, shutdown, and idle/busy checks.
- Add prompt-facing promoted reflection context only behind `REFLECTION_CONTEXT_ENABLED`.
- Add CLI for production dry-run/write modes:

```powershell
python src\scripts\run_reflection_cycle.py hourly --dry-run
python src\scripts\run_reflection_cycle.py daily --dry-run
python src\scripts\run_reflection_cycle.py promote --dry-run
```

- Add tests for reflection persistence, worker idle behavior, daily two-lane promotion validation, privacy rejection, boundary rejection, memory API calls, prompt-facing context filtering, logging level separation, and Cache2 invalidation after promotion.
- Add a deterministic score-availability test proving `run_global_reflection_promotion(enable_memory_writes=True)` records skipped/deferred output and performs no memory write when `find_active_memory_units` does not return `(score, document)` tuples.
- Add FastAPI scheduling tests proving lifespan starts/stops the reflection worker by default, skips startup when explicitly disabled, and defers while primary chat/consolidation work is busy.

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
| Stage 1c DB interface | bigbang | Production reflection-run persistence uses `db.reflection_cycle`; no direct collection writes from `reflection_cycle`. |
| Reflection persistence | bigbang | Production reflection uses DB run documents, not local artifacts. |
| Memory writes | bigbang | Daily two-lane promotion uses Stage 1b memory APIs; no candidate-only production write path. |
| Monitored-channel index | bigbang | Add production index support for latest character-message lookup. |
| Service worker schedule | compatible | On by default after Stage 1c is deployed; explicitly disabled only with `REFLECTION_CYCLE_DISABLED=true`; started only from FastAPI lifespan. |
| Prompt-facing context | compatible | Off by default via feature flag; raw hourly summaries are never prompt-facing context. |
| Autonomous messaging | compatible | Not implemented. |

## Agent Autonomy Boundaries

- The agent must verify Stage 1a and Stage 1b evidence before coding.
- The agent must use the public interfaces named in this plan. If current code lacks a named public entry point, update the plan or ICD before inventing a replacement.
- The agent must not add a second memory writer.
- The agent must not write directly to `db.memory` except through Stage 1b APIs.
- The agent must not write directly to `character_reflection_runs` from `reflection_cycle` modules; all reflection-run collection operations must go through `kazusa_ai_chatbot.db.reflection_cycle`.
- The agent must not import `service.py` from `reflection_cycle`. Service-owned state is exposed only through the busy-probe callback passed into the worker.
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

## Public Interfaces

Stage 1c public entry points must be exported from
`kazusa_ai_chatbot.reflection_cycle.__init__`:

```python
async def run_hourly_reflection_cycle(
    *,
    now: datetime | None = None,
    dry_run: bool,
) -> ReflectionWorkerResult: ...

async def run_daily_channel_reflection_cycle(
    *,
    character_local_date: str,
    dry_run: bool,
) -> ReflectionWorkerResult: ...

async def run_global_reflection_promotion(
    *,
    character_local_date: str,
    dry_run: bool,
    enable_memory_writes: bool,
) -> ReflectionPromotionResult: ...

def start_reflection_cycle_worker(
    *,
    is_primary_interaction_busy: Callable[[], bool],
) -> ReflectionWorkerHandle: ...

async def stop_reflection_cycle_worker(
    handle: ReflectionWorkerHandle,
) -> None: ...
```

Rules:

- CLI and service code must call these public facades.
- Tests may import internals only for focused module-unit coverage.
- `enable_memory_writes` must remain a required keyword-only argument with no
  default. Callers must choose explicitly for every global promotion call.
- The global promotion live LLM mini-gate is a procedural implementation and
  release gate recorded in `Execution Evidence`. Runtime code does not read a
  mini-gate or release marker. After that gate is recorded,
  `enable_memory_writes=True` is the caller's explicit per-run approval for
  memory writes.
- `dry_run=True` may write local CLI artifacts but must not write memory rows.
  Production worker dry-runs may write `character_reflection_runs` with
  `status="dry_run"` only when the CLI or test explicitly asks for DB dry-run
  evidence.

## DB Interface Contract

`kazusa_ai_chatbot.db.reflection_cycle` owns all MongoDB operations for
`character_reflection_runs`. Approved DB-interface functions:

```python
async def ensure_reflection_run_indexes() -> None: ...
async def upsert_reflection_run(document: CharacterReflectionRunDoc) -> None: ...
async def find_reflection_run_by_id(run_id: str) -> CharacterReflectionRunDoc | None: ...
async def list_hourly_runs_for_channel_day(
    *,
    scope_ref: str,
    character_local_date: str,
) -> list[CharacterReflectionRunDoc]: ...
async def list_daily_channel_runs(
    *,
    character_local_date: str,
) -> list[CharacterReflectionRunDoc]: ...
async def list_existing_run_ids(run_ids: list[str]) -> set[str]: ...
```

Repository and worker modules must import this module as a DB interface. They
must not import `get_db` or use collection methods directly.

`character_reflection_runs` indexes:

```text
reflection_run_id_unique:
  [("run_id", 1)], unique

reflection_kind_date_status:
  [("run_kind", 1), ("character_local_date", 1), ("status", 1)]

reflection_scope_hour:
  [("scope.scope_ref", 1), ("hour_start", 1)]

reflection_source_run_ids:
  [("source_reflection_run_ids", 1)]
```

All upserts must set `_id=run_id`. The unique `run_id` index is still required
so tests and projections can assert idempotency without relying on `_id`
semantics alone.

`db.conversation_reflection` remains the only DB interface for
conversation-history reflection reads. Stage 1c may add persistence-only
message reference support there only if the prompt-facing allowlist remains
unchanged and README tests prove `_id` is not sent to the LLM.

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
    boundary_assessment: ReflectionBoundaryAssessment
    privacy_review: MemoryPrivacyReview
    evidence_refs: list[MemoryEvidenceRef]
```

```python
class ReflectionBoundaryAssessment(TypedDict):
    verdict: Literal["acceptable", "needs_human_review", "blocked"]
    affects_identity_or_boundaries: bool
    reason: str
```

`MemoryPrivacyReview`, `MemoryEvidenceRef`, and `MemoryEvidenceMessageRef` are
imported from `kazusa_ai_chatbot.memory_evolution`. Stage 1c must not invent a
parallel privacy/evidence schema.

Run-kind ownership:

- `hourly_slot` stores approved hourly observational output and
  `source_message_refs` for the source messages in that hour.
- `daily_channel` stores approved per-channel daily synthesis output and
  `source_reflection_run_ids` for the channel's hourly slots.
- `daily_global_promotion` stores the new Stage 1c global promotion prompt
  output and `source_reflection_run_ids` for all included `daily_channel` docs.

Run idempotency:

- `hourly_slot.run_id` must be deterministic from
  `("hourly_slot", scope_ref, hour_start, prompt_version)`.
- `daily_channel.run_id` must be deterministic from
  `("daily_channel", scope_ref, character_local_date, prompt_version)`.
- `daily_global_promotion.run_id` must be deterministic from
  `("daily_global_promotion", character_local_date, prompt_version)`.
- Repository writes must upsert by `run_id`. A retry updates the same run
  document's `attempt_count`, `status`, and output fields; it must not create a
  second run document for the same logical slot.

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
class ChannelDailySynthesisCard(TypedDict):
    daily_run_id: str
    scope_ref: str
    channel_type: str
    character_local_date: str
    confidence: Literal["low", "medium", "high"]
    day_summary: str
    cross_hour_topics: list[str]
    conversation_quality_patterns: list[str]
    privacy_risk_labels: list[str]
    validation_warning_labels: list[str]


class ReflectionEvidenceCard(TypedDict, total=False):
    evidence_card_id: str
    source_reflection_run_ids: list[str]
    scope_ref: str
    channel_type: str
    character_local_date: str
    active_character_utterance: str
    sanitized_observation: str
    supports: list[Literal["lore", "self_guidance"]]
    private_detail_risk: Literal["low", "medium", "high"]


class PromotionLimits(TypedDict):
    max_lore: Literal[1]
    max_self_guidance: Literal[1]
    max_total_decisions: Literal[2]


class GlobalPromotionPromptPayload(TypedDict):
    evaluation_mode: Literal["daily_global_promotion"]
    character_local_date: str
    character_time_zone: str
    channel_daily_syntheses: list[ChannelDailySynthesisCard]
    evidence_cards: list[ReflectionEvidenceCard]
    promotion_limits: PromotionLimits
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
- Deterministic payload builders cap `channel_daily_syntheses` at 25 channels,
  each `ChannelDailySynthesisCard` at 600 serialized characters, and
  `evidence_cards` at 40 cards with 360 serialized characters per card. If the
  cap drops data, the prompt payload must include validation warning labels
  that make the omitted count visible.

Lane rules:

- `PROMOTION_LANE_MEMORY_TYPE = {"lore": "fact", "self_guidance": "defense_rule"}` must live in `promotion.py` and must be used by validators, memory-document builders, tests, and prompt-facing context filters. Do not duplicate these string literals in multiple places.
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

Similarity resolution is deterministic:

- Query filters must include `source_kind="reflection_inferred"`,
  `source_global_user_id=""`, and the lane's `memory_type`.
- `PROMOTION_DUPLICATE_SCORE_THRESHOLD = 0.92`.
- `PROMOTION_MERGE_SCORE_THRESHOLD = 0.88`.
- `PROMOTION_REVIEW_BAND_SCORE_THRESHOLD = 0.82`.
- If two or more active rows in the same lane score at or above `0.88`, the
  resolver uses `merge_memory_units`.
- If exactly one active row scores at or above `0.92`, the resolver uses
  `supersede_memory_unit`.
- If the top score is at or above `0.82` but below the mutation threshold,
  record `status="skipped"` with a duplicate-review warning and do not write.
- Only when every returned score is below `0.82` may the resolver call
  `insert_memory_unit` for a new lineage.
- `decision="reject"` or `decision="no_action"` from the LLM always means no
  memory mutation. Deterministic code may reject or skip an unsafe promote row,
  but it must not rewrite rejected/no-action rows into writes.
- The LLM decision is not allowed to bypass these thresholds. The deterministic
  resolver owns final insert/supersede/merge mode for any validated promote row.
- If `find_active_memory_units` returns any row that is not a two-item tuple
  with a numeric score and memory document, or if semantic search raises before
  returning scores, the resolver records a skipped/deferred promotion and must
  not write memory.

Memory ids:

- New reflection memory unit ids must use
  `memory_evolution.identity.deterministic_memory_unit_id("reflection", parts)`.
- `parts` must include lane, character-local date, sanitized memory name,
  sanitized content, and sorted source reflection run ids.
- New-lineage inserts set `lineage_id=memory_unit_id` and `version=1`.
- Supersede replacements use the target row's `lineage_id`; merge replacements
  follow the Stage 1b merge lineage rules.

Additional `lore` requirements:

- character-spoken or character-agreed evidence
- acceptable boundary assessment

Additional `self_guidance` requirements:

- behavior guidance is phrased as an instruction to the active character's future responses
- no deterministic keyword gate may rewrite user statements into guidance; the LLM must emit the lane decision directly
- boundary assessment is acceptable when guidance affects character identity, intimacy, attachment, safety, or setting behavior

### Global Promotion Prompt Contract

`promotion.py` must keep the global promotion LLM stage as one local block:

```text
GLOBAL_PROMOTION_PROMPT_VERSION
GLOBAL_PROMOTION_SYSTEM_PROMPT
_global_promotion_llm
run_global_promotion_llm(...)
```

The system prompt must use this exact section skeleton, with Simplified Chinese
instructions and English schema keys:

```text
# 角色
你负责审阅每日频道反思，只输出可验证、去隐私、可长期使用的全局晋升决定。

# 核心任务
在 lore 与 self_guidance 两个通道中，各最多选择一条高信号内容；没有足够证据时输出 no_action 或 reject。

# 语言政策
JSON key 和枚举值必须保持英文。你新生成的自由文本字段必须使用简体中文。证据片段保持原文。

# 生成步骤
1. 检查 channel_daily_syntheses，只把它当作压缩后的反思证据。
2. 检查 evidence_cards，确认是否有 active_character_utterance 支持角色说过或同意过的内容。
3. 排除用户事实、用户偏好、关系承诺、健康信息、私密身份信息。
4. 分别判断 lore 与 self_guidance 是否有 high signal。
5. 输出 promotion_decisions；不要输出数据库字段、Mongo 查询、embedding、source_global_user_id。

# 输入格式
<GlobalPromotionPromptPayload JSON schema summary>

# 输出格式
只输出 JSON：{"promotion_decisions": [ReflectionPromotionDecision, ...]}

# 禁止事项
不要编造证据。不要从用户发言改写成角色长期规则。不要把 reject/no_action 改成 promote。
```

The prompt-render test must assert these section headers exist and that the
rendered prompt contains `evaluation_mode`, `channel_daily_syntheses`,
`evidence_cards`, `promotion_limits`, and `promotion_decisions`.

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
| Hourly reflection worker | 1 LLM call per message-bearing hour, capped at 3 slots per worker tick by default | Background only | 8000 chars per hour prompt | Uses Stage 1a approved hourly cap and schema. |
| Per-channel daily synthesis worker | 1 LLM call per monitored channel/day | Background only | 25000 chars per daily prompt | Uses Stage 1a approved per-channel daily cap and schema. |
| Global promotion worker | 1 LLM call per character-local day | Background only | 25000 chars per promotion prompt | New Stage 1c prompt consuming compact `daily_channel` docs and bounded evidence cards; requires one-by-one real LLM approval before writes. |
| Promotion persistence | 0 LLM calls | Background only | N/A | Deterministic validation and Stage 1b API writes only; no semantic reinterpretation of lane decisions. |
| Normal chat reflection context | 0 new LLM calls | Response path only when flag enabled | Max 3 promoted lore rows and 3 promoted self-guidance rows | Adds bounded promoted memory context to existing cognition; raw hourly data is forbidden. |

No Stage 1c reflection LLM call is allowed on the live response critical path.
The worker must defer while chat is queued or processing.

`character_local_date` and daily run gates use `CHARACTER_TIME_ZONE` from
`config.py` via `ZoneInfo(CHARACTER_TIME_ZONE)`. Stage 1a hourly buckets remain
UTC, but daily grouping assigns each UTC hourly slot to a character-local date
after converting the slot's `hour_start` into `CHARACTER_TIME_ZONE`. The
`04:30` and `05:00` run gates are character-local wall-clock times, not UTC and
not server-local time.

## Runtime Scheduling And Priority Contract

FastAPI lifespan owns the production worker:

```text
service.lifespan
  -> bootstrap DB, cache, graph, dispatcher, scheduler
  -> start chat input queue worker
  -> if not REFLECTION_CYCLE_DISABLED: asyncio.create_task(reflection worker loop)
  -> yield
  -> stop reflection worker
  -> stop chat input queue worker, scheduler, MCP, DB
```

The approved async pattern is one `asyncio.create_task(...)` created from
FastAPI lifespan and owned by `ReflectionWorkerHandle`. The handle must contain
the task and a cancellation/stop signal. Shutdown must signal stop, cancel only
after a bounded wait, and suppress `asyncio.CancelledError`.

Default schedule:

| Config | Default | Meaning |
|---|---:|---|
| `REFLECTION_CYCLE_DISABLED` | `false` | Reflection worker runs by default. Set to `true` to explicitly disable it. |
| `REFLECTION_WORKER_INTERVAL_SECONDS` | `900` | One scheduling tick every 15 minutes. |
| `REFLECTION_HOURLY_SLOTS_PER_TICK` | `3` | Maximum hourly LLM calls per tick. |
| `REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME` | `04:30` | Earliest character-local time to synthesize the previous local day. |
| `REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME` | `05:00` | Earliest character-local time to run previous-day global promotion. |

Tick order:

1. If `is_primary_interaction_busy()` is true, skip the tick before selecting
   work and do not start an LLM call.
2. Run due hourly slots first, capped by `REFLECTION_HOURLY_SLOTS_PER_TICK`.
3. Run due per-channel daily syntheses for the previous character-local day
   only after available hourly slots for that channel/day have terminal run
   documents. Terminal statuses are `succeeded`, `failed`, `skipped`, and
   `dry_run`; a failed hourly slot must not block daily synthesis forever. If
   non-terminal hourly documents are present, use the terminal subset and
   record a `partial_hourly_input` validation warning on the daily run
   document.
4. Run global promotion for the previous character-local day only after all
   available `daily_channel` docs for that day are terminal.
5. Check `is_primary_interaction_busy()` again before every LLM call and before
   every memory write. If it becomes true during an in-flight LLM call, let the
   current LLM call complete, record the current unit of work if needed, and do
   not start the next LLM call or memory write in that tick.

Primary interaction priority order:

1. `/chat` queue processing and graph execution.
2. Assistant message persistence and conversation progress recording.
3. Post-dialog consolidation, including user memory and character state writes.
4. Existing scheduled-event dispatch.
5. Reflection hourly, daily, promotion, and prompt-facing context refresh.

Reflection must never block a chat response, delay consolidation of the
current turn, or create/send scheduled messages.

## Logging Contract

Stage 1c logging must make promoted memory visible without turning normal
operation logs into raw evidence dumps.

`INFO` logs are operator-critical and must include:

- worker startup, shutdown, and explicit disable state;
- one summary per hourly, daily-channel, and global-promotion worker pass with
  character-local date, run counts, succeeded/failed/skipped counts, and defer
  reason when present;
- every actual memory mutation from global promotion, including lane, action
  (`insert`, `supersede`, or `merge`), `memory_unit_id`, `lineage_id`,
  `memory_type`, sanitized memory name, sanitized content preview capped at
  160 characters, source reflection run count, and run id;
- every global-promotion no-write outcome that needs operator attention:
  privacy rejection, boundary rejection, score-unavailable skip, memory lock
  deferral, lane disabled by flag, dry-run mode, or caller-disabled memory
  writes.

`DEBUG` logs may include supporting evidence only:

- prompt character counts and cap/omission diagnostics;
- daily synthesis confidence, validation warning labels, and selected channel
  counts;
- promotion candidate confidence, boundary/privacy review fields, top
  similarity scores, merge/supersede source ids, and evidence-card ids;
- source run ids and scope refs.

Forbidden at every log level:

- raw transcript text;
- raw hourly reflection output;
- full `character_reflection_runs` documents;
- raw `source_message_refs`;
- `platform_user_id`, `global_user_id`, display names, or private user facts;
- unsanitized evidence text or attachment descriptions.

Tests must capture logs and prove:

- a successful lore mutation emits one `INFO` log that names the promoted lore
  via sanitized name/content preview;
- confidence, scores, evidence-card ids, boundary review, and validation
  warnings are present only at `DEBUG`;
- forbidden raw/user-identifying fields are absent from both `INFO` and
  `DEBUG` logs.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/db/reflection_cycle.py`
- `src/kazusa_ai_chatbot/reflection_cycle/repository.py`
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
- `src/kazusa_ai_chatbot/reflection_cycle/context.py`
- `src/scripts/run_reflection_cycle.py`
- `tests/test_reflection_cycle_stage1c_repository.py`
- `tests/test_reflection_cycle_stage1c_worker.py`
- `tests/test_reflection_cycle_stage1c_promotion.py`
- `tests/test_reflection_cycle_stage1c_promotion_live_llm.py`
- `tests/test_reflection_cycle_stage1c_reflection_context.py`
- `tests/test_reflection_cycle_stage1c_service.py`
- `tests/test_reflection_cycle_stage1c_integration.py`

### Modify

- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/db/bootstrap.py`
- `src/kazusa_ai_chatbot/db/__init__.py`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/reflection_cycle/__init__.py`
- `src/kazusa_ai_chatbot/reflection_cycle/models.py`
- `src/kazusa_ai_chatbot/reflection_cycle/runtime.py`
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- Minimal cognition planning prompt module that consumes promoted reflection context.

### Keep

- Stage 1b memory schema and APIs except normal imports.
- Dispatcher send-message path.
- `/chat` request handling.
- User-memory consolidator writers.

## Implementation Order

1. Verify Stage 1a approval artifact and Stage 1b completion evidence; record exact paths and commands before editing code.
2. Update `reflection_cycle/README.md` with the Stage 1c production ICD sections for public interfaces, DB interface, worker schedule, memory integration, and prompt-facing context.
3. Add failing deterministic tests for the Stage 1c DB boundary, reflection-run repository idempotency, promotion validation, similarity thresholds, lock deferral, and worker idle behavior.
4. Add failing FastAPI/service tests for worker lifespan start/stop, default-on startup, explicit-disable behavior, and primary-interaction busy deferral.
5. Add `character_reflection_runs` schema/indexes and `db.reflection_cycle`; repository code must call only this DB interface.
6. Implement public facades and production runtime using Stage 1a approved monitored-channel selector, hour-slot projection, per-channel daily projection, and core prompts. Keep the 168-hour fallback disabled in production mode.
7. Add per-slot worker error isolation with bounded external-LLM retry and failed-run persistence.
8. Add global promotion prompt, promotion validators, deterministic similarity resolver, and dry-run artifact output. Do not enable memory writes yet.
9. Run the global promotion prompt real LLM mini-gate one case at a time and record inspection evidence.
10. Add two-lane promotion integration through Stage 1b memory APIs only.
11. Add FastAPI lifespan worker wiring and feature flags after module and worker tests pass.
12. Add prompt-facing promoted reflection context behind `REFLECTION_CONTEXT_ENABLED`; raw hourly summaries must not be retrievable by live cognition.
13. Run focused deterministic tests, static boundary greps, and compile checks.
14. Run dry-run against monitored-channel data.
15. Enable writes only after dry-run evidence and promotion prompt approval are reviewed and recorded.

## Progress Checklist

- [x] Stage 1 — handoff evidence and ICD updated.
  - Covers: Stage 1a/1b evidence review and `reflection_cycle/README.md`.
  - Verify: exact evidence paths are recorded; README diff has Stage 1c public interfaces and boundaries.
  - Evidence: record paths, commands, and README review result in `Execution Evidence`.
  - Handoff: complete.
  - Sign-off: Codex / 2026-05-05 after execution evidence was recorded.
- [x] Stage 2 — module and DB-interface tests added.
  - Covers: repository, DB boundary, promotion validation, worker idle logic.
  - Verify: tests fail for missing Stage 1c implementation or pass only for unchanged Stage 1a contracts.
  - Evidence: record test names and baseline result.
  - Handoff: complete.
  - Sign-off: Codex / 2026-05-05 after focused Stage 1c tests passed.
- [x] Stage 3 — reflection-run DB interface and repository implemented.
  - Covers: `db.reflection_cycle`, schema/indexes, run idempotency.
  - Verify: repository tests and boundary grep pass.
  - Evidence: record changed files and test output.
  - Handoff: complete.
  - Sign-off: Codex / 2026-05-05 after repository and boundary tests passed.
- [x] Stage 4 — production runtime and worker implemented.
  - Covers: hourly/daily persistence, fallback disabled, failure isolation, retry cap, busy deferral.
  - Verify: worker tests pass.
  - Evidence: record test output and skip/defer behavior.
  - Handoff: complete.
  - Sign-off: Codex / 2026-05-05 after worker tests and CLI dry-run passed.
- [x] Stage 5 — global promotion dry-run and live LLM mini-gate approved.
  - Covers: promotion prompt, validators, privacy/boundary rejection, deterministic similarity resolver.
  - Verify: deterministic promotion tests pass; live LLM tests run one by one and meet every mini-gate pass criterion in `Verification`.
  - Evidence: record trace paths and human/agent judgment notes.
  - Handoff: complete.
  - Sign-off: Codex / 2026-05-05 after one-by-one live LLM traces were inspected and accepted.
- [x] Stage 6 — memory writes integrated through Stage 1b APIs.
  - Covers: insert/supersede/merge, lock deferral, Cache2 invalidation.
  - Verify: tests prove no legacy memory path and no direct memory DB calls.
  - Evidence: record memory API call assertions and grep output.
  - Handoff: complete.
  - Sign-off: Codex / 2026-05-05 after memory API and legacy-path boundary tests passed.
- [x] Stage 7 — FastAPI scheduling and flags wired.
  - Covers: `service.py`, config flags, lifespan start/stop, default-on worker startup, explicit disable flag, busy-probe callback.
  - Verify: `tests/test_reflection_cycle_stage1c_service.py` passes.
  - Evidence: record service test output.
  - Handoff: complete.
  - Sign-off: Codex / 2026-05-05 after FastAPI scheduling tests passed.
- [x] Stage 8 — prompt-facing promoted context implemented.
  - Covers: bounded promoted lore/self-guidance context behind flag.
  - Verify: reflection-context tests and raw-hourly exclusion greps pass.
  - Evidence: record test and grep output.
  - Handoff: complete.
  - Sign-off: Codex / 2026-05-05 after reflection-context and raw-hourly exclusion checks passed.
- [x] Stage 9 — final deterministic verification and dry-runs complete.
  - Covers: focused tests, static greps, monitored-channel dry-run, MongoDB explain.
  - Verify: every command in `Verification` passes or has an allowed documented exception.
  - Evidence: record command outputs and dry-run summary.
  - Handoff: implementation complete.
  - Sign-off: Codex / 2026-05-05 after deterministic tests, live LLM tests, dry-runs, static greps, py_compile, import smoke, and MongoDB explain passed.

## Verification

```powershell
pytest tests\test_reflection_cycle_stage1c_repository.py -q
pytest tests\test_reflection_cycle_stage1c_worker.py -q
pytest tests\test_reflection_cycle_stage1c_promotion.py -q
pytest tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_skips_memory_write_when_scores_are_unavailable -q
pytest tests\test_reflection_cycle_stage1c_promotion.py::test_promotion_logs_info_for_memory_mutation_and_debug_for_evidence -q
pytest tests\test_reflection_cycle_stage1c_reflection_context.py -q
pytest tests\test_reflection_cycle_stage1c_service.py -q
pytest tests\test_reflection_cycle_stage1c_integration.py -q
```

Run real LLM promotion-prompt tests one by one and inspect each artifact before
continuing:

```powershell
pytest tests\test_reflection_cycle_stage1c_promotion_live_llm.py::test_global_promotion_live_normal_case -q -s
pytest tests\test_reflection_cycle_stage1c_promotion_live_llm.py::test_global_promotion_live_privacy_rejection_case -q -s
pytest tests\test_reflection_cycle_stage1c_promotion_live_llm.py::test_global_promotion_live_no_signal_case -q -s
```

Mini-gate pass criteria:

- All three live tests must write durable trace artifacts containing rendered
  prompt, input payload, raw output, parsed output, validation warnings, and
  inspector notes.
- All parsed outputs must be strict JSON with top-level
  `promotion_decisions`.
- All generated free-text fields must follow the language policy: Simplified
  Chinese for new free text, original language preserved only inside evidence
  snippets, English for keys and enum values.
- No parsed decision may contain `source_global_user_id`, MongoDB query fields,
  embedding fields, raw transcript text, user identifiers, user commitments,
  health details, or private relationship facts.
- Every non-reject lore decision must have `lane="lore"`,
  `signal_strength="high"`, `memory_type="fact"`,
  `privacy_review.user_details_removed=true`,
  `privacy_review.private_detail_risk in {"low", "medium"}`,
  `boundary_assessment.verdict="acceptable"`, and at least one evidence ref
  tied to an `evidence_card` with non-empty `active_character_utterance`.
- Every non-reject self-guidance decision must have
  `lane="self_guidance"`, `signal_strength="high"`,
  `memory_type="defense_rule"`, behavior-only content addressed to the active
  character's future responses, privacy review with user details removed, and
  acceptable boundary assessment when identity, intimacy, attachment, safety,
  or boundary-setting is implicated.
- `test_global_promotion_live_normal_case` must produce exactly one lore
  promote candidate and exactly one self-guidance promote candidate from the
  fixture; both must satisfy the lane criteria above.
- `test_global_promotion_live_privacy_rejection_case` must produce no promote
  candidate. If it emits a row, every row must have `decision="reject"` and
  `privacy_review.private_detail_risk="high"`.
- `test_global_promotion_live_no_signal_case` must produce no promote
  candidate; allowed decisions are only `no_action` or `reject`.
- If any live case passes pytest assertions but violates these bars during
  artifact inspection, the mini-gate fails and memory writes remain disabled.

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

Run source greps proving Stage 1c respects DB boundaries:

```powershell
rg "get_db|\\.find\\(|\\.aggregate\\(|insert_one\\(|update_one\\(|update_many\\(|delete_one\\(|delete_many\\(|replace_one\\(|count_documents\\(" src\kazusa_ai_chatbot\reflection_cycle src\scripts\run_reflection_cycle.py
```

Allowed result: no matches. MongoDB operations must appear only in
`src\kazusa_ai_chatbot\db\conversation_reflection.py` and
`src\kazusa_ai_chatbot\db\reflection_cycle.py`.

Run service scheduling checks:

```powershell
pytest tests\test_reflection_cycle_stage1c_service.py::test_lifespan_starts_reflection_worker_by_default -q
pytest tests\test_reflection_cycle_stage1c_service.py::test_lifespan_does_not_start_reflection_worker_when_explicitly_disabled -q
pytest tests\test_reflection_cycle_stage1c_service.py::test_lifespan_stops_reflection_worker_on_shutdown -q
pytest tests\test_reflection_cycle_stage1c_service.py::test_reflection_worker_defers_while_primary_interaction_is_busy -q
```

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
- `reflection_cycle/README.md` documents the Stage 1c production interfaces, DB boundaries, worker schedule, feature flags, and memory integration rules before service wiring lands.
- Monitored-channel selection uses latest character message time and does not rely on counters or a dedicated monitoring collection.
- Production monitoring disables the read-only fallback window and idles when no monitor-active channel exists.
- Hourly production reflection writes only `character_reflection_runs`.
- Hourly production reflection evaluates every message-bearing hour in monitored channels and skips only empty hours.
- Per-channel daily production reflection writes `daily_channel` run documents and does not write memory.
- Global promotion writes `daily_global_promotion` run documents and is the only reflection step allowed to produce promotion decisions.
- The global promotion prompt has recorded one-by-one real LLM approval evidence before memory writes are enabled.
- Hourly, daily-channel, and global-promotion production run documents include
  prompt version and attempt metadata.
- Run documents persist raw scope triples in addition to hashed scope refs.
- Evidence refs and source message refs are repository/input-derived, not LLM-derived.
- Source message refs use composite message identity fields unless a future DB interface explicitly provides `conversation_history_id`.
- Promotion duplicate detection uses Stage 1b similarity scores before choosing `promote_new`, `supersede`, or `merge`.
- Promotion writes are skipped/deferred with no memory API mutation when Stage 1b semantic search scores are unavailable or malformed.
- Memory write lock contention records a skipped/deferred promotion and does not fail the whole worker pass.
- Stage 1c writers construct `EvolvingMemoryDoc` directly and do not call `build_memory_doc` or `save_memory`.
- `decision="reject"` never mutates an existing active memory unit.
- One failed hourly slot, daily channel, or global promotion run records a failed/skipped run document and does not stop unrelated work in the same worker pass.
- Global promotion writes at most one lore mutation and at most one self-guidance mutation through Stage 1b APIs per character-local day.
- Private/user-specific details are rejected before all global memory writes.
- Lore promotion requires character agreement and boundary assessment.
- Self-guidance promotion requires behavior-only content and boundary assessment when it can affect character identity, intimacy, attachment, safety, or setting behavior.
- Reflection worker runs by default, can be disabled only by
  `REFLECTION_CYCLE_DISABLED=true`, and defers while chat is busy.
- FastAPI lifespan starts the reflection worker by default, skips it when
  explicitly disabled, stops it on shutdown, and passes a busy-probe callback
  instead of exposing service internals.
- Reflection worker scheduling is lower priority than chat, post-dialog persistence, consolidation, and scheduled-event dispatch.
- `INFO` logs show operator-critical lifecycle and promotion outcomes,
  including what lore or self-guidance was promoted; `DEBUG` logs contain
  supporting confidence, score, boundary/privacy, validation, and evidence-card
  details only.
- Logs at every level exclude raw transcripts, raw hourly outputs, user
  identifiers, and unsanitized private details.
- `character_reflection_runs` uses `_id=run_id`, has the indexes listed in this plan, and upserts are idempotent by `run_id`.
- `character_local_date` and daily/promotion run gates use `CHARACTER_TIME_ZONE`, while Stage 1a hourly slots remain UTC.
- Prompt-facing reflection context is disabled by default and bounded when enabled.
- Prompt-facing reflection context never includes raw hourly summaries or raw `hourly_slot` run outputs.
- No autonomous messages are sent.

## Validation Against Current Implementation

Stage 1c is implemented and verified in the current workspace. The live
database has the monitored-channel production index and reflection-run indexes
ensured through the approved DB/bootstrap interfaces.

| Entry Gate Or Component | Expected by plan | Current implementation | Status |
|---|---|---|---|
| Stage 1a approval | Required before coding | Stage 1a plan is completed and signed off with private/group/daily live LLM traces plus a monitored-channel artifact accepted on 2026-05-04 | Ready |
| Stage 1b completion | Required before coding | `memory_evolution` package, reset CLI, active-only retrieval, scored semantic search, ICD, and completion evidence exist in the current workspace | Ready |
| `character_reflection_runs` repository | Required | `reflection_cycle/repository.py` builds idempotent run documents and calls only `db.reflection_cycle` | Complete |
| Stage 1c DB interface | Required | `db/reflection_cycle.py` owns reflection-run persistence and indexes | Complete |
| Production worker | Required | `reflection_cycle/worker.py` implements hourly, daily, and promotion scheduling with busy deferral | Complete |
| Two-lane promotion | Required | `reflection_cycle/promotion.py` implements prompt, validators, repository-derived evidence refs, score-based duplicate handling, and Stage 1b API writes | Complete |
| Global promotion prompt approval | Required | One-by-one live LLM mini-gate passed for normal, privacy rejection, and no-signal cases | Complete |
| Per-slot error isolation | Required | Worker tests cover no-fallback idle behavior, dry-run persistence, failed-hour terminal gating, and primary-interaction deferral | Complete |
| Prompt-facing promoted context | Required | `reflection_cycle/context.py` exposes bounded promoted lore/self-guidance only behind `REFLECTION_CONTEXT_ENABLED` | Complete |
| Production CLI | Required | `src/scripts/run_reflection_cycle.py` supports `hourly`, `daily`, and `promote` dry-run/write modes | Complete |
| Feature flags | Required | Reflection default-on disable flag, context flag, lane flags, and schedule timing values are implemented in `config.py` | Complete |
| FastAPI scheduling tests | Required | `tests/test_reflection_cycle_stage1c_service.py` covers lifespan start, explicit disable, shutdown stop, and busy deferral | Complete |
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

Stage 1c implementation is complete. Future changes to reflection promotion,
memory schema, or autonomous behavior require a new or superseding plan.

## Plan Sign-Off

- Approved for implementation: 2026-05-05
- Implementation completed: 2026-05-05
- Sign-off scope: Stage 1c implementation, deterministic tests, live LLM
  promotion mini-gate, production dry-runs, DB/index verification, and
  interface-control documentation.
- Sign-off basis: this plan now pins the Stage 1a/1b handoff artifacts,
  requires Stage 1c ICD updates before implementation, defines public module
  facades, isolates MongoDB access behind DB-interface modules, specifies the
  global promotion prompt skeleton and mini-gate pass criteria, defines typed
  promotion payload shapes, pins prompt version and character-local date
  handling, specifies deterministic promotion thresholds and idempotent ids,
  defines FastAPI lifespan scheduling and primary-interaction priority,
  requires default-on scheduling, explicit disable, logging-level separation,
  score-availability, and service scheduling tests, and preserves the Stage 1b
  memory API boundary.
- Implementation sign-off basis: every Stage 1c verification command completed
  successfully with the explicit `live_llm` marker for real LLM tests; live
  traces were inspected one by one; static greps found no legacy writer or raw
  DB operations in Stage 1c production code; MongoDB explain used
  `conv_role_ts_platform_channel`; dry-run CLI commands completed; and memory
  writes remain routed through Stage 1b public APIs with repository-derived
  evidence refs.

## Execution Evidence

- Stage 1a approval artifact: `development_plans/character_reflection_cycle_stage1a_plan.md`, status `completed`; private, group, and daily live LLM traces are recorded in that plan's `Execution Evidence`, with Stage 1a signed off on 2026-05-04.
- Stage 1b completion evidence: `development_plans/memory_evolution_stage1b_plan.md`, status `completed`; focused Stage 1b command in that plan passed with `56 passed` after scoring fixes. During Stage 1c final verification, `venv\Scripts\pytest.exe tests\test_memory_evolution_module_boundary.py tests\test_memory_evolution_repository.py -q` passed with `8 passed`.
- Focused test results:
  - `venv\Scripts\pytest.exe tests\test_reflection_cycle_stage1c_repository.py tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_promotion.py tests\test_reflection_cycle_stage1c_reflection_context.py tests\test_reflection_cycle_stage1c_service.py tests\test_reflection_cycle_stage1c_integration.py -q` passed with `25 passed`.
  - `venv\Scripts\pytest.exe tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_skips_memory_write_when_scores_are_unavailable -q` passed.
  - `venv\Scripts\pytest.exe tests\test_reflection_cycle_stage1c_promotion.py::test_promotion_logs_info_for_memory_mutation_and_debug_for_evidence -q` passed.
  - Service scheduling checks named in `Verification` passed one by one: default startup, explicit disable, shutdown stop, and busy deferral.
  - Full default non-live suite `venv\Scripts\pytest.exe -q` passed with `661 passed, 179 deselected`.
  - `venv\Scripts\python.exe -m py_compile` passed for every changed Stage 1c runtime, DB, service, cognition, CLI, and test file.
  - Import smoke passed for `run_hourly_reflection_cycle`, `run_daily_channel_reflection_cycle`, `run_global_reflection_promotion`, `build_promoted_reflection_context`, and lazy legacy `db.save_memory` / `db.search_memory` exports.
- Static boundary evidence:
  - `rg "build_memory_doc|save_memory" src\kazusa_ai_chatbot\reflection_cycle src\scripts\run_reflection_cycle.py` returned no matches.
  - `rg "get_db|\.find\(|\.aggregate\(|insert_one\(|update_one\(|update_many\(|delete_one\(|delete_many\(|replace_one\(|count_documents\(" src\kazusa_ai_chatbot\reflection_cycle src\scripts\run_reflection_cycle.py` returned no matches.
  - `tests\test_reflection_cycle_stage1c_promotion.py::test_promotion_uses_repository_evidence_refs_not_llm_refs` passed, proving memory writes replace LLM evidence refs with repository-derived evidence refs.
- Monitored-channel dry-run summary:
  - `venv\Scripts\python.exe src\scripts\run_reflection_cycle.py hourly --dry-run` returned `processed=3`, `skipped=3`, `failed=0`, `deferred=False`, with three hourly dry-run run ids.
  - `venv\Scripts\python.exe src\scripts\run_reflection_cycle.py daily --dry-run` returned `processed=1`, `failed=0`, `deferred=False`, with one daily-channel dry-run run id.
- MongoDB explain summary:
  - Ensured `conv_role_ts_platform_channel` and reflection-run indexes in the live database.
  - `conversation_history` indexes include `conv_role_ts_platform_channel: {"role":1,"timestamp":-1,"platform":1,"platform_channel_id":1}`.
  - MongoDB explain for the monitored-channel aggregate used `IXSCAN` on `conv_role_ts_platform_channel`, `nReturned=3`, `totalKeysExamined=20`, `totalDocsExamined=20`, and `executionTimeMillis=4`.
  - Semantic `find_active_memory_units` dry-run returned score tuples: `rows=3`, `score_tuple_shape=True`, `scores=[0.6684, 0.6546, 0.6526]`.
- Global promotion prompt real LLM approval:
  - `venv\Scripts\pytest.exe -m live_llm tests\test_reflection_cycle_stage1c_promotion_live_llm.py::test_global_promotion_live_normal_case -q -s` passed and was inspected. Trace: `test_artifacts/llm_traces/reflection_cycle_stage1c_promotion_live_llm__normal_case__20260505T013648505032Z.json`. Output produced exactly one `lore` promote candidate and one `self_guidance` promote candidate; both used Simplified Chinese generated text, acceptable boundary review, low privacy risk, and evidence refs from `hourly-run-1`.
  - `venv\Scripts\pytest.exe -m live_llm tests\test_reflection_cycle_stage1c_promotion_live_llm.py::test_global_promotion_live_privacy_rejection_case -q -s` passed and was inspected. Trace: `test_artifacts/llm_traces/reflection_cycle_stage1c_promotion_live_llm__privacy_rejection_case__20260505T013757882160Z.json`. Output rejected both lanes with `private_detail_risk="high"` and no promote candidates.
  - `venv\Scripts\pytest.exe -m live_llm tests\test_reflection_cycle_stage1c_promotion_live_llm.py::test_global_promotion_live_no_signal_case -q -s` passed and was inspected. Trace: `test_artifacts/llm_traces/reflection_cycle_stage1c_promotion_live_llm__no_signal_case.json`. Output contained no promote candidates.
- Promotion dry-run summary:
  - `venv\Scripts\python.exe src\scripts\run_reflection_cycle.py promote --dry-run` returned `processed=1`, `skipped=1`, `failed=0`, `deferred=False`, `defer reason: memory writes disabled`, with run id `reflection_run_aa3403dffe50dce00ec4f96e80c95fa9`.
- Prompt-facing context filtering evidence:
  - `tests\test_reflection_cycle_stage1c_reflection_context.py` passed, proving disabled context returns empty without querying memory and enabled context projects only reflection-promoted `fact` and `defense_rule` lanes.
  - The raw-hourly/DB-boundary grep in `tests\test_reflection_cycle_stage1c_integration.py` passed.
- Post-completion audit evidence on 2026-05-05:
  - Code-guideline and plan-alignment audit fixed service imports so service depends only on package-level reflection facades, added worker-to-promotion busy-probe propagation before memory writes, and removed plan-label wording from Python docstrings/test strings.
  - `venv\Scripts\pytest.exe tests\test_reflection_cycle_stage1c_repository.py tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_promotion.py tests\test_reflection_cycle_stage1c_reflection_context.py tests\test_reflection_cycle_stage1c_service.py tests\test_reflection_cycle_stage1c_integration.py -q` passed with `28 passed`.
  - `venv\Scripts\pytest.exe -q` passed with `664 passed, 179 deselected`.
  - `venv\Scripts\pytest.exe tests\test_memory_evolution_module_boundary.py tests\test_memory_evolution_repository.py -q` passed with `8 passed`.
  - `venv\Scripts\python.exe -m py_compile` passed for the audited reflection, service, CLI, and Stage 1c test files; `git diff --check` passed with line-ending warnings only; static greps again found no legacy writer, no raw DB operations in reflection/CLI, and no service imports of reflection internals.
- Follow-up quality audit evidence on 2026-05-05:
  - Fixed hourly parent scope parsing to strip only the terminal hourly timestamp suffix, replaced retry prompt construction through skipped-result wrapping with direct prompt construction, removed keyword-based privacy-risk classification from evidence cards, documented process-loaded lane flags and global synthetic scope in the ICD, and clarified the mini-gate as a procedural release gate rather than a runtime marker.
  - Added deterministic tests for timestamp-suffix scope parsing, direct hourly retry prompt construction, negated privacy-note handling, and partial hourly input warnings on daily run documents.
  - `venv\Scripts\pytest.exe tests\test_reflection_cycle_stage1c_repository.py tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_promotion.py tests\test_reflection_cycle_stage1c_reflection_context.py tests\test_reflection_cycle_stage1c_service.py tests\test_reflection_cycle_stage1c_integration.py -q` passed with `32 passed`.
  - `venv\Scripts\pytest.exe tests\test_memory_evolution_module_boundary.py tests\test_memory_evolution_repository.py -q` passed with `8 passed`.
  - `venv\Scripts\pytest.exe -q` passed with `668 passed, 179 deselected`.
  - `venv\Scripts\python.exe -m py_compile` passed for the audited reflection and Stage 1c test files; static greps found no legacy writer, no raw DB operations in reflection/CLI, no service imports of reflection internals, and no Stage 1c plan/ICD release-marker references.
- Any deviations:
  - The plan's listed live LLM commands were run with `-m live_llm` because `pytest.ini` deselects `live_llm` tests by default.
  - The initial live normal case exposed that the prompt needed a stricter evidence-ref rule; the prompt and tests were tightened, runtime writes now replace LLM evidence refs with repository-derived refs, and the live mini-gate was rerun successfully.
  - The broad non-live suite exposed stale test fixtures missing the existing required `time_context` state field and one monkeypatch signature drift; those tests were updated to the current state contract, with no production behavior change.
  - No scope deviations: no autonomous messages, `/chat` calls, raw hourly prompt-facing context, legacy memory writer use, or direct DB operations inside `kazusa_ai_chatbot.reflection_cycle` were added.
