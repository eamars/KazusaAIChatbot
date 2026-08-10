# cognition core v2 context fade and sleep phase plan

## Summary

- Goal: make group-scene and conversation-progress context fade by elapsed time
  through deterministic discard before projection, and make the character sleep
  phase and morning refresh first-class cognition v2 concerns.
- Status: completed
- Scope boundary: `cognition_core_v2` owns the sleep-phase descriptor and the
  morning-refresh transition; `conversation_progress` owns age-based discard of
  its own turns, events, and narrative; `time_boundary` owns shared local-clock
  math; `config` owns the new thresholds.
- Change direction: replace "project everything and let the model judge age"
  with "discard aged content deterministically, then project"; replace deep
  reducer reach-in and duplicated clock parsers with owned entrypoints.
- Acceptance state: no aged group-scene turn, progress event, or stale
  narrative reaches any prompt; cognition receives a validated sleep phase;
  morning refresh runs through a `cognition_core_v2` entrypoint that validates
  its own output.

## Scope And Change Direction

Four defects drive this plan.

1. Group-scene turns are selected by count and characters only. `occurred_at`
   is parsed for sort order and then discarded, and the rendered scene contains
   no timestamps. A channel idle for a week renders week-old turns as the
   current scene.
2. Conversation-progress content never ages inside its rolling 48-hour packet
   TTL. Events are ordered by retention tier with recency as a tie-break only,
   and the narrative fields have no age gate.
3. `project_conversation_progress_evidence` stamps every progress event's
   `evidence_ref.occurred_at` with the current episode timestamp instead of the
   event's own time, and `scene_context.semantic_temporal_context` is
   hardcoded `"immediate"`. Aged content is therefore presented as current.
4. The sleep window is invisible to `cognition_core_v2`, and the morning
   refresh reaches into `state_reducers.apply_sleep_recovery` from
   `reflection_cycle` without any output validation.

End state: discard is mechanical and happens before projection; cognition never
sees aged conversational context; the sleep phase is a validated part of the
cognition input; morning refresh is a `cognition_core_v2` public entrypoint.

Excluded: the persisted group style image, response suppression during sleep,
and any LLM call added for summarization or discount.

## Confirmed Decisions

These are settled by the plan owner and are not open for reinterpretation.

| Topic | Decision |
|---|---|
| Fade mechanism | Deterministic mechanical discard. Do not present aged data to a cognition stage and let the model discount it. Do not add freshness labels as a decision mechanism. |
| Discard cost | No LLM call. No scheduled summarization job. Lazy discard on the read path only. |
| Narrative fields | When stale, clear all of them as one set rather than serving a stale summary or re-summarizing. |
| Thresholds | Tiered and env-overridable, with the values in `Contracts And Data Shapes`. |
| `decision_critical` age | 48 hours, equal to the existing rolling `EPISODE_TTL`, so that tier's effective behavior is unchanged and only lower tiers begin expiring early. |
| Group style image | Persistent by design. It is not intended to expire and is flushed only by a new image. Out of scope. |
| Sleep enforcement scope | Unchanged. The sleep window continues to gate only the two self-cognition trigger lanes. The reply path is never blocked. |
| Sleep phase integration | The phase becomes a validated field of the cognition v2 input and reaches goal cognition. |
| Morning refresh ownership | `cognition_core_v2` owns the transition semantics through a public entrypoint. `reflection_cycle` keeps scheduling, idempotency, persistence, and the audit row. |
| Pre-wake offset | `AFFECT_SETTLING_WAKE_PREP_MINUTES` default stays `30`. Operators set another value by environment; no code change. |

## Mandatory Skills

- `development-plan`: before editing this plan or executing any work item.
- `local-llm-architecture`: before changing what reaches a prompt, including
  the sleep-phase descriptor and every discard that changes projected content.
- `py-style`: before editing any Python file.
- `cjk-safety`: before editing any Python file that contains Chinese text,
  which includes the sleep-phase label vocabulary.
- `test-style-and-execution`: before adding, changing, or running tests.
- `python-venv`: before running Python.

## Mandatory Rules

Project rules specific to this change surface:

- Discard decisions belong to deterministic code. No prompt receives aged
  content together with an instruction to discount it.
- Discarded content leaves no trace a model could reason about. Age-discarded
  group-scene turns are excluded from `omitted_turn_count`, not reported as
  omitted.
- Deterministic code may drop a whole event or clear a whole narrative field
  set. It must never rewrite, paraphrase, summarize, or synthesize progress
  content.
- The read-path prune is presentation and hand-off state only. It must not
  attempt its own database write; the guarded replace in
  `db/conversation_progress.py` requires `turn_count` to strictly increase, so
  the pruned form persists only when the next recorded turn writes it.
- Every new threshold is a named env-backed constant. No bare numeric literal
  for an age, cap, or floor.
- The sleep phase label vocabulary is frozen Chinese text following the
  existing `project_duration` and `project_numeric_band` style: keyword-only
  arguments, no config reads inside the projector, `ValueError` on
  out-of-domain input.
- Do not change `compute_affect_settling_due_local_time`,
  `_sleep_period_bounds`, `validate_affect_settling_timing`, or the
  import-time timing validation in `reflection_cycle/affect_settling.py`.
- Preserve the public name, signature, and observable behavior of
  `is_self_cognition_sleep_period`.
- Preserve unrelated worktree changes. `docs/CODING_AGENT_CAPABILITY_ASSESSMENT.md`
  and `src/scripts/count_code.py` are untracked and must remain untouched.

## Must Do

- Discard ambient group-scene turns older than the configured age inside
  `build_group_scene_context`, before the count and character fitting, and
  exclude them from the ambient total that feeds `omitted_turn_count`.
- Discard conversation-progress events by retention tier age on the read path,
  inside `ConversationProgressRuntime.load`, immediately after packet
  selection, so the pruned packet feeds both the prompt projection and any
  later `prior_episode_state`.
- Clear the narrative field set when the newest surviving event is older than
  the narrative threshold, reusing the existing canonical empty shape.
- Pass each progress event's own timestamp as its `evidence_ref.occurred_at`
  instead of the episode timestamp.
- Derive `scene_context.semantic_temporal_context` instead of hardcoding
  `"immediate"`.
- Add shared local-clock helpers to `time_boundary` and remove the duplicate
  `HH:MM` parsers from `self_cognition/sleep_period.py`.
- Add a `cognition_core_v2` sleep-phase projector and carry its result as a
  new validated optional `scene_context` field.
- Add a `cognition_core_v2` public morning-refresh entrypoint that owns the
  scope guard, the reducer call, and output validation, and route
  `reflection_cycle/affect_settling.py` through it.
- Remove the two parameters `apply_sleep_recovery` accepts and ignores.
- Add the four threshold constants and one narrative threshold to `config`,
  surfaced through `conversation_progress/policy.py`.
- Update `docs/HOWTO.md` and `src/kazusa_ai_chatbot/cognition_core_v2/README.md`.
- Update every test listed in `Change Surface`.

## Deferred

- Do not add staleness, TTL, cache expiry, or a freshness gate to the group
  style image, `group_engagement_action_context`, or `engagement_guidelines`.
- Do not block, delay, or suppress a reply based on the sleep phase.
- Do not add the sleep phase to the appraisal prompts, the surface prompts,
  `CharacterOperationalContextV1`, or `TextSurfaceInputV2`.
- Do not add an age cutoff to `get_ambient_conversation_history`. The database
  query stays count-bounded; discard happens in projection.
- Do not add a calendar job, background worker, or scheduled summarization.
- Do not change `EPISODE_TTL`, the Mongo TTL indexes, the compaction limits,
  `_event_tier`, or the tier ordering.
- Do not change `past_dialog_cognition_context`, which has its own structural
  triggers and no age gate. Its aging remains indirect through trace TTL.
- Do not extend the morning refresh to user scope, and do not change
  `elapsed_sleep_seconds` from the configured window length to real elapsed
  time. Both remain as they are; they are recorded as known asymmetries in
  `Execution Evidence` for a future decision.
- Do not make `apply_sleep_recovery` re-derive affect activations or prune
  terminal entities. Its current dampening semantics are preserved exactly.
- Do not deduplicate the third `HH:MM` parser in
  `reflection_cycle/affect_settling.py`. Leaving it avoids destabilizing the
  schedule math and its import-time validator; the residual duplication is
  accepted.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Group-scene turn selection | bigbang | Age filter applies unconditionally. No legacy unfiltered path. |
| Progress event and narrative discard | bigbang | Prune applies on every load. No opt-out and no unpruned read path. |
| Progress evidence `occurred_at` | bigbang | Event timestamp replaces the episode timestamp directly. |
| `scene_context.semantic_temporal_context` | bigbang | Derived value replaces the hardcoded literal. |
| `scene_context` sleep phase field | compatible | Added as an optional key following the existing `current_user_role` precedent, so inputs without it stay valid. |
| Persisted packets | compatible | No migration and no backfill. Stored packets are pruned lazily on read and persisted by the next recorded turn. |
| Clock helper relocation | bigbang | `self_cognition/sleep_period.py` loses its private parsers and uses the shared helpers. |
| Morning refresh entrypoint | bigbang | `reflection_cycle` stops importing the reducer directly and calls the v2 entrypoint. |
| Tests | bigbang | Replace assertions for removed behavior with assertions for the new contract. |

Rollback is a source revert. No runtime flag, environment switch, or dual path
is introduced. Because persisted packets are only ever pruned toward a smaller
valid shape, a revert leaves stored data valid.

## Target State

```text
group scene build
  ambient turns
    -> drop turns older than GROUP_SCENE_MAX_TURN_AGE_MINUTES vs trigger time
    -> existing count and character fitting
    -> render (omitted_turn_count counts only count-based truncation)

conversation progress load
  stored packet (unexpired by rolling EPISODE_TTL)
    -> drop events whose age exceeds their retention tier threshold
    -> if newest surviving event older than narrative threshold, clear the
       narrative field set to the canonical empty shape
    -> prompt projection and prior-episode hand-off both use the pruned packet
    -> next recorded turn persists the pruned form

cognition input build
  scene_context.semantic_temporal_context  derived, not hardcoded
  scene_context.character_sleep_phase      derived from the configured window
  progress evidence occurred_at            the event's own timestamp

morning refresh
  reflection worker tick (schedule, idempotency, CAS, audit, callback)
    -> cognition_core_v2.run_character_morning_refresh
         scope guard -> sleep recovery reducer -> validate_cognition_state
         -> recovered state + bounded transition summary
```

Cognition never receives a group-scene turn, progress event, or narrative
summary older than its configured threshold, and never receives a fabricated
`occurred_at`.

## Contracts And Data Shapes

### Thresholds

Add to `src/kazusa_ai_chatbot/config.py` using the existing
`_positive_int_from_env` helper, and re-export through
`src/kazusa_ai_chatbot/conversation_progress/policy.py` so the module keeps
ownership of its mechanical limits:

```python
GROUP_SCENE_MAX_TURN_AGE_MINUTES = 120
CONVERSATION_PROGRESS_BACKGROUND_MAX_AGE_MINUTES = 120
CONVERSATION_PROGRESS_ACTIVE_SCENE_MAX_AGE_MINUTES = 360
CONVERSATION_PROGRESS_DECISION_CRITICAL_MAX_AGE_MINUTES = 2880
CONVERSATION_PROGRESS_NARRATIVE_MAX_AGE_MINUTES = 360
```

`ConversationProgressEventV2.retention` has three values, so all three need a
threshold. Age is measured from the event's `updated_at`, which is the field
`_ordered_events` already uses for its recency tie-break.
`CONVERSATION_PROGRESS_DECISION_CRITICAL_MAX_AGE_MINUTES` equals the existing
48-hour `EPISODE_TTL`, so that tier expires with the packet as it does today.

Age is evaluated against the `current_timestamp_utc` already threaded through
`load_progress_context` and `ConversationProgressRuntime.load`. Age for group
scene is evaluated against `trigger_occurred_at`, which
`build_group_scene_context` already parses as its effective present.

### Progress prune

Add to `src/kazusa_ai_chatbot/conversation_progress/policy.py`:

```python
def prune_aged_progress_packet(
    packet: ConversationProgressStateV2,
    *,
    current_timestamp_utc: str,
) -> tuple[ConversationProgressStateV2, int, bool]:
    """Drop aged events and clear a stale narrative before projection."""
```

Returns the pruned packet, the number of dropped events, and whether the
narrative set was cleared. Rules:

- Drop an event when `current_timestamp_utc - event["updated_at"]` exceeds the
  threshold for `event["retention"]`. Drop the whole event dictionary; an event
  cannot be blanked in place because `_validate_stored_event` requires
  non-empty `event_id`, `semantic_summary`, `actor`, `action`, `object`,
  `first_seen_at`, and `updated_at`. The `events` list may become empty.
- After dropping, if there is no surviving event, or the newest surviving
  `updated_at` is older than `CONVERSATION_PROGRESS_NARRATIVE_MAX_AGE_MINUTES`,
  set `current_thread`, `character_stance`, `user_goal`, `current_blocker`,
  `emotional_trajectory`, and `episode_narrative` to `""` and `overused_moves`
  to `[]`. Empty strings are legal: `validate_active_packet` calls
  `_bounded_text` without `required=True` for each of these. Keys must remain
  present; `None` and absent keys are invalid.
- Do not change `created_at`, `updated_at`, `expires_at`, `turn_count`,
  `status`, or any block reference. `turn_count` must stay at or above 1 while
  `status == "active"`.
- The returned packet must satisfy `validate_active_packet`.

Call it in `ConversationProgressRuntime.load` immediately after `_select_packet`,
so the same pruned object flows into `build_progress_prompt`,
`continuation_projection_chars`, and the returned
`ConversationProgressLoadResult`. Do not add a parameter to
`build_progress_prompt`.

### Group-scene age filter

In `src/kazusa_ai_chatbot/conversation_progress/projection.py`, inside
`build_group_scene_context`, filter within the existing ambient loop where
`occurred_time` is already parsed, before the row is appended to `ambient`.
Discarded turns must not enter `ambient`, so `total_ambient_count=len(ambient)`
and therefore `omitted_turn_count` reflect only count-based truncation. The
trigger turn is never age-filtered. `_fit_group_scene_turns` and
`project_group_scene_prompt` are unchanged and stay timestamp-free.

### Progress evidence provenance

In `project_conversation_progress_evidence`, set
`evidence_ref["occurred_at"]` from the event's own `updated_at` rather than the
`occurred_at` argument. Keep the argument, which still supplies the episode
time for any row that needs it and preserves the signature.

`EvidenceRefV2.occurred_at` is format-validated only. `_validate_evidence_ref`
requires an ISO-8601 UTC timestamp ending in `Z` and imposes no relation to the
episode timestamp; `state_reducers.py` never reads `occurred_at`, and the causal
event identity hash excludes it. Event timestamps must therefore be normalized
to the same `Z`-suffixed, second-truncated form the episode timestamp uses.

The one behavioral consequence to accept and verify:
`state_projection._relationship_evidence_freshness` renders
`project_duration(max(occurred_at), effective_at)`, so relationship evidence
freshness will now reflect real ages. Because aged events are discarded before
projection, surviving rows are within their tier threshold and the rendered
freshness stays bounded.

### Derived temporal context

Replace the hardcoded `"semantic_temporal_context": "immediate"` in
`nodes/persona_supervisor2_cognition.py` with a value derived from the newest
surviving progress event age, using the existing `project_duration` vocabulary.
When there is no surviving event, the value describes the current turn only.
The field stays a bounded string; `_validate_scene_context` is unchanged for it.

### Shared clock helpers

Add to `src/kazusa_ai_chatbot/time_boundary.py`:

```python
def local_period_bounds(local_period: str) -> tuple[int, int]:
    """Parse exact HH:MM-HH:MM text into local minutes after midnight."""


def local_minutes_in_zone(now: datetime, *, time_zone: str) -> int:
    """Project one timezone-aware instant into local minutes after midnight."""
```

`self_cognition/sleep_period.py` uses both and deletes its private
`_local_period_bounds` and `_local_time_minutes`. Its public
`is_self_cognition_sleep_period` keeps its exact signature, its timezone-aware
requirement, its empty-period-disabled behavior, and its half-open
`[start, end)` semantics including the midnight wrap.

### Sleep-phase projector

Add to `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`:

```python
def project_character_sleep_phase(
    now: datetime,
    *,
    sleep_local_period: str,
    character_time_zone: str,
    wake_prep_minutes: int,
) -> str:
    """Translate the configured sleep window into a frozen phase label."""
```

Deterministic, config-free, keyword-only, `ValueError` on out-of-domain input,
matching `project_duration` and `project_numeric_band` style. Returns one of
three frozen Chinese labels for: outside the window; inside the window but not
within `wake_prep_minutes` of its end; inside the window and within
`wake_prep_minutes` of its end. An empty `sleep_local_period` returns the
outside-window label. The union of the two in-window labels is exactly the
window that `is_self_cognition_sleep_period` reports, so the two stay
consistent by construction.

### Scene context field

Extend `SceneContextV2` in `cognition_core_v2/contracts.py`:

```python
character_sleep_phase: NotRequired[str]
```

`_validate_scene_context` adds it to the required set only when present,
following the existing `current_user_role` pattern, and validates it as bounded
text. `nodes/persona_supervisor2_cognition.py` populates it from
`project_character_sleep_phase`, passing `CHARACTER_SLEEP_LOCAL_PERIOD`,
`CHARACTER_TIME_ZONE`, and `AFFECT_SETTLING_WAKE_PREP_MINUTES`.

`scene_context` reaches goal cognition and not appraisal or surface. That reach
is the intended scope: the sleep phase shapes motive, not evidence appraisal.
`goal_cognition` strips role fields from `scene_context` and must pass the new
field through unchanged.

### Morning-refresh entrypoint

Add a `cognition_core_v2` public entrypoint and export it from
`cognition_core_v2/__init__.py`:

```python
def run_character_morning_refresh(
    state: Mapping[str, Any],
    *,
    elapsed_sleep_seconds: int,
    updated_at: str,
) -> CharacterMorningRefreshResultV2:
    """Own the complete deterministic character morning-refresh transition."""
```

`CharacterMorningRefreshResultV2` is a `TypedDict` carrying a
`schema_version` literal, the recovered state, the applied
`elapsed_sleep_seconds`, and bounded deterministic transition counts sufficient
for the caller's audit row. The entrypoint owns the character-scope guard, the
`apply_sleep_recovery` call, and `validate_cognition_state` on its output,
closing the current gap where the recovered state is returned unvalidated. It
knows nothing about local dates, run identifiers, or persistence.

`reflection_cycle/affect_settling.py` stops importing `apply_sleep_recovery`
and calls this entrypoint from its existing `sleep_recovery` adapter, which
keeps building the audit document from the returned summary plus its own
`local_date_key`, `started_at`, and `completed_at`. Scheduling, the
`character_reflection_runs` idempotency row, the compare-and-replace write, and
the refresh callback are unchanged.

`apply_sleep_recovery` drops its `character_constraints` and
`relationship_context` parameters, which it accepts and ignores. Its dampening
arithmetic is unchanged.

## Runtime Or Resource Constraints

The constraint source is the project's own prompt caps, not a generic default.
This change adds no model call and no scheduled job.

- `scene_context.character_sleep_phase` adds a short bounded label to the goal
  cognition payload only. `GOAL_COGNITION_PROMPT_CAP` is 36000 characters with
  `scene_context` reducible under existing floors, so the addition is absorbed.
- Every discard strictly reduces projected characters in
  `public_group_scene` (capped 1800) and `conversation_continuity` (capped
  2200), and reduces progress evidence row count.
- Acceptance evidence is that the existing prompt-budget contract tests still
  pass and that no new context-limit failure appears.

## Change Surface

Target ownership boundary: `conversation_progress` for discard,
`cognition_core_v2` for the sleep phase and morning refresh.

Changes outside those boundaries and why: `config.py` is the project's only
environment reader, so the thresholds cannot live elsewhere without duplicating
the config contract. `time_boundary.py` already owns local-clock projection and
is the correct home for helpers shared by `self_cognition` and
`cognition_core_v2`, avoiding a new dependency between those two.
`nodes/persona_supervisor2_cognition.py` and `nodes/persona_supervisor2.py` are
the only builders of the cognition input and the group scene.
`reflection_cycle/affect_settling.py` is the sole caller of the reducer being
re-homed.

### Modify

- `src/kazusa_ai_chatbot/config.py` — add five threshold constants.
- `src/kazusa_ai_chatbot/time_boundary.py` — add the two shared clock helpers.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py` — surface the
  thresholds; add `prune_aged_progress_packet`.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py` — group-scene age
  filter; event-own `occurred_at` in progress evidence.
- `src/kazusa_ai_chatbot/conversation_progress/__init__.py` — export the shared
  group-scene ambient-turn filter used by the persona stage.
- `src/kazusa_ai_chatbot/conversation_progress/runtime.py` — call the prune
  after packet selection in `load`.
- `src/kazusa_ai_chatbot/self_cognition/sleep_period.py` — use the shared
  helpers; delete the two private parsers; preserve the public contract.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py` — add
  `project_character_sleep_phase`.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` — add the optional
  `character_sleep_phase` field and its validation.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py` — remove the two
  ignored parameters from `apply_sleep_recovery`.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` — pass the new
  scene field through the existing role-field stripping.
- `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py` — export the
  morning-refresh entrypoint and its result type.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` — derive
  `semantic_temporal_context`; populate `character_sleep_phase`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` — apply the shared
  age-filtered ambient sequence before group scope and Stage 0 prompt
  construction.
- `src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py` — call the v2
  entrypoint instead of importing the reducer.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and `docs/HOWTO.md` —
  document the thresholds, the discard contract, the sleep-phase field, and the
  morning-refresh entrypoint.
- `tests/test_conversation_progress_v2_live_llm.py` — add individually-run
  live acceptance cases for stale group ambient turns and stale private
  progress events.

### Create

- One `cognition_core_v2` module for the morning-refresh entrypoint and its
  result type, or an addition to an existing v2 module if the implementation
  owner finds a better-fitting one within this boundary.

### Delete

- `self_cognition/sleep_period.py` private `_local_period_bounds` and
  `_local_time_minutes`.
- The `character_constraints` and `relationship_context` parameters of
  `apply_sleep_recovery`.
- The direct `apply_sleep_recovery` import in
  `reflection_cycle/affect_settling.py`.

### Keep

- `conversation_progress` compaction limits, `_event_tier`, `EPISODE_TTL`, the
  Mongo TTL indexes, and `db/conversation.py`.
- `db/interaction_style_images.py` and everything serving the group style image.
- `reflection_cycle/affect_settling.py` schedule math, window predicates,
  idempotency, persistence, and import-time validation.
- The two self-cognition sleep enforcement call sites.
- `nodes/persona_supervisor2_l3_surface.py` and `TextSurfaceInputV2`.

### Tests

Existing tests that must be updated because their asserted behavior changes:

- `tests/test_conversation_progress_group_scene.py` — age filtering and
  `omitted_turn_count` semantics.
- `tests/test_conversation_progress_cognition_evidence.py` — `occurred_at` now
  comes from the event.
- `tests/test_conversation_progress_cognition.py`,
  `tests/test_conversation_progress_runtime.py` — prune on load.
- `tests/test_conversation_progress_v2_contract.py`,
  `tests/test_conversation_progress_compaction.py` — packet validity after
  pruning and after narrative clearing.
- `tests/test_cognition_core_v2_contracts.py` — `scene_context` optional field.
- `tests/test_cognition_core_v2_failures.py` —
  `test_elapsed_decay_and_sleep_recovery_are_scope_specific` for the reducer
  signature change.
- `tests/test_self_cognition_sleep_period.py` — behavior must remain identical
  after the helper relocation; these five tests are the regression gate.
- `tests/test_reflection_affect_settling.py` — the entrypoint indirection.
- `tests/test_reflection_cycle_stage1c_worker.py`,
  `tests/test_self_cognition_integration.py` — these monkeypatch
  `is_self_cognition_sleep_period` and `_run_daily_affect_settling` by name;
  preserve those names or update the patch targets.
- `tests/cognition_baseline_worker.py` — calls `run_daily_affect_settling`.

New coverage required for: each retention tier's age boundary, narrative
clearing and its non-clearing case, group-scene age boundary including that the
trigger turn is never dropped, `omitted_turn_count` excluding aged turns,
pruned-packet validity, sleep-phase label boundaries including the empty-period
and midnight-wrap cases, and morning-refresh output validation.

### Lifecycle

- this plan;
- `development_plans/README.md`.

## Agent Autonomy Boundaries

The responsible agent may choose local mechanics, decomposition, work order,
helper placement inside the named modules, and verification breadth, provided
the contracts in `Contracts And Data Shapes` and the change surface hold.

The agent must not: add a compatibility path, fallback, feature flag, or dual
read; add an LLM call, scheduled job, or cache; introduce a freshness label or
any other model-visible age signal used to delegate a discard decision; change
a threshold to a value not stated here; alter the sleep enforcement scope or the
reply path; extend the sleep phase to appraisal or surface; rewrite, paraphrase,
or synthesize any progress content; change the affect-settling schedule math.

If the plan and code disagree, if a stated contract cannot be implemented inside
the boundary, or if pruning cannot produce a packet that satisfies
`validate_active_packet`, the agent records the conflict and requests a decision
or plan amendment rather than reinterpreting the plan.

## Verification

The affected contracts are: conversation-progress packet validity, group-scene
projection, cognition input validation, goal-cognition prompt budget, the
self-cognition sleep gate, and the morning-refresh state transition. The
implementation owner selects exact commands and breadth; the checks below are
the acceptance-bearing ones.

- **Deterministic contract checks.** Age boundaries for all three retention
  tiers, narrative clearing and non-clearing, group-scene age boundary and
  trigger retention, `omitted_turn_count` semantics, pruned-packet validity
  through `validate_active_packet`, sleep-phase label boundaries, and
  morning-refresh output validation.
- **Behavior-preservation gate.** The five existing sleep-period tests must
  pass unchanged in intent after the helper relocation. Any required edit there
  is a signal that behavior drifted and needs review.
- **Regression radius.** The `cognition_core_v2`, `conversation_progress`,
  `reflection_cycle`, and `self_cognition` suites, plus the cognition prompt
  budget contract tests, because the change touches prompt content, persisted
  packet shape, and a shared reducer.
- **Static checks.** Compile every edited Python file, immediately for any file
  containing Chinese text. Confirm no remaining direct import of
  `apply_sleep_recovery` outside `cognition_core_v2`, no remaining private
  clock parser in `self_cognition/sleep_period.py`, and no remaining hardcoded
  `"immediate"` temporal context.
- **Persistence check.** A read-path prune followed by a recorded turn persists
  the pruned packet; a read-path prune alone performs no write.
- **Live check.** One group-channel turn and one long-running private
  conversation, inspected to confirm that aged turns and aged events are absent
  from the rendered prompts and that the sleep-phase label is correct for the
  wall-clock time of the run. Run live cases one at a time and inspect each
  artifact.

Record the checks actually run, their results, and any residual risk.

## Acceptance Criteria

- No ambient group-scene turn older than `GROUP_SCENE_MAX_TURN_AGE_MINUTES`
  reaches any prompt, and `omitted_turn_count` reflects only count-based
  truncation.
- No progress event older than its retention tier threshold reaches any prompt
  or any evidence row.
- When the newest surviving event is older than the narrative threshold, or no
  event survives, the narrative field set is empty in the projected prompt.
- A pruned packet satisfies `validate_active_packet`; a read-path prune issues
  no database write; the next recorded turn persists the pruned form.
- Every progress evidence row carries the originating event's own timestamp.
- `scene_context.semantic_temporal_context` is derived, and no hardcoded
  `"immediate"` remains.
- `scene_context.character_sleep_phase` is populated, validated, optional for
  backward-valid inputs, and present in the goal-cognition payload.
- `is_self_cognition_sleep_period` keeps its signature and observable behavior,
  and the sleep window still gates exactly the two self-cognition lanes.
- The reply path is unaffected by the sleep phase.
- `reflection_cycle` invokes the morning refresh only through the
  `cognition_core_v2` public entrypoint, and that entrypoint validates its
  output.
- `apply_sleep_recovery` has no unused parameters and unchanged arithmetic.
- No LLM call, scheduled job, cache, or freshness label was added.
- The group style image is untouched.
- Documentation describes the final thresholds and contracts.

## Progress Checklist

- [x] Thresholds and shared clock helpers added; `self_cognition` parsers
  removed with its five tests passing unchanged in intent.
- [x] `prune_aged_progress_packet` implemented and wired into
  `ConversationProgressRuntime.load`; packet validity and tier boundaries
  covered.
- [x] Group-scene age filter implemented; trigger retention and
  `omitted_turn_count` covered.
- [x] Progress evidence `occurred_at` and derived
  `semantic_temporal_context` implemented; relationship freshness impact
  checked.
- [x] Sleep-phase projector and `scene_context` field implemented and reaching
  goal cognition.
- [x] Morning-refresh entrypoint implemented, exported, and consumed by
  `reflection_cycle`; reducer parameters removed.
- [x] Documentation updated.
- [x] Regression radius and static checks pass; live cases inspected.
- [x] Independent code review complete and findings resolved.

Each item records its changed surface, verification result, and any deviation
in `Execution Evidence` before it is marked complete.

## Independent Code Review

This gate is required because the change alters persisted packet shape,
prompt-visible content, a cognition input contract, and a shared state reducer.

The review checks: that no discard decision was delegated to a prompt; that
deterministic code never rewrites or synthesizes progress content; that pruning
always yields a valid packet and never writes on the read path; that the sleep
enforcement scope and the reply path are unchanged; that the affect-settling
schedule math is untouched; that no compatibility path, fallback, cache, LLM
call, or unrelated cleanup was added; and that the sleep-period behavior gate
genuinely passed rather than being edited to fit.

The responsible owner resolves findings inside the approved boundary and reruns
affected verification. Findings that need a new contract return to plan
amendment.

## Execution Evidence

Populate during execution. Do not pre-fill.

- Pre-change baseline and git state: `git status --short` was clean before
  delegation. After the explicit user execution command, the parent moved this
  plan and its registry row from `draft` to `in_progress`. DeepSeek completed
  the acknowledgement turn and began the bounded implementation turn with the
  explicit owned-file list; its 600-second hard deadline expired, so the parent
  interrupted the handoff, preserved the workspace patch, and completed the
  review locally.
- Threshold and clock-helper relocation result, including sleep-period gate:
  the five configured thresholds and `time_boundary` helpers are present;
  `tests/test_self_cognition_sleep_period.py` passed all five cases, and the
  new sleep-phase/shared-clock coverage passed.
- Progress prune result, tier boundaries, packet validity: all retention-tier
  boundary, stale-narrative, non-clearing, validation, and read-path tests
  passed. `tests/test_conversation_progress_v2_contract.py`,
  `tests/test_conversation_progress_runtime.py`, and
  `tests/test_conversation_progress_compaction.py` passed 41 tests together.
- Group-scene filter result: the age boundary, trigger retention, and
  age-excluded `omitted_turn_count` tests passed; the complete non-live
  conversation-progress suite passed 171 tests.
- Evidence provenance and temporal-context result, relationship freshness
  check: event-owned UTC-Z timestamps and newest-survivor duration projection
  passed the cognition connector/evidence checks. The cognition V2
  deterministic suite passed 458 tests, with four explicitly excluded live or
  benchmark files.
- Sleep-phase projector and scene-field result: same-day, overnight, empty,
  invalid-input, shared-clock, and self-cognition-union cases passed. The
  optional contract validates backward-compatible inputs, and the connector
  test confirms the field reaches goal cognition.
- Morning-refresh entrypoint result and output validation: the public export,
  character-scope guard, reducer signature, state-output validation, bounded
  transition counts, and reflection adapter route passed 40 focused tests.
- Persistence check result: read-path pruning performed no database write in
  the runtime tests, and the next recorded turn persisted the pruned packet.
- Regression and static-check results (final): the non-live conversation-
  progress regression radius passed 173 tests; the non-live Cognition V2
  suite passed 513 tests with two expected day-wide trace-inventory skips and
  four live/benchmark deselections; the reflection, self-cognition, and
  persona regression batch passed 377 tests; and the post-correction prompt
  budget/guidance checks passed 55 tests. All 139 changed Python files
  compiled under `venv\Scripts\python`; `git diff --check` passed. Production
  search found no direct `apply_sleep_recovery` import outside
  `cognition_core_v2`, no private clock parser in
  `self_cognition/sleep_period.py`, and no hardcoded
  `semantic_temporal_context` value of `immediate`.
- Repository-wide default baseline: `pytest -q -rA --maxfail=1` stopped after
  1,504 passed, 21 skipped, and 1,134 deselected at
  `tests/test_cognition_preference_adapter.py::test_preference_stage_owns_visible_boundaries_only`.
  The isolated failure is a stale preference-prompt expectation outside this
  plan's change surface; the test file is untouched, and every plan-specific
  regression batch passed.
- Live case artifacts and judgment (initial attempts): the group case passed
  individually and wrote
  `test_artifacts/llm_traces/test_live_interleaved_group_multifragment_continuation__group_multifragment__20260805T095421231240Z.json`;
  its trace showed two accepted recorder calls, eight participant turns, the
  unrelated source excluded, and coherent downstream output. The first
  private long-thread attempt wrote
  `test_artifacts/llm_traces/test_live_asuna_houjing_long_thread_regression__asuna_houjing_cognition_semantic_failure__20260805T095603271546Z.json`;
  its contract reached both relevant events, but the model-selected bid
  omitted the completed event and repeated its completed location. The
  failure is retained as semantic-quality evidence in the human-readable
  closure review.
- Live closure reruns: after a bounded prompt-only correction in
  `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`, the private
  long-thread replay passed in 86.57 seconds with completed-event citation,
  distinct-location selection, current-input grounding, no accidental reopen,
  and intact source lineage. The final group replay passed in 37.51 seconds
  with two accepted recorder calls, eight participant logical turns, seven
  final fragments, `unrelated_group_source_absent=true`, and
  `critical_event_state=completed`. Both final traces and the parent-authored
  semantic judgments are recorded in
  `test_artifacts/diagnostics/cognition_core_v2_context_fade_sleep_phase_closure_review_2026-08-09.md`.
- Known asymmetries recorded for a future decision: morning refresh is
  character-scope only, so user-scope affect receives no nightly relief; and
  `elapsed_sleep_seconds` is the configured window length rather than real
  elapsed time.
- Review findings, fixes, verification rerun: parent review replaced string
  timestamp ordering with parsed-instant ordering, normalized invalid timezone
  errors to `ValueError`, removed the remaining production temporal literal in
  the validation CLI, and made the morning-refresh scope guard explicit. The
  affected focused suites and compilation checks passed after each fix. The
  live group fixture also received its missing required `public_group_scene`
  field so the prescribed case could reach the changed builder.
- Default-agent independent review and follow-up: the reviewer identified a
  stale group ambient sequence still reaching Stage 0, temporal context being
  derived from the capped prompt subset, and non-integer morning-refresh input
  being accepted. A shared parsed-time group filter now feeds scope, scene,
  and Stage 0; temporal context now reads the full pruned episode packet; and
  morning refresh rejects booleans and non-integers. The reviewer also required
  stale-data live acceptance coverage. The new group and private cases passed
  individually, and their durable traces were inspected; the focused
  deterministic suite passed 31 tests, the conversation/cognition batch
  passed 632 tests, the persona/cognition integration batch passed 58 tests,
  and the reflection/self-cognition batch passed 54 tests.
- Residual-risk disposition: the initial private semantic failure was resolved
  by the bounded goal-prompt correction and the replay passed on the second
  attempt. The missing identity-growth replay source remains unavailable, so
  its live-only module is explicitly skipped rather than fabricated. The
  repository-wide baseline preference-prompt failure is outside this plan's
  ownership boundary and remains unmodified. No deterministic semantic rewrite
  or parser-side repair was added.
- Closure follow-up after remote integration: the initial implementation was
  committed as `dc8915ea`, then rebased onto remote commits
  `1b443317` and `ecdd885d` as `d5a48e2e`; the branch is one commit ahead of
  `origin/cognition_core_v2`. Follow-up workspace changes fix the canonical
  self-cognition relationship vocabulary, update the stale prompt assertion,
  guard collection on the unavailable identity-growth manifest, and add
  reusable recorder and selection-stage prompt guidance. The focused
  deterministic contract suites remain green. The live traces and a
  human-readable review are recorded in
  `test_artifacts/diagnostics/cognition_core_v2_closure_long_thread_review_2026-08-05.md`.
- Thinking diagnostic: the live route resolved to
  `CONSOLIDATION_LLM_THINKING_ENABLED=False`, `model_family=gemma4`, and
  `thinking_strategy=disabled`. Visible correction notes and duplicate JSON
  objects in the raw traces are model output, not provider-side thinking
  enablement.
- Final diff summary and sign-off: thresholds, lazy progress pruning,
  group-scene age filtering, event timestamp provenance, derived temporal
  context, shared clock helpers, cognition-owned sleep phase, validated
  morning refresh, reflection routing, documentation, the bounded goal-prompt
  correction, and focused tests are complete. The deterministic contract
  surface and final live semantic gates are signed off; residuals are
  explicitly classified above.
