# self cognition memory semantics plan

## Summary

- Goal: Enable approved persistence categories for self-cognition by extending
  the existing per-lane consolidation write policy to recognize an
  `internal_thought` origin, and by adding the one new write lane the
  agency-loop playground already identifies as legitimate
  (`conversation_progress`).
- Plan class: large
- Status: draft
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `no-prepost-user-input`
- Overall cutover strategy: compatible
- Highest-risk areas: choosing the correct allowed lanes, scoping the new
  `conversation_progress` write contract, keeping self-cognition out of the
  upstream consolidator LLM lanes, duplicate suppression, and auditability
- Acceptance criteria: not executable yet; this draft is ready for approval
  only after the `conversation_progress` write contract and the per-lane
  decision table are accepted.

## Context

Self-cognition is intended to generate durable side effects, but the existing
architecture already specifies *how* non-`user_message` origins should be gated
at the persistence boundary. Stage 06 of
`development_plans/archive/completed/short_term/multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md`
introduced a per-lane policy infrastructure that is already in `main`:

- `ConsolidationWritePolicy` is a `TypedDict` with seven persistence keys
  (`character_state`, `relationship_insight`, `user_memory_units`,
  `task_dispatch`, `affinity`, `character_image`, `cache_invalidation`) in
  `persona_supervisor2_consolidator_origin_policy.py`.
- `build_consolidation_write_policy(...)` is the single decision point that
  `db_writer` consults before each durable write.
- The current implementation denies every lane for any origin other than
  `user_message`. The archived Stage 06 plan explicitly hands off to "later
  approved plans" to flip individual lanes on for non-chat origins.

This plan is that later approved plan for `internal_thought` (self-cognition)
origin. It does **not** rebuild a parallel memory writer. It also does not
adopt the prior draft's proposal to enable `user_memory_units` writes for
self-cognition: the agency-loop playground artifacts already in repo
(`test_artifacts/self_cognition_shared_poc/*/self_cognition_consolidation_candidate.json`)
explicitly block `user_memory_units` for self-cognition with the documented
reason *"self-cognition is not user-authored evidence; future production
origin policy must verify source lineage before writes."* Real cognition
output for those cases produces character-introspective content
(`internal_monologue`, `action_directives`, `character_intent`,
`emotional_appraisal`, `logical_stance`) and does not synthesise new
user-authored facts; the user-relevant facts already live in
`memory_evidence` as input from prior live turns.

The same playground artifacts identify the one new write lane self-cognition
*should* be able to produce: `conversation_progress`, keyed by topic, carrying
`progression_guidance`, `action_status`, `logical_stance`, and
`character_intent`. This lane does not currently exist in `db_writer`. Adding
it is the actual net-new persistence work for this plan.

The prerequisite bugfix plan removes `no_remember` from self-cognition-created
state. That removal does not by itself cause any writes, because the per-lane
policy still denies every lane for `internal_thought` origin. This plan owns
the policy decision for which lanes to flip on and the `conversation_progress`
write that goes with it.

## Discovery Decisions Needed Before Approval

This plan is intentionally draft. Resolve these decisions before marking it
`approved`:

- The exact persisted shape of a `conversation_progress` record: required
  fields, upsert key (proposed:
  `(platform, platform_channel_id, global_user_id, topic_id)`), retention.
- Where the `conversation_progress` writer lives: a new function under
  `src/kazusa_ai_chatbot/db/` consumed by `db_writer`, or a focused write
  helper inside `persona_supervisor2_consolidator_persistence.py`.
- Whether self-cognition enters `db_writer` through a focused per-lane apply
  function that runs only allowed lanes, or through a trimmed graph entry that
  skips `global_state_updater`, `relationship_recorder`, and
  `facts_harvester`. Proposed default: focused per-lane apply function; do not
  call the full graph.
- Whether `conversation_progress` extraction is purely deterministic
  (re-using `character_intent`, `logical_stance`, and progression-guidance
  fields already present in the cognition output) or requires an extra bounded
  LLM judgment call. Proposed default: deterministic only; the fields already
  exist.
- Duplicate suppression key for `conversation_progress`: per-topic overwrite
  with monotonic `written_at`, or append-with-idempotency on
  `(topic_id, origin_episode_id)`.
- Confirmation that no existing module already persists conversation progress
  under a different name. Stage 0 must grep for it before implementation.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing prompt, graph, cognition,
  consolidation, memory, or background LLM behavior.
- `no-prepost-user-input`: load before changing memory extraction, promise
  persistence, or any path that decides whether user-facing instructions,
  preferences, commitments, or accepted actions become durable state.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Do not reintroduce `no_remember` into self-cognition-created state.
- Do not enable the `user_memory_units` lane for `internal_thought` origin in
  this plan. The playground evidence explicitly blocks it; enabling it without
  source-lineage verification is out of scope.
- Do not enable fact reinforcement, fact invention, or self-authored
  `future_promises` memory writes for `internal_thought` origin. These
  patterns are denied by construction; see `Writable Channels Principle`.
- Do not call `global_state_updater`, `relationship_recorder`, or
  `facts_harvester` for self-cognition. The corresponding state fields are
  already produced by the shared cognition graph; re-running those lanes would
  spend background LLM calls re-extracting content already present.
- Do not call the full `call_consolidation_subgraph` for self-cognition.
- Do not use `/chat`, adapter delivery, or synthetic conversation-history rows
  as the persistence path.
- Do not persist memory from raw reflection rows, raw source packets, raw DB
  documents, embeddings, or full self-cognition artifacts.
- Do not add deterministic keyword rules that reinterpret LLM-judged channels
  after the fact. If extraction is wrong, fix the prompt/schema and structural
  validation.
- Do not add retry loops, model-context increases, alternate LLM routes, or
  fallback prompts as the primary strategy for extraction failures.
- Do not build a parallel memory orchestrator. Reuse
  `build_consolidation_write_policy` and the `db_writer` write paths gated by
  it. The previously proposed `src/kazusa_ai_chatbot/self_cognition/memory.py`
  is not created.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or
  final reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Add `build_self_cognition_consolidation_origin(...)` alongside
  `build_user_message_consolidation_origin(...)` in
  `persona_supervisor2_consolidator_origin.py`. It must validate
  `trigger_source=internal_thought`,
  `input_sources=("internal_monologue",)`, and `output_mode=preview` and
  raise `ConsolidationOriginError` on mismatch.
- Extend `build_consolidation_write_policy(...)` to return per-lane decisions
  for `internal_thought` origin, with `conversation_progress` allowed and all
  seven existing lanes denied with per-lane documented reasons taken from the
  playground artifacts.
- Add `conversation_progress` to the `WritePolicyKey` literal and to the
  `ConsolidationWritePolicy` `TypedDict`.
- Add a focused `conversation_progress` write contract (DB-layer upsert + a
  `db_writer` branch that consults the policy decision) so the new lane lives
  in one place.
- Thread the self-cognition consolidation origin from the self-cognition
  runner into a focused per-lane apply path so only the
  `conversation_progress` write runs and no upstream consolidator LLM lanes
  fire.
- Persist `self_cognition_consolidation_candidate.json` in production runs in
  the same shape as the existing playground artifact, recording
  `allowed_lanes`, `blocked_lanes`, candidate payload, and `evidence_lineage`.
- Add deterministic tests for per-lane policy decisions (both origins), the
  new write, upsert idempotency, dry-run behavior, runner threading, and
  worker wiring.

## Deferred

- Do not enable the `user_memory_units` lane for self-cognition.
  Source-lineage verification is required before any such change; the
  agency-loop playground artifacts document this constraint.
- Do not enable `character_state`, `relationship_insight`, `affinity`, or
  `character_image` lanes for self-cognition.
- Do not enable `task_dispatch` for self-cognition. It is already owned by
  the existing action-candidate handoff path.
- Do not redesign live-chat consolidation.
- Do not add new MongoDB collections beyond what `conversation_progress`
  requires; if a new collection is needed, this plan must be updated with a
  data migration section before implementation.
- Do not migrate historical self-cognition artifacts.
- Do not change reflection promotion semantics.
- Do not add response-path LLM calls.
- Do not add visual-directive behavior (already disabled by the bugfix plan).

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Per-lane policy extension | compatible | Add `internal_thought` branch alongside the existing `user_message` branch in `build_consolidation_write_policy`. Keep `user_message` behavior identical. |
| New `conversation_progress` lane | bigbang for the new lane | Add the lane in exactly one place. Do not maintain a parallel writer. |
| Self-cognition consolidator entry | compatible | Enter through a focused per-lane apply function. Do not run upstream LLM lanes. Do not call the full subgraph. |
| Dry runs | compatible | Preserve dry-run/no-write behavior. Record policy decisions and candidate payloads without DB writes. |
| Database | compatible | Use the existing connection. Add a single new collection or field for `conversation_progress` only if Stage 0 confirms no existing storage. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the approved contracts in this plan.
- The agent must not invent additional allowed lanes, expand the per-origin
  policy beyond `internal_thought`, or change the playground-documented
  reasons for blocked lanes.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, extra feature flags, or
  unrelated prompt rewrites.
- The agent must treat changes outside the files listed in `Change Surface`
  as out of scope unless the plan is updated first.
- If existing helpers exactly satisfy a needed projection, evaluator, DB
  writer, cache invalidation, or prompt-budget contract, reuse them.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker
  instead of inventing a substitute.

## Target State

Self-cognition produces a `consolidation_origin` with
`trigger_source=internal_thought` and threads it into a focused per-lane apply
path. That path calls `build_consolidation_write_policy(origin)`, which
returns a policy where only `conversation_progress` is allowed. The
`conversation_progress` write persists a record keyed by topic from the
cognition output's `character_intent`, `logical_stance`, and progression-
guidance fields. All other seven existing write lanes are denied for
self-cognition origin and produce no DB write, no dispatcher call, no
character-image side effect, and no cache invalidation.

Self-cognition does not call `global_state_updater`,
`relationship_recorder`, `facts_harvester`, or the full live-chat
consolidator graph. It does not call adapters, does not write conversation
rows, and does not redispatch tasks.

Each production self-cognition run writes a
`self_cognition_consolidation_candidate.json` artifact in the same shape as
the existing playground artifacts, recording `allowed_lanes`, `blocked_lanes`,
the candidate payload, and the evidence lineage.

## Proposed Per-Lane Decision Table For `internal_thought` Origin

This section is a draft proposal, aligned with the existing playground
artifacts. It is not an executable contract until accepted.

| Lane | Decision for `internal_thought` | Reason |
|---|---|---|
| `conversation_progress` (new) | Allow | The cognition output already contains the fields needed (`character_intent`, `logical_stance`, progression-guidance). This is what self-cognition meaningfully changes turn-over-turn. Playground artifacts explicitly allow it. |
| `user_memory_units` | Deny | Playground reason: *"self-cognition is not user-authored evidence; future production origin policy must verify source lineage before writes."* Real cognition output is character-introspective; user-relevant facts are already in `memory_evidence` from prior live turns. |
| `cache_invalidation` | Deny | Coupled to `user_memory_units` writes that do not fire. No retrieval cache becomes stale from a denied write. |
| `character_state` | Deny | Playground reason: *"hourly playground run cannot mutate stable character state."* Mood/vibe drift from idle reasoning needs a separate plan. |
| `relationship_insight` | Deny | Subjective drift risk from idle reasoning without a live user turn. |
| `affinity` | Deny | Same source-lineage argument as `user_memory_units`; no user-authored signal. |
| `task_dispatch` | Deny | Already owned by the existing action-candidate handoff (`self_cognition/handoff.py`). A memory lane must not redispatch. |
| `character_image` | Deny | Visual directives are disabled for self-cognition per the bugfix plan; no upstream signal exists to persist. |

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Persistence-policy mechanism | Reuse `build_consolidation_write_policy`. Add a branch for `internal_thought` origin; do not build a parallel writer. | Stage 06 already built per-lane policy as the canonical gate. Two writers would drift over time. |
| Upstream consolidator LLM lanes | Do not run for self-cognition. | The cognition output already contains the relevant fields. Re-running burns local-model budget and risks divergence. |
| Allowed lanes | Only `conversation_progress`. | Real playground evidence and real cognition output content. |
| Conversation-progress extraction | Deterministic projection from cognition output (no extra LLM call). | Local Gemma; minimise extraction risk. The needed fields already exist. |
| Audit artifact | Persist `self_cognition_consolidation_candidate.json` in production runs. | Existing playground pattern; reuse the schema. |
| Future expansion (more lanes) | Out of scope. | Each new lane requires its own source-lineage analysis and a follow-up plan. |
| Full consolidator graph | Do not call. | Even gated by policy, the upstream nodes (`global_state_updater`, `relationship_recorder`, `facts_harvester`) make additional LLM calls that are not needed when the cognition output already carries the equivalent fields. |

## Writable Channels Principle

The per-lane decision table is governed by a single rule: self-cognition may
write only to channels where both of the following hold.

1. Author identity: self-cognition is the legitimate author of the content.
   The cognition output expresses self-cognition's own interpretive
   judgment, not a derived claim about external (user-authored) facts.
2. No duplicate source of truth: no other production path already owns the
   channel for content that originates in self-cognition.

Only `conversation_progress` satisfies both for `internal_thought` origin
today. The eight existing lanes fail at least one of these conditions, as
recorded in the decision table reasons.

### Patterns explicitly out of scope

Three patterns that may otherwise look like natural extensions of
self-cognition memory are denied by construction. They are listed by name so
future plans must argue for the corrective before lifting the restriction.

- Fact reinforcement: boosting confidence, refreshing recency, or
  re-marking existing `user_memory_units`. Subsumed by the
  `user_memory_units` deny. Required corrective: a decay model so
  reinforcement competes with forgetting. Without decay, reinforcement is
  either a no-op or a read-write feedback loop on the same channel.

- Fact invention: writing new `user_memory_units` derived from idle
  reasoning rather than from a user-authored live turn. Subsumed by the
  `user_memory_units` deny. Required corrective: source-monitoring or a
  multi-memory cross-check so model-inferred content does not accumulate
  as if user-authored.

- Self-authored `future_promises` memory writes: persisting a
  character-initiated future commitment as a `user_memory_units`
  `future_promise`. Subsumed by the `user_memory_units` and `task_dispatch`
  denies. Required corrective: a scheduler-memory sync mechanism. The
  existing `action_candidate` -> handoff -> scheduler path already owns
  character-initiated future actions and provides duplicate suppression; a
  memory write would create a second source of truth with no sync.

### Conditions for lifting a denial

A future plan may flip a denied lane to allowed only if it presents:

- A named corrective that prevents the documented failure mode.
- Tests that prove the corrective behaves as claimed under the failure
  scenario.
- A scope explicit enough that the change to
  `build_consolidation_write_policy` is a single per-origin branch update,
  not a parallel writer.

## Contracts And Data Shapes

The final approved plan must replace this draft interface with exact accepted
fields and enums.

### Self-cognition consolidation origin builder

```python
def build_self_cognition_consolidation_origin(
    *,
    episode: CognitiveEpisode,
) -> ConsolidationOriginMetadata:
    """Project an internal-thought episode into consolidation origin metadata.

    Validates trigger_source=internal_thought,
    input_sources=("internal_monologue",), and output_mode=preview. Raises
    ConsolidationOriginError otherwise.
    """
```

### Extended write-policy function

```python
def build_consolidation_write_policy(
    *,
    origin: ConsolidationOriginMetadata,
) -> ConsolidationWritePolicy:
    """Return per-lane allow/deny decisions for the given origin.

    For trigger_source=user_message: existing behavior unchanged.
    For trigger_source=internal_thought: conversation_progress allowed; all
    other lanes denied with documented per-lane reasons.
    For any other origin: all lanes denied.
    """
```

### Extended write-policy keys

```python
WritePolicyKey = Literal[
    "character_state",
    "relationship_insight",
    "user_memory_units",
    "task_dispatch",
    "affinity",
    "character_image",
    "cache_invalidation",
    "conversation_progress",  # new
]
```

### Conversation-progress write request (proposed)

```python
class ConversationProgressWriteRequest(TypedDict):
    topic_id: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    character_intent: str
    logical_stance: str
    progression_guidance: str
    action_status: str
    origin_episode_id: str
    written_at: str
```

Upsert key:
`(platform, platform_channel_id, global_user_id, topic_id)`.

### Self-cognition consolidation candidate artifact

Match the existing playground artifact shape at
`test_artifacts/self_cognition_shared_poc/*/self_cognition_consolidation_candidate.json`,
with `origin`, `allowed_lanes`, `blocked_lanes`, and `evidence_lineage`
keys. The production artifact records the same decision shape as the
playground but reflects the actual write outcome (allowed-and-written,
allowed-but-dry-run, denied) rather than playground placeholders.

## LLM Call And Context Budget

Use `50k tokens` as the overall context-window assumption.

| Call | Path | Before | After |
|---|---|---|---|
| Conversation-progress extraction | self-cognition | 0 calls | **0 calls** — derived deterministically from cognition output |
| Global state updater | self-cognition | 0 calls | 0 calls — lane not invoked |
| Relationship recorder | self-cognition | 0 calls | 0 calls — lane not invoked |
| Facts harvester (+ evaluator loop) | self-cognition | 0 calls | 0 calls — lane not invoked |
| Full live-chat consolidator graph | self-cognition | 0 calls | 0 calls — graph not invoked |

This plan adds **zero** new background LLM calls. The
`conversation_progress` write is a deterministic projection over the
existing shared-cognition output.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
  - Add `build_self_cognition_consolidation_origin(*, episode)`.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
  - Add `conversation_progress` to `WritePolicyKey` and
    `ConsolidationWritePolicy`.
  - Branch `build_consolidation_write_policy` on origin:
    - `user_message`: existing behavior unchanged.
    - `internal_thought`: `conversation_progress` allowed; the seven existing
      lanes denied with per-lane reasons mirroring the playground artifacts.
    - other origins: all lanes denied.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - Add a `conversation_progress` write branch in `db_writer`, gated by
    `policy["conversation_progress"]["allowed"]`.
  - Add a focused `apply_self_cognition_consolidation(*, state, origin)`
    entry that runs only allowed lanes for the self-cognition path, so the
    upstream LLM lanes do not fire. Exact entry shape is fixed in Stage 0.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`
  - Add a `conversation_progress_candidate` field to `ConsolidatorState`
    carrying the deterministic projection from cognition output.

- `src/kazusa_ai_chatbot/db/` (exact module fixed in Stage 0)
  - Add `upsert_conversation_progress(...)` write function and the
    accompanying read helpers required by tests.

- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Build a `consolidation_origin` from the existing self-cognition cognitive
    episode and call the focused per-lane apply path with policy gating.
  - Persist a `self_cognition_consolidation_candidate.json` artifact
    recording `allowed_lanes`, `blocked_lanes`, the candidate payload, and
    evidence lineage.

- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Enable production `conversation_progress` writes through an explicit
    parameter.
  - Preserve dry-run no-write behavior; in dry-run, the artifact records the
    candidate without invoking the DB writer.

- `src/kazusa_ai_chatbot/self_cognition/models.py`
  - Add `ARTIFACT_CONSOLIDATION_CANDIDATE` constant and any new status
    enums required by the artifact.

- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Document the allowed lane, the blocked lanes with their reasons, the
    deterministic-projection contract, and the consolidation-candidate audit
    artifact.

- `tests/test_consolidation_origin_metadata.py`
  - Add tests for `build_self_cognition_consolidation_origin` validation
    (accept on matching episode, raise on mismatched fields).

- `tests/test_consolidator_origin_policy_db_writer.py`
  - Add per-origin lane decision tests covering `internal_thought` origin.
  - Assert `user_message` origin behavior is unchanged.

- `tests/test_self_cognition_*.py`
  - Add consolidation-candidate artifact tests, policy-gating tests, runner
    threading tests, dry-run tests, and worker wiring tests.

### Create

- No new orchestrator module is approved. The previous draft proposed
  `src/kazusa_ai_chatbot/self_cognition/memory.py`; that file is **not**
  created. The per-lane policy and the `db_writer` write paths are the only
  persistence mechanism.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
  - Do not call the full graph for self-cognition.

- `global_state_updater`, `relationship_recorder`, `facts_harvester` and
  their evaluator loop — unchanged and not invoked for self-cognition.

- Live `/chat` service path behavior for user-message consolidation.
- Scheduler and dispatcher validation behavior.
- Adapter delivery behavior.
- Reflection promotion behavior.
- Visual-directive default from the bugfix plan.

## Implementation Order

This plan is not executable until Stage 0 resolves the
`conversation_progress` write contract. After approval, implementation order
must follow:

1. Stage 0: confirm no existing module persists conversation progress under a
   different name; fix the upsert key, retention, exact field set, writer
   module location, and runner entry path.
2. Add failing tests for `build_self_cognition_consolidation_origin`
   validation.
3. Implement the origin builder.
4. Add failing tests for the extended per-lane policy decisions
   (`internal_thought` allows only `conversation_progress`; `user_message`
   unchanged; all other origins fully denied).
5. Implement the policy extension and the new `WritePolicyKey` entry.
6. Add failing tests for the conversation-progress write (allow path, deny
   path, upsert idempotency, missing-required-field rejection).
7. Implement `upsert_conversation_progress` and wire the lane through
   `db_writer` and the focused `apply_self_cognition_consolidation` entry.
8. Add failing tests for runner threading and the consolidation-candidate
   artifact shape.
9. Wire the runner and worker integration; preserve dry-run no-write.
10. Update README and docs.
11. Run focused and regression verification.
12. Run independent code review.

## Progress Checklist

- [ ] Stage 0 - discovery decisions resolved
  - Covers: conversation-progress write contract, upsert key, writer module
    location, runner entry path, duplicate suppression, confirmation no
    existing storage exists.
  - Verify: this draft no longer contains unresolved discovery decisions.
  - Evidence: record accepted decisions and updated plan diff.
  - Handoff: next agent starts at Stage 1.
  - Sign-off: `<agent/date>`.

- [ ] Stage 1 - executable plan finalized
  - Covers: exact contracts, implementation steps, verification commands, and
    acceptance criteria.
  - Verify: independent plan review passes.
  - Evidence: record review findings and approval status.
  - Handoff: implementation may start only after this stage.
  - Sign-off: `<agent/date>`.

- [ ] Stage 2 - implementation complete
  - Covers: approved implementation steps after plan finalization.
  - Verify: focused and regression tests pass.
  - Evidence: record command output.
  - Handoff: next agent starts independent code review.
  - Sign-off: `<agent/date>`.

- [ ] Stage 3 - independent code review complete
  - Verify: full diff reviewed against the approved plan and affected tests
    rerun after any review fixes.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, residual
    risks, and approval status.
  - Handoff: plan can be marked completed only after this stage is signed off.
  - Sign-off: `<agent/date>`.

## Verification

Final verification commands must be filled in after Stage 0 decisions. The
minimum expected gates are:

```powershell
venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py -q
venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_integration.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_*.py -q
```

Expected after implementation:

- `internal_thought` origin returns a policy where only
  `conversation_progress` is allowed.
- `user_message` origin policy is unchanged across all seven existing lanes.
- The conversation-progress write upserts under the documented key and is a
  no-op when the lane is denied.
- Self-cognition runs persist a `self_cognition_consolidation_candidate.json`
  artifact whose `blocked_lanes` reasons match the playground artifacts.
- No new MongoDB collection appears without an updated plan section.

## Independent Plan Review

Run this gate before approval or execution. Prefer a reviewer that did not
draft the plan. If no separate reviewer is available, the active agent must
reread the parent self-cognition architecture reference, this plan, the
archived Stage 06 plan, and relevant source/test context from a fresh-review
posture.

Review scope:

- The per-lane decision table for `internal_thought` matches the playground
  artifacts and the documented source-lineage constraint on
  `user_memory_units`.
- The plan does not call the full consolidator graph or the upstream LLM
  lanes for self-cognition.
- The plan does not introduce a parallel memory writer.
- The conversation-progress write contract is explicit and has an upsert key.
- Dry-run and live-worker behavior are distinct.
- Tests prove allowed writes, denied lanes, dry-run behavior, runner
  threading, upsert idempotency, and the consolidation-candidate artifact
  shape.

## Independent Code Review

Run this gate after all final `Verification` commands pass and before final
sign-off. Prefer a reviewer that did not implement the change. If no separate
reviewer is available, the active agent must reread this plan, inspect the
full diff from a fresh-review posture, and record that no separate reviewer
was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/context leaks, persistence
  risk, duplicate dispatcher side effects, brittle fixtures, and avoidable
  blast radius.
- Alignment with the approved per-lane decision table, `Must Do`, `Deferred`,
  `Change Surface`, verification gates, and acceptance criteria.
- Regression and handoff quality, including static-grep accuracy, execution
  evidence, and lifecycle registry updates.

## Acceptance Criteria

This draft is ready for approval when:

- Stage 0 decisions are resolved and encoded as directives.
- The per-lane decision table for `internal_thought` origin is accepted
  as-is or replaced with documented reasons.
- The `conversation_progress` write contract is explicit (fields, upsert
  key, writer module).
- The implementation order contains exact tests, expected failures, exact
  source edits, and final verification commands.

The implemented feature is complete only after the finalized acceptance
criteria replace this draft section.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Self-cognition is mistakenly allowed to write `user_memory_units` | Encode the per-lane denial with the playground's documented reason; assert the deny in tests | Per-origin policy tests |
| New `conversation_progress` writer duplicates an existing storage path | Stage 0 explicitly greps for existing storage before implementation | Stage 0 sign-off |
| Self-cognition accidentally triggers upstream LLM lanes | Enter via a focused per-lane apply function; do not call the full graph; static-grep for forbidden call sites | Runner tests and static greps |
| Per-lane policy regresses for `user_message` origin | Keep the `user_message` branch unchanged; add the `internal_thought` branch alongside | Existing user-message policy tests must continue to pass unchanged |
| Conversation-progress writes accumulate without bound | Upsert by `(platform, platform_channel_id, global_user_id, topic_id)`; do not append | Upsert idempotency tests |
| Dry-run path writes to production | Gate writes behind the explicit worker parameter; record the candidate-only artifact in dry-run | Dry-run test |
| Future plans drift back to a parallel memory writer | Acceptance criteria forbid creating `self_cognition/memory.py` or any other parallel orchestrator | Static grep in the code review gate |

## Execution Evidence

- Not started.

## Execution Handoff

Execution is not started. The next agent should begin at Stage 0 by
resolving the `conversation_progress` write contract with the owner before
this plan is approved.
