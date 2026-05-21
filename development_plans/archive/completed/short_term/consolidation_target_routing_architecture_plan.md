# consolidation target routing architecture plan

## Summary

- Goal: split consolidation by durable target so user, group-channel,
  character-self, and internal/background cognition do not share the
  user-profile write path.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `systematic-debugging`, `database-data-pull`, `py-style`,
  `test-style-and-execution`, `no-prepost-user-input`; apply `cjk-safety`
  before editing Python files containing CJK prompt text.
- Overall cutover strategy: staged bigbang for target planning and validation,
  compatible for current valid user-message consolidation, migration for
  existing malformed synthetic-user rows.
- Highest-risk areas: recreating fake user targets, weakening fail-fast
  user-profile invariants, and adding routing complexity that deterministic
  target validation does not need.
- Acceptance criteria: every durable write is validated against a deterministic
  target plan; group writes do not use user lanes; `self_cognition` is never
  used as a user id.

## Context

The 2026-05-20 failure was caused by a lifecycle mismatch, not by a missing
database default. A group-origin self-cognition path eventually wrote to
`user_profiles` as `global_user_id="self_cognition"`, creating a partial user
row. `get_affinity()` then raised `KeyError: 'affinity'`. That crash is the
intended fail-fast signal for malformed user-profile lifecycle data.

The architecture now has multiple cognition origins:

```text
/chat user message
  -> self-cognition source case
  -> group-chat review window
  -> scheduled future cognition
  -> character self-image or reflection-driven state
```

`origin_kind` explains why cognition ran. It does not identify the durable
entity that can be written. The missing contract is `target_kind`.

The failed path showed the key execution decision:

- group-image routing is deterministic and must not be produced by an LLM;
- a group-review consolidation window may write group-channel state, but it
  must not fabricate a user target from participant presence;
- source labels such as `self_cognition` are provenance only and never durable
  user identities.

This plan preserves the crash as a useful invariant and fixes the lifecycle
path that let a non-user become a user target.

## Mandatory Skills

- `development-plan-writing`: load before modifying or executing this plan.
- `local-llm-architecture`: load before changing consolidation prompts, graph
  routing, target routing, or background LLM behavior.
- `systematic-debugging`: load before changing the failure path so the fix
  addresses lifecycle root cause rather than the crash symptom.
- `database-data-pull`: load before read-only production diagnostics or cleanup
  planning.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `no-prepost-user-input`: load before changing user-message interpretation,
  accepted preferences, commitments, or memory-lifecycle semantics.
- `cjk-safety`: load before editing Python files containing CJK prompt text.

## Mandatory Rules

- Do not add defaults to `get_affinity()` or other user-profile readers to hide
  malformed user rows.
- Do not use `"self_cognition"` or any other origin/source label as a
  `global_user_id`.
- Keep `origin_kind` as provenance only. It must not grant write lanes.
- Deterministic code owns target construction, target validation, write-lane
  permission, scheduler materialization, cache invalidation, and persistence.
- Target construction is deterministic. LLM output must not create
  `target_kind`, target ids, write lanes, database operations, or group targets.
- Participant presence in a message window is not enough to create a user
  write.
- Group image routing is deterministic for group-scoped consolidation.
- Do not let group-channel consolidation write `affinity`,
  `last_relationship_insight`, or `user_memory_units`.
- Do not let character-self consolidation write user affinity or user memory.
- Do not change L1, L2, L2d, L3, dialog wording, adapters, or delivery behavior
  unless an implementation blocker proves the target contract cannot be wired
  without a bounded change.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Use PowerShell `-LiteralPath` for filesystem paths that may contain spaces.
- Do not read `.env`.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major checklist stage, reread this entire plan before
  starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.

## Must Do

- Move the core consolidator entrypoint into a first-class
  `kazusa_ai_chatbot.consolidation` package.
- Add a deterministic `ConsolidationTargetPlan` contract.
- Build eligible targets before any durable consolidation write.
- Preserve valid current user-message consolidation behavior.
- Validate every internal write intent against the target plan.
- Remove the missing-`source_user_id` fallback to `"self_cognition"`.
- Add a deterministic group-channel target path for group-scoped consolidation.
- Add read-only diagnostics and a dry-run-first cleanup path for malformed
  synthetic-user rows.
- Add focused tests and static checks for target planning, forbidden synthetic
  users, write-lane validation, and group/user/character separation.

## Deferred

- Do not build a generic memory-provider abstraction.
- Do not add a trigger-source registry beyond fields required for target
  planning.
- Do not add group affinity.
- Do not store group facts in `user_memory_units`.
- Do not add new external tools, image generation, motor surfaces, or adapter
  behavior.
- Do not redesign group reflection source selection.
- Do not backfill group images from historical chat without a separate approved
  migration plan.
- Do not perform production data mutation from implementation tests.

## Cutover Policy

Overall strategy: staged bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Consolidation package entrypoint | bigbang | Runtime callers import the core entrypoint from `kazusa_ai_chatbot.consolidation.core`. Do not preserve the node module as the public entrypoint. |
| Target plan construction | bigbang | Every consolidation run receives a deterministic target plan before durable writes. |
| Write-intent validation | bigbang | Every proposed write intent is validated against target alias, target kind, and allowed lane before persistence. |
| Existing valid user-message writes | compatible | Preserve current behavior for real validated user targets. |
| Group target behavior | bigbang | Group-scoped consolidation receives a deterministic group-channel target. The LLM does not output group targets. |
| Missing durable target | bigbang | Missing or ambiguous durable target produces `internal` or no write, never a fake user. |
| Existing malformed rows | migration | Clean only through approved dry-run/apply operator flow after code prevents re-creation. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- For `bigbang` areas, delete or rewrite legacy references instead of adding
  fallback compatibility paths.
- For `compatible` areas, preserve only the current valid user-message behavior
  listed in this plan.
- For `migration` areas, follow the exact diagnostics and cleanup gates in this
  plan.
- Any change to cutover policy requires user approval before implementation.

## Overdesign Guardrail

- Actual problem: generic cognition sources can reach a user-shaped
  consolidator even when the durable target is a group channel, character-self
  state, internal artifact, or no durable entity.
- Minimal change: create a consolidation package boundary, add deterministic
  target planning, and validate proposed writes before persistence.
- Ownership boundaries: deterministic code builds targets and allowed lanes;
  database helpers persist already-validated writes and remain fail-fast.
- Rejected complexity: no DB defaulting, fake users, prompt-built targets,
  group affinity, memory-provider registry, trigger registry, broad prompt
  rewrite, retry loop, feature flag, compatibility shim, or L1/L2/L3 redesign.
- Evidence threshold: add a broader provider abstraction, new group-image
  collection, or larger prompt contract only after a separate approved plan
  proves deterministic target planning cannot represent required behavior.

## Agent Autonomy Boundaries

- The agent may choose local helper names only when the public contracts in
  this plan remain intact.
- The agent must not let the LLM create targets, target ids, target kinds,
  write lanes, group targets, or persistence operations.
- The agent must not add alternate execution paths, fallback users, feature
  flags, compatibility shims, or unrelated cleanup.
- Changes outside the listed Change Surface require a plan revision or explicit
  user approval.
- If code and plan disagree, preserve the plan's stated intent and record the
  discrepancy before expanding scope.

## Target State

```text
cognitive episode or self-cognition case
  -> kazusa_ai_chatbot.consolidation.core
  -> deterministic ConsolidationTargetPlan
  -> existing extraction
  -> deterministic write-intent projection
  -> deterministic validation
  -> target-specific persistence
```

`nodes` remains the live persona graph owner. `kazusa_ai_chatbot.consolidation`
owns post-turn/background consolidation entrypoint, target planning,
write-intent validation, and persistence dispatch.

Minimum target mappings:

| Source case | Deterministic base target |
|---|---|
| Normal `/chat` with validated real author | `user` for the current author |
| Group-scoped `/chat` consolidation | `group_channel` plus current author user target when valid |
| User-scoped active commitment | `user` for the validated commitment owner |
| Group review | `group_channel` |
| Scheduled future cognition from real user source | `user` |
| Scheduled future cognition from group source | `group_channel` |
| Character self-check | `character` |
| No validated durable target | `internal` or no write |

Allowed write lanes:

| `target_kind` | Allowed write lanes |
|---|---|
| `user` | `relationship_insight`, `user_memory_units`, `affinity`, `user_style_image` |
| `group_channel` | `group_channel_style_image` |
| `character` | `character_state`, `character_self_image` |
| `internal` | audit/local artifact only |

## Design Decisions

| Topic | Decision | Why |
|---|---|---|
| DB crash behavior | Keep user-profile readers fail-fast. | The crash revealed malformed lifecycle data. Defaulting would hide the root cause. |
| Origin versus target | Keep `origin_kind` separate from `target_kind`. | A source such as self-cognition can reason about a user, group, character, or internal artifact. |
| Group path | Build group-channel target deterministically for group-scoped consolidation. | A group has group image/state, not affinity or user memory. This does not need LLM judgment. |
| Core module boundary | Move `call_consolidation_subgraph(...)` to `kazusa_ai_chatbot.consolidation.core`. | Consolidation is a subsystem boundary, not a persona-node implementation detail. |

## Contracts And Data Shapes

Deterministic target plan:

```python
class ConsolidationTarget(TypedDict):
    target_alias: str
    target_kind: Literal["user", "group_channel", "character", "internal"]
    target_id: dict[str, str]
    write_lanes: list[str]


class ConsolidationTargetPlan(TypedDict):
    origin_kind: str
    targets: list[ConsolidationTarget]
```

Internal write intent, projected by deterministic code from extraction output
and validated target context:

```python
class ConsolidationWriteIntent(TypedDict):
    target_alias: str
    write_lane: str
    payload: dict
```

Validation requirements:

- `target_alias` must exist in the target plan.
- `write_lane` must be listed for that target.
- `user` targets must resolve to real validated user profiles.
- `group_channel` targets must not call user-profile or user-memory helpers.
- `character` targets must not call user-profile or user-memory helpers.
- `internal` targets must not persist durable user, group, or character writes.
- Synthetic user ids, including `"self_cognition"`, are rejected before DB
  helper calls.

## LLM Call And Context Budget

- Target-plan construction adds zero LLM calls.
- Existing consolidation extraction remains a background post-response path.
- This plan adds no new LLM call, prompt, retry loop, or real-data LLM
  quality-review gate.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/consolidation/__init__.py`
  - package export for the consolidation subsystem.
- `src/kazusa_ai_chatbot/consolidation/README.md`
  - subsystem ICD for target planning, write validation, and forbidden
    synthetic user targets.
- `src/kazusa_ai_chatbot/consolidation/core.py`
  - new public home for `call_consolidation_subgraph(...)`.
- `src/kazusa_ai_chatbot/consolidation/target.py`
  - target types, target-plan construction, and write-intent validation.
- `src/kazusa_ai_chatbot/consolidation/group_channel.py`
  - group-channel projection and persistence dispatch for allowed group lanes.
- `tests/test_consolidation_target_routing.py`
  - target-plan and forbidden synthetic-user tests.
- `tests/test_consolidator_group_channel_branch.py`
  - group-channel write-lane and negative user-lane tests.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
  - move the core entrypoint and graph assembly into `consolidation/core.py`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`
  - add target-plan state only if shared state requires an explicit field.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - validate write intents before DB helper calls.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
  - keep origin policy as provenance; do not grant target lanes from origin.
- `src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py`
  - remove missing-user fallback to `"self_cognition"`.
- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - preserve group scope and carried user metadata without fabricating a user.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - pass target-plan metadata into consolidation state.
- `src/kazusa_ai_chatbot/service.py`
  - import the consolidator entrypoint from `kazusa_ai_chatbot.consolidation.core`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - update consolidation imports if needed.
- Existing direct-import tests for consolidator origin, efficiency, and service
  background consolidation
  - update imports to the new public package path.
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - document target-kind semantics and forbidden synthetic user identity.
- `src/kazusa_ai_chatbot/nodes/README.md`
  - document that nodes produce consolidation input while the consolidation
    package owns durable write routing.
- `src/kazusa_ai_chatbot/db/README.md`
  - clarify that `user_profiles` is user-only and group-channel style image is
    the first group durable lane.

### Keep

- `src/kazusa_ai_chatbot/db/users.py`
  - keep fail-fast behavior for malformed user profiles.
- `src/kazusa_ai_chatbot/db/interaction_style_images.py`
  - reuse existing group-channel style-image storage for the first group lane.
- Existing consolidation helper-node modules for facts, images, memory units,
  origin, and reflection
  - keep in place unless an import update is mechanically required.
- L1, L2, L2d, L3, dialog, RAG, dispatcher, and adapter implementations
  - no behavioral redesign in this plan.

### Operator Or Migration Surface

- Add a maintenance script or `db.script_operations` helper only after target
  prevention is implemented and tested.
- The cleanup helper must support dry-run and apply modes.
- Apply mode requires explicit user approval after dry-run evidence is reviewed.

## Data Migration

No production mutation is approved by this implementation plan. Apply mode
requires separate explicit approval after dry-run evidence is reviewed.

Required dry-run diagnostics:

- count `user_profiles` where `global_user_id == "self_cognition"`;
- count `user_profiles` missing `affinity`;
- count `scheduled_events` where `source_user_id == "self_cognition"`;
- count `user_memory_units` where `global_user_id == "self_cognition"`;
- count affected self-cognition action attempts whose future-cognition target
  scope lacks a real source user;
- verify no real platform account is linked to `global_user_id="self_cognition"`.

Approved apply behavior, after user approval:

- fail synthetic scheduled events with a migration reason and remove their
  synthetic source-user ownership;
- remove or quarantine malformed synthetic user-profile and memory rows;
- leave real UUID-like users untouched;
- rerun diagnostics and record zero remaining synthetic user-owned rows.

## Implementation Order

1. Complete plan review and owner approval.
2. Run read-only production diagnostics and record sanitized counts.
3. Add focused failing tests for target-plan construction and forbidden
   synthetic user targets.
4. Create `kazusa_ai_chatbot.consolidation` and move
   `call_consolidation_subgraph(...)` to `consolidation/core.py`.
5. Implement `consolidation/target.py` with deterministic target construction
   and write-intent validation.
6. Remove future-cognition fallback from missing user id to `"self_cognition"`.
7. Wire target plan into consolidation state before durable writes.
8. Connect group-channel write support through existing interaction-style image
   storage.
9. Connect character-self target lanes without user-profile dependency.
10. Update subsystem READMEs and direct-import tests.
11. Add migration dry-run helper; do not implement apply mode until dry-run
    output is reviewed.
12. Run verification.
13. Run independent code review and fix in-scope findings.
14. Run production cleanup apply only after explicit user approval.

## Progress Checklist

- [x] Stage 1 - plan approval gate complete
  - Covers: Implementation Order step 1.
  - Verify: independent plan review has no remaining blockers and owner approves
    the draft for execution.
  - Evidence: record approval status in Execution Evidence.
  - Sign-off: `Codex/2026-05-21`.

- [x] Stage 2 - read-only diagnostics complete
  - Covers: step 2.
  - Verify: diagnostic counts recorded without production mutation.
  - Evidence: record sanitized counts only.
  - Sign-off: `Codex/2026-05-21`.

- [x] Stage 3 - target contract and package boundary complete
  - Covers: steps 3-5.
  - Verify: target routing focused tests pass and consolidator imports resolve
    from `kazusa_ai_chatbot.consolidation.core`.
  - Evidence: record red/green focused test output and import grep output.
  - Sign-off: `Codex/2026-05-21`.

- [x] Stage 4 - synthetic identity fallback removed
  - Covers: step 6.
  - Verify: tests prove missing user ids do not become `"self_cognition"` user
    targets.
  - Evidence: record focused test output.
  - Sign-off: `Codex/2026-05-21`.

- [x] Stage 5 - validation and target lanes wired
  - Covers: steps 7-9.
  - Verify: invalid aliases, invalid lanes, group-to-user writes,
    character-to-user writes, and synthetic user targets fail before DB helper
    calls.
  - Evidence: record tests and static grep results.
  - Sign-off: `Codex/2026-05-21`.

- [x] Stage 6 - docs and migration dry-run complete
  - Covers: steps 10-11.
  - Verify: subsystem READMEs match the implemented target contract and dry-run
    cleanup reports exact planned mutation without applying it.
  - Evidence: record docs diff summary and dry-run output path/counts.
  - Sign-off: `Codex/2026-05-21`.

- [x] Stage 7 - full verification complete
  - Covers: step 12.
  - Verify: every Verification command passes or has an approved blocker.
  - Evidence: record command outputs.
  - Sign-off: `Codex/2026-05-21`.

- [x] Stage 8 - independent code review complete
  - Covers: step 13.
  - Verify: review findings are closed or explicitly accepted as residual risk.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, and final
    approval status.
  - Sign-off: `Codex/2026-05-21`.

- [x] Stage 9 - approved production cleanup complete
  - Covers: step 14.
  - Verify: apply mode runs only after explicit user approval and post-cleanup
    diagnostics show no synthetic user-owned rows.
  - Evidence: record sanitized before/after counts.
  - Sign-off: `Codex/2026-05-21`.

## Verification

Run from repository root with the project virtual environment.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\consolidation\core.py src\kazusa_ai_chatbot\consolidation\target.py src\kazusa_ai_chatbot\consolidation\group_channel.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\action_spec\handlers\future_cognition.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\runner.py`

### Static Greps

- `rg -n "or \"self_cognition\"|or 'self_cognition'|source_user_id.*self_cognition|global_user_id.*self_cognition" src\kazusa_ai_chatbot\action_spec src\kazusa_ai_chatbot\self_cognition src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\consolidation`

  Expected: no match where `"self_cognition"` is used as a fallback user id or
  user-profile owner. Matches are allowed only for event names, artifact ids,
  case ids, source labels, or log component names.

- `rg -n "from kazusa_ai_chatbot\.nodes\.persona_supervisor2_consolidator import call_consolidation_subgraph|import kazusa_ai_chatbot\.nodes\.persona_supervisor2_consolidator" src tests`

  Expected: no matches. Callers must import the core entrypoint from
  `kazusa_ai_chatbot.consolidation.core`.

- `rg -n "update_affinity|update_last_relationship_insight|update_user_memory_units_from_state" src\kazusa_ai_chatbot\consolidation\group_channel.py`

  Expected: no matches. Exit code `1` is acceptable.

- `rg -n "CONSOLIDATION_LLM|llm|prompt|ainvoke|invoke" src\kazusa_ai_chatbot\consolidation\target.py`

  Expected: no matches that perform LLM or prompt-based target construction.

- `git diff --check`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_consolidation_target_routing.py -q`
- `venv\Scripts\python -m pytest tests\test_consolidator_group_channel_branch.py -q`
- `venv\Scripts\python -m pytest tests\test_action_spec_future_cognition.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py -q`

Required assertions:

- target planning does not call an LLM client;
- missing user id never becomes `"self_cognition"`;
- group review always has deterministic group-channel eligibility;
- group-channel writes cannot use user lanes;
- character-self writes cannot use user lanes.

### Adjacent Tests

- `venv\Scripts\python -m pytest tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py -q`
- `venv\Scripts\python -m pytest tests\test_interaction_style_images.py tests\test_cognition_interaction_style_context.py -q`
- `venv\Scripts\python -m pytest tests\test_consolidator_character_image.py tests\test_service_background_consolidation.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py tests\test_self_cognition_tracking.py -q`

### Live DB Diagnostics

Use only when MongoDB is available and the user authorizes production
inspection:

- Count malformed user profiles missing `affinity`.
- Count rows where `global_user_id == "self_cognition"` in user-owned
  collections.
- Count scheduled future-cognition events where `source_user_id ==
  "self_cognition"`.
- After cleanup, expected result is zero synthetic user-owned rows.

Do not print raw message bodies, raw reflection text, or secrets.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the active agent
must reread this plan, the development-plan registry, README, HOWTO,
self-cognition README, nodes README, brain-service README, and DB README from
a fresh-review posture.

Review scope:

- The plan addresses target lifecycle and write validation, not only the crash.
- `origin_kind` and `target_kind` stay separate.
- Group-channel target creation is deterministic and outside LLM output.
- LLM output cannot create targets, target kinds, write lanes, or DB
  operations.
- Synthetic `self_cognition` user identity is removed without weakening
  fail-fast profile invariants.
- The core consolidator entrypoint moves to a consolidation package.
- The plan does not expand L1/L2/L3/dialog, adapters, dispatcher, or RAG.
- Migration cleanup is dry-run first and cannot touch real users.
- Verification gates map to the risks above.

### 2026-05-21 Review Result

Reviewer mode: active agent fresh-review; no separate reviewer was available.

Inputs reviewed: `git status --short`, `README.md`, `docs/HOWTO.md`,
`development_plans/README.md`, `src/kazusa_ai_chatbot/self_cognition/README.md`,
`src/kazusa_ai_chatbot/nodes/README.md`,
`src/kazusa_ai_chatbot/brain_service/README.md`, and
`src/kazusa_ai_chatbot/db/README.md`.

Findings fixed in this revision:

- Blocker: the previous plan allowed target construction paths that could
  recreate fake users. The plan now requires deterministic target construction
  and write-intent validation before persistence.
- Non-blocking: the previous plan carried trend projection and broad future
  direction not needed for execution. This revision removed that material and
  retained only decision rationale.

Approval status: no remaining plan-readiness blockers found in this review.
The owner approved execution on 2026-05-21, and the plan is now `in_progress`.

## Independent Code Review

Run this gate after all Verification commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project style, prompt safety, CJK safety, test style, and path-safe command
  compliance.
- Alignment with Must Do, Deferred, target contracts, cutover policy, change
  surface, verification, and acceptance criteria.
- Design risk: hidden fallback users, prompt metadata leaks, fake identities,
  invalid lanes, group-to-user writes, and accidental L3/dialog changes.
- Regression coverage: focused target-plan tests, future-cognition tests,
  group-channel tests, character-self tests, and sanitized DB diagnostics.

Fix findings directly only when the fix is inside this plan's approved change
surface. If a finding requires a broader contract or new architecture, stop and
revise the plan before changing code.

## Acceptance Criteria

This plan is complete when:

- The public consolidator entrypoint is
  `kazusa_ai_chatbot.consolidation.core.call_consolidation_subgraph`.
- No runtime caller imports `call_consolidation_subgraph` from the old node
  module.
- Every consolidation run has an explicit deterministic
  `ConsolidationTargetPlan`.
- Durable writes are materialized as internal write intents only after
  deterministic lane projection and validation.
- Deterministic validation rejects unknown aliases, forbidden lanes, fake user
  ids, and unvalidated user targets before DB writes.
- Valid current user-message consolidation behavior is preserved.
- Group-scoped consolidation has deterministic group-channel eligibility.
- Group-channel intents write only group-channel state.
- Character-self consolidation targets character-owned state only.
- Missing durable targets no longer become `global_user_id="self_cognition"`.
- User-profile helpers remain fail-fast for malformed rows.
- Approved cleanup removes or quarantines existing synthetic user-owned rows
  only after code prevents re-creation.
- All Verification gates pass or have owner-approved blockers.
- Independent code review approves the implementation.
- Registry and plan lifecycle records follow `development_plans/README.md`.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Fake user semantics return through another path | Forbid synthetic ids and add static greps plus target-plan tests | Static Greps and focused tests |
| Group-channel writes call user helpers | Validate target kind and lane before persistence | Group-channel negative tests |
| Existing valid user writes regress | Preserve compatible user lane and run adjacent consolidator tests | Adjacent tests |
| Core package move becomes broad helper migration | Move only public entrypoint and target modules | Import greps and py_compile |
| Cleanup touches real users | Dry-run first, exact synthetic filters, no linked platform accounts | Live DB diagnostics |

## Execution Evidence

- 2026-05-21: Owner approved execution by requesting implementation start.
  Status changed to `in_progress`; Stage 1 signed off by Codex.
- 2026-05-21: Read-only lifecycle diagnostic dry-run wrote
  `test_artifacts\consolidation_target_lifecycle_dry_run.json`.
  Sanitized counts: `synthetic_user_profiles=1`,
  `user_profiles_missing_affinity=1`, `synthetic_scheduled_events=3`,
  `synthetic_user_memory_units=1`,
  `future_cognition_attempts_missing_user=3`,
  `synthetic_user_profiles_with_platform_accounts=0`. No apply mode ran.
- 2026-05-21: Production-data dry-run rerun wrote
  `test_artifacts\consolidation_target_lifecycle_production_dry_run_20260521_214920.json`
  with the same sanitized counts. Field-limited detail exports confirmed the
  synthetic scheduled events are `running`, not `pending`; the synthetic
  active commitment has no matching conversation-history rows; and cleanup is
  not blocked by linked platform accounts.
- 2026-05-21: Moved the public consolidator entrypoint to
  `src\kazusa_ai_chatbot\consolidation\core.py`, added deterministic target
  validation in `consolidation\target.py`, and retired the old node module as
  a non-entrypoint module.
- 2026-05-21: Removed future-cognition missing-user fallback to
  `"self_cognition"`. Scheduled real-user cases now read the real
  `user_profile`; missing profiles remain empty and fail validation instead of
  receiving a dry-run default.
- 2026-05-21: Wired explicit `consolidation_target_plan` into consolidator
  state and required `db_writer` to read it with plain indexing.
- 2026-05-21: Added group-channel write validation and persistence dispatch via
  `consolidation\group_channel.py`; group review with a group target no longer
  receives an accidental character target.
- 2026-05-21: Strengthened real user-target validation so
  `user_profile.global_user_id` must match the selected `global_user_id`.
- 2026-05-21: Scope correction narrowed this plan to deterministic target
  construction, write-intent validation, group-channel separation, and
  synthetic-user cleanup only.
- 2026-05-21: Independent code review found four issues: group-channel
  persistence was declared but not called; `origin_kind` still allowed group
  internal-thought cases to update character lanes; real-user validation was
  shape-only; and the plan still contained stale draft text. Fixes were made
  in scope and rerun through focused and adjacent verification.
- 2026-05-21: Final independent code review reported no blocking findings.
  Residual risks recorded: group-channel style-image extraction is not yet
  end-to-end beyond direct persistence wiring, some adjacent service tests
  connect to the configured MongoDB.
- 2026-05-21: Verification passed:
  `py_compile` over touched consolidation, persistence, self-cognition,
  service, action-spec, and diagnostic modules; `git diff --check` with only
  LF/CRLF warnings; static greps for old consolidator imports, synthetic
  `self_cognition` user fallbacks, user DB helpers in group-channel module,
  and LLM dependency in target planning.
- 2026-05-21: Focused and adjacent tests passed:
  `tests\test_consolidation_target_routing.py`,
  `tests\test_consolidator_group_channel_branch.py`,
  `tests\test_action_spec_future_cognition.py`,
  `tests\test_self_cognition_integration.py`,
  `tests\test_self_cognition_delivery_target.py`,
  `tests\test_consolidation_lifecycle_diagnostics.py`,
  `tests\test_consolidation_origin_policy.py`,
  `tests\test_consolidator_origin_policy_db_writer.py`,
  `tests\test_db_writer_cache2_invalidation.py`,
  `tests\test_consolidator_origin_selection.py`,
  `tests\test_consolidation_origin_metadata.py`,
  `tests\test_consolidator_efficiency.py`,
  `tests\test_interaction_style_images.py`,
  `tests\test_cognition_interaction_style_context.py`,
  `tests\test_consolidator_character_image.py`,
  `tests\test_service_background_consolidation.py`,
  `tests\test_self_cognition_group_review_source.py`, and
  `tests\test_self_cognition_tracking.py`.
- 2026-05-21: Implemented approved cleanup apply support in
  `db.script_operations` and
  `scripts.inspect_consolidation_target_lifecycle --apply`. Apply mode blocks
  when the synthetic profile has linked platform accounts, fails synthetic
  scheduled events with migration metadata, removes the synthetic
  `source_user_id`, deletes exact synthetic profile and user-memory rows, and
  reports sanitized before/after counts. Verification passed:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\db\script_operations.py src\scripts\inspect_consolidation_target_lifecycle.py tests\test_consolidation_lifecycle_diagnostics.py`,
  `venv\Scripts\python -m pytest tests\test_consolidation_lifecycle_diagnostics.py -q`,
  `venv\Scripts\python -m pytest tests\test_script_db_boundary.py -q`,
  `venv\Scripts\python -m scripts.inspect_consolidation_target_lifecycle --help`,
  and scoped `git diff --check` with only LF/CRLF warnings.
- 2026-05-21: Owner explicitly approved Stage 9 production cleanup. Fresh
  pre-apply diagnostic wrote
  `test_artifacts\consolidation_target_lifecycle_stage9_before_20260521_225632.json`.
  Sanitized counts before apply: `synthetic_user_profiles=1`,
  `user_profiles_missing_affinity=1`, `synthetic_scheduled_events=3`,
  `synthetic_user_memory_units=1`,
  `future_cognition_attempts_missing_user=3`,
  `synthetic_user_profiles_with_platform_accounts=0`.
  Approved apply wrote
  `test_artifacts\consolidation_target_lifecycle_stage9_apply_20260521_225632.json`
  with `apply_status=applied` and `synthetic_user_owned_rows_after=0`.
  Post-apply diagnostic wrote
  `test_artifacts\consolidation_target_lifecycle_stage9_after_20260521_225632.json`.
  Sanitized counts after apply: `synthetic_user_profiles=0`,
  `user_profiles_missing_affinity=0`, `synthetic_scheduled_events=0`,
  `synthetic_user_memory_units=0`,
  `future_cognition_attempts_missing_user=3`,
  `synthetic_user_profiles_with_platform_accounts=0`.
- 2026-05-21: Post-Stage-9 independent code review found no blocking or
  important issues. Follow-up fixes normalized scheduled synthetic
  `source_user_id` before profile lookup, target construction, and delivery
  binding; kept targetless scheduled future-cognition `user_profile` empty;
  removed the dormant optional Stage-6 routing API surface; and updated stale
  self-cognition docs. Residual risks accepted: the entire repo test suite was
  not rerun, and some adjacent service tests connect to the configured MongoDB.
  Final verification passed: `py_compile` over touched files; 46
  future/self-cognition target tests; 15 lifecycle/target-routing tests; 38
  consolidation origin/persistence/cache tests; 97 service/image/tracking
  adjacent tests; `git diff --check`; and static greps for synthetic
  `self_cognition` user fallbacks, old consolidator imports, group-channel
  user-profile helper use, and removed optional routing API names.

## Glossary

- `origin_kind`: why cognition ran.
- `target_kind`: durable entity a validated write may modify: user,
  group channel, character, or internal artifact.
- `target_alias`: prompt-safe handle for one deterministic target.
- `write_intent`: internal projected change naming a target alias and write
  lane after deterministic validation.
- `group image`: group-scoped durable state; first slice uses existing
  group-channel interaction-style image storage.
- `synthetic user`: any non-user label, including `"self_cognition"`, used as
  a `global_user_id`. This is forbidden.
