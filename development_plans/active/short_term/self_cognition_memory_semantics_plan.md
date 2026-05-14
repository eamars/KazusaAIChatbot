# self cognition memory semantics plan

## Summary

- Goal: Treat self-cognition as an `internal_thought` input trigger that
  reuses the same cognition, private finalization, consolidation, persistence,
  scheduler, and cache-invalidation infrastructure used after user input, while
  keeping adapter delivery and conversation-history writes out of the internal
  path.
- Plan class: large
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`,
  `cjk-safety` when editing Python files containing CJK strings
- Overall cutover strategy: compatible
- Highest-risk areas: preserving live-chat behavior, preventing a parallel
  memory writer, keeping origin/source identity clear for internal-thought
  evidence, and avoiding accidental adapter delivery from private cognition.
- Acceptance criteria: implementation is complete only when self-cognition can
  enter the existing consolidation writer lanes through `internal_thought`
  origin metadata, no new database collection or sidecar persistence path is
  introduced, all verification gates pass, and independent code review approves
  the result.

## Context

Self-cognition already builds background `CognitiveEpisode` values with
`trigger_source=internal_thought`, `input_sources=["internal_monologue"]`, and
`output_mode=preview`. The current dry-run path runs shared RAG/cognition and
may build an action candidate, but production self-cognition does not call the
existing post-dialog consolidation subgraph.

The previous version of this plan incorrectly narrowed self-cognition to one
new `conversation_progress` effect and denied relationship, mood, affinity,
memory, task, image, and cache effects. That was not the intended product
direction. The approved direction is that self-cognition follows the same path
as user input: it is a normal internal trigger for the character brain, not a
separate low-power reflection artifact.

This plan updates the target design:

- Reuse the existing consolidation subgraph and `db_writer` lanes.
- Reuse the existing MongoDB collections and DB facades.
- Do not create a new `conversation_progress` writer, memory module, DB helper,
  migration, or collection.
- Allow repeated internal rumination over unresolved relationship events.
- Do not add intentional outbound-spam mitigation in this plan. User-facing
  repetition control is recorded as long-term design observation only.

## Mandatory Skills

- `development-plan-writing`: preserve this approved work contract and record
  execution evidence before lifecycle changes.
- `local-llm-architecture`: keep background LLM work bounded, source-aware,
  and compatible with local/weaker model behavior.
- `no-prepost-user-input`: do not add deterministic semantic filters that
  rewrite, drop, or reclassify LLM-chosen facts, commitments, preferences, or
  relationship effects.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files that contain CJK prompt text
  or adding CJK string literals to Python.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Keep the target boundary to `self_cognition`, consolidation origin/policy,
  the existing consolidation subgraph, and focused tests.
- Do not add a new MongoDB collection, migration, DB facade helper, persistent
  storage service, or sidecar memory orchestrator.
- Do not create `src/kazusa_ai_chatbot/self_cognition/memory.py`.
- Do not add a `conversation_progress` lane to `db_writer`.
- Do not add a deterministic self-cognition conversation-progress writer in
  this plan.
- Do not add intentional self-cognition spam prevention, rate limiting,
  cooldown, topic suppression, outbound message variation, or new idempotency
  logic in this plan. Existing dispatcher/scheduler validation and existing
  `self_cognition_action_attempts` behavior remain unchanged.
- Do not suppress repeated internal rumination over unresolved relationship
  events. Repeated internal cognition is an intended capability.
- Do not save self-cognition private finalization text as a visible
  conversation-history assistant row.
- Do not deliver self-cognition private finalization through adapters unless
  existing scheduler/dispatcher infrastructure independently schedules an
  outward action.
- Do not reintroduce `no_remember` into self-cognition-created state.
- Do not add new response-path LLM calls. All new consolidation work remains
  background-only.
- Do not add code-side semantic filtering, keyword classification, or
  post-LLM channel rewriting over user-authored input or internal-thought
  evidence. Structural validation, schema checks, enum checks, timestamps,
  identifiers, and origin metadata projection are allowed.
- Update prompt-facing payloads only to make source identity explicit and to
  avoid falsely describing internal-thought input as user-authored speech.
  Do not create separate self-cognition-only consolidator prompts.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in
  `Execution Evidence`.

## Must Do

- Add `build_self_cognition_consolidation_origin(...)` beside
  `build_user_message_consolidation_origin(...)` in
  `persona_supervisor2_consolidator_origin.py`.
- Update `call_consolidation_subgraph(...)` so it selects the user-message or
  self-cognition origin builder from the existing `CognitiveEpisode` trigger
  source and still rejects unsupported origins.
- Extend `build_consolidation_write_policy(...)` so supported
  `internal_thought` preview origins allow the same seven existing
  `db_writer` lanes as supported user-message origins:
  `character_state`, `relationship_insight`, `user_memory_units`,
  `task_dispatch`, `affinity`, `character_image`, and `cache_invalidation`.
- Preserve the existing `db_writer` lane implementations. Do not fork,
  duplicate, or reimplement the writer for self-cognition.
- Make consolidation prompt payloads source-aware by exposing concise
  `consolidation_origin` fields where the prompt currently needs to know
  whether the input came from user speech or internal thought.
- Adjust prompt wording in existing consolidator prompts only as needed so
  `decontexualized_input` can mean either a user turn or an internal-thought
  trigger. Keep one shared prompt per existing consolidator stage.
- Wire self-cognition runner/worker production paths to merge shared cognition
  output into a consolidation-ready state, produce private finalization through
  the existing dialog/finalization path when needed by consolidation, and call
  the existing consolidation subgraph through an injectable seam.
- Preserve dry-run artifact writing for local runs and production no-local-file
  behavior for worker runs.
- Record sanitized self-cognition event metadata for consolidation outcomes
  without raw source packet text, raw conversation bodies, private
  finalization text, generated action-candidate text, or raw DB documents.
- Add or update deterministic tests for origin validation, policy decisions,
  shared consolidator entry, source-aware payloads, self-cognition production
  consolidation wiring, no conversation-history write, no adapter delivery, no
  new collection, and sanitized event logging.
- Update `development_plans/long_term/todo.md` with the outbound-repetition
  observation captured by this plan.

## Deferred

- Do not implement self-cognition spam prevention, repeated-topic suppression,
  cooldowns, message variation, or new outbound idempotency in this plan.
- Do not redesign live-chat conversation-progress recording.
- Do not add self-cognition conversation-progress persistence.
- Do not redesign live-chat consolidation semantics beyond making the existing
  consolidator origin-aware.
- Do not migrate historical self-cognition artifacts.
- Do not change reflection promotion semantics.
- Do not add a new autonomous-contact permission system.
- Do not add adapter changes, service endpoints, operator auth, config flags,
  environment flags, or feature toggles.
- Do not add a new memory, relationship, affinity, or scheduler database.
- Do not remove existing dispatcher/scheduler validation.

## Long-Term Observation

Self-cognition is allowed to revisit the same unresolved relationship event
over time. That repeated rumination can intentionally deepen mood, affinity,
relationship stance, possessiveness, or task priority. The architecture
concern is user-facing repetition, not internal repetition.

Future proactive-contact design should govern whether repeated internal
rumination produces repeated outbound messages, and whether outward messages
need rate limits, variation, quiet hours, or user-facing controls. This plan
does not implement that mitigation.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Origin metadata | compatible | Add an `internal_thought` origin builder beside the existing user-message builder. Preserve user-message validation. |
| Consolidation entry | compatible | Route supported user-message and internal-thought episodes into the same existing consolidation subgraph. Unsupported origins still fail closed. |
| Write policy | compatible | Allow the same existing seven `db_writer` lanes for supported internal-thought preview origins. Preserve user-message lane behavior. |
| `db_writer` | compatible | Reuse the existing writer and persistence facades. Do not add self-cognition-specific branches except origin-policy checks already used by the writer. |
| Self-cognition worker | compatible | Add consolidation application after cognition/private finalization. Preserve existing action-candidate and dispatcher behavior. |
| Conversation history | compatible | Keep internal finalization private. Do not save it as an assistant chat row. |
| Database | compatible | Reuse existing collections and indexes. No migration, backfill, or new collection is approved. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- If a verification failure suggests a different cutover strategy is needed,
  stop and update the plan before implementation continues.
- Any change to this policy requires user approval before implementation.

## Data Migration

No data migration is approved or required.

- Reuse existing collections such as `character_state`, `user_profiles`,
  `user_memory_units`, `scheduled_events`, `self_cognition_action_attempts`,
  and existing cache invalidation infrastructure.
- Do not backfill historical self-cognition artifacts.
- Do not create or bootstrap a self-cognition memory/progress collection.

## Agent Autonomy Boundaries

- The agent may choose local helper names and assertion ordering when the
  public contracts in this plan remain unchanged.
- The agent must not invent additional storage paths, prompt families,
  mitigation rules, feature flags, fallback paths, compatibility shims, or
  unrelated cleanup.
- The agent must reuse existing writer, scheduler, cache invalidation, event
  logging, and dry-run artifact infrastructure.
- The agent must not reinterpret LLM output semantically in deterministic code.
  If source confusion appears, fix the shared prompt contract or payload shape.
- The agent may add small structural helpers only for origin projection,
  payload projection, test seams, and sanitized event metadata.
- If existing helper behavior already satisfies a needed projection,
  validation, cache, repository, or scheduler contract, reuse it instead of
  duplicating storage logic.
- If the plan and code disagree, preserve the plan's stated same-path
  architecture and record the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

Self-cognition production runs can produce an internal-thought episode, execute
the existing shared cognition graph, produce a private finalization state when
the consolidation graph needs `final_dialog`, and then call the existing
consolidation subgraph.

The consolidation subgraph uses `build_self_cognition_consolidation_origin(...)`
for supported `internal_thought` preview episodes. The existing `db_writer`
then applies the same seven lane decisions used by user-message origins:

- character mood/vibe/reflection state
- last relationship insight
- user memory units
- future task dispatch
- affinity delta
- character image
- cache invalidation

The internal path does not save a visible assistant message and does not call
adapters directly. Any later user-facing action remains owned by existing
scheduler/dispatcher infrastructure. Repeated internal rumination is allowed;
this plan adds no new outbound-spam mitigation.

## Per-Lane Decision Table For `internal_thought`

| Lane | Decision | Reason |
|---|---|---|
| `character_state` | Allow | Internal cognition is allowed to change mood, global vibe, and reflection summary. |
| `relationship_insight` | Allow | The character may form relationship conclusions from remembered evidence and unresolved events. |
| `user_memory_units` | Allow | Internal-thought evidence may update existing user-memory-unit semantics through the same source-aware shared consolidator path. |
| `task_dispatch` | Allow | Internal cognition may produce future obligations through the same scheduler path used by accepted user-turn commitments. |
| `affinity` | Allow | Relationship appraisal from self-cognition may change affinity through the existing scaled affinity updater. |
| `character_image` | Allow | The same writer lane remains available. Current self-cognition debug modes may prevent visual directives from producing changes. |
| `cache_invalidation` | Allow | Existing cache invalidation must run after successful durable writes. |

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Primary architecture | Same-path consolidation | Self-cognition is a real internal trigger for the character brain, not a sidecar preview writer. |
| Storage | Existing infrastructure only | The user explicitly rejected additional database/storage infrastructure. |
| Write policy | Allow seven existing lanes for supported `internal_thought` | This gives self-cognition autonomy over mood, relationship, memory, task, affinity, image, and cache effects. |
| Conversation history | Do not write private finalization rows | Internal thought should affect state without pretending that a visible chat message occurred. |
| Conversation progress | No new progress writer | The prior progress-only design was the wrong direction and created a side path. |
| Prompt contract | Shared prompts with source-aware payloads | The model needs origin identity, but separate prompt families would drift from the same-path goal. |
| Spam prevention | Long-term observation only | Repeated rumination is desired; outbound repetition control is future proactive-contact design. |

## Contracts And Data Shapes

### Self-cognition consolidation origin builder

```python
def build_self_cognition_consolidation_origin(
    *,
    episode: CognitiveEpisode,
) -> ConsolidationOriginMetadata:
    """Project an internal-thought preview episode into consolidation origin."""
```

Required validation:

- `trigger_source == "internal_thought"`
- `input_sources == ["internal_monologue"]`
- `output_mode == "preview"`
- `validate_cognitive_episode(episode)` succeeds

The builder projects the same identifier fields as
`build_user_message_consolidation_origin(...)`. Empty
`active_turn_platform_message_ids` and `active_turn_conversation_row_ids` are
valid for self-cognition.

### Origin-aware consolidation selection

`call_consolidation_subgraph(global_state)` must select the origin builder from
`global_state["cognitive_episode"]["trigger_source"]`.

Supported selections:

- `user_message` -> `build_user_message_consolidation_origin(...)`
- `internal_thought` -> `build_self_cognition_consolidation_origin(...)`

Unsupported origins raise the existing consolidation-origin error path.

### Write policy reason

Use stable reason strings:

- `user_message_chat_input` for supported user-message origins.
- `internal_thought_same_path` for supported internal-thought preview origins.
- `origin_not_enabled` for denied origins.

### Prompt-facing source identity

Prompt payloads that currently describe `decontexualized_input` as user speech
must receive a compact origin block:

```python
{
    "trigger_source": "user_message | internal_thought",
    "input_sources": ["..."],
    "output_mode": "...",
    "episode_id": "...",
}
```

The prompt wording must define:

- For `user_message`, `decontexualized_input` is the user's current turn.
- For `internal_thought`, `decontexualized_input` is the internal trigger text
  summarizing the character's current cognitive focus and evidence.
- `final_dialog` may be visible dialog for user-message turns or private
  finalization for internal-thought turns.

### Self-cognition consolidation outcome

Self-cognition runner/worker event metadata must expose only structural and
sanitized fields:

```python
{
    "consolidation_called": bool,
    "write_success": dict[str, bool],
    "scheduled_event_count": int,
    "cache_evicted_count": int,
    "origin_trigger_source": "internal_thought",
    "origin_episode_id": str,
}
```

Do not log raw internal packets, full chat bodies, private finalization text,
or generated action-candidate text.

## LLM Call And Context Budget

This plan adds no response-path LLM calls and no new prompt family.

Background self-cognition cases that enter consolidation reuse existing calls:

| Stage | Before | After |
|---|---:|---:|
| RAG supervisor | 0 or 1 existing call | unchanged |
| Shared cognition | 1 existing call | unchanged |
| Dialog/private finalization | 0 or 1 existing call | 1 existing shared dialog/finalization call when consolidation is applied |
| Consolidator global-state updater | 0 for self-cognition production | 1 existing consolidator call |
| Consolidator relationship recorder | 0 for self-cognition production | 1 existing consolidator call |
| Facts harvester and evaluator | 0 for self-cognition production | existing consolidator bounded loop |
| Memory-unit updater | 0 for self-cognition production | existing updater when the shared writer has evidence |
| Task dispatcher LLM | 0 for self-cognition production | existing dispatcher call only when future promises and tools are present |

The worker's existing `SELF_COGNITION_MAX_CASES_PER_TICK` remains the case-count
cap. No response-path latency budget changes.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
  - Add `build_self_cognition_consolidation_origin(...)`.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
  - Allow the seven existing write/effect lanes for supported
    `internal_thought` preview origins.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
  - Select the origin builder by trigger source.
  - Preserve the same subgraph and node sequence.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
  - Add source-aware prompt payload fields and minimal wording updates.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
  - Add source-aware prompt payload fields and minimal wording updates.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
  - Add source-aware prompt payload fields and minimal wording updates.

- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Build a consolidation-ready state after cognition/private finalization.
  - Add injectable consolidation seam for tests and production reuse.
  - Preserve dry-run artifact writing behavior.

- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Apply consolidation in production worker ticks through the runner seam.
  - Preserve existing action-candidate handoff behavior.
  - Preserve production no-local-file behavior.

- `src/kazusa_ai_chatbot/self_cognition/models.py`
  - Add exact artifact/status constants only when needed for sanitized
    consolidation outcome records.

- `src/kazusa_ai_chatbot/self_cognition/tracking.py`
  - Add sanitized consolidation outcome record construction.

- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Document same-path persistence, private finalization, no new DB, and the
    long-term outbound-repetition observation.

- `development_plans/long_term/todo.md`
  - Add the outbound-repetition observation under autonomous agency.

- Focused tests under `tests/`
  - Update or add tests named in `Implementation Order`.

### Create

- No new production module is approved.
- No new DB submodule, MongoDB collection, migration, or storage helper is
  approved.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - Keep existing lane implementations. Do not fork the writer for
    self-cognition.

- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
  - Live chat post-turn behavior remains unchanged.

- `src/kazusa_ai_chatbot/conversation_progress/`
  - Do not add self-cognition persistence in this plan.

- Adapter delivery, service endpoints, reflection promotion, DB bootstrap, and
  config/environment handling remain unchanged.

## Implementation Order

1. Add failing tests for `build_self_cognition_consolidation_origin(...)` in
   `tests/test_consolidation_origin_metadata.py`.
2. Implement the self-cognition origin builder.
3. Add policy tests in `tests/test_consolidation_origin_policy.py` proving
   supported `internal_thought` allows all seven existing lanes and unsupported
   origins deny all lanes.
4. Implement the origin policy update.
5. Add tests for `call_consolidation_subgraph(...)` origin selection in
   `tests/test_consolidator_origin_selection.py`.
6. Implement origin selection in `call_consolidation_subgraph(...)`.
7. Add prompt-payload tests in
   `tests/test_consolidator_source_aware_payloads.py` proving consolidator
   prompt payloads include source identity and do not label internal-thought
   input as user speech.
8. Implement source-aware payload and minimal prompt wording updates.
9. Add self-cognition runner tests for consolidation-ready state construction,
   private finalization, injectable consolidation seam, and sanitized outcome
   artifacts.
10. Implement runner integration.
11. Add worker integration tests for production consolidation calls, no local
    file writes, unchanged action-candidate handoff, no assistant conversation
    row, and no adapter delivery.
12. Implement worker integration.
13. Update self-cognition README and long-term roadmap observation.
14. Run all `Verification` commands.
15. Run the `Independent Code Review` gate, fix in-scope findings, rerun
    affected verification, and record evidence.

## Progress Checklist

- [ ] Stage 1 - origin contract complete
  - Covers: Implementation Order steps 1-6.
  - Verify: origin metadata, origin policy, and consolidator origin-selection
    tests pass.
  - Evidence: record expected red failures and green command output.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 2 - source-aware shared prompts complete
  - Covers: Implementation Order steps 7-8.
  - Verify: prompt-payload tests pass and no separate self-cognition prompt
    family is created.
  - Evidence: record changed prompts and focused test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 3 - self-cognition runner integration complete
  - Covers: Implementation Order steps 9-10.
  - Verify: runner/tracking tests pass.
  - Evidence: record private-finalization handling and sanitized artifact
    shape.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 4 - worker production integration complete
  - Covers: Implementation Order steps 11-13.
  - Verify: worker integration, event logging, README, and roadmap checks pass.
  - Evidence: record no-file production assertion and no delivery/history
    assertions.
  - Handoff: next agent starts Stage 5 verification.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 5 - full verification complete
  - Covers: Implementation Order step 14.
  - Verify: every command in `Verification` passes or has an explicitly
    allowed no-match result.
  - Evidence: record command output and static grep results.
  - Handoff: next agent starts Stage 6 independent code review.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 6 - independent code review complete
  - Covers: Implementation Order step 15.
  - Verify: full diff reviewed against this plan; affected tests rerun after
    review fixes.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, residual
    risks, and approval status.
  - Handoff: plan can be marked completed only after this stage is signed off.
  - Sign-off: `<agent/date>` after evidence is recorded.

## Verification

Run from repository root with the project virtual environment.

### Static Compile

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin_policy.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_reflection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\self_cognition\models.py src\kazusa_ai_chatbot\self_cognition\tracking.py src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\self_cognition\worker.py
```

### Static Greps

```powershell
rg -n "self_cognition[/\\]memory|class .*Memory|memory orchestrator" src\kazusa_ai_chatbot\self_cognition
```

Expected result: no matches. `rg` exit code `1` is acceptable.

```powershell
rg -n "record_self_cognition_progress|last_self_cognition_origin_episode_id|SelfCognitionProgressRecordInput" src\kazusa_ai_chatbot
```

Expected result: no matches. `rg` exit code `1` is acceptable.

```powershell
rg -n "conversation_progress" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin_policy.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py
```

Expected result: no matches. `rg` exit code `1` is acceptable. A match here
means the implementation added a `conversation_progress` policy or persistence
lane, which is forbidden by this plan.

```powershell
rg -n "create_collection|conversation_episode_state|self_cognition_memory|self_cognition_progress" src\kazusa_ai_chatbot\db src\kazusa_ai_chatbot\self_cognition
```

Expected result: existing collection/bootstrap references only. New collection
creation for this feature is a blocker.

```powershell
rg -n "cooldown|rate limit|rate_limit|spam|suppress repeated|topic suppression|last_self_cognition_origin_episode_id" src\kazusa_ai_chatbot\self_cognition src\kazusa_ai_chatbot\nodes
```

Expected result: no new intentional mitigation added by this plan. Existing
unrelated occurrences must be listed in `Execution Evidence`.

```powershell
git diff --check
```

### Focused Tests

```powershell
venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py tests\test_consolidation_origin_policy.py -q
venv\Scripts\python -m pytest tests\test_consolidator_origin_selection.py tests\test_consolidator_source_aware_payloads.py -q
venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py -q
```

### Adjacent Regression Tests

```powershell
venv\Scripts\python -m pytest tests\test_service_background_consolidation.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_dry_run_cli.py -q
venv\Scripts\python -m pytest tests\test_conversation_progress_recorder.py tests\test_conversation_progress_cognition.py tests\test_conversation_progress_flow.py -q
```

### Full Deterministic Regression

```powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

Live DB and live LLM tests are not required for this plan unless implementation
changes DB bootstrap behavior or prompt contracts in a way that deterministic
tests cannot cover. If that happens, update this plan before running
additional live gates.

## Independent Plan Review

Review performed on 2026-05-14 from a fresh-review posture after the user
requested independent review and approval.

Inputs inspected:

- `README.md`
- `docs/HOWTO.md`
- `development_plans/README.md`
- `development_plans/long_term/todo.md`
- completed Stage 06 origin-policy plan
- current self-cognition runner and worker
- current consolidator origin, policy, subgraph, persistence, reflection,
  facts, and memory-unit code
- current brain-service post-turn helpers

Findings and fixes:

- Blocker resolved during this review: Implementation Order steps for
  consolidator origin selection and source-aware prompt payloads left test-file
  placement open. The plan now names
  `tests/test_consolidator_origin_selection.py` and
  `tests/test_consolidator_source_aware_payloads.py`.
- Blocker resolved during this review: the conversation-progress static grep
  would have matched existing source-packet and cognition-context references.
  The verification gate now checks exact forbidden symbols and exact forbidden
  consolidator policy/persistence files.
- Blocker resolved: the prior approved plan denied the very autonomy lanes the
  user wanted. This plan allows the existing seven `db_writer` lanes for
  supported `internal_thought` origins.
- Blocker resolved: the prior plan introduced a new conversation-progress
  persistence path. This plan removes that side path and reuses existing
  persistence infrastructure only.
- Blocker resolved: the prior plan treated repeated internal rumination as a
  loop to suppress. This plan allows repeated internal rumination and records
  outward repetition control as long-term observation only.
- Non-blocking risk: source-aware prompt payloads are required so internal
  thoughts are not misread as user-authored speech while still using the same
  prompt families.

Plan self-review:

- Coverage: every `Must Do` item maps to implementation steps and verification
  gates.
- Placeholder scan: no unresolved decisions remain.
- Contract consistency: origin builder, policy reasons, lane list, change
  surface, and verification commands are aligned.
- Granularity: checklist stages split origin contract, prompt payloads,
  runner integration, worker integration, verification, and review.
- Verification: tests and greps cover same-path policy, no new DB, no
  conversation-progress persistence lane, no intentional spam mitigation, and
  sanitized event logging.

Approval decision: approved for implementation.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/context leaks, persistence risk,
  accidental delivery/history writes, brittle fixtures, and avoidable blast
  radius.
- Alignment with the same-path design, `Must Do`, `Deferred`, `Change Surface`,
  verification gates, and acceptance criteria.
- Regression and handoff quality, including static-grep accuracy, execution
  evidence, no new DB, no self-cognition progress side path, no intentional
  spam mitigation, and lifecycle registry accuracy.

Fix concrete findings directly only when the fix is inside the approved Change
Surface. If a finding requires a new contract, boundary, or change surface,
stop and update this plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `build_self_cognition_consolidation_origin(...)` exists and validates only
  internal-thought preview episodes.
- `call_consolidation_subgraph(...)` routes supported user-message and
  internal-thought episodes into the same subgraph and rejects unsupported
  origins.
- `build_consolidation_write_policy(...)` allows the same seven existing
  writer lanes for supported `internal_thought` preview origins as for
  supported user-message origins.
- Existing user-message consolidation behavior remains unchanged.
- Self-cognition production runs call the existing consolidation subgraph
  through an injectable seam after shared cognition/private finalization.
- Self-cognition private finalization is not saved as a conversation-history
  assistant row and is not delivered through adapters.
- No new MongoDB collection, migration, DB facade helper, conversation-progress
  self-cognition writer, or memory sidecar is created.
- No new intentional spam-prevention, cooldown, topic-suppression, or outbound
  dedupe logic is implemented.
- The long-term roadmap records outbound repetition governance as future
  observation.
- All Verification commands pass or have explicitly allowed no-match results.
- Independent code review passes and evidence is recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Internal thought is mistaken for user-authored speech | Add source identity to shared prompt payloads without creating separate prompt families | Prompt-payload tests |
| Live user-message consolidation regresses | Preserve user-message origin validation and policy reasons | Origin/policy regression tests |
| A sidecar memory or progress writer appears | Forbid new DB collections, self-cognition memory module, and progress writer | Static greps and code review |
| Private finalization leaks to chat history or adapters | Keep self-cognition finalization private and test no delivery/history writes | Worker integration tests |
| Repeated internal rumination produces repeated outward messages | Not mitigated in this plan by user request; recorded as long-term observation | Roadmap check and no-spam-mitigation grep |
| Background worker cost increases | Reuse existing LLM stages only and keep current per-tick case cap | LLM budget review and worker tests |

## Execution Evidence

- Independent plan review: updated by Codex on 2026-05-14 after user clarified
  same-path self-cognition, repeated rumination, no new DB, and no intentional
  spam mitigation in this plan.
- Independent plan review: refreshed by Codex on 2026-05-14 after final review
  request; test-file placement and static-grep expectations were tightened;
  no blockers remain.
- Approval status: approved for implementation.
- Implementation: not started.

## Execution Handoff

Intended execution mode: sequential implementation in the current workspace or
on a feature branch.

Next action: reread this approved plan, load mandatory skills, check
`git status --short`, then start at Progress Checklist Stage 1.
