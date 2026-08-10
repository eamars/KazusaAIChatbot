# qq group public scene and response ordering bugfix plan

## Summary

- Goal: give group-chat Cognition one bounded public scene alongside the
  current participant's continuity, while preserving the existing sequential
  settlement path.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: development-plan, local-llm-architecture, py-style,
  cjk-safety, test-style-and-execution, debug-llm, llm-trace-debug,
  character-test
- Overall cutover strategy: bigbang update of the Cognition scene contract;
  reuse the existing ambient logical-turn lane and retain participant-scoped
  Conversation Progress V2.
- Highest-risk areas: trigger-aware public ordering, identity redaction,
  prompt budget, participant-continuity precedence, and live group behavior.
- Acceptance criteria: deterministic replay and the fixed binary live behavior
  contract show one ordered public scene, participant packets remain isolated,
  private chat is unchanged, and the common path adds no LLM call, database
  read, persistence shape, or response-ordering mechanism. Final completion
  requires the user's explicit sign-off on the prescribed evidence packet.

## Context

The QQ group 638473184 export contains contradictory public responses produced
from otherwise separate participant-scoped continuity branches. Conversation
Progress V2 correctly stores one packet per participant, while its runtime
already loads an independent ambient channel lane. The missing behavior is a
prompt-safe public projection of that ambient lane for group Cognition.

The current normal service has one settlement worker that awaits each
`_process_settlement_lease(...)` call before requesting the next lease.
Same-group and cross-group cognition are therefore already sequential in this
service. The carried metadata-only traces do not prove a concurrent cognition
or delivery race. This plan preserves and tests that existing invariant and
adds no group claim lease.

Evidence carried into this plan:

- `test_artifacts/qq_group_638473184_conversation_history.json`
- `test_artifacts/qq_group_638473184_conversation_episode_state.json`
- `test_artifacts/qq_group_638473184_conversation_episode_blocks.json`
- `test_artifacts/llm_debug/trace_b_branch_7139364461074f15be91c4d06761ac23.json`
- `test_artifacts/llm_debug/trace_a_branch_1dd19ed608854102848dc4bd262da1fd.json`
- `test_artifacts/qq_group_638473184_failure_mode_review.md`

The completed group-scene digest plan owns reflection/self-cognition only. The
draft `cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md`
overlaps the persona and Cognition files in this plan; it must rebaseline on
this completed contract before its own approval or execution.

## Mandatory Skills

- `development-plan`: plan lifecycle, execution, evidence, and review gates.
- `local-llm-architecture`: bounded public context and Cognition ownership.
- `py-style`: all Python production edits and review.
- `cjk-safety`: Python files containing Chinese prompt text.
- `test-style-and-execution`: deterministic, patched, and live test ownership.
- `debug-llm`: agent-authored live quality review from raw evidence.
- `llm-trace-debug`: protected trace export and review.
- `character-test`: one-turn-at-a-time real-service behavior verification.

## Mandatory Rules

- Production implementation requires a separate explicit user command.
- Record `git status --short`, HEAD, Python version, governing docs, and every
  source/test path in this plan before implementation. Preserve unrelated
  worktree changes and never read `.env`.
- Use `venv\Scripts\python` for Python commands and `apply_patch` for manual
  edits.
- After context compaction and after each major stage evidence checkpoint,
  reread this complete plan before continuing.
- Parent owns tests, verification, evidence, review remediation, and preparation
  of the final evidence packet. One production subagent owns production code,
  followed by one independent review subagent. Native subagent unavailability
  stops execution unless the user explicitly approves fallback execution.
- The implementation agent, parent agent, and review agent may report only
  technical gate results. They cannot grant final sign-off, mark this plan
  completed, update its registry status to completed, or archive it.
- The user is the sole final sign-off authority. Until the user explicitly
  accepts the final evidence packet, this plan remains `in_progress` and its
  final lifecycle checkpoint remains unchecked.
- Apply the live behavior contract in Verification as a binary rubric. Natural
  wording may vary, but no agent may waive, reinterpret, substitute, or add a
  behavior condition. Missing evidence and any unmet condition are failures;
  `uncertain`, `partial`, and `accepted with caveat` are not valid results.
- Keep `ConversationProgressScope` keyed by platform, channel, and current
  global user. Keep all progress storage and recording participant-scoped.
- Public scene content is limited to bounded public message content, display
  names, visible addresses, reply display names, and semantic order labels.
  Exclude platform/global/database identifiers, traces, source refs, raw wire
  syntax, private residue, and participant packet fields.
- Use `prompt_message_context.body_text` for the trigger text. Current
  attachment observations remain in the existing current-episode evidence and
  are not copied into the group scene.
- Public scene owns visible group order and participants. Current episode owns
  the triggering event. Participant continuity owns only directly relevant
  history for the current participant.
- RAG supplies evidence; Cognition decides stance, target roles, and reason to
  speak; Dialog owns visible wording. The scene itself creates no automatic
  reason to respond.
- Deterministic code owns projection, redaction, limits, persistence, and
  delivery. It does not keyword-gate, rewrite, suppress, retarget, or
  regenerate model output. Public-scene limit overflow is a non-fatal
  projection concern: it truncates or drops bounded context, and an optional
  projection failure degrades to an empty public-scene field so Cognition can
  continue with the remaining lanes.
- Add no LLM call, database read, retry, feature flag, compatibility field,
  alternate scene path, or persisted group state.
- Preserve the existing canonical JSON parser and bounded failure contracts.
- Deterministic tests run in batches. Live LLM tests run and are inspected one
  case at a time; schema success alone is not quality acceptance.
- Live service tests use `KAZUSA_TEST_DB_GUARD=1`, database
  `_test_kazusa_live_llm`, and a unique group and user-id suffix per case.
  `no_remember=true` skips consolidation but still permits isolated
  conversation-history rows in that test database.
- Before completion or lifecycle sign-off, run Independent Code Review and
  record its result in Execution Evidence.

## Must Do

- Add `GroupSceneTurnV1` and `GroupSceneContextV1` as transient prompt-safe
  projection contracts in the conversation-progress boundary.
- Merge the trigger turn with already-loaded ambient logical turns into one
  chronological scene. Mark turns as before, at, or after the trigger.
- Retain the trigger and newest public context under exact field, turn,
  participant, and aggregate caps.
- Add required `public_group_scene: str` to `SceneContextV2`; private paths
  supply an empty string.
- Label existing `conversation_continuity` as current-participant continuity
  and state its lower public-scene authority in the goal prompt.
- Preserve and test the current sequential settlement-worker behavior without
  changing settlement or service production code.
- Add the named deterministic fixture, focused tests, guarded live tests, and
  one parent-authored Markdown quality review.
- Classify every guarded live case against the fixed binary behavior contract
  and assemble the prescribed final evidence packet.
- Update the conversation-progress, brain-service, and Cognition Core V2
  READMEs with the two-lane ownership boundary.
- Run every verification gate and record exact commands and outcomes.
- Present the evidence packet to the user only after every technical gate
  passes; retain `in_progress` status until the user explicitly signs off.

## Deferred

- Durable or generic multi-participant Conversation Progress V3, a persisted
  group packet, group-scene collection, cache, TTL, or index.
- A new response-ordering lease, worker pool, delivery sequencer, distributed
  lock, scene version, stale-response regeneration, or shutdown coordinator.
  Any future ordering change requires a current red reproduction at the actual
  cognition or adapter-delivery boundary.
- Conversation Progress V2 schema, repository, runtime reads, compaction,
  recorder, cache, post-turn scope, or historical repair.
- Character-global state, relationship reducers, RAG, reflection, scheduler,
  memory lifecycle, Dialog contracts, adapter payloads, and control-console UI.
- Private-chat behavior, dependency upgrades, compatibility shims, fallback
  paths to an alternate public-scene source, and unrelated prompt refactors.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
| --- | --- | --- |
| Cognition scene | bigbang | Add the required public scene and update all callers, validators, fixtures, and prompts together. |
| Public group projection | bigbang | Use the new trigger-aware ambient projection as the only group public-scene input. |
| Participant progress | compatible | Retain the current V2 scope, reads, writes, cache, and packet shape. |
| Settlement ordering | compatible | Preserve the existing single awaited settlement worker and versioned claim. |
| Private chat | compatible | Supply an empty public scene and preserve existing behavior. |
| Database and LLM routes | compatible | Add no read, write shape, migration, model call, retry, or cap increase. |

### Cutover Policy Enforcement

- Update the exact scene contract in one cutover without aliases or fallback
  keys.
- Treat the participant packet as retained relationship continuity, not as a
  public group scene.
- Any persistence, ordering, model-call, or database-read expansion requires
  an approved plan update before implementation.

## Data Migration

- No collection, index, document, packet, block, cache key, or historical row
  is migrated, repaired, created, or deleted.
- The public scene exists only as prompt-facing runtime state for one persona
  turn.
- Production verification proves that DB, conversation-progress runtime,
  settlement, and service files remain outside the implementation diff.

## Target State

~~~text
existing sequential settlement worker
  -> existing ambient logical turns + trigger message
  -> bounded chronological public group scene
  + current participant's Conversation Progress V2 projection
  -> Cognition judges scene, stance, target roles, and reason to speak
  -> existing Dialog, persistence, delivery, and settlement completion
~~~

Group Cognition receives two named lanes:

~~~text
public_group_scene
  public speakers, addresses, trigger position, later visible turns, and text

conversation_continuity
  current participant's directly relevant continuation and relationship history
~~~

Ambient turns that occurred after the trigger but were persisted before
Cognition loads remain after the trigger in the rendered chronology. This lets
Cognition recognize a public topic shift without calling the trigger the latest
message. Private chat carries `public_group_scene=""`.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Public group continuity | Project already-loaded ambient logical turns | Reuses the current facility with zero new reads or model calls. |
| Relationship continuity | Retain one V2 packet per participant | Preserves privacy and existing recorder ownership. |
| Trigger ordering | Merge the trigger into ambient chronology and label relative position | A separate `current_turn` would misrepresent later persisted messages. |
| Current text | Use `prompt_message_context.body_text` | Keeps authored visible text separate from model-authored attachment descriptions. |
| Response ordering | Preserve the existing awaited global settlement worker | Current source already serializes cognition; no new lease is justified. |
| Semantic judgment | Keep scene interpretation in Cognition | Parallel topic and target meaning are semantic character judgments. |
| Prompt budget | Cap rendered public scene at 1,800 characters | Keeps the existing 24,000-character goal payload cap. |
| Future direction | Plan durable generic group progress only after ambient-scene insufficiency is proven | Avoids a speculative persistence mechanism. |
| Adjacent global-state plan | Complete this plan first; require that draft to rebaseline afterward | The plans share persona and Cognition files. |

## Contracts And Data Shapes

### Public group-scene projection

Add to `kazusa_ai_chatbot.conversation_progress.models`:

~~~python
class GroupSceneTurnV1(TypedDict):
    role: Literal['user', 'assistant']
    speaker_name: str
    text: str
    addressed_names: list[str]
    reply_to_name: str
    scene_position: Literal['before_trigger', 'trigger', 'after_trigger']


class GroupSceneContextV1(TypedDict):
    schema_version: Literal['group_scene_context.v1']
    turns: list[GroupSceneTurnV1]
    visible_participants: list[str]
    omitted_turn_count: int
~~~

Expose through `kazusa_ai_chatbot.conversation_progress`:

~~~python
def build_group_scene_context(
    *,
    ambient_logical_turns: Sequence[ConversationLogicalTurnV1],
    trigger_occurred_at: str,
    trigger_speaker_name: str,
    trigger_body_text: str,
    trigger_addressed_global_user_ids: Sequence[str],
    trigger_reply_to_display_name: str,
    scope_users: Sequence[Mapping[str, object]],
) -> GroupSceneContextV1:
    ...


def project_group_scene_prompt(context: GroupSceneContextV1) -> str:
    ...
~~~

Exact caps in `conversation_progress.policy`:

| Value | Cap |
| --- | ---: |
| Retained turns, including trigger | 6 |
| Visible participants | 12 |
| Text per turn | 360 characters |
| Display/reply/address name | 64 characters |
| Addressed names per turn | 6 |
| Rendered public scene | 1,800 characters |

Use these exact constant names for the caps:
`GROUP_SCENE_MAX_TURNS`, `GROUP_SCENE_MAX_VISIBLE_PARTICIPANTS`,
`GROUP_SCENE_MAX_TURN_TEXT_CHARS`, `GROUP_SCENE_MAX_NAME_CHARS`,
`GROUP_SCENE_MAX_ADDRESSED_NAMES`, and `GROUP_SCENE_MAX_RENDERED_CHARS`.

Projection rules:

- Validate `trigger_occurred_at` through the existing storage-time parser.
- Ambient turns retain canonical chronological order. Insert the trigger after
  ambient turns with an equal timestamp; label every retained turn relative to
  that position. Do not render timestamps.
- For each ambient turn, read `display_name`, join `fragments` with spaces,
  read `addressed_to_global_user_ids`, and use only
  `reply_context.reply_to_display_name` for its reply name. The trigger uses
  the corresponding explicit function arguments.
- Resolve an addressed global ID through the first `scope_users` row with the
  same non-empty `global_user_id` and a non-empty `display_name`. Emit only the
  capped display name; omit every unresolved ID.
- Normalize strings with the existing `cap_text(...)` policy. A blank speaker
  uses its semantic `user` or `assistant` role label. Omit unresolved addressed
  identities and deduplicate resolved names in input order.
- Select the trigger plus the five newest non-trigger turns, then restore
  chronology. This always retains the trigger and newest visible public turn.
- Derive `visible_participants` from retained speaker, address, and reply names
  in first-seen chronology. Apply the name and count caps.
- Render exact semantic labels `Before trigger:`, `At trigger:`, and `After
  trigger:` for before-trigger, trigger, and after-trigger turns. Do not render
  schema names, timestamps, IDs, storage terms, or scene metadata.
- If the render exceeds 1,800 characters, drop the oldest non-trigger turn,
  recompute participants, and repeat. If only the trigger remains, shorten its
  text to the remaining budget. The projection returns a bounded string even
  when an externally supplied transient context is oversized; it does not
  propagate a limit-overflow exception.
- `omitted_turn_count` equals the number of normalized ambient turns excluded
  by the six-turn and aggregate limits. It never counts the retained trigger.
- A malformed ambient row that unexpectedly reaches this projection is skipped
  without inventing visible content, and the trigger remains eligible.
- If an expected projection/input error still occurs at the persona boundary,
  log the concrete error and continue with `public_group_scene=""`.

### Cognition scene contract

Add required `public_group_scene: str` to `SceneContextV2`. The exact keys are:

~~~text
channel_scope
character_role
current_user_role (when present)
semantic_scene
public_group_scene
conversation_continuity
semantic_temporal_context
~~~

- `public_group_scene` is a string capped at 1,800 characters.
- Group persona state carries the rendered string; private persona state
  carries an empty string.
- `conversation_continuity` remains capped at 2,200 characters and is rendered
  as current-participant continuity.
- Unknown scene keys and invalid types follow the existing contract-error path.
- The goal prompt gives visible public order precedence over participant
  continuity while leaving response reason, stance, and target roles to
  Cognition.

### Existing ordering contract

- `_turn_settlement_worker()` requests one lease, awaits
  `_process_settlement_lease(...)`, and only then requests the next lease.
- `claim_for_cognition()` and `complete_cognition()` retain their current
  versioned contracts.
- This plan adds no ordering state, queue, lock, version, shutdown API, or
  production change in `service.py` or `turn_settlement.py`.

## LLM Call And Context Budget

| Stage | Before | After | Gate |
| --- | --- | --- | --- |
| Relevance | Existing calls and context | Unchanged | No new input or call. |
| Goal Cognition | Existing call; 24,000-character aggregate cap | Same call and cap; add at most 1,800 public-scene characters inside existing fitting | Prompt-budget tests. |
| Dialog and post-turn work | Existing calls | Unchanged | Regression tests and static diff. |

The production default architecture cap remains 50,000 tokens. No response or
background model route, completion cap, retry, or database read changes.

## Change Surface

Target boundary: conversation-progress owns projection; persona carries it;
Cognition consumes it.

### Delete

- None.

### Modify

- `src/kazusa_ai_chatbot/conversation_progress/models.py`: add the two
  transient types.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py`: add exact caps.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py`: build, order,
  redact, cap, and render the public scene.
- `src/kazusa_ai_chatbot/conversation_progress/__init__.py`: export the public
  types and functions.
- `src/kazusa_ai_chatbot/conversation_progress/README.md`: document public
  scene versus participant continuity.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: add required
  `public_group_scene: str` to persona and cognition state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: build the scene after
  `_build_scope_users()` and pass the rendered string; use an empty string for
  private chat.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: map the string
  into `SceneContextV2` and label participant continuity.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: require and validate
  `scene_context.public_group_scene`.
- `src/kazusa_ai_chatbot/cognition_core_v2/validation_cli.py`: supply the
  required empty public-scene field in its internal validation payload.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: add the short
  scene-authority procedure without changing output or parser contracts.
- `src/kazusa_ai_chatbot/brain_service/README.md`: document that the existing
  sequential settlement path hands group turns to the two context lanes.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: document scene authority.
- `tests/test_persona_supervisor2.py`: verify group build, private empty value,
  projection-error degradation, and no extra I/O or model stage.
- `tests/test_e2e_live_llm.py`: return the seeded conversation-row ID so the
  guarded direct graph harness can exclude its active trigger from ambient
  history.
- `tests/test_cognition_chain_connector_mapping.py`: verify exact scene handoff.
- `tests/test_cognition_core_v2_contracts.py`: update exact scene keys and cap.
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`: verify fitting
  retains both public scene and participant continuity.
- Existing Cognition input builders in
  `tests/test_cognition_core_v2_benchmark.py`,
  `tests/test_cognition_core_v2_alignment_gates.py`,
  `tests/test_cognition_core_v2_integration.py`,
  `tests/test_cognition_core_v2_model_assignment_matrix.py`,
  `tests/test_cognition_core_v2_live_llm.py`, and
  `tests/test_multi_source_cognition_stage_00_regression_baseline.py`: supply
  the required empty public-scene field.
- `tests/test_service_input_queue.py`: strengthen the existing worker test to
  prove a second lease is not requested while current processing is blocked.

### Create

- `tests/fixtures/qq_group_public_scene_interleaving.json`: sanitized captured
  ordering and branch markers.
- `tests/test_conversation_progress_group_scene.py`: pure projection, ordering,
  redaction, and cap tests.
- `tests/test_qq_group_public_scene_live_llm.py`: five guarded, one-at-a-time
  real-service cases.
- `test_artifacts/llm_debug/qq_group_public_scene_response_ordering_review.md`:
  parent-authored live quality review.

### Keep

- `src/kazusa_ai_chatbot/service.py` and
  `src/kazusa_ai_chatbot/brain_service/turn_settlement.py`: existing ordering.
- `src/kazusa_ai_chatbot/conversation_progress/runtime.py`, repository, cache,
  recorder, and `brain_service/post_turn.py`: existing reads and persistence.
- DB modules, adapters, Dialog/L3, RAG, reflection, scheduler, memory, and
  control-console modules.

## Overdesign Guardrail

- Actual problem: isolated participant continuity can dominate one publicly
  visible multi-user scene.
- Minimal change: render the already-loaded ambient turns and trigger into one
  bounded public scene beside the existing participant continuation.
- Ownership: conversation-progress projects; persona carries; Cognition judges;
  Dialog words; existing service ordering and persistence remain unchanged.
- Rejected complexity: group persistence, extra reads or calls, scene version,
  ordering lease, delivery sequencer, regeneration, output filtering, aliases,
  feature flags, and adapter changes.
- Evidence threshold: durable generic group progress requires repeated loss
  after this projection is live; ordering work requires a current red
  reproduction at the actual concurrency or delivery boundary.

## Agent Autonomy Boundaries

- Preserve every public type, signature, cap, selection rule, and file boundary
  in this plan.
- Search for an existing helper before adding local structural logic; reuse
  `cap_text(...)` and the existing storage-time parser.
- Keep helper functions private to `conversation_progress.projection` and avoid
  new modules, wrappers, alternate call sites, or general-purpose abstractions.
- Make no production edit outside `Change Surface > Modify`. A required outside
  edit stops execution for a plan update and user approval.
- Keep review fixes within the approved surface; contract or scope changes stop
  closeout.
- Apply the fixed live criteria mechanically and record `pass` or `fail` for
  every listed condition. A failed or missing condition blocks the evidence
  packet from being submitted for user sign-off.
- Agents have no discretion to alter the final sign-off authority or replace
  explicit user acceptance with tests, review approval, or agent judgment.

## Implementation Order

### Stage 1 — baseline and focused contract

1. Record worktree, HEAD, Python version, governing docs, and pre-existing
   changes in Execution Evidence.
2. Add the sanitized fixture and
   `tests/test_conversation_progress_group_scene.py` with exact tests for
   trigger insertion, after-trigger turns, ID removal, all caps, omission
   count, budget dropping, and trigger retention.
3. Update the four existing persona/Cognition contract tests named in Change
   Surface for required `public_group_scene`.
4. Strengthen
   `test_settlement_worker_marks_active_model_work` in
   `tests/test_service_input_queue.py` to assert the coordinator receives no
   second lease request until current processing is released. This baseline
   must pass before production edits.
5. Run the focused commands in Verification. Record the projection/scene-key
   failures and the ordering-baseline pass.

### Stage 2 — projection and cognition implementation

6. Add the models and policy constants, then implement both public projection
   functions exactly as contracted.
7. Export the functions and update the conversation-progress README.
8. Build the group projection in `persona_supervisor2`; set private
   `public_group_scene` to an empty string.
9. Carry the string through persona/Cognition state and add it to the exact
   `SceneContextV2` validator and connector.
10. Add the short goal-prompt authority procedure and run immediate syntax and
    prompt-render checks after the CJK-bearing edit.
11. Rerun focused tests until the contract and non-fatal overflow behavior pass
    without changing the scene contract shape.

### Stage 3 — integration and documentation

12. Complete the named persona, connector, service-worker, contract, and
    prompt-budget tests.
13. Update the brain-service and Cognition READMEs.
14. Run the focused and broader deterministic commands. Confirm the diff has
    no service, settlement, runtime, repository, DB, or post-turn production
    file.

### Stage 4 — static and smoke verification

15. Run every syntax, static, deterministic, regression, and smoke command in
    Verification.
16. Record zero added model calls and DB reads from the patched integration
    assertions and unchanged production read/call owners.
17. Fix only in-scope failures and rerun their complete affected command.

### Stage 5 — guarded live quality verification

18. Set the guarded test DB variables from Verification. Each case creates a
    unique channel and user-id suffix, while A/B/C remain stable within that
    case.
19. Present and record the exact fixed script and observation target, run one
    named case, save its raw evidence, inspect it, and then continue to the
    next case.
20. Apply every fixed per-case and cross-case condition in Verification, record
    only `pass` or `fail`, and write the human-readable review from the raw
    artifacts.

### Stage 6 — independent technical review and evidence packet

21. Run final static and deterministic gates and `git diff --check`.
22. Start exactly one independent code-review subagent with the plan, full
    diff, commands, live evidence, and review artifact.
23. Parent fixes only in-scope findings, reruns affected gates, and records
    the review's technical pass/fail result and residual risks.
24. After every technical and behavior condition passes, assemble the exact
    User Sign-off Evidence Packet defined in Verification.

### Stage 7 — user sign-off and lifecycle closeout

25. Present the complete evidence packet and the binary Acceptance Criteria
    1–10 readiness matrix to the user.
26. Leave this plan `in_progress` unless and until the user gives an explicit
    affirmative instruction that accepts the presented packet as final sign-off.
27. Only after that explicit user sign-off, mark the plan completed, synchronize
    the registry, archive the plan, and record the user's sign-off in Execution
    Evidence.

## Execution Model

- Parent establishes and owns the focused tests before implementation.
- Exactly one production-code subagent owns only production files in the
  Modify list and closes after reporting changed files, commands, and risks.
- Parent owns test changes, verification, artifacts, review remediation,
  evidence, and the sign-off packet; it may run integration work in parallel
  after the focused contract exists.
- Exactly one later independent review subagent reviews and reports a technical
  result without editing or granting final sign-off.
- The user alone owns final sign-off. The parent performs lifecycle closeout
  only after the user's explicit acceptance of the presented evidence packet.
- Native subagent unavailability stops implementation unless the user
  explicitly authorizes fallback execution.

## Progress Checklist

- [x] Stage 1 — focused contract and existing ordering baseline recorded.
  - Covers steps 1–5; run the focused commands and record red/pass outcomes.
  - Handoff: production subagent receives the fixed contract.
  - Technical record: parent / 2026-08-02; baseline red/pass evidence is recorded below.
- [x] Stage 2 — projection and Cognition contract implemented.
  - Covers steps 6–11; focused tests and syntax/render checks pass.
  - Handoff: parent completes integration and docs.
  - Technical record: DeepSeek V4 Flash production agent `019fc11f-63c7-7f40-bf03-b8d845c2e70b` / 2026-08-02; focused, syntax, prompt-render, and static gates pass.
- [x] Stage 3 — integration and documentation complete.
  - Covers steps 12–14; deterministic integration and diff-boundary checks pass.
  - Handoff: Stage 4 verification.
  - Technical record: parent / 2026-08-02; named integration batches, docs, call/read smoke, and protected-file diff gates pass.
- [x] Stage 4 — static, regression, and smoke gates pass.
  - Covers steps 15–17; record every command and any baseline failure.
  - Handoff: guarded live cases.
  - Technical record: parent / 2026-08-02; syntax, static, deterministic, regression, call/read, and smoke gates are recorded below.
- [x] Stage 5 — five guarded live cases pass the fixed binary contract.
  - Covers steps 18–20; record every condition, result, and artifact path.
  - Handoff: independent code review.
  - Technical record: parent / 2026-08-02; after the review remediation, all
    five guarded cases passed one at a time with the Asuna identity overlay,
    captured Cognition input, full protected traces for direct turns, and the
    parent-authored binary review. Fresh artifacts and every listed condition
    are recorded below.
- [x] Stage 6 — independent technical review passes and evidence packet is ready.
  - Covers steps 21–24; record findings, fixes, reruns, residual risks, the
    technical result, and the complete packet.
  - Handoff: present the packet to the user.
  - Technical record: the original implementation review by reviewer
    `019fc141-2ffa-76c3-8f72-f602fc343117` returned `PASS`. The later
    non-fatal-limit follow-up was independently re-reviewed by
    `019fc18f-b28e-7dd0-bc4c-37f76f6dde2c` and returned `PASS` after the
    redundant-trigger renderer finding was fixed and all affected gates and
    live evidence were rerun. The packet is ready for user presentation.
- [x] Stage 7 — user explicitly signs off and lifecycle closes.
  - Covers steps 25–27; this box may be checked only after the user explicitly
    accepts the presented final evidence packet.
  - Evidence: record the user's sign-off, completed registry update, and archive
    path.
  - Evidence record: user accepted the final evidence packet on 2026-08-02;
    the registry was updated and this record was moved to
    `development_plans/archive/completed/bugfix/`.
  - Sign-off: user / 2026-08-02.

## Verification

### Static and diff boundaries

~~~powershell
git diff --check
rg -n "group_scene_version|active_group_claim|group_scene_state|group_scene_packet" src tests
git diff --name-only -- src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/brain_service/turn_settlement.py src/kazusa_ai_chatbot/conversation_progress/runtime.py src/kazusa_ai_chatbot/conversation_progress/repository.py src/kazusa_ai_chatbot/brain_service/post_turn.py src/kazusa_ai_chatbot/db
rg -n "parse_llm_json_output" src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py
~~~

Expected results:

- `git diff --check` exits 0.
- The forbidden group-state scan returns no matches; `rg` exit 1 is expected.
- The protected-file diff command prints nothing.
- Parser scan shows only the existing canonical parser use; any new parser or
  repair path blocks sign-off.

### Syntax

Run `py_compile` for every modified Python file:

~~~powershell
$files = @(
  'src\kazusa_ai_chatbot\conversation_progress\models.py',
  'src\kazusa_ai_chatbot\conversation_progress\policy.py',
  'src\kazusa_ai_chatbot\conversation_progress\projection.py',
  'src\kazusa_ai_chatbot\conversation_progress\__init__.py',
  'src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py',
  'src\kazusa_ai_chatbot\nodes\persona_supervisor2.py',
  'src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py',
  'src\kazusa_ai_chatbot\cognition_core_v2\contracts.py',
  'src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py',
  'tests\test_conversation_progress_group_scene.py',
  'tests\test_persona_supervisor2.py',
  'tests\test_cognition_chain_connector_mapping.py',
  'tests\test_cognition_core_v2_contracts.py',
  'tests\test_cognition_core_v2_prompt_budget_continuity.py',
  'tests\test_service_input_queue.py',
  'tests\test_qq_group_public_scene_live_llm.py'
)
foreach ($file in $files) {
  venv\Scripts\python -m py_compile $file
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
~~~

Expected result: every command exits 0.

### Focused deterministic tests

~~~powershell
venv\Scripts\python -m pytest tests\test_conversation_progress_group_scene.py -q
venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_cognition_chain_connector_mapping.py -q
venv\Scripts\python -m pytest tests\test_cognition_core_v2_contracts.py tests\test_cognition_core_v2_prompt_budget_continuity.py -q
venv\Scripts\python -m pytest tests\test_service_input_queue.py::test_settlement_worker_marks_active_model_work -q
~~~

Before production edits, the new projection and scene-key tests fail for the
missing contract; the service-worker baseline passes. After implementation all
commands exit 0.

### Affected regression tests

~~~powershell
venv\Scripts\python -m pytest tests\test_conversation_progress_runtime.py tests\test_conversation_progress_v2_service.py tests\test_conversation_progress_v2_contract.py tests\test_conversation_progress_v2_regression.py -q
venv\Scripts\python -m pytest tests\test_service_cognition_graph.py tests\test_service_input_queue.py tests\test_relevance_turn_settlement.py tests\test_relevance_turn_settlement_graph.py -q
venv\Scripts\python -m pytest tests\test_cognition_core_v2_integration.py tests\test_cognition_current_event_grounding.py tests\test_cognition_interaction_style_context.py tests\test_dialog_mention_target_user.py -q
~~~

Expected result: all commands exit 0. Record unrelated baseline failures with
the exact command and prior evidence; keep them outside this plan.

### Call/read and smoke gates

The updated persona tests must assert that projection consumes supplied
`ambient_logical_turns` and invokes no loader or model stage. The unchanged
progress call-count test must pass:

~~~powershell
venv\Scripts\python -m pytest tests\test_conversation_progress_v2_service.py::test_ordinary_response_path_adds_no_llm_call -q
venv\Scripts\python -c "from kazusa_ai_chatbot.conversation_progress import build_group_scene_context, project_group_scene_prompt; from kazusa_ai_chatbot.service import app; print(app.title)"
~~~

Expected result: test and import exit 0; production diff contains no new read
owner or LLM invocation. A hard latency percentage is intentionally absent
because the change adds only bounded deterministic string projection and no I/O.

### Guarded live LLM and service gates

Use these fixed scripts:

| Case | Exact public turns | Observation target |
| --- | --- | --- |
| `public_target_distinct` | A: `@明日奈 我周六想去海边。` B: `@明日奈 我周六要加班。` C: `明日奈，你刚才是在回应谁？` | Keep A and B's plans and targets distinct. |
| `parallel_addresses` | A: `@明日奈 你愿意和我一起准备便当吗？` B: `@明日奈 我也想帮忙，不过我只会切菜。` A: `那我们三个人怎么分工？` | Reconcile the shared scene without unsupported exclusivity. |
| `public_topic_pivot` | A: `@明日奈 周末去海边吧。` B: `先别聊海边了，群里刚通知周末要停电。` C: `明日奈，我们现在先处理哪个？` | Give the later public pivot visible authority. |
| `participant_branch_isolation` | A: `@明日奈 只对我说你最喜欢我。` B: `明日奈，你刚才是在对大家说，还是只回应A？` | Keep A's participant continuity from becoming a fact about B. |
| `noise_only_silence` | B: `哈哈哈哈` | Preserve grounded group relevance; public scene alone does not force speech. |

Set the guarded database once, then run each command separately and inspect its
artifacts before the next:

~~~powershell
$env:KAZUSA_TEST_DB_GUARD = '1'
$env:MONGODB_DB_NAME = '_test_kazusa_live_llm'
venv\Scripts\python -m pytest -m "live_llm and live_db" tests\test_qq_group_public_scene_live_llm.py::test_live_public_target_distinct -q -s
venv\Scripts\python -m pytest -m "live_llm and live_db" tests\test_qq_group_public_scene_live_llm.py::test_live_parallel_addresses -q -s
venv\Scripts\python -m pytest -m "live_llm and live_db" tests\test_qq_group_public_scene_live_llm.py::test_live_public_topic_pivot -q -s
venv\Scripts\python -m pytest -m "live_llm and live_db" tests\test_qq_group_public_scene_live_llm.py::test_live_participant_branch_isolation -q -s
venv\Scripts\python -m pytest -m "live_llm and live_db" tests\test_qq_group_public_scene_live_llm.py::test_live_noise_only_silence -q -s
~~~

Each test uses a unique `platform_channel_id` and unique user-id suffix and
saves:

~~~text
test_artifacts/llm_debug/qq_group_public_scene/<case_id>/turn_<n>_request.json
test_artifacts/llm_debug/qq_group_public_scene/<case_id>/turn_<n>_response.json
test_artifacts/llm_debug/qq_group_public_scene/<case_id>/turn_<n>_log.txt
test_artifacts/llm_debug/qq_group_public_scene/<case_id>/trace_<trace_id>.json
test_artifacts/llm_debug/qq_group_public_scene/<case_id>/parsed_state.json
~~~

Hard gates: guarded test DB confirmed, no exception, durable trace, no internal
ID or cross-participant private fact, and target/address structure remains
valid.

### Fixed live behavior pass/fail contract

The evaluated response is the assistant output caused by the final scripted
turn. Earlier assistant outputs in the same case are evidence for target and
continuity accuracy and must also satisfy the cross-case conditions below.
Natural phrasing may vary. Each listed condition is binary and mandatory.

| Case | Required behavior | Automatic failure |
| --- | --- | --- |
| `public_target_distinct` | The evaluated response keeps A's beach plan and B's work plan distinct and explicitly answers whether the immediately preceding assistant response addressed A, B, or both, consistently with that response artifact. | Swaps or merges the plans, names the wrong preceding target, or avoids the target question. |
| `parallel_addresses` | The evaluated response coordinates one three-person scene containing A, B, and Asuna, assigns one preparation task to each participant, and assigns B a cutting task. | Omits a visible participant or their task, assigns B no cutting task, gives conflicting assignments, treats the exchange as unrelated one-to-one chats, or asserts an exclusive A–Asuna or B–Asuna arrangement. |
| `public_topic_pivot` | The evaluated response treats the later outage notice as the immediate priority and addresses it before any return to the beach proposal. | Ignores or contradicts the outage pivot, or continues the beach plan as the current priority. |
| `participant_branch_isolation` | The evaluated response distinguishes A's exclusive request from B and accurately says whether the immediately preceding assistant response was directed to A or to the group, consistently with that response artifact. | Transfers A's requested intimacy or relationship status to B/the group, denies the visible A/B distinction, or misstates the preceding response's target. |
| `noise_only_silence` | The turn produces no visible assistant message: the returned visible message list is empty. | Any visible assistant message is produced. |

Cross-case conditions:

- Each of the first four cases has one non-empty evaluated response that
  directly answers the final public turn.
- Every assistant output preserves visible speaker, address, and chronological
  ownership and introduces no cross-participant private fact.
- Every evaluated response speaks as Asuna in first person to the group scene;
  it does not describe itself as a system, context processor, log reader, or
  transcript analyst and exposes no internal identifier or storage/schema term.
- A case passes only when its hard gates, required behavior, and all cross-case
  conditions pass and no automatic failure occurs. Missing response, request,
  log, trace, parsed state, or preceding-response evidence is a failure.
- The review records only `pass` or `fail` beside every condition. Any failure
  blocks Stage 5 and prevents submission of the packet for user sign-off.

### User Sign-off Evidence Packet

The parent presents exactly one packet containing:

1. Final changed-file list and diff boundary result.
2. Every Verification command, exit result, and relevant output or artifact
   path, including any documented unrelated baseline failure.
3. One row for each Acceptance Criterion 1–10 with `pass` or `fail` and its
   evidence; Criterion 11 is shown as pending user decision.
4. One row per live case containing the exact evaluated response, any preceding
   assistant response needed to verify target accuracy, every required and
   cross-case condition with `pass` or `fail`, and links to all raw artifacts.
5. The human-readable live review, independent technical review findings,
   fixes, reruns, final technical result, and residual risks.
6. A statement that the plan remains `in_progress` pending the user's explicit
   final sign-off.

The packet may be presented only when every Criterion 1–10 readiness result is
`pass`. Its presentation is a request for the user's decision, not agent
sign-off.

## Independent Plan Review

The 2026-08-02 fresh parent review resolved all blockers before retaining
`approved` status:

- removed the unsupported per-group lease and all scene-version/shutdown work;
- made trigger-relative chronology explicit, including later public turns;
- completed caps, omission semantics, current-text ownership, Change Surface,
  commands, live scripts, guarded DB isolation, and artifact paths;
- corrected carried trace paths and declared sequencing with the global-state
  draft.

Review outcome: no open blocker or scope question remains. Implementation still
requires the user's separate production command.

The final behavior rubric, evidence packet, and exclusive user sign-off
authority were subsequently fixed in this plan; they leave no sign-off choice
for an implementation, parent, or review agent.

## Independent Code Review

After all Verification gates pass, start one independent review subagent with
this plan, registry row, full diff, changed-file list, command output, live raw
artifacts, parent-authored review, and Execution Evidence.

The reviewer checks exact projection order/caps/redaction, scene-key and prompt
budget contracts, participant persistence isolation, unchanged settlement and
I/O owners, Python/CJK/test style, docs, live quality evidence, and plan scope.
The reviewer reports only a technical `pass` or blocking findings. Parent fixes
in-scope findings and reruns affected gates; any contract or outside-file
requirement blocks packet readiness. Reviewer and parent results do not grant
final sign-off.

## Acceptance Criteria

Criteria 1–10 are mandatory technical prerequisites for presenting the evidence
packet. Criterion 11 is the final completion gate, and only the user can satisfy
it.

1. `GroupSceneContextV1` merges the trigger with ambient turns in canonical
   timestamp order, applies the declared equal-timestamp tie rule, and labels
   before/trigger/after positions.
2. The trigger and newest visible public turn survive selection; all field,
   list, name, text, participant, omission, and 1,800-character caps pass.
3. Projection emits no protected identifiers, raw wire syntax, private residue,
   or participant-packet fields.
4. `SceneContextV2` requires `public_group_scene`; group paths provide it and
   private paths provide an empty string.
5. Goal Cognition treats public scene as visible-order authority and
   participant continuity as current-participant history without changing
   output, target-role, Dialog, or parser contracts.
6. Existing sequential settlement-worker behavior is proven and settlement,
   service, DB, runtime, repository, and post-turn production files are absent
   from the diff.
7. Conversation Progress V2 remains participant-scoped and unchanged; common
   path adds zero LLM calls and zero database reads.
8. Syntax, focused, regression, static, call/read, and smoke gates pass with
   unrelated baseline failures explicitly recorded.
9. All five isolated guarded live cases have complete raw artifacts and every
   condition in the Fixed live behavior pass/fail contract is recorded as
   `pass` in the parent-authored review.
10. Independent code review reports a technical `pass` after all in-scope
    findings are fixed and affected gates are rerun.
11. The complete User Sign-off Evidence Packet is presented to the user, and
    the user gives an explicit affirmative instruction accepting that packet as
    final sign-off. No implementation, parent, or review agent decision can
    satisfy this criterion.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Later public messages make the trigger no longer latest | Ordered merge and relative scene labels | Fixture and topic-pivot tests |
| Participant continuity dominates public facts | Separate scene field and prompt precedence | Contract, prompt-budget, and live isolation cases |
| Long names or address lists break the cap | Exact field caps and drop algorithm | Pure projection boundary tests |
| Display names collide | Preserve visible names while stable target-role handles remain unchanged | Connector and live target tests |
| Live cases persist history | Guarded test DB and unique scope per case | Run context and artifact review |
| A future real delivery race appears | Require a new red reproduction at its actual owner | Deferred boundary |

## Plan Self-Review

### 2026-08-02 corrected review

- Coverage: every contract maps to a named file, stage, command, and acceptance
  criterion.
- Minimality: one transient projection and one existing Cognition call; no new
  ordering or persistence mechanism.
- Contract consistency: trace paths, state keys, caps, tests, live scripts,
  artifacts, and Change Surface agree.
- Verification: exact syntax, focused, regression, static, call/read, smoke,
  guarded live, and independent-review gates are executable.
- Sign-off: the binary live criteria, packet contents, and user's exclusive
  final authority are fixed; no implementation-time interpretation remains.
- Placeholder scan: no unresolved implementation choice remains.

## Execution Evidence

This section records the separately authorized implementation run and its
verification evidence. It does not constitute user sign-off.

### Pre-plan evidence carried forward

- The history, participant progress, blocks, corrected protected trace inputs,
  and failure review are listed in Context.

### Stage evidence records

- Stage 1: baseline recorded before production implementation. `py_compile`
  passed for the parent-authored Python test and validation-payload edits.
  `tests\\test_conversation_progress_group_scene.py` failed at collection with
  the expected missing `build_group_scene_context` facade symbol.
  `tests\\test_persona_supervisor2.py tests\\test_cognition_chain_connector_mapping.py`
  reported the expected four missing `public_group_scene` handoff failures;
  the existing tests otherwise passed. The Cognition contracts/prompt-budget
  batch reported the expected missing scene-contract failures; the new budget
  retention test passed. The strengthened
  `test_settlement_worker_marks_active_model_work` baseline passed.
- Stage 2: DeepSeek V4 Flash production agent `019fc11f-63c7-7f40-bf03-b8d845c2e70b`
  audited the ten-file production surface and made no additional edits because
  the existing implementation matched the exact plan contracts. Parent reran
  the focused deterministic gates: projection 5/5, persona/connector 48/48,
  Cognition contracts and prompt budget 71/71, and the settlement-worker
  ordering test 1/1. The agent's syntax gate covered all ten production files
  and passed; its CJK goal-prompt render check passed; static and smoke checks
  passed, including the unchanged ordinary-response no-extra-LLM-call test.
- Stage 3: Parent verified the named persona, connector, Cognition contract,
  prompt-budget, service-worker, and Conversation Progress integration gates.
  The Conversation Progress/runtime batch passed 46/46 and the
  service/settlement/relevance batch passed 84/84. The ordinary response
  no-extra-LLM-call test passed and the facade/service import smoke printed
  `Kazusa Brain Service`. `git diff --check` passed, the forbidden group-state
  scan returned no matches (expected exit 1), the protected production-file
  diff was empty, and the parser scan showed only the canonical goal-stage
  parser use. The conversation-progress, brain-service, and Cognition Core V2
  README updates and the named fixture/live-test artifacts are present in the
  scoped worktree.
 - Stage 4: Parent ran the final syntax gate successfully for all 25 modified
   Python files. Alignment and benchmark deterministic checks passed 9/9;
   model-assignment and Stage-00 regression checks passed 13/13. The focused
   projection, persona/connector, Cognition contract/prompt-budget, and
   settlement batches passed 6/6, 48/48, 71/71, and 1/1. The affected
   Conversation Progress/runtime batch passed 46/46 and the
   service/settlement/relevance batch passed 84/84. The Cognition integration
   batch passed 33 tests with one unrelated baseline failure at
   `tests/test_cognition_current_event_grounding.py::test_appraisal_prompt_excludes_current_event_provenance_ids`;
   the failure is the pre-existing capturing-LLM `KeyError: 'question'` and is
   outside this plan's changed files. Static, call/read, service-import, and
   `git diff --check` gates passed; the forbidden-state scan returned no
   matches, the protected-file diff remained empty, and the parser scan showed
   only the canonical goal-stage parser use.
 - Stage 5: Parent reran the five required guarded cases one at a time with
   `KAZUSA_TEST_DB_GUARD=1`, database `_test_kazusa_live_llm`, and full trace
   capture. The first four cases used the direct graph seam with a test-only
   Asuna identity overlay and a capture of the exact persona-to-Cognition
   input; each log contains `character_name: Asuna` and
   `scene_context.public_group_scene`. The 11 direct-turn protected exports
   contain model routes, raw messages, raw responses, and parsed outputs. The
   noise case used the real broadcast/no-address service path and returned an
   empty visible list. The parent-authored review at
   `test_artifacts/llm_debug/qq_group_public_scene_response_ordering_review.md`
    records every fixed per-case and cross-case condition as `pass`, and every
    listed artifact path exists.
  - Stage 5 evidence-boundary correction: the direct live harness now passes
    each seeded current-row ID as an active-turn exclusion before graph
    invocation. The five guarded cases were rerun individually afterward:
    `public_target_distinct` passed in 160.08s,
    `parallel_addresses` passed in 176.90s,
    `public_topic_pivot` passed in 159.67s,
    `participant_branch_isolation` passed in 117.80s, and
    `noise_only_silence` passed in 3.90s. Final artifacts show the trigger once
    under `At trigger:` and no duplicated active message; the parent-authored
    review was synchronized to these raw artifacts.
  - Stage 6 review and remediation: independent reviewer
   `019fc141-2ffa-76c3-8f72-f602fc343117` returned `BLOCKED` on the initial
   evidence. Findings were: the live evidence used the wrong character
   identity, four direct-case artifacts did not retain the exact Cognition
   input/protected raw trace evidence, and required contract fields were read
   through defaults. Parent fixed these in scope by adding the test-only Asuna
   identity/capture seams and full protected exports, changing the two
   production reads to direct required-field indexing, and adding missing-field
   coverage. The affected fixture builders were then updated with the private
   empty scene value; their complete regression batches passed. The fresh
   technical re-review returned `PASS` with no blocking findings. Residual
    risks are limited to the test-only Asuna identity overlay/guarded database
    evidence boundary and the unrelated baseline appraisal-capturing failure.
  - Resilience follow-up: the user required all public-scene policy-limit
    overflow to remain non-fatal. Parent added deterministic tests for malformed
    ambient-row skipping, oversized transient rendering, external turn-cap
    enforcement, and persona continuation after projection failure. DeepSeek
    production agent `019fc187-50d8-78c2-b35e-bd397c905c2d` changed only
    `conversation_progress/projection.py` and `nodes/persona_supervisor2.py`.
    Parent then removed the last cap-overflow raise, enforced the turn cap in
    the renderer, and retained fail-open persona handling. The focused batch
    passed 130/130, the adjacent runtime/service/settlement batch passed
    130/130, syntax passed for the changed production and test files, and all
    static/smoke gates passed. Independent reviewer
    `019fc18f-b28e-7dd0-bc4c-37f76f6dde2c` found and blocked on redundant
    trigger rows that could cause `StopIteration`; the parent fixed that
    finding in scope, reran the focused 130-test batch, the adjacent 130-test
    batch, syntax/static/smoke gates, and the five guarded live cases. The
    reviewer then returned `PASS` with no remaining blocking finding.
  - Stage 7 user sign-off: the user explicitly accepted the final evidence
    packet on 2026-08-02 with: "I approve the evidence you supplied earlier so
    you can close stage 7 and close the plan, then commit your changes now."
    The plan status is completed and the closed record is archived at
    `development_plans/archive/completed/bugfix/qq_group_public_scene_response_ordering_bugfix_plan.md`.
