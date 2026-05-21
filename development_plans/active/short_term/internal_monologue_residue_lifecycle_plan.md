# internal monologue residue lifecycle plan

## Summary

- Goal: Introduce an explicit internal monologue residue lifecycle that carries
  the reason behind the active character's mood across self-cognition, chat,
  speech, and dialog episodes without using `reflection_summary` as hidden
  live cognition carry-over.
- Plan class: high_risk_migration
- Status: draft
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `database-data-pull`, `py-style`, `test-style-and-execution`; apply
  `cjk-safety` before editing Python files containing CJK prompt text.
- Overall cutover strategy: staged compatible for storage and diagnostics,
  bigbang for prompt consumption once tests prove L2a-only residue loading.
- Highest-risk areas: letting stale private thought become a reason to speak,
  widening residue into L1/L2b/L2d/dialog, storing raw monologue as durable
  memory, and hiding the existing `reflection_summary` behavior instead of
  replacing it with an explicit contract.
- Acceptance criteria: current mood can be explained by a bounded causal
  residue card; L2a is the only direct live cognition consumer; downstream
  stages see only the current-turn derived monologue/stance/intent; legacy
  `reflection_summary` no longer acts as live carry-over; all verification and
  independent review gates pass.

## Context

Recent read-only MongoDB diagnostics showed the current failure mode.

Artifacts:

- `test_artifacts/diagnostics/current_character_runtime_state_fresh.json`
- `test_artifacts/diagnostics/current_self_cognition_event_log_events.json`
- `test_artifacts/diagnostics/current_self_cognition_action_attempts.json`
- `test_artifacts/diagnostics/current_self_cognition_scheduled_future_events.json`

The current singleton `character_state` contained:

```text
mood: 倦怠
global_vibe: 冷淡
reflection_summary:
杏山千纱观察到群内无人互动的沉闷氛围，对突如其来的单向分享保持旁观态度。
updated_at: 2026-05-20 21:49:49 +12:00
```

The self-cognition event log contained 150 recent self-cognition events from
`2026-05-20T08:50:02+12:00` through `2026-05-20T21:50:48+12:00`. Of those,
137 were `audit_only / silent / character_state_write=true`, and 13 were
`action_candidate / scheduled_action_request / character_state_write=true`.
This proves silent self-cognition is repeatedly shaping global runtime
character state.

The action-attempt ledger also showed actual private causal reasons, including:

```text
当前内心有强烈不安但已决定回避，依赖未来群聊信息才能进一步处理身份冒用问题

蚝爹油暂时离开群聊，当前无法继续调侃与赌约钉死；需要等他重新出现才能延续活跃节奏并落实挑战

截止时间已过且群聊无人提及，角色决定就此放下
```

The current system therefore has enough information to explain affective state,
but the information is split across `character_state.reflection_summary`,
self-cognition action attempts, source packets, and completed cognition state.
Normal `/chat` can see mood and global vibe, but it does not receive a
first-class explanation of why the character is in that state.

The target behavior is:

```text
self-cognition loop
  -> character becomes low, guarded, tired, anxious, or depressed for a reason
  -> recorder writes compact private causal residue
  -> next relevant self-cognition or /chat loads the residue into L2a
  -> L2a forms current internal_monologue using mood plus reason
  -> downstream stages receive only the current-turn interpretation
```

The target is not to make the character more likely to speak. The target is to
make the character's current mood intelligible to herself in the next
cognition.

## Mandatory Skills

- `development-plan-writing`: preserve lifecycle, progress checklist,
  verification gates, execution evidence, and review requirements.
- `local-llm-architecture`: keep local-model prompts bounded, semantic, and
  explicit; keep deterministic decisions outside prompts.
- `database-data-pull`: use read-only exports for production evidence and
  diagnostics; do not read `.env`.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files containing CJK prompt text.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Use PowerShell `-LiteralPath` for filesystem paths that may contain spaces;
  prefer repo-relative paths.
- Do not read `.env`.
- Do not mutate production data from tests.
- Do not store raw hidden chain-of-thought. In this system,
  `internal_monologue` means the explicit generated character-facing thought
  artifact produced by L2a, not provider-hidden reasoning.
- Do not persist full raw internal monologue as the carry-over object. Persist
  compact prompt-safe causal residue.
- Do not make residue an action trigger, delivery rule, permission rule,
  scheduler rule, adapter rule, or response-ratio control.
- Deterministic code owns scope matching, TTL, prompt budget, status
  transitions, collection access, and selection of at most the approved number
  of residue cards.
- LLM stages own semantic interpretation of why a completed cognition left
  unresolved private affect, but only inside the approved recorder contract.
- L2a Consciousness is the only direct live cognition consumer of
  prompt-facing internal monologue residue.
- L1, L2b, L2c1, L2c2, L2d, L3, dialog, adapter delivery, and scheduler
  dispatch must not receive raw prior residue directly.
- Downstream cognition may see residue influence only after L2a has rewritten
  it into current-turn `internal_monologue`, `logical_stance`, and
  `character_intent`.
- `reflection_summary` must not remain a live carry-over input once the new
  residue path is active. It may remain as legacy/audit character-state text
  until a later approved cleanup removes or redefines it.
- Raw reflection output must not enter normal cognition directly. This plan
  handles completed cognition residue, not raw reflection promotion.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or
  final reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the Independent Code Review gate and record the result
  in Execution Evidence.

## Must Do

- Add a first-class internal monologue residue module and storage contract.
- Define the residue as affective causal context: what the character feels,
  why she feels it, what private thread remains unresolved, and how it should
  be used as soft background.
- Support at least these residue scopes:
  - `global_private_state`
  - `self_cognition_chain`
  - `same_group_scene`
  - `same_user_thread`
- Load `global_private_state` residue into normal `/chat` when it is active,
  fresh, and prompt-budget eligible, so user messages can explain current mood
  caused by self-cognition.
- Feed the prompt-facing residue only to L2a Consciousness.
- Keep L1 current-stimulus focused and remove its dependence on
  `reflection_summary` as emotional carry-over.
- Keep L2b boundary judgment current-scene focused; do not let old residue
  create stale boundary defensiveness.
- Keep L2d action selection free of raw old residue; residue is not a reason
  to speak.
- Keep L3/dialog free of raw old residue; they consume only current-turn
  monologue and action directives.
- Add a residue recorder that consumes completed cognition/dialog/speech state
  and writes compact causal residue after the current episode completes.
- Use existing completed persona/self-cognition state as recorder input:
  `internal_monologue`, `emotional_appraisal`, `logical_stance`,
  `character_intent`, `judgment_note`, `boundary_core_assessment`,
  `social_distance`, `emotional_intensity`, `action_specs`,
  `action_results`, `surface_outputs`, `final_dialog`, trigger/source scope,
  and delivery/result status.
- Preserve the current bounded live response path. No new LLM call may block
  visible `/chat` response delivery without explicit user approval.
- Add experiment evidence under `experiments/` before production prompt
  rewiring, using exported diagnostics or dry-run artifacts to compare current
  `reflection_summary`, L2a-only residue, and no-residue cases.
- Add focused deterministic tests for scope selection, TTL, prompt projection,
  recorder output validation, and non-consumer boundaries.
- Add prompt contract tests proving the L2a payload receives residue and L1,
  L2b, L2d, L3, and dialog do not.
- Add self-cognition integration tests proving a silent self-cognition can
  write global private causal residue and a later `/chat` can load it into L2a.

## Deferred

- Do not redesign L1, L2b, L2c1, L2c2, L2d, L3, dialog, RAG, adapters, or the
  dispatcher beyond the specific residue contracts named in this plan.
- Do not add a response-ratio controller or deterministic silence/speak rule.
- Do not add a generic memory-provider abstraction.
- Do not add a broad emotional-state ontology beyond the fields listed in the
  residue data shape.
- Do not migrate existing historical `reflection_summary` values into residue
  cards unless a separate approved migration plan authorizes it.
- Do not expose raw message bodies, raw source packets, prompt text, or full
  generated dialog through ops endpoints or event logs.
- Do not remove the `reflection_summary` field from persisted
  `character_state` in this plan. This plan changes prompt consumption first.
- Do not make scheduled events carry raw residue. Scheduled future cognition
  may carry source refs and semantic follow-up context only through approved
  self-cognition contracts.
- Do not make dialog explain the residue directly unless L3 content anchors
  produced from the current L2 decision authorize that surface behavior.

## Cutover Policy

Overall strategy: staged compatible for storage and diagnostics, bigbang for
prompt consumption after verification.

| Area | Policy | Instruction |
|---|---|---|
| New residue collection/module | compatible | Add alongside existing `character_state`; tolerate empty or missing residue. |
| Recorder | compatible | Start by writing bounded residue without deleting `reflection_summary`. |
| L2a consumption | bigbang | Once tests pass, L2a consumes `internal_monologue_residue_context` as the only direct residue input. |
| L1 `reflection_summary` consumption | bigbang | Remove `reflection_summary` from L1 live carry-over when L2a residue is wired. Do not leave dual carry-over. |
| Existing `reflection_summary` storage | compatible | Keep the field for legacy/audit/runtime display; do not use it as prompt-facing carry-over. |
| Existing historical data | compatible | Do not backfill historical rows in this plan. Missing residue means empty context. |
| Tests | bigbang | Update prompt contract tests to enforce the new ownership boundary. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative dual-prompt strategy by
  default.
- If an area is `bigbang`, delete or rewrite live prompt references instead
  of preserving both old and new carry-over inputs.
- If an area is `compatible`, preserve only the compatibility surfaces
  explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local helper names only when the public contracts and
  ownership boundaries in this plan remain intact.
- The agent must not add feature flags, alternate fallback paths,
  compatibility prompt shims, broad registries, extra LLM calls, or unrelated
  cleanup.
- Changes outside the listed Change Surface require plan revision or explicit
  user approval.
- If the implementation discovers an existing equivalent helper, the agent may
  reuse or move it only when doing so stays inside the approved change surface.
- If the plan and code disagree, preserve the plan's stated intent and record
  the discrepancy before changing code.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

```text
completed cognition/chat/dialog/self-cognition episode
  -> internal_monologue_residue recorder
  -> compact scoped residue state
  -> next relevant trigger loads prompt-facing residue
  -> L2a consumes residue as soft causal background
  -> L2a emits current internal_monologue / stance / intent
  -> downstream stages consume only current derived fields
```

The residue answers:

```text
What does the character feel?
Why does she feel it?
What private thread remains unresolved?
How should the next cognition treat that reason without turning it into an
automatic action?
```

`mood` and `global_vibe` continue to answer what the character feels.
`internal_monologue_residue` answers why the character feels that way.
`conversation_progress` continues to answer what is happening in the external
conversation episode.
Durable memory/reflection continues to answer what should survive after this
short private state expires or is promoted.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Direct live consumer | L2a Consciousness only | L2a owns current-turn internal monologue, stance, and intent. |
| L1 role | Do not feed full residue to L1 | L1 should react to the current stimulus, not replay prior private narrative. |
| Boundary role | Do not feed raw residue to L2b | Old residue can create stale boundary defensiveness. |
| Action role | Do not feed raw residue to L2d | Residue is not a reason to speak or act. |
| Dialog role | Do not feed raw residue to dialog | Dialog should render directives, not reinterpret private carry-over. |
| Storage | Separate short-lived residue collection | `character_state.reflection_summary` is a global singleton and is too blunt for scoped causal carry-over. |
| Raw monologue | Do not persist raw monologue as the carry unit | The carry unit must be compact, prompt-safe, and scoped. |
| Recorder timing | Post-episode/background | Normal `/chat` response latency must stay bounded; self-cognition already runs outside user-visible latency. |
| Existing `reflection_summary` | Keep storage, remove prompt carry-over | Avoid data migration while eliminating unintended live behavior. |
| Self-cognition to chat | Allow `global_private_state` residue | The user's core case requires self-cognition affective reasons to be visible to later `/chat` L2a. |

## Contracts And Data Shapes

### Storage Document

Create a collection owned by the new module, tentatively:

```text
internal_monologue_residue_state
```

Conceptual document shape:

```python
{
    "residue_id": str,
    "character_id": str,
    "scope_kind": (
        "global_private_state"
        | "self_cognition_chain"
        | "same_group_scene"
        | "same_user_thread"
    ),
    "scope": {
        "platform": str,
        "platform_channel_id": str,
        "channel_type": str,
        "global_user_id": str,
        "source_chain_id": str,
    },
    "status": "active" | "settled" | "stale" | "expired",
    "affect": str,
    "causal_reason": str,
    "unresolved_private_thread": str,
    "settled_judgment": str,
    "trajectory": "worsening" | "steady" | "softening" | "resolved" | "unknown",
    "surface_policy": str,
    "carry_instruction": str,
    "source_refs": list[dict[str, str]],
    "created_at": str,
    "updated_at": str,
    "expires_at": str,
}
```

Hard caps:

- `affect`: 80 characters
- `causal_reason`: 240 characters
- `unresolved_private_thread`: 240 characters
- `settled_judgment`: 200 characters
- `surface_policy`: 160 characters
- `carry_instruction`: 160 characters
- `source_refs`: at most 5 sanitized refs, no message bodies

### Prompt-Facing Projection

L2a receives at most two compact cards, rendered as semantic context rather
than raw database rows:

```python
{
    "internal_monologue_residue_context": [
        {
            "scope_match": str,
            "affect": str,
            "causal_reason": str,
            "unresolved_private_thread": str,
            "settled_judgment": str,
            "trajectory": str,
            "carry_instruction": (
                "Use only as soft private background. Current input and "
                "current evidence remain primary."
            ),
        }
    ]
}
```

Forbidden prompt-facing fields:

- raw MongoDB `_id`
- scheduler ids
- action-attempt ids
- delivery metadata
- raw message bodies
- raw source packets
- raw full internal monologue from previous turns
- raw `reflection_summary` as a substitute for residue

### Public Module Interface

Create a module such as:

```text
src/kazusa_ai_chatbot/internal_monologue_residue/
```

Public facade:

```python
async def load_residue_context(
    *,
    trigger_scope: ResidueTriggerScope,
    current_timestamp_utc: str,
) -> ResidueLoadResult: ...

async def record_completed_episode_residue(
    *,
    completed_state: Mapping[str, Any],
    current_timestamp_utc: str,
) -> ResidueRecordResult: ...

def project_residue_for_l2a(
    *,
    cards: Sequence[InternalMonologueResidue],
    budget_chars: int,
) -> list[dict[str, str]]: ...
```

Runtime code must import from the module facade, not storage internals.

### Lifecycle Status

- `active`: eligible for prompt projection.
- `settled`: retained for audit/debug but not prompt-projected.
- `stale`: retained until TTL cleanup but not prompt-projected.
- `expired`: not prompt-projected and eligible for pruning.

Default TTLs:

- `global_private_state`: 24 hours
- `self_cognition_chain`: 24 hours or until superseded by the next chain card
- `same_group_scene`: 12 hours
- `same_user_thread`: 48 hours

The implementation may use shorter TTLs if focused tests prove the card is
suppressed correctly; it must not use longer TTLs without plan revision.

## LLM Call And Context Budget

Before:

- Normal `/chat` enters L1 with `character_profile.reflection_summary` as live
  affective background.
- L2a does not receive a first-class causal residue card.
- Self-cognition consolidation can update `mood`, `global_vibe`, and
  `reflection_summary`.

After:

- Normal `/chat` and self-cognition load at most two residue cards through
  deterministic scope selection.
- L1 no longer receives `reflection_summary` as live carry-over.
- L2a receives `internal_monologue_residue_context`.
- Recorder runs after episode completion. It may reuse existing consolidation
  model work or run as background-only work. It must not add a new blocking
  response-path LLM call without explicit user approval.

Budget:

- Prompt-facing residue budget: maximum 900 characters total in L2a human
  payload.
- Storage strings are capped before persistence.
- Normal response-path LLM call count must not increase.
- Self-cognition background call count may increase by one only if the
  implementation records explicit before/after runtime budget and tests prove
  normal `/chat` latency is unaffected.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/internal_monologue_residue/README.md`
  - Subsystem ICD, lifecycle, data ownership, prompt-facing projection,
    recorder/loader contracts, and non-consumer boundaries.
- `src/kazusa_ai_chatbot/internal_monologue_residue/__init__.py`
  - Public facade exports.
- `src/kazusa_ai_chatbot/internal_monologue_residue/models.py`
  - TypedDict/dataclass contracts and validation constants.
- `src/kazusa_ai_chatbot/internal_monologue_residue/projection.py`
  - Prompt-facing projection and character-budget enforcement.
- `src/kazusa_ai_chatbot/internal_monologue_residue/storage.py`
  - DB facade integration only; no raw MongoDB access outside DB-owned helpers
    if new helpers are added.
- `src/kazusa_ai_chatbot/internal_monologue_residue/loader.py`
  - Scope matching, TTL filtering, ranking, and load result.
- `src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py`
  - Completed-episode residue candidate generation, validation, and write
    coordination.
- `experiments/internal_monologue_residue_probe.py`
  - Diagnostic experiment comparing current `reflection_summary`, L2a-only
    residue, and no-residue prompt payloads using exported artifacts.
- `tests/test_internal_monologue_residue_projection.py`
  - Projection caps, forbidden fields, prompt shape.
- `tests/test_internal_monologue_residue_loader.py`
  - Scope selection, TTL, ranking, status filtering.
- `tests/test_internal_monologue_residue_recorder.py`
  - Completed-state validation and compact causal card output.
- `tests/test_internal_monologue_residue_prompt_boundaries.py`
  - L2a receives residue; L1/L2b/L2d/L3/dialog do not.

### Modify

- `src/kazusa_ai_chatbot/db/README.md`
  - Add collection ownership for `internal_monologue_residue_state`.
- `src/kazusa_ai_chatbot/db/__init__.py` and a narrow DB submodule
  - Add named semantic helpers for residue read/write if the new module
    persists through DB facade.
- `src/kazusa_ai_chatbot/service.py` or the relevant brain-service post-turn
  module
  - Load residue before persona cognition and record residue after completed
    response/private finalization.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
  - Integrate recorder if post-turn is the selected owner for normal `/chat`.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Load residue for self-cognition state and record residue after the
    completed self-cognition episode.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Wire production worker telemetry/result handling if needed.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
  - Remove live `reflection_summary` carry-over from L1 prompt input.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - Add `internal_monologue_residue_context` to L2a payload and prompt
    instructions.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
  - Stop treating next-round live psychological background as
    `reflection_summary`; if reused for recorder, output compact residue
    fields through the new contract.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - Keep `reflection_summary` persistence compatible, but do not make it the
    live carry-over path.
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Document self-cognition residue production and consumption.
- `src/kazusa_ai_chatbot/nodes/README.md`
  - Document L2a-only residue consumption and the non-consumer boundaries.
- `tests/test_cognition_live_llm_prompt_contracts.py`
  - Update prompt contract assertions if they currently require
    `reflection_summary` in L1.
- `tests/test_cognition_prompt_contract_text.py`
  - Add static prompt-boundary assertions.
- `tests/test_self_cognition_integration.py`
  - Add focused integration coverage for self-cognition residue handoff.
- `tests/test_service_background_consolidation.py`
  - Add normal `/chat` post-turn recorder/load integration coverage if this
    is the selected service integration point.

### Keep

- `src/kazusa_ai_chatbot/dialog_agent.py`
  - Dialog must not consume raw residue directly.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
  - L2d must not consume raw residue directly.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2c2.py`
  - L2c2 must not consume raw residue directly in this plan.
- `src/kazusa_ai_chatbot/conversation_progress/`
  - Keep external conversation-flow memory separate from private affective
    causal residue.
- Adapter implementations
  - No adapter changes are needed.
- Scheduler/dispatcher behavior
  - No delivery, permission, or scheduling behavior changes are authorized.

## Data Migration

No production data mutation is approved by this draft.

Storage migration is additive:

- create or bootstrap `internal_monologue_residue_state` indexes;
- tolerate the collection being empty;
- do not backfill old `reflection_summary` rows;
- do not delete `reflection_summary` from `character_state`.

Optional cleanup after future approval:

- inspect whether any runtime prompt still receives `reflection_summary`;
- if no live consumer remains, a later plan may redefine
  `reflection_summary` as an audit/display field or remove it from runtime
  profile projection.

## Overdesign Guardrail

- Actual problem: the character can carry mood from repeated self-cognition
  into `/chat`, but the next cognition lacks the private causal reason behind
  that mood unless it reads the overloaded `reflection_summary` singleton.
- Minimal change: add a scoped, short-lived causal residue card and feed it
  only to L2a, while removing `reflection_summary` from live carry-over.
- Ownership boundaries: deterministic code selects eligible residue and
  enforces limits; recorder LLM or existing consolidation logic summarizes
  completed cognition into compact causal fields; L2a interprets residue for
  the current turn; downstream stages consume only current-turn derived
  cognition.
- Rejected complexity: no action trigger, no response-ratio control, no raw
  monologue persistence, no direct L1/L2b/L2d/L3/dialog consumption, no
  scheduled-event residue payload, no adapter changes, no global emotional
  ontology, and no broad memory-provider abstraction.
- Evidence threshold: add new consumers, longer TTLs, a richer emotion schema,
  or direct boundary/social residue only after focused experiments or
  production diagnostics show L2a-only causal residue fails to preserve needed
  continuity.

## Implementation Order

1. Independent plan review and approval.
   - Review this draft against the self-cognition README, nodes README, DB
     README, conversation-progress README, and current diagnostics.
   - Resolve blockers before production code changes.
2. Stage 0 experiment.
   - Add or run `experiments/internal_monologue_residue_probe.py`.
   - Use current diagnostic artifacts to compare current
     `reflection_summary`, L2a-only residue, and no-residue prompt payloads.
   - Record whether the L2a-only shape carries causal reasons without
     increasing action pressure.
3. Add module contract tests before implementation.
   - Add projection, loader, and recorder tests.
   - Run them and record expected missing-module failures.
4. Implement the residue module contract.
   - Add models, projection, loader, recorder skeleton, validation, and caps.
   - Keep persistence behind public DB helpers.
5. Add DB facade helpers and bootstrap/index support.
   - Add semantic read/write helpers for residue state.
   - Update DB README with collection ownership.
6. Wire loader into normal `/chat` state.
   - Load eligible residue before cognition.
   - Do not add response-path LLM calls.
7. Wire loader into self-cognition state.
   - Load `global_private_state`, `self_cognition_chain`, or relevant scoped
     residue based on the case.
8. Wire L2a prompt consumption.
   - Add `internal_monologue_residue_context` to L2a payload.
   - Add explicit prompt guidance: soft background only, current input
     primary, residue is not action pressure.
9. Remove `reflection_summary` live carry-over from L1.
   - Update L1 prompt and prompt contract tests.
   - Keep character-state storage compatible.
10. Wire post-episode recorder.
    - Normal `/chat`: record after completed persona state and final visible
      or private surface result exists.
    - Self-cognition: record after completed self-cognition cognition/action/
      dialog state exists.
11. Add telemetry.
    - Record sanitized event-log labels for residue loaded/written/skipped,
      scope kind, status, and disposition. Do not log residue text.
12. Run focused tests and prompt contract tests.
13. Run adjacent cognition, self-cognition, and service tests.
14. Run independent code review.

## Progress Checklist

- [ ] Stage 1 - plan review complete
  - Covers: Implementation Order step 1.
  - Verify: independent plan review records no approval blockers.
  - Evidence: record reviewer mode and findings in Execution Evidence.
  - Sign-off: `<agent/date>`.

- [ ] Stage 2 - experiment evidence complete
  - Covers: step 2.
  - Verify: experiment output compares current `reflection_summary`,
    L2a-only residue, and no-residue payloads using exported diagnostics.
  - Evidence: record artifact path and summarized findings.
  - Sign-off: `<agent/date>`.

- [ ] Stage 3 - module contract tests red
  - Covers: step 3.
  - Verify: focused tests fail for missing symbols or missing behavior before
    implementation.
  - Evidence: record failing command output.
  - Sign-off: `<agent/date>`.

- [ ] Stage 4 - residue module contract implemented
  - Covers: step 4.
  - Verify: projection, loader, and recorder unit tests pass.
  - Evidence: record changed files and test output.
  - Sign-off: `<agent/date>`.

- [ ] Stage 5 - DB facade and bootstrap wired
  - Covers: step 5.
  - Verify: DB helper tests pass without raw MongoDB access outside DB-owned
    internals.
  - Evidence: record tests and static import greps.
  - Sign-off: `<agent/date>`.

- [ ] Stage 6 - loader integrated into chat and self-cognition
  - Covers: steps 6-7.
  - Verify: service and self-cognition integration tests prove eligible
    residue is loaded and missing residue degrades to empty context.
  - Evidence: record focused test output.
  - Sign-off: `<agent/date>`.

- [ ] Stage 7 - L2a prompt consumption wired
  - Covers: step 8.
  - Verify: prompt contract tests prove only L2a receives
    `internal_monologue_residue_context`.
  - Evidence: record prompt-render checks.
  - Sign-off: `<agent/date>`.

- [ ] Stage 8 - legacy L1 carry-over removed
  - Covers: step 9.
  - Verify: static greps and tests prove L1 no longer uses
    `reflection_summary` as live carry-over.
  - Evidence: record grep and test output.
  - Sign-off: `<agent/date>`.

- [ ] Stage 9 - recorder integrated
  - Covers: step 10.
  - Verify: completed chat and self-cognition episodes write compact causal
    residue with validated caps.
  - Evidence: record focused integration output.
  - Sign-off: `<agent/date>`.

- [ ] Stage 10 - telemetry complete
  - Covers: step 11.
  - Verify: event-log tests prove no residue text, prompt text, or raw message
    bodies are logged.
  - Evidence: record event-log test output.
  - Sign-off: `<agent/date>`.

- [ ] Stage 11 - full verification complete
  - Covers: steps 12-13.
  - Verify: every Verification command passes or has an approved blocker.
  - Evidence: record command outputs.
  - Sign-off: `<agent/date>`.

- [ ] Stage 12 - independent code review complete
  - Covers: step 14.
  - Verify: review findings are closed or explicitly accepted as residual
    risk.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, and
    approval status.
  - Sign-off: `<agent/date>`.

## Verification

Run from repository root with the project virtual environment.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\internal_monologue_residue\models.py src\kazusa_ai_chatbot\internal_monologue_residue\projection.py src\kazusa_ai_chatbot\internal_monologue_residue\loader.py src\kazusa_ai_chatbot\internal_monologue_residue\recorder.py`

Expected: all listed files compile.

### Static Greps

- `rg -n "internal_monologue_residue_context" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\nodes\dialog_agent.py`

Expected: no matches. Exit code `1` is acceptable.

- `rg -n "reflection_summary|character_reflection_summary" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py`

Expected: no live prompt consumption of `reflection_summary`. Matches are
allowed only for comments explaining decommissioned behavior if such comments
are approved.

- `rg -n "scheduled_events.*internal_monologue_residue|internal_monologue_residue.*scheduled_events" src\kazusa_ai_chatbot`

Expected: no scheduled-event raw residue payload path.

- `rg -n "_id|action_attempt_id|delivery_tracking_id|adapter_message_id" src\kazusa_ai_chatbot\internal_monologue_residue\projection.py`

Expected: no prompt-facing projection of raw storage/action/delivery ids.

- `git diff --check`

Expected: no whitespace errors.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_internal_monologue_residue_projection.py -q`
- `venv\Scripts\python -m pytest tests\test_internal_monologue_residue_loader.py -q`
- `venv\Scripts\python -m pytest tests\test_internal_monologue_residue_recorder.py -q`
- `venv\Scripts\python -m pytest tests\test_internal_monologue_residue_prompt_boundaries.py -q`

Expected: all pass after implementation. The initial red run should fail only
for missing planned symbols or missing planned behavior.

### Integration Tests

- `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py -q`
- `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py -q`
- `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py -q`
- `venv\Scripts\python -m pytest tests\test_cognition_live_llm_prompt_contracts.py -q`

Expected: all non-live tests pass. Live LLM tests, if selected from these
files, must be run one case at a time with output inspected according to the
test skill.

### Experiment

- `venv\Scripts\python experiments\internal_monologue_residue_probe.py`

Expected: writes an artifact under `test_artifacts/diagnostics/` or
`test_artifacts/experiments/` comparing current `reflection_summary`,
L2a-only residue, and no-residue payloads. The artifact must not include raw
secrets, embeddings, base64 media, or full private transcripts.

### Live DB Diagnostics

Use only when MongoDB is available and the user authorizes production
inspection:

- Export latest `character_state` singleton.
- Export bounded recent self-cognition event counts.
- Export bounded recent self-cognition action-attempt reasons.
- Export bounded recent `internal_monologue_residue_state` rows after
  implementation.

Expected after implementation:

- active residue rows exist only when recorder produced validated compact
  causal cards;
- prompt-facing projection excludes raw ids and message bodies;
- a self-cognition-created `global_private_state` residue can be selected for
  later `/chat`.

Do not print raw private message bodies or raw source packets.

## Independent Plan Review

Run this gate before approving this draft for implementation. Prefer a
reviewer that did not draft the plan. If no separate reviewer is available, the
active agent must reread this plan, the self-cognition README, nodes README,
DB README, conversation-progress README, and current diagnostic artifacts from
a fresh-review posture.

Review scope:

- The plan addresses emotional causality, not just mood intensity.
- The plan keeps L2a as the only direct live cognition consumer.
- The plan removes `reflection_summary` from live carry-over instead of adding
  a second parallel carry-over input.
- The plan does not let residue become action pressure, response gating,
  delivery logic, scheduler logic, or dialog content by itself.
- The storage shape is bounded, prompt-safe, short-lived, and scoped.
- The self-cognition-to-chat use case is covered by `global_private_state`.
- Experiment evidence precedes production prompt rewiring.
- Verification commands cover consumer boundaries, prompt shape, storage caps,
  and self-cognition integration.

Record blockers, non-blocking findings, required edits, and approval status in
Execution Evidence. Do not approve while blockers remain.

## Independent Code Review

Run this gate after all Verification commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer
is available, the active agent must reread this plan and inspect the full diff
from a fresh-review posture.

Review scope:

- Project rules and style compliance for changed Python, tests, prompts,
  documentation, scripts, and diagnostic artifacts.
- Plan alignment with Must Do, Deferred, L2a-only consumption, storage
  contract, cutover policy, and verification gates.
- Code quality and design risk, especially hidden fallback paths, stale
  `reflection_summary` prompt use, raw id leakage, raw monologue persistence,
  direct L2d/dialog consumption, response-path LLM call increases, and
  accidental adapter/scheduler changes.
- Regression quality, including focused residue tests, prompt contract tests,
  self-cognition integration tests, service post-turn tests, and sanitized DB
  diagnostics.

Fix findings only when the fix is inside this plan's approved change surface.
If a finding requires a new public contract or broader architecture, stop and
revise the plan before changing code.

## Acceptance Criteria

This plan is complete when:

- `internal_monologue_residue_state` or its approved equivalent exists as a
  separate short-lived residue lane.
- The residue card carries compact affective causality: affect, causal reason,
  unresolved private thread, settled judgment, trajectory, and carry policy.
- Residue scope matching supports `global_private_state`,
  `self_cognition_chain`, `same_group_scene`, and `same_user_thread`.
- A silent self-cognition episode can write a `global_private_state` residue
  explaining the reason behind a low mood such as `倦怠 / 冷淡`.
- A later `/chat` can load that active residue into L2a.
- L2a is the only direct live cognition consumer of prompt-facing residue.
- L1 no longer uses `reflection_summary` as live emotional carry-over.
- L2b, L2d, L3, dialog, adapters, and scheduler do not receive raw prior
  residue directly.
- Downstream stages receive residue influence only through current-turn
  `internal_monologue`, `logical_stance`, `character_intent`, judgment, social
  fields, and selected surface directives.
- Normal `/chat` response-path LLM call count does not increase.
- Prompt-facing residue projection stays within the approved character budget
  and excludes raw ids, raw source packets, raw previous monologue, and raw
  message bodies.
- `reflection_summary` remains storage-compatible but is no longer the live
  carry-over mechanism.
- All Verification gates pass.
- Independent code review approves the implementation.
- Registry and plan lifecycle records are updated according to
  `development_plans/README.md`.

## Plan Self-Review

Performed by Codex on 2026-05-20 while drafting the plan.

- Coverage: the plan carries forward the user's clarified requirement that the
  system carry the reason behind mood, not merely the mood label.
- Evidence: the plan cites current read-only DB artifacts showing
  `倦怠 / 冷淡`, a reason-bearing `reflection_summary`, repeated silent
  self-cognition character-state writes, and action-attempt reasons.
- Minimality: direct live consumption is limited to L2a; new storage is
  short-lived and scoped; downstream stages remain derived consumers.
- Placeholder scan: no unfinished placeholder token or open option remains.
  The plan is still `draft` because owner approval is required before
  implementation.
- Contract consistency: data shape, change surface, implementation order,
  verification, and acceptance criteria all use the same residue vocabulary.
- Verification: static greps, focused tests, integration tests, experiment
  evidence, live DB diagnostics, and independent review gates map to the
  identified risks.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Residue becomes a hidden reason to speak | L2d does not receive raw residue; prompt says current scene remains primary | Prompt-boundary tests and L2d static grep |
| Stale private reason colors unrelated chats | Deterministic scope, TTL, status, and ranking before projection | Loader tests |
| L1 overreacts to old residue | L1 does not consume residue or `reflection_summary` carry-over | L1 static grep and prompt tests |
| Raw monologue becomes durable memory | Store compact causal cards only, with caps and forbidden fields | Recorder/projection tests |
| Dialog exposes private reasoning | Dialog receives only current directives; raw residue is excluded | Dialog payload tests/static grep |
| Existing valid character state display breaks | Keep `reflection_summary` storage-compatible | Service/consolidation tests |
| Normal chat latency increases | Recorder runs post-episode/background; no new response-path LLM call | LLM budget review and service tests |
| Group/private scope leakage | Scope matching is deterministic and tested | Loader tests |
| Event logs leak sensitive text | Log labels/status only, not residue text | Event-log tests |

## Execution Evidence

No implementation has started. Record plan review, experiment output, red/green
test results, static greps, live DB diagnostics, code review, and any approved
follow-up evidence here during execution.

## Glossary

- `internal_monologue`: the explicit generated character-facing thought
  artifact produced by L2a for the current cognition.
- `internal_monologue_residue`: compact private causal carry-over distilled
  from completed cognition/dialog/self-cognition state.
- `affective causal residue`: the reason behind the current affect, including
  unresolved private thread and carry policy.
- `global_private_state`: character-level private residue created by
  self-cognition that may explain mood in later `/chat`.
- `self_cognition_chain`: residue scoped to a scheduled or repeated
  self-cognition continuation.
- `same_group_scene`: residue scoped to a group channel scene.
- `same_user_thread`: residue scoped to one user's short private or channel
  interaction.
- `reflection_summary`: legacy/current singleton character-state summary that
  must stop acting as live cognition carry-over in this plan.
