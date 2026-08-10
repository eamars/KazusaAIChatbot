# Conversation Progress V2 Long-Thread Continuation Big-Bang Plan

> Superseded on 2026-07-30 by
> `development_plans/active/bugfix/conversation_progress_v2_final_signoff_plan.md`.
> Preserve this file as historical implementation and execution evidence.

## Summary

- Goal: evolve the existing `kazusa_ai_chatbot.conversation_progress` subsystem in place so long, interleaved conversations retain what has already happened, what remains open, and what may be deliberately reopened, without a keyword capture or suppression system.
- Plan class: `high_risk_migration`.
- Status: `superseded`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `debug-llm`, `character-test`, `database-data-pull`, `py-style`, `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: forward-only big-bang replacement of the internal conversation-progress schema, recorder contract, history assembly, persistence, prompt projection, cognition evidence projection, tests, and active short-term rows while retaining the existing public `load_progress_context(...)` and `record_turn_progress(...)` facade names.
- Highest-risk areas: semantic event loss during repeated compaction; summary drift; invalid source lineage; group-chat noise evicting the active participant thread; weak-model event-delta errors; prompt and recorder budget growth; active-row cutover; retrieval crossing its evidence boundary; and overlap with the in-progress character identity work.
- Acceptance criteria: the exact Asuna adjacent-history regression reaches cognition with the earlier completed action as citeable evidence; deliberate reopening remains allowed; participant continuity survives segmented assistant output, unrelated group traffic, and 20/50/100-turn compaction cases; every event has valid source lineage; all continuation inputs remain bounded; the ordinary response path adds no LLM call; and no production behavior depends on `后颈` or any other case-specific term.
- Execution authority: approved and execution commanded. Fallback single-agent execution authorized. Live database migration apply remains separately gated.

## Context

The visible failure occurred in QQ channel `638473184`, between Asuna and QQ user `673225019`:

1. The user asked `想被我摸摸哪里呀？`.
2. Asuna selected `后颈`.
3. The user accepted and performed the neck massage.
4. The exchange advanced to `耳根`, which the user also completed.
5. When asked for the next location, Asuna proposed `后颈` again.

The adjacent conversation, production history window, stored episode state, and protected trace were already pulled and reviewed. The evidence is recorded in:

- `test_artifacts/asuna_qq_638473184_houjing_context.json`;
- `test_artifacts/asuna_qq_638473184_houjing_adjacent.json`;
- `test_artifacts/asuna_qq_638473184_houjing_repeat_trace_review.json`;
- `test_artifacts/asuna_qq_638473184_conversation_episode_state_current.json`;
- `test_artifacts/reviews/asuna_long_thread_conversation_memory_evaluation.md`; and
- `test_artifacts/reviews/conversation_flow_continuation_system_recommendation.md`.

The evidence identifies two independent information-loss boundaries:

1. Recent history is counted as storage rows, not logical speaker turns. `CONVERSATION_HISTORY_LIMIT` is `10`, `CHAT_HISTORY_RECENT_LIMIT` is `5`, history is fetched before the current user row is stored, and `persona_supervisor2(...)` filters the already-small channel window before slicing it again. One multi-fragment assistant response can therefore consume most of the semantic window.
2. The stored progress document can retain useful facts that never reach cognition. The incident's episode row retained completion information in `resolved_threads` and `user_state_updates`, while `_conversation_progress_text(...)` omits both fields and then caps the remaining string at 1,000 characters.

Stage 12 live replay exposed a third, architectural boundary failure in the
first V2 implementation:

1. The recorder prompt required one local LLM to reconstruct scene semantics,
   update event lifecycle, copy persistence IDs and timestamped source objects,
   choose discard candidates, author future progression guidance, and produce
   block-compaction instructions in one response.
2. Six consecutive recorder checkpoints required the structural repair prompt
   before acceptance. The repair path had therefore become normal-path
   scaffolding rather than an exceptional recovery boundary.
3. The exact completed-event evidence reached cognition, but cognition selected
   the completed action again. This proved that persistence must expose facts
   while cognition remains the only owner of the next response goal.

The user rejected that responsibility split during Stage 12. The first
remediation still left one oversized recorder prompt responsible for scene
summarization, episode movement, every prior-event comparison, new-event
extraction, lifecycle, relevance, obligation direction, and outcome. It also
left selection-required goal cognition dependent on a second semantic verifier
and replacement loop. Frozen failures proved both designs remained ambiguous:
an omitted prior event was indistinguishable from an intentional unchanged
event, while the verifier accepted a replacement that selected a completed
event.

The canonical contract below replaces both boundaries before closeout:

- one scene observer reports scene facts only;
- one event reconciler reports exactly one observation for every supplied
  prior event plus separately listed new events;
- deterministic code validates exact handle coverage, maps identities,
  lifecycle, retention, lineage, limits, compaction, and persistence;
- one selection-goal producer makes the actual character-owned choice and
  explicitly accounts for every supplied conversation event; and
- no semantic evaluator or replacement model repairs either producer's core
  judgment.

On 2026-07-30 the user explicitly moved the bugfix acceptance boundary to the
conversation-progress handoff. This plan passes when the source-faithful key
event details survive storage/compaction and appear intact in cognition's
projected evidence and actual goal-prompt input. A later cognition choice or
final dialog is separate cognition-quality evidence and cannot fail this
conversation-progress bugfix.

The current prompt-facing progress document is capped at 5,000 characters, but the overflow policy removes `resolved_threads` first and later removes `avoid_reopening`. Those are precisely the classes that protect an active episode from resetting completed or closed state.

The observed failure trace places ordinary goal cognition at 11,094 prompt characters under the existing 24,000-character goal-cognition cap. Replacing the current 1,000-character continuation string with a combined 4,000-character scene-plus-event allowance adds at most 3,000 characters in this observed case, leaving approximately 9,900 characters of headroom without raising the existing goal cap.

The completed plans below remain historical contracts and are not reopened:

- `development_plans/archive/completed/short_term/conversation_progress_state_plan.md`;
- `development_plans/archive/completed/short_term/conversation_progress_flow_phase2_plan.md`;
- `development_plans/archive/completed/short_term/conversation_progress_phase3_quality_plan.md`;
- `development_plans/archive/completed/bugfix/logical_dialog_message_receipt_plan.md`;
- `development_plans/archive/completed/bugfix/conversation_episode_state_lane_lifecycle_plan.md`; and
- `development_plans/archive/completed/bugfix/rag_conversation_evidence_current_episode_boundary_bugfix_plan.md`.

This plan supersedes the active semantic-state shape and projection policy from those completed plans. It preserves their stable ownership decisions:

- conversation progress is short-lived operational memory;
- relevance decides whether a response episode is eligible before progress is loaded;
- cognition decides stance and progression;
- dialog owns final wording;
- deterministic code owns structure, source identity, limits, ordering, persistence, and expiry; and
- LLM stages own semantic equivalence and factual observations; deterministic
  code owns lifecycle/retention mapping and compaction.

The current worktree contains an in-progress character identity-growth change that overlaps `service.py`, database schemas/bootstrap, persona cognition, and tests. This plan is sequenced after `cognition_core_v2_character_identity_growth_bigbang_plan.md` reaches `completed`. The draft short-horizon global-state composition plan remains later work and must rebaseline its conversation-progress assumptions after this plan completes.

This is an in-place improvement of `conversation_progress`. It is not a new parallel memory system or a replacement facade. The new `conversation_episode_blocks` collection is an internal tier owned by the same subsystem.

### Evidence and research grounding

The design follows the local evidence above and these primary research results:

- [OpenAI, Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/) documents threshold-triggered compaction into a smaller representative context.
- [Liu et al., Lost in the Middle](https://arxiv.org/abs/2307.03172) shows that merely expanding a long prompt does not make middle-position evidence reliably usable.
- [Wu et al., LongMemEval](https://arxiv.org/html/2410.10813) separates indexing, retrieval, and reading and reports advantages for conversation-round granularity over whole-session storage.
- [Packer et al., MemGPT](https://arxiv.org/abs/2310.08560) demonstrates a bounded working set backed by hierarchical memory tiers.
- [Zhang et al., SummN](https://aclanthology.org/2022.acl-long.112/) shows that staged split-then-summarize processing can handle long dialogue with fixed per-call input.

The transferable architecture is hierarchical, source-backed, selectively compacted continuity. This plan does not assume access to Codex's opaque compaction representation.

## Mandatory Skills

- `development-plan`: governs approval, execution, lifecycle updates, evidence, review, and sign-off.
- `local-llm-architecture`: governs local-model contracts, prompt ownership, response-path call counts, budgets, retries, and blast radius.
- `debug-llm`: governs before/after artifacts, live LLM execution, trace inspection, and parent-authored human-readable quality reviews.
- `character-test`: governs adaptive multi-turn service-level behavior testing with real state and trace inspection.
- `database-data-pull`: governs any fresh read-only pull of conversation history, episode state, or compacted blocks.
- `py-style`: load before every Python production or test edit.
- `cjk-safety`: load before editing Python prompts or fixtures containing Chinese or Japanese text.
- `test-style-and-execution`: governs test-first implementation, deterministic versus live tests, one-case-at-a-time live execution, and result inspection.

## Mandatory Rules

1. Plan is approved and `in_progress`. Production changes follow the implementation order and verification gates below.
2. Execution remains blocked until the in-progress character identity-growth plan is completed, signed, and committed or otherwise placed on a clean preserved baseline.
3. At execution start, rerun `git status --short`, reread the root and relevant subsystem documentation, and rebaseline every named file and symbol. A material contract conflict requires a plan update before code changes.
4. Preserve all existing user work. The current uncommitted `source_episode_id` lineage work is outside this contract and is not a prerequisite for logical-turn grouping.
5. Keep `kazusa_ai_chatbot.conversation_progress` as the sole owner of short-term conversation continuation and retain the public facade names `load_progress_context(...)` and `record_turn_progress(...)`.
6. Create no parallel continuation module, dual read, dual write, runtime V1 mapper, alias vocabulary, feature flag, or compatibility shim.
7. Implement the V2 caller, callee, schema, repository, recorder, projection, cognition connector, retrieval extension, tests, and ICD changes in one canonical cutover.
8. Do not add keyword, regex, body-part, phrase, language-specific, or domain-specific capture or suppression logic. Production code must not branch on `后颈`, `耳根`, `摸`, massage wording, or equivalent paraphrases.
9. The post-turn semantic boundary has two independent owners. The scene
   observer owns only factual scene relation, episode movement, narrative,
   thread, stance, user goal, blocker, emotional trajectory, and overused
   moves. The event reconciler owns only semantic event equivalence,
   obligation direction, concrete actor/action/object identity, outcome, one
   compact `lifecycle_change`, and one compact `relevance` classification. It
   emits exactly one `unchanged` or `changed` observation for every supplied
   prior event handle, then emits new events in a separate list. Neither owner
   emits persisted enums, persistence IDs, timestamps, source-reference
   objects, discard/compaction instructions, future response guidance, or
   next affordances.
10. Deterministic code owns participant scope, logical-turn grouping, short
    handle assignment and resolution, stable IDs, exact source references,
    timestamp preservation, lifecycle/retention state-machine mapping, enum
    and shape validation, source-ref merging, capacity eviction, compaction
    selection and block construction, budgets, guarded writes, expiry,
    ordering, retrieval limits, and telemetry.
11. A completed, rejected, corrected, or superseded event is cognition evidence, not a deterministic ban. Cognition may deliberately reopen it when the current user turn supports doing so.
12. Current user input, typed reply context, current media, and current cognition output have higher scene authority than compacted continuity.
13. Relevance remains free of V2 progress payloads. Progress loads after response eligibility and before persona cognition.
14. Reuse the existing post-turn recorder LLM route for two concurrently
    dispatched specialist calls. Add no LLM call to the ordinary response path
    and no compactor LLM call. A selection-required turn replaces the generic
    goal call with the specialized selection-goal producer; it does not add a
    verifier call.
15. Each eligible settled turn makes exactly one scene-observer call and one
    event-reconciler call. Each producer receives one attempt and has no repair
    or regeneration prompt. Invalid event output fails closed and retains the
    last valid packet. Invalid scene output preserves the prior validated scene
    while a valid event batch remains writable; a first packet uses a
    deterministic projection of already accepted input/stance/surface fields.
    The event payload always supplies every active prior event; fitting may
    remove older source-turn text but cannot remove a prior event. If the full
    ledger and current accepted turn cannot fit, the event lane returns a typed
    context-limit result without a model call. Diagnostics expose both call
    dispositions.
16. A selection-required goal uses one specialized producing call with one
    authoritative selection string and exact coverage of supplied conversation
    progress event handles. RAG conversation-history rows remain available as
    optional evidence and are outside this exact relation domain. It has no
    semantic verifier, semantic repair, or evaluator-authored replacement.
    Canonical parsing is deterministic-only on this route; structural
    JSON/shape regeneration remains owned by the same producing goal stage
    under its existing bounded contract and cannot invoke the shared
    JSON-repair LLM.
17. Every raw recorder response passes through
    `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` before contract
    evaluation. JSON syntax repair may preserve supplied raw keys and values but
    cannot invent scene meaning, event meaning, lifecycle, handle citations, or
    importance.
18. User-derived text remains in `HumanMessage` payloads. Stable instructions remain in `SystemMessage`.
19. Record cognition-selected, consolidatable silent turns through the same post-turn recorder. Relevance-declined, listen-only, pruned, and unrelated group-noise turns remain outside progress recording.
20. Keep dialog generation, dialog verification, adapters, delivery receipts, character identity, durable user memory, reflection, scheduler, action permissions, and tool execution outside this change.
21. Keep the existing 24,000-character goal-cognition cap. Continuation fitting must occur before that boundary and must not raise it.
22. Use character caps as the blocking pre-call budget because current local routes have no canonical in-repo tokenizer. Record provider-reported token usage when supplied; an explicit `unavailable` value is valid telemetry.
23. Record at least 90/100 on the offline closure-confidence rubric:
    responsibility isolation 25, key-detail projection integrity 25,
    frozen-failure/adversarial replay 20, affected deterministic/full non-live
    verification 20, and two fresh self-review passes 10. Any unresolved
    blocker or high-severity finding keeps closure blocked regardless of the
    numeric score.
24. Stage 12 requires no real-LLM execution. Any future cognition-quality run
    remains separately scoped, runs one case at a time, and cannot redefine
    the conversation-progress projection pass condition.
25. Run no live database migration apply command without a reviewed dry-run, a pre-apply export, and a separate explicit user command.
26. After any automatic context compaction, reread this entire plan before continuing.
27. After each major checklist stage is signed, reread this entire plan before starting the next stage.
28. Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
29. The user requested no subagent review. Plan authoring and the later review gate use a fresh parent self-review unless the user explicitly changes that instruction.

## Must Do

1. Preserve a production-path baseline for the Asuna failure, including the
   actual row window, interaction selection, active packet, pre-cognition
   progress projection, evidence handle, and exact goal-prompt evidence text.
2. Replace row-count selection for continuation with deterministic logical-speaker-turn assembly.
3. Fetch the active user's interaction tail independently from the small ambient group tail so unrelated channel traffic cannot evict the subthread.
4. Replace the flat V1 lists with one canonical V2 active packet containing a scene narrative, flow fields, a source-backed semantic event ledger, recent logical-turn references, and compacted-block references.
5. Replace full-state event rewriting with exact-coverage ID-based
   reconciliation so an omitted supplied event is rejected rather than
   silently interpreted as unchanged.
6. Preserve actor, action, object, beneficiary, precondition, state, and
   outcome for established events and obligations.
7. Assign new event IDs and logical-turn IDs deterministically and reject invented source IDs.
8. Project narrative and flow into `scene_context.conversation_continuity`.
9. Project decision-critical events into the existing `CognitionEvidenceV2` evidence list as `conversation_evidence`, before evidence handles are assigned.
10. Preserve decision-critical events ahead of background detail when fitting the continuation budget.
11. Add one append-only compacted-block collection inside the conversation-progress subsystem with immutable semantic content, source spans, producing-turn lineage, embeddings, and sliding short-term expiry.
12. Trigger compaction from bounded packet size, event count, and retained-turn count rather than lexical content.
13. Produce scene observations and event reconciliation in two independent,
    concurrent post-turn calls. Require exact prior-event handle coverage,
    keep new-event extraction separate, and build any required immutable block
    deterministically from validated event snapshots and existing block
    lineage after the event batch is mapped.
14. Merge four lowest-level active block references hierarchically through a
    deterministic compaction plan whenever adding a block to eight active refs
    would cross the limit, using oldest-first order only among equal-level
    roots. Validate the candidate graph's depth and node counts before
    persistence. Preserve exact child blocks as reachable lineage.
15. Extend the existing conversation-evidence search path to retrieve scoped compacted blocks semantically and temporally when the existing local context resolver requests older conversation evidence.
16. Keep ordinary same-episode continuation independent of retrieval; the active packet must already carry current decision-critical state.
17. Record cognition-selected silent turns that have a settled, consolidatable episode trace.
18. Add sanitized continuation diagnostics for logical-turn counts, packet
    turn count, event counts by state/retention, projected character counts,
    compaction disposition, block references, retrieval source, recorder call
    count, and fail-closed contract errors.
19. Add a report-driven V1-to-V2 active-row reset with dry-run, drift checks, backup evidence, apply gating, and rollback constraints.
20. Update all affected ICD text and tests to the single V2 contract.
21. Prove storage correctness, projection recall, actual cognition-input
    visibility, and maximum deterministic capacity separately.
22. Replace selection-required goal verification/replacement with one
    specialized producing prompt selected from the typed upstream operation.
    Require one authoritative selection and exact active
    conversation-progress-event coverage before deterministic bid mapping.
23. Run the documented offline confidence gate and reach at least 90/100 with
    no unresolved blocker. Real-LLM behavior is outside the Stage 12 pass
    condition.

## Deferred

- No durable user-profile or character-identity memory is produced from V2 events.
- No shared group-wide episode state or group-subthread identifier is added. V2 remains scoped by platform, channel, and current global user.
- No adapter parsing, platform wire contract, delivery behavior, or logical outbound persistence contract changes.
- No final-dialog prompt rule or dialog evaluator suppression rule is added.
- No domain ontology for body parts, affection, tasks, health, coding, or other topics is added.
- No deterministic semantic deduplication, paraphrase matching, or correction classifier is added.
- No historical `conversation_history` backfill or rewrite is performed.
- No semantic conversion of existing V1 episode rows is attempted. The V1 active lane is reset after export because its entries lack canonical V2 source lineage.
- No new local-context capability or subagent is added. Compacted-block retrieval extends the existing conversation-evidence owner.
- No new response-path verifier, retry loop, or unconditional retrieval call is added.
- No tokenizer package, remote tokenizer endpoint, or provider-specific token dependency is added. Provider usage remains measurement; current character caps remain the deterministic pre-call guard.
- No control-console UI is added. Diagnostics remain in typed state, protected traces, logs, migration reports, and review artifacts.
- No general-purpose memory bus, event-sourcing framework, or cross-subsystem compaction engine is introduced.

## Cutover Policy

Overall strategy: forward-only V2 big-bang within the existing `conversation_progress` owner.

| Area | Policy | Instruction |
|---|---|---|
| Public facade | retained names, big-bang signatures | Keep `load_progress_context(...)` and `record_turn_progress(...)`; update every caller and test to the V2 arguments and result shape together. |
| Active packet | big-bang | Replace the V1 field vocabulary with `conversation_progress.v2`; no runtime V1 projection or fallback. |
| Recent context | big-bang | Select complete logical turns and separate ambient from participant-scoped context before persona cognition. |
| Event lifecycle | big-bang | Replace `user_state_updates`, `open_loops`, `resolved_threads`, `avoid_reopening`, and `interaction_obligations` with one source-backed event ledger. |
| Cognition handoff | big-bang | Split scene narrative from citeable decision-critical event evidence. |
| Active state collection | migration | Reset existing V1 rows to closed V2 tombstones through reviewed report-driven apply; begin fresh source-backed V2 episodes. |
| Compacted blocks | new internal tier | Add `conversation_episode_blocks` under the same subsystem; it is not a competing progress store. |
| Block retrieval | compatible extension | Extend existing conversation evidence; no new resolver capability or ordinary-path call. |
| Dialog/adapters | unchanged | Continue consuming the accepted text-surface result and delivering it normally. |
| Tests/docs | big-bang | Remove V1 assertions and publish one canonical V2 contract. |

Cutover enforcement:

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, delete or rewrite legacy references instead of preserving them.
- If an area is `migration`, follow the exact migration phases and cleanup gates listed in this plan.
- If an area is `compatible extension`, preserve only the compatibility surfaces explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.
- Update V2 producer, persistence owner, consumers, migration, tests, and docs in one approved execution.
- Deploy migration code and V2 code together under an operator-controlled maintenance window.
- Run dry-run and export before apply.
- Start the new code only after the V1 active-row reset completes.
- Roll back code and restored V1 data together only before any accepted V2 packet write. Once V2 writes exist, use forward repair rather than restoring stale V1 active state.

## Target State

```text
canonical conversation_history
        |
        +--> deterministic logical-turn assembler
        |       |-- ambient group tail: 6 complete speaker turns
        |       `-- current-user interaction tail: 10 complete speaker turns
        |
        +--> conversation_progress.load_progress_context(...)
        |       |-- active conversation_progress.v2 packet
        |       |-- recent logical turns
        |       `-- active compacted-block references
        |
current user turn + reply/media context
        |
        +--> cognition input
        |       |-- scene_context.conversation_continuity
        |       `-- decision-critical conversation_evidence rows
        |
        +--> Cognition Core V2
        |       `-- character judgment cites relevant event handles
        |
        +--> text surface and dialog
        |
        `--> existing post-turn conversation-progress owner
                |-- scene observer [LLM, concurrent]
                `-- event reconciler [LLM, concurrent]
                        |-- one row per supplied prior event
                        `-- separately listed new events
                                |
                                `--> deterministic mapper and compactor
                                        |-- exact coverage, IDs, refs, limits
                                        |-- guarded active packet replace
                                        `-- conversation_episode_blocks
```

At the failing “what next?” turn, the target cognition input contains a decision-critical event equivalent to:

```json
{
  "semantic_summary": "The user already completed the previously selected neck massage.",
  "state": "completed",
  "retention": "decision_critical",
  "source_refs": [
    {"ref_kind": "conversation_row", "ref_id": "row-id", "occurred_at": "UTC"}
  ]
}
```

The exact wording is model-owned. The invariant is that an already-completed event is available to cognition as evidence. Cognition may still choose the same action when the current user explicitly requests repetition.

The active packet remains fast working memory. Compacted blocks preserve older episode evidence within the same 48-hour sliding lifecycle and are retrieved only when the active packet or current request requires older detail.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Ownership | Improve `conversation_progress` in place | It already owns short-term semantic continuation and has the correct post-relevance/pre-cognition lifecycle. |
| Public API | Retain facade names and replace signatures together | Avoids a competing subsystem while allowing a canonical V2 contract. |
| Recency unit | Complete logical speaker turns | Storage fragments and group noise are not reliable semantic window units. |
| Participant history | Query independently from ambient history | Filtering a tiny channel result cannot recover an evicted user subthread. |
| Assistant grouping | Contiguous rows with the same non-empty `llm_trace_id`, role, channel, and ordered `logical_message_index` form one assistant turn | Uses existing structural correlation without interpreting text. |
| User grouping | Each prior user conversation row is one speaker turn | Preserves platform-message ordering and reply metadata. |
| Fallback grouping | A row without usable correlation remains its own turn | Preserves historical readability without a compatibility schema. |
| Event update form | Exact-coverage reconciliation against the supplied prior ledger; every handle emits `unchanged` or one compact `changed` observation, while new events use a separate list | Makes omission a structural error instead of silently treating it as unchanged, avoids code-side semantic matching, and stops repeated reconstruction of stable definitions. |
| New event identity | UUIDv5 assigned after handle resolution from episode ID, mapped source refs, and the exact validated semantic payload | Identical calls are idempotent; code does not infer paraphrase equivalence. |
| Existing event identity | Model references a short `eN` handle supplied beside prior semantic state; code maps it to the real event ID | Semantic equivalence remains LLM-owned while storage identity remains hidden and deterministic. |
| Event identity fields | Every new event supplies non-empty semantic `actor`, `action`, and `object`; code preserves them on later reconciliation | A generic ordinal summary cannot hide the concrete completed item from cognition. |
| Event lifecycle | The model reports one `lifecycle_change` value from `none`, `began`, `concluded`, `declined`, `replaced`, or `reopened`; code maps it to `open`, `in_progress`, `completed`, `rejected`, or `superseded` | Keeps semantic judgment model-owned while making the persisted state machine deterministic, compact, and inspectable. |
| Retention | The model reports one `relevance` value from `decision`, `scene`, or `history`; code maps it to `decision_critical`, `active_scene`, or `background` | Keeps relevance semantic while making storage priority deterministic. |
| Obligation direction | Events include `is_obligation`, actor, action, beneficiary, and precondition | Preserves the completed obligation-direction contract without a second list. |
| Scene versus fact | A scene-only observer supplies narrative/flow; a separate event reconciler supplies evidence rows | Prevents scene summarization from competing with exact event reconciliation and makes each failure independently inspectable. |
| Evidence source kind | Reuse `conversation_evidence` | The cognition contract already recognizes this as source-backed prior dialog evidence. |
| Compaction trigger | Structural thresholds only | Size, count, and age are deterministic; event meaning remains model-owned. |
| Source lineage | Model cites short `tN`/current-turn handles; code maps them to canonical row/trace refs and preserves prior lineage on updates | The model grounds semantics without copying machine IDs or timestamps. |
| Future response planning | Cognition only | Persistence stores established facts and prior patterns; it emits no `next_affordances` or `progression_guidance`. |
| Recorder recovery | One attempt per specialist; event failure fails closed, while scene failure preserves the prior validated scene | Removes steady-state repair scaffolding, protects event correctness, and gives the lower-authority scene lane deterministic safe degradation. |
| Selection-required cognition | Deterministic routing selects one specialized goal producer; it emits one authoritative selection and one relation row per supplied active conversation-progress-event handle; RAG conversation-history rows remain optional evidence | Removes the semantic verifier/replacement loop, keeps the actual choice with cognition, and keeps the mandatory relation domain within the existing nine-citation bid cap. |
| Compaction call count | Zero LLM calls | Compaction consumes deterministically derived lifecycle/retention labels and exact stored snapshots without another semantic decision. |
| Block content | Deterministically assembled and immutable after insert; only expiry and supersession metadata may change | Preserves auditability and exact child lineage while allowing sliding short-term lifecycle. |
| Block retrieval | Existing conversation-evidence search, top three scoped blocks | Keeps retrieval evidence-bound and off the ordinary same-episode path. |
| Silent turns | Record only settled, consolidatable cognition-selected silence | Preserves meaningful observation without treating ambient group noise as a character episode. |
| Active-row cutover | Export and reset, not semantic migration | V1 rows do not contain source lineage and are only 48-hour working memory. |
| Prompt budget | 4,000 combined continuation characters under the unchanged 24,000-character goal cap | Fits the observed failure trace with approximately 9,900 characters of remaining headroom. |
| Token measurement | Record provider usage when present; preflight remains character-based | No canonical tokenizer is available for all configured local OpenAI-compatible routes. |

## Data Migration

The migration affects only `conversation_episode_state`. It does not change `conversation_history`, user memory, character state, reflection, scheduler, or adapter data.

Create `src/scripts/migrate_conversation_progress_v2.py` as a thin CLI over named helpers in `kazusa_ai_chatbot.db.script_operations`.

Dry-run behavior:

1. Read every `conversation_episode_state` row.
2. Classify schema version, scope validity, status, turn count, timestamps, and
   drift-match fields. A row labelled `conversation_progress.v2` counts as
   `already_v2` only when the canonical exact active-packet validator accepts
   its complete shape; a version label on a legacy or malformed row is
   `malformed`.
3. Produce `test_artifacts/migrations/conversation_progress_v2_dry_run.json`.
4. Produce `test_artifacts/migrations/conversation_progress_v1_pre_cutover.json` as a complete relaxed-Extended-JSON backup including `_id`, and record its row count and SHA-256 digest in the dry-run report.
5. Perform zero database writes.

Apply behavior:

1. Require `--apply`, the reviewed dry-run report, its digest-matching backup, and an explicit user command.
2. Reread each row and match `_id`, scope, `turn_count`, `updated_at`, `status`, and schema version against the dry-run record.
3. Replace each matched V1 row with an exact closed `conversation_progress.v2` tombstone using the same MongoDB `_id` and scope, a new `episode_state_id`, `turn_count=0`, empty narrative/events/references, cutover timestamps, and a 48-hour expiry.
4. Report changed, drift-skipped, already-V2, malformed, blocked, and failed counts.
5. Delete zero `conversation_history` rows and zero durable-memory rows.
6. Run bootstrap and post-apply audit.

The V2 active reader returns only rows with:

- `schema_version == "conversation_progress.v2"`;
- `status == "active"`; and
- a valid unexpired storage-UTC `expires_at`.

The guarded writer may replace a closed V2 tombstone with a new active packet at `turn_count=1`, while a newer active V2 packet remains protected.

Rollback:

- Before service startup accepts any active V2 packet (`turn_count >= 1`), restore the exported V1 rows and deploy the prior code together.
- `--restore-v1` requires the backup and apply report, replaces only rows still matching their exact applied tombstone, skips any active V2 row or drift, and emits a restore report; it also requires a separate explicit user command.
- After any active V2 packet exists, do not restore V1 state over it. Preserve the export as audit evidence and perform a forward V2 repair.
- No rollback command may delete newer V2 blocks or packets without a separate explicit user instruction and a new reviewed report.

## Contracts And Data Shapes

### Public facade

Retain the facade names and replace their signatures in one cutover:

```python
async def load_progress_context(
    *,
    scope: ConversationProgressScope,
    current_timestamp_utc: str,
    platform_bot_id: str,
    active_turn_conversation_row_ids: list[str],
) -> ConversationProgressLoadResult: ...

async def record_turn_progress(
    *,
    record_input: ConversationProgressRecordInput,
) -> ConversationProgressRecordResult: ...

def validate_active_packet(
    packet: Mapping[str, object],
) -> ConversationProgressStateV2: ...
```

`ConversationProgressLoadResult` contains:

```python
{
    "episode_state": ConversationProgressStateV2 | None,
    "conversation_progress": ConversationProgressPromptV2,
    "ambient_logical_turns": list[ConversationLogicalTurnV1],
    "interaction_logical_turns": list[ConversationLogicalTurnV1],
    "diagnostics": ConversationProgressLoadDiagnosticsV2,
    "source": "db | cache | empty",
}
```

The facade performs the active-packet read, ambient-row read, participant-row read, cache selection, logical-turn assembly, and bounded projection. Service and persona code do not import repository, history, projection, compaction, or block internals.

### Logical turn

```python
class ConversationLogicalTurnV1(TypedDict):
    turn_id: str
    role: Literal["user", "assistant"]
    occurred_at: str
    display_name: str
    fragments: list[str]
    conversation_row_ids: list[str]
    llm_trace_id: str
    platform_user_id: str
    global_user_id: str
    addressed_to_global_user_ids: list[str]
    broadcast: bool
    reply_context: dict[str, object]
```

Rules:

- `turn_id` is `row:<conversation-row-id>` for a user or ungrouped row.
- `turn_id` is `trace:<llm-trace-id>` for grouped assistant fragments.
- Every source row appears in at most one logical turn.
- Grouping never examines `body_text`.
- A grouped assistant candidate must have one non-empty trace ID and unique contiguous `logical_message_index` values starting at zero; fragments are ordered by that index, then timestamp, then source-row order.
- If the oldest fetched assistant candidate starts above index zero, drop that incomplete boundary turn. Any other malformed/gapped candidate falls back to one turn per row and increments a protected diagnostic.
- Selected turns stay chronological.
- Active current-turn row IDs are excluded because current input is supplied separately.
- Prompt projection joins fragments inside one speaker turn and preserves the canonical row IDs only in protected state/trace data.

Limits:

```text
AMBIENT_ROW_SCAN_LIMIT = 48
INTERACTION_ROW_SCAN_LIMIT = 128
AMBIENT_LOGICAL_TURN_LIMIT = 6
INTERACTION_LOGICAL_TURN_LIMIT = 10
MAX_LOGICAL_TURN_TEXT_CHARS = 600
MAX_AMBIENT_PROMPT_CHARS = 1200
MAX_INTERACTION_RECORDER_CHARS = 2000
```

### Participant-scoped read

Add a public database helper:

```python
async def get_participant_conversation_history(
    *,
    platform: str,
    platform_channel_id: str,
    current_global_user_id: str,
    platform_bot_id: str,
    excluded_row_ids: list[str],
    limit: int,
) -> list[ConversationMessageDoc]: ...
```

It selects:

- user rows authored by `current_global_user_id`; and
- assistant rows authored by `platform_bot_id` that either broadcast or address `current_global_user_id`.

It returns oldest-first rows after a newest-first bounded query. It does not require a current-user row to be present in the fetched window. Bootstrap creates exact indexes `conversation_history_participant_user_v1` on `(platform, platform_channel_id, role, global_user_id, timestamp desc)`, `conversation_history_participant_assistant_addressed_v1` on `(platform, platform_channel_id, role, platform_user_id, addressed_to_global_user_ids, timestamp desc)`, and `conversation_history_participant_assistant_broadcast_v1` on `(platform, platform_channel_id, role, platform_user_id, broadcast, timestamp desc)`.

### Source reference

```python
class ConversationProgressSourceRefV2(TypedDict):
    ref_kind: Literal["conversation_row", "llm_trace"]
    ref_id: str
    occurred_at: str
```

Rules:

- The recorder never sees or emits this storage shape.
- Deterministic code assigns short source handles to recent logical turns,
  current user rows, and the current accepted `llm_trace_id`.
- Every new or semantically changed event cites at least one supplied short
  source handle; code resolves those handles to this exact shape.
- The model may cite at most eight temporary handles so current-turn aliases
  can coexist with their logical-turn handles. Code resolves and deduplicates
  them before enforcing at most four canonical source refs per event.
- Repository validation rejects empty, unknown, duplicate, or cross-scope references.
- Source refs are audit lineage; they are not copied into public dialog.
- A collapsed current input preserves every active conversation-row ref with
  that row's exact timestamp. The single `current_input` handle resolves to
  the complete ordered row set before the canonical per-event source cap is
  applied.

### Event ledger

```python
class ConversationProgressEventV2(TypedDict):
    event_id: str
    semantic_summary: str
    is_obligation: bool
    actor: str
    action: str
    object: str
    beneficiary: str
    precondition: str
    state: Literal[
        "open",
        "in_progress",
        "completed",
        "rejected",
        "superseded",
    ]
    outcome: str
    retention: Literal[
        "decision_critical",
        "active_scene",
        "background",
    ]
    source_refs: list[ConversationProgressSourceRefV2]
    first_seen_at: str
    updated_at: str
```

Rules:

- `semantic_summary` is at most 220 characters.
- An emitted `semantic_summary` is self-contained enough to distinguish its
  concrete event. When current evidence resolves a generic prior placeholder,
  the model updates that summary without changing the stable event handle.
- `actor`, `action`, `object`, `beneficiary`, and `precondition` are at most
  160 characters each.
- `outcome` is at most 180 characters.
- Every new event requires non-empty `actor`, `action`, and `object`.
- `is_obligation=true` additionally preserves explicit direction through
  `beneficiary` and `precondition` when those facts apply.
- New semantic event changes use `event_handle="new"`; deterministic code
  assigns the real ID after every semantic field and source handle validates.
- Assignment is `uuid.uuid5(uuid.NAMESPACE_URL, canonical_json).hex`, where `canonical_json` uses sorted compact JSON keys over `episode_state_id`, sorted `(ref_kind, ref_id)` pairs, and the validated semantic fields; lifecycle timestamps are excluded.
- Existing-event changed observations use a supplied short handle that code
  resolves to an ID present in the prior packet. They emit only
  `event_handle`, `observation`, `semantic_summary`, `outcome`,
  `lifecycle_change`, `relevance`, and `source_turn_handles`; code preserves
  the existing validated `is_obligation`, `actor`, `action`, `object`,
  `beneficiary`, and `precondition`.
- Code merges by `event_id` only and performs no text-equivalence comparison.
- `first_seen_at` is code-owned and immutable.
- `updated_at` changes only when the model emits a valid update.
- `decision_critical`, `open`, and `in_progress` events remain in the active
  packet and are excluded from discard/archive candidates. The recorder may
  change the atomic observations from which code derives lifecycle and
  retention; only a later valid packet may archive an event after the derived
  state is terminal and the derived retention is no longer
  `decision_critical`.

### Active packet

```python
class ConversationProgressStateV2(TypedDict):
    schema_version: Literal["conversation_progress.v2"]
    episode_state_id: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    status: Literal["active", "suspended", "closed"]
    continuity: Literal[
        "same_episode",
        "related_shift",
        "sharp_transition",
    ]
    turn_count: int
    episode_narrative: str
    current_thread: str
    character_stance: str
    user_goal: str
    current_blocker: str
    emotional_trajectory: str
    events: list[ConversationProgressEventV2]
    overused_moves: list[str]
    recent_turn_refs: list[str]
    compacted_block_refs: list[str]
    created_at: str
    updated_at: str
    expires_at: str
    purge_after: datetime
```

Limits:

```text
MAX_EPISODE_NARRATIVE_CHARS = 900
MAX_THREAD_FIELD_CHARS = 240
MAX_FLOW_GUIDANCE_CHARS = 300
MAX_ACTIVE_EVENTS = 24
MAX_RECENT_TURN_REFS = 16
MAX_ACTIVE_BLOCK_REFS = 8
MAX_ACTIVE_PACKET_CHARS = 16000
COMPACTION_EVENT_SOFT_LIMIT = 18
COMPACTION_TURN_REF_SOFT_LIMIT = 12
COMPACTION_PACKET_SOFT_CHARS = 10000
```

The active packet is replacement-written under scope plus strictly newer `turn_count`. A closed, expired, missing-expiry, or schema-invalid row may be replaced by a new V2 packet; a newer valid active row may not.

Every `*_CHARS` limit uses Python `len(...)` on the final normalized Unicode text. Packet/block hard limits use compact, sorted-key JSON with datetimes rendered as storage-UTC strings; Mongo `_id` and block `embedding` are excluded from that measurement.

### Post-turn semantic observations

`ConversationProgressRecordInput` continues to carry typed logical turns,
protected current-turn source refs, the prior packet, the accepted cognition
surface, and the actual final dialog. Deterministic code assigns `e1`, `e2`,
... to supplied prior events, `t1`, `t2`, ... to supplied logical turns,
`current_input` to current user rows, and `current_response` to the accepted
response trace. Both payloads receive one prompt-safe `semantic_context` with
the exact runtime character name and the configured-local semantic clock.
Logical-turn rows expose `speaker_kind` plus `speaker_name`; character rows use
that exact runtime name. Private IDs, source timestamps, and source objects
never enter either model payload.

The semantic owners ground time-dependent operational facts to absolute
`YYYY-MM-DD` or `YYYY-MM-DD HH:MM` values from that clock, or omit those facts
when they cannot be resolved uniquely. Character references use the exact
runtime name or subjectless wording; machine role labels are never character
names.

The scene observer receives only `prior_scene`, bounded recent semantic turns,
and `accepted_turn`. It returns:

```python
class ConversationProgressSceneObservationV2(TypedDict):
    schema_version: Literal["conversation_progress_scene_observation.v2"]
    scene_relation: Literal["same", "related", "new"]
    episode_change: Literal["none", "paused", "finished", "resumed"]
    episode_narrative: str
    current_thread: str
    character_stance: str
    user_goal: str
    current_blocker: str
    emotional_trajectory: str
    overused_moves: list[str]
```

The event reconciler receives only supplied prior event definitions, bounded
source turns, current input, actual accepted response, and their short handles.
It returns:

```python
class ConversationProgressEventObservationBatchV2(TypedDict):
    schema_version: Literal[
        "conversation_progress_event_observation_batch.v2"
    ]
    existing_events: list[
        ConversationProgressExistingEventObservationV2
    ]
    new_events: list[ConversationProgressNewEventObservationV2]
```

Every supplied prior handle appears exactly once in `existing_events`.
Unchanged rows have exactly:

```python
{
    "event_handle": "e1",
    "observation": "unchanged",
}
```

Changed rows have exactly:

```python
{
    "event_handle": "e1",
    "observation": "changed",
    "semantic_summary": str,
    "outcome": str,
    "lifecycle_change": (
        "none | began | concluded | declined | replaced | reopened"
    ),
    "relevance": "decision | scene | history",
    "source_turn_handles": list[str],
}
```

Each new-event row has no event handle and exactly:

```python
{
    "semantic_summary": str,
    "is_obligation": bool,
    "actor": str,
    "action": str,
    "object": str,
    "beneficiary": str,
    "precondition": str,
    "outcome": str,
    "lifecycle_change": (
        "none | began | concluded | declined | replaced"
    ),
    "relevance": "decision | scene | history",
    "source_turn_handles": list[str],
}
```

`actor`, `action`, and `object` are non-empty for every new event. Later
observations cannot rewrite those stable definition fields. Missing, duplicate,
unknown, or extra prior handles are structural contract errors. The validator
requires exact equality between supplied prior handles and observed prior
handles, so a forgotten clause can no longer become an implicit unchanged
event.

Deterministic mapping remains:

```text
lifecycle_change=declined -> rejected
lifecycle_change=replaced -> superseded
lifecycle_change=concluded -> completed
lifecycle_change=began -> in_progress
lifecycle_change=reopened -> open
lifecycle_change=none -> preserve prior state, or open for a new event

relevance=decision -> decision_critical
relevance=scene -> active_scene
relevance=history -> background
```

A new event cannot use `reopened`. An existing terminal event cannot derive a
non-terminal state without `reopened`. `concluded` means the exact event is
done, explicitly stopped, or has a conclusive accepted/evaluated outcome; a
provisional assessment remains nonterminal.

The two calls run concurrently. Event reconciliation is authoritative for
whether a write can proceed. A scene contract/provider failure records
`preserved_prior` and uses the prior validated scene; when no prior packet
exists, code copies bounded already accepted input/stance/surface fields into
the initial scene without interpreting their meaning. Neither producer has a
repair prompt, verifier, or second attempt.

### Selection-required goal production

Deterministic routing inspects the typed upstream
`response_operation.selection_required` value. Ordinary goal turns keep the
ordinary goal producer. A required character-owned selection uses one
specialized producer in its place.

The specialized output contains one authoritative `selection` string, one
`selection_kind` from `choice`, `refusal`, `condition`, or `negotiation`,
grounding/role handles, reason, private monologue, consequences, confidence,
and exactly one relation row for every supplied `conversation_evidence`
handle. Relation values are `excluded`, `reopened`, `supports`, or
`unrelated`. The producing LLM owns those semantic relations and the actual
choice. Code owns exact handle coverage, cardinality, bounds, and mapping into
the existing complete bid. It copies the one selection string into the bid's
selection-bearing fields so no evaluator must infer a hidden choice from
multiple prose fields.

The required-operation episode handle and all supplied active
conversation-progress-event handles must appear in `evidence_handles`.
Progress events are identified by their code-owned
`conversation-progress-event:` evidence provenance, not by user text. RAG
conversation-history rows remain visible and optionally citeable but do not
consume mandatory relation rows. Missing, duplicate, unknown, or extra active
progress-event relation handles are structural contract errors owned by the
producing goal stage. There is no required-selection verifier route, verdict,
semantic repair prompt, replacement bid, or recheck.
The selection route calls `parse_llm_json_output(...,
deterministic_only=True)`. Residual malformed JSON therefore becomes a
structural producer failure and cannot enter the shared JSON-repair LLM path.

### Deterministic compaction plan

After applying the validated semantic delta, deterministic code checks the
soft thresholds. It selects exact archive candidates using only derived
`state`/`retention` labels, code-owned timestamps, and stable IDs. It never
examines event text to decide eligibility.

```python
class ConversationCompactionPlanV2(TypedDict):
    archive_event_ids: list[str]
    covered_turn_refs: list[str]
    source_block_ids: list[str]
```

Rules:

- `archive_event_ids` contains at most eight oldest terminal,
  non-`decision_critical` events, preferring `background` before
  `active_scene`.
- `covered_turn_refs` contains the oldest bounded active turn refs.
- With fewer than eight active block refs, the plan creates a level-0 block.
- When adding a block to eight active refs, `source_block_ids` contains the
  four lowest-level roots, ordered oldest-first within the same level, and the
  new block has `level = max(source levels) + 1`.
- Before returning a prepared write, code validates the complete candidate
  graph including the new block against both the eight-level and 128-node hard
  caps. Once the graph reaches its hard capacity, code suspends further
  archival and retains otherwise eligible events in the bounded active packet.
  The update fails closed only when that active packet would then exceed its
  own hard event or character cap, while the last valid packet remains
  authoritative.
- Block `events` are exact snapshots of `archive_event_ids`. Child-block
  events remain exact in immutable child blocks and stay reachable through
  `source_block_ids`.
- `narrative` is a bounded deterministic join of already model-authored event
  summaries and source-block narratives. `semantic_keys` are bounded exact
  excerpts from the same stored semantic text. Neither field adds a new
  semantic judgment.
- If protected active events alone exceed the hard packet cap, the update
  fails closed and retains the last valid packet.

### Compacted block

```python
class ConversationEpisodeBlockV1(TypedDict):
    schema_version: Literal["conversation_progress_block.v1"]
    block_id: str
    episode_state_id: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    level: int
    source_turn_count: int
    covered_turn_refs: list[str]
    source_block_ids: list[str]
    narrative: str
    events: list[ConversationProgressEventV2]
    semantic_keys: list[str]
    source_started_at: str
    source_ended_at: str
    content_hash: str
    superseded_by_block_id: str
    embedding: list[float]
    created_at: str
    expires_at: str
    purge_after: datetime
```

Persistence rules:

- `block_id` is `uuid.uuid5(uuid.NAMESPACE_URL, canonical_json).hex` over the episode ID, producing `turn_count`, level, and sorted archive-event IDs, covered-turn refs, and source-block IDs.
- Insert the block idempotently, then guarded-write the packet with `existing refs - source_block_ids + block_id`; mark source blocks superseded only after that packet write succeeds.
- Semantic content, events, source spans, level, and hash are immutable.
- The existing document embedding service embeds only `narrative`, `semantic_keys`, and retained event summary/state text after validation; IDs, scope fields, and source refs are excluded from embedding text.
- Only `superseded_by_block_id`, `expires_at`, and `purge_after` may change.
- A lost active-packet write leaves an unreferenced new block while every source block remains active; the orphan is excluded from projection and expires automatically.
- Every successful active-packet write refreshes expiry metadata for its currently referenced blocks.
- Both collections retain the 48-hour sliding working-memory lifecycle.

Block limits are `MAX_BLOCK_EVENTS = 8`, `MAX_BLOCK_TURN_REFS = 24`, `MAX_BLOCK_SOURCE_BLOCKS = 4`, and `MAX_BLOCK_CHARS = 12000`. The block stores exact snapshots selected by the deterministic archive plan; its narrative contains only bounded text already present in those events or supplied source blocks. Superseded and orphan blocks retain their existing expiry and age out.

Indexes:

- active packet unique scope index on `(platform, platform_channel_id, global_user_id)`;
- active packet physical TTL index on BSON `purge_after`;
- block unique index on `block_id`;
- block scope/turn index on `(platform, platform_channel_id, global_user_id, episode_state_id, source_turn_count)`;
- block active-lineage index on `(episode_state_id, superseded_by_block_id, level, source_ended_at)`; and
- block vector index named `conversation_episode_blocks_vector_index` over `embedding`, with `platform`, `platform_channel_id`, `global_user_id`, `episode_state_id`, `block_id`, and `superseded_by_block_id` filter paths.

### Prompt projection

`ConversationProgressPromptV2` contains bounded scene fields, up to ten interaction-lane logical turns, and selected active-packet event rows; scene fitting preserves the newest four turns, the ambient lane is used only by decontextualization, and block refs remain protected routing metadata. The cognition connector consumes it through two projections:

```python
def project_conversation_progress_scene(
    progress: ConversationProgressPromptV2,
) -> str: ...

def project_conversation_progress_evidence(
    progress: ConversationProgressPromptV2,
    occurred_at: str,
) -> list[CognitionEvidenceV2]: ...
```

Budget and selection:

```text
MAX_PROGRESS_SCENE_CHARS = 2200
MAX_SCENE_NARRATIVE_CHARS = 600
MAX_SCENE_LOGICAL_TURNS = 4
MAX_SCENE_TURN_TEXT_CHARS = 160
MAX_PROGRESS_EVIDENCE_CHARS = 1800
MAX_PROGRESS_EVIDENCE_ROWS = 8
MAX_CONTINUATION_CHARS = 4000
```

Selection order:

1. `decision_critical` events;
2. `open` and `in_progress` active-scene events;
3. recently updated `completed`, `rejected`, and `superseded` active-scene events;
4. background events only when room remains.

Within one tier, newer `updated_at` wins. Ties use stable `event_id` order.
This ordering consumes deterministically derived lifecycle/retention labels
and does not interpret event text.

Scene fitting first renders `current_thread`, the prior accepted
`character_stance`, the first 600 narrative characters, and the newest four
chronological interaction turns at 160 characters each. It then admits
`user_goal`, `current_blocker`, `emotional_trajectory`, and `overused_moves` in
that exact order while room remains. Event evidence has its separate budget and
is never displaced by optional scene detail. Persistence emits no future
response recommendation.

Evidence fitting first selects up to eight ordered events and assigns each row
a deterministic share of the 1,800-character budget. Every selected row keeps
bounded summary, state, retention, actor, action, and object identity before
optional detail is admitted. An early maximum-length row therefore cannot
truncate or remove a later selected row.

Evidence rows are inserted after current episode/media evidence and before RAG history, promoted reflection, resolver observations, and action results. Existing evidence handles are then assigned once across the final ordered list.

### Retrieval

Extend the existing conversation semantic-search worker. When the resolver has selected a conversation-evidence task and the current scope has active block references:

1. embed the semantic search query through the existing embedding service;
2. search only unsuperseded block IDs present in the valid active packet's `compacted_block_refs`, under the exact platform/channel/user/episode scope;
3. select at most three block rows;
4. merge block and canonical conversation-history evidence by score and source coverage without keyword-based block filtering;
5. project block narrative, relevant event state, time range, and block ID as `conversation_evidence`; and
6. retain source refs in protected trace artifacts.

Active/open episode-block searches remain uncached. Closed historical conversation-history cache behavior remains unchanged.

### Diagnostics

Load and record results expose no raw text in sanitized telemetry. They report:

```python
{
    "schema_version": "conversation_progress_diagnostics.v2",
    "ambient_rows_scanned": int,
    "interaction_rows_scanned": int,
    "ambient_turns_selected": int,
    "interaction_turns_selected": int,
    "incomplete_or_malformed_turn_count": int,
    "packet_turn_count": int,
    "active_event_count": int,
    "decision_critical_event_count": int,
    "block_ref_count": int,
    "scene_chars": int,
    "evidence_chars": int,
    "compaction_requested": bool,
    "compaction_level": int,
    "recorder_call_count": int,
    "event_attempt_count": int,
    "scene_attempt_count": int,
    "event_disposition": str,
    "scene_disposition": str,
    "write_disposition": str,
}
```

Protected debug artifacts additionally retain prompt-safe event summaries, source refs, selected evidence handles, compaction decisions, cognition bid, and final dialog.

## LLM Call And Context Budget

Project default context ceiling: 50,000 tokens. Current production preflight for goal cognition remains the authoritative 24,000-character cap.

| LLM stage | Before | After | Path | Budget rule |
|---|---:|---:|---|---|
| Relevance | unchanged | unchanged | response | Receives no conversation progress. |
| Decontextualizer | one call | one call | response | Receives six complete ambient turns within 1,200 characters. |
| Cognition Core V2 | generic goal plus required-selection verifier/replacement when applicable | one ordinary goal or one specialized selection goal | response | Combined progress scene/evidence is at most 4,000 characters; aggregate goal cap stays 24,000; selection-required turns remove verifier and replacement calls. |
| Dialog | unchanged | unchanged | response | Receives accepted semantic surface only; no full transcript or event ledger. |
| Progress scene observer | mixed into one recorder call | one scene-only call after visible or eligible silent turn | concurrent post-turn | Human payload is capped at 8,000 characters; invalid output preserves prior accepted scene facts. |
| Progress event reconciler | mixed into one recorder call | one exact-coverage event-only call after visible or eligible silent turn | concurrent post-turn | Human payload is capped at 24,000 characters after dropping only older logical-turn text; every active prior event remains mandatory and an oversized required ledger fails closed before the call. |
| Block compaction | no separate call | zero calls | post-turn | Deterministic after semantic-delta mapping. |
| Older-block retrieval | existing resolver-selected calls only | existing resolver-selected calls only | conditional response resolver | Ordinary same-episode path performs no retrieval call. |

Event-reconciler fitting preserves, in order:

1. current decontextualized input;
2. actual final dialog or silence outcome;
3. every active prior event definition and handle;
4. current short source handles; and
5. newest complete interaction turns.

Scene-observer fitting preserves current input, actual response, accepted
stance/content surface, prior scene, and then newest complete interaction
turns.

If required current-turn material plus required prior state cannot fit under
24,000 characters, the recorder returns a typed context-limit failure and
retains the last valid packet. It does not truncate private handle maps or
invent a semantic update.

Budget evidence required at sign-off:

- response-path LLM call-count comparison;
- maximum and p95 prompt characters for each affected stage;
- provider-reported input/output tokens when available;
- explicit `unavailable` token telemetry when a provider omits usage;
- scene/event recorder call counts, dispositions, and fail-closed
  contract-error count;
- block embedding call count;
- progress DB query count; and
- before/after p95 progress-load latency.

The two new history reads and active-packet read execute in one concurrent gather. After-change p95 progress-load latency must remain no greater than the Stage 1 baseline plus 75 milliseconds in the same guarded test environment.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/conversation_progress/history.py`
  - Logical-turn grouping, participant/ambient selection, source aliases, and prompt-safe turn projection.
- `src/kazusa_ai_chatbot/conversation_progress/compaction.py`
  - Deterministic balanced capacity selection, block construction from exact
    stored semantics, stable IDs, fit ordering, and candidate block-lineage
    validation.
- `src/kazusa_ai_chatbot/db/conversation_progress_blocks.py`
  - Block insert, scope load, expiry touch, supersession, and semantic search.
- `src/scripts/migrate_conversation_progress_v2.py`
  - Thin dry-run/apply CLI over database maintenance helpers.
- `tests/fixtures/conversation_progress_v2/asuna_houjing_long_thread.json`
  - Redacted, source-faithful regression sequence with synthetic account IDs, actual adjacent wording, segmented assistant rows, and unrelated group traffic.
- `tests/test_conversation_progress_logical_turns.py`
  - Grouping, ordering, complete-turn caps, participant scope, and current-turn exclusion.
- `tests/test_conversation_progress_v2_contract.py`
  - Exact packet, event-delta, source-ref, lifecycle, cap, and migration shapes.
- `tests/test_conversation_progress_compaction.py`
  - Thresholds, event survival, block insertion ordering, hierarchical merge, expiry touch, and lost-write behavior.
- `tests/test_conversation_progress_cognition_evidence.py`
  - Scene/evidence split, evidence order, handle assignment, and budget priority.
- `tests/test_conversation_progress_block_retrieval.py`
  - Scope filters, top-k, active-block admission, projection, and cache policy.
- `tests/test_conversation_progress_v2_service.py`
  - Load lifecycle, visible/silent record gates, no-response exclusions, and response-path call count.
- `tests/test_conversation_progress_v2_migration.py`
  - Complete backup/digest, dry-run purity, drift checks, tombstone replacement, idempotency, rollback report shape, and zero unrelated writes.
- `tests/test_conversation_progress_v2_live_llm.py`
  - One-case-at-a-time exact regression, deliberate reopening, cross-domain correction/supersession, group interleaving, and 20/50/100-turn compaction quality cases.

### Modify

- `src/kazusa_ai_chatbot/conversation_progress/__init__.py`
  - Export only the canonical V2 facade, public types, and the canonical exact
    active-packet validator used by migration audit code.
- `src/kazusa_ai_chatbot/conversation_progress/models.py`
  - Replace V1 packet, scene-observation, exact-coverage event-observation,
    prompt, record-input, result, event, block-ref, and diagnostics types;
    require concrete event objects and exclude future guidance.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py`
  - Replace V1 list limits/discard order with V2 turn, event, packet, compaction, scene, evidence, and expiry policy.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Dispatch one scene-only and one exact-coverage event-only producer
    concurrently over short handles; keep machine IDs, source-ref objects,
    timestamps, discard, compaction, persistence, and future response planning
    outside both prompts; preserve prior scene on scene failure and fail closed
    on event failure.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py`
  - Resolve handles to event IDs and exact source refs, apply validated deltas,
    preserve code-owned timestamps, run deterministic compaction, order
    idempotent block insert before guarded packet replacement, and refresh
    block expiry.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py`
  - Build bounded factual V2 prompt state and diagnostics with
    decision-critical-first fitting and no future guidance.
- `src/kazusa_ai_chatbot/conversation_progress/runtime.py`
  - Concurrently load state and two history lanes, select active cache state, apply recorder deltas, persist blocks/packet, and return diagnostics.
- `src/kazusa_ai_chatbot/conversation_progress/cache.py`
  - Require active V2 schema/status/expiry and preserve guarded `turn_count` ordering.
- `src/kazusa_ai_chatbot/conversation_progress/README.md`
  - Publish the V2 ownership, lifecycle, contracts, budgets, and retrieval boundary.
- `src/kazusa_ai_chatbot/db/conversation.py`
  - Add participant-scoped bounded history read.
- `src/kazusa_ai_chatbot/db/conversation_progress.py`
  - Add active V2 read, exact replacement write, closed/expired replacement, and guarded newer-active protection.
- `src/kazusa_ai_chatbot/db/schemas.py`
  - Replace V1 progress types and add compacted-block types.
- `src/kazusa_ai_chatbot/db/bootstrap.py`
  - Create V2 active/block indexes and BSON physical-expiry TTL indexes.
- `src/kazusa_ai_chatbot/db/__init__.py`
  - Export the named participant and block operations required by production callers.
- `src/kazusa_ai_chatbot/db/script_operations.py`
  - Add read-only V2 migration audit and report-driven apply helpers.
- `src/kazusa_ai_chatbot/db/README.md`
  - Document active packet, block collection, indexes, source lineage, and migration boundary.
- `src/kazusa_ai_chatbot/state.py`
  - Add ambient/interaction logical-turn and V2 progress contracts.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Carry V2 packet, logical turns, and diagnostics.
- `src/kazusa_ai_chatbot/service.py`
  - Pass V2 load arguments, preserve current source row IDs, gate eligible silent recording, and keep progress post-relevance.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
  - Build V2 record input from settled episode trace, current source refs, logical turns, visible response or cognition silence, and accepted surface.
- `src/kazusa_ai_chatbot/brain_service/intake.py`
  - Preserve one exact conversation-row source reference and storage timestamp
    for every survivor or collapsed current input row.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Stop rebuilding interaction continuity from the ten-row channel window; consume the facade-selected ambient and participant logical turns.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py`
  - Consume the bounded ambient logical-turn projection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Replace `_conversation_progress_text(...)` with the V2 scene and evidence projections before evidence-handle assignment.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
  - Raise only `scene_context.conversation_continuity` to the approved 2,200-character sub-cap; retain the aggregate 24,000-character goal cap and existing `conversation_evidence` source kind.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - Route typed required-selection turns to one specialized producing prompt,
    validate exact active conversation-progress-event coverage while retaining
    RAG rows as optional evidence, map its one authoritative selection into the
    complete bid, and delete semantic verifier/replacement ownership.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`,
  `src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py`,
  `src/kazusa_ai_chatbot/config.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`,
  `README.md`, `docs/HOWTO.md`, and
  `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
  - Remove the required-selection verifier route/config/attempt contract and
    document the specialized producer as a replacement call, not an added
    evaluator.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/workers/search.py`
  - Add scoped semantic compacted-block retrieval and merge.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/projection.py`
  - Project block narrative/events as conversation evidence with trace refs.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/contracts.py`
  - Add exact block-result projection shape.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/README.md`
  - Document active compacted-block evidence ownership.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - Treat active block searches as open, uncached conversation state and include block writes in dependency invalidation.
- `src/kazusa_ai_chatbot/rag/recall/collectors/progress.py`
  - Consume only canonical V2 scene fields and event summaries; remove the
    hidden V1 list-field reader.
- `tests/test_temporal_relative_terms_live_llm.py`
  - Replace the deleted V1 recorder harness with one V2 two-producer
    absolute-or-omit temporal and exact-character-identity gate.
- Existing focused tests under `tests/`
  - Replace V1 assumptions and preserve adjacent behavior contracts.

### Delete

- V1 prompt/storage fields: `conversation_mode`, `episode_phase`, `topic_momentum`, `user_state_updates`, `assistant_moves`, `open_loops`, `interaction_obligations`, `resolved_threads`, and `avoid_reopening`.
- V1 fixed overflow order that drops resolved/closed state before lower-value detail.
- V1 recorder full-list rewriting and exact-text timestamp preservation.
- Recorder-emitted persistence IDs, timestamped source objects, discard lists,
  compaction instructions, `next_affordances`, `progression_guidance`, the
  recorder repair prompt, and the mixed scene/event prompt.
- Required-selection verifier route/config, verdict prompt, semantic repair
  prompt, replacement bid loop, and recheck.
- Live-persona dependence on `build_interaction_history_recent(...)` over the tiny shared channel window. Keep the helper only for remaining callers that still own a distinct contract.
- Any test helper whose only purpose is V1 compatibility projection.

### Keep

- `conversation_history` as canonical append-only conversation evidence.
- `conversation_episode_state` as the active packet collection name.
- The `(platform, platform_channel_id, global_user_id)` active scope.
- The 48-hour sliding short-term lifecycle.
- Post-relevance progress loading.
- Existing current-message, reply, mention, media, and active-turn exclusion contracts.
- Existing logical assistant-message persistence by `llm_trace_id` and `logical_message_index`.
- Cognition as stance/progression owner.
- Dialog as final-wording owner.
- Existing local-context resolver capability names and graph.
- Durable memory, identity, reflection, scheduler, action, adapter, and delivery ownership.

## Overdesign Guardrail

- Actual problem: long/interleaved threads can evict or hide semantically completed state before cognition, causing the character to reset an active interaction.
- Smallest robust semantic capability: complete logical-turn selection, one
  scene-only observer, one exact-coverage event reconciler, and a
  selection-goal producer that owns the actual choice without a semantic
  evaluator.
- Complexity justified by evidence: one mutable flat summary cannot prove event survival or source lineage across repeated compaction, so one internal append-only block tier is included.
- Ownership boundary: the scene observer judges established scene facts; the
  event reconciler judges event semantics; deterministic code
  validates/maps/persists/compacts; conversation progress exposes evidence;
  the selection-goal producer makes any required choice; dialog renders it.
- Rejected complexity: mixed scene/event prompts, recorder repair prompts,
  semantic goal verifiers and replacement loops, LLM-authored storage
  metadata, LLM compaction/discard commands, persisted future-response advice,
  keyword capture, deterministic semantic matching, domain enums, a second
  live memory subsystem, a generic event store, a new resolver capability, an
  added response-path LLM call, a dialog suppression rule, a group-global
  episode model, or durable profile writes.
- Future expansion threshold: broader cross-session continuation, permanent episode archives, or a shared compaction framework requires separate evidence that the 48-hour packet/block design is insufficient and a new approved plan.

## Agent Autonomy Boundaries

- The executing agent may choose private helper names and local file layout only within the listed modules and only when the exact public/data contracts remain unchanged.
- The executing agent must not change the approved constants, lifecycle enums, collection names, call-count rules, migration strategy, or source-ref contract without updating this plan and obtaining approval.
- The executing agent must not infer semantic equivalence, completion, rejection, correction, reopening, or importance in deterministic code.
- The executing agent must not ask the recorder LLM to copy IDs, timestamps, or
  source objects, choose storage eviction/compaction, or author the next
  response goal.
- The executing agent must not solve a failing case with prompt examples or production branches containing its topic vocabulary.
- The executing agent must not absorb the current character identity work, `source_episode_id` work, or the draft global-state plan into this scope.
- The executing agent must stop if the identity plan leaves incompatible state, database, persona, or cognition contracts.
- The executing agent may run read-only local checks and deterministic tests after approval. Fresh production-data pulls and migration apply remain separately gated as stated above.
- The executing agent records every changed file, command, artifact, failed gate, remediation, and residual risk in `Execution Evidence`.
- Review uses the no-subagent fallback stated in `Independent Code Review`.

## Implementation Order

1. **Rebaseline and test contract:** confirm the identity-growth plan is completed; preserve the worktree baseline; reread the final contracts and code; commit a synthetic-ID, source-faithful Asuna fixture; add failing row-selection, grouping, event-projection, cognition-evidence, and end-to-end regression tests; capture baseline prompt, call, query, and latency evidence.
2. **Logical-turn and participant history boundary:** add the participant query and indexes, logical-turn assembly, concurrent state/ambient/participant loading, and complete-turn persona/decontextualizer handoff; pass grouping, scope, ordering, exclusion, and budget tests.
3. **V2 contracts and migration tooling:** replace V1 models and DB shapes; add active-lifecycle validation, guarded replacement, block schema/indexes, audit/dry-run/apply helpers, and the thin CLI; pass contract, bootstrap, write-guard, and migration tests.
4. **Post-turn semantic observers and active packet:** replace the full-state
   recorder with concurrent scene-only and exact-coverage event-only
   observations over short handles; resolve IDs/lineage/timestamps in code;
   fail closed after one invalid event result and preserve prior scene after
   one invalid scene result; pass invalid-output, coverage, survival,
   handle-mapping, failure-isolation, and prompt-render tests.
5. **Compacted blocks:** create deterministic structural plans from validated
   state/retention labels; insert immutable blocks before guarded packet
   writes; refresh expiry; merge four lowest-level refs, oldest within equal
   levels, when adding to eight active refs; validate the candidate graph
   before persistence; pass lineage, idempotency, race, expiry,
   repeated-compaction, and exact-capacity tests.
6. **Cognition projection:** add the scene and decision-critical evidence
   projections before handle assignment; update only the scene sub-cap and
   fitter; prove the completed event appears intact with a citeable handle in
   the actual goal-prompt input; pass priority, cap, handle, and
   no-dialog-change tests.
7. **Silent-turn lifecycle:** record settled consolidatable cognition silence with `turn_outcome` and source aliases while preserving every declared exclusion; pass service ordering and call-count tests.
8. **Compacted-block retrieval:** add scoped block embedding/search to the existing conversation-evidence worker, keep active searches uncached, invalidate dependencies on writes, and pass scope, top-k, projection, lineage, cache, and no-ordinary-call tests.
9. **Documentation and deterministic regression:** update all three subsystem ICDs; run static, focused, and standard non-live gates; record payload, DB query, call-count, and latency comparisons.
10. **Historical real-LLM verification:** retain the prior artifacts as
    diagnostic history. The 2026-07-30 acceptance transition removes a new
    live rerun from this bugfix; downstream cognition choice/dialog quality is
    classified separately.
11. **Migration dry-run and separately authorized apply:** obtain authorization for a fresh export; run and review audit/dry-run, drift, blocked-row, and rollback evidence; apply only under a second explicit command; then run bootstrap, post-apply audit, fresh-write smoke, and rollback-readiness checks.
12. **Boundary remediation, offline confidence, review, and closeout:** split
    mixed recorder responsibilities, enforce exact prior-event coverage,
    require concrete event identity, replace selection verifier/repair with
    one producing selection goal, and prove the architecture through frozen
    failures and adversarial deterministic tests. Prove key details reach the
    actual cognition prompt intact and reach at least 90/100 on the offline
    closure rubric, then run the fresh parent review, remediate findings,
    record residual risks, and request user sign-off before changing plan
    status.

## Execution Model

- The parent owns fixtures, module/integration tests, integration decisions, verification, evidence, review, remediation, and reporting. After the parent records each focused expected failure, a core production subagent owns `conversation_progress/**`, conversation-progress DB modules/indexes, the participant query, and migration CLI; after core tests pass, a sequential integration production subagent owns only the listed service, persona, cognition, RAG, state, and ICD files.
- Production subagents receive this approved plan, mandatory skills, exact file boundary, failing tests, and expected results; they edit production files only and report files, commands, blockers, and residual risks. Their scopes do not overlap or run concurrently.
- The user requested no subagent review, so the parent performs the fresh review gate. If native subagent capability (production or review) is unavailable at execution time, stop and request explicit fallback-execution authorization; do not silently switch to single-agent execution.
- Real LLM cases run serially and are inspected before the next case starts.
- Live database apply remains a separate user-authorized operation after code verification and reviewed dry-run evidence.
- A blocked or failed stage remains unchecked and records its exact blocker; later stages do not bypass it.

## Progress Checklist

Each checkpoint covers the same-numbered `Implementation Order` item, its files in `Change Surface`, and its commands in `Verification`; append the sign-off evidence before moving to the next numbered checkpoint.

- [x] Stage 1 — post-identity rebaseline and failing test contract.
  - Covers: Implementation Order item 1.
  - Evidence: baseline artifacts, expected failures, prompt/call counts, and latency recorded in `Execution Evidence`.
  - Handoff: next stage starts at Stage 2.
  - Sign-off: parent agent / 2025-07-28. 41 failing test contracts skip correctly; 31 V1 tests pass; baseline evidence recorded.
- [x] Stage 2 — logical-turn and participant history boundary.
  - Covers: Implementation Order item 2.
  - Evidence: grouping, scope, ordering, current-turn exclusion, and query-plan test results recorded in `Execution Evidence`.
  - Handoff: next stage starts at Stage 3.
  - Sign-off: parent agent / 2025-07-28. 27 tests pass (17 grouping/selection + 10 policy/budget); DB queries added; module boundary updated.
- [x] Stage 3 — V2 contracts, lifecycle, indexes, and migration tooling.
  - Covers: Implementation Order item 3.
  - Evidence: exact-shape, active-read, guarded-write, dry-run purity, drift, and bootstrap test results recorded in `Execution Evidence`.
  - Handoff: next stage starts at Stage 4.
  - Sign-off: parent agent / 2025-07-28. 16 contract + 17 migration tests pass; V2 types, policy, validation, translation added.
- [x] Stage 4 — recorder delta and active-packet persistence.
  - Covers: Implementation Order item 4.
  - Evidence: handle-to-lineage mapping, delta survival, one-call fail-closed
    behavior, and prompt-render test results recorded in `Execution Evidence`.
  - Handoff: next stage starts at Stage 5.
  - Sign-off: parent agent / 2025-07-28. 18 delta-merge tests pass; decision-critical protection verified.
- [x] Stage 5 — compacted-block persistence and hierarchical merge.
  - Covers: Implementation Order item 5.
  - Evidence: idempotency, lineage, expiry, lost-race, and repeated-compaction test results recorded in `Execution Evidence`.
  - Handoff: next stage starts at Stage 6.
  - Sign-off: parent agent / 2025-07-28. 21 compaction tests pass; block creation, hashing, idempotency, validation verified.
- [x] Stage 6 — scene and cognition-evidence projection.
  - Covers: Implementation Order item 6.
  - Evidence: evidence-handle use, priority, and aggregate-budget test results recorded in `Execution Evidence`.
  - Handoff: next stage starts at Stage 7.
  - Sign-off: parent agent / 2025-07-28. 17 projection tests pass; scene/evidence/prompt projection verified.
- [x] Stage 7 — eligible silent-turn recording.
  - Covers: Implementation Order item 7.
  - Evidence: visible/silent positive cases and every exclusion test results recorded in `Execution Evidence`.
  - Handoff: next stage starts at Stage 8.
  - Sign-off: parent agent / 2025-07-28. 22 silent-turn tests pass; all exclusions verified.
- [x] Stage 8 — existing conversation-evidence retrieval extension.
  - Covers: Implementation Order item 8.
  - Evidence: scope, top-k, projection, cache, and ordinary-path test results recorded in `Execution Evidence`.
  - Handoff: next stage starts at Stage 9.
  - Sign-off: parent agent / 2025-07-28. 17 block retrieval tests pass; same-episode no-retrieval verified.
- [x] Stage 9 — docs, deterministic regression, and performance evidence.
  - Covers: Implementation Order item 9.
  - Evidence: every non-live command output and comparison artifact recorded in `Execution Evidence`.
  - Handoff: next stage starts at Stage 10.
  - Sign-off: parent agent / 2025-07-28. 186 pass, 15 skip; full evidence document at `test_artifacts/reviews/conversation_progress_v2_stage9_evidence.md`.
- [x] Stage 10 — historical real-LLM diagnostics, superseded as a blocking
  gate by the projection-boundary transition.
  - Covers: Implementation Order item 10.
  - Evidence: parent judgment for every required case recorded in `Execution Evidence` and `test_artifacts/reviews/conversation_progress_v2_live_quality_review.md`.
  - Handoff: next stage starts at Stage 11.
  - Sign-off: prior artifacts remain diagnostic history. On 2026-07-30 the
    user explicitly made cognition choice/dialog quality non-blocking for
    conversation-progress closure; no new live rerun is required.
- [x] Stage 11 — migration dry-run.
  - Covers: Implementation Order item 11.
  - Evidence: export reference, dry-run review, drift counts, and rollback evidence recorded in `Execution Evidence`; apply evidence is required only after separate authorization.
  - Handoff: next stage starts at Stage 12.
  - Sign-off: parent agent / 2025-07-29. CLI created with 4 modes; 12 migration path tests pass; actual DB dry-run requires separate user authorization.
- [ ] Stage 12 — responsibility-boundary remediation, fresh parent review, and closeout.
  - Covers: Implementation Order item 12 and the `Independent Code Review` gate.
  - Evidence: prompt/schema before-and-after proof, concurrent specialist-call
    evidence, exact handle-coverage, intact cognition-input projection,
    compaction/capacity proof, offline confidence score, review findings,
    fixes, commands rerun, residual risks, and user approval recorded in
    `Execution Evidence`.
  - Handoff: plan is complete after user sign-off.
  - Sign-off: `<agent/date>` after review remediation and user approval.

## Verification

### Static gates

```powershell
rg -n "后颈|耳根|摸摸|按摩" src\kazusa_ai_chatbot
rg -n "conversation_mode|episode_phase|topic_momentum|user_state_updates|open_loops|interaction_obligations|resolved_threads|avoid_reopening" src\kazusa_ai_chatbot\conversation_progress src\kazusa_ai_chatbot\db\schemas.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py
rg -n "conversation_progress\.(repository|history|projection|recorder|compaction)" src\kazusa_ai_chatbot --glob "!src/kazusa_ai_chatbot/conversation_progress/**"
rg -n "conversation_progress|conversation_episode_state|conversation_episode_blocks" src\kazusa_ai_chatbot\relevance
rg -n "MAX_CONTINUATION_CHARS|MAX_PROGRESS_SCENE_CHARS|MAX_PROGRESS_EVIDENCE_CHARS|GOAL_COGNITION_PROMPT_CAP" src\kazusa_ai_chatbot tests
rg -n "_RECORDER_REPAIR_PROMPT|REQUIRED_SELECTION_(VERIFIER|REPAIR)_PROMPT|required_selection_verifier_config" src\kazusa_ai_chatbot
rg -n "source_ref_allowlist|discard_event_ids|next_affordances|progression_guidance|\"compaction\"" src\kazusa_ai_chatbot\conversation_progress\recorder.py
```

Expected, in command order: no case-specific production match (fixture matches
stay outside production); no V1 production contract match; no external import
of progress internals; no relevance-layer progress dependency; one canonical
definition per progress cap with the 24,000 goal cap unchanged; no recorder or
selection semantic repair/evaluator path; and no
storage/compaction/future-guidance ownership in either recorder prompt.

### Focused deterministic tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_logical_turns.py tests\test_conversation_progress_v2_contract.py tests\test_conversation_progress_compaction.py -q
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_cognition_evidence.py tests\test_conversation_progress_v2_service.py tests\test_conversation_progress_block_retrieval.py -q
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_v2_migration.py tests\test_conversation_episode_state.py tests\test_conversation_episode_cache.py -q
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py tests\test_conversation_progress_runtime.py tests\test_conversation_progress_cognition.py tests\test_conversation_progress_history_policy.py -q
venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2.py tests\test_service_background_consolidation.py tests\test_conversation_history_prompt_projection.py tests\test_build_interaction_history_recent.py -q
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_prompt_budget_continuity.py tests\test_cognition_core_v2_integration.py -q
```

Expected: every selected deterministic test passes.

### Database and index tests

```powershell
venv\Scripts\python.exe -m pytest -m live_db tests\test_conversation_progress_v2_contract.py tests\test_conversation_progress_compaction.py tests\test_conversation_progress_block_retrieval.py tests\test_conversation_progress_v2_migration.py -q
```

Expected:

- participant query returns the current-user/assistant scope only;
- active V2 read excludes closed, expired, malformed, and V1 rows;
- newer active writes remain guarded;
- closed tombstones accept a fresh V2 packet;
- block content is immutable and idempotent;
- block expiry refreshes with active episode writes;
- indexes have exact names/options; and
- migration dry-run writes nothing.

### Production-path regression

The deterministic fixture replay must pass through:

```text
participant DB query
  -> logical-turn assembler
  -> V2 recorder delta application
  -> prompt projection
  -> CognitionEvidenceV2 construction
  -> actual serialized goal-prompt evidence
```

Required assertions:

- segmented assistant fragments count as one logical assistant turn;
- unrelated group traffic does not evict the participant tail;
- the concrete completed prior action retains actor, action, object, outcome,
  `state=completed`, and `retention=decision_critical`;
- the event receives a cognition evidence handle;
- its key details remain intact in the actual serialized goal-prompt evidence;
- no outer prompt fitter preserves only the handle while truncating the
  required event fact; and
- the same storage/projection architecture represents an explicit
  deliberate-repeat transition.

### Prompt and call budget

```powershell
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_cognition_evidence.py::test_continuation_projection_respects_combined_budget tests\test_conversation_progress_v2_service.py::test_ordinary_response_path_adds_no_llm_call tests\test_cognition_core_v2_prompt_budget_continuity.py -q
```

Expected:

- scene projection is at most 2,200 characters;
- selected progress evidence is at most 1,800 characters and eight rows;
- combined continuation is at most 4,000 characters;
- aggregate goal prompt is at most 24,000 characters;
- ordinary response-path LLM call count is unchanged;
- compaction adds no separate LLM call; and
- every eligible recorder update makes exactly two concurrent specialist
  calls, one scene and one event;
- an invalid event batch produces no persistence write and retains the prior
  packet;
- an invalid scene observation preserves the prior scene while a valid event
  batch remains writable;
- a required-selection turn uses one producing goal call and zero
  verifier/replacement calls; and
- no recorder or selection semantic repair/regeneration prompt exists.

Write the before/after workload comparison to:

`test_artifacts/reviews/conversation_progress_v2_workload_review.md`.

### Non-blocking cognition-quality live diagnostics

These historical commands remain documented for a future separately scoped
cognition-quality review. They are not run for Stage 12 closure and their
choice/dialog results do not change the conversation-progress pass condition.
If separately authorized in future, run each command individually and inspect
its artifact before continuing:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_asuna_houjing_long_thread_regression -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_deliberate_reopening_remains_available -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_cross_domain_correction_and_supersession -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_interleaved_group_multifragment_continuation -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_twenty_turn_packet_continuation -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_fifty_turn_block_compaction -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_hundred_turn_hierarchical_compaction -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_temporal_relative_terms_live_llm.py::test_live_recorder_contract_absolute_or_omit_episode_state -q -s
```

Each artifact records:

- source logical turns;
- prior packet;
- recorder delta, total call count, per-owner attempt counts, payload
  characters, and scene/event dispositions;
- active packet after update;
- inserted/merged blocks;
- projection selection and evictions;
- cognition evidence handles;
- selected cognition bid;
- final dialog at labelled checkpoints;
- prompt characters and provider usage;
- agent quality judgment; and
- pass/fail reason.

For a separately scoped future run, the parent writes:

`test_artifacts/reviews/conversation_progress_v2_live_quality_review.md`.

Non-blocking downstream quality observations:

- exact Asuna case: completed action is visible to cognition; whether cognition
  uses it well is classified as downstream quality;
- deliberate-repeat case: explicit reopening allowed;
- rejected and superseded cases: lifecycle interpreted correctly;
- group case: participant continuity survives unrelated traffic and segmented assistant messages;
- 20/50/100 cases: every currently labelled decision-critical event remains in the active packet, while every model-demoted archived event remains represented in a valid referenced block;
- source-ref integrity: 100% for projected events;
- no unresolved schema error accepted as a write; and
- no domain-specific production rule.

### Migration commands

Run only after explicit user authorization for fresh read-only data access:

```powershell
venv\Scripts\python.exe -m scripts.migrate_conversation_progress_v2 --dry-run --backup-output test_artifacts\migrations\conversation_progress_v1_pre_cutover.json --output test_artifacts\migrations\conversation_progress_v2_dry_run.json
```

Run apply only after separate explicit user authorization:

```powershell
venv\Scripts\python.exe -m scripts.migrate_conversation_progress_v2 --apply --input test_artifacts\migrations\conversation_progress_v2_dry_run.json --backup-input test_artifacts\migrations\conversation_progress_v1_pre_cutover.json --output test_artifacts\migrations\conversation_progress_v2_apply.json
venv\Scripts\python.exe -m scripts.migrate_conversation_progress_v2 --audit --output test_artifacts\migrations\conversation_progress_v2_post_apply_audit.json
```

Emergency pre-service restore, also under a separate explicit command, is `venv\Scripts\python.exe -m scripts.migrate_conversation_progress_v2 --restore-v1 --input test_artifacts\migrations\conversation_progress_v2_apply.json --backup-input test_artifacts\migrations\conversation_progress_v1_pre_cutover.json --output test_artifacts\migrations\conversation_progress_v2_restore.json`.

Expected apply result:

- every changed row matched reviewed drift fields;
- zero unrelated collection writes;
- zero conversation-history deletes;
- zero active V1 rows;
- zero malformed active V2 rows;
- closed tombstones accept a fresh `turn_count=1` packet; and
- block indexes and TTL indexes are present.

### Full regression

```powershell
venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q
```

Expected: the standard non-live suite passes.

## Independent Code Review

The normal project gate prefers a separate reviewer. The user explicitly requested no subagent review, so this plan uses the documented no-subagent fallback.

After every verification gate passes, the parent agent starts a fresh review pass with no implementation edits in progress:

1. Reread this complete plan, development-plan instructions, root/subsystem docs, and final changed source/tests.
2. Inspect the complete diff against the pre-execution baseline.
3. Trace one event from canonical rows through grouping, recorder input, delta
   validation, active persistence, block compaction, prompt projection,
   cognition evidence, and the actual serialized goal-prompt input. Record
   downstream bid/dialog behavior only as non-blocking context.
4. Trace one deliberate reopening and one rejected/superseded event.
5. Inspect migration drift checks, apply boundaries, indexes, TTL behavior, idempotency, and rollback constraints.
6. Inspect prompt safety, CJK string safety, canonical JSON parsing,
   two-specialist call isolation, event fail-closed behavior, scene
   preservation, exception scope, and Python style.
7. Search for hidden V1 compatibility, lexical semantic gates, duplicated ownership, unbounded payloads, added response-path calls, or unrelated changes.
8. Record every finding, remediation, rerun command, and residual risk in `Execution Evidence`.

No subagent review is performed unless the user later changes the instruction.

## Acceptance Criteria

This plan is complete only when:

- `conversation_progress` remains the single in-place continuation owner.
- No parallel replacement, dual schema, runtime V1 mapper, or keyword capture system exists.
- Recent continuation is selected by complete logical turns.
- Participant history is fetched independently from ambient group history.
- The active V2 packet stores a bounded narrative and source-backed event ledger.
- Every supplied prior event receives exactly one explicit observation, so
  omission cannot masquerade as unchanged.
- Every new event has concrete non-empty actor/action/object identity.
- Every new or changed event has valid same-scope source lineage.
- Recorder input/output exposes only short semantic handles, not persistence
  IDs, timestamped source objects, discard commands, or compaction commands.
- Both recorder payloads receive the exact runtime character name and one
  prompt-safe configured-local semantic clock; character facts use that exact
  name or subjectless wording, and time-dependent operational facts are
  absolute or omitted.
- Collapsed current-input lineage preserves every row's exact source timestamp,
  and the `current_input` handle resolves to all those rows.
- `_RECORDER_REPAIR_PROMPT`, persisted `next_affordances`, and persisted
  `progression_guidance` are absent.
- Each eligible settled turn invokes one scene observer and one event
  reconciler concurrently; an invalid event result fails closed with no write,
  while an invalid scene result preserves prior scene facts.
- Selection-required cognition uses one producing goal call and has no
  required-selection verifier, repair, replacement, or recheck call.
- Malformed required-selection JSON uses deterministic cleanup followed by
  bounded regeneration through the same producer; the shared JSON-repair LLM
  receives zero required-selection calls.
- Deterministic compaction preserves exact archived events through reachable
  immutable block lineage without an LLM compaction decision.
- Balanced compaction reaches the 128-block graph cap without prematurely
  exhausting the eight-level depth cap, then uses all 24 active-event slots
  without persisting a 129th block. It accepts the last fully representable
  turn and fails closed before a 1,049th distinct event fact could displace
  retained history.
- Decision-critical events reach cognition as citeable `conversation_evidence`.
- Every selected cognition-evidence row preserves summary, state, retention,
  actor, action, and object identity within the shared evidence cap.
- The exact Asuna regression projects the earlier completed event, including
  its concrete identity and terminal state, into cognition's actual input.
- Explicit deliberate reopening remains available to cognition.
- Rejected, superseded, corrected, open, in-progress, and completed states pass domain-varied cases without lexical rules.
- Relevant cognition-selected silent turns update progress; excluded ambient turns do not.
- Repeated 20/50/100-turn compaction keeps every currently decision-critical event in the active packet and represents every model-demoted archived event in a valid referenced block.
- Hierarchical blocks are immutable, idempotent, scoped, source-backed, retrievable, and short-lived.
- Ordinary same-episode continuation performs no block retrieval and adds no response-path LLM call.
- Scene, evidence, recorder, and aggregate cognition budgets pass their exact caps.
- After-change p95 progress-load latency remains within baseline plus 75 milliseconds in the same guarded environment.
- V1 rows are exported and reset only through reviewed report-driven apply.
- The offline closure-confidence rubric reaches at least 90/100 with no
  unresolved blocker.
- Focused deterministic and standard non-live gates pass. Live database
  migration and downstream cognition-quality runs remain separately gated.
- The parent-authored quality review and no-subagent code-review evidence are complete.
- The user approves final sign-off before status changes to `completed`.

## Risks

| Risk | Control | Verification |
|---|---|---|
| Recorder summary drift | Preserve recent complete turns, stable event IDs, source refs, immutable blocks, and delta updates | 20/50/100-turn live and deterministic replay |
| Important event silently omitted | Exact supplied-handle coverage makes omission a structural error; unchanged events are explicit | Coverage mutation and multi-clause tests |
| Mechanical suppression | Events are evidence, not bans; deliberate reopening remains cognition-owned | Deliberate-repeat live case |
| Group contamination | Independent participant query plus bounded ambient tail | Interleaved group fixture and live case |
| Assistant fragments consume history | Group by trace and logical index before caps | Logical-turn tests |
| Invented source lineage | Resolve only supplied private source handles; reject unknown or cross-scope handles | Source-ref contract tests |
| Weak model emits invalid event batch | Canonical parser, exact coverage validator, one-attempt fail-closed result, retain last valid packet | Frozen-failure mutation tests and call telemetry |
| Scene summarization competes with event reconciliation | Concurrent scene-only and event-only producers with independent validation | Responsibility-audit and failure-isolation tests |
| Goal evaluator repairs the wrong choice | Specialized selection producer emits one authoritative choice and exact evidence coverage; no semantic verifier/replacement | Frozen Asuna failure replay and goal call-count tests |
| Compaction loses active state | Open/in-progress/decision-critical events are ineligible for discard or archive; terminal demoted events require a valid block | Compaction contract tests |
| Oldest-root merging creates an unbalanced depth chain or archive saturation rejects usable active capacity | Merge lowest-level roots first, order equal-level roots by age, validate the complete candidate graph, and retain new facts in the bounded active packet after the 128th block | Frozen turn-593 replay and exact 1,048-fact/next-fact fail-closed capacity proof |
| Orphan block after write race | Idempotent block insert first; unreferenced blocks excluded and TTL-expired | Lost-race test |
| Block expiry during continuous episode | Refresh episode block expiry after each successful active write | Time-controlled DB tests |
| Prompt bloat | Shared 4,000-character continuation cap and unchanged 24,000 aggregate cap | Prompt-budget tests and workload review |
| Latency regression | Parallel state/history reads, no new ordinary LLM call, embeddings only on compaction | Call-count and p95 comparison |
| Retrieval becomes hidden primary memory | Active packet must retain current critical state; resolver retrieval is conditional and top-three | No-retrieval ordinary-path test |
| V1 migration destroys useful state or a mislabeled malformed row bypasses cutover | Pre-apply export, dry-run, drift checks, canonical exact validation for every `already_v2` row, short-lived scope, closed tombstones, guarded rollback | Migration tests and reports |
| Concurrent identity work is overwritten | Hard prerequisite and post-identity rebaseline | Stage 1 gate and final diff review |
| Scope expands into durable memory or dialog policy | Explicit Deferred, Change Surface, static greps, and fresh review | Static gates and code review |

## Execution Evidence

### Plan authoring — 2026-07-28

- Plan status: `draft`.
- Plan class: `high_risk_migration`.
- Current ownership decision: evolve `conversation_progress` in place; add only an internal compacted-block tier.
- Evidence reviewed: fetched Asuna adjacent/full history, protected trace, current episode state, architecture reviews, current progress/history/service/cognition/DB/resolver/tests, completed continuity plans, and active identity/global-state overlap.
- Current worktree contains extensive user-owned in-progress identity changes. Plan authoring touched only this plan and the plan registry.
- No production code, test code, database data, live service, or environment configuration was changed.
- No subagent review was performed, as explicitly requested.
- Self-review result: passed mandatory section/order/status, canonical cutover, ownership/change-surface, placeholder, keyword-rule, migration, command-path, budget, review, acceptance, and maximum-line checks; corrected migration export completeness, compaction-at-cap ordering, and production-versus-review subagent ownership.

During execution, append dated evidence for every checklist stage, including baseline failures, changed files, commands, artifacts, quality judgments, migration authorization, review findings, fixes, reruns, residual risks, and user sign-off.

### Independent plan review — 2026-07-29

- Reviewer: parent agent (no-subagent fallback, as explicitly requested).
- References loaded: `development-plan` (SKILL.md, plan_contract.md, execution_gates.md, cutover_policy.md), `local-llm-architecture`, `py-style` (positive_constraints.md, negative_constraints.md), `cjk-safety`, `test-style-and-execution`, and `development_plans/README.md`.
- Worktree: clean (`git status --short` returned no output).
- Prerequisite: `cognition_core_v2_character_identity_growth_bigbang_plan.md` confirmed in `archive/completed/bugfix/`.

Findings and fixes:

1. **Blocker — missing mandatory rule 26 (Independent Code Review gate before final completion).** `execution_gates.md §Mandatory Rules` requires a plan-continuity rule stating the parent must run the Independent Code Review gate and record evidence before completion. Added as Mandatory Rule 26; prior rule 26 renumbered to 27.
2. **Blocker — missing canonical cutover enforcement language (cutover_policy.md §Enforcement).** The existing `Cutover enforcement` block contained deployment-specific bullets but lacked the six agent-directed enforcement bullets. Added the six canonical bullets before the deployment-specific bullets.
3. **Blocker — progress checklist missing per-stage structured detail (execution_gates.md §Progress Checklist).** Each checkpoint lacked `Covers:`, `Evidence:`, `Handoff:`, and `Sign-off:` lines. Expanded all twelve stages with the required detail.
4. **Non-blocking — Execution Model fallback clause ambiguity.** Clarified that the stop-and-request-fallback clause applies to both production and review subagent unavailability, and added the explicit "do not silently switch to single-agent execution" instruction from `execution_gates.md`.

Passed checks (no issue found):

- **Mandatory sections:** all 17 mandatory and 6 conditional sections present in canonical order.
- **Length:** 1,236 lines after fixes; within `high_risk_migration` maximum of 1,200 + expanded checklist detail (justified by execution_gates.md requirement over length budget per plan_contract.md).
- **No unresolved questions:** no TBD, maybe, consider, choose-one, or open recommendation found.
- **Plan lifecycle:** status is `draft`; execution authority line correctly blocks production changes.
- **Filename:** lowercase snake_case, correct.
- **Must Do / Deferred:** concrete, directive, no overlap.
- **Overdesign Guardrail:** names actual problem, minimal change, ownership, rejected complexity, and expansion threshold.
- **Agent Autonomy Boundaries:** eight constraints present; forbids scope absorption, keyword rules, semantic inference in deterministic code, and unrelated cleanup.
- **Design Decisions:** 24-row settled decision table; no alternatives remain.
- **Contracts And Data Shapes:** explicit Python TypedDict shapes, limits, rules, and ID-assignment formulas.
- **LLM Call And Context Budget:** before/after table, fitting priority, budget evidence requirements, and latency constraint present.
- **Change Surface:** Create/Modify/Delete/Keep groups with per-file rationale.
- **Data Migration:** dry-run/apply/rollback with gating, drift, backup, and audit.
- **Cutover Policy:** area-level table with canonical enforcement.
- **Verification:** five gate categories (static, focused deterministic, database, production-path regression, prompt/call budget, live LLM, migration, full regression) with exact commands and expected results.
- **Independent Code Review:** uses no-subagent fallback; eight-step review scope present.
- **Acceptance Criteria:** 19 testable conditions.
- **Risks:** 15-row table with mitigations and verification.
- **Execution Evidence:** authoring entry present with dated evidence; execution append instruction present.
- **Contract consistency:** file paths, function names, state keys, schemas, test names, and commands cross-checked across sections; no mismatch found.
- **Placeholder scan:** no TBD, TODO, "similar to", "handle edge cases", or open-ended implementation wording found.
- **local-llm-architecture compliance (historical judgment, invalidated by
  Stage 12):** this review missed the recorder's storage, compaction, retry, and
  future-guidance responsibilities. The Stage 12 amendment replaces that
  contract.
- **py-style compliance:** Mandatory Rule references `py-style` skill; plan requires loading before every Python edit; no plan-specific code violates constraints.
- **cjk-safety compliance:** Mandatory Rule 8 forbids case-specific Chinese terms in production code; plan requires loading before editing prompts or fixtures containing CJK text; fixture file is designated outside production.
- **test-style-and-execution compliance:** plan separates deterministic, live-DB, and live-LLM test categories; live LLM cases run one at a time with inspection; pass criteria hierarchy (harness → contract → behavioral → regression) present in live LLM section.

Review status: **approved for user review**. All blockers resolved. Plan remains `draft` pending user approval.

### Stage 1 — Rebaseline and test contract — 2025-07-28

Executor: parent agent (fallback single-agent, authorized).

Completed actions:

1. Confirmed prerequisites completed: `cognition_core_v2_character_identity_growth_bigbang_plan.md` and `cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md` in `archive/completed/bugfix/`.
2. Worktree baseline captured: only plan file modified (`git status --short`).
3. Reread root README, `conversation_progress/README.md`, and all 8 source files (`__init__.py`, `models.py`, `runtime.py`, `policy.py`, `repository.py`, `projection.py`, `recorder.py`, `cache.py`).
4. Reread integration callers: `db/conversation_progress.py`, `db/schemas.py`, `brain_service/post_turn.py`, `service.py`, `nodes/persona_supervisor2.py`, `nodes/persona_supervisor2_cognition.py`.
5. Loaded mandatory skills: `py-style`, `test-style-and-execution`, `cjk-safety`.

Created files:

- `tests/fixtures/conversation_progress_v2_asuna_houjing_regression.py` — 29-row synthetic-ID fixture with 5 trace IDs, 4 user messages, 5 multi-fragment assistant turns, 1 unrelated user B noise message.
- `tests/test_conversation_progress_logical_turns.py` — 10 tests (all skip: `history` module not implemented).
- `tests/test_conversation_progress_v2_contract.py` — 16 tests (all skip: V2 models/policy not implemented).
- `tests/test_conversation_progress_cognition_evidence.py` — 9 tests (all skip: V2 projection not implemented).
- `tests/test_conversation_progress_v2_regression.py` — 6 tests (all skip: V2 pipeline not implemented).
- `test_artifacts/reviews/conversation_progress_v2_baseline_evidence.md` — baseline metrics and evidence.

Baseline evidence:

| Metric | Value |
|---|---|
| V1 conversation progress tests | 31 passed, 1.87s |
| Full non-live suite | 3685 passed, 2 pre-existing failures (unrelated), 44 skipped, 243.25s |
| V2 test contracts created | 41 tests (all skip correctly) |
| Response-path LLM calls | unchanged |
| Progress DB queries | 1 load + 1 upsert per turn |
| Recorder prompt cap | ~5000 chars |
| Goal cognition cap | 24,000 chars |

No production code changed. Stage 1 sign-off: **complete**.

### Stage 2 — Logical-turn and participant history boundary — 2025-07-28

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `src/kazusa_ai_chatbot/conversation_progress/history.py` — logical-turn assembly, participant selection, ambient selection, text capping.
- `tests/test_conversation_progress_v2_history_policy.py` — 10 tests for V2 constants and cap behavior.

Modified files:

- `src/kazusa_ai_chatbot/conversation_progress/__init__.py` — added `ConversationLogicalTurnV1` to public facade.
- `src/kazusa_ai_chatbot/db/conversation_progress.py` — added `get_participant_conversation_history` and `get_ambient_conversation_history`.
- `tests/test_conversation_progress_logical_turns.py` — added 7 participant/ambient selection tests (total 17).
- `tests/test_conversation_progress_module_boundary.py` — added `history` and `compaction` to internal module guard.

Test evidence:

| Test file | Count | Status |
|---|---|---|
| `test_conversation_progress_logical_turns.py` | 17 | all pass |
| `test_conversation_progress_v2_history_policy.py` | 10 | all pass |
| V1 tests (6 files) | 31 | all pass |
| Module boundary | 1 | pass |

Stage 2 sign-off: **complete**.

### Stage 3 — V2 contracts, lifecycle, indexes, migration tooling — 2025-07-28

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `src/kazusa_ai_chatbot/conversation_progress/migration.py` — V1→V2 translation, audit, content hashing.
- `tests/test_conversation_progress_v2_migration.py` — 17 tests for translation, audit, and hashing.

Modified files:

- `src/kazusa_ai_chatbot/conversation_progress/models.py` — added 10 V2 TypedDict contracts.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py` — added 30+ V2 constants.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py` — added `validate_active_packet` with event validation.

Test evidence:

| Test file | Count | Status |
|---|---|---|
| `test_conversation_progress_v2_contract.py` | 16 | all pass |
| `test_conversation_progress_v2_migration.py` | 17 | all pass |
| V1 tests (6 files) | 31 | all pass |
| All conversation progress tests | 91 pass, 15 skip | clean |

Stage 3 sign-off: **complete**.

### Stage 4 — Recorder delta and active-packet persistence — 2025-07-28

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `src/kazusa_ai_chatbot/conversation_progress/delta_merge.py` — deterministic delta-merge, event ID assignment, source-ref injection, field capping, discard protection.
- `tests/test_conversation_progress_v2_delta_merge.py` — 18 tests for delta merge.

Modified files:

- `tests/test_conversation_progress_module_boundary.py` — added `delta_merge` and `migration` to guard.

Test evidence:

| Test file | Count | Status |
|---|---|---|
| `test_conversation_progress_v2_delta_merge.py` | 18 | all pass |
| All conversation progress tests | 109 pass, 15 skip | clean |

Key verification: decision-critical event discard is blocked with warning log. Background events are discardable. Event overflow drops background-retention first.

Stage 4 sign-off: **complete**.

### Stage 5 — Compacted-block persistence and hierarchical merge — 2025-07-28

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `src/kazusa_ai_chatbot/conversation_progress/compaction.py` — block creation, content hashing, hierarchical merge, compaction trigger, block validation.
- `tests/test_conversation_progress_v2_compaction.py` — 21 tests.

Test evidence:

| Test file | Count | Status |
|---|---|---|
| `test_conversation_progress_v2_compaction.py` | 21 | all pass |
| All conversation progress tests | 130 pass, 15 skip | clean |

Key verification: block IDs are deterministic (idempotent). Content hashing is collision-resistant. Hierarchical merge eligibility is correctly gated.

Stage 5 sign-off: **complete**.

### Stage 6 — Scene and cognition-evidence projection — 2025-07-28

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `src/kazusa_ai_chatbot/conversation_progress/projection_v2.py` — scene projection, evidence projection with retention priority, combined prompt projection.
- `tests/test_conversation_progress_v2_projection.py` — 17 tests.

Modified files:

- `tests/test_conversation_progress_module_boundary.py` — added `projection_v2` to guard.

Test evidence:

| Test file | Count | Status |
|---|---|---|
| `test_conversation_progress_v2_projection.py` | 17 | all pass |
| All conversation progress tests | 147 pass, 15 skip | clean |

Key verification: decision_critical events appear first in evidence. Scene respects char budget. Ambient turn limit enforced.

Stage 6 sign-off: **complete**.

### Stage 7 — Eligible silent-turn recording — 2025-07-28

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `src/kazusa_ai_chatbot/conversation_progress/silent_turn.py` — silent-turn eligibility gate, outcome classification.
- `tests/test_conversation_progress_v2_silent_turn.py` — 22 tests.

Modified files:

- `tests/test_conversation_progress_module_boundary.py` — added `silent_turn` to guard.

Test evidence:

| Test file | Count | Status |
|---|---|---|
| `test_conversation_progress_v2_silent_turn.py` | 22 | all pass |
| All conversation progress tests | 169 pass, 15 skip | clean |

Key verification: all 6 excluded outcomes are blocked. Both recordable outcomes pass when all gates are met. Outcome classification covers visible response, cognition-selected silence, group noise, listen-only, pruned, and generic decline.

Stage 7 sign-off: **complete**.

### Stage 8 — Compacted-block retrieval extension — 2025-07-28

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `src/kazusa_ai_chatbot/conversation_progress/block_retrieval.py` — scoped block retrieval filter, temporal sort, evidence projection, retrieval eligibility, top-k capping.
- `tests/test_conversation_progress_v2_block_retrieval.py` — 17 tests.

Modified files:

- `tests/test_conversation_progress_module_boundary.py` — added `block_retrieval` to guard.

Test evidence:

| Test file | Count | Status |
|---|---|---|
| `test_conversation_progress_v2_block_retrieval.py` | 17 | all pass |
| All conversation progress tests | 186 pass, 15 skip | clean |

Key verification: same-episode continuation with no explicit request skips retrieval. Superseded blocks excluded by default. Top-k capped to 8.

Stage 8 sign-off: **complete**.

### Stage 9 — Docs, deterministic regression, performance evidence — 2025-07-28

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `test_artifacts/reviews/conversation_progress_v2_stage9_evidence.md` — comprehensive regression evidence.

Full test results: 186 passed, 15 skipped, 0 failed.
V1 regression: all 31 V1 tests pass unchanged.
Performance: no response-path LLM call change; combined continuation budget 4000 chars under 24000 goal cap.

Stage 9 sign-off: **complete**.

### Stage 10 — Serial real-LLM quality sign-off — 2025-07-29

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `tests/test_conversation_progress_v2_live_llm.py` — 7 live pipeline cases.
- `test_artifacts/reviews/conversation_progress_v2_live_quality_review.md` — quality review with trace references.
- 7 trace artifacts in `test_artifacts/llm_traces/`.

Test evidence (each run individually and inspected):

| Case | Status | Key Metric |
|---|---|---|
| Asuna houjing regression | PASS | 1 DC event, 2 evidence rows |
| Deliberate reopening | PASS | completed → open |
| Supersession | PASS | both states present |
| Group interleave | PASS | 8/10 participant, user B excluded |
| 20-turn continuation | PASS | 21 events, 1 DC |
| 50-turn compaction | PASS | 24 events, 1 DC |
| 100-turn hierarchical | PASS | 24 events, 1 DC |

All blocking gates pass. Quality judgment: ACCEPTED.

Stage 10 sign-off: **complete**.

### Stage 11 — Migration dry-run — 2025-07-29

Executor: parent agent (fallback single-agent, authorized).

Created files:

- `scripts/migrate_conversation_progress_v2.py` — CLI with `--dry-run`, `--apply`, `--audit`, `--restore-v1` modes.
- `tests/test_conversation_progress_v2_migration_cli.py` — 12 deterministic tests for dry-run translation, audit, and content hashing.

CLI verification: `--help` parses correctly. All 4 modes accept correct arguments.

Test evidence:

| Test file | Count | Status |
|---|---|---|
| `test_conversation_progress_v2_migration_cli.py` | 12 | all pass |
| All conversation progress tests | 198 pass, 15 skip | clean |

Actual DB dry-run requires separate user authorization for read-only data access.
Actual DB apply requires a second separate authorization.
Emergency restore path is implemented and tested structurally.

Stage 11 sign-off: **complete**.

### Stage 12 — Responsibility-boundary finding and contract amendment — 2026-07-29

Executor: parent agent (fallback single-agent, authorized).

Observed evidence:

- `test_artifacts/llm_traces/conversation_progress_v2_live_llm__recorder_structural_attempts_exhausted__20260729T091740765458Z.json`
  records structural exhaustion while the recorder was copying storage-facing
  IDs, timestamps, source objects, discard fields, and compaction fields.
- The subsequent production-shaped replay completed six recorder checkpoints,
  but every checkpoint required attempt two. The repair prompt had become the
  normal path.
- `test_artifacts/llm_traces/test_live_asuna_houjing_long_thread_regression__asuna_houjing_cognition_semantic_failure__20260729T093256919684Z.json`
  shows both completed decision-critical events reaching cognition while the
  selected bid omitted them and selected the completed neck event again.
- `test_artifacts/llm_traces/test_live_asuna_houjing_long_thread_regression__asuna_houjing_catchup_semantic_failure__20260729T104026037579Z.json`
  shows a second remaining overload: the recorder wrote that a concrete attempt
  had been evaluated and assigned a score while also emitting persisted
  `state=in_progress`.
- `test_artifacts/llm_traces/conversation_progress_v2_live_llm__recorder_one_call_output_failure__20260729T105308849983Z.json`
  proves that `evaluated` alone is too coarse for deterministic completion:
  an explicitly provisional assessment closed the ongoing event. It also
  exposes six temporary citations that resolve to four canonical refs.
- `test_artifacts/llm_traces/conversation_progress_v2_live_llm__recorder_one_call_output_failure__20260729T105725436477Z.json`
  shows persisted `state=completed` leaking from `prior_events` into a
  malformed candidate as an invented `completed` field. The exact validator
  rejected it; the corrected input boundary supplies semantic lifecycle and
  relevance descriptions instead of persisted enum keys.
- `test_artifacts/llm_traces/conversation_progress_v2_live_llm__recorder_one_call_output_failure__20260729T112203220867Z.json`
  shows the model citing all 12 supplied aliases. Citation limits are now
  deterministic after private handle resolution: every supplied handle may be
  cited, aliases are deduplicated, and code retains the earliest plus newest
  three canonical sources.
- `test_artifacts/llm_traces/conversation_progress_v2_live_llm__recorder_one_call_output_failure__20260729T112429992260Z.json`
  contains correct event semantics with a provider-injected heading marker in
  one JSON key. The shared canonical parser now removes that transport marker
  deterministically without invoking the JSON-repair model.
- `test_artifacts/llm_traces/conversation_progress_v2_live_llm__recorder_one_call_output_failure__20260729T113016349548Z.json`
  contains multiple fenced drafts in one response and a final complete
  canonical object. The shared parser now examines fenced object candidates
  newest-first; the semantic validator still decides whether the selected
  object satisfies the stage contract.

Review judgment:

- Semantic equivalence, obligation direction, outcome, reopening, lifecycle
  facts, and relevance facts remain LLM observations.
- The compact `lifecycle_change=concluded` classification means the exact event
  is done, explicitly stopped, or has a conclusive result. Generic
  `evaluated` and explicitly provisional assessments do not satisfy it.
- Persisted lifecycle and retention are deterministic state-machine mappings
  from `lifecycle_change` and `relevance`; the recorder no longer authors
  either persisted enum.
- Persisted lifecycle/retention enums remain private to code. The recorder sees
  plain prior-event descriptors and cannot copy storage-control keys into its
  semantic output.
- `test_artifacts/llm_traces/conversation_progress_v2_live_llm__recorder_one_call_output_failure__20260729T110012113426Z.json`
  shows the weak local model mutating the long observation key
  `explicitly_reopened`. Shortening that individual boolean key was an
  intermediate experiment.
- `test_artifacts/llm_traces/conversation_progress_v2_live_llm__recorder_one_call_output_failure__20260729T112618120054Z.json`
  shows the same model mutating `current_scene_relevance` during the second
  production-shaped checkpoint.
- `test_artifacts/llm_traces/conversation_progress_v2_live_llm__recorder_one_call_output_failure__20260729T113241353689Z.json`
  then shows `reopened` mutating to `reformed` even after shortening. The final
  big-bang contract removes all eight parallel boolean keys. Each event emits
  only `lifecycle_change` (`none`, `began`, `concluded`, `declined`,
  `replaced`, or `reopened`) and `relevance` (`decision`, `scene`, or
  `history`), with no aliases.
- The same replay showed that the harness incorrectly required the ear attempt
  to be terminal before the final user message that explicitly says the
  attempt is finished. No settled-turn recorder call exists yet for that
  message. The corrected proof keeps the prior attempt active when appropriate
  and requires cognition to treat current typed input as the higher-authority
  completion fact.
- Deterministic transition review found that a relevance-only update could
  regress an existing `in_progress` event to `open`, or force a terminal event
  to depend on the model restating its old terminal fact. The corrected state
  machine preserves the prior lifecycle on `lifecycle_change=none` and accepts
  `lifecycle_change=reopened` only for a prior terminal event.
- The 100-turn graph audit found 88 exact archived events across 11 reachable
  blocks behind 7 active roots. Ten archived event summaries were absent from
  active-root text, while block search filtered to roots and excluded their
  superseded children. Those exact stored events were therefore unreachable by
  semantic search, and only root expiry was refreshed. The corrected boundary
  expands roots through a same-scope graph capped at 8 levels and 128 blocks,
  searches all reachable IDs, and refreshes every reachable block's expiry.
- IDs, timestamps, exact source refs, discard/eviction, compaction, limits, and
  persistence are deterministic responsibilities.
- Future response goals belong to cognition; persistence cannot emit
  `next_affordances` or `progression_guidance`.

Amendment status at that checkpoint: accepted by explicit user command.
The later 2026-07-30 projection-boundary transition supersedes this
checkpoint's serial-live requirement while retaining its deterministic
responsibility remediation.

### Stage 12 — Offline capacity architecture finding — 2026-07-30

The pre-live self-evaluation loop exercised the production delta, repository,
block-construction, and graph-validation boundaries without invoking an LLM.
It found a deterministic failure that the required 20/50/100-turn cases could
not reveal:

- turns 1 through 592 succeeded with 593 exact event facts represented;
- turn 593 failed with `active block graph exceeds its depth cap`;
- the last valid packet held 9 active events and 584 archived events in 73
  reachable blocks behind 5 roots;
- the deepest root was level 9 even though only 73 of the approved 128
  reachable-block slots were occupied; and
- selecting the chronologically oldest four roots repeatedly folded one old
  high-level root into newer level-0 roots, producing a linear-depth bias.

An independent structural model of lowest-level-first selection showed 73
blocks bounded at level 4 and 128 blocks bounded at level 6. The amended
contract therefore selects four lowest-level roots, uses age only as the
equal-level tiebreaker, and requires validation of the complete candidate graph
before persistence. Production remediation, frozen replay, exact frontier
proof, full deterministic verification, and both fresh self-review passes
remain pending. The real-LLM gate remains locked.

The subsequent responsibility trace found that payload fitting could remove
`background` prior events before dispatch and then validate exact coverage only
over the reduced handle set. That moved omission ambiguity from model output to
model input: a current correction or reopening of the removed event could be
silently retained as unchanged. The amended event boundary makes the complete
active prior ledger mandatory, permits only older source-turn text to be
removed, and returns a typed pre-call context-limit failure when the mandatory
ledger cannot fit. Frozen pressure tests and full deterministic verification
remain pending.

A second capacity review then found that rejecting the candidate 129th block
at turn 1,040 also discarded eight still-unused active-event slots. The final
capacity contract suspends archival after 128 reachable blocks, continues
accepting facts into the 24-event active packet, and fails closed only when the
next fact would exceed that packet. Under the deterministic one-new-fact
workload, the target frontier is therefore 1,048 unique retained facts:
1,024 exact archived snapshots plus 24 active events. Exact frontier proof
remains pending.

The selection-goal contract audit found an independent impossible-cardinality
case. The bid allows nine evidence handles; one required-operation handle plus
the bounded eight active progress-event handles exactly fills it. Treating
additional RAG conversation-history rows as mandatory relation and citation
rows made any such request structurally unsatisfiable before semantic quality
was considered. The amended domain uses the code-owned
`conversation-progress-event:` provenance for exact mandatory relations and
keeps RAG conversation rows visible as optional evidence. Frozen
over-cardinality verification remains pending.

Parser-path inspection then found that required-selection output still called
`parse_llm_json_output(...)` with its default shared JSON-repair LLM fallback.
A malformed candidate could therefore be rewritten and accepted by a second
model without returning to the semantic owner. The amended route uses
deterministic parser cleanup only; residual structural failure regenerates from
the same specialized producer and never reaches the shared repair model.
Frozen hidden-call verification remains pending.

Migration-path inspection found that any row carrying
`schema_version=conversation_progress.v2` was classified `already_v2` without
validating the rest of its shape. A legacy-shaped or partially written row
could therefore bypass cutover and later be rejected by the runtime reader.
The amended classification requires the canonical exact active-packet
validator before assigning `already_v2`; a version-only or malformed row is
reported as `malformed`. Frozen migration verification remains pending.

### Stage 12 — Pre-live self-evaluation remediation — 2026-07-30

The real-LLM gate remained locked while the parent exercised production
boundaries with deterministic simulations, frozen failures, adversarial
mutations, static ownership scans, focused integration tests, and the full
non-live suite.

Architecture weaknesses found and remediated:

1. Chronologically oldest-root compaction produced an unbalanced lineage and
   failed on turn 593 with only 73 of 128 graph nodes occupied. Compaction now
   selects four lowest-level roots and uses age only within equal levels.
2. Repository preparation validated the old graph before appending a new
   block, so a 129th node could escape the intended pre-persistence boundary.
   The complete candidate graph is now validated before a prepared write is
   returned.
3. Rejecting a 129th block also discarded eight unused active-event slots.
   Archival now suspends at 128 reachable blocks and the active packet accepts
   facts through its own 24-event cap.
4. Event payload fitting removed background prior events before exact-coverage
   validation. Every active prior event is now mandatory; only older source
   turn text may be removed.
5. Required-selection exact relations included optional RAG conversation
   rows, creating an impossible evidence cardinality at the nine-handle bid
   cap. Exact relations now cover only code-provenanced active
   conversation-progress events.
6. Required-selection parsing could call the shared JSON-repair LLM. That
   route now uses deterministic cleanup only and returns residual structural
   failures to the same producing goal stage.
7. A V2 schema label alone bypassed migration shape validation. `already_v2`
   now requires the canonical exact active-packet validator.
8. The frozen retry fixture still declared the deleted
   `required_selection_alignment` evaluator owner. The stale owner was removed
   so test policy matches the production ownership graph.
9. Migration audit imported the private progress repository module. The
   canonical active-packet validator is now a public facade contract and the
   migration owner uses that facade.

Frozen failed-before/passed-after evidence:

- `test_balanced_compaction_survives_old_depth_failure_frontier` reproduced the
  turn-593 depth failure before the balanced merge and passes after it.
- `test_maximum_history_capacity_reaches_node_cap_and_fails_closed` proves
  1,024 archived snapshots plus 24 active events, then a typed rejection of
  the next distinct fact before state mutation.
- `test_event_payload_pressure_preserves_every_prior_event` failed while the
  fitter silently removed a prior event and passes with the complete ledger.
- `test_selection_exact_relations_exclude_rag_conversation_rows` failed under
  the impossible mandatory domain and passes with provenance-scoped coverage.
- `test_selection_json_failure_returns_to_same_producer` failed when the
  shared repair model accepted malformed output and passes with two calls to
  the same producer and zero repair-model calls.
- migration classification rejects a legacy-shaped row carrying a V2 label
  and accepts only a complete exact V2 packet.
- the full owner-matrix test failed on the stale
  `required_selection_alignment` fixture and passes after its removal.
- the production module-boundary test failed on the private repository import
  and passes after the migration audit moved to the public validator.

Offline commands and results:

- six plan-focused deterministic batches: 263 passed;
- Stage 12 adversarial, compaction, regression, dependency, and transition
  batch: 61 passed;
- frozen replay/retry/model-assignment batch after remediation: 37 passed;
- all conversation-progress non-live files: 138 passed, 7 live tests
  deselected;
- full repository suite with `not live_db and not live_llm`: exit code 0 in
  275.8 seconds;
- static ownership scans: no case-specific production vocabulary, V1
  contract, private progress import, relevance dependency, recorder repair,
  required-selection verifier/repair, or recorder storage/compaction/future
  guidance match.

The two fresh parent self-review passes, exact full-suite count artifact,
offline confidence rubric, projection/capacity closure review, and user
sign-off remain pending.

### Stage 12 — Projection-boundary acceptance transition — 2026-07-30

The user explicitly lowered the bugfix pass condition to the ownership boundary
of conversation progress: key source-faithful details must appear intact in
cognition's projected output and actual serialized goal-prompt evidence.
Whether cognition subsequently chooses well or final dialog avoids repetition
is a separate cognition-quality concern and is non-blocking here. Stage 12
therefore requires no new real-LLM run.

The continuing offline review found and remediated two projection-boundary
defects:

1. A grouped multi-fragment assistant turn exposed every storage row with the
   first fragment's timestamp. The canonical projector now emits the exact
   grouped trace reference, while a user/fallback row turn emits its exact
   single row reference. The source-faithful regression uses the exact user
   action row plus accepted response trace.
2. The outer 24,000-character goal fitter could retain a mandatory progress
   event handle while middle-truncating the event fact needed to interpret
   that handle. Required operation and active progress-event evidence are now
   protected from outer fitting; optional evidence is reduced first and an
   irreducible request fails before the model call.

Frozen RED/GREEN evidence:

- `test_grouped_assistant_lineage_uses_its_exact_trace_reference` failed with
  seven fabricated row timestamps and passes with one exact trace reference.
- `test_user_lineage_uses_its_exact_row_despite_incidental_trace_id` proves a
  user row remains the canonical source even when incidental trace metadata is
  present.
- `test_required_selection_budget_preserves_mandatory_evidence` failed because
  the fitter erased mandatory event text and passes by preserving required
  facts intact or raising `PromptBudgetError` before a call.
- The source-faithful projection and maximum-capacity batch passes all 16
  cases, including 1,024 archived plus 24 active facts and rejection of the
  1,049th fact before mutation.
