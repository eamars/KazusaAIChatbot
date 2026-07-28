# Conversation Progress V2 Long-Thread Continuation Big-Bang Plan

## Summary

- Goal: evolve the existing `kazusa_ai_chatbot.conversation_progress` subsystem in place so long, interleaved conversations retain what has already happened, what remains open, and what may be deliberately reopened, without a keyword capture or suppression system.
- Plan class: `high_risk_migration`.
- Status: `draft`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `debug-llm`, `character-test`, `database-data-pull`, `py-style`, `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: forward-only big-bang replacement of the internal conversation-progress schema, recorder contract, history assembly, persistence, prompt projection, cognition evidence projection, tests, and active short-term rows while retaining the existing public `load_progress_context(...)` and `record_turn_progress(...)` facade names.
- Highest-risk areas: semantic event loss during repeated compaction; summary drift; invalid source lineage; group-chat noise evicting the active participant thread; weak-model event-delta errors; prompt and recorder budget growth; active-row cutover; retrieval crossing its evidence boundary; and overlap with the in-progress character identity work.
- Acceptance criteria: the exact Asuna adjacent-history regression reaches cognition with the earlier completed action as citeable evidence; deliberate reopening remains allowed; participant continuity survives segmented assistant output, unrelated group traffic, and 20/50/100-turn compaction cases; every event has valid source lineage; all continuation inputs remain bounded; the ordinary response path adds no LLM call; and no production behavior depends on `后颈` or any other case-specific term.
- Execution authority: this draft authorizes documentation only. Production edits, database writes, live-service changes, migration apply mode, and plan execution require plan approval plus an explicit implementation command.

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
- LLM stages own semantic equivalence, lifecycle, importance, and compaction judgment.

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

1. Production changes remain blocked while this plan is `draft`.
2. Execution remains blocked until the in-progress character identity-growth plan is completed, signed, and committed or otherwise placed on a clean preserved baseline.
3. At execution start, rerun `git status --short`, reread the root and relevant subsystem documentation, and rebaseline every named file and symbol. A material contract conflict requires a plan update before code changes.
4. Preserve all existing user work. The current uncommitted `source_episode_id` lineage work is outside this contract and is not a prerequisite for logical-turn grouping.
5. Keep `kazusa_ai_chatbot.conversation_progress` as the sole owner of short-term conversation continuation and retain the public facade names `load_progress_context(...)` and `record_turn_progress(...)`.
6. Create no parallel continuation module, dual read, dual write, runtime V1 mapper, alias vocabulary, feature flag, or compatibility shim.
7. Implement the V2 caller, callee, schema, repository, recorder, projection, cognition connector, retrieval extension, tests, and ICD changes in one canonical cutover.
8. Do not add keyword, regex, body-part, phrase, language-specific, or domain-specific capture or suppression logic. Production code must not branch on `后颈`, `耳根`, `摸`, massage wording, or equivalent paraphrases.
9. The recorder/compactor LLM owns semantic event identity, lifecycle, retention priority, narrative updates, deliberate reopening, and which bounded events should move into a compacted block.
10. Deterministic code owns participant scope, logical-turn grouping, stable IDs, allowed source references, enum validation, timestamps, budgets, compaction triggers, guarded writes, expiry, ordering, retrieval limits, and telemetry.
11. A completed, rejected, corrected, or superseded event is cognition evidence, not a deterministic ban. Cognition may deliberately reopen it when the current user turn supports doing so.
12. Current user input, typed reply context, current media, and current cognition output have higher scene authority than compacted continuity.
13. Relevance remains free of V2 progress payloads. Progress loads after response eligibility and before persona cognition.
14. Reuse the existing post-turn recorder LLM route. Add no LLM call to the ordinary response path and no second ordinary compactor call.
15. Structural recorder failure may receive one stage-owned full regeneration, for a maximum of two background attempts. Semantic uncertainty does not trigger a retry.
16. Every raw recorder response passes through `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` before contract evaluation. JSON repair cannot invent event meaning, IDs, lifecycle, or source references.
17. User-derived text remains in `HumanMessage` payloads. Stable instructions remain in `SystemMessage`.
18. Record cognition-selected, consolidatable silent turns through the same post-turn recorder. Relevance-declined, listen-only, pruned, and unrelated group-noise turns remain outside progress recording.
19. Keep dialog generation, dialog verification, adapters, delivery receipts, character identity, durable user memory, reflection, scheduler, action permissions, and tool execution outside this change.
20. Keep the existing 24,000-character goal-cognition cap. Continuation fitting must occur before that boundary and must not raise it.
21. Use character caps as the blocking pre-call budget because current local routes have no canonical in-repo tokenizer. Record provider-reported token usage when supplied; an explicit `unavailable` value is valid telemetry.
22. Run live LLM cases one at a time and inspect each complete artifact. Harness completion alone is not a quality pass.
23. Run no live database migration apply command without a reviewed dry-run, a pre-apply export, and a separate explicit user command.
24. After any automatic context compaction, reread this entire plan before continuing.
25. After each major checklist stage is signed, reread this entire plan before starting the next stage.
26. The user requested no subagent review. Plan authoring and the later review gate use a fresh parent self-review unless the user explicitly changes that instruction.

## Must Do

1. Preserve a production-path baseline for the Asuna failure, including the actual row window, interaction selection, pre-cognition progress projection, evidence handles, selected cognition bid, and final dialog.
2. Replace row-count selection for continuation with deterministic logical-speaker-turn assembly.
3. Fetch the active user's interaction tail independently from the small ambient group tail so unrelated channel traffic cannot evict the subthread.
4. Replace the flat V1 lists with one canonical V2 active packet containing a scene narrative, flow fields, a source-backed semantic event ledger, recent logical-turn references, and compacted-block references.
5. Replace full-state event rewriting with ID-based event deltas so an omitted event is not silently erased.
6. Preserve actor, action, beneficiary, precondition, state, and outcome for events the model marks as obligations.
7. Assign new event IDs and logical-turn IDs deterministically and reject invented source IDs.
8. Project narrative and flow into `scene_context.conversation_continuity`.
9. Project decision-critical events into the existing `CognitionEvidenceV2` evidence list as `conversation_evidence`, before evidence handles are assigned.
10. Preserve decision-critical events ahead of background detail when fitting the continuation budget.
11. Add one append-only compacted-block collection inside the conversation-progress subsystem with immutable semantic content, source spans, producing-turn lineage, embeddings, and sliding short-term expiry.
12. Trigger compaction from bounded packet size, event count, and retained-turn count rather than lexical content.
13. Produce active-packet updates and any required block summary in the same recorder call.
14. Merge the oldest four blocks hierarchically in the same compaction output whenever adding a block to eight active refs would cross the limit.
15. Extend the existing conversation-evidence search path to retrieve scoped compacted blocks semantically and temporally when the existing local context resolver requests older conversation evidence.
16. Keep ordinary same-episode continuation independent of retrieval; the active packet must already carry current decision-critical state.
17. Record cognition-selected silent turns that have a settled, consolidatable episode trace.
18. Add sanitized continuation diagnostics for logical-turn counts, packet turn count, event counts by state/retention, projected character counts, compaction disposition, block references, retrieval source, and structural regeneration count.
19. Add a report-driven V1-to-V2 active-row reset with dry-run, drift checks, backup evidence, apply gating, and rollback constraints.
20. Update all affected ICD text and tests to the single V2 contract.
21. Prove storage correctness, projection recall, cognition use, and visible flow separately.

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
        `--> existing post-turn conversation-progress recorder
                |-- current packet delta
                |-- event updates/retirements
                `-- required compaction block in the same LLM call
                        |
                        +--> guarded active packet replace
                        `--> append-only conversation_episode_blocks
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
| Event update form | ID-based deltas against the prior ledger | Prevents silent loss from full-list omission and avoids code-side semantic matching. |
| New event identity | UUIDv5 assigned after validation from episode ID, allowed source refs, and the exact validated semantic payload | Identical retries are idempotent; code does not infer paraphrase equivalence. |
| Existing event identity | Model must reference an ID supplied in prior state | Semantic equivalence remains LLM-owned while ID validity remains deterministic. |
| Event lifecycle | `open`, `in_progress`, `completed`, `rejected`, `superseded` | Represents progression without a domain-specific ontology. |
| Retention | `decision_critical`, `active_scene`, `background` | Lets the model express semantic importance while code enforces fit order. |
| Obligation direction | Events include `is_obligation`, actor, action, beneficiary, and precondition | Preserves the completed obligation-direction contract without a second list. |
| Scene versus fact | Narrative/flow go to scene context; event rows go to evidence | Prevents key state from being buried in prose and makes cognition use measurable. |
| Evidence source kind | Reuse `conversation_evidence` | The cognition contract already recognizes this as source-backed prior dialog evidence. |
| Compaction trigger | Structural thresholds only | Size, count, and age are deterministic; event meaning remains model-owned. |
| Compaction call count | Same post-turn recorder call | Preserves ordinary response latency and avoids another semantic stage. |
| Block content | Immutable after insert; only expiry and supersession metadata may change | Preserves auditability while allowing sliding short-term lifecycle. |
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
2. Classify schema version, scope validity, status, turn count, timestamps, and drift-match fields.
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

- The recorder may emit only source refs supplied in prior events, recent logical turns, active current user row IDs, or the current accepted `llm_trace_id`.
- Every new or semantically changed event update has at least one source ref.
- Each event has at most four source refs.
- Repository validation rejects empty, unknown, duplicate, or cross-scope references.
- Source refs are audit lineage; they are not copied into public dialog.

### Event ledger

```python
class ConversationProgressEventV2(TypedDict):
    event_id: str
    semantic_summary: str
    is_obligation: bool
    actor: str
    action: str
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
- `actor`, `action`, `beneficiary`, and `precondition` are at most 160 characters each.
- `outcome` is at most 180 characters.
- `is_obligation=true` requires non-empty `actor` and `action`.
- New recorder events use an empty `event_id`; deterministic code assigns the ID after every other field validates.
- Assignment is `uuid.uuid5(uuid.NAMESPACE_URL, canonical_json).hex`, where `canonical_json` uses sorted compact JSON keys over `episode_state_id`, sorted `(ref_kind, ref_id)` pairs, and the validated semantic fields; lifecycle timestamps are excluded.
- Existing-event updates use an ID present in the supplied prior packet.
- Code merges by `event_id` only and performs no text-equivalence comparison.
- `first_seen_at` is code-owned and immutable.
- `updated_at` changes only when the model emits a valid update.
- `decision_critical`, `open`, and `in_progress` events remain in the active packet and are excluded from discard/archive candidates. The recorder may change their lifecycle or retention semantically; only a later valid packet may archive them after they are terminal and no longer `decision_critical`.

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
    next_affordances: list[str]
    progression_guidance: str
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

### Recorder delta

`ConversationProgressRecordInput` replaces `chat_history_recent` with typed logical turns and adds current-turn source aliases:

```python
{
    "scope": ConversationProgressScope,
    "storage_timestamp_utc": str,
    "character_name": str,
    "prior_episode_state": ConversationProgressStateV2 | None,
    "decontextualized_input": str,
    "interaction_logical_turns": list[ConversationLogicalTurnV1],
    "current_turn_source_refs": list[ConversationProgressSourceRefV2],
    "turn_outcome": "visible_response | cognition_silence",
    "content_plan": dict[str, str],
    "logical_stance": str,
    "character_intent": str,
    "final_dialog": list[str],
    "boundary_profile": BoundaryProfileDoc,
    "compaction_request": ConversationCompactionRequestV2 | None,
}
```

The recorder returns:

```python
class ConversationProgressRecorderDeltaV2(TypedDict):
    schema_version: Literal["conversation_progress_recorder_delta.v2"]
    continuity: Literal[
        "same_episode",
        "related_shift",
        "sharp_transition",
    ]
    status: Literal["active", "suspended", "closed"]
    episode_narrative: str
    current_thread: str
    character_stance: str
    user_goal: str
    current_blocker: str
    emotional_trajectory: str
    event_updates: list[ConversationProgressEventUpdateV2]
    discard_event_ids: list[str]
    overused_moves: list[str]
    next_affordances: list[str]
    progression_guidance: str
    compaction: ConversationCompactionOutputV2 | None
```

`ConversationProgressEventUpdateV2` has the event fields above except `first_seen_at` and `updated_at`. A new event uses `event_id=""`.

`discard_event_ids` may contain only prior events whose model-emitted state is `completed`, `rejected`, or `superseded` and whose retention is `background`. All other prior events remain unless updated or archived through the required compaction output.

### Compaction request and output

Deterministic code sets `compaction_request` when any soft threshold is crossed. The request supplies exact archive-candidate event snapshots, turn refs, and the oldest unsuperseded block summaries plus their retained event snapshots. It never supplies lexical matching rules.

```python
class ConversationCompactionOutputV2(TypedDict):
    archive_event_ids: list[str]
    retain_event_ids: list[str]
    covered_turn_refs: list[str]
    block_narrative: str
    semantic_keys: list[str]
    source_block_ids: list[str]
```

Rules:

- A required compaction output must be present and structurally valid.
- `archive_event_ids`, `retain_event_ids`, `covered_turn_refs`, and `source_block_ids` must be subsets of supplied candidates; retained IDs must come from archived active events or supplied source-block events.
- `archive_event_ids` contains at most eight terminal, non-`decision_critical` events; `retain_event_ids` contains at most eight exact event snapshots for the new block.
- `block_narrative` is at most 900 characters.
- `semantic_keys` contains at most eight model-owned semantic phrases of at most 80 characters each.
- With no source blocks, the output creates a level-0 block from active-packet events and turn refs. When creating a block from eight existing active block refs, the request must include the oldest four as `source_block_ids`; the one output block covers those source blocks plus the current archive candidates, uses `level = max(source levels) + 1`, and leaves at most five active refs.
- Invalid compaction output triggers the one permitted structural regeneration. Exhaustion retains the last valid packet and records a typed failed update.

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

Block limits are `MAX_BLOCK_EVENTS = 8`, `MAX_BLOCK_TURN_REFS = 24`, `MAX_BLOCK_SOURCE_BLOCKS = 4`, and `MAX_BLOCK_CHARS = 12000`. The block stores exact snapshots selected by `retain_event_ids`; its narrative covers every archived candidate and supplied source block. Superseded and orphan blocks retain their existing expiry and age out.

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

Within one tier, newer `updated_at` wins. Ties use stable `event_id` order. This ordering consumes model-authored lifecycle/retention labels and does not interpret event text.

Scene fitting first renders `current_thread`, `character_stance`, `progression_guidance`, the first 600 narrative characters, and the newest four chronological interaction turns at 160 characters each. It then admits `user_goal`, `current_blocker`, `emotional_trajectory`, `next_affordances`, and `overused_moves` in that exact order while room remains. The recorder prompt requires the narrative's most decision-relevant content first. Event evidence has its separate budget and is never displaced by optional scene detail.

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
    "structural_attempt_count": int,
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
| Cognition Core V2 | unchanged call count | unchanged call count | response | Combined progress scene/evidence is at most 4,000 characters; aggregate goal cap stays 24,000. |
| Dialog | unchanged | unchanged | response | Receives accepted semantic surface only; no full transcript or event ledger. |
| Progress recorder | one call after visible responsive turn | one call after visible or eligible silent turn; second attempt only for structural failure | post-turn | Human payload is capped at 24,000 characters after dropping older block hints, background events, and older logical-turn text in that order. |
| Block compaction | no separate call | no separate call | post-turn | Returned in the recorder delta when structurally requested. |
| Older-block retrieval | existing resolver-selected calls only | existing resolver-selected calls only | conditional response resolver | Ordinary same-episode path performs no retrieval call. |

Recorder fitting preserves, in order:

1. current decontextualized input;
2. current accepted content plan and actual final dialog or silence outcome;
3. prior decision-critical/open/in-progress events;
4. current source-reference aliases;
5. prior narrative and active-scene events;
6. recent complete interaction turns;
7. block hints and background events.

If required current-turn material plus required prior state cannot fit under 24,000 characters, the recorder returns a typed context-limit failure and retains the last valid packet. It does not truncate source IDs or invent a semantic update.

Budget evidence required at sign-off:

- response-path LLM call-count comparison;
- maximum and p95 prompt characters for each affected stage;
- provider-reported input/output tokens when available;
- explicit `unavailable` token telemetry when a provider omits usage;
- recorder structural regeneration count;
- block embedding call count;
- progress DB query count; and
- before/after p95 progress-load latency.

The two new history reads and active-packet read execute in one concurrent gather. After-change p95 progress-load latency must remain no greater than the Stage 1 baseline plus 75 milliseconds in the same guarded test environment.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/conversation_progress/history.py`
  - Logical-turn grouping, participant/ambient selection, source aliases, and prompt-safe turn projection.
- `src/kazusa_ai_chatbot/conversation_progress/compaction.py`
  - Event-delta application, structural compaction requests, block construction, stable IDs, fit ordering, and block lineage validation.
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
  - Export only the canonical V2 facade and public types.
- `src/kazusa_ai_chatbot/conversation_progress/models.py`
  - Replace V1 packet, prompt, record-input, result, event, block-ref, and diagnostics types.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py`
  - Replace V1 list limits/discard order with V2 turn, event, packet, compaction, scene, evidence, and expiry policy.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Replace the full-state V1 prompt/output with the V2 narrative/event-delta and optional compaction contract; add bounded structural regeneration.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py`
  - Apply validated deltas by event ID, preserve code-owned timestamps, build blocks, order idempotent block insert before guarded packet replacement, and refresh block expiry.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py`
  - Build bounded V2 prompt state and diagnostics with decision-critical-first fitting.
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
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Stop rebuilding interaction continuity from the ten-row channel window; consume the facade-selected ambient and participant logical turns.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py`
  - Consume the bounded ambient logical-turn projection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Replace `_conversation_progress_text(...)` with the V2 scene and evidence projections before evidence-handle assignment.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
  - Raise only `scene_context.conversation_continuity` to the approved 2,200-character sub-cap; retain the aggregate 24,000-character goal cap and existing `conversation_evidence` source kind.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - Preserve continuation event evidence priority during aggregate fitting without changing call count or goal semantics.
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
- Existing focused tests under `tests/`
  - Replace V1 assumptions and preserve adjacent behavior contracts.

### Delete

- V1 prompt/storage fields: `conversation_mode`, `episode_phase`, `topic_momentum`, `user_state_updates`, `assistant_moves`, `open_loops`, `interaction_obligations`, `resolved_threads`, and `avoid_reopening`.
- V1 fixed overflow order that drops resolved/closed state before lower-value detail.
- V1 recorder full-list rewriting and exact-text timestamp preservation.
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
- Smallest robust semantic capability: complete logical-turn selection plus a source-backed active narrative/event packet whose decision-critical rows reach cognition as evidence.
- Complexity justified by evidence: one mutable flat summary cannot prove event survival or source lineage across repeated compaction, so one internal append-only block tier is included.
- Ownership boundary: conversation progress summarizes and retrieves evidence; cognition judges its meaning; dialog renders the chosen response.
- Rejected complexity: keyword capture, deterministic semantic matching, domain enums, a second live memory subsystem, a generic event store, a new resolver capability, a new response-path LLM stage, a dialog suppression rule, a group-global episode model, or durable profile writes.
- Future expansion threshold: broader cross-session continuation, permanent episode archives, or a shared compaction framework requires separate evidence that the 48-hour packet/block design is insufficient and a new approved plan.

## Agent Autonomy Boundaries

- The executing agent may choose private helper names and local file layout only within the listed modules and only when the exact public/data contracts remain unchanged.
- The executing agent must not change the approved constants, lifecycle enums, collection names, call-count rules, migration strategy, or source-ref contract without updating this plan and obtaining approval.
- The executing agent must not infer semantic equivalence, completion, rejection, correction, reopening, or importance in deterministic code.
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
4. **Recorder delta and active packet:** replace the full-state recorder with the V2 delta, source allowlist, deterministic IDs/timestamps, ID-only merge, and maximum-two-attempt structural regeneration; pass invalid-output, survival, and prompt-render tests.
5. **Compacted blocks:** create structural requests; require same-call compaction output; insert immutable blocks before guarded packet writes; refresh expiry; merge the oldest four when adding to eight active refs; pass lineage, idempotency, race, expiry, and repeated-compaction tests.
6. **Cognition projection:** add the scene and decision-critical evidence projections before handle assignment; update only the scene sub-cap and fitter; prove the selected bid can cite the completed event; pass priority, cap, handle, and no-dialog-change tests.
7. **Silent-turn lifecycle:** record settled consolidatable cognition silence with `turn_outcome` and source aliases while preserving every declared exclusion; pass service ordering and call-count tests.
8. **Compacted-block retrieval:** add scoped block embedding/search to the existing conversation-evidence worker, keep active searches uncached, invalidate dependencies on writes, and pass scope, top-k, projection, lineage, cache, and no-ordinary-call tests.
9. **Documentation and deterministic regression:** update all three subsystem ICDs; run static, focused, and standard non-live gates; record payload, DB query, call-count, and latency comparisons.
10. **Serial real-LLM verification:** run and inspect the Asuna, deliberate-repeat, interleaved group/multi-fragment, and 20/50/100-turn compaction cases one at a time; inspect packets, blocks, evidence, cognition, and dialog; author and sign the quality comparison.
11. **Migration dry-run and separately authorized apply:** obtain authorization for a fresh export; run and review audit/dry-run, drift, blocked-row, and rollback evidence; apply only under a second explicit command; then run bootstrap, post-apply audit, fresh-write smoke, and rollback-readiness checks.
12. **Review and closeout:** run the fresh parent review, remediate findings, rerun affected gates, record residual risks, and request user sign-off before changing plan status.

## Execution Model

- The parent owns fixtures, module/integration tests, integration decisions, verification, evidence, review, remediation, and reporting. After the parent records each focused expected failure, a core production subagent owns `conversation_progress/**`, conversation-progress DB modules/indexes, the participant query, and migration CLI; after core tests pass, a sequential integration production subagent owns only the listed service, persona, cognition, RAG, state, and ICD files.
- Production subagents receive this approved plan, mandatory skills, exact file boundary, failing tests, and expected results; they edit production files only and report files, commands, blockers, and residual risks. Their scopes do not overlap or run concurrently.
- The user requested no subagent review, so the parent performs the fresh review gate. If native production-subagent capability is unavailable at execution time, stop and request explicit fallback-execution authorization.
- Real LLM cases run serially and are inspected before the next case starts.
- Live database apply remains a separate user-authorized operation after code verification and reviewed dry-run evidence.
- A blocked or failed stage remains unchecked and records its exact blocker; later stages do not bypass it.

## Progress Checklist

Each checkpoint covers the same-numbered `Implementation Order` item, its files in `Change Surface`, and its commands in `Verification`; append the sign-off evidence before moving to the next numbered checkpoint.

- [ ] Stage 1 — post-identity rebaseline and failing test contract; sign-off: baseline artifacts, expected failures, prompt/call counts, and latency.
- [ ] Stage 2 — logical-turn and participant history boundary; sign-off: grouping, scope, ordering, current-turn exclusion, and query-plan tests.
- [ ] Stage 3 — V2 contracts, lifecycle, indexes, and migration tooling; sign-off: exact-shape, active-read, guarded-write, dry-run purity, drift, and bootstrap tests.
- [ ] Stage 4 — recorder delta and active-packet persistence; sign-off: source-ref, delta-survival, structural regeneration, and prompt-render tests.
- [ ] Stage 5 — compacted-block persistence and hierarchical merge; sign-off: idempotency, lineage, expiry, lost-race, and repeated-compaction tests.
- [ ] Stage 6 — scene and cognition-evidence projection; sign-off: evidence-handle use, priority, and aggregate-budget tests.
- [ ] Stage 7 — eligible silent-turn recording; sign-off: visible/silent positive cases and every exclusion.
- [ ] Stage 8 — existing conversation-evidence retrieval extension; sign-off: scope, top-k, projection, cache, and ordinary-path tests.
- [ ] Stage 9 — docs, deterministic regression, and performance evidence; sign-off: every non-live command and comparison artifact.
- [ ] Stage 10 — serial real-LLM quality sign-off; sign-off: parent judgment for every required case.
- [ ] Stage 11 — migration dry-run; sign-off: export reference, dry-run review, drift counts, and rollback evidence; apply evidence is required only after separate authorization.
- [ ] Stage 12 — fresh parent review, remediation, and closeout; sign-off: findings, fixes, reruns, residual risks, and user approval.

## Verification

### Static gates

```powershell
rg -n "后颈|耳根|摸摸|按摩" src\kazusa_ai_chatbot
rg -n "conversation_mode|episode_phase|topic_momentum|user_state_updates|open_loops|interaction_obligations|resolved_threads|avoid_reopening" src\kazusa_ai_chatbot\conversation_progress src\kazusa_ai_chatbot\db\schemas.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py
rg -n "conversation_progress\.(repository|history|projection|recorder|compaction)" src\kazusa_ai_chatbot --glob "!src/kazusa_ai_chatbot/conversation_progress/**"
rg -n "conversation_progress|conversation_episode_state|conversation_episode_blocks" src\kazusa_ai_chatbot\relevance
rg -n "MAX_CONTINUATION_CHARS|MAX_PROGRESS_SCENE_CHARS|MAX_PROGRESS_EVIDENCE_CHARS|GOAL_COGNITION_PROMPT_CAP" src\kazusa_ai_chatbot tests
```

Expected, in command order: no case-specific production match (fixture matches stay outside production); no V1 production contract match; no external import of progress internals; no relevance-layer progress dependency; and one canonical definition per progress cap with the 24,000 goal cap unchanged.

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
  -> goal cognition patched boundary
  -> text-surface patched boundary
```

Required assertions:

- segmented assistant fragments count as one logical assistant turn;
- unrelated group traffic does not evict the participant tail;
- the completed prior action remains a `decision_critical` event;
- the event receives a cognition evidence handle;
- the selected bid cites that handle;
- the next semantic plan does not present the completed event as new; and
- the same architecture permits an explicit deliberate-repeat fixture.

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
- only structural recorder failure can produce attempt two.

Write the before/after workload comparison to:

`test_artifacts/reviews/conversation_progress_v2_workload_review.md`.

### One-case-at-a-time live LLM tests

Run each command separately and inspect its artifact before continuing:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_asuna_houjing_long_thread_regression -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_deliberate_reopening_remains_available -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_cross_domain_correction_and_supersession -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_interleaved_group_multifragment_continuation -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_twenty_turn_packet_continuation -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_fifty_turn_block_compaction -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_hundred_turn_hierarchical_compaction -q -s
```

Each artifact records:

- source logical turns;
- prior packet;
- recorder delta and structural attempt count;
- active packet after update;
- inserted/merged blocks;
- projection selection and evictions;
- cognition evidence handles;
- selected cognition bid;
- final dialog at labelled checkpoints;
- prompt characters and provider usage;
- agent quality judgment; and
- pass/fail reason.

The parent writes:

`test_artifacts/reviews/conversation_progress_v2_live_quality_review.md`.

Blocking live quality gates:

- exact Asuna case: completed action recalled and used; accidental reopen absent;
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
3. Trace one event from canonical rows through grouping, recorder input, delta validation, active persistence, block compaction, prompt projection, cognition evidence, selected bid, and final dialog.
4. Trace one deliberate reopening and one rejected/superseded event.
5. Inspect migration drift checks, apply boundaries, indexes, TTL behavior, idempotency, and rollback constraints.
6. Inspect prompt safety, CJK string safety, canonical JSON parsing, bounded regeneration, exception scope, and Python style.
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
- Event deltas cannot silently erase unchanged prior events.
- Every new or changed event has valid same-scope source lineage.
- Decision-critical events reach cognition as citeable `conversation_evidence`.
- The exact Asuna regression retains and uses the earlier completed event.
- Explicit deliberate reopening remains available to cognition.
- Rejected, superseded, corrected, open, in-progress, and completed states pass domain-varied cases without lexical rules.
- Relevant cognition-selected silent turns update progress; excluded ambient turns do not.
- Repeated 20/50/100-turn compaction keeps every currently decision-critical event in the active packet and represents every model-demoted archived event in a valid referenced block.
- Hierarchical blocks are immutable, idempotent, scoped, source-backed, retrievable, and short-lived.
- Ordinary same-episode continuation performs no block retrieval and adds no response-path LLM call.
- Scene, evidence, recorder, and aggregate cognition budgets pass their exact caps.
- After-change p95 progress-load latency remains within baseline plus 75 milliseconds in the same guarded environment.
- V1 rows are exported and reset only through reviewed report-driven apply.
- Focused, database, live-LLM, migration, and standard non-live gates pass.
- The parent-authored quality review and no-subagent code-review evidence are complete.
- The user approves final sign-off before status changes to `completed`.

## Risks

| Risk | Control | Verification |
|---|---|---|
| Recorder summary drift | Preserve recent complete turns, stable event IDs, source refs, immutable blocks, and delta updates | 20/50/100-turn live and deterministic replay |
| Important event silently omitted | Delta merge retains unchanged events; disposal requires explicit valid IDs and lifecycle/retention rules | Event-survival tests |
| Mechanical suppression | Events are evidence, not bans; deliberate reopening remains cognition-owned | Deliberate-repeat live case |
| Group contamination | Independent participant query plus bounded ambient tail | Interleaved group fixture and live case |
| Assistant fragments consume history | Group by trace and logical index before caps | Logical-turn tests |
| Invented source lineage | Allowlist supplied refs; reject unknown/cross-scope refs | Source-ref contract tests |
| Weak model emits invalid delta | Canonical parser, exact evaluator, one full structural regeneration, retain last valid packet | Invalid-output tests and attempt telemetry |
| Compaction loses active state | Open/in-progress/decision-critical events are ineligible for discard or archive; terminal demoted events require a valid block | Compaction contract tests |
| Orphan block after write race | Idempotent block insert first; unreferenced blocks excluded and TTL-expired | Lost-race test |
| Block expiry during continuous episode | Refresh episode block expiry after each successful active write | Time-controlled DB tests |
| Prompt bloat | Shared 4,000-character continuation cap and unchanged 24,000 aggregate cap | Prompt-budget tests and workload review |
| Latency regression | Parallel state/history reads, no new ordinary LLM call, embeddings only on compaction | Call-count and p95 comparison |
| Retrieval becomes hidden primary memory | Active packet must retain current critical state; resolver retrieval is conditional and top-three | No-retrieval ordinary-path test |
| V1 migration destroys useful state | Pre-apply export, dry-run, drift checks, short-lived scope, closed tombstones, guarded rollback | Migration tests and reports |
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
