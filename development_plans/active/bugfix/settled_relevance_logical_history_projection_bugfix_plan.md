# settled relevance logical-history projection bugfix plan

## Summary

- Goal: make settled relevance consume complete logical conversation turns so
  one fragmented character response occupies one history slot and recent human
  input remains visible.
- Plan class: `large`.
- Status: `in_progress`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, and `debug-llm`.
- Overall cutover strategy: big-bang replacement of the settled-relevance
  raw-row suffix with the existing logical-turn projection.
- Highest-risk areas: merging separate character responses, losing temporal or
  participant evidence, evicting newest history under character pressure, and
  overlapping active relevance or Conversation Progress lifecycle ownership.
- Acceptance criteria: settled relevance receives at most ten complete logical
  turns; complete assistant fragments from one response occupy one slot;
  independent responses remain separate; the history sub-budget is 6,000
  characters; deterministic fitting removes oldest whole turns first; and all
  other conversation-history consumers retain their current behavior.

## Context

`conversation_history` stores every item in `final_dialog` as a separate
assistant row. Rows from one accepted response share an `llm_trace_id` and
carry contiguous zero-based `logical_message_index` values. The latest
four-hour production export inspected during discovery contained 49 assistant
rows but only 13 assistant responses. All 13 responses had three to five
complete fragments with consistent targets. Across rolling ten-row windows,
assistant fragments occupied a median of six rows and as many as eight.

Conversation Progress already solves this for its ambient and participant
lanes:

```text
bounded canonical rows
  -> assemble complete logical turns
  -> select newest complete turns
  -> project one history row per turn
```

Settled relevance remains the exposed live consumer. The service retrieves ten
raw rows, excludes active-turn rows, and passes the last ten remaining rows as
`fresh_history`. The relevance projector then applies a ten-item limit and a
4,000-character history sub-budget. Fragmented character speech therefore
occupies multiple evidence handles and displaces human turns before relevance
decides whether the character has a grounded reason to speak.

Two active plans have completed their code and verification but still await
explicit lifecycle closure:

- `relevance_evidence_grounded_admission_over_sensitivity_bugfix_plan.md`;
- `conversation_progress_v2_final_signoff_plan.md`.

Execution begins only after both plans are completed and archived, or after
their execution records explicitly release `service.py`,
`relevance/persona_relevance_agent.py`, and
`conversation_progress/__init__.py` to this plan. The implementation baseline
must contain relevance merge commit `305c84b1` or a descendant with the same
contracts.

## Mandatory Skills

- `development-plan`: load before approval, execution, review, lifecycle
  changes, sign-off, or archive.
- `local-llm-architecture`: load before changing response-path context
  projection, character budgets, or settled relevance behavior.
- `py-style`: load before editing or reviewing Python production code.
- `cjk-safety`: load before editing
  `relevance/persona_relevance_agent.py`, which contains CJK prompt literals.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before the one-at-a-time real relevance regression and its
  human-readable inspection record.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for every Python and pytest command.
- Use `apply_patch` for manual source, test, documentation, and plan edits.
- Check `git status --short` before every implementation stage and final
  sign-off; preserve unrelated user work.
- Keep `.env` contents outside agent inspection. Runtime and test entrypoints
  may load project configuration through their established path.
- Treat relevance as the sole semantic admission owner. Deterministic code
  owns row scanning, active-row exclusion, logical-turn grouping, limits,
  chronological ordering, and character fitting.
- Reuse the existing `conversation_progress.history` grouping rules and expose
  the required functions through the package facade. Maintain one canonical
  grouping implementation.
- Group only contiguous assistant rows sharing one non-empty `llm_trace_id`
  whose `logical_message_index` values are complete, unique, and contiguous
  from zero.
- Keep separate response traces separate even when their addressees are
  identical.
- Preserve the existing malformed-group behavior: an oldest cut-off suffix is
  omitted; an internally malformed group remains represented by individual
  rows and contributes the existing diagnostic behavior.
- Preserve user rows as one logical turn per canonical row.
- Preserve active-turn exclusion, typed speaker/target/reply fields,
  attachment descriptions, chronological order, and
  `before_active_turn|during_active_turn|after_active_turn|unknown`.
- Keep the settled history logical-turn limit at ten and each projected
  `body_text` cap at 500 characters.
- Set the settled history compact-JSON sub-budget to exactly 6,000 characters.
- When the sub-budget is exceeded, remove oldest whole projected turns until
  the list fits, renumber participant handles, and preserve chronological
  order. Apply the same role-neutral rule to user and assistant turns.
- Keep the settled rendered-input hard cap at 16,000 characters, completion
  cap at 512 tokens, thinking disabled, and the existing outer fitting order.
- Keep the frontline relevance path, prompts, public relevance decisions,
  model route, call count, retry behavior, evidence validation, coordinator,
  cognition claim, persistence, and delivery behavior unchanged.
- Keep all RAG, reflection, conversation search, conversation progress runtime,
  consolidation, audit/export, and background history consumers unchanged.
- After automatic context compaction, reread this entire plan before
  implementation, verification, handoff, lifecycle changes, or reporting.
- After signing off each major checklist stage, reread this entire plan before
  starting the next stage.
- Before completion, run the `Independent Code Review` gate and record its
  result in `Execution Evidence`.
- Execute through the parent-led native subagent model in `Execution Model`.

## Must Do

- Expose `assemble_logical_turns(...)` and
  `select_recent_logical_turns(...)` through
  `kazusa_ai_chatbot.conversation_progress`.
- Replace only the settled-relevance generic history read with the existing
  embedding-excluding ambient-history reader and a relevance-specific fixed
  scan of 48 rows.
- Exclude active-turn rows before logical-turn assembly using the existing row
  and platform-message identities.
- Assemble external rows through the canonical Conversation Progress function,
  select the newest ten complete logical turns, and project them through
  `logical_turns_as_history_rows(...)`.
- Assign temporal relations to the projected logical turns from their
  canonical timestamps relative to the active-turn timestamp interval.
- Raise `_HISTORY_TOTAL_CHARS` from 4,000 to 6,000 and make
  `_project_history(...)` preserve the newest whole turns under that budget.
- Rebuild sequential participant handles and final interaction evidence from
  the exact retained history.
- Add focused deterministic coverage, run the selected live relevance
  regression alone, and update the relevance and brain-service contracts.
- Record every verification and review result before lifecycle completion.

## Deferred

- RAG conversation evidence, memory retrieval, reflection, consolidation,
  exports, audit, background jobs, and Conversation Progress runtime behavior
  remain unchanged.
- Conversation-history storage rows, indexes, embeddings, schemas, migrations,
  delivery receipts, and adapter behavior remain unchanged.
- Frontline history, cognition history, dialog history, prompt wording, public
  decision schemas, and participant vocabulary remain unchanged.
- Pagination, adaptive database scans, database-side grouping, role-weighted
  eviction, assistant-specific suppression, summarization, and additional LLM
  calls remain outside this plan.
- General extraction of conversation-history projection into a new shared
  package remains outside this focused reuse.

## Cutover Policy

Overall strategy: `bigbang`.

| Area | Policy | Instruction |
|---|---|---|
| Settled history intake | bigbang | Replace the ten-raw-row suffix with one 48-row scan followed by canonical logical-turn assembly and a ten-turn limit. |
| History fitting | bigbang | Replace prefix-favoring row fitting with newest-whole-turn retention under 6,000 characters. |
| Other consumers | retained | Preserve every conversation-history consumer outside settled relevance. |
| Public relevance contract | retained | Preserve actions, fields, refs, model route, call count, and 16,000-character outer cap. |
| Persistence and data | retained | Preserve stored rows and require no migration or backfill. |

The execution agent applies each area policy directly. Any policy change
requires user approval before implementation.

## Target State

The live settled path becomes:

```text
one channel-scoped database read, newest 48 raw rows
  -> remove active-turn rows
  -> canonical complete logical-turn assembly
  -> newest 10 logical turns
  -> one prompt-facing row per logical turn
  -> temporal and participant projection
  -> 6,000-character newest-first whole-turn fitting
  -> existing 16,000-character settled relevance fitting
  -> existing ignore | proceed | wait decision
```

One three-fragment character response becomes one history row whose
`body_text` joins the fragments in logical-message order. Three independent
responses to the same user remain three history rows. A user row remains one
history row. Final model-facing rows stay chronological.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Group identity | Existing contiguous `llm_trace_id` plus complete logical indexes | This reverses the dialog persistence operation without merging independent responses. |
| Scan size | Fixed 48 raw rows for settled relevance only | It matches the proven ambient scan scale while isolating other consumers. |
| Logical limit | Ten newest complete turns | It preserves the settled relevance cardinality while recovering human slots. |
| Character budget | 6,000 compact-JSON characters | The user approved the increase from 4,000 while retaining the 16,000 outer cap. |
| Eviction | Remove oldest complete turns, role-neutral | It is deterministic and preserves the freshest scene evidence. |
| Oversized text | Preserve the existing 500-character per-turn body cap | It keeps a bounded individual turn without adding summarization. |
| Temporal relation | Use canonical logical-turn timestamp against the active interval | Grouped turns no longer have one raw-row index and already expose a canonical occurrence time. |
| Reuse boundary | Export existing assembly and selection through the Conversation Progress facade | Service callers use a public package boundary and share one implementation. |

## Contracts And Data Shapes

The settled `fresh_history` state key remains `list[dict]`. Each retained item
continues to expose the fields consumed by relevance:

```python
{
    "role": "user | assistant",
    "timestamp": str,
    "display_name": str,
    "body_text": str,
    "platform_user_id": str,
    "global_user_id": str,
    "addressed_to_global_user_ids": list[str],
    "broadcast": bool,
    "reply_context": dict,
    "llm_trace_id": str,
    "turn_temporal_relation": (
        "before_active_turn | during_active_turn | "
        "after_active_turn | unknown"
    ),
}
```

Assistant `body_text` is `"\n".join(turn["fragments"])`. No raw row id, trace
id, or platform identity is added to the model-facing payload; the existing
relevance projector continues converting identities to stable relations.

History-budget fitting measures:

```python
len(json.dumps(projected_history, ensure_ascii=False, separators=(",", ":")))
```

The final list must be at most 6,000 characters. Fitting repeatedly removes
index `0`, renumbers `participant_n` handles, and reserializes until the
contract holds.

## LLM Call And Context Budget

| Dimension | Before | After |
|---|---|---|
| Settled LLM calls | At most one per assessment plus the existing bounded authoritative repair | Unchanged |
| Database calls | One generic channel history read | One ambient channel history read with active-row exclusions |
| Raw database row cap | 10 shared-config rows | 48 relevance-specific rows |
| Model-facing history count | At most 10 raw rows | At most 10 logical turns |
| Per-turn body cap | 500 characters | 500 characters |
| History sub-budget | 4,000 characters | 6,000 characters |
| Settled rendered-input hard cap | 16,000 characters | 16,000 characters |
| Completion and thinking | 512 tokens; disabled | Unchanged |

The default architecture context cap is 50,000 tokens. The existing
16,000-character settled hard cap remains far below that ceiling. The change
adds at most 38 rows to one bounded database result and adds no model,
retrieval, embedding, cache, repair, or retry call. Prompt fitting continues
to rebuild all citable evidence after outer-cap reductions.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/conversation_progress/__init__.py`
  - Export the existing assembly and newest-turn selection functions.
- `src/kazusa_ai_chatbot/service.py`
  - Add the relevance-specific 48-row scan constant, call the existing
    embedding-excluding ambient-history reader, use the public logical history
    functions in `_process_settlement_lease(...)` and
    `_settled_state_from_lease(...)`, and preserve temporal projection.
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`
  - Set the 6,000-character history budget and retain newest whole rows under
    the exact compact-JSON measurement.
- `src/kazusa_ai_chatbot/relevance/README.md`
  - Document logical-turn intake, cardinality, fitting, and unchanged outer
    contract.
- `src/kazusa_ai_chatbot/brain_service/README.md`
  - Document the service-owned scan, active-row exclusion, assembly, and
    temporal projection.
- `docs/HOWTO.md`
  - Document the settled history sub-budget and bounded scan behavior beside
    the existing relevance route limits.
- `tests/test_conversation_progress_logical_turns.py`
  - Prove the required assembly and selection functions are available through
    the public facade.
- `tests/test_service_input_queue.py`
  - Prove query size, active exclusion, fragment collapse, same-addressee
    separation, chronology, and temporal relations.
- `tests/test_persona_relevance_agent.py`
  - Prove the 6,000-character constant, newest-whole-turn fitting, participant
    renumbering, evidence rebuilding, and unchanged outer cap.
- This plan and `development_plans/README.md`
  - Record lifecycle, execution evidence, review, and archive transition.

### Create

- No production module, schema, fixture, migration, or runtime flag.

### Delete

- No file or persisted data.

### Keep

- Every source and test path outside the exact modify list remains unchanged,
  except an in-scope review correction explicitly authorized below.

## Overdesign Guardrail

- Actual problem: fragmented character response rows consume the fixed settled
  relevance history slots and displace recent human input.
- Minimal change: reuse existing logical-turn assembly in the one exposed raw
  list consumer, increase its local sub-budget, and apply deterministic oldest
  whole-turn eviction.
- Ownership boundaries: database storage retains raw rows; service owns
  bounded retrieval and active-row exclusion; shared deterministic code owns
  logical grouping; relevance owns prompt-safe identity projection and
  semantic admission.
- Rejected complexity: adaptive scans, aggregation pipelines, new storage
  fields, alternate grouping keys, summarizers, role quotas, compatibility
  paths, new prompts, model calls, retries, caches, and broad shared-history
  refactors.
- Evidence threshold: a captured case where the 48-row scan cannot recover
  ten useful logical turns, or where a complete response exceeds established
  grouping metadata, is required before expanding scan or grouping policy.

## Agent Autonomy Boundaries

- Local implementation mechanics may vary only while preserving every exact
  contract in this plan.
- Production changes stay within the listed modify paths and named symbols.
- The existing Conversation Progress implementation remains the single
  grouping owner; equivalent grouping code is reused through its facade.
- Review fixes may update listed tests or documentation when they preserve the
  approved runtime contract.
- Architecture, grouping identity, scan size, limits, drop order, prompt
  fields, and outer relevance fitting remain fixed decisions.
- Unrelated cleanup, formatting churn, dependency changes, and prompt rewrites
  remain outside execution.
- A source/plan discrepancy, active lifecycle ownership collision, required
  change outside the allowlist, or failing contract requiring new semantics
  stops execution and returns to plan reconciliation.

## Implementation Order

1. Parent rechecks lifecycle ownership, clean baseline, ICDs, source, and tests;
   records the exact baseline commit and current target hashes.
2. Parent changes imports in
   `tests/test_conversation_progress_logical_turns.py` to the package facade,
   adds the focused service and relevance tests named in `Change Surface`, and
   records the expected pre-implementation failures.
3. Parent starts exactly one production-code subagent with this plan, mandatory
   skills, test contract, and production-only change surface.
4. Production-code subagent exports the existing functions, changes the
   service scan/assembly projection, changes deterministic relevance fitting,
   and updates the three approved contract documents.
5. Parent reruns focused deterministic tests, applies only listed integration
   test corrections, and records exact results.
6. Parent runs the one selected real-LLM regression alone, inspects its durable
   trace, and records a human-readable judgment.
7. Parent runs adjacent and full non-live regressions plus static diff gates.
8. Parent starts exactly one independent code-review subagent, remediates
   in-scope findings, reruns affected gates, and records approval.
9. Parent requests user sign-off, marks the plan completed only after approval,
   archives it, and updates the registry.

## Execution Model

- Parent agent owns orchestration, test code, baseline failures, verification,
  evidence, review remediation, lifecycle updates, and final sign-off.
- Parent establishes the focused deterministic test contract before production
  implementation begins.
- Production-code subagent: exactly one native subagent; starts after the
  focused test contract; edits production code and approved contract docs only;
  reports files, commands, blockers, and residual risks; then closes.
- Parent may continue integration-test and verification preparation while the
  production-code subagent works.
- Independent code-review subagent: exactly one native subagent after planned
  verification; reviews the plan, diff, and evidence; reports findings without
  implementing fixes.
- Native subagent unavailability stops execution until the user explicitly
  authorizes fallback execution.

## Progress Checklist

- [x] Stage 1 - focused contract established.
  - Covers: implementation steps 1-2.
  - Verify: focused selectors fail only for the planned missing facade export,
    raw-row behavior, and 4,000-character budget.
  - Evidence: baseline commit, target hashes, test names, commands, and
    expected failures recorded in `Execution Evidence`.
  - Handoff: production-code subagent starts Stage 2.
  - Sign-off: parent agent / 2026-07-30.
- [x] Stage 2 - production projection and integration complete.
  - Covers: implementation steps 3-5.
  - Verify: facade, service, and relevance focused suites pass; documentation
    matches runtime contracts.
  - Evidence: changed paths, focused results, and diff-scope audit recorded.
  - Handoff: parent starts Stage 3 after rereading this plan.
  - Sign-off: parent agent / 2026-07-30.
- [x] Stage 3 - live and regression verification complete.
  - Covers: implementation steps 6-7.
  - Verify: selected real-LLM case passes after individual inspection; adjacent
    and full non-live commands pass; static gates match expectations.
  - Evidence: trace/review path, human judgment, counts, and static output
    recorded.
  - Handoff: independent reviewer starts Stage 4 after plan reread.
  - Sign-off: parent agent / 2026-07-30, with user-approved unrelated
    exceptions recorded in `Execution Evidence`.
- [ ] Stage 4 - independent code review and lifecycle sign-off complete.
  - Covers: implementation steps 8-9 and `Independent Code Review`.
  - Verify: no unresolved review finding; affected gates rerun after fixes;
    user approval recorded.
  - Evidence: reviewer identity, findings, fixes, reruns, residual risks, user
    decision, archive path, and registry update recorded.
  - Handoff: complete.
  - Sign-off: `<agent/date>` after verification and evidence.

## Verification

### Focused deterministic tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_logical_turns.py tests\test_persona_relevance_agent.py tests\test_service_input_queue.py -q
```

Expected: exit code 0. New tests prove one slot per complete assistant response,
separation across response traces, active-row exclusion, ten-turn selection,
6,000-character newest-turn retention, handle renumbering, and unchanged
16,000-character outer fitting.

### Selected real-LLM regression

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_relevance_turn_settlement_live_llm.py::test_live_other_user_answer_makes_pending_reply_redundant -q -s
```

Expected: run alone; inspect the durable trace and confirm external history
still lets settled relevance identify that another participant resolved the
pending request. Record behavioral judgment in addition to pytest status. A
pre-existing model-quality failure may proceed only when the user explicitly
accepts it as unrelated to this deterministic projection change and the
failure is preserved in a human-readable review artifact.

### Adjacent deterministic regression

```powershell
venv\Scripts\python.exe -m pytest tests\test_relevance_participation_evidence.py tests\test_frontline_relevance_agent.py tests\test_relevance_turn_settlement.py tests\test_relevance_turn_settlement_graph.py -q
```

Expected: exit code 0; public evidence, frontline, coordinator, and graph
contracts remain unchanged.

### Full non-live regression

```powershell
venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q
```

Expected: exit code 0; record exact passed, skipped, and deselected counts.
Pre-existing failures outside the changed paths may proceed only with explicit
user approval and exact failure evidence.

### Static and diff gates

```powershell
rg -n "assemble_logical_turns|select_recent_logical_turns|logical_turns_as_history_rows" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\conversation_progress\__init__.py
rg -n "_HISTORY_TOTAL_CHARS = 6000" src\kazusa_ai_chatbot\relevance\persona_relevance_agent.py
git diff --check
git status --short
```

Expected: the first search shows facade exports and service use; the second
shows exactly one relevance-local assignment; `git diff --check` exits zero;
status contains only the approved change surface. Any extra path blocks
sign-off.

## Independent Code Review

After all verification gates pass, the parent starts one independent
code-review subagent. The reviewer receives this plan, baseline, complete diff,
test results, live trace judgment, and execution evidence.

Review scope:

- project and mandatory-skill compliance across every changed file;
- exact grouping reuse and absence of same-addressee over-merging;
- active-row exclusion, malformed-boundary behavior, temporal relations,
  identity projection, and chronology;
- exact ten-turn, 500-character, 6,000-character, 16,000-character, and
  48-row contracts;
- deterministic oldest-whole-turn eviction and handle/evidence rebuilding;
- unchanged consumers, prompts, LLM calls, decisions, persistence, RAG,
  reflection, and Conversation Progress runtime;
- test realism, live evidence quality, documentation accuracy, diff scope, and
  lifecycle readiness.

The parent remediates findings only inside the approved change surface. A
finding requiring a new contract or outside path returns the plan to user
approval. Record findings, fixes, reruns, residual risks, and reviewer approval
in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- settled relevance scans at most 48 raw channel rows in one existing database
  call and applies the newest-ten limit after canonical logical-turn assembly;
- one complete fragmented assistant response occupies one `fresh_history`
  slot and preserves fragment order, targets, reply context, and attachments;
- separate assistant responses to the same addressee remain separate turns;
- user rows remain individual turns and recent user turns gain the recovered
  slots;
- active rows stay excluded and temporal relations remain correct with or
  without active rows inside the scanned window;
- model-facing history stays chronological, each body stays within 500
  characters, and compact history stays within 6,000 characters;
- sub-budget and outer-cap fitting remove oldest whole history turns,
  renumber handles, and rebuild citable evidence deterministically;
- the settled outer cap remains 16,000 characters with unchanged model calls,
  completion budget, decisions, prompts, and coordinator behavior;
- every other conversation-history consumer and all persisted data remain
  unchanged;
- focused, adjacent, static, and diff gates pass; live and full non-live gates
  either pass or retain exact user-approved unrelated exceptions;
- independent review has no unresolved finding; and
- user approval and lifecycle archive are recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Separate responses merge because the same user is addressed | Trace-and-index grouping remains canonical | Same-addressee/different-trace test |
| A scan begins inside a fragmented response | Existing cut-off suffix rule omits that incomplete oldest group | Facade and service boundary tests |
| Grouping loses target, reply, image, or speaker evidence | Reuse canonical logical turn and history-row projections | Structured service assertions |
| Newest turns disappear under the history sub-budget | Compact-list sizing and repeated index-zero eviction | 6,000-character fitting test |
| Participant refs become stale after eviction | Renumber handles and rebuild evidence from final payload | Cap-dropped-ref regression |
| More history displaces required outer payload fields | Existing 16,000-character fitter and exact evidence rebuild remain authoritative | Worst-case projection test |
| Larger database result adds response latency | One bounded read grows from 10 to 48 rows through the existing embedding-excluding projection | Query-argument test and live timing observation |
| Parallel active plans still own target files | Lifecycle release gate precedes Stage 1 | Baseline evidence and registry audit |

## Execution Evidence

Planning evidence recorded on 2026-07-30:

- current branch `cognition_core_v2` is clean at merge commit `305c84b1`;
- the latest four-hour QQ conversation export contained 93 rows, including 49
  assistant rows representing 13 complete assistant response traces;
- rolling ten-row measurements confirmed median assistant occupancy of six and
  maximum occupancy of eight;
- current Conversation Progress assembly, facade, policy, runtime, tests, and
  ICD were inspected;
- current service settled-history retrieval, active exclusion, temporal
  projection, and tests were inspected;
- current relevance history projection, 4,000/16,000-character fitting,
  evidence rebuilding, tests, and ICD were inspected;
- the active relevance and Conversation Progress plans have completed code and
  review evidence and retain only user lifecycle sign-off;
- user decisions fixed the target consumer, existing grouping rule, 6,000
  history budget, and deterministic oldest-whole-turn policy; and
- plan self-review found complete coverage, one canonical grouping owner,
  explicit non-overlap, exact limits, named tests, and no unresolved design
  choice.

Stage 1 execution evidence recorded on 2026-07-30:

- the user explicitly released the overlapping active-plan ownership gate and
  directed execution of this plan;
- execution baseline: branch `cognition_core_v2`, commit `305c84b1`;
- pre-implementation target hashes:
  `conversation_progress/__init__.py`
  `1bb937eed2845d4e293b6ce48827dee22db33f81`,
  `service.py` `7e03c046a5598fb719e3cb2ea833378aeb757c2e`,
  and `persona_relevance_agent.py`
  `4d31a107d84e0951a47d3e7cda5c33a7332b99b1`;
- public-facade test failed with missing `assemble_logical_turns`;
- service query test failed because settled relevance called the generic
  history reader instead of the 48-row ambient reader;
- service projection test failed because one complete two-fragment assistant
  response remained two `fresh_history` rows; and
- relevance fitting test failed because `_HISTORY_TOTAL_CHARS` remained 4,000
  instead of 6,000.

Stage 2 execution evidence recorded on 2026-07-30:

- production-code subagent `/root/production_implementation` changed exactly
  the six approved production and contract-document paths;
- the package facade exports the three logical-history helpers, settled
  relevance alone uses one 48-row embedding-excluded ambient query, active
  platform and row identities are excluded, and the service projects the
  newest ten canonical logical turns with timestamp relations;
- relevance now measures compact JSON against exactly 6,000 characters,
  removes oldest whole rows, and recomputes opaque participant handles before
  returning the retained chronological list;
- parent-owned fixture corrections added required canonical row ids, patched
  the newly selected ambient reader, and retained the existing deterministic
  group-attention policy for the expanded user-heavy fixture;
- focused deterministic result: `88 passed in 3.10s`;
- `py_compile` passed for all three changed production files and all three
  focused test files;
- facade/service/budget static searches matched the expected symbols and exact
  constant, `git diff --check` passed, and `git status --short` contains only
  the approved change surface; and
- the three approved ICD/HOWTO documents state the 48-row scan, logical
  assembly, newest-ten limit, exact 6,000-character sub-budget, and unchanged
  16,000-character outer cap.

Stage 3 verification evidence recorded on 2026-07-30:

- the selected live-LLM pytest case passed twice, but both raw outputs ignored
  `history_1`, justified `ignore` from generic non-address, and failed the
  recipient-evidence contract; the bounded evaluator therefore returned
  fail-closed `ignore`;
- one additional production-shaped diagnostic supplied typed participant
  identity, current-author targeting, and `after_active_turn`; the model again
  ignored history and failed the same recipient-evidence contract;
- human-readable quality review:
  `test_artifacts/reviews/settled_relevance_logical_history_live_review.md`;
- live traces:
  `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__L10_answered_by_other_user.json`,
  `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__L10_answered_by_other_user__20260730T070104647761Z.json`,
  and
  `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__L10_logical_history_production_shape.json`;
- adjacent deterministic result: `56 passed in 1.73s`;
- full non-live result:
  `3846 passed, 3 skipped, 859 deselected, 2 failed, 1 warning in 289.49s`;
- the two full-suite failures are outside this plan's changed paths:
  `test_post_fix_corpus_binds_current_checkout_revision` requires the missing
  external checkout `C:\workspace\kazusa_ai_chatbot_v2_prefix`, and
  `test_selection_producer_retry_reuses_goal_route_and_trace` fails an
  existing Cognition Core V2 prompt-identity assertion; and
- Stage 3 initially remained unsigned because the required live behavioral
  judgment and full non-live exit-zero gates did not pass. Independent code
  review had not started before the user's exception decision.

Stage 3 exception approval recorded on 2026-07-30:

- the user approved proceeding with exceptions for unrelated changes;
- the live-model history-use quality failure remains documented and causes no
  prompt, relevance-decision, retry, or model-route change in this plan;
- the two unrelated full-suite failures remain unchanged and outside this
  plan's source and test diff; and
- Stage 3 is accepted with those exact exceptions so the mandatory independent
  code review may proceed.

Stage 4 independent review evidence recorded on 2026-07-30:

- independent reviewer `/root/independent_code_review` inspected the complete
  plan, diff, tests, documents, live review, traces, exception evidence, and
  lifecycle state;
- initial review found two low-severity in-scope issues and no runtime,
  medium, or high finding:
  - `_project_history(...)` required a complete Args/Returns docstring under
    Python style rule P-003;
  - two ambient-reader monkeypatches used `raising=False`, weakening the
    missing-symbol contract;
- the parent expanded the docstring and restored normal raising monkeypatch
  behavior without changing runtime semantics;
- post-fix `py_compile` and `git diff --check` passed, and the focused suite
  passed `88 passed in 1.93s`;
- final independent review outcome: `APPROVED`, with no remaining finding;
- residual risks are limited to the user-approved live-model history-use
  quality exception and the two unchanged unrelated full-suite failures; and
- lifecycle completion, archive move, and registry transition await explicit
  final user sign-off.
