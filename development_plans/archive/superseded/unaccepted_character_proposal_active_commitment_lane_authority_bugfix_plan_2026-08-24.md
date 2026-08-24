# Unaccepted Character Proposal And Active-Commitment Lane Authority Bugfix Plan

- Status: superseded
- Superseded on: 2026-08-24
- Superseded by:
  `asuna_semantic_authority_and_memory_feedback_consolidated_bugfix_plan_2026-08-24.md`
- Lifecycle note: the false-commitment RCA, lane contract, and real-LLM gates
  are carried forward by the consolidated plan. This record grants no current
  production or database authority.
- Date: 2026-08-24
- Type: active bugfix plan
- RCA evidence:
  `test_artifacts/reviews/asuna_exchange_fixation_l2_replay_system_rca_20260824.md`
- RCA evidence SHA-256:
  `5E390687FB2D0BBA74D781EBC35B1A54A5F5F85DD609BC803B19DF6B3E469A8F`
- Production implementation authority: requires a separate explicit user
  command after review of this plan.
- Database mutation authority: none. Any cleanup of the reproduced debug-user
  row requires a separate exact apply decision.
- Architecture and acceptance owner: parent agent.
- Fixed implementation and test executor after approval: the existing fresh
  GPT-5.6 Luna subagent at maximum reasoning and standard normal speed.

## Summary

A real memory-enabled service replay produced a role-inverted durable memory:

```text
用户同意在获得一之濑明日奈的决策协助后，提供一份具体的奖励作为交换。
```

The user never accepted the reward proposal. The background lane router
correctly selected `user_memory_units`, not `active_commitment`. The shared
memory-unit extractor nevertheless emitted an `active_commitment`, and the
candidate validator accepted it because it validates against the global unit
type enum rather than the router-approved lane set.

The fix preserves the hybrid agentic flow and the current model-call budget:

1. The existing lane-router LLM remains the semantic owner of coarse durable
   lane selection.
2. Deterministic code converts the accepted lane set into a small canonical
   memory-unit write contract containing the exact allowed candidate types.
3. The existing one-call extractor receives that contract and remains the
   semantic owner of candidate meaning and wording.
4. The deterministic candidate validator rejects output types outside the
   accepted lane contract.
5. The extractor prompt explicitly distinguishes character-owned accepted
   commitments from unanswered character proposals and user-owned promises.

No new LLM stage, retry, reviewer call, foreground database read, persistent
collection, or output schema is introduced.

## Failure Evidence

### Service replay

- Channel: `semantic-progression-l2-replay-20260824_121234`
- Global user: `4e468b8d-2fca-44df-9a8f-31c3ca87cb5c`
- Turn 2 trace: `llmtrace_243c44872a804fd894fa5497d3cb650e`
- Pre-turn user-memory rows: `0`
- Post-turn user-memory rows: `1`
- Persisted unit ID: `e42c715b95e44fb58be86d91df3845cf`
- Persisted type: `active_commitment`

The exact user input was:

```text
我已经把任务都列出来了：今天要交的报告、下周会议的材料，还有十几封待回邮件。你会先挑哪一个，为什么？
```

The user answered Asuna's request for task details. They did not accept,
reference, or respond to Asuna's reward proposal.

### Router and writer evidence

The background log reported the accepted durable lanes as:

```text
user_memory_units
character_identity_growth
```

The router did not select `active_commitment`. Its `user_memory_units` reason
was the user's disclosed near-term workload and deadlines. That is a valid
coarse user-memory decision.

`persistence.db_writer(...)` enables the same generic memory-unit pipeline when
either `user_memory_units` or `active_commitment` is present. The extractor
prompt and output schema always offer all five unit types. `_valid_candidates`
accepts any member of `VALID_EXTRACTED_USER_MEMORY_UNIT_TYPES`; it has no input
representing which lane the router approved.

The pre-extraction source policy therefore cannot protect the candidate:

- `active_commitment` lane policy correctly requires user plus assistant
  acceptance sources;
- only `user_memory_units` was selected, so that stronger policy never ran;
- the later generic extractor created an active commitment under the accepted
  user-memory write path;
- the candidate validator saw a globally valid enum and persisted it.

This is a contract-continuity failure between an agentic semantic decision and
deterministic persistence admission.

## Historical Design Review

### Decisions to carry forward

| Source | Required design carried into this plan |
|---|---|
| `consolidator_lane_router_memory_pollution_bigbang_plan.md` | Router chooses coarse lanes; lane-local semantics must remain separated; user memory excludes `active_commitment`; active commitment requires current user plus assistant acceptance evidence; deterministic source validation must not interpret prose. |
| `user_memory_units_lane_data_integrity_plan.md` | User-memory units require durable meaning, source refs, bounded candidates, and one-at-a-time live LLM review for ambiguous commitment semantics. |
| `docs/architecture/character_memory_architecture.md` | Commitments require a request, character acceptance, explicit owner, intended outcome, and lifecycle. Consolidation cannot invent unsupported meaning. |
| `no-prepost-user-input` policy | LLM stages retain semantic ownership. Deterministic code may enforce the selected enum/lane contract and must not classify or rewrite user text with keywords, regexes, or prose rules. |
| `local-llm-architecture` policy | Give the local model one compact, explicit contract; preserve the existing bounded call; avoid another semantic stage or retry. |

The completed big-bang plan intended a `user_memory_units` candidate shape that
excluded `active_commitment` and a distinct active-commitment candidate shape.
The current implementation converged both onto one extractor without carrying
the router's lane constraint into that extractor or its validator. This plan
repairs that missing boundary without restoring the old monolithic harvester
or adding another call.

### Mechanisms to leave retired

- Deterministic keyword or phrase checks over user input, dialog, facts, or
  commitment prose.
- A post-generation semantic classifier or output rewrite.
- A second extractor or reviewer call on the live consolidation path.
- Compatibility aliases, dual output shapes, fallback lane names, or a
  parallel memory collection.
- Treating any continuation of the same task as implicit acceptance.
- Treating a character-authored demand as a character commitment to the user.

## Root-Cause Decisions

### 1. `active_commitment` is character-owned

For the current canonical user-memory unit schema, `active_commitment` means a
future behavior, promise, reminder, address rule, or ongoing interaction rule
that the active character accepted for the current user.

It requires:

- a user or scene request;
- evidence that the character accepted or assumed the behavior;
- the character as obligation owner;
- a recognizable outcome;
- a future lifecycle when applicable.

A demand, offer, bargain, or condition authored by the character does not meet
that contract merely because the user continues the underlying topic. A user
promise to the character is also not this character-owned unit type. If the
user explicitly makes a durable promise, another user-memory meaning may be
appropriate; this plan does not create a new user-obligation lane.

### 2. Silence and task continuation are not acceptance

The extractor must return no active commitment when the user:

- does not respond to the character's proposal;
- supplies information the character requested for the underlying task;
- asks the next connected practical question;
- changes topic without adopting the proposal.

Explicit acceptance, rejection, clarification, or negotiation remains semantic
evidence and is interpreted by the owning LLM stage. This plan adds no local
phrase matcher.

### 3. Router output constrains candidate admission

The accepted lane set is trusted structured control state. Deterministic code
derives exactly these allowed unit types:

| Accepted lane | Allowed candidate types |
|---|---|
| `user_memory_units` | `stable_pattern`, `recent_shift`, `objective_fact`, `milestone` |
| `active_commitment` | `active_commitment` |
| both | union of the two rows |
| neither | empty; extraction/persistence fails closed |

This mapping validates the output contract selected by the router. It does not
interpret user text or override a semantic result into another lane.

### 4. One canonical extractor call remains

The existing extractor receives a compact code-owned
`memory_unit_write_contract` in its human payload:

```json
{
  "enabled_lanes": ["user_memory_units"],
  "allowed_unit_types": [
    "stable_pattern",
    "recent_shift",
    "objective_fact",
    "milestone"
  ]
}
```

The order is canonical and bounded. The extractor must emit only allowed
types, or an empty `memory_units` list. The output schema itself remains the
single current schema.

The deterministic validator receives the same allowed set and drops any
off-lane candidate with a typed diagnostic. It preserves valid candidates and
does not rewrite an invalid type.

### 5. Source authority remains lane-owned

`validate_lane_source_policy(...)` remains the source-class owner. An
`active_commitment` candidate is possible only when the router selected that
lane after its user-plus-assistant source requirement passed. The new
candidate-type validation closes the path that previously used the weaker
`user_memory_units` source policy.

## Scope

### In scope

- Canonical allowed-type projection from accepted consolidation lanes.
- Extractor prompt input and semantic guidance for lane/type authority,
  commitment ownership, explicit acceptance, and inverse negative cases.
- Deterministic rejection of candidate types outside the accepted lane set.
- Exact diagnostics for rejected off-lane candidates.
- Documentation of the router-to-extractor-to-validator boundary.
- Focused deterministic tests, one-at-a-time real LLM tests, a memory-enabled
  service gate, protected evidence inspection, and test-impact governance.

### Explicitly out of scope

- Conversation Progress scene or event relevance. That remains in the
  semantic-progression plan's separately approved amendment.
- Reflection promotion, shared-memory authority, cross-user applicability, or
  the global exchange-condition row. Those remain in the separate cross-user
  character-memory plan.
- Database cleanup, reset, migration, or mutation.
- New memory unit types, a user-obligation lane, new collection, or persistent
  schema change.
- Router prompt redesign, router call count, source-policy meaning, merge/
  rewrite semantics, scheduling, recall, cognition, L3, dialog, adapters, or
  multi-emotion state.
- A deterministic semantic filter over input or generated prose.

## Change Surface

### Production files to modify after approval

- `src/kazusa_ai_chatbot/consolidation/memory_units.py`
  - define the exact lane-to-unit-type mapping;
  - build the canonical bounded write contract from
    `enabled_consolidation_write_lanes`;
  - include it in `_json_payload(...)`;
  - clarify `_EXTRACTOR_PROMPT` and its input contract;
  - pass allowed types through `_validated_candidates(...)` and
    `_valid_candidates(...)`;
  - reject off-lane candidates without rewriting them.
- `src/kazusa_ai_chatbot/consolidation/README.md`
  - document lane authority, commitment owner, and fail-closed candidate
    validation.

No other production file is planned. Discovery of a required production caller
or contract owner pauses execution for parent analysis and user approval.

### Tests and governance to modify or add

- `tests/test_user_memory_units_rag_flow.py`
- `tests/test_user_memory_units_live_llm.py`
- `tests/test_consolidation_source_policy.py` only if an existing assertion
  needs a direct integration link; source-policy behavior itself remains
  unchanged.
- `tests/ownership/source_test_impact_manifest.json`
- `test_artifacts/live_llm/unaccepted_proposal_commitment_authority_20260824/`
- this plan and `development_plans/README.md`

## Deterministic Test Contract

Write the exact tests before production changes and record genuine RED, then
GREEN:

1. `user_memory_units` only projects four non-commitment types in canonical
   order.
2. `active_commitment` only projects `active_commitment`.
3. Both accepted lanes project the exact union without duplicates.
4. Neither lane fails closed before extraction or produces an empty allowed
   set according to the existing caller boundary.
5. An extractor response containing `active_commitment` under
   `user_memory_units` is rejected with an off-lane diagnostic.
6. A valid user fact beside an off-lane commitment remains accepted; one bad
   candidate does not erase a valid independent row.
7. A valid active commitment remains accepted when the active-commitment lane
   is enabled.
8. The prompt contains the exact character-owner, acceptance, unanswered
   proposal, underlying-task continuation, and empty-output rules.
9. Existing malformed-output, timestamp, merge, rewrite, and source-ref tests
   remain green.
10. Source/test impact validation maps the production owner to the exact tests.

## Required Real LLM Gates

Every live case runs separately with output and trace inspected before the next
case starts. Direct extractor gates do not call persistence and do not mutate
MongoDB.

### L1 — captured false-commitment negative control

Run the real extractor with the captured two-turn task/reward context, the
actual character name, and only `user_memory_units` enabled.

Pass:

- no `active_commitment` candidate;
- no claim that the user accepted or owes a reward;
- a legitimate user-memory candidate about the disclosed workload is allowed,
  and an empty list is also allowed when the model judges it insufficiently
  durable;
- one extractor call, no retry or reviewer.

### L2 — unanswered character proposal generalization

Use a distinct synthetic exchange or condition, then let the user continue the
underlying task without responding to it. Keep only `user_memory_units`
enabled.

Pass: no active commitment and no fabricated user acceptance.

### L3 — accepted current-user character commitment positive control

Use a user-scoped request such as an address or response-order rule and an
assistant final dialog that explicitly accepts it. Enable only
`active_commitment`.

Pass:

- exactly one character-owned active commitment is allowed;
- the fact states what the character accepted for the user;
- source evidence and due-time behavior remain valid;
- the negative-case fix does not suppress a legitimate commitment.

### L4 — ordinary user fact positive control

Use a durable user fact with only `user_memory_units` enabled.

Pass: a permitted non-commitment type is emitted and retained by validation.

### L5 — memory-enabled service regression

After the separately governed Conversation Progress amendment is available,
run the semantic-progression L2 sequence under a new identity.

Pass:

- visible response satisfies the progression gate;
- post-turn memory contains no fabricated reward acceptance or user promise;
- any written memory uses a router-authorized unit type;
- the protected background trace shows the unchanged call roster;
- identity and addressee remain exact.

## Verification Order

1. Record `git status --short`, owned-file hashes, and overlapping user edits.
2. Add exact deterministic tests; run the smallest nodes and record RED.
3. Modify only the two planned production/documentation owners.
4. Run exact nodes, syntax, style, static no-filter checks, manifest validation,
   and focused non-live consolidation regression.
5. Run L1-L4 one at a time with raw outputs inspected.
6. Reload the candidate service only after deterministic and component live
   gates pass.
7. Run L5 under a brand-new identity and inspect post-turn memory.
8. Parent reviews the complete diff, traces, call counts, prompt caps, and
   multi-emotion non-touch evidence.
9. Present sign-off evidence to the user. Archive only after explicit
   acceptance.

## Suggested Deterministic Commands

Exact node names are finalized during test-first implementation.

```powershell
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_rag_flow.py -q
venv\Scripts\python.exe -m pytest tests\test_consolidation_source_policy.py -q
venv\Scripts\python.exe -m pytest tests\test_test_impact_manifest.py -q
venv\Scripts\python.exe -m scripts.validate_test_impact --check-all --run
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\consolidation\memory_units.py tests\test_user_memory_units_rag_flow.py tests\test_user_memory_units_live_llm.py
git diff --check
```

Live nodes run one at a time with `-m live_llm -q -s -o addopts=""` and raw
artifact inspection between commands.

## Runtime And Overhead Budget

- Foreground response calls: unchanged.
- Consolidation router calls: unchanged.
- Memory extractor calls: unchanged at one when an eligible user-memory or
  active-commitment lane is selected.
- Merge/rewrite/stability calls: unchanged and candidate-gated as today.
- Retries/reviewers/evaluators: none added.
- Foreground and background database reads: unchanged.
- Persistent schema and collections: unchanged.
- Prompt increase: one small bounded contract plus concise inverse-case rules;
  it must fit the existing extractor cap.

## Acceptance Criteria

1. Router-approved lanes deterministically bound extractor candidate types.
2. `user_memory_units` alone can never persist an `active_commitment` output.
3. `active_commitment` remains character-owned and acceptance-backed.
4. Unanswered character proposals and underlying-task continuation produce no
   fabricated acceptance.
5. Legitimate user facts and accepted character commitments both pass their
   positive controls.
6. LLM semantic ownership is preserved; deterministic code validates enum and
   lane contracts only.
7. No additional call, retry, persistent field, collection, or DB read exists.
8. All required live gates pass one at a time with inspected trace artifacts.
9. The exact service replay writes no false reward commitment.
10. The diff stays inside the approved change surface.

## Rollback And Recovery

The production change is prompt and deterministic admission logic in one
module. Before implementation, record the exact owned-file hash and diff. If a
required gate fails, retain the artifacts, restore only this plan's attributable
edits through a reviewed patch, and keep unrelated working-tree changes.

No database rollback is part of this plan. The reproduced false row remains an
explicit diagnostic record until the user approves an exact lifecycle action.

## Execution Roles And Approval Boundaries

### Parent architect

- owns RCA, plan scope, historical-design alignment, output-quality judgment,
  diff review, and sign-off;
- decides ordinary test-message progression under the user's delegated
  discretion;
- asks the user only when semantic output quality requires their opinion or a
  production/database scope change needs authority.

### Luna implementation and test executor

- owns only the files listed in the approved change surface;
- records RED before production edits, then GREEN;
- runs deterministic and live LLM tests one at a time as specified;
- reports any required unplanned production owner before editing it;
- preserves concurrent and user-authored working-tree changes.

### User authority

- approves production implementation of this draft;
- separately approves any unplanned production owner;
- separately approves any exact database lifecycle manifest;
- decides ambiguous semantic quality when the parent cannot make a confident
  architectural judgment.

## Progress Checklist

- [x] Reproduce the false commitment with a real memory-enabled service turn.
- [x] Verify the exact pre/post memory state.
- [x] Verify the lane router selected `user_memory_units`, not
  `active_commitment`.
- [x] Trace the missing lane constraint through extractor and validator.
- [x] Review the character-memory architecture and completed consolidation
  big-bang plan.
- [x] Draft the bounded no-extra-call solution and real-LLM gates.
- [ ] User approves production implementation.
- [ ] Luna records deterministic RED.
- [ ] Luna implements the approved change surface.
- [ ] Deterministic gates pass.
- [ ] Real LLM L1-L4 pass one at a time.
- [ ] Memory-enabled service L5 passes after the progression amendment.
- [ ] Parent review and user sign-off complete.
