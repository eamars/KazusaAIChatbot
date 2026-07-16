# relevance input scope robustness bugfix plan

## Summary

- Goal: make frontline and settled relevance distinguish character-grounded
  participation from merely answerable group traffic while preserving turn
  settlement, private input, and first-ready cognition ordering.
- Plan class: medium.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `cjk-safety`, and
  `test-style-and-execution`.
- Overall cutover strategy: bigbang.
- Highest-risk areas: group false positives, direct-address false negatives,
  production history projection, stale continuity, and private-message drift.
- Acceptance criteria: production-shaped real-LLM gates pass individually;
  deterministic relevance and settlement regressions pass; LLM call counts and
  caps remain unchanged.

## Context

The captured incident produced two independent semantic decisions: a direct
group mention was ignored, then a later unaddressed complaint in another group
entered cognition. Exact-contract live probes reproduce the unaddressed
complaint as `start` at frontline and `proceed` at settled relevance even when
the service-normalized target label is `none`.

Code inspection also found that settled history currently loses participant
and target relationships, discarded explicit-third-party input can become a
promotable prelude, bot continuity has no recency bound, and the prompts do not
receive an explicit private/group scope. Existing live fixtures manually
supply fields that production does not project, so they do not gate the real
service contract.

The completed relevance turn-settlement plan remains historical. This plan is
the bounded follow-up bugfix contract.

## Mandatory Skills

- `development-plan`: govern execution evidence, lifecycle, and review.
- `local-llm-architecture`: keep both LLM questions compact and semantic.
- `no-prepost-user-input`: keep natural-language participation judgment in the
  LLM prompts; deterministic code may validate typed scope and slot contracts.
- `py-style`: govern every Python edit.
- `cjk-safety`: govern Python tests and prompts containing CJK text.
- `test-style-and-execution`: govern test taxonomy and one-at-a-time live runs.

## Mandatory Rules

- Establish every real-LLM gate before production implementation.
- Run and inspect real-LLM gates one at a time; retain durable trace artifacts.
- Use deterministic tests for semantic projection, target labels, recency,
  candidate eligibility, validation, timing, and claim behavior.
- Keep user text as model evidence. Do not add keyword classification,
  post-LLM action rewriting, retry prompts, repair calls, or a third LLM agent.
- Deterministic code owns typed target/reply scope, same-author/channel slot
  eligibility, time windows, schema validation, and fail-closed invalid output.
- Keep prompt constants triple-single-quoted, static, role-neutral, and concise.
  Put runtime character identity and conversation facts in the human payload.
- Keep frontline and settled on `RELEVANCE_AGENT_LLM`; preserve 8,000/256 and
  16,000/512 input/completion caps and thinking disabled.
- Preserve private immediate readiness, group six-second quiet timing,
  ten-second hard deadline, one optional wait, FIFO relevance work, and the
  single atomic cognition lane.
- Use `venv\Scripts\python` for Python and pytest commands.
- Use `apply_patch` for manual edits and preserve unrelated workspace changes.
- After automatic context compaction or a major stage sign-off, reread this
  entire plan before continuing.
- Before completion, run the parent-only independent code review authorized by
  the user and record findings and remediation in Execution Evidence.

## Must Do

- Capture production-shaped live gates for the incident and uncovered
  input-driven failures before changing production.
- Add private/group scope and runtime character identity to bounded relevance
  projections without exposing platform identifiers.
- Make group target `none` require a character-specific participation basis;
  generic usefulness, empathy, or answerability is insufficient.
- Preserve clear whole-group invitations, direct character address, replies to
  the character, and exact open-turn continuity.
- Correct settled history projection to retain current-author, character, and
  other-participant relations plus semantic target/reply relations.
- Prevent explicit-third-party discarded messages from becoming prelude
  candidates.
- Bound latest-bot continuity to the active scene window.
- Distinguish unresolved reply targets from absence of a reply.
- Align wait availability with the prompt contract without adding another LLM
  call or exposing operational timing.
- Update relevance and brain-service ICDs to the final canonical contract.

## Deferred

- Cognition cancellation or revision after a turn reaches `RUNNING`.
- Cross-author collaborative turn assembly.
- Adapter parsing changes, delivery changes, persistence migrations, and base
  infrastructure error handling.
- New LLM routes, extra relevance agents, retries, feature flags, compatibility
  shims, or prompt-specific keyword filters.
- Alias and nickname resolution beyond the runtime canonical character name.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Relevance payloads | bigbang | Update callers, projections, prompts, and tests to one canonical semantic shape. |
| History projection | bigbang | Replace artificial `speaker`/summary fixture assumptions with production relation fields. |
| Continuity and preludes | bigbang | Expose only eligible semantic candidates; preserve no legacy candidate shape. |
| Tests | bigbang | Make production-shaped gates authoritative and update obsolete fixtures directly. |

## Cutover Policy Enforcement

- Follow the selected policy for every area.
- Rewrite callers and tests together; create no compatibility mapper or dual
  payload.
- A cutover-policy change requires user approval.

## Target State

```text
typed envelope and deterministic semantic projection
  -> compact frontline LLM: discard | start | append
  -> same-author/same-channel settlement coordinator
  -> settled LLM: ignore | proceed | wait when available
  -> version check and atomic cognition claim
```

Frontline sees conversation scope, the runtime character name, current typed
target/reply/media evidence, bounded eligible open turns and preludes, and
recent same-author/same-channel bot continuity. Settled relevance sees the
assembled turn, correct participant/target relations in fresh history, bounded
scene and relationship evidence, and an action contract that reflects whether
wait is currently available.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Agent count | Keep two LLM stages | Current failures are evidence and contract defects; another call adds local-model load without new ownership. |
| Group `none` | Default to silence absent a grounded participation basis | Answerability is not permission for the character to join a group thread. |
| Private input | Expose `private` scope and retain typed character address | Conservative group rules must not suppress DMs. |
| Natural name address | Supply the runtime canonical name as evidence | The frontline needs identity distinction without a hard-coded persona name. |
| Typed third-party scope | Exclude from prelude eligibility and reject invalid slots | This is deterministic scope validation, not semantic text interpretation. |
| Continuity | Retain only within the existing active-scene window | Old dialog must not authorize unrelated later group speech. |
| Wait | Render the allowed action vocabulary for the current assessment | The model should know whether wait is available without reasoning over clocks or phase labels. |
| Invalid LLM output | Fail closed | The user accepts conservative discard over wrong-scope cognition. |

## Contracts And Data Shapes

Frontline dynamic payload adds:

```python
{
    "conversation_scope": "group|private",
    "active_character_name": str,
    "current_message": {...},
    "open_turns": list[dict],
    "recent_preludes": list[dict],
    "latest_bot_continuity": str,
}
```

Frontline renders separate static `group` and `private` instruction branches.
Its output action schema omits `append` when the exact capped payload contains
no open slot, and absent append/prelude slot references fail closed.

Settled dynamic payload adds `conversation_scope`, `active_character_name`, and
an assembled-turn relation contract:

```python
{
    "author_relation": "current_human",
    "effective_latest_fragment": {...},
    "fragments": list[dict],
    "earlier_context_present": bool,
    "media": {...},
}
```

Each fresh-history row becomes:

```python
{
    "speaker_relation": "character|current_author|other_participant",
    "body_text": str,
    "target_summary": str,
    "reply_summary": str,
    "turn_relation": (
        "before_active_turn|during_active_turn|after_active_turn|unknown"
    ),
}
```

The reply label vocabulary adds `unknown_participant` for a present typed reply
whose author is unresolved. Output action vocabularies stay unchanged.

## LLM Call And Context Budget

| Call | Before | After | Blocking and caps |
|---|---|---|---|
| Frontline | One call per active input | Unchanged | Shared FIFO relevance executor; scope-specific and candidate-aware static prompt branch; <=8,000 rendered characters, <=256 completion tokens, thinking off. |
| Settled relevance | One call normally, two after wait | Unchanged | Admitted turns only; <=16,000 rendered characters, <=512 completion tokens, thinking off. |

New semantic descriptors replace lost or ambiguous evidence and add no model
call. Character-based caps conservatively remain below the 50k-token planning
cap.

## Change Surface

### Delete

- No production module deletion.

### Modify

- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py`: canonical
  payload and participation prompt.
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`: canonical
  settled prompt, wait contract, and history relations.
- `src/kazusa_ai_chatbot/brain_service/turn_settlement.py`: eligible preludes
  and bounded bot continuity.
- `src/kazusa_ai_chatbot/service.py`: semantic scope, identity, reply, and
  settled-history inputs required by relevance.
- `src/kazusa_ai_chatbot/relevance/README.md` and
  `src/kazusa_ai_chatbot/brain_service/README.md`: final runtime contract.
- Focused deterministic relevance, settlement, and service tests.
- `development_plans/README.md`: active and completed lifecycle rows.

### Create

- `tests/test_relevance_cross_channel_failure_live_llm.py`: captured and
  production-shaped real-LLM release gates.
- This active bugfix plan; archive it after completion.

### Keep

- Adapters, message-envelope schemas, cognition, dialog, persistence, RAG,
  consolidation, and base runtime coordination.

## Overdesign Guardrail

- Actual problem: unaddressed group speech can claim cognition while direct
  character speech is discarded, and production context loses scope relations.
- Minimal change: correct semantic projection and tighten the two existing LLM
  contracts plus typed candidate eligibility.
- Ownership boundaries: LLMs judge semantic participation; deterministic code
  owns typed scope, time, candidates, validation, and claims.
- Rejected complexity: another LLM, retries, keyword rules, compatibility
  payloads, cross-author turns, cognition preemption, and adapter changes.
- Evidence threshold: a production-shaped gate that still fails after this
  contract and cannot be traced to projection or implementation is required
  before another agent or call is proposed.

## Agent Autonomy Boundaries

- The parent performs the user-approved single-agent fallback execution and
  parent-only final review.
- Local mechanics may change only inside this plan's contracts and paths.
- Equivalent helpers must be reused after project search; thin wrappers and
  speculative abstractions are forbidden.
- Unrelated cleanup, formatting churn, new routes, fallbacks, and broad prompt
  changes are forbidden.
- A required change outside the listed surface requires plan update and user
  approval.

## Implementation Order

1. Add all captured and boundary real-LLM gates to the dedicated live test.
2. Run each new live gate individually, inspect its trace, and record baseline.
3. Add deterministic tests for scope projection, history relations, reply
   `unknown_participant`, prelude eligibility, continuity expiry, and wait
   rendering.
4. Implement the minimal production contract across relevance, coordinator,
   and service.
5. Run focused deterministic tests and syntax/prompt-render checks.
6. Rerun every affected live gate individually and inspect each trace.
7. Run affected non-live regressions and static scope checks.
8. Perform parent-only independent code review, remediate findings, and rerun
   affected verification.
9. Update ICDs, evidence, registry, and archive the completed plan.

## Execution Model

- User-approved fallback execution: the parent owns tests, production code,
  verification, evidence, review remediation, lifecycle updates, and sign-off.
- The parent establishes the focused real-LLM and deterministic contracts
  before production implementation.
- The parent performs a fresh-posture independent code review after planned
  verification and records every finding and fix.

## Progress Checklist

- [x] Stage 1 - real-LLM gates captured and baselined individually.
  - Verify: each named case runs alone with an inspected durable trace.
  - Evidence: record action, reason, model, prompt size, and judgment below.
  - Handoff: add deterministic contract tests.
  - Sign-off: parent, 2026-07-16.
- [x] Stage 2 - deterministic contracts established and failing as expected.
  - Verify: focused tests identify the missing semantic fields and eligibility.
  - Evidence: record exact commands and failures.
  - Handoff: implement production contract.
  - Sign-off: parent, 2026-07-16.
- [x] Stage 3 - production contract and ICDs implemented.
  - Verify: syntax, prompt rendering, and focused deterministic tests pass.
  - Evidence: record changed files and commands.
  - Handoff: rerun live and regression gates.
  - Sign-off: parent, 2026-07-16.
- [x] Stage 4 - live and non-live delivery gates pass.
  - Verify: live cases run individually; focused and affected suites pass.
  - Evidence: record trace review and test totals.
  - Handoff: independent code review.
  - Sign-off: parent, 2026-07-16.
- [x] Stage 5 - parent-only independent code review and closeout complete.
  - Verify: diff, plan alignment, skill compliance, test realism, private
    behavior, workload, and residual risks reviewed; findings remediated.
  - Evidence: record findings, fixes, reruns, and approval below.
  - Handoff: archive plan and update registry.
  - Sign-off: parent, 2026-07-16.

## Verification

### Static And Syntax

- `venv\Scripts\python -m py_compile` on every changed Python file succeeds.
- Runtime calls to both prompt builders produce valid bounded JSON payloads.
- `rg "keyword|retry|compat"` review finds no newly introduced semantic
  keyword classifier, retry path, or compatibility layer in changed source.

### Deterministic Tests

- Run focused relevance, settlement, graph, and service-input tests with the
  project virtual environment.
- Expected result: all pass; existing private timing and first-ready claim
  behavior remain unchanged.

### Real LLM Tests

- Run every test in `test_relevance_cross_channel_failure_live_llm.py` one at a
  time with `-m live_llm -q -s` and inspect the newest artifact before the next.
- Expected result: no wrong-scope group input starts or proceeds; direct,
  private, whole-group, and exact-continuation inputs remain admitted.

## Independent Code Review

After verification, the parent rereads this plan and reviews the full diff and
evidence from a fresh posture. Review project/style compliance, LLM ownership,
projection correctness, private/group behavior, wait alignment, candidate
scope, workload, prompt/test leakage, fixture realism, and plan scope. Fix
in-scope findings directly and rerun affected checks. Record approval below.

## Acceptance Criteria

This plan is complete when:

- Captured direct mention proceeds and captured unaddressed complaint is
  discarded at frontline and ignored at settled relevance.
- Private input remains character-addressed and immediately ready.
- Whole-group invitations remain eligible while ambient group statements,
  unsolicited media, stale continuity, prompt injection, and unknown replies
  fail closed without another basis.
- Explicit-third-party preludes are unavailable and ambiguous multi-turn input
  never false-appends.
- Fresh history preserves participant, target, and reply relations and can
  suppress a redundant response before, during, or after interleaved active
  fragments when the temporal evidence supports it.
- Wait is available once only and the hard-deadline prompt cannot emit wait.
- Existing timing, FIFO, first-ready, claim, media, and private regressions pass.
- LLM call count, route, input caps, and completion caps remain unchanged.
- Parent-only independent review has no unresolved finding.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Conservative group policy discards valid ambient participation | Preserve whole-group, direct-name, reply, scene, and continuity bases in the LLM contract | Positive live gates |
| Prompt fix passes only incident wording | Use varied boundary fixtures and forbid incident-specific examples in prompts | Trace review and prompt diff |
| Private messages inherit group silence | Explicit scope and private live/deterministic gates | Private gates |
| Correct history increases final-model load | Use bounded semantic relations and existing 4,000-character history budget | Prompt cap tests |
| Wait contract becomes inconsistent | Render current allowed actions and validate existing phase rule | Prompt-render and deadline tests |

## Execution Evidence

- Pre-implementation exact-label incident baseline:
  - Frontline target `none`: `start`, 2,123 prompt characters, 1,229 ms.
  - Settled target `none`: `proceed`, 2,945 prompt characters, 2,270 ms.
- Stage 1 real-LLM baseline used
  `gemma-4-26b-a4b-it-claude-opus-distill-v2`; every case ran alone and its
  newest JSON trace under `test_artifacts/llm_traces` was inspected before the
  next case:

| Case | Baseline | Prompt chars | Gate judgment |
|---|---:|---:|---|
| RCA00 unaddressed complaint/frontline | `start` | 2,123 | fail: answerability was treated as participation |
| RCA01 captured direct mention/settled | `proceed` | 2,901 | pass |
| RCA02 unaddressed complaint/settled | `proceed` | 2,945 | fail: helpfulness overrode target scope |
| RCA03 whole-group invitation | `discard` | 2,134 | fail: valid whole-group listener basis lost |
| RCA04 canonical-name address | `start` | 2,133 | pass |
| RCA05 private message/frontline | `start` | 2,126 | pass |
| RCA06 explicit-third-party prelude | `start`, no selected prelude | 2,209 | model pass; deterministic candidate leak remains |
| RCA07 stale continuity | invalid `append`, validator `discard` | 2,131 | fail: semantic result depended on validator fallback |
| RCA08 recent continuity | invalid `append`, validator `discard` | 2,131 | fail: bot continuity was mistaken for an open slot |
| RCA09 two ambiguous open turns | `append open_1` | 2,526 | fail: unsupported parent selection |
| RCA10 unaddressed prompt injection | `discard` | 2,174 | pass |
| RCA11 unaddressed media | `discard` | 2,125 | pass |
| RCA12 latest recipient switches away | `proceed` | 2,233 | fail: other participant was treated as response authority |
| RCA13 production-shaped other-user answer | `ignore` | 2,185 | output pass; deterministic projection still needs relation proof |
| RCA14 private emotional message/settled | `proceed` | 2,078 | pass |
| RCA15 unresolved reply target | `ignore` | 2,095 | model pass; service label still needs deterministic proof |

- RCA09 was rerun after correcting its current fragment from an inherited
  character target to the intended `none`/`none` shape; it still selected an
  arbitrary parent, confirming a real semantic failure.
- Stage 2 command:
  `venv\Scripts\python -m pytest tests\test_frontline_relevance_agent.py
  tests\test_persona_relevance_agent.py
  tests\test_relevance_turn_settlement.py tests\test_service_input_queue.py -q`.
  Result: 10 failed, 66 passed. The failures prove the missing canonical
  scope/name projections, participation prompt contract, phase-specific wait
  rendering, production history relations, third-party prelude exclusion,
  continuity expiry, unresolved-reply label, and service-to-relevance identity
  fields. Existing private readiness, FIFO work, interleaved-author slot
  isolation, settlement versions, watermark, and cognition claim tests passed.
- Stage 3 implemented the canonical scope/name payload, group participation
  prompts, phase-specific settled action prompts, production history relation
  projection, typed unresolved-reply label, third-party prelude exclusion, and
  180-second bot-continuity expiry. Both relevance and brain-service ICDs now
  describe the contract.
- Focused post-implementation result: 76 passed across frontline, settled,
  coordinator, and service-input tests. All changed Python files compiled;
  `git diff --check` passed with line-ending notices only. The changed-source
  scan introduced no keyword classifier, retry, or compatibility bridge.
- Stage 4 surfaced and fixed three implementation gaps without adding an LLM
  call: the local model needed explicit current-human authorship, a bounded
  `effective_latest_fragment`, and before/after/unknown history relation to
  apply recipient correction and intervening-answer redundancy reliably. A
  final prompt clarification distinguishes an explicit whole-group request
  from a statement that merely permits reactions.
- Final real-LLM delivery result: RCA00-RCA21 all passed one case at a time on
  `gemma-4-26b-a4b-it-claude-opus-distill-v2`; each newest trace was inspected.
  Final frontline prompts were about 3,300-3,800 characters for group and
  2,037 for private; affected settled prompts were 4,405-4,504, under the
  unchanged 8,000 and 16,000 caps. Direct, private, canonical-name,
  whole-group, recent-continuity, bare-summon follow-up, and interleaved-answer
  cases passed. Ambient group, stale continuity, ambiguous parents, prompt
  injection, unsolicited media, recipient switch, redundant answer, and
  unresolved reply cases failed closed as intended. The partial-opener gate
  accepts only `append(open_1)` or a valid conservative `discard`; `start` is a
  release failure under the user-approved weak-model policy.
- Final affected non-live result: 87 passed across relevance, settlement,
  service input, private multimodal integration, and graph claim/ignore tests.
- Stage 5 parent-only independent review findings and remediation:
  - The plan schema omitted implemented current-human, effective-latest, and
    temporal-history fields. The canonical data-shape section now records all
    three.
  - A ten-row busy-group history window could evict the active row and erase
    before/after evidence. Persisted first/latest active timestamps now provide
    a fail-safe fallback; equal or invalid timestamps remain `unknown`.
  - The existing mention-follow-up live fixture incorrectly retargeted the
    second message to the character. RCA19 and RCA20 now use production-shaped
    target-`none` follow-ups. The ordered frontline DAG keeps the bare summon in
    one turn and prevents a weak partial-completion miss from starting a split
    turn.
  - A combined group/private prompt caused the local model to apply the private
    start rule to group input. Static scope-specific branches remove that role
    conflict and reduce private prompt load.
  - The always-open action schema encouraged invented append/prelude slots, and
    the live test could pass despite the contract violation. Candidate-aware
    action vocabularies plus validation against the exact capped model payload
    now fail closed; deterministic tests cover both invented slot kinds.
  - An external answer between two active fragments was previously labeled
    before the final fragment. The canonical `during_active_turn` relation and
    RCA21 preserve this interleaving evidence without cross-author assembly.
  - Private immediate readiness, coalesced input, one-call routing, shared FIFO
    relevance work, and the atomic cognition lane remain unchanged. No retry,
    repair call, keyword classifier, compatibility bridge, or extra agent was
    introduced.
- Final static review: all changed Python compiled, `git diff --check` passed
  with line-ending notices only, changed-source scan found no added LLM call
  site and no new retry/keyword/compatibility path, and no unresolved review
  finding remains.
