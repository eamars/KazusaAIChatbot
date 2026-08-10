# settled relevance pre-active answer attribution bugfix plan

## Summary

- Goal: prevent settled relevance from using pre-active-turn answers or links
  to suppress a new character-directed request.
- Plan class: large (prompt contract, production projection, live-LLM gates,
  ICDs, and independent review; no migration).
- Status: completed.
- Mandatory skills: `development-plan`, `debug-llm`, `llm-trace-debug`,
  `local-llm-architecture`, `no-prepost-user-input`, `py-style`,
  `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: bigbang for the private model-facing history
  projection; no database migration.
- Highest risks: false `ignore` for direct requests and loss of genuine
  after-turn redundancy suppression.
- Acceptance: H01 returns `proceed` in three individually inspected real-LLM
  runs; a genuine after-turn answer still returns `ignore`; tests, evidence,
  and independent review pass; no code overrides the LLM action.

## Context

The incident is QQ group `638473184`, platform message `1509040787`:

`@杏山千纱 千纱老师，能帮 @3945160191 搜几个bilibili的部署astrbot的视频么？`

Protected trace `llmtrace_7ae1154b3a734de89933972b19732622` contains frontline
and settled persona relevance only. Cognition, RAG/Bilibili, dialog, and
delivery never ran, so the no-response decision belongs to settled relevance.

The production-shaped reproduction
`tests/test_relevance_turn_settlement_live_llm.py::test_live_recreates_astrbot_request_no_response_gate`
returned `proceed`, `ignore`, and `proceed` across three individually run
trials. The false `ignore` reason said that another participant had already
fulfilled the AstrBot request. The cited links and answer summary were
`before_active_turn`. The existing contract exposes that relation in a flat
history list, which the local model sometimes misattributes as current-turn
completion evidence.

The completed
`development_plans/archive/completed/bugfix/relevance_input_scope_robustness_bugfix_plan.md`
established typed participant and temporal relations and remains historical.
This plan is the narrow follow-up. It changes settled evidence shaping and
tests only. Frontline still makes its own LLM admission decision; settled
relevance still makes the semantic `ignore|proceed` decision and `wait` when
available.

## Mandatory Skills

- `development-plan`: lifecycle, test-first execution, evidence, and review.
- `debug-llm`: human-readable review from real raw model outputs.
- `llm-trace-debug`: protected trace and dialog-RCA boundary.
- `local-llm-architecture`: compact semantic prompts and bounded local-model
  context.
- `no-prepost-user-input`: keep message ownership and reply judgment in the
  LLM; deterministic code may expose typed temporal provenance only.
- `py-style` and `cjk-safety`: every Python and CJK fixture/prompt edit.
- `test-style-and-execution`: deterministic versus real-LLM test taxonomy and
  one-at-a-time live execution.

## Mandatory Rules

- Execute only after this plan is approved or `in_progress` and the user
  explicitly authorizes implementation.
- The settled relevance LLM remains the sole semantic chooser of `ignore`,
  `proceed`, and available `wait`. Deterministic code only projects typed
  evidence, validates the closed contract, applies limits, persists state, and
  routes or claims a validated action.
- Do not add keyword/URL/topic classifiers, answer detectors, direct-mention
  force-proceed branches, post-LLM action rewrites, retries, repair prompts,
  or another relevance call.
- Use only typed `turn_temporal_relation` for partitioning. Whether text
  answers the current request remains an LLM judgment. Message text stays
  evidence in the human prompt, never judge instructions.
- Preserve the current route, temperature, thinking setting, completion/input
  caps, action enums, native-reply semantics, settlement timing, bounded wait,
  and atomic cognition claim. Keep `service.py`'s source `fresh_history` list
  unchanged; grouping is private to the relevance projection.
- Use `venv\Scripts\python`, `apply_patch`, and the existing user test change.
  Run live LLM cases one at a time, inspect each result, and author the
  human-readable review from raw artifacts. Deterministic tests may batch.
- After compaction or each major checklist sign-off, reread this plan. Before
  completion or lifecycle sign-off, run Independent Code Review and record
  findings, fixes, reruns, and residual risks. Use parent-led native subagents;
  stop if unavailable unless the user authorizes fallback.

## Must Do

- Capture the H01 baseline before implementation.
- Replace the flat model-facing settled history list with four explicit
  temporal partitions and update its prompt/cap fallback atomically.
- Require a current-turn participation basis and allow redundancy evidence only
  from during/after partitions; preserve LLM action ownership.
- Update projection, prompt, fallback, production-shaped live tests, and both
  relevance ICDs.
- Retain a genuine after-turn answer regression and author a human-readable
  before/after debug review.
- Run focused, integration, static, one-at-a-time live, and independent-review
  gates before sign-off.

## Deferred

- Do not modify frontline, message parsing, queue/settlement ownership or
  timing, service routing, graph, cognition, RAG/web/Bilibili, dialog,
  persistence, consolidation, delivery, or adapters.
- Do not change MongoDB data/indexes/retention, trace capture, participant or
  nickname resolution, or cross-author turn assembly.
- Do not add agents, routes, retries, flags, compatibility payloads, semantic
  filters, or broad prompt refactors.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Model-facing history | bigbang | Replace the flat list with the canonical four-key object in projection, prompt, tests, and ICDs together. |
| Service/database state | unchanged | Keep the source list and persisted rows; perform no migration or compatibility translation. |
| Action semantics | unchanged | Use the validated relevance LLM action as the only semantic reply decision. |

No rollout flag or dual payload is authorized. Changing this policy requires
user approval and a plan revision.

## Target State

```text
typed envelope -> frontline relevance LLM
  -> settled projection with temporal partitions
  -> settled relevance LLM: ignore | proceed | wait
  -> deterministic validation/claim/routing
  -> cognition only for validated proceed
```

The model-facing `fresh_history` is:

```json
{
  "before_active_turn_context": [],
  "during_active_turn_evidence": [],
  "after_active_turn_evidence": [],
  "unknown_timing_context": []
}
```

Before-turn and unknown-timing rows are context only and cannot prove that the
current request was answered. During/after rows are candidate evidence only;
the LLM must judge same-request and recipient equivalence. A typed current
character address remains a valid `proceed` basis when no eligible row resolves
the current request.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Semantic authority | Keep settled relevance as sole action chooser. | The failure is unstable semantic attribution, not missing deterministic routing. |
| Evidence shaping | Partition existing temporal relations before prompt rendering. | A flat list did not reliably preserve the evidence boundary for the local model. |
| Redundancy | During/after is necessary provenance, not automatic suppression. | Existing legitimate answer suppression remains required. |
| Text interpretation | No deterministic body, URL, or topic classification. | Request ownership and answer equivalence belong to the LLM. |
| Budget | Keep the existing frontline and settled calls, including the existing optional settled reassessment, and all caps. | A retry or extra judge adds latency and masks the contract defect. |

## Contracts And Data Shapes

- `_project_history(state: Mapping[str, Any])` returns a dict with exactly
  `before_active_turn_context`, `during_active_turn_evidence`,
  `after_active_turn_evidence`, and `unknown_timing_context`, each a list.
- Each row retains `speaker_relation`, `body_text`, `target_summary`,
  `reply_summary`, and `turn_relation`; it is placed from normalized
  `turn_temporal_relation` as follows:
  `before_active_turn` -> context, `during_active_turn` -> during evidence,
  `after_active_turn` -> after evidence, missing/invalid -> unknown context.
- The projection still uses at most ten newest external rows and the existing
  history character cap. No identifiers, timestamps, rows, or semantic text
  labels are added. Chronological order is preserved within every partition.
- Cap fallback uses the same four-key empty object, never a legacy list.
- `build_settled_relevance_messages` keeps its signature and system/human pair;
  decision validation keeps its output fields and action vocabulary.
- `service._settled_state_from_lease` remains the source of the list and
  temporal labels; persistence and public service contracts do not change.

The settled prompt defines the four keys and applies this procedure: read the
effective latest fragment; identify its typed participation basis; treat the
assembled turn as current human input; inspect only during/after evidence for
a possible same-request answer; use the LLM to judge equivalence; treat
before/unknown as context only; allow a during row to resolve only an earlier
fragment, never meaning introduced by a later fragment; then emit the action
allowed by the observation phase with native-reply anchoring separate from
action choice.

## LLM Call And Context Budget

| Call | Before | After |
|---|---|---|
| Frontline | One existing `RELEVANCE_AGENT_LLM` call; hard maximum 8,000 input chars/256 completion tokens, thinking off. | Unchanged. |
| Settled | One call per assessment; hard maximum 16,000 input chars/512 completion tokens, thinking off, with the optional bounded wait. | Same route, caps, settings, and wait contract; only bounded history JSON and concise prompt procedure change. |

The four partitions share the current history budget and existing drop order;
the hard `SETTLED_RELEVANCE_MAX_INPUT_CHARS` cap and default 50k-token context
ceiling do not increase. There is no new response-path latency, database call,
blocking stage, retry, or judge. Complete observation still exposes only
`ignore|proceed`.

## Change Surface

### Delete

- None.

### Modify

- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`: settled prompt, `_project_history`, payload, and cap fallback; target production boundary.
- `tests/test_persona_relevance_agent.py`: update production-relation/worst-case tests; add exact unknown-timing and prompt-boundary checks.
- `tests/test_relevance_turn_settlement_live_llm.py`: H01 requires `proceed` and retains production-shaped trace evidence.
- `src/kazusa_ai_chatbot/relevance/README.md` and `src/kazusa_ai_chatbot/brain_service/README.md`: canonical projection and unchanged ownership.
- `development_plans/README.md`: registry row for this draft.

### Create

- This plan file and, during execution, the ignored parent-authored `test_artifacts/debug/relevance_pre_active_answer_bugfix_review.md` from raw live traces; test code emits evidence, not the Markdown judgment.

### Keep

- `src/kazusa_ai_chatbot/service.py`: its typed temporal projection already supplies the required source list.
- Frontline, settlement, graph, cognition/RAG/web/dialog, persistence, delivery, adapters, and existing after/during live regressions, including `test_live_production_history_answer_makes_turn_redundant`.

## Overdesign Guardrail

- Actual problem: a weak model uses pre-active-turn answer content to suppress
  a new explicit character request.
- Minimal change: partition existing temporal provenance and state its
  redundancy proof rule in the settled prompt.
- Ownership: LLM decides participation, equivalence, stance, and action;
  deterministic code supplies provenance, validates output, and controls
  lifecycle.
- Rejected complexity: classifiers, overrides, retries, extra fields, agents,
  routes, flags, compatibility payloads, migrations, and broad refactors.
- Evidence threshold: a separately approved plan requires repeated post-fix
  failures before adding any new judge or semantic classifier.

## Agent Autonomy Boundaries

- Parent owns orchestration, tests, live inspection, debug review,
  verification, review remediation, registry, and sign-off.
- The production subagent edits only the target relevance module and changes
  only the specified prompt/projection behavior.
- Agents may choose local mechanics only while preserving the four-key payload,
  caps, action contract, and listed scope. No compatibility layer, fallback
  path, semantic filter, or unrelated cleanup is allowed.
- A source/plan discrepancy stops execution for plan revision; it is not a
  reason to widen the change silently.

## Implementation Order

1. Parent updates
   `test_settled_history_projects_production_participant_relations`,
   `test_settled_worst_case_projection_remains_valid_json`,
   `test_settled_history_partitions_unknown_temporal_relation`, and
   `test_settled_prompt_defines_temporal_history_evidence` in
   `tests/test_persona_relevance_agent.py`; run them before production code and
   record the expected flat-list failure.
2. Parent runs H01 three separate times before implementation and records raw
   outputs, parsed actions, prompt sizes, and false-ignore observations.
3. Parent starts exactly one native production subagent with this plan,
   mandatory skills, failing contract, and ownership limited to
   `persona_relevance_agent.py`; the subagent edits no tests.
4. Subagent implements the partitions, prompt procedure, and same-shape cap
   fallback, then reports symbols, commands, and residual risks.
5. Parent reruns the module tests, then verifies the unchanged service source
   contract with `test_settled_fresh_history_excludes_active_turn_fragments`
   and `test_settled_history_uses_timestamps_when_active_row_is_outside_window`
   in `tests/test_service_input_queue.py`, plus
   `test_proceed_requires_claim_before_downstream_cognition` and
   `test_ignore_ends_before_claim_and_cognition` in
   `tests/test_relevance_turn_settlement_graph.py`.
6. Parent updates/runs H01 three separate times, runs
   `test_live_production_history_answer_makes_turn_redundant` and
   `test_live_interleaved_other_user_answer_makes_turn_redundant` individually,
   and appends each inspected result to the debug review before the next live
   case.
7. Parent updates both ICDs, runs all remaining gates, starts the independent
   code-review subagent, remediates only in-scope findings, reruns affected
   gates, and records closeout evidence.

## Execution Model

- Parent owns orchestration, test-contract setup, tests, verification,
  live-output inspection, debug review, remediation, lifecycle, and sign-off.
- Exactly one native production subagent edits production code only; exactly one
  independent native review subagent reviews after verification and does not
  implement fixes. Parent may run tests, evidence, static checks, and docs in
  parallel with production editing.
- If native subagents are unavailable, stop until the user explicitly approves
  fallback execution.

## Progress Checklist

- [x] Stage 1 — H01 baseline and named projection/prompt tests; verify expected
  flat-list failure, three individual H01 runs, and raw paths; record evidence,
  sign off, and reread this plan before Stage 2. Evidence is recorded below.
- [x] Stage 2 — `persona_relevance_agent.py` implementation; verify focused
  tests and canonical fallback; record symbols/risks, sign off, and reread plan.
- [x] Stage 3 — H01/after-turn live regressions and debug review; verify H01
  `proceed` three times, after-turn `ignore`, and no rewrite; record raw review,
  sign off, and reread plan.
- [x] Stage 4 — both ICDs, deterministic suite, static checks, and exact scope;
  verify all gates below and record results before reviewer handoff.
- [x] Stage 5 — independent review and closeout; reviewer checks plan, diff,
  tests, prompts, boundaries, artifacts, and evidence; parent records findings,
  fixes, reruns, risks, approval, and checks this final box.

## Execution Evidence

### Stage 1 — baseline and focused contract

- The historical H01 baseline was run as three separate live invocations. Each
  produced frontline `start`; settled actions were `proceed`, `ignore`, and
  `proceed`. The false `ignore` cited a pre-active answer as if it completed
  the current request. Raw summaries:
  - `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__H01_astrbot_request_summary__20260722T142231953149Z.json`
  - `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__H01_astrbot_request_summary__20260722T142306355413Z.json`
  - `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__H01_astrbot_request_summary__20260722T142333035918Z.json`
- A current pre-fix invocation reproduced the same failure with settled
  `ignore`; its raw summary is
  `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__H01_astrbot_request_summary__20260722T220217470152Z.json`.
- `venv\Scripts\python -m pytest tests/test_persona_relevance_agent.py -q`
  was run after the focused contract edit: 9 passed and 4 failed. The four
  expected failures were the temporal partition projection, unknown-timing
  partition, prompt-key contract, and bounded fallback-shape assertions.
- The plan and repository context were reread after Stage 1. Production code
  remained unchanged at the handoff to the implementation subagent.

### Stage 2 — production implementation

- Exactly one native production subagent edited only
  `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`.
- The implementation normalizes missing/invalid timing to
  `unknown_timing_context`, preserves the existing row fields and newest-ten
  history bound, keeps chronological order within each partition, and uses the
  same four-key object for cap fallback.
- The settled prompt now defines the partition contract and keeps the LLM as
  the sole `ignore|proceed|wait` chooser. No service, graph, route, retry, or
  action-rewrite code changed.
- `venv\Scripts\python -m py_compile
  src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py` passed.
- Focused persona relevance tests passed: 13.
- Stage 2 sign-off: implementation scope and focused contract passed; this
  plan was reread before entering live regressions.

### Stage 3 — live regression and quality review

- H01 was run three times individually after the fix. All three returned
  frontline `start` and settled `proceed`, with prompt size 7,246 chars and
  direct-address rationales. Raw settled traces:
  - `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__H01_astrbot_request_settled__20260722T221208821414Z.json`
  - `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__H01_astrbot_request_settled__20260722T221238714570Z.json`
  - `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__H01_astrbot_request_settled__20260722T221259322782Z.json`
- Genuine after-turn answer: `ignore`, prompt 5,526 chars; raw trace
  `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__RCA13_production_history_answer__20260722T221348286596Z.json`.
- Genuine during-turn answer: `ignore`, prompt 5,625 chars; raw trace
  `test_artifacts/llm_traces/relevance_turn_settlement_live_llm__RCA21_interleaved_other_user_answer__20260722T221416224392Z.json`.
- Individual controls passed: direct address (`proceed`), recipient switch
  (`ignore`), private emotional input (`proceed`), and group noise with native
  reply anchor (`proceed`, anchor `true`). Their inspected raw traces and
  quality judgments are in
  `test_artifacts/debug/relevance_pre_active_answer_bugfix_review.md`.
- No deterministic conversion of any returned relevance action was observed.
- Stage 3 sign-off: strict H01 and redundancy gates passed; this plan was
  reread before final deterministic/static verification.

### Stage 4 — deterministic, static, and scope verification

- `tests/test_persona_relevance_agent.py`: 13 passed.
- `tests/test_relevance_turn_settlement.py`,
  `tests/test_relevance_turn_settlement_graph.py`, and
  `tests/test_service_input_queue.py`: 64 passed.
- `tests/test_frontline_relevance_agent.py`: 11 passed.
- Python syntax checks passed for the modified production and test modules.
- `git diff --check` passed; only expected LF/CRLF conversion warnings were
  emitted.
- Canonical partition keys were found in the target module, tests, both ICDs,
  and the active plan. The action audit found only the existing validation,
  compatibility derivation, routing, and trace fields; no new action rewrite.
- Tracked scope is the registry, target production module, two ICDs, and two
  planned test files, with the active plan as the expected untracked file.
  The ignored parent-authored debug review is the only execution artifact
  outside that tracked scope.
- Stage 4 sign-off: all pre-review gates passed; the plan is ready for the
  independent native code review.

### Post-review remediation and final verification

- Reviewer P3 documentation finding: the existing production subagent expanded
  `_project_history`'s docstring with the exact four keys, invalid/missing
  normalization, partition ordering, and retained bounds; immediate
  `py_compile` passed.
- Reviewer P3 evidence finding: the live settled harness now records bounded
  rendered-history keys and row counts in each raw trace. The final H01 traces
  show keys in canonical order with counts `9/0/1/0`.
- A final during-turn rerun initially returned `proceed` despite the correct
  `0/1/0/0` projection. Raw inspection showed the model treated a later
  same-request clarification as requiring a new answer. The settled prompt was
  clarified to distinguish same-request clarification from a distinct later
  request or withdrawal; the final rerun returned `ignore` and cited
  `during_active_turn_evidence`.
- A final group-noise control initially returned `proceed` with an omitted
  native anchor twice. The prompt was clarified narrowly for noisy-group direct
  questions where anchoring materially identifies the target message; the
  final rerun returned `proceed` with `use_reply_feature: true`.
- After remediation, persona relevance tests passed: 13; settlement,
  service, and graph tests passed: 64; frontline relevance tests passed: 11;
  all modified Python modules compiled; `git diff --check` passed.
- Final inspected live raw traces:
  - H01 settled: `...H01_astrbot_request_settled__20260722T223612028453Z.json`,
    `...20260722T223634638466Z.json`, and
    `...20260722T223657320820Z.json` — all `proceed`.
  - After-turn: `...RCA13_production_history_answer__20260722T223548824850Z.json` — `ignore`.
  - During-turn: `...RCA21_interleaved_other_user_answer__20260722T223528230677Z.json` — `ignore`.
  - Group-noise anchor: `...L21_specific_group_reply_anchor__20260722T224036018170Z.json` — `proceed`, anchor `true`.

## Verification

### Static and scope checks

- `git diff --check` exits zero; line-ending warnings may be recorded.
- `git diff --name-only` contains only the tracked registry, target module, two
  ICDs, and two test files. `git status --short` also shows this plan as the
  expected untracked/new plan; ignored `test_artifacts/debug` output is allowed.
  Any other path blocks execution pending plan revision.
- `rg -n "before_active_turn_context|during_active_turn_evidence|after_active_turn_evidence|unknown_timing_context" src tests development_plans` finds the canonical keys in target code, tests, and docs.
- Review `rg -n "response_action|should_respond" src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py src/kazusa_ai_chatbot/service.py`; matches remain validation, compatibility derivation, routing, or unchanged contract, with no new action rewrite.

### Deterministic tests

- `venv\Scripts\python -m pytest tests/test_persona_relevance_agent.py -q`
- `venv\Scripts\python -m pytest tests/test_relevance_turn_settlement.py tests/test_relevance_turn_settlement_graph.py tests/test_service_input_queue.py -q`
- `venv\Scripts\python -m pytest tests/test_frontline_relevance_agent.py -q`

### Real-LLM gates

Run each invocation separately with `-q -s`, inspecting raw trace and parsed
output before the next. The strict three-run H01 gate is intentional: this
named regression is a wrong-owner false `ignore` for an unambiguous direct
address, not harmless wording variance.

1. `venv\Scripts\python -m pytest -m live_llm tests/test_relevance_turn_settlement_live_llm.py::test_live_recreates_astrbot_request_no_response_gate -q -s`
2. Repeat command 1 twice as separate invocations; all three H01 decisions are
   `proceed`.
3. `venv\Scripts\python -m pytest -m live_llm tests/test_relevance_cross_channel_failure_live_llm.py::test_live_production_history_answer_makes_turn_redundant -q -s`
   returns `ignore` for a genuine after-turn answer.
4. `venv\Scripts\python -m pytest -m live_llm tests/test_relevance_cross_channel_failure_live_llm.py::test_live_interleaved_other_user_answer_makes_turn_redundant -q -s`
   preserves the during-turn answer boundary.
5. Run existing direct-address, recipient-switch, private, and group-noise
   relevance cases individually and inspect each output.

After each live case, parent records raw evidence and appends the inspected
decision to `test_artifacts/debug/relevance_pre_active_answer_bugfix_review.md`;
the final review includes run context, exact input identity, before/after
actions, prompt shape, quality judgment, validation, and raw paths. Test code
writes only raw/structured evidence.

### Database and service boundary

- No database migration or write is part of this plan.
- Service tests continue to show source-list temporal relations; only the
  private relevance payload groups them.
- `ignore` produces no cognition/RAG/dialog/delivery path, while validated H01
  `proceed` remains eligible for the existing claim path.

## Independent Plan Review

### Review record — 2026-07-23

- Reviewer: parent fresh-review posture; no native reviewer tool was available.
- Inputs: registry, prior completed relevance plan, RCA review, current source,
  service projection, tests, and ICDs.
- Fixed blockers: named exact module/integration tests and combined tracked plus
  untracked scope checks.
- Fixed non-blockers: chronological/during-turn semantics, strict H01 rationale,
  exact call caps, tracked/untracked scope wording, and per-case live review.
- Result: no unresolved plan blockers; status is now `in_progress` under the
  user's explicit execution instruction, and production source remains
  untouched.

## Independent Code Review

After verification, exactly one independent native code-review subagent reviews
the approved plan, diff, focused/regression output, live raw traces, debug
review, and ICDs. It checks style, CJK safety, prompt trust boundaries,
partition correctness, caps/calls, no action rewrite, regressions, scope, and
evidence. Parent fixes only in-scope findings, reruns affected gates, and
records reviewer identity, findings, fixes, reruns, residual risks, and status.

### Review record — 2026-07-23

- Reviewer: native independent agent `Confucius`, id
  `019f8be8-12e0-7db1-a805-8ccc4c9f0deb`.
- Initial result: `NEEDS-FIX for closeout`; no functional P0–P2 defects.
- Findings and disposition:
  - Process-gate finding that a native reviewer was unavailable: resolved as
    not applicable. The parent launched this reviewer through the native
    subagent interface, and this record identifies the completed reviewer.
  - P3 `_project_history` docstring underspecified: fixed by the same sole
    production implementation agent; compilation passed.
  - P3 live evidence omitted rendered history shape: fixed in the parent-owned
    live harness; rerun artifacts now include canonical keys and bounded counts.
  - During-turn live sensitivity found during remediation: fixed with a narrow
    prompt clarification; final raw run returned `ignore`.
  - Native-reply anchor sensitivity found during remediation: fixed with a
    narrow prompt clarification; final raw run returned `proceed` and anchor
    `true`.
- No reviewer finding required a service, database, graph, delivery, adapter,
  routing, retry, classifier, or action-rewrite change.
- Final focused re-review: `APPROVE`; the same reviewer found no residual
  in-scope issue. It confirmed the docstring invariants, rendered-history
  evidence, final during-turn and native-reply prompt clarifications, final
  live decisions, deterministic tests, caps/calls, unchanged service/graph
  ownership, and absence of action rewrites.
- Final disposition: approved for closeout.

## Acceptance Criteria

- H01 no longer false-ignores because of pre-active history and returns
  `proceed` in three separate inspected real-LLM runs.
- The genuine after-turn answer regression returns `ignore`.
- Before/unknown rows are not in candidate partitions; during/after rows remain
  LLM-judged redundancy evidence.
- Parsed relevance action remains authoritative; no deterministic conversion,
  retry, or replacement exists.
- Existing frontline/private/group/wait/native-reply/settlement/claim and
  cognition/RAG/dialog/delivery/persistence regressions pass.
- Route, call count, caps, truncation/drop behavior, and ownership docs remain
  unchanged except for the canonical partition description.
- Static checks, deterministic tests, individually inspected live traces,
  human-readable review, and independent code review are recorded before the
  plan leaves its draft execution lifecycle.
