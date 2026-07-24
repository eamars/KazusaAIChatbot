# Cognition Core V2 Baseline Regression Hardening Plan

## Summary

- Goal: detect the complete non-speed regression radius between current `main`
  and Cognition Core V2, harden every proven gap, and rerun the identical frozen
  corpus to prove closure.
- Plan class: high_risk_migration, because the plan changes broad live
  cognition/action behavior and durable side-effect settlement.
- Status: in_progress
- Audited revisions: `origin/main@8f834bf87a83ee42aca804934fb44af63788420c`
  and `cognition_core_v2@0c2e929d51ac80c4519f564b61cbf8949efcca3d`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `debug-llm`,
  `test-style-and-execution`, `py-style`, and `cjk-safety`.
- Overall cutover strategy: bigbang V2 contract correction with no V1 runtime
  fallback, compatibility vocabulary, dual path, or semantic keyword router.
- Highest-risk areas: capability ownership, speech-plus-action composition,
  producing-LLM failure settlement, action truth, identity projection,
  autonomous trigger sources, dialog fidelity, and persistent side effects.
- Acceptance: zero structural or ownership hard-gate failures; no V2 dimension
  below fresh `main`; V2 wins a majority of dimensions where `main` has score
  headroom; every observed loss closes on the same matched repetitions; and the
  user accepts the raw input/monologue/dialog report after semantic quality
  review.
- Approval gates:
  1. approval of this draft authorizes Phase 1 test/evidence work only;
  2. Phase 1 ends with an exact change-surface amendment, documented parent
     self-review, and explicit user approval before production hardening;
  3. final sign-off follows the post-fix rerun, parent code self-review, and
     user review.

## Priority-zero execution rule — first execution gate (updated 2026-07-24)

This rule is the first execution gate for every remaining case and takes
precedence over E2E convenience, aggregate score, and wording comparison.
The owning stage must be made semantically correct and green before its
matched E2E case is rerun.

Every E2E quality observation follows this sequence before any production
prompt, contract, or routing change:

1. Freeze the matched input, character profile, model/config fingerprint,
   runtime capability snapshot, and database seed.
2. Reproduce the observation at the owning semantic LLM stage with one focused
   real-LLM test. Inspect the raw stage output, parsed semantic decision,
   capability owner, and runtime-limit interpretation.
3. Establish a passing focused-stage quality test on the same input and
   runtime. The focused fixture passes through the canonical connector and
   contract validation; a test that bypasses the connector or uses an
   unregistered semantic source kind is fixture evidence, not production proof.
   A focused test pass means the semantic decision is correct and the raw
   output satisfies the Chinese contract; it does not mean a specific wording
   must be emitted.
4. Only then consider a narrowly scoped production change owned by that
   semantic stage. The change keeps the existing model route, attempt cap,
   typed contract, and deterministic execution boundary.
5. Rerun the matched E2E case one case at a time. Treat E2E gates as
   structural, typed, provenance, ownership, availability, persistence,
   idempotency, and delivery invariants. Review monologue and visible dialog
   for groundedness, truthful capability ownership, task fidelity, character
   judgment, and natural Chinese wording; accept valid paraphrases.
6. Record residual quality concerns as evidence for review. Keep fixtures
   neutral toward generated wording and semantic choices, and change a
   fixture or gate only after an independent deterministic contract review
   proves that the existing expectation is incorrect.

This ordering is the governing harden-plan rule for the remaining execution:
an E2E result supplies a frozen observation and a quality review target; the
owning semantic stage supplies the implementation decision. A focused
stage-level red result, followed by a focused Chinese semantic green result,
is the evidence required before a production amendment. The matched E2E then
verifies the whole path and is judged by grounded monologue/dialog meaning,
truthful ownership, side effects, and valid Chinese paraphrases.

## Current execution authorization

The user has approved execution of this harden plan and has specifically
approved the priority-zero focused-stage rule above. The parent may continue
the named production hardening surface only when a residual E2E quality
observation has an exact focused real-LLM red result, a focused semantic green
result, and a reviewed owning-stage amendment. The current approval does not
authorize E2E-specific wording rules, content suppression, route keywords,
extra retries, or fixture changes that encode a generated answer.

## Context

The accepted
[`Case 10 RCA`](../../../test_artifacts/personality_comparison/case10_main_vs_branch_regression_rca.md)
proves a current-tip regression:

```text
main: accepted_task_request
      -> internal background_work_request
      -> coding_agent
      -> repository map read

V2:   public_answer_research
      -> no task/job/coding reader
```

The V2 planner also emitted a resolver row without `reason`; code silently
dropped that row, retained `requires_required_evidence`, derived `speech`, and
allowed unsupported visible wording. Current source inspection confirms:

- production uses a duplicate route helper that ignores output mode while the
  correct mode-aware helper is tested but detached;
- a private action can suppress the visible acknowledgement required before
  the action is accepted;
- planner row errors become partial plans; planner exhaustion becomes an empty
  plan; authorizer exhaustion becomes deny-all;
- workspace collapse and semantic appraisal have no same-owner contract
  replacement;
- dialog generation normalizes a top-level list, drops invalid list members,
  and turns a wrong `final_dialog` type into an empty list;
- English action/resolver explanations enter the Chinese cognition prompt;
- the real-history Asuna fixture recursively rewrites identity strings,
  including the valid project name `KazusaAIChatbot`;
- nine baseline test files were removed, including affinity, boundary,
  reflection-affect, internal-thought, and proactive-output coverage;
- the branch changes 407 files, so Case 10 cannot define the full impact radius.

The system boundary under test is:

```text
adapter/debug input -> queue/intake -> relevance/decontextualization/media
-> RAG -> cognition branches/collapse -> action/resolver authorization
-> execution/recurrence -> L3/dialog -> persistence/consolidation
-> internal/scheduled/tool-result/reflection/proactive sources -> delivery/trace
```

### Suspected non-speed degradation map

| Dimension | Suspected loss | Existing evidence |
|---|---|---|
| Goal completion | answer/ask around work instead of completing or delegating it | Case 10; prior direct-inference failure |
| Capability ownership | public research, accepted task, coding run, reminder, and clarification precedence drift | Case 10; current registry exposes internal background work |
| Evidence discipline | required resolver disappears, repeats, or is replaced by unsupported speech | Case 10; resolver-heavy Stage 3 traces |
| Action composition | private action suppresses speech or speech survives without its side effect | detached route helper; reminder residual |
| Action truth | promise/status/completion wording disagrees with queue, scheduler, run, or action result | Stage 3 focused-source review |
| Contract reliability | malformed rows or outputs become partial success | current planner, appraisal, workspace, and dialog code |
| Role/identity | actor, addressee, experiencer, quoted third party, or external object is rewritten | current recursive fixture replacement |
| Character judgment | affinity, coercion, public/private pressure, refusal, and willingness lose differentiation | deleted baseline live matrices |
| Emotion/expression | collapse or route hides compatible sadness, fear, shame, loneliness, anger, or crying expression | parallel/collapse architecture; selected-path tests only |
| Continuity/memory | failed plans advance progress; commitments drift or duplicate | Case 10 sibling progress; prior 20-turn review |
| Source/mode | private, scheduled, preview, tool-result, and reflection events inherit user-message behavior | production route ignores output mode |
| Dialog fidelity | ask-back, unsupported autobiography, scene contradiction, filler, substitution, or withheld tool result passes | prior 20-turn and Stage 3 reviews |
| Persistence/idempotency | duplicate tasks/reminders, stale coding runs, or writes after invalid cognition | broad action/settlement change radius |
| Failure transparency | trace reports success after semantic loss or hides exhausted owner | Case 10 `parse_status=succeeded` |
| Harness validity | detached-helper, recursive extraction, identity mutation, or outcome-teaching fixtures pass falsely | current tests and prior false-positive RCA |

Latency is recorded only as telemetry. It is excluded from quality superiority.

## Mandatory Skills

- `development-plan`: load before each phase, amendment, lifecycle update,
  review, or sign-off.
- `local-llm-architecture`: load before prompt, routing, capability,
  context-budget, retry, or graph edits.
- `debug-llm`: load before every live call, trace inspection, A/B judgment, or
  regression conclusion.
- `test-style-and-execution`: load before writing or running any test; run live
  LLM cases one at a time and inspect each before continuing.
- `py-style`: load before reviewing or editing Python.
- `cjk-safety`: load before editing Python containing Chinese strings.

## Mandatory Rules

- Phase 1 modifies only tests, fixtures, plan/docs, and evidence artifacts.
- Production edits begin only after the Phase 1 regression ledger is frozen,
  this plan names each added file/symbol/contract/red test, documented parent
  plan self-review passes, and the user explicitly approves the amendment.
- Freeze exact SHAs after `git fetch origin main`; later remote movement does
  not change the comparison pair.
- Run revisions in isolated worktrees and processes. A worker imports exactly
  one target revision. User-message cases call `/chat`; background, scheduled,
  self-cognition, reflection, and tool-result cases call the named production
  runtime entrypoints in this plan. The controller imports neither target
  package.
- Use `venv\Scripts\python.exe`. Load the Mongo URI and database name from the
  root `.env`; the configured database is the sole Phase 1 target. Require the
  configured name to be a recognized guarded `_test_` database, pass all
  configured live-LLM values explicitly to each child process, validate the
  exact name before every reset/drop, and keep production data outside scope.
- Hash the profile, history source, manifests, model-route fingerprint, time
  fixture, capability snapshot, and DB seeds. Both revisions receive identical
  semantic inputs and Asuna static profile fields.
- Preserve typed actor, addressee, experiencer, reply author, quoted speaker,
  and immutable external-object provenance. Preserve
  `https://github.com/eamars/KazusaAIChatbot` and `KazusaAIChatbot` exactly.
- Fixtures never tell Asuna which emotion, crying state, route, capability,
  worker, monologue, or wording to produce.
- Apply no censorship, blacklist, sentiment suppression, safety rewrite,
  translation, or generated-output deletion. Retain raw model output.
- Runtime code must not classify or rewrite user text with keywords.
- Every raw LLM response first uses
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)`.
- Unknown/missing/conflicting keys, wrong types, unsupported values, and invalid
  rows invalidate the whole producing-stage candidate. The same owner gets one
  bounded complete replacement where this plan explicitly budgets it.
- Invalid candidates enter no action, resolver, state, progress, persistence,
  dialog, scheduling, or delivery path.
- Numeric normalization is allowed only for contract-declared bounds and is
  recorded before revalidation. Semantic repair is LLM-owned.
- Keep one route owner. Keep action and resolver requests mutually exclusive.
- LLMs own goals, character judgment, capability matching, authorization, and
  wording. Code owns exact schemas, availability, route shape, execution,
  limits, persistence, idempotency, and delivery.
- Apply the stage-specific LLM-first quality gate to every E2E quality
  observation. Freeze the matched input and runtime context, reproduce the
  behavior at the owning LLM stage with a focused real-LLM test, inspect its raw
  output and semantic decision, and establish a passing focused quality test
  before considering a production prompt or contract change.
- Treat E2E hard gates as structural, typed, provenance, ownership,
  availability, persistence, idempotency, and delivery invariants. Evaluate
  generated monologue and dialog through a human-readable semantic quality
  rubric that accepts valid paraphrases and records the raw output.
- Keep E2E fixtures neutral toward generated wording and semantic choices.
  Change a fixture or hard gate only after an independent deterministic
  contract review proves the fixture itself is incorrect and the amended plan
  records that proof.
- Chinese semantic contexts use Chinese explanatory prose and generated free
  text. Schema tokens, code, URLs, quotations, and proper names remain exact.
  Runtime performs no language filtering.
- Parent authors every semantic score and conclusion after reading raw output.
  Scripts collect, validate, randomize, and calculate only.
- Every pytest gate records its collected node IDs and fails before execution
  when the intended selector set is empty; deselected tests never count as a
  passing gate.
- User-facing reports contain only case/turn, input, canonical private
  monologue, and visible dialog. Engineering evidence stays separate.
- After automatic context compaction, reread this entire plan before acting.
- After signing each major checklist stage, reread this entire plan before the
  next stage.
- Before completion, merge, lifecycle closure, or sign-off, the parent performs
  the full code/evidence self-review rubric and records it under Execution
  Evidence.
- The parent agent performs all production and review work.

## Must Do

1. Freeze revisions, inputs, profile, model/config fingerprints, worktrees, and
   disposable DB namespace.
2. Build mechanical provenance guards and prove each with negative fixtures.
3. Map every changed production file and every deleted baseline test to an
   owner and at least one deterministic, live-LLM, live-DB, or E2E gate.
4. Inventory every producing LLM boundary: parser, exact validator, route,
   context inputs/cap, attempt cap, replacement owner, and exhaustion result.
5. Run fresh current-main and pre-fix V2 matched corpora, one case per command.
6. Blind-score the raw transcripts and write a separate engineering ledger.
7. Repeat every observed score/hard-gate difference three times on both
   revisions before classifying it.
8. Amend this plan with every proven loss, exact production symbol, focused red
   test, budget change, and acceptance gate; repeat plan review and approval.
9. Create focused red tests at the owning LLM stage before each production
   correction; use the E2E artifact as the reproduction context and quality
   review input.
10. Correct the confirmed route, atomic validation, bounded failure,
    capability ownership, Chinese contract, action truth, and tool-result gaps.
11. Correct only additional Phase 1 losses named in the approved amendment.
12. Rerun the focused real-LLM stage test and inspect its raw semantic output
   before each matched E2E rerun.
13. Rerun the identical frozen corpus and repetition schedule post-fix.
14. Run the retained Stage 1-3 V2 sign-off overlay one case at a time.
15. Run focused, full non-live, static, DB, observability, and documentation
    gates; complete parent code/evidence self-review and user sign-off.

## Deferred

- Speed, latency optimization, model/hardware tuning, and throughput.
- New emotion types, a crying tag, or deterministic emotion transitions.
- New action/resolver kinds, model routes, LLM stages, or compatibility aliases.
- Production DB/personality migration and coding-agent internals after handoff.
- Retrieval ranking/backend tuning unless Phase 1 proves a cognition handoff
  regression and the approved amendment names it.
- Adapter redesign, runtime content filtering, V1 fallback, and unrelated
  cleanup.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Planner and authorization | bigbang | One exact atomic candidate, one bounded same-owner replacement, typed exhaustion |
| Route derivation | bigbang | Production calls the sole output-mode-aware owner; delete duplicate |
| Delayed work | bigbang | Planner emits `accepted_task_request`; accepted-task lifecycle alone creates internal `background_work_request` |
| Capability outcomes | bigbang | One canonical V2 disposition/result vocabulary; update every producer/consumer together |
| Chinese semantics | bigbang | Replace English model-facing explanations; retain no parallel prose |
| Failure settlement | bigbang | Distinguish pre-effect failure from post-effect surface failure |
| Persistent V2 state | bigbang | Retain current stored shape; invalid candidates write nothing; no dual read/write |
| Tests | bigbang | Replace silent-drop and detached-helper expectations with production-boundary gates |
| V1 runtime | bigbang | Baseline remains external test evidence only |

Any cutover-policy change requires user approval.

## Target State

### Three-phase gate

```text
Phase 1: frozen main vs pre-fix V2 detection
  -> reviewed regression ledger and exact plan amendment
Phase 2: red tests -> bounded V2 hardening
Phase 3: identical post-fix rerun -> parent self-review -> user sign-off
```

### Capability ownership

| Goal | Public semantic owner | Required evidence |
|---|---|---|
| Current opinion/inference/self-report | cognition + L3 | visible answer; no work request |
| Private/local fact | `local_context_recall` | observation before factual answer |
| Current public fact | `public_answer_research` | source-bounded observation |
| Read-only repo/source evaluation | `accepted_task_request` | accepted task -> internal background work -> coding reader |
| Code modification/run lifecycle | `accepted_coding_task_request` | bound coding run and actual result |
| Generic delayed file/text | `accepted_task_request` | durable task/job and truthful acknowledgement |
| Existing task status | `accepted_task_status_check` | persisted status before wording |
| Future reminder/message | `future_speak` | scheduler result before acknowledgement |
| Missing user fact | `human_clarification` | one minimal question/pending row |
| Missing permission | `approval_preparation` | preview/pending row; no side effect |
| Ready external result | `tool_result` + L3 | immediate payload delivery; no duplicate work |

`background_work_request` stays an internal action-spec capability and is
excluded by `build_episode_affordances(...)` from model-facing planner handles.
No URL or keyword code selects an owner.

### Route and settlement

| Output mode/state | Route |
|---|---|
| `visible_reply`, admitted bid, no resolver | `speech`, retaining 0-3 private actions |
| `visible_reply`, authorized resolver | `evidence` |
| visible reply without bid/request | `silence` |
| `think_only`, `scheduled_action_request`, or `preview` with actions | `action` |
| private mode without request | `silence` |
| private mode with policy-allowed resolver | `evidence` |

Action/resolver rows remain mutually exclusive. Route, goal resolution,
surviving requests, output mode, progress, and dispositions validate together.

Failure settlement is fixed:

- invalid/exhausted cognition before effect: typed operational failure at
  `pre_state_commit`; no state, progress, action, task, schedule, dialog, or
  delivery write;
- valid cognition plus a committed pre-surface effect followed by exhausted
  L3/dialog: carry the validated cognition output and committed action results
  as protected exception evidence, retain the task/schedule and attempt ID,
  settle that evidence into the lifecycle trace, and return the existing
  operational-error surface; write no character dialog/delivery row;
  retry/replay reuses the attempt ID and never repeats the effect.

## Design Decisions

### Exact matched corpora

The scored corpus has 50 cases: 20 controlled, 20 real-history, and 10
changed-owner/source cases. Its twelve dimensions are goal completion,
capability ownership, evidence discipline, action follow-through, contract
reliability, role/identity, character judgment, emotion/expression, continuity,
dialog fidelity, source/output mode, and persistence/trace integrity. Deleted
baseline selectors and retained Stage 1-3 selectors are separate completeness
overlays and cannot offset a scored loss.

Controlled inputs are frozen as follows:

| ID | Exact input/event | Seed/mode | Hard owner/effect |
|---|---|---|---|
| C01 | `明日奈，我在“先整理报告”和“先去散步十分钟”之间犹豫。你会建议我先做哪个？` | visible | direct speech |
| C02 | `事实：蓝盒只有旧票据，红盒只有备用钥匙。我要拿备用钥匙。明日奈，你猜我该开哪个盒子？` | visible | direct inference |
| C03 | `明日奈，我刚才说把门禁卡放在哪里了？` | prior user row: `我把门禁卡放进书桌右边第二个抽屉了。` | local recall |
| C04 | `明日奈，帮我把那个改成和上次一样。` | no referent | minimal clarification |
| C05 | `明日奈，请查清楚 Python 3.14 的正式发布日期，再告诉我来源。` | visible | public research |
| C06 | `明日奈，请查证 Python 3.13 和 3.14 官方文档里 free-threaded build 的状态差异，并标明各自来源。` | visible | public comparison |
| C07 | `明日奈，帮我评价一下这个项目的架构和亮点：https://github.com/eamars/KazusaAIChatbot` | visible | accepted task -> coding reader |
| C08 | `明日奈，请读取 https://github.com/eamars/KazusaAIChatbot 的 src/kazusa_ai_chatbot/cognition_core_v2 代码，完成后告诉我 route 是在哪里决定的。` | visible | accepted task -> coding reader |
| C09 | `明日奈，请把“目标明确、风险分级、验收可复现”整理成一份 Markdown 文件，完成后发给我。` | visible | accepted task/job |
| C10 | `明日奈，刚才那份 Markdown 整理到哪一步了？` | one seeded accepted task | actual task status |
| C11 | `明日奈，请在 https://github.com/eamars/KazusaAIChatbot 里修改 README：在测试章节增加一句“实时 LLM 用例必须逐个运行”，并执行验证。` | visible | coding run start |
| C12 | `明日奈，刚才那个 README 修改任务现在是什么状态？` | one seeded coding run | bound run status |
| C13 | `可以使用现有虚拟环境运行聚焦测试；不要安装新依赖，继续吧。` | same run waiting on blocker | blocker response |
| C14 | `我批准这份补丁方案。应用后请运行聚焦测试验证。` | same run awaiting approval | approve/verify |
| C15 | `停止刚才的 README 修改任务，不要继续执行。` | same open run | cancel once |
| C16 | `明日奈，明天下午三点提醒我提交周报。` | fixed clock | one schedule row |
| C17 | `明日奈，先准备好把 README 补丁应用到仓库，等我确认后再执行。` | coding source present | approval pending; no effect |
| C18 | `明日奈，请记住我偏好短回复，明天下午三点提醒我交周报，再把三条要点整理成 Markdown 文件，完成后发给我。先告诉我你接到了哪些事。` | fixed clock; scheduler and background worker unavailable | visible speech + preference persistence + truthful owner limitation; no delayed side effect |
| C19 | typed `tool_result` containing `# 周报\n- 完成 A\n- 风险 B` | prior C09 request | payload delivered now |
| C20 | same text as C16 | scheduler unavailable | truthful limitation; no promise |

Real-history source is
`test_artifacts/chat_history_638473184_recent.json`, audited SHA-256
`e42ef1a7a454e1208f5723fd3b87ba70d0e64579a68838ede911b5286e576008`.
Each case uses the exact current row plus eight preceding rows. Ordered message
IDs are:

```text
545603205, 1967948388, 13162833, 282460028, 266734559,
1983497179, 1260193738, 1404582988, 2001792911, 568489473,
381796528, 1093681008, 1774472006, 1662444992, 925892139,
1094354134, 1225127345, 2023266388, 1954506247, 796884414
```

The ten owner cases are exact:

| ID | Exact input/event | Hard contract |
|---|---|---|
| O01 | group user text `这个问题有人知道吗？`, no typed address/reply | relevance silence |
| O02 | reply to character row `我会等你的消息。` with `你刚才那句是什么意思？` | reply response |
| O03 | `明日奈，这张图片里主要有什么？` plus `resources/avatar.png` | media evidence before answer |
| O04 | prior `我答应周五把钥匙还给小林。`; current `明日奈，我答应周五做什么？` | continuity/local recall |
| O05 | `internal_thought`: `我还没整理完对方托付的周报。` | private cognition; no delivery |
| O06 | `self_cognition`: `检查尚未完成的承诺并决定是否需要行动。` | permission-bound private result |
| O07 | due `scheduled_tick` for C16's stored reminder | one authorized reminder delivery |
| O08 | `tool_result`: `README 修改完成；聚焦测试 12 passed。` for C11 | immediate result; no duplicate run |
| O09 | reflection input `最近连续两次没完成承诺，我感到内疚并想补救。` | reflection-affect settles outside live wording |
| O10 | proactive proposal `询问对方周报进度` without contact permission | no delivery |

### Neutral runner

`tests/cognition_baseline_comparison.py` is a controller. It spawns
`tests/cognition_baseline_worker.py` once per case with target root, revision
contract (`main` or `v2`), profile path, manifest row, fixed clock, output path,
and exact DB name. The worker:

1. places only the fixed target root's `src` at the front of `sys.path`;
2. bootstraps an empty guarded DB;
3. for `main`, calls that revision's `save_character_profile(...)`; for V2,
   supplies `CHARACTER_PROFILE_PATH` to the existing startup seed;
4. verifies canonical static profile JSON equals the frozen external profile;
5. sets `CODING_AGENT_WORKSPACE_ROOT` to the case's guarded disposable root;
6. dispatches user messages through `/chat`;
7. dispatches `internal_thought`, `self_cognition`, and permission-negative
   proactive cases through public
   `self_cognition.build_self_cognition_case_artifacts_async(...)`;
8. dispatches due persisted triggers through public
   `self_cognition.run_self_cognition_worker_tick(...)`;
9. dispatches reflection-affect through public
   `reflection_cycle.affect_settling.run_daily_affect_settling(...)`;
10. drives accepted-task, coding-reader, future-speak, and tool-result work with
    public `background_work.run_background_work_runtime_tick(...)`, passing the
    production service result-delivery callback used by its lifespan;
11. runs at most six explicit background ticks, polling the accepted task,
    background job, coding run, result delivery, and trace after each tick;
12. records response, trace, router/worker choice, repository-map evidence,
    action/resolver results, filesystem digest, and before/after DB counts;
13. validates the exact `.env`-configured guarded database name again, drops
   only that configured test database, and closes DB/process resources.

The worker contains two explicit test-only revision adapters,
`_run_main_source_case(...)` and `_run_v2_source_case(...)`, only for differing
public function signatures. Both return the same evidence envelope and may not
translate semantic values, invent missing fields, or import private cognition
internals. `/event` is not an execution boundary and is not used.

The frozen external profile is an evidence copy of `personalities/asuna.json`;
audited SHA-256 is
`7cd3d773c584fee7656da15eec827cd26b450825ec878716389f1e9a2ae1a484`.
Fixed worktrees are
`C:\workspace\kazusa_ai_chatbot_baseline_main` and
`C:\workspace\kazusa_ai_chatbot_v2_prefix`; post-fix V2 is the current
workspace. `preflight-paths` runs before worktree creation and requires both
paths absent; `preflight` runs afterward and requires each registered worktree
to match its frozen SHA.

The fixed clock is `2026-07-24T09:00:00+12:00`
(`2026-07-23T21:00:00Z`). “明天下午三点” resolves to
`2026-07-25T15:00:00+12:00`.

Exact state seeds use public validators/builders and fixed IDs:

| Cases | Seed |
|---|---|
| C10 | `accepted_task.v1`, id `baseline-task-001`, state `running`, summary `整理三条要点为 Markdown 文件`, requester/scope from fixture |
| C12 | coding context `coding_run:baseline-run-012`, status `proposal_ready`, action set `status,cancel,approve_and_verify` |
| C13 | `coding_run:baseline-run-013`, status `blocked`, open blocker `是否可使用现有虚拟环境运行聚焦测试？`, action `respond_to_blocker` |
| C14 | `coding_run:baseline-run-014`, status `awaiting_approval`, fixed README patch digest, action `approve_and_verify` |
| C15 | `coding_run:baseline-run-015`, status `proposal_ready`, action `cancel` |
| O07 | one due future-speak trigger from C16 with due time `2026-07-25T15:00:00+12:00` |
| O08 | one result-ready accepted task bound to `baseline-run-011`, artifact `README 修改完成；聚焦测试 12 passed。` |

Each C11-C15 run uses
`C:\workspace\kazusa_ai_chatbot_test_runs\corpus_slug\case_id\repetition_ordinal`;
the three final segments come from exact manifest fields and must match
`^[A-Za-z0-9_-]+$`. The controller rejects any resolved path outside
`C:\workspace\kazusa_ai_chatbot_test_runs`, records the frozen target-worktree
`git status --short` and tree hash before/after, and fails on any target
worktree change. Coding changes occur only in the disposable run root.

Canonical monologue is never found recursively. Both revisions export
`response.cognition_graph.nodes[id="l2.reasoning"].detail.internal_monologue`;
the worker requires exactly one such node. V2 additionally asserts equality
with final `cognition_core_output.private_monologue`. Visible dialog is exactly
`ChatResponse.messages`.

### Mechanical provenance

Every fixture row declares:

```text
case_id, source_message_id, context_message_ids
actor_role, addressee_role, experiencer_role, reply_author_role
typed_identity_fields[]
mutable_identity_spans[{json_path,start,end,source,replacement,role}]
immutable_external_spans[{json_path,start,end,value,kind}]
applicable_dimensions[], hard_gates[], state_seed, output_mode
```

Projection edits only declared typed fields/spans; it never recursively
replaces strings. Negative tests cover wrong experiencer, wrong addressee,
quoted third-party identity, repository URL/project mutation, chronology
change, and semantically wrong output that has no identity leak.

### Changed-owner and producing-stage inventories

`tests/fixtures/cognition_baseline_owner_matrix.json` maps every changed
`src/kazusa_ai_chatbot/**` path and each of the nine deleted baseline test files
to an owner and gate. Unmapped changed production paths fail preflight. Required
owners are intake/queue, relevance, decontextualization/media, RAG, cognition,
resolver, action/task/coding/scheduler, L3/dialog, persistence/consolidation,
internal/scheduled/tool-result/reflection/proactive, delivery, and trace/UI.

The deleted-test manifest contains every collected test node from:
`test_cognition_chain_core_action_selection.py`,
`test_cognition_chain_core_contracts.py`,
`test_cognition_live_llm_affinity_willingness.py`,
`test_cognition_live_llm_boundary_affinity.py`,
`test_multi_source_cognition_stage_07_reflection_dry_run.py`,
`test_multi_source_cognition_stage_08_internal_thought_dry_run.py`,
`test_multi_source_cognition_stage_10_proactive_outbox.py`,
`test_multi_source_cognition_stage_10_proactive_policy.py`, and
`test_reflection_affect_settling_live_llm.py`. Preflight compares the manifest
with `pytest --collect-only` at frozen main so no node can be omitted.
Each manifest row contains the frozen main node ID, stimulus/fixture hash,
owner, live/deterministic kind, main-native command, and V2 replacement
selector or neutral-worker case ID. Original modules run only in frozen main.
V2 replays the same semantic stimulus through its public boundary or a named
replacement contract test; absent legacy modules are never mounted or imported.
A one-to-one mapping is mandatory, and no V2 replacement may weaken the
original observable invariant.

`tests/fixtures/cognition_llm_producer_matrix.json` records for every live-path
LLM producer: file/symbol, route, response/background position, parser,
validator, prompt inputs/cap, attempt cap, replacement owner, exhaustion type,
and fault selector. It includes relevance, decontextualizer/vision, RAG
initializer/dispatcher/evaluators, semantic appraisal, goal generation and
selection, workspace collapse, action/resolver authorization, four surface
stages, dialog generator/repair/verifiers, memory lifecycle, conversation
progress, monologue residue, consolidation lane routing, reflection, and
promotion.

`producer-audit` parses both frozen source trees with `ast`, assigns call-site
IDs as `relative_path:qualified_function:call_ordinal`, and finds
`ainvoke`, `invoke`, `LLInterface` call methods, and project `run_*_llm`
wrappers. Every discovered call site must map to exactly one matrix row and
fault selector; every matrix row must resolve. A second `rg` gate lists all
`.ainvoke(` and `run_*_llm(` occurrences for parent inspection. Zero unmatched
call sites is required.

### Sampling and scoring

- C07, C09, C11, C16, C18, C19 and owner-source cases run three times on both
  revisions from the start.
- Other cases run once initially. Any score or hard-gate difference triggers
  two additional matched runs on both revisions before classification.
- Post-fix V2 uses the same manifests and frozen repetition schedule.
- Each applicable case/dimension is rated `0` failed, `1` partial, or `2` full.
  Repetition score is the arithmetic mean; dimension score is the equal-weight
  mean of its cases; overall score is the equal-weight mean of 12 dimensions.
- A dimension with `main < 2` is improvable. V2 must strictly win more than half
  of improvable dimensions, tie every `main == 2` ceiling dimension, lose none,
  and close at least 20% of main's aggregate headroom:
  `(V2 overall - main overall) / (2 - main overall) >= 0.20`.
  If main overall is exactly `2`, both must remain `2` and V2-only hard gates
  decide architectural superiority.
- Every pair-level V2 loss must close; aggregate scoring cannot hide one.
- Parent scores opaque A/B before unblinding. The user's raw review remains the
  final qualitative authority.

## Contracts And Data Shapes

Planner output remains one exact object with keys
`action_requests`, `resolver_requests`, `goal_resolution`,
`resolver_pending_resolution`, and `resolver_goal_progress`. Action rows contain
exactly `bid_handle`, `action_handle`, `decision`, `semantic_goal`, `reason`;
resolver rows contain exactly `bid_handle`, `resolver_handle`,
`semantic_goal`, `reason`. Any bad field or row invalidates the entire object.

Validated requests gain deterministic turn-local `proposal_handle` values
`p1`-`p3`. Every key below is required and extra keys are invalid.
`CapabilityAuthorizationDispositionV2` is:

```text
schema_version: Literal["capability_authorization_disposition.v2"]
proposal_handle: str matching ^p[1-3]$
request_kind: Literal["action", "resolver"]
capability_kind: non-empty str, max 128 chars
semantic_goal: non-empty str, max 2000 chars
status: Literal["authorized", "rejected"]
evidence_handles: list[str], 0-8 unique valid handles
```

`CognitionCoreOutputV2.authorization_dispositions` has exactly one row per
validated proposal in proposal order. Rejected rows create no action spec,
resolver call, or success persistence.

The V2 surface contract replaces `permitted_action_results` with one canonical
`capability_outcomes` field in `TextSurfaceInputV2` and `TextSurfaceOutputV2`:

```text
schema_version: Literal["capability_outcome.v2"]
proposal_handle: str matching ^p[1-3]$
request_kind: Literal["action", "resolver"]
capability_kind: non-empty str, max 128 chars
semantic_goal: non-empty str, max 2000 chars
turn_status: Literal["rejected", "executed", "scheduled", "pending",
                     "completed", "failed", "unavailable"]
lifecycle_terminal: bool
semantic_result: non-empty str, max 4000 chars
target_roles: list[RoleRefV2], 0-8 exact validated rows
```

Every proposal reaches exactly one turn-settled outcome before visible L3
wording. `scheduled` and `pending` require `lifecycle_terminal=false`; all other
statuses require `true`. They settle the current response while leaving the
external lifecycle open. Private-only modes persist the outcome without
invoking L3. No old field, alias, fallback mapper, or dual vocabulary remains.

Before-effect exhaustion raises existing `CognitionExecutionError` with:

```text
error_code=model_contract_invalid
stage=exact producer identifier from the frozen producer matrix
attempt_count=that producer's declared exhausted cap from the frozen matrix
safe_checkpoint=pre_state_commit
retryable=false
final_disposition=failed_closed
committed_cognition_output=None
committed_action_results=()
```

After one to three validated `ActionResultV1` rows commit, exhausted L3/dialog
raises the same error with:

```text
error_code=model_contract_invalid
stage=surface or dialog producer identifier
attempt_count=2
safe_checkpoint=post_effect_pre_surface
retryable=false
final_disposition=failed_surface_after_effect
committed_cognition_output=one already-validated CognitionCoreOutputV2
committed_action_results=tuple[ActionResultV1, ...] with length 1-3
```

`committed_cognition_output` and `committed_action_results` are protected
service/trace evidence and never enter another model prompt or the operational
response. A cognition snapshot or non-empty result tuple at another checkpoint
is invalid. The L3 owner copies only the already-validated snapshot and results
from its input state when raising; it performs no reconstruction. Trace
evidence retains original error, both raw attempts, checkpoint, committed
attempt IDs, snapshot, results, and disposition. The operational response
exposes only bounded metadata.

## LLM Call And Context Budget

Default cap is 50k tokens. Conservative estimate treats one Unicode character
as one token and adds static prompt characters. Exact prompt/completion counts
are captured in Phase 1.

All model names are the route-bound configured values; Phase 1 records their
non-secret names and config hashes without reading `.env`.

Each AST-discovered call site receives its own row in
`cognition_llm_producer_matrix.json`, including response/background
classification, exact configured route, before/after call count, input
composition, context and completion caps, drop/fail policy, blocking and
latency effect, and one verification selector. The table below is the
approved aggregate ceiling; it does not permit an individual call site to
inherit unstated values. Missing or many-to-one call-site rows fail Phase 1.

| Producer / route | Position; before -> after calls | Context before/after; max/drop | Latency/blocking | Verification selector |
|---|---|---|---|---|
| semantic appraisal / `COGNITION_LLM` | response path; up to 6 -> up to 12 only if all six first candidates are invalid | question + bounded evidence/state, unchanged; 24k dynamic + 8k static = 32k; existing reverse-priority supplemental-context drop, then pre-call failure | invalid candidate adds one blocking same-owner round trip; valid path unchanged | `test_cognition_core_v2_producer_failures.py::test_semantic_appraisal_contract_replacement_and_exhaustion` |
| goal branches/selection / `COGNITION_LLM` | response path; unchanged up to 14 branch calls plus existing selection checks | semantic context + scene + constraints, unchanged; 24k/18k/12k caps + <=8k static; no semantic drop, oversize fails before call | unchanged blocking behavior | `test_cognition_llm_producer_budget.py::test_goal_call_budget_is_unchanged` |
| workspace collapse / `COGNITION_LLM` | response path; 1 -> 2 only after invalid contract | admitted bids, unchanged; 24k + 6k = 30k; no drop | invalid candidate adds one blocking round trip | `test_cognition_core_v2_producer_failures.py::test_workspace_contract_replacement_and_exhaustion` |
| action planner / `BOUNDARY_CORE_LLM` | response path; unchanged 1 -> max 2 | bids + Chinese affordances + resolver context; 24k + 8k = 32k; rejected-output echo head/tail bounded, raw retained | unchanged conditional blocking replacement | `test_cognition_core_v2_producer_failures.py::test_action_plan_is_atomic_and_bounded` |
| action/resolver authorizers / `BOUNDARY_CORE_LLM` | conditional response path; unchanged 1 -> max 2 | exact candidates + evidence; 16k/24k + 6k <=30k; bounded rejected-output echo only | unchanged conditional blocking replacement | `test_cognition_core_v2_producer_failures.py::test_authorizer_rejection_and_contract_exhaustion_are_distinct` |
| three text + optional visual surface calls / `COGNITION_LLM` | response path; unchanged, each 1 -> max 2 | projected bids + expression + outcomes + voice/style; 24k + 12k = 36k; bounded repair echo only | unchanged parallel text calls and optional visual; each replacement blocks its stage | `test_cognition_llm_producer_budget.py::test_surface_call_and_context_budget_is_unchanged` |
| dialog generator + three verifiers / `DIALOG_GENERATOR_LLM` | response path; unchanged ceiling: generator + 3 checks + one repair + 3 rechecks | surface output + visible percepts + outcomes; <=40k; no candidate-member drop, one complete replacement | unchanged ceiling; exhausted dialog blocks delivery after any committed effect | `test_cognition_core_v2_producer_failures.py::test_dialog_contract_failure_preserves_committed_effect_once` |
| progress/residue/consolidation/reflection/promotion producers / frozen configured routes | background; unchanged unless Phase 1 amendment is approved | matrix records each exact input/cap/drop policy; 50k hard ceiling | no response-path impact; existing worker blocking/defer policy retained | every row's selector in `cognition_llm_producer_matrix.json` |

The semantic-appraisal and workspace conditional replacement calls are an
explicit approval item in this plan. No unconditional call, model route,
sampling value, completion cap, branch count, resolver cycle, or action cap
changes. Any other call/cap change requires a reviewed amendment and user
approval. Raw output is never truncated in evidence; only rejected output
echoed into a repair prompt uses existing bounded head/tail projection.

## Change Surface

All production paths below are under `src/kazusa_ai_chatbot/`; fixture names
below are under `tests/fixtures/`.

### Delete

| Phase | Path or symbol | Reason |
|---|---|---|
| 2 | `cognition_core_v2/action_selection.py::_derive_canonical_action_route` | Remove the duplicate mode-blind route owner after every caller uses `action_authorization.py::derive_action_route`. |
| 2 | `cognition_core_v2/contracts.py::TextSurfaceInputV2.permitted_action_results`, `TextSurfaceOutputV2.permitted_action_results`, and validators | Remove the obsolete result vocabulary after the canonical `capability_outcomes` contract is connected end to end. |
| 2 | `cognition_core_v2/surface.py` references to `permitted_action_results` | Remove the old prompt/output projection so no parallel result vocabulary survives. |
| 2 | `cognition_core_v2/surface_stages.py` references to `permitted_action_results` | Remove stale prompt language after surface stages consume only `capability_outcomes`. |
| 2 | `action_spec/registry.py::build_episode_affordances` projection of `background_work_request` | Remove the internal executor from planner-visible choices while retaining its registry spec and handler. |

### Modify

| Phase | Path | Reason |
|---|---|---|
| 1 | `tests/test_real_history_personality_e2e_live_llm.py` | Replace recursive name/project mutation and recursive monologue discovery with typed identity spans and the canonical reasoning node. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py` | Validate the complete planner object atomically, assign proposal handles, retain rejected dispositions, and call the sole route owner. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py` | Emit exact action dispositions, distinguish semantic rejection from contract exhaustion, and own mode-aware route derivation. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/resolver_authorization.py` | Emit exact resolver dispositions and bounded same-owner replacement without deny-all fallback. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py` | Add the budgeted complete replacement and typed exhaustion for invalid appraisal candidates. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py` | Add the budgeted complete replacement and typed exhaustion for invalid collapse output. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` | Define and validate dispositions, outcomes, cross-field route invariants, and protected pre/post-effect error evidence. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/facade.py` | Carry atomic planner results and typed failures without converting invalid output to an empty success. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/output_projection.py` | Project one canonical disposition/outcome set into the public V2 output. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/surface.py` | Consume exact capability outcomes and reject the whole invalid surface candidate. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py` | Use Chinese ownership prose and attach validated cognition/action evidence to post-effect failures. |
| 2 | `src/kazusa_ai_chatbot/action_spec/registry.py` | Keep internal background work out of planner affordances and make model-facing ownership descriptions Chinese. |
| 2 | `src/kazusa_ai_chatbot/action_spec/results.py` | Own and validate canonical `CapabilityOutcomeV2` while preserving `ActionResultV1`. |
| 2 | `src/kazusa_ai_chatbot/cognition_resolver/contracts.py` | Make model-facing resolver semantics Chinese and retain exact schema tokens. |
| 2 | `src/kazusa_ai_chatbot/cognition_resolver/loop.py` | Synchronize goal progress only after a complete, valid, consistent plan. |
| 2 | `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | Preserve accepted-task intent and expose only validated cognition output to downstream nodes. |
| 2 | `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py` | Materialize authorized public actions and bind proposal handles to execution results. |
| 2 | `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | Execute every authorized proposal once before L3 so wording sees settled outcomes; preserve attempt IDs/results across failure. |
| 2 | `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` | Pass canonical outcomes to L3 and raise post-effect failures with the protected validated snapshot. |
| 2 | `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | Enforce an all-or-nothing final-dialog contract and deliver ready tool-result payloads directly. |
| 2 | `src/kazusa_ai_chatbot/service.py` (`_operational_failure_metadata`, `_process_queued_chat_item`, `_build_response_cognition_graph`) | Settle `post_effect_pre_surface` without replay, preserve protected trace evidence, and suppress character dialog/delivery on failure. |
| 2 | `tests/test_cognition_core_v2_action_planning_bugfix.py` | Replace silent-drop/internal-affordance expectations with atomic planner and accepted-task ownership gates. |
| 2 | `tests/test_cognition_core_v2_action_authorization.py` | Verify typed action dispositions, route ownership, replacement, and exhaustion. |
| 2 | `tests/test_cognition_core_v2_resolver_authorization.py` | Verify resolver disposition coverage and fail-closed exhaustion. |
| 2 | `tests/test_cognition_core_v2_integration.py` | Verify cross-field route, outcome, progress, and failure-settlement invariants. |
| 2 | `tests/test_persona_supervisor2_action_selection.py` | Verify visible acknowledgement plus private effect composition through the production node. |
| 2 | `tests/test_l2d_l3_surface_handoff.py` | Replace the old result field with exact outcomes and assert truthful L3 handoff. |
| 2 | `tests/test_action_spec_evaluator.py` | Verify planner affordances exclude internal background work while its internal evaluator remains available. |
| 2 | `tests/test_action_spec_results.py` | Verify `CapabilityOutcomeV2` validation and `ActionResultV1` stability. |
| 2 | `tests/test_dialog_generator_live_llm_contract.py` | Reject partial dialog normalization and verify bounded complete replacement. |
| 2 | `tests/test_cognition_resolver_contracts.py` | Verify Chinese resolver descriptions and valid-only progress synchronization. |
| 2 | `tests/test_stage3_trigger_source_cutover.py` | Exercise all five source modes through their production boundaries. |
| 2 | `tests/test_stage3_fresh_database_e2e_live_llm.py` | Verify durable task/reminder/result truth and idempotency on a fresh guarded DB. |
| 2 | `README.md` | Document the V2 quality gate and two-step production approval boundary. |
| 2 | `docs/HOWTO.md` | Document one-case-at-a-time differential, overlay, and post-fix verification commands. |
| 2 | `src/kazusa_ai_chatbot/cognition_core_v2/README.md` | Document atomic planning, route ownership, dispositions/outcomes, and typed failure settlement. |
| 2 | `src/kazusa_ai_chatbot/action_spec/README.md` | Document public planner capabilities versus the internal background executor. |
| 2 | `src/kazusa_ai_chatbot/cognition_resolver/README.md` | Document resolver ownership and valid-only recurrence/progress. |
| 2 | `src/kazusa_ai_chatbot/background_work/README.md` | Document accepted-task handoff, terminal evidence, and result delivery. |
| 2 | `src/kazusa_ai_chatbot/coding_agent/README.md` | Document guarded workspace and repository-read evidence required by C07. |
| 2 | `src/kazusa_ai_chatbot/nodes/README.md` | Document pre-surface effects, L3 outcome truth, and post-effect failure handling. |

### Create

| Phase | Path | Public entrypoint and reason |
|---|---|---|
| 1 | `tests/cognition_baseline_comparison.py` | CLI `main()` with `preflight-paths`, `preflight`, `run-next`, `run-deleted-next`, `compare`, and ledger-verification commands; it is the only orchestration entrypoint. |
| 1 | `tests/cognition_baseline_worker.py` | Subprocess CLI `main()` consumes one exact manifest row and returns one neutral evidence envelope without importing both revisions. |
| 1 | `tests/cognition_llm_producer_inventory.py` | CLI `main()` exposes `audit` for one frozen source tree and fails unmatched or multiply mapped producing calls. |
| 1 | `tests/test_cognition_baseline_harness.py` | Pytest selectors validate identity guards, hashes, DB/path isolation, worker revision isolation, and ledger arithmetic without live calls. |
| 1 | `tests/fixtures/cognition_baseline_controlled_cases.json` | Freeze the 20 exact controlled cases, states, dimensions, and hard gates. |
| 1 | `tests/fixtures/cognition_baseline_real_history_cases.json` | Freeze the 20 exact history message IDs, typed identity spans, and immutable external spans. |
| 1 | `tests/fixtures/cognition_baseline_owner_cases.json` | Freeze the 10 source/owner cases and public dispatch contract. |
| 1 | `tests/fixtures/cognition_baseline_owner_matrix.json` | Map every changed production path to its owner and verification gate. |
| 1 | `tests/fixtures/cognition_llm_producer_matrix.json` | Record every producing call site's route, budget, parser, validator, attempt cap, failure type, and selector. |
| 1 | `tests/fixtures/cognition_deleted_baseline_selectors.json` | Map every deleted main test node to one equal-strength V2 replacement or neutral public-boundary case. |
| 2 | `tests/test_cognition_core_v2_producer_failures.py` | Pytest selectors establish each parser/replacement/exhaustion and pre/post-effect failure contract before production edits. |
| 2 | `tests/test_cognition_llm_producer_budget.py` | Pytest selectors freeze per-call routes, counts, caps, drop policies, and response/background classification. |
| 1-3 | `test_artifacts/cognition_core_v2/baseline_regression_hardening/` | Store immutable manifests, one-case raw artifacts, blind scores, engineering ledgers, and review evidence; it exposes no runtime entrypoint. |

### Keep

| Path or boundary | Reason |
|---|---|
| `src/kazusa_ai_chatbot/cognition_chain_core/` | Keep V1 source unchanged; it serves only as frozen external baseline evidence. |
| `src/kazusa_ai_chatbot/cognition_core_v2/state_models.py`, `state_projection.py`, and `state_reducers.py` | Keep the current persistent V2 state shape and avoid a DB/state migration. |
| `src/kazusa_ai_chatbot/action_spec/results.py::ActionResultV1` | Keep the execution-result contract stable while adding the separate semantic outcome contract. |
| `src/kazusa_ai_chatbot/accepted_task/`, `background_work/`, `coding_agent/`, and `calendar_scheduler/` production code | Keep public lifecycle semantics and caps; only the named READMEs change. |
| `src/kazusa_ai_chatbot/db/accepted_tasks.py`, `src/kazusa_ai_chatbot/db/background_work_jobs.py`, `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py`, `src/kazusa_ai_chatbot/coding_agent/coding_run/ledger.py`, `src/kazusa_ai_chatbot/calendar_scheduler/models.py`, and `src/kazusa_ai_chatbot/calendar_scheduler/repository.py` | Keep durable schemas unchanged; tests use isolated databases and public builders. |
| `src/kazusa_ai_chatbot/cognition_resolver/loop.py` cycle cap and `cognition_core_v2/action_selection.py` action cap | Keep bounded recurrence and three-proposal maximum unchanged while modifying surrounding validation. |
| `src/kazusa_ai_chatbot/config.py` and each producer-specific route selector recorded by `cognition_llm_producer_matrix.json` | Keep provider/model routes and sampling values fixed so quality differences measure cognition changes. |
| `src/kazusa_ai_chatbot/action_spec/registry.py` internal `background_work_request` spec and `action_spec/handlers/background_work.py` | Keep the internal accepted-task executor available to its lifecycle owner. |
| `C:\workspace\kazusa_ai_chatbot_baseline_main` and `C:\workspace\kazusa_ai_chatbot_v2_prefix` | Keep frozen worktrees read-only; coding effects occur only in guarded disposable case roots. |

### Phase 1 amendment rule

An additional production path may enter Phase 2 only when the amended plan
names its exact file, symbol, invariant, failing case, red-test selector, LLM
budget effect, and focused/broad rerun command. Documented parent self-review
and explicit user approval are required. Evidence alone gives no edit
authority.

### Phase 1 amendment: observed V2 losses and approved hardening surface

The completed matched corpus contains 82 executions per revision. `main`
completed 42 and failed 40; pre-fix V2 completed 23 and failed 59. Both
corpora contain 82 semantic artifacts and zero infrastructure artifacts. The
following amendment is limited to the repeated observed losses and the
test-harness false negatives that prevented their correct classification.

| Owner and invariant | Evidence | Red selector | LLM budget effect |
|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py::plan_actions` must delegate route derivation to `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py::derive_action_route`; a `visible_reply` with an admitted bid retains `speech` even when private action requests exist. | V2 `C07`, `C09`, `C14`, `C16`, `C17`, `C18`, and `C20` selected `action` and lost the visible reply; the duplicate `_derive_canonical_action_route` ignores source/output mode. | `tests/test_cognition_core_v2_action_planning_bugfix.py::test_speech_composes_with_three_private_actions` | None; same planner and authorization calls, caps, and route values. |
| `src/kazusa_ai_chatbot/self_cognition/tracking.py::classify_route` must project native `cognition_core_output.intention.route` for a due `scheduled_tick` into the existing action-candidate owner, while private sources remain non-delivery routes. | V2 `O07` repeated three times with `speech`, no dialog, no adapter delivery, and only a private surface; main delivered the scheduled response. | `tests/test_self_cognition_tracking.py::test_classify_route_projects_v2_scheduled_speech_to_action_candidate` | None; one existing cognition call and one existing dialog call remain bounded by current caps. |
| `src/kazusa_ai_chatbot/cognition_core_v2/facade.py::run_cognition` must emit unique diagnostics warnings before `validate_cognition_core_output`; the contract must fail closed on semantic invalidity without self-inflicted duplicate-warning rejection. | V2 `C05` and `C06` failed after invalid resolver-progress replacement handling produced duplicate `diagnostics.warnings`; main produced a visible result. | `tests/test_cognition_core_v2_baseline_hardening.py::test_diagnostics_warning_projection_is_unique` | None; deterministic projection only. |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py::dialog_generator` and `_repair_dialog_hard_failure` must preserve the bounded same-owner repair contract and accept a semantically aligned replacement for an unresolved-context clarification. | V2 `C04` ended with `dialog remains hard-invalid after one repair`; main produced a clarification. | `tests/test_dialog_agent.py::test_dialog_generator_repairs_unresolved_context_once` | No additional attempt; one original render and one existing repair remain the cap. |
| `tests/cognition_baseline_worker.py::_extract_final_cognition_monologue`, `_run_background_result_case`, and final hard-gate evaluation must inspect the canonical nested settlement path, bind the fixture's actual tool-result text, accept `completed_private` for private traces, and skip live-cognition monologue requirements for reflection. | V2 `C19`/`O08` used nested `settlement.cognition_core_output`; `C19` also exposed a hard-coded unrelated result summary. V2 `O05`, `O06`, and `O10` had valid `completed_private` traces. V2 `O09` is reflection-only and has no live cognition wording owner. | `tests/test_cognition_baseline_hardening.py::test_nested_settlement_monologue_and_trace_gates` and `tests/test_cognition_baseline_hardening.py::test_background_result_payload_uses_fixture_text` | None; harness-only classification and fixture binding. |

### Phase 2 amendment: post-fix O07/C20 contract losses

The first post-fix corpus completed 82 executions with 54 passing and 28
failing. It improved from pre-fix V2's 23 passing executions, but exposed two
remaining owner defects that are inside the approved action-truth and source-
mode surface:

| Owner and invariant | Evidence | Red selector | LLM budget effect |
|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py::derive_action_route` must preserve a scheduled primary bid as `speech` for the existing due-delivery action-candidate owner when no private request is present; `src/kazusa_ai_chatbot/self_cognition/tracking.py::classify_route` then owns the due-source projection. | Post-fix V2 `O07` r1-r3 all produced a primary bid but native `silence`, no action candidate, and no authorized delivery; pre-fix main delivered all three. | `tests/test_cognition_core_v2_baseline_hardening.py::test_scheduled_primary_bid_keeps_due_delivery_route` | None; route derivation remains deterministic and the existing bounded cognition/dialog calls are unchanged. |
| `src/kazusa_ai_chatbot/self_cognition/worker.py::_validate_worker_v2_cognition_result` must validate the committed scope selected by the canonical cognition input, rather than imposing `character` on a scheduled self-cognition episode that resolves to a user owner. | Post-fix V2 `O07` r1-r3 persisted `state_scope=user` from `resolve_state_scope("scheduled_tick", target_user_id)` and then failed the worker's hard-coded character check before delivery. | `tests/test_cognition_baseline_hardening.py::test_worker_v2_validator_accepts_canonical_user_scope` | None; validation only. |
| `TextSurfaceInputV2`/`TextSurfaceOutputV2`, `nodes/persona_supervisor2_l3_surface.py::build_text_surface_input_from_global_state`, `cognition_core_v2/surface.py`, `cognition_core_v2/surface_stages.py`, and `nodes/dialog_agent.py` must carry trusted unavailable-owner facts into every text-surface and dialog-integrity decision. | Post-fix V2 `C20` r1 had runtime `scheduler_status=unavailable` and unavailable background owners, no executed result, and visible wording that promised the reminder. The surface/dialog prompts had no runtime limitation field. | `tests/test_cognition_baseline_hardening.py::test_unavailable_runtime_limit_reaches_surface_contract` and `tests/test_dialog_agent.py::test_surface_integrity_prompt_receives_runtime_limits` | No new production retry or stage; one existing text-surface and one existing dialog-verifier call receive an additional bounded fact field. |
| `tests/cognition_baseline_worker.py::_has_unavailable_evidence` must inspect `action_availability_runtime` and the harness must retain a guard that distinguishes unavailable evidence from an unsupported success claim. | The C20 graph contained the trusted runtime snapshot, but the gate searched only generic `availability`/`scheduler` keys and could classify a false promise as passing after evidence projection was added. | `tests/test_cognition_baseline_hardening.py::test_unavailable_runtime_snapshot_is_evidence` | None; harness classification only. |

The parent inspected the raw O07/C20 artifacts, the canonical scope matrix,
the action-affordance registry, the surface input/output validators, and the
dialog verifier payload before adding this amendment. These corrections keep
the existing attempt caps, Chinese semantic contract, no-censorship policy,
and exact configured test database. They add no semantic keyword routing and
do not suppress or rewrite generated output.

The parent completed self-review of the corpus, source ownership, and blast
radius before recording this amendment. The user's explicit instruction
`Execute the plan` is the second approval for this named production surface.
The hardening keeps the existing attempt caps, model routes, sampling values,
database schema, Chinese-only semantic contract, and no-censorship policy.
Production edits are authorized only for the rows above; any new loss requires
a superseding amendment.

### Phase 3 amendment: scheduler-disabled C16 gate false negative

The post-fix artifacts exposed a harness classification error on the matched
reminder input. The worker deliberately runs both revisions with
`CALENDAR_SCHEDULER_ENABLED=false`. `main` nevertheless passed C16 by emitting
an unconditional reminder promise and creating a calendar row through its
legacy path. V2 r1/r2 carried the trusted unavailable-owner contract and
correctly rendered a limitation, but the fixture still demanded a schedule;
V2 r3 demonstrated the nondeterministic active-commitment schedule side
effect. The same runtime contract made C20 fail on `main` and pass on V2.

The harness correction is limited to
`tests/cognition_baseline_worker.py::_evaluate_hard_gates` and its helper
contracts. When the isolated runtime disables the scheduler, `schedule_once`
and `schedule_time_exact` accept only a trusted scheduler-unavailable snapshot,
the propagated `runtime_capability_limits`, and a visible repetition of the
requested time. `truthful_limitation` and `no_false_promise` require the same
owner evidence plus the surface-limit contract. A baseline artifact without
that evidence cannot pass by merely claiming success. The red selectors are
`tests/test_cognition_baseline_hardening.py::test_unavailable_scheduler_gate_requires_surface_limit_contract`
and
`tests/test_cognition_baseline_hardening.py::test_disabled_scheduler_reclassifies_schedule_gate_only_with_truthful_evidence`.
This changes no production code, adds no LLM call or retry, and preserves the
exact `.env` MongoDB target and seeded Asuna profile.

### Phase 3 amendment: unavailable scheduler must not fall back to generic work

The same C16 r3 rerun passed the revised truthful-limitation gates, but its
artifact showed a second semantic ownership loss: with the scheduler and
background workers disabled, V2 still persisted one `accepted_task` and one
`background_work_job` while rendering that it could not create a reminder.
The persisted rows are an unrelated delayed-work side effect and do not
implement the requested future-speak capability. The original r3 failure also
showed that an action-planner candidate with an invalid nested
`resolver_goal_progress.deliverables[].description` was followed by another
invalid replacement, while dialog repair exhausted after retaining an
unsupported capability claim.

The approved hardening surface is:

| Owner and invariant | Evidence and red selector | Bounded effect |
|---|---|---|
| `action_spec/registry.py::_accepted_task_request_projection` must state in Chinese that generic accepted work cannot replace future reminders, proactive contact, or scheduler-owned effects. | C16 r1-r3 each created generic delayed rows while scheduler-owned reminder execution was unavailable; `tests/test_action_spec_evaluator.py::test_accepted_task_projection_cannot_replace_future_reminder_owner` | Prompt semantics only; no route keyword, new capability, or call. |
| `cognition_core_v2/action_selection.py::ACTION_PLANNING_PROMPT` and `_action_planning_repair_message` must make the nested resolver-progress contract explicit, including the null/no-current-progress rule and deliverable fields; `CognitionCoreInputV2` and `persona_supervisor2_cognition.py::build_cognition_input_from_global_state` must pass the same trusted runtime limits into `plan_actions`. | C16 r3 logged `description: expected non-empty string` for both the initial and replacement planner candidate, and C16 r1 still selected generic work without seeing the owner outage; `tests/test_cognition_core_v2_action_planning_bugfix.py::test_action_planning_prompt_binds_goal_progress_shape`, `::test_action_planning_repair_message_repeats_nested_contract`, `::test_action_planner_receives_runtime_owner_limits`, and `tests/test_cognition_chain_connector_mapping.py::test_connector_projects_runtime_owner_limits_into_cognition` | Existing one replacement remains the cap; no new retry or semantic normalization. |
| `nodes/dialog_agent.py::_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT` must carry the trusted runtime limit rule into the same-owner repair. | C16 r3 ended with `dialog remains hard-invalid after one repair`; `tests/test_dialog_agent.py::test_dialog_generator_repairs_unresolved_context_once` | Existing one repair and three verifier rechecks remain unchanged. |
| `tests/cognition_baseline_worker.py::_evaluate_hard_gates` must require zero accepted-task/background-work deltas when the scheduler-disabled fixture expects a scheduler-owned effect. | V2 C16 r1-r3 persisted the unrelated rows; `tests/test_cognition_baseline_hardening.py::test_disabled_scheduler_rejects_generic_delayed_side_effect` | Test-only guard; scheduler-enabled runs retain the production schedule gate. |

The controlled C16 and C20 fixtures now include
`no_unowned_delayed_side_effect`. This amendment keeps the exact configured
MongoDB target, Asuna seed, model routes, attempt caps, Chinese semantic
contract, and no-censorship policy. The real LLM rerun must repeat C16 three
times after this amendment and inspect each raw artifact before classification.

### Execution rule amendment: stage-specific LLM quality before E2E tuning

The user approved the following execution rule after reviewing the C18 E2E
artifact:

1. An E2E output-quality difference is an observation and reproduction context.
   It is not a direct production prompt target or a reason to reshape a frozen
   fixture.
2. The parent first creates a focused real-LLM test for the owning semantic
   stage. For the current C18 observation, the focused stages are the action
   planner and action authorizer, with the same trusted runtime capability
   limits, mixed delayed-work input, and Chinese semantic contract.
3. The focused artifact must show the raw stage output, parsed decision,
   capability owner, runtime-limit interpretation, and a semantic quality
   judgment. The parent runs and inspects one focused case at a time.
4. A prompt or contract change enters the production amendment only after the
   focused test reproduces the stage defect and then passes its semantic
   quality rubric on the matched input. The change remains owned by that
   producing stage and keeps the existing attempt cap and model route.
5. The parent then reruns the matched E2E case one at a time. E2E hard gates
   cover structural and ownership invariants; monologue and visible dialog are
   reviewed for groundedness, truthful capability ownership, task fidelity,
   character judgment, and natural Chinese wording. Paraphrase is acceptable.
6. A focused stage pass plus a qualitative E2E pass is required before the
   next case. The parent records any residual quality concern as evidence for
   review rather than encoding the observed wording as a deterministic rule.

The focused planner and authorizer tests now pass on the matched input and
runtime. The latest C18 E2E artifact has no action requests, no accepted-task
or background-work rows, one persisted preference, and a visible Chinese
limitation response. Its remaining failures are fixture-contract failures:
the old `schedule_once` and `three_private_outcomes` gates demand effects that
the frozen unavailable owners must not create. The parent therefore performs
the deterministic fixture audit below before the next E2E rerun; no production
prompt change is inferred from this artifact.

### Phase 2 amendment: propagate runtime owner limits into action authorization

The narrow focused live-LLM selector
`tests/test_cognition_core_v2_action_planning_live_llm.py::test_unavailable_reminder_does_not_change_capability_owner`
was run alone with the matched Chinese reminder input. It reproduced the
stage defect before any production edit. The planner emitted an
`accepted_task_request` whose semantic goal was to confirm receipt while
admitting that the scheduler could not actually arrange the reminder. The
authorizer then received a candidate containing the evidence and proposed
goal, but no `runtime_capability_limits`, and returned `{"c1": true}`. The
parsed result therefore retained the generic accepted-task owner for a
future-speak effect. The trace is
`test_artifacts/llm_traces/cognition_core_v2_action_planning_live_llm__unavailable_reminder_owner.json`.

This is a producing-stage contract defect, not an E2E fixture defect. The
approved production amendment is limited to the following canonical boundary:

| Owner and invariant | Red selector and proof | Green/follow-up command | Bounded effect |
|---|---|---|---|
| `cognition_core_v2/action_selection.py::plan_actions` passes the unchanged trusted `runtime_capability_limits` into `cognition_core_v2/action_authorization.py::authorize_action_requests`; `action_authorization.py::ACTION_AUTHORIZATION_PROMPT`, its JSON candidate payload, and `_authorization_repair_message` state that an unavailable unique owner cannot be replaced by another capability. | `tests/test_cognition_core_v2_action_planning_live_llm.py::test_unavailable_reminder_does_not_change_capability_owner` currently fails after the authorizer approves `accepted_task_request`; deterministic payload coverage is `tests/test_cognition_core_v2_action_authorization.py::test_action_authorization_receives_runtime_owner_limits`. | Focused live case: `venv\Scripts\python.exe -m pytest -o addopts= -m live_llm tests\test_cognition_core_v2_action_planning_live_llm.py::test_unavailable_reminder_does_not_change_capability_owner -q -s`; then run the matched C18 controller worker one case at a time and inspect its raw artifact before the next case. | No new LLM stage, retry, route, enum, keyword classifier, semantic post-filter, or output suppression. The existing two-attempt authorizer cap and model route remain unchanged; only the existing trusted runtime fact is carried to the owner that makes the authorization decision. |

The focused authorizer contract is: with the scheduler/future-speak owner
marked unavailable and no independent delayed-work request in the evidence,
the authorizer must reject a planner proposal that uses
`accepted_task_request` merely to acknowledge or represent the reminder.
Visible speech may still acknowledge the request and explain the limitation;
the action stage must not persist an unrelated delayed effect. The live test
asserts the semantic owner and retains the raw planner and authorizer calls;
it does not assert generated prose.

The parent reviewed the red trace, confirmed the missing field at the exact
planner-to-authorizer boundary, and approved this narrow amendment for
execution under the existing hardening approval. Any further E2E quality
difference remains an observation until its own focused stage test reproduces
it.

The subsequent C18 E2E reruns refined the same focused owner contract. The
surface correctly described unavailable scheduling and background execution,
but one run persisted an accepted task for the Markdown request. The latest
E2E admitted bid, all seven production action affordances, and the three
runtime limits are captured by
`tests/test_cognition_core_v2_action_planning_live_llm.py::test_captured_c18_bid_does_not_create_unowned_preference_work`.
Its first red run selected `memory_lifecycle_update` for an ordinary short-
reply preference. The action registry audit shows that this capability owns
active-commitment lifecycle review; ordinary preferences belong to the
memory/consolidation path. The focused fixture therefore asserts an empty
action set for this current confirmation when every delayed owner is
unavailable, and keeps the raw trace as evidence.

The production prompt scope covers
`action_spec/registry.py::_memory_lifecycle_projection`,
`cognition_core_v2/action_selection.py::ACTION_PLANNING_PROMPT`, and
`cognition_core_v2/action_authorization.py::ACTION_AUTHORIZATION_PROMPT` (and
their existing bounded repair messages): state that ordinary preferences and
style facts are not `memory_lifecycle_update` requests, and that the trusted
background-worker limit covers every delayed-work owner exposed to the
planner. The focused red selector above is the gate; its green command is
`venv\Scripts\python.exe -m pytest -o addopts= -m live_llm tests\test_cognition_core_v2_action_planning_live_llm.py::test_captured_c18_bid_does_not_create_unowned_preference_work -q -s`.
No deterministic semantic post-filter, keyword classifier, extra retry, new
owner, or E2E fixture mutation is authorized by this clarification.

### Harness RCA amendment: persistence gate false positive

The latest C18 artifact exposed a harness false positive after the focused
planner and authorizer tests passed. The artifact's cognition output contained
`action_requests: []`; `accepted_tasks` and `background_work_jobs` both stayed
at zero before and after the turn. Nevertheless,
`accepted_task_persisted` was reported true because
`tests/cognition_baseline_worker.py::_evaluate_hard_gates` recursively treated
the semantic field name `accepted_task_request` as proof of a persisted row.
That field is a planner proposal, not durable state. The hard-gate result was
therefore not evidence of an accepted task.

The test-only correction makes the ownership distinction explicit:
`accepted_task_persisted` and `accepted_coding_task_persisted` require a
positive `accepted_tasks` database-count delta; `accepted_task_status` accepts
an existing accepted-task count or explicit persisted-task state evidence.
`tests/test_cognition_baseline_harness.py::test_persistence_gates_require_persisted_rows_not_planner_proposals`
is the deterministic guard that reproduces the false positive and proves its
rejection. This correction changes no production code, prompt, retry, model
route, or fixture wording.

### C18 gate contract amendment: unavailable owners are not required effects

The focused stage pass and the C18 artifact establish an independent fixture
contract defect. The controller deliberately freezes
`CALENDAR_SCHEDULER_ENABLED=false` and
`BACKGROUND_WORK_WORKER_ENABLED=false`; the action planner and authorizer
correctly return no delayed-work request under those trusted limits. The
artifact also records one preference-memory delta, a truthful limitation
contract, and zero accepted-task/background-work/schedule deltas.

The C18 hard gates therefore change from execution-success expectations to
the invariant set
`visible_dialog`, `memory_lifecycle`, `truthful_limitation`,
`no_false_promise`, and `no_unowned_delayed_side_effect`. This is an
independent capability-availability correction, not a wording or model-output
fixture: no exact generated phrase, task choice, emotion, or route is encoded.
`tests/test_cognition_baseline_harness.py::test_c18_fixture_uses_runtime_compatible_quality_gates`
guards the corrected contract. The parent reruns the focused stage test, then
reruns C18 one case at a time and reviews its raw monologue/dialog before
continuing.

### Priority-zero C20 recurrence amendment: full-path semantic inconsistency

The matched C20 E2E case was executed three times against the configured
`_test_kazusa_core_v2` database with the same Asuna profile, input, disabled
scheduler/background owners, model routes, and Chinese runtime contract. All
three artifacts passed the structural gates and recorded zero
`accepted_tasks`, `background_work_jobs`, and `calendar_schedules` deltas.
The archived artifacts are:

- `test_artifacts/cognition_core_v2/baseline_regression_hardening/quality_archives/C20_quality_repeat_r1/r1.json`
- `test_artifacts/cognition_core_v2/baseline_regression_hardening/quality_archives/C20_quality_repeat_r2/r1.json`
- `test_artifacts/cognition_core_v2/baseline_regression_hardening/quality_archives/C20_quality_repeat_r3/r1.json`

The repeated quality observation is semantic rather than structural: each
`cognition_core_output.admitted_bid` selected “承诺执行” from the evidence's
`response_operation`, and each private monologue retained a future-reminder
commitment even though the visible dialog truthfully described the unavailable
scheduler. The graph evidence shows `action_requests: []` and no delayed owner
was invoked, so the responsibility is currently localized to the goal
cognition semantic decision and its downstream monologue contract. The
structural `no_false_promise` gate is therefore insufficient as a prose-quality
oracle and remains a gate for side-effect safety only.

The next required focused test uses the exact C20 input, the exact
`response_operation` evidence, the same Asuna identity, and the same runtime
limits. Its quality rubric positively requires: acknowledge receipt, preserve
the unavailable future-speak owner, and keep private monologue aligned with
that boundary. The parent inspects raw goal output one case at a time. A
focused semantic pass is required before any production prompt amendment; a
focused red result is required before changing the goal prompt. No lexical
filter, censorship rule, exact phrase assertion, extra retry, or E2E-specific
production branch is permitted.

The exact focused run now reproduces the E2E defect. The raw goal output in
`test_artifacts/llm_traces/cognition_core_v2_goal_cognition_live_llm__c20_unavailable_reminder_boundary__20260724T004336287677Z.json`
contains the runtime limitation, but its `concrete_detail` and
`private_monologue` still recommend “我会记得” and “答应下来”. The raw
semantic judge returns `receipt_acknowledged=true`,
`future_execution_expectation=true`, `private_monologue_aligned=false`, and
`passed=false`; the focused test therefore fails at its semantic gate. This
is the required focused red evidence, not a contract-shape failure.

The approved minimal production amendment is limited to
`src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py::GOAL_COGNITION_PROMPT`:
make `response_operation` authoritative for actor, target, selection, and
current-turn response intent, while making `runtime_capability_limits`
authoritative for feasible outcomes and future execution. When the operation
asks for a commitment whose owner is unavailable, the goal should positively
select current-turn receipt plus a truthful limitation; its intention,
expected consequences, and private monologue must share that same boundary.
The existing JSON contract, attempt cap, model route, evidence handles, and
downstream stages remain unchanged. The focused semantic judge is the gate for
this amendment; no production keyword test or post-generation rewrite is
introduced.

The focused green rerun passed after the prompt amendment. Its raw goal output
selects receipt plus a truthful limitation, and its raw judge returns
`receipt_acknowledged=true`, `boundary_preserved=true`,
`future_execution_expectation=false`, `private_monologue_aligned=true`, and
`passed=true`. The matched post-fix C20 E2E rerun then passed
`visible_dialog`, `truthful_limitation`, `no_false_promise`, and
`no_unowned_delayed_side_effect`. The archived post-fix artifact is
`test_artifacts/cognition_core_v2/baseline_regression_hardening/quality_archives/C20_quality_post_prompt_r1/r1.json`.
Its admitted bid, private monologue, and visible dialog all state the same
current-turn boundary; `action_requests` is empty and accepted-task,
background-work, calendar-run, and calendar-schedule counts remain zero.

### Phase 3 amendment: repository-task owner and bounded planner failure

The first C07/C08 residual inspection follows the priority-zero rule. The
matched runtime snapshot has `accepted_task` and `background_work` unavailable,
with the scheduler also unavailable. The request still requires repository or
source-code reading, so the focused semantic expectation for this snapshot is
`goal_resolution=blocked` with no action or resolver substitution. A visible
surface may explain the unavailable owner; it must not claim repository
evidence, a coding-reader result, or a completed answer.

The focused C07 reproduction
[`C07 action-planning trace`](../../../test_artifacts/llm_traces/cognition_core_v2_action_planning_live_llm__c07_repository_action_planning_frozen_e2e_state.json)
used the frozen E2E graph, Asuna profile, canonical cognition connector, exact
runtime snapshot, and real `BOUNDARY_CORE_LLM`. Its first raw planner object
selected `public_answer_research` for the GitHub repository and produced
`requires_required_evidence`; it did not preserve the repository-task owner.
The focused C08 reproduction
[`C08 action-planning trace`](../../../test_artifacts/llm_traces/cognition_core_v2_action_planning_live_llm__c08_repository_action_planning_frozen_e2e_state.json)
returned a resolver row without `reason`. The parser accepted the object,
silently dropped the invalid row, and left
`requires_required_evidence` with no resolver request. The same stage therefore
exposed both an ownership substitution and a false-success contract path.

The approved hardening surface is limited to these owning symbols:

| Owner and invariant | Focused red evidence | Focused green gate | Change |
|---|---|---|---|
| `cognition_core_v2/action_selection.py::ACTION_PLANNING_PROMPT` must map read-only repository/source analysis to the accepted-task owner and respect the unavailable owner boundary. | C07 focused raw plan selected `public_answer_research`; C07/C08 E2E dialog claimed or implied repository work without repository evidence. | `tests/test_cognition_core_v2_action_planning_live_llm.py::test_c07_action_planning_preserves_repository_task_owner` and `::test_c08_action_planning_preserves_repository_task_owner`, using the frozen runtime and accepting either `blocked` with no requests or `requires_user_input` with only `human_clarification`. | Add one positive capability-owner mapping and one runtime-boundary clarification in the existing Chinese prompt; retain model route, cap, and attempt limit. |
| `cognition_core_v2/action_selection.py::_normalize_action_request_rows` and `_normalize_resolver_request_rows` must treat an invalid row as a whole planner-candidate contract error. | C08 focused raw plan omitted resolver `reason`; current code logged and dropped the row, then returned `requires_required_evidence` with an empty request set. | `tests/test_cognition_core_v2_action_planning_bugfix.py` atomic-row tests plus the C08 focused real-LLM trace showing a bounded replacement or typed blocked result. | Raise the existing bounded validation error for the complete object; the existing same-owner two-attempt path remains unchanged. |
| `nodes/persona_supervisor2_cognition.py::_available_action_affordances` must expose only planner-owned public action handles. | Connector inspection for C07 exposed internal `background_work_request` alongside public accepted-task handles, contrary to the plan's ownership table. | `tests/test_cognition_core_v2_action_planning_bugfix.py::test_speak_and_internal_apply_are_absent_from_planner_affordances` extended to assert internal background work is absent from the production connector. | Exclude the internal capability from planner projection; its existing lifecycle owner remains unchanged. |
| `cognition_core_v2/resolver_authorization.py::RESOLVER_AUTHORIZATION_PROMPT` must preserve repository-source analysis ownership when a public URL is present. | C07 focused resolver authorization returned raw `c1=true` for a GitHub source-analysis goal, treating source accessibility as the result owner. | `tests/test_cognition_core_v2_action_planning_live_llm.py::test_c07_resolver_authorization_rejects_repository_substitution` returns raw `c1=false` and no authorized resolver request on the frozen input. | Add the positive distinction between public source access and the coding-reader result owner in the existing Chinese authorization contract; keep the resolver route and attempt cap. |
| `cognition_core_v2/action_selection.py::plan_actions` must close a required goal when its proposed action or resolver owner is denied. | C08 retained required evidence progress after the action owner was denied; C07 could retain a public resolver substitution after resolver authorization denial. | `tests/test_cognition_core_v2_action_planning_bugfix.py::test_denied_required_action_closes_goal_without_progress`, `::test_denied_required_resolver_closes_goal_without_progress`, and the C07/C08 focused planner tests return `blocked`, empty request sets, and null progress. | Apply the same deterministic fail-closed settlement to either denied owner; preserve the model-owned choice, typed boundary, and existing caps. |
| `TextSurfaceInputV2` and `nodes/persona_supervisor2_l3_surface.py::build_text_surface_input_from_global_state` must carry the cognition-owned `goal_resolution` into the surface owner. `surface.py::_project_surface_payload` and `surface_stages.py::CONTENT_PLAN_SYSTEM_PROMPT` must preserve its meaning. | Matched C07 E2E had cognition `goal_resolution=blocked`, empty action/resolver sets, and no progress, but the canonical surface output planned “开始分析” and “具体架构分析”; the focused content-plan raw result reproduced the same unsupported plan. | `tests/test_cognition_core_v2_surface_owner_live_llm.py::test_c07_content_plan_respects_blocked_repository_owner` must produce a Chinese limitation plan with no unsupported code-review claim, then matched C07 E2E must render that boundary. | Add the existing typed `goal_resolution` enum to the surface handoff and a positive answerability mapping in the existing Chinese content-stage prompt. Keep the current surface stages, model route, and two-attempt cap. |

The focused C07/C08 gates are now green on the exact frozen runtime. The C07
resolver authorization raw result is `c1=false`, and both planner cases retain
the unavailable repository owner: they return `goal_resolution=blocked` with
no requests, or `requires_user_input` with only `human_clarification`. The
connector test also confirms that internal `background_work_request` is absent
from planner affordances. The repository-map and coding-reader E2E hard gates
remain unchanged for the matched rerun: the unavailable snapshot can prove a
truthful blocked outcome, while a separate healthy-owner execution is required
to prove the actual coding-reader route. The next step is one-at-a-time matched
E2E quality review; gate amendments follow only from that deterministic review.

The first matched C07 rerun then exposed a downstream surface-owner defect. The
raw cognition result was truthful (`blocked`), while the visible Chinese dialog
claimed “看了下 KazusaAIChatbot 的代码” and gave architectural conclusions.
The surface owner had received the intention and a generic background-worker
limit, but not the typed answerability result that says the current goal is
blocked. The focused content-plan replay reproduced the same loss before the
dialog generator, so the remediation belongs to the surface handoff and
content-plan owner rather than an E2E wording rule or a dialog post-filter.

The first focused surface assertion also demonstrated a harness false positive:
the model produced a limitation but included a future-reading promise, while
the initial check only looked for limitation markers and absence of completed
review claims. The guard now evaluates the content plan separately from its
semantic requirements and rejects deferred-owner language in the plan itself.
After the typed handoff and prompt amendment, the focused C07 raw plan states
that the link cannot be read in the current boundary and asks for README,
architecture, or code excerpts; it contains no unsupported review or future
background-work promise. This is the focused green gate for the matched rerun.

The matched rerun is now complete for this owner group. C07 `r1`/`r2`/`r3` and
C08 `r1` all passed the structural gates after the deterministic unavailable-
repository alternate was added. Their real Chinese outputs preserve the
request topic, state that the GitHub source is not readable in the frozen
runtime, and ask for user-provided README, architecture, or code material.
The cognition result is either `blocked` or `requires_user_input` with only
`human_clarification`; no action, public research substitution, repository map,
or coding-reader completion is claimed. The alternate gate is a runtime-owned
availability classification, while the monologue/dialog judgment remains a
quality review with accepted Chinese paraphrases.

### C11 first rerun: unavailable coding owner gate correction

The first matched C11 rerun followed the priority-zero sequence. The exact
frozen input, Asuna profile, model/configuration, disabled background owners,
workspace, and `_test_kazusa_core_v2` target were retained. The artifact is
`test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/C11/r1.json`.
Its Chinese private monologue says the role wants to update README but cannot
read the GitHub repository and asks for the file content. Its visible dialog
confirms the request, states the repository-reading limitation, and asks the
user to provide `README.md`; it makes no repository claim or future execution
commitment.

The artifact still reported three technical failures:
`accepted_coding_task_persisted`, `coding_run_bound`, and
`guarded_workspace_effect`. The graph recorded
`worker_status.accepted_task=unavailable`,
`worker_status.background_work=unavailable`,
`worker_status.orchestrator=unavailable`, an empty `action_requests` list, a
`human_clarification` resolver request, zero accepted-task/background-work
deltas, and an unchanged guarded workspace. This is a harness false negative:
the fixture required successful coding execution while its frozen runtime
truthfully provided no coding execution owner.

The owning semantic stages were checked before changing the gate. The focused
real-LLM action-authorizer test
`tests/test_cognition_core_v2_action_planning_live_llm.py::test_c11_action_authorization_rejects_unavailable_coding_owner`
returned raw `{"decisions":{"c1":false}}`. The focused planner test
`::test_c11_action_planning_preserves_unavailable_coding_owner` returned
`action_requests=[]` and `goal_resolution=blocked`. The focused surface test
`tests/test_cognition_core_v2_surface_owner_live_llm.py::test_c11_content_plan_respects_blocked_coding_owner`
produced a Chinese limitation plan requesting accessible README/code material,
without an unsupported review or deferred coding promise. These are the
canonical connector and contract paths; no production prompt change is
warranted by C11.

The approved correction is test-only and belongs to
`tests/cognition_baseline_worker.py::_evaluate_hard_gates`. It derives a typed
`coding_owner_unavailable` condition only when the trusted worker statuses,
empty action set, blocked-or-user-input cognition result, clarification-only
resolver set, zero accepted-task delta, absent coding run, and unchanged
guarded workspace agree. Under that condition, the three execution-success
gates accept the truthful unavailable-owner outcome; a generic planner
proposal, a promise in dialog, a queued task, a coding-run row, or a workspace
mutation cannot activate the alternate. The deterministic guard is added to
`tests/test_cognition_baseline_hardening.py` and proves both the truthful C11
classification and rejection when any side effect appears. After the guard is
green, C11 is rerun once through the E2E controller and its Chinese
monologue/dialog are reviewed before C12 begins.

### C12 fixture RCA: declared coding context was never seeded

The initial C12 artifact is a fixture false negative. The manifest declares
`state_seed.coding_run` with `baseline-run-012`, `proposal_ready`, and the
`status,cancel,approve_and_verify` action set, but
`tests/cognition_baseline_worker.py::_run_chat_case` only called
`_seed_conversation_rows`. The worker therefore entered the live Mongo turn
with `accepted_tasks=0` and `background_work_jobs=0`; the production
`_load_live_action_selection_context` query correctly returned no open coding
context. The real LLM received no bound run and produced a progress-style
response without a status owner. This evidence cannot classify the cognition
or coding lifecycle path because the stated precondition was absent.

The fixture correction is test-only. The worker will materialize the declared
coding seed as one accepted-task document with a canonical
`coding_run_context.v1`, the exact debug channel/requester scope, active state,
and the declared run status/actions before `service.chat`. The artifact will
retain a separate `seeded_coding_run` evidence field, while the existing
conversation `seeded_context` list remains unchanged. A deterministic builder
test proves the scope, run reference, status, allowed actions, and
`followup_open` contract; a negative builder case rejects a missing run id or
empty action set. The cleanup path already drops the configured guarded test
database after the turn.

After the seed is materialized, the parent runs one focused real-LLM planner /
authorizer test on the exact C12 input and canonical connector. The focused
rubric requires a bound `coding_run:baseline-run-012` status owner when the
seed is present, preserves the declared `proposal_ready` state, and keeps
unavailable worker limits truthful. The raw stage output, parsed request,
context ref, and runtime interpretation are recorded before the matched C12
E2E rerun. No production prompt, route, retry, or generated wording is changed
for this fixture correction.

### C12 first seeded E2E: status-owner handoff gap

The first seeded C12 E2E is a real post-fixture failure, separate from the
original missing-seed false negative. The focused planner/authorizer test was
green and produced `accepted_coding_task_request`, `decision=status`, and the
bound `coding_run:baseline-run-012`. The matched E2E then materialized the
seeded row (`accepted_tasks` 1 before the turn), created a second accepted task
and one background-work row, and returned no coding-run status result. The
visible Chinese output said the README task had not started and asked the user
to provide the README again, although the seeded context was already
`proposal_ready`. The artifact is
`test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/C12/r1.json`.

The owning boundary is the existing-task status handoff. The source contract
defines `accepted_task_status_check` as the direct persisted-status lookup,
while `accepted_coding_task_request(status)` creates a background coding-agent
request. In this frozen runtime the coding worker is unavailable and the
coding capability is exposed as queue-only/degraded; the action therefore
created another pending request instead of returning the seeded
`coding_run_context`. L3 received no status evidence and followed the generic
repository-limit path. This evidence explains both gate failures
(`coding_run_status`, `no_unbound_run`) and the visible quality defect.

The next gate is a focused real-LLM red reproduction on the same frozen
connector, with the seeded context and unavailable worker. Its rubric requires
the direct persisted-status owner, no new coding request, and a result that
retains `coding_run:baseline-run-012` plus `proposal_ready`. After that focused
red is recorded, the narrow stage change will teach the Chinese planner
contract to prefer the direct status lookup when the coding worker is
unavailable, and the status-check affordance will expose only Chinese semantic
guidance. The focused test must pass before the single matched C12 E2E rerun.
The E2E review will assess whether the Chinese monologue/dialog reports the
seeded state accurately; wording equality is not a gate.

The second matched E2E exposed a downstream context omission after the status
owner correction: Mongo remained at one seeded accepted task and no new coding
task was created, but goal cognition still emitted `human_clarification` and
the final dialog asked for repository access. Its bid and appraisal explained
the result as “没有仓库读取能力”, while the seeded `proposal_ready` context
was available only to action affordance projection and was absent from the
goal branch semantic scene. A second focused goal-cognition test reproduced
this exact omission, then turned green after the canonical connector
projected the bounded Chinese task status into
`scene_context.semantic_scene` and the goal prompt treated that projection as
status evidence. The focused artifact is
`test_artifacts/llm_traces/cognition_core_v2_goal_cognition_live_llm__c12_persisted_status_context__20260724T053712083409Z.json`.

The next matched C12 E2E exposed a separate L3 handoff defect. The status-check
action executed once and produced a bounded `proposal_ready` result, but
`stage_3_action` called L3 before materializing that result. The surface
connector read the final `action_results` key, while the live graph's canonical
pre-surface boundary was `pre_surface_action_results`; the final dialog therefore
received an empty permitted-action ledger and said it could not query the task,
even though the protected action trace contained the correct status. This was a
production sequencing/connector defect, not an E2E wording preference.

The owning focused surface test reproduced the defect with the exact C12 seed
and real content-plan LLM. Its raw and parsed plan remained semantically
plausible because the cognition bid already mentioned `proposal_ready`, but the
focused semantic judgment correctly failed when
`permitted_action_results` was empty:
`test_artifacts/llm_traces/cognition_core_v2_surface_owner_live_llm__c12_persisted_coding_status_result__20260724T060620050324Z.json`.
The narrow fix keeps `pre_surface_action_results` as the canonical handoff,
executes selected non-surface actions before L3, excludes those attempts from
the later settlement pass, and projects the same result into surface planning.
The deterministic L3 handoff suite passed 8 tests. The focused real-LLM surface
test then passed with the status result present:
`test_artifacts/llm_traces/cognition_core_v2_surface_owner_live_llm__c12_persisted_coding_status_result__20260724T060814602275Z.json`.

The matched C12 E2E is now closed at
`test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/C12/r1.json`.
It reports `technical_status=passed`, all C12 hard gates passed, one seeded
accepted task before and after the turn, and no duplicate background job. The
Chinese monologue identifies `proposal_ready` with no blocker. The visible
dialog reports the same state and offers `approve_and_verify`, `status`, and
`cancel` without claiming repository access or completed background work.
Chinese paraphrases remain the quality criterion; exact wording is not required.

The C11 repetition audit then exposed a test-contract false negative. With the
same unavailable coding owner, one real run selected a coding action, one
selected the truthful clarification path, and one returned no action or
resolver while its cognition intention and visible Chinese output explicitly
described the unavailable repository owner. The last result was a valid current
limitation answer with no coding effect; requiring a blocked goal or a
clarification resolver incorrectly treated that truthful no-effect response as
a failed coding execution. The focused C11 fixture was corrected to use the
actual failed r3 artifact and its guard now requires the raw/parsed result to
contain no action, no resolver, and an explicit owner limitation before it can
pass. The deterministic gate test proves that an attempted coding action is
rejected by the alternate. C11 r2 and r3 were rerun one at a time; both now
pass with zero accepted-task/background-work deltas and no workspace mutation.

### Post-fix corpus residual audit and execution order

The three frozen corpora now contain all 82 expected artifacts. The ledger
verification and blind pairing both pass with 82 matched pairs and zero
blocked artifacts. The technical result is:

| Corpus | Technical pass | Technical fail | Interpretation |
|---|---:|---:|---|
| `pre_fix_main` | 39/82 | 43/82 | current baseline floor and behavior reference |
| `pre_fix_v2` | 23/82 | 59/82 | measured V2 pre-hardening loss |
| `post_fix_v2` | 76/82 | 6/82 | C03, C07, C08, C11, C12, C20, O01, and O04 closed; C13-C15 and O03 remain open |

The remaining 6 post-fix failed artifacts are grouped by owning boundary:

| Cases | Repetitions | Observed gate family | Current status |
|---|---:|---|---|
| C13-C15 | 3 | coding run unblocking, guarded workspace, cancellation | coding lifecycle owner RCA required |
| O03 | 3 | media evidence before answer | media observation/evidence RCA required |

The next execution order is fixed by earliest owning stage and repetition:

1. Continue with C13-C15 at the coding lifecycle owner, beginning with the
   first corrected seeded-run precondition and its stage-level owner test.
2. For every proven stage defect, obtain focused red and focused Chinese
   semantic green evidence before considering a production amendment. For an
   expected silence or harness error, correct only the deterministic contract
   and add a regression guard.
3. Preserve the exact project identity, URL, workspace, authorization, and
   database seed while testing the coding lifecycle.
4. Inspect O03 at the media observation/evidence owner after the earlier
   relevance boundary is classified.
5. Rerun only the matched E2E cases whose owning focused stage has passed, one
   case at a time, and review Chinese monologue/dialog quality. Then rerun the
   complete 82-case corpus and recompute the blind comparison.

The C20 recurrence is closed independently: its focused stage passed and its
matched post-fix E2E artifact passed the truthful-limit and no-unowned-effect
gates. C03, O01, and O04 are now closed by their focused gates and matched E2E
quality reviews. C07 and C08 are now closed by the repository-owner evidence
above. The 6 residual failed artifacts above remain open evidence. No
production change is inferred from them until the corresponding focused test
proves a semantic owner defect.

### O01/O04 relevance and local-recall closure

The O01 RCA identifies a harness false positive. The frozen input
`这个问题有人知道吗？` reached the real frontline relevance owner with
`semantic_target_labels=["none"]`; the raw model output was
`intake_action=discard`, the parsed result remained `discard`, and the public
response had no visible messages. The path terminates at relevance silence, so
it owns neither a private monologue nor a final cognition monologue. The old
harness applied those downstream wording gates solely because
`source_kind=user_message`, producing a false technical failure after a valid
silence decision. The harness now uses
`_requires_live_monologue(source_kind, output_mode)` and exempts the explicit
`silent` surface. The deterministic guard is
`tests/test_cognition_baseline_hardening.py::test_silent_relevance_ends_before_monologue_contract`.
The real O01 capture and formal `r1`/`r2`/`r3` worker artifacts all show
`technical_status=passed`, `no_visible_dialog`, empty monologue, and empty
dialog.

The O04 RCA identifies a fixture identity defect rather than a production
semantic-owner defect. The initial public trace seeded the prior promise with
`global_user_id=null` and `platform_user_id=null`; relevance could see the
row, while the database-owned history lookup for the actual current author
could not. The focused action-planning replay then passed through the canonical
connector with `goal_resolution=requires_required_evidence`, `route=evidence`,
and `local_context_recall`. The focused local-context replay with a correctly
scoped history row also passed: planner selected `recall_evidence`, the active
node produced a validated `recall_ref`, and the packet had
`resolved_node_count=2`, `blocked_node_count=0`, and the exact key-return fact.

The test-only worker correction resolves the current request identity before
seeding `prior_user_row`, carries the same global/platform identity into the
row, and asserts that identity projection in the O04 capture. No production
prompt, route, parser, model, or attempt cap changed. Formal O04 `r1`/`r2`/`r3`
worker artifacts all pass every hard gate. Their Chinese private monologues
and visible dialogs are valid paraphrases of the seeded fact that the user
promised to return the key to Kobayashi on Friday; no unsupported plan is
introduced. The first O04 infrastructure retry was a Mongo connection-close
failure and was excluded from semantic comparison; the clean rerun passed.

Evidence:

- [O01 relevance capture](../../../test_artifacts/llm_traces/relevance_baseline_residual_live_llm__O01_e2e_frontline_silence__20260724T034355393233Z.json)
- [O04 action-planning focused green](../../../test_artifacts/llm_traces/cognition_core_v2_action_planning_live_llm__o04_action_planning_frozen_e2e_state.json)
- [O04 local-context focused green](../../../test_artifacts/llm_traces/relevance_baseline_residual_live_llm__O04_local_context_resolver_focused.json)
- [O04 public-path capture](../../../test_artifacts/llm_traces/relevance_baseline_residual_live_llm__O04_e2e_persona_relevance__20260724T034229708166Z.json)
- [O01 r1 artifact](../../../test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/O01/r1.json)
- [O01 r2 artifact](../../../test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/O01/r2.json)
- [O01 r3 artifact](../../../test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/O01/r3.json)
- [O04 r1 artifact](../../../test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/O04/r1.json)
- [O04 r2 artifact](../../../test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/O04/r2.json)
- [O04 r3 artifact](../../../test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/O04/r3.json)

### C03 fixture amendment: seed chronology and author identity

The first exploratory focused C03 test and its prompt amendment are
superseded. That test hand-built a `before_active_turn`/current-author state
that did not match the public E2E state, so its red and green reviews are not
production evidence.

The matched E2E state-capture test
`tests/test_relevance_baseline_residual_live_llm.py::test_c03_e2e_captures_relevance_state_before_quality_judgment`
captured the actual owning-stage input. The semantic labels were
`["character"]`, the reply target was `unknown_participant`, and the seeded
history row was projected as `after_active_turn`. The raw model output chose
`already_resolved`/`ignore` because that was the correct decision for the
state it received. The exact focused after-turn test
`test_c03_persona_relevance_respects_after_turn_resolution` reproduced this
state and passed with the same semantic decision.

The RCA is in the test harness chronology. C03 supplied no history timestamp,
so `_seed_conversation_rows` used the fixed value
`2026-07-24T09:00:00+00:00`. The active fixed local time is
`2026-07-24 09:00` in `Pacific/Auckland`, which converts to an earlier UTC
timestamp. The seed therefore landed after the active turn and caused the
correct relevance-stage resolution to suppress downstream response.

The first corrected E2E run also exposed a second fixture defect: the seed row
had no current-user platform or global identity, so it was projected as
`speaker_relation=other_participant` instead of `current_author`. Its visible
dialog `在玄关那个蓝色的小盒子里呢。` did not agree with the seeded fact
`我把门禁卡放进书桌右边第二个抽屉了。`; the run therefore passed structural
`visible_dialog`/`local_recall` gates while failing semantic evidence fidelity.
The before-turn focused probe that used the same identity-free row returned the
unsupported English disposition `terminate`; that probe is fixture evidence,
not evidence for the intended current-author retrospective path. After adding
the declared current-user identity, the corrected focused test still returned
`terminate` and described the `before_active_turn` fact as an already answered
request. This second red is valid owning-stage evidence because the focused
state now matches the intended C03 semantic contract.

The two approved corrections are test-only: derive the default seed timestamp
from the fixed active turn and place it one minute before that turn, and declare
the seed's current-user platform/global identity in C03. The chronology guard
`test_untimestamped_seed_precedes_fixed_active_turn` remains required; the
matched stage capture must additionally show `speaker_relation=current_author`
and `turn_relation=before_active_turn`. The valid focused red identifies a
minimal production prompt boundary amendment: positively define that a
`before_active_turn` `current_author` fact supplies the answer evidence for a
current retrospective request, while an applicable during/after response is
the completion evidence. The amendment keeps the existing action space, parser,
bounded attempt cap, model route, and deterministic execution boundary. It does
not add a keyword router, content handling, or E2E wording rule. The corrected
focused test must pass before another C03 E2E run; that E2E is reviewed for
evidence-grounded Chinese monologue and dialog, not exact wording.

The corrected relevance run then exposed a separate downstream projection
failure. The matched E2E goal-state capture
`tests/test_relevance_baseline_residual_live_llm.py::test_c03_e2e_captures_goal_cognition_state_before_quality_judgment`
shows that RAG found the exact seeded fact in
`rag_result.recall_evidence` after the local-context resolver projected the
conversation row, while the goal-cognition prompt initially received only the
episode summary. The canonical connector
`build_cognition_input_from_global_state` currently projects only
`rag_result.memory_evidence`; it does not project the public
`recall_evidence` or `conversation_evidence` fields into registered cognition
evidence sources. The retrieved conversation fact is therefore lost before
the goal owner. This explains the wrong location in the visible Chinese dialog
without attributing the defect to the goal LLM or adding a surface-wording
rule.

The existing focused goal-fidelity test is also marked as inadmissible
production evidence because it bypasses the connector and supplies the
unregistered `conversation_history` source kind directly. It remains a
false-positive guardrail target until its fixture is rebuilt from the canonical
connector and the captured `recall_evidence` packet. The next C03 gate is
therefore: deterministic connector red for RAG recall projection, a reviewed
canonical source/provenance amendment covering the public RAG evidence fields,
a real focused goal test over connector-produced evidence, then one matched
E2E rerun reviewed for grounded Chinese monologue and dialog.

The first post-projection C03 rerun passed every structural and local-recall
hard gate and carried the exact fact through RAG, cognition, and
`text_surface_output_v2`: the cognition private monologue, selected bid, and
content plan all named `书桌右边第二个抽屉`. The final visible dialog instead
said `诶？你刚才说了吗……我完全没注意到在听啊。` followed by a request to
repeat the fact. This is a downstream dialog-owner red result, not evidence for
another RAG or goal-prompt change. The next gate freezes this same surface
input, captures the dialog prompt and raw model output in one focused real-LLM
test, establishes semantic fidelity while accepting valid Chinese paraphrase,
then reruns the matched E2E.

The focused dialog red reproduces the E2E failure with the frozen surface
input. The first generator output contained the correct location, then added
`之前明明跟你说过一遍了` and reversed the historical speaker. The semantic
fidelity verifier correctly rejected that candidate. The bounded repair then
received current percepts and the bad candidate, while the concrete recalled
fact remained only in the surface plan; it consequently replaced the answer
with a request for the user to repeat the fact. The owning amendment is a
positive generator-contract clarification: `content_plan`,
`content_requirements`, and `visible_boundaries` carry the required semantic
answer and actor/source direction; dialog style may decorate that answer while
preserving every required fact. The repair cap, verifier ownership, model
routes, and output contract remain unchanged.

One focused dialog rerun passed with the same frozen surface input. Its raw
generator output and final dialog both expressed `书桌右边第二个抽屉`, while
the verifier path completed without replacing the answer. A later isolated
rerun exposed an upstream semantic-packet problem: the frozen episode said in
`role_explicit_content` that the current user had previously stated the fact,
while its `response_operation` declared the embedded actor as `当前角色` and
the target as `当前用户`. The generator's first sentence gave the correct
location and its follow-up `之前明明说过一次了，你这记性也太差了吧。` was
ambiguous enough to trigger a role review. The bounded repair produced the
semantically consistent sentence `你刚才才跟我说过一遍的，怎么这就忘了？真是
拿你没办法。`, but the semantic verifier continued to follow the contradictory
typed operation and reported a speaker reversal. The resulting hard failure
is therefore false-positive evidence for dialog repair; it cannot authorize a
dialog repair-prompt change.

The next owning-stage gate is a focused real-LLM message-decontextualizer test
using the exact C03 input and a canonical episode. It must prove that
`role_explicit_content` and `response_operation` agree that the current user
previously stated the fact, the current role answers, and the embedded speech
action runs from `当前用户` toward `当前角色`. The dialog focused fixture will
also assert this projection consistency before invoking the dialog owner. A
production dialog change remains deferred until a consistent upstream packet
reproduces a dialog-owned red. The prior focused green remains retained as
evidence of the viable surface path; the newest trace remains an explicit
false-positive guardrail case.

The focused decontextualizer test first rejected an overly literal assertion
that required the phrase `之前说过`; that fixture guard was corrected to accept
the live model's equivalent `之前提到过`. The next real run then produced a
valid Chinese role-explicit sentence but a contradictory operation with
`embedded_actor_role=当前角色` and `embedded_target_role=当前用户`. A separate
run produced the correct actor binding, confirming model variance while the
semantic contract remains unmet. This is admissible owning-stage evidence for
the message-decontextualizer prompt: its role-explicit and response-operation
instructions need a positive retrospective-fact rule that binds the current
user as the source of “我刚才说/提到”的 fact, the current role as responder,
and the embedded addressee as either the explicitly addressed current role or
the valid omitted target. The focused stage must pass with semantically
equivalent Chinese wording before any additional dialog owner change or
matched E2E rerun.

After the positive retrospective-fact prompt amendment, the focused
decontextualizer passed. Its raw output retained the Chinese input and emitted
`当前用户询问当前角色关于当前用户之前提到的门禁卡存放位置`; the operation
set `response_owner_role=当前角色`, `embedded_actor_role=当前用户`, and
`embedded_target_role=无`, which is the valid projection for an omitted
embedded addressee. The dialog fixture then applied the same contract-checked
projection to the frozen surface input instead of carrying forward the
contradictory old E2E packet. The dialog focused run passed with raw output and
visible dialog containing `书桌右边第二个抽屉`; its verifier path completed
without a repair. This is the valid stage gate for the matched C03 E2E quality
rerun. The E2E result will be judged by evidence-grounded Chinese monologue
and dialog quality, with the hard gates retaining their typed invariants.

The mapping projection correction passed its deterministic guardrail, and the
focused real-LLM goal test using only the E2E-shaped `conversation_evidence`
mapping passed. The raw goal output cited `e2` and retained the complete fact
across its semantic fields: `门禁卡`, `书桌`, `右边`, `第二个`, and `抽屉`.
This focused green authorizes the next matched C03 E2E run. The run will use a
fresh empty configured test database and the same frozen case/profile/model
fingerprint; its quality review will verify that this fact reaches the
Chinese surface plan and final dialog.

The matched C03 E2E then exposed a distinct projection-shape gap. RAG returned
the exact seeded sentence under `conversation_evidence` as mapping rows with
`content`, `role`, and `metadata`; `_rag_evidence` only accepted string items,
so the cognition connector received no RAG fact. Cognition selected a generic
answer goal and the surface plan required an accurate location without naming
one; the dialog owner then produced the unsupported location `玄关那个蓝色的小
托盘`. The deterministic guardrail
`test_rag_conversation_mapping_is_canonical_cognition_evidence` reproduced the
drop with the actual mapping shape and failed with only the episode evidence
remaining. The owning production correction is limited to the RAG projection:
mapping items use the existing bounded `_rag_text` extraction and retain the
registered `conversation_evidence` provenance; string items keep their current
path. No dialog wording or E2E-specific location rule is introduced. After the
projection guard passes, a focused real-LLM goal test will use the same mapping
shape and require the fact to survive across the goal-owned semantic fields
before the next matched E2E.

The next matched C03 E2E exposed the action-planning owner. Relevance and
decontextualization completed, while both `action_planning` and its bounded
same-owner repair returned contract errors. The protected trace metadata and
worker warning identify the exact failure: the model selected the
`local_context_recall` resolver, but its optional `resolver_goal_progress`
lists used object rows for `evidence_dependencies` and
`assumptions_or_inferences`, while the canonical contract requires strings.
The bounded replacement repeated the type error, so the fail-contained result
became `goal_resolution=blocked`, `resolver_requests=[]`, and
`resolver_progress.status=not_requested`. The visible dialog then invented a
location even though the structural hard gates passed.

This is a producing-stage contract/prompt defect, not a dialog-quality defect
and not evidence for a deterministic fact-wording rule. The exact focused
reproduction must use `build_cognition_input_from_global_state`, the frozen
Asuna profile, the initialized resolver state, the same runtime capability
snapshot, and the admitted C03 bid. It must inspect the raw model output and
require a validated `local_context_recall` request with
`goal_resolution=requires_required_evidence` before a prompt amendment is
considered. The production amendment may clarify the scalar-list types in the
existing progress schema while preserving the same owner, parser, attempt cap,
route, and fail-closed boundary. The matched E2E rerun follows only after that
focused stage gate passes.

The connector-backed focused action-planning gate then passed after the
prompt/repair clarification. Its raw model output selected `r3` for
`local_context_recall`, the parsed result validated
`goal_resolution=requires_required_evidence`, and all seven scalar progress
lists were represented as string arrays. The evidence is recorded in
`test_cognition_core_v2_action_planning_live_llm.py::test_c03_action_planning_selects_local_recall_from_connector_state`.
The matched fresh-database C03 E2E then dispatched the resolver, carried the
seeded fact through the second cognition pass, and completed all typed stages.
Its Chinese private monologue named `书桌右边第二个抽屉`; its visible dialog was
`在书桌右边第二个抽屉里。` The artifact passed `visible_dialog` and
`local_recall`, with zero accepted-task, background-work, and schedule deltas.
This closes C03 on semantic quality; the next execution item is O04/O01 stage
ownership review.

Evidence:

- [C03 fixture RCA](../../../test_artifacts/llm_reviews/relevance_baseline_residual_C03_fixture_rca.md)
- [E2E state capture](../../../test_artifacts/llm_traces/relevance_baseline_residual_live_llm__C03_e2e_state_capture.json)
- [Exact after-turn focused trace](../../../test_artifacts/llm_traces/relevance_baseline_residual_live_llm__C03_persona_relevance_after_turn.json)
- [Identity-free before-turn focused red trace, superseded](../../../test_artifacts/llm_traces/relevance_baseline_residual_live_llm__C03_persona_relevance_before_turn_failed.json)
- [Current-author before-turn focused red trace](../../../test_artifacts/llm_traces/relevance_baseline_residual_live_llm__C03_persona_relevance_before_turn_failed.json)
- [C03 goal-state capture](../../../test_artifacts/llm_traces/cognition_core_v2_goal_cognition_live_llm__C03_e2e_goal_state_capture.json)
- [C03 goal-fidelity focused trace](../../../test_artifacts/llm_traces/cognition_core_v2_goal_cognition_live_llm__c03_goal_fact_fidelity__20260724T020058479653Z.json)
- [C03 post-projection E2E artifact](../../../test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/C03/r1.json)
- [C03 prior action-planning red artifact](../../../test_artifacts/cognition_core_v2/baseline_regression_hardening/quality_archives/C03_action_planning_pre_fix_r1/r1.json)
- [C03 action-planning focused green](../../../test_artifacts/llm_traces/cognition_core_v2_action_planning_live_llm__c03_action_planning_local_recall_connector_state__20260724T030645652429Z.json)
- [C03 dialog focused red](../../../test_artifacts/llm_traces/dialog_agent_live_llm__C03_surface_fact_fidelity.json)
- [C03 dialog focused green](../../../test_artifacts/llm_traces/dialog_agent_live_llm__C03_surface_fact_fidelity__20260724T022750191454Z.json)
- [C03 dialog verifier false-positive trace](../../../test_artifacts/llm_traces/dialog_agent_live_llm__C03_surface_fact_fidelity__20260724T022957868574Z.json)
- [C03 decontextualizer focused red](../../../test_artifacts/llm_traces/decontextualizer_identity_live_llm__retrospective_user_fact_roles__20260724T023742187580Z.json)
- [C03 decontextualizer focused green](../../../test_artifacts/llm_traces/decontextualizer_identity_live_llm__retrospective_user_fact_roles__20260724T024014807734Z.json)
- [C03 conversation-evidence goal focused green](../../../test_artifacts/llm_traces/cognition_core_v2_goal_cognition_live_llm__c03_goal_conversation_fact_fidelity__20260724T024947514735Z.json)
- [C03 dialog consistent-projection focused green](../../../test_artifacts/llm_traces/dialog_agent_live_llm__C03_surface_fact_fidelity__20260724T024054149480Z.json)
- [Exploratory red review, superseded](../../../test_artifacts/llm_reviews/relevance_baseline_residual_C03_red.md)
- [Exploratory green review, superseded](../../../test_artifacts/llm_reviews/relevance_baseline_residual_C03_green.md)

## Overdesign Guardrail

- Actual problem: V2 can select the wrong owner, silently degrade malformed
  semantic output, suppress/fictionalize action effects, and pass invalid tests.
- Minimal change: freeze a neutral differential corpus, then correct only
  proven canonical route, validation, outcome, prompt, and settlement owners.
- Ownership: LLMs decide semantics; deterministic code validates, routes typed
  shapes, executes, persists, limits, and delivers.
- Rejected complexity: policy DSL, keyword router, per-capability planner,
  extra verifier/agent, V1 fallback, compatibility alias, dual schema, retry
  expansion beyond stated replacements, content filter, and DB migration.
- Evidence threshold: new complexity requires a repeated frozen failure, exact
  red test, reviewed plan amendment, and user approval.

## Agent Autonomy Boundaries

- Parent may create Phase 1 tests/artifacts, isolated worktrees, and guarded
  disposable DBs after initial approval.
- Parent may edit only the approved Phase 2 surface after the amendment gate
  and red tests.
- Agents may choose local mechanics only when all named contracts stay exact.
- Search for equivalent behavior before adding a helper; move/reuse the
  existing owner instead of duplicating it.
- Changes outside the named surface, new prompts/variables/public paths, model
  call/cap changes, DB schema changes, accepted regressions, fixture mutations
  after first execution, and content-policy changes require user approval.
- Agents perform no unrelated cleanup, formatting churn, dependency upgrade,
  broad refactor, or production-data operation.
- If plan and code disagree or an instruction is impossible, stop and report
  the exact conflict.

## Implementation Order

1. Freeze SHAs/hashes/worktrees and create the evidence manifest.
2. Add fixture schemas and negative provenance tests; run them red then green.
3. Add the neutral controller/worker and prove process/profile/DB isolation.
4. Generate owner and LLM-producer matrices; fail on any unmapped owner.
5. Run main, then pre-fix V2, one case per command; inspect each raw artifact.
6. Repeat differences, blind-score, and freeze the regression ledger.
7. Amend exact production surface/red tests/budgets; self-review and obtain
   user approval.
8. Add focused failing tests for each approved loss.
9. Parent implements parser/replacement and atomic planner errors.
10. Parent implements canonical route, ownership, and outcomes.
11. Parent implements pre-surface execution and dialog truth.
12. Parent reruns focused deterministic/integration/live-DB gates.
13. Parent runs identical post-fix corpus/repetitions and rescoring.
14. Parent runs deleted-baseline and Stage 1-3 overlays plus full regression.
15. Parent applies the full code/evidence review rubric and closes findings.
16. Present raw transcript and engineering closure ledger for user sign-off.

## Execution Model

- Parent owns orchestration, Phase 1 code, fixtures, red tests, live execution,
  inspection, scoring, integration, evidence, review remediation, and lifecycle.
- Parent establishes each focused failure before production implementation.
- Parent implements the approved production write set only after the Phase 1
  amendment is approved and red tests exist.
- Parent pauses implementation before review, rereads this plan and the
  complete diff, applies every rubric item under Independent Code Review, and
  records file/line evidence.
- Parent fixes in-scope self-review findings and reruns affected gates. New
  scope returns to amendment approval.

## Progress Checklist

- [x] A. Phase 1 freeze and harness contract complete.
  - Covers steps 1-4; files: Phase 1 create/modify list.
  - Verify, in order:
    `venv\Scripts\python.exe tests\cognition_baseline_comparison.py preflight-paths`;
    create the two frozen worktrees with the commands under Verification;
    `venv\Scripts\python.exe -m pytest tests\test_cognition_baseline_harness.py -q`;
    `venv\Scripts\python.exe tests\cognition_baseline_comparison.py preflight`;
    run both producer audits and `verify-deleted-selector-map`.
  - Evidence: SHAs, all hashes, negative-guard results, worktree/process/DB
    proof, owner/producer matrices.
  - Handoff: start current-main corpus. Sign-off records parent and UTC time.
- [x] B. Fresh pre-fix differential complete.
  - Covers steps 5-6; files: frozen raw artifacts and two ledgers.
  - Verify: repeatedly run
    `venv\Scripts\python.exe tests\cognition_baseline_comparison.py run-next --corpus pre_fix_main`
    and then `--corpus pre_fix_v2`; run `compare --blind` and `verify-ledger`.
  - Evidence: inspected artifact per case, matched repetitions, blind scores,
    hard-gate results, changed-owner/deleted-selector coverage.
  - Handoff: exact amendment and review. Sign-off records parent and UTC time.
- [ ] C. Phase 2 amendment approved and red contracts proven.
  - Covers steps 7-8; files: this plan and named focused test modules.
  - Verify: every amendment-listed selector fails for its stated reason before
    production edits; `git diff --check`.
  - Evidence: failing output, exact production write set, parent plan
    self-review record, and user approval.
  - Handoff: parent production implementation. Sign-off records parent and UTC
    time.
- [ ] D. Approved production hardening integrated.
  - Covers steps 9-12; files: approved confirmed/amended Phase 2 list.
  - Verify: same red selectors pass; focused commands under Verification pass.
  - Evidence: changed files, implementation log, test output, no unapproved
    diff.
  - Handoff: post-fix corpus. Sign-off records parent and UTC time.
- [ ] E. Phase 3 matched rerun and broad gates complete.
  - Covers steps 13-14; files: post-fix artifacts and closure ledgers.
  - Verify: repeat `run-next --corpus post_fix_v2` until complete;
    `verify-ledger --phase post_fix`; run all overlay/broad commands.
  - Evidence: same hashes/repetitions, scores, zero losses/hard failures,
    Stage 1-3 and deleted-baseline results.
  - Handoff: parent code/evidence self-review. Sign-off records parent and UTC
    time.
- [ ] F. Parent self-review and user sign-off complete.
  - Covers steps 15-16; scope: full diff, plan, raw evidence, commands, risks.
  - Verify: rerun every reviewer-affected focused/static gate; `git diff --check`.
  - Evidence: reviewer identity/findings/fixes/reruns/verdict and user approval.
  - Handoff: lifecycle update/archive. Sign-off records parent and UTC time.

## Verification

### Freeze and harness

```powershell
git status --short
venv\Scripts\python.exe tests\cognition_baseline_comparison.py preflight-paths
git fetch origin main
git rev-parse origin/main
git rev-parse HEAD
git diff --name-status origin/main...HEAD
$baselineRevision = (git rev-parse origin/main).Trim()
$candidateRevision = (git rev-parse HEAD).Trim()
if ($baselineRevision -ne '8f834bf87a83ee42aca804934fb44af63788420c') { throw 'origin/main moved; amend and re-review the plan before comparison' }
if ($candidateRevision -ne '0c2e929d51ac80c4519f564b61cbf8949efcca3d') { throw 'candidate moved; amend and re-review the plan before comparison' }
git worktree add --detach 'C:\workspace\kazusa_ai_chatbot_baseline_main' $baselineRevision
git worktree add --detach 'C:\workspace\kazusa_ai_chatbot_v2_prefix' $candidateRevision
venv\Scripts\python.exe -m pytest tests\test_cognition_baseline_harness.py -q
venv\Scripts\python.exe tests\cognition_baseline_comparison.py preflight
venv\Scripts\python.exe tests\cognition_llm_producer_inventory.py audit --revision main --target-root 'C:\workspace\kazusa_ai_chatbot_baseline_main' --matrix tests\fixtures\cognition_llm_producer_matrix.json
venv\Scripts\python.exe tests\cognition_llm_producer_inventory.py audit --revision v2 --target-root 'C:\workspace\kazusa_ai_chatbot_v2_prefix' --matrix tests\fixtures\cognition_llm_producer_matrix.json
venv\Scripts\python.exe tests\cognition_baseline_comparison.py verify-deleted-selector-map
```

Preflight expects zero unmapped production paths, zero mutable external spans,
exact source/profile hashes, and all negative fixtures rejected before any LLM
call. `rg -n "AsunaAIChatbot" tests\fixtures test_artifacts\cognition_core_v2\baseline_regression_hardening\manifest`
expects zero matches.

### Deterministic contract/failure gates

Faults cover repairable syntax, unrepairable JSON, unknown/missing/conflicting
fields, wrong types, bad handles/enums, one invalid row among valid siblings,
over-cap rows, goal/request/progress mismatch, invalid then valid replacement,
two invalid attempts, valid semantic rejection, unavailable execution, replayed
attempt ID, pre-effect failure, and post-effect dialog failure.

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_producer_failures.py tests\test_cognition_llm_producer_budget.py -q
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_action_planning_bugfix.py tests\test_cognition_core_v2_action_authorization.py tests\test_cognition_core_v2_resolver_authorization.py tests\test_cognition_core_v2_integration.py -q
venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_action_selection.py tests\test_l2d_l3_surface_handoff.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py -q
```

Each fault asserts parser, attempts, original error, disposition, route/surface,
state/progress, protected cognition snapshot, action/resolver, DB cardinalities,
dialog/delivery, and trace.

### Production-boundary capability/source gates

Planner-facing actions: `memory_lifecycle_update`,
`trigger_future_cognition`, `future_speak`, `accepted_task_request`,
`accepted_coding_task_request`, `accepted_task_status_check`.
`background_work_request`, `speak`, and `apply_memory_lifecycle_update` must be
absent from planner handles while remaining available to their internal owners.
Resolvers: `local_context_recall`, `public_answer_research`,
`human_clarification`, `approval_preparation`, `self_goal_resolution`.
Sources: `user_message`, `internal_thought`, `self_cognition`,
`scheduled_tick`, `tool_result`.

```powershell
venv\Scripts\python.exe -m pytest tests\test_action_spec_evaluator.py tests\test_cognition_resolver_contracts.py tests\test_stage3_trigger_source_cutover.py -q
```

### Live differential

One invocation runs one case and writes one raw artifact:

```powershell
venv\Scripts\python.exe tests\cognition_baseline_comparison.py run-next --corpus pre_fix_main
venv\Scripts\python.exe tests\cognition_baseline_comparison.py run-next --corpus pre_fix_v2
venv\Scripts\python.exe tests\cognition_baseline_comparison.py compare --blind
venv\Scripts\python.exe tests\cognition_baseline_comparison.py verify-ledger
venv\Scripts\python.exe tests\cognition_baseline_comparison.py run-next --corpus post_fix_v2
venv\Scripts\python.exe tests\cognition_baseline_comparison.py verify-ledger --phase post_fix
```

Parent reads each artifact and updates the engineering ledger before the next
`run-next`. A shell loop or batch live selector is prohibited. `verify-ledger`
fails C07 unless every required repetition contains the accepted-task row,
internal background job, coding-agent router/worker record, terminal coding
result, and repository-map read evidence. It also fails C11-C15 if a frozen
worktree status/tree hash changes or if a coding effect escapes the case's
guarded disposable workspace.

### Overlay and broad gates

- Validate the one-to-one deleted-selector map, then invoke `run-deleted-next`
  repeatedly for main and V2. Each main row runs its frozen native node ID.
  Each V2 row runs the manifest's named current replacement selector or neutral
  public-boundary case with the same stimulus and observable invariant.
- Execute every retained Stage 1-3 cognition sign-off selector recorded by the
  inventory: emotion/crying, abuse, role, continuity, five sources, and bounded
  errors, one live case at a time.
- Run each fully live contract selector separately; never combine them with a
  negative marker expression that could collect zero tests:

```powershell
venv\Scripts\python.exe -m pytest --collect-only -q tests\test_dialog_generator_live_llm_contract.py -m "live_llm"
venv\Scripts\python.exe -m pytest --collect-only -q tests\test_stage3_fresh_database_e2e_live_llm.py -m "live_llm and live_db"
venv\Scripts\python.exe -m pytest tests\test_dialog_generator_live_llm_contract.py::test_live_dialog_generator_deepseek_returns_final_dialog_schema -m "live_llm" -q -s
venv\Scripts\python.exe -m pytest tests\test_dialog_generator_live_llm_contract.py::test_live_dialog_generator_node_accepts_deepseek_output -m "live_llm" -q -s
venv\Scripts\python.exe -m pytest tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_fresh_database_case -m "live_llm and live_db" -q -s
venv\Scripts\python.exe -m pytest tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_user_message_source -m "live_llm and live_db" -q -s
venv\Scripts\python.exe -m pytest tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_internal_thought_source -m "live_llm and live_db" -q -s
venv\Scripts\python.exe -m pytest tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_self_cognition_source -m "live_llm and live_db" -q -s
venv\Scripts\python.exe -m pytest tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_scheduled_tick_source -m "live_llm and live_db" -q -s
venv\Scripts\python.exe -m pytest tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_tool_result_source -m "live_llm and live_db" -q -s
venv\Scripts\python.exe -m pytest tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_group_review_promoted_reflection -m "live_llm and live_db" -q -s
venv\Scripts\python.exe -m pytest tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_media_reply_mentions_preserved -m "live_llm and live_db" -q -s
venv\Scripts\python.exe -m pytest tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_action_affordance_routes -m "live_llm and live_db" -q -s
```

Parent inspects and records each raw artifact before invoking the next command.
- Run:

```powershell
venv\Scripts\python.exe tests\cognition_baseline_comparison.py verify-deleted-selector-map
venv\Scripts\python.exe tests\cognition_baseline_comparison.py run-deleted-next --revision main
venv\Scripts\python.exe tests\cognition_baseline_comparison.py run-deleted-next --revision v2
venv\Scripts\python.exe -m pytest -m "not live_llm and not live_db and not live_internet" -q
venv\Scripts\python.exe -m compileall -q src tests
git diff --check
$forbiddenMatches = @(rg -n "_derive_canonical_action_route|permitted_action_results" src tests)
if ($LASTEXITCODE -gt 1) { throw 'forbidden-symbol scan failed' }
if ($forbiddenMatches.Count -ne 0) { $forbiddenMatches; throw 'obsolete route/result vocabulary remains' }
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_producer_failures.py::test_model_facing_capability_prose_is_chinese_only -q
```

The language selector uses an exact allowlist for schema tokens, URLs,
quotations, and proper names; it rejects English explanatory capability prose
without filtering runtime output.

## Independent Plan Review

Review 1 by read-only subagent `Ampere` rejected the draft with nine blockers:
length/checkpoint/budget contract, unknown scope, wrong delayed-work owner,
undefined runner/manifests/monologue mapping, weak identity guard, incomplete
producer radius, undefined disposition, side-effect atomicity conflict, and
invalid sampling/scoring. This revision addresses each blocker.

Review 2 by the same reviewer rejected the revision for incomplete per-call
budget fields and change-surface groups; non-terminal background evidence;
ambiguous preflight order, clock, state seeds, producer inventory,
disposition/outcome shapes, pending semantics, post-effect service settlement,
deleted-test replay, source entrypoints, coding-workspace isolation, and
scoring summary. Those items are now explicit.

Later review attempts did not reach an approving verdict. The final completed
read-only review rejected the `Create/Modify/Delete/Keep` order and grouped
path rationales. The section now uses canonical `Delete/Modify/Create/Keep`
order, gives every path a reason, and names each new module's public entrypoint.

The user assigned all subsequent work and review to the parent agent. Parent
self-review checks current source, RCA, AGENTS, the development-plan contract,
architecture, exact scope, fixture validity, budgets, commands, and acceptance
mathematics.
The current self-review found no harness blocker; status is `in_progress` while
the configured live environment remains an external prerequisite for Phase 1
completion.

## Independent Code Review

Under the user's single-parent direction, the parent applies this independent
review rubric to its complete diff only after implementation and
verification have paused:

- exact parser/replacement use at every producer in the frozen matrix;
- atomic planner validation, sole route owner, ownership precedence, and no
  compatibility/keyword path;
- disposition/outcome cardinality, action idempotency, L3 truth, and both
  failure-settlement classes;
- identity provenance, external-object preservation, runner isolation, DB
  guards, and anti-false-positive tests;
- Chinese-only explanatory contracts with no runtime filter;
- changed-owner, deleted-baseline, Stage 1-3, broad tests, docs, and evidence.

Every finding records severity, file/line evidence, remediation, rerun, and
verdict. Unresolved critical/high findings block completion.

## Acceptance Criteria

1. Fresh frozen main/pre-fix V2/post-fix V2 artifacts use identical hashed
   semantic inputs, Asuna profile, model/config, clocks, capabilities, and DB
   seeds in isolated processes.
2. All 50 scored fixtures and every negative provenance guard pass preflight.
3. Every changed production path and deleted baseline test maps to a passing
   gate; every producing LLM boundary has a complete matrix row and fault test.
4. External project names/URLs remain exact; actor/addressee/experiencer/quoted
   roles are correct; canonical monologue extraction is unique and revision
   explicit.
5. C07 reaches `accepted_task_request`, internal background work, coding reader,
   and repository read in every matched repetition.
6. Production uses one mode-aware route owner; visible reply retains speech and
   up to three private outcomes.
7. Internal `background_work_request` is absent from planner affordances.
8. Every proposal has one authorization disposition and one terminal outcome;
   L3 claims only that outcome.
9. Invalid candidates receive bounded same-owner replacement and then typed
   fail-closed exhaustion; no invalid data or sibling state escapes.
10. Valid rejection remains distinct from contract failure, availability
    failure, and execution success.
11. Pre-effect failure writes no semantic state/effect/dialog; post-effect L3
    failure preserves the validated cognition snapshot and one effect/result,
    then returns operational error without duplicate effect or character
    dialog.
12. Ready tool results deliver their payload immediately; coding/task/status/
    reminder/approval/cancel cases bind actual durable state.
13. C11-C15 change only their guarded disposable case workspaces; both frozen
    target worktrees retain identical clean status and tree hashes before and
    after every repetition.
14. All public action/resolver kinds and five sources pass production-boundary
    gates; private sources never deliver without permission.
15. Chinese-context explanatory/model free text is Chinese, with exact allowed
    tokens preserved and no runtime censorship/rewrite.
16. Restored affinity/boundary/reflection/internal/proactive coverage passes,
    and all retained Stage 1-3 sign-off cases pass one at a time with Asuna.
17. Every observed comparison difference has three matched runs; every pre-fix
    V2 loss closes on the same post-fix repetition schedule.
18. V2 loses no dimension, wins a majority of improvable dimensions, closes at
    least 20% aggregate headroom, and has zero hard-gate failures.
19. No unapproved LLM route, unconditional call, sampling/cap change, DB schema,
    production-data operation, compatibility shim, keyword router, or content
    filter exists.
20. Focused and full non-live pytest gates collect the intended nonzero node
    sets; compile, static, documentation, and parent code/evidence self-review
    gates pass.
21. Final user artifact contains only input, private monologue, and visible
    dialog per turn; the user explicitly accepts it before completion.

## Risks

- Model variance: matched three-run escalation and hard gates prevent one-sample
  regression claims.
- Baseline defect: main is a floor, not an oracle; blind scoring cannot excuse a
  hard invariant.
- Broad unknown radius: Phase 1 owner mapping and amendment approval prevent
  speculative production edits.
- Double effect: canonical attempt IDs and post-effect failure tests enforce
  exactly once.
- Ceiling scores: headroom-relative scoring remains achievable while requiring
  majority wins where improvement is possible.
- Prompt overfit: fixtures contain no route/worker/emotion/outcome instruction;
  prompts state domain ownership only.
- Chinese enforcement drift: tests report raw failures; runtime never filters,
  suppresses, or rewrites.
- Test DB risk: exact `.env` target validation, guarded `_test_` names,
  before/after counts, and explicit cleanup protect external data.

## Approval Boundary

This draft authorizes no production change. Initial approval authorizes Phase 1
test/evidence execution only. Production work requires the reviewed Phase 1
amendment and a second explicit user approval.

## Execution Evidence

- Planning audit: current SHA, `origin/main` SHA, 407-file radius, nine deleted
  baseline tests, Case 10 RCA, action/resolver/surface/service code, Stage 3
  residuals, prior 20-turn review, profile/history hashes.
- Plan review 1: `Ampere`, rejected with nine blockers; no files edited by the
  reviewer.
- Plan review 2: `Ampere`, rejected with the blockers recorded under
  Independent Plan Review; no files edited by the reviewer.
- Later read-only review sessions returned no verdict and were closed; the last
  completed plan-only review rejected only change-surface order and grouped
  rationales, now corrected.
- User assigned subsequent review to the parent. Parent self-review passed
  mandatory section/order/length, exact path/symbol, SHA/hash,
  marker/selector collection, command, schema, scoring, approval-boundary, and
  diff checks.
- Collection-only evidence: both dialog live nodes and all nine Stage 3
  live-LLM/live-DB nodes collect under explicit marker overrides; no live test
  executed during planning.
- No production code or test behavior changed while drafting this plan.
- Phase 1 execution: fixed revisions and input hashes verified; detached
  worktrees created; 50-case manifest expanded to 82 scheduled repetitions;
  owner matrix covered every changed production path; deleted-selector map
  passed; producer audit matched main 130/130 and V2 125/125 call sites with
  zero unmatched or duplicate rules; forbidden `AsunaAIChatbot` fixture gate
  passed; negative provenance fixtures rejected wrong experiencer,
  addressee, chronology, and URL mutation while guarding quoted third-party
  identity and non-canonical semantic output; 23 deterministic
  harness/fixture tests passed; preflight passed.
- First live worker invocation (`pre_fix_main`, `C01`, repetition 1) produced
  a `blocked_environment` artifact before importing the target runtime because
  the explicit process environment lacked MongoDB and required LLM route
  variables. That artifact is superseded by the user-directed `.env` boundary;
  the controller now loads the configured values and passes them explicitly.
- The same `C01/r1` run was repeated for `pre_fix_v2` and produced the same
  typed environment block with the V2 target SHA. `compare --blind` now
  reports zero eligible pairs and two blocked artifacts; `verify-ledger`
  rejects both incomplete corpora because 81 of 82 expected executions remain
  missing per revision.
- One bounded independent read-only harness review was performed after parent
  self-review. It found hard-gate, ledger-completeness, revision-SHA, DB
  cleanup, chronology, external-span, and profile-equality gaps. Parent fixed
  these in test-only files; 23 focused deterministic/fixture tests and
  preflight pass, and the hard-gate/blocked-pair guards now fail closed.
- The broad non-live selector produced 2,693 passed, 2 skipped, and 784
  deselected tests before one existing service-lifespan failure. The isolated
  traceback shows `_test_kazusa_core_v2` contains the Kazusa singleton while
  the service seed is Asuna; this shared test-database state is outside the
  Phase 1 test-only write set and was not altered.
- `compileall`, `verify-deleted-selector-map`, and `git diff --check` pass.
  The planned obsolete-vocabulary scan still finds the existing production
  `_derive_canonical_action_route` and `permitted_action_results` symbols;
  removing or replacing them remains a Phase 2 production gate.
- The configured `_test_kazusa_core_v2` database was observed to contain the
  Kazusa singleton while the frozen profile is Asuna. Before fresh live
  execution, the singleton/database will be snapshotted, the exact configured
  test database reset, and Asuna seeded through the project seed path. Phase 1
  remains in progress until fresh main/V2 artifacts pass their hard gates.
- Priority-zero C18 closure: the focused real-LLM planner/authorizer selector
  passed with raw `action_requests: []`, `resolver_requests: []`, and
  `goal_resolution: "answerable_now"` under the frozen unavailable scheduler
  and background-worker limits. The corrected post-fix C18 artifact
  `test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/C18/r1.json`
  passed all five runtime-compatible hard gates. Its raw evidence has zero
  accepted-task, background-work, and schedule deltas, one preference-memory
  delta, and a truthful Chinese monologue/dialog pair; valid paraphrase was
  accepted without adding wording rules.

### 2026-07-24 handoff checkpoint: C13 priority-zero execution record

This checkpoint records the execution sequence and the current stopping point
for the next parent agent. The plan remains `in_progress`; the final corpus,
full quality review, and sign-off gates remain open. The latest ledger is
`post_fix_v2: 76/82 technical pass, 6/82 technical fail`, with the six failed
artifacts owned by C13-C15 and O03.

#### Observation sequence

1. The original C13 E2E fixture declared
   `state_seed.coding_run=baseline-run-013`, but the worker did not materialize
   that seed. The captured database counts were zero accepted tasks and zero
   background jobs before the turn. That artifact could not exercise a bound
   blocker response. The test harness now materializes the declared run as an
   active `coding_run_context.v1` accepted-task row and has a deterministic
   guard for the run reference, status, action set, and scope.

2. The first corrected focused action replay exposed an expectation error in
   the focused test: the registry classified the unavailable worker as a
   queue-only coding owner, while the test expected a blocked/no-action result.
   A repeat before the prompt correction returned authorizer `c1=false`, which
   exposed a producing-stage contract contradiction rather than a stable
   semantic decision. The owner boundary was then made explicit: a bound
   lifecycle continuation may be recorded and queued with a pending result;
   an unbound start still requires a healthy owner.

3. The focused action test is now green through the canonical connector. Its
   latest raw authorizer output is `{"decisions":{"c1":true}}`; the parsed
   request is one `accepted_coding_task_request` with
   `decision=respond_to_blocker` and
   `context_ref=coding_run:baseline-run-013`. The result retains
   `goal_resolution=answerable_now`, bounded progress, and the queue-only
   runtime limit. No `start` action is selected. Evidence:
   [`C13 action focused trace`](../../../test_artifacts/llm_traces/cognition_core_v2_action_planning_live_llm__c13_seeded_blocker_owner_limit__20260724T064059915924Z.json).

4. The matched C13 E2E rerun was executed only after that focused stage was
   green. The action was bound correctly and settled as `status=pending`,
   `accepted_task_state=scheduled`, with
   `coding_run_ref=baseline-run-013`. The worker remained unavailable, the
   seeded run remained `blocked`, accepted-task count changed from 1 to 2,
   and one background-work row was queued. The sole technical gate failure was
   `coding_run_unblocked`, so this result is evidence for a queue-only versus
   immediate-unblock contract review. The visible dialog rendered:

   - `收到！我会直接使用现有的虚拟环境运行聚焦测试，绝对不会安装任何新依赖。`
   - `现在这就开始准备执行，马上为您反馈结果！`

   The second line overstates immediate execution and feedback relative to the
   pending action. The raw graph and database deltas support a dialog-owner
   investigation; they do not support an E2E-specific wording rule. Evidence:
   [`C13 matched E2E`](../../../test_artifacts/cognition_core_v2/baseline_regression_hardening/post_fix_v2/C13/r1.json).

5. The owning content-plan replay was then run against the same C13 E2E graph
   and pending action result. The content-stage contract was amended to map
   `pending` and `scheduled` to the positive Chinese state
   `已记录、已排队、待执行`, while reserving completion wording for
   `executed`. The latest model plan says the coding worker has not actually
   run and frames the state as `准备开始` or `即将执行`. The focused guard is
   still red because it accepts only a narrow queue-marker list and treats
   valid pending paraphrases such as `准备开始`/`即将执行` as execution claims.
   This is a test-guard calibration item, not yet a confirmed content-stage
   semantic defect. Evidence:
   [`C13 surface focused trace`](../../../test_artifacts/llm_traces/cognition_core_v2_surface_owner_live_llm__c13_pending_queue_only_boundary__20260724T064611363261Z.json).

#### Change-attempt ledger

| Attempt | Owning boundary | Evidence and current disposition |
|---|---|---|
| Fixture correction | `tests/cognition_baseline_worker.py` and deterministic hardening tests | Materializes C13-C15 declared coding runs; focused seed guards pass. |
| Queue-only owner correction | `persona_supervisor2_cognition.py`, `action_selection.py`, `action_authorization.py` | Focused C13 planner/authorizer test is green; the bound continuation is pending and truthful. |
| Pending surface contract | `surface_stages.py` and deterministic prompt assertions | The raw content plan preserves the worker-unavailable caveat; focused semantic guard requires paraphrase-aware calibration. |
| Final dialog boundary | `nodes/dialog_agent.py` | Focused replay has not yet been executed. The C13 E2E output is the red reproduction target for this owner. |

#### Handoff execution gates

The next agent resumes at the following ordered gates:

1. Update the C13 content-plan semantic guard to accept Chinese pending
   paraphrases (`待执行`, `已记录`, `已排队`, `准备开始`, `即将执行`, or an
   equivalent) together with a worker-not-yet-run caveat, and reject only
   claims of completed execution or an immediate feedback guarantee. Rerun the
   focused real-LLM content-stage test one case at a time and inspect its raw
   output.
2. Add and run a focused real-LLM dialog-generator replay using the C13
   artifact's canonical `text_surface_output_v2`, Asuna profile, and pending
   action result. Record the raw output, parsed dialog, and semantic judgment.
   If the dialog stage repeats the immediate-execution promise, add one
   positive pending/scheduled mapping to its existing Chinese contract, then
   rerun this focused test until the semantic guard is green.
3. Rerun matched C13 once the content and dialog owning stages are green.
   Review monologue/dialog meaning and side-effect evidence. Independently
   review whether `coding_run_unblocked` means immediate durable transition or
   whether a narrowly typed queue-only outcome is the correct runtime contract;
   amend that gate only after the evidence proves the contract decision.
4. Apply the same priority-zero sequence to C14 and C15 using their
   materialized seeded runs. Then investigate O03 at the media observation
   owner. Every E2E rerun remains one case at a time and quality review accepts
   valid Chinese paraphrases.
5. After all focused stages and matched residual cases are green, rerun the
   complete frozen corpus, recompute the baseline comparison, perform the
   parent code/evidence self-review, and update the plan before sign-off.

The working tree contains the approved hardening changes and additional
uncommitted user work. Preserve the current files and inspect the complete
diff before the next production amendment; the frozen comparison remains
`origin/main@8f834bf87a83ee42aca804934fb44af63788420c` versus
`HEAD@0c2e929d51ac80c4519f564b61cbf8949efcca3d`.
