# self cognition group digest context evidence bugfix plan

## Summary

- Goal: produce evidence for whether wider group digest context fixes ambient
  self-cognition misreads, then measure whether pre-cognition conversation
  evidence adds value.
- Plan class: large
- Status: in_progress
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: compatible source-hydration expansion; no action
  gate, silence rule, adapter behavior, persistence migration, or dialog
  change.
- Highest-risk areas: widening context into action pressure, over-structuring
  LLM-to-LLM handoff, relying on optional resolver RAG after cognition has
  already misread the scene, and hiding the value comparison behind one green
  test.
- Acceptance criteria: the plan implements and tests all four requested parts:
  expanded digest input, real LLM test of that expansion, pre-cognition
  conversation evidence from `summary`, and real LLM test of that evidence
  value.

## Context

The reproduced failure case is:

```text
group_activity_window:scope_e13fdf80a90b:2026-06-12T00:15:00+00:00:2026-06-12T00:30:00+00:00
```

The reviewed 15-minute window before Kazusa spoke contained only:

```text
W: 你这拉稀还能骑就不对
W: 正常拉稀是没力气的
```

The source rows had no structured address, mention, or reply-to-Kazusa
metadata. The labels correctly said `bot_addressing=ambient_group_context`.

The missing antecedent was immediately before the reviewed window:

```text
温格高艾菲波加查: 现在也是受凉额
温格高艾菲波加查: 山里爬坡 出汗 下坡 那山风呼呼吹肚子上
温格高艾菲波加查: 穿衣服又热 骑行服又冷
温格高艾菲波加查: 不穿骑行裤屁股完蛋
1816: 那应该就是肠胃保护了
W: 你这拉稀还能骑就不对
W: 正常拉稀是没力气的
```

Under the production backend, the current real LLM test reproduces the failure:
L2a rewrites W's raw `你` into `调侃我` / `拿我开涮`, L2c promotes that to
`BANTAR`, and L2d emits a visible `speak`.

The user's confirmed design direction:

1. Keep `group_digest` as the existing digest, but expand its input scope to
   the same exposure size used by relevance-style channel context.
2. Add a separate string-only `summary`, intentionally short,
   describing general topics like relevance's `channel_topic`.
3. Feed `summary` to conversation search to retrieve possible
   relevant topics from even earlier conversation.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, or reviewing
  this plan.
- `local-llm-architecture`: load before changing prompt-facing context, LLM
  prompts, cognition source packets, or RAG/evidence flow.
- `debug-llm`: load before running or reviewing live LLM evidence.
- `py-style`: load before editing Python production or test files.
- `cjk-safety`: load before editing Python files that contain CJK prompt text.
- `test-style-and-execution`: load before changing or running tests.

## Mandatory Rules

- Do not edit production code from this draft until the user explicitly
  approves implementation and the plan status is `approved` or `in_progress`.
- Use `venv\Scripts\python`; use `apply_patch`; check `git status --short`;
  do not read `.env`.
- LLM-to-LLM handoff for this plan is string-only. Do not add arrays such as
  `main_participants`, `literal_anchors`, `addressing_assessment`, topic
  objects, nested summary structures, or new model-facing schemas unless the
  user explicitly changes the contract.
- `group_digest` remains the current digest concept. The change is its input
  scope, not a replacement of the digest contract.
- `summary` is one intentionally short string. It describes the
  general visible topic only; it must not decide who should speak, whether
  Kazusa should respond, or whether a boundary was crossed.
- `conversation_progress.group_scene_digest` must keep the existing shape
  `{"digest": str}`. Store `summary` as a separate optional string field, not
  nested inside `group_scene_digest`.
- Conversation evidence is evidence only. It must not become persona stance,
  action recommendation, deterministic target selection, response ratio, or
  silence gate.
- Keep typed message-envelope authority visible. `ambient_group_context` must
  remain in the prompt-facing source packet when no structured address exists.
- Pre-cognition conversation evidence is allowed only for this group-review
  source hydration path. Do not change normal live `/chat` RAG behavior.
- Real LLM tests must run one case at a time. Inspect and summarize each trace
  in a human-authored review artifact before claiming quality.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing. After signing off any major checklist stage,
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.

## Must Do

- Preserve the 15-minute group activity window as the reviewed source and
  idempotency unit.
- Expand the input rows used by `group_digest` to a same-channel exposure that
  matches the rows relevance exposes as `chat_history_wide`: chronological rows
  from the same channel, ending at the latest reviewed-window message,
  excluding future rows, capped by `CONVERSATION_HISTORY_LIMIT`.
- Add `summary` as a separate short string generated by the group
  digest module without changing the existing `group_scene_digest.digest`
  contract.
- Use `summary` as the plain-string task seed for pre-cognition
  conversation evidence search over earlier same-channel conversation before
  the reviewed window start.
- Build evidence artifacts that compare:
  1. baseline current behavior,
  2. expanded digest only,
  3. expanded digest plus conversation evidence.
- Include real LLM output for the laxi failure case in the evidence artifacts.

## Deferred

- No deterministic suppression based on body text, pronouns, `你`, topic
  keywords, or speaker names.
- No new action router gate, dialog prompt change, adapter change, delivery
  change, persistence migration, or new database collection.
- No rich LLM summary schema.
- No change to normal live-turn relevance/decontextualizer behavior.
- No reliance on second-cycle optional RAG after L2d. The evidence path is
  pre-cognition source hydration for group review only.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Group review source id | compatible | Keep the 15-minute window source id and ledger behavior unchanged. |
| `group_digest` | compatible | Keep existing digest output; expand only the input rows used to write it. |
| `summary` | compatible | Add one optional string-only source-hydration field. Omit it when unavailable. |
| Conversation evidence | compatible | Add bounded pre-cognition evidence only to group-review source packets. |
| Cognition/action/dialog | compatible | Preserve cognition graph, action router, dialog, and delivery contracts. |
| Tests | bigbang | Add focused tests and live LLM evidence for each planned stage. |

## Target State

```text
selected 15-minute group window
  -> same-channel exposure rows ending at latest reviewed row
  -> existing group_digest string over expanded exposure
  -> short summary string
  -> bounded pre-cognition conversation evidence from summary
  -> self-cognition source packet
  -> existing cognition resolver unchanged
```

The model-facing handoff remains text-first. `group_digest` and
`summary` are strings. Conversation evidence enters as existing
projected evidence text, not as a new decision schema.

## Design Decisions

| Topic | Decision | Evidence / Rationale |
|---|---|---|
| Reviewed unit | Keep 15-minute activity window | Reflection phase and self-cognition ledgers already use it for bounded review and idempotency. |
| Digest input scope | Use the last `CONVERSATION_HISTORY_LIMIT` same-channel rows ending at the latest reviewed-window row | Relevance already exposes `chat_history_wide` to the model; current 15-minute bucket cut off the antecedent. This uses the prompt exposure size, not relevance's internal 180-second noise-scoring suffix. |
| Digest output | Keep existing `digest` string | Current self-cognition ICD says group digest is optional one-string source hydration. |
| Summary output | Add one short string `summary` | Mirrors relevance `channel_topic` style without introducing structured LLM-to-LLM payloads. |
| Retrieval seed | Use only the short summary string plus deterministic scope limits | Avoids overdesigned LLM-to-LLM schemas while giving conversation search a topic. |
| Retrieval timing | Pre-cognition | The reproduced failure happens before the model asks for RAG; relying on L2d to request RAG is not sufficient. |

## Contracts And Data Shapes

The source packet may include:

```python
conversation_progress["group_scene_digest"] = {
    "digest": str,
}
conversation_progress["summary"] = str
```

Rules:

- `digest` is the existing observational digest string.
- `summary` is a short topic string.
- Both strings are optional; omitted strings must not block the case.
- No additional model-facing fields are allowed under this plan.
- Deterministic code may use normal runtime metadata for scoping and exclusion,
  but it must not expose raw internal ids as model-facing digest content.

Conversation evidence task string format:

```text
Find earlier same-channel conversation before <reviewed window start local time> that helps
explain this group topic: <summary>
```

The executor may add deterministic non-model constraints such as channel,
`timestamp < reviewed_window.window_start`, reviewed-window source-row
exclusion, future-row exclusion, and bounded result limits. The search must not
retrieve the reviewed W rows as antecedent evidence.

## LLM Call And Context Budget

- Stage 1 adds no new source type. It widens the rows passed into the existing
  group digest path.
- Stage 1 adds one short `summary` string. It may be generated in
  the group digest module, but `group_scene_digest` storage remains
  `{"digest": str}`. Any parser contract used to obtain the summary is internal
  to the digest module and must not become model-facing source structure.
- Stage 3 adds one bounded pre-cognition conversation evidence call for
  group-review cases where `summary` is non-empty.
- Normal live `/chat` relevance, decontextualizer, resolver RAG, dialog, and
  adapter delivery budgets stay unchanged.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/reflection_cycle/activity_windows.py`
  - Preserve reviewed-window behavior.
  - Add or expose a helper that builds digest exposure rows from same-channel
    scope messages in chronological order, ending at the latest reviewed-window
    message, capped by `CONVERSATION_HISTORY_LIMIT`, and excluding future rows.

- `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`
  - Keep the existing digest role.
  - Accept expanded exposure rows for prompt input.
  - Add one short string output named `summary`.
  - Keep output validation strict and string-only.

- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Keep `conversation_progress.group_scene_digest = {"digest": str}`.
  - Attach `conversation_progress.summary = str` when available.
  - Add the bounded pre-cognition conversation evidence call in Stage 3.

- `src/kazusa_ai_chatbot/self_cognition/projection.py`
  - Render the summary and pre-cognition evidence as source context when
    present, preserving `ambient_group_context` labels.

### Test

- `tests/test_reflection_cycle_group_scene_digest.py`
  - Deterministic digest exposure and summary validation tests.

- `tests/test_self_cognition_group_review_source.py`
  - Source packet shape tests for expanded digest and optional evidence.

- `tests/test_self_cognition_response_sensitivity_live_llm.py`
  - Real LLM laxi case evidence modes:
    `baseline_15m_digest`, `expanded_digest_only`, and
    `expanded_digest_plus_conversation_evidence`.

## Overdesign Guardrail

Actual problem:

- A 15-minute group-review source can start mid-thread, so L2a may treat raw
  pronouns in the reviewed rows as Kazusa-directed even when typed metadata is
  ambient.

Minimal change:

- Widen only group-review source hydration, add one short topic string, and
  add one bounded pre-cognition conversation evidence lookup from that string.

Ownership boundaries:

- Deterministic code owns same-channel scope, row caps, reviewed-window source
  identity, timestamp bounds, future-row exclusion, and reviewed-row exclusion.
- Digest LLM owns only the observational digest and short topic string.
- Conversation evidence owns what earlier messages say and nearby relation.
- Cognition owns whether the evidence gives Kazusa a reason to speak.

Rejected complexity:

- No rich summary structures, no participant arrays, no literal-anchor arrays,
  no addressing object, no new RAG worker, no action gate, no dialog change, no
  deterministic pronoun/keyword suppression, and no fallback prompt retries for
  parsed-but-invalid summary output.

Evidence threshold:

- Stage 2 and Stage 4 must each produce trace-backed real LLM review artifacts
  before any quality claim is made.

## Agent Autonomy Boundaries

- Execution agents may adjust helper names and exact private function placement
  inside the listed files to fit existing code style.
- Execution agents may not add new model-facing fields, new public schemas, new
  runtime config flags, or new behavior outside group-review source hydration
  without user approval.
- Stages 1 through 4 are all in scope once this plan is approved. Stage 2
  evidence is a comparison checkpoint, not a stop condition.
- Execution agents must not decide that Stage 3 is unnecessary. The user asked
  for evidence comparing expanded digest alone against expanded digest plus
  conversation evidence.

## Execution Model

- Execution is parent-led and test-contract-first.
- The parent agent owns baseline reproduction, focused tests, live LLM evidence
  artifacts, integration verification, execution evidence, and final sign-off.
- A production-code subagent may edit only the approved production files in
  `Change Surface` after the parent establishes the focused deterministic and
  live LLM test contracts.
- The independent code-review subagent reviews the final diff and evidence
  only. It must not implement fixes.
- If native subagent capability is unavailable during execution, stop unless
  the user explicitly approves fallback inline execution.

## Implementation Order

### Stage 0 - Baseline Evidence

1. Add or use a baseline evidence mode that forces the legacy 15-minute digest
   input and disables pre-cognition conversation evidence.
2. Run the current laxi live LLM reproduction one case at a time.
3. Save the trace path with mode label `baseline_15m_digest`.
4. Write or update a human-authored review artifact showing:
   - raw source rows,
   - `group_digest`,
   - L2a output,
   - L2c output,
   - L2d action result.

Verification command:

```powershell
venv\Scripts\python.exe -m pytest tests/test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_laxi_baseline_15m_digest_reproduces_speak -q -s -m "live_llm and live_db"
```

Expected baseline: under production backend, the baseline test passes by
reproducing visible speech. The visible speech is expected reproduction
evidence, not a completed fix gate.

### Stage 1 - Expand Group Digest Input Scope

1. Add deterministic tests proving the digest prompt payload receives the last
   `CONVERSATION_HISTORY_LIMIT` same-channel rows ending at the latest
   reviewed-window row, includes earlier 温格 context for the laxi fixture, and
   excludes future rows after the latest reviewed-window row.
2. Implement the minimal helper/wiring so `group_digest` sees expanded exposure
   rows while the reviewed source id remains the 15-minute window.
3. Add one separate short `summary` string output and validation.
4. Keep `summary` string-only and intentionally short.

Focused deterministic verification:

```powershell
venv\Scripts\python.exe -m pytest tests/test_reflection_cycle_group_scene_digest.py -q
venv\Scripts\python.exe -m pytest tests/test_self_cognition_group_review_source.py -q
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/reflection_cycle/activity_windows.py src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py src/kazusa_ai_chatbot/self_cognition/sources.py src/kazusa_ai_chatbot/self_cognition/projection.py
```

### Stage 2 - Test Stage 1 With Real LLM Output

1. Run the laxi live LLM case one case at a time with mode label
   `expanded_digest_only`.
2. Inspect the trace.
3. Write a human-authored review artifact comparing baseline vs Stage 1:
   - digest text before/after,
   - `summary`,
   - L2a pronoun interpretation,
   - L2c intent,
   - L2d action count,
   - visible speak selected or not.

Verification command:

```powershell
venv\Scripts\python.exe -m pytest tests/test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_laxi_expanded_digest_only_records_output -q -s -m "live_llm and live_db"
```

Decision evidence required before Stage 3 implementation begins:

- If Stage 1 no longer selects visible speech and L2a no longer rewrites W's
  `你` as Kazusa, record that expanded digest may be enough for that run.
- If Stage 1 still selects visible speech or still misattributes `你`, record
  the exact L2 stage where the misread remains.
- Continue to Stage 3 after recording this evidence because the approved plan
  requires measuring the added value of conversation evidence.

### Stage 3 - Add Conversation Evidence From Summary

1. Add deterministic tests for building the conversation evidence task string
   from `summary`.
2. Add the bounded pre-cognition conversation evidence call for group-review
   cases with a non-empty summary.
3. Attach projected conversation evidence to the self-cognition source packet
   as evidence text.
4. Keep deterministic scope constraints outside the model-facing task string:
   same channel, `timestamp < reviewed_window.window_start`, bounded results,
   exclude reviewed-window source rows and all future rows.
5. Add a deterministic test proving the laxi Stage 3 retrieval cannot return
   the two reviewed W rows as antecedent evidence.

Focused deterministic verification:

```powershell
venv\Scripts\python.exe -m pytest tests/test_self_cognition_group_review_source.py -q
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/self_cognition/sources.py src/kazusa_ai_chatbot/self_cognition/projection.py
```

### Stage 4 - Test Stage 3 With Real LLM Output

1. Run the laxi live LLM case one case at a time with mode label
   `expanded_digest_plus_conversation_evidence`.
2. Inspect the trace.
3. Write a human-authored review artifact comparing Stage 1 vs Stage 3:
   - `summary` used as retrieval seed,
   - conversation evidence selected summary,
   - whether evidence retrieves earlier 温格 riding/diarrhea context,
   - L2a pronoun interpretation,
   - L2c intent,
   - L2d action count,
   - visible speak selected or not,
   - added latency and retrieval count.

Verification command:

```powershell
venv\Scripts\python.exe -m pytest tests/test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_laxi_expanded_digest_with_evidence_records_output -q -s -m "live_llm and live_db"
```

## Progress Checklist

- [x] Stage 0 - baseline reproduction evidence recorded.
  - Covers: current failing behavior under production backend.
  - Verify: `baseline_15m_digest` trace selects visible speech.
  - Evidence: trace path and human-authored review artifact.
  - Sign-off: Codex / 2026-06-12. The baseline live replay passed by
    reproducing the historical visible-speech failure under
    `baseline_15m_digest`.

- [x] Stage 1 - expanded digest input and summary implemented.
  - Covers: expanded digest exposure, string-only summary, source packet
    rendering.
  - Verify: deterministic tests and `py_compile` pass.
  - Evidence: changed files, test output, prompt payload sample.
  - Sign-off: Codex / 2026-06-12. Deterministic checks passed:
    `test_reflection_cycle_group_scene_digest.py` 19 passed,
    `test_self_cognition_group_review_source.py` 18 passed, and `py_compile`
    passed.

- [x] Stage 2 - expanded digest real LLM evidence recorded.
  - Covers: laxi case with expanded digest only.
  - Verify: one-at-a-time live LLM run, trace inspection, review artifact.
  - Evidence: baseline vs Stage 1 comparison.
  - Sign-off: Codex / 2026-06-12. Expanded digest improves the source packet
    by exposing the 温格高艾菲波加查 / 1816 antecedent thread. The final-code
    Stage 2 run stayed silent with no action specs, but earlier Stage 2 runs
    were mixed, so expansion alone should be treated as trace-backed evidence
    for this case rather than a universal stability proof.

- [x] Stage 3 - conversation evidence from summary implemented.
  - Covers: summary-derived task string, bounded pre-cognition evidence, source
    packet rendering.
  - Verify: deterministic tests and `py_compile` pass.
  - Evidence: changed files, test output, retrieval task string sample.
  - Sign-off: Codex / 2026-06-12. `summary` now seeds a bounded
    same-channel, pre-window conversation evidence request. Deterministic
    tests pass and prove the source packet receives rendered evidence.

- [x] Stage 4 - conversation evidence real LLM value recorded.
  - Covers: laxi case with expanded digest plus conversation evidence.
  - Verify: one-at-a-time live LLM run, trace inspection, review artifact.
  - Evidence: Stage 1 vs Stage 3 comparison and value judgment.
  - Sign-off: Codex / 2026-06-12. The final-code Stage 4 replay retrieved and
    exposed relevant earlier diarrhea evidence, including the `蒙脱石散` /
    `拉水` antecedent, and stayed silent with no action specs. Compared with
    Stage 2, it produced cleaner `EVADE` /旁观 reasoning.

- [x] Independent Code Review gate completed.
  - Scope: full diff, plan compliance, prompt safety, string-only handoff,
    test quality, real LLM evidence quality, and regression risk.
  - Verify: reviewer findings recorded and all accepted findings remediated or
    explicitly deferred with user approval.
  - Evidence: review artifact and rerun verification output.
  - Sign-off: Codex / 2026-06-12. Independent gpt-5.5 xhigh review completed;
    all findings were remediated and focused verification was rerun.

## Verification

Required deterministic checks:

```powershell
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/reflection_cycle/activity_windows.py src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py src/kazusa_ai_chatbot/self_cognition/sources.py src/kazusa_ai_chatbot/self_cognition/projection.py tests/test_reflection_cycle_group_scene_digest.py tests/test_self_cognition_group_review_source.py tests/test_self_cognition_response_sensitivity_live_llm.py
venv\Scripts\python.exe -m pytest tests/test_reflection_cycle_group_scene_digest.py -q
venv\Scripts\python.exe -m pytest tests/test_self_cognition_group_review_source.py -q
```

Required real LLM checks:

```powershell
venv\Scripts\python.exe -m pytest tests/test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_laxi_expanded_digest_only_records_output -q -s -m "live_llm and live_db"
venv\Scripts\python.exe -m pytest tests/test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_laxi_expanded_digest_with_evidence_records_output -q -s -m "live_llm and live_db"
```

Real LLM runs must be one case at a time. Each run must have a trace and a
human-authored review artifact. Trace payloads and artifact names must include
one of these mode labels:

- `baseline_15m_digest`
- `expanded_digest_only`
- `expanded_digest_plus_conversation_evidence`

## Independent Code Review

Before completion, run an independent code review against:

- approved plan scope,
- string-only LLM-to-LLM handoff,
- no deterministic keyword/pronoun suppression,
- no action/dialog/delivery behavior changes,
- deterministic source scoping and future-row exclusion,
- real LLM evidence artifacts,
- focused deterministic test coverage.

The reviewer must not implement fixes. The parent agent records findings,
remediates accepted findings inside this plan's change surface, reruns affected
verification, and records final review status.

## Acceptance Criteria

- Stage 1 produces real LLM evidence showing whether expanded digest alone
  prevents the laxi misread.
- Stage 4 produces real LLM evidence showing whether pre-cognition
  conversation evidence adds value over expanded digest alone.
- The final evidence includes raw source summary, digest text, summary string,
  retrieved conversation evidence when applicable, L2a/L2c/L2d outputs, and
  action count.
- No model-facing structured summary fields are introduced.
- No production behavior outside group-review source hydration changes.

## Execution Evidence

- Baseline:
  - Command: `venv\Scripts\python.exe -m pytest tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_laxi_baseline_15m_digest_reproduces_speak -q -s -m "live_llm and live_db"`
  - Result: 1 passed. `observed_user_visible_speak=true`, `action_spec_count=1`.
  - Trace: `test_artifacts/llm_traces/self_cognition_group_response_sensitivity_live_llm_baseline_15m_digest__group_activity_window_scope_e13fdf80a90b_2026-06-12T00_15_00_00_00_2026-06-12T00_30_00_0.json`
  - Review artifact: `test_artifacts/self_cognition_laxi_stage0_stage4_review_20260612.md`
- Stage 1 deterministic:
  - Command: `venv\Scripts\python.exe -m py_compile ...`; `venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_group_scene_digest.py -q`; `venv\Scripts\python.exe -m pytest tests\test_self_cognition_group_review_source.py -q`
  - Result: `py_compile` passed; digest tests 19 passed; group-review source tests 18 passed.
- Stage 1 live LLM:
  - Command: `venv\Scripts\python.exe -m pytest tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_laxi_expanded_digest_only_records_output -q -s -m "live_llm and live_db"`
  - Result: 1 passed. `observed_user_visible_speak=false`, `action_spec_count=0`.
  - Trace: `test_artifacts/llm_traces/self_cognition_group_response_sensitivity_live_llm_expanded_digest_only__group_activity_window_scope_e13fdf80a90b_2026-06-12T00_15_00_00_00_2026-06-12T00_30_00___20260612T024145542440Z.json`
  - Review artifact: `test_artifacts/self_cognition_laxi_stage0_stage4_review_20260612.md`
  - Stage 1 quality judgment: expanded digest alone improves the prompt
    context. The final-code run stayed silent, but earlier runs were mixed, so
    this is positive evidence for the original case rather than a global
    stability proof.
- Stage 3 deterministic:
  - Command: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\self_cognition\sources.py tests\test_self_cognition_group_review_source.py tests\test_self_cognition_response_sensitivity_live_llm.py`; `venv\Scripts\python.exe -m pytest tests\test_self_cognition_group_review_source.py -q`
  - Result: `py_compile` passed; group-review source tests 18 passed.
- Stage 4 live LLM:
  - Command: `venv\Scripts\python.exe -m pytest tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_laxi_expanded_digest_with_evidence_records_output -q -s -m "live_llm and live_db"`
  - Result: 1 passed as an evidence-recording test. `observed_user_visible_speak=false`, `action_spec_count=0`.
  - Trace: `test_artifacts/llm_traces/self_cognition_group_response_sensitivity_live_llm_expanded_digest_plus_conversation_evidence__group_activity_window_scope_e13fdf80a90b_2026-06-12T00_15_00_00_0__20260612T024117158447Z.json`
  - Review artifact: `test_artifacts/self_cognition_laxi_stage0_stage4_review_20260612.md`
  - Stage 4 quality judgment: summary-seeded conversation evidence retrieved
    and exposed relevant earlier context, including the `蒙脱石散` / `拉水`
    antecedent. The final-code run stayed silent and produced cleaner
    `EVADE`/旁观 reasoning.
- Independent code review:
  - Reviewer: subagent `019eb9aa-5554-73c1-bf95-6cc3204dba93`, gpt-5.5 xhigh.
  - Findings: inclusive pre-window `to_timestamp`, invalid optional summary
    suppressing valid digest, ambiguous digest prompt wording, and plan
    lifecycle/evidence text inconsistencies.
  - Remediation: search end now subtracts one microsecond from reviewed window
    start; invalid optional summaries are omitted while valid digests remain;
    prompt wording now says `digest` required and `summary` optional; plan
    registry and baseline wording were updated.
  - Final status: all findings addressed; deterministic and live comparison
    tests rerun.

## Risks

- Expanded digest may be enough for the laxi case but insufficient for older or
  slower topic continuations.
- Conversation search may retrieve nearby but irrelevant earlier discussion if
  the summary is too generic.
- Adding pre-cognition evidence may improve correctness but increase
  reflection worker latency.
- Wider digest input may include Kazusa's earlier assistant line and increase
  self-reference pressure if the digest prompt is not clear about reviewed
  window versus prior exposure.
