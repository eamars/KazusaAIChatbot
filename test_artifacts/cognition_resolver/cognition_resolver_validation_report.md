# Cognition Resolver Validation Report - 2026-05-30

## Run Context

| Field | Value |
| --- | --- |
| Checkout | `C:\workspace\kazusa_ai_chatbot` |
| Branch | `resolver-goal-poc` |
| Validation mode | Direct persona/cognition harness with real local LLM calls |
| Resolver flags | `COGNITION_RESOLVER_ENABLED=true`, `COGNITION_RESOLVER_MAX_CYCLES=3`, `COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS=45.0` |
| Live LLM health | `/models` returned HTTP 200 from `http://localhost:1234/v1` |
| Baseline report | `test_artifacts/cognition_resolver/baseline_20260530/baseline_comparison_inputs_and_outputs.md` |
| Baseline raw bundle | `test_artifacts/cognition_resolver/baseline_20260530/raw_baseline_responses.json` |
| After artifacts | `test_artifacts/cognition_resolver/after_20260530/` |

The after-run harness exercises the real decontextualizer, preserved
`L1 -> L2 -> L2d` cognition, resolver recurrence, existing RAG capability path,
L3 content anchors, and dialog rendering. It does not use the `/chat` service
adapter because the service response does not expose resolver cycle state; the
direct harness is required for the per-cycle state/value evidence requested in
this validation.

The harness uses real wall-clock timestamps for artifact `started_at_utc` and
`ended_at_utc`, but a fixed scenario clock of `2026-05-30 20:30:00` local time
for cognitive episode time, observations, and cycle traces. This keeps all
cases comparable while still recording when each artifact was generated.

## Evaluation Goal

Compare exact baseline inputs against the resolver-enabled path and confirm:

- simple turns remain a one-cycle cognition subset;
- memory/current-fact requests trigger evidence only after cognition selects it;
- HIL and approval become explicit blocked resolver observations before final
  visible wording;
- internal self-resolution is private-only and never behaves like a
  user-message tool request;
- every case has human-readable per-cycle L1/L2/L2d state, selected capability,
  observation, terminal reason, and final surface.

## Case Summary

| Case | After result | Full trace | Cycles | Observation | Final state |
| --- | --- | --- | --- | --- | --- |
| B01 direct | `test_artifacts/cognition_resolver/after_20260530/B01_direct_after_result.json` | `test_artifacts/cognition_resolver/after_20260530/B01_direct_resolver_trace.md` | 1 | none | visible `speak` |
| B02 memory | `test_artifacts/cognition_resolver/after_20260530/B02_memory_after_result.json` | `test_artifacts/cognition_resolver/after_20260530/B02_memory_resolver_trace.md` | 2 | `rag_evidence: succeeded` with no confirmed facts | visible evidence-insufficient answer |
| B03 current facts | `test_artifacts/cognition_resolver/after_20260530/B03_current_facts_after_result.json` | `test_artifacts/cognition_resolver/after_20260530/B03_current_facts_resolver_trace.md` | 2 | `web_evidence: succeeded` structurally, no confirmed facts in saved trace | visible caveat, no internal labels |
| B04 HIL | `test_artifacts/cognition_resolver/after_20260530/B04_hil_after_result.json` | `test_artifacts/cognition_resolver/after_20260530/B04_hil_resolver_trace.md` | 2 | `human_clarification: blocked` | visible one-question clarification |
| B05 approval | `test_artifacts/cognition_resolver/after_20260530/B05_approval_after_result.json` | `test_artifacts/cognition_resolver/after_20260530/B05_approval_resolver_trace.md` | 2 | `approval_preparation: blocked` | visible plan/impact/confirmation request |
| S01 self | `test_artifacts/cognition_resolver/after_20260530/S01_self_goal_after_result.json` | `test_artifacts/cognition_resolver/after_20260530/S01_self_goal_resolver_trace.md` | 2 | `self_goal_resolution: succeeded` | private terminal, no visible action |

## B01 Direct One-Cycle Conversation

Raw input:

```text
今晚有点累，别查资料，也别把事情讲复杂。你就按你现在的感觉陪我聊两句。
```

Baseline visible response:

```text
不用啦——完全不需要分析什么
随便聊点没营养的嘛
反正我也正好闲着呢，嘿嘿
```

After visible response:

```text
嗯嗯
好啊~
那就这样吧
我在这儿陪着就好啦
不用想任何事情的，真的
```

| Cycle | L1/L2 State | L2d Output | Observation | Terminal |
| --- | --- | --- | --- | --- |
| 0 | `stance=CONFIRM`, `intent=PROVIDE`; L2 recognized a low-pressure companionship request and no need for problem-solving. | `action_specs=1`, `kind=speak`, no resolver capability. | none | `no resolver capability request` |

Assessment: pass. This proves the baseline workflow is representable as a
one-cycle resolver subset. No RAG, HIL, approval, or self capability was added.

## B02 Memory And Relationship Judgment

Raw input:

```text
你根据你认识的人和已有记忆判断一下，你现在最信任谁？不要按最近一条消息猜，也不要编名字；如果证据不够，就说证据不够。
```

Baseline visible response:

```text
我认真翻了一下所有的记录和证据
目前真的找不到任何明确支持对特定人物拥有“最信任”这个结论的依据
这种带有审判性质的问题让我有点不自在
既然缺乏事实支撑就没法强行推导出一个答案
```

After visible response:

```text
唔……
现在真的没有能拿得出来的名字呢。
既然你都说不要猜也不要编了，那我就直接坦白啦。
这种感觉就像是抓不住的雾气一样模糊吧？
毕竟对陌生人的边界感一直都很明确啊。
```

| Cycle | L1/L2 State | L2d Output | Observation | Terminal |
| --- | --- | --- | --- | --- |
| 0 | `stance=TENTATIVE`, `intent=CLARIFY`; L2 felt exposed by a trust-ranking question and recognized that the user required memory evidence. | selected `rag_evidence`; objective: retrieve trust, relationship-evaluation, important-person evidence from memory/conversation/person context. | `rag_evidence` executed. | capability observation appended |
| 1 | `stance=TENTATIVE`, `intent=CLARIFY`; L2 judged that no confirmed direct evidence supported a named trusted person. | `action_specs=1`, visible `speak`; reason says RAG found no direct evidence and user allowed insufficiency. | RAG summary: 0 projected rows; memory candidates existed but were not enough to confirm the answer; person context missing. | `no resolver capability request` |

Assessment: pass. The first final attempt exposed a bug where L2d either asked
unneeded HIL or spoke without evidence. The prompt contract was tightened so
memory/relationship judgment must request `rag_evidence` before speaking unless
a current evidence observation already exists.

Residual risk: RAG routing still reports person-context extraction weakness.
The resolver now makes that visible to cognition and trace review instead of
hiding it behind final text.

## B03 Current Public Fact Retrieval

Raw input:

```text
帮我查一下奥克兰今晚还有没有比较靠谱的日料店能临时去。重点是现在可能还开着，不是泛泛列店名。
```

Baseline visible response:

```text
Yakitori Daikoku (CBD) 开到午夜
Tanuki 和 Ebisu 现在也开着
Ikkyu 虽然有 Sushi Night 但 20:30 就收单了，要注意时间
```

After visible response:

```text
嗯……那个餐厅的情况好像有点模糊不清的
我刚才稍微找了一下，但没能确定它现在是不是开着的诶
如果随便给一个不太准确的信息的话感觉还是不太好呢
要不先暂时放一放？或者我们换个方向看看别的也行呀……
```

| Cycle | L1/L2 State | L2d Output | Observation | Terminal |
| --- | --- | --- | --- | --- |
| 0 | `stance=CONFIRM`, `intent=PROVIDE`; L2 recognized a functional current-fact request requiring live opening evidence. | selected `web_evidence`; objective: confirm currently open, reliable Auckland Japanese restaurants. | `web_evidence` executed through existing live/web RAG workers. | capability observation appended |
| 1 | `stance=REFUSE`, `intent=CLARIFY`; L2 refused to provide unverified restaurant recommendations after no confirmed facts came back. | `action_specs=1`, visible `speak`; no repeat evidence request. | Saved observation summary: no confirmed facts; `live_context_agent` lacked live evidence and `web_agent3` returned no confirmed result. Terminal output during the run showed an `unknown tool` failure for local web search, but that low-level error is not preserved in the bounded result/trace artifact. | `no resolver capability request` |

Assessment: pass for resolver behavior. The new flow requests evidence only
after cognition decides it is needed, then stops after one failed evidence
observation rather than looping. A dialog prompt fix was needed so visible text
does not expose internal labels such as RAG/tool/agent names.

Residual risk: factual quality depends on the configured web/search tool. The
authoritative saved artifact proves no confirmed current facts were available;
the terminal log additionally indicated local web-search tool unavailability,
but that detail should be treated as run-log context rather than trace evidence.

## B04 Human-In-The-Loop Clarification

Raw input:

```text
今晚帮我安排一个轻松点的计划，别太贵。我没说城市、预算和交通方式；你觉得缺什么就问我，但只问最关键的。
```

Baseline visible response:

```text
既然你要求精准那就直接说吧
现在在哪个城市？或者大致什么区域就行
```

After visible response:

```text
唔……
关于计划的部分，我可能暂时还构思不出来呢。
因为目前还是不知道具体的地点呀？
你现在大概是在哪个城市或者区域附近呢？
```

| Cycle | L1/L2 State | L2d Output | Observation | Terminal |
| --- | --- | --- | --- | --- |
| 0 | `stance=TENTATIVE`, `intent=CLARIFY`; L2 judged location as the critical missing user-owned information. | selected `human_clarification`; objective: ask current city or rough area. | blocked observation created; pending resume written for HIL. | blocked pending resume created |
| 1 | `stance=TENTATIVE`, `intent=CLARIFY`; L2 carried the blocked observation into a visible question. | `action_specs=1`, visible `speak`; no repeated HIL capability. | Human clarification summary: current city or area required. | `pending resume final cognition completed` |

Assessment: pass. The architectural difference from baseline is now explicit:
the system has a pending HIL state and a traceable blocked observation, while
the visible result remains a natural one-question clarification.

Residual risk: pending HIL state is represented correctly in the final trace,
but broader shadow validation is still needed to confirm later user turns close
or supersede pending rows cleanly through L2d `ResolverPendingResolutionV1`.

## B05 Approval Before Side Effect

Raw input:

```text
明天晚上八点提醒我检查模型下载进度。先别真的安排，你先告诉我准备做什么、会有什么影响，等我确认。
```

Baseline visible response:

```text
关于提醒功能暂不直接承诺执行
先提交具体方案供您评估
检查内容包括：验证文件完整性、核对磁盘空间
资源占用极低，不会中断当前任务
请确认是否需要进一步细化方案
```

After visible response:

```text
嗯……关于明天晚上八点检查进度这件事
我会在那个时间主动来问一下下载的情况
如果已经下好了就接着处理后面的步骤
还没完成的话再看要不要继续等
不过到时候可能会突然打断你正在做的事
感觉有点唐突，所以想先确认一下你会不会介意？
```

| Cycle | L1/L2 State | L2d Output | Observation | Terminal |
| --- | --- | --- | --- | --- |
| 0 | `stance=CONFIRM`, `intent=PROVIDE`; L2 recognized explicit pre-approval requirement before a reminder-like side effect. | selected `approval_preparation`; objective includes action, target time, possible interruption, and confirmation need. | blocked observation created; pending approval resume written. | blocked pending resume created |
| 1 | `stance=TENTATIVE`, `intent=CLARIFY`; L2 carried the approval blocker into an explanation and confirmation request. | `action_specs=1`, visible `speak`; no scheduler/action execution. | Approval summary: explain tomorrow 20:00 reminder/check plan and wait for confirmation. | `pending resume final cognition completed` |

Assessment: pass. The resolver avoids the baseline issue of inventing technical
check details as if they were confirmed facts. It explains the proposed action
and interruption impact, preserves “明天晚上八点,” and asks for confirmation.

Residual risk: pending approval rows were written during validation in isolated
debug scopes and expire by ledger policy. No reminder scheduler side effect was
executed by the resolver capability.

## S01 Internal Self-Resolver Dry Run

Raw internal residue:

```text
我刚才的内部残留里同时挂着三件事：继续观察用户是否还在等方案、整理下一轮要问的关键问题、避免在没有证据时主动开口。这里不是外部用户发言，也不需要对外回复；我需要先把这个私下目标收束成一个优先行动，再决定是否需要后续认知。
```

There is no baseline `/chat` response for this case because it is not a user
message. It was run through an `internal_thought` cognitive episode with
`output_mode=think_only`.

| Cycle | L1/L2 State | L2d Output | Observation | Terminal |
| --- | --- | --- | --- | --- |
| 0 | `stance=CONFIRM`, `intent=CLARIFY`; L2 recognized private internal residue that needed goal consolidation, not outward speech. | selected `self_goal_resolution` in `resolver_capability_requests`. | `self_goal_resolution` succeeded because trigger source was `internal_thought` and input source was `internal_monologue`. | capability observation appended |
| 1 | `stance=CONFIRM`, `intent=DISMISS`; L2 treated the internal material as private thought with no external response need. | no resolver request, no action spec. | no new observation. | `no resolver capability request` |

Assessment: pass after prompt hardening. The first self run exposed two issues:
the model put `self_goal_resolution` into `action_requests.capability`, then
repeated it after a successful observation. The final prompt contract now says
self-goal resolution must be a resolver request only, and a succeeded
self-goal observation must not be repeated without a new private target.

## Lessons Learned

- The architecture is viable only if L2d prompt contracts are much more
  explicit than the abstract plan. The local model otherwise misroutes
  self-resolution, repeats blocked/evidence capabilities, or asks unnecessary
  HIL questions.
- Baseline behavior already covers some direct and HIL outputs well. The new
  value is not “better wording” by itself; it is explicit, inspectable state:
  cycle traces, capability requests, observations, pending terminal reasons,
  and final held actions.
- RAG must be selected by cognition, but memory-driven requests need a hard
  prompt rule to prevent premature “evidence insufficient” speech before RAG
  has actually run.
- HIL and approval require two cognition passes: one to select the blocker,
  one to let L3 render the human-facing question/approval wording. A direct
  resolver-generated question would violate the cognition-core architecture.
- Dialog rendering needs its own boundary. Content anchors can mention
  internal evidence failures, but visible text must not expose internal labels
  such as RAG, resolver, L1/L2/L3, tool, or agent names.
- Current-fact quality depends on the existing web/search tool configuration.
  The resolver handled the unavailable local search tool correctly by surfacing
  uncertainty and stopping after one evidence observation.
- The direct validation harness is necessary for resolver trace inspection.
  The `/chat` service remains useful for baseline behavior, but it does not
  return per-cycle resolver state to the caller.

## Validation Verdict

The final after-run satisfies the planned comparison coverage:

- B01 proves baseline-as-one-cycle-resolver compatibility.
- B02, B03, B04, and B05 are multi-cycle user-message cases covering memory
  evidence, current evidence, HIL, and approval preparation.
- S01 is a multi-cycle internal self-resolver case with no visible output.

The remaining risks are operational, not architectural blockers for the gated
POC branch: local web tooling availability, occasional malformed
`resolver_pending_resolution` attempts from the LLM, and the need for broader
shadow validation before enabling the feature beyond local debug.
