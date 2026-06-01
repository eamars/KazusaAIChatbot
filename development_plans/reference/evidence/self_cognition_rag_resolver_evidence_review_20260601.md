# Self-Cognition RAG Resolver Evidence Review - 2026-06-01

## Scope

This review covers the `resolver-self-cognition-rag` branch after extending the
existing RAG cognitive episode adapter to accept cognition-selected
`internal_thought` + `internal_monologue` RAG requests.

Raw diagnostic artifacts:

- `test_artifacts/cognition_resolver/real_db_comparison_20260601/self_cognition_l2d_diagnostics.json`
- `test_artifacts/cognition_resolver/real_db_comparison_20260601/R04_self_cognition_price_topic_resolver_trace.md`
- `test_artifacts/cognition_resolver/real_db_comparison_20260601/R05_self_cognition_photo_topic_resolver_trace.md`

## Prompt Audit

The reviewed `_ACTION_INITIALIZER_PROMPT` remains generic. It does not list
validation cases, does not map specific user examples to actions, and does not
ask the LLM to copy raw durable ids.

The prompt was reorganized so `internal_thought` is treated as a cognition
source rather than a `self_goal_resolution` default:

- missing conversation, memory, relationship, or material evidence may choose
  `rag_evidence`;
- private target consolidation may choose `self_goal_resolution`;
- enough grounded reason may choose ordinary actions;
- weak scene pressure may return no action.

This keeps action direction inside L2d and avoids deterministic Python rules
such as "if context is insufficient, lookup evidence."

## Implementation Summary

The change extends `build_text_chat_rag_request(...)` only at the episode
projection boundary:

- `user_message` RAG request shape remains unchanged;
- unsupported trigger sources such as `reflection_signal` remain rejected;
- `internal_thought` with `internal_monologue` can now project `target_scope`,
  `origin_metadata`, turn clock fields, and existing prompt-safe state into the
  same RAG2 request boundary;
- no new RAG worker, direct DB lookup path, or semantic Python classifier was
  introduced.

## Deterministic Verification

Command:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py tests\test_rag_cognitive_episode_adapter.py tests\test_cognition_resolver_loop.py tests\test_cognition_prompt_contract_text.py
```

Result: passed.

Command:

```powershell
venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py tests\test_cognition_resolver_loop.py tests\test_cognition_prompt_contract_text.py tests\test_persona_supervisor2_action_initializer.py -q
```

Result: 74 passed.

## Real LLM Diagnostic

Command:

```powershell
venv\Scripts\python test_artifacts\cognition_resolver\real_db_comparison_20260601\diagnose_l2d_self_cognition.py
```

Result:

- `SUMMARY_PATH=test_artifacts\cognition_resolver\real_db_comparison_20260601\self_cognition_l2d_diagnostics.json`
- `R04_self_cognition_price_topic: l2d_calls=3 route=audit_only action_specs=0`
- `R05_self_cognition_photo_topic: l2d_calls=2 route=audit_only action_specs=1`

### R04: Self-Cognition Price Topic

Pass criterion: positive. L2d selected `rag_evidence` from an
`internal_thought` case and the resolver executed existing RAG2.

Cycle 0 L2d output:

```json
{
  "capability_kind": "rag_evidence",
  "objective": "获取当前聊天窗口中关于 STM32C8T6 价格讨论的具体上下文，确认小钳子（或相关对话参与者）具体在问什么、语境如何以及是否需要回应",
  "reason": "虽然内心已有介入意向，但目前仅基于 internal_monologue 的主观判断，缺乏对当前聊天窗口实际内容的客观证据支撑；必须先读取真实对话内容才能决定是直接提供信息还是保持观察"
}
```

RAG2 path:

- initializer requested conversation evidence for `STM32C8T6`;
- initializer requested person context for `小钳子`;
- RAG2 dispatched `conversation_evidence_agent`,
  `person_context_agent`, then `conversation_evidence_agent` again for context
  around the resolved person;
- final RAG observation succeeded and was fed back to cognition.

Key RAG answer:

```text
用户“小钳子”在消息记录中曾询问：“话说现在STM32C8T6什么价啊”。
目前无法从检索到的证据中确认 STM32C8T6 价格询问的具体上下文语境或是否需要回应。
```

Cycle 1 and cycle 2 behavior:

- L2d attempted an equivalent `rag_evidence` request after inconclusive
  evidence.
- The resolver duplicate guard blocked that repeated request.
- Because the source was non-user `internal_thought`, the terminal duplicate
  blocker stayed private and produced no visible speech.

Judgment:

This satisfies the branch acceptance condition because cognition selected RAG
for a self-cognition evidence gap and the existing RAG path executed. The
residual risk is local-model over-retry after inconclusive evidence; it is
bounded by duplicate detection and did not create a visible regression.

### R05: Self-Cognition Photo Topic

Pass criterion: positive. L2d did not call RAG, but its opt-out path was
coherent and cognition-owned.

Cycle 0 L2d output:

```json
{
  "capability_kind": "self_goal_resolution",
  "objective": "收束当前内部思考，明确在观察期内是否需要主动介入或继续等待证据（照片）出现",
  "reason": "当前处于 internal_thought 状态且有初步裁决意图，需要正式整理并确认下一步行动策略"
}
```

Cycle 1 L2d action:

```json
{
  "capability": "trigger_future_cognition",
  "decision": "等待群聊新动态出现",
  "detail": "持续观察聊天窗口，重点关注是否有照片流出或话题进一步展开的信号",
  "reason": "当前决策已明确倾向于观望，且缺乏介入所需的实质性证据（照片），主动发言可能显得突兀"
}
```

Judgment:

This is an acceptable no-RAG outcome. The model did not need historical memory
or conversation lookup to know its current blocker: no photo had appeared and
speaking would be premature. It chose a private future cognition action instead
of forcing speech or a redundant evidence call.

## Final Review

The implementation satisfies the requested production gap:

- cognition can now select RAG from self-cognition/internal-thought;
- Python does not decide when evidence is needed;
- existing RAG2 is reused;
- prompt edits are generic and not case-tuned;
- real LLM evidence is positive under the agreed relaxed criterion.

Merge into `resolver-goal-poc` is justified.
