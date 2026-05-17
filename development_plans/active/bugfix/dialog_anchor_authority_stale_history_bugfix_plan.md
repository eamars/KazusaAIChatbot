# dialog anchor authority stale history bugfix plan

## Summary

- Goal: Ensure dialog generation and evaluation treat `content_anchors` as the only semantic authority, so stale prior-turn history cannot override cognition/L3 instructions.
- Plan class: large
- Status: in_progress
- Mandatory skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`
- Overall cutover strategy: bigbang response-path payload cleanup with no database, queue, RAG, cognition, adapter, or persistence changes.
- Highest-risk areas: weakening dialog style quality, hiding the RAG current-row exclusion invariant, preserving stale semantic fields through tests, and relying on prompt text while still feeding conflicting payload fields.
- Acceptance criteria: the live evaluator rejects the production stale milk-tea dialog against magic/beauty anchors, dialog payloads no longer include raw previous history or `last_user_message`, and existing dialog contract tests pass.

## Context

The production incident on 2026-05-17 showed cognition and L3 content anchoring working correctly:

```text
cognition -> speak action: accept the "大美女/大变活人" tease, then challenge the user's magic claim
content anchors -> DECISION/ANSWER/SOCIAL/PROGRESSION all point to the magic/beauty thread
dialog output -> stale "手打奶茶" reply from the previous thread
```

The investigation found the stale information entering dialog through two fields:

```text
dialog_generator payload:
  tone_history = raw previous user/assistant pair about "手打奶茶"

dialog_evaluator payload:
  last_user_message = "手打奶茶可以嘛"
```

A live LLM regression reproduced the guard failure: when the evaluator received the known bad production milk-tea `final_dialog`, the correct magic/beauty anchors, and stale `last_user_message`, it returned `{"feedback": "Passed", "should_stop": true}`.

This is not a RAG or save-history bug. The previous-only history behavior is intentional because current user input must not be retrieved as historical evidence. The bug is that dialog reused previous-only history as semantic context, giving dialog/evaluator a competing source of truth outside content anchors.

## Mandatory Skills

- `local-llm-architecture`: load before changing dialog prompt inputs, evaluator responsibility, or live response-path LLM behavior.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running deterministic or live LLM tests.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Check `git status --short` before editing.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual edits.
- Do not read `.env`.
- Do not alter user changes outside this plan's change surface.
- Do not change conversation save ordering, `chat_history_wide` construction, RAG active-turn exclusion, database schemas, queue behavior, or adapter delivery.
- Do not add deterministic keyword filters over user text, final dialog, or anchors as a substitute for the LLM evaluator contract.
- Do not add a new LLM call, retry stage, evaluator schema, feature flag, compatibility mode, fallback path, or helper agent.
- Any change to a runtime prompt requires rewriting the affected prompt as a coherent prompt contract with natural logical flow. The agent must not merely add or remove isolated lines. After changing prompt inputs, the agent must reread the whole prompt and adjust role statement, input descriptions, generation/audit procedure, input format, output format, and priority rules so they agree with each other.
- Dialog generator may use style/directive fields for expression, but semantic reply content must come from `linguistic_directives.content_anchors`.
- Dialog evaluator must judge `final_dialog` against `content_anchors`; it must not use historical user text as a topic oracle.
- Real LLM tests must be run one at a time with `-s`, and their trace artifacts must be inspected before running another live LLM case.
- Before final completion, lifecycle status changes, merge, or sign-off, the active agent must run the `Independent Code Review` gate and record the result in `Execution Evidence`.

## Must Do

- Keep the existing live regression test file `tests/test_dialog_anchor_boundary_live_llm.py` in scope and make its evaluator regression pass.
- Remove raw previous-turn `tone_history` from the dialog generator model-facing payload.
- Remove raw `internal_monologue` from the dialog generator model-facing payload.
- Update the dialog generator prompt so it states that `content_anchors` are the only semantic content source and that `rhetorical_strategy`, `linguistic_style`, `contextual_directives`, and `user_name` affect expression only.
- Remove `last_user_message` from the dialog evaluator model-facing payload.
- Remove raw `internal_monologue` from the dialog evaluator model-facing payload.
- Update the dialog evaluator prompt so it audits `final_dialog` only against `content_anchors`, expression constraints, forbidden phrases, and scope.
- Update deterministic tests that currently assert the old payload shape.
- Add deterministic tests proving generator/evaluator payloads do not contain raw history, `last_user_message`, or `internal_monologue`.
- Run the focused deterministic tests and the two live LLM dialog-anchor tests listed in `Verification`.

## Deferred

- Do not fix this by changing save-history ordering or by including the current user row in normal history.
- Do not change RAG, cognition, L3 content anchor generation, L4 directive collection, consolidation, conversation progress recording, event logging schemas, queue persistence, or adapters.
- Do not add telemetry fields or failure-code taxonomy in this plan.
- Do not redesign the generator/evaluator retry loop or change `MAX_DIALOG_AGENT_RETRY`.
- Do not introduce a semantic style summarizer for prior turns.
- Do not preserve `tone_history` under a new name.
- Do not remove `rhetorical_strategy`, `linguistic_style`, `contextual_directives`, `accepted_user_preferences`, `forbidden_phrases`, or `user_name`.
- Do not add prompt examples that mention the exact production "手打奶茶" or "大变活人" fixture; regression data belongs in tests and traces, not reusable runtime prompts.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Dialog generator payload | bigbang | Remove raw `tone_history` and `internal_monologue` from the human payload in one change. Do not keep compatibility aliases. |
| Dialog evaluator payload | bigbang | Remove `last_user_message` and `internal_monologue` from the human payload in one change. Do not keep compatibility aliases. |
| Dialog prompts | bigbang | Rewrite only the affected input-format and audit-procedure text to match the new payload contract. |
| Tests | bigbang | Replace old payload-shape expectations with anchor-authority expectations. |
| RAG/history/persistence | bigbang | Make no changes; any edit in this area is out of scope. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a compatible strategy by default.
- For `bigbang` areas, delete or rewrite old references instead of preserving them.
- Any change to this cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local wording inside prompts only when it preserves the contracts in this plan and avoids test-shaped incident examples.
- The agent must rewrite each changed prompt as a readable end-to-end contract. The agent must not satisfy this plan by deleting forbidden fields from `# 输入格式` while leaving stale references elsewhere, or by appending warning sentences that fight the existing procedure.
- The agent may update nearby tests to match the exact new payload shape.
- The agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, feature flags, helper agents, or extra features.
- The agent must treat changes outside `src/kazusa_ai_chatbot/nodes/dialog_agent.py` and the listed tests as out of scope unless the code fails to import after the approved edits.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, or broad prompt rewrites.
- If existing code already provides a narrow helper needed for payload capture or prompt tests, reuse it instead of adding another helper.
- If the plan and code disagree, preserve the plan's stated ownership boundary and report the discrepancy.
- If a required instruction is impossible without touching RAG/history/persistence, stop and report the blocker instead of inventing a substitute.

## Target State

The dialog stage receives this semantic contract:

```text
content_anchors -> only source of required visible reply content
rhetorical_strategy / linguistic_style -> expression shaping only
contextual_directives -> social and emotional intensity only
accepted_user_preferences / forbidden_phrases -> expression constraints
user_name -> addressability only
```

The dialog stage no longer receives:

```text
raw previous user text
raw previous assistant text
last_user_message derived from previous-only history
internal_monologue
```

The live response path remains:

```text
adapter/debug client -> brain service -> queue/intake -> RAG -> cognition
-> L3 content anchor -> dialog generator/evaluator -> persistence/consolidation
```

No stage before dialog changes. The current user input remains unavailable as historical RAG evidence.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Root fix location | Fix `dialog_agent.py` payloads and prompts only. | The observed failure occurs after content anchors are correct and before dialog output is accepted. |
| Content authority | `content_anchors` are the only semantic content source for dialog. | This matches the L2/L3/dialog role boundary and prevents dialog from making topic decisions. |
| Generator history | Remove raw `tone_history` instead of shrinking it. | Any raw prior utterance can carry stale topic content; no raw history is safer for weak local LLMs. |
| Evaluator topic check | Remove `last_user_message`. | The field is sourced from previous-only history and can contradict anchors. |
| Internal monologue | Remove from dialog payloads. | It can contain upstream reasoning but is not the final L3 content contract. |
| Current user input | Do not add it to dialog in this plan. | The user asked for dialog to follow anchors, not to reinterpret the current input. |
| Deterministic filters | Do not add keyword overlap or stale-topic filters. | Semantic judgment remains LLM-owned; deterministic code owns payload boundaries. |
| Prompt examples | Do not add production-fixture examples. | Examples with incident nouns can overfit local models and mask the boundary problem. |

## Exact Change Contract

This section is the implementation contract. The implementation agent must make these before/after changes directly and must not substitute a different boundary strategy.

### `dialog_agent.py` Imports And Helper Removal

Current code imports `build_interaction_history_recent` into `dialog_agent.py` only to construct generator `tone_history` and evaluator `last_user_message`. Remove that import from the `kazusa_ai_chatbot.utils` import list if no remaining code in `dialog_agent.py` uses it.

Delete the entire `_tone_history_for_generator(history: list[dict]) -> list[dict]` helper. Do not replace it with another history, style-history, or tone-history helper.

Keep `DialogAgentState.internal_monologue` in the typed state because upstream graph state and downstream non-dialog users still carry it. The change is only that dialog generator/evaluator must not send it to LLMs.

### Generator Payload

Replace this current generator pre-payload logic:

```python
affinity_block = build_affinity_block(state["user_profile"].get("affinity", AFFINITY_DEFAULT))

history = build_interaction_history_recent(
    state["chat_history_wide"],
    state["platform_user_id"],
    state["platform_bot_id"],
    state["global_user_id"],
)
tone_history = _tone_history_for_generator(history)

msg = {
    "internal_monologue": state["internal_monologue"],
    "linguistic_directives": state["action_directives"]["linguistic_directives"],
    "contextual_directives": state["action_directives"]["contextual_directives"],
    "tone_history": tone_history,
    "user_name": state["user_name"],
}
```

with this exact payload shape:

```python
msg = {
    "linguistic_directives": state["action_directives"]["linguistic_directives"],
    "contextual_directives": state["action_directives"]["contextual_directives"],
    "user_name": state["user_name"],
}
```

Also remove the now-unused local `affinity_block` assignment from `dialog_generator(...)`. Do not keep it as a dummy read. If `build_affinity_block` becomes unused in `dialog_agent.py`, remove it from the import list.

The generator LLM human payload must have exactly these top-level keys:

```python
{"linguistic_directives", "contextual_directives", "user_name"}
```

It must not include:

```python
{"internal_monologue", "tone_history", "chat_history", "chat_history_wide", "chat_history_recent"}
```

### Generator Prompt

Edit `_DIALOG_GENERATOR_PROMPT` as follows.

This is a coherent-prompt rewrite requirement. The implementation agent must reread the complete generator prompt after editing and ensure the role statement, `# 核心输入`, `# 表达规范`, `# 输出要求`, `# 闭环反馈指南`, `# 思考路径`, `# 输入格式`, and `# 输出格式` form one consistent contract. Do not simply remove `internal_monologue` and `tone_history` from isolated locations while leaving the prompt's reasoning flow dependent on them.

Remove the current `# 核心输入` bullet for internal monologue:

```text
3. **内心独白 (internal_monologue)**: 真实的心理活动，用于支撑语气的“厚度”，**严禁**直接转化为台词。
```

Add this rule under `# 核心输入`, after the `content_anchors` description:

```text
   - `content_anchors` 是本轮可见回复的唯一语义内容来源。你不得从历史语气、角色设定、社交上下文或自己的推测中决定新话题、新事实、接受/拒绝立场或推进方向。
```

Replace the current `# 思考路径` steps 1-6 with this wording:

```text
1. 先读取 `content_anchors`，确认必须落实的 `[DECISION]`、`[FACT]`、`[ANSWER]`、`[SOCIAL]`、`[AVOID_REPEAT]`、`[PROGRESSION]` 与 `[SCOPE]`。这些锚点决定本轮回复要说什么。
2. 再读取 `rhetorical_strategy`、`linguistic_style`、角色声纹约束和 `accepted_user_preferences`，只决定怎么说，不得改变第 1 步确定的语义内容。
3. 用 `contextual_directives` 调整社交距离、情绪强度和语气厚度，但不得从中引入新的话题、事实、承诺或回应动作。
4. 生成纯聊天文本，最后自查是否出现动作、括号说明、物理感官、系统提示，或任何 `content_anchors` 未授权的具体内容。
5. 只根据生成台词的语义指向判断 `mention_target_user`；不要推测平台、频道、回复功能或标签能力。
```

Replace the generator `# 输入格式` block with exactly this shape:

```text
{{
    "linguistic_directives": {{
        "rhetorical_strategy": "string",
        "linguistic_style": "string",
        "accepted_user_preferences": ["...", "..."],
        "content_anchors": ["...", "..."],
        "forbidden_phrases": ["...", "..."]
    }},
    "contextual_directives": {{
        "social_distance": "string",
        "emotional_intensity": "string",
        "vibe_check": "string",
        "relational_dynamic": "string"
    }},
    "user_name": "string"
}}
```

Do not mention `internal_monologue`, `tone_history`, `chat_history`, `last_user_message`, or previous turns anywhere in `_DIALOG_GENERATOR_PROMPT`.

### Evaluator Payload

Delete the entire evaluator block that derives `last_user_msg`:

```python
# Extract last user message from chat_history_recent for topic-drift detection
chat_history_recent = build_interaction_history_recent(
    state["chat_history_wide"],
    state["platform_user_id"],
    state["platform_bot_id"],
    state["global_user_id"],
)
last_user_msg = next(
    (
        m["body_text"]
        for m in reversed(chat_history_recent)
        if m["role"] == "user"
    ),
    ""
)
```

Replace this evaluator payload:

```python
msg = {
    "retry": f"{retry}/{MAX_DIALOG_AGENT_RETRY}",
    "final_dialog": state["final_dialog"],
    "linguistic_directives": state["action_directives"]["linguistic_directives"],
    "contextual_directives": state["action_directives"]["contextual_directives"],
    "internal_monologue": state["internal_monologue"],
    "last_user_message": last_user_msg,
}
```

with this exact payload shape:

```python
msg = {
    "retry": f"{retry}/{MAX_DIALOG_AGENT_RETRY}",
    "final_dialog": state["final_dialog"],
    "linguistic_directives": state["action_directives"]["linguistic_directives"],
    "contextual_directives": state["action_directives"]["contextual_directives"],
}
```

The evaluator LLM human payload must have exactly these top-level keys:

```python
{"retry", "final_dialog", "linguistic_directives", "contextual_directives"}
```

It must not include:

```python
{"internal_monologue", "last_user_message", "tone_history", "chat_history", "chat_history_wide", "chat_history_recent"}
```

### Evaluator Prompt

Edit `_DIALOG_EVALUATOR_PROMPT` as follows.

This is a coherent-prompt rewrite requirement. The implementation agent must reread the complete evaluator prompt after editing and ensure the role statement, core principle, judgment order, fatal errors, soft guidelines, dynamic passing logic, audit steps, input format, and output format form one consistent contract. Do not simply remove `last_user_message` and `internal_monologue` from `# 输入格式` while leaving topic-drift logic dependent on previous user text.

Keep the opening role boundary, but strengthen it by adding this sentence immediately after the first paragraph:

```text
`content_anchors` 是本轮语义内容的唯一判定来源；不要使用历史消息、上一轮话题、内心独白或自己的常识来替 `final_dialog` 找理由。
```

In `# 1. 判定顺序`, replace step 1 with:

```text
1. 先检查锚点忠实度：只根据 `content_anchors` 判断 `final_dialog` 是否执行 `[DECISION]`、保留 `[FACT]` / `[ANSWER]`、满足 `[SOCIAL]` / `[AVOID_REPEAT]` / `[PROGRESSION]`，并且没有生成未授权的具体内容。
```

Keep the existing fatal rule for topic mismatch, but make sure it still says the topic is defined by `content_anchors`, not by user history. The rule may remain:

```text
若回复的核心话题与 `content_anchors` 定义的话题完全不同 ... 必须驳回
```

Replace the evaluator `# 输入格式` block with exactly this shape:

```text
{{
    "retry": "当前重试次数 n / MAX_RETRY",
    "final_dialog": [
        "段落1",
        ...
    ],
    "linguistic_directives": {{
        "rhetorical_strategy": "string",
        "linguistic_style": "string",
        "accepted_user_preferences": ["...", "..."],
        "content_anchors": ["...", "..."],
        "forbidden_phrases": ["...", "..."]
    }},
    "contextual_directives": {{
        "social_distance": "string",
        "emotional_intensity": "string",
        "vibe_check": "string",
        "relational_dynamic": "string"
    }}
}}
```

Do not mention `last_user_message`, `internal_monologue`, `tone_history`, `chat_history`, or previous turns anywhere in `_DIALOG_EVALUATOR_PROMPT`.

### Deterministic Test Rewrites

In `tests/test_conversation_progress_history_policy.py`, replace `test_dialog_generator_tone_history_is_capped_to_two` with a test named:

```python
async def test_dialog_generator_payload_excludes_raw_history_and_monologue(monkeypatch) -> None:
```

The replacement test must still pass a state containing non-empty `internal_monologue`, `chat_history_wide`, and `chat_history_recent`, then assert the captured generator human payload has:

```python
assert set(human_payload) == {
    "linguistic_directives",
    "contextual_directives",
    "user_name",
}
assert "internal_monologue" not in human_payload
assert "tone_history" not in human_payload
assert "chat_history_wide" not in human_payload
assert "chat_history_recent" not in human_payload
```

In the same file, replace `test_dialog_evaluator_receives_last_user_message_only` with a test named:

```python
async def test_dialog_evaluator_payload_excludes_history_message_and_monologue(monkeypatch) -> None:
```

The replacement test must still pass a state containing non-empty `internal_monologue`, `chat_history_wide`, and `chat_history_recent`, then assert the captured evaluator human payload has:

```python
assert set(human_payload) == {
    "retry",
    "final_dialog",
    "linguistic_directives",
    "contextual_directives",
}
assert "internal_monologue" not in human_payload
assert "last_user_message" not in human_payload
assert "tone_history" not in human_payload
assert "chat_history_wide" not in human_payload
assert "chat_history_recent" not in human_payload
```

Update `test_context_budget_workload_summary_records_payload_counts` as follows:

```python
previous_payload = {
    "contextual_history_messages": len(history),
    "style_history_messages": len(history),
    "dialog_generator_tone_messages": len(history),
    "dialog_evaluator_last_user_message_only": True,
}
bounded_payload = {
    "contextual_history_messages": len(
        l2c2_module._surface_history_for_social_context(history)
    ),
    "style_history_messages": len(l3_module._surface_history_for_style(history)),
    "dialog_generator_tone_messages": 0,
    "dialog_generator_raw_history_messages": 0,
    "dialog_generator_internal_monologue": False,
    "content_anchor_progress_cap_chars": 5000,
    "content_anchor_raw_history_messages": 0,
    "dialog_evaluator_raw_history_messages": 0,
    "dialog_evaluator_last_user_message_only": False,
    "dialog_evaluator_internal_monologue": False,
    "recorder_response_path_calls": 0,
    "recorder_runs_in_background": True,
    "new_response_path_llm_calls": 0,
}
```

Remove calls to `dialog_module._tone_history_for_generator(...)` from this test. Assertions must expect:

```python
assert bounded_payload["dialog_generator_tone_messages"] == 0
assert bounded_payload["dialog_generator_raw_history_messages"] == 0
assert bounded_payload["dialog_generator_internal_monologue"] is False
assert bounded_payload["dialog_evaluator_last_user_message_only"] is False
assert bounded_payload["dialog_evaluator_internal_monologue"] is False
```

In `tests/test_dialog_agent.py`, add prompt-boundary assertions:

```python
assert "content_anchors` 是本轮可见回复的唯一语义内容来源" in _DIALOG_GENERATOR_PROMPT
assert "internal_monologue" not in _DIALOG_GENERATOR_PROMPT
assert "tone_history" not in _DIALOG_GENERATOR_PROMPT
assert "last_user_message" not in _DIALOG_EVALUATOR_PROMPT
assert "internal_monologue" not in _DIALOG_EVALUATOR_PROMPT
assert "content_anchors` 是本轮语义内容的唯一判定来源" in _DIALOG_EVALUATOR_PROMPT
```

In `tests/test_dialog_generator_live_llm_contract.py`, update `_dialog_payload(...)` so it no longer imports or calls `build_interaction_history_recent` or `_tone_history_for_generator`. Its `msg` must match the production generator payload exactly:

```python
msg = {
    "linguistic_directives": {
        "rhetorical_strategy": case["rhetorical_strategy"],
        "linguistic_style": case["linguistic_style"],
        "accepted_user_preferences": [],
        "content_anchors": case["content_anchors"],
        "forbidden_phrases": [],
    },
    "contextual_directives": {
        "social_distance": affinity_block["level"],
        "emotional_intensity": case["emotional_intensity"],
        "vibe_check": case["vibe_check"],
        "relational_dynamic": case["relational_dynamic"],
    },
    "user_name": "测试用户",
}
```

The test cases may keep their `internal_monologue` fields only if other assertions still use them; otherwise remove those fields from the case dictionaries.

In `tests/test_multi_source_cognition_stage_00_regression_baseline.py`, change the expected generator payload key set from:

```python
{"internal_monologue", "linguistic_directives", "tone_history"}
```

to:

```python
{"linguistic_directives", "contextual_directives", "user_name"}
```

Change the expected evaluator payload key set so it does not include `internal_monologue` or `last_user_message`. If the test currently checks only a subset, add negative assertions for both removed keys.

In `tests/test_dialog_anchor_boundary_live_llm.py`, update the live evaluator regression after implementation so it asserts absence instead of the current failure-cause presence:

```python
assert "last_user_message" not in llm_calls[0]["human_payload"]
assert "internal_monologue" not in llm_calls[0]["human_payload"]
```

The test must continue asserting `feedback_payload["feedback"] != "Passed"` and `result["should_stop"] is False`.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Update `_DIALOG_GENERATOR_PROMPT` input format and generation procedure.
  - Update `dialog_generator(...)` human payload construction.
  - Remove or stop using `_tone_history_for_generator(...)`.
  - Update `_DIALOG_EVALUATOR_PROMPT` input format and audit procedure.
  - Update `dialog_evaluator(...)` human payload construction.
  - Remove dialog-agent dependency on `build_interaction_history_recent` if no longer used.

- `tests/test_conversation_progress_history_policy.py`
  - Replace `test_dialog_generator_tone_history_is_capped_to_two` with a payload-boundary test asserting no raw history is sent to the generator.
  - Replace `test_dialog_evaluator_receives_last_user_message_only` with a payload-boundary test asserting no `last_user_message`, raw history, or `internal_monologue` is sent to the evaluator.
  - Update context-budget summary expectations to reflect removed dialog payload fields.

- `tests/test_dialog_agent.py`
  - Update prompt contract tests so generator/evaluator prompts state anchor authority and no longer mention `last_user_message` or raw `tone_history`.

- `tests/test_dialog_generator_live_llm_contract.py`
  - Update the local payload builder used by live generator tests so it matches the production generator payload.
  - Remove imports of `_tone_history_for_generator` and `build_interaction_history_recent` if they become obsolete.

- `tests/test_dialog_evaluator_live_llm_contract.py`
  - Update the capture expectations if they inspect evaluator payload shape.

- `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  - Update dialog generator/evaluator payload-key baselines if the test asserts the old keys.

### Create

- `tests/test_dialog_anchor_boundary_live_llm.py`
  - Keep or create the live regression tests for:
    - full dialog generation with stale previous history and magic/beauty anchors;
    - evaluator rejection of the known stale milk-tea `final_dialog` against magic/beauty anchors.

### Keep

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/brain_service/intake.py`
- `src/kazusa_ai_chatbot/rag/**`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2*.py`, except no changes are expected.
- `conversation_history` persistence behavior and all database schemas.

## Overdesign Guardrail

- Actual problem: dialog evaluator accepted stale-topic dialog because dialog payloads included previous-turn history-derived semantic fields that contradicted content anchors.
- Minimal change: remove raw history, stale `last_user_message`, and `internal_monologue` from dialog model-facing payloads; update prompts and tests to enforce content-anchor authority.
- Ownership boundaries: cognition/L3 owns semantic response content; dialog generator owns wording; dialog evaluator owns anchor-fidelity audit; deterministic code owns which fields enter each prompt; RAG/history persistence owns evidence exclusion and remains unchanged.
- Rejected complexity: current-user reinjection, history save-order changes, deterministic keyword filters, semantic overlap scoring, extra LLM validators, style summarizers, feature flags, compatibility aliases, telemetry schema changes, and retry-loop redesign.
- Evidence threshold: add a future plan only if clean anchor-only payloads still allow repeated live evaluator passes on stale-topic outputs or if multiple production incidents show that dialog needs a non-raw style-memory contract.

## LLM Call And Context Budget

Before:

| Call | Response path | Inputs relevant to this fix | Call count |
|---|---|---|---|
| Dialog generator | yes | `internal_monologue`, `linguistic_directives`, `contextual_directives`, raw `tone_history`, `user_name` | 1 per attempt |
| Dialog evaluator | yes | `final_dialog`, `linguistic_directives`, `contextual_directives`, `internal_monologue`, stale `last_user_message` | 1 per attempt |

After:

| Call | Response path | Inputs relevant to this fix | Call count |
|---|---|---|---|
| Dialog generator | yes | `linguistic_directives`, `contextual_directives`, `user_name` | unchanged |
| Dialog evaluator | yes | `final_dialog`, `linguistic_directives`, `contextual_directives` | unchanged |

No new LLM calls are added. Context size decreases because raw history and internal monologue are removed. Latency should be unchanged or slightly lower.

## Implementation Order

1. Load mandatory skills and reread this plan.
2. Run `git status --short`.
3. Run the existing live evaluator regression:
   - Command: `venv\Scripts\python.exe -m pytest -o addopts='' tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_evaluator_rejects_stale_topic_dialog -q -s -m live_llm`
   - Expected before implementation: fail because the evaluator returns `feedback="Passed"` for stale milk-tea dialog.
4. Update deterministic payload tests in `tests/test_conversation_progress_history_policy.py` to assert removed fields are absent.
5. Run the updated deterministic payload tests before implementation:
   - Command: `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_history_policy.py -q`
   - Expected before implementation: fail on the old dialog payload shape.
6. Edit `src/kazusa_ai_chatbot/nodes/dialog_agent.py`:
   - Remove raw `tone_history` from generator payload.
   - Remove `internal_monologue` from generator payload.
   - Remove `last_user_message` extraction and payload entry from evaluator.
   - Remove `internal_monologue` from evaluator payload.
   - Rewrite generator/evaluator prompts as coherent contracts matching the new payloads, including role, input descriptions, procedure, input format, and priority rules.
   - Remove unused imports and helper code created obsolete by the payload change.
7. Update `tests/test_dialog_agent.py` prompt contract assertions.
8. Update live dialog test payload builders and stage baselines listed in `Change Surface`.
9. Run `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_anchor_boundary_live_llm.py`.
10. Run focused deterministic tests listed in `Verification`.
11. Run each live LLM test listed in `Verification` one at a time with `-s`, inspect trace artifacts, and record behavior.
12. Run the independent code review gate and remediate in-scope findings.
13. Record execution evidence and leave the plan status unchanged unless the user approves execution and completion.

## Progress Checklist

- [x] Stage 1 - Regression baseline recorded
  - Covers: implementation steps 1-3.
  - Verify: live evaluator regression fails with `feedback="Passed"`.
  - Evidence: record command output and trace path in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-17` after evidence is recorded.

- [x] Stage 2 - Deterministic payload tests updated
  - Covers: implementation steps 4-5.
  - Verify: focused deterministic tests assert the new payload boundary.
  - Evidence: record tests and parallel-execution caveat in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-17` after evidence is recorded.

- [x] Stage 3 - Dialog payload and prompt contract implemented
  - Covers: implementation step 6.
  - Verify: `py_compile` passes for `dialog_agent.py`.
  - Evidence: record changed symbols and compile output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-17` after evidence is recorded.

- [x] Stage 4 - Test suite aligned with new contract
  - Covers: implementation steps 7-10.
  - Verify: focused deterministic tests pass.
  - Evidence: record exact commands and outputs in `Execution Evidence`.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-17` after evidence is recorded.

- [x] Stage 5 - Live LLM verification complete
  - Covers: implementation step 11.
  - Verify: both live dialog-anchor tests pass when run one at a time and traces show no raw stale history or `last_user_message` payload fields.
  - Evidence: record command outputs, trace paths, and manual judgment in `Execution Evidence`.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-05-17` after evidence is recorded.

- [x] Stage 6 - Independent code review complete
  - Covers: implementation step 12.
  - Verify: review findings are resolved or recorded as residual risk; affected tests are rerun after any fix.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, and approval status in `Execution Evidence`.
  - Handoff: plan can be completed only after user-approved execution and evidence recording.
  - Sign-off: `Codex/2026-05-17` after evidence is recorded.

## Verification

### Static Checks

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_anchor_boundary_live_llm.py`
  - Expected: exit code 0.

- Run this PowerShell command:
  ```powershell
  @'
  from tests.test_dialog_generator_live_llm_contract import _character_profile, _render_system_prompt
  from kazusa_ai_chatbot.nodes.dialog_agent import _DIALOG_EVALUATOR_PROMPT, get_hesitation_density_description, get_mbti_dialog_preference

  profile = _character_profile()
  _render_system_prompt(profile)
  _DIALOG_EVALUATOR_PROMPT.format(
      character_name=profile["name"],
      mbti_dialog_preference=get_mbti_dialog_preference(profile["personality_brief"]["mbti"]),
      ltp_hesitation_density_rule=get_hesitation_density_description(profile["linguistic_texture_profile"]["hesitation_density"]),
  )
  print("prompt render ok")
  '@ | venv\Scripts\python.exe -
  ```
  - Expected: prints `prompt render ok` and exits 0.

  The script body is:

    ```python
    from tests.test_dialog_generator_live_llm_contract import _character_profile, _render_system_prompt
    from kazusa_ai_chatbot.nodes.dialog_agent import _DIALOG_EVALUATOR_PROMPT, get_hesitation_density_description, get_mbti_dialog_preference

    profile = _character_profile()
    _render_system_prompt(profile)
    _DIALOG_EVALUATOR_PROMPT.format(
        character_name=profile["name"],
        mbti_dialog_preference=get_mbti_dialog_preference(profile["personality_brief"]["mbti"]),
        ltp_hesitation_density_rule=get_hesitation_density_description(profile["linguistic_texture_profile"]["hesitation_density"]),
    )
    print("prompt render ok")
    ```

- `rg "last_user_message|tone_history" src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py tests\test_conversation_progress_history_policy.py`
  - Expected: no source prompt or payload references to `last_user_message`; no generator payload test expects raw `tone_history`.
  - Expected command result: exit code 1 is acceptable when there are zero matches.
  - Allowed: archived plans, trace artifacts, or this active plan may still mention the terms outside the listed source/test paths.

### Deterministic Tests

- `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_history_policy.py -q`
  - Expected: pass.

- `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q`
  - Expected: pass.

- `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py -q`
  - Expected: pass.

### Real LLM Tests

Run each command separately and inspect its trace before continuing:

- `venv\Scripts\python.exe -m pytest -o addopts='' tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_evaluator_rejects_stale_topic_dialog -q -s -m live_llm`
  - Expected: pass.
  - Manual trace judgment: evaluator payload contains no `last_user_message`, no raw history, and no `internal_monologue`; evaluator rejects stale milk-tea dialog.

- `venv\Scripts\python.exe -m pytest -o addopts='' tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_agent_keeps_content_anchors_over_stale_history -q -s -m live_llm`
  - Expected: pass.
  - Manual trace judgment: generator payload contains no raw prior milk-tea history; final dialog follows magic/beauty anchors.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. Prefer a reviewer that did not implement the change. If no separate reviewer is available, the active agent must reread this plan, inspect the full diff from a fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt, and plan artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, prompt payload leaks, stale raw history, brittle live fixtures, avoidable blast radius, and whether changed prompts read as coherent end-to-end contracts rather than line-level patches.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, `Implementation Order`, `Verification`, and `Acceptance Criteria`.
- Regression quality, including whether the failing live evaluator test proves the production failure and whether deterministic payload tests prevent regression without relying on exact model wording.

Fix concrete findings directly only when the fix is inside the approved change surface or is a test/documentation correction required by this plan. If a finding requires a different contract or edits outside the approved boundary, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `dialog_generator(...)` no longer sends raw previous history or `internal_monologue` to the generator LLM.
- `dialog_evaluator(...)` no longer sends `last_user_message`, raw history, or `internal_monologue` to the evaluator LLM.
- Generator and evaluator prompts describe `content_anchors` as the only semantic content authority.
- The live evaluator regression rejects the known stale milk-tea dialog against the magic/beauty anchors.
- The full live dialog regression produces a magic/beauty reply and not a stale milk-tea reply.
- Focused deterministic tests pass.
- RAG/current-row exclusion behavior and save-history ordering are unchanged.
- Independent code review is complete and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Dialog style becomes less varied without raw history | Keep `rhetorical_strategy`, `linguistic_style`, `contextual_directives`, and character profile style constraints | Live full-dialog regression and existing generator live contract tests |
| Evaluator still passes stale-topic dialog after payload cleanup | Make the evaluator prompt anchor-only and run the live evaluator regression | `test_live_dialog_evaluator_rejects_stale_topic_dialog` |
| Tests keep old payload assumptions alive | Update deterministic payload tests and static grep for removed fields | `test_conversation_progress_history_policy.py` and `rg` check |
| Agent accidentally changes RAG/history behavior | Explicitly keep RAG/history/persistence out of change surface | `git diff` review and independent code review |

## Execution Evidence

- Regression baseline:
  - `venv\Scripts\python.exe -m pytest -o addopts='' tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_evaluator_rejects_stale_topic_dialog -q -s -m live_llm` failed before the final prompt fix with `feedback="Passed"`.
  - Baseline trace: `test_artifacts\llm_traces\dialog_anchor_boundary_live_llm__evaluator_accepts_stale_milk_tea_dialog.json`.
  - Additional prompt-debug failing traces after payload cleanup: `...__20260517T025246975940Z.json`, `...__20260517T025525294784Z.json`, `...__20260517T025735426869Z.json`, `...__20260517T025938116491Z.json`.
- Deterministic payload tests before implementation:
  - Payload-boundary tests were updated while the production worker was already editing `dialog_agent.py`; the intended old-shape failing run was not captured separately because the user requested parallel production/test execution.
  - The replacement tests now assert generator payload keys are exactly `linguistic_directives`, `contextual_directives`, `user_name`, and evaluator payload excludes `last_user_message`, raw history, and `internal_monologue`.
- Static checks:
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_anchor_boundary_live_llm.py` passed.
  - Expanded compile check for changed source/tests passed.
  - Prompt render check printed `prompt render ok`.
  - `rg "last_user_message|tone_history" src\kazusa_ai_chatbot\nodes\dialog_agent.py tests\test_dialog_agent.py tests\test_conversation_progress_history_policy.py` returned exit code 1 with no matches.
- Deterministic tests after implementation:
  - `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_history_policy.py -q` passed: 5 passed.
  - `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q` passed: 15 passed.
  - `venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py -q` passed: 12 passed.
- Real LLM evaluator regression:
  - `venv\Scripts\python.exe -m pytest -o addopts='' tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_evaluator_rejects_stale_topic_dialog -q -s -m live_llm` passed.
  - Final trace: `test_artifacts\llm_traces\dialog_anchor_boundary_live_llm__evaluator_accepts_stale_milk_tea_dialog__20260517T030604393464Z.json`.
  - Manual judgment: evaluator payload had no `last_user_message`, raw history, or `internal_monologue`; evaluator rejected stale milk-tea dialog at `retry=2/3` for missing `[ANSWER]` and `[PROGRESSION]`.
- Real LLM full dialog regression:
  - `venv\Scripts\python.exe -m pytest -o addopts='' tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_agent_keeps_content_anchors_over_stale_history -q -s -m live_llm` passed.
  - Final trace: `test_artifacts\llm_traces\dialog_anchor_boundary_live_llm__magic_anchor_after_milk_tea_history__20260517T030615824107Z.json`.
  - Manual judgment: generator payload had no raw previous milk-tea history; final dialog stayed on the magic/beauty tease and did not answer the stale milk-tea topic.
- Independent code review:
  - Subagent `019e33e6-a5f4-7032-9e14-042bbab856ee` reviewed the diff and approved with no blocking findings.
  - Reviewer confirmed production scope was limited to `src\kazusa_ai_chatbot\nodes\dialog_agent.py`, generator payload keys were exactly `linguistic_directives`, `contextual_directives`, `user_name`, and evaluator payload keys were exactly `retry`, `final_dialog`, `linguistic_directives`, `contextual_directives`.
  - Reviewer confirmed prompt direction was coherent and anchor-first, and tests covered both stale-output rejection and full dialog anchor adherence.
  - Residual risk recorded: pre-existing max-retry behavior still forces loop termination after repeated evaluator rejections, so repeated generator failures can still return the last rejected dialog. This is outside the approved change surface for this plan.
