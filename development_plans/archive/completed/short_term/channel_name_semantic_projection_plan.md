# channel name semantic projection plan

## Summary

- Goal: project a usable group/channel name into existing LLM-facing semantic
  text without adding a new prompt field or source-packet structure.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: bigbang semantic projection with compatible
  optional conversation-row metadata for future group-review reads.
- Highest-risk areas: leaking platform ids into prompts, feeding group labels
  to RAG as query context, duplicating source text in self-cognition, and
  expanding the change into routing, delivery, memory, or scheduler behavior.
- Acceptance criteria: live `/chat` and self-cognition group review can render
  a usable group name inside existing semantic text; synthetic labels are
  dropped; no new LLM-facing group-name field exists; focused tests and static
  greps pass.

## Context

`ChatRequest.channel_name` already exists and live service state already
receives it. The relevance agent already sees it as a weak scene/topic clue.
Downstream shared state uses `channel_topic` as the existing scene text field.

The implementation must preserve that contract:

```text
raw adapter channel_name + raw relevance topic
-> deterministic prompt-safe scene text
-> existing LLM-facing channel_topic or self-cognition instruction
```

The canonical state value `channel_topic` must remain the topic produced by
relevance unless a specific prompt builder is rendering scene text for an LLM.
This avoids accidentally feeding the group label to RAG and other consumers
that only need the topic.

Self-cognition group review already uses an instruction sentence as the
model-facing source boundary. A usable group name belongs in that sentence,
not in a new source-packet field.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  or signing off this plan.
- `local-llm-architecture`: load before changing prompt-facing context,
  LLM prompt contracts, RAG/cognition/dialog payloads, or LLM budget.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files or tests containing CJK
  prompt strings or CJK test data.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires user approval, status `approved` or `in_progress`,
  and a direct user instruction to execute.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off any major checklist stage, reread this entire plan before
  starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in
  `Execution Evidence`.
- Execution must use parent-led native subagent execution. If native subagent
  capability is unavailable, stop before execution unless the user explicitly
  approves fallback execution.
- Use `venv\Scripts\python` for Python commands. Use `apply_patch` for manual
  edits. Check `git status --short` before production-code edits. Do not read
  `.env`.
- Do not add a new LLM call, model route, feature flag, collection, migration,
  prompt stage, compatibility shim, fallback mapper, or parallel prompt
  vocabulary.
- Do not add LLM-facing fields such as `group_chat_name`, `scene_name`,
  `channel_label`, `prompt_channel_topic`, or `group_name`.
- The group/channel name is a label only. It must not decide relevance,
  routing, delivery, permission, persistence, memory writes, action
  feasibility, reply anchoring, or scheduler behavior.
- Deterministic code owns label sanitization and default/synthetic label
  rejection. LLMs must not infer whether `Group 123` is a real group name.
- `channel_topic` remains weak scene text in prompts. Visible body text, reply
  context, mentions, explicit addressees, and attachment facts stay stronger.
- RAG, consolidation, reflection promotion, scheduler, dispatcher, and delivery
  must not receive a new raw `channel_name` prompt field.

## Must Do

- Create `src/kazusa_ai_chatbot/channel_scene_projection.py` with:
  - `usable_channel_label(channel_type: str, channel_name: str) -> str`
  - `project_channel_topic_text(channel_type: str, channel_name: str, channel_topic: str) -> str`
  - `project_group_review_instruction_preamble(channel_name: str) -> str`
- The helper must:
  - return an empty label for private/system channels;
  - reject empty labels, multiline labels, pure ids, `Private`, and synthetic
    `Group <id>` / `QQ group <id>` labels;
  - bound label length before rendering;
  - produce group phrasing only for group channels with usable labels;
  - contain no LLM calls, DB reads, adapter API calls, routing decisions, or
    delivery decisions.
- Keep relevance output JSON shape unchanged. The relevance LLM returns the
  topic only; deterministic code must not ask it to format the group-name
  sentence.
- Pass existing `channel_name` through persona state as internal metadata so
  prompt builders can render scene text without mutating canonical
  `state["channel_topic"]`.
- Use the projected scene text only in LLM-facing prompt payloads that already
  consume `channel_topic` for scene understanding:
  - message decontextualizer input payload;
  - cognition-chain scene payload;
  - selected text-surface planning payload through cognition-chain state.
- Keep RAG/capability request builders on the raw canonical `state["channel_topic"]`.
- Persist sanitized usable `channel_name` as optional metadata on new
  conversation-history rows. This scalar is not embedded, indexed, or used for
  routing.
- Include optional `channel_name` in reflection message reads and carry one
  usable group label into `GroupActivityWindow`.
- In group-review case construction, store the sanitized carried label in
  existing `case["channel_topic"]`. For group-review cases only, this value is
  the channel label used by the source instruction, not a topic summary.
- Render group-review instruction as:
  `下面是我在“<label>”群聊里看到的现场观察资料。`
  when a usable label exists, otherwise preserve the existing fallback
  instruction.
- Add focused deterministic tests listed in `Verification`.
- Update only the subsystem docs that define `/chat.channel_name`,
  `conversation_history`, or group-review source projection.

## Deferred

- Do not add a new public request field, cognition scene field, source-packet
  field, RAG request field, memory field, or prompt schema field for group
  name.
- Do not redesign relevance, decontextualization, cognition, L2d, L3, dialog,
  RAG, memory consolidation, reflection promotion, scheduler, dispatcher, or
  delivery.
- Do not add adapter group-name discovery, NapCat `get_group_info` lookup, or
  adapter persistent group-name cache in this plan.
- Do not treat channel name as a topic by itself when messages contain no
  topic.
- Do not backfill historical conversation rows.
- Do not store or project raw platform channel ids as names.
- Do not make group labels stronger than visible chat evidence.

## Cutover Policy

Overall strategy: bigbang prompt-facing projection with compatible optional
row metadata.

| Area | Policy | Instruction |
|---|---|---|
| Projection helper | bigbang | Use one deterministic helper for sanitization and scene-text rendering. |
| Live prompt payloads | bigbang | Render projected text at prompt boundaries; do not mutate canonical `state["channel_topic"]`. |
| Relevance output | unchanged | Keep existing JSON fields and raw topic output. |
| RAG/capability requests | unchanged | Keep raw canonical topic; do not add group labels. |
| Conversation rows | compatible | New rows may store optional sanitized `channel_name`; old rows omit it. |
| Group-review source instruction | bigbang | Use label-aware preamble when available; keep fallback when absent. |
| Public schemas | unchanged | No new `/chat`, cognition scene, RAG, or source-packet field. |

## Cutover Policy Enforcement

- Bigbang areas must be rewritten directly to the plan contract.
- Compatible areas preserve only the optional scalar metadata named here.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

Live group chat with usable label:

```text
raw channel_topic: 新番角色和剧情走向
LLM-facing channel_topic: “动画讨论群”群聊中正在讨论：新番角色和剧情走向
```

Live group chat with no usable label:

```text
raw channel_topic: HPE 网络研讨会截图
LLM-facing channel_topic: HPE 网络研讨会截图
```

Live group chat with usable label and no topic:

```text
LLM-facing channel_topic: “动画讨论群”群聊里刚出现这条消息，但还没有形成明确连续话题。
```

Self-cognition group review with usable label:

```text
下面是我在“动画讨论群”群聊里看到的现场观察资料。
```

No target state includes a new LLM-facing group-name field.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Projection lane | Existing `channel_topic` text and group-review instruction | Matches the requested non-overdesigned semantic text projection. |
| Canonical topic | Keep raw topic in shared state | Prevents RAG and non-scene consumers from receiving group labels. |
| Label validation | Deterministic helper | Local LLM should not classify synthetic or unsafe labels. |
| Self-cognition carrier | Existing case/context data, rendered into instruction | Avoids a new source-packet field. |
| Historical rows | No backfill | Missing labels degrade to current generic wording. |
| Adapter lookup | Deferred | Semantic projection should not require adapter API discovery in the first implementation. |

## Contracts And Data Shapes

Helper contract:

```python
def usable_channel_label(channel_type: str, channel_name: str) -> str: ...

def project_channel_topic_text(
    channel_type: str,
    channel_name: str,
    channel_topic: str,
) -> str: ...

def project_group_review_instruction_preamble(channel_name: str) -> str: ...
```

Allowed `project_channel_topic_text(...)` outputs:

```text
<topic>
“<label>”群聊中正在讨论：<topic>
“<label>”群聊里刚出现这条消息，但还没有形成明确连续话题。
```

Forbidden prompt outputs:

```text
“Group 227608960”群聊中正在讨论：...
“QQ group 227608960”群聊中正在讨论：...
“227608960”群聊中正在讨论：...
“Private”群聊中正在讨论：...
```

Conversation-history rows:

```python
{
    "channel_name": str,  # optional sanitized metadata
}
```

Rules:

- `channel_name` is optional.
- It is not part of embedding text.
- It is not indexed by this plan.
- It is not used for identity, delivery, routing, memory target selection, or
  cache invalidation.

Group activity windows:

```python
channel_name: str
```

Rules:

- Internal source-preparation metadata only.
- Derived from bounded reflection input rows.
- Used only to render group-review instruction text.
- Not rendered as a separate source-packet field.

## Data Migration

No migration or historical backfill is required. Existing rows without
`channel_name` continue to use fallback group-review wording.

## LLM Call And Context Budget

- Live `/chat` call count: unchanged.
- Self-cognition/reflection call count: unchanged.
- New live or background LLM calls: none.
- Context increase: one bounded quoted label inside existing scene text.
- Blocking behavior: no new service-side blocking I/O.
- Prompt strength: projected `channel_topic` remains weak scene context.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/channel_scene_projection.py`
  - Deterministic label validation and scene-text projection.

- `tests/test_channel_scene_projection.py`
  - Label acceptance/rejection, private/system no-op behavior, topic
    composition, fallback wording, and length bounding.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Carry existing `channel_name` as internal persona state metadata.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Pass existing top-level `channel_name` into persona state.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - Render projected `channel_topic` in the LLM payload only.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Render projected scene `channel_topic` for cognition-chain input only.

- `src/kazusa_ai_chatbot/brain_service/intake.py`
  - Persist sanitized usable `channel_name` on new inbound user rows.

- `src/kazusa_ai_chatbot/db/schemas.py`
  - Document optional `ConversationMessageDoc.channel_name`.

- `src/kazusa_ai_chatbot/db/conversation_reflection.py`
  - Include optional `channel_name` in reflection message projection.

- `src/kazusa_ai_chatbot/reflection_cycle/activity_windows.py`
  - Carry one usable group label on `GroupActivityWindow`.

- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Set `case["channel_topic"]` to the sanitized window label for group-review
    cases.

- `src/kazusa_ai_chatbot/self_cognition/projection.py`
  - Render label-aware group-review instruction preamble from
    `case["channel_topic"]`.

- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Prevent group-review label carrier data from also seeding cognition-scene
    `channel_topic`.

- `src/kazusa_ai_chatbot/brain_service/README.md`
  - Clarify `channel_name` as optional human-readable label metadata.

- `src/kazusa_ai_chatbot/db/README.md`
  - Document optional non-embedded `conversation_history.channel_name`.

- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Document group-review instruction projection.

### Keep

- `ChatRequest` public field shape.
- `CognitionChainInputV1.scene` shape.
- RAG request shape.
- Source packet public shape.
- Dispatcher, delivery receipt, scheduler, consolidation, memory evolution,
  reflection promotion, and adapter behavior.

## Overdesign Guardrail

- Actual problem: Kazusa needs a prompt-safe named-group scene hint in live
  chat and group-review self-cognition.
- Minimal change: sanitize an existing label and fold it into existing
  `channel_topic` or source-instruction text.
- Ownership boundaries: adapters provide labels; deterministic code validates
  and formats labels; LLM stages read weak scene text; RAG retrieves evidence;
  cognition judges stance/action; dialog renders wording; delivery uses ids.
- Rejected complexity: no new prompt field, scene contract, source-packet
  field, RAG field, LLM call, migration, adapter lookup, compatibility shim,
  fallback mapper, feature flag, or group-name memory.
- Evidence threshold: add richer channel identity only after a concrete
  failure proves existing semantic text projection is insufficient.

## Agent Autonomy Boundaries

- The responsible agent must keep behavior inside the named change surface.
- The responsible agent must not introduce alternate migration strategies,
  compatibility layers, fallback paths, or extra features.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If equivalent helper behavior already exists, reuse or move it instead of
  duplicating it.
- If code inspection shows that a listed file is not the correct owner, stop
  and update this plan before editing a different owner.

## Implementation Order

1. Add `tests/test_channel_scene_projection.py`.
   - Expected before implementation: import failure for missing helper module.
2. Add focused live prompt-projection tests.
   - Files: `tests/test_msg_decontexualizer.py` and
     `tests/test_persona_supervisor2.py`.
   - Expected before implementation: group label is absent from LLM-facing
     scene text or raw state is mutated.
3. Add persistence and reflection-window tests.
   - Files: `tests/test_service_input_queue.py`,
     `tests/test_reflection_cycle_activity_windows.py`.
   - Expected before implementation: no carried usable label.
4. Add self-cognition source-instruction tests.
   - File: `tests/test_self_cognition_group_review_source.py`.
   - Expected before implementation: instruction remains generic.
5. Start the production-code subagent with this approved plan and failing
   focused tests.
6. Implement helper, live prompt projection, persistence metadata,
   reflection-window carry, and group-review instruction projection.
7. Run focused tests and fix only plan-scoped failures.
8. Update the listed docs.
9. Run all verification commands.
10. Run independent code review and remediate in-scope findings.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract before production
  implementation starts.
- Production-code subagent: exactly one native subagent, started after focused
  tests are established; owns production code changes only.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused tests established
  - Covers: implementation steps 1 through 4.
  - Verify: focused tests run and expected failures/baseline are recorded.
  - Evidence: record commands and expected failures in `Execution Evidence`.
  - Sign-off: `Codex/2026-07-02` after evidence is recorded.

- [x] Stage 2 - production implementation complete
  - Covers: implementation steps 5 through 7.
  - Verify: focused tests pass.
  - Evidence: record changed files and focused test output.
  - Sign-off: `Codex/2026-07-02` after evidence is recorded.

- [x] Stage 3 - docs and full verification complete
  - Covers: implementation steps 8 and 9.
  - Verify: all commands in `Verification`.
  - Evidence: record docs touched, static grep output, and test output.
  - Sign-off: `Codex/2026-07-02` after evidence is recorded.

- [x] Stage 4 - review gate complete by no-subagent fallback
  - Covers: implementation step 10.
  - Verify: review approval plus rerun affected tests after fixes.
  - Evidence: record findings, fixes, rerun commands, residual risks, and
    approval status.
  - Sign-off: `Codex/2026-07-02` after fallback review evidence is recorded.

## Verification

### Static Greps

- `rg -n "group_chat_name|scene_name|prompt_channel_topic" src tests`
  - Expected: no matches.
- `rg -n "\"channel_label\"|'channel_label'" src tests`
  - Expected: no matches.
- `rg -n "Group <id>|QQ group|Group \\{" src\\kazusa_ai_chatbot tests`
  - Expected: no brain prompt projection of synthetic group names. Test
    fixtures may contain rejected-label examples.
- `rg -n "\"channel_name\"" src\\kazusa_ai_chatbot\\rag src\\kazusa_ai_chatbot\\consolidation src\\kazusa_ai_chatbot\\dispatcher src\\kazusa_ai_chatbot\\calendar_scheduler`
  - Expected: no new raw group-name consumers in these subsystems.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_channel_scene_projection.py -q`
- `venv\Scripts\python -m pytest tests\test_msg_decontexualizer.py tests\test_persona_supervisor2.py -q`
- `venv\Scripts\python -m pytest tests\test_service_input_queue.py tests\test_reflection_cycle_activity_windows.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_msg_decontexualizer.py tests\test_cognition_chain_core_contracts.py tests\test_cognition_resolver_loop.py -q`
- `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py -q`
- `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py tests\test_reflection_cycle_prompt_contracts.py -q`

## Independent Plan Review

Review completed during drafting. Findings addressed:

- Removed adapter group-name lookup from scope; it is not required for semantic
  projection and would expand adapter behavior.
- Changed live projection from canonical-state mutation to prompt-boundary
  rendering so RAG keeps the raw topic.
- Removed duplicate examples, discovery rationale, live LLM inspection, and
  broad adapter verification.
- Made helper ownership explicit in `channel_scene_projection.py`.
- Tightened change surface and verification to implementation-owned files.
- Adjusted plan class to `large` because the remaining scope crosses prompt,
  persistence, reflection-window, and self-cognition boundaries.

Approval status: in progress after explicit user approval for fallback
execution without subagents.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for changed Python, tests, prompts,
  documentation, and commands.
- Plan alignment: `Must Do`, `Deferred`, change surface, exact contracts,
  implementation order, verification gates, and acceptance criteria.
- Architecture risks: prompt/RAG payload leaks, synthetic label leakage,
  persistence misuse, source-packet shape changes, and avoidable blast radius.
- Regression quality: focused tests, static greps, evidence, and lifecycle
  records.

The parent agent fixes review findings only when the fix is inside the
approved change surface. Contract or boundary changes require plan update and
user approval before implementation.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Live group `/chat` renders usable labels into existing LLM-facing
  `channel_topic` text without mutating canonical raw topic state.
- Private/system channels do not receive group-label phrasing.
- Synthetic labels such as `Group <id>`, `QQ group <id>`, `Private`, and pure
  ids are not projected into prompts.
- Self-cognition group review renders a usable group label in the existing
  instruction sentence and falls back to current generic wording when absent.
- No new LLM-facing group-name field exists.
- RAG, consolidation, scheduler, dispatcher, and delivery have no new raw
  `channel_name` prompt consumer.
- Focused tests, regression tests, static greps, and the user-approved
  no-subagent fallback review pass with evidence recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Group label leaks into RAG | Render projection only at prompt boundaries | Static grep and focused prompt-payload tests |
| Synthetic ids leak into prompts | Deterministic usable-label gate | Helper tests and static grep |
| Self-cognition duplicates digest/topic text | Render name only in instruction preamble | Group-review rendering tests |
| Optional row metadata becomes control data | Document metadata-only rule and avoid indexes/embedding | DB docs and grep checks |

## Execution Evidence

- Plan review performed during drafting.
- 2026-07-02: User explicitly approved executing this plan without subagents.
  Status moved to `in_progress`; execution will use parent-led fallback with
  self-review evidence instead of native subagent execution.
- 2026-07-02 Stage 1 focused tests established. Baseline commands:
  - `venv\Scripts\python -m pytest tests\test_channel_scene_projection.py -q`
    failed during collection with missing
    `kazusa_ai_chatbot.channel_scene_projection`.
  - `venv\Scripts\python -m pytest tests\test_msg_decontexualizer.py tests\test_persona_supervisor2.py -q`
    failed 3 tests: decontextualizer raw topic payload, missing persona
    `channel_name` carry, cognition-chain raw scene topic.
  - `venv\Scripts\python -m pytest tests\test_service_input_queue.py tests\test_reflection_cycle_activity_windows.py -q`
    failed 2 tests: missing persisted usable `channel_name`, missing
    `GroupActivityWindow.channel_name`.
  - `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`
    failed 2 tests: missing group-review case label carry and generic
    group-review instruction preamble.
- 2026-07-02 Stage 2 inspection found `self_cognition.runner` also consumes
  `case["channel_topic"]` as cognition-scene topic. Added it to the change
  surface to keep group-review label carrier data instruction-only.
- 2026-07-02 Stage 2 implementation complete. Changed production files:
  `channel_scene_projection.py`, persona state/prompt builders, intake,
  DB schemas/reflection projection, activity windows, self-cognition sources,
  projection, and runner. Focused tests passed:
  - `venv\Scripts\python -m pytest tests\test_channel_scene_projection.py -q`
    passed 6.
  - `venv\Scripts\python -m pytest tests\test_msg_decontexualizer.py tests\test_persona_supervisor2.py -q`
    passed 28.
  - `venv\Scripts\python -m pytest tests\test_service_input_queue.py tests\test_reflection_cycle_activity_windows.py -q`
    passed 41.
  - `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`
    passed 24.
- 2026-07-02 Stage 3 verification prep found the original `channel_label`
  static grep conflicted with the required helper name `usable_channel_label`
  and an existing unrelated local variable. Verification now forbids quoted
  `channel_label` field tokens instead.
- 2026-07-02 Stage 3 docs updated:
  `src/kazusa_ai_chatbot/brain_service/README.md`,
  `src/kazusa_ai_chatbot/db/README.md`, and
  `src/kazusa_ai_chatbot/self_cognition/README.md`.
- 2026-07-02 Stage 3 focused tests passed after docs:
  - `venv\Scripts\python -m pytest tests\test_channel_scene_projection.py -q`
    passed 6.
  - `venv\Scripts\python -m pytest tests\test_msg_decontexualizer.py tests\test_persona_supervisor2.py -q`
    passed 28.
  - `venv\Scripts\python -m pytest tests\test_service_input_queue.py tests\test_reflection_cycle_activity_windows.py -q`
    passed 41.
  - `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`
    passed 24.
- 2026-07-02 Stage 3 regression tests passed:
  - `venv\Scripts\python -m pytest tests\test_msg_decontexualizer.py tests\test_cognition_chain_core_contracts.py tests\test_cognition_resolver_loop.py -q`
    passed 58.
  - `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py -q`
    passed 29.
  - `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py tests\test_reflection_cycle_prompt_contracts.py -q`
    passed 35.
- 2026-07-02 Stage 3 static greps:
  - `rg -n "group_chat_name|scene_name|prompt_channel_topic" src tests`
    returned no matches.
  - `rg -n "\"channel_label\"|'channel_label'" src tests` returned no
    matches.
  - `rg -n "Group <id>|QQ group|Group \\{" src\kazusa_ai_chatbot tests`
    returned only test fixture/documentation examples, with no
    `src\kazusa_ai_chatbot` prompt-projection matches.
  - `rg -n "\"channel_name\"" src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\consolidation src\kazusa_ai_chatbot\dispatcher src\kazusa_ai_chatbot\calendar_scheduler`
    returned no matches.
- 2026-07-02 Stage 4 no-subagent fallback review completed after user
  instruction to execute without subagents. Review checked plan alignment,
  helper determinism, canonical `channel_topic` preservation, prompt-boundary
  projection only, source-packet shape preservation, optional sanitized
  metadata only, and static evidence that RAG/consolidation/scheduler/
  dispatcher do not receive a new raw `channel_name` consumer. Findings: no
  in-scope fixes required. Residual risk: review was not independent because
  the user explicitly requested no subagent execution.
