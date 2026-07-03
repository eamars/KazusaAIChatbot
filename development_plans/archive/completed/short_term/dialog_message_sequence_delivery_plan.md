# dialog message sequence delivery plan

## Summary

- Goal: Replace the one-bubble dialog rendering assumption with a
  platform-neutral outbound message sequence contract for normal chat delivery.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`.
- Overall cutover strategy: bigbang for L3/dialog prompt contract, adapter
  normal-chat delivery, and tests. The public `ChatResponse.messages` shape is
  retained.
- Highest-risk areas: moving sequence decisions into dialog, introducing
  platform-specific brain prompt assumptions, blocking adapter event handlers
  during follow-up delays, applying native reply to every follow-up message,
  breaking inline mention rendering across follow-up messages, and weakening
  fixed-format/code preservation.
- Acceptance criteria: `final_dialog` and `ChatResponse.messages` are treated
  as ordered outbound messages; L3 owns the intended send sequence shape;
  dialog renders each sequence item without deciding delivery mechanics;
  Discord and NapCat normal-chat adapters send message lists as multiple
  platform sends; native reply anchoring applies only to the first outbound
  message; inline delivery mention candidates are rendered only where authored
  `@display_name` tokens appear in each outbound message; follow-up delay is
  adapter-owned, deterministic, non-blocking, and length-based; focused tests
  and prompt greps prove the old one-bubble contract is gone from the active
  prompt path.

## Context

The current product intent is:

```text
one cognition iteration
  -> one L3 content plan
  -> dialog renders final_dialog as 1-N outbound messages
  -> normal chat adapter sends each message separately
  -> adapter applies native reply only to the first message
  -> adapter renders inline mention candidates inside each message independently
  -> adapter delays follow-up sends without blocking the inbound handler
```

The current codebase has a split contract:

- The service/API contract already exposes `ChatResponse.messages: list[str]`
  and the Brain Service ICD says `use_reply_feature` applies to the first
  outbound message.
- The completed inline delivery mention work defines `delivery_mentions` as
  platform-neutral inline render candidates. Dialog may author visible
  `@display_name` tokens; adapters replace exact authored tokens with native
  mentions where those tokens appear.
- L3 and dialog prompts still describe `final_dialog` as one chat bubble with
  newline-joined layout fragments.
- Discord and NapCat normal-chat adapters currently join returned messages with
  newline before sending, which prevents logical follow-up delivery and risks
  applying inline mention replacement to a combined body instead of each
  outbound message.
- Tests explicitly assert the obsolete one-bubble prompt contract.

The chat-history research artifact remains evidence for online text-chat
surface behavior only. Runtime prompts must stay platform-neutral and must not
mention QQ, Discord, or any specific target adapter as the brain's assumed
delivery channel.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  or signing off this plan.
- `local-llm-architecture`: load before changing L3, dialog, prompt contracts,
  response-path LLM context, adapter/brain ownership boundaries, or tests that
  lock prompt behavior.
- `debug-llm`: load before any live LLM prompt comparison, prompt quality
  artifact, or generated-dialog quality review.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files or tests containing CJK prompt
  strings or CJK test data.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off any major checklist stage, reread this entire plan before
  starting the next stage.
- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires user approval, status `approved` or `in_progress`,
  and a direct user instruction to execute.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.
- Use parent-led native subagent execution. If native subagents are unavailable,
  stop before execution unless the user explicitly approves fallback execution.
- Keep the brain platform-agnostic. L3 and dialog prompts must use
  platform-neutral wording such as ordinary online text chat or outbound chat
  messages. Do not add QQ-specific, Discord-specific, or adapter-specific
  assumptions to brain prompts.
- Keep dialog as a renderer. Dialog must follow upstream `content_plan`,
  `rendering`, voice, and linguistic directives. It must not decide whether to
  speak, change stance, choose native reply behavior, calculate delay, infer the
  platform, or independently choose delivery mechanics.
- L3 owns the intended visible message sequence shape in `content_plan.rendering`.
  Dialog owns wording for each message in that sequence.
- Deterministic adapter code owns native reply rendering, inline mention
  rendering, platform chunking, send ordering, delivery receipts, follow-up
  delay, task scheduling, and transport errors.
- Prompt edits must prefer positive rendering instructions and concise
  ownership statements. Do not add long lists of forbidden style examples or
  platform-specific negative constraints.
- Keep essential fixed-format protection: code, JSON, logs, commands,
  patches, tables, and fenced blocks must survive as complete message strings
  without semantic or formatting corruption.
- Preserve existing `ChatResponse` field names and types. Do not add
  per-message reply targets, per-message mention targets, delay fields, channel
  fields, or platform fields to the brain response in this plan.

## Must Do

- Update the L3 content-plan prompt so `rendering` describes outbound message
  sequence shape, message count intent, compactness, and fixed-format handling
  instead of single-bubble layout.
- Update the dialog generator prompt so `final_dialog` means ordered outbound
  messages. Each string is one message the adapter may send separately.
- Remove active prompt wording that says runtime joins `final_dialog` with
  newlines, that each item is a single-bubble layout unit, or that final output
  is one visible chat bubble.
- Keep `content_plan` as semantic authority and keep dialog free of decision
  ownership.
- Keep online text-chat surface rules platform-neutral.
- Update normal Discord adapter delivery so `ChatResponse.messages` is sent as
  a logical message sequence.
- Update normal NapCat adapter delivery so `ChatResponse.messages` is sent as
  a logical message sequence.
- Apply native reply behavior only to the first logical outbound message.
- Preserve inline delivery mention rendering for every logical outbound
  message. Each message replaces only exact authored `@display_name` tokens
  that appear in that message.
- Keep service-side inline mention candidate construction compatible with
  multi-message `final_dialog`; candidate discovery may scan the joined visible
  response, while adapter delivery must preserve the message list and render
  candidates per logical message.
- Delay only follow-up logical messages, not platform chunks within the same
  logical message.
- Calculate follow-up delay in adapter-owned deterministic code from the
  follow-up message length with a clamped 1-5 second range.
- Schedule delayed follow-up delivery through adapter-owned async tasks so the
  inbound event handler does not wait on the delay.
- Post the normal delivery receipt using the first successfully sent platform
  message id, preserving the existing delivery receipt model.
- Update tests that currently assert one-bubble prompt wording or newline
  joining behavior.
- Add focused tests for the shared delay calculation, Discord/NapCat
  first-message-only native reply behavior, and per-message inline mention
  rendering.

## Deferred

- Do not add a new LLM stage, evaluator, retry loop, repair prompt, or style
  classifier.
- Do not add a large render-policy schema or per-message metadata schema.
- Do not add new `ChatResponse` fields.
- Do not change relevance, L2, L2d, RAG, consolidation, memory lifecycle,
  scheduler, or persistence semantics.
- Do not change `persona_relevance_agent` ownership of `use_reply_feature`.
- Do not make dialog decide reply anchoring, mentions, delay, or platform
  delivery.
- Do not reintroduce `mention_target_user`, prefix mention fallback behavior,
  or any per-message mention metadata in this plan.
- Do not change the dispatcher `send_message` public contract in this plan.
- Do not update background artifact result-ready dispatch sequence semantics in
  this plan; that path uses the dispatcher single-message contract and needs a
  separate dispatcher/adapter callback plan if multi-message delayed delivery is
  required there.
- Do not add typing indicators, read receipts, typing-state simulation, or
  platform-specific delivery affordances.
- Do not mine or embed real private chat messages into automated tests. Use
  synthetic or paraphrased fixtures.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| L3 rendering prompt | bigbang | Replace single-bubble wording with outbound message-sequence wording in the active prompt. |
| Dialog generator prompt | bigbang | Replace one-bubble/newline-fragment wording with ordered outbound-message wording. |
| Normal `/chat` response API | compatible | Preserve existing `ChatResponse.messages`, `use_reply_feature`, `delivery_mentions`, and `delivery_tracking_id` field names and types. |
| Discord normal-chat adapter delivery | bigbang | Stop joining `messages` into one text body before send. Send logical messages in order. |
| NapCat normal-chat adapter delivery | bigbang | Stop joining `messages` into one text body before send. Send logical messages in order. |
| Native reply | bigbang | Apply native reply anchoring to the first logical outbound message only. |
| Inline delivery mentions | bigbang | Preserve `delivery_mentions` candidates and apply exact inline token replacement independently inside each logical outbound message. |
| Follow-up delay | bigbang | Add adapter-owned deterministic delay for follow-up logical messages. |
| Dispatcher/background artifact delivery | compatible | Preserve current single-message dispatcher contract and leave sequence support to a separate plan. |
| Tests | bigbang | Rewrite obsolete one-bubble tests into message-sequence contract tests. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- If an area is `bigbang`, rewrite legacy prompt/test/adapter behavior instead
  of preserving fallback behavior.
- If an area is `compatible`, preserve only the explicitly listed existing
  public fields or contracts.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The normal live chat path has this observable behavior:

```text
adapter event
  -> brain service /chat
  -> persona graph
  -> L3 content_plan with rendering as message sequence instruction
  -> dialog_agent returns final_dialog: list[str]
  -> ChatResponse.messages preserves that list
  -> adapter starts an outbound sequence task
  -> messages[0] is sent immediately
  -> native reply applies to messages[0] only
  -> inline delivery mention replacement runs independently for each message
  -> messages[1:] are sent as ordinary follow-up messages after clamped
     length-based adapter delay
  -> delivery receipt records the first successful platform message id
```

Prompt-facing examples use platform-neutral language:

```json
{
  "content_plan": {
    "visible_goal": "先短反应，再补一句轻边界",
    "semantic_content": "不喜欢用户把钱和亲近关系绑在一起；要求对方收住",
    "voice": "防御，轻微不舒服，没有升级成严重冲突",
    "rendering": "生成 2 条连续发送的普通文字消息；第一条短反应，第二条补充边界"
  }
}
```

Dialog output:

```json
{
  "final_dialog": [
    "别拿钱说这个",
    "听着怪不舒服的"
  ]
}
```

Adapter delivery:

```text
message 1: native reply when requested; inline mention replacement if the text contains an authored tag
delay based on message 2 length
message 2: normal follow-up send; inline mention replacement if the text contains an authored tag
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Brain platform assumption | Use platform-neutral online text-chat wording only | The brain service is shared by Discord, NapCat QQ, debug UI, and future adapters. |
| Message cardinality owner | L3 owns intended sequence shape through existing `rendering` text | This avoids adding schema knobs and keeps dialog as renderer. |
| Dialog ownership | Dialog renders ordered message strings and preserves content-plan authority | Dialog must not become a delivery planner or second cognition stage. |
| Public response shape | Keep `ChatResponse.messages: list[str]` | The existing Brain Service ICD already models outbound messages as a list. |
| Reply anchoring | Apply native reply to the first logical outbound message only | Follow-ups inherit context from ordered delivery and should not repeatedly reply to the same user. |
| Inline mention rendering | Apply `delivery_mentions` as exact inline token replacements inside each logical message | Mentions should appear only where dialog authored the visible `@display_name` token, and message sequencing must not collapse those tokens into a single body. |
| Delay owner | Adapter owns deterministic delay | Delay is delivery mechanics, not cognition or dialog semantics. |
| Delay formula | Use hard-coded constants and clamp to 1-5 seconds | The requirement needs inspectable timing, not an LLM decision or config surface. |
| Adapter blocking | Use adapter-owned background delivery tasks | Inbound event handling should not remain open while follow-up delay sleeps. |
| Delivery receipt | Record first successful platform message id | Existing service model has one delivery tracking id for one assistant row. |
| Dispatcher scope | Leave dispatcher single-message contract unchanged | Extending runtime callback delivery requires a separate API/dispatcher plan. |

## Contracts And Data Shapes

### L3 Content Plan

Keep the existing content plan object shape:

```json
{
  "content_plan": {
    "visible_goal": "string",
    "semantic_content": "string",
    "voice": "string",
    "rendering": "string"
  }
}
```

Change the model-facing meaning of `rendering`:

```text
Before: single chat bubble layout, length, and fixed-format protection.
After: outbound message sequence shape, length, compactness, and fixed-format
       protection.
```

L3 may write prose such as:

```text
生成 1 条普通文字消息；短反应即可。
生成 2 条连续发送的普通文字消息；第一条短反应，第二条补充边界。
生成 2 条消息；第一条引入，第二条完整保留 fenced code block。
```

### Dialog Output

Keep the existing JSON shape:

```json
{
  "final_dialog": ["message 1", "message 2"]
}
```

New meaning:

- `final_dialog[i]` is one logical outbound message.
- `final_dialog` order is send order.
- A fixed-format block may occupy one complete message string.
- Dialog may include visible `@display_name` tokens only when the upstream
  content plan and conversation context call for addressing that user in the
  wording. Dialog does not emit mention metadata.
- Inline mention candidates remain a service/adapter delivery concern, not a
  dialog output field.

### Brain Service Response

Keep the existing `ChatResponse` fields and types: `messages`, `content_type`,
`attachments`, `use_reply_feature`, `delivery_mentions`,
`scheduled_followups`, `delivery_tracking_id`, and optional
`cognition_graph`. No new response fields are added.

### Adapter Sequence Delivery

Each normal-chat adapter implements a private sequence-delivery path. Exact
signatures may differ per adapter when platform objects differ, but behavior
must match:

- send `messages[0]` immediately;
- apply native reply to `messages[0]` only;
- apply inline delivery mention replacement independently to each logical
  message before platform chunking;
- send `messages[1:]` as normal follow-up messages;
- sleep before each follow-up using the follow-up text length;
- post delivery receipt using the first successful platform message id;
- log and stop the sequence if the first send fails;
- log later follow-up failures without changing the first delivery receipt.

### Shared Delay Helper

Create a small shared adapter helper with these constants:
`FOLLOWUP_MIN_DELAY_SECONDS = 1.0`,
`FOLLOWUP_MAX_DELAY_SECONDS = 5.0`, and
`FOLLOWUP_VISIBLE_CHARS_PER_SECOND = 12.0`.

`followup_delay_seconds(text: str) -> float` returns:

```text
min(5.0, max(1.0, len(text.strip()) / 12.0))
```

This helper must not inspect platform, channel, user, relationship, model
output metadata, or semantic content beyond visible string length.

## LLM Call And Context Budget

No new LLM calls are added.

| Stage | Before | After | Response path | Context impact |
|---|---:|---:|---|---|
| L3 content plan agent | 1 call when selected text surface runs | 1 call | yes | Prompt wording changes only; target length stays same or shorter. |
| Dialog generator | 1 call when selected text surface runs | 1 call | yes | Prompt wording changes only; target length stays same or shorter. |

Default context cap: 50k tokens.

Prompt edits must not add examples or policy blocks that increase the active
system prompt by more than 10 percent. If implementation needs a larger prompt
increase, stop and update this plan for approval before editing production
prompts.

## Change Surface

### Create

- `src/adapters/outbound_sequence.py`
  - Owns deterministic follow-up delay constants and
    `followup_delay_seconds`.
- `tests/test_adapter_outbound_sequence.py`
  - Covers delay clamp behavior and whitespace handling.

### Modify

- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`
  - Update `_CONTENT_PLAN_AGENT_PROMPT` wording for `rendering`.
  - Replace single-bubble prompt language with outbound message-sequence
    language.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Update `_DIALOG_GENERATOR_PROMPT`.
  - Align the `DialogAgentOutput.final_dialog` comment and logging wording
    with message sequence semantics.
  - Preserve parser and output shape.
- `src/adapters/discord_adapter.py`
  - Preserve `ChatResponse.messages` as logical messages.
  - Add adapter-owned outbound sequence task handling for normal `/chat`
    responses.
  - Apply native reply only to the first logical message.
  - Apply inline delivery mention replacement independently to each logical
    message before platform chunking.
  - Keep `_split_message` as platform chunking inside each logical message.
  - Post delivery receipt for the first successful platform message id.
- `src/adapters/napcat_qq_adapter/ws_adapter.py`
  - Preserve `ChatResponse.messages` as logical messages.
  - Add adapter-owned outbound sequence task handling for normal `/chat`
    responses.
  - Apply native reply only to the first logical message.
  - Apply inline delivery mention replacement independently to each logical
    message before platform chunking.
  - Post delivery receipt for the first successful platform message id.
- `src/kazusa_ai_chatbot/service.py`
  - Preserve `ChatResponse.messages` as the original `final_dialog` list.
  - Ensure inline delivery mention candidates are built from the complete
    visible response text across all `final_dialog` messages without joining
    the list for delivery.
- `src/kazusa_ai_chatbot/brain_service/README.md`
  - Clarify that adapters render `messages` as ordered outbound messages,
    apply native reply to the first message only, preserve inline mention
    rendering per message, and own follow-up delay.
- `tests/test_dialog_agent.py`
  - Replace one-bubble prompt contract assertions with message-sequence
    assertions.
- `tests/test_l3_dialog_content_plan_contract.py`
  - Replace fixture text and assertions that use `单个聊天气泡`.
- `tests/test_runtime_adapter_registration.py`
  - Add or update Discord and NapCat normal-chat sequence tests.
  - Assert first-message-only native reply behavior.
  - Assert inline delivery mentions are replaced independently in each logical
    message that contains an exact authored tag.
  - Assert follow-up delay is awaited inside the adapter sequence task using
    patched sleep/delay helpers.
- `tests/test_inline_delivery_mentions.py`
  - Preserve existing inline mention candidate construction tests.
  - Add or update a multi-message service case if adapter sequence work exposes
    a candidate construction regression.

### Keep

- `src/kazusa_ai_chatbot/brain_service/contracts.py`
  - Keep `ChatResponse` schema unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`
  - Keep `use_reply_feature` decision ownership unchanged.
- `src/kazusa_ai_chatbot/dispatcher/adapter_iface.py`
  - Keep single-message dispatcher adapter contract unchanged.
- `src/kazusa_ai_chatbot/dispatcher/remote_adapter.py`
  - Keep single-message runtime callback delivery unchanged.
- `src/kazusa_ai_chatbot/dispatcher/handlers.py`
  - Keep single-message dispatcher handler unchanged.

### Delete

- No source files are deleted in this plan.

## Overdesign Guardrail

- Fix only the current mismatch: prompts and adapters collapse `final_dialog`
  into one newline-joined bubble while the service API and target behavior use
  ordered outbound messages.
- Keep the minimal boundary: L3 owns sequence intent, dialog owns wording,
  service preserves response fields and builds inline mention candidates, and
  adapters own delivery mechanics.
- New LLM calls, response fields, per-message metadata, dispatcher sequence
  APIs, platform prompt assumptions, typing indicators, configurable delay, and
  compatibility shims require a separate approved plan.

## Agent Autonomy Boundaries

- Local implementation mechanics are allowed only when they preserve this
  plan's contracts and change surface.
- New architecture, alternate cutover strategies, compatibility layers,
  fallback paths, extra features, or target modules outside the listed change
  surface require stopping and updating this plan before implementation.
- Create only the shared delay helper named in this plan. Additional helper
  modules, wrappers, task managers, or abstractions are out of scope.
- The responsible agent must search for existing equivalent behavior before
  adding helper code. If equivalent delay or adapter task tracking already
  exists, use or move that behavior instead of duplicating it.
- Avoid unrelated cleanup, formatting churn, dependency upgrades, prompt
  rewrites, adapter modularization, and broad refactors.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy. If an instruction is impossible, stop and report the blocker.

## Implementation Order

1. Parent updates focused prompt contract tests.
   - File: `tests/test_dialog_agent.py`.
   - Replace `test_dialog_generator_prompt_describes_message_sequence_contract`
     with `test_dialog_generator_prompt_describes_message_sequence_contract`.
   - Expected pre-implementation result: fails because the prompt still says
     one chat bubble and newline join.
2. Parent updates L3 prompt contract fixtures.
   - File: `tests/test_l3_dialog_content_plan_contract.py`.
   - Replace `单个聊天气泡` fixture expectations with outbound message-sequence
     wording.
   - Expected pre-implementation result: fails because the current L3 prompt
     still describes single-bubble layout.
3. Parent adds focused delay helper tests.
   - File: `tests/test_adapter_outbound_sequence.py`.
   - Add tests for minimum clamp, maximum clamp, ordinary length calculation,
     and stripped empty text.
   - Expected pre-implementation result: fails because the helper module does
     not exist.
4. Parent adds adapter sequence tests.
   - File: `tests/test_runtime_adapter_registration.py`.
   - Add Discord normal-chat test proving two logical messages become two sends,
     native reply applies only to the first logical message, and delivery
     receipt records the first sent id.
   - Add Discord inline mention test proving `delivery_mentions` replacements
     happen independently in each logical message that contains an exact
     authored `@display_name` token.
   - Add NapCat normal-chat test proving two logical messages become two
     `send_msg` calls, native reply applies only to the first logical message,
     and delivery receipt records the first sent id.
   - Add NapCat inline mention test proving `delivery_mentions` replacements
     happen independently in each logical message that contains an exact
     authored `@display_name` token.
   - Patch sleep or delay helpers so tests run without real 1-5 second waits.
   - Expected pre-implementation result: fails because adapters still join the
     list before sending.
5. Parent starts the production-code subagent after the focused tests establish
   the expected failing contract.
6. Production-code subagent creates `src/adapters/outbound_sequence.py`.
   - Implement constants and `followup_delay_seconds`.
   - Keep the helper independent of platform and brain semantics.
7. Production-code subagent updates L3 prompt text.
   - Replace single-bubble `rendering` wording with outbound message-sequence
     wording.
   - Preserve content-plan authority and fixed-format protection.
8. Production-code subagent updates dialog prompt text.
   - Replace one-bubble/newline-fragment wording with ordered outbound-message
     wording.
   - Preserve JSON output shape and `content_plan` authority.
9. Production-code subagent updates Discord normal-chat adapter delivery.
   - Stop joining logical messages before delivery.
   - Add private outbound sequence task tracking.
   - Send first logical message immediately, with native reply if applicable.
   - Apply inline delivery mention replacement inside each logical message
     before platform chunking.
   - Send follow-up logical messages after `followup_delay_seconds`.
   - Keep `_split_message` only for platform chunking within one logical
     message.
   - Post delivery receipt for first successful platform message id.
10. Production-code subagent updates NapCat normal-chat adapter delivery.
    - Stop joining logical messages before delivery.
    - Add private outbound sequence task tracking.
    - Send first logical message immediately, with native reply if applicable.
    - Apply inline delivery mention replacement inside each logical message
      before platform chunking.
    - Send follow-up logical messages after `followup_delay_seconds`.
    - Post delivery receipt for first successful platform message id.
11. Production-code subagent updates service multi-message inline mention
    handling.
    - Preserve `ChatResponse.messages` as the original ordered list.
    - Build inline delivery mention candidates from the complete visible
      response text across all `final_dialog` messages.
    - Keep candidate construction platform-neutral and schema-compatible.
12. Production-code subagent updates Brain Service ICD and adapter ICD wording.
    - Clarify ordered message delivery, first-message native reply,
      per-message inline mention rendering, and adapter-owned delay without
      changing schema.
13. Parent runs focused tests.
14. Parent runs static greps for obsolete prompt and adapter join wording.
15. Parent runs broader regression tests listed in `Verification`.
16. Parent runs one-at-a-time live LLM prompt checks only after deterministic
    tests pass and records a human-readable `debug-llm` review artifact.
17. Parent starts independent code-review subagent after all planned
    verification passes.
18. Parent remediates review findings within the approved change surface and
    reruns affected verification.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused prompt and delay tests established
  - Covers: implementation steps 1-4.
  - Verify: run the focused tests and record expected failures.
  - Evidence: record failing test names and failure reasons in `Execution
    Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-07-01` after focused baseline failures were recorded.
- [x] Stage 2 - L3/dialog prompt contract updated
  - Covers: implementation steps 7-8.
  - Verify: focused dialog and L3 prompt tests pass.
  - Evidence: record test output and static grep results.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-07-01` after focused prompt tests and prompt greps passed.
- [x] Stage 3 - adapter sequence delivery implemented
  - Covers: implementation steps 6 and 9-11.
  - Verify: adapter delay helper tests, Discord/NapCat sequence tests, and
    multi-message inline mention tests pass.
  - Evidence: record test output and changed files.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-07-01` after delay, adapter sequence, per-message mention, and non-blocking tests passed.
- [x] Stage 4 - docs and regression verification complete
  - Covers: implementation steps 12-16.
  - Verify: static greps, focused tests, regression tests, and live LLM review
    artifact if run.
  - Evidence: record commands, outputs, and artifact path.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-07-01` after docs, static checks, prompt render, and regression tests passed.
- [x] Stage 5 - independent code review complete
  - Covers: implementation steps 17-18.
  - Verify: independent review completed, findings remediated, affected tests
    rerun.
  - Evidence: record review summary, fixes, commands rerun, and residual risks.
  - Handoff: plan can be marked completed only after this stage passes.
  - Sign-off: `Codex/2026-07-01`; performed as fallback single-agent review because the user explicitly instructed execution without subagents.

## Verification

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests/test_adapter_outbound_sequence.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_dialog_agent.py::test_dialog_generator_prompt_describes_message_sequence_contract -q`
- `venv\Scripts\python.exe -m pytest tests/test_l3_dialog_content_plan_contract.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_discord_handle_message_sends_brain_messages_as_sequence_with_first_reply_only -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_discord_on_message_replaces_inline_delivery_mentions_across_message_sequence -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_discord_on_message_does_not_wait_for_followup_delay -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_napcat_handle_event_sends_brain_messages_as_sequence_with_first_reply_only -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_napcat_handle_event_replaces_inline_delivery_mentions_across_message_sequence -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_napcat_handle_event_does_not_wait_for_followup_delay -q`
- `venv\Scripts\python.exe -m pytest tests/test_inline_delivery_mentions.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_cognition_prompt_contract_text.py::test_l3_content_plan_scope_preserves_complete_plan_deliverables -q`

### Regression Tests

- `venv\Scripts\python.exe -m pytest tests/test_dialog_agent.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_l3_dialog_content_plan_contract.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_inline_delivery_mentions.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_service_background_consolidation.py::test_chat_response_adds_inline_delivery_mentions_without_channel_gate tests/test_service_background_consolidation.py::test_chat_response_reply_feature_keeps_inline_delivery_mentions tests/test_service_background_consolidation.py::test_chat_response_adds_multiple_inline_delivery_mentions tests/test_service_background_consolidation.py::test_chat_response_preserves_message_sequence_for_inline_mentions tests/test_service_background_consolidation.py::test_chat_response_tracks_deliverable_assistant_row -q`

### Prompt Render Checks

- Run a prompt-render check for `_CONTENT_PLAN_AGENT_PROMPT`.
- Run a prompt-render check for `_DIALOG_GENERATOR_PROMPT`.
- Expected result: both prompts render without `.format(...)` placeholder
  errors and contain platform-neutral message-sequence wording.

### Static Greps

- `rg "一个聊天气泡|运行时会用换行连接|组织单气泡布局|布局单位|单个聊天气泡" src/kazusa_ai_chatbot/nodes/dialog_agent.py src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py tests/test_dialog_agent.py tests/test_l3_dialog_content_plan_contract.py tests/test_cognition_prompt_contract_text.py`
  - Expected result: no matches. `rg` exit code 1 is acceptable.
- `rg '"\\n"\\.join\\(messages\\)|"\\n"\\.join\\(replies\\)' src/adapters`
  - Expected result: no matches. `rg` exit code 1 is acceptable.
- `rg "mention_target_user" src/kazusa_ai_chatbot/nodes/dialog_agent.py src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`
  - Expected result: no matches. `rg` exit code 1 is acceptable.
- `rg "QQ|NapCat|Discord" src/kazusa_ai_chatbot/nodes/dialog_agent.py src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`
  - Expected result: no matches introduced by this plan. Existing mentions, if
    any, require explicit review and justification before sign-off.

### Live LLM Quality Gate

Run live LLM checks one case at a time only after deterministic tests pass.
Use `debug-llm` and produce a human-readable review artifact.

Cases:

- one-message casual online text reply;
- two-message reaction plus light boundary;
- two-message serious boundary;
- technical answer with one introduction message plus one fixed-format block.

Expected quality:

- L3 describes sequence shape without platform assumptions.
- Dialog follows the L3 sequence shape.
- Dialog does not mention delivery mechanics, delay, adapter, platform, or
  native reply.
- Fixed-format content remains intact as a complete message string.

## Independent Plan Review

Plan review performed on 2026-07-01 against the current dirty worktree and the
completed inline delivery mention plan. Native subagent review was not used
because the available tool requires explicit user authorization for delegation.

Findings addressed:

- Replaced stale `mention_target_user` and mention-prefix semantics with the
  current inline mention contract.
- Separated first-message native reply from per-message inline mention
  replacement.
- Added the missing `service.py` verification boundary for multi-message inline
  candidate construction.
- Replaced stale prefix/dialog-flag test commands with current inline mention
  and adapter sequence tests.
- Tightened ownership wording so dialog follows upstream rendering and does
  not decide reply, mention metadata, delay, platform, or delivery mechanics.
Status: review blockers remediated. The plan entered fallback single-agent
execution after explicit user approval on 2026-07-01.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for changed Python, tests, prompts, docs,
  and command artifacts.
- Alignment with `Must Do`, `Deferred`, `Change Surface`, exact contracts,
  implementation order, verification gates, and acceptance criteria.
- Ownership boundaries: L3 sequence intent, dialog wording, service response
  assembly and inline candidates, adapters delivery mechanics.
- Hidden fallback risks: no newline-join normal delivery, per-message schema
  expansion, platform-specific prompt assumption, or one-bubble compatibility
  shim.
- Adapter reliability: sequence tasks are tracked, logged, and cleaned up; tests
  patch delay instead of sleeping for real 1-5 second waits.
- Handoff quality: focused tests map to risks, static greps are current, live
  LLM evidence is readable if run, and deferred dispatcher work stays deferred.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows
review-only fixture/documentation corrections. If a finding requires new
schema, dispatcher API changes, prompt ownership changes, or a broader adapter
contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- L3 prompt and tests use outbound message-sequence wording instead of
  single-bubble layout wording.
- Dialog prompt and tests define `final_dialog` as ordered outbound messages.
- Active L3/dialog prompts contain no one-bubble or newline-join contract
  wording.
- Normal Discord `/chat` delivery sends `ChatResponse.messages` as logical
  messages in order.
- Normal NapCat `/chat` delivery sends `ChatResponse.messages` as logical
  messages in order.
- Native reply applies only to the first logical outbound message.
- Inline delivery mention candidates are applied independently to each logical
  outbound message and only replace exact authored `@display_name` tokens.
- Follow-up logical messages are delayed by adapter-owned deterministic
  length-based timing with a 1-5 second clamp.
- Adapter inbound handlers do not wait on follow-up delay before returning.
- Delivery receipt records the first successful platform message id.
- Existing `ChatResponse` schema remains unchanged.
- Dispatcher/background artifact single-message behavior remains unchanged and
  documented as deferred.
- Focused tests, regression tests, static greps, prompt render checks, and the
  independent code review gate pass.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Dialog starts deciding message count without upstream instruction | Keep L3 `rendering` as the sequence-shape owner and update dialog prompt to execute it | L3/dialog prompt tests and live LLM cases |
| Brain prompt gains platform-specific assumptions | Use platform-neutral wording and static grep for platform names in active prompts | Static grep and code review |
| Follow-up delay blocks adapter event handling | Use adapter-owned background sequence tasks | Adapter tests and code review |
| Native reply repeats on every follow-up | First-message-only sequence contract | Discord/NapCat sequence tests |
| Inline mention candidates are lost or applied to the wrong logical message | Per-message exact token replacement contract | Discord/NapCat inline mention tests |
| Platform chunking is confused with logical follow-up messages | Keep chunking inside each logical message and delay only between logical messages | Discord `_split_message` tests |
| First send fails and follow-ups become detached | Stop sequence when first send fails and log the error | Adapter failure-path test or code review evidence |
| Delivery receipt points at a follow-up instead of the first message | Record first successful platform message id | Existing and updated delivery receipt tests |
| Prompt edits increase local-model instruction burden | Keep prompts same size or shorter and avoid extra examples | Prompt render check and diff review |

## Execution Evidence

- Focused test baseline:
  - `tests/test_adapter_outbound_sequence.py -q` initially failed with
    `ModuleNotFoundError: No module named 'adapters.outbound_sequence'`.
  - `tests/test_dialog_agent.py::test_dialog_generator_prompt_describes_message_sequence_contract -q`
    initially failed because the dialog prompt still described one-bubble
    layout and newline joining.
  - `tests/test_l3_dialog_content_plan_contract.py::test_content_plan_prompt_describes_message_sequence_rendering -q`
    initially failed because L3 still described single-bubble rendering.
  - Adapter sequence tests initially failed because adapters had no
    `_normal_chat_delivery_tasks` and still joined returned message lists.
  - `tests/test_service_background_consolidation.py::test_chat_response_preserves_message_sequence_for_inline_mentions -q`
    passed at baseline, confirming service response assembly already preserved
    the message list while candidate discovery scanned visible text.
- Production implementation summary:
  - Added `src/adapters/outbound_sequence.py` with 1-5 second clamped
    length-based follow-up delay constants.
  - Updated L3 `content_plan.rendering` prompt wording to describe outbound
    message sequence shape and fixed-format handling.
  - Updated dialog generator prompt so each `final_dialog` string is one
    ordered outbound online text message and removed active newline-join /
    one-bubble wording.
  - Updated Discord and NapCat normal `/chat` paths to send the first logical
    message immediately, apply native reply only to that first message, post
    delivery receipt using the first sent id, and schedule follow-ups through
    tracked adapter-owned tasks.
  - Applied inline delivery mention rendering per logical message before
    Discord chunking or NapCat segment conversion.
  - Added task finalizers that discard completed follow-up tasks and log
    unexpected task failures.
  - Updated Brain Service and Adapter ICDs for ordered message delivery,
    first-message reply behavior, per-message mention rendering, adapter-owned
    delay, and single-message runtime callback scope.
- Static grep results:
  - `venv\Scripts\python.exe -m py_compile ...` passed for all changed Python
    source and test files.
  - Prompt render script for `_CONTENT_PLAN_AGENT_PROMPT` and
    `_DIALOG_GENERATOR_PROMPT` printed `prompt render ok`.
  - `rg "一个聊天气泡|运行时会用换行连接|组织单气泡布局|布局单位|单个聊天气泡" ...`
    over active L3/dialog prompt files and deterministic prompt-contract tests
    returned no matches; `rg` exit code 1 accepted.
  - `rg '"\\n"\\.join\\(messages\\)|"\\n"\\.join\\(replies\\)' src/adapters`
    returned no matches; `rg` exit code 1 accepted.
  - `rg "mention_target_user" src/kazusa_ai_chatbot/nodes/dialog_agent.py src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`
    returned no matches; `rg` exit code 1 accepted.
  - `rg "QQ|NapCat|Discord" src/kazusa_ai_chatbot/nodes/dialog_agent.py src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`
    returned no matches; `rg` exit code 1 accepted.
  - `git diff --check` passed with line-ending warnings only.
- Prompt render checks:
  - Rendered both active prompt templates with dummy values and no
    `.format(...)` placeholder errors.
- Focused test results:
  - `venv\Scripts\python.exe -m pytest tests/test_adapter_outbound_sequence.py tests/test_dialog_agent.py::test_dialog_generator_prompt_describes_message_sequence_contract tests/test_l3_dialog_content_plan_contract.py::test_content_plan_prompt_describes_message_sequence_rendering -q`
    passed, 6 tests.
  - `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_discord_handle_message_sends_brain_messages_as_sequence_with_first_reply_only tests/test_runtime_adapter_registration.py::test_discord_on_message_replaces_inline_delivery_mentions_across_message_sequence tests/test_runtime_adapter_registration.py::test_napcat_handle_event_sends_brain_messages_as_sequence_with_first_reply_only tests/test_runtime_adapter_registration.py::test_napcat_handle_event_replaces_inline_delivery_mentions_across_message_sequence -q`
    passed, 4 tests.
  - `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_napcat_handle_event_does_not_wait_for_followup_delay -q`
    passed.
  - `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_discord_on_message_does_not_wait_for_followup_delay -q`
    passed.
  - `venv\Scripts\python.exe -m pytest tests/test_service_background_consolidation.py::test_chat_response_preserves_message_sequence_for_inline_mentions -q`
    passed.
  - `venv\Scripts\python.exe -m pytest tests/test_cognition_prompt_contract_text.py::test_l3_content_plan_scope_preserves_complete_plan_deliverables -q`
    passed.
- Regression test results:
  - `venv\Scripts\python.exe -m pytest tests/test_dialog_agent.py -q`
    passed, 22 tests.
  - `venv\Scripts\python.exe -m pytest tests/test_l3_dialog_content_plan_contract.py -q`
    passed, 7 tests.
  - `venv\Scripts\python.exe -m pytest tests/test_inline_delivery_mentions.py -q`
    passed, 7 tests.
  - `venv\Scripts\python.exe -m pytest tests/test_service_background_consolidation.py::test_chat_response_adds_inline_delivery_mentions_without_channel_gate tests/test_service_background_consolidation.py::test_chat_response_reply_feature_keeps_inline_delivery_mentions tests/test_service_background_consolidation.py::test_chat_response_adds_multiple_inline_delivery_mentions tests/test_service_background_consolidation.py::test_chat_response_preserves_message_sequence_for_inline_mentions tests/test_service_background_consolidation.py::test_chat_response_tracks_deliverable_assistant_row -q`
    passed, 5 tests.
  - `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py -q`
    passed, 69 tests.
  - `venv\Scripts\python.exe -m pytest tests/test_cognition_prompt_contract_text.py::test_l3_content_plan_scope_preserves_complete_plan_deliverables tests/test_persona_supervisor2_action_selection.py::test_l2d_prompt_does_not_offer_missing_scene_context tests/test_cognition_prompt_contract_text.py::test_dialog_agent_prompt_contract_is_platform_neutral -q`
    passed, 3 tests, after updating stale prompt assertions.
  - `venv\Scripts\python.exe -m pytest tests/control_console_e2e/test_page_navigation_e2e.py::test_each_sidebar_page_has_connected_or_explicitly_gated_state tests/control_console_e2e/test_service_lifecycle_e2e.py::test_service_cards_start_stop_and_dependency_states tests/control_console_e2e/test_service_lifecycle_e2e.py::test_service_action_click_shows_operator_feedback_while_pending tests/control_console_e2e/test_visual_product_acceptance_e2e.py::test_desktop_visual_acceptance_for_cards_buttons_and_branding -q`
    passed, 4 tests, after remediating surfaced E2E selector and layout
    issues.
  - `venv\Scripts\python.exe -m pytest -q` passed:
    2559 passed, 2 skipped, 360 deselected.
- Live LLM review artifact:
  - Migrated `tests/test_dialog_one_bubble_layout_live_llm.py` to
    `tests/test_dialog_message_sequence_live_llm.py` and rewrote legacy live
    fixtures around the message-sequence contract.
  - Added a two-message follow-up live LLM case with explicit
    `expected_message_count` validation.
  - `venv\Scripts\python.exe -m pytest tests/test_l3_dialog_content_plan_live_llm.py::test_live_dialog_content_plan_technical_golden -m live_llm -q -s`
    passed.
  - `venv\Scripts\python.exe -m pytest tests/test_dialog_message_sequence_live_llm.py::test_live_dialog_message_sequence_two_message_followup -m live_llm -q -s`
    passed.
  - `venv\Scripts\python.exe -m pytest tests/test_dialog_message_sequence_live_llm.py::test_live_dialog_message_sequence_group_casual_reply -m live_llm -q -s`
    passed.
  - `venv\Scripts\python.exe -m pytest tests/test_l3_dialog_content_plan_live_llm.py::test_live_l3_content_plan_technical_comparison -m live_llm -q -s`
    passed.
  - Human-readable review artifact:
    `test_artifacts/llm_traces/dialog_message_sequence_review_20260701.md`.
- Direct dialog-agent live LLM unit review:
  - Added `tests/test_dialog_agent_direct_live_llm.py` with fabricated,
    complete L3 dialog state inputs that call the dialog agent directly.
  - Covered seven direct dialog cases: group broadcast, named participant
    inline tag, two-message follow-up boundary, technical numeric comparison,
    Python code block preservation, unknown referent clarification, and
    privacy-boundary code handling.
  - `venv\Scripts\python.exe -m py_compile tests\test_dialog_agent_direct_live_llm.py`
    passed.
  - `venv\Scripts\python.exe -m pytest tests\test_dialog_agent_direct_live_llm.py --collect-only -m live_llm -q`
    collected 7 live LLM tests.
  - Each of the seven live LLM cases was run individually with
    `venv\Scripts\python.exe -m pytest <case> -m live_llm -q -s` and passed
    after output inspection.
  - `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q`
    passed, 22 tests.
  - Human-readable review artifact:
    `test_artifacts/llm_traces/dialog_agent_direct_live_llm_review_20260701.md`.
- Final closeout verification:
  - `venv\Scripts\python.exe -m pytest -q` passed:
    2559 passed, 2 skipped, 367 deselected.
  - `git diff --check` passed with line-ending warnings only.
- Independent code review:
  - User explicitly requested execution without subagents, so the planned
    independent subagent review was replaced by fallback single-agent review.
  - Review finding 1: adapter tests proved delayed follow-ups but did not prove
    the inbound handler returned before the delay. Fixed by adding Discord and
    NapCat blocking-sleep non-blocking tests; reran affected tests and full
    adapter suite.
  - Review finding 2: follow-up tasks were discarded but unexpected task
    exceptions would rely on event-loop warnings. Fixed by adding adapter task
    finalizers that retrieve and log unexpected task exceptions; reran compile
    and full adapter suite.
  - Review finding 3: `tests/test_cognition_prompt_contract_text.py` had one
    active L3 prompt-contract assertion for the retired rendering wording.
    Fixed the assertion and reran the targeted L3 prompt-contract test.
  - No remaining in-scope findings after fallback review.
- Residual risks:
  - Dispatcher and background artifact callback delivery remain intentionally
    single-message scope per this plan.
